#!/usr/bin/env python3
"""Audit native training augmentation and row survival on CPU; never train.

The immutable PR137 training labels and r61 acquisition helpers are reused.
Three fixed passes are a new diagnostic sample, not historical training replay.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import MethodType
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
import run_quality_focal_r61 as r61
from pose_checkpoint import get_recipe, recipe_identity

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'baomai-augmentation-audit-r63'
PASSES = 3
SEED = 46
NATIVE_BLOBS = {
    'data/augment.py': '14d574c52f2d92809a5c9f7d0fa1e78ec3889560',
    'data/base.py': '541781d541f6f0f8fc2a2e85c57c3fcd1f327e17',
    'data/build.py': '2e0ff237bc35a6f878b3a3113e6e62113830e3cd',
    'data/dataset.py': '3f8d96688bd0bdef7a78639bc2daf870507607de',
    'utils/instance.py': 'da3a50063f81fbae88ff50139ee621034bee349d',
}
PROTOCOL = {
    'passes_per_training_photo': PASSES, 'seed_base': SEED,
    'sample_seed': '46 + pass_index * 142 + native_dataset_index; indices start at zero',
    'dataset': 'exact PR137 candidate training photos and labels used by r61',
    'training_photos': 142, 'source_fronts': 3999,
    'control': 'native LetterBox and Format only, on the same training rectangular canvas',
    'treatment': 'unchanged native training augmentation; no new augmentation',
    'native_box_filter': {'width_strictly_greater_px': 2, 'height_strictly_greater_px': 2,
                          'area_ratio_strictly_greater': .10, 'aspect_ratio_strictly_less': 100},
    'native_coordinate_bounds': '0 <= x <= width and 0 <= y <= height; upper boundaries are inclusive',
    'panel_selection': 'first qualifying samples in native traversal order; at most three removed-row and three retained-clipping/visibility cases, with unique photographs across both groups',
    'maximum_panels': 6, 'weights_loaded': False, 'training': False, 'inference': False,
    'annotation_edits': False, 'historical_training_rng_reproduced': False,
    'drop_implies_annotation_error': False, 'box_area_ratio_is_physical_visibility': False,
}
need, sha, checked, encoded, put, read = r61.need, r61.sha, r61.checked, r61.encoded, r61.put, r61.read


def source_check(root, commit, driver_sha):
    need(root == ROOT and re.fullmatch('[0-9a-f]{40}', commit or '')
         and re.fullmatch('[0-9a-f]{64}', driver_sha or ''), 'Complete published source identities required')
    need(r61.rev(root) == commit, 'Wrong source checkout preserved')
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root, text=True)
    need(not status.strip(), 'Changed source preserved; use a fresh audit source folder')
    checked(Path(__file__), driver_sha)
    closure = {**r61.local_source_files(root), 'scripts/' + Path(__file__).name: driver_sha}
    tree = subprocess.check_output(['git', 'ls-tree', '-r', 'HEAD'], cwd=root, text=True)
    blobs = {line.split('\t', 1)[1]: line.split()[2] for line in tree.splitlines()}
    for name, expected in closure.items():
        data = checked(root / name, expected).read_bytes()
        need(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blobs.get(name),
             'Source differs from published Git object: ' + name)
    return {'commit': commit, 'driver_sha256': driver_sha, 'files_sha256': closure}


def make_plan(root, archive, source):
    upstream, payload, provenance = r61.label_plan(root, archive)
    entries = [row for row in upstream['roles']['reference'] if row['split'] == 'train']
    need(len(entries) == 142 and upstream['roles']['reference'] == upstream['roles']['candidate'], 'Training set differs')
    plan = {'schema': 'training-augmentation-audit-r63-v1', 'protocol': PROTOCOL,
            'source': source, 'entries': entries, 'counts': upstream['counts']['reference']['train'],
            'recipe': recipe_identity(r61.RECIPE), 'recipe_settings': get_recipe(r61.RECIPE),
            'input_archive': upstream['input_archive'],
            'provenance_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in provenance.items()}}
    need(plan['counts'] == [142, 3999, 7259, 739, 13], 'Training inventory differs')
    return plan, payload, provenance


def prepare(root, work, archive, assets, source, download=False):
    plan, payload, provenance = make_plan(root, archive, source)
    if download:
        def obtain(entry):
            path = assets / entry['image_source']
            if path.exists():
                checked(path, entry['image_sha256'])
                return
            url = f"https://media.githubusercontent.com/media/Siuuuers/dl/{r61.r47.SOURCE_INPUTS}/{entry['image_source']}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            need(hashlib.sha256(data).hexdigest() == entry['image_sha256'], 'Image download differs: ' + entry['id'])
            put(path, data)
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(obtain, [row for row in plan['entries'] if row['source_kind'] == 'original']))
        r61.base.obtain_supplementary(root, assets, plan['entries'])
    folder = work / 'dataset'
    for row in plan['entries']:
        image = checked(assets / row['image_source'], row['image_sha256'])
        need(r61.image_size(image.read_bytes()) == (1280, 720), 'Image dimensions differ: ' + row['id'])
        target = folder / row['image']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(image, target)
        checked(target, row['image_sha256'])
        put(folder / row['label'], payload[row['id']])
    put(folder / 'data.yaml', encoded(r61.dataset_config(folder)))
    for name, data in provenance.items():
        put(work / 'inputs' / name, data)
    put(work / 'plan.json', encoded(plan))
    verify_stage(root, work, archive, source)
    print('Prepared 142 training photographs, 3999 fronts; no model is needed.', flush=True)


def verify_stage(root, work, archive, source):
    expected, _, provenance = make_plan(root, archive, source)
    plan = read(work / 'plan.json')
    need(plan == expected, 'Saved input/source plan differs')
    folder = work / 'dataset'
    for row in plan['entries']:
        checked(folder / row['image'], row['image_sha256'])
        checked(folder / row['label'], row['label_sha256'])
    for name, data in provenance.items():
        checked(work / 'inputs' / name, hashlib.sha256(data).hexdigest())
    for field, subfolder, suffixes in (('image', 'images', {'.png', '.jpg', '.jpeg'}), ('label', 'labels', {'.txt'})):
        found = {p.relative_to(folder).as_posix() for p in (folder / subfolder).rglob('*') if p.suffix.lower() in suffixes}
        need(found == {row[field] for row in plan['entries']}, 'Unexpected staged membership: ' + subfolder)
    need(read(folder / 'data.yaml') == r61.dataset_config(folder), 'Dataset configuration differs')
    return plan


def native_contract():
    import ultralytics
    from ultralytics.data.augment import RandomPerspective
    import numpy as np
    need(ultralytics.__version__ == '8.4.182', 'This audit requires Ultralytics 8.4.182')
    package = Path(ultralytics.__file__).parent
    files = {}
    for name, expected in NATIVE_BLOBS.items():
        data = (package / name).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        need(actual == expected, 'Pinned augmentation source differs: ' + name)
        files[name] = {'git_blob_sha1': actual, 'sha256': hashlib.sha256(data).hexdigest()}
    packages = {}
    for name in ('ultralytics', 'torch', 'torchvision', 'opencv-python', 'numpy', 'Pillow', 'albumentations'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Independent arithmetic checks the native, strict filtering inequalities.
    source = np.array([[0, 0, 10, 10]] * 6, dtype=np.float32).T
    transformed = np.array([[0, 0, 10, 10], [0, 0, 2, 10], [0, 0, 10, 2],
                            [0, 0, 3, 3], [0, 0, 10, 1], [0, 0, 1000, 3]], dtype=np.float32).T
    need(RandomPerspective.box_candidates(source, transformed).tolist() == [True, False, False, False, False, False],
         'Native filter fixture changed')
    points = np.array([[[4, 5, 2], [0, 0, 0]], [[10, 10, 2], [11, 5, 2]]], dtype=np.float32)
    out = RandomPerspective().apply_keypoints(points.copy(), np.eye(3), (10, 10))
    need(out[:, :, 2].tolist() == [[2, 0], [2, 0]], 'Native visibility or boundary fixture changed')
    return {'packages': packages, 'python': platform.python_version(), 'native_files': files,
            'native_filter_and_missing_point_fixtures': 'passed', 'device': 'cpu'}


def walk_transforms(value):
    yield value
    for item in getattr(value, 'transforms', []):
        yield from walk_transforms(item)


def transform_description(observer):
    result = []
    for node in observer.nodes:
        item = {'name': type(node).__name__}
        for key in ('p', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'direction', 'flip_idx'):
            if hasattr(node, key):
                value = getattr(node, key)
                if isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
                    item[key] = value
        if node in observer.albumentations:
            item['enabled'] = node.transform is not None
            item['transforms'] = str(node.transform) if node.transform is not None else None
        result.append(item)
    return result


class Observer:
    """Observe real native calls; compare row filtering independently without changing inputs."""
    def __init__(self, transforms):
        import numpy as np
        from ultralytics.data.augment import RandomPerspective, RandomFlip, Albumentations
        self.np = np
        self.nodes = list(walk_transforms(transforms))
        self.affines = [x for x in self.nodes if isinstance(x, RandomPerspective)]
        need(len(self.affines) == 1, 'Expected exactly one native affine transform')
        self.flips = [x for x in self.nodes if isinstance(x, RandomFlip)]
        self.albumentations = [x for x in self.nodes if isinstance(x, Albumentations)]
        need(all(not x.contains_spatial for x in self.albumentations if x.transform is not None),
             'Unexpected extra spatial transform; row lineage must be implemented before auditing it')
        self.original_affine = self.affines[0].apply_instances
        self.original_flips = [x.get_params for x in self.flips]
        self.enabled = True
        def observe_affine(this, labels, params):
            if not self.enabled:
                return self.original_affine(labels, params)
            self.capture(labels, params)
            result = self.original_affine(labels, params)
            actual = result['instances']
            need(np.allclose(actual.bboxes, self.expected.bboxes[self.keep], atol=1e-5, rtol=0)
                 and np.array_equal(actual.keypoints, self.expected.keypoints[self.keep]), 'Native affine replay differs')
            self.survivors = deepcopy(actual)
            return result
        self.affines[0].apply_instances = MethodType(observe_affine, self.affines[0])
        for transform, original in zip(self.flips, self.original_flips):
            def observe_flip(this, labels, _original=original):
                params = _original(labels)
                if self.enabled:
                    self.flip_parameters.append(deepcopy(params))
                return params
            transform.get_params = MethodType(observe_flip, transform)

    def begin(self):
        self.capture_count, self.flip_parameters = 0, []

    def capture(self, labels, params):
        from ultralytics.utils.instance import Instances
        np = self.np
        self.capture_count += 1
        need(self.capture_count == 1, 'Repeated affine transform for one image')
        source = deepcopy(labels['instances'])
        source.convert_bbox('xyxy')
        source.denormalize(*params['orig_shape'][::-1])
        need(not len(source.segments), 'This audit is for boxes and two corners, not masks')
        self.before, self.params = source, deepcopy(params)
        self.unclipped = self.affines[0].apply_bboxes(source.bboxes.copy(), params['M'])
        points = source.keypoints.copy()
        xy = np.concatenate((points[..., :2], np.ones((*points.shape[:2], 1), dtype=points.dtype)), axis=-1) @ params['M'].T
        xy = xy[..., :2] / xy[..., 2:3]
        width, height = params['size']
        outside = (xy[..., 0] < 0) | (xy[..., 0] > width) | (xy[..., 1] < 0) | (xy[..., 1] > height)
        points[..., :2] = xy
        points[..., 2][outside] = 0
        self.expected = Instances(self.unclipped.copy(), source.segments.copy(), points, bbox_format='xyxy', normalized=False)
        self.expected.clip(width, height)
        original_wh = source.bboxes[:, 2:] - source.bboxes[:, :2]
        wh = self.expected.bboxes[:, 2:] - self.expected.bboxes[:, :2]
        self.width_height = wh
        self.area_ratio = wh.prod(1) / ((original_wh * params['scale']).prod(1) + 1e-16)
        self.aspect = np.maximum(wh[:, 0] / (wh[:, 1] + 1e-16), wh[:, 1] / (wh[:, 0] + 1e-16))
        self.keep = (wh[:, 0] > 2) & (wh[:, 1] > 2) & (self.area_ratio > .10) & (self.aspect < 100)
        native = self.affines[0].box_candidates((source.bboxes * params['scale']).T, self.expected.bboxes.T)
        need(np.array_equal(native, self.keep), 'Independent filtering arithmetic differs from native')

    def final_check(self, output):
        np = self.np
        need(self.capture_count == 1 and len(self.flip_parameters) == 2, 'Incomplete native transform observation')
        expected = deepcopy(self.survivors)
        slot_order = [0, 1]
        for p in self.flip_parameters:
            expected.convert_bbox('xywh')
            if p['flip']:
                need(p['direction'] == 'horizontal', 'Vertical flipping is disabled by this recipe')
                expected.fliplr(p['w'])
                expected.keypoints = np.ascontiguousarray(expected.keypoints[:, p['flip_idx'], :])
                slot_order = [slot_order[i] for i in p['flip_idx']]
        expected.convert_bbox('xywh')
        expected.normalize(output['img'].shape[2], output['img'].shape[1])
        need(np.allclose(output['bboxes'].numpy(), expected.bboxes, atol=1e-7, rtol=0)
             and np.allclose(output['keypoints'].numpy(), expected.keypoints, atol=1e-7, rtol=0),
             'Final native target geometry/order differs from affine plus recorded slot permutation')
        need(len(output['cls']) == int(self.keep.sum()) and bool((output['cls'] == 0).all()), 'Final class/row count differs')
        return slot_order


def set_sample_seed(seed, observer):
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(seed, deterministic=True)
    for transform in observer.albumentations:
        if transform.transform is not None and hasattr(transform.transform, 'set_random_seed'):
            transform.transform.set_random_seed(seed)


def pixel_targets(output):
    import numpy as np
    width, height = output['img'].shape[2], output['img'].shape[1]
    boxes = output['bboxes'].numpy() * np.array([width, height, width, height], dtype=np.float32)
    boxes = np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), axis=1)
    points = output['keypoints'].numpy() * np.array([width, height, 1], dtype=np.float32)
    return boxes, points


def summarize(rows, images):
    by_source = {str(n): Counter() for n in (0, 1, 2)}
    reasons, transitions = Counter(), Counter()
    for row in rows:
        group = by_source[str(row['source_known_corners'])]
        group['source_presentations'] += 1
        group['retained' if row['retained'] else 'removed'] += 1
        group['box_clipped'] += row['box_clipped']
        group['known_corners_outside_canvas'] += row['known_corners_made_unavailable']
        reasons.update(row['filter_failure_reasons'])
        dest = str(sum(v > 0 for v in row['final_corner_flags'])) if row['retained'] else 'removed'
        transitions[f"{row['source_known_corners']}->{dest}"] += 1
    retained = sum(row['retained'] for row in rows)
    removed_positive = [row for row in rows if not row['retained']
                        and row['affine_clipped_box_xyxy'][2] > row['affine_clipped_box_xyxy'][0]
                        and row['affine_clipped_box_xyxy'][3] > row['affine_clipped_box_xyxy'][1]]
    return {'source_object_presentations': len(rows), 'retained_object_presentations': retained,
            'removed_object_presentations': len(rows) - retained,
            'removed_distinct_source_rows': len({(r['image_id'], r['source_row']) for r in rows if not r['retained']}),
            'removed_positive_clipped_box_area': len(removed_positive),
            'removed_zero_clipped_box_area': len(rows) - retained - len(removed_positive),
            'removed_positive_area_maximum_short_side_px': max((min(r['affine_clipped_box_xyxy'][2] - r['affine_clipped_box_xyxy'][0],
                r['affine_clipped_box_xyxy'][3] - r['affine_clipped_box_xyxy'][1]) for r in removed_positive), default=0.0),
            'retained_clipped_presentations': sum(row['retained'] and row['box_clipped'] for row in rows),
            'known_corners_outside_canvas_before_row_filter': sum(row['known_corners_made_unavailable'] for row in rows),
            'source_unavailable_corners_restored': sum(row['unavailable_corners_restored'] for row in rows),
            'source_corner_groups': {key: dict(value) for key, value in by_source.items()},
            'source_to_final_corner_count': dict(sorted(transitions.items())),
            'filter_failure_reasons_nonexclusive': dict(reasons),
            'images_with_removed_rows': sum(any(not r['retained'] for r in rows if r['sample_id'] == image['sample_id']) for image in images),
            'zero_source_fronts_preserved': all(image['final_rows'] == 0 for image in images if image['source_rows'] == 0)}


def render_panel(original, augmented, rows, output):
    """Show raw/augmented raster views with geometrically verified box/corner overlays."""
    from PIL import Image, ImageDraw
    import numpy as np
    source = Image.fromarray(np.ascontiguousarray(original[..., ::-1]))
    target = Image.fromarray(augmented['img'].permute(1, 2, 0).numpy())
    canvas = Image.new('RGB', (2560, max(source.height, target.height) + 70), 'white')
    canvas.paste(source, (0, 50)); canvas.paste(target, (1280, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 12), 'SOURCE: archived labels; red = removed later, orange = clipped/known point lost', fill='black')
    draw.text((1295, 12), 'NATIVE AUGMENTATION: colored rows require review; not proven wrong labels', fill='black')
    draw.text((15, 30), 'Known corners only: L cyan, R magenta. Missing corners are not drawn. JPEG display copy; originals are hash-verified.', fill='black')
    draw.text((1295, 30), 'Corner labels follow final image-left/image-right slots after horizontal reflection.', fill='black')
    interesting = [r for r in rows if not r['retained'] or r['box_clipped'] or r['known_corners_made_unavailable']]
    for row in interesting:
        color = 'red' if not row['retained'] else 'orange'
        b = row['source_box_xyxy']
        draw.rectangle((b[0], b[1] + 50, b[2], b[3] + 50), outline=color, width=2)
        draw.text((b[0], b[1] + 37), str(row['source_row']), fill=color)
        for side, point in enumerate(row['source_keypoints_xyv']):
            if point[2] > 0:
                x, y = point[0], point[1] + 50
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=('cyan', 'magenta')[side], outline='black')
        b = row['final_box_xyxy'] if row['retained'] else row['affine_clipped_box_xyxy']
        if not row['retained'] and row['slot_order'] == [1, 0]:
            b = [target.width - b[2], b[1], target.width - b[0], b[3]]
        if b[2] > b[0] and b[3] > b[1]:
            draw.rectangle((1280 + b[0], b[1] + 50, 1280 + b[2], b[3] + 50), outline=color, width=2)
            draw.text((1280 + b[0], b[1] + 37), str(row['source_row']), fill=color)
        if row['retained']:
            for side, point in enumerate(row['final_keypoints_xyv']):
                if point[2] > 0:
                    x, y = 1280 + point[0], point[1] + 50
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=('cyan', 'magenta')[side], outline='black')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, 'JPEG', quality=94, subsampling=0, optimize=True)


def audit(work):
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.augment import Compose, LetterBox
    runtime = native_contract()
    torch.set_num_threads(2)
    plan, folder = read(work / 'plan.json'), work / 'dataset'
    cfg = get_cfg(overrides={**plan['recipe_settings'], 'task': 'pose', 'mode': 'train', 'workers': 0})
    dataset = build_yolo_dataset(cfg, str(folder / 'images/train'), 4, r61.dataset_config(folder), mode='train', stride=32)
    loader_plan = {'roles': {'reference': plan['entries']}, 'counts': {'reference': {'train': plan['counts']}}}
    inventory = r61.verify_loader(folder, loader_plan, 'reference', 'train', dataset)
    observer = Observer(dataset.transforms)
    control = Compose([LetterBox(new_shape=1280, scaleup=False), deepcopy(dataset.transforms.transforms[-1])])
    rows, images, controls, guarded = [], [], [], set()
    panels, selected_photos = [], set()
    candidates = {'removed': [], 'clipped_or_visibility': []}
    entries = {r['id']: r for r in plan['entries']}
    for repeat in range(PASSES):
        for index in range(len(dataset)):
            seed = SEED + repeat * len(dataset) + index
            source = dataset.get_image_and_label(index)
            identity = Path(source['im_file']).stem
            sample = f'{identity}/pass{repeat + 1:02d}'
            if repeat == 0:
                before = deepcopy(source['instances'])
                baseline = control(deepcopy(source))
                baseline_boxes, baseline_points = pixel_targets(baseline)
                need(len(baseline['cls']) == len(before) and np.array_equal(baseline_points[..., 2], before.keypoints[..., 2]),
                     'Unaugmented control changed row inventory or visibility')
                controls.append({'image_id': identity, 'rows': len(before), 'shape_chw': list(baseline['img'].shape),
                                 'boxes_xyxy': baseline_boxes.tolist(), 'keypoints_xyv': baseline_points.tolist()})
            observer.begin()
            set_sample_seed(seed, observer)
            output = dataset.transforms(deepcopy(source))
            raw = deepcopy(source['instances'])
            raw.convert_bbox('xyxy')
            raw.denormalize(source['img'].shape[1], source['img'].shape[0])
            need(tuple(source['img'].shape[:2]) == (720, 1280)
                 and np.array_equal(observer.before.bboxes, raw.bboxes)
                 and np.array_equal(observer.before.keypoints, raw.keypoints),
                 'Source panel coordinates differ from the raw input raster')
            slot_order = observer.final_check(output)
            final_boxes, final_points = pixel_targets(output)
            source_flags = observer.before.keypoints[..., 2]
            n = len(observer.before)
            need(n == len(dataset.labels[index]['cls']), 'Unexpected mixing or pre-affine row removal')
            categories = {'empty' if n == 0 else 'nonempty'}
            if not observer.keep.all():
                categories.add('removed')
            if categories - guarded:
                observer.enabled = False
                try:
                    set_sample_seed(seed, observer)
                    untouched = dataset.transforms(deepcopy(source))
                    for field in ('img', 'cls', 'bboxes', 'keypoints', 'batch_idx'):
                        need(torch.equal(output[field], untouched[field]), 'Observer changed native output: ' + field)
                finally:
                    observer.enabled = True
                guarded |= categories
            sample_rows, final_index = [], 0
            for j in range(n):
                after = observer.expected.keypoints[j, :, 2]
                clipped = not np.array_equal(observer.unclipped[j], observer.expected.bboxes[j])
                failures = []
                if observer.width_height[j, 0] <= 2: failures.append('width_le_2')
                if observer.width_height[j, 1] <= 2: failures.append('height_le_2')
                if observer.area_ratio[j] <= .10: failures.append('area_ratio_le_0_10')
                if observer.aspect[j] >= 100: failures.append('aspect_ratio_ge_100')
                retained = bool(observer.keep[j])
                need(retained == (not failures), 'Filter reason inventory differs')
                row = {'sample_id': sample, 'image_id': identity, 'pass': repeat + 1, 'seed': seed,
                       'source_row': j + 1, 'label_sha256': entries[identity]['label_sha256'],
                       'source_box_xyxy': observer.before.bboxes[j].tolist(),
                       'source_keypoints_xyv': observer.before.keypoints[j].tolist(),
                       'source_corner_flags': source_flags[j].astype(int).tolist(),
                       'source_known_corners': int((source_flags[j] > 0).sum()),
                       'affine_unclipped_box_xyxy': observer.unclipped[j].tolist(),
                       'affine_clipped_box_xyxy': observer.expected.bboxes[j].tolist(),
                       'affine_corner_flags': after.astype(int).tolist(),
                       'box_clipped': bool(clipped), 'native_filter_area_ratio': float(observer.area_ratio[j]),
                       'native_filter_aspect_ratio': float(observer.aspect[j]),
                       'filter_failure_reasons': failures, 'retained': retained,
                       'known_corners_made_unavailable': int(((source_flags[j] > 0) & (after == 0)).sum()),
                       'unavailable_corners_restored': int(((source_flags[j] == 0) & (after > 0)).sum()),
                       'slot_order': slot_order, 'final_box_xyxy': final_boxes[final_index].tolist() if retained else None,
                       'final_keypoints_xyv': final_points[final_index].tolist() if retained else None,
                       'final_corner_flags': final_points[final_index, :, 2].astype(int).tolist() if retained else None}
                final_index += retained
                sample_rows.append(row)
            rows.extend(sample_rows)
            image = {'sample_id': sample, 'image_id': identity, 'pass': repeat + 1, 'seed': seed,
                     'image_sha256': entries[identity]['image_sha256'], 'source_rows': n, 'final_rows': len(output['cls']),
                     'output_shape_chw': list(output['img'].shape), 'matrix': observer.params['M'].tolist(),
                     'scale': observer.params['scale'], 'canvas_wh': list(observer.params['size']), 'slot_order': slot_order,
                     'augmented_rgb_sha256': hashlib.sha256(output['img'].numpy().tobytes()).hexdigest()}
            images.append(image)
            category = ('removed' if any(not r['retained'] for r in sample_rows) else
                        'clipped_or_visibility' if any(r['box_clipped'] or r['known_corners_made_unavailable'] for r in sample_rows) else None)
            if category and identity not in selected_photos and len(candidates[category]) < 3:
                selected_photos.add(identity)
                name = f'panels/{len(panels) + 1:02d}-{identity}-pass{repeat + 1:02d}.jpg'
                render_panel(source['img'], output, sample_rows, work / 'results' / name)
                record = {'path': name, 'sample_id': sample, 'category': category, 'source_rows': [r['source_row'] for r in sample_rows
                          if not r['retained'] or r['box_clipped'] or r['known_corners_made_unavailable']]}
                candidates[category].append(record)
                panels.append(record)
        print(f'Completed native augmentation pass {repeat + 1}/{PASSES}: {len(images)} image presentations', flush=True)
    counts = summarize(rows, images)
    need(len(rows) == PASSES * 3999 and len(images) == PASSES * 142 and len(controls) == 142, 'Audit budget differs')
    need(counts['source_unavailable_corners_restored'] == 0 and counts['zero_source_fronts_preserved'], 'Missing points or empty scenes changed unexpectedly')
    result = {'schema': 'training-augmentation-audit-result-r63-v1', 'status': 'complete', 'plan_sha256': sha(work / 'plan.json'),
              'runtime': runtime, 'native_loader_inventory': inventory, 'protocol': PROTOCOL,
              'image_presentations': len(images), 'counts': counts, 'observer_equivalence_cases': sorted(guarded),
              'source_panel_geometry_equals_raw_input_all_samples': True,
              'actual_transform_pipeline': transform_description(observer), 'panels': panels,
              'interpretation': 'Native transformed-label arithmetic and row lineage are verified. A removed row or changed visible-point count is not by itself an error. Box area ratios do not measure physical visible material. Human image review is required to judge recognizability after cropping; this audit cannot prove which augmentation would improve a detector.'}
    put(work / 'results' / 'summary.json', encoded(result))
    put(work / 'results' / 'details.json.gz', gzip.compress(encoded({'rows': rows, 'images': images, 'controls': controls}), mtime=0))
    put(work / 'results' / 'results.md', markdown(result).encode())
    verify_result(work)
    return result


def markdown(result):
    c = result['counts']
    lines = ['# Native augmentation and label-survival audit', '',
             'This CPU diagnostic runs the existing augmentation. It does not train or score a model, change annotations, or replay historical training draws.', '',
             f"Three fixed passes contain {result['image_presentations']} image presentations and {c['source_object_presentations']} source-object presentations.",
             f"Native filtering retains {c['retained_object_presentations']} and removes {c['removed_object_presentations']}; {c['retained_clipped_presentations']} retained boxes are clipped.", '',
             f"Of the removed presentations, {c['removed_zero_clipped_box_area']} have zero clipped box area. The remaining {c['removed_positive_clipped_box_area']} retain positive box area, with a largest short side of {c['removed_positive_area_maximum_short_side_px']:.3f} pixels. These are box measurements, not physical object-visibility measurements.", '',
             '| Source known corners | Presentations | Retained | Removed | Box clipped | Known corners outside canvas |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group, row in c['source_corner_groups'].items():
        lines.append('| ' + group + ' | ' + ' | '.join(str(row.get(k, 0)) for k in ('source_presentations', 'retained', 'removed', 'box_clipped', 'known_corners_outside_canvas')) + ' |')
    lines += ['', '## Interpretation', '', result['interpretation'], '',
              'Source `v=0` means an unavailable annotation, not a negative physical existence label. A newly cropped-out known corner becomes unavailable in the transformed sample. Native internal coordinates for unavailable slots need not remain `0 0`; those coordinates are masked by the visibility flag. Canonical source labels remain unchanged.', '',
              'The native box filter uses width and height > 2 pixels, area ratio > 0.10 and aspect ratio < 100. Reasons may overlap. The area ratio compares the clipped transformed axis-aligned box to the scaled source box; it is not an occlusion fraction.', '',
              '## Review panels', '', 'Red marks source rows removed by the native filter; orange marks retained rows affected by clipping or lost known corners. These colors flag review cases, not annotation verdicts. Row numbers refer to the frozen training label file. Cyan/magenta dots show known left/right corner slots; unavailable points are not drawn. Panels are high-quality JPEG display copies; all measurement uses verified original images and numerical target geometry.', '']
    for row in result['panels']:
        lines.append(f"- [{row['sample_id']}]({row['path']}) — {row['category']}.")
    return '\n'.join(lines) + '\n'


def verify_result(work):
    result = read(work / 'results/summary.json')
    detail = json.loads(gzip.decompress((work / 'results/details.json.gz').read_bytes()))
    need(result['protocol'] == PROTOCOL and result['plan_sha256'] == sha(work / 'plan.json'), 'Saved audit identity differs')
    need(result['counts'] == summarize(detail['rows'], detail['images']), 'Aggregate replay differs')
    need(len(detail['rows']) == PASSES * 3999 and len(detail['images']) == PASSES * 142, 'Saved audit membership differs')
    need(len({(r['sample_id'], r['source_row']) for r in detail['rows']}) == len(detail['rows']), 'Duplicate row identity')
    need((work / 'results/results.md').read_text() == markdown(result), 'Result explanation differs from counts')
    for row in result['panels']:
        need((work / 'results' / row['path']).is_file(), 'Missing review panel')
    return result


def package(work):
    verify_result(work)
    paths = {'plan.json': work / 'plan.json'}
    paths.update({'results/' + p.relative_to(work / 'results').as_posix(): p for p in (work / 'results').rglob('*') if p.is_file()})
    inventory = {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(paths.items())}
    put(work / 'return-manifest.json', encoded({'schema': 'augmentation-audit-return-r63-v1', 'files': inventory}))
    paths['return-manifest.json'] = work / 'return-manifest.json'
    target = work / (PREFIX + '-results.zip')
    if not target.exists():
        with ZipFile(target, 'w', compression=ZIP_DEFLATED) as z:
            for name, path in sorted(paths.items()): z.write(path, name)
    with ZipFile(target) as z:
        need(set(z.namelist()) == set(paths) and z.testzip() is None, 'Return archive inventory differs')
        for name, path in paths.items():
            need(hashlib.sha256(z.read(name)).hexdigest() == sha(path), 'Return archive bytes differ: ' + name)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('source-check', 'prepare', 'audit', 'verify', 'package', 'native-check'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--inputs-dir', type=Path)
    parser.add_argument('--download-inputs', action='store_true')
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    source = source_check(root, args.source_revision, args.script_sha256)
    need(root != work and not work.is_relative_to(root) and not root.is_relative_to(work), 'Source and output directories must be separate')
    if args.command == 'source-check':
        print(json.dumps(source, indent=2)); return
    if args.command == 'native-check':
        print(json.dumps(native_contract(), indent=2)); return
    archive = args.archive.resolve() if args.archive else r61.prior.obtain_archive(work / 'cache/pr137')
    if args.command == 'prepare':
        prepare(root, work, archive, (args.inputs_dir or work / 'cache/assets').resolve(), source, args.download_inputs)
    else:
        verify_stage(root, work, archive, source)
        if args.command == 'audit':
            if (work / 'results/summary.json').exists():
                result = verify_result(work)
                need(result['runtime'] == native_contract(), 'Saved native runtime differs; use a fresh work folder')
            else:
                result = audit(work)
            print(json.dumps(result['counts'], indent=2))
        elif args.command == 'verify':
            print(json.dumps(verify_result(work)['counts'], indent=2))
        else:
            print(package(work))


if __name__ == '__main__':
    main()
