#!/usr/bin/env python3
"""Train a controlled original-versus-reviewed label comparison on 137 photos."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED

sys.dont_write_bytecode = True
from evaluate_pose import INFERENCE, _prediction_records, parse_pose_labels
from pose_checkpoint import get_recipe, recipe_identity, check_loaded_pose_model, check_training_settings
from predict_pose_folder import postclip_nms
from run_corner_average_r58 import average_predictions, report
from run_flip_inference_r56 import inverse_horizontal_predictions

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = '72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926'
OVERLAY = 'data/candidates/front-review-2026-11-26-r37/manifest.json'
OVERLAY_SHA = 'ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3'
INITIAL_SHA = 'e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e'
ASSET_COMMIT = 'ea4ac955f2dfaf1add24c243599715f802eb41a7'
RECIPE_ID = 'reviewed-incremental-fronts-seed46-v1'
RECIPE_SHA = 'f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8'
ARMS = ('original_labels', 'reviewed_labels')
PREFIX = 'baomai-training-label-comparison-r61'
REFERENCE_FILES = {
    'frozen666': ('data/references/sequence-refined-reference-2026-11-18/manifest.json',
                  '0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96', 666),
    'r50': ('data/reviews/fronts-2026-11-27-r50/manifest.json',
            'f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f', 654),
}
PROTOCOL = {
    'box_confidence_strictly_greater': .25, 'endpoint_confidence_inclusive': .5,
    'matching_iou_inclusive': .5, 'native_nms_iou': .7,
    'postclip_suppression_iou_strictly_greater': .7,
    'image_size_px': [1280, 720], 'imgsz': 1280, 'precision': 'float32', 'max_det': 300,
    'association_iou_inclusive': .5, 'coordinate_weights': [.5, .5],
    'primary_reference': 'r50', 'sensitivity_references': ['supplied664', 'frozen666'],
    'reference_coordinates_used_for_prediction': False,
    'scores_recalibrated_after_coordinate_averaging': False,
    'development_only': True, 'automatic_promotion': False,
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(path, expected):
    path = Path(path)
    need(path.is_file() and sha(path) == expected, 'Input differs: ' + str(path))
    return path


def frozen(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'Different existing file preserved: ' + str(path))
    else:
        path.write_bytes(data)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def read_references(root, original_records):
    """Return (references, provenance), reading only three pinned label sets.

    Records are the original data/manifest.json records, including or excluding
    train records. References are {name: {image_id: parsed target records}}.
    """
    root = Path(root).resolve()
    checked(root / 'data/manifest.json', MANIFEST_SHA)
    entries = {row['id']: row for row in original_records if row['split'] == 'val'}
    need(len(entries) == 20, 'Expected the same 20 development photographs')
    expected_records = {row['id']: row for row in read(root / 'data/manifest.json')['records'] if row['split'] == 'val'}
    need(entries == expected_records, 'Development photograph source records differ')
    references, provenance = {'supplied664': {}}, {'supplied664': {
        'manifest_path': 'data/manifest.json', 'manifest_sha256': MANIFEST_SHA,
        'objects': 664, 'labels': {}}}
    for identity, row in sorted(entries.items()):
        relative = 'data/' + row['label']
        data = checked(root / relative, row['label_sha256']).read_bytes()
        references['supplied664'][identity] = parse_pose_labels(data.decode(), 1280, 720, identity)
        provenance['supplied664']['labels'][identity] = {'path': relative, 'sha256': digest(data)}
    need(sum(map(len, references['supplied664'].values())) == 664, 'Supplied reference count differs')
    for name, (relative, expected, count) in REFERENCE_FILES.items():
        path = checked(root / relative, expected)
        rows = [row for row in read(path)['files'] if row.get('split', 'val') == 'val']
        need(len(rows) == 20 and {row['image_id'] for row in rows} == set(entries), 'Reference membership differs: ' + name)
        references[name], provenance[name] = {}, {'manifest_path': relative, 'manifest_sha256': expected,
                                                   'objects': count, 'labels': {}}
        for row in rows:
            identity = row['image_id']
            image_sha = row.get('source_image_sha256', row.get('image_sha256'))
            need(image_sha == entries[identity]['image_sha256'], 'Reference photograph differs: ' + identity)
            label = (path.parent / row['label_path']).resolve()
            need(label.is_relative_to(root), 'Reference label escapes source root')
            data = checked(label, row['label_sha256']).read_bytes()
            references[name][identity] = parse_pose_labels(data.decode(), 1280, 720, identity)
            provenance[name]['labels'][identity] = {'path': label.relative_to(root).as_posix(), 'sha256': digest(data)}
        need(sum(map(len, references[name].values())) == count, 'Reference object count differs: ' + name)
    return references, provenance


def score_native(references, native_by_arm, baseline=None):
    """Score any paired arms identically, returning (measurement, replay data).

    native_by_arm[arm]['original'|'reflected'][image_id] contains native
    _prediction_records outputs. Reflected records MUST still be in reflected
    image coordinates. This function inverts them, filters both views, then
    averages corners using predicted boxes alone. It never pools predictions.
    """
    need(len(native_by_arm) >= 2, 'At least two comparison arms are required')
    baseline = baseline or next(iter(native_by_arm))
    need(baseline in native_by_arm, 'Unknown baseline arm')
    identities = set(next(iter(references.values())))
    need(all(set(ref) == identities for ref in references.values()), 'Reference photograph sets differ')
    predictions, suppression, associations = {}, {}, {}
    for arm, native in native_by_arm.items():
        need(set(native) == {'original', 'reflected'} and all(set(view) == identities for view in native.values()),
             'Inference view or photograph membership differs: ' + arm)
        predictions[arm] = {'original': {}, 'two_view': {}}
        suppression[arm], associations[arm] = {}, {}
        for identity in sorted(identities):
            original = [p for p in native['original'][identity] if p['confidence'] > .25]
            reflected = [p for p in inverse_horizontal_predictions(native['reflected'][identity], 1280) if p['confidence'] > .25]
            kept, original_suppression = postclip_nms(original)
            flipped, reflected_suppression = postclip_nms(reflected)
            averaged, links = average_predictions(kept, flipped)
            predictions[arm]['original'][identity] = kept
            predictions[arm]['two_view'][identity] = averaged
            suppression[arm][identity] = {'original': original_suppression, 'reflected': reflected_suppression}
            associations[arm][identity] = links
    measurement = {'protocol': PROTOCOL, 'baseline_arm': baseline, 'references': {}, 'paired_changes': {}}
    per_image, cases = {}, {}
    for reference, gold in references.items():
        measurement['references'][reference], measurement['paired_changes'][reference] = {}, {}
        per_image[reference], cases[reference] = {}, {}
        for arm, outputs in predictions.items():
            measurement['references'][reference][arm], per_image[reference][arm], cases[reference][arm] = {}, {}, {}
            for output, values in outputs.items():
                summary, images, states = report(gold, values)
                summary['detection'].pop('endpoint_errors_px', None)
                measurement['references'][reference][arm][output] = summary
                per_image[reference][arm][output], cases[reference][arm][output] = images, states
        for arm in native_by_arm:
            if arm == baseline:
                continue
            measurement['paired_changes'][reference][arm] = {}
            for output in ('original', 'two_view'):
                changes = {}
                old, new = cases[reference][baseline][output], cases[reference][arm][output]
                need(len(old) == len(new), 'Paired case count differs')
                for tolerance in (5, 10):
                    gains, losses = [], []
                    for a, b in zip(old, new):
                        need((a['image'], a['reference_index']) == (b['image'], b['reference_index']), 'Paired case order differs')
                        identity = a['image'] + '/R' + str(a['reference_index'] + 1).zfill(3)
                        key = 'within' + str(tolerance)
                        if a['complete_reference'] and not a[key] and b[key]:
                            gains.append(identity)
                        if a['complete_reference'] and a[key] and not b[key]:
                            losses.append(identity)
                    changes[str(tolerance)] = {'gained': len(gains), 'lost': len(losses), 'gained_ids': gains, 'lost_ids': losses}
                measurement['paired_changes'][reference][arm][output] = changes
    details = {'native-predictions': native_by_arm, 'predictions': predictions, 'suppression': suppression,
               'associations': associations, 'per-image': per_image, 'cases': cases}
    return measurement, details


def write_zip(path, members, compressed=True):
    """Write deterministic ZIP bytes; preserve any different existing result."""
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED if compressed else ZIP_STORED) as archive:
        for name, data in sorted(members.items()):
            need(not name.startswith('/') and '..' not in Path(name).parts, 'Unsafe archive member')
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED if compressed else ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    frozen(path, buffer.getvalue())


def package_evidence(work, files, checkpoints, prefix=PREFIX):
    """Package {name: bytes} evidence and {arm: checkpoint Path} in 16 MiB parts.

    Returns the verified return manifest. All small wrappers and one compact
    evidence ZIP are also placed in <prefix>-upload.zip for a single download.
    """
    work = Path(work)
    need(files and checkpoints and 'inventory.json' not in files, 'Invalid package members')
    members = dict(files)
    members['inventory.json'] = encoded({name: {'bytes': len(data), 'sha256': digest(data)} for name, data in members.items()})
    evidence = work / (prefix + '-results.zip')
    write_zip(evidence, members)
    manifest = {'schema': 'paired-model-return-r61-v1', 'status': 'completed', 'models': {},
                'evidence_file': evidence.name, 'evidence_bytes': evidence.stat().st_size, 'evidence_sha256': sha(evidence),
                'instructions': 'Verify each wrapper and member in listed order for each model; concatenate member bytes and verify the complete model hash.'}
    wrapped = {}
    for arm, path in checkpoints.items():
        need(re.fullmatch('[a-zA-Z0-9_-]+', arm) is not None, 'Unsafe model arm name')
        payload = Path(path).read_bytes()
        need(payload, 'Empty checkpoint')
        model_name = arm + '-last.pt'
        descriptor = {'model_file': model_name, 'model_bytes': len(payload), 'model_sha256': digest(payload), 'parts': []}
        for index, begin in enumerate(range(0, len(payload), 16 * 1024 * 1024)):
            chunk = payload[begin:begin + 16 * 1024 * 1024]
            member = model_name + '.part' + str(index).zfill(2)
            target = work / (member + '.zip')
            write_zip(target, {member: chunk})
            descriptor['parts'].append({'file': target.name, 'bytes': target.stat().st_size, 'sha256': sha(target),
                                        'member': member, 'member_bytes': len(chunk), 'member_sha256': digest(chunk)})
            wrapped[target.name] = target.read_bytes()
        manifest['models'][arm] = descriptor
    frozen(work / 'return-manifest.json', encoded(manifest))
    write_zip(work / (prefix + '-upload.zip'), {evidence.name: evidence.read_bytes(),
        'return-manifest.json': encoded(manifest), **wrapped}, compressed=False)
    verify_package(work, prefix)
    return manifest


def verify_package(work, prefix=PREFIX):
    work = Path(work)
    manifest = read(work / 'return-manifest.json')
    need(manifest['status'] == 'completed', 'Package is incomplete')
    evidence = checked(work / manifest['evidence_file'], manifest['evidence_sha256'])
    need(evidence.stat().st_size == manifest['evidence_bytes'], 'Evidence length differs')
    with ZipFile(evidence) as archive:
        names = archive.namelist()
        inventory = json.loads(archive.read('inventory.json'))
        need(len(names) == len(set(names)) and set(names) == set(inventory) | {'inventory.json'} and archive.testzip() is None,
             'Evidence membership or CRC differs')
        for name, descriptor in inventory.items():
            data = archive.read(name)
            need(descriptor == {'bytes': len(data), 'sha256': digest(data)}, 'Evidence member differs: ' + name)
    upload_names = {evidence.name, 'return-manifest.json'}
    for arm, model in manifest['models'].items():
        h, size = hashlib.sha256(), 0
        for part in model['parts']:
            wrapper = checked(work / part['file'], part['sha256'])
            need(wrapper.stat().st_size == part['bytes'], 'Wrapper length differs')
            with ZipFile(wrapper) as archive:
                need(archive.namelist() == [part['member']] and archive.testzip() is None, 'Checkpoint wrapper differs')
                data = archive.read(part['member'])
            need(len(data) == part['member_bytes'] and digest(data) == part['member_sha256'], 'Checkpoint member differs')
            h.update(data)
            size += len(data)
            upload_names.add(part['file'])
        need(size == model['model_bytes'] and h.hexdigest() == model['model_sha256'], 'Assembled checkpoint differs: ' + arm)
    with ZipFile(work / (prefix + '-upload.zip')) as archive:
        need(len(archive.namelist()) == len(upload_names) and set(archive.namelist()) == upload_names and archive.testzip() is None,
             'Upload archive membership differs')
        for name in upload_names:
            need(digest(archive.read(name)) == sha(work / name), 'Upload archive content differs: ' + name)
    return manifest


def training_budget(photographs):
    recipe = get_recipe(RECIPE_ID)
    batch, epochs = recipe['batch'], recipe['epochs']
    nb = math.ceil(photographs / batch)
    warmup = round(min(recipe['warmup_epochs'], max(epochs - 1, 0)) * nb)
    accumulate, last, indices = max(round(recipe['nbs'] / batch), 1), -1, []
    for index in range(nb * epochs):
        if index < warmup:
            accumulate = max(1, round(1 + (recipe['nbs'] / batch - 1) * index / warmup))
        if index - last >= accumulate:
            indices.append(index)
            last = index
    return {'epochs': epochs, 'batch': batch, 'batches_per_epoch': nb, 'batches': nb * epochs,
            'images': photographs * epochs, 'steps': len(indices), 'optimizer_batch_indices': indices}


def count_groups(groups):
    objects = sum(map(len, groups))
    known = sum(p[2] == 2 for group in groups for row in group for p in row['keypoints'])
    return {'images': len(groups), 'objects': objects, 'known_corners': known,
            'unavailable_corners': 2 * objects - known, 'empty_images': sum(not group for group in groups)}


def label_plan(root):
    root = Path(root).resolve()
    source = read(checked(root / 'data/manifest.json', MANIFEST_SHA))
    overlay = read(checked(root / OVERLAY, OVERLAY_SHA))
    need(source['model_sha256'] == INITIAL_SHA and source['model_epoch'] == 173, 'Initializer declaration differs')
    need(recipe_identity(RECIPE_ID)['recipe_sha256'] == RECIPE_SHA, 'Training recipe changed')
    records = source['records']
    train_ids = {r['id'] for r in records if r['split'] == 'train'}
    val_ids = {r['id'] for r in records if r['split'] == 'val'}
    need(len(records) == 157 and len(train_ids) == 137 and len(val_ids) == 20 and not train_ids & val_ids,
         'Original split or unique photograph membership differs')
    need(len({r['image_sha256'] for r in records}) == 157, 'Duplicate original photograph bytes')
    effective = {row['image_id']: row for row in overlay['image_sources']}
    need(set(effective) == train_ids and len(overlay['changes']) == 92, 'Reviewed training overlay membership differs')
    roles, counts, changed = {}, {}, []
    inputs = {'data/manifest.json': MANIFEST_SHA, OVERLAY: OVERLAY_SHA}
    for arm in ARMS:
        roles[arm], groups = [], {'train': [], 'val': []}
        for row in records:
            identity, split = row['id'], row['split']
            need(row['original_size'] == [1280, 720], 'Declared photo dimensions differ')
            relative, expected = 'data/' + row['label'], row['label_sha256']
            checked(root / relative, expected)
            if split == 'train':
                current = effective[identity]
                need(current['source_image_sha256'] == row['image_sha256'], 'Reviewed image provenance differs')
                if arm == 'reviewed_labels':
                    relative, expected = current['effective_label']['path'], current['effective_label']['sha256']
            data = checked(root / relative, expected).read_bytes()
            inputs[relative] = expected
            targets = parse_pose_labels(data.decode(), 1280, 720, identity)
            groups[split].append(targets)
            roles[arm].append({'id': identity, 'split': split,
                'image': 'images/' + split + '/' + identity + '.png', 'label': 'labels/' + split + '/' + identity + '.txt',
                'image_source': 'data/' + row['image'], 'image_sha256': row['image_sha256'],
                'label_source': relative, 'label_sha256': expected})
        counts[arm] = {split: count_groups(values) for split, values in groups.items()}
    for old, new in zip(roles[ARMS[0]], roles[ARMS[1]]):
        need({k: v for k, v in old.items() if k not in ('label_source', 'label_sha256')} ==
             {k: v for k, v in new.items() if k not in ('label_source', 'label_sha256')}, 'A non-label treatment was introduced')
        if old['split'] == 'val':
            need(old == new, 'Trainer-validation labels differ between arms')
        elif old['label_sha256'] != new['label_sha256']:
            before = [[float(x) for x in line.split()] for line in (root / old['label_source']).read_text().splitlines() if line.strip()]
            after = [[float(x) for x in line.split()] for line in (root / new['label_source']).read_text().splitlines() if line.strip()]
            changed.append({'image_id': old['id'], 'original_objects': len(before), 'reviewed_objects': len(after),
                            'numeric_content_changed': before != after,
                            'original_sha256': old['label_sha256'], 'reviewed_sha256': new['label_sha256']})
    need(len(changed) == 92, 'Changed training-label file count differs')
    need(counts['original_labels']['train'] == {'images': 137, 'objects': 3772, 'known_corners': 6839,
         'unavailable_corners': 705, 'empty_images': 13}, 'Original train counts differ')
    need(counts['reviewed_labels']['train'] == {'images': 137, 'objects': 3784, 'known_corners': 6837,
         'unavailable_corners': 731, 'empty_images': 13}, 'Reviewed train counts differ')
    references, provenance = read_references(root, records)
    for descriptor in provenance.values():
        inputs[descriptor['manifest_path']] = descriptor['manifest_sha256']
        for label in descriptor['labels'].values():
            inputs[label['path']] = label['sha256']
    return {'schema': 'controlled-training-label-plan-r61-v1', 'roles': roles, 'counts': counts,
        'source_files': inputs, 'references': provenance, 'primary_reference': 'r50',
        'initial_model': {'path': source['model_checkpoint'], 'sha256': INITIAL_SHA, 'epoch': 173},
        'recipe': recipe_identity(RECIPE_ID), 'recipe_settings': get_recipe(RECIPE_ID),
        'budget_per_arm': training_budget(137), 'changed_training_files': changed,
        'treatment': 'Only the original 137 photographs training labels change: supplied labels versus the frozen effective labels used by the current detector.',
        'additional_training_photographs': 0, 'annotation_values_modified_by_this_experiment': False,
        'trainer_validation_labels_equal': True, 'final_epoch_selection': True,
        'inference_protocol': PROTOCOL, 'development_only': True, 'automatic_promotion': False,
        'interpretation': 'One paired seed estimates this frozen label intervention under the current recipe. It does not estimate the effect of the five additional photographs, every new definition rule, or transfer to a new capture session.'}


def source_identity(root, commit, driver_sha):
    root = Path(root).resolve()
    need(root == ROOT and re.fullmatch('[0-9a-f]{40}', commit or '') and re.fullmatch('[0-9a-f]{64}', driver_sha or ''),
         'Complete immutable source and driver identities are required')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    need(git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain', '--untracked-files=all'), 'Source checkout changed')
    checked(Path(__file__), driver_sha)
    pending, seen = {Path(__file__).name}, set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((root / 'scripts' / name).read_text())):
            modules = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = module.split('.')[0] + '.py'
                if (root / 'scripts' / candidate).is_file() and candidate not in seen:
                    pending.add(candidate)
    files = {}
    for name in sorted(seen):
        path = root / 'scripts' / name
        relative = path.relative_to(root).as_posix()
        expected = git('rev-parse', 'HEAD:' + relative)
        data = path.read_bytes()
        need(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected,
             'Source file differs from published Git object: ' + relative)
        files[relative] = digest(data)
    return {'commit': commit, 'driver_sha256': driver_sha, 'files_sha256': files}


def obtain_assets(root, cache, plan):
    assets = {row['image_source']: row['image_sha256'] for row in plan['roles'][ARMS[0]]}
    assets[plan['initial_model']['path']] = INITIAL_SHA
    def obtain(item):
        relative, expected = item
        path = cache / relative
        if not path.exists():
            url = f'https://media.githubusercontent.com/media/Siuuuers/dl/{ASSET_COMMIT}/{relative}'
            with urllib.request.urlopen(url, timeout=180) as response:
                data = response.read()
            need(digest(data) == expected, 'Downloaded asset differs: ' + relative)
            frozen(path, data)
        checked(path, expected)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(obtain, sorted(assets.items())))


def prepare(root, work, cache, source):
    plan = label_plan(root)
    frozen(work / 'plan.json', encoded(plan))
    frozen(work / 'source.json', encoded(source))
    obtain_assets(root, cache, plan)
    for arm, rows in plan['roles'].items():
        directory = work / 'datasets' / arm
        for row in rows:
            image = checked(cache / row['image_source'], row['image_sha256'])
            header = image.read_bytes()[:24]
            need(header[:8] == b'\x89PNG\r\n\x1a\n' and int.from_bytes(header[16:20], 'big') == 1280
                 and int.from_bytes(header[20:24], 'big') == 720, 'Actual image dimensions differ: ' + row['id'])
            for source_path, relative, expected in ((image, row['image'], row['image_sha256']),
                    (root / row['label_source'], row['label'], row['label_sha256'])):
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source_path, target)
                checked(target, expected)
        frozen(directory / 'data.yaml', encoded({'path': str(directory), 'train': 'images/train', 'val': 'images/val',
            'names': ['cassette_front'], 'kpt_shape': [2, 3], 'flip_idx': [1, 0]}))
    initializer = work / 'initializer/cassette_lower_epoch173.pt'
    initializer.parent.mkdir(parents=True, exist_ok=True)
    if not initializer.exists():
        shutil.copyfile(cache / plan['initial_model']['path'], initializer)
    verify_stage(root, work)
    frozen(work / 'preflight.json', encoded({'status': 'passed', 'source': source, 'plan_sha256': sha(work / 'plan.json'),
        'counts': plan['counts'], 'budget_per_arm': plan['budget_per_arm'], 'changed_training_files': len(plan['changed_training_files']),
        'numeric_changed_training_files': sum(row['numeric_content_changed'] for row in plan['changed_training_files']),
        'images_and_initializer_verified': True, 'training_executed': False, 'fresh_inference_executed': False}))
    print(json.dumps({'preflight': 'passed', 'counts': plan['counts'], 'budget_per_arm': plan['budget_per_arm'],
                      'changed_training_files': len(plan['changed_training_files'])}, indent=2), flush=True)
    return plan


def verify_stage(root, work):
    plan = read(work / 'plan.json')
    need(plan == label_plan(root), 'Saved plan differs from frozen labels or recipe')
    checked(work / 'initializer/cassette_lower_epoch173.pt', INITIAL_SHA)
    for arm, rows in plan['roles'].items():
        directory = work / 'datasets' / arm
        need(read(directory / 'data.yaml') == {'path': str(directory), 'train': 'images/train', 'val': 'images/val',
             'names': ['cassette_front'], 'kpt_shape': [2, 3], 'flip_idx': [1, 0]}, 'Dataset configuration differs')
        for row in rows:
            checked(directory / row['image'], row['image_sha256'])
            checked(directory / row['label'], row['label_sha256'])
        for kind, folder, suffix in (('image', 'images', '.png'), ('label', 'labels', '.txt')):
            actual = {p.relative_to(directory).as_posix() for p in (directory / folder).rglob('*' + suffix)}
            need(actual == {row[kind] for row in rows}, 'Unexpected staged files: ' + arm + '/' + kind)
    return plan


def verify_arm(work, arm, plan, runtime=None, trial=False):
    directory = work / ('resource-trial' if trial else arm)
    receipt = read(directory / 'receipt.json')
    need(receipt['role'] == arm and receipt['trial'] == trial and receipt['plan_sha256'] == sha(work / 'plan.json')
         and receipt['source_sha256'] == sha(work / 'source.json'), 'Training receipt identity differs')
    need(receipt['status'] == ('trial-complete' if trial else 'complete') and receipt['updated'], 'Training did not complete')
    if runtime is not None:
        need(receipt['runtime'] == runtime, 'The two arms must use the same runtime')
    if trial:
        need(receipt['batches'] == receipt['steps'] == 4 and receipt['optimizer_batch_indices'] == [0, 1, 2, 3],
             'Resource trial budget differs')
    else:
        budget = plan['budget_per_arm']
        need(receipt['epochs'] == list(range(1, budget['epochs'] + 1))
             and all(receipt[key] == budget[key] for key in ('batches', 'images', 'steps', 'optimizer_batch_indices')),
             'Completed training budget differs')
        need(len(receipt['epoch_sample_order_sha256']) == budget['epochs'], 'Missing sample-order records')
        need(set(receipt['artifacts']) == {'fit/weights/last.pt', 'fit/results.csv'}, 'Unexpected training artifacts')
        for relative, expected in receipt['artifacts'].items():
            checked(directory / relative, expected)
        if arm == 'reviewed_labels':
            control = verify_arm(work, 'original_labels', plan, runtime)
            need(receipt['initial_state_sha256'] == control['initial_state_sha256'], 'Initial parameter states differ')
            need(receipt['epoch_sample_order_sha256'] == control['epoch_sample_order_sha256'], 'Training sample order differs')
    return receipt


def train(root, work, arm, trial=False):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import img2label_paths
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.utils import SETTINGS
    from rebuild_additional_front_pilot_r45 import runtime as native_runtime
    plan, runtime = verify_stage(root, work), native_runtime()
    need(arm in ARMS and (not trial or arm == 'reviewed_labels'), 'Unknown training role')
    directory, out = work / 'datasets' / arm, work / ('resource-trial' if trial else arm)
    budget = plan['budget_per_arm']
    if (out / 'receipt.json').exists():
        verify_arm(work, arm, plan, runtime, trial)
        print('Reusing verified completed', out.name, flush=True)
        return
    if not trial:
        verify_arm(work, 'reviewed_labels', plan, runtime, trial=True)
        if arm == 'reviewed_labels':
            verify_arm(work, 'original_labels', plan, runtime)
    need(not out.exists(), 'Incomplete run preserved; choose a fresh work directory: ' + str(out))
    out.mkdir(parents=True)
    initializer = checked(work / 'initializer/cassette_lower_epoch173.pt', INITIAL_SHA)
    progress = {'role': arm, 'trial': trial, 'runtime': runtime, 'plan_sha256': sha(work / 'plan.json'),
        'source_sha256': sha(work / 'source.json'), 'status': 'starting', 'epochs': [],
        'batches': 0, 'images': 0, 'steps': 0, 'optimizer_batch_indices': [],
        'epoch_sample_order_sha256': [], 'updated': False}
    def record():
        save(out / 'progress.json', progress)
    class TrialDone(Exception):
        pass
    class Trainer(PoseTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            need(self.world_size in (0, 1) and not self.ddp, 'Use one explicit GPU')
            for event, callback in (('on_train_epoch_start', self.start), ('on_train_batch_end', self.batch_done),
                                    ('on_train_epoch_end', self.epoch_done), ('on_model_save', self.saved)):
                self.add_callback(event, callback)
        def _build_train_pipeline(self):
            super()._build_train_pipeline()
            need(self.batch_size == budget['batch'] and len(self.train_loader) == budget['batches_per_epoch'], 'Actual batch count differs')
            for split, loader in (('train', self.train_loader), ('val', self.test_loader)):
                entries = {row['image']: row for row in plan['roles'][arm] if row['split'] == split}
                seen, cached = set(), loader.dataset.labels
                objects = sum(len(item['cls']) for item in cached)
                known = sum(int((item['keypoints'][:, :, 2] == 2).sum()) for item in cached)
                observed = {'images': len(loader.dataset), 'objects': objects, 'known_corners': known,
                            'unavailable_corners': 2 * objects - known, 'empty_images': sum(not len(item['cls']) for item in cached)}
                need(observed == plan['counts'][arm][split], 'Actual loader counts differ: ' + split)
                need(len(cached) == len(loader.dataset.im_files), 'Loader cache membership differs')
                for image, label, loaded in zip(loader.dataset.im_files, img2label_paths(loader.dataset.im_files), cached):
                    key = Path(image).resolve().relative_to(directory).as_posix()
                    row = entries[key]
                    need(Path(label).resolve() == directory / row['label'], 'Loader label path differs')
                    checked(image, row['image_sha256'])
                    text = checked(label, row['label_sha256']).read_text()
                    expected = np.array([[float(token) for token in line.split()] for line in text.splitlines() if line.strip()], dtype=np.float32).reshape(-1, 11)
                    actual = np.concatenate((loaded['cls'], loaded['bboxes'], loaded['keypoints'].reshape(-1, 6)), axis=1)
                    need(loaded.get('normalized') is True and loaded.get('bbox_format') == 'xywh' and actual.shape == expected.shape
                         and bool(np.allclose(actual, expected, rtol=0, atol=1e-7)), 'Cached coordinates differ: ' + row['id'])
                    seen.add(key)
                need(seen == set(entries), 'Loader photograph membership differs')
            check_training_settings(dict(vars(self.args)), RECIPE_ID, model=initializer, data=directory / 'data.yaml', device='0', project=out)
            progress['settings'] = dict(vars(self.args))
            h = hashlib.sha256()
            for name, tensor in sorted(self.model.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                h.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes())
            progress['initial_state_sha256'] = h.hexdigest()
            record()
        def start(self, trainer):
            self.seen, self.sample_order, self.nb = Counter(), [], 0
        def preprocess_batch(self, batch):
            value = super().preprocess_batch(batch)
            self.ids = [Path(path).stem for path in value['im_file']]
            need(len(self.ids) == value['img'].shape[0], 'Batch photograph count differs')
            return value
        def optimizer_step(self):
            need(bool(torch.isfinite(self.loss).all()), 'Nonfinite training loss')
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            need(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Missing or nonfinite gradients')
            parameter = next((p for group in self.optimizer.param_groups if group['lr'] > 0 for p in group['params']
                              if p.grad is not None and bool(torch.count_nonzero(p.grad))), None)
            need(parameter is not None, 'No nonzero gradient at positive learning rate')
            before = parameter.detach().clone()
            super().optimizer_step()
            need(bool(torch.isfinite(parameter).all()), 'Nonfinite parameter after update')
            progress['updated'] |= bool(torch.any(before != parameter.detach()))
            progress['steps'] += 1
            progress['optimizer_batch_indices'].append(progress['batches'])
        def batch_done(self, trainer):
            need(bool(torch.isfinite(self.loss).all()), 'Nonfinite batch loss')
            self.seen.update(self.ids)
            self.sample_order.append(self.ids)
            self.nb += 1
            progress['batches'] += 1
            progress['images'] += len(self.ids)
            record()
            if trial and progress['batches'] == 4:
                need(progress['updated'] and progress['steps'] == 4, 'Trial did not update parameters')
                raise TrialDone()
        def epoch_done(self, trainer):
            need(self.nb == budget['batches_per_epoch'] and self.seen == Counter(row['id'] for row in plan['roles'][arm] if row['split'] == 'train'),
                 'Per-epoch image exposure differs')
            progress['epoch_sample_order_sha256'].append(digest(encoded(self.sample_order)))
        def saved(self, trainer):
            need(self.last.is_file() and self.epoch + 1 == len(progress['epochs']) + 1, 'Saved epoch sequence differs')
            progress['epochs'].append(self.epoch + 1)
            record()
    SETTINGS.update({name: False for name in ('sync', 'clearml', 'comet', 'dvc', 'mlflow', 'raytune', 'tensorboard', 'wandb')})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = YOLO(str(initializer))
    check_loaded_pose_model(model)
    try:
        model.train(trainer=Trainer, **get_recipe(RECIPE_ID), data=str(directory / 'data.yaml'), device='0', workers=0,
                    plots=False, verbose=False, project=str(out), name='fit', exist_ok=False)
        need(not trial, 'Trial did not stop at its fixed budget')
        need(progress['epochs'] == list(range(1, budget['epochs'] + 1)) and progress['updated']
             and all(progress[key] == budget[key] for key in ('batches', 'images', 'steps', 'optimizer_batch_indices')),
             'Completed training budget differs')
        with (out / 'fit/results.csv').open() as stream:
            need([int(row['epoch']) for row in csv.DictReader(stream)] == list(range(1, 31)), 'Training CSV epochs differ')
        verify_stage(root, work)
        progress['artifacts'] = {name: sha(out / name) for name in ('fit/weights/last.pt', 'fit/results.csv')}
        progress['status'] = 'complete'
    except TrialDone:
        progress['status'] = 'trial-complete'
    except Exception as error:
        progress['status'], progress['error'] = 'failed', str(error)
        record()
        raise
    record()
    save(out / 'receipt.json', progress)
    verify_arm(work, arm, plan, runtime, trial)
    print(arm, progress['status'], flush=True)


def evaluate(root, work):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from rebuild_additional_front_pilot_r45 import runtime as native_runtime
    plan, runtime = verify_stage(root, work), native_runtime()
    receipts = {arm: verify_arm(work, arm, plan, runtime) for arm in ARMS}
    references, _ = read_references(root, read(root / 'data/manifest.json')['records'])
    native = {}
    for arm in ARMS:
        native[arm] = {'original': {}, 'reflected': {}}
        model = YOLO(str(work / arm / 'fit/weights/last.pt'))
        check_loaded_pose_model(model)
        for row in plan['roles'][arm]:
            if row['split'] != 'val':
                continue
            image = cv2.imread(str(work / 'datasets' / arm / row['image']))
            need(image is not None and image.shape == (720, 1280, 3), 'Evaluation image decoding differs')
            for view, array in (('original', image), ('reflected', np.ascontiguousarray(image[:, ::-1, :]))):
                result = model.predict(source=array, device='0', conf=.25, **INFERENCE)[0]
                need(not model.predictor.model.end2end and not model.predictor.model.fp16, 'Expected ordinary NMS in FP32')
                need(tuple(result.orig_shape) == (720, 1280), 'Prediction dimensions differ')
                native[arm][view][row['id']] = _prediction_records(result)
        del model
        torch.cuda.empty_cache()
    measurement, details = score_native(references, native, 'original_labels')
    summary = {'schema': 'controlled-training-label-result-r61-v1', 'status': 'completed',
        'training': True, 'fresh_inference': True, 'automatic_promotion': False, 'development_only': True,
        'plan_sha256': sha(work / 'plan.json'), 'source_sha256': sha(work / 'source.json'),
        'runtime': runtime, 'model_sha256': {arm: receipts[arm]['artifacts']['fit/weights/last.pt'] for arm in ARMS},
        'measurement': measurement, 'interpretation': plan['interpretation']}
    files = {'summary.json': encoded(summary), 'plan.json': (work / 'plan.json').read_bytes(),
             'source.json': (work / 'source.json').read_bytes(), 'preflight.json': (work / 'preflight.json').read_bytes()}
    for name, value in details.items():
        files[name + '.json.gz'] = gzip.compress(encoded(value), mtime=0)
    for relative in plan['source_files']:
        files['source/' + relative] = (root / relative).read_bytes()
    for relative in read(work / 'source.json')['files_sha256']:
        files['source/' + relative] = (root / relative).read_bytes()
    for arm in ('resource-trial', *ARMS):
        for relative in ('receipt.json', 'progress.json') + (() if arm == 'resource-trial' else ('fit/results.csv',)):
            files['training/' + arm + '/' + relative] = (work / arm / relative).read_bytes()
    frozen(work / 'summary.json', encoded(summary))
    package_evidence(work, files, {arm: work / arm / 'fit/weights/last.pt' for arm in ARMS})
    return verify_result(root, work)


def verify_result(root, work):
    plan = verify_stage(root, work)
    manifest = verify_package(work)
    references, _ = read_references(root, read(root / 'data/manifest.json')['records'])
    with ZipFile(work / manifest['evidence_file']) as archive:
        summary = json.loads(archive.read('summary.json'))
        need(summary == read(work / 'summary.json') and summary['plan_sha256'] == sha(work / 'plan.json')
             and summary['source_sha256'] == sha(work / 'source.json'), 'Saved result identity differs')
        need(archive.read('plan.json') == (work / 'plan.json').read_bytes()
             and archive.read('source.json') == (work / 'source.json').read_bytes(), 'Packaged plan/source differs')
        native = json.loads(gzip.decompress(archive.read('native-predictions.json.gz')))
        measurement, details = score_native(references, native, 'original_labels')
        need(summary['measurement'] == measurement, 'Saved score replay differs')
        for name, value in details.items():
            need(json.loads(gzip.decompress(archive.read(name + '.json.gz'))) == value, 'Saved evidence replay differs: ' + name)
        for arm in ARMS:
            receipt = verify_arm(work, arm, plan, summary['runtime'])
            need(receipt['artifacts']['fit/weights/last.pt'] == summary['model_sha256'][arm]
                 == manifest['models'][arm]['model_sha256'], 'Returned model identity differs')
            need(json.loads(archive.read('training/' + arm + '/receipt.json')) == receipt, 'Packaged training receipt differs')
        for relative, expected in {**plan['source_files'], **read(work / 'source.json')['files_sha256']}.items():
            need(digest(archive.read('source/' + relative)) == expected, 'Packaged source/label differs: ' + relative)
    receipt = {'status': 'passed', 'models': 2, 'references': 3, 'outputs_per_model': 2,
               'all_saved_scores_replayed': True, 'training_label_only_plan_verified': True,
               'equal_initial_state_and_sample_order': True, 'archive_sha256': sha(work / (PREFIX + '-upload.zip'))}
    frozen(work / 'verification.json', encoded(receipt))
    print(json.dumps(receipt, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'prepare', 'trial', 'train', 'evaluate', 'verify'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    args.root, args.work, args.cache_dir = args.root.resolve(), args.work.resolve(), args.cache_dir.resolve()
    need(not args.work.is_relative_to(args.root) and not args.cache_dir.is_relative_to(args.root)
         and not args.root.is_relative_to(args.work) and not args.root.is_relative_to(args.cache_dir), 'Keep source, work and cache separate')
    need(args.work != args.cache_dir and not args.work.is_relative_to(args.cache_dir)
         and not args.cache_dir.is_relative_to(args.work), 'Keep work and cache separate')
    source = source_identity(args.root, args.source_commit, args.script_sha256)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.action == 'plan':
        plan = label_plan(args.root)
        frozen(args.work / 'plan.json', encoded(plan))
        frozen(args.work / 'source.json', encoded(source))
        print(json.dumps({'status': 'labels_verified', 'counts': plan['counts'], 'budget_per_arm': plan['budget_per_arm'],
              'changed_training_files': len(plan['changed_training_files']), 'assets_downloaded': False,
              'training_executed': False}, indent=2))
        return
    if args.action == 'prepare':
        prepare(args.root, args.work, args.cache_dir, source)
        return
    need(read(args.work / 'source.json') == source, 'Saved source identity differs')
    if args.action in ('trial', 'train'):
        need(args.action == 'trial' or args.arm is not None, 'Choose a training arm')
        train(args.root, args.work, 'reviewed_labels' if args.action == 'trial' else args.arm, args.action == 'trial')
    elif args.action == 'evaluate':
        if (args.work / 'return-manifest.json').exists():
            verify_result(args.root, args.work)
        else:
            evaluate(args.root, args.work)
    else:
        verify_result(args.root, args.work)


if __name__ == '__main__':
    main()
