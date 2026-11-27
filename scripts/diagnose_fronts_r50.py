#!/usr/bin/env python3
"""R50: CPU input audit, then PR137 same-tensor hard/linear-Soft-NMS diagnosis.

No training, confidence sweep, new images or automatic promotion. The primary
frozen666 score stays separate from historical663, R38 and the R50 proposal.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile

sys.dont_write_bytecode = True
CONF, OVERLAP, MATCH, CAP = .13, .70, .50, 300
ULTRALYTICS = '8.4.182'
RESULT_NAME = 'baomai-r50-diagnosis-results.zip'
R50_REFERENCE = 'data/reviews/fronts-2026-11-27-r50/manifest.json'
R50_SHA = 'f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f'
R48_REFERENCE = 'data/reviews/fronts-2026-11-27-r48/manifest.json'
R48_SHA = '21838bd979a03f41a82ea17496f29bb4d4bd6357c94064112a91124a5a4723e4'
ROW_MAP_PATH = 'results/review/front-review-20261126-r38/row-maps.json'
ROW_MAP_SHA = '56e7662a1facbeedcdfe29210c263c4bd1b75ce8b0149e5317138a942667dc9d'
ORIGINAL_MANIFEST_SHA = '72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926'
OVERLAY_PATH = 'data/candidates/front-review-2026-11-26-r37/manifest.json'
OVERLAY_SHA = 'ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3'
LEGACY_SCRIPT_SHA = '188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811'
VIEWS = ('saved_raw', 'saved', 'hard_raw', 'hard', 'soft_raw', 'soft')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def encoded(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def save(path, obj):
    path.write_bytes(encoded(obj))


def checked(path, expected):
    need(path.is_file(), 'Required file missing: ' + str(path))
    data = path.read_bytes()
    need(sha(data) == expected, 'File hash differs; preserve this directory and choose a fresh base: ' + str(path))
    return data


def put(path, data):
    """Populate an absent immutable input; preserve incompatible existing data."""
    if path.exists():
        need(path.is_file() and path.read_bytes() == data, 'Existing file differs; preserved: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.part')
        tmp.write_bytes(data)
        tmp.replace(path)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def source_check(root, revision, script_sha, reference_sha):
    """Verify the exact clean checkout and source/data closure without GPU imports."""
    need(re.fullmatch('[0-9a-f]{40}', revision) is not None, 'SOURCE_REVISION is not pinned; publication is incomplete')
    need(re.fullmatch('[0-9a-f]{64}', script_sha) is not None, 'SCRIPT_SHA256 is not pinned')
    need(reference_sha == R50_SHA, 'R50 reference pin differs')
    need(git(root, 'rev-parse', 'HEAD') == revision, 'Wrong source revision; existing checkout preserved. Choose a fresh base directory.')
    need(not git(root, 'status', '--porcelain', '--untracked-files=all'), 'Source checkout has changes; preserve it and choose a fresh base directory.')
    checked(root / 'scripts/diagnose_fronts_r50.py', script_sha)
    need(Path(__file__).resolve() == (root / 'scripts/diagnose_fronts_r50.py').resolve(), 'Run the script from the pinned source checkout')
    blobs = {}
    for line in git(root, 'ls-tree', '-r', 'HEAD').splitlines():
        meta, relative = line.split('\t', 1)
        mode, kind, object_id = meta.split()
        if relative.startswith('scripts/') and relative.endswith('.py'):
            need(kind == 'blob' and mode in ('100644', '100755'), 'Unexpected source file type: ' + relative)
            path = root / relative
            need(path.is_file() and not path.is_symlink(), 'Missing local source: ' + relative)
            data = path.read_bytes()
            git_digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            need(git_digest == object_id, 'Source differs from pinned Git object: ' + relative)
            blobs[relative] = sha(data)
    need('scripts/diagnose_fronts_r50.py' in blobs, 'R50 script is not tracked by pinned source')
    # These imports are pure Python. All repository Python files were verified above,
    # including lazy imports that are otherwise missed by existence-only closure scans.
    from audit_additional_front_pilot_r47_results import REFERENCES, REVIEW, RELEASE_SHA
    from evaluate_pose import INFERENCE, parse_pose_labels
    files = dict(blobs)
    def read(relative, expected):
        data = checked(root / relative, expected)
        files[relative] = expected
        return json.loads(data)
    original = read('data/manifest.json', ORIGINAL_MANIFEST_SHA)
    all_records = original['records']
    need(len(all_records) == len({r['id'] for r in all_records}) == 157, 'Original image inventory differs')
    records = {r['id']: r for r in all_records if r['split'] == 'val'}
    need(len(records) == 20, 'Expected exactly 20 development scenes')
    for r in all_records:
        relative = 'data/' + r['label']
        checked(root / relative, r['label_sha256']); files[relative] = r['label_sha256']
    overlay = read(OVERLAY_PATH, OVERLAY_SHA)
    for r in overlay['changes']:
        relative = str(Path(OVERLAY_PATH).parent / r['candidate_label'])
        checked(root / relative, r['candidate_label_sha256']); files[relative] = r['candidate_label_sha256']
    release = read(REVIEW + '/release-status.json', RELEASE_SHA)
    # The generated R47 manifest lives inside the immutable return archive;
    # audit() verifies it there. These are the release files present in source.
    for name, expected in {**release['files'], 'audit-ledger.json': release['audit_sha256'],
                           'outlines.json': release['outlines_sha256']}.items():
        relative = REVIEW + '/' + name
        checked(root / relative, expected); files[relative] = expected
    refs, descriptors = {}, dict(REFERENCES)
    descriptors['r50'] = (R50_REFERENCE, reference_sha, 654)
    descriptors['r48_parent'] = (R48_REFERENCE, R48_SHA, 654)
    reference_contract = {}
    for name, (relative, expected, count) in descriptors.items():
        manifest = read(relative, expected)
        selected = [r for r in manifest['files'] if r.get('split', 'val') == 'val']
        need(len(selected) == 20 and {r['image_id'] for r in selected} == set(records), 'Reference membership differs: ' + name)
        gold, labels = {}, {}
        for r in selected:
            identity = r['image_id']
            need(r.get('image_sha256', r.get('source_image_sha256')) == records[identity]['image_sha256'], 'Reference image differs: ' + identity)
            label = str(Path(relative).parent / r['label_path'])
            data = checked(root / label, r['label_sha256']); files[label] = r['label_sha256']
            gold[identity] = parse_pose_labels(data.decode(), 1280, 720, name + '/' + identity)
            labels[identity] = {'path': label, 'sha256': r['label_sha256']}
        need(sum(map(len, gold.values())) == count, 'Reference object count differs: ' + name)
        refs[name] = gold
        reference_contract[name] = {'manifest_path': relative, 'manifest_sha256': expected, 'objects': count, 'labels': labels}
    maps = read(ROW_MAP_PATH, ROW_MAP_SHA)
    r38 = json.loads((root / REFERENCES['r38'][0]).read_text())
    need(r38['row_maps'] == {'path': ROW_MAP_PATH, 'sha256': ROW_MAP_SHA}, 'R38 row-map declaration differs')
    for identity, gold in refs['r38'].items():
        rows = maps[identity]['rows']
        need(len(rows) == len(gold) and sorted(r['current_row'] for r in rows) == list(range(1, len(gold) + 1)), 'Row-map inventory differs: ' + identity)
        need(len(gold) == len(refs['r50'][identity]), 'R50 stable row membership differs')
    r50 = json.loads((root / R50_REFERENCE).read_text())
    need(r50['parent'] == {'manifest_path': R48_REFERENCE, 'manifest_sha256': R48_SHA} and r50['development_only'] and not r50['changed_training_rows'], 'R50 reference lineage differs')
    changes = []
    for identity in refs['r50']:
        before, after = refs['r48_parent'][identity], refs['r50'][identity]
        need(len(before) == len(after), 'R50 row membership changed')
        changes.extend((identity, j + 1) for j, (a, b) in enumerate(zip(before, after)) if a != b)
    need(changes == [('new_000010', 29)], 'R50 treatment differs from its one-row proposal')
    before, after = refs['r48_parent']['new_000010'][28], refs['r50']['new_000010'][28]
    need(before['keypoints'] == after['keypoints'] and before['class_id'] == after['class_id'] and
         all(abs(before['bbox_xyxy'][k] - after['bbox_xyxy'][k]) < .00001 for k in range(3)) and
         abs(after['bbox_xyxy'][3] - 714) < .00001, 'R50 amendment includes more than physical bottom ymax=714')
    del refs['r48_parent']
    contract = {'schema': 'baomai-r50-diagnosis-inputs-v1', 'source': revision, 'script_sha256': script_sha,
                'legacy_r49_script_sha256': LEGACY_SCRIPT_SHA, 'source_files_sha256': dict(sorted(files.items())),
                'references': reference_contract, 'row_maps': {'path': ROW_MAP_PATH, 'sha256': ROW_MAP_SHA},
                'images': {i: r['image_sha256'] for i, r in sorted(records.items())},
                'settings': {**INFERENCE, 'conf': CONF, 'matching_iou': MATCH, 'endpoint_conf': .5,
                    'soft_nms': 'linear, IoU > .70, score *= 1-IoU, strict conf > .13, max_det 300, same postclip'},
                'primary_reference': 'frozen666', 'proposal_reference': 'r50', 'training': False, 'automatic_promotion': False}
    print('Source preflight passed:', revision, len(files), 'source/data files; row-map hash verified', flush=True)
    return {'contract': contract, 'records': records, 'refs': refs, 'maps': maps}


def prepare(args, context):
    """Restore and audit original pixels, the exact archive and checkpoint on CPU."""
    from audit_additional_front_pilot_r47_results import obtain_archive, audit, ARCHIVE_SHA, ARCHIVE_BYTES, MODEL_HASHES, INPUT_SOURCE
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    cache = (args.cache_dir or work / 'cache').resolve(); cache.mkdir(parents=True, exist_ok=True)
    originals = (args.originals_dir or cache / 'originals').resolve(); originals.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve() if args.archive else obtain_archive(cache)
    archive_data = checked(archive, ARCHIVE_SHA)
    need(len(archive_data) == ARCHIVE_BYTES, 'PR137 archive size differs')
    receipt = audit(archive, args.root)
    need(receipt['status'] == 'passed', 'PR137 CPU audit failed')
    receipt = json.loads(encoded(receipt))  # The inherited audit contains tuple counts.
    with zipfile.ZipFile(archive) as z:
        prior_data = z.read('evaluation/predictions.json')
        summary_data = z.read('evaluation/summary.json')
        prior = json.loads(prior_data)['candidate']
        old_summary = json.loads(summary_data)
        weight = z.read('candidate/fit/weights/last.pt')
    need(set(prior) == set(context['records']), 'Saved prediction membership differs')
    need(sha(weight) == MODEL_HASHES['candidate'], 'Wrong PR137 candidate checkpoint')
    weights = cache / 'pr137-candidate.pt'; put(weights, weight)
    for identity, record in sorted(context['records'].items()):
        path = originals / (identity + '.png')
        if not path.exists():
            url = f'https://media.githubusercontent.com/media/Siuuuers/dl/{INPUT_SOURCE}/data/{record["image"]}'
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            need(sha(data) == record['image_sha256'], 'Original download differs: ' + identity)
            put(path, data)
        data = checked(path, record['image_sha256'])
        need(data[:8] == b'\x89PNG\r\n\x1a\n' and data[12:16] == b'IHDR' and struct.unpack('>II', data[16:24]) == (1280, 720), 'Original dimensions differ: ' + identity)
    context['contract'].update({'archive_sha256': ARCHIVE_SHA, 'archive_bytes': ARCHIVE_BYTES,
        'checkpoint_sha256': MODEL_HASHES['candidate'], 'saved_predictions_sha256': sha(prior_data),
        'saved_summary_sha256': sha(summary_data), 'original_source': INPUT_SOURCE})
    contract = context['contract']
    receipt_path = work / 'preflight.json'
    if not receipt_path.exists():
        need(not any((work / n).exists() for n in ('summary.json', 'predictions.json', 'decoded', RESULT_NAME)),
             'Outputs exist without an input contract; preserved. Choose a fresh work directory.')
    preflight = {'contract': contract, 'contract_sha256': sha(encoded(contract)), 'input_audit': receipt,
                 'ready_for_inference': True, 'fresh_inference': False, 'fresh_training': False}
    put(receipt_path, encoded(preflight))
    context.update({'prior': prior, 'old_summary': old_summary, 'receipt': receipt, 'preflight': preflight,
                    'weights': weights, 'originals': originals, 'work': work, 'root': args.root})
    print('Data preflight passed: PR137 archive + checkpoint, 20 original PNGs, four score references; no GPU imports.', flush=True)
    return context


def score_reports(refs, outputs, old_summary):
    from audit_additional_front_pilot_r45_results import corner_groups
    from pose_scoring import score_image, aggregate_scores
    reports = {}
    for name, gold in refs.items():
        reports[name] = {}
        for view in VIEWS:
            scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
            for identity, targets in gold.items():
                pred = outputs[identity]['saved']['raw' if view == 'saved_raw' else 'postclip'] if view.startswith('saved') else outputs[identity][view]
                scores[identity] = score_image(targets, pred, 1280, 720, MATCH, .5)
                for key, values in corner_groups(targets, pred, scores[identity]).items(): groups[key].update(values)
            report = {'detection': aggregate_scores(list(scores.values())), 'corners_by_locatable_count': groups, 'per_image': scores}
            if view.startswith('saved') and name != 'r50':
                need(report == old_summary['references'][name]['raw' if view == 'saved_raw' else 'postclip']['candidate'], 'Saved replay differs: ' + name + '/' + view)
            reports[name][view] = report
        a, b = [reports[name][k]['per_image'] for k in ('hard', 'soft')]
        reports[name]['paired_delta'] = {i: {'recovered_rows': [j+1 for j in sorted(set(a[i]['unmatched_gold_indices']) - set(b[i]['unmatched_gold_indices']))],
             'newly_missed_rows': [j+1 for j in sorted(set(b[i]['unmatched_gold_indices']) - set(a[i]['unmatched_gold_indices']))]} for i in gold}
    return reports


def verify_return(target, context):
    """Verify membership, immutable identities and every returned score without a GPU."""
    from predict_pose_folder import postclip_nms
    images = set(context['records'])
    expected_names = {'summary.json', 'predictions.json', 'trace.json', 'missed-cases.json', 'timing.json',
                      'input-audit.json', 'preflight.json', 'diagnose_fronts_r50.py'} | {'decoded/' + i + '.npy' for i in images}
    with zipfile.ZipFile(target) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == expected_names | {'inventory.json'} and z.testzip() is None, 'Return membership/CRC differs; existing archive preserved')
        inventory = json.loads(z.read('inventory.json'))
        need(set(inventory) == expected_names, 'Return inventory differs')
        for name, item in inventory.items():
            data = z.read(name)
            need(item == {'bytes': len(data), 'sha256': sha(data)}, 'Return member differs: ' + name)
        need(z.read('diagnose_fronts_r50.py') == Path(__file__).read_bytes(), 'Returned script differs')
        need(json.loads(z.read('preflight.json')) == context['preflight'], 'Returned input contract differs')
        need(json.loads(z.read('input-audit.json')) == context['receipt'], 'Returned PR137 audit differs')
        summary = json.loads(z.read('summary.json'))
        need(summary['contract'] == context['contract'] and summary['contract_sha256'] == sha(encoded(context['contract'])), 'Returned source/cache/reference identities differ')
        need(summary['same_tensor_comparison'] and summary['fresh_inference'] and not summary['training'] and not summary['automatic_promotion'], 'Returned experiment declaration differs')
        outputs = json.loads(z.read('predictions.json')); traces = json.loads(z.read('trace.json'))
        need(set(outputs) == set(traces) == images, 'Returned prediction membership differs')
        for identity in sorted(images):
            row = outputs[identity]
            need(row['saved'] == context['prior'][identity], 'Returned saved cache differs: ' + identity)
            for arm in ('hard', 'soft'):
                kept, audit = postclip_nms(row[arm + '_raw'])
                need(kept == row[arm] and audit == row[arm + '_audit'], 'Returned postclip replay differs: ' + identity)
            need(traces[identity]['native_reconstruction_exact'] and traces[identity]['image_sha256'] == context['records'][identity]['image_sha256'], 'Returned trace differs')
            # NPY payload hash and shape are bound to the trace without importing NumPy.
            data = z.read('decoded/' + identity + '.npy')
            import ast
            need(data[:8] == b'\x93NUMPY\x01\x00', 'Unexpected decoded tensor format')
            length = struct.unpack('<H', data[8:10])[0]
            header = ast.literal_eval(data[10:10+length].decode('latin1').strip())
            need(header['descr'] == '<f4' and not header['fortran_order'] and list(header['shape']) == traces[identity]['shape'] and tuple(header['shape'][:2]) == (1, 11), 'Returned decoded tensor shape/type differs')
            payload = data[10+length:]
            need(len(payload) == 4 * 11 * header['shape'][2] and sha(payload) == traces[identity]['raw_tensor_sha256'], 'Returned decoded tensor content differs')
        need(summary['reports'] == score_reports(context['refs'], outputs, context['old_summary']), 'Returned score replay differs')
    print('Verified completed R50 return; preserving and reusing:', target, 'SHA256:', sha(target.read_bytes()), flush=True)
    return summary


def soft_select(boxes, confidence):
    """One fixed linear Soft-NMS: IoU>.70 -> score*(1-IoU), strict .13 gate."""
    import numpy as np
    scores = np.asarray(confidence, dtype=np.float64).copy()
    boxes = np.asarray(boxes, dtype=np.float64)
    need(boxes.shape == (len(scores), 4) and np.isfinite(boxes).all()
         and np.isfinite(scores).all(), 'Invalid decoded inputs')
    active = np.flatnonzero(scores > CONF)
    need(len(active) <= 30000, 'Native max_nms cap exceeded; pilot stopped')
    chosen, values = [], []
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    while len(active) and len(chosen) < CAP:
        pos = int(np.argmax(scores[active]))
        i = int(active[pos])
        chosen.append(i); values.append(float(scores[i]))
        active = np.delete(active, pos)
        intersection = np.prod(np.maximum(np.minimum(boxes[active, 2:], boxes[i, 2:]) -
                                          np.maximum(boxes[active, :2], boxes[i, :2]), 0), axis=1)
        union = area[i] + area[active] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        scores[active] *= np.where(iou > OVERLAP, 1 - iou, 1)
        active = active[scores[active] > CONF]
    return chosen, values, bool(len(active))


def stage(eligible, scores, ordinary, final, capped):
    above = {i for i in eligible if scores[i] > CONF}
    if not eligible:
        return 'no_iou_eligible_decoded_box'
    if not above:
        return 'confidence_gate'
    if not above.intersection(ordinary):
        return 'ordinary_nms_or_cap' if capped else 'ordinary_nms'
    if not above.intersection(final):
        return 'postclip_suppression'
    return 'retained_assignment'


def selftest():
    import numpy as np
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 10, 8]])
    chosen, scores, capped = soft_select(boxes, [.9, .8, .7, .75])
    need(chosen == [0, 2, 3] and abs(scores[2] - .15) < 1e-10 and not capped, 'Soft-NMS synthetic test failed')
    need(soft_select(boxes[:2], [CONF, .12])[0] == [], 'Strict score gate failed')
    need(stage([], [.8], set(), set(), False) == 'no_iou_eligible_decoded_box', 'Stage test')
    need(stage([0], [.1], set(), set(), False) == 'confidence_gate', 'Stage test')
    need(stage([0], [.8], set(), set(), False) == 'ordinary_nms', 'Stage test')
    need(stage([0], [.8], set(), set(), True) == 'ordinary_nms_or_cap', 'Stage test')
    need(stage([0], [.8], {0}, set(), False) == 'postclip_suppression', 'Stage test')
    need(stage([0], [.8], {0}, {0}, False) == 'retained_assignment', 'Stage test')
    print('Synthetic selection and stage checks passed')


def run(context, device):
    work, weights = context['work'], context['weights']
    target = work / RESULT_NAME
    if target.exists():
        return verify_return(target, context)
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.utils import nms, ops
    from evaluate_pose import INFERENCE, _prediction_records
    from pose_checkpoint import check_loaded_pose_model
    from pose_scoring import score_image, _box_iou
    from predict_pose_folder import postclip_nms
    need(md.version('ultralytics') == ULTRALYTICS, 'Use pinned Ultralytics')
    selftest()
    records, refs, maps, prior = [context[k] for k in ('records', 'refs', 'maps', 'prior')]
    originals_dir = context['originals']

    class TracePredictor(PosePredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            self.trace = (raw.detach().clone(), img, orig_imgs)
            return super().postprocess(preds, img, orig_imgs, **kwargs)

    torch.set_num_threads(4)
    model = YOLO(str(weights)); check_loaded_pose_model(model)
    model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), predictor=TracePredictor, device=device, conf=CONF, **INFERENCE)
    predictor = model.predictor
    need(not predictor.model.end2end and not predictor.model.fp16, 'Expected ordinary one-to-many FP32 inference')
    def sync():
        if device != 'cpu': torch.cuda.synchronize()
    outputs, traces, cases, runtime_rows = {}, {}, [], []
    raw_tensors = work / 'decoded'; raw_tensors.mkdir(exist_ok=True)
    for i, record in sorted(records.items()):
        path = originals_dir / (i + '.png')
        result = model.predict(source=str(path), device=device, conf=CONF, **INFERENCE)[0]
        need(tuple(result.orig_shape) == (720, 1280), 'Prediction dimensions differ')
        primary = _prediction_records(result)
        raw, image_tensor, originals = predictor.trace
        need(raw.ndim == 3 and tuple(raw.shape[:2]) == (1, 11) and raw.shape[2] > 0
             and raw.dtype == torch.float32, 'Expected a nonempty FP32 (1, 11, N) decoded tensor')
        raw_np = np.ascontiguousarray(raw.cpu().numpy())
        need(np.isfinite(raw_np).all(), 'Nonfinite decoded tensor')
        np.save(raw_tensors / (i + '.npy'), raw_np, allow_pickle=False)
        input_boxes = ops.xywh2xyxy(raw[0, :4].T.clone())
        scaled = ops.scale_boxes(image_tensor.shape[2:], input_boxes.clone(), result.orig_img.shape).cpu().tolist()
        confidences = raw[0, 4].cpu().tolist()
        sync(); t = time.perf_counter()
        detections, indices = nms.non_max_suppression(raw.clone(), conf_thres=CONF, iou_thres=OVERLAP,
            classes=None, agnostic=False, max_det=CAP, nc=1, end2end=False, rotated=False, return_idxs=True)
        keep = [int(j) for j in indices[0].tolist()]
        ordinary = _prediction_records(predictor.construct_results([d.clone() for d in detections], image_tensor, originals)[0])
        need(ordinary == primary, 'Native NMS index reconstruction differs: ' + i)
        hard, hard_audit = postclip_nms(ordinary)
        sync(); hard_ms = (time.perf_counter() - t) * 1000
        hard_keep = [keep[j-1] for j in hard_audit['retained_primary_prediction_ids']]
        sync(); t = time.perf_counter()
        soft_ids, values, soft_capped = soft_select(input_boxes.cpu().numpy(), confidences)
        ix = torch.tensor(soft_ids, dtype=torch.long, device=raw.device)
        value_tensor = torch.tensor(values, dtype=raw.dtype, device=raw.device).reshape(-1, 1)
        cls = torch.zeros_like(value_tensor)
        rows = torch.cat([input_boxes[ix], value_tensor, cls, raw[0, 5:, :].T[ix]], dim=1)
        soft_raw = _prediction_records(predictor.construct_results([rows.clone()], image_tensor, originals)[0])
        soft, soft_audit = postclip_nms(soft_raw)
        sync(); soft_ms = (time.perf_counter() - t) * 1000
        soft_keep = [soft_ids[j-1] for j in soft_audit['retained_primary_prediction_ids']]
        outputs[i] = {'saved': prior[i], 'hard_raw': ordinary, 'hard': hard, 'soft_raw': soft_raw, 'soft': soft,
                      'hard_audit': hard_audit, 'soft_audit': soft_audit, 'ordinary_indices': keep,
                      'hard_indices': hard_keep, 'soft_raw_indices': soft_ids, 'soft_indices': soft_keep}
        for reference_name in ('r38', 'r50'):
            gold = refs[reference_name][i]
            score_old = score_image(gold, prior[i]['postclip'], 1280, 720, MATCH, .5)
            score_hard = score_image(gold, hard, 1280, 720, MATCH, .5)
            score_soft = score_image(gold, soft, 1280, 720, MATCH, .5)
            old_miss, hard_miss, soft_miss = [set(s['unmatched_gold_indices']) for s in (score_old, score_hard, score_soft)]
            above_count = sum(c > CONF for c in confidences)
            capped = above_count > 30000 or len(keep) == CAP
            rowmap = {r['current_row']: r for r in maps[i]['rows']}
            for j in sorted(old_miss | hard_miss):
                g = gold[j]
                overlap = [_box_iou(b, g['bbox_xyxy']) for b in scaled]
                eligible = [k for k, v in enumerate(overlap) if v >= MATCH]
                def item(k):
                    return {'anchor_index': k, 'bbox_xyxy': scaled[k], 'iou': overlap[k], 'confidence': confidences[k],
                            'ordinary': k in keep, 'postclip': k in hard_keep, 'soft_postclip': k in soft_keep}
                best = max(range(len(overlap)), key=lambda k: (overlap[k], confidences[k], -k))
                cases.append({'reference': reference_name, 'image': i, 'row': j + 1, 'r38_row': j + 1, 'stable_id': rowmap[j+1]['stable_id'], 'gold': g,
                              'saved_miss': j in old_miss, 'fresh_hard_miss': j in hard_miss, 'soft_miss': j in soft_miss,
                              'locatable_corners': sum(k[2] == 2 for k in g['keypoints']),
                              'mechanism': stage(eligible, confidences, set(keep), set(hard_keep), capped) if j in hard_miss else 'fresh_baseline_matched',
                              'best_decoded': item(best), 'eligible': [item(k) for k in eligible], 'output_cap_possible': capped})
        traces[i] = {'raw_tensor_sha256': sha(raw_np.tobytes()), 'shape': list(raw_np.shape), 'input_shape': list(image_tensor.shape),
                     'image_sha256': record['image_sha256'], 'above_confidence': above_count, 'soft_cap_reached': soft_capped,
                     'native_reconstruction_exact': True, 'saved_raw_exact': ordinary == prior[i]['raw'],
                     'saved_postclip_exact': hard == prior[i]['postclip']}
        runtime_rows.append({'image': i, 'hard_suppression_and_construct_ms': hard_ms, 'soft_suppression_and_construct_ms': soft_ms})
        print(i, 'R50 proposal fresh hard/soft misses:', len(hard_miss), len(soft_miss), 'unmatched:', score_hard['fp'], score_soft['fp'], flush=True)
    reports = score_reports(refs, outputs, context['old_summary'])
    summary = {'contract': context['contract'], 'contract_sha256': sha(encoded(context['contract'])),
               'source': context['contract']['source'], 'script_sha256': context['contract']['script_sha256'],
               'archive_sha256': context['contract']['archive_sha256'], 'checkpoint_sha256': context['contract']['checkpoint_sha256'],
               'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
               'ultralytics': md.version('ultralytics'), 'torchvision': md.version('torchvision'), 'device': device,
               'gpu': torch.cuda.get_device_name(0) if device != 'cpu' else None,
               'nms_source_sha256': sha(Path(nms.__file__).read_bytes()),
               'ops_source_sha256': sha(Path(ops.__file__).read_bytes())},
               'settings': context['contract']['settings'], 'primary_reference': 'frozen666',
               'same_tensor_comparison': True, 'fresh_inference': True, 'training': False, 'automatic_promotion': False,
               'saved_baseline_exact_images': sum(t['saved_postclip_exact'] for t in traces.values()),
               'fresh_miss_mechanisms': {name: dict(Counter(c['mechanism'] for c in cases if c['reference'] == name and c['fresh_hard_miss'])) for name in ('r38', 'r50')},
               'reports': reports, 'timing_scope': 'Illustrative per-image suppression and result construction only; not full end-to-end latency.'}
    for name, obj in [('summary.json', summary), ('predictions.json', outputs), ('trace.json', traces),
                      ('missed-cases.json', cases), ('timing.json', runtime_rows), ('input-audit.json', context['receipt'])]:
        save(work / name, obj)
    files = [work / name for name in ('summary.json', 'predictions.json', 'trace.json', 'missed-cases.json', 'timing.json', 'input-audit.json', 'preflight.json')]
    files += [raw_tensors / (i + '.npy') for i in sorted(records)]
    script_copy = work / 'diagnose_fronts_r50.py'; put(script_copy, Path(__file__).read_bytes()); files.append(script_copy)
    inventory = {str(p.relative_to(work)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())} for p in files}
    tmp = target.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p, str(p.relative_to(work)))
        z.writestr('inventory.json', encoded(inventory))
    verify_return(tmp, context)
    need(not target.exists(), 'A completed result appeared during execution; preserved both results for review')
    tmp.replace(target)
    print_summary(summary)
    print('Return:', target, 'SHA256:', sha(target.read_bytes()))
    return summary


def print_summary(summary):
    for name, report in summary['reports'].items():
        for view in ('saved', 'hard', 'soft'):
            d = report[view]['detection']; c = report[view]['corners_by_locatable_count']['2']
            print(name, view, {k: d[k] for k in ('tp', 'fn', 'fp', 'duplicate_fp', 'other_fp')},
                  'both-corner 5px/10px:', c['within5'], c['within10'], '/', c['eligible'])
    print('Fresh miss mechanisms:', summary['fresh_miss_mechanisms'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work', type=Path)
    parser.add_argument('--source-revision')
    parser.add_argument('--script-sha256')
    parser.add_argument('--r50-reference-sha256', default=R50_SHA)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--originals-dir', type=Path)
    parser.add_argument('--device', default='0')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--source-check-only', action='store_true')
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--verify-return', action='store_true')
    mode.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    if args.selftest:
        selftest(); return
    if not args.source_revision or not args.script_sha256 or (not args.source_check_only and not args.work):
        parser.error('--source-revision, --script-sha256 and (except source-check-only) --work are required')
    args.root = args.root.resolve()
    context = source_check(args.root, args.source_revision, args.script_sha256, args.r50_reference_sha256)
    if args.source_check_only:
        return
    context = prepare(args, context)
    target = context['work'] / RESULT_NAME
    if args.preflight_only:
        if target.exists(): verify_return(target, context)
        return
    if args.verify_return:
        need(target.is_file(), 'No completed R50 return exists: ' + str(target))
        print_summary(verify_return(target, context)); return
    run(context, args.device)


if __name__ == '__main__':
    main()
