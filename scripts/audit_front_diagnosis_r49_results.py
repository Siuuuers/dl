#!/usr/bin/env python3
"""Audit the PR143 R49 return and replay its fixed tensors/references on CPU.

No model loading, fresh inference, training, annotation mutation or promotion.
The returned R38 diagnosis and the later R50 reference are kept separate.
"""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

import numpy as np

from audit_additional_front_pilot_r47_results import (
    ARCHIVE_SHA as PRIOR_SHA, MODEL_HASHES, REFERENCES, audit as audit_prior,
    obtain_archive as obtain_prior,
)
from audit_additional_front_pilot_r45_results import corner_groups
from evaluate_pose import INFERENCE, parse_pose_labels
from pose_scoring import score_image, aggregate_scores, _box_iou
from predict_pose_folder import postclip_nms

ROOT = Path(__file__).resolve().parents[1]
NAME = 'baomai-r49-diagnosis-results.zip'
PR_HEAD = 'a59f07dce21c5aedcb088c725dc14225a53a301c'
SOURCE = '8cf5a2afc9fe28ba0f4f4596c27944761a790b76'
ARCHIVE_SHA = '892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3'
ARCHIVE_BYTES = 15631804
SCRIPT_SHA = '188df92036d5dc8fd8e13fc8d8815728a56c33ced654de2cf4716788eac47811'
NOTEBOOK_SHA = '63060941e7834a60562d97ddf3dd1bb46f2feb921b737ae744aaca0d329026d2'
ORIGINAL_MANIFEST_SHA = '72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926'
R50_PATH = 'data/reviews/fronts-2026-11-27-r50/manifest.json'
R50_SHA = 'f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f'
ROW_MAP_SHA = '56e7662a1facbeedcdfe29210c263c4bd1b75ce8b0149e5317138a942667dc9d'
VIEWS = ('saved_raw', 'saved', 'hard_raw', 'hard', 'soft_raw', 'soft')
CONF, NMS, MATCH, CAP = .13, .70, .50, 300
BASE_URL = f'https://raw.githubusercontent.com/Siuuuers/dl/{PR_HEAD}/results/t4/baomai-r49-diagnosis-t4-001/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def checked(path, expected):
    data = Path(path).read_bytes()
    need(sha(data) == expected, 'Hash mismatch: ' + str(path))
    return data


def obtain_archive(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory / NAME
    if not target.exists():
        with urllib.request.urlopen(BASE_URL + NAME, timeout=45) as response:
            data = response.read()
        need(len(data) == ARCHIVE_BYTES and sha(data) == ARCHIVE_SHA, 'Downloaded return differs')
        target.write_bytes(data)
    checked(target, ARCHIVE_SHA)
    return target


def expected_script(root):
    notebook = json.loads(checked(root / 'notebooks/baomai_r49_diagnosis.ipynb', NOTEBOOK_SHA))
    texts = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code': continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCRIPT_TEXT' for t in node.targets):
                texts.append(ast.literal_eval(node.value).encode())
    need(len(texts) == 1 and sha(texts[0]) == SCRIPT_SHA, 'Historical notebook script differs')
    return texts[0]


def load_references(root):
    original = json.loads(checked(root / 'data/manifest.json', ORIGINAL_MANIFEST_SHA))
    records = {r['id']: r for r in original['records'] if r['split'] == 'val'}
    descriptors = {**REFERENCES, 'r50': (R50_PATH, R50_SHA, 654)}
    refs, contract = {}, {}
    for name, (relative, expected, count) in descriptors.items():
        path = root / relative
        manifest = json.loads(checked(path, expected)); refs[name] = {}
        labels = {}
        for r in manifest['files']:
            if r.get('split', 'val') != 'val': continue
            identity = r['image_id']; data = checked(path.parent / r['label_path'], r['label_sha256'])
            need(r.get('image_sha256', r.get('source_image_sha256')) == records[identity]['image_sha256'], 'Reference image differs')
            refs[name][identity] = parse_pose_labels(data.decode(), 1280, 720, name + '/' + identity)
            labels[identity] = r['label_sha256']
        need(set(refs[name]) == set(records) and sum(map(len, refs[name].values())) == count, 'Reference population differs')
        contract[name] = {'manifest_path': relative, 'manifest_sha256': expected, 'objects': count, 'label_sha256': labels}
    maps = json.loads(checked(root / 'results/review/front-review-20261126-r38/row-maps.json', ROW_MAP_SHA))
    need(len(records) == 20, 'Expected twenty original photographs')
    return records, refs, maps, contract


def decoded_geometry(raw, input_shape):
    """Replay FP32 xywh decoding and letterbox inversion without a model runtime."""
    need(raw.dtype == np.dtype('float32') and raw.shape[:2] == (1, 11) and np.isfinite(raw).all(), 'Invalid decoded tensor')
    xywh = raw[0, :4].T
    boxes = np.empty_like(xywh)
    half = xywh[:, 2:] / np.float32(2)
    boxes[:, :2] = xywh[:, :2] - half
    boxes[:, 2:] = xywh[:, :2] + half
    h, w = input_shape[-2:]
    gain = min(h / 720, w / 1280)
    pad = (round((w - 1280 * gain) / 2 - .1), round((h - 720 * gain) / 2 - .1))
    scaled = boxes.copy()
    scaled[:, [0, 2]] -= pad[0]; scaled[:, [1, 3]] -= pad[1]
    scaled /= gain
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, 1280)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, 720)
    points = raw[0, 5:].T.reshape(-1, 2, 3).copy()
    points[:, :, 0] -= (w - 1280 * gain) / 2
    points[:, :, 1] -= (h - 720 * gain) / 2
    points[:, :, :2] /= gain
    points[:, :, 0] = points[:, :, 0].clip(0, 1280)
    points[:, :, 1] = points[:, :, 1].clip(0, 720)
    return boxes, scaled, points, raw[0, 4]


def overlaps(box, boxes):
    inter = np.prod(np.maximum(np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2]), 0), axis=1)
    area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
    union = np.prod(np.maximum(box[2:] - box[:2], 0)) + area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def check_hard_indices(boxes, scores, selected):
    """Validate each native greedy choice, allowing equal-score implementation ties."""
    active = np.flatnonzero(scores > CONF)
    need(len(active) <= 30000, 'Unsupported native candidate-cap case')
    need(len(selected) == len(set(selected)) <= CAP, 'Invalid native selected indices')
    for i in selected:
        need(i in active and scores[i] == max(scores[active]), 'Native order/score selection differs')
        remaining = active[active != i]
        active = remaining[overlaps(boxes[i], boxes[remaining]) <= NMS]
    need(not len(active) or len(selected) == CAP, 'Native NMS left an unsuppressed candidate')


def soft_indices(boxes, confidence):
    boxes = boxes.astype(np.float64); scores = confidence.astype(np.float64).copy()
    active = np.flatnonzero(scores > CONF)
    need(len(active) <= 30000, 'Unsupported soft candidate-cap case')
    indices, values = [], []
    while len(active) and len(indices) < CAP:
        i = int(active[np.argmax(scores[active])])
        indices.append(i); values.append(float(scores[i]))
        active = active[active != i]
        iou = overlaps(boxes[i], boxes[active])
        scores[active] *= np.where(iou > NMS, 1 - iou, 1)
        active = active[scores[active] > CONF]
    return indices, values, bool(len(active))


def reconstructed(indices, scores, boxes, points):
    return [{'class_id': 0, 'bbox_xyxy': boxes[i].tolist(), 'confidence': float(np.float32(s)),
             'keypoints': points[i].tolist()} for i, s in zip(indices, scores)]


def score_reports(refs, outputs):
    reports = {}
    for name, gold in refs.items():
        reports[name] = {}
        for view in VIEWS:
            per_image, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
            for identity, targets in gold.items():
                pred = outputs[identity]['saved']['raw' if view == 'saved_raw' else 'postclip'] if view.startswith('saved') else outputs[identity][view]
                per_image[identity] = score_image(targets, pred, 1280, 720, MATCH, .5)
                for key, values in corner_groups(targets, pred, per_image[identity]).items(): groups[key].update(values)
            reports[name][view] = {'detection': aggregate_scores(list(per_image.values())), 'corners_by_locatable_count': groups, 'per_image': per_image}
        reports[name]['paired_delta'] = {}
        for identity in gold:
            a, b = (reports[name][v]['per_image'][identity] for v in ('hard', 'soft'))
            reports[name]['paired_delta'][identity] = {'recovered_rows': [n+1 for n in sorted(set(a['unmatched_gold_indices'])-set(b['unmatched_gold_indices']))],
                'newly_missed_rows': [n+1 for n in sorted(set(b['unmatched_gold_indices'])-set(a['unmatched_gold_indices']))]}
    return reports


def stage(eligible, confidence, ordinary, final, capped):
    above = {i for i in eligible if confidence[i] > CONF}
    if not eligible: return 'no_iou_eligible_decoded_box'
    if not above: return 'confidence_gate'
    if not above.intersection(ordinary): return 'ordinary_nms_or_cap' if capped else 'ordinary_nms'
    if not above.intersection(final): return 'postclip_suppression'
    return 'retained_assignment'


def missed_cases(refs, maps, outputs, reports, geometry):
    cases = []
    for name in ('r38', 'r50'):
        for identity, gold in refs[name].items():
            row = outputs[identity]; _, boxes, _, confidence = geometry[identity]
            scores = [reports[name][v]['per_image'][identity] for v in ('saved', 'hard', 'soft')]
            old, hard, soft = [set(s['unmatched_gold_indices']) for s in scores]
            capped = sum(confidence > CONF) > 30000 or len(row['ordinary_indices']) == CAP
            rowmap = {r['current_row']: r for r in maps[identity]['rows']}
            for j in sorted(old | hard):
                target = gold[j]; iou = [_box_iou(b, target['bbox_xyxy']) for b in boxes.tolist()]
                eligible = [k for k, overlap in enumerate(iou) if overlap >= MATCH]
                def item(k):
                    return {'anchor_index': k, 'bbox_xyxy': boxes[k].tolist(), 'iou': iou[k], 'confidence': float(confidence[k]),
                            'ordinary': k in row['ordinary_indices'], 'postclip': k in row['hard_indices'], 'soft_postclip': k in row['soft_indices']}
                best = max(range(len(iou)), key=lambda k: (iou[k], float(confidence[k]), -k))
                cases.append({'reference': name, 'image': identity, 'r38_row': j+1, 'stable_id': rowmap[j+1]['stable_id'],
                    'gold': target, 'saved_miss': j in old, 'fresh_hard_miss': j in hard, 'soft_miss': j in soft,
                    'locatable_corners': sum(k[2] == 2 for k in target['keypoints']),
                    'mechanism': stage(eligible, confidence, set(row['ordinary_indices']), set(row['hard_indices']), capped) if j in hard else 'fresh_baseline_matched',
                    'best_decoded': item(best), 'eligible': [item(k) for k in eligible], 'output_cap_possible': bool(capped)})
    return cases


def filter_delta(outputs, refs, reports):
    details, assignment_changes = [], []
    count = Counter()
    for identity, row in outputs.items():
        hr, sr, hf, sf = [set(row[k]) for k in ('ordinary_indices', 'soft_raw_indices', 'hard_indices', 'soft_indices')]
        for label, ids in [('raw_added', sr-hr), ('raw_removed', hr-sr), ('final_added', sf-hf), ('final_removed', hf-sf)]: count[label] += len(ids)
        count['hard_raw'] += len(hr); count['soft_raw'] += len(sr); count['hard_final'] += len(hf); count['soft_final'] += len(sf)
        added = []
        for anchor in sorted(sr-hr):
            k = row['soft_raw_indices'].index(anchor)
            suppressed = next((s for s in row['soft_audit']['suppressed_predictions'] if s['primary_prediction_id'] == k+1), None)
            added.append({'anchor_index': anchor, 'soft_raw_prediction_number': k+1, 'prediction': row['soft_raw'][k],
                'survives_postclip': anchor in sf, 'suppressor_anchor_index': row['soft_raw_indices'][suppressed['suppressor_primary_prediction_id']-1] if suppressed else None})
        removed = []
        for anchor in sorted(hf-sf):
            k = row['hard_indices'].index(anchor); p = row['hard'][k]
            status = {}
            for name, gold in refs.items():
                score = reports[name]['hard']['per_image'][identity]
                best = max(range(len(gold[identity])), key=lambda j: _box_iou(p['bbox_xyxy'], gold[identity][j]['bbox_xyxy']))
                status[name] = {'category': 'matched' if k not in score['unmatched_prediction_indices'] else 'duplicate' if k in score['duplicate_prediction_indices'] else 'other_unmatched',
                    'best_reference_row': best+1, 'best_iou': _box_iou(p['bbox_xyxy'], gold[identity][best]['bbox_xyxy'])}
            removed.append({'anchor_index': anchor, 'hard_final_prediction_number': k+1, 'hard_raw_prediction_number': row['ordinary_indices'].index(anchor)+1, 'prediction': p, 'reference_status': status})
        if hr != sr or hf != sf: details.append({'image': identity, 'raw_added': added, 'raw_removed_anchor_indices': sorted(hr-sr), 'final_removed': removed, 'final_added_anchor_indices': sorted(sf-hf)})
        for name in refs:
            for av, bv, ai, bi in [('hard_raw', 'soft_raw', 'ordinary_indices', 'soft_raw_indices'), ('hard', 'soft', 'hard_indices', 'soft_indices')]:
                left, right = [reports[name][v]['per_image'][identity] for v in (av,bv)]
                pairs = lambda s,key: {m['gold_index']: row[key][m['prediction_index']] for m in s['matches']}
                a, b = pairs(left,ai), pairs(right,bi)
                for j in sorted(set(a)|set(b)):
                    if a.get(j) != b.get(j): assignment_changes.append({'reference': name, 'image': identity, 'row': j+1, 'comparison': av+' -> '+bv, 'before_anchor': a.get(j), 'after_anchor': b.get(j)})
    return {'counts': dict(count), 'per_image': details, 'matched_anchor_assignment_changes': assignment_changes,
        'meaning': 'Proposal IDs are decoded-anchor indices; output P numbers are one-based within the named arm/stage.'}


def audit(archive, root=ROOT, prior_archive=None, originals=None):
    root, archive = Path(root), Path(archive)
    data = checked(archive, ARCHIVE_SHA); need(len(data) == ARCHIVE_BYTES, 'Return size differs')
    records, refs, maps, contract = load_references(root)
    need(prior_archive is not None, 'A verified PR137 archive is required')
    # The producer returns tuples for counts; compare their actual JSON contract.
    prior_receipt = json.loads(encoded(audit_prior(prior_archive, root)))
    with zipfile.ZipFile(prior_archive) as z:
        prior = json.loads(z.read('evaluation/predictions.json'))['candidate']
        prior_summary = json.loads(z.read('evaluation/summary.json'))
    expected = {'summary.json','predictions.json','trace.json','missed-cases.json','timing.json','input-audit.json','diagnose_r49.py'} | {'decoded/'+i+'.npy' for i in records}
    geometry = {}; checked_images = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) == 28 and set(names) == expected | {'inventory.json'} and z.testzip() is None, 'Archive members/CRC differ')
        need(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names), 'Unsafe archive member')
        inventory = json.loads(z.read('inventory.json')); need(set(inventory) == expected, 'Inventory membership differs')
        for name, value in inventory.items():
            raw = z.read(name); need(value == {'bytes':len(raw),'sha256':sha(raw)}, 'Member hash/size differs: '+name)
        need(z.read('diagnose_r49.py') == expected_script(root), 'Returned script differs from historical notebook')
        returned = json.loads(z.read('summary.json')); outputs = json.loads(z.read('predictions.json')); traces = json.loads(z.read('trace.json'))
        original_cases = json.loads(z.read('missed-cases.json')); timings = json.loads(z.read('timing.json'))
        need(json.loads(z.read('input-audit.json')) == prior_receipt, 'PR137 input audit differs')
        for key, value in {'source':SOURCE,'script_sha256':SCRIPT_SHA,'archive_sha256':PRIOR_SHA,'checkpoint_sha256':MODEL_HASHES['candidate'],
                'same_tensor_comparison':True,'fresh_inference':True,'training':False,'automatic_promotion':False}.items(): need(returned[key] == value, 'Experiment identity differs: '+key)
        settings = {**INFERENCE,'conf':CONF,'matching_iou':MATCH,'endpoint_conf':.5,'soft_nms':'linear, IoU > .70, score *= 1-IoU, strict conf > .13, max_det 300, same postclip'}
        need(returned['settings'] == settings and set(outputs) == set(traces) == set(records), 'Settings/image inventory differs')
        need(returned['runtime']['ultralytics'] == '8.4.182' and returned['runtime']['gpu'] == 'Tesla T4', 'Runtime receipt differs')
        for identity in sorted(records):
            row, trace = outputs[identity], traces[identity]
            need(row['saved'] == prior[identity] and trace['image_sha256'] == records[identity]['image_sha256'], 'Saved/image identity differs')
            raw = np.load(io.BytesIO(z.read('decoded/'+identity+'.npy')), allow_pickle=False)
            need(list(raw.shape) == trace['shape'] and sha(raw.tobytes()) == trace['raw_tensor_sha256'], 'Decoded tensor differs')
            geometry[identity] = decoded_geometry(raw, trace['input_shape'])
            boxes, scaled, points, confidence = geometry[identity]
            check_hard_indices(boxes, confidence, row['ordinary_indices'])
            need(reconstructed(row['ordinary_indices'], confidence[row['ordinary_indices']], scaled, points) == row['hard_raw'], 'Native decoded output differs: '+identity)
            ids, values, capped = soft_indices(boxes, confidence)
            need(ids == row['soft_raw_indices'] and capped == trace['soft_cap_reached'], 'Soft selection differs: '+identity)
            need(reconstructed(ids, values, scaled, points) == row['soft_raw'], 'Soft decoded output differs: '+identity)
            for view, key in [('hard','ordinary_indices'),('soft','soft_raw_indices')]:
                final, filtering = postclip_nms(row[view+'_raw'])
                need(final == row[view] and filtering == row[view+'_audit'], 'Postclip replay differs')
                need([row[key][j-1] for j in filtering['retained_primary_prediction_ids']] == row[view+'_indices'], 'Postclip anchor mapping differs')
            need(trace['above_confidence'] == int(sum(confidence>CONF)) and trace['native_reconstruction_exact'], 'Native trace metadata differs')
            need(trace['saved_raw_exact'] == (row['hard_raw'] == prior[identity]['raw']) and trace['saved_postclip_exact'] == (row['hard'] == prior[identity]['postclip']), 'Baseline exactness receipt differs')
            if originals:
                checked(Path(originals)/(identity+'.png'), records[identity]['image_sha256']); checked_images += 1
        need({t['image'] for t in timings} == set(records) and len(timings) == 20, 'Timing image inventory differs')
    reports = score_reports(refs, outputs)
    need({k:reports[k] for k in REFERENCES} == returned['reports'], 'Returned score/paired replay differs')
    for name in REFERENCES:
        for v in ('saved_raw','saved'):
            need(reports[name][v] == prior_summary['references'][name]['raw' if v=='saved_raw' else 'postclip']['candidate'], 'PR137 score baseline differs')
    cases = missed_cases(refs, maps, outputs, reports, geometry)
    need([{k:v for k,v in c.items() if k!='reference'} for c in cases if c['reference']=='r38'] == original_cases, 'Returned missed-case trace differs')
    stages = {name:dict(Counter(c['mechanism'] for c in cases if c['reference']==name and c['fresh_hard_miss'])) for name in ('r38','r50')}
    need(stages['r38'] == returned['fresh_miss_mechanisms'], 'Returned miss-stage aggregate differs')
    exact = sum(r['hard'] == r['saved']['postclip'] for r in outputs.values())
    need(exact == returned['saved_baseline_exact_images'] == 20, 'Fresh/saved baseline differs')
    compact = {}
    for name, views in reports.items():
        compact[name] = {'paired_delta': views['paired_delta']}
        for view in VIEWS:
            report = views[view]
            compact[name][view] = {'detection': {k:v for k,v in report['detection'].items() if k!='endpoint_errors_px'}, 'corners_by_locatable_count':report['corners_by_locatable_count'],
                'per_image': {i:{k:s[k] for k in ('tp','fn','fp','duplicate_fp','other_fp')} for i,s in report['per_image'].items()}}
    delta = filter_delta(outputs, refs, reports)
    summary = {'schema':'pr143-r49-audit-summary-v1','source_pr':143,'pr_head':PR_HEAD,'source':SOURCE,'archive_sha256':ARCHIVE_SHA,
        'runtime':returned['runtime'],'settings':returned['settings'],'primary_reference':'frozen666','references':contract,
        'fresh_saved_exact_images':exact,'miss_stages':stages,'reports':{n:{v:{k:compact[n][v][k] for k in ('detection','corners_by_locatable_count')} for v in VIEWS} for n in refs},
        'filter_delta_counts':delta['counts'],'matched_anchor_assignment_changes':delta['matched_anchor_assignment_changes'],
        'fresh_inference_performed_by_this_audit':False,'returned_experiment_is_fresh_inference':True,'training':False,'automatic_promotion':False}
    verification = {'schema':'pr143-r49-return-audit-v1','status':'passed','pr':143,'pr_head':PR_HEAD,'archive_sha256':ARCHIVE_SHA,'archive_bytes':ARCHIVE_BYTES,
        'verified_inventory_members':27,'historical_script_sha256':SCRIPT_SHA,'historical_notebook_sha256':NOTEBOOK_SHA,
        'independent_numpy_native_selection_checks':20,'exact_native_box_confidence_keypoint_reconstructions':20,
        'exact_linear_soft_selection_and_output_reconstructions':20,'replayed_postclip_filters':40,'replayed_original_score_tables':18,'additional_r50_score_tables':6,
        'reconstructed_original_miss_traces':len(original_cases),'original_images_checked':checked_images,'fresh_inference':False,'fresh_training':False,'automatic_promotion':False,
        'timing_limit':'Returned timings measure suppression and construction only; CPU NumPy Soft-NMS versus native runtime is not an optimized deployment comparison.'}
    return {'summary.json':summary,'reference-replay.json':compact,'missed-cases.json':cases,'filter-delta.json':delta,'verification.json':verification}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--archive',type=Path)
    p.add_argument('--prior-archive',type=Path); p.add_argument('--originals',type=Path)
    p.add_argument('--download-dir',type=Path,default=Path('/tmp/pr143-r49-audit'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    archive=a.archive or obtain_archive(a.download_dir)
    prior=a.prior_archive or obtain_prior(a.download_dir/'pr137')
    results=audit(archive,a.root,prior,a.originals)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,obj in results.items():
        path=a.output/name; data=encoded(obj)
        need(not path.exists() or path.read_bytes()==data,'Existing incompatible audit preserved: '+str(path))
    for name,obj in results.items(): (a.output/name).write_bytes(encoded(obj))
    print(json.dumps(results['verification.json'],indent=2))
    print('Miss stages:',results['summary.json']['miss_stages'])
    print('Filter delta:',results['filter-delta.json']['counts'])


if __name__ == '__main__': main()
