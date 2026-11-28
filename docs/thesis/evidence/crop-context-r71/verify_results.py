#!/usr/bin/env python3
"""Independent R71 arithmetic replay. Imports no repository implementation."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(root, name):
    data = (root / name).read_bytes()
    return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)


def equivalent(first, second, label='value'):
    if isinstance(first, dict) and isinstance(second, dict):
        assert set(first) == set(second), (label, 'keys', set(first) ^ set(second))
        for key in first:
            equivalent(first[key], second[key], label + '/' + str(key))
    elif isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        assert len(first) == len(second), (label, 'length', len(first), len(second))
        for index, (a, b) in enumerate(zip(first, second)):
            equivalent(a, b, label + '/' + str(index))
    elif isinstance(first, float) or isinstance(second, float):
        if first is None or second is None:
            assert first is second, (label, first, second)
        else:
            assert math.isclose(first, second, rel_tol=0., abs_tol=1e-10), (label, first, second)
    else:
        assert first == second, (label, first, second)


def iou(a, b):
    intersect = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    if not intersect:
        return 0.
    aa, ba = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    scale = max(aa, ba)
    return (intersect / scale) / (aa / scale + ba / scale - intersect / scale)


def pose_overlap(a, b):
    if a['class_id'] != b['class_id']:
        return 0.
    a, b = a['bbox_xyxy'], b['bbox_xyxy']
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.


def point_ok(point):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in point) and 0 <= point[0] < 1280 and 0 <= point[1] < 720 and point[2] >= .5


def suppress(pool):
    kept, removed = [], []
    for index in sorted(range(len(pool)), key=lambda i: (-pool[i]['confidence'], i)):
        partner = None
        for other in kept:
            if pool[index]['class_id'] == pool[other]['class_id'] and iou(pool[index]['bbox_xyxy'], pool[other]['bbox_xyxy']) > .70:
                partner = other
                break
        if partner is None:
            kept.append(index)
        else:
            removed.append({'primary_prediction_id': index+1, 'suppressor_primary_prediction_id': partner+1,
                            'postclip_prediction_iou': iou(pool[index]['bbox_xyxy'], pool[partner]['bbox_xyxy']),
                            'prediction': pool[index]})
    return [{**deepcopy(pool[i]), 'primary_prediction_id': i+1} for i in kept], kept, removed


def complete_baseline(native):
    out = {}
    for image, records in native['original'].items():
        original = suppress(records)[0]
        reflected = deepcopy(native['reflected'][image])
        for p in reflected:
            x0, y0, x1, y1 = p['bbox_xyxy']
            p['bbox_xyxy'] = [1280-x1, y0, 1280-x0, y1]
            p['keypoints'] = [[1280-k[0] if k[0] is not None else None, k[1], k[2]] for k in p['keypoints'][::-1]]
        reflected = suppress(reflected)[0]
        matrix = [[pose_overlap(a, b) for b in reflected] for a in original]
        merged = deepcopy(original)
        for i, a in enumerate(original):
            if not reflected:
                continue
            j = max(range(len(reflected)), key=lambda k: (matrix[i][k], -k))
            reverse = max(range(len(original)), key=lambda k: (matrix[k][j], -k))
            if reverse == i and matrix[i][j] >= .50 and all(point_ok(k) for k in a['keypoints']) and all(point_ok(k) for k in reflected[j]['keypoints']):
                for corner in (0, 1):
                    for axis in (0, 1):
                        merged[i]['keypoints'][corner][axis] = (a['keypoints'][corner][axis] + reflected[j]['keypoints'][corner][axis])/2
        out[image] = merged
    return out


def parse_labels(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        v = list(map(float, line.split()))
        assert len(v) == 11
        x, y, w, h = v[1]*1280, v[2]*720, v[3]*1280, v[4]*720
        rows.append({'class_id': int(v[0]), 'bbox_xyxy': [x-w/2, y-h/2, x+w/2, y+h/2],
                     'keypoints': [[v[5]*1280, v[6]*720, int(v[7])], [v[8]*1280, v[9]*720, int(v[10])]]})
    return rows


def corners(target, prediction):
    result = []
    for i, gt in enumerate(target['keypoints']):
        if gt[2] != 2:
            continue
        p = prediction['keypoints'][i] if prediction is not None else None
        valid = p is not None and point_ok(p)
        err = math.hypot(p[0]-gt[0], p[1]-gt[1]) if valid else None
        result.append({'corner_index': i, 'reference_xy': gt[:2], 'prediction': p, 'available': valid,
                       'error_px': err, 'within5': err is not None and err <= 5., 'within10': err is not None and err <= 10.})
    return result


def percentile(values, q):
    if not values:
        return None
    seq = sorted(values)
    pos = q*(len(seq)-1)
    lower, upper = math.floor(pos), math.ceil(pos)
    return seq[lower]+(seq[upper]-seq[lower])*(pos-lower)


def score(targets, predictions, image):
    remaining = set(range(len(targets)))
    matches = []
    overlaps = [[iou(p['bbox_xyxy'], gt['bbox_xyxy']) for gt in targets] for p in predictions]
    for pi in sorted(range(len(predictions)), key=lambda i: (-predictions[i]['confidence'], i)):
        candidates = [gi for gi in remaining if targets[gi]['class_id'] == predictions[pi]['class_id'] and overlaps[pi][gi] >= .5]
        if candidates:
            gi = max(candidates, key=lambda j: (overlaps[pi][j], -j))
            remaining.remove(gi)
            matches.append({'gold_index': gi, 'prediction_index': pi, 'iou': overlaps[pi][gi]})
    matched_pi, matched_gt = {m['prediction_index'] for m in matches}, {m['gold_index'] for m in matches}
    extras = [i for i in range(len(predictions)) if i not in matched_pi]
    dups = [i for i in extras if any(targets[j]['class_id'] == predictions[i]['class_id'] and overlaps[i][j] >= .5 for j in matched_gt)]
    by_gt = {m['gold_index']: m['prediction_index'] for m in matches}
    errors = []
    for m in matches:
        errors.extend(k['error_px'] for k in corners(targets[m['gold_index']], predictions[m['prediction_index']]) if k['available'])
    visible = sum(k[2] == 2 for gt in targets for k in gt['keypoints'])
    measurement = {'gold_count': len(targets), 'prediction_count': len(predictions), 'tp': len(matches), 'fn': len(remaining),
                   'fp': len(extras), 'duplicate_fp': len(dups), 'other_fp': len(extras)-len(dups), 'visible_endpoints': visible,
                   'returned_endpoints': len(errors), 'missing_endpoints': visible-len(errors),
                   'endpoint_coverage': len(errors)/visible if visible else None,
                   'conditional_median_error_px': percentile(errors, .5), 'conditional_p95_error_px': percentile(errors, .95),
                   'endpoint_errors_px': errors, 'matches': matches, 'unmatched_gold_indices': sorted(remaining),
                   'unmatched_prediction_indices': extras, 'duplicate_prediction_indices': dups,
                   'other_fp_prediction_indices': [i for i in extras if i not in dups]}
    states = []
    for gi, gt in enumerate(targets):
        pi = by_gt.get(gi)
        pred = predictions[pi] if pi is not None else None
        cc = corners(gt, pred)
        complete = len(cc) == 2
        maximum = max(c['error_px'] for c in cc) if complete and all(c['available'] for c in cc) else None
        states.append({'image': image, 'reference_index': gi, 'prediction_index': pi,
                       'complete_reference': complete, 'detected': pi is not None, 'maximum_corner_error_px': maximum,
                       'within5': maximum is not None and maximum <= 5., 'within10': maximum is not None and maximum <= 10.})
    return measurement, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--execution-receipt', type=Path)
    args = parser.parse_args()
    root = args.results
    inventory = read(root, 'inventory.json')
    for name, spec in inventory.items():
        data = (root/name).read_bytes()
        assert len(data) == spec['bytes'] and sha(data) == spec['sha256'], name
    summary = read(root, 'summary.json')
    assert summary['protocol_sha256'] == sha((root/'protocol.json').read_bytes())
    archive_path = args.archive or root/'baomai-crop-context-r71-results.zip'
    package = None
    if archive_path.is_file():
        with ZipFile(archive_path) as archive:
            zipped = {name: archive.read(name) for name in archive.namelist()}
        assert len(zipped) == len(ZipFile(archive_path).namelist())
        archive_inventory = json.loads(zipped['inventory.json'])
        assert set(zipped) == set(archive_inventory) | {'inventory.json'}
        for name, spec in archive_inventory.items():
            assert spec == {'bytes':len(zipped[name]),'sha256':sha(zipped[name])}, name
        for name in inventory:
            assert (root/name).read_bytes() == zipped[name], name
        source = json.loads(zipped['source.json'])
        package = {'filename':archive_path.name,'bytes':archive_path.stat().st_size,'sha256':sha(archive_path.read_bytes())}
    else:
        source = read(root, 'source.json')
    assert source['analysis_source_sha256'] == summary['analysis_source_sha256']
    assert source['protocol_sha256'] == summary['protocol_sha256'] and source['checkpoint'] == summary['checkpoint']
    source_commit = source['source_commit']
    assert len(source_commit)==40 and all(c in '0123456789abcdef' for c in source_commit)
    execution_binding = None
    if args.execution_receipt:
        execution_data = args.execution_receipt.read_bytes()
        execution = json.loads(execution_data)
        assert execution['status']=='passed' and execution['source_commit']==source_commit
        assert execution['protocol_sha256']==summary['protocol_sha256'] and execution['checkpoint']==summary['checkpoint']
        assert execution['source_sha256']==summary['analysis_source_sha256']
        assert execution['input_notebook_unchanged'] and execution['source_checkout_clean']
        if package is not None:
            assert execution['package']==package
        assert len(execution['cells'])==4
        for cell in execution['cells']:
            assert cell['status']=='passed' and cell['source_unchanged']
            assert sha((args.execution_receipt.parent/cell['log']).read_bytes())==cell['log_sha256']
        executed_path=args.execution_receipt.parent/execution['executed_notebook']['path']
        executed_data=executed_path.read_bytes()
        assert len(executed_data)==execution['executed_notebook']['bytes'] and sha(executed_data)==execution['executed_notebook']['sha256']
        executed=json.loads(executed_data)
        for cell in execution['cells']:
            code=executed['cells'][cell['notebook_cell_index']]['source']
            assert sha((''.join(code) if isinstance(code,list) else code).encode())==cell['source_sha256']
        execution_binding={'schema':execution['schema'],'status':'passed','source_commit':source_commit,
                           'receipt_sha256':sha(execution_data),'notebook':execution['notebook'],
                           'executed_notebook':execution['executed_notebook'],'code_cells_passed':4}
    outputs = read(root, 'predictions.json.gz')
    refs = {r: {im: parse_labels(text) for im, text in images.items()} for r, images in read(root, 'reference-labels.json.gz').items()}
    native = read(root, 'native-baseline.json.gz')
    expected_baseline = complete_baseline(native)
    equivalent(expected_baseline, outputs['baseline'], 'baseline_reconstruction')
    crop_rows, merges = read(root, 'crop-lineage.json.gz'), read(root, 'merge-lineage.json.gz')
    calls = read(root, 'inference-calls.json.gz')
    call_index = {x['call_id']: x for x in calls}
    assert len(calls) == len(call_index) == 120
    boundary_cases, invalid_points, crop_records, kept_total = 0, 0, 0, 0
    artificial_invalid, photographedge_invalid = 0, 0
    for variant, images in crop_rows.items():
        size = 768 if variant == 'crop_context_768' else 1280
        for image, crops in images.items():
            pool = deepcopy(expected_baseline[image])
            lineage = [{'source_id': f'baseline/{image}/P{i+1:03d}', 'source': 'baseline', 'source_index': i, 'crop': None} for i in range(len(pool))]
            for crop_name, x0, x1 in (('left', 0, 768), ('right', 512, 1280)):
                call = call_index[f'{variant}/{crop_name}/{image}']
                assert call['image_size'] == size
                expected_shape = [1,3,736,768] if size == 768 else [1,3,1216,1280]
                assert call['input_tensor_shape'] == expected_shape
                expected_transform = {'original_size_wh': [768,720], 'input_size_wh': [expected_shape[3], expected_shape[2]],
                                      'resized_size_wh': [768,720] if size == 768 else [1280,1200],
                                      'gain_xy': [1.,1.] if size == 768 else [5/3,5/3],
                                      'pad_left_top_right_bottom': [0,8,0,8]}
                equivalent(call['transform'], expected_transform, 'transform')
                rows = crops[crop_name]
                assert len(rows) == call['native_output_count'] == len(call['unclipped_keypoints'])
                for index, row in enumerate(rows):
                    crop_records += 1
                    assert row['native_index'] == index and row['crop_xyxy'] == [x0,0,x1,720]
                    p = row['native_prediction']
                    b = p['bbox_xyxy']
                    assert .25 < p['confidence'] <= 1 and p['class_id'] == 0
                    reasons = []
                    if x0 and b[0] <= 1:
                        reasons.append('box_touches_artificial_left_edge')
                    if x1 < 1280 and b[2] >= 767:
                        reasons.append('box_touches_artificial_right_edge')
                    assert reasons == row['boundary_rejection_reasons']
                    assert row['retained_for_pool'] == (not reasons)
                    boundary_cases += bool(reasons)
                    mapped = deepcopy(p)
                    mapped['bbox_xyxy'] = [b[0]+x0,b[1],b[2]+x0,b[3]]
                    for corner, (up, nativep, pm) in enumerate(zip(call['unclipped_keypoints'][index], p['keypoints'], row['point_mapping'])):
                        x,y,q = up
                        assert pm['unclipped_crop_xy'] == [x,y] and pm['native_clipped_crop_xy'] == nativep[:2]
                        assert q == nativep[2] == pm['confidence']
                        finite = all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y))
                        invalid = ['nonfinite_unclipped_crop_coordinate'] if not finite else ['unclipped_coordinate_outside_crop'] if not (0 <= x < 768 and 0 <= y < 720) else []
                        assert invalid == pm['invalid_reasons']
                        mappedp = [None,None,q] if invalid else [x+x0,y,q]
                        if invalid:
                            invalid_points += 1
                            if finite:
                                artificial_invalid += (crop_name == 'left' and x >= 768) or (crop_name == 'right' and x < 0)
                                photographedge_invalid += y < 0 or y >= 720 or (crop_name == 'left' and x < 0) or (crop_name == 'right' and x >= 768)
                        else:
                            assert nativep == up
                        mapped['keypoints'][corner] = mappedp
                        equivalent(mappedp, pm['mapped_keypoint'], 'point_mapping')
                    equivalent(mapped, row['mapped_prediction'], 'mapped_prediction')
                    if not reasons:
                        pool.append(mapped)
                        lineage.append({'source_id': f'{variant}/{crop_name}/{image}/P{index+1:03d}', 'source': 'crop', 'source_index': index, 'crop': crop_name})
            expected, kept, removed = suppress(pool)
            kept_total += len(kept)
            audit = merges[variant][image]
            equivalent(pool, audit['pool_predictions'], 'pool')
            equivalent(lineage, audit['pool_lineage'], 'pool_lineage')
            assert kept == audit['kept_pool_indices']
            assert [lineage[i]['source_id'] for i in kept] == audit['kept_source_ids']
            equivalent(removed, audit['suppression']['suppressed_predictions'], 'suppression')
            equivalent(expected, outputs[variant][image], 'merged_outputs')
    saved_scores, saved_states = read(root, 'per-image.json.gz'), read(root, 'target-states.json.gz')
    states, scores = {}, {}
    matrix = []
    for variant, images in outputs.items():
        states[variant], scores[variant] = {}, {}
        for ref, gold in refs.items():
            agg = Counter(); target_states=[]; group = {str(k): {'fronts':0,'matched':0,'missed':0} for k in range(3)}
            scores[variant][ref] = {}
            for image, targets in sorted(gold.items()):
                sc, st = score(targets, images[image], image)
                equivalent(sc, saved_scores[variant][ref][image], f'score/{variant}/{ref}/{image}')
                scores[variant][ref][image] = sc
                target_states.extend(st)
                for key in ('gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp'):
                    agg[key] += sc[key]
                for gt, state in zip(targets, st):
                    g = group[str(sum(p[2]==2 for p in gt['keypoints']))]
                    g['fronts'] += 1; g['matched'] += state['detected']; g['missed'] += not state['detected']
            equivalent(target_states, saved_states[variant][ref], f'target_states/{variant}/{ref}')
            states[variant][ref] = target_states
            complete = [s for s in target_states if s['complete_reference']]
            row = {'reference':ref,'method':variant,'fronts':agg['gold_count'],'outputs':agg['prediction_count'],
                   'matched':agg['tp'],'missed':agg['fn'],'extras':agg['fp'],'duplicates':agg['duplicate_fp'],
                   'other_extras':agg['other_fp'],'eligible_pairs':len(complete),'within5':sum(s['within5'] for s in complete),
                   'within10':sum(s['within10'] for s in complete),'complete_pair_missed_boxes':sum(not s['detected'] for s in complete),
                   'matched_pair_unavailable':sum(s['detected'] and s['maximum_corner_error_px'] is None for s in complete),
                   'known_corner_groups':group,'empty_scene_outputs':sum(len(images[i]) for i in gold if not gold[i])}
            matrix.append(row)
    equivalent({(r['method'],r['reference']):r for r in matrix},
               {(r['method'],r['reference']):r for r in summary['matrix']}, 'matrix')
    target_ledger = read(root, 'paired-targets.json.gz')
    reference_lineage = read(root, 'reference-lineage.json.gz')
    stable = {(r['image'],r['candidate_index']): r.get('stable_reference_id',r.get('stable_id')) for r in reference_lineage}
    paired = {v: {r: Counter() for r in refs} for v in outputs if v != 'baseline'}
    known = {v: {r: {str(k):Counter({n:0 for n in ('fronts','reference_corners','before_available','after_available','before_within5','after_within5','before_within10','after_within10')}) for k in range(3)} for r in refs} for v in paired}
    indexed_states = {v:{r:{(s['image'],s['reference_index']):s for s in ss} for r,ss in re.items()} for v,re in states.items()}
    change_keys = ('detection_gain','detection_loss','pair_within5_gain','pair_within5_loss','pair_within10_gain','pair_within10_loss','source_changed')
    seen=set()
    for row in target_ledger:
        variant,ref,image,gi = row['variant'],row['reference'],row['image'],row['reference_index']
        key=(variant,ref,image,gi)
        assert key not in seen
        seen.add(key)
        before,after = indexed_states['baseline'][ref][(image,gi)],indexed_states[variant][ref][(image,gi)]
        bi,ai=before['prediction_index'],after['prediction_index']
        bp=outputs['baseline'][image][bi] if bi is not None else None
        ap=outputs[variant][image][ai] if ai is not None else None
        bid=f'baseline/{image}/P{bi+1:03d}' if bi is not None else None
        aid=merges[variant][image]['kept_source_ids'][ai] if ai is not None else None
        gt=refs[ref][image][gi]
        bc,ac=corners(gt,bp),corners(gt,ap)
        equivalent(row['before'],{**before,'source_id':bid,'known_corner_outcomes':bc},'paired_before')
        equivalent(row['after'],{**after,'source_id':aid,'known_corner_outcomes':ac},'paired_after')
        expected={'detection_gain':not before['detected'] and after['detected'], 'detection_loss':before['detected'] and not after['detected'],
                  'pair_within5_gain':not before['within5'] and after['within5'],'pair_within5_loss':before['within5'] and not after['within5'],
                  'pair_within10_gain':not before['within10'] and after['within10'],'pair_within10_loss':before['within10'] and not after['within10'],
                  'source_changed':bid!=aid}
        for name,value in expected.items():
            assert row[name] == value
            paired[variant][ref][name] += value
        nk=len(bc)
        assert row['known_corners'] == nk and row['stable_reference_id'] == (stable[(image,gi)] if ref=='r67' else f'{ref}/{image}/R{gi+1:03d}')
        group=known[variant][ref][str(nk)];group['fronts']+=1;group['reference_corners']+=nk
        for prefix,cc in (('before',bc),('after',ac)):
            for metric in ('available','within5','within10'):
                group[prefix+'_'+metric]+=sum(c[metric] for c in cc)
    assert len(seen)==2*sum(len(g) for images in refs.values() for g in images.values())
    for variant in paired:
        for ref in refs:
            for key in change_keys:
                assert paired[variant][ref][key] == summary['paired_changes'][variant][ref][key]
            for suffix in ('gain','loss'):
                ids=[r['stable_reference_id'] for r in target_ledger if r['variant']==variant and r['reference']==ref and r['detection_'+suffix]]
                assert ids==summary['paired_changes'][variant][ref]['detection_'+suffix+'_ids']
    equivalent(known, summary['known_corner_totals'], 'known_corner_totals')
    saved_outputs=read(root,'output-states.json.gz')
    for row in saved_outputs:
        v,r,i,pi=row['variant'],row['reference'],row['image'],row['prediction_index']
        sc=scores[v][r][i]
        match={m['prediction_index']:m['gold_index'] for m in sc['matches']}
        status='matched' if pi in match else 'duplicate' if pi in sc['duplicate_prediction_indices'] else 'other_extra'
        assert row['status']==status and row['reference_index']==match.get(pi)
        sid=f'baseline/{i}/P{pi+1:03d}' if v=='baseline' else merges[v][i]['kept_source_ids'][pi]
        assert row['source_id']==sid
    receipt={'schema':'r71-independent-numerical-verification-v1','status':'passed','passed':True,'verifier_sha256':sha(Path(__file__).read_bytes()),
             'source_commit':source_commit,'package':package,'notebook_execution':execution_binding,
             'result_directory':str(root),'input_inventory_sha256':sha((root/'inventory.json').read_bytes()),
             'summary_sha256':sha((root/'summary.json').read_bytes()),'protocol_sha256':summary['protocol_sha256'],
             'checkpoint':summary['checkpoint'],'repository_functions_imported':False,
             'verified':{'baseline_native_reflection_nms_and_average':True,'crop_mapping_edge_filter_and_whole_record_nms':True,
                         'crop_native_records':crop_records,'boundary_rejected':boundary_cases,'invalidated_native_points':invalid_points,
                         'invalidated_artificial_edge_points':artificial_invalid,'invalidated_photo_edge_points':photographedge_invalid,
                         'matrix_rows':len(matrix),'per_image_scores':sum(len(x) for v in scores.values() for x in v.values()),
                         'paired_target_records':len(target_ledger),'output_state_records':len(saved_outputs),'all_known_corner_totals':True},
             'r67_matrix':[m for m in matrix if m['reference']=='r67'],
             'r67_paired_changes':{v:summary['paired_changes'][v]['r67'] for v in paired}}
    args.receipt.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
