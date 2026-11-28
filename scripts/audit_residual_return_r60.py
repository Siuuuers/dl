#!/usr/bin/env python3
"""Audit the returned residual-corner experiment and compare fixed saved outputs."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from zipfile import ZipFile

SOURCE = '06e55d9c1b2557014aff7ff0d68b5096890fe546'
DRIVER = '54cdf49281abdc5409c9a8828d1a3bb00c5f9da8610e8767645b69315a78e5d3'
RETURN_HEAD = '244e553b2ea0ae6fe35d518c33da8998c2a31e5e'
RETURN_SHA = 'dbf5057d765c2f38bc2a28fcf98104c51c49a09afa5a83a2268495784e08b79c'
RETURN_BYTES = 3971513
ORIGINAL_COMPARISON_SHA = 'aae05d14298afad0263197fe4ab7e0e753f63445d9d3503dc1b006030e16c516'
MEMBERS = {'baseline-equivalence.json', 'cases.json', 'corner_refinement_r58.py',
    'correction-traces.json', 'detector-evaluation.json', 'detector-receipt.json',
    'detector-runtime.json', 'detector-training.json', 'input-audit.json', 'inventory.json',
    'model-smoke.json', 'pair-accounting-common025.json', 'pair-accounting.json', 'predictions.json',
    'preflight.json', 'residual_corner_r59.py', 'run_corner_refinement_r58.py',
    'run_residual_corner_r59.py', 'staging-r58.json', 'summary.json', 'training-crops.json',
    'training-log.json', 'training-receipt.json', 'zero-output-gate.json'}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def valid_point(point):
    return (len(point) == 3 and all(isinstance(x, (float, int)) and math.isfinite(x) for x in point)
            and 0 <= point[0] < 1280 and 0 <= point[1] < 720 and point[2] >= .5)


def independent_geometry(box):
    x0, y0, x1, y1 = box
    w, h = x1-x0, y1-y0
    bounds = [max(0, math.floor(x0-w/4)), max(0, math.floor(y0-h/4)),
              min(1280, math.ceil(x1+w/4)), min(720, math.ceil(y1+h/4))]
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    scale = min(256/width, 256/height)
    resized = [max(1, min(256, round(width*scale))), max(1, min(256, round(height*scale)))]
    return {'xyxy': bounds, 'resized_wh': resized, 'scale_xy': [resized[0]/width, resized[1]/height],
            'pad_xy': [(256-resized[0])//2, (256-resized[1])//2], 'canvas_wh': [256, 256]}


def replay_independent_offsets(arms, traces):
    seen, states = set(), Counter()
    for trace in traces:
        image, index = trace['image_id'], trace['raw_index']
        need((image, index) not in seen, 'Repeated correction identity')
        seen.add((image, index))
        before, after = arms['control'][image]['raw'][index], arms['treatment'][image]['raw'][index]
        geom = independent_geometry(before['bbox_xyxy'])
        need(geom == trace['geometry'], 'Independent crop geometry differs')
        expected = deepcopy(before)
        for slot, (point, delta) in enumerate(zip(before['keypoints'], trace['canvas_offsets'])):
            if not valid_point(point):
                continue
            finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in delta)
            if finite and all(x == 0 for x in delta):
                continue
            position = [point[axis]+delta[axis]/geom['scale_xy'][axis] for axis in range(2)] if finite else None
            available = position is not None and 0 <= position[0] < 1280 and 0 <= position[1] < 720
            expected['keypoints'][slot] = [*position, point[2]] if available else [None, None, point[2]]
        need(expected == after and trace['original_points'] == before['keypoints']
             and trace['output_points'] == after['keypoints'], 'Independent correction application differs')
        states.update(trace['endpoint_states'])
    need(len(seen) == sum(len(r['raw']) for r in arms['control'].values()), 'Missing correction record')
    return {'records': len(seen), 'endpoints': 2*len(seen), 'states': dict(states)}


def model_parts(directory, data):
    raw = (directory/'return-manifest.json').read_bytes()
    manifest = json.loads(raw)
    need(manifest['status'] == 'completed' and manifest['evidence_sha256'] == RETURN_SHA
         and manifest['evidence_file'] == 'baomai-residual-corner-r59-results.zip', 'Return manifest differs')
    need(len(manifest['parts']) == 6, 'Checkpoint part count differs')
    digest, length = hashlib.sha256(), 0
    for index, part in enumerate(manifest['parts']):
        name = f'correction-final.pt.part{index:02d}'
        need(part['file'] == name+'.zip' and part['member'] == name, 'Checkpoint order differs')
        payload = (directory/part['file']).read_bytes()
        need(len(payload) == part['bytes'] and sha(payload) == part['sha256'], 'Checkpoint wrapper differs')
        with ZipFile(directory/part['file']) as stream:
            need(stream.namelist() == [name] and stream.testzip() is None, 'Checkpoint wrapper membership differs')
            content = stream.read(name)
        need(len(content) == part['member_bytes'] and sha(content) == part['member_sha256'], 'Checkpoint part differs')
        digest.update(content)
        length += len(content)
    receipt = data['training-receipt']
    need(length == manifest['model_bytes'] == receipt['model_bytes']
         and digest.hexdigest() == manifest['model_sha256'] == receipt['model_sha256'], 'Full model identity differs')
    return {'parts': 6, 'model_bytes': length, 'model_sha256': digest.hexdigest(), 'manifest_sha256': sha(raw)}


def pair_comparison(refs, outputs, score_image):
    reports, all_rows = {}, {}
    for reference, gold in refs.items():
        rows = []
        for image, targets in sorted(gold.items()):
            scores = {arm: score_image(targets, values[image]['postclip'], 1280, 720, .5, .5)
                      for arm, values in outputs.items()}
            matches = {arm: {m['gold_index']: m['prediction_index'] for m in result['matches']}
                       for arm, result in scores.items()}
            for index, target in enumerate(targets):
                if not all(p[2] == 2 for p in target['keypoints']):
                    continue
                row = {'image': image, 'row': index+1, 'id': f'{image}/R{index+1:03d}', 'arms': {}}
                for arm, values in outputs.items():
                    pi = matches[arm].get(index)
                    pred = values[image]['postclip'][pi] if pi is not None else None
                    valid = pred is not None and all(valid_point(p) for p in pred['keypoints'])
                    errors = [math.hypot(p[0]-g[0], p[1]-g[1]) for p, g in zip(pred['keypoints'], target['keypoints'])] if valid else None
                    row['arms'][arm] = {'prediction_index': pi, 'errors_px': errors,
                        'maximum_error_px': max(errors) if valid else None,
                        'within5': valid and max(errors) <= 5, 'within10': valid and max(errors) <= 10}
                rows.append(row)
        report = {'eligible': len(rows), 'arms': {}, 'residual_changes_from': {}}
        for arm in outputs:
            errors = [r['arms'][arm]['maximum_error_px'] for r in rows if r['arms'][arm]['maximum_error_px'] is not None]
            report['arms'][arm] = {'within5': sum(r['arms'][arm]['within5'] for r in rows),
                'within10': sum(r['arms'][arm]['within10'] for r in rows), 'matched_available_pairs': len(errors),
                'mean_maximum_error_px': statistics.mean(errors), 'median_maximum_error_px': statistics.median(errors),
                'maximum_error_px': max(errors), 'per_image': {
                    image: {'eligible': sum(r['image'] == image for r in rows),
                            'within5': sum(r['image'] == image and r['arms'][arm]['within5'] for r in rows),
                            'within10': sum(r['image'] == image and r['arms'][arm]['within10'] for r in rows)}
                    for image in gold}}
        for baseline in ('control', 'proposed', 'supplied'):
            report['residual_changes_from'][baseline] = {}
            for tolerance in (5, 10):
                key = f'within{tolerance}'
                gains = [r['id'] for r in rows if not r['arms'][baseline][key] and r['arms']['residual'][key]]
                losses = [r['id'] for r in rows if r['arms'][baseline][key] and not r['arms']['residual'][key]]
                report['residual_changes_from'][baseline][str(tolerance)] = {
                    'gains': len(gains), 'losses': len(losses), 'gained_ids': gains, 'lost_ids': losses}
        reports[reference], all_rows[reference] = report, rows
    return reports, all_rows


def audit(root, directory, original_archive, comparison_archive, overrides=None):
    sys.path.insert(0, str(root/'scripts'))
    from audit_corner_refinement_returns import checked_source
    from audit_reflection_return import load_references
    import audit_additional_front_pilot_r47_results as prior
    import run_inference_ablation_r53 as r53
    import run_residual_corner_r59 as runner
    from evaluate_pose import parse_pose_labels
    from pose_scoring import score_image
    import torch
    archive = directory/'baomai-residual-corner-r59-results.zip'
    payload = archive.read_bytes()
    need(len(payload) == RETURN_BYTES and sha(payload) == RETURN_SHA, 'Returned evidence ZIP differs')
    with ZipFile(archive) as stream:
        need(set(stream.namelist()) == MEMBERS and len(stream.namelist()) == len(MEMBERS)
             and stream.testzip() is None, 'Evidence ZIP membership/CRC differs')
        members = {name: stream.read(name) for name in stream.namelist()}
    inventory = json.loads(members['inventory.json'])
    need(set(inventory) == MEMBERS-{'inventory.json'}, 'Inventory members differ')
    for name, expected in inventory.items():
        need(expected == {'bytes': len(members[name]), 'sha256': sha(members[name])}, 'Evidence member differs: '+name)
    data = {name[:-5]: json.loads(value) for name, value in members.items() if name.endswith('.json')}
    summary, contract = data['summary'], data['summary']['contract']
    need(summary['status'] == 'completed' and summary['baseline_passed'] and summary['interpretation_allowed']
         and not summary['automatic_promotion'], 'Run status differs')
    need(contract['source'] == SOURCE and contract['script_sha256'] == DRIVER, 'Published source identity differs')
    for relative, expected in contract['source_files_sha256'].items():
        checked_source(root, relative, expected, overrides)
    for name in runner.SOURCE_MEMBERS:
        need(members[name] == checked_source(root, 'scripts/'+name,
             contract['source_files_sha256']['scripts/'+name], overrides), 'Saved implementation differs')
    need(contract == data['preflight']['contract'] and sha(encoded(contract)) == data['preflight']['contract_sha256'], 'Preflight differs')
    need(data['input-audit'] == data['preflight']['input_audit'] and runner.RECIPE == contract['settings']['recipe'], 'Recipe/input receipt differs')
    need(sha(members['staging-r58.json']) == contract['training_staging_sha256'], 'Staging identity differs')
    refs = load_references(root, contract)
    maps = json.loads(checked_source(root, contract['row_maps']['path'], contract['row_maps']['sha256'], overrides))
    need(json.loads(encoded(prior.audit(original_archive, root))) == data['input-audit'], 'Prior archive audit differs')
    with ZipFile(original_archive) as stream:
        saved = stream.read('evaluation/predictions.json')
        need(sha(saved) == contract['saved_predictions_sha256'] and sha(stream.read('evaluation/summary.json')) == contract['saved_summary_sha256'], 'Saved detector source differs')
        entries = json.loads(stream.read('staging.json'))['roles']['candidate']
        labels = {e['id']: stream.read('datasets/candidate/'+e['label']) for e in entries}
    arms = data['predictions']
    need(arms['control'] == data['detector-evaluation'], 'Control cache differs')
    runner.shared.frozen_detection_check(arms['control'], arms['treatment'])
    for output in [*arms.values(), data['detector-training']]:
        runner.shared.replay_postclip(output)
    runner.replay_transforms(arms['control'], arms['treatment'], data['correction-traces'])
    independent = replay_independent_offsets(arms, data['correction-traces'])
    need(runner.zero_output_gate(arms['control']) == data['zero-output-gate'], 'Zero identity differs')
    gate = r53.baseline_equivalence(refs, json.loads(saved)['candidate'], arms['control'])
    need(gate['passed'] and gate == data['baseline-equivalence'], 'Baseline-equivalence replay differs')
    scored = {arm: r53.reports(refs, output) for arm, output in arms.items()}
    common = {arm: runner.at_confidence(output) for arm, output in arms.items()}
    common_scores = {arm: r53.reports(refs, output) for arm, output in common.items()}
    need(scored == summary['reports'] and common_scores == summary['reports_at_common_confidence_025'], 'Aggregate/per-image scores differ')
    need(data['cases'] == r53.case_inventory(refs, maps, arms, scored), 'Case replay differs')
    need(data['pair-accounting'] == runner.shared.pair_accounting(refs, arms)
         and data['pair-accounting-common025'] == runner.shared.pair_accounting(refs, common), 'Offered-pair accounting differs')
    stage = data['staging-r58']
    by_id = {row['id']: row for row in stage['records']}
    need(len(by_id) == 162 and set(by_id) == {e['id'] for e in entries}, 'Training/development membership differs')
    with tempfile.TemporaryDirectory(prefix='residual-audit-') as temporary:
        work = Path(temporary)
        fronts = known = 0
        for entry in entries:
            row = by_id[entry['id']]
            need(row == {'id': entry['id'], 'split': entry['split'], 'image': 'dataset/'+entry['image'],
                'label': 'dataset/'+entry['label'], 'image_sha256': entry['image_sha256'],
                'label_sha256': entry['label_sha256']}, 'Staged entry differs')
            label = labels[row['id']]
            need(sha(label) == row['label_sha256'], 'Label hash differs')
            path = work/row['label'];path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(label)
            if row['split'] == 'train':
                targets = parse_pose_labels(label.decode(), 1280, 720, row['id'])
                fronts += len(targets)
                known += sum(p[2] == 2 for target in targets for p in target['keypoints'])
        need(fronts == 3999 and known == 7259, 'Training target count differs')
        crop_replay = runner.training_crops({'stage': stage, 'work': work}, data['detector-training'], save=False)
    need(crop_replay == data['training-crops'], 'Training targets/masks/residual distributions differ')
    receipt, history = data['training-receipt'], data['training-log']
    for key, name in [('training_log_sha256', 'training-log.json'), ('training_crops_sha256', 'training-crops.json'), ('preflight_sha256', 'preflight.json')]:
        need(receipt[key] == sha(members[name]), 'Training input receipt differs')
    count = len(crop_replay['records']);steps = math.ceil(count/runner.BATCH)
    need(receipt == summary['training_receipt'] and receipt['recipe'] == runner.RECIPE
         and receipt['epochs'] == 30 and receipt['samples'] == 30*count
         and receipt['updates'] == receipt['expected_updates'] == 30*steps, 'Training budget differs')
    need([row['epoch'] for row in history] == list(range(1, 31)), 'Epoch membership differs')
    for row in history:
        order = torch.randperm(count, generator=torch.Generator().manual_seed(runner.SEED+row['epoch'])).tolist()
        need(sha(encoded([crop_replay['records'][i]['identity'] for i in order])) == row['sample_order_sha256'], 'Epoch order differs')
        rate = runner.RECIPE['minimum_learning_rate']+(runner.RECIPE['learning_rate']-runner.RECIPE['minimum_learning_rate'])*(1+math.cos(math.pi*row['epoch']/30))/2
        need(row['samples'] == count and row['updates'] == steps and math.isfinite(row['mean_loss'])
             and math.isclose(row['next_learning_rate'], rate, rel_tol=1e-12, abs_tol=1e-15), 'Epoch receipt differs')
    detector = data['detector-receipt']
    need(detector['preflight_sha256'] == sha(members['preflight.json']) and detector['runtime'] == data['detector-runtime'], 'Detector runtime receipt differs')
    for name, digest in detector['files'].items():
        need(sha(members[name]) == digest, 'Detector output hash differs')
    model = model_parts(directory, data)
    smoke = data['model-smoke']
    initial_state = '2982de6f862dbbb1c40819b0155475cbdbe842012eaa50851ddf5f200875a959'
    need(smoke['status'] == 'passed' and smoke['initial_offsets_exactly_zero']
         and smoke['original_records_exactly_preserved'] and smoke['unavailable_endpoint_gradient_zero']
         and smoke['head_gradient_finite_nonzero'] and smoke['parameters_changed']
         and smoke['backbone_sha256'] == runner.BACKBONE_SHA
         and smoke['initial_model']['state_sha256'] == initial_state,
         'Pretrained mechanics or published initial state differs')
    need(sha(comparison_archive.read_bytes()) == ORIGINAL_COMPARISON_SHA, 'Completed original comparison differs')
    with ZipFile(comparison_archive) as stream:
        compared = json.loads(gzip.decompress(stream.read('predictions.json.gz')))
    compared_outputs = {'control': common['control'], 'residual': common['treatment'],
        'supplied': {image: {'postclip': rows} for image, rows in compared['original'].items()},
        'proposed': {image: {'postclip': rows} for image, rows in compared['proposed'].items()}}
    paired, rows = pair_comparison(refs, compared_outputs, score_image)
    compact = {reference: {arm: {
        'detection': {key: tables[reference]['postclip']['detection'][key] for key in ('gold_count', 'prediction_count', 'tp', 'fn', 'fp', 'duplicate_fp', 'other_fp')},
        'complete_pairs': tables[reference]['postclip']['corners_by_locatable_count']['2']}
        for arm, tables in common_scores.items()} for reference in refs}
    result = {'schema': 'cassette-residual-return-audit-r60-v1', 'status': 'passed',
        'input': {'pull_request': 155, 'head': RETURN_HEAD, 'archive_bytes': RETURN_BYTES, 'archive_sha256': RETURN_SHA},
        'implementation_commit': SOURCE, 'driver_sha256': DRIVER, 'source_sha256': sha(Path(__file__).read_bytes()),
        'verified_archive_members': len(MEMBERS), 'verified_source_data_files': len(contract['source_files_sha256']),
        'replayed_aggregate_tables': 32, 'replayed_per_image_tables': 640, 'independent_coordinate_replay': independent,
        'baseline_equivalence': gate, 'model': model, 'recorded_runtime': data['detector-runtime'],
        'sample_order_replay_torch': str(torch.__version__), 'training_photographs': 142, 'training_fronts': fronts,
        'known_training_endpoints': known, 'training_crop_counts': crop_replay['counts'], 'usable_training_crops': count,
        'epochs': 30, 'sample_presentations': receipt['samples'], 'updates': receipt['updates'],
        'sample_order_hashes_replayed': 30, 'training_loss_first': history[0]['mean_loss'],
        'training_loss_last': history[-1]['mean_loss'], 'common_confidence_strictly_above': .25,
        'recorded_initial_state_matches_published_bootstrap': True, 'initial_model_state_sha256': initial_state,
        'results': compact, 'paired_comparison': paired, 'all_original_detection_records_preserved': True,
        'automatic_promotion': False, 'model_forward_recomputed': False,
        'interpretation': 'Completed development residual-correction comparison. Main original-baseline methods are supplied and proposed; research control is separate.',
        'verification_limit': 'Saved offsets, geometry and scores replayed independently. Actual model forward replay is recorded separately; no training rerun is claimed.'}
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--original-archive', type=Path, required=True)
    parser.add_argument('--comparison-archive', type=Path, required=True)
    parser.add_argument('--pinned-sources', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result, rows = audit(args.root.resolve(), args.directory.resolve(), args.original_archive.resolve(), args.comparison_archive.resolve(), args.pinned_sources)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'audit.json').write_bytes(encoded(result))
    (args.output_dir/'paired-case-changes.json.gz').write_bytes(gzip.compress(encoded(rows), mtime=0))
    print(json.dumps({'status': result['status'], 'source_files': result['verified_source_data_files'],
        'counts': {ref: {arm: {key: values[key] for key in ('within5', 'within10', 'mean_maximum_error_px')}
                   for arm, values in r['arms'].items()} for ref, r in result['paired_comparison'].items()}}, indent=2))


if __name__ == '__main__':
    main()
