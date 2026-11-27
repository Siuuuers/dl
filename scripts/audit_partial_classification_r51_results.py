#!/usr/bin/env python3
"""Audit the returned PR146 paired training and replay its saved predictions.

This CPU audit verifies archived source, labels, loss/counter receipts and score
calculations. It does not rerun training or infer new predictions from weights.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import urllib.request
import zipfile

import run_partial_classification_r51 as experiment
import audit_additional_front_pilot_r47_results as prior
from evaluate_pose import INFERENCE
from pose_checkpoint import check_training_settings
from pose_scoring import score_image, aggregate_scores, _box_iou
from predict_pose_folder import postclip_nms
from partial_classification_r51 import NATIVE_SOURCE_HASHES

ROOT = Path(__file__).resolve().parents[1]
PR_HEAD = 'b7f64852e3de2a3ac8299783065ad56c03e0dea6'
SOURCE = 'da5a67d4bec8171246c56c6b03b34ee20fc519e0'
DRIVER_SHA = '382b51e48013ec82b30eb1a44175546ae30ce716bd75882646338ae8b48b0852'
ARCHIVE_SHA = '9538932145e90201d407c339b7da1f73c5c905935853191ab9683b221bd5a09e'
ARCHIVE_BYTES = 43528788
STAGING_SHA = 'dcaf95c7dd1e1d39bffae8bebed50068efbdee5f2cd071a5d166787f32348855'
NAME = 'baomai-partial-classification-r51-results.zip'
BASE_URL = f'https://raw.githubusercontent.com/Siuuuers/dl/{PR_HEAD}/results/t4/baomai-partial-classification-r51-t4-001/'
PART_HASHES = (
    '1ed57925e332efeaaaeb82b5b92d723ce64a4ba74d267c1d9454fa706d847ca4',
    '18253e6e7b1a1c0da12f42a9fabe43a0f619dd88a3ae012335a04857fdabb0a0',
    '53f18493b19f5a0b9cf76a1434b6c34e78a7c2dbaf33d377c21166e3ae7cac2e',
)
MODEL_HASHES = {
    'reference': '0a1546050a56e6b1eab0d8555dcafa0f76378c64193a25648fd8d66ff713a3cb',
    'candidate': 'b37738005c47eebe489ab5a06723e2e32b3c50cfdfaae88a178518262bd494ab',
}
need, digest, encoded = experiment.need, experiment.digest, experiment.encoded


def obtain_archive(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / NAME
    if target.exists():
        data = target.read_bytes()
    else:
        parts = []
        for index, expected in enumerate(PART_HASHES):
            member = NAME + f'.part{index:02d}'
            path = directory / (member + '.zip')
            data = path.read_bytes() if path.exists() else urllib.request.urlopen(BASE_URL + path.name, timeout=120).read()
            need(digest(data) == expected, 'Wrapped part differs: ' + path.name)
            if not path.exists():
                path.write_bytes(data)
            with zipfile.ZipFile(io.BytesIO(data)) as wrapped:
                need(wrapped.namelist() == [member] and wrapped.testzip() is None, 'Wrapped part membership/CRC differs')
                parts.append(wrapped.read(member))
        data = b''.join(parts)
    need(len(data) == ARCHIVE_BYTES and digest(data) == ARCHIVE_SHA, 'PR146 archive differs')
    if not target.exists():
        target.write_bytes(data)
    return target


def check_source(root, staging, archive):
    closure = staging['source_files_sha256']
    need(closure == experiment.local_source_files(root), 'Current helper closure differs from the trained source')
    listing = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', SOURCE, '--', *closure], text=True)
    objects = {line.split('\t', 1)[1]: line.split()[2] for line in listing.splitlines()}
    need(set(objects) == set(closure), 'Published source tree membership differs')
    for name, expected in closure.items():
        data = (root / name).read_bytes()
        need(digest(data) == expected and archive.read('source/' + name) == data, 'Archived/local source differs: ' + name)
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        need(blob == objects[name], 'Source bytes differ from published Git object: ' + name)
    need(closure['scripts/run_partial_classification_r51.py'] == DRIVER_SHA, 'Driver differs')


def corner_state(target, prediction):
    slots = [index for index, point in enumerate(target['keypoints']) if point[2] == 2]
    errors = []
    for index in slots:
        value = None
        if prediction is not None:
            x, y, confidence = prediction['keypoints'][index]
            if x is not None and y is not None and 0 <= x < 1280 and 0 <= y < 720 and confidence >= .5:
                gx, gy, _ = target['keypoints'][index]
                value = math.hypot(x - gx, y - gy)
        errors.append(value)
    success = bool(slots) and all(value is not None for value in errors)
    return dict(locatable_corners=len(slots), matched=prediction is not None,
                errors_px=errors, within5=bool(success and max(errors) <= 5),
                within10=bool(success and max(errors) <= 10))


def case_inventory(gold, predictions, scores):
    cases = dict(reference='r50', prediction_stage='postclip', source_pr=146,
                 numbering='one-based rows and prediction positions within their named arm',
                 recoveries=[], newly_missed=[], remaining_misses=[], corner_status_changes=[],
                 candidate_corner_failures=[], candidate_unmatched=[], per_image=[])
    cases['unmatched_decomposition'] = {
        role: {key: scores[role]['detection'][key] for key in ('fp', 'duplicate_fp', 'other_fp')}
        for role in experiment.ROLES
    }
    cases['unmatched_definition'] = (
        'Formal duplicates are unmatched predictions with IoU >= 0.5 to an already matched same-class target. '
        'Other unmatched predictions are the remaining unmatched outputs; this is not a physical-background verdict.'
    )
    for identity, targets in gold.items():
        arms = {role: predictions[role][identity]['postclip'] for role in experiment.ROLES}
        image_scores = {role: scores[role]['per_image'][identity] for role in experiment.ROLES}
        matches = {role: {m['gold_index']: m['prediction_index'] for m in image_scores[role]['matches']} for role in experiment.ROLES}
        for index, target in enumerate(targets):
            states, displayed = {}, {}
            for role in experiment.ROLES:
                position = matches[role].get(index)
                prediction = arms[role][position] if position is not None else None
                states[role] = corner_state(target, prediction)
                displayed[role] = dict(prediction_number=position + 1 if position is not None else None, prediction=prediction, **states[role])
            row = dict(image=identity, row=index + 1, target=target, **displayed)
            if not states['reference']['matched'] and states['candidate']['matched']:
                cases['recoveries'].append(row)
            if states['reference']['matched'] and not states['candidate']['matched']:
                cases['newly_missed'].append(row)
            if not states['candidate']['matched']:
                cases['remaining_misses'].append(row)
            if states['candidate']['matched'] and states['candidate']['locatable_corners'] and not states['candidate']['within5']:
                cases['candidate_corner_failures'].append(row)
            if any(states['reference'][key] != states['candidate'][key] for key in ('within5', 'within10')):
                cases['corner_status_changes'].append(row)
        duplicates = set(image_scores['candidate']['duplicate_prediction_indices'])
        assigned = set(matches['candidate'])
        for position in image_scores['candidate']['unmatched_prediction_indices']:
            prediction = arms['candidate'][position]
            overlaps = [_box_iou(prediction['bbox_xyxy'], target['bbox_xyxy']) for target in targets]
            best = max(range(len(overlaps)), key=lambda j: (overlaps[j], -j)) if overlaps else None
            cases['candidate_unmatched'].append(dict(image=identity, prediction_number=position + 1,
                category='formal_duplicate' if position in duplicates else 'other_unmatched', prediction=prediction,
                nearest_row=best + 1 if best is not None and overlaps[best] > 0 else None,
                nearest_iou=overlaps[best] if best is not None else 0,
                nearest_row_matched=best in assigned if best is not None and overlaps[best] > 0 else None,
                physical_background_verdict=None))
        cases['per_image'].append(dict(image=identity, **{role: {key: image_scores[role][key] for key in
            ('gold_count', 'prediction_count', 'tp', 'fn', 'fp', 'duplicate_fp', 'other_fp')} for role in experiment.ROLES}))
    cases['counts'] = {key: len(cases[key]) for key in ('recoveries', 'newly_missed', 'remaining_misses',
                        'corner_status_changes', 'candidate_corner_failures', 'candidate_unmatched')}
    cases['remaining_misses_by_locatable_count'] = dict(Counter(str(row['candidate']['locatable_corners'])
                                                            for row in cases['remaining_misses']))
    cases['candidate_corner_failures_by_locatable_count'] = dict(Counter(str(row['candidate']['locatable_corners'])
                                                                      for row in cases['candidate_corner_failures']))
    need(len(cases['candidate_unmatched']) == scores['candidate']['detection']['fp']
         and sum(row['category'] == 'formal_duplicate' for row in cases['candidate_unmatched'])
         == scores['candidate']['detection']['duplicate_fp'], 'Unmatched inventory decomposition differs')
    return cases


def audit(root, archive, prior_archive):
    root, archive = Path(root).resolve(), Path(archive)
    data = archive.read_bytes()
    need(len(data) == ARCHIVE_BYTES and digest(data) == ARCHIVE_SHA, 'PR146 archive differs')
    expected_plan, labels, provenance = experiment.label_plan(root, prior_archive)
    with zipfile.ZipFile(io.BytesIO(data)) as z, zipfile.ZipFile(prior_archive) as old:
        names = z.namelist()
        need(len(names) == len(set(names)) == 519 and z.testzip() is None, 'ZIP membership or CRC differs')
        need(all(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts for name in names), 'Unsafe archive member')
        inventory = json.loads(z.read('inventory.json'))
        need(set(inventory) == set(names) - {'inventory.json'}, 'Inventory membership differs')
        for name, entry in inventory.items():
            value = z.read(name)
            need(entry == dict(bytes=len(value), sha256=digest(value)), 'Inventory bytes differ: ' + name)
        staging = json.loads(z.read('staging.json'))
        need(digest(z.read('staging.json')) == STAGING_SHA and staging['source'] == SOURCE, 'Staging/source differs')
        check_source(root, staging, z)
        expected_plan['source'] = SOURCE  # Source bytes were checked against this immutable Git tree above.
        need(staging == expected_plan, 'Training protocol differs from the published paired plan')
        for name, value in provenance.items():
            need(z.read('inputs/' + name) == value, 'Prior-input provenance differs: ' + name)
        need(z.read('inputs/source-manifest.json') == (root / 'data/manifest.json').read_bytes(),
             'Original input image manifest differs')
        groups = {str(n): 0 for n in (0, 1, 2)}
        for role, entries in staging['roles'].items():
            grouped = {'train': [], 'val': []}
            for entry in entries:
                value = z.read('datasets/' + role + '/' + entry['label'])
                need(value == labels[entry['id']] and digest(value) == entry['label_sha256'], 'Staged label differs')
                parsed = experiment.labels(value, entry['id'])
                grouped[entry['split']].append(parsed)
                if role == 'reference' and entry['split'] == 'train':
                    for target in parsed:
                        groups[str(sum(point[2] == 2 for point in target['keypoints']))] += 1
            need({split: list(experiment.counts(rows)) for split, rows in grouped.items()} == staging['counts'][role], 'Actual label counts differ')
        receipts, controller_checks = {}, {}
        with tempfile.TemporaryDirectory(prefix='r51-return-cpu-') as temporary:
            work = Path(temporary)
            for name in ('staging.json', 'cpu-loader-preflight.json', 'loss-guard.json'):
                (work / name).write_bytes(z.read(name))
            cpu_hashes = experiment.verify_cpu_gates(root, work, staging)
            for role, factor in (('resource-trial', 2), ('reference', 1), ('candidate', 2)):
                receipt = json.loads(z.read(role + '/receipt.json'))
                need(receipt == json.loads(z.read(role + '/progress.json')), 'Progress differs: ' + role)
                arm = 'candidate' if role == 'resource-trial' else role
                trial = role == 'resource-trial'
                need(receipt['source'] == SOURCE and receipt['staging_sha256'] == STAGING_SHA
                     and receipt['role'] == arm and receipt['trial'] is trial and receipt['updated'] is True
                     and receipt['classification_factor'] == factor, 'Run identity differs: ' + role)
                if trial:
                    need(receipt['status'] == 'trial-complete' and receipt['batches'] == receipt['steps'] == 4
                         and receipt['images'] == 16 and receipt['optimizer_batch_indices'] == [0, 1, 2, 3], 'Trial budget differs')
                else:
                    budget = staging['budgets'][role]
                    need(receipt['status'] == 'complete' and receipt['epochs'] == list(range(1, 31))
                         and all(receipt[key] == budget[key] for key in ('batches', 'images', 'steps', 'optimizer_batch_indices')), 'Completed budget differs')
                    need(set(receipt['artifacts']) == {'fit/weights/last.pt', 'fit/results.csv'}, 'Completed artifact membership differs')
                    for name, expected in receipt['artifacts'].items():
                        need(digest(z.read(role + '/' + name)) == expected, 'Completed artifact differs')
                    need(receipt['artifacts']['fit/weights/last.pt'] == MODEL_HASHES[role], 'Checkpoint identity differs')
                    rows = list(csv.DictReader(io.StringIO(z.read(role + '/fit/results.csv').decode())))
                    need([int(row['epoch']) for row in rows] == list(range(1, 31)) and
                         all(math.isfinite(float(value)) for row in rows for value in row.values() if value.strip()), 'CSV completion/numerics differ')
                settings = receipt['settings']
                need(Path(settings['model']).name == 'cassette_lower_epoch173.pt', 'Initialization path differs')
                check_training_settings(settings, experiment.RECIPE, model=settings['model'], data=settings['data'], device='0', project=settings['project'])
                experiment.verify_controller(work, staging, arm, receipt, trial=trial)
                controller_checks[role] = dict(factor=factor, batches=receipt['batches'], images=receipt['images'],
                    steps=receipt['steps'], epochs=len(receipt['epochs']), cpu_preflight_sha256=receipt['cpu_preflight_sha256'],
                    assignment_totals=receipt['classification_controller']['assignment_totals'])
                receipts[role] = receipt
        runtime = receipts['reference']['runtime']
        need(all(receipt['runtime'] == runtime for receipt in receipts.values()), 'Mixed training runtime')
        need(runtime['gpu'] == 'Tesla T4' and runtime['native'] == NATIVE_SOURCE_HASHES
             and runtime['packages']['ultralytics'] == '8.4.182', 'Training runtime/native code differs')
        summary = json.loads(z.read('evaluation/summary.json'))
        predictions = json.loads(z.read('evaluation/predictions.json'))
        need(summary['source'] == SOURCE and summary['staging_sha256'] == STAGING_SHA and summary['runtime'] == runtime
             and summary['model_sha256'] == MODEL_HASHES and summary['treatment'] == staging['treatment']
             and summary['budgets'] == staging['budgets'] and summary['experiment'] == 'partial-classification-r51', 'Evaluation identity differs')
        need(summary['inference'] == {**INFERENCE, 'conf': .13, 'matching_iou': .5, 'endpoint_conf': .5}, 'Inference settings differ')
        validation = {entry['id'] for entry in staging['roles']['reference'] if entry['split'] == 'val'}
        need(set(predictions) == set(experiment.ROLES), 'Prediction arms differ')
        for role in experiment.ROLES:
            need(set(predictions[role]) == validation, 'Prediction image membership differs')
            for record in predictions[role].values():
                kept, filtering = postclip_nms(record['raw'])
                need(kept == record['postclip'] and filtering == record['filter'], 'Postclip reconstruction differs')
        need(predictions['reference'] == json.loads(old.read('evaluation/predictions.json'))['candidate'], 'Reference differs from the prior PR137 candidate cache')
        refs = experiment.read_refs(root, staging['roles']['reference'])
        need(set(summary['references']) == set(refs), 'Reference membership differs')
        compact = {}
        for reference, (manifest_sha, gold) in refs.items():
            relative, _, _ = experiment.REFS[reference]
            manifest = root / relative
            need(z.read('references/' + reference + '/manifest.json') == manifest.read_bytes(), 'Reference manifest differs')
            for entry in json.loads(manifest.read_text())['files']:
                if entry.get('split', 'val') == 'val':
                    need(z.read('references/' + reference + '/' + entry['label_path']) == (manifest.parent / entry['label_path']).read_bytes(), 'Reference label differs')
            report = summary['references'][reference]
            need(report['manifest_sha256'] == manifest_sha, 'Reference hash differs')
            compact[reference] = dict(manifest_sha256=manifest_sha)
            for view in ('raw', 'postclip'):
                compact[reference][view] = {}
                for role in experiment.ROLES:
                    scores, strata = {}, {str(n): Counter() for n in (0, 1, 2)}
                    for identity, targets in gold.items():
                        pred = predictions[role][identity][view]
                        scores[identity] = score_image(targets, pred, 1280, 720, .5, .5)
                        for key, value in experiment.r45.corners(targets, pred, scores[identity]).items():
                            strata[key].update(value)
                    result = dict(detection=aggregate_scores(list(scores.values())), corners_by_locatable_count=strata, per_image=scores)
                    need(result == report[view][role], 'Score/assignment replay differs: ' + '/'.join((reference, view, role)))
                    compact[reference][view][role] = dict(detection={k: v for k, v in result['detection'].items() if k != 'endpoint_errors_px'},
                        corners_by_locatable_count=strata,
                        missed_rows={identity: [j + 1 for j in score['unmatched_gold_indices']] for identity, score in scores.items() if score['fn']},
                        empty_scene_outputs={identity: score['prediction_count'] for identity, score in scores.items() if not score['gold_count']})
                delta = {}
                for identity in gold:
                    before, after = (set(compact[reference][view][role]['missed_rows'].get(identity, [])) for role in experiment.ROLES)
                    delta[identity] = dict(recovered_rows=sorted(before - after), newly_missed_rows=sorted(after - before))
                need(delta == report[view]['identity_delta'], 'Changed miss identities differ')
                compact[reference][view]['identity_delta'] = {identity: value for identity, value in delta.items() if any(value.values())}
        cases = case_inventory(refs['r50'][1], predictions, summary['references']['r50']['postclip'])
    verification = dict(schema='pr146-r51-return-cpu-audit-v1', status='passed', pr=146, pr_head=PR_HEAD,
        source=SOURCE, driver_sha256=DRIVER_SHA, auditor_sha256=digest(Path(__file__).read_bytes()),
        archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES, inventory_entries_verified=518,
        source_files_verified=len(staging['source_files_sha256']), staging_sha256=STAGING_SHA,
        prior_archive_sha256=prior.ARCHIVE_SHA, full_training_protocol_equals_published_plan=True,
        exact_same_prior_candidate_inputs_both_arms=True, source_bytes_match_published_git_tree=True,
        cpu_preflight_sha256=cpu_hashes, counts=staging['counts'], training_corner_groups=groups,
        runtime=runtime, model_sha256=MODEL_HASHES, run_checks=controller_checks,
        replayed_postclip_filters=40, replayed_score_tables=16, replayed_image_score_reports=320,
        prior_candidate_reference_predictions_exact=dict(prior_pr=137, prior_role='candidate', current_role='reference',
            images=20, views=['raw', 'postclip', 'filter'], exact_parsed_values=True, checkpoint_byte_identity_claimed=False),
        scope='Archive/source/label/receipt integrity and exact saved-score replay. No new inference, training, or independent fresh image-loader execution. Original image identity is bound by the audited source manifests and archived actual-loader receipts.',
        screen_discrepancy='Independent historical663 reference replay is 633 TP and 30 FN out of 663; the README-reported screen value 31 is not used.',
        automatic_promotion=False, fresh_training=False, fresh_inference=False)
    result = dict(schema='pr146-r51-results-v1', source_pr=146, archive_sha256=ARCHIVE_SHA,
        primary_reference='frozen666', reports=compact,
        conclusion='The classification-only treatment recovers four R38/R50 partial fronts but increases unmatched outputs by 28; it is a tradeoff, not a clean improvement or automatic promotion.',
        interpretation='All four references reuse the same development photographs. Candidate inference contains no full decoded tensor trace, so prior R49 stage labels cannot be asserted as the remaining candidate misses current causes.')
    return {'verification.json': verification, 'summary.json': result, 'case-inventory.json': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--prior-archive', type=Path)
    parser.add_argument('--download-dir', type=Path, default=Path('/tmp/pr146-r51-return'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive or obtain_archive(args.download_dir)
    previous = args.prior_archive or prior.obtain_archive(args.download_dir / 'pr137')
    results = audit(args.root, archive, previous)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in results.items():
        data, path = encoded(value), args.output / name
        need(not path.exists() or path.read_bytes() == data, 'Different existing audit preserved: ' + str(path))
        path.write_bytes(data)
    print(json.dumps({'status': 'passed', 'output': str(args.output), 'cases': results['case-inventory.json']['counts']}, indent=2))


if __name__ == '__main__':
    main()
