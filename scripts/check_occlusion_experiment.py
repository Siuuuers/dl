#!/usr/bin/env python3
"""Check occlusion artifact eligibility with explicit synthetic fixtures; never train."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON report path')
    args = parser.parse_args()
    ROOT = args.repo_root.resolve()
    if ROOT != Path(__file__).resolve().parents[1]:
        raise ValueError('--repo-root must identify the executing checkout')
    if args.output.exists() or args.output.is_symlink():
        raise ValueError('--output must be a fresh regular file path')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from check_pose_checkpoint import bundle_fixture
    from evaluate_pose import file_sha256
    from occlusion_experiment import (experiment_identity, load_protocol, metadata_preflight,
                                     package_pair, source_identity, verified_label_inputs, verify_run)
    from pose_checkpoint import write_checkpoint_manifest
    from pose_input_corruption import PoseInputCorruption
    from stage_pose_training import yaml_bytes
    from train_pose_comparison import descriptor


    def read(path):
        return json.loads(path.read_text())


    def write(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


    context = load_protocol(ROOT)
    document = context['document']
    labels = verified_label_inputs(ROOT, context)
    preflight = metadata_preflight(ROOT)
    actual_source = source_identity(ROOT)
    source = {**actual_source, 'working_tree_status': ''}
    before_policy = file_sha256(ROOT / 'docs/annotation-policy.md')
    with tempfile.TemporaryDirectory(prefix='occlusion-synthetic-package-') as temporary:
        workspace = Path(temporary)
        template_dir = workspace / 'template'
        template_dir.mkdir()
        _, template, _, base_receipt, base_staging, _ = bundle_fixture(template_dir, recipe_id=document['recipe_id'])
        source_binding = {
            'reference_manifest_sha256': document['source_manifest_sha256'],
            'model_card_sha256': preflight['model_card_sha256'],
            'initial_checkpoint_sha256': preflight['model_sha256'],
            'candidate_id': document['candidate_id'],
            'candidate_manifest': document['candidate_manifest'],
            'candidate_manifest_sha256': document['candidate_manifest_sha256'],
        }


        def make_run(role):
            run = workspace / role
            shutil.copytree(template, run)
            (run / 'checkpoint-manifest.json').unlink()
            (run / 'fit/weights/last.pt').write_bytes(('SYNTHETIC ' + role + '; NEVER DESERIALIZE; NO TRAINING\n').encode())
            (run / 'train.log').write_text('SYNTHETIC COMPLETED-RECEIPT FIXTURE. No GPU, training, or inference was performed.\n')
            stage = copy.deepcopy(base_staging)
            stage['fixture_only'] = 'Synthetic receipt validation; no source PNG bytes copied.'
            stage['output_root'] = str(run / 'staged-data')
            stage['source'] = source_binding
            for arm in ('reference', 'candidate'):
                files = []
                for record in labels['records']:
                    for kind in ('image', 'label'):
                        digest = (labels['labels'][record['id']][1] if arm == 'candidate' and kind == 'label'
                                  else record[kind + '_sha256'])
                        files.append({'path': record[kind], 'source_sha256': digest, 'output_sha256': digest})
                stage['variants'][arm] = {
                    'data_yaml': arm + '/data.yaml',
                    'data_yaml_sha256': hashlib.sha256(yaml_bytes(run / 'staged-data' / arm)).hexdigest(),
                    'counts': labels['candidate_counts'] if arm == 'candidate' else labels['original_counts'],
                    'files': files,
                }
            write(run / 'staging-manifest.json', stage)
            receipt = copy.deepcopy(base_receipt)
            receipt['fixture_only'] = 'SYNTHETIC: NO GPU, TRAINING, MODEL DESERIALIZATION, OR ACCURACY EVIDENCE.'
            for field in ('reference_manifest_sha256', 'model_card_sha256', 'initial_checkpoint_sha256'):
                receipt[field] = source_binding[field]
            receipt['staging_manifest_sha256'] = file_sha256(run / 'staging-manifest.json')
            receipt['data_yaml_sha256'] = stage['variants']['candidate']['data_yaml_sha256']
            receipt['source'] = source
            receipt['training_settings'].update(model=preflight['weights'], data=str(run / 'staged-data/candidate/data.yaml'),
                                                device='0', project=str(run), save_dir=str(run / 'fit'))
            receipt['runtime'].update(torch='2.10.0+cu128', torch_cuda='12.8', device='cuda:0',
                                      device_name='SYNTHETIC_GPU_FIXTURE_NO_TRAINING', device_capability=[8, 9])
            receipt['runtime']['backend']['cudnn_version'] = 91000
            receipt['experiment'] = experiment_identity(context, role)
            receipt['images_processed'], receipt['batches_completed'] = 4110, 1050
            state = PoseInputCorruption(document['augmentation'], role).state()
            state['counters'].update(batches_seen=1050, images_seen=4110, images_planned=1500,
                                     images_applied=1500 if role == 'candidate' else 0,
                                     empty_images=390, probability_skipped=1900, geometry_rejected=320,
                                     source_missing_slots_seen=21030, known_endpoints_covered_planned=2200,
                                     collateral_known_endpoints_planned=700,
                                     known_endpoints_covered_applied=2200 if role == 'candidate' else 0,
                                     collateral_known_endpoints_applied=700 if role == 'candidate' else 0)
            state['examples'] = [{'status': 'planned', 'fixture_only': True, 'example': i} for i in range(8)]
            state['decision_sha256'] = hashlib.sha256(b'SYNTHETIC identical paired mask trace').hexdigest()
            receipt['input_corruption'] = state
            for field, relative, checkpoint in [('checkpoint', 'fit/weights/last.pt', True),
                                                 ('results', 'fit/results.csv', False), ('log', 'train.log', False)]:
                receipt[field] = descriptor(run / relative, run, checkpoint=checkpoint)
            write(run / 'training-receipt.json', receipt)
            write_checkpoint_manifest(run, preflight)
            return run


        control, candidate = make_run('control'), make_run('candidate')
        checks = []
        archive = workspace / 'SYNTHETIC_ONLY_results.zip'
        result = package_pair(ROOT, control, candidate, archive)
        assert result['status'] == 'verified' and not result['accuracy_evaluated']
        checks.append({'case': 'valid_synthetic_completed_pair', 'result': 'accepted'})
        assert package_pair(ROOT, control, candidate, archive, verify_existing=True)['sha256'] == result['sha256']
        checks.append({'case': 'existing_archive_exact_verification', 'result': 'accepted'})


        def reject(name, operation):
            try:
                operation()
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                checks.append({'case': name, 'result': 'rejected', 'error_type': type(error).__name__, 'reason': str(error)})
            else:
                raise AssertionError('INVALID FIXTURE WAS ACCEPTED: ' + name)


        reject('swapped_experimental_roles', lambda: package_pair(ROOT, candidate, control, workspace / 'swapped.zip'))


        def changed(name, mutate):
            run = workspace / ('invalid-' + name)
            shutil.copytree(candidate, run)
            receipt, staging, manifest = (read(run / n) for n in ['training-receipt.json', 'staging-manifest.json', 'checkpoint-manifest.json'])
            mutate(receipt, staging)
            write(run / 'staging-manifest.json', staging)
            receipt['staging_manifest_sha256'] = file_sha256(run / 'staging-manifest.json')
            write(run / 'training-receipt.json', receipt)
            manifest['training_receipt'] = descriptor(run / 'training-receipt.json', run)
            manifest['staging_manifest'] = descriptor(run / 'staging-manifest.json', run)
            write(run / 'checkpoint-manifest.json', manifest)
            reject(name, lambda: verify_run(ROOT, run, 'candidate', context))


        changed('missing_hook_evidence', lambda r, s: r.pop('input_corruption'))
        changed('changed_protocol_identity', lambda r, s: r['experiment'].update(protocol_sha256='f' * 64))


        def wrong_label(receipt, staging):
            entry = next(f for f in staging['variants']['candidate']['files'] if f['path'].endswith('.txt'))
            entry['source_sha256'] = entry['output_sha256'] = 'f' * 64


        changed('rehashed_wrong_training_label', wrong_label)


        def cpu_full(receipt, staging):
            receipt['training_settings']['device'] = 'cpu'
            receipt['runtime'].update(device='cpu', device_name='cpu', device_capability=None)


        changed('full_cpu_receipt', cpu_full)


        def partial(receipt, staging):
            receipt['images_processed'], receipt['batches_completed'] = 4, 1
            state = receipt['input_corruption']
            state['counters'] = dict.fromkeys(state['counters'], 0)
            state['counters'].update(images_seen=4, batches_seen=1, images_planned=1, images_applied=1,
                                     probability_skipped=3, known_endpoints_covered_planned=1, known_endpoints_covered_applied=1)
            state['examples'] = state['examples'][:1]


        changed('partial_hook_exposure_claimed_complete', partial)
        assert file_sha256(ROOT / 'docs/annotation-policy.md') == before_policy
        report = {'scope': 'SYNTHETIC FIXTURES ONLY. No GPU, model deserialization, training, inference, source-image validation, or accuracy claim.',
                  'status': 'passed', 'checks': checks, 'count': len(checks), 'protocol_sha256': context['sha256'],
                  'source_files_sha256': actual_source['files_sha256'], 'live_policy_unchanged': True,
                  'harness_sha256': file_sha256(Path(__file__))}
        report_path = args.output.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write(report_path, report)
        print(json.dumps({'report': str(report_path), 'count': len(checks), 'status': 'passed', 'checks': checks}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
