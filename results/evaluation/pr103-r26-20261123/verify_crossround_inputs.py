#!/usr/bin/env python3
"""Audit R25/R26 staged inputs and the L4/A100 matched-control reproducibility risk."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--r25-bundle', type=Path, required=True)
    parser.add_argument('--r26-bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    bundles = {'r25': args.r25_bundle.resolve(), 'r26': args.r26_bundle.resolve()}
    indexes = {name: read(path / 'gpu-results-manifest.json') for name, path in bundles.items()}
    a, b = indexes.values()
    assert a['shared_training_settings'] == b['shared_training_settings']
    runtime_changes = {key: {'r25': a['runtime'].get(key), 'r26': b['runtime'].get(key)}
                       for key in sorted(a['runtime'].keys() | b['runtime'].keys())
                       if a['runtime'].get(key) != b['runtime'].get(key)}
    source_a = a['training_source']['files_sha256']
    source_b = b['training_source']['files_sha256']
    report = {
        'script_sha256': sha(__file__),
        'recorded_training_runtime_equal': a['runtime'] == b['runtime'],
        'recorded_training_runtimes': {name: index['runtime'] for name, index in indexes.items()},
        'recorded_runtime_differences': runtime_changes,
        'training_settings_equal': True,
        'shared_training_settings': a['shared_training_settings'],
        'recorded_source_file_differences': {path: {'r25': source_a.get(path), 'r26': source_b.get(path)}
            for path in sorted(source_a.keys() | source_b.keys()) if source_a.get(path) != source_b.get(path)},
        'roles': {},
    }
    for role in ('reference', 'candidate'):
        manifests = {name: read(path / role / 'staging-manifest.json') for name, path in bundles.items()}
        files = {name: {f['path']: f for f in m['variants'][role]['files']} for name, m in manifests.items()}
        assert files['r25'].keys() == files['r26'].keys()
        receipts = {name: read(path / role / 'training-receipt.json') for name, path in bundles.items()}
        compared_fields = ('recipe_id', 'recipe_sha256', 'initial_checkpoint_sha256', 'checkpoint_selection',
                           'completed_epochs', 'optimizer_steps', 'batches_completed', 'images_processed')
        for field in compared_fields:
            assert receipts['r25'][field] == receipts['r26'][field], field
        changes = []
        for path, old in files['r25'].items():
            new = files['r26'][path]
            for item in (old, new):
                assert sha(root / item['source_path']) == item['source_sha256'] == item['output_sha256']
            if old['output_sha256'] == new['output_sha256']:
                continue
            assert path.startswith('labels/train/')
            oldlines = (root / old['source_path']).read_text().splitlines()
            newlines = (root / new['source_path']).read_text().splitlines()
            assert len(oldlines) == len(newlines)
            rows = []
            for row, (x, y) in enumerate(zip(oldlines, newlines), 1):
                if x == y:
                    continue
                xx, yy = x.split(), y.split()
                assert len(xx) == len(yy) == 11
                changed = [i for i in range(11) if xx[i] != yy[i]]
                assert set(changed) <= {1, 3, 5, 6}
                assert any(i in changed for i in (5, 6))
                rows.append({'row': row, 'before': x, 'after': y, 'changed_fields': changed,
                             'left_point_changed': True, 'box_changed': any(i in changed for i in (1, 3))})
            changes.append({'path': path, 'r25_source': old['source_path'], 'r26_source': new['source_path'],
                            'r25_sha256': old['output_sha256'], 'r26_sha256': new['output_sha256'], 'rows': rows})
        report['roles'][role] = {'changes': changes,
                                'equal_receipt_fields': {field: receipts['r26'][field] for field in compared_fields},
                                'files_verified_per_arm': len(files['r25'])}
    assert not report['roles']['reference']['changes']
    changes = report['roles']['candidate']['changes']
    assert [change['path'] for change in changes] == ['labels/train/new_000013.txt']
    assert [row['row'] for row in changes[0]['rows']] == [29, 30, 31]
    assert [row['row'] for row in changes[0]['rows'] if row['box_changed']] == [29, 30]
    report.update({'changed_training_files': 1, 'changed_training_rows': 3,
                   'changed_training_left_endpoints': 3, 'changed_training_boxes': 2,
                   'original_images_and_trainer_validation_unchanged': True,
                   'development_review_changes_not_used_for_training': True})
    import torch
    states = {}
    for name, path in bundles.items():
        checkpoint = path / 'reference/fit/weights/last.pt'
        manifest = read(path / 'reference/checkpoint-manifest.json')
        assert sha(checkpoint) == manifest['checkpoint']['sha256']
        data = torch.load(checkpoint, map_location='cpu', weights_only=False)
        states[name] = (data.get('ema') or data['model']).state_dict()
    x, y = states.values()
    assert list(x) == list(y)
    different = []
    for name in x:
        assert x[name].shape == y[name].shape and x[name].dtype == y[name].dtype
        if torch.equal(x[name], y[name]):
            continue
        delta = (x[name].float() - y[name].float()).abs()
        different.append({'name': name, 'shape': list(x[name].shape), 'dtype': str(x[name].dtype),
                          'max_absolute_difference': float(delta.max()),
                          'mean_absolute_difference': float(delta.mean())})
    report['matched_control_tensors'] = {
        'compared': len(x), 'identical': len(x) - len(different), 'changed_count': len(different),
        'all_identical': not different, 'changed': different,
        'interpretation': ('The same control supervision and recipe produced different saved tensors across these recorded runtimes; '
                           'a cross-round candidate difference cannot be attributed solely to the three label corrections.'
                           if different else 'The matched controls have identical saved tensors despite different recorded hardware. '
                           'This limits observed control drift, but does not prove hardware invariance for the changed candidate.'),
    }
    report['limits'] = [
        'R26 versus its own control tests accumulated eligible supervision in the same recorded A100 runtime.',
        'R25-to-R26 changes three training rows and the GPU architecture. This is not an isolated three-row causal trial.',
        'Identical software versions, deterministic settings, and a shared seed do not by themselves establish identical training arithmetic across devices.',
        'Scoring all returned models in one runtime removes an inference-environment mismatch; it does not remove their training-runtime difference.',
        'A single pair per round is descriptive evidence, not a population-level estimate.',
    ]
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'changed_training_rows': 3, 'changed_training_points': 3, 'changed_training_boxes': 2,
                      'recorded_runtime_differences': runtime_changes,
                      'control_tensors_compared': len(x), 'control_tensors_changed': len(different)}, indent=2))


if __name__ == '__main__':
    main()
