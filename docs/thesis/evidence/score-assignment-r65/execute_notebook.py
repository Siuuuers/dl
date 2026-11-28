"""Execute the four real published-pin notebook cells unchanged in one shell."""
import argparse
from collections import Counter
from zoneinfo import ZoneInfo
import gzip
import re
import contextlib
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile

import IPython
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextlib.contextmanager
def capture_fds(path):
    sys.stdout.flush()
    sys.stderr.flush()
    previous = [os.dup(1), os.dup(2)]
    with path.open('wb') as log:
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(previous[0], 1)
            os.dup2(previous[1], 2)
            os.close(previous[0])
            os.close(previous[1])


MODEL_HASHES = {
    'reference': '15d7c97d5c58f16e40462807231c7bb01a5ee9a719225cb756493e1b41714d67',
    'candidate': '8760602df814f47685dde25a2e389a554f623a7280473c90d5e94a0f30fe519f',
}
INPUT_ARCHIVE_SHA = 'cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a'
RETURN_MANIFEST_SHA = '8e9b60725a701f79048435772df4fdc07385077dc8d5efd7049833ababba3473'
RESULT_FILES = ('summary.json', 'targets.json.gz', 'images.json', 'groups.csv', 'results.md', 'manifest.json')
HEADS = ('one2many', 'one2one')
MODES = ('native', 'box_only_counterfactual')


def verify_cache(cache):
    expected = {role + '-last.pt': digest for role, digest in MODEL_HASHES.items()}
    expected.update({'return-manifest.json': RETURN_MANIFEST_SHA,
                     'baomai-quality-focal-r61-results.zip': INPUT_ARCHIVE_SHA})
    observed = {}
    for name, digest in expected.items():
        data = (cache / 'quality-focal-pr157' / name).read_bytes()
        check(sha(data) == digest, 'Cached input changed: ' + name)
        observed[name] = {'bytes': len(data), 'sha256': digest}
    return observed


def verify_package(base, expected_source, expected_driver):
    source, work = base / 'source', base / 'work'
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    check(commit == expected_source and not status, 'Final source identity or cleanliness differs')
    driver = source / 'scripts/audit_score_assignment_r65.py'
    check(sha(driver.read_bytes()) == expected_driver, 'Published driver identity differs')
    archive_path = work / 'baomai-score-assignment-r65-results.zip'
    expected_names = {'plan.json', 'inputs/pr157-return-manifest.json',
                      'source/audit_score_assignment_r65.py', 'smoke/summary.json',
                      *('results/' + name for name in RESULT_FILES)}
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        check(len(names) == len(set(names)) and set(names) == expected_names, 'ZIP membership differs')
        check(archive.testzip() is None, 'Result ZIP CRC failed')
        members = {name: archive.read(name) for name in sorted(names)}
    for name, data in members.items():
        actual = driver if name == 'source/audit_score_assignment_r65.py' else work / name
        check(data == actual.read_bytes(), 'Packaged/source bytes differ: ' + name)
    manifest = json.loads(members['results/manifest.json'])
    check(set(manifest) == set(RESULT_FILES) - {'manifest.json'}, 'Result manifest membership differs')
    for name, item in manifest.items():
        data = members['results/' + name]
        check(item == {'bytes': len(data), 'sha256': sha(data)}, 'Manifest mismatch: ' + name)
    plan = json.loads(members['plan.json'])
    summary = json.loads(members['results/summary.json'])
    smoke = json.loads(members['smoke/summary.json'])
    images = json.loads(members['results/images.json'])
    rows = json.loads(gzip.decompress(members['results/targets.json.gz']))
    check(plan['source'] == summary['source'] == smoke['source'], 'Plan/result sources disagree')
    check(plan['source']['commit'] == expected_source and plan['source']['driver_sha256'] == expected_driver,
          'Result source pin differs')
    tree = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-r', 'HEAD'], text=True)
    blobs = {line.split('\t', 1)[1]: line.split()[2] for line in tree.splitlines()}
    for name, digest in plan['source']['files_sha256'].items():
        data = (source / name).read_bytes()
        check(sha(data) == digest, 'Source closure SHA differs: ' + name)
        check(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blobs.get(name),
              'Source closure Git object differs: ' + name)
    check(plan['counts'] == [142, 3999, 7259, 739, 13], 'Training counts differ')
    check(plan['models_sha256'] == summary['models_sha256'] == smoke['models_sha256'] == MODEL_HASHES,
          'Checkpoint identities differ')
    check(sha(members['inputs/pr157-return-manifest.json']) == RETURN_MANIFEST_SHA, 'Return manifest changed')
    check(plan['input_archive']['sha256'] == INPUT_ARCHIVE_SHA, 'Input archive identity differs')
    check(summary['plan_sha256'] == smoke['plan_sha256'] == sha(members['plan.json']), 'Plan SHA differs')
    check(plan['protocol'] == summary['protocol'] == smoke['protocol'], 'Protocol differs')
    check(summary['protocol']['training'] is False and summary['protocol']['optimizer_updates'] == 0
          and summary['protocol']['annotation_edits'] is False, 'Training or annotation scope changed')
    check(summary['completed'] and summary['full142_completed'] and summary['scope'] == 'all142_training_photos'
          and summary['photographs_per_model'] == 142 and summary['forward_calls'] == 284
          and summary['source_fronts_per_model'] == 3999, 'Full audit incomplete or wrong scope')
    check(smoke['completed'] and not smoke['full142_completed'] and smoke['scope'] == 'three_photo_smoke'
          and smoke['photographs_per_model'] == 3 and smoke['forward_calls'] == 6, 'Smoke scope wrong')
    check(summary['runtime']['device'] == 'cpu', 'Expected CPU for this execution receipt')
    check(summary['runtime']['native_files_sha256'] == smoke['runtime']['native_files_sha256'], 'Native source mismatch')
    check(summary['runtime']['deterministic_algorithms'] and not summary['runtime']['tf32_matmul']
          and not summary['runtime']['tf32_cudnn'], 'Arithmetic scope differs')
    identities = {entry['id'] for entry in plan['entries']}
    check(len(identities) == len(plan['entries']) == 142, 'Image identities duplicated or missing')
    target_lineage, strata = {}, Counter()
    for entry in plan['entries']:
        check(sha((work / 'dataset' / entry['image']).read_bytes()) == entry['image_sha256'], 'Photo changed')
        data = (work / 'dataset' / entry['label']).read_bytes()
        check(sha(data) == entry['label_sha256'], 'Label changed')
        label_rows = [line.split() for line in data.decode().splitlines() if line.strip()]
        for index, fields in enumerate(label_rows, 1):
            check(len(fields) == 11, 'Unexpected label format')
            known = sum(float(fields[k]) > 0 for k in (7, 10))
            target_lineage[entry['id'], index] = known
            strata[known] += 1
    check(len(target_lineage) == 3999 and strata == {0: 137, 1: 465, 2: 3397}, 'Source strata differ')
    expected_images = {(role, head, identity) for role in MODEL_HASHES for head in HEADS for identity in identities}
    check(len(images) == 568 and {(r['role'], r['branch'], r['image_id']) for r in images} == expected_images,
          'Per-image/head membership differs')
    check(all(r['input_shape'] == [1, 3, 736, 1280] for r in images), 'Input geometry differs')
    check(len(rows) == 31992, 'Target record count differs')
    seen, group_counts = set(), Counter()
    for row in rows:
        key = row['role'], row['branch'], row['mode'], row['image_id'], row['source_row']
        check(key not in seen, 'Duplicate target record')
        seen.add(key)
        check(row['role'] in MODEL_HASHES and row['branch'] in HEADS and row['mode'] in MODES, 'Unknown mode')
        check(target_lineage.get((row['image_id'], row['source_row'])) == row['known_corners'], 'Lineage differs')
        group_counts[key[:3] + (row['known_corners'],)] += 1
    check(len(summary['groups']) == 24, 'Missing summary groups')
    for group in summary['groups']:
        key = group['role'], group['branch'], group['mode'], group['known_corners']
        check(group['targets'] == group_counts[key] == strata[group['known_corners']], 'Summary stratum mismatch')
    for role, digest in MODEL_HASHES.items():
        check(sha((work / 'models' / (role + '-last.pt')).read_bytes()) == digest, 'Staged checkpoint changed')
    for result, name in ((summary, 'full'), (smoke, 'smoke')):
        check(set(result['invariants']) == set(MODEL_HASHES), 'Frozen-state role coverage differs')
        for role, state in result['invariants'].items():
            check(re.fullmatch('[0-9a-f]{64}', state['parameter_and_buffer_sha256_before'])
                  and state['parameter_and_buffer_sha256_before'] == state['parameter_and_buffer_sha256_after']
                  and state['all_modules_eval'] and state['all_parameter_gradients_none'], 'Frozen model changed')
            for head in HEADS:
                guards = [g for g in state['guards'] if g['branch'] == head]
                check({str(c) for g in guards for c in g['categories']} == {'0', '1', '2', 'empty'}
                      and all(g['native_inputs_exact'] and g['native_returned_assignments_exact'] for g in guards),
                      'Native guard coverage differs: ' + name + '/' + role + '/' + head)
    package = {
        'path': archive_path.name, 'bytes': archive_path.stat().st_size, 'sha256': sha(archive_path.read_bytes()),
        'member_count': len(names), 'member_set_exact': True, 'CRC': 'passed',
        'members': {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in members.items()},
        'manifest_hashes': 'All declared member sizes and SHA256 independently verified.',
        'package_disk_identity': 'Every member equals its corresponding source or work file.',
        'source_closure_files_verified': len(plan['source']['files_sha256']),
        'full_image_head_records_verified': len(images), 'full_target_records_verified': len(rows),
        'training_images_verified': len(identities), 'source_fronts_verified': len(target_lineage),
        'source_corner_strata': dict(sorted(strata.items())),
    }
    return archive_path, package, summary, smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--root', type=Path, default=Path('/workspace/scratch/f5388ec04e55/dl'))
    parser.add_argument('--base', type=Path, default=Path('/workspace/scratch/f5388ec04e55/r65-score-notebook-run'))
    args = parser.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    notebook_path = root / 'notebooks/baomai_score_assignment_r65.ipynb'
    evidence = root / 'docs/thesis/evidence/score-assignment-r65'
    check(not base.exists(), 'Fresh base already exists; preserve it and select another base')
    check(not (evidence / 'notebook-execution.json').exists(), 'Existing execution receipt preserved')
    payload = notebook_path.read_bytes()
    notebook = json.loads(payload)
    check(notebook['metadata']['source_commit'] == args.source_commit, 'Notebook source pin differs from expected publication')
    check(sum(cell['cell_type'] == 'code' for cell in notebook['cells']) == 4, 'Expected exactly four actual code cells')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<actual-notebook-cell>', 'exec')
    os.environ['BAOMAI_SCORE_ASSIGNMENT_BASE'] = str(base)
    os.environ['BAOMAI_R61_CACHE'] = str(root.parent / 'r63-cache')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    evidence.mkdir(parents=True, exist_ok=True)
    executed = copy.deepcopy(notebook)
    shell = InteractiveShell.instance()
    receipt = {
        'schema': 'cassette-score-assignment-notebook-execution-r65-v1', 'status': 'running',
        'started_at': datetime.datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(), 'timezone': 'Asia/Hong_Kong',
        'notebook': {'path': str(notebook_path.relative_to(root)), 'sha256': sha(payload), 'bytes': len(payload)},
        'source_commit': args.source_commit,
        'execution': 'All four actual code cells, unchanged, sequentially in one IPython InteractiveShell namespace.',
        'environment': {'python': platform.python_version(), 'executable': sys.executable,
                        'platform': platform.platform(), 'IPython': IPython.__version__, 'device': 'CPU'},
        'fresh_base_directory': str(base), 'source_checkout_initially_absent': True,
        'fresh_frozen_model_forwards': True, 'training': False, 'optimizer_updates': 0, 'annotation_edits': False, 'cells': [],
        'runner_sha256': sha(Path(__file__).read_bytes()),
        'verified_cache_before': verify_cache(root.parent / 'r63-cache'),
        'limitations': ['Executed locally in IPython, not in a hosted Google Colab runtime.',
                        'The Colab browser download interface is not exercised; the non-Colab FileLink output is captured.',
                        'Frozen unaugmented training-image diagnostics, not historical training replay or recognition accuracy evaluation.'],
    }
    try:
        code_number = 0
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            code_number += 1
            code = ''.join(cell['source'])
            log_path = evidence / f'notebook-cell-{code_number}.txt'
            start = time.monotonic()
            with capture_fds(log_path):
                with capture_output(stdout=True, stderr=True, display=True) as captured:
                    result = shell.run_cell(code, store_history=False)
            external = log_path.read_text(errors='replace')
            combined = external + captured.stdout
            outputs = []
            if combined:
                outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': combined.splitlines(keepends=True)})
            if captured.stderr:
                outputs.append({'output_type': 'stream', 'name': 'stderr', 'text': captured.stderr.splitlines(keepends=True)})
            for rich in captured.outputs:
                outputs.append({'output_type': 'display_data', 'data': rich.data, 'metadata': rich.metadata or {}})
            error = result.error_before_exec or result.error_in_exec
            if error:
                outputs.append({'output_type': 'error', 'ename': type(error).__name__, 'evalue': str(error),
                                'traceback': traceback.format_exception(type(error), error, error.__traceback__)})
            executed['cells'][index]['execution_count'] = code_number
            executed['cells'][index]['outputs'] = outputs
            log_path.write_text(combined + captured.stderr)
            receipt['cells'].append({
                'code_cell': code_number, 'notebook_cell_index': index, 'source_sha256': sha(code.encode()),
                'source_unchanged': executed['cells'][index]['source'] == cell['source'],
                'status': 'failed' if error else 'passed', 'seconds': round(time.monotonic() - start, 3),
                'log': log_path.name, 'log_sha256': sha(log_path.read_bytes()),
                'rich_output_count': len(captured.outputs),
            })
            print(f'Cell {code_number}: {receipt["cells"][-1]["status"]}', flush=True)
            if error:
                raise RuntimeError(f'Notebook code cell {code_number} failed') from error
        check(notebook_path.read_bytes() == payload, 'Input notebook changed during execution')
        check(shell.user_ns['SOURCE_COMMIT'] == args.source_commit, 'Executed cell source pin differs')
        check(shell.user_ns['DRIVER_SHA256'] == notebook['metadata']['driver_sha256'], 'Notebook metadata and code driver hashes differ')
        archive_path, package, stage_summary, geometry_summary = verify_package(
            base, args.source_commit, shell.user_ns['DRIVER_SHA256'])
        with ZipFile(archive_path) as z:
            for name in z.namelist():
                target = evidence / name
                check(not target.exists(), 'Existing analytic output preserved: ' + name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
        (evidence / 'package-manifest.json').write_text(json.dumps(package, indent=2) + '\n')
        check(verify_cache(root.parent / 'r63-cache') == receipt['verified_cache_before'], 'Input cache changed')
        receipt['final_source_checkout'] = {'commit': args.source_commit, 'clean': True}
        receipt['package'] = package
        receipt['driver_sha256'] = shell.user_ns['DRIVER_SHA256']
        receipt['runtime'] = stage_summary['runtime']
        receipt['frozen_model_invariants'] = stage_summary['invariants']
        receipt['shared_cached_inputs_unchanged'] = True
        receipt['analysis_summary'] = {
            'full142_completed': True, 'photographs_per_model': 142, 'models': 2,
            'heads_per_model': 2, 'assignment_variants_per_head': 2,
            'source_fronts_per_model': 3999, 'full_forward_calls': 284,
            'smoke_forward_calls': 6, 'target_records': 31992,
            'image_head_records': 568, 'groups': 24,
        }
        receipt['input_notebook_unchanged'] = True
        receipt['status'] = 'passed'
    except Exception as error:
        receipt['status'] = 'failed'
        receipt['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        raise
    finally:
        executed_path = evidence / 'executed-notebook.ipynb'
        executed_path.write_text(json.dumps(executed, indent=1) + '\n')
        receipt['executed_notebook'] = {'path': executed_path.name, 'sha256': sha(executed_path.read_bytes()),
                                        'bytes': executed_path.stat().st_size}
        receipt['finished_at'] = datetime.datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
        (evidence / 'notebook-execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'status': receipt['status'], 'cells': [x['status'] for x in receipt['cells']], 'receipt': str(evidence / 'notebook-execution.json'), 'receipt_sha256': sha((evidence / 'notebook-execution.json').read_bytes()), 'package_sha256': receipt.get('package', {}).get('sha256'), 'error': receipt.get('error', {}).get('message')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
