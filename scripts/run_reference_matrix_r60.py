#!/usr/bin/env python3
"""Score unchanged supplied/proposed predictions against three annotation references."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

sys.dont_write_bytecode = True
from audit_reflection_return import digest, encoded, require
from evaluate_pose import parse_pose_labels
from run_original_comparison_r59 import assemble, measure

ARCHIVE_PATH = 'docs/thesis/evidence/original-baseline/fresh/baomai-original-comparison-r59-results.zip'
ARCHIVE_SHA256 = 'aae05d14298afad0263197fe4ab7e0e753f63445d9d3503dc1b006030e16c516'
ARCHIVE_BYTES = 1133014
MANIFEST_SHA256 = '72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926'
UPSTREAM_COMMIT = '351eb3cf960ccc34829e2bc72e1fa4645acc6295'
REFERENCE_NAMES = {'supplied664': 'Supplied annotations', 'frozen666': 'Earlier fixed annotations', 'r50': 'Reviewed annotations'}
RESULT_NAME = 'baomai-annotation-reference-r60-results.zip'


def source_check(root, commit, driver_sha):
    root = root.resolve()
    require(root == Path(__file__).resolve().parents[1], 'Driver source root differs')
    require(re.fullmatch('[0-9a-f]{40}', commit) and re.fullmatch('[0-9a-f]{64}', driver_sha), 'Complete source identities required')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == commit and not git('status', '--porcelain', '--untracked-files=all'), 'Source checkout changed')
    require(digest(Path(__file__).read_bytes()) == driver_sha, 'Driver identity differs')
    files = {}
    for line in git('ls-tree', '-r', 'HEAD', 'scripts').splitlines():
        meta, relative = line.split('\t', 1)
        mode, kind, oid = meta.split()
        if not relative.endswith('.py'):
            continue
        b = (root / relative).read_bytes()
        require(kind == 'blob' and hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == oid,
                'Source file differs: ' + relative)
        files[relative] = digest(b)
    return {'commit': commit, 'driver_sha256': driver_sha, 'files_sha256': files}


def checked(root, relative, sha):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Input path escapes source root')
    data = path.read_bytes()
    require(digest(data) == sha, 'Input differs: ' + relative)
    return data


def verified_zip(path, expected_sha=None, expected_bytes=None):
    payload = Path(path).read_bytes()
    if expected_sha is not None:
        require(digest(payload) == expected_sha and len(payload) == expected_bytes, 'Input archive identity differs')
    with ZipFile(path) as z:
        names = z.namelist()
        inventory = json.loads(z.read('inventory.json'))
        require(len(names) == len(set(names)) and set(names) == set(inventory) | {'inventory.json'} and z.testzip() is None,
                'Archive membership or CRC differs')
        members = {name: z.read(name) for name in names}
    for name, entry in inventory.items():
        require(entry == {'bytes': len(members[name]), 'sha256': digest(members[name])}, 'Inventory differs: ' + name)
    return members


def inputs(root):
    root = root.resolve()
    members = verified_zip(root / ARCHIVE_PATH, ARCHIVE_SHA256, ARCHIVE_BYTES)
    prior = json.loads(members['summary.json'])
    require(prior['status'] == 'completed' and prior['fresh_inference'] and not prior['training'], 'Source comparison is incomplete')
    native = json.loads(gzip.decompress(members['native-predictions.json.gz']))
    arms, suppression, associations = assemble(native)
    for name, actual in [('predictions', arms), ('suppression', suppression), ('associations', associations)]:
        require(actual == json.loads(gzip.decompress(members[name + '.json.gz'])), 'Source reconstruction differs: ' + name)

    manifest = json.loads(checked(root, 'data/manifest.json', MANIFEST_SHA256))
    require(manifest['model_sha256'] == prior['contract']['original_model_sha256'] and
            manifest['split_summary']['val'] == {'images': 20, 'instances': 664, 'visible_endpoints': 1227, 'missing_endpoints': 101, 'empty_images': 3},
            'Supplied manifest contract differs')
    refs = {'supplied664': {}}
    label_text = {'supplied664': {}}
    descriptors = {'supplied664': {'manifest_path': 'data/manifest.json', 'manifest_sha256': MANIFEST_SHA256,
        'objects': 664, 'upstream_repository': 'an94-doll/embedding_vision', 'upstream_commit': UPSTREAM_COMMIT,
        'upstream_manifest_path': 'data/manifest.json', 'labels': {}}}
    images = {}
    for record in manifest['records']:
        if record['split'] != 'val':
            continue
        identity = record['id']
        require(record['original_size'] == [1280, 720], 'Image size differs')
        path = 'data/' + record['label']
        b = checked(root, path, record['label_sha256'])
        refs['supplied664'][identity] = parse_pose_labels(b.decode(), 1280, 720, identity)
        label_text['supplied664'][identity] = b.decode()
        descriptors['supplied664']['labels'][identity] = {'path': path, 'sha256': digest(b)}
        images[identity] = record['image_sha256']
        require(prior['contract']['assets']['data/' + record['image']]['sha256'] == record['image_sha256'], 'Photograph identity differs')
    require(len(images) == 20 and all(set(predictions) == set(images) for predictions in arms.values()), 'Photograph membership differs')
    for key in ('frozen666', 'r50'):
        descriptor = prior['contract']['reference_contract'][key]
        checked(root, descriptor['manifest_path'], descriptor['manifest_sha256'])
        refs[key], label_text[key] = {}, {}
        for identity, label in descriptor['labels'].items():
            b = checked(root, label['path'], label['sha256'])
            refs[key][identity] = parse_pose_labels(b.decode(), 1280, 720, key + '/' + identity)
            label_text[key][identity] = b.decode()
        require(set(refs[key]) == set(images) and sum(map(len, refs[key].values())) == descriptor['objects'], 'Reference membership differs')
        descriptors[key] = descriptor
    require(sum(map(len, refs['supplied664'].values())) == 664, 'Supplied reference count differs')
    contract = {'source_prediction_archive': {'path': ARCHIVE_PATH, 'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA256},
        'photographs': images, 'references': descriptors, 'protocol': prior['measurement']['protocol'],
        'prediction_runtime': prior['runtime'], 'original_model_sha256': prior['contract']['original_model_sha256'],
        'adapted_model_sha256': prior['contract']['trained_model_sha256'],
        'prediction_bytes_unchanged_between_references': True,
        'primary_reference': 'r50', 'sensitivity_references': ['supplied664', 'frozen666'],
        'inference_this_analysis': False, 'training_this_analysis': False, 'labels_modified': False}
    return {'contract': contract, 'references': refs, 'native': native, 'prior': prior, 'label_text': label_text}


def calculate(context):
    measured, records = measure(context['references'], context['native'])
    for key in ('frozen666', 'r50'):
        require(measured['references'][key] == context['prior']['measurement']['references'][key], 'Previously measured score changed: ' + key)
    matrix = []
    for reference in REFERENCE_NAMES:
        for arm in ('original', 'proposed'):
            row = measured['references'][reference][arm]
            detection, pairs = row['detection'], row['pairs']
            matrix.append({'reference': reference, 'reference_name': REFERENCE_NAMES[reference], 'method': arm,
                'fronts': detection['gold_count'], 'outputs': detection['prediction_count'],
                'matched': detection['tp'], 'missed': detection['fn'], 'extras': detection['fp'],
                'duplicates': detection['duplicate_fp'], 'eligible_pairs': pairs['eligible'],
                'within5': pairs['within5'], 'within10': pairs['within10'], 'complete_pair_missed_boxes': pairs['missed_box']})
    summary = {'schema': 'cassette-annotation-reference-r60-v1', 'status': 'completed',
        'fresh_inference': False, 'training': False, 'labels_modified': False, 'automatic_promotion': False,
        'development_only': True, 'contract': context['contract'], 'matrix': matrix,
        'original_to_proposed_changes': {key: measured['references'][key]['original_to_proposed_changes'] for key in REFERENCE_NAMES},
        'association_counts': measured['association_counts'],
        'existing_reference_scores_exactly_reproduced': True,
        'interpretation': 'Changing the evaluation annotations changes measured agreement, not model parameters or predictions. References describe the same photographs and are not independent test sets. This analysis does not isolate the effect of changing training labels.'}
    return summary, {**records, 'reference-labels': context['label_text']}


def write_result(work, summary, records, preflight):
    payload = {'summary.json': encoded(summary), 'preflight.json': encoded(preflight)}
    payload.update({name + '.json.gz': gzip.compress(encoded(value), mtime=0) for name, value in records.items()})
    payload['inventory.json'] = encoded({name: {'bytes': len(b), 'sha256': digest(b)} for name, b in payload.items()})
    target = work / RESULT_NAME
    with ZipFile(target.with_suffix('.part'), 'w', ZIP_DEFLATED) as z:
        for name, b in sorted(payload.items()):
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, b)
    target.with_suffix('.part').replace(target)
    (work / 'summary.json').write_bytes(payload['summary.json'])
    return target


def verify(work, context, preflight):
    members = verified_zip(work / RESULT_NAME)
    summary, records = calculate(context)
    require(json.loads(members['summary.json']) == summary, 'Saved summary differs')
    require(json.loads(members['preflight.json']) == preflight, 'Saved preflight differs')
    for name, value in records.items():
        require(json.loads(gzip.decompress(members[name + '.json.gz'])) == value, 'Saved record differs: ' + name)
    require((work / 'summary.json').read_bytes() == members['summary.json'], 'External summary differs')
    return {'status': 'passed', 'matrix_cells': len(summary['matrix']), 'reference_count': len(REFERENCE_NAMES),
        'photographs': 20, 'fresh_inference': False, 'training': False, 'labels_modified': False,
        'archive_sha256': digest((work / RESULT_NAME).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--script-sha256', required=True)
    args = parser.parse_args()
    source = source_check(args.root, args.source_commit, args.script_sha256)
    require(not args.work.resolve().is_relative_to(args.root.resolve()), 'Keep outputs outside the source checkout')
    context = inputs(args.root)
    preflight = {'schema': 'cassette-annotation-reference-preflight-r60-v1', 'status': 'passed', 'source': source,
        'contract': context['contract'], 'runtime': {'python': platform.python_version(), 'implementation': platform.python_implementation()},
        'requirements': 'Python standard library only; no GPU, model execution, training or image downloads.'}
    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / 'preflight.json'
    if path.exists():
        require(json.loads(path.read_bytes()) == preflight, 'Existing preflight differs; choose a new output directory')
    else:
        path.write_bytes(encoded(preflight))
    if args.action == 'prepare':
        print('Verified unchanged predictions and three annotation references for all 20 photographs.')
    elif args.action == 'run':
        summary, records = calculate(context)
        target = write_result(args.work, summary, records, preflight)
        for row in summary['matrix']:
            print(row['reference'], row['method'], 'missed', row['missed'], 'extras', row['extras'],
                  'duplicates', row['duplicates'], 'pairs5', str(row['within5']) + '/' + str(row['eligible_pairs']),
                  'pairs10', str(row['within10']) + '/' + str(row['eligible_pairs']))
        print('Result:', target)
    else:
        receipt = verify(args.work, context, preflight)
        (args.work / 'verification.json').write_bytes(encoded(receipt))
        print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
