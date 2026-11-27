#!/usr/bin/env python3
"""Prepare and reproduce the R52 annotation study on CPU, then bundle reports.

Only saved predictions and original images are read. No model is loaded and no
training or fresh inference is performed. All generated work stays outside the
source checkout; incompatible existing inputs and completed work are preserved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

import audit_partial_classification_r51_results as result_audit
import build_front_review_r52 as geometry

ROOT = Path(__file__).resolve().parents[1]
NAME = 'baomai-annotation-study-r52.zip'
IMAGE_SOURCE = 'ea4ac955f2dfaf1add24c243599715f802eb41a7'
SOURCE_MANIFEST_SHA = '72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926'
VISUAL_OBSERVATIONS = 'results/evaluation/pr146-r51-20261127/visual-observations.json'
DOCUMENTS = ('docs/thesis-plan-r52.md', 'docs/annotation-policy.md', 'docs/partial-classification-r51-results.md')
SOURCE_FILES = (
    'scripts/run_annotation_study_r52.py',
    'scripts/audit_partial_classification_r51_results.py',
    'scripts/audit_front_geometry_r52.py',
    'scripts/build_front_review_r52.py',
    'scripts/export_annotation_study_r52.py',
)
need, encoded = result_audit.need, result_audit.encoded


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def put(path, payload):
    path = Path(path)
    if path.exists():
        need(path.read_bytes() == payload, 'Different existing file preserved: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def download(path, url, expected, size=None):
    path = Path(path)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        need(hashlib.sha256(payload).hexdigest() == expected
             and (size is None or len(payload) == size), 'Downloaded file differs: ' + path.name)
        put(path, payload)
    need(sha(path) == expected and (size is None or path.stat().st_size == size),
         'Different cached input preserved: ' + str(path))
    return path


def locations(args):
    return dict(
        archive=args.archive or args.work / 'cache/pr146' / result_audit.NAME,
        prior_archive=args.prior_archive or args.work / 'cache/pr137' / result_audit.prior.NAME,
        r44_archive=args.r44_archive or args.work / 'cache/r44-review.zip',
        originals=args.originals or args.work / 'originals',
        supplementary=args.supplementary or args.work / 'supplementary',
    )


def prepare(args):
    paths = locations(args)
    if args.archive:
        need(sha(paths['archive']) == result_audit.ARCHIVE_SHA
             and paths['archive'].stat().st_size == result_audit.ARCHIVE_BYTES, 'PR146 archive differs')
    else:
        result_audit.obtain_archive(paths['archive'].parent)
    if args.prior_archive:
        need(sha(paths['prior_archive']) == result_audit.prior.ARCHIVE_SHA
             and paths['prior_archive'].stat().st_size == result_audit.prior.ARCHIVE_BYTES, 'PR137 archive differs')
    else:
        result_audit.prior.obtain_archive(paths['prior_archive'].parent)
    manifest_path = args.root / 'data/manifest.json'
    need(sha(manifest_path) == SOURCE_MANIFEST_SHA, 'Original image manifest differs')
    entries = [entry for entry in json.loads(manifest_path.read_bytes())['records'] if entry['split'] == 'val']
    need(len(entries) == len({entry['id'] for entry in entries}) == 20, 'Development photograph membership differs')

    def obtain(entry):
        url = f'https://media.githubusercontent.com/media/Siuuuers/dl/{IMAGE_SOURCE}/data/{entry["image"]}'
        path = download(paths['originals'] / (entry['id'] + '.png'), url, entry['image_sha256'])
        need(result_audit.experiment.image_size(path.read_bytes()) == (1280, 720), 'Original image dimensions differ')

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(obtain, entries))
    rows, _ = geometry.audit.load_rows(args.root)
    download(paths['r44_archive'], geometry.R44_URL, geometry.R44_SHA, geometry.R44_BYTES)
    geometry.restore_originals(paths['r44_archive'], paths['supplementary'], rows)
    supplementary = {row['image_id']: row['source_image_sha256'] for row in rows}
    need(len(supplementary) == 12, 'Supplementary photograph membership differs')
    for identity in supplementary:
        need(result_audit.experiment.image_size((paths['supplementary'] / (identity + '.jpg')).read_bytes()) == (1280, 720),
             'Supplementary image dimensions differ')
    receipt = dict(schema='r52-review-inputs-v1', original_manifest_sha256=SOURCE_MANIFEST_SHA,
        original_image_source_revision=IMAGE_SOURCE,
        original_image_sha256={entry['id'] + '.png': entry['image_sha256'] for entry in entries},
        supplementary_image_sha256={identity + '.jpg': expected for identity, expected in supplementary.items()},
        pr146=dict(sha256=result_audit.ARCHIVE_SHA, bytes=result_audit.ARCHIVE_BYTES, head=result_audit.PR_HEAD),
        pr137=dict(sha256=result_audit.prior.ARCHIVE_SHA, bytes=result_audit.prior.ARCHIVE_BYTES, head=result_audit.prior.PR_HEAD),
        supplementary_capsule=dict(sha256=geometry.R44_SHA, bytes=geometry.R44_BYTES, url=geometry.R44_URL),
        model_loaded=False, training=False, fresh_inference=False)
    put(args.work / 'inputs.json', encoded(receipt))
    print('Verified inputs: 20 development PNGs, 12 supplementary JPEGs, PR146 + PR137 results and R44 image capsule.', flush=True)
    return paths


def source_identity(root):
    return dict(revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_clean=not subprocess.check_output(
                    ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True).strip(),
                files_sha256={name: sha(root / name) for name in SOURCE_FILES},
                documents_sha256={name: sha(root / name) for name in DOCUMENTS},
                visual_observations_sha256=sha(root / VISUAL_OBSERVATIONS),
                r52_ledger_sha256=sha(root / geometry.SOURCE / 'audit-ledger.json'),
                r52_manifest_sha256=sha(root / geometry.SOURCE / 'manifest.json'))


def file_inventory(directory):
    return {path.relative_to(directory).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
            for path in sorted(directory.rglob('*')) if path.is_file() and path.name != 'review-receipt.json'}


def verify_review(args):
    output = args.work / 'review'
    receipt = json.loads((output / 'review-receipt.json').read_bytes())
    need(receipt['source'] == source_identity(args.root) and receipt['inputs_sha256'] == sha(args.work / 'inputs.json'),
         'Completed review used different inputs/source; preserved. Choose a fresh work folder.')
    need(receipt['files'] == file_inventory(output), 'Completed review was changed; preserved. Choose a fresh work folder.')
    return receipt


def rank_csv(report):
    rows = {row['stable_id']: row for row in report['rows']}
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['rank', 'stable_id', 'image_id', 'source_row', 'output_row', 'flags',
                                            'strict_interior_events', 'edge_contact_events'])
    writer.writeheader()
    for rank, identity in enumerate(report['priority_queue'], 1):
        row = rows[identity]
        writer.writerow(dict(rank=rank, stable_id=identity, image_id=row['image_id'], source_row=row['source_row'],
            output_row=row['output_row'], flags=';'.join(row['flags']),
            strict_interior_events=len(row['foreign_interior_corners']),
            edge_contact_events=len(row['foreign_edge_contacts'])))
    return out.getvalue().encode()


def review(args):
    paths = prepare(args)
    output = args.work / 'review'
    if output.exists():
        need((output / 'review-receipt.json').is_file(), 'Incomplete review preserved; choose a fresh work folder.')
        verify_review(args)
        print('Verified and reused the completed CPU review.', flush=True)
        return
    temporary = Path(tempfile.mkdtemp(prefix='review-in-progress-', dir=args.work))
    try:
        print('Replaying PR146 training receipts and all four saved-score references...', flush=True)
        reports = result_audit.audit(args.root, paths['archive'], paths['prior_archive'])
        for name, value in reports.items():
            put(temporary / 'audit' / name, encoded(value))
        print('Replaying all 507 supplementary labels and before/after geometry screens...', flush=True)
        geometry.run(args.root, args.root / geometry.SOURCE, paths['supplementary'], check=True)
        geometry.run(args.root, temporary / 'geometry', paths['supplementary'])
        for stage in ('before', 'after'):
            report = json.loads((temporary / 'geometry' / f'geometry-{stage}.json').read_bytes())
            put(temporary / 'geometry' / f'priority-{stage}.csv', rank_csv(report))
        print('Rendering both PDF reports from verified original images...', flush=True)
        subprocess.run([sys.executable, '-B', str(args.root / 'scripts/export_annotation_study_r52.py'),
            '--root', str(args.root), '--archive', str(paths['archive']), '--originals', str(paths['originals']),
            '--supplementary', str(paths['supplementary']), '--audit-dir', str(temporary / 'audit'),
            '--output-dir', str(temporary / 'reports')], check=True)
        for name in ('r52-cassette-amendments.pdf', 'r52-results-and-thesis.pdf'):
            need((temporary / 'reports' / name).is_file(), 'Missing PDF: ' + name)
        receipt = dict(schema='r52-cpu-review-v1', status='passed', source=source_identity(args.root),
            inputs_sha256=sha(args.work / 'inputs.json'), files=file_inventory(temporary),
            training=False, fresh_inference=False, model_loaded=False,
            geometry_is_review_cue_not_error_verdict=True, automatic_promotion=False)
        put(temporary / 'review-receipt.json', encoded(receipt))
        temporary.rename(output)
    except Exception:
        print('Incomplete review preserved for inspection:', temporary, file=sys.stderr)
        raise
    print('Completed CPU review:', output, flush=True)


def package(args):
    receipt = verify_review(args)
    members = {'inputs.json': (args.work / 'inputs.json').read_bytes()}
    for path in sorted((args.work / 'review').rglob('*')):
        if path.is_file():
            members['review/' + path.relative_to(args.work / 'review').as_posix()] = path.read_bytes()
    for name in SOURCE_FILES:
        members['source/' + name] = (args.root / name).read_bytes()
    for name in (*DOCUMENTS, VISUAL_OBSERVATIONS, geometry.SOURCE + '/audit-ledger.json'):
        members['source/' + name] = (args.root / name).read_bytes()
    members['README.txt'] = (
        'R52 CPU annotation study\n\n'
        'Start with review/reports/r52-cassette-amendments.pdf and r52-results-and-thesis.pdf.\n'
        'The review/audit directory replays PR146; review/geometry includes ranked review CSVs and exact label proposals.\n'
        'inputs.json and review/review-receipt.json identify every source and input.\n'
        'No model was loaded, trained or promoted. No fresh inference was performed.\n'
        'Geometry cues prioritize visual review; they are not annotation-error verdicts.\n'
        'Repository base: https://github.com/Siuuuers/dl/tree/' + receipt['source']['revision'] + '\n'
        'Source checkout clean: ' + str(receipt['source']['working_tree_clean']) + '\n'
        'For a development run with local changes, the executed study-source hashes and files are included.\n'
    ).encode()
    members['inventory.json'] = encoded({name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
                                         for name, value in sorted(members.items())})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name, value in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, value)
    data = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as zipped:
        need(zipped.testzip() is None and set(zipped.namelist()) == set(members), 'Packaged ZIP differs')
    destination = args.work / NAME
    put(destination, data)
    print(json.dumps(dict(archive=str(destination), bytes=len(data), sha256=sha(destination), files=len(members)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'review', 'package'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path, required=True)
    for name in ('archive', 'prior-archive', 'r44-archive', 'originals', 'supplementary'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    args.root, args.work = args.root.resolve(), args.work.resolve()
    need(args.root != args.work and args.root not in args.work.parents and args.work not in args.root.parents,
         'Use a work directory outside the source checkout')
    args.work.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'review': review, 'package': package}[args.action](args)


if __name__ == '__main__':
    main()
