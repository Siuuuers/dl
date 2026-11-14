#!/usr/bin/env python3
"""Read-only audit of the returned PR35 payload and the labels its receipts identify."""
import argparse
import csv
import hashlib
import io
import json
import os
import struct
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--archive', type=Path, required=True,
                    help='Actual returned PR35 ZIP, not a Git LFS pointer')
parser.add_argument('--output', type=Path, required=True,
                    help='New or empty directory for the audit evidence')
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = args.output.resolve()
ARCHIVE = args.archive.resolve()
if OUT.exists() and any(OUT.iterdir()):
    parser.error('--output must be new or empty; existing evidence is preserved')
PR35_REV = '8e8bb99cabcb44ed0ec0acd14db911aa44013fd8'
ARCHIVE_REL = 'results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip'
EXPECTED_ZIP_SHA = 'df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91'
SELECTED_REL = 'data/candidates/annotations-2026-11-10/manifest.json'
LATER_REL = 'data/candidates/remaining-fronts-2026-11-14/manifest.json'
FROZEN_REL = 'data/manifest.json'
SOURCE_REL = 'results/review/target-clarification-20261114/resolution.json'

def sha(b):
    return hashlib.sha256(b).hexdigest()

cache = {}
def file_sha(p):
    p = Path(p)
    if p not in cache:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        cache[p] = h.hexdigest()
    return cache[p]

def git_bytes(spec):
    env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'}
    return subprocess.check_output(['git', 'show', spec], cwd=ROOT, env=env, timeout=45)

def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def count_labels(records, label_bytes):
    counts = {s: dict(images=0, objects=0, visible_endpoints=0, missing_endpoints=0, empty_images=0)
              for s in ('train', 'val')}
    for rec in records:
        rows = [r.split() for r in label_bytes[rec['id']].decode().splitlines() if r.strip()]
        c = counts[rec['split']]
        c['images'] += 1
        c['objects'] += len(rows)
        c['empty_images'] += not rows
        for row in rows:
            assert len(row) == 11 and row[0] == '0'
            vis = [int(row[7]), int(row[10])]
            assert all(v in (0, 2) for v in vis)
            c['visible_endpoints'] += sum(v == 2 for v in vis)
            c['missing_endpoints'] += sum(v == 0 for v in vis)
    return counts

def geometry(row, size):
    t = row.split()
    n = list(map(float, t))
    w, h = size
    return {
        'bbox_xyxy_px': [(n[1] - n[3]/2)*w, (n[2] - n[4]/2)*h,
                        (n[1] + n[3]/2)*w, (n[2] + n[4]/2)*h],
        'endpoints_xyv_px': [[n[5]*w, n[6]*h, int(n[7])], [n[8]*w, n[9]*h, int(n[10])]],
    }

OUT.mkdir(parents=True, exist_ok=True)
archive_sha = file_sha(ARCHIVE)
assert archive_sha == EXPECTED_ZIP_SHA and ARCHIVE.stat().st_size == 39776456
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    members = {n: z.read(n) for n in z.namelist()}
package = json.loads(members['gpu-results-manifest.json'])
assert set(members) == {'gpu-results-manifest.json'} | {r['path'] for r in package['files']}
archive_files = []
for item in package['files']:
    b = members[item['path']]
    actual = dict(path=item['path'], sha256=sha(b), size_bytes=len(b))
    assert actual == item
    archive_files.append(actual)

frozen_bytes = (ROOT / FROZEN_REL).read_bytes()
frozen = json.loads(frozen_bytes)
selected_bytes = (ROOT / SELECTED_REL).read_bytes()
selected = json.loads(selected_bytes)
later = json.loads((ROOT / LATER_REL).read_bytes())
records = frozen['records']
records_by_id = {r['id']: r for r in records}
changes = {c['image_id']: c for c in selected['changes']}
later_changes = {c['image_id']: c for c in later['changes']}
assert len(records) == 157 and len(changes) == 8
assert selected['source_manifest_sha256'] == sha(frozen_bytes)
assert sha((ROOT / Path(SELECTED_REL).parent / selected['review']['path']).read_bytes()) == selected['review']['sha256']

payloads = {}
for arm in ('reference', 'candidate'):
    prefix = arm + '/'
    receipt = json.loads(members[prefix + 'training-receipt.json'])
    staging = json.loads(members[prefix + 'staging-manifest.json'])
    checkpoint = json.loads(members[prefix + 'checkpoint-manifest.json'])
    for field in ('checkpoint', 'training_receipt', 'staging_manifest'):
        link = checkpoint[field]
        assert link['sha256'] == sha(members[prefix + link['path']])
        if 'size_bytes' in link:
            assert link['size_bytes'] == len(members[prefix + link['path']])
    for field in ('checkpoint', 'log', 'results'):
        link = receipt[field]
        assert link['sha256'] == sha(members[prefix + link['path']])
    assert receipt['staging_manifest_sha256'] == sha(members[prefix + 'staging-manifest.json'])
    assert receipt['reference_manifest_sha256'] == staging['source']['reference_manifest_sha256'] == sha(frozen_bytes)
    assert receipt['model_card_sha256'] == staging['source']['model_card_sha256'] == file_sha(ROOT / 'metadata/model_card.json')
    assert receipt['initial_checkpoint_sha256'] == staging['source']['initial_checkpoint_sha256'] == frozen['model_sha256']
    assert staging['source']['candidate_id'] == selected['candidate_id']
    assert staging['source']['candidate_manifest_sha256'] == sha(selected_bytes)
    assert receipt['recipe_id'] == 'reviewed-annotations-seed46-v1'
    assert receipt['training_settings']['seed'] == 46
    assert receipt['training_settings']['epochs'] == 30
    assert receipt['training_settings']['imgsz'] == 1280
    assert receipt['training_settings']['fraction'] == 1.0
    assert receipt['arm'] == arm and receipt['status'] == 'complete'
    assert receipt['completed_epochs'] == list(range(1,31))
    assert receipt['images_processed'] == 137 * 30
    assert receipt['execution']['exit_code'] == 0
    assert receipt['inputs_reverified_after_worker'] is True
    assert receipt['training_returned'] is True and receipt['finalization_completed'] is True
    assert receipt['optimizer_parameter_change_observed'] is True
    assert receipt['validation_executed'] is True
    result_rows = list(csv.DictReader(io.StringIO(members[prefix+'fit/results.csv'].decode())))
    epoch_key = next(k for k in result_rows[0] if k.strip() == 'epoch')
    assert [int(r[epoch_key]) for r in result_rows] == list(range(1,31))
    payloads[arm] = dict(receipt=receipt, staging=staging, checkpoint=checkpoint)

candidate = payloads['candidate']
receipt = candidate['receipt']
staging = candidate['staging']
baseline_path = ROOT / frozen['model_checkpoint']
assert file_sha(baseline_path) == receipt['initial_checkpoint_sha256']
assert baseline_path.stat().st_size == 106838542

# Every staged label is checked against actual repository bytes, with the selected
# eight-file overlay substituted only for the candidate arm. No staging directory
# or training process is recreated.
inventory = []
arm_labels = {}
for variant in ('reference','candidate'):
    v = staging['variants'][variant]
    fs = {f['path']:f for f in v['files']}
    assert len(fs) == 314
    assert set(fs) == {r[k] for r in records for k in ('image','label')}
    label_bytes = {}
    for rec in records:
        c = changes.get(rec['id']) if variant == 'candidate' else None
        label_rel = (Path(SELECTED_REL).parent / c['candidate_label']) if c else Path('data') / rec['label']
        b = (ROOT / label_rel).read_bytes()
        source_b = (ROOT / 'data' / rec['label']).read_bytes()
        assert sha(source_b) == rec['label_sha256']
        f = fs[rec['label']]
        assert f['source_path'] == label_rel.as_posix()
        assert f['source_sha256'] == f['output_sha256'] == sha(b)
        if c:
            assert c['source_label_sha256'] == sha(source_b)
            assert c['candidate_label_sha256'] == sha(b)
        label_bytes[rec['id']] = b
        im = fs[rec['image']]
        assert im['source_path'] == 'data/' + rec['image']
        assert im['source_sha256'] == im['output_sha256'] == rec['image_sha256']
        image_path = ROOT / 'data' / rec['image']
        actual_image_available = image_path.exists() and image_path.stat().st_size > 200
        actual_image_sha = file_sha(image_path) if actual_image_available else None
        if actual_image_available:
            assert actual_image_sha == rec['image_sha256']
            with image_path.open('rb') as image_file:
                header = image_file.read(24)
            assert header[:8] == b'\x89PNG\r\n\x1a\n'
            assert list(struct.unpack('>II', header[16:24])) == rec['original_size']
        inventory.append(dict(variant=variant, image_id=rec['id'], split=rec['split'],
            label_source_path=label_rel.as_posix(), actual_label_sha256=sha(b),
            recorded_staged_label_sha256=f['output_sha256'], label_bytes_match=True,
            objects=len(b.decode().splitlines()), actual_image_available=actual_image_available,
            actual_image_sha256=actual_image_sha, recorded_staged_image_sha256=im['output_sha256']))
    counts = count_labels(records, label_bytes)
    assert counts == v['counts']
    arm_labels[variant] = label_bytes
    # YAML itself is not carried in the ZIP. Reconstruct its bytes using the
    # hash-verified historical yaml_bytes function and recorded staging root.
    arm_root = str(Path(staging['output_root']) / variant)
    yaml = (f'path: {json.dumps(arm_root, ensure_ascii=False)}\n'
            'train: images/train\nval: images/val\nnames:\n  0: cassette_front\n'
            'kpt_shape: [2, 3]\nflip_idx: [1, 0]\nkpt_names:\n'
            '  0: [front_lower_left, front_lower_right]\n').encode()
    assert sha(yaml) == v['data_yaml_sha256']
    if variant == 'candidate':
        assert sha(yaml) == receipt['data_yaml_sha256']
    (OUT / f'{variant}-dataset-yaml-reconstructed.txt').write_bytes(yaml)

changed_ids = sorted(i for i in records_by_id if arm_labels['reference'][i] != arm_labels['candidate'][i])
assert changed_ids == sorted(changes)
assert all(arm_labels['reference'][r['id']] == arm_labels['candidate'][r['id']]
           for r in records if r['split'] == 'val')
for arm, p in payloads.items():
    for variant in ('reference','candidate'):
        # Both returned arm staging inventories must identify the same data.
        assert p['staging']['variants'][variant]['files'] == staging['variants'][variant]['files']
        assert p['staging']['variants'][variant]['counts'] == staging['variants'][variant]['counts']
    expected = staging['variants'][arm]['counts']
    for loader in p['receipt']['loader_verification']:
        for key in ('images', 'objects', 'visible_endpoints'):
            assert loader[key] == expected[loader['split']][key]

source_files = []
for path, expected in receipt['source']['files_sha256'].items():
    b = git_bytes(receipt['source']['revision'] + ':' + path)
    assert sha(b) == expected
    source_files.append(dict(path=path, sha256=sha(b), historical_git_bytes_match=True))
policy_historical = git_bytes(receipt['source']['revision'] + ':' + selected['annotation_policy']['path'])
assert sha(policy_historical) == selected['annotation_policy']['sha256']
assert file_sha(ROOT / 'docs/policies/annotation-policy-pr35-2026-11-10.md') == sha(policy_historical)

source_038 = arm_labels['candidate']['new_000038']
assert source_038 == arm_labels['reference']['new_000038']
assert 'new_000038' not in changes and 'new_000038' not in later_changes
rows_038 = source_038.decode().splitlines()
assert len(rows_038) == 15
resolution = json.loads((ROOT / SOURCE_REL).read_bytes())
assert resolution['new_000038']['source_label_sha256'] == sha(source_038)
assert resolution['new_000038']['source_sha256'] == records_by_id['new_000038']['image_sha256']

examples = []
for image_id in ('new_000029','new_000033','new_000037','new_000040','old_000042'):
    c = later_changes[image_id]
    before = arm_labels['candidate'][image_id]
    after = (ROOT / Path(LATER_REL).parent / c['candidate_label']).read_bytes()
    assert sha(before) == c['source_label_sha256']
    assert sha(after) == c['candidate_label_sha256']
    assert before != after and image_id not in changes
    operations = []
    for op in c['replacements']:
        assert before.decode().splitlines()[op['original_row']-1] == op['source_row_text']
        assert after.decode().splitlines()[op['candidate_row']-1] == op['candidate_row_text']
        operations.append(dict(kind='replace', original_row=op['original_row'],
            changed_fields=op['changed_fields'],
            before=geometry(op['source_row_text'],records_by_id[image_id]['original_size']),
            after=geometry(op['candidate_row_text'],records_by_id[image_id]['original_size']),
            review_status=op['review_status'], reason=op['reason']))
    for op in c['additions']:
        assert op['candidate_row'] > len(before.decode().splitlines())
        assert after.decode().splitlines()[op['candidate_row']-1] == op['candidate_row_text']
        operations.append(dict(kind='append', candidate_row=op['candidate_row'],
            after=geometry(op['candidate_row_text'],records_by_id[image_id]['original_size']),
            review_status=op['review_status'], reason=op['reason']))
    examples.append(dict(image_id=image_id, trained_label_sha256=sha(before),
        later_label_sha256=sha(after), trained_rows=len(before.decode().splitlines()),
        later_rows=len(after.decode().splitlines()), correction_absent_from_pr35_training=True,
        operations=operations))

archive_git_binding = {'revision':PR35_REV,'path':ARCHIVE_REL,'checked':False}
try:
    env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'}
    tree_record = subprocess.check_output(['git','ls-tree',PR35_REV,ARCHIVE_REL],
        cwd=ROOT,env=env,timeout=45,text=True).strip()
    mode, kind, committed_oid, path = tree_record.split(maxsplit=3)
    assert kind == 'blob' and path == ARCHIVE_REL
    git_hash = hashlib.sha1(f'blob {ARCHIVE.stat().st_size}\0'.encode())
    with ARCHIVE.open('rb') as archive_file:
        for chunk in iter(lambda:archive_file.read(1024*1024),b''):
            git_hash.update(chunk)
    assert git_hash.hexdigest() == committed_oid
    archive_git_binding.update(checked=True,
        method='Actual ZIP bytes reproduce the Git blob object ID in the immutable PR35 commit tree; no ZIP redownload.',
        committed_blob_oid=committed_oid,actual_archive_git_blob_oid=git_hash.hexdigest(),
        git_tree_record=tree_record)
except (subprocess.SubprocessError, FileNotFoundError) as exc:
    archive_git_binding['limitation'] = str(exc)

audit = {
 'schema_version':1,'kind':'PR35-label-training-provenance-audit',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only artifact, receipt and label-byte audit. No model loading, training, inference, or fresh visual adjudication.',
 'repository':str(ROOT),
 'local_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'archive':dict(path=str(ARCHIVE),size_bytes=ARCHIVE.stat().st_size,sha256=archive_sha,
    crc_ok=True,member_count=len(members),payload_hashes_verified=len(archive_files),
    published_url=f'https://github.com/Siuuuers/dl/blob/{PR35_REV}/{ARCHIVE_REL}',
    git_binding=archive_git_binding),
 'selected_checkpoint':dict(arm='candidate',recipe_id=receipt['recipe_id'],recipe_sha256=receipt['recipe_sha256'],
    checkpoint=receipt['checkpoint'],checkpoint_manifest_sha256=sha(members['candidate/checkpoint-manifest.json']),
    receipt_sha256=sha(members['candidate/training-receipt.json']),
    staging_manifest_sha256=sha(members['candidate/staging-manifest.json']),
    epochs=30,seed=46,checkpoint_selection=receipt['checkpoint_selection'],
    completed_epochs=receipt['completed_epochs'],images_processed=receipt['images_processed'],
    optimizer_steps=receipt['optimizer_steps'],batches_completed=receipt['batches_completed'],
    csv_epoch_rows=30,loader_verification=receipt['loader_verification'],
    inputs_reverified_after_worker=receipt['inputs_reverified_after_worker'],
    source_revision=receipt['source']['revision'],source_files=source_files,
    recorded_device=receipt['runtime']['device_name'],hardware_independently_attested=False),
 'packaged_default':dict(path=frozen['model_checkpoint'],sha256=file_sha(baseline_path),
    size_bytes=baseline_path.stat().st_size,epoch=173,bytes_verified=True,
    relationship='Distinct earlier checkpoint; the PR35 receipt records it as initialization for both arms. Generic predict_pose_folder default still selects it without --checkpoint-manifest.',
    runtime_source='scripts/predict_pose_folder.py',runtime_source_sha256=file_sha(ROOT/'scripts/predict_pose_folder.py'),
    generic_confidence_default=0.25,selected_review_workflow_confidence=0.13,
    original_epoch173_training_rerun_or_173_epoch_receipt_audited=False),
 'dataset':dict(frozen_manifest_path=FROZEN_REL,frozen_manifest_sha256=sha(frozen_bytes),
    overlay_manifest_path=SELECTED_REL,overlay_manifest_sha256=sha(selected_bytes),
    historical_policy_sha256=sha(policy_historical),historical_policy_git_bytes_verified=True,
    changed_train_label_ids=changed_ids,counts={v:staging['variants'][v]['counts'] for v in ('reference','candidate')},
    unique_label_bytes_verified=157+len(changes),effective_labels_verified_per_arm=157,
    validation_labels_identical=True,validation_images_identical=True,
    actual_original_pngs_freshly_hashed=sum(r['actual_image_available'] for r in inventory if r['variant']=='candidate'),
    image_inventory_hashes_match_frozen_manifest=157,
    data_yaml=dict(sha256=receipt['data_yaml_sha256'],mode='Reconstructed bytes match recorded hash; original staged YAML is not a ZIP member.'),
    physical_policy_certification=staging['physical_policy_conformance']),
 'new_000038':dict(label_sha256=sha(source_038),rows=15,
    image_sha256=records_by_id['new_000038']['image_sha256'],
    original_label_byte_identical_in_selected_training=True,
    no_selected_overlay_change=True,no_PR63_remaining_overlay_change=True,
    source_row1_text=rows_038[0],source_row1_geometry=geometry(rows_038[0],records_by_id['new_000038']['original_size']),
    training_provenance_conclusion='The selected PR35 candidate staging inventory records this exact complete 15-row source label file, including row1; no two extra rows for the thumb-adjacent stack were added by its eight-file overlay.',
    visual_conclusion_source=SOURCE_REL,visual_conclusion_source_sha256=file_sha(ROOT/SOURCE_REL),
    visual_conclusion='Published corrected review identifies three thumb-adjacent cassette fronts where only source row1 describes the region; two local instances are absent or merged and row1 box/point ownership is inconsistent.',
    visual_scope='This audit verifies the review is attached to the same image/label hashes. It does not independently re-adjudicate those pixels or certify exact repair coordinates. The separate bottom-left frame-clipped front is not counted among these two instances.',
    exact_new_labels_ready=resolution['new_000038']['exact_new_labels_ready']),
 'later_accepted_corrections_absent_from_training':examples,
 'conclusion':'Yes: the selected model was trained with residual annotation defects now identified by later review, including the unchanged new_000038 file. The selected eight-file overlay did correct earlier defects, but did not certify every training row. Later edits do not change existing model weights until a new training experiment is completed.',
 'limitations':[
   'Hashes bind actual returned files and the data inventories recorded by the trainer. They do not independently attest the remote GPU process or prove every recorded event physically occurred.',
   'The returned ZIP omits staged image/label copies and data.yaml. Actual current frozen/overlay source label bytes match every recorded staged label hash; image byte availability is reported above; YAML bytes were reconstructed.',
   'Only a sample of batches is detailed in the receipt. Completion, loader counts, 30 CSV epochs and 4,110 image presentations are consistent, but this audit did not replay training.',
   'Semantic defects and accepted repairs rely on their cited visual review; this provenance audit is not whole-dataset visual certification.',
   'The effect of any particular erroneous row on predictions is not measured here; no claim that all labels or the whole model are unusable follows.',
   'The earlier epoch173 original training run is not newly reconstructed or independently attested by this audit.'
 ]
}
dump('audit.json', audit)
dump('archive-files.json', archive_files)
dump('label-image-hash-inventory.json', inventory)
for name in ('training-receipt.json', 'staging-manifest.json', 'checkpoint-manifest.json'):
    (OUT / ('candidate-' + name)).write_bytes(members['candidate/' + name])
print(json.dumps({
 'status':'passed','out':str(OUT),'archive_sha256':archive_sha,
 'archive_git_binding_checked':archive_git_binding['checked'],
 'payload_hashes_verified':len(archive_files),
 'candidate_counts':audit['dataset']['counts']['candidate'],
 'actual_original_pngs_freshly_hashed':audit['dataset']['actual_original_pngs_freshly_hashed'],
 'new_000038_label_unchanged':True,'examples':len(examples),
 'source_code_hashes_verified':len(source_files)
},indent=2))
