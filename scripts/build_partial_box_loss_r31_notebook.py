#!/usr/bin/env python3
"""Build the five-code-cell R31 notebook from a published parent and explicit text payloads."""
import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zlib

BASE_REVISION = '337a12d8b11f9925b493cba794f29a4ffcf848ed'
BASE_NOTEBOOK = 'notebooks/baomai_partial_target_loss_r30.ipynb'
BASE_NOTEBOOK_SHA256 = '152dac24a9f4d9b46d62bd7fb8766352c1c9b2a4cc380b06d158b6788d15cfc9'
PROTOCOL_PATH = 'experiments/partial-box-loss-2026-11-24-r31.json'
OUTPUT_PATH = 'notebooks/baomai_partial_box_loss_r31.ipynb'
ASSEMBLY_NAME = 'R31 notebook assembly'
ASSEMBLY_EMAIL = 'r31-notebook@localhost'
ASSEMBLY_DATE = '2026-11-24T12:00:00+00:00'
ASSEMBLY_MESSAGE = 'Assemble verified R31 box-only loss notebook sources locally'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def git(root, *args, data=None, env=None):
    return subprocess.check_output(['git', *args], cwd=root, input=data, env=env)


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing assignment: ' + name)


def replace_assignment(source, name, value):
    lines = source.splitlines(keepends=True)
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    assert len(matches) == 1, name
    node = matches[0]
    lines[node.lineno - 1:node.end_lineno] = [name + ' = ' + repr(value) + '\n']
    return ''.join(lines)


def publication_files(root):
    files = {
        PROTOCOL_PATH,
        'scripts/front_review_r31.py', 'scripts/partial_box_loss_r31.py',
        'scripts/train_pose_partial_box_loss_r31.py', 'scripts/check_partial_box_loss_r31.py',
        'scripts/build_partial_box_loss_r31_notebook.py',
        'scripts/stage_pose_training.py', 'scripts/train_pose_comparison.py',
        'scripts/verify_incremental_pose_candidate.py',
        # These exact R30 prerequisites were embedded in the prior notebook but
        # omitted from the published parent. Keep their original verified bytes.
        'results/review/front-review-20261124-r30/row-maps.json',
        'results/review/front-review-20261124-r30/review-status.json',
    }
    for directory in ('data/reviews/fronts-2026-11-24-r31', 'data/candidates/front-review-2026-11-24-r31'):
        files.update(p.relative_to(root).as_posix() for p in (root / directory).rglob('*') if p.is_file())
    ledger = 'results/review/front-review-20261124-r31/'
    files.update(ledger + name for name in (
        'row-review.json', 'independent-a.json', 'independent-b.json', 'independent-root-cases.json',
        'operations.json', 'row-maps.json', 'request-coverage.json', 'review-status.json',
        'amendment-summary.json', 'training-input-delta.json'))
    for name in files:
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        assert path.suffix in ('.json', '.txt', '.py'), 'Only explicit text source is embedded.'
    return sorted(files)


def assemble_identity(root, members):
    """Create a deterministic object/commit using a temporary index, without editing HEAD or the worktree."""
    embedded = {}
    with tempfile.TemporaryDirectory(prefix='r31-notebook-index-') as tmp:
        env = {**os.environ, 'GIT_INDEX_FILE': str(Path(tmp) / 'index'),
               'GIT_AUTHOR_NAME': ASSEMBLY_NAME, 'GIT_AUTHOR_EMAIL': ASSEMBLY_EMAIL,
               'GIT_COMMITTER_NAME': ASSEMBLY_NAME, 'GIT_COMMITTER_EMAIL': ASSEMBLY_EMAIL,
               'GIT_AUTHOR_DATE': ASSEMBLY_DATE, 'GIT_COMMITTER_DATE': ASSEMBLY_DATE}
        git(root, 'read-tree', BASE_REVISION, env=env)
        for relative in members:
            payload = (root / relative).read_bytes()
            tracked = git(root, 'ls-tree', BASE_REVISION, '--', relative).decode().strip()
            mode = tracked.split()[0] if tracked else '100644'
            assert mode in ('100644', '100755')
            blob = git(root, 'hash-object', '-w', '--stdin', data=payload).decode().strip()
            git(root, 'update-index', '--add', '--cacheinfo', mode, blob, relative, env=env)
            embedded[relative] = {'sha256': digest(payload), 'size_bytes': len(payload), 'git_mode': mode,
                                  'zlib_base64': base64.b64encode(zlib.compress(payload, 9)).decode()}
        tree = git(root, 'write-tree', env=env).decode().strip()
        source = git(root, '-c', 'commit.gpgsign=false', 'commit-tree', tree, '-p', BASE_REVISION,
                     data=(ASSEMBLY_MESSAGE + '\n').encode(), env=env).decode().strip()
    return source, tree, embedded


def build(root):
    template = (root / BASE_NOTEBOOK).read_bytes()
    assert digest(template) == BASE_NOTEBOOK_SHA256, 'R30 template changed'
    notebook = json.loads(template)
    protocol_bytes = (root / PROTOCOL_PATH).read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol['experiment_id'] == 'partial-box-loss-r31'
    for descriptor in (protocol['accepted_review'], protocol['evaluation']['current_review_reference']):
        assert digest((root / descriptor['path']).read_bytes()) == descriptor['sha256']
    assert digest((root / protocol['candidate_manifest']).read_bytes()) == protocol['candidate_manifest_sha256']
    assert protocol['loss_weighting']['classification'] == 'native_bce_unchanged_for_all_anchors'
    members = publication_files(root)
    source, tree, embedded = assemble_identity(root, members)
    old_helpers = literal_assignment(''.join(notebook['cells'][1]['source']), 'BASE_HELPER_SHA256')
    helper_names = (set(old_helpers) | {'scripts/front_review_r30.py', 'scripts/partial_target_loss.py',
                                      'scripts/train_pose_partial_loss.py'}) - set(embedded)
    helpers = {name: digest(git(root, 'show', BASE_REVISION + ':' + name)) for name in sorted(helper_names)}
    for cell in notebook['cells']:
        value = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            value = (value.replace('baomai-partial-target-loss-r30', 'baomai-partial-box-loss-r31')
                     .replace('from partial_target_loss import', 'from partial_box_loss_r31 import')
                     .replace('scripts/train_pose_partial_loss.py', 'scripts/train_pose_partial_box_loss_r31.py')
                     .replace('R30', 'R31'))
            cell['outputs'], cell['execution_count'] = [], None
        cell['source'] = value.splitlines(keepends=True)
    first = ''.join(notebook['cells'][1]['source'])
    for key, value in {
        'SOURCE_REVISION': source, 'PROTOCOL_PATH': PROTOCOL_PATH, 'PROTOCOL_SHA256': digest(protocol_bytes),
        'EXPERIMENT_ID': 'partial-box-loss-r31', 'BASE_REVISION': BASE_REVISION,
        'EMBEDDED_FILES': embedded, 'BASE_HELPER_SHA256': helpers,
        'ASSEMBLY_NAME': ASSEMBLY_NAME, 'ASSEMBLY_EMAIL': ASSEMBLY_EMAIL,
        'ASSEMBLY_DATE': ASSEMBLY_DATE, 'ASSEMBLY_MESSAGE': ASSEMBLY_MESSAGE,
    }.items():
        first = replace_assignment(first, key, value)
    notebook['cells'][1]['source'] = first.splitlines(keepends=True)
    preflight = ''.join(notebook['cells'][5]['source'])
    preflight = preflight.replace('source_identity, verify_run, verify_trial, package_pair)',
                                 'source_identity, verify_run, verify_trial, package_pair, verify_native_sources)')
    preflight = preflight.replace('CONTEXT = load_protocol(REPO, PROTOCOL_PATH)',
                                 'verify_native_sources()\nCONTEXT = load_protocol(REPO, PROTOCOL_PATH)')
    preflight = preflight.replace('assert document["loss_weighting"]["candidate_factor"] == 2',
        'assert document["loss_weighting"]["candidate_factor"] == 2\n'
        'assert document["loss_weighting"]["classification"] == "native_bce_unchanged_for_all_anchors"')
    preflight = preflight.replace('only the assigned-partial detection-loss factor differs.',
                                 'only the assigned-partial box-regression factor differs; BCE stays native.')
    notebook['cells'][5]['source'] = preflight.splitlines(keepends=True)
    markdown = {
      0: '''# Cassette fronts: R31 partial box loss

**Upload this notebook to Colab, select an L4 or A100 GPU, choose Run all, and return `baomai-partial-box-loss-r31-results.zip`.**

This notebook tests one bounded question: **does extra box-regression weight for partial fronts improve the detection trade-off when classification keeps its native objective?** The R30 candidate found more targets but also produced more unmatched outputs. That motivates this test; it does not prove that classification weighting caused the extra outputs.

Both roles use the **same accepted R31 training labels** and start independently from original epoch173. The control uses factor1. The candidate uses factor2 only for the native CIoU and normalized-L1 box losses of anchors already assigned to a target with fewer than two visible keypoints after augmentation. Classification, keypoint/RLE losses, image exposure and the native branch schedule remain fixed. Shared features can still change prediction scores; the returned results decide whether the trade-off improved.

Each role uses137 training images once per epoch, seed46, batch4, image size1280, FP32, AdamW and initial learning rate0.0001: **30 epochs,1050 batches,4110 image exposures and78 optimizer updates**. A four-batch resource trial runs first and its weights are discarded.

The five code cells handle Drive backup, verified preparation, preflight, the matched pair and ZIP delivery. Completed runs can be recovered and repackaged. Interrupted runs stay intact for diagnosis and restart fresh; partial optimizer state is never resumed. No unpublished GitHub commit is needed: the notebook fetches a published parent and assembles its embedded, hash-verified source locally.
''',
      2: '''## 1. Prepare the verified source and original inputs

Fetch the published project parent, verify the embedded R31 source and accepted supervision, and assemble their exact local commit. Obtain the original photographs and fixed epoch173 initializer through Git LFS, verify every hash, and confirm the index still contains the original pointers. Existing incompatible or incomplete folders are preserved; the error identifies the source folder to archive before retrying.
''',
      4: '''## 2. Prepare the runtime and verify the shared supervision

Keep Colab's matching CUDA PyTorch/torchvision installation and install the pinned trainer. Preflight verifies the original inputs, immutable review, eligible training overlay and fixed protocol. Both roles use identical labels. Development amendments remain separate from the original trainer-validation labels.

The initializer's native head is Pose26 with two E2ELoss branches and `reg_max=1`; its box auxiliary is normalized L1. The runtime checks the exact native source files before training. Classification and keypoint/RLE objectives retain their native implementation.
''',
      6: '''## 3. Run the resource trial and matched comparison

The four-batch candidate trial checks actual training on CUDA and its weights are discarded. Both complete roles then start afresh from epoch173 with identical staged R31 supervision and the same GPU/software runtime. The only prescribed difference is the partial-target box-regression multiplier.

Completed runs are verified before reuse and backed up to Drive. Interrupted folders stay intact for diagnosis; archive only the indicated incomplete folder before restarting. A runtime mismatch blocks a mixed pair. The notebook keeps the fixed thresholds, learning rate and epoch budget.
''',
      8: '''## 4. Save and return the result

The ZIP contains both final portable checkpoints, verified receipts, source/supervision identities, the fixed protocol, per-branch assignment counts, ordinary image exposure and training results. Packaging verifies identical staged labels, matching runtimes and equal update budgets. Keep the Drive copy and downloaded ZIP.
''',
      10: '''### How the returned models will be judged

Use fixed confidence0.13, NMS IoU0.70, matching IoU0.50 and endpoint-confidence0.50. Report the frozen666-front primary reference, historical663-front reference and current R31 review separately. The repeatedly reviewed development photos remain exploratory evidence.

Compare zero-, one- and two-corner recall, gained/lost identities,5/10-pixel pair success with eligible misses retained, unmatched outputs, duplicates and empty-scene outputs. Read the branch-assignment receipts alongside these results. All18376 observed partial targets received a one2one assignment in both PR109 roles, so training assignment starvation is not an established explanation for its remaining inference misses.

The new pair tests box weighting against native factor1 under R31 labels. Cross-round comparisons also contain label changes and are not a controlled attribution to removal of classification weighting. The ZIP provides training evidence; promotion requires reviewing the returned comparison. No automatic promotion or new-session reliability claim is made.
''',
    }
    # Keep ordinary prose readable without changing the exact source identifiers.
    for index, text in markdown.items():
        for before, after in (
            ('factor1', 'factor 1'), ('factor2', 'factor 2'), ('uses137', 'uses 137'),
            ('seed46', 'seed 46'), ('batch4', 'batch 4'), ('size1280', 'size 1280'),
            ('rate0.0001', 'rate 0.0001'), ('epochs,1050', 'epochs, 1050'),
            ('batches,4110', 'batches, 4110'), ('and78', 'and 78'),
            ('confidence0.13', 'confidence 0.13'), ('IoU0.70', 'IoU 0.70'),
            ('IoU0.50', 'IoU 0.50'), ('endpoint-confidence0.50', 'endpoint-confidence 0.50'),
            ('frozen666', 'frozen 666'), ('historical663', 'historical 663'),
            ('identities,5/10', 'identities, 5/10'), ('All18376', 'All 18,376'),
            ('a one2one', 'a one-to-one'), ('epoch173', 'epoch 173')):
            text = text.replace(before, after)
        notebook['cells'][index]['source'] = text.splitlines(keepends=True)
    notebook['metadata']['r31_handoff'] = {
        'base_revision': BASE_REVISION, 'source_revision': source, 'source_tree': tree,
        'protocol_sha256': digest(protocol_bytes), 'embedded_file_count': len(embedded),
        'published_parent_only': True, 'classification_objective_unchanged': True,
    }
    assert len([c for c in notebook['cells'] if c['cell_type'] == 'code']) == 5
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell-{i}', 'exec')
    payload = (json.dumps(notebook, indent=1, ensure_ascii=False) + '\n').encode()
    return payload, notebook['metadata']['r31_handoff']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / OUTPUT_PATH
    payload, identity = build(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(json.dumps({'notebook': str(output), 'sha256': digest(payload), **identity}, indent=2))


if __name__ == '__main__':
    main()
