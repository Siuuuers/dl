#!/usr/bin/env python3
"""Build a self-contained, five-cell native-loss R33 training notebook."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

from build_partial_box_loss_r31_notebook import git, literal_assignment, replace_assignment
from build_partial_box_replication_r32_notebook import publication_files as r32_files

BASE_REVISION = "337a12d8b11f9925b493cba794f29a4ffcf848ed"
TEMPLATE_PATH = "notebooks/baomai_partial_box_replication_r32.ipynb"
TEMPLATE_SHA256 = "162c2c7c343761071dba734ab4609afc0a2058fd92d1e6b4673fe890f03e7fb3"
PROTOCOL_PATH = "experiments/front-review-2026-11-25-r33.json"
OUTPUT_PATH = "notebooks/baomai_front_review_r33.ipynb"
ASSEMBLY_NAME = "R33 notebook assembly"
ASSEMBLY_EMAIL = "r33-notebook@localhost"
ASSEMBLY_DATE = "2026-11-25T07:00:00+08:00"
ASSEMBLY_MESSAGE = "Assemble verified R33 native supervision comparison locally"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def publication_files(root):
    files = set(r32_files(root)) | {
        TEMPLATE_PATH, PROTOCOL_PATH, "scripts/front_review_r33.py",
        "scripts/front_review_r33_training.py", "scripts/native_front_review_r33.py",
        "scripts/build_front_review_r33_notebook.py", "scripts/triage_front_geometry_r33.py",
        "docs/annotation-boundaries-r33.md",
    }
    for directory in ("data/reviews/fronts-2026-11-25-r33", "data/candidates/front-review-2026-11-25-r33"):
        files.update(p.relative_to(root).as_posix() for p in (root/directory).rglob("*") if p.is_file())
    ledger = "results/review/front-review-20261125-r33/"
    files.update(ledger+name for name in (
        "row-review.json", "independent-review-a.json", "independent-review-b.json",
        "root-endorsement.json", "coverage.json", "geometry-triage.json",
        "operations.json", "row-maps.json", "request-coverage.json", "review-status.json",
        "amendment-summary.json", "training-input-delta.json",
    ))
    for name in files:
        path = root/name
        assert path.is_file() and not path.is_symlink(), name
        assert path.suffix in (".json", ".txt", ".py", ".md", ".ipynb"), name
    return sorted(files)


def assemble_identity(root, members):
    embedded = {}
    with tempfile.TemporaryDirectory(prefix="r33-notebook-index-") as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp)/"index"),
               "GIT_AUTHOR_NAME": ASSEMBLY_NAME, "GIT_AUTHOR_EMAIL": ASSEMBLY_EMAIL,
               "GIT_COMMITTER_NAME": ASSEMBLY_NAME, "GIT_COMMITTER_EMAIL": ASSEMBLY_EMAIL,
               "GIT_AUTHOR_DATE": ASSEMBLY_DATE, "GIT_COMMITTER_DATE": ASSEMBLY_DATE}
        git(root, "read-tree", BASE_REVISION, env=env)
        for relative in members:
            payload = (root/relative).read_bytes()
            tracked = git(root, "ls-tree", BASE_REVISION, "--", relative).decode().strip()
            mode = tracked.split()[0] if tracked else "100644"
            assert mode in ("100644", "100755")
            blob = git(root, "hash-object", "-w", "--stdin", data=payload).decode().strip()
            git(root, "update-index", "--add", "--cacheinfo", mode, blob, relative, env=env)
            embedded[relative] = {"sha256": digest(payload), "size_bytes": len(payload), "git_mode": mode,
                                  "zlib_base64": base64.b64encode(zlib.compress(payload, 9)).decode()}
        tree = git(root, "write-tree", env=env).decode().strip()
        source = git(root, "-c", "commit.gpgsign=false", "commit-tree", tree, "-p", BASE_REVISION,
                     data=(ASSEMBLY_MESSAGE+"\n").encode(), env=env).decode().strip()
    return source, tree, embedded


PREFLIGHT_TAIL = '''
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_pose import preflight_dataset
from stage_pose_training import source_state
from partial_box_loss_r31 import verify_native_sources
from native_front_review_r33 import load_protocol, source_identity, verify_run, verify_trial, package_pair
verify_native_sources()  # Verify pinned native code; no custom criterion is attached.
CONTEXT = load_protocol(REPO, PROTOCOL_PATH)
assert CONTEXT["sha256"] == PROTOCOL_SHA256
document = CONTEXT["document"]
assert document["experiment_id"] == EXPERIMENT_ID
assert document["recipe_sha256"] == RECIPE_SHA256
assert all(document["recipe_id"] == ROLE_RECIPES[role] for role in ROLES)
assert document["execution"]["loss"] == "unmodified_native_Pose26"
PREFLIGHT = preflight_dataset(REPO, require_weights=True)
assert PREFLIGHT["model_sha256"] == document["initial_checkpoint_sha256"]
STATE = source_state(REPO, REPO / document["candidate"]["path"])
for role in ROLES:
    assert STATE["counts"][role] == document["roles"][role]["counts"]
    assert STATE["counts"][role]["train"]["images"] == 137
    assert STATE["counts"][role]["train"]["empty_images"] == 13
    assert STATE["counts"][role]["val"]["objects"] == 664
DELTA = CONTEXT["candidate_verification"]["training_input_delta"]["report"]
assert DELTA["summary"]["changed_rows"] > 0
assert DELTA["summary"]["classes_visibility_preserved"] is True
DELTA_PATH = RUN_BASE / "training-input-delta.json"
delta_bytes = (REPO / document["training_input_delta"]["path"]).read_bytes()
assert not DELTA_PATH.exists() or DELTA_PATH.read_bytes() == delta_bytes
DELTA_PATH.write_bytes(delta_bytes)
save_to_drive(DELTA_PATH)
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
print("Verified training change:", DELTA["summary"])
print("Reference uses previous R31 labels; candidate uses the exact eligible R33 corrections.")
print("Native Pose26 loss in both arms. Missing landmarks remain unavailable supervision.")
print("Each arm: seed 46; 137 images once per epoch; 30 epochs; 4110 image presentations; 1050 batches; 78 optimizer updates.")
print("Batch 4, image size 1280, FP32, AdamW, initial learning rate 0.0001. Resource-trial weights are discarded.")
'''


def build(root):
    root = Path(root).resolve()
    template = (root/TEMPLATE_PATH).read_bytes()
    assert digest(template) == TEMPLATE_SHA256, "Exact recovered R32 notebook differs"
    notebook = json.loads(template)
    protocol_bytes = (root/PROTOCOL_PATH).read_bytes()
    protocol = json.loads(protocol_bytes)
    from native_front_review_r33 import load_protocol
    context = load_protocol(root)
    delta = context["candidate_verification"]["training_input_delta"]["report"]
    source, tree, embedded = assemble_identity(root, publication_files(root))
    first = "".join(notebook["cells"][1]["source"])
    old_helpers = literal_assignment(first, "BASE_HELPER_SHA256")
    helpers = {name: digest(git(root, "show", BASE_REVISION+":"+name))
               for name in sorted(set(old_helpers)-set(embedded))}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            value = "".join(cell["source"])
            value = value.replace("baomai-partial-box-replication-r32", "baomai-front-review-r33")
            value = value.replace("Verified local R32 replication source:", "Verified local R33 native comparison source:")
            value = value.replace("baomai-front-review-r33-20261124", "baomai-front-review-r33-20261125")
            cell["source"], cell["outputs"], cell["execution_count"] = value.splitlines(keepends=True), [], None
    first = "".join(notebook["cells"][1]["source"])
    for key, value in {
        "SOURCE_REVISION": source, "PROTOCOL_PATH": PROTOCOL_PATH, "PROTOCOL_SHA256": digest(protocol_bytes),
        "EXPERIMENT_ID": "front-review-r33", "BASE_REVISION": BASE_REVISION,
        "ROLES": ("reference", "candidate"),
        "ROLE_RECIPES": {role: protocol["recipe_id"] for role in ("reference", "candidate")},
        "RECIPE_SHA256": protocol["recipe_sha256"], "EMBEDDED_FILES": embedded, "BASE_HELPER_SHA256": helpers,
        "ASSEMBLY_NAME": ASSEMBLY_NAME, "ASSEMBLY_EMAIL": ASSEMBLY_EMAIL,
        "ASSEMBLY_DATE": ASSEMBLY_DATE, "ASSEMBLY_MESSAGE": ASSEMBLY_MESSAGE,
    }.items():
        first = replace_assignment(first, key, value)
    notebook["cells"][1]["source"] = first.splitlines(keepends=True)
    preflight = "".join(notebook["cells"][5]["source"]).split("sys.dont_write_bytecode = True", 1)[0]
    notebook["cells"][5]["source"] = (preflight+PREFLIGHT_TAIL).splitlines(keepends=True)
    training = "".join(notebook["cells"][7]["source"])
    training = training.replace('    assert receipt["experiment"] == experiment_identity(CONTEXT, role)\n', '')
    training = training.replace('scripts/train_pose_partial_box_replication_r32.py', 'scripts/train_pose_comparison.py')
    training = training.replace('"--protocol", PROTOCOL_PATH, "--trial" if trial else "--train", "--role", role,',
        '"--candidate", document["candidate"]["path"], "--recipe", ROLE_RECIPES[role], "--trial" if trial else "--train", "--arm", role,')
    training = training.replace('with the shared R31 candidate labels.', 'with its verified label arm and native loss.')
    training = training.replace('"control"', '"reference"').replace('CONTROL_RUN', 'REFERENCE_RUN').replace('control_receipt', 'reference_receipt')
    start = training.index('assert reference_receipt["training_labels"] == candidate_receipt["training_labels"]')
    end = training.index('assert reference_receipt["optimizer_steps"]', start)
    training = training[:start] + '''assert reference_receipt["training_labels"]["label_manifest"] == document["control"]["manifest"]
assert candidate_receipt["training_labels"]["label_manifest"] == document["candidate"]["path"]
assert reference_receipt["training_labels"]["training_protocol"] == candidate_receipt["training_labels"]["training_protocol"]
assert [s["image_ids"] for s in reference_receipt["batch_samples"]] == [s["image_ids"] for s in candidate_receipt["batch_samples"]]
''' + training[end:]
    notebook["cells"][7]["source"] = training.splitlines(keepends=True)
    package = "".join(notebook["cells"][9]["source"]).replace("CONTROL_RUN", "REFERENCE_RUN")
    notebook["cells"][9]["source"] = package.splitlines(keepends=True)
    count, photos = delta["summary"]["changed_rows"], delta["summary"]["changed_photos"]
    markdown = {
        0: f'''# Cassette fronts: R33 native training comparison

**Upload this notebook to Colab, choose an L4 or A100 GPU, select Run all, and return `baomai-front-review-r33-results.zip`.**

One question: **do the verified R33 training-boundary corrections improve localization under the unchanged native loss?** The reference uses the previous R31 training labels; the candidate uses **{count} corrected rows across {photos} training photographs**. Every other training setting is shared.

PR113 was the second-seed check of doubled partial-box loss. It did not meet the prespecified clean-improvement screen: on the frozen 666-object reference, misses changed from 32 to 33 and corner-pair successes within 5 pixels changed from 524 to 522, although unmatched outputs fell from 105 to 100. The weight escalation stops here. This notebook uses the **unmodified native Pose26 loss in both arms**.

Both arms start independently from original epoch 173. Each receives 137 images once per epoch, including 13 empty scenes: **30 epochs, 1,050 batches, 4,110 image presentations and 78 optimizer updates**, seed 46, batch 4, image size 1280, FP32, AdamW and initial learning rate 0.0001. Final-epoch selection is fixed in advance.

The five code cells assemble verified source files, restore original inputs, check the runtime, run a discarded four-batch resource trial and the matched pair, then package the return ZIP. Completed runs are verified before reuse and backed up to Drive when available.
''',
        2: '''## 1. Restore verified source and original inputs

Only a published parent commit is fetched. The R33 source, labels and review ledgers are embedded with hashes, so this notebook does not depend on an unpublished commit being accessible. All 157 original photographs and the original initializer are verified. Interrupted or incompatible folders are preserved for inspection.
''',
        4: f'''## 2. Verify the exact treatment

The comparison changes only the accepted boundaries in {count} eligible training rows. All 137 training photographs remain, with 3,784 objects and 13 empty scenes. Both arms retain prior supervision for held photograph 015 and the 31 old-series photographs. Original trainer validation remains 20 images with 664 objects.

The complete R33 annotation review is separate from training eligibility. Its development-label amendments never enter training or trainer validation. The current R33 development reference is exploratory; frozen 666 and historical 663 remain unchanged.
''',
        6: '''## 3. Run the matched native-loss pair

Use one GPU runtime for both arms. The four-batch trial verifies image loading, gradients, parameter updates and memory before full training; its weights are discarded. Each full arm starts fresh from original epoch 173. Changing loss, seed, image size, thresholds or checkpoint-selection rules would create a different experiment.
''',
        8: '''## 4. Return the verified result ZIP

Return `baomai-front-review-r33-results.zip` with both final checkpoints, raw logs, receipts, protocol and exact training delta. Packaging checks the actual bytes and equality of the two runtimes. A completed, compatible ZIP can be verified and downloaded again without retraining.
''',
        10: '''## How the result will be judged

Compare both fresh arms and retained PR35 on frozen 666, historical 663, preserved R32 and amended R33, reporting each reference separately. Include misses, unmatched outputs, duplicates, groups with zero, one or two visible corners, empty scenes and both-corner successes at 5 and 10 native pixels with eligible misses kept in denominators. Inspect exact gained/lost rows and continuous distances; do not select a seed or epoch after seeing results.

Keep the current one-to-many inference path, confidence 0.13 and NMS 0.70. The exploratory NMS 0.50 check removed many duplicates but lost real partial fronts; the already-trained one-to-one head also worsened recall. These diagnostics do not justify a default switch. Hidden-corner estimates remain permitted, but they are unscorable against unavailable annotation slots and are never substituted into ground truth.

Keep PR35 as the retained selected checkpoint until a separate review supports changing it. Repeated development-image review is not independent evidence of reliability on new capture sessions. This notebook performs no automatic model promotion.

**Annotation standard:** box the observed frontal face and its own visible molded lip/rim. Exclude standalone strips belonging only to rack, jaw, hand or another cassette; preserve an AABB when owned fragments or an owned lip require its extrema. Use color as contextual evidence. Stored image y increases downward: top = stored ymin, bottom = stored ymax.
''',
    }
    for i, value in markdown.items():
        notebook["cells"][i]["source"] = value.splitlines(keepends=True)
    notebook["metadata"].pop("r32_handoff", None)
    notebook["metadata"]["r33_handoff"] = {
        "base_revision": BASE_REVISION, "source_revision": source, "source_tree": tree,
        "protocol_sha256": digest(protocol_bytes), "recipe_sha256": protocol["recipe_sha256"],
        "embedded_file_count": len(embedded), "published_parent_only": True,
        "training_delta": delta["summary"], "native_loss_both_arms": True,
        "seed": 46, "automatic_promotion": False,
    }
    assert len([c for c in notebook["cells"] if c["cell_type"] == "code"]) == 5
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{i}", "exec")
    return (json.dumps(notebook, indent=1, ensure_ascii=False)+"\n").encode(), notebook["metadata"]["r33_handoff"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload, meta = build(args.root)
    output = args.output or args.root/OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(json.dumps({"notebook": str(output), "sha256": digest(payload), **meta}, indent=2))
