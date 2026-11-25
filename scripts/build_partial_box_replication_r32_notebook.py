#!/usr/bin/env python3
"""Build the five-cell, self-contained R32 seed47 replication notebook."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

from build_partial_box_loss_r31_notebook import (
    build as build_r31, publication_files as r31_files, git,
    literal_assignment, replace_assignment,
)

BASE_REVISION = "337a12d8b11f9925b493cba794f29a4ffcf848ed"
R31_TEMPLATE_SHA256 = "3487253855abe19b4199e420dd7e9717caa7ada4198f8dfd256595d8809ca047"
PROTOCOL_PATH = "experiments/partial-box-replication-2026-11-25-r32.json"
OUTPUT_PATH = "notebooks/baomai_partial_box_replication_r32.ipynb"
ASSEMBLY_NAME = "R32 notebook assembly"
ASSEMBLY_EMAIL = "r32-notebook@localhost"
ASSEMBLY_DATE = "2026-11-25T02:00:00+08:00"
ASSEMBLY_MESSAGE = "Assemble verified R32 seed47 replication sources locally"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def publication_files(root):
    files = set(r31_files(root)) | {
        PROTOCOL_PATH, "scripts/pose_checkpoint.py", "scripts/front_review_r32.py",
        "scripts/partial_box_replication_r32.py",
        "scripts/train_pose_partial_box_replication_r32.py",
        "scripts/build_partial_box_replication_r32_notebook.py",
        "docs/annotation-boundaries-r32.md",
    }
    directory = root/"data/reviews/fronts-2026-11-25-r32"
    files.update(p.relative_to(root).as_posix() for p in directory.rglob("*") if p.is_file())
    ledger = "results/review/front-review-20261125-r32/"
    files.update(ledger+name for name in (
        "row-review.json", "independent-surface.json", "independent-surface-extra.json",
        "independent-root-endorsement.json", "independent-points.json", "local-color-reference.json",
        "operations.json", "row-maps.json", "request-coverage.json", "review-status.json",
        "amendment-summary.json",
    ))
    for name in files:
        path = root/name
        assert path.is_file() and not path.is_symlink(), name
        assert path.suffix in (".json", ".txt", ".py", ".md"), name
    return sorted(files)


def assemble_identity(root, members):
    embedded = {}
    with tempfile.TemporaryDirectory(prefix="r32-notebook-index-") as tmp:
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
            embedded[relative] = {
                "sha256": digest(payload), "size_bytes": len(payload), "git_mode": mode,
                "zlib_base64": base64.b64encode(zlib.compress(payload, 9)).decode(),
            }
        tree = git(root, "write-tree", env=env).decode().strip()
        source = git(root, "-c", "commit.gpgsign=false", "commit-tree", tree,
                     "-p", BASE_REVISION, data=(ASSEMBLY_MESSAGE+"\n").encode(), env=env).decode().strip()
    return source, tree, embedded


def build(root):
    template, _ = build_r31(root)
    assert digest(template) == R31_TEMPLATE_SHA256, "Immutable R31 notebook template differs"
    notebook = json.loads(template)
    protocol_bytes = (root/PROTOCOL_PATH).read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["experiment_id"] == "partial-box-replication-r32"
    assert protocol["recipe_id"] == "reviewed-incremental-fronts-seed47-v1"
    for ref in (protocol["accepted_review"], protocol["evaluation"]["current_review_reference"]):
        assert digest((root/ref["path"]).read_bytes()) == ref["sha256"]
    assert digest((root/protocol["candidate_manifest"]).read_bytes()) == protocol["candidate_manifest_sha256"]
    assert protocol["replication"]["report_both_seeds"] and not protocol["replication"]["select_better_seed"]
    source, tree, embedded = assemble_identity(root, publication_files(root))
    old_helpers = literal_assignment("".join(notebook["cells"][1]["source"]), "BASE_HELPER_SHA256")
    helpers = {name: digest(git(root, "show", BASE_REVISION+":"+name))
               for name in sorted(set(old_helpers)-set(embedded))}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        value = "".join(cell["source"])
        value = value.replace("baomai-partial-box-loss-r31", "baomai-partial-box-replication-r32")
        value = value.replace("from partial_box_loss_r31 import", "from partial_box_replication_r32 import")
        value = value.replace("scripts/train_pose_partial_box_loss_r31.py", "scripts/train_pose_partial_box_replication_r32.py")
        value = value.replace("Verified local R31 training source:", "Verified local R32 replication source:")
        value = value.replace("Fixed: seed 46,", "Fixed: seed 47,")
        cell["source"] = value.splitlines(keepends=True)
        cell["outputs"], cell["execution_count"] = [], None
    first = "".join(notebook["cells"][1]["source"])
    for key, value in {
        "SOURCE_REVISION": source, "PROTOCOL_PATH": PROTOCOL_PATH,
        "PROTOCOL_SHA256": digest(protocol_bytes), "EXPERIMENT_ID": "partial-box-replication-r32",
        "BASE_REVISION": BASE_REVISION, "EMBEDDED_FILES": embedded, "BASE_HELPER_SHA256": helpers,
        "ROLE_RECIPES": {role: protocol["recipe_id"] for role in ("control", "candidate")},
        "RECIPE_SHA256": protocol["recipe_sha256"],
        "ASSEMBLY_NAME": ASSEMBLY_NAME, "ASSEMBLY_EMAIL": ASSEMBLY_EMAIL,
        "ASSEMBLY_DATE": ASSEMBLY_DATE, "ASSEMBLY_MESSAGE": ASSEMBLY_MESSAGE,
    }.items():
        first = replace_assignment(first, key, value)
    notebook["cells"][1]["source"] = first.splitlines(keepends=True)
    preflight = "".join(notebook["cells"][5]["source"])
    preflight += "\nassert document['replication']['prespecified_seed'] == 47\n"
    preflight += "assert document['replication']['training_labels_changed'] == 0\n"
    notebook["cells"][5]["source"] = preflight.splitlines(keepends=True)
    markdown = {
        0: """# Cassette fronts: R32 seed 47 replication

**Upload this notebook to Colab, choose an L4 or A100 GPU, select Run all, and return `baomai-partial-box-replication-r32-results.zip`.**

This run asks one question: **does PR110's small partial-front detection change repeat with a second, prespecified seed?** It repeats the same comparison with seed 47. The control uses native box weight 1; the candidate uses weight 2 only for the native box-regression contributions of already-assigned partial fronts. The R31 loss implementation is reused unchanged.

On the frozen 666-object reference, PR110 reduced misses from 36 to 35, increased unmatched outputs from 108 to 111, and left 5/10-pixel corner-pair success unchanged. On the preserved R31 development labels, misses fell from 15 to 12, but pair success at 5 pixels fell from 572 to 570. This is a small trade-off, not a demonstrated improvement. Several changes cross a scoring threshold by a small amount.

Both roles use exactly the **same existing R31 training labels** and start independently from original epoch 173. R32's 23 changed development rows do not enter training. Each role uses 137 images once per epoch: **30 epochs, 1,050 batches, 4,110 image presentations and 78 optimizer updates**, at batch 4, image size 1280, FP32, AdamW and initial learning rate 0.0001.

The five code cells prepare verified inputs, check the runtime, run the matched pair and package the return ZIP. A four-batch resource trial runs first; its weights are discarded. Source files are embedded and hash-verified against a published parent, so an unpublished GitHub commit is not required.
""",
        2: """## 1. Prepare the verified source and inputs

Fetch the published parent and assemble the embedded R31 prerequisites plus R32 protocol and review. Verify all 157 original photographs, the epoch 173 initializer and the Git LFS pointer identities. Existing incompatible or interrupted folders are preserved; an error identifies the folder to archive before retrying.
""",
        4: """## 2. Verify the runtime and unchanged training supervision

Keep Colab's matching CUDA PyTorch and torchvision installation, then install the pinned trainer. Preflight checks the exact native loss sources, protocol, recipe, accepted R31 training overlay and clarified R32 development reference.

Both roles train on the same 3,784 objects in 137 images. Original trainer validation remains 20 images with 664 objects. The R32 development reference contains 661 objects and 583 eligible two-corner pairs, including one explicitly removed clipped landmark. These three inventories serve different purposes and remain separate.

The native Pose26 head, its E2ELoss branches, normalized L1 box auxiliary, classification, keypoint and RLE objectives remain unchanged. Seed 47 is the only numerical recipe change from PR110; the within-seed treatment remains the box multiplier.
""",
        6: """## 3. Run the resource trial and matched pair

Run the four-batch candidate trial on CUDA and discard its weights. Then start both complete roles afresh from epoch 173 with the same seed, runtime, labels, image order and update budget.

Post-augmentation target counts may differ from seed 46. The two seed 47 roles must match each other; actual counts are recorded rather than copied from the earlier run. Completed runs can be verified and recovered from Drive. Interrupted runs remain available for diagnosis and restart fresh; partial optimizer state is not resumed.
""",
        8: """## 4. Save and return the results

The ZIP contains both final portable checkpoints, source and supervision identities, the fixed protocol, runtime receipts, per-branch target assignments, image exposure and training results. Packaging verifies equal budgets and identical staged labels. Keep the Drive copy and downloaded ZIP, then return the ZIP for evaluation.
""",
        10: """### How to read the returned result

Keep confidence 0.13, NMS IoU 0.70, matching IoU 0.50 and endpoint confidence 0.50. Report the frozen 666-object reference, historical 663-object reference, preserved R31 reference and R32 development reference separately. The frozen references allow like-for-like historical comparisons; they are not a claim that every old boundary is correct. The repeatedly reviewed development photographs remain exploratory.

Report seed 46 and seed 47 independently and together. **Do not choose whichever seed looks better.** Inspect partial-front misses, exact gained/lost identities, continuous IoU and corner distances, 5/10-pixel pair success with eligible misses retained, unmatched outputs, duplicates and empty scenes. A threshold crossing alone is not a new visual capability.

The prospective diagnostic screen asks for fewer frozen-reference misses without more unmatched or empty-scene outputs, and without a reduction in 5/10-pixel pair success. It is not a statistical significance test or a promotion rule. A favorable seed 47 result cannot erase a seed 46 penalty.

If this second seed does not show a clean favorable trade-off, stop increasing partial-box weighting. Keep the native training default and retained PR35 checkpoint while reviewing the evidence. Two seeds cannot establish reliability on new capture sessions, and there is no automatic model promotion.

**Annotation language:** in this review, the user's `ymax` means the top and `ymin` means the bottom. Stored image y increases downward: top = stored `ymin`, bottom = stored `ymax`. The accepted standard includes the target's own visible frontal lip, rim and protrusion; it excludes the broad perforated upper plane, neighbors, rack and gripper. Color supports component ownership but is not a universal classifier.
""",
    }
    for i, text in markdown.items():
        notebook["cells"][i]["source"] = text.splitlines(keepends=True)
    notebook["metadata"].pop("r31_handoff", None)
    notebook["metadata"]["r32_handoff"] = {
        "base_revision": BASE_REVISION, "source_revision": source, "source_tree": tree,
        "protocol_sha256": digest(protocol_bytes), "recipe_sha256": protocol["recipe_sha256"],
        "embedded_file_count": len(embedded), "published_parent_only": True,
        "training_labels_changed_from_r31": 0, "prespecified_seed": 47,
        "r31_criterion_reused_unchanged": True, "classification_objective_unchanged": True,
    }
    assert len([c for c in notebook["cells"] if c["cell_type"] == "code"]) == 5
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{i}", "exec")
    return (json.dumps(notebook, indent=1, ensure_ascii=False)+"\n").encode(), notebook["metadata"]["r32_handoff"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root/OUTPUT_PATH
    payload, metadata = build(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(json.dumps({"notebook": str(output), "sha256": digest(payload), **metadata}, indent=2))
