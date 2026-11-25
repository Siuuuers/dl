#!/usr/bin/env python3
"""Build the slim R37 notebook from an immutable published source commit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_partial_box_loss_r31_notebook import replace_assignment
from front_review_r37_training import CANDIDATE_PATH, PROTOCOL_PATH, REVIEW_PATH
from native_front_review_r37 import load_protocol
from train_pose_comparison import SOURCE_FILES

TEMPLATE_PATH = "notebooks/baomai_front_review_r36.ipynb"
TEMPLATE_SHA256 = "26226fc729aa2a0b29689db236e17d61ff1ffe7efb33c1deb7311903154c55db"
OUTPUT_PATH = "notebooks/baomai_front_review_r37.ipynb"

SOURCE_LOADER = '''SOURCE_URL = "https://github.com/Siuuuers/dl.git"

def assemble_source(repo, origin=SOURCE_URL):
    repo = Path(repo)
    assert not repo.is_symlink(), f"Source folder is a symlink: {repo}. Choose a fresh RUN_BASE."
    assert not repo.exists() or repo.is_dir(), f"Preserve the existing file at {repo} and choose a fresh RUN_BASE."
    git_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    if not repo.exists():
        run(["git", "init", repo], env=git_env)
        run(["git", "remote", "add", "origin", origin], cwd=repo)
        run(["git", "fetch", "--depth=1", "origin", SOURCE_REVISION], cwd=repo, env=git_env)
        run(["git", "checkout", "--detach", SOURCE_REVISION], cwd=repo, env=git_env)
    else:
        probe = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=repo,
                               capture_output=True, text=True)
        assert probe.returncode == 0, (f"Source checkout is incomplete at {repo}. Preserve or rename only "
            "that source folder, then rerun. Keep the run and Drive folders intact.")
        assert probe.stdout.strip() == SOURCE_REVISION, (
            "Existing source belongs to different pins; preserve it and choose a fresh RUN_BASE.")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == SOURCE_REVISION
    assert sha256(repo / PROTOCOL_PATH) == PROTOCOL_SHA256
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip(), (
        "Existing source has local changes. Preserve or rename only the source folder; keep completed runs intact.")
    print("Verified published R37 source:", SOURCE_REVISION)
    return SOURCE_REVISION
'''


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify_source_commit(root, revision, protocol):
    assert re.fullmatch(r"[0-9a-f]{40}", revision), "Pin the complete published source commit."
    names = set(SOURCE_FILES) | {PROTOCOL_PATH, CANDIDATE_PATH, REVIEW_PATH,
                                 protocol["training_input_delta"]["path"]}
    for relative in (CANDIDATE_PATH, REVIEW_PATH):
        document = json.loads((root / relative).read_bytes())
        parent = Path(relative).parent
        if relative == CANDIDATE_PATH:
            names.update((parent / row["candidate_label"]).as_posix() for row in document["changes"])
        else:
            names.update((parent / row["label_path"]).as_posix() for row in document["files"])
    for name in sorted(names):
        published = subprocess.check_output(["git", "show", revision+":"+name], cwd=root)
        assert published == (root / name).read_bytes(), "Source commit differs: " + name


def build(root, revision):
    root = Path(root).resolve()
    template = (root / TEMPLATE_PATH).read_bytes()
    assert digest(template) == TEMPLATE_SHA256, "Published R36 notebook template differs."
    context = load_protocol(root)
    protocol = context["document"]
    verify_source_commit(root, revision, protocol)
    notebook = json.loads(template)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            value = "".join(cell["source"]).replace("baomai-front-review-r36", "baomai-front-review-r37")
            value = value.replace("native_front_review_r36", "native_front_review_r37")
            value = value.replace("baomai-front-review-r37-20261125", "baomai-front-review-r37-20261126")
            cell["source"], cell["outputs"], cell["execution_count"] = value.splitlines(keepends=True), [], None
    first = "".join(notebook["cells"][1]["source"])
    assert first.count("SOURCE_URL = ") == 1, "Published R36 source-loader prefix differs."
    first = first.split("SOURCE_URL = ", 1)[0] + SOURCE_LOADER
    for name, value in {"SOURCE_REVISION": revision, "PROTOCOL_PATH": PROTOCOL_PATH,
                        "PROTOCOL_SHA256": context["sha256"], "EXPERIMENT_ID": "front-review-r37"}.items():
        first = replace_assignment(first, name, value)
    notebook["cells"][1]["source"] = first.splitlines(keepends=True)
    inputs = "".join(notebook["cells"][3]["source"])
    restore_marker = "def verify_original_inputs(repo):"
    assert inputs.count(restore_marker) == 1, "Original-input restore function differs."
    inputs = restore_marker + inputs.split(restore_marker, 1)[1]
    notebook["cells"][3]["source"] = inputs.splitlines(keepends=True)
    preflight = "".join(notebook["cells"][5]["source"])
    preflight = preflight.replace('assert DELTA["summary"]["classes_visibility_preserved"] is True',
                                  'assert DELTA["summary"]["classes_preserved"] is True')
    preflight = preflight.replace("Reference uses accepted R35 labels; candidate uses the exact eligible R36 corrections.",
                                  "Reference uses accepted R36 labels; candidate uses the exact eligible R37 corrections.")
    delta_check = 'assert DELTA["summary"]["classes_preserved"] is True'
    assert preflight.count(delta_check) == 1, "Training-delta preflight differs."
    assert preflight.count('print("Request scope:"') == 1, "Pending-request display differs."
    assert preflight.count('assert DELTA["held_request_rows"] == document["held_request_rows"]') == 1
    notebook["cells"][5]["source"] = preflight.splitlines(keepends=True)
    delta = context["candidate_verification"]["training_input_delta"]["report"]["summary"]
    count, photos = delta["changed_rows"], delta["changed_photos"]
    assert delta["visibility_changes"] == 0 and delta["classes_visibility_preserved"] is True
    notebook["metadata"]["colab"]["name"] = Path(OUTPUT_PATH).name
    notebook["metadata"].pop("r36_handoff", None)
    notebook["metadata"]["r37_handoff"] = {
        "source_revision": revision, "published_source_revision": revision,
        "protocol_sha256": context["sha256"], "recipe_sha256": protocol["recipe_sha256"],
        "embedded_file_count": 0, "training_delta": delta,
        "native_loss_both_arms": True, "seed": 46, "automatic_promotion": False,
        "request_completion": protocol["request_completion"],
        "held_request_rows": protocol["held_request_rows"],
    }
    pending_lines = ["| Image / row | Fields kept from R36 | Reason |",
                     "|---|---|---|"]
    for pending in protocol["held_request_rows"]:
        fields = "Whole row" if pending["preserve_row"] else ", ".join(pending["pending_fields"])
        reason = pending["reason"].replace("|", "\\|").replace("\n", " ")
        pending_lines.append(f"| `{pending['image_id']}:{pending['row']}` | {fields} | {reason} |")
    pending_details = ("\n### Requests still pending\n\n" + "\n".join(pending_lines) +
        "\n\nThese fields remain unchanged. The verified corrections can run independently; "
        "a later clarified amendment requires a separately reviewed source snapshot.\n") if protocol["held_request_rows"] else ""
    markdown = {
        0: f'''# Cassette fronts: R37 reviewed-annotation comparison

**Choose an L4 or A100 GPU in Colab, select Run all, and return `baomai-front-review-r37-results.zip`.**

This is a **regression check for two marked annotation corrections**: the right box edge in `new_000006:53` and the left corner in `new_000034:35`. The reference uses eligible R36 training labels. The candidate changes **{count} rows across {photos} training photographs**, with unchanged object, class and corner-availability inventories. Every other training setting is shared. These local corrections do not promise to solve all missed partial fronts.

Both marked training requests are applied. The inherited development-point request listed below remains pending and retains its entire R36 row.

Both arms use the unmodified native Pose26 loss and start independently from original epoch 173. Each receives 137 training images, including 13 empty scenes, once per epoch: **30 epochs, 1,050 batches, 4,110 image presentations and 78 optimizer updates**, seed 46, batch 4, image size 1280, FP32 and AdamW. Final-epoch selection is fixed.

The notebook loads a published immutable source commit. Labels and review records live in the repository, keeping this notebook small. Five code cells prepare the workspace, restore inputs, verify the runtime, run a discarded four-batch trial and the matched pair, then package the result. Compatible completed runs are reused and backed up to Drive when available.
''',
        2: '''## 1. Restore source and original inputs

The exact source commit, protocol hash, 157 original image hashes and unchanged epoch-173 initializer are checked. Existing incompatible or incomplete folders are preserved for inspection. The notebook does not embed source files, labels, photos or weights.
''',
        4: f'''## 2. Verify the exact training change

Only the accepted corrections in {count} eligible training rows differ. All 137 training photographs remain, with 3,784 objects and 13 empty scenes. Held photograph 015 and the 31 excluded old-series photographs retain their exact previous supervision in both arms. The original trainer-validation split remains 20 images with 664 objects.

All 20 development label files remain byte-identical to R36. Earlier R35 development deletions stay outside both training arms. The inherited pending request retains its R36 row. Frozen 666 and historical 663 evaluation references remain unchanged. The current development review remains exploratory evidence. A corrected training label is not a measured model improvement.
{pending_details}
''',
        6: '''## 3. Run the matched native-loss pair

Use one GPU runtime for both arms. The four-batch resource trial checks real loading, gradients, parameter updates and memory; its weights are discarded. Each full arm then starts afresh from original epoch 173. This experiment changes annotation supervision only.

These are fresh fine-tuning sessions with reset optimizer and learning-rate state, not resumed epochs 174–203. Native gradient accumulation explains why 1,050 batches give 78 optimizer updates. A random-weight restart or a new COCO-pretrained training run would be a separate experiment with a different initialization and a suitable declared budget.
''',
        8: '''## 4. Return the verified result ZIP

Return `baomai-front-review-r37-results.zip` with both final checkpoints, raw logs, receipts, protocol and exact training delta. Packaging verifies artifact bytes, supervision and runtime equality. A completed compatible ZIP can be verified and downloaded again without retraining.
''',
        10: '''## How to interpret the returned run

Keep inference fixed: box confidence 0.13, endpoint confidence 0.50, image size 1280, one-to-many output and NMS IoU 0.70. Matching IoU remains 0.50. Report frozen 666 and historical 663 separately, with R36/R37 development references clearly marked exploratory.

Prioritize missed visible fronts, especially fronts with zero or one locatable corner. Report unmatched outputs and their duplicate subset, empty-scene outputs, and both-corner successes at 5/10 original-image pixels alongside misses. Keep missed eligible objects in success denominators. A box-localization mismatch can create both an unmatched output and a missed reference; it is not automatically a completely absent prediction.

PR35 remains selected. This notebook does not promote a model, change inference thresholds, add hidden annotation points or claim reliability on unseen camera scenes. A hidden-corner estimate at inference has no coordinate truth when the annotation is unavailable.

The existing checkpoint already contains trained one-to-many and one-to-one heads. A bounded PR121 one-to-one probe reduced total unmatched outputs but increased misses and duplicates, lost a 10-pixel pair and returned one invalid zero-area box. The default one-to-many plus NMS path is therefore retained.
''',
    }
    for index, value in markdown.items():
        notebook["cells"][index]["source"] = value.splitlines(keepends=True)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"R37 cell {index}", "exec")
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert code.count("def assemble_source(") == 1, "A later cell overrides the published-source loader."
    assert all(name not in code for name in ("BASE_REVISION", "PUBLISHED_SOURCE_REVISION", "SOURCE_TREE",
                                             "ASSEMBLY_NAME", "ASSEMBLY_EMAIL", "ASSEMBLY_DATE", "ASSEMBLY_MESSAGE")), \
        "Historical commit-reconstruction state survived notebook generation."
    payload = (json.dumps(notebook, ensure_ascii=False, indent=1)+"\n").encode()
    assert len(payload) < 50000 and b"EMBEDDED_FILES" not in payload
    output = root / OUTPUT_PATH
    output.write_bytes(payload)
    return {"path": str(output), "size_bytes": len(payload), "sha256": digest(payload),
            "source_revision": revision, "protocol_sha256": context["sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.source_revision), indent=2))
