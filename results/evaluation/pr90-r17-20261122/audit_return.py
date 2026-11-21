#!/usr/bin/env python3
"""Verify PR90 r17 archive, completed bundles and actual staged source bytes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ARCHIVE_SHA = "339eec148cc5d56d9bd2351f2562f09658685dd9117073338c434af63ad4917f"
SOURCE_REVISION = "08921b4ccd122798675293221145f2be41171db2"
PROTOCOL_PATH = "experiments/front-review-2026-11-22-r17.json"
PROTOCOL_SHA = "320d9e97805ddea968dd04b033037272c0fe41d77f1c149c2b05bc654f4ca12a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, archive, bundle = args.root.resolve(), args.archive.resolve(), args.bundle.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import preflight_dataset
    from pose_checkpoint import select_checkpoint
    from stage_pose_training import yaml_bytes

    require(archive.stat().st_size == 39782260 and sha(archive) == ARCHIVE_SHA, "Wrong PR90 archive")
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, "ZIP CRC failure")
        members = z.namelist()
        require(len(members) == len(set(members)), "Duplicate ZIP member")
        require(all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in members), "Unsafe ZIP member")
        require(z.read("gpu-results-manifest.json") == (bundle / "gpu-results-manifest.json").read_bytes(), "Extracted manifest differs")
    index = json.loads((bundle / "gpu-results-manifest.json").read_text())
    require(set(members) == {"gpu-results-manifest.json", *(f["path"] for f in index["files"])}, "ZIP inventory differs")
    for f in index["files"]:
        q = bundle / f["path"]
        require(q.stat().st_size == f["size_bytes"] and sha(q) == f["sha256"], "Member identity differs: " + f["path"])
    require(index["training_source"]["revision"] == SOURCE_REVISION, "Unexpected source revision")
    require(index["training_source"]["working_tree_status"] == "", "Training source was dirty")
    for path, expected in index["training_source"]["files_sha256"].items():
        preserved = subprocess.run(["git", "-C", str(root), "show", SOURCE_REVISION + ":" + path], capture_output=True, check=True).stdout
        require(hashlib.sha256(preserved).hexdigest() == expected == sha(root / path), "Training-source file differs: " + path)
    protocol_path = root / PROTOCOL_PATH
    require(sha(protocol_path) == PROTOCOL_SHA, "Protocol differs")
    protocol = json.loads(protocol_path.read_text())
    preflight = preflight_dataset(root, require_weights=False)
    original = json.loads((root / "data/manifest.json").read_text())
    for record in original["records"]:
        require(sha(root / "data" / record["image"]) == record["image_sha256"], "Original image differs: " + record["id"])
        require(sha(root / "data" / record["label"]) == record["label_sha256"], "Original label differs: " + record["id"])
    report = {
        "schema_version": 1,
        "kind": "pr90-r17-return-audit",
        "pr": {"number": 90, "head_sha": "ae390faccbf59731214933212fdce1a98868a630", "url": "https://github.com/Siuuuers/dl/pull/90"},
        "archive": {"size_bytes": archive.stat().st_size, "sha256": sha(archive), "crc_passed": True, "indexed_members_verified": len(index["files"])},
        "training_source": index["training_source"],
        "recorded_source_files_verified_against_git_revision": True,
        "protocol_sha256": PROTOCOL_SHA,
        "original_validation": original["split_summary"]["val"],
        "original_images_verified": len(original["records"]),
        "runtime": index["runtime"],
        "arms": {},
    }
    staged = {}
    for role in ("reference", "candidate"):
        selected = select_checkpoint(preflight, bundle / role / "checkpoint-manifest.json", require_weights=True)
        receipt = json.loads((bundle / role / "training-receipt.json").read_text())
        require(receipt["status"] == "complete" and receipt["training_returned"] and receipt["finalization_completed"], "Arm is incomplete: " + role)
        require(receipt["source"]["revision"] == SOURCE_REVISION, "Arm source differs: " + role)
        require(receipt["execution"]["exit_code"] == 0, "Arm failed: " + role)
        rows = list(csv.DictReader((bundle / role / "fit/results.csv").open()))
        require([int(x["epoch"]) for x in rows] == list(range(1, 31)), "Epoch inventory differs: " + role)
        staging = json.loads((bundle / role / "staging-manifest.json").read_text())
        for name, variant in staging["variants"].items():
            expected_yaml = yaml_bytes(Path(staging["output_root"]) / name)
            require(hashlib.sha256(expected_yaml).hexdigest() == variant["data_yaml_sha256"], "Staged YAML differs: " + role + "/" + name)
            for entry in variant["files"]:
                require(sha(root / entry["source_path"]) == entry["source_sha256"] == entry["output_sha256"], "Staged bytes differ: " + entry["source_path"])
        for path, expected in staging["code_sha256"].items():
            require(sha(root / path) == expected, "Staging code differs: " + path)
        staged[role] = staging
        report["arms"][role] = {
            "checkpoint_sha256": selected["model_sha256"],
            "selection": receipt["checkpoint_selection"], "epochs": len(rows),
            "batches_completed": receipt["batches_completed"], "images_processed": receipt["images_processed"],
            "optimizer_steps": receipt["optimizer_steps"], "counts": staging["variants"][role]["counts"],
            "final_epoch_csv": rows[-1], "runtime": receipt["runtime"],
            "execution": receipt["execution"], "recipe_id": receipt["recipe_id"], "recipe_sha256": receipt["recipe_sha256"],
        }
    for name in ("reference", "candidate"):
        for key in ("files", "counts", "data_yaml"):
            require(staged["reference"]["variants"][name][key] == staged["candidate"]["variants"][name][key], "Arm staging inventories differ: " + name + "/" + key)
    report["staging_yaml_reconstructed_and_verified"] = True
    report["staging_yaml_path_note"] = "Each run stages both arms under its own absolute output_root; only that root changes the YAML bytes. All image and label inventories, source paths and hashes agree across both staging receipts."
    by_role = {role: {entry["path"]: entry for entry in staged["reference"]["variants"][role]["files"]} for role in ("reference", "candidate")}
    changed = sorted(path for path, entry in by_role["reference"].items() if entry["output_sha256"] != by_role["candidate"][path]["output_sha256"])
    require(all(path.startswith("labels/train/") for path in changed), "Treatment changed images or validation labels")
    held = protocol["retained_legacy_image_ids"]
    excluded = protocol["excluded_training_image_ids"]
    require(not set(held) & set(excluded), "Held/excluded inventories overlap")
    control_path = root / index["source"]["control"]["manifest"]
    control = json.loads(control_path.read_text())
    control_changes = {entry["image_id"]: entry for entry in control["changes"]}
    records = {entry["id"]: entry for entry in original["records"]}
    for image_id in held + excluded:
        require(records[image_id]["split"] == "train", "Held/excluded image is not training data")
        path = "labels/train/" + image_id + ".txt"
        change = control_changes.get(image_id)
        expected = change["candidate_label_sha256"] if change else records[image_id]["label_sha256"]
        require(by_role["reference"][path]["output_sha256"] == by_role["candidate"][path]["output_sha256"] == expected, "Held/excluded source is not exact PR35 control: " + image_id)
    report["training_difference"] = {
        "changed_label_files": len(changed), "changed_paths": changed,
        "held_train_photos_identical_to_PR35": len(held), "held_ids": held,
        "excluded_old_train_photos_identical_to_PR35": len(excluded), "excluded_ids": excluded,
        "all_original_images_identical": True, "validation_labels_identical": True,
    }
    initial = root / original["model_checkpoint"]
    report["initial_checkpoint_bytes_reverified"] = initial.is_file() and sha(initial) == original["model_sha256"]
    require(report["initial_checkpoint_bytes_reverified"], "Initial checkpoint bytes differ")
    report["note"] = "No training repeated. Model comparison is recorded separately. Equal held supervision controls treatment difference; it does not certify physical label correctness."
    require(not args.output.exists(), "Choose a fresh audit output")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"archive": report["archive"], "changed_training_files": len(changed), "held": len(held), "excluded_old": len(excluded), "initial_checkpoint_bytes_reverified": report["initial_checkpoint_bytes_reverified"]}, indent=2))


if __name__ == "__main__":
    main()
