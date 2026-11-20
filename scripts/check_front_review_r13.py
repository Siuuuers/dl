#!/usr/bin/env python3
"""Stage the r13 comparison and check held and excluded photographs separately."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from pose_checkpoint import get_recipe, recipe_identity
from stage_pose_training import stage_datasets
from verify_front_review_r13 import (
    CANDIDATE, PARENT, PROTOCOL_PATH, REVIEW_PATH, verify_candidate,
)
from verify_pose_candidate import require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists() and not output.is_relative_to(root),
            "Choose a fresh verification directory outside the checkout.")
    inspection = verify_candidate(root)
    source = json.loads((root / "data/manifest.json").read_text())
    records = source["records"]
    original_before = {row["id"]: sha(root / "data" / row["label"]) for row in records}
    require(all(original_before[row["id"]] == row["label_sha256"] for row in records),
            "An original label differs from the source manifest.")
    manifest = json.loads((root / CANDIDATE).read_text())
    protocol = json.loads((root / PROTOCOL_PATH).read_text())
    prior = json.loads((root / "experiments/front-review-2026-11-19-r8.json").read_text())
    review = json.loads((root / REVIEW_PATH).read_text())
    held = set(manifest["retained_legacy_image_ids"])
    excluded = set(manifest["excluded_training_image_ids"])
    old_ids = {row["id"] for row in records if row["id"].startswith("old_")}
    require(len(old_ids) == 31 and excluded == old_ids and not held.intersection(excluded),
            "The 31 confirmed old photos must be separate treatment exclusions, not holds.")
    require(protocol["excluded_training_image_ids"] == inspection["excluded_training_image_ids"]
            == sorted(excluded), "Protocol, review verification and treatment exclusions differ.")
    require(not (held | excluded).intersection(manifest["changed_images_vs_control"]),
            "A held or excluded photo became a treatment-only change.")

    output.mkdir(parents=True)
    staged = stage_datasets(root, root / CANDIDATE, output / "staging")
    require(staged["actual_original_pngs_verified"] == 157,
            "The actual 157 original PNGs were not all verified.")
    control_path = root / manifest["control"]["manifest"]
    control = json.loads(control_path.read_text())
    changes = {change["image_id"]: change for change in control["changes"]}
    val_count = held_count = excluded_count = old_review_count = 0
    for row in records:
        iid = row["id"]
        original = (root / "data" / row["label"]).read_bytes()
        arms = [(output / "staging" / arm / row["label"]).read_bytes()
                for arm in ("reference", "candidate")]
        if row["split"] == "val":
            require(arms[0] == arms[1] == original, f"Original validation changed: {iid}")
            val_count += 1
        if iid in held | excluded:
            change = changes.get(iid)
            expected = ((control_path.parent / change["candidate_label"]).read_bytes()
                        if change else original)
            require(arms[0] == arms[1] == expected,
                    f"Held/excluded photo does not use exact PR35 in both arms: {iid}")
            held_count += iid in held
            excluded_count += iid in excluded
        if iid in old_ids:
            current = (root / REVIEW_PATH).parent / row["label"]
            previous = (root / PARENT["path"]).parent / row["label"]
            require(current.read_bytes() == previous.read_bytes(),
                    f"A confirmed old review label changed from r12: {iid}")
            old_review_count += 1
    require(val_count == 20 and held_count == len(held) and excluded_count == 31
            and old_review_count == 31, "Staged cohort accounting differs.")

    for role, objects in (("primary", 666), ("diagnostic", 663)):
        for suffix in ("reference", "reference_sha256", "counts"):
            require(protocol["evaluation"][role + "_" + suffix]
                    == prior["evaluation"][role + "_" + suffix],
                    f"Frozen {role} reference changed.")
        require(sha(root / protocol["evaluation"][role + "_reference"])
                == protocol["evaluation"][role + "_reference_sha256"]
                and protocol["evaluation"][role + "_counts"]["objects"] == objects,
                f"Frozen {role} reference bytes/count differ.")
    recipe_id = "reviewed-incremental-fronts-seed46-v1"
    recipe = get_recipe(recipe_id)
    expected_settings = {"seed": 46, "epochs": 30, "batch": 4, "imgsz": 1280,
                         "amp": False, "optimizer": "AdamW", "lr0": 0.0001}
    require(all(recipe[key] == value for key, value in expected_settings.items()),
            "The fixed numerical recipe changed.")
    require(all(protocol["roles"][role]["recipe_id"] == recipe_id
                and protocol["roles"][role]["recipe_sha256"]
                == recipe_identity(recipe_id)["recipe_sha256"]
                for role in ("reference", "candidate")), "Comparison recipes differ.")
    require(source["model_epoch"] == 173
            and protocol["initial_checkpoint_sha256"] == source["model_sha256"]
            and protocol["execution"]["checkpoint_selection"] == "final_epoch"
            and protocol["execution"]["initialization"] == "original_epoch173_independently_for_each_arm"
            and protocol["execution"]["automatic_promotion"] is False,
            "Initialization, final-epoch selection or promotion policy changed.")
    require(sha(root / PARENT["path"]) == PARENT["sha256"], "Parent review drifted.")
    confirmed = (root / REVIEW_PATH).parent / "labels/train/old_000013.txt"
    require(confirmed.read_bytes() == (root / "data/labels/train/old_000013.txt").read_bytes(),
            "The confirmed original old_000013 bytes changed.")
    require(all(sha(root / "data" / row["label"]) == original_before[row["id"]]
                for row in records), "Verification modified an original label.")
    staging = json.loads((output / "staging/staging-manifest.json").read_text())
    require(staging["source"]["excluded_training_image_ids"] == sorted(excluded),
            "Staging provenance omitted the separate treatment exclusions.")
    report = {
        "status": "verified", "training_run": False,
        "review": {"path": REVIEW_PATH, "sha256": sha(root / REVIEW_PATH)}, "parent": PARENT,
        "candidate_manifest_sha256": sha(root / CANDIDATE),
        "protocol_sha256": sha(root / PROTOCOL_PATH),
        "original_images_verified": 157, "all_original_label_bytes_unchanged": True,
        "staged_counts": staged["counts"], "complete_review_counts": review["expected_counts"],
        "original_validation_photos_byte_identical_in_both_arms": val_count,
        "held_training_photos_exact_pr35_in_both_arms": held_count,
        "held_training_image_ids": sorted(held),
        "excluded_training_photos_exact_pr35_in_both_arms": excluded_count,
        "excluded_training_image_ids": sorted(excluded),
        "confirmed_old_review_photos_byte_identical_to_r12": old_review_count,
        "old_000013_review_exact_original_bytes": True,
        "frozen_reference_object_counts": {"primary": 666, "diagnostic": 663},
        "fixed_training_recipe": expected_settings,
        "changed_training_photos_vs_control": len(manifest["changed_images_vs_control"]),
        "staging_manifest_sha256": sha(output / "staging/staging-manifest.json"),
        "source_files_sha256": {f"scripts/{name}": sha(root / "scripts" / name) for name in (
            "check_front_review_r13.py", "verify_front_review_r13.py", "build_front_review_r13.py",
            "front_review_experiment.py", "stage_pose_training.py", "train_pose_comparison.py",
            "verify_incremental_pose_candidate.py")},
        "limits": ["Mechanical checks bind the saved review decisions; they do not certify every physical boundary.",
                   "This check stages images and labels and runs no model training."],
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
