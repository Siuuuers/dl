#!/usr/bin/env python3
"""Verify PR76 returned artifacts against the recorded r8 source and protocol."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_pose import preflight_dataset
from pose_checkpoint import select_checkpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    archive = HERE / "returned.zip"
    assert archive.stat().st_size == 39782118
    assert sha(archive) == "02442e197cf619768a3d14d933ff69841aa99387d744686bde0f6e1dfe5cddc3"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        archive_names = set(z.namelist())
    bundle = HERE / "returned"
    index = json.loads((bundle / "gpu-results-manifest.json").read_text())
    assert archive_names == {"gpu-results-manifest.json", *(f["path"] for f in index["files"])}
    for f in index["files"]:
        p = bundle / f["path"]
        assert p.stat().st_size == f["size_bytes"] and sha(p) == f["sha256"], f["path"]
    assert index["training_source"]["revision"] == "f870296f13f7c40f03786bc421aeefb9866be6de"
    assert index["training_source"]["working_tree_status"] == ""
    for p, expected in index["training_source"]["files_sha256"].items():
        assert sha(ROOT / p) == expected, p
    protocol_path = ROOT / "experiments/front-review-2026-11-19-r8.json"
    protocol_sha = sha(protocol_path)
    assert protocol_sha == "96426b9d0ba7a07b902698e320a82449c051c6790c4bac25dda709d2f181b7d3"
    protocol = json.loads(protocol_path.read_text())
    preflight = preflight_dataset(ROOT, require_weights=False)
    report = {
        "archive": {"size_bytes": archive.stat().st_size, "sha256": sha(archive), "crc_passed": True,
                    "indexed_members_verified": len(index["files"])},
        "training_source": index["training_source"],
        "protocol_sha256": protocol_sha,
        "original_validation": json.loads((ROOT / "data/manifest.json").read_text())["split_summary"]["val"],
        "runtime": index["runtime"],
        "arms": {},
    }
    for role in ("reference", "candidate"):
        selected = select_checkpoint(preflight, bundle / role / "checkpoint-manifest.json", require_weights=True)
        receipt = json.loads((bundle / role / "training-receipt.json").read_text())
        assert receipt["status"] == "complete"
        assert receipt["training_returned"] and receipt["finalization_completed"]
        assert receipt["source"]["revision"] == index["training_source"]["revision"]
        assert receipt["execution"]["exit_code"] == 0
        rows = list(csv.DictReader((bundle / role / "fit/results.csv").open()))
        assert [int(x["epoch"]) for x in rows] == list(range(1, 31))
        staging = json.loads((bundle / role / "staging-manifest.json").read_text())
        for variant in staging["variants"].values():
            for entry in variant["files"]:
                assert sha(ROOT / entry["source_path"]) == entry["source_sha256"] == entry["output_sha256"], entry["source_path"]
        report["arms"][role] = {
            "checkpoint_sha256": selected["model_sha256"],
            "selection": receipt["checkpoint_selection"],
            "epochs": len(rows),
            "batches_completed": receipt["batches_completed"],
            "images_processed": receipt["images_processed"],
            "optimizer_steps": receipt["optimizer_steps"],
            "counts": staging["variants"][role]["counts"],
            "final_epoch_csv": rows[-1],
            "runtime": receipt["runtime"],
            "recipe_id": receipt["recipe_id"],
            "recipe_sha256": receipt["recipe_sha256"],
        }
    staging = json.loads((bundle / "reference/staging-manifest.json").read_text())
    by_role = {role: {entry["path"]: entry for entry in staging["variants"][role]["files"]} for role in ("reference", "candidate")}
    changed = [p for p, entry in by_role["reference"].items()
               if entry["output_sha256"] != by_role["candidate"][p]["output_sha256"]]
    assert all(p.startswith("labels/train/") for p in changed)
    held = protocol["retained_legacy_image_ids"]
    for iid in held:
        p = "labels/train/" + iid + ".txt"
        assert by_role["reference"][p]["output_sha256"] == by_role["candidate"][p]["output_sha256"]
    report["training_difference"] = {"changed_label_files": len(changed), "changed_paths": changed,
                                     "held_train_photos_identical": len(held), "held_ids": held,
                                     "all_original_images_identical": True, "validation_labels_identical": True}
    report["initial_checkpoint_bytes_reverified"] = False
    report["note"] = "The initial checkpoint identity is verified in source and receipts; this return does not include its bytes. No training repeated. Model comparison is recorded separately."
    (HERE / "verified-return.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"archive": report["archive"], "changed_train_files": len(changed), "held_train_photos": len(held),
                      "arms": {k: {x:v[x] for x in ("epochs", "batches_completed", "images_processed", "optimizer_steps", "counts")} for k,v in report["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
