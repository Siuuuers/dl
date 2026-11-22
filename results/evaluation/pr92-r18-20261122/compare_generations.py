#!/usr/bin/env python3
"""Separate PR92 model changes from the r16-to-r18 reference-only score change."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return rows[0], {r["image_id"]: r["predictions"] for r in rows[1:]}


def delta(old, new):
    return {
        "detection": {k: new["detection"][k] - old["detection"][k]
                      for k in ("tp", "fn", "fp", "duplicate_fp", "other_fp")},
        "gained_matches": sorted(set(new["matched_identities"]) - set(old["matched_identities"])),
        "lost_matches": sorted(set(old["matched_identities"]) - set(new["matched_identities"])),
        "corner_success_changes": {r: {
            "gained": sorted(set(new["success_identities"][r]) - set(old["success_identities"][r])),
            "lost": sorted(set(old["success_identities"][r]) - set(new["success_identities"][r])),
        } for r in ("5", "10")},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--audit", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, audit = args.root.resolve(), args.audit.resolve()
    prior = root / "results/evaluation/pr90-r17-20261122"
    old = read(prior / "fixed-comparison.json")
    replay = read(audit / "previous-r17-on-r18.json")
    new = read(audit / "fixed-comparison.json")
    assert new["model_set"] == "returned_pr92_r18" and replay["model_set"] == "previous_pr90_r17"
    assert new["thresholds"] == replay["thresholds"] == old["thresholds"]
    assert new["protocol"] == replay["protocol"]
    for role in ("pr35", "reference", "candidate"):
        assert replay["models"][role]["predictions_sha256"] == old["models"][role]["predictions_sha256"]
        assert sha(prior / "inference" / ("eval-" + role) / "predictions.jsonl") == old["models"][role]["predictions_sha256"]
        assert sha(audit / "inference" / ("eval-" + role) / "predictions.jsonl") == new["models"][role]["predictions_sha256"]
    for label in ("primary", "diagnostic"):
        assert replay["references"][label] == old["references"][label]
    report = {
        "schema_version": 1, "kind": "pr92-r18-reference-and-generation-separation",
        "measurement_script_sha256": sha(__file__),
        "inputs": {"prior_pr90_fixed_sha256": sha(prior / "fixed-comparison.json"),
                   "previous_r17_on_r18_sha256": sha(audit / "previous-r17-on-r18.json"),
                   "pr92_fixed_sha256": sha(audit / "fixed-comparison.json")},
        "prior_frozen_reference_results_reproduced_exactly": True,
        "thresholds": new["thresholds"],
        "reference_only_effect_on_unchanged_r17_predictions": {
            "from": {"id": old["references"]["current_r16_exploratory"]["reference_id"],
                     "sha256": old["references"]["current_r16_exploratory"]["manifest_sha256"]},
            "to": {"id": replay["references"]["current_r18_exploratory"]["reference_id"],
                   "sha256": replay["references"]["current_r18_exploratory"]["manifest_sha256"]},
            "models": {role: delta(old["references"]["current_r16_exploratory"]["models"][role],
                                   replay["references"]["current_r18_exploratory"]["models"][role])
                       for role in ("pr35", "reference", "candidate")},
        },
        "same_reference_model_deltas": {}, "training_runtimes": {}, "representative_cases": [],
        "limits": [
            "Both PR92 arms have the same A100 training runtime; previous PR90 arms trained on L4.",
            "Cross-generation differences combine changed training supervision with changed training hardware.",
            "The previous model outputs are preserved inference arrays, not newly loaded prior weights.",
            "The pre-return r18 exploratory reference is distinct from both frozen references and all r19 edits.",
        ],
    }
    for label, reference in new["references"].items():
        before = replay["references"][label]
        assert reference["manifest_sha256"] == before["manifest_sha256"]
        report["same_reference_model_deltas"][label] = {
            "reference_manifest_sha256": reference["manifest_sha256"],
            "models": {role: delta(before["models"][role], reference["models"][role])
                       for role in ("pr35", "reference", "candidate")},
        }
    for generation, directory in (("r17", prior), ("r18", audit)):
        report["training_runtimes"][generation] = {
            role: read(directory / "checkpoint-receipts" / role / "training-receipt.json")["runtime"]
            for role in ("pr35", "reference", "candidate")}
    assert report["training_runtimes"]["r18"]["reference"] == report["training_runtimes"]["r18"]["candidate"]
    current_arrays = {role: predictions(audit / "inference" / ("eval-" + role) / "predictions.jsonl")[1]
                      for role in ("pr35", "reference")}
    assert current_arrays["pr35"] == current_arrays["reference"]
    report["fresh_control_and_pr35_prediction_arrays"] = {
        "images": 20, "exactly_equal_images": 20, "maximum_coordinate_difference_px": 0,
        "maximum_confidence_difference": 0,
        "scope": "Observed outputs at the fixed inference operating point; checkpoint artifact hashes differ.",
    }
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels, preflight_dataset
    from pose_scoring import score_image, _box_iou
    records = {r["id"]: r for r in preflight_dataset(root, require_weights=False)["records"]}
    review_path = root / "data/reviews/fronts-2026-11-22-r18/manifest.json"
    assert sha(review_path) == new["references"]["current_r18_exploratory"]["manifest_sha256"]
    labels = {r["image_id"]: r for r in read(review_path)["files"]}
    candidate_outputs = {generation: predictions(directory / "inference/eval-candidate/predictions.jsonl")[1]
                         for generation, directory in (("r17", prior), ("r18", audit))}
    for image_id, row in (("new_000011", 55), ("new_000008", 10), ("new_000008", 4),
                          ("new_000008", 28), ("new_000090", 19), ("new_000090", 45), ("new_000090", 46)):
        record = records[image_id]
        path = review_path.parent / labels[image_id]["label_path"]
        assert sha(path) == labels[image_id]["label_sha256"]
        gold = parse_pose_labels(path.read_text(), record["width"], record["height"], str(path))
        target = gold[row - 1]
        case = {"image_id": image_id, "row": row, "gold": target, "models": {}}
        for generation, outputs in candidate_outputs.items():
            predicted = outputs[image_id]
            score = score_image(gold, predicted, record["width"], record["height"], .5, .5)
            matched = next((v for v in score["matches"] if v["gold_index"] == row - 1), None)
            index, best = max(enumerate(predicted), key=lambda v: _box_iou(target["bbox_xyxy"], v[1]["bbox_xyxy"]))
            case["models"][generation] = {
                "matched": matched, "best_iou_prediction_index": index,
                "best_iou": _box_iou(target["bbox_xyxy"], best["bbox_xyxy"]), "prediction": best,
                "locatable_corner_errors_px": [math.hypot(x-gx, y-gy) if v == 2 else None
                    for (gx, gy, v), (x, y, _) in zip(target["keypoints"], best["keypoints"])],
            }
        report["representative_cases"].append(case)
    assert not args.output.exists(), "Choose a fresh output"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"same_r18_reference_candidate_delta": report["same_reference_model_deltas"]["current_r18_exploratory"]["models"]["candidate"],
                      "current_control_equals_pr35": True}, indent=2))


if __name__ == "__main__":
    main()
