#!/usr/bin/env python3
"""Compare PR90 fixed results with preserved PR88 predictions on identical labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def predictions(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return rows[0], {r["image_id"]: r["predictions"] for r in rows[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--previous-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, audit = args.root.resolve(), args.audit.resolve()
    prior = root / "results/evaluation/pr88-r15-20261122"
    current = load(audit / "fixed-comparison.json")
    previous = load(prior / "fixed-comparison.json")
    prior_review = load(prior / "post-r16-diagnostic.json")
    assert sha(args.previous_replay) == sha(prior / "post-r16-diagnostic.json")
    assert current["thresholds"] == previous["thresholds"] == prior_review["thresholds"]
    assert current["references"]["current_r16_exploratory"]["manifest_sha256"] == prior_review["reference"]["sha256"]
    for role in ("pr35", "reference", "candidate"):
        assert sha(prior / "inference" / ("eval-" + role) / "predictions.jsonl") == previous["models"][role]["predictions_sha256"]
        assert sha(audit / "inference" / ("eval-" + role) / "predictions.jsonl") == current["models"][role]["predictions_sha256"]
    report = {
        "schema_version": 1,
        "kind": "pr90-versus-pr88-same-reference-comparison",
        "measurement_script_sha256": sha(__file__),
        "prior_fixed_comparison_sha256": sha(prior / "fixed-comparison.json"),
        "prior_r16_diagnostic_sha256": sha(prior / "post-r16-diagnostic.json"),
        "prior_r16_replay_sha256": sha(args.previous_replay),
        "prior_r16_replay_byte_identical": True,
        "current_fixed_comparison_sha256": sha(audit / "fixed-comparison.json"),
        "thresholds": current["thresholds"],
        "new_prior_model_inference": False,
        "baseline_prediction_consistency": {},
        "references": {},
        "representative_cases": [],
        "limits": [
            "Previous-generation inference is the preserved PR88 output, not newly run weights.",
            "Recorded runtime versions match, but fresh baseline arrays show small numeric differences.",
            "A one-match development-set change is not proof of improved generalization.",
            "The comparison is r15 versus r17; it does not isolate the r16-to-r17 annotation delta.",
        ],
    }
    for role in ("pr35", "reference"):
        old_header, old = predictions(prior / "inference" / ("eval-" + role) / "predictions.jsonl")
        new_header, new = predictions(audit / "inference" / ("eval-" + role) / "predictions.jsonl")
        assert old_header["runtime"] == new_header["runtime"]
        assert set(old) == set(new)
        assert all(len(old[k]) == len(new[k]) for k in old)
        coords, confidence = [], []
        for image_id in old:
            for a, b in zip(old[image_id], new[image_id]):
                assert a["class_id"] == b["class_id"]
                coords.extend(abs(x-y) for x, y in zip(a["bbox_xyxy"], b["bbox_xyxy"]))
                confidence.append(abs(a["confidence"] - b["confidence"]))
                for ak, bk in zip(a["keypoints"], b["keypoints"]):
                    coords.extend(abs(x-y) for x, y in zip(ak[:2], bk[:2]))
                    confidence.append(abs(ak[2]-bk[2]))
        report["baseline_prediction_consistency"][role] = {
            "recorded_runtime_equal": True,
            "same_count_per_image": True,
            "prediction_order_comparison": "original emitted order",
            "exactly_equal_image_arrays": sum(old[k] == new[k] for k in old),
            "images": len(old),
            "maximum_coordinate_difference_px": max(coords, default=0),
            "maximum_confidence_difference": max(confidence, default=0),
        }
    for label, current_reference in current["references"].items():
        old_reference = prior_review if label == "current_r16_exploratory" else previous["references"][label]
        old = old_reference["models"]["candidate"]
        new = current_reference["models"]["candidate"]
        for role in ("pr35", "reference"):
            a, b = old_reference["models"][role], current_reference["models"][role]
            for key in ("tp", "fn", "fp", "duplicate_fp", "other_fp", "returned_endpoints"):
                assert a["detection"][key] == b["detection"][key]
            assert a["matched_identities"] == b["matched_identities"]
            assert a["success_identities"] == b["success_identities"]
        old_scenes = {r["image_id"]: r for r in old["scenes"]}
        report["references"][label] = {
            "reference_manifest_sha256": current_reference["manifest_sha256"],
            "baseline_detection_counts_and_success_identities_unchanged": True,
            "candidate_detection_delta": {k: new["detection"][k] - old["detection"][k] for k in ("tp", "fn", "fp", "duplicate_fp", "other_fp")},
            "gained_matches": sorted(set(new["matched_identities"]) - set(old["matched_identities"])),
            "lost_matches": sorted(set(old["matched_identities"]) - set(new["matched_identities"])),
            "corner_success_changes": {radius: {
                "gained": sorted(set(new["success_identities"][radius]) - set(old["success_identities"][radius])),
                "lost": sorted(set(old["success_identities"][radius]) - set(new["success_identities"][radius])),
            } for radius in ("5", "10")},
            "changed_scene_counts": [{"previous": old_scenes[r["image_id"]], "current": r}
                for r in new["scenes"] if any(r[k] != old_scenes[r["image_id"]][k]
                for k in ("matched", "misses", "extras", "duplicates"))],
        }
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels, preflight_dataset
    from pose_scoring import score_image, _box_iou
    records = {r["id"]: r for r in preflight_dataset(root, require_weights=False)["records"]}
    review_path = root / "data/reviews/fronts-2026-11-22-r16/manifest.json"
    assert sha(review_path) == current["references"]["current_r16_exploratory"]["manifest_sha256"]
    labels = {r["image_id"]: r for r in load(review_path)["files"]}
    outputs = {role: predictions(path)[1] for role, path in {
        "r15": prior / "inference/eval-candidate/predictions.jsonl",
        "r17": audit / "inference/eval-candidate/predictions.jsonl",
    }.items()}
    for image_id, row in [("new_000089", 49), ("new_000059", 11), ("new_000090", 19),
                          ("new_000090", 45), ("new_000090", 46), ("new_000008", 10)]:
        record = records[image_id]
        path = review_path.parent / labels[image_id]["label_path"]
        assert sha(path) == labels[image_id]["label_sha256"]
        gold = parse_pose_labels(path.read_text(), record["width"], record["height"], str(path))
        target = gold[row - 1]
        case = {"image_id": image_id, "row": row, "gold": target, "models": {}}
        for role, all_predictions in outputs.items():
            predicted = all_predictions[image_id]
            score = score_image(gold, predicted, record["width"], record["height"], .5, .5)
            matched = next((p for p in score["matches"] if p["gold_index"] == row - 1), None)
            index, best = max(enumerate(predicted), key=lambda p: _box_iou(target["bbox_xyxy"], p[1]["bbox_xyxy"]))
            case["models"][role] = {
                "matched": matched,
                "best_iou_prediction_index": index,
                "best_iou": _box_iou(target["bbox_xyxy"], best["bbox_xyxy"]),
                "prediction": best,
                "locatable_corner_errors_px": [math.hypot(x-gx, y-gy) if visibility == 2 else None
                    for (gx, gy, visibility), (x, y, _) in zip(target["keypoints"], best["keypoints"])],
            }
        report["representative_cases"].append(case)
    assert not args.output.exists(), "Choose a fresh output"
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baseline_consistency": report["baseline_prediction_consistency"],
                      "r16_delta": report["references"]["current_r16_exploratory"]["candidate_detection_delta"]}, indent=2))


if __name__ == "__main__":
    main()
