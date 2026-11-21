#!/usr/bin/env python3
"""Replay unchanged PR86 predictions on the explicitly post-return r15 review."""
import argparse
import json
from pathlib import Path
import sys

from measure_pr86 import MODEL_SHA, SCORER_SHA, require, sha, success_rows, select_preserved_checkpoint

REVIEW_PATH = "data/reviews/fronts-2026-11-21-r15/manifest.json"
REVIEW_SHA = "6012d09767851151312537c2046022b972bf6519f36d110675d3e82f74694e93"
FIXED_COMPARISON_SHA = "b5c546d8f1da26d09e3b3016e7c2f44e50f254cf44fd76cc16c73e96e16c057d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, audit = args.root.resolve(), args.audit.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import _read_replay, parse_pose_labels, preflight_dataset
    from pose_scoring import aggregate_scores, score_image
    from diagnose_pose_quality import aggregate_quality, quality_for_image

    for name, expected in SCORER_SHA.items():
        require(sha(root / "scripts" / name) == expected, "Scorer changed: " + name)
    preflight = preflight_dataset(root, require_weights=False)
    records = {r["id"]: r for r in preflight["records"]}
    fixed_path = audit / "fixed-comparison.json"
    require(sha(fixed_path) == FIXED_COMPARISON_SHA, "Pre-return comparison changed")
    fixed = json.loads(fixed_path.read_text())
    require(fixed["thresholds"] == {"box_confidence": .13, "endpoint_confidence": .5,
            "nms_iou": .7, "matching_iou": .5, "imgsz": 1280, "radii_px": [5, 10]},
            "Fixed scoring settings changed")
    review_path = root / REVIEW_PATH
    require(sha(review_path) == REVIEW_SHA, "Post-return r15 review changed")
    review = json.loads(review_path.read_text())
    status_path = root / review["review_status"]["path"]
    require(sha(status_path) == review["review_status"]["sha256"], "Review status changed")
    status = json.loads(status_path.read_text())
    review_files = {r["image_id"]: r for r in review["files"] if r["split"] == "val"}
    require(set(review_files) == set(records), "Development photo inventory differs")
    gold = {}
    for image_id, record in records.items():
        entry = review_files[image_id]
        label = review_path.parent / entry["label_path"]
        require(sha(label) == entry["label_sha256"], "Review label changed: " + image_id)
        require(entry["source_image_sha256"] == record["image_sha256"], "Original image changed: " + image_id)
        gold[image_id] = parse_pose_labels(label.read_text(), record["width"], record["height"], str(label))
    counts = review["expected_counts"]["val"]
    actual_counts = {"images": len(gold), "objects": sum(len(g) for g in gold.values()),
        "visible_endpoints": sum(point[2] == 2 for objects in gold.values() for obj in objects for point in obj["keypoints"]),
        "missing_endpoints": sum(point[2] != 2 for objects in gold.values() for obj in objects for point in obj["keypoints"]),
        "empty_images": sum(not objects for objects in gold.values())}
    require(actual_counts == counts, "Reference counts differ")
    report = {
        "schema_version": 1,
        "kind": "pr86-predictions-post-return-r15-diagnostic",
        "status": "measured_exploratory_only",
        "reference": {"path": REVIEW_PATH, "sha256": REVIEW_SHA, "id": review["review_id"],
                      "counts": counts, "timing": "amended_after_PR86_return",
                      "remaining_uncertain_cases": [c for c in status["cases"] if c["split"] == "val" and c["status"] != "resolved"]},
        "pre_return_comparison_sha256": sha(fixed_path),
        "measurement_script_sha256": sha(__file__),
        "scorers_sha256": SCORER_SHA,
        "thresholds": fixed["thresholds"],
        "new_inference_or_training": False,
        "models": {},
        "limits": ["This reference was amended after the PR86 return; it cannot replace the preregistered primary score.",
                   "The same cached predictions are reused with no threshold, NMS or checkpoint search.",
                   "These repeatedly used development scenes do not establish unseen-session or robot performance."],
    }
    parent_path = root / review["parent"]["path"]
    require(sha(parent_path) == review["parent"]["sha256"], "Parent r14 review changed")
    parent = {r["image_id"]: r for r in json.loads(parent_path.read_text())["files"]}
    report["changed_development_labels"] = [image_id for image_id, r in sorted(review_files.items())
                                             if r["label_sha256"] != parent[image_id]["label_sha256"]]
    for role, expected_model in MODEL_SHA.items():
        pred_path = audit / "inference" / ("eval-" + role) / "predictions.jsonl"
        require(sha(pred_path) == fixed["models"][role]["predictions_sha256"], "Predictions changed: " + role)
        selected = select_preserved_checkpoint(preflight, audit / "checkpoint-receipts" / role / "checkpoint-manifest.json")
        require(selected["model_sha256"] == expected_model, "Checkpoint changed: " + role)
        header, rows, thresholds = _read_replay(pred_path, selected, [.13])
        require(thresholds == [.13] and len(rows) == 20, "Incomplete cached predictions")
        predicted = {r["image_id"]: r["predictions"] for r in rows}
        scores, qualities, scenes = [], [], []
        matches, successes = set(), {5: set(), 10: set()}
        groups = {str(k): {"gold": 0, "matched": 0, "misses": 0} for k in range(3)}
        for image_id, record in sorted(records.items()):
            target, output = gold[image_id], predicted[image_id]
            width, height = record["width"], record["height"]
            score = score_image(target, output, width, height, .5, .5)
            scores.append(score)
            qualities.append(quality_for_image(target, output, width, height, [5, 10]))
            matched = {p["gold_index"] for p in score["matches"]}
            matches.update(f"{image_id}:{i+1}" for i in matched)
            for i, obj in enumerate(target):
                group = groups[str(sum(p[2] == 2 for p in obj["keypoints"]))]
                group["gold"] += 1
                group["matched"] += int(i in matched)
                group["misses"] += int(i not in matched)
            for radius in successes:
                successes[radius].update(f"{image_id}:{i}" for i in success_rows(target, output, score, width, height, radius))
            scenes.append({"image_id": image_id, "gold": len(target), "predictions": len(output),
                           "matched": score["tp"], "misses": score["fn"], "extras": score["fp"],
                           "duplicates": score["duplicate_fp"], "missed_rows": [i+1 for i in score["unmatched_gold_indices"]]})
        aggregate = aggregate_scores(scores)
        aggregate.pop("endpoint_errors_px", None)
        prior = fixed["references"]["current_r14_exploratory"]["models"][role]
        report["models"][role] = {
            "checkpoint_sha256": expected_model, "predictions_sha256": sha(pred_path),
            "detection": aggregate, "corner_quality": aggregate_quality(qualities),
            "locatable_corner_groups": groups, "scenes": scenes,
            "matched_identities": sorted(matches),
            "success_identities": {str(k): sorted(v) for k, v in successes.items()},
            "empty_scene_outputs": sum(s["predictions"] for s in scenes if s["gold"] == 0),
            "miss_free_nonempty_scenes": sum(s["misses"] == 0 for s in scenes if s["gold"] > 0),
            "difference_from_pre_return_r14": {
                "detection": {k: aggregate[k] - prior["detection"][k] for k in ("tp", "fn", "fp", "duplicate_fp", "other_fp")},
                "gained_matches": sorted(matches - set(prior["matched_identities"])),
                "lost_matches": sorted(set(prior["matched_identities"]) - matches),
                "corner_success_changes": {str(k): {"gained": sorted(v - set(prior["success_identities"][str(k)])),
                                                    "lost": sorted(set(prior["success_identities"][str(k)]) - v)} for k, v in successes.items()},
            },
        }
    require(not args.output.exists(), "Choose a fresh output")
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"reference": report["reference"]["id"], "changed_development_labels": report["changed_development_labels"],
                      "models": {k: {"counts": {q: v["detection"][q] for q in ("tp", "fn", "fp", "duplicate_fp")},
                                     "pairs": {str(r["radius_px"]): r["both_visible_objects_within_radius"] for r in v["corner_quality"]["radii"]},
                                     "changes": v["difference_from_pre_return_r14"]} for k, v in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
