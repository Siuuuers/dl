#!/usr/bin/env python3
"""Rescore the four fixed PR101 prediction sets on the separate R26 review."""
import argparse
import hashlib
import json
from pathlib import Path

from diagnose_pose_quality import aggregate_quality, quality_for_image
from evaluate_pose import parse_pose_labels
from front_review_r26 import REVIEW_PATH, verify_candidate
from pose_scoring import aggregate_scores, score_image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root):
    root = root.resolve()
    verify_candidate(root)
    evidence = root / "results/evaluation/pr101-r25-20261123"
    comparison_path = evidence / "fixed-comparison.json"
    comparison = json.loads(comparison_path.read_text())
    review_path = root / REVIEW_PATH
    review = json.loads(review_path.read_text())
    files = {item["image_id"]: item for item in review["files"] if item["split"] == "val"}
    report = {"schema_version": 1, "kind": "r26_exploratory_rescore", "training_run": False,
              "new_inference": False, "model_selection": "retain_pr35_under_frozen_protocol",
              "reference": {"path": REVIEW_PATH, "sha256": sha(review_path),
                            "counts": review["expected_counts"]["val"]},
              "prediction_evidence": {"path": comparison_path.relative_to(root).as_posix(),
                                      "sha256": sha(comparison_path)},
              "thresholds": comparison["thresholds"], "models": {},
              "interpretation": "Same saved predictions; R25-to-R26 score differences here come from reference amendments, not model learning. Row008:36 is deleted; all other current row identities remain stable. Frozen666/663 results are unchanged."}
    for role, metadata in comparison["models"].items():
        path = evidence / "inference" / f"eval-{role}" / "predictions.jsonl"
        if sha(path) != metadata["predictions_sha256"]:
            raise ValueError(f"Prediction records changed: {role}")
        header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
        if header["model_sha256"] != metadata["metadata"]["model_sha256"]:
            raise ValueError(f"Checkpoint identity changed: {role}")
        if len(rows) != 20 or {r["image_id"] for r in rows} != set(files):
            raise ValueError("Exact development inventory required")
        scores, quality, matches = [], [], set()
        for row in rows:
            image_id = row["image_id"]
            item = files[image_id]
            label = review_path.parent / item["label_path"]
            if row["threshold"] != .13 or row["image_sha256"] != item["source_image_sha256"] or sha(label) != item["label_sha256"]:
                raise ValueError(f"Reference or prediction input drift: {image_id}")
            gold = parse_pose_labels(label.read_text(), row["width"], row["height"])
            score = score_image(gold, row["predictions"], row["width"], row["height"], .5, .5)
            scores.append(score)
            quality.append(quality_for_image(gold, row["predictions"], row["width"], row["height"], [5, 10]))
            matches.update(f"{image_id}:{pair['gold_index'] + 1}" for pair in score["matches"])
        old = comparison["references"]["current_r25_exploratory"]["models"][role]
        prior = set(old["matched_identities"])
        detection = aggregate_scores(scores)
        report["models"][role] = {
            "checkpoint_sha256": header["model_sha256"], "predictions_sha256": sha(path),
            "detection": detection, "corner_quality": aggregate_quality(quality),
            "matched_identities": sorted(matches),
            "reference_only_change_from_r25": {
                "matched_delta": detection["tp"] - old["detection"]["tp"],
                "miss_delta": detection["fn"] - old["detection"]["fn"],
                "unmatched_output_delta": detection["fp"] - old["detection"]["fp"],
                "newly_matched_identities": sorted(matches - prior),
                "no_longer_matched_identities": sorted(prior - matches),
                "deleted_target_identity": "new_000008:36"}}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({role: {"misses": item["detection"]["fn"], "unmatched_outputs": item["detection"]["fp"],
                            "pairs": [x["both_visible_objects_within_radius"] for x in item["corner_quality"]["radii"]]}
                      for role, item in result["models"].items()}, indent=2))
