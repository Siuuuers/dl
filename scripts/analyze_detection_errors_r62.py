#!/usr/bin/env python3
"""Partition cached detection errors without changing predictions or references."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from audit_reflection_return import require
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image
from run_original_comparison_r59 import assemble, measure
from run_reference_matrix_r60 import verified_zip


ARCHIVE_PATH = "docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip"
ARCHIVE_SHA256 = "ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601"
ARCHIVE_BYTES = 972658
REFERENCES = ("supplied664", "frozen666", "r50")
METHODS = ("original", "proposed")
BOX_CONFIDENCE = 0.25
MATCHING_IOU = 0.50
OVERLAP_CUTOFF = 0.10
LIMITATIONS = [
    "Overlap strata are geometric review cues, not physical background, label-error or object-identity verdicts.",
    "The native cache was already filtered at confidence 0.25, ordinary NMS 0.70 and max_det 300, then clipped. It cannot expose discarded native proposals or lower-score responses.",
    "No eligible cached candidate does not establish that the network produced no candidate. New pre-filter decoding would be needed to separate confidence, native NMS and proposal failures.",
    "Pipeline stages identify where eligible records disappear within the saved cache only; they do not establish a physical cause.",
    "Three annotation references describe the same repeatedly inspected 20 development photographs and are not independent test sets.",
    "Corner availability is reference-defined. A missing corner does not remove its front from detection evaluation.",
    "Reflected-view complementarity is an annotation-based diagnostic, not an implementable oracle or evidence that merging views improves the result.",
    "This replay changes no labels, predictions, model, matching rule, confidence cutoff or deployment setting.",
]


def nearest_same_class(query, candidates):
    eligible = [index for index, candidate in enumerate(candidates)
                if candidate["class_id"] == query["class_id"]]
    if not eligible:
        return None, 0.0
    index = max(eligible, key=lambda i: (_box_iou(query["bbox_xyxy"], candidates[i]["bbox_xyxy"]), -i))
    overlap = _box_iou(query["bbox_xyxy"], candidates[index]["bbox_xyxy"])
    return (index if overlap > 0 else None), overlap


def available_corners(target):
    return sum(point[2] == 2 for point in target["keypoints"])


def overlap_stratum(overlap, duplicate):
    if duplicate:
        return "duplicate"
    require(overlap < MATCHING_IOU, "An extra overlaps an unclaimed eligible reference")
    return "near_reference_overlap" if overlap >= OVERLAP_CUTOFF else "low_reference_overlap"


def miss_stage(target, native, retained):
    selected = [prediction for prediction in native if prediction["confidence"] > BOX_CONFIDENCE]
    stages = {}
    for name, values in (("native_cached", native), ("above_box_cutoff", selected), ("retained", retained)):
        index, overlap = nearest_same_class(target, values)
        stages[name] = {"best_iou": overlap, "prediction_index": index,
                        "prediction_position": index + 1 if index is not None else None,
                        "prediction": values[index] if index is not None else None}
    if stages["native_cached"]["best_iou"] < MATCHING_IOU:
        reason = "no_eligible_cached_candidate"
    elif stages["above_box_cutoff"]["best_iou"] < MATCHING_IOU:
        reason = "explicit_confidence_filter"
    elif stages["retained"]["best_iou"] < MATCHING_IOU:
        reason = "extra_postclip_suppression"
    else:
        reason = "retained_assignment_contention"
    return {"cache_stage": reason, "stages": stages}


def inspect_image(reference, method, image, gold, predictions, native):
    score = score_image(gold, predictions, 1280, 720, MATCHING_IOU, 0.5)
    matched = {pair["gold_index"]: pair["prediction_index"] for pair in score["matches"]}
    duplicates = set(score["duplicate_prediction_indices"])
    counts = {key: score[key] for key in
              ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp")}
    counts["reference"] = reference
    counts["method"] = method
    counts["image"] = image
    counts["reference_empty"] = not gold
    counts["by_available_corners"] = {
        str(n): {"fronts": sum(available_corners(g) == n for g in gold),
                 "missed": sum(available_corners(gold[i]) == n for i in score["unmatched_gold_indices"])}
        for n in range(3)}
    extra_counts = Counter()
    stage_counts = Counter()
    cases = []
    for pi in score["unmatched_prediction_indices"]:
        prediction = predictions[pi]
        gi, overlap = nearest_same_class(prediction, gold)
        stratum = overlap_stratum(overlap, pi in duplicates)
        extra_counts[stratum] += 1
        cases.append({
            "case_id": f"{reference}/{method}/{image}/P{pi + 1:03d}",
            "kind": "extra_prediction", "reference": reference, "method": method, "image": image,
            "prediction_index": pi, "prediction_position": pi + 1,
            "primary_prediction_id": prediction.get("primary_prediction_id"),
            "prediction": prediction, "stratum": stratum, "max_same_class_iou": overlap,
            "nearest_reference_index": gi, "nearest_reference_row": gi + 1 if gi is not None else None,
            "nearest_reference": gold[gi] if gi is not None else None,
            "nearest_reference_matched": gi in matched if gi is not None else None,
            "nearest_reference_matched_prediction_index": matched.get(gi),
        })
    for gi in score["unmatched_gold_indices"]:
        target = gold[gi]
        trace = miss_stage(target, native, predictions)
        stage_counts[trace["cache_stage"]] += 1
        cases.append({
            "case_id": f"{reference}/{method}/{image}/R{gi + 1:03d}",
            "kind": "missed_reference", "reference": reference, "method": method, "image": image,
            "reference_index": gi, "reference_row": gi + 1, "target": target,
            "available_corners": available_corners(target), **trace,
        })
    counts["extra_strata"] = {name: extra_counts[name] for name in
                             ("duplicate", "near_reference_overlap", "low_reference_overlap")}
    counts["miss_cache_stages"] = dict(sorted(stage_counts.items()))
    require(sum(counts["extra_strata"].values()) == score["fp"], "Extra partition does not close")
    require(sum(stage_counts.values()) == score["fn"], "Miss partition does not close")
    return counts, cases, score


def matrix_row(reference, method, measured):
    row = measured["references"][reference][method]
    detection, pairs = row["detection"], row["pairs"]
    return {"reference": reference, "method": method,
            "fronts": detection["gold_count"], "outputs": detection["prediction_count"],
            "matched": detection["tp"], "missed": detection["fn"], "extras": detection["fp"],
            "duplicates": detection["duplicate_fp"], "eligible_pairs": pairs["eligible"],
            "within5": pairs["within5"], "within10": pairs["within10"],
            "complete_pair_missed_boxes": pairs["missed_box"]}


def analyze(root):
    members = verified_zip(root / ARCHIVE_PATH, ARCHIVE_SHA256, ARCHIVE_BYTES)
    def read(name):
        payload = members[name]
        return json.loads(gzip.decompress(payload) if name.endswith(".gz") else payload)
    saved_summary = read("summary.json")
    native = read("native-predictions.json.gz")
    labels = read("reference-labels.json.gz")
    references = {name: {image: parse_pose_labels(text, 1280, 720, name + "/" + image)
                         for image, text in labels[name].items()} for name in REFERENCES}
    arms, suppression, associations = assemble(native)
    for name, actual in (("predictions", arms), ("suppression", suppression), ("associations", associations)):
        require(actual == read(name + ".json.gz"), "Reconstructed " + name + " differs")
    measured, records = measure(references, native)
    for name, actual in records.items():
        require(actual == read(name + ".json.gz"), "Replayed " + name + " differs")
    rows, cases = [], []
    for reference in REFERENCES:
        for method in METHODS:
            baseline = matrix_row(reference, method, measured)
            saved = next(row for row in saved_summary["matrix"]
                         if (row["reference"], row["method"]) == (reference, method))
            require(all(saved[key] == value for key, value in baseline.items()), "Baseline row differs")
            per_image = []
            for image, gold in sorted(references[reference].items()):
                native_method = "trained" if method == "proposed" else "original"
                counts, image_cases, score = inspect_image(reference, method, image, gold,
                                                          arms[method][image], native[native_method][image])
                require(score == records["per-image"][reference][method][image], "Image replay differs")
                per_image.append(counts)
                cases.extend(image_cases)
            strata = {name: sum(row["extra_strata"][name] for row in per_image)
                      for name in ("duplicate", "near_reference_overlap", "low_reference_overlap")}
            availability = {str(n): {key: sum(row["by_available_corners"][str(n)][key] for row in per_image)
                                     for key in ("fronts", "missed")} for n in range(3)}
            stages = Counter()
            for row in per_image:
                stages.update(row["miss_cache_stages"])
            require(sum(strata.values()) == baseline["extras"] and strata["duplicate"] == baseline["duplicates"],
                    "Aggregate extra partition differs")
            require(sum(v["fronts"] for v in availability.values()) == baseline["fronts"] and
                    sum(v["missed"] for v in availability.values()) == baseline["missed"], "Availability partition differs")
            rows.append({**baseline, "extra_strata": strata, "by_available_corners": availability,
                         "miss_cache_stages": dict(sorted(stages.items())), "per_image": per_image})
    primary = next(row for row in rows if (row["reference"], row["method"]) == ("r50", "proposed"))
    top_images = sorted(primary["per_image"], key=lambda row: (-row["fp"], row["image"]))[:4]
    complementarity = {}
    for reference in REFERENCES:
        original_missed, reflected_missed = set(), set()
        for method, misses in (("proposed", original_missed), ("reflected", reflected_missed)):
            for image, score in records["per-image"][reference][method].items():
                misses.update(image + f"/R{i + 1:03d}" for i in score["unmatched_gold_indices"])
        complementarity[reference] = {
            "proposed_misses_matched_by_reflected_view": sorted(original_missed - reflected_missed),
            "proposed_matches_missed_by_reflected_view": sorted(reflected_missed - original_missed),
            "missed_by_both": sorted(original_missed & reflected_missed),
            "reflected_detection": {key: measured["references"][reference]["reflected"]["detection"][key]
                                    for key in ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp")}}
    summary = {
        "schema": "cassette-detection-error-audit-r62-v1", "status": "completed",
        "fresh_inference": False, "training": False, "labels_modified": False, "development_only": True,
        "source_archive": {"path": ARCHIVE_PATH, "sha256": ARCHIVE_SHA256, "bytes": ARCHIVE_BYTES},
        "analysis_sources_sha256": {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                                    for relative in ("scripts/analyze_detection_errors_r62.py", "scripts/pose_scoring.py",
                                                     "scripts/evaluate_pose.py", "scripts/run_original_comparison_r59.py",
                                                     "scripts/run_corner_average_r58.py", "scripts/predict_pose_folder.py",
                                                     "scripts/run_reference_matrix_r60.py", "scripts/audit_reflection_return.py")},
        "source_protocol": saved_summary["contract"]["protocol"],
        "source_prediction_runtime": saved_summary["contract"]["prediction_runtime"],
        "source_models": {key: saved_summary["contract"][key] for key in
                          ("original_model_sha256", "adapted_model_sha256")},
        "source_photographs": saved_summary["contract"]["photographs"],
        "source_references": saved_summary["contract"]["references"],
        "verification": {"archive_members_and_hashes": "passed", "native_output_reconstruction": "exact",
                         "saved_score_and_case_replay": "exact", "original_proposed_matrix_rows": len(rows),
                         "per_image_scores_replayed": sum(len(images) for methods in records["per-image"].values()
                                                          for images in methods.values())},
        "definitions": {"duplicate": "Existing scorer's extra prediction with same-class IoU >= 0.50 to an already matched reference.",
                        "near_reference_overlap": "Nonduplicate extra with maximum same-class reference IoU >= 0.10 and < 0.50.",
                        "low_reference_overlap": "Nonduplicate extra with maximum same-class reference IoU < 0.10, including no reference.",
                        "indices": "Index fields are zero-based. Row and position fields are one-based. primary_prediction_id belongs to that method/view and is not a cross-model identity."},
        "matrix": rows,
        "primary_headline": {key: primary[key] for key in
                             ("fronts", "outputs", "matched", "missed", "extras", "duplicates", "eligible_pairs", "within5", "within10", "extra_strata", "by_available_corners", "miss_cache_stages")},
        "primary_top_four_extra_images": [{key: row[key] for key in ("image", "fp", "fn", "duplicate_fp")}
                                          for row in top_images],
        "reflected_view_complementarity": complementarity,
        "review_case_count": len(cases), "limitations": LIMITATIONS,
    }
    return summary, {"schema": "cassette-detection-error-cases-r62-v1", "cases": cases}


def markdown(summary):
    lines = ["# Cached detection error audit", "", "All six baseline rows reproduce exactly; no inference or training was performed.", "",
             "| Reference | Method | Missed | Extras | Duplicate subset | Other IoU 0.10–<0.50 | Other IoU <0.10 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["matrix"]:
        strata = row["extra_strata"]
        lines.append(f"| {row['reference']} | {row['method']} | {row['missed']} | {row['extras']} | {strata['duplicate']} | {strata['near_reference_overlap']} | {strata['low_reference_overlap']} |")
    lines += ["", "## Current method under the reviewed reference", "",
              "| Locatable reference corners | Fronts | Missed |", "| --- | ---: | ---: |"]
    for count, row in summary["primary_headline"]["by_available_corners"].items():
        lines.append(f"| {count} | {row['fronts']} | {row['missed']} |")
    lines += ["", "## Interpretation limits", ""] + ["- " + limit for limit in summary["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, cases = analyze(args.root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {"summary.json": summary, "cases.json": cases}
    for name, value in outputs.items():
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    (args.output / "results.md").write_text(markdown(summary))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(args.output.iterdir()) if path.name in {"summary.json", "cases.json", "results.md"}}
    print(json.dumps({"status": summary["status"], "verification": summary["verification"],
                      "primary_headline": summary["primary_headline"], "output_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
