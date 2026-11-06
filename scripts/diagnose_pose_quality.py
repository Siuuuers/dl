#!/usr/bin/env python3
"""Describe saved baomai predictions at illustrative pixel radii; never infer or approve quality."""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_pose import (
    _json_write, _read_replay, _source_identity, create_output_directory,
    file_sha256, preflight_dataset,
)
from pose_scoring import score_image
from pose_checkpoint import select_checkpoint


MATCHING_IOU = 0.50
ENDPOINT_CONFIDENCE = 0.50
COUNT_FIELDS = (
    "gold_count", "prediction_count", "tp", "fn", "fp", "visible_endpoints",
    "returned_visible_endpoints", "both_visible_objects", "both_visible_objects_with_both_returned",
    "unlocatable_reference_endpoints", "unlocatable_on_matched_objects", "returned_on_unlocatable_reference",
)
RADIUS_COUNTS = ("visible_endpoints_within_radius", "both_visible_objects_within_radius")


def _validated_radii(radii):
    if (not isinstance(radii, (list, tuple)) or not radii
            or any(isinstance(radius, bool) or not isinstance(radius, (int, float))
                   or not math.isfinite(radius) or radius < 0 for radius in radii)
            or len(set(radii)) != len(radii)):
        raise ValueError("radii must be unique finite nonnegative numbers")
    return list(radii)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _returned(point, width, height):
    x, y, confidence = point
    return (confidence >= ENDPOINT_CONFIDENCE and x is not None and y is not None
            and math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height)


def quality_for_image(gold, predictions, width, height, radii):
    """Keep all eligible reference objects/points in denominators, including misses."""
    radii = _validated_radii(radii)
    scored = score_image(gold, predictions, width, height, MATCHING_IOU, ENDPOINT_CONFIDENCE)
    errors = {}
    unlocatable_matched = returned_unlocatable = 0
    for pair in scored["matches"]:
        gold_index = pair["gold_index"]
        predicted = predictions[pair["prediction_index"]]["keypoints"]
        for side, ((gx, gy, visibility), point) in enumerate(zip(gold[gold_index]["keypoints"], predicted)):
            returned = _returned(point, width, height)
            if visibility == 0:
                unlocatable_matched += 1
                returned_unlocatable += returned
            elif returned:
                errors[(gold_index, side)] = math.hypot(point[0] - gx, point[1] - gy)
    both_visible = [index for index, item in enumerate(gold) if all(point[2] == 2 for point in item["keypoints"])]
    both_returned = [index for index in both_visible if (index, 0) in errors and (index, 1) in errors]
    return {
        **{name: scored[name] for name in ("gold_count", "prediction_count", "tp", "fn", "fp", "visible_endpoints")},
        "returned_visible_endpoints": len(errors),
        "both_visible_objects": len(both_visible),
        "both_visible_objects_with_both_returned": len(both_returned),
        "unlocatable_reference_endpoints": 2 * len(gold) - scored["visible_endpoints"],
        "unlocatable_on_matched_objects": unlocatable_matched,
        "returned_on_unlocatable_reference": returned_unlocatable,
        "radii": [{"radius_px": radius,
                   "visible_endpoints_within_radius": sum(error <= radius for error in errors.values()),
                   "both_visible_objects_within_radius": sum(
                       errors[(index, 0)] <= radius and errors[(index, 1)] <= radius for index in both_returned)}
                  for radius in radii],
    }


def aggregate_quality(scores):
    """Pool counts before dividing; empty scenes still contribute false detections."""
    radii = _validated_radii([row["radius_px"] for row in scores[0]["radii"]]) if scores else []
    if any([row["radius_px"] for row in score["radii"]] != radii for score in scores):
        raise ValueError("all images must use the same radii in the same order")
    result = {name: sum(score[name] for score in scores) for name in COUNT_FIELDS}
    result.update(
        image_count=len(scores),
        object_precision=_rate(result["tp"], result["prediction_count"]),
        object_recall=_rate(result["tp"], result["gold_count"]),
        visible_endpoint_coverage=_rate(result["returned_visible_endpoints"], result["visible_endpoints"]),
        both_visible_objects_returned_fraction=_rate(result["both_visible_objects_with_both_returned"], result["both_visible_objects"]),
        returned_on_unlocatable_reference_fraction=_rate(result["returned_on_unlocatable_reference"], result["unlocatable_on_matched_objects"]),
    )
    result["radii"] = []
    for index, radius in enumerate(radii):
        counts = {name: sum(score["radii"][index][name] for score in scores) for name in RADIUS_COUNTS}
        result["radii"].append({"radius_px": radius, **counts,
            "visible_endpoint_success_rate": _rate(counts["visible_endpoints_within_radius"], result["visible_endpoints"]),
            "both_visible_object_success_rate": _rate(counts["both_visible_objects_within_radius"], result["both_visible_objects"])})
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--predictions", type=Path, required=True, help="existing verified predictions.jsonl")
    parser.add_argument("--checkpoint-manifest", type=Path,
                        help="matching completed paired-training bundle; omitted requires baseline predictions")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are refused")
    parser.add_argument("--radii", type=float, nargs="+", default=[2, 5, 10], help="illustrative pixel radii, not acceptance criteria")
    parser.add_argument("--conf", type=float, nargs="+", help="subset of box thresholds already present in the replay")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = None
    try:
        radii = _validated_radii(args.radii)
        if args.conf is not None and (len(set(args.conf)) != len(args.conf)
                or any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.conf)):
            raise ValueError("box confidences must be unique finite numbers in [0, 1]")
        preflight = preflight_dataset(args.root, require_weights=False)
        if args.checkpoint_manifest:
            preflight = select_checkpoint(preflight, args.checkpoint_manifest, require_weights=False)
        header, items, thresholds = _read_replay(args.predictions, preflight, args.conf)
        records = {record["id"]: record for record in preflight["records"]}
        per_image = []
        for item in items:
            record = records[item["image_id"]]
            quality = quality_for_image(record["gold"], item["predictions"], record["width"], record["height"], radii)
            per_image.append({"image_id": item["image_id"], "box_confidence": item["threshold"], **quality})
        aggregates = [{"box_confidence": threshold, **aggregate_quality(
            [row for row in per_image if row["box_confidence"] == threshold])} for threshold in thresholds]
        script = Path(__file__).resolve()
        source = _source_identity(script.parents[1])
        source["files_sha256"]["scripts/diagnose_pose_quality.py"] = file_sha256(script)
        provenance = header.get("replay", {})
        summary = {
            "schema_version": 1, "diagnostic_only": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "command": [sys.executable, str(script), *(sys.argv[1:] if argv is None else argv)],
            "predictions": {"path": str(args.predictions.resolve()), "sha256": file_sha256(args.predictions),
                "original_inference_source": provenance.get("inference_source", header["source"]),
                "original_inference_created_at_utc": provenance.get("inference_created_at_utc", header["created_at_utc"]),
                "inference": header["inference"], "runtime": header["runtime"]},
            "reference": {**{name: preflight[name] for name in ("manifest_sha256", "model_card_sha256")},
                "model_sha256": preflight.get("reference_model_sha256", preflight["model_sha256"]),
                "weights_loaded_or_verified_in_this_diagnostic": False,
                "images": [{name: record[name] for name in ("id", "image", "label", "image_sha256", "label_sha256", "width", "height")}
                           for record in preflight["records"]]},
            "procedure": {"matching_iou": MATCHING_IOU, "endpoint_confidence": ENDPOINT_CONFIDENCE,
                "radii_px": radii, "radii_are_illustrative_not_acceptance_criteria": True,
                "endpoint_success": "Corresponding returned endpoint has Euclidean original-pixel error <= radius; denominator includes all gold-visible endpoints and misses.",
                "both_point_success": "Both corresponding endpoints returned and each error <= radius; denominator is all objects with both gold points visible, including misses.",
                "availability": "returned_on_unlocatable_reference counts eligible coordinates for gold v0 endpoints only on matched objects; denominator is unlocatable_on_matched_objects. No hidden coordinate error is computed.",
                "returned": "Endpoint confidence >= 0.5, finite coordinates, 0 <= x < width and 0 <= y < height on a matched object.",
                "zero_denominator": "null; empty scenes still contribute detection false positives",
                "new_inference_or_training": False},
            "limits": ["Pixel radii are illustrative; no acceptance pass/fail decision is made.",
                "Development validation already used for checkpoint selection; points and related scenes are not independent trials.",
                "Reference-v0 output availability is a disagreement diagnostic, not proof of a physical false positive or its coordinate error."],
            "metrics": aggregates,
        }
        if "checkpoint_identity" in preflight:
            summary["evaluated_checkpoint"] = preflight["checkpoint_identity"]
        output = create_output_directory(args.output)
        _json_write(output / "quality-summary.json", summary)
        _json_write(output / "quality-per-image.json", per_image)
        rate_fields = ("object_precision", "object_recall", "visible_endpoint_coverage",
                       "both_visible_objects_returned_fraction", "returned_on_unlocatable_reference_fraction")
        with (output / "quality-per-image.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_id", "box_confidence", "radius_px", *COUNT_FIELDS,
                *RADIUS_COUNTS, *rate_fields, "visible_endpoint_success_rate", "both_visible_object_success_rate"), extrasaction="ignore")
            writer.writeheader()
            for row in per_image:
                rates = aggregate_quality([row])
                for radius in rates.pop("radii"):
                    writer.writerow({"image_id": row["image_id"], "box_confidence": row["box_confidence"], **rates, **radius})
        _json_write(output / "run_status.json", {"status": "complete", "exit_code": 0,
            "scored_image_threshold_pairs": len(per_image), "diagnostic_only": True})
        print(json.dumps({"output": str(output.resolve()), "diagnostic_only": True, "metrics": aggregates}, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        if output is not None:
            try:
                _json_write(output / "run_status.json", {"status": "failed", "exit_code": 1,
                    "diagnostic_only": True, "error_type": type(error).__name__, "error": str(error)})
            except OSError as status_error:
                print(f"Could not write failure status: {status_error}", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
