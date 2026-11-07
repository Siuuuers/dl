#!/usr/bin/env python3
"""Audit saved M0/M1/M2 predictions and summarize the predeclared comparison.

Example: python scripts/summarize_pose_comparison.py --evidence ../comparison-evidence \
    --root . --output ../comparison-evidence/comparison.json
Requires the original validation PNGs/labels and small checkpoint evidence bundles,
but neither model weights nor a machine-learning runtime. Existing output is refused.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

from diagnose_pose_quality import aggregate_quality, quality_for_image
from evaluate_pose import _read_replay, _source_identity, file_sha256, preflight_dataset
from package_gpu_results import check_pair
from pose_checkpoint import _contained_path, _read_json, _require, _same, select_checkpoint
from pose_scoring import _percentile, aggregate_scores, score_image


BASELINE_SHA256 = "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"
ARMS = {"m0": None, "m1": "reference", "m2": "candidate"}
THRESHOLDS = [0.15, 0.25, 0.40]
FIELDS = ("fn", "fp", "missing_endpoints", "conditional_median_error_px", "conditional_p95_error_px")
COMPARISONS = (("m1", "m0"), ("m2", "m0"), ("m2", "m1"))
IMAGE_FIELDS = ("id", "image", "label", "image_sha256", "label_sha256", "width", "height")


def indexed_rows(path, threshold_field):
    rows = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(rows, list) and len(rows) == 60, f"{path}: expected 60 rows")
    indexed = {(row["image_id"], row[threshold_field]): row for row in rows}
    _require(len(indexed) == 60, f"{path}: duplicate image/threshold")
    return indexed


def statistics(values):
    values = sorted(values)
    return {"count": len(values), "mean": math.fsum(values) / len(values) if values else None,
            "median": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}


def dominance(new, old):
    available = all(new[field] is not None and old[field] is not None for field in FIELDS)
    deltas = {field: new[field] - old[field] if available else None for field in FIELDS}
    return {"passes": available and all(value <= 0 for value in deltas.values())
            and any(value < 0 for value in deltas.values()), "statistics_available": available,
            "new_minus_old": deltas}


def audit_run(name, selected, evidence, input_file, code_root):
    header, items, thresholds = _read_replay(input_file(f"{name}/predictions.jsonl"), selected, None)
    lines = (evidence / name / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    _require(len([line for line in lines if line.strip()]) == 61 and len(items) == 60
             and _same(thresholds, THRESHOLDS), f"{name}: expected exactly three full 20-image sweeps")
    _require("replay" not in header, f"{name}: supply the original inference records")
    records = {record["id"]: record for record in selected["records"]}
    images = [{key: record[key] for key in IMAGE_FIELDS} for record in selected["records"]]
    _require(_same(header["images"], images), f"{name}: prediction header image inventory differs")
    source = header["source"]
    _require(re.fullmatch(r"[0-9a-f]{40}", source["revision"] or "") is not None
             and source["working_tree_status"] == "", f"{name}: inference did not record a clean revision")
    expected_sources = {"scripts/evaluate_pose.py", "scripts/pose_scoring.py", "scripts/pose_checkpoint.py"}
    _require(set(source["files_sha256"]) == expected_sources, f"{name}: evaluation source inventory differs")
    for path, digest in source["files_sha256"].items():
        _require(file_sha256(_contained_path(code_root, path)) == digest,
                 f"{name}: recorded evaluation code differs from the replay implementation: {path}")
    runtime = header["runtime"]
    _require(runtime["ultralytics"] == "8.4.182" and runtime["backend_end2end"] is False
             and runtime["backend_fp16"] is False and runtime["loaded_kpt_shape"] == [2, 3],
             f"{name}: inference runtime is incompatible")

    for directory in (name, f"{name}-quality"):
        status = _read_json(input_file(f"{directory}/run_status.json"))
        _require(status["status"] == "complete" and type(status["exit_code"]) is int
                 and status["exit_code"] == 0 and type(status["scored_image_threshold_pairs"]) is int
                 and status["scored_image_threshold_pairs"] == 60, f"{directory}: incomplete execution")

    summary = _read_json(input_file(f"{name}/summary.json"))
    _require(all(_same(summary.get(key), value) for key, value in header.items() if key != "record_type"),
             f"{name}: summary and prediction provenance differ")
    _require(_same(summary["matching_iou"], 0.50) and _same(summary["endpoint_confidence"], 0.50),
             f"{name}: scoring settings differ from the predeclared comparison")
    saved_scores = indexed_rows(input_file(f"{name}/per_image.json"), "threshold")
    quality = _read_json(input_file(f"{name}-quality/quality-summary.json"))
    saved_quality = indexed_rows(input_file(f"{name}-quality/quality-per-image.json"), "box_confidence")
    procedure = quality["procedure"]
    radii = procedure["radii_px"]
    _require(_same(procedure["matching_iou"], 0.50) and _same(procedure["endpoint_confidence"], 0.50)
             and radii == [2, 5, 10] and quality["diagnostic_only"] is True,
             f"{name}: quality diagnostic settings differ")
    _require(quality["predictions"]["sha256"] == file_sha256(evidence / name / "predictions.jsonl")
             and _same(quality.get("evaluated_checkpoint"), selected.get("checkpoint_identity")),
             f"{name}: quality diagnostic identifies different predictions/checkpoint")
    for key, expected in (("original_inference_source", source), ("inference", header["inference"]),
                          ("runtime", runtime), ("original_inference_created_at_utc", header["created_at_utc"])):
        _require(_same(quality["predictions"][key], expected), f"{name}: quality provenance differs: {key}")
    for key, expected in (("images", images), ("model_sha256", BASELINE_SHA256),
                          ("manifest_sha256", header["manifest_sha256"]),
                          ("model_card_sha256", header["model_card_sha256"])):
        _require(_same(quality["reference"][key], expected), f"{name}: quality reference differs: {key}")
    _require(quality["source"]["revision"] == source["revision"]
             and quality["source"]["working_tree_status"] == "", f"{name}: quality source differs")
    _require(set(quality["source"]["files_sha256"]) == expected_sources | {"scripts/diagnose_pose_quality.py"},
             f"{name}: quality source inventory differs")
    for path, digest in quality["source"]["files_sha256"].items():
        _require(file_sha256(_contained_path(code_root, path)) == digest,
                 f"{name}: quality implementation differs: {path}")

    scores, qualities, endpoints = {}, {}, {}
    for item in items:
        key = item["image_id"], item["threshold"]
        record = records[key[0]]
        arguments = record["gold"], item["predictions"], record["width"], record["height"]
        scored = score_image(*arguments, matching_iou=0.50, endpoint_conf=0.50)
        _require(all(_same(saved_scores[key].get(field), value) for field, value in scored.items()),
                 f"{name}: saved per-image scores differ from replay: {key}")
        diagnosed = quality_for_image(*arguments, radii)
        _require(_same(saved_quality[key], {"image_id": key[0], "box_confidence": key[1], **diagnosed}),
                 f"{name}: saved per-image quality differs from replay: {key}")
        scores[key], qualities[key] = scored, diagnosed
        if key[1] != 0.25:
            continue
        for match in scored["matches"]:
            gold_index = match["gold_index"]
            predicted = item["predictions"][match["prediction_index"]]["keypoints"]
            for side, (gold, point) in enumerate(zip(record["gold"][gold_index]["keypoints"], predicted)):
                x, y, confidence = point
                if (gold[2] == 2 and confidence >= 0.50 and x is not None and y is not None
                        and math.isfinite(x) and math.isfinite(y)
                        and 0 <= x < record["width"] and 0 <= y < record["height"]):
                    dx, dy = x - gold[0], y - gold[1]
                    endpoints[(key[0], gold_index, side)] = {"error_px": math.hypot(dx, dy), "dx_px": dx, "dy_px": dy}

    _require(len(summary["metrics"]) == len(quality["metrics"]) == 3, f"{name}: aggregate count differs")
    metrics, diagnostics = {}, {}
    for index, threshold in enumerate(THRESHOLDS):
        aggregate = aggregate_scores([scores[item["image_id"], threshold] for item in items if item["threshold"] == threshold])
        aggregate.pop("endpoint_errors_px")
        aggregate["threshold"] = threshold
        _require(all(_same(summary["metrics"][index].get(key), value) for key, value in aggregate.items()),
                 f"{name}: aggregate scores differ from replay at {threshold}")
        diagnosed = {"box_confidence": threshold, **aggregate_quality(
            [qualities[item["image_id"], threshold] for item in items if item["threshold"] == threshold])}
        _require(_same(quality["metrics"][index], diagnosed), f"{name}: aggregate quality differs at {threshold}")
        metrics[threshold], diagnostics[threshold] = aggregate, diagnosed
    pair_success = {(image_id, index) for image_id, record in records.items()
                    for index, obj in enumerate(record["gold"]) if all(point[2] == 2 for point in obj["keypoints"])
                    and all((image_id, index, side) in endpoints
                            and endpoints[image_id, index, side]["error_px"] <= 5 for side in (0, 1))}
    return {"header": header, "quality_source": quality["source"], "procedure": procedure,
            "metrics": metrics, "diagnostics": diagnostics, "scores": scores,
            "endpoints": endpoints, "pair_success": pair_success}


def summarize(root, evidence):
    evidence = evidence.resolve()
    code_root = Path(__file__).resolve().parents[1]
    input_hashes = {}

    def input_file(relative):
        path = _contained_path(evidence, relative)
        digest = file_sha256(path)
        _require(relative not in input_hashes or input_hashes[relative] == digest, f"input changed: {relative}")
        input_hashes[relative] = digest
        return path

    preflight = preflight_dataset(root, require_weights=False)
    _require(preflight["model_sha256"] == BASELINE_SHA256, "frozen reference is not the planned epoch173 initialization")
    selected, bundles = {"m0": preflight}, {}
    for name, arm in ARMS.items():
        if arm is None:
            continue
        prefix = f"checkpoints/{arm}/"
        manifest_path = input_file(prefix + "checkpoint-manifest.json")
        selected[name] = select_checkpoint(preflight, manifest_path, require_weights=False)
        _require(selected[name]["checkpoint_identity"]["arm"] == arm, f"{name}: wrong training arm")
        manifest = _read_json(manifest_path)
        receipt = _read_json(input_file(prefix + manifest["training_receipt"]["path"]))
        staging = _read_json(input_file(prefix + manifest["staging_manifest"]["path"]))
        for descriptor in (receipt["results"], receipt["log"]):
            input_file(prefix + descriptor["path"])
        bundles[arm] = {"receipt": receipt, "staging": staging}
    check_pair(bundles["reference"], bundles["candidate"])
    runs = {name: audit_run(name, selected[name], evidence, input_file, code_root) for name in ARMS}
    for name in ("m1", "m2"):
        for key in ("source", "runtime", "images", "inference", "box_confidences", "warmup",
                    "manifest_sha256", "model_card_sha256"):
            _require(_same(runs[name]["header"][key], runs["m0"]["header"][key]),
                     f"{name}: inference differs from M0: {key}")
        for key in ("quality_source", "procedure"):
            _require(_same(runs[name][key], runs["m0"][key]), f"{name}: quality diagnostic differs: {key}")

    comparisons = [{"box_confidence": threshold, "primary": threshold == 0.25,
                    "comparisons": {f"{new}_vs_{old}": dominance(runs[new]["metrics"][threshold], runs[old]["metrics"][threshold])
                                    for new, old in COMPARISONS}} for threshold in THRESHOLDS]
    primary = next(row["comparisons"] for row in comparisons if row["primary"])
    eligible = [name for name in ("m1", "m2") if primary[f"{name}_vs_m0"]["passes"]]
    common = sorted(set.intersection(*(set(run["endpoints"]) for run in runs.values())))
    common_metrics = {name: {field: statistics(run["endpoints"][key][field] for key in common)
                            for field in ("error_px", "dx_px", "dy_px")} for name, run in runs.items()}
    paired = {}
    for new, old in COMPARISONS:
        shared = sorted(set(runs[new]["endpoints"]) & set(runs[old]["endpoints"]))
        changes = [runs[new]["endpoints"][key]["error_px"] - runs[old]["endpoints"][key]["error_px"] for key in shared]
        old_pairs, new_pairs = runs[old]["pair_success"], runs[new]["pair_success"]
        paired[f"{new}_vs_{old}"] = {"common_returned_endpoints": len(shared), "error_change_px": statistics(changes),
            "worse_endpoints": sum(value > 0 for value in changes), "better_endpoints": sum(value < 0 for value in changes),
            "equal_endpoints": sum(value == 0 for value in changes),
            "returned_only_old": len(set(runs[old]["endpoints"]) - set(shared)),
            "returned_only_new": len(set(runs[new]["endpoints"]) - set(shared)),
            "both_corners_within_5px": {"old_success": len(old_pairs), "new_success": len(new_pairs),
                "success_both": len(old_pairs & new_pairs), "became_failure": len(old_pairs - new_pairs),
                "became_success": len(new_pairs - old_pairs)}}
    per_image = []
    for record in preflight["records"]:
        identifier = record["id"]
        keys = [key for key in common if key[0] == identifier]
        per_image.append({"image_id": identifier, "common_returned_endpoints": len(keys),
            "median_error_px": {name: statistics(run["endpoints"][key]["error_px"] for key in keys)["median"] for name, run in runs.items()},
            "mean_signed_y_error_px": {name: statistics(run["endpoints"][key]["dy_px"] for key in keys)["mean"] for name, run in runs.items()},
            "both_corners_within_5px": {name: sum(key[0] == identifier for key in run["pair_success"]) for name, run in runs.items()}})

    source = _source_identity(code_root)
    for name in ("summarize_pose_comparison.py", "diagnose_pose_quality.py", "package_gpu_results.py"):
        source["files_sha256"][f"scripts/{name}"] = file_sha256(code_root / "scripts" / name)
    for path, digest in input_hashes.items():
        _require(file_sha256(evidence / path) == digest, f"input changed during audit: {path}")
    return {"schema_version": 1, "kind": "baomai-three-model-comparison", "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "input_files_sha256": dict(sorted(input_hashes.items())),
        "verification_scope": "Saved-prediction replay, complete-run and paired-checkpoint evidence consistency; no model weights loaded or verified here.",
        "reference": {key: preflight[key] for key in ("model_sha256", "manifest_sha256", "model_card_sha256", "reference_validation_images")},
        "inference": {key: runs["m0"]["header"][key] for key in ("source", "runtime", "inference", "images", "warmup")},
        "models": {name: {"model_sha256": run["header"]["model_sha256"],
            "checkpoint_identity": selected[name].get("checkpoint_identity"),
            "metrics": list(run["metrics"].values()), "quality": list(run["diagnostics"].values())} for name, run in runs.items()},
        "decision": "retain-M0" if not eligible else "review-eligible-models",
        "baseline_replacement_eligible": eligible,
        "candidate_meets_rule_against_both_controls": primary["m2_vs_m0"]["passes"] and primary["m2_vs_m1"]["passes"],
        "comparison_rule": {"primary_box_confidence": 0.25, "lower_is_better_fields": list(FIELDS),
            "dominance": "All five unrounded metrics no higher, at least one strictly lower; unavailable statistics cannot pass.",
            "other_thresholds_are_diagnostic": True}, "comparisons": comparisons,
        "common_returned_endpoints": {"box_confidence": 0.25, "count": len(common), "metrics": common_metrics,
            "signed_error": "prediction minus reference; negative Y is upward in the image", "per_image": per_image},
        "paired_primary_diagnostics": paired,
        "empty_scene_false_detections": {record["id"]: {name: {str(threshold): run["scores"][record["id"], threshold]["fp"]
            for threshold in THRESHOLDS} for name, run in runs.items()} for record in preflight["records"] if not record["gold"]},
        "limits": ["Development validation already used for selection; this is not an independent test.",
                   "One training seed; related images and endpoints are not independent trials.",
                   "Pixel radii are illustrative, not acceptance tolerances. Signed residuals do not establish a cause.",
                   "Unlocatable-reference output counts are availability disagreement, not measurable hidden-coordinate errors."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="m0/m1/m2, their quality runs and checkpoints/{reference,candidate}")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="frozen validation repository")
    parser.add_argument("--output", type=Path, required=True, help="new comparison JSON; existing files are refused")
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        result = summarize(args.root, args.evidence)
        result["command"] = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                          "decision": result["decision"], "common_returned_endpoints": result["common_returned_endpoints"]["count"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        print(f"Comparison refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
