#!/usr/bin/env python3
"""Frozen PR35 donor and right-anchor diagnostic; never edits source labels."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import platform
import sys

import numpy as np

import context_corner_pilot as geometry


PilotError = geometry.PilotError
STRATEGIES = ("direct_pr35", "model_donor_homography", "right_anchored_homography")
SLOTS = ("left", "right")
RADII = (5, 10)
COMPARISONS = ((STRATEGIES[1], STRATEGIES[0]), (STRATEGIES[2], STRATEGIES[0]),
               (STRATEGIES[2], STRATEGIES[1]))


def read_pinned_json(root, relative_path, expected_sha256):
    """Verify the exact bytes before accepting a frozen dependency."""
    path = geometry._safe_path(Path(root), relative_path)
    data = path.read_bytes()
    digest = geometry.sha256(data)
    if digest != expected_sha256:
        raise PilotError(f"Input hash mismatch: {relative_path}")
    result = json.loads(data)
    return result, {"path": relative_path, "sha256": digest, "bytes": len(data)}


def _finite_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _point(xy, dimensions, *, confidence=None, source, reason=None):
    """Retain finite raw points even when rejected; never clip coordinates."""
    finite = (xy is not None and len(xy) == 2 and all(_finite_number(value) for value in xy))
    raw_xy = list(xy) if finite else None
    width, height = dimensions
    in_frame = (bool(0 <= raw_xy[0] < width and 0 <= raw_xy[1] < height)
                if finite else None)
    if reason is None and not finite:
        reason = "No finite endpoint coordinates"
    if reason is None and not in_frame:
        reason = "Endpoint outside half-open image bounds; no clipping"
    return {"xy": raw_xy, "confidence": confidence, "source": source,
            "available": reason is None, "in_frame": in_frame, "reason": reason}


def direct_endpoint(keypoint, dimensions, confidence_threshold=0.5):
    """Apply the fixed model endpoint gate, preserving its exact finite values."""
    if keypoint is None:
        return _point(None, dimensions, source="direct_pr35", reason="No matched model endpoint")
    if len(keypoint) != 3:
        raise PilotError("A model keypoint must have x, y, confidence")
    x, y, confidence = keypoint
    reason = None
    if not _finite_number(confidence):
        reason = "Non-finite model endpoint confidence"
        confidence = None
    elif confidence < confidence_threshold:
        reason = "Model endpoint confidence below fixed cutoff"
    return _point([x, y], dimensions, confidence=confidence, source="direct_pr35", reason=reason)


def _strategy(endpoints, *, structural_validity=None, failure_reason=None, target_points=()):
    available = sum(point["available"] for point in endpoints.values())
    pair = [endpoints[slot]["xy"] for slot in SLOTS]
    return {"status": "complete" if available == 2 else "partial" if available else "unavailable",
            "endpoints": endpoints, "raw_pair_xy": pair if all(point is not None for point in pair) else None,
            "structurally_valid_geometry": structural_validity, "geometry_failure_reason": failure_reason,
            "target_model_endpoints_used": list(target_points),
            "target_reference_endpoints_used": False, "target_bbox_used_for_geometry": False}


def direct_prediction(target_keypoints, dimensions, confidence_threshold=0.5):
    points = target_keypoints if target_keypoints is not None else [None, None]
    return _strategy({slot: direct_endpoint(points[index], dimensions, confidence_threshold)
                      for index, slot in enumerate(SLOTS)}, target_points=SLOTS)


def model_donor_homography(earlier_keypoints, later_keypoints, t, dimensions, *,
                           confidence_threshold=0.5, allow_extrapolation=False):
    """Only designated donor endpoints and t enter; no target or box argument."""
    donor_points = []
    failures = []
    for name, keypoints in (("earlier", earlier_keypoints), ("later", later_keypoints)):
        points = keypoints if keypoints is not None else [None, None]
        checked = [direct_endpoint(points[index], dimensions, confidence_threshold) for index in range(2)]
        donor_points.append(checked)
        failures.extend(f"{name}.{slot}: {point['reason']}" for slot, point in zip(SLOTS, checked)
                        if not point["available"])
    pair = None
    failure = "; ".join(failures) if failures else None
    if failure is None:
        try:
            pair = geometry.estimate_pair("homography",
                [point["xy"] for point in donor_points[0]],
                [point["xy"] for point in donor_points[1]], t,
                allow_extrapolation=allow_extrapolation).tolist()
        except (PilotError, np.linalg.LinAlgError) as error:
            failure = str(error)
    endpoints = {slot: _point(pair[index] if pair is not None else None, dimensions,
                              source="estimated_from_context", reason=failure)
                 for index, slot in enumerate(SLOTS)}
    result = _strategy(endpoints, structural_validity=pair is not None, failure_reason=failure)
    result["donor_endpoint_checks"] = {name: {slot: point for slot, point in zip(SLOTS, points)}
                                        for name, points in zip(("earlier", "later"), donor_points)}
    return result


def anchor_to_right(geometric_result, target_right_keypoint, dimensions, *, confidence_threshold=0.5):
    """R stays exactly direct; L = R + (geoL - geoR). No target-left input."""
    right = direct_endpoint(target_right_keypoint, dimensions, confidence_threshold)
    failure = None
    left_xy = None
    vector = None
    if not right["available"]:
        failure = "Target right endpoint unavailable: " + right["reason"]
    elif not geometric_result["structurally_valid_geometry"]:
        failure = "Designated donor geometry unavailable: " + str(geometric_result["geometry_failure_reason"])
    else:
        # A finite raw H pair can provide a vector even if either absolute point is out of frame.
        pair = np.asarray(geometric_result["raw_pair_xy"], dtype=float)
        vector = (pair[0] - pair[1]).tolist()
        left_xy = (np.asarray(right["xy"]) + np.asarray(vector)).tolist()
    left = _point(left_xy, dimensions, source="estimated_from_context", reason=failure)
    result = _strategy({"left": left, "right": right},
                        structural_validity=geometric_result["structurally_valid_geometry"],
                        failure_reason=geometric_result["geometry_failure_reason"], target_points=("right",))
    result["geometry_left_minus_right_xy"] = vector
    result["geometry_raw_pair_xy"] = copy.deepcopy(geometric_result["raw_pair_xy"])
    result["raw_geometry_out_of_frame_vector_permitted"] = True
    return result


def evaluate_outputs(target_keypoints, earlier_keypoints, later_keypoints, t, dimensions, *,
                     confidence_threshold=0.5, allow_extrapolation=False):
    """Keep the direct baseline separate from the two narrow contextual APIs."""
    contextual = model_donor_homography(earlier_keypoints, later_keypoints, t, dimensions,
        confidence_threshold=confidence_threshold, allow_extrapolation=allow_extrapolation)
    right_keypoint = target_keypoints[1] if target_keypoints is not None else None
    anchored = anchor_to_right(contextual, right_keypoint, dimensions, confidence_threshold=confidence_threshold)
    direct = direct_prediction(target_keypoints, dimensions, confidence_threshold)
    return {STRATEGIES[0]: direct, STRATEGIES[1]: contextual, STRATEGIES[2]: anchored}


def score_outputs(outputs, target):
    """Read reference coordinates only after all three strategy outputs exist."""
    reference = {}
    for index, slot in enumerate(SLOTS):
        visible = bool(target["visible"][index])
        reference[slot] = {"observed": visible,
                           "xy": target["xy"][index].tolist() if visible else None}
    for result in outputs.values():
        for slot, point in result["endpoints"].items():
            observed = reference[slot]["observed"]
            error = (float(np.linalg.norm(np.asarray(point["xy"]) - reference[slot]["xy"]))
                     if observed and point["xy"] is not None else None)
            point.update(reference_observed=observed, error_px=error,
                within_radius={str(radius): (point["available"] and error is not None and error <= radius)
                               if observed else None for radius in RADII})
        errors = [result["endpoints"][slot]["error_px"] for slot in SLOTS]
        result["pair_maximum_error_px"] = max(errors) if all(value is not None for value in errors) else None
        both_observed = all(reference[slot]["observed"] for slot in SLOTS)
        result["both_within_radius"] = {
            str(radius): all(result["endpoints"][slot]["within_radius"][str(radius)] for slot in SLOTS)
            if both_observed else None for radius in RADII}
    return reference


def _summarize_strategy(cases, strategy):
    records = [case["strategies"][strategy] for case in cases]
    result = {"target_denominator": len(cases), "slots": {}}
    for slot in SLOTS:
        points = [record["endpoints"][slot] for record in records]
        result["slots"][slot] = {
            "denominator": len(cases), "available": sum(point["available"] for point in points),
            "unavailable": sum(not point["available"] for point in points),
            "within_radius": {str(radius): sum(point["within_radius"][str(radius)] for point in points)
                              for radius in RADII},
            "finite_raw_error": geometry._error_stats([point["error_px"] for point in points
                                                         if point["error_px"] is not None]),
            "available_error": geometry._error_stats([point["error_px"] for point in points
                                                        if point["available"] and point["error_px"] is not None]),
            "unavailability_reasons": dict(sorted(Counter(point["reason"] for point in points
                                                           if not point["available"]).items()))}
    result["pairs"] = {
        "denominator": len(cases), "available": sum(record["status"] == "complete" for record in records),
        "unavailable": sum(record["status"] != "complete" for record in records),
        "both_within_radius": {str(radius): sum(record["both_within_radius"][str(radius)] for record in records)
                               for radius in RADII},
        "finite_raw_maximum_error": geometry._error_stats([record["pair_maximum_error_px"] for record in records
                                                            if record["pair_maximum_error_px"] is not None])}
    result["all_endpoints_finite_raw_error"] = geometry._error_stats([
        record["endpoints"][slot]["error_px"] for record in records for slot in SLOTS
        if record["endpoints"][slot]["error_px"] is not None])
    result["structural_geometry_failures"] = sum(record["structurally_valid_geometry"] is False for record in records)
    result["error_statistic_basis"] = (
        "Finite raw errors retain low-confidence/out-of-frame points; available errors apply the fixed gates. "
        "Success requires availability; failures remain in every fixed denominator.")
    return result


def _scope_result(result, scope):
    if scope == "pair":
        return result["status"] == "complete", result["pair_maximum_error_px"], result["both_within_radius"]
    point = result["endpoints"][scope]
    return point["available"], point["error_px"], point["within_radius"]


def paired_comparison(cases, candidate, baseline):
    result = {"candidate": candidate, "baseline": baseline, "denominator": len(cases), "scopes": {}}
    for scope in (*SLOTS, "pair"):
        rows = []
        for case in cases:
            ca, ce, cs = _scope_result(case["strategies"][candidate], scope)
            ba, be, bs = _scope_result(case["strategies"][baseline], scope)
            rows.append({"case_id": case["case_id"], "candidate_available": ca, "baseline_available": ba,
                         "candidate_error": ce, "baseline_error": be, "candidate_success": cs, "baseline_success": bs})
        deltas = [row["candidate_error"] - row["baseline_error"] for row in rows
                  if row["candidate_available"] and row["baseline_available"]
                  and row["candidate_error"] is not None and row["baseline_error"] is not None]
        thresholds = {}
        for radius in RADII:
            key = str(radius)
            gains = [row["case_id"] for row in rows if row["candidate_success"][key] and not row["baseline_success"][key]]
            losses = [row["case_id"] for row in rows if row["baseline_success"][key] and not row["candidate_success"][key]]
            thresholds[key] = {"gains": len(gains), "losses": len(losses), "net_success_change": len(gains) - len(losses),
                "gain_case_ids": gains, "loss_case_ids": losses,
                "both_succeed": sum(row["candidate_success"][key] and row["baseline_success"][key] for row in rows),
                "both_fail": sum(not row["candidate_success"][key] and not row["baseline_success"][key] for row in rows)}
        result["scopes"][scope] = {
            "denominator": len(cases), "at_radius": thresholds,
            "availability_gains": [row["case_id"] for row in rows if row["candidate_available"] and not row["baseline_available"]],
            "availability_losses": [row["case_id"] for row in rows if row["baseline_available"] and not row["candidate_available"]],
            "common_available_error_change": {"count": len(deltas), "sign": "candidate minus baseline; negative is better",
                "mean_delta_px": float(np.mean(deltas)) if deltas else None,
                "median_delta_px": float(np.median(deltas)) if deltas else None,
                "minimum_delta_px": float(min(deltas)) if deltas else None,
                "maximum_delta_px": float(max(deltas)) if deltas else None,
                "lower_error": sum(value < -1e-9 for value in deltas),
                "higher_error": sum(value > 1e-9 for value in deltas),
                "equal_within_1e_minus_9_px": sum(abs(value) <= 1e-9 for value in deltas)}}
    return result


def summarize(cases):
    return {"target_pair_denominator": len(cases), "endpoint_denominator": 2 * len(cases),
            "strategies": {strategy: _summarize_strategy(cases, strategy) for strategy in STRATEGIES},
            "paired_comparisons": [paired_comparison(cases, candidate, baseline) for candidate, baseline in COMPARISONS]}


def validate_model_inputs(bundle, plan, cohort, loaded, identities):
    if bundle.get("schema_version") != 1:
        raise PilotError("Unsupported model-input schema")
    if bundle["inference"]["threshold"] != plan["box_confidence_threshold"]:
        raise PilotError("Model-input box threshold differs from frozen plan")
    expected = {(stack["image_id"], member["row"]) for stack in cohort["stacks"] for member in stack["members"]}
    expected_images = {image_id for image_id, _ in expected}
    image_entries = {entry["image_id"]: entry for entry in bundle["images"]}
    if len(image_entries) != len(bundle["images"]) or set(image_entries) != expected_images:
        raise PilotError("Model-input images do not exactly cover the frozen stacks")
    identity_by_id = {entry["image_id"]: entry for entry in identities}
    for image_id, entry in image_entries.items():
        data = loaded[image_id]
        if [entry["width"], entry["height"]] != [data["width"], data["height"]]:
            raise PilotError(f"Model-input image dimensions mismatch: {image_id}")
        if entry["image_sha256"] != identity_by_id[image_id]["image_sha256"]:
            raise PilotError(f"Model-input image hash mismatch: {image_id}")
        if entry["threshold"] != plan["box_confidence_threshold"]:
            raise PilotError(f"Model-input image threshold mismatch: {image_id}")
    rows, prediction_ids = {}, set()
    for record in bundle["rows"]:
        key = (record["image_id"], record["row"])
        if key in rows:
            raise PilotError(f"Duplicate model-input row: {key}")
        rows[key] = record
        if not record["matched"]:
            if record["prediction"] is not None:
                raise PilotError("Unmatched input row cannot supply an assigned prediction")
            continue
        index = record["prediction_index_zero_based"]
        if type(index) is not int or not 0 <= index < image_entries[record["image_id"]]["source_prediction_count"]:
            raise PilotError("Invalid source prediction index")
        identity = (record["image_id"], index)
        if identity in prediction_ids:
            raise PilotError("Prediction assigned to more than one frozen reference row")
        prediction_ids.add(identity)
        overlap = record["matching_iou"]
        prediction = record["prediction"]
        if not _finite_number(overlap) or not plan["association_iou_threshold"] <= overlap <= 1:
            raise PilotError("Input association does not meet the fixed reference-box matching threshold")
        if prediction["class_id"] != 0 or not _finite_number(prediction["confidence"]) or not (
                plan["box_confidence_threshold"] <= prediction["confidence"] <= 1):
            raise PilotError("Input prediction class or box confidence is invalid")
        bbox = prediction["bbox_xyxy"]
        if len(bbox) != 4 or not all(_finite_number(value) for value in bbox) or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise PilotError("Invalid model bbox metadata")
        if len(prediction["keypoints"]) != 2 or any(len(point) != 3 for point in prediction["keypoints"]):
            raise PilotError("Model input must contain two x/y/confidence keypoints")
    if set(rows) != expected:
        raise PilotError("Model-input rows do not exactly cover the frozen stack members")
    return rows


def _case_metadata(baseline_case, stack, excluded, *, extrapolation):
    """Reuse the pinned donor identities; reference donor coordinates are ignored."""
    members = {member["row"]: member["rank"] for member in stack["members"]}
    row, rank = baseline_case["row"], baseline_case["rank"]
    if baseline_case["image_id"] != stack["image_id"] or members.get(row) != rank:
        raise PilotError("Baseline target row/rank differs from its frozen stack")
    if baseline_case["case_id"] != f"{stack['image_id']}:{row}" or baseline_case["stack_id"] != stack["id"]:
        raise PilotError("Baseline target identity differs from its frozen stack")
    if "excluded_rows" in baseline_case and set(baseline_case["excluded_rows"]) != set(excluded):
        raise PilotError("Baseline excluded-row set differs from the frozen cohort")
    donors = [{"row": donor["row"], "rank": donor["rank"]} for donor in baseline_case["donors"]]
    if len(donors) != 2 or any(members.get(donor["row"]) != donor["rank"] or donor["row"] in excluded for donor in donors):
        raise PilotError("Baseline designated donors have invalid membership or exclusion")
    earlier, later = donors
    if earlier["rank"] >= later["rank"]:
        raise PilotError("Baseline donor ranks are not increasing")
    if extrapolation:
        if later["rank"] >= min(members[value] for value in excluded):
            raise PilotError("Extrapolation donors are not above all excluded targets")
    elif not earlier["rank"] < rank < later["rank"]:
        raise PilotError("Interpolation donors do not bracket the target")
    t = (rank - earlier["rank"]) / (later["rank"] - earlier["rank"])
    if t != baseline_case["rank_ratio"] or not (0 < t < 1 or extrapolation and 1 < t <= 3):
        raise PilotError("Baseline rank ratio differs from frozen donor ranks")
    return {"case_id": baseline_case["case_id"], "image_id": stack["image_id"], "stack_id": stack["id"],
            "row": row, "rank": rank, "excluded_rows": sorted(excluded), "rank_ratio": t,
            "allow_extrapolation": extrapolation, "frozen_donor_identities": donors,
            "manual_stack_and_rank_used": True, "reference_box_matching_used": True,
            "reference_boxes_used_for_geometry": False, "reference_donor_coordinates_used": False,
            "target_reference_coordinates_used_for_prediction": False}


def _model_keypoints(record):
    return record["prediction"]["keypoints"] if record["matched"] else None


def _association(record):
    return {"matched": record["matched"], "prediction_index_zero_based": record["prediction_index_zero_based"],
            "matching_iou": record["matching_iou"],
            "box_confidence": record["prediction"]["confidence"] if record["matched"] else None}


def evaluate_case(baseline_case, stack, excluded, loaded, rows, plan, *, extrapolation=False):
    record = _case_metadata(baseline_case, stack, excluded, extrapolation=extrapolation)
    data = loaded[record["image_id"]]
    target_model = rows[(record["image_id"], record["row"])]
    donor_models = [rows[(record["image_id"], donor["row"])] for donor in record["frozen_donor_identities"]]
    outputs = evaluate_outputs(_model_keypoints(target_model), *[_model_keypoints(donor) for donor in donor_models],
        record["rank_ratio"], (data["width"], data["height"]),
        confidence_threshold=plan["endpoint_confidence_threshold"], allow_extrapolation=extrapolation)
    record["strategies"] = outputs
    record["target_reference_for_scoring_only"] = score_outputs(outputs, data["rows"][record["row"]])
    record["target_association"] = _association(target_model)
    record["donors"] = [{**identity, **_association(model), "model_keypoints": _model_keypoints(model)}
                        for identity, model in zip(record["frozen_donor_identities"], donor_models)]
    return record


def evaluate_hidden(case, baseline_case, stacks, loaded, rows, plan):
    shared = {**case, "case_id": f"{case['image_id']}:{case['row']}", "status": "review_only",
              "accepted_for_training": False, "accepted_as_observed_reference": False,
              "hidden_endpoint_accuracy_known": False, "confidence_calibrated": False}
    if case["disposition"] == "abstain":
        shared.update(status="abstained", donors=[], strategies={})
        return shared
    stack = stacks[case["stack_id"]]
    if baseline_case["rank"] != case["rank"] or baseline_case["disposition"] != case["disposition"]:
        raise PilotError("Hidden baseline identity differs from the frozen cohort")
    record = evaluate_case(baseline_case, stack, {case["row"]}, loaded, rows, plan,
                           extrapolation=case["disposition"] == "estimate_extrapolation")
    shared.update(record)
    shared["original_cohort_reason"] = case["reason"]
    shared["reason"] = (
        "Frozen hidden-corner case: reference endpoint coordinates are withheld from every predictor. "
        "The donor-only strategy uses designated model donors; the anchored strategy additionally uses "
        "the predicted target right. Missing-left reference accuracy remains unknown.")
    shared["right_check_limitation"] = (
        "Anchored right is the exact direct PR35 right, not an independent validation of the hidden left. "
        "Observed right reference coordinates are used only for scoring.")
    for strategy, output in shared["strategies"].items():
        output["missing_endpoint_proposals"] = [
            {"slot": slot, "xy": point["xy"], "confidence": point["confidence"],
             "status": point["source"] if point["available"] else "unavailable", "reason": point["reason"],
             "accepted_for_training": False, "accepted_as_observed_reference": False,
             "ground_truth_available": False, "geometric_error_px": None}
            for slot, point in output["endpoints"].items() if not point["reference_observed"]]
    return shared


def run_experiment(repo_root, plan_path, model_inputs_path, pilot_root=None):
    root = Path(pilot_root or Path(__file__).resolve().parents[1]).resolve()
    plan_path = Path(plan_path).resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema_version") != 1 or plan.get("strategies") != list(STRATEGIES):
        raise PilotError("Unsupported plan schema or strategy selection")
    if not plan.get("frozen_before_new_strategy_errors"):
        raise PilotError("Plan is not frozen before strategy errors")
    if plan["radii_px"] != list(RADII) or plan["endpoint_confidence_threshold"] != 0.5:
        raise PilotError("Plan differs from the fixed endpoint gates")
    if Path(model_inputs_path).resolve() != geometry._safe_path(root, plan["model_inputs_path"]):
        raise PilotError("CLI model-input path differs from the frozen plan")
    cohort, cohort_identity = read_pinned_json(root, plan["cohort_path"], plan["cohort_sha256"])
    baseline, baseline_identity = read_pinned_json(root, plan["baseline_evaluation_path"], plan["baseline_evaluation_sha256"])
    bundle, bundle_identity = read_pinned_json(root, plan["model_inputs_path"], plan["model_inputs_sha256"])
    if geometry.sha256(Path(geometry.__file__).read_bytes()) != baseline["provenance"]["script_sha256"]:
        raise PilotError("The reused geometry helper differs from the pinned baseline implementation")
    if baseline["provenance"]["cohort"]["sha256"] != cohort_identity["sha256"]:
        raise PilotError("Baseline used a different frozen cohort")
    loaded, identities = geometry.load_reference(repo_root, cohort)
    stacks = {stack["id"]: stack for stack in cohort["stacks"]}
    if len(stacks) != len(cohort["stacks"]):
        raise PilotError("Duplicate frozen stack ID")
    for stack in stacks.values():
        geometry._validate_stack(stack, loaded[stack["image_id"]])
    rows = validate_model_inputs(bundle, plan, cohort, loaded, identities)
    lane_results = {}
    for lane, selection in (("interpolation", "holdout_rows"), ("extrapolation", "extrapolation_holdout_rows")):
        expected = [(stack["id"], row) for stack in cohort["stacks"] for row in stack[selection]]
        cases = baseline["lanes"][lane]["cases"]
        if [(case["stack_id"], case["row"]) for case in cases] != expected:
            raise PilotError("Baseline lane targets or order differ from the frozen cohort")
        evaluated = []
        for case in cases:
            stack = stacks[case["stack_id"]]
            excluded = set(stack[selection]) if lane == "extrapolation" else {case["row"]}
            evaluated.append(evaluate_case(case, stack, excluded, loaded, rows, plan, extrapolation=lane == "extrapolation"))
        lane_results[lane] = {"cases": evaluated, "summary": summarize(evaluated),
            "per_stack": {stack_id: summarize([case for case in evaluated if case["stack_id"] == stack_id]) for stack_id in stacks}}
    hidden_baseline = {(case["image_id"], case["row"]): case for case in baseline["hidden_cases"]}
    if set(hidden_baseline) != {(case["image_id"], case["row"]) for case in cohort["hidden_cases"]}:
        raise PilotError("Baseline hidden-case identities differ from the frozen cohort")
    hidden = [evaluate_hidden(case, hidden_baseline[(case["image_id"], case["row"])], stacks, loaded, rows, plan)
              for case in cohort["hidden_cases"]]
    return {"schema_version": 1, "experiment_id": plan["experiment_id"], "status": "complete",
        "coordinate_system": "Original image pixels; finite half-open image bounds; no clipping or resizing",
        "strategies": list(STRATEGIES), "radii_px": list(RADII), "endpoint_confidence_threshold": 0.5,
        "purpose": "Compare fixed cached PR35, designated model-donor homography and right-anchored homography",
        "oracle_assistance": "Manual stack membership/ranks and frozen reference-box associations; neither reference boxes nor reference endpoint coordinates enter the contextual predictors",
        "limits": list(plan["limits"]) + [
            "Unavailable designated donors are not replaced by farther rows",
            "Finite raw H points outside the image may supply an anchored vector; each final endpoint is checked independently",
            "Context-derived endpoints have confidence null and never inherit donor confidence",
            "Model inference used intact target pixels; this is not a pixel-occlusion experiment",
            "No inferred endpoint is approved for training or as an observed reference"],
        "provenance": {
            "plan": {"filename": plan_path.name, "sha256": geometry.sha256(plan_bytes), "bytes": len(plan_bytes)},
            "cohort": cohort_identity, "baseline_evaluation": baseline_identity, "model_inputs": bundle_identity,
            "script_sha256": geometry.sha256(Path(__file__).read_bytes()),
            "reused_geometry_script_sha256": geometry.sha256(Path(geometry.__file__).read_bytes()),
            "reference_manifest": cohort["reference_manifest"], "reference_manifest_sha256": cohort["reference_manifest_sha256"],
            "base_commit": plan.get("base_commit"), "source_model": bundle["source_model"], "model_sha256": bundle["model_sha256"],
            "model_evidence": {key: bundle[key] for key in ("source_evidence_path", "source_evidence_commit", "source_files", "inference", "association")},
            "verified_source_files": identities, "model_input_row_count": len(rows),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__}},
        "lanes": lane_results, "hidden_cases": hidden}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--model-inputs", required=True, type=Path)
    parser.add_argument("--pilot-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path, help="New JSON output file outside the source repository")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.is_relative_to(args.repo_root.resolve()):
            raise PilotError("Output must be outside the source repository")
        if output.exists():
            raise PilotError("Output already exists")
        result = run_experiment(args.repo_root, args.plan, args.model_inputs, args.pilot_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "complete", "output": str(output),
            "lane_pair_counts": {name: value["summary"]["target_pair_denominator"] for name, value in result["lanes"].items()}}))
    except (PilotError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Model-anchor pilot failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
