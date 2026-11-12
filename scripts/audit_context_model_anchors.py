#!/usr/bin/env python3
"""Independently replay the frozen PR35 model-anchor diagnostic audit.

Requires Python 3.12+ and NumPy. Source predictions/scoring may be extracted from
the pinned PR53 evidence archive. Reads source files and writes only a new audit
JSON. The original PNGs must be restored rather than Git LFS pointers.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np


FOLLOWUP = "results/review/context-model-anchors-20261112"
PLAN_SHA = "45c12b2e2e8617ad76a0b2c812122b9156eb29d908e721bc79e334bac4fb22cd"
INPUT_SHA = "7cbed9aef5970e8d2bbb85a19770fdfebec5ea7d1801b0c422d49f9534025004"
PREDICTION_SHA = "45587e9bd1c194fe44730a5b5404408856aa80bfb4e582632acaefb19e95c700"
SCORING_SHA = "4b97981807f0d7baddc84e9cba8e8e4b79c33bc0686aa316ac45c899b2ec4344"
STRATEGIES = ("direct_pr35", "model_donor_homography", "right_anchored_homography")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0

    def equal(self, actual, expected, label):
        self.count += 1
        if actual != expected:
            raise AssertionError((label, actual, expected))

    def close(self, actual, expected, label, tolerance=1e-8):
        self.count += 1
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=tolerance):
            raise AssertionError((label, actual, expected))

    def structure(self, actual, expected, label):
        if isinstance(expected, dict):
            self.equal(set(actual), set(expected), label + "/keys")
            for key in expected:
                self.structure(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, list):
            self.equal(len(actual), len(expected), label + "/length")
            for index, (first, second) in enumerate(zip(actual, expected)):
                self.structure(first, second, label + "/" + str(index))
        elif isinstance(expected, float):
            self.close(actual, expected, label)
        else:
            self.equal(actual, expected, label)


def bbox_iou(first, second):
    def area(box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    intersection = area([max(first[0], second[0]), max(first[1], second[1]),
                         min(first[2], second[2]), min(first[3], second[3])])
    return intersection / (area(first) + area(second) - intersection)


def available(point, width, height, threshold=0.5):
    x, y, confidence = point
    return (all(math.isfinite(value) for value in point) and confidence >= threshold
            and 0 <= x < width and 0 <= y < height)


def in_frame(point, width, height):
    return bool(all(math.isfinite(value) for value in point) and 0 <= point[0] < width and 0 <= point[1] < height)


def independent_homography(first, second, ratio):
    """Direct eight-equation solution; no DLT, SVD or implementation imports."""
    source = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    destination = [first[0], first[1], second[1], second[0]]
    design, right_hand = [], []
    for (x, y), (u, v) in zip(source, destination):
        design.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        right_hand.append(u)
        design.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        right_hand.append(v)
    matrix = np.append(np.linalg.solve(np.asarray(design), np.asarray(right_hand)), 1).reshape(3, 3)
    homogeneous = np.array([[0.0, ratio, 1.0], [1.0, ratio, 1.0]]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def audit_source_bundle(root, repo, prediction_path, scoring_path, checks):
    directory = root / FOLLOWUP
    plan_raw = (directory / "plan.json").read_bytes()
    input_raw = (directory / "model-inputs.json").read_bytes()
    checks.equal(digest(plan_raw), PLAN_SHA, "frozen plan hash")
    checks.equal(digest(input_raw), INPUT_SHA, "frozen input bundle hash")
    plan, inputs = json.loads(plan_raw), json.loads(input_raw)
    checks.equal(plan["model_inputs_sha256"], INPUT_SHA, "plan binds input bundle")
    checks.equal(plan["strategies"], list(STRATEGIES), "fixed three strategies")
    checks.equal(plan["endpoint_confidence_threshold"], 0.5, "endpoint threshold")
    checks.equal(plan["box_confidence_threshold"], 0.13, "box threshold")
    checks.equal(plan["association_iou_threshold"], 0.5, "matching threshold")
    checks.equal(plan["radii_px"], [5, 10], "success radii")
    cohort_raw = (root / plan["cohort_path"]).read_bytes()
    baseline_raw = (root / plan["baseline_evaluation_path"]).read_bytes()
    checks.equal(digest(cohort_raw), plan["cohort_sha256"], "frozen cohort")
    checks.equal(digest(baseline_raw), plan["baseline_evaluation_sha256"], "frozen baseline evaluation")
    cohort, baseline = json.loads(cohort_raw), json.loads(baseline_raw)
    prediction_raw, scoring_raw = prediction_path.read_bytes(), scoring_path.read_bytes()
    checks.equal(digest(prediction_raw), PREDICTION_SHA, "original predictions hash")
    checks.equal(digest(scoring_raw), SCORING_SHA, "original scorer hash")
    source_raw = {"evaluation/predictions-current.jsonl": prediction_raw,
                  "evaluation/scoring-current-primary.json": scoring_raw}
    for record in inputs["source_files"]:
        checks.equal(digest(source_raw[record["member"]]), record["sha256"], "bundle source hash")
        checks.equal(len(source_raw[record["member"]]), record["bytes"], "bundle source size")
    predictions = [json.loads(line) for line in prediction_raw.splitlines()]
    header = predictions[0]
    by_image = {record["image_id"]: record for record in predictions[1:]}
    score_images = {record["image_id"]: record for record in json.loads(scoring_raw)["0.13"]["per_image"]}
    checks.equal(inputs["model_sha256"], header["model_sha256"], "checkpoint identity")
    checks.equal(header["thresholds"], [0.13], "cached inference threshold")
    checks.equal({key: value for key, value in inputs["inference"].items() if key != "threshold"},
                 header["inference"], "inference arguments")
    checks.equal(inputs["inference"]["threshold"], 0.13, "bundle threshold")
    manifest_path = repo / cohort["reference_manifest"]
    manifest_raw = manifest_path.read_bytes()
    checks.equal(digest(manifest_raw), cohort["reference_manifest_sha256"], "corrected reference manifest")
    manifest = {entry["image_id"]: entry for entry in json.loads(manifest_raw)["files"]}
    labels, dimensions, image_identities = {}, {}, []
    expected_members = {(stack["image_id"], member["row"]) for stack in cohort["stacks"] for member in stack["members"]}
    checks.equal(len(inputs["rows"]), 36, "member count")
    checks.equal({(record["image_id"], record["row"]) for record in inputs["rows"]}, expected_members, "exact member set")
    checks.equal({entry["image_id"] for entry in inputs["images"]}, {key[0] for key in expected_members}, "exact image set")
    for image in inputs["images"]:
        identity, source = manifest[image["image_id"]], by_image[image["image_id"]]
        for key in ("image_id", "width", "height", "image_sha256", "threshold"):
            checks.equal(image[key], source[key], "image extraction " + key)
        checks.equal(image["source_prediction_count"], len(source["predictions"]), "source proposal count")
        checks.equal(image["image_sha256"], identity["image_sha256"], "reference/image identity")
        original = repo / "data/images/val" / (image["image_id"] + ".png")
        png = original.read_bytes()
        checks.equal(digest(png), image["image_sha256"], "original PNG hash")
        checks.equal(png[:8], b"\x89PNG\r\n\x1a\n", "actual PNG rather than pointer")
        width, height = struct.unpack(">II", png[16:24])
        checks.equal([width, height], [image["width"], image["height"]], "PNG dimensions")
        label_raw = (manifest_path.parent / identity["label_path"]).read_bytes()
        checks.equal(digest(label_raw), identity["label_sha256"], "corrected label hash")
        rows = {}
        for row, line in enumerate(label_raw.decode().splitlines(), 1):
            fields = list(map(float, line.split()))
            cx, cy, bw, bh = fields[1:5]
            rows[row] = {"class_id": int(fields[0]),
                         "bbox": [(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height],
                         "xy": np.array([[fields[5]*width, fields[6]*height], [fields[8]*width, fields[9]*height]]),
                         "visible": [fields[7] == 2, fields[10] == 2]}
        labels[image["image_id"]], dimensions[image["image_id"]] = rows, (width, height)
        image_identities.append({"image_id": image["image_id"], "image_sha256": image["image_sha256"],
                                 "label_sha256": identity["label_sha256"]})
    max_iou_difference, minimum_confidence = 0.0, 1.0
    members = {}
    for row in inputs["rows"]:
        image_id, number = row["image_id"], row["row"]
        official = [record for record in score_images[image_id]["objects"] if record["reference_row_one_based"] == number]
        checks.equal(len(official), 1, "unique saved association")
        official = official[0]
        checks.equal(row["matched"], True, "included object matched")
        for key in ("matched", "prediction_index_zero_based", "matching_iou"):
            checks.equal(row[key], official[key], "association " + key)
        checks.equal(row["source_endpoint_returned"], official["endpoint_returned"], "saved availability")
        prediction = by_image[image_id]["predictions"][row["prediction_index_zero_based"]]
        checks.equal(row["prediction"], prediction, "complete exact source prediction")
        reference = labels[image_id][number]
        overlap = bbox_iou(reference["bbox"], prediction["bbox_xyxy"])
        checks.close(overlap, row["matching_iou"], "independent reference-box IoU", 1e-12)
        max_iou_difference = max(max_iou_difference, abs(overlap - row["matching_iou"]))
        checks.equal(overlap >= 0.5, True, "association IoU passes")
        checks.equal(prediction["confidence"] >= 0.13, True, "object confidence passes")
        checks.equal(prediction["class_id"], reference["class_id"], "class matches")
        width, height = dimensions[image_id]
        returned = [available(point, width, height) for point in prediction["keypoints"]]
        checks.equal(returned, row["source_endpoint_returned"], "independent endpoint availability")
        minimum_confidence = min(minimum_confidence, *(point[2] for point in prediction["keypoints"]))
        members[image_id, number] = row
    result = {"status": "pass", "member_records": len(members), "images": len(inputs["images"]),
              "exact_source_prediction_records": len(members), "exact_saved_scorer_associations": len(members),
              "independently_recomputed_bbox_ious": len(members), "maximum_bbox_iou_difference": max_iou_difference,
              "minimum_included_endpoint_confidence": minimum_confidence,
              "source_prediction_sha256": digest(prediction_raw), "source_scoring_sha256": digest(scoring_raw),
              "image_and_label_identities": image_identities,
              "matching_policy": "Use frozen reference-box scorer association; never rematch using endpoint errors."}
    return plan, inputs, cohort, baseline, members, labels, dimensions, result


def derive_case(case, members, labels, dimensions):
    image_id, row = case["image_id"], case["row"]
    donor_rows = [donor["row"] for donor in case["donors"]]
    donor_points = [np.asarray(members[image_id, donor]["prediction"]["keypoints"])[:, :2] for donor in donor_rows]
    direct = np.asarray(members[image_id, row]["prediction"]["keypoints"])[:, :2]
    geometric = independent_homography(*donor_points, case["rank_ratio"])
    anchored = np.array([direct[1] + geometric[0] - geometric[1], direct[1]])
    truth = labels[image_id][row]
    width, height = dimensions[image_id]
    strategies = {}
    for name, pair in zip(STRATEGIES, (direct, geometric, anchored)):
        returned = [in_frame(point, width, height) for point in pair]
        errors = [math.hypot(*(point-reference)) if visible else None
                  for point, reference, visible in zip(pair, truth["xy"], truth["visible"])]
        strategies[name] = {"raw_pair_xy": pair.tolist(), "endpoint_available": returned,
                            "endpoint_errors_px": errors,
                            "endpoint_within_radius": {str(radius): [(available_ and error <= radius) if error is not None else None
                                for available_, error in zip(returned, errors)] for radius in (5, 10)}}
    return {"case_id": case["case_id"], "stack_id": case["stack_id"], "image_id": image_id,
            "row": row, "donor_rows": donor_rows, "ratio": case["rank_ratio"], "strategies": strategies}


def error_stats(values):
    ordered = sorted(values)
    def quantile(fraction):
        location = (len(ordered) - 1) * fraction
        lower, upper = math.floor(location), math.ceil(location)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)
    return {"count": len(ordered), "median_px": quantile(0.5) if ordered else None,
            "p90_px": quantile(0.9) if ordered else None, "maximum_px": max(ordered) if ordered else None}


def independent_summary(cases):
    # Every included real-cohort endpoint passes the frozen confidence and frame gates.
    # Explicit denominators use all cases, not a filtered successful subset.
    total = len(cases)
    result = {"target_pair_denominator": total, "endpoint_denominator": 2 * total, "strategies": {}}
    for strategy in STRATEGIES:
        records = [case["strategies"][strategy] for case in cases]
        item = {"target_denominator": total, "slots": {}}
        for index, slot in enumerate(("left", "right")):
            errors = [record["endpoint_errors_px"][index] for record in records]
            availability = [record["endpoint_available"][index] for record in records]
            item["slots"][slot] = {"denominator": total, "available": sum(availability),
                "unavailable": total - sum(availability),
                "within_radius": {str(radius): sum(record["endpoint_within_radius"][str(radius)][index] for record in records)
                                  for radius in (5, 10)},
                "finite_raw_error": error_stats(errors),
                "available_error": error_stats([error for error, valid in zip(errors, availability) if valid]),
                "unavailability_reasons": {}}
        maxima = [max(record["endpoint_errors_px"]) for record in records]
        pair_available = [all(record["endpoint_available"]) for record in records]
        item["pairs"] = {"denominator": total, "available": sum(pair_available),
            "unavailable": total - sum(pair_available),
            "both_within_radius": {str(radius): sum(all(record["endpoint_within_radius"][str(radius)]) for record in records)
                                   for radius in (5, 10)},
            "finite_raw_maximum_error": error_stats(maxima)}
        item["all_endpoints_finite_raw_error"] = error_stats([error for record in records for error in record["endpoint_errors_px"]])
        item["structural_geometry_failures"] = 0
        item["error_statistic_basis"] = (
            "Finite raw errors retain low-confidence/out-of-frame points; available errors apply the fixed gates. "
            "Success requires availability; failures remain in every fixed denominator.")
        result["strategies"][strategy] = item
    result["paired_comparisons"] = []
    for candidate, baseline in ((STRATEGIES[1], STRATEGIES[0]), (STRATEGIES[2], STRATEGIES[0]), (STRATEGIES[2], STRATEGIES[1])):
        comparison = {"candidate": candidate, "baseline": baseline, "denominator": total, "scopes": {}}
        for scope in ("left", "right", "pair"):
            differences, rows = [], []
            for case in cases:
                first, second = case["strategies"][candidate], case["strategies"][baseline]
                if scope == "pair":
                    first_valid, second_valid = all(first["endpoint_available"]), all(second["endpoint_available"])
                    first_error, second_error = max(first["endpoint_errors_px"]), max(second["endpoint_errors_px"])
                    first_success = {str(radius): all(first["endpoint_within_radius"][str(radius)]) for radius in (5, 10)}
                    second_success = {str(radius): all(second["endpoint_within_radius"][str(radius)]) for radius in (5, 10)}
                else:
                    index = ("left", "right").index(scope)
                    first_valid, second_valid = first["endpoint_available"][index], second["endpoint_available"][index]
                    first_error, second_error = first["endpoint_errors_px"][index], second["endpoint_errors_px"][index]
                    first_success = {str(radius): first["endpoint_within_radius"][str(radius)][index] for radius in (5, 10)}
                    second_success = {str(radius): second["endpoint_within_radius"][str(radius)][index] for radius in (5, 10)}
                if first_valid and second_valid:
                    differences.append(first_error - second_error)
                rows.append((case["case_id"], first_valid, second_valid, first_success, second_success))
            at_radius = {}
            for radius in ("5", "10"):
                gains = [row[0] for row in rows if row[3][radius] and not row[4][radius]]
                losses = [row[0] for row in rows if row[4][radius] and not row[3][radius]]
                at_radius[radius] = {"gains": len(gains), "losses": len(losses),
                    "net_success_change": len(gains) - len(losses), "gain_case_ids": gains, "loss_case_ids": losses,
                    "both_succeed": sum(row[3][radius] and row[4][radius] for row in rows),
                    "both_fail": sum(not row[3][radius] and not row[4][radius] for row in rows)}
            comparison["scopes"][scope] = {"denominator": total, "at_radius": at_radius,
                "availability_gains": [row[0] for row in rows if row[1] and not row[2]],
                "availability_losses": [row[0] for row in rows if row[2] and not row[1]],
                "common_available_error_change": {"count": len(differences),
                    "sign": "candidate minus baseline; negative is better",
                    "mean_delta_px": sum(differences) / len(differences) if differences else None,
                    "median_delta_px": error_stats(differences)["median_px"],
                    "minimum_delta_px": min(differences) if differences else None,
                    "maximum_delta_px": max(differences) if differences else None,
                    "lower_error": sum(value < -1e-9 for value in differences),
                    "higher_error": sum(value > 1e-9 for value in differences),
                    "equal_within_1e_minus_9_px": sum(abs(value) <= 1e-9 for value in differences)}}
        result["paired_comparisons"].append(comparison)
    return result


def audit_case(actual, expected, baseline_case, members, labels, checks, hidden=False):
    label = actual["case_id"]
    image_id, row = expected["image_id"], expected["row"]
    target_source = members[image_id, row]
    for key in ("case_id", "stack_id", "image_id", "row"):
        checks.equal(actual[key], expected[key], label + "/" + key)
    checks.equal(actual["rank"], baseline_case["rank"], label + "/rank")
    checks.equal(actual["rank_ratio"], expected["ratio"], label + "/ratio")
    checks.equal(actual["excluded_rows"], baseline_case.get("excluded_rows", [row]), label + "/excluded rows")
    checks.equal([donor["row"] for donor in actual["donors"]], expected["donor_rows"], label + "/exact fixed donor rows")
    checks.equal(set(expected["donor_rows"]) & set(actual["excluded_rows"]), set(), label + "/donors exclude targets")
    checks.equal(actual["target_reference_coordinates_used_for_prediction"], False, label + "/no reference target inputs")
    checks.equal(actual["reference_boxes_used_for_geometry"], False, label + "/no reference bbox geometry")
    checks.equal(actual["reference_donor_coordinates_used"], False, label + "/no reference donor coordinates")
    checks.equal(actual["allow_extrapolation"], expected["ratio"] > 1, label + "/explicit extrapolation")
    def association(record):
        return {"matched": record["matched"], "prediction_index_zero_based": record["prediction_index_zero_based"],
                "matching_iou": record["matching_iou"], "box_confidence": record["prediction"]["confidence"]}
    checks.equal(actual["target_association"], association(target_source), label + "/target exact saved association")
    for donor, old_donor in zip(actual["donors"], baseline_case["donors"]):
        source = members[image_id, donor["row"]]
        checks.equal(donor, {"row": old_donor["row"], "rank": old_donor["rank"], **association(source),
                            "model_keypoints": source["prediction"]["keypoints"]}, label + "/exact donor source")
    max_difference = 0.0
    for strategy in STRATEGIES:
        output, want = actual["strategies"][strategy], expected["strategies"][strategy]
        checks.equal(output["target_reference_endpoints_used"], False, label + "/no reference endpoints")
        checks.equal(output["target_bbox_used_for_geometry"], False, label + "/no target bbox")
        target_dependencies = ["left", "right"] if strategy == STRATEGIES[0] else [] if strategy == STRATEGIES[1] else ["right"]
        checks.equal(output["target_model_endpoints_used"], target_dependencies, label + "/explicit target dependencies")
        checks.equal(output["status"], "complete", label + "/availability status")
        checks.equal(output["geometry_failure_reason"], None, label + "/no geometry failure")
        checks.equal(output["structurally_valid_geometry"], None if strategy == STRATEGIES[0] else True, label + "/geometry status")
        checks.structure(output["raw_pair_xy"], want["raw_pair_xy"], label + "/" + strategy + "/independent pair")
        max_difference = max(max_difference, float(np.max(np.abs(np.asarray(output["raw_pair_xy"]) - want["raw_pair_xy"]))))
        for index, slot in enumerate(("left", "right")):
            point = output["endpoints"][slot]
            direct = strategy == STRATEGIES[0] or (strategy == STRATEGIES[2] and index == 1)
            source_point = target_source["prediction"]["keypoints"][index]
            if direct:
                checks.equal(point["xy"], source_point[:2], label + "/exact copied direct coordinates")
            else:
                checks.structure(point["xy"], want["raw_pair_xy"][index], label + "/context coordinates")
            checks.equal(point["confidence"], source_point[2] if direct else None, label + "/confidence provenance")
            checks.equal(point["source"], "direct_pr35" if direct else "estimated_from_context", label + "/point provenance")
            checks.equal(point["available"], want["endpoint_available"][index], label + "/availability")
            checks.equal(point["in_frame"], True, label + "/in frame")
            checks.equal(point["reason"], None, label + "/endpoint reason")
            checks.equal(point["reference_observed"], labels[image_id][row]["visible"][index], label + "/observability")
            checks.structure(point["error_px"], want["endpoint_errors_px"][index], label + "/independent error")
            checks.equal(point["within_radius"], {radius: values[index] for radius, values in want["endpoint_within_radius"].items()},
                         label + "/independent endpoint successes")
        truth = labels[image_id][row]
        for index, slot in enumerate(("left", "right")):
            checks.equal(actual["target_reference_for_scoring_only"][slot],
                         {"observed": truth["visible"][index], "xy": truth["xy"][index].tolist() if truth["visible"][index] else None},
                         label + "/target reference is scoring only")
        maxima = None if hidden else max(want["endpoint_errors_px"])
        checks.structure(output["pair_maximum_error_px"], maxima, label + "/independent pair maximum")
        checks.equal(output["both_within_radius"], {radius: None if hidden else all(values)
                     for radius, values in want["endpoint_within_radius"].items()}, label + "/independent pair successes")
        if hidden:
            proposal = output["missing_endpoint_proposals"]
            checks.equal(len(proposal), 1, label + "/single unknown left proposal")
            checks.equal(proposal[0]["slot"], "left", label + "/unknown left")
            checks.equal(proposal[0]["xy"], output["endpoints"]["left"]["xy"], label + "/proposal matches raw left")
            for key in ("accepted_for_training", "accepted_as_observed_reference", "ground_truth_available"):
                checks.equal(proposal[0][key], False, label + "/unapproved hidden proposal")
            checks.equal(proposal[0]["geometric_error_px"], None, label + "/hidden error remains unknown")
    checks.equal(actual["strategies"][STRATEGIES[2]]["endpoints"]["right"],
                 actual["strategies"][STRATEGIES[0]]["endpoints"]["right"], label + "/anchored right is exact direct right")
    return max_difference


def audit_evaluation(root, plan, inputs, cohort, baseline, members, labels, dimensions, derived, hidden, checks):
    raw = (root / FOLLOWUP / "evaluation.json").read_bytes()
    evaluation = json.loads(raw)
    script = root / "scripts/context_corner_model_anchors.py"
    provenance = evaluation["provenance"]
    checks.equal(provenance["script_sha256"], digest(script.read_bytes()), "evaluation binds audited implementation")
    checks.equal(provenance["reused_geometry_script_sha256"],
                 digest((root / "scripts/context_corner_pilot.py").read_bytes()), "evaluation binds original geometry helper")
    for key, expected_hash in (("plan", PLAN_SHA), ("model_inputs", INPUT_SHA),
                               ("cohort", plan["cohort_sha256"]), ("baseline_evaluation", plan["baseline_evaluation_sha256"])):
        checks.equal(provenance[key]["sha256"], expected_hash, "evaluation dependency " + key)
    checks.equal(provenance["model_sha256"], inputs["model_sha256"], "evaluation model identity")
    checks.equal(provenance["model_input_row_count"], 36, "evaluation member count")
    checks.equal(evaluation["strategies"], list(STRATEGIES), "evaluation strategies")
    checks.equal(evaluation["radii_px"], [5, 10], "evaluation radii")
    checks.equal(evaluation["endpoint_confidence_threshold"], 0.5, "evaluation endpoint cutoff")
    maximum_difference, count = 0.0, 0
    summaries, substitutions = {}, {}
    for lane in ("interpolation", "extrapolation"):
        expected_count = 26 if lane == "interpolation" else 8
        actual_cases = evaluation["lanes"][lane]["cases"]
        checks.equal(len(actual_cases), expected_count, lane + "/fixed denominator")
        checks.equal([case["case_id"] for case in actual_cases],
                     [case["case_id"] for case in baseline["lanes"][lane]["cases"]], lane + "/frozen case order")
        displacements, detailed = [], []
        for actual, expected, prior in zip(actual_cases, derived[lane], baseline["lanes"][lane]["cases"]):
            maximum_difference = max(maximum_difference, audit_case(actual, expected, prior, members, labels, checks))
            pair = expected["strategies"][STRATEGIES[1]]["raw_pair_xy"]
            original = prior["methods"]["homography"]["raw_pair_xy"]
            differences = [math.hypot(a[0]-b[0], a[1]-b[1]) for a, b in zip(pair, original)]
            displacements.extend(differences)
            detailed.append({"case_id": actual["case_id"], "left_displacement_px": differences[0],
                             "right_displacement_px": differences[1]})
            count += 1
        summaries[lane] = independent_summary(derived[lane])
        checks.structure(evaluation["lanes"][lane]["summary"], summaries[lane], lane + "/all summary fields")
        for stack in cohort["stacks"]:
            expected = independent_summary([case for case in derived[lane] if case["stack_id"] == stack["id"]])
            checks.structure(evaluation["lanes"][lane]["per_stack"][stack["id"]], expected, lane + "/" + stack["id"] + "/summary")
        substitutions[lane] = {
            "meaning": "Change in H estimates when replacing the frozen manual donor points with PR35 donor points; not hidden-point accuracy.",
            "all_endpoint_displacement": error_stats(displacements),
            "cases": detailed}
    distinct = {(case["image_id"], case["row"]) for rows in derived.values() for case in rows}
    checks.equal(len(distinct), 30, "distinct observed objects across overlapping lanes")
    actual_hidden = [case for case in evaluation["hidden_cases"] if case["disposition"] == "estimate_extrapolation"]
    prior_hidden = [case for case in baseline["hidden_cases"] if case["disposition"] == "estimate_extrapolation"]
    checks.equal(len(actual_hidden), 2, "actual hidden cases")
    for actual, expected, prior in zip(actual_hidden, hidden, prior_hidden):
        maximum_difference = max(maximum_difference, audit_case(actual, expected, prior, members, labels, checks, hidden=True))
        for key in ("accepted_for_training", "accepted_as_observed_reference", "hidden_endpoint_accuracy_known", "confidence_calibrated"):
            checks.equal(actual[key], False, "unknown hidden-left status")
        count += 1
    abstentions = [case for case in evaluation["hidden_cases"] if case["disposition"] == "abstain"]
    checks.equal(len(abstentions), 3, "frozen abstentions")
    checks.equal([case["case_id"] for case in evaluation["hidden_cases"]],
                 [case["case_id"] for case in baseline["hidden_cases"]], "all hidden/abstained identities")
    for case in abstentions:
        checks.equal(case["strategies"], {}, "abstention has no coordinate claims")
        checks.equal(case["donors"], [], "abstention has no donors")
        checks.equal(case["status"], "abstained", "abstention status")
    return evaluation, {
        "status": "pass", "script_sha256": digest(script.read_bytes()), "evaluation_sha256": digest(raw),
        "scenarios_independently_calculated": count, "strategy_pairs_compared": count * 3,
        "homography_fits_independently_solved": count,
        "maximum_absolute_coordinate_difference_px": maximum_difference,
        "lane_denominators": {"interpolation": 26, "extrapolation": 8},
        "distinct_observed_objects": len(distinct), "actual_hidden_cases_with_unknown_left_error": 2,
        "abstentions": 3, "all_lane_and_per_stack_summaries_and_paired_comparisons_recomputed": True,
        "summary": {lane: {strategy: {
            "left_5px": summary["strategies"][strategy]["slots"]["left"]["within_radius"]["5"],
            "left_10px": summary["strategies"][strategy]["slots"]["left"]["within_radius"]["10"],
            "right_5px": summary["strategies"][strategy]["slots"]["right"]["within_radius"]["5"],
            "right_10px": summary["strategies"][strategy]["slots"]["right"]["within_radius"]["10"],
            "pair_5px": summary["strategies"][strategy]["pairs"]["both_within_radius"]["5"],
            "pair_10px": summary["strategies"][strategy]["pairs"]["both_within_radius"]["10"],
            "left_error": summary["strategies"][strategy]["slots"]["left"]["finite_raw_error"]}
            for strategy in STRATEGIES} for lane, summary in summaries.items()},
        "donor_substitution_sensitivity": substitutions,
        "actual_hidden_estimates": hidden}


def audit_dependencies(root, plan, cohort, baseline, members, labels, dimensions, evaluation, checks):
    # These targeted input perturbations audit information flow on the frozen cases.
    # They are not image occlusion tests and never contribute to accuracy denominators.
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("model_anchor_implementation_under_audit",
                                                root / "scripts/context_corner_model_anchors.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stacks = {stack["id"]: stack for stack in cohort["stacks"]}
    loaded = {image_id: {"width": dimensions[image_id][0], "height": dimensions[image_id][1], "rows": rows}
              for image_id, rows in labels.items()}
    records = [(case, lane == "extrapolation") for lane in ("interpolation", "extrapolation")
               for case in baseline["lanes"][lane]["cases"]]
    records += [(case, True) for case in baseline["hidden_cases"] if case["disposition"] == "estimate_extrapolation"]
    left_checks, reference_checks, right_checks = 0, 0, 0
    for prior, extrapolation in records:
        stack, key = stacks[prior["stack_id"]], (prior["image_id"], prior["row"])
        excluded = set(prior.get("excluded_rows", [prior["row"]]))
        original = module.evaluate_case(prior, stack, excluded, loaded, members, plan, extrapolation=extrapolation)
        altered_members = copy.deepcopy(members)
        for number in excluded:
            altered_members[key[0], number]["prediction"]["keypoints"][0] = [float("nan"), float("nan"), 0.0]
            altered_members[key[0], number]["prediction"]["bbox_xyxy"] = [0.0, 0.0, 1.0, 1.0]
        altered = module.evaluate_case(prior, stack, excluded, loaded, altered_members, plan, extrapolation=extrapolation)
        for strategy in STRATEGIES[1:]:
            checks.equal(altered["strategies"][strategy], original["strategies"][strategy],
                         prior["case_id"] + "/context unchanged when target left and target box fields are corrupted")
        left_checks += 1
        altered_loaded = copy.deepcopy(loaded)
        for reference in altered_loaded[key[0]]["rows"].values():
            reference["xy"] = np.array([[987654.0, -876543.0], [-765432.0, 654321.0]])
            reference["bbox"] = [0.0, 0.0, 1.0, 1.0]
        altered = module.evaluate_case(prior, stack, excluded, altered_loaded, members, plan, extrapolation=extrapolation)
        for strategy in STRATEGIES:
            checks.equal(altered["strategies"][strategy]["raw_pair_xy"], original["strategies"][strategy]["raw_pair_xy"],
                         prior["case_id"] + "/reference donor and target coordinate fields do not enter prediction")
        reference_checks += 1
        altered_members = copy.deepcopy(members)
        shift = [1.25, -2.5]
        for index in (0, 1):
            altered_members[key]["prediction"]["keypoints"][1][index] += shift[index]
        altered = module.evaluate_case(prior, stack, excluded, loaded, altered_members, plan, extrapolation=extrapolation)
        checks.equal(altered["strategies"][STRATEGIES[1]], original["strategies"][STRATEGIES[1]],
                     prior["case_id"] + "/target right does not enter donor-only H")
        shifted = original["strategies"][STRATEGIES[2]]["raw_pair_xy"]
        expected = [[point[0] + shift[0], point[1] + shift[1]] for point in shifted]
        checks.structure(altered["strategies"][STRATEGIES[2]]["raw_pair_xy"], expected,
                         prior["case_id"] + "/anchor translates exactly with supplied target right")
        checks.equal(altered["strategies"][STRATEGIES[2]]["endpoints"]["right"]["xy"],
                     altered_members[key]["prediction"]["keypoints"][1][:2], prior["case_id"] + "/changed right copied exactly")
        right_checks += 1
    # One real-case failure injection tests the all-cases denominator and copied-right
    # behavior, which the unmodified high-confidence cohort cannot exercise.
    prior = baseline["lanes"]["interpolation"]["cases"][0]
    key = (prior["image_id"], prior["donors"][0]["row"])
    altered_members = copy.deepcopy(members)
    altered_members[key]["prediction"]["keypoints"][0][2] = 0.49
    failed = module.evaluate_case(prior, stacks[prior["stack_id"]], {prior["row"]}, loaded, altered_members, plan)
    cases = copy.deepcopy(evaluation["lanes"]["interpolation"]["cases"])
    cases[0] = failed
    summary = module.summarize(cases)
    checks.equal(summary["target_pair_denominator"], 26, "failure injection preserves complete denominator")
    checks.equal(summary["endpoint_denominator"], 52, "failure injection preserves endpoint denominator")
    for strategy in STRATEGIES:
        checks.equal(summary["strategies"][strategy]["target_denominator"], 26, "failure retained in strategy denominator")
        for slot in ("left", "right"):
            checks.equal(summary["strategies"][strategy]["slots"][slot]["denominator"], 26, "failure retained in slot denominator")
    checks.equal(failed["strategies"][STRATEGIES[1]]["status"], "unavailable", "designated donor failure has no fallback")
    checks.equal(failed["strategies"][STRATEGIES[2]]["status"], "partial", "anchor preserves available direct right")
    checks.equal(failed["strategies"][STRATEGIES[2]]["endpoints"]["right"],
                 failed["strategies"][STRATEGIES[0]]["endpoints"]["right"], "failure preserves exact direct right")
    checks.equal(summary["strategies"][STRATEGIES[1]]["slots"]["left"]["available"], 25, "failed geometry left counted")
    checks.equal(summary["strategies"][STRATEGIES[1]]["slots"]["right"]["available"], 25, "failed geometry right counted")
    checks.equal(summary["strategies"][STRATEGIES[2]]["slots"]["left"]["available"], 25, "failed inferred left counted")
    checks.equal(summary["strategies"][STRATEGIES[2]]["slots"]["right"]["available"], 26, "copied right remains counted")
    return {"status": "pass", "excluded_target_left_and_box_perturbation_cases": left_checks,
            "reference_coordinate_perturbation_cases": reference_checks, "target_right_translation_cases": right_checks,
            "single_designated_donor_failure_injection": {
                "case_id": prior["case_id"], "designated_earlier_donor_row": key[1],
                "altered_endpoint_confidence": 0.49, "all_case_denominator_preserved": 26,
                "no_donor_fallback": True, "direct_right_preserved_exactly": True},
            "scope": "Coordinate-field dependency checks only; cached inference still used intact original images."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path,
                        help="Checkout with frozen reference labels and the four original PNGs")
    parser.add_argument("--prediction-jsonl", required=True, type=Path)
    parser.add_argument("--scoring-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new audit path")
    checks = Checks()
    root, repo = args.pilot_root.resolve(), args.repo_root.resolve()
    plan, inputs, cohort, baseline, members, labels, dimensions, source_audit = audit_source_bundle(
        root, repo, args.prediction_jsonl, args.scoring_json, checks)
    derived = {lane: [derive_case(case, members, labels, dimensions) for case in section["cases"]]
               for lane, section in baseline["lanes"].items()}
    hidden = [derive_case(case, members, labels, dimensions) for case in baseline["hidden_cases"]
              if case["disposition"] == "estimate_extrapolation"]
    evaluation, numerical_audit = audit_evaluation(root, plan, inputs, cohort, baseline, members, labels, dimensions,
                                                   derived, hidden, checks)
    dependencies = audit_dependencies(root, plan, cohort, baseline, members, labels, dimensions, evaluation, checks)
    result = {"schema_version": 1, "audit_status": "pass",
              "audit_script_sha256": digest(Path(__file__).read_bytes()),
              "plan_sha256": PLAN_SHA, "model_inputs_sha256": INPUT_SHA,
              "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
              "source_extraction_audit": source_audit,
              "checks": checks.count, "numerical_and_measurement_audit": numerical_audit,
              "dependency_and_denominator_audit": dependencies,
              "limits": [
                  "Manual stack/rank and frozen reference-box associations remain oracle-assisted.",
                  "Withheld coordinate fields do not remove target pixels from cached model inference.",
                  "All 36 included matched predictions have both endpoints available; the single failure injection is a software audit, not an accuracy observation.",
                  "Actual 010:23/24 left errors are unknown. Copying the right prediction cannot certify them.",
                  "Success counts from overlapping interpolation/extrapolation lanes must not be pooled."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit_status": result["audit_status"], "checks": checks.count,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
