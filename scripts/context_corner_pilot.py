#!/usr/bin/env python3
"""Fixed, context-only corner geometry diagnostic; never edits source labels."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import struct
import sys

import numpy as np


METHODS = ("linear", "homography")
RADII = (5, 10)


class PilotError(ValueError):
    """Invalid input or unsupported geometric operation."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise PilotError(f"Path leaves its input root: {relative}")
    return path


def _validate_donors(earlier_xy, later_xy, t, allow_extrapolation=False):
    earlier = np.asarray(earlier_xy, dtype=float)
    later = np.asarray(later_xy, dtype=float)
    if earlier.shape != (2, 2) or later.shape != (2, 2):
        raise PilotError("Donor endpoints must each have shape (2, 2)")
    if not np.isfinite(earlier).all() or not np.isfinite(later).all():
        raise PilotError("Non-finite donor coordinates")
    if not np.isfinite(t):
        raise PilotError("Non-finite rank ratio")
    if not (0 < t < 1 or (allow_extrapolation and 1 < t <= 3)):
        raise PilotError("Rank ratio requires bracketing, or explicit 1 < t <= 3 extrapolation")
    if not (earlier[0, 0] < earlier[1, 0] and later[0, 0] < later[1, 0]):
        raise PilotError("Donor left/right image-X order is invalid")
    quad = np.array([earlier[0], earlier[1], later[1], later[0]])
    edges = np.roll(quad, -1, axis=0) - quad
    following = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    tolerance = 1e-10 * max(1.0, float(np.ptp(quad, axis=0).max()) ** 2)
    if not (np.all(crosses > tolerance) or np.all(crosses < -tolerance)):
        raise PilotError("Donor quadrilateral is non-convex or degenerate")
    return earlier, later, quad


def _normalize_points(points):
    center = points.mean(axis=0)
    mean_distance = float(np.linalg.norm(points - center, axis=1).mean())
    if mean_distance <= 1e-12:
        raise PilotError("Degenerate normalization geometry")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array([[scale, 0, -scale * center[0]],
                          [0, scale, -scale * center[1]], [0, 0, 1]])
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (transform @ homogeneous.T).T[:, :2], transform


def _homography(source, destination):
    src, src_transform = _normalize_points(source)
    dst, dst_transform = _normalize_points(destination)
    design = []
    for (x, y), (u, v) in zip(src, dst):
        design.extend([[-x, -y, -1, 0, 0, 0, u * x, u * y, u],
                       [0, 0, 0, -x, -y, -1, v * x, v * y, v]])
    _, singular_values, vh = np.linalg.svd(np.asarray(design), full_matrices=True)
    if singular_values[-1] <= 1e-12 * singular_values[0]:
        raise PilotError("Rank-deficient four-point homography")
    normalized = vh[-1].reshape(3, 3)
    result = np.linalg.inv(dst_transform) @ normalized @ src_transform
    result /= np.linalg.norm(result)
    if not np.isfinite(result).all():
        raise PilotError("Non-finite homography")
    return result


def estimate_pair(method, earlier_xy, later_xy, t, *, allow_extrapolation=False):
    """Accept only two donors and rank ratio; target coordinates/bbox cannot enter."""
    earlier, later, quad = _validate_donors(earlier_xy, later_xy, t, allow_extrapolation)
    if method == "linear":
        result = (1 - t) * earlier + t * later
    elif method == "homography":
        source = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        homography = _homography(source, quad)
        # Reject a projective horizon anywhere between the donors and the target.
        extent = max(1., float(t))
        domain = np.array([[0, 0, 1], [1, 0, 1], [0, extent, 1], [1, extent, 1]])
        weights = (homography @ domain.T)[2]
        if not (np.all(weights > 1e-12) or np.all(weights < -1e-12)):
            raise PilotError("Projective horizon crosses donor-to-target domain")
        projected = (homography @ np.array([[0, t, 1], [1, t, 1]]).T).T
        result = projected[:, :2] / projected[:, 2, None]
    else:
        raise PilotError(f"Unknown method: {method}")
    if not np.isfinite(result).all() or result[0, 0] >= result[1, 0]:
        raise PilotError("Estimated endpoint pair is non-finite or reverses image-X order")
    return result


def load_reference(repo_root, cohort):
    """Bind selected images and labels to the exact cohort-pinned reference bytes."""
    repo_root = Path(repo_root).resolve()
    manifest_path = _safe_path(repo_root, cohort["reference_manifest"])
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes) != cohort["reference_manifest_sha256"]:
        raise PilotError("Reference manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    entries = {entry["image_id"]: entry for entry in manifest["files"]}
    needed = {stack["image_id"] for stack in cohort["stacks"]}
    needed.update(case["image_id"] for case in cohort.get("hidden_cases", []))
    loaded, identities = {}, []
    for image_id in sorted(needed):
        if image_id not in entries:
            raise PilotError(f"Image absent from reference: {image_id}")
        entry = entries[image_id]
        image_path = _safe_path(repo_root, f"data/images/val/{image_id}.png")
        image_bytes = image_path.read_bytes()
        if sha256(image_bytes) != entry["image_sha256"]:
            raise PilotError(f"Original image hash mismatch: {image_id}")
        if len(image_bytes) < 24 or image_bytes[:8] != b"\x89PNG\r\n\x1a\n" or image_bytes[12:16] != b"IHDR":
            raise PilotError(f"Original is not a PNG: {image_id}")
        width, height = struct.unpack(">II", image_bytes[16:24])
        if (width, height) != (entry["width"], entry["height"]):
            raise PilotError(f"Original dimensions mismatch: {image_id}")
        label_path = _safe_path(manifest_path.parent, entry["label_path"])
        label_bytes = label_path.read_bytes()
        if sha256(label_bytes) != entry["label_sha256"]:
            raise PilotError(f"Reference label hash mismatch: {image_id}")
        rows = {}
        for number, line in enumerate(label_bytes.decode("utf-8").splitlines(), 1):
            values = np.asarray([float(value) for value in line.split()])
            if len(values) != 11 or not np.isfinite(values).all() or values[0] != 0:
                raise PilotError(f"Invalid reference row: {image_id}:{number}")
            visibility = values[[7, 10]]
            if not np.isin(visibility, [0, 2]).all():
                raise PilotError(f"Unsupported visibility: {image_id}:{number}")
            points = np.array([[values[5] * width, values[6] * height],
                               [values[8] * width, values[9] * height]])
            if not ((visibility != 0) | np.all(points == 0, axis=1)).all():
                raise PilotError(f"Noncanonical missing point: {image_id}:{number}")
            rows[number] = {"xy": points, "visible": visibility == 2}
        loaded[image_id] = {"rows": rows, "width": width, "height": height}
        identities.append({"image_id": image_id, "image_path": str(image_path.relative_to(repo_root)),
                           "image_sha256": sha256(image_bytes), "image_bytes": len(image_bytes),
                           "dimensions": [width, height],
                           "label_path": str(label_path.relative_to(repo_root)),
                           "label_sha256": sha256(label_bytes), "label_bytes": len(label_bytes)})
    return loaded, identities


def _validate_stack(stack, data):
    members = stack["members"]
    rows = [member["row"] for member in members]
    ranks = [member["rank"] for member in members]
    if any(type(value) is not int for value in rows + ranks):
        raise PilotError("Stack rows and ranks must be integers")
    if len(set(rows)) != len(rows) or len(set(ranks)) != len(ranks) or min(ranks) < 0:
        raise PilotError(f"Duplicate or invalid stack members: {stack['id']}")
    if any(row not in data["rows"] for row in rows):
        raise PilotError(f"Stack row absent from reference: {stack['id']}")
    for key in ("holdout_rows", "extrapolation_holdout_rows"):
        targets = stack.get(key, [])
        if len(set(targets)) != len(targets) or any(row not in rows for row in targets):
            raise PilotError(f"Invalid {key}: {stack['id']}")
        if any(not data["rows"][row]["visible"].all() for row in targets):
            raise PilotError(f"Holdout does not have two observed endpoints: {stack['id']}")
    return {member["row"]: member["rank"] for member in members}


def select_donors(stack, data, target_rank, excluded_rows, *, extrapolation=False, first_rank=None):
    # Exclude whole target objects before inspecting any coordinates or visibility.
    eligible = [member for member in stack["members"] if member["row"] not in excluded_rows]
    eligible = [member for member in eligible if data["rows"][member["row"]]["visible"].all()]
    eligible.sort(key=lambda member: member["rank"])
    if extrapolation:
        candidates = [member for member in eligible if member["rank"] < first_rank]
        if len(candidates) < 2:
            raise PilotError("Fewer than two fully labelled donors above the excluded targets")
        selected = candidates[-2:]
    else:
        above = [member for member in eligible if member["rank"] < target_rank]
        below = [member for member in eligible if member["rank"] > target_rank]
        if not above or not below:
            raise PilotError("No fully labelled bracketing donors")
        selected = [above[-1], below[0]]
    donors = [{"row": member["row"], "rank": member["rank"],
               "xy": data["rows"][member["row"]]["xy"].tolist()} for member in selected]
    t = (target_rank - donors[0]["rank"]) / (donors[1]["rank"] - donors[0]["rank"])
    return donors, float(t)


def _estimate_methods(donors, t, dimensions, extrapolation=False, donor_failure=None):
    results = {}
    width, height = dimensions
    for method in METHODS:
        if donor_failure:
            results[method] = {"status": "abstained", "reason": donor_failure, "raw_pair_xy": None}
            continue
        try:
            pair = estimate_pair(method, donors[0]["xy"], donors[1]["xy"], t,
                                 allow_extrapolation=extrapolation)
            in_frame = bool(np.all((pair[:, 0] >= 0) & (pair[:, 0] < width)
                                   & (pair[:, 1] >= 0) & (pair[:, 1] < height)))
            results[method] = {"status": "estimated" if in_frame else "rejected",
                               "reason": None if in_frame else "Estimated point outside image; no clipping",
                               "raw_pair_xy": pair.tolist(), "in_frame": in_frame}
        except (PilotError, np.linalg.LinAlgError) as error:
            results[method] = {"status": "abstained", "reason": str(error), "raw_pair_xy": None}
    return results


def _disagreement(methods):
    left, right = [methods[method]["raw_pair_xy"] for method in METHODS]
    if left is None or right is None:
        return None
    distances = np.linalg.norm(np.asarray(left) - np.asarray(right), axis=1)
    return {"endpoint_distances_px": distances.tolist(), "maximum_px": float(distances.max()),
            "interpretation": "Uncalibrated method disagreement; not an error bound"}


def evaluate_holdout(stack, data, row, ranks, excluded, extrapolation=False):
    case = {"case_id": f"{stack['image_id']}:{row}", "stack_id": stack["id"],
            "image_id": stack["image_id"], "row": row, "rank": ranks[row],
            "excluded_rows": sorted(excluded), "target_bbox_used": False,
            "target_endpoints_used_for_prediction": False}
    failure = None
    try:
        donors, t = select_donors(stack, data, ranks[row], excluded, extrapolation=extrapolation,
                                  first_rank=min(ranks[target] for target in excluded))
    except PilotError as error:
        donors, t, failure = [], None, str(error)
    methods = _estimate_methods(donors, t, (data["width"], data["height"]), extrapolation, failure)
    # Read withheld target endpoints only after every estimate is complete.
    target = data["rows"][row]["xy"]
    for result in methods.values():
        if result["raw_pair_xy"] is None:
            result.update(endpoint_errors_px=None, maximum_endpoint_error_px=None)
        else:
            distances = np.linalg.norm(np.asarray(result["raw_pair_xy"]) - target, axis=1)
            result.update(endpoint_errors_px=distances.tolist(), maximum_endpoint_error_px=float(distances.max()))
        result["both_within_radius"] = {str(radius): result["status"] == "estimated"
                                        and result["maximum_endpoint_error_px"] <= radius for radius in RADII}
    case.update(donors=donors, rank_ratio=t, target_xy_for_scoring_only=target.tolist(),
                methods=methods, method_disagreement=_disagreement(methods))
    return case


def _error_stats(values):
    if not values:
        return {"count": 0, "median_px": None, "p90_px": None, "maximum_px": None}
    a = np.asarray(values)
    return {"count": len(values), "median_px": float(np.median(a)),
            "p90_px": float(np.percentile(a, 90)), "maximum_px": float(np.max(a))}


def summarize(cases):
    result = {"target_pair_denominator": len(cases), "endpoint_denominator": 2 * len(cases), "methods": {}}
    for method in METHODS:
        records = [case["methods"][method] for case in cases]
        errors = [value for record in records for value in (record["endpoint_errors_px"] or [])]
        pair_errors = [record["maximum_endpoint_error_px"] for record in records
                       if record["maximum_endpoint_error_px"] is not None]
        result["methods"][method] = {
            "estimated_pairs": sum(record["status"] == "estimated" for record in records),
            "failed_pairs": sum(record["status"] != "estimated" for record in records),
            "failure_reasons": dict(sorted(Counter(record["reason"] for record in records
                                                    if record["status"] != "estimated").items())),
            "both_within_radius": {str(radius): sum(record["both_within_radius"][str(radius)] for record in records)
                                   for radius in RADII},
            "endpoints_within_radius": {str(radius): sum(value <= radius for record in records
                                                          if record["status"] == "estimated"
                                                          for value in record["endpoint_errors_px"])
                                        for radius in RADII},
            "endpoint_error": _error_stats(errors), "pair_maximum_error": _error_stats(pair_errors),
            "error_statistic_basis": "All finite raw estimates, including rejected out-of-frame outputs; failures remain in success denominators."}
    disagreement = [case["method_disagreement"]["maximum_px"] for case in cases if case["method_disagreement"]]
    result["maximum_pair_method_disagreement"] = _error_stats(disagreement)
    return result


def evaluate_hidden(case, stacks, loaded):
    image_id, row = case["image_id"], case["row"]
    data = loaded[image_id]
    if row not in data["rows"]:
        raise PilotError(f"Hidden-case row absent: {image_id}:{row}")
    record = {**case, "case_id": f"{image_id}:{row}", "kind": "context_estimate_sidecar",
              "accepted_for_training": False, "accepted_as_observed_reference": False,
              "target_bbox_used": False, "target_endpoints_used_for_prediction": False,
              "confidence": None, "confidence_calibrated": False}
    if case["disposition"] == "abstain":
        record.update(status="abstained", donors=[], methods={}, method_disagreement=None)
        return record
    if case["disposition"] not in ("estimate", "estimate_extrapolation"):
        raise PilotError("Unknown hidden-case disposition")
    stack = stacks[case["stack_id"]]
    if stack["image_id"] != image_id:
        raise PilotError("Hidden case and donor stack have different images")
    member_ranks = {member["row"]: member["rank"] for member in stack["members"]}
    if type(case.get("rank")) is not int or member_ranks.get(row) != case["rank"]:
        raise PilotError("Hidden-case row/rank does not match its declared stack member")
    extrapolation = case["disposition"] == "estimate_extrapolation"
    # All hidden rows have at least one absent endpoint, so they cannot become donors.
    excluded = {row}
    failure = None
    try:
        donors, t = select_donors(stack, data, case["rank"], excluded,
                                  extrapolation=extrapolation, first_rank=case["rank"])
    except PilotError as error:
        donors, t, failure = [], None, str(error)
    methods = _estimate_methods(donors, t, (data["width"], data["height"]), extrapolation, failure)
    target = data["rows"][row]
    if target["visible"].all():
        raise PilotError(f"Hidden case has no missing endpoint: {image_id}:{row}")
    for result in methods.values():
        result["proposals"] = []
        result["visible_endpoint_checks"] = []
        if result["raw_pair_xy"] is not None:
            for slot, name in enumerate(("left", "right")):
                point = result["raw_pair_xy"][slot]
                if target["visible"][slot]:
                    result["visible_endpoint_checks"].append({"slot": name,
                        "reference_xy": target["xy"][slot].tolist(),
                        "error_px": float(np.linalg.norm(np.asarray(point) - target["xy"][slot])),
                        "used_in_predictor": False, "certifies_hidden_side_accuracy": False})
                else:
                    result["proposals"].append({"slot": name, "xy": point,
                        "status": "estimated_from_context" if result["status"] == "estimated" else "rejected",
                        "accepted_for_training": False, "ground_truth_available": False,
                        "geometric_error_px": None})
    record.update(status="review_only", donors=donors, rank_ratio=t, methods=methods,
                  method_disagreement=_disagreement(methods))
    return record


def run_pilot(repo_root, cohort_path):
    cohort_path = Path(cohort_path).resolve()
    cohort_bytes = cohort_path.read_bytes()
    cohort = json.loads(cohort_bytes)
    if cohort.get("schema_version") != 1:
        raise PilotError("Unsupported cohort schema")
    loaded, identities = load_reference(repo_root, cohort)
    stacks = {stack["id"]: stack for stack in cohort["stacks"]}
    if len(stacks) != len(cohort["stacks"]):
        raise PilotError("Duplicate stack ID")
    lanes = {"interpolation": [], "extrapolation": []}
    for stack in cohort["stacks"]:
        data = loaded[stack["image_id"]]
        ranks = _validate_stack(stack, data)
        for row in stack.get("holdout_rows", []):
            lanes["interpolation"].append(evaluate_holdout(stack, data, row, ranks, {row}))
        excluded = set(stack.get("extrapolation_holdout_rows", []))
        for row in stack.get("extrapolation_holdout_rows", []):
            lanes["extrapolation"].append(evaluate_holdout(stack, data, row, ranks, excluded, True))
    lane_results = {}
    for lane, cases in lanes.items():
        if len({case["case_id"] for case in cases}) != len(cases):
            raise PilotError(f"Duplicate target identity in {lane}")
        lane_results[lane] = {"cases": cases, "summary": summarize(cases),
                              "per_stack": {stack_id: summarize([case for case in cases if case["stack_id"] == stack_id])
                                            for stack_id in stacks}}
    return {"schema_version": 1, "experiment_id": cohort.get("experiment_id"), "status": "complete",
            "coordinate_system": "Original image pixels; no clipping, resizing or model predictions",
            "purpose": "Contextual geometry diagnostic; source observations and labels remain unchanged",
            "methods": list(METHODS), "radii_px": list(RADII),
            "assumptions": ["Manually supplied stack membership and layer ranks are correct",
                            "Rank differences represent equal physical layer pitch",
                            "Homography additionally assumes equal width and one local planar projective surface"],
            "limits": ["Interpolation and extrapolation are separate; their cases overlap and must not be pooled",
                       "Errors concern visible withheld endpoints, not accuracy of hidden corners",
                       "No calibrated confidence or uncertainty interval; disagreement is diagnostic only",
                       "Development images and adjacent stack rows are not independent test observations",
                       "No inferred point is approved for training or as an observed reference"],
            "provenance": {"cohort": {"filename": cohort_path.name, "sha256": sha256(cohort_bytes), "bytes": len(cohort_bytes)},
                           "script_sha256": sha256(Path(__file__).read_bytes()),
                           "reference_manifest": cohort["reference_manifest"],
                           "reference_manifest_sha256": cohort["reference_manifest_sha256"],
                           "source_commit": cohort.get("source_commit"),
                           "source_files": identities,
                           "runtime": {"python": platform.python_version(), "numpy": np.__version__}},
            "lanes": lane_results,
            "hidden_cases": [evaluate_hidden(case, stacks, loaded) for case in cohort.get("hidden_cases", [])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSON output file; existing files are refused")
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.is_relative_to(args.repo_root.resolve()):
            raise PilotError("Output must be outside the source repository")
        if output.exists():
            raise PilotError("Output already exists")
        result = run_pilot(args.repo_root, args.cohort)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "complete", "output": str(output),
                          "lane_pair_counts": {name: lane["summary"]["target_pair_denominator"]
                                               for name, lane in result["lanes"].items()}}))
    except (PilotError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Context pilot failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
