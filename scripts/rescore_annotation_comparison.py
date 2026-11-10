#!/usr/bin/env python3
"""Audit one fresh annotation-training pair against original and PR30 references.

The evidence layout is m0/m1/m2, their matching -quality directories, and
checkpoints/{reference,candidate}. M0 is freshly inferred epoch173; M1 is the
original-label control; M2 uses the explicitly selected annotation candidate.
All inference must follow completion of both training arms and identify one
clean evaluation revision. This entry point never trains, infers, loads model
weights, rewrites historical evidence, or automatically promotes a model.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

from diagnose_pose_quality import _returned, aggregate_quality
from evaluate_pose import _read_replay, file_sha256, preflight_dataset
from pose_checkpoint import _contained_path, _read_json, _require, _same, recipe_identity, select_checkpoint
from pose_scoring import aggregate_scores
from rescore_reference_amendment import (RADII, THRESHOLDS, compare, describe_image,
                                         load_api, verify_amendment)
from summarize_pose_comparison import COMPARISONS, statistics, summarize as audit_comparison
from train_pose_comparison import SOURCE_FILES


REFERENCE_ID = "baomai-visible-front-2026-11-10"
REFERENCE_MANIFEST_SHA256 = "edf2d01624b96d0a4e652b1fa1fdfbc861c453f825343c840972493a908e6ed2"
MODEL_ROLES = {"m0": "fresh epoch173 baseline", "m1": "fresh original-label control",
               "m2": "fresh annotation-candidate model"}
PRIMARY_CONFIDENCE = 0.25
PAIR_RADII = (5, 10)


def utc_time(value, name):
    """Refuse missing/naive timestamps rather than silently accepting old sweeps."""
    _require(isinstance(value, str), f"{name}: UTC timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name}: invalid ISO timestamp") from error
    _require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
             f"{name}: timestamp must explicitly identify UTC")
    return parsed


def verify_fresh_inference(headers, receipts, evaluation_revision):
    """Bind the new M0/control/candidate sweeps to one post-training evaluation."""
    _require(set(headers) == set(MODEL_ROLES) and set(receipts) == {"reference", "candidate"},
             "fresh comparison requires all three inference runs and both training receipts")
    _require(re.fullmatch(r"[0-9a-f]{40}", evaluation_revision or "") is not None,
             "--evaluation-revision must be a full lowercase 40-hex commit SHA")
    completed = max(utc_time(receipt.get("created_at_utc"), f"{arm} training receipt")
                    for arm, receipt in receipts.items())
    for name, header in headers.items():
        _require("replay" not in header, f"{name}: original fresh inference is required")
        _require(header["source"]["revision"] == evaluation_revision
                 and header["source"]["working_tree_status"] == "",
                 f"{name}: inference does not identify the requested clean evaluation revision")
        _require(_same(header["source"], headers["m0"]["source"]),
                 f"{name}: inference source differs from the fresh M0 sweep")
        _require(utc_time(header.get("created_at_utc"), f"{name} inference") >= completed,
                 f"{name}: inference predates completion of this training pair; run a fresh sweep")
    return {"evaluation_revision": evaluation_revision,
            "both_training_receipts_completed_by_utc": completed.isoformat(),
            "inference_created_at_utc": {name: header["created_at_utc"] for name, header in headers.items()},
            "rule": "Every original inference header is from the same clean revision and is timestamped after both completed training receipts.",
            "clock_limit": "Recorded UTC chronology is checked; this is not independent attestation of machine clocks or execution."}


def returned_endpoints(gold, predictions, score, record):
    result = {}
    for match in score["matches"]:
        index = match["gold_index"]
        predicted = predictions[match["prediction_index"]]["keypoints"]
        for side, (target, point) in enumerate(zip(gold[index]["keypoints"], predicted)):
            if target[2] == 2 and _returned(point, record["width"], record["height"]):
                dx, dy = point[0] - target[0], point[1] - target[1]
                result[record["id"], index, side] = {"error_px": math.hypot(dx, dy),
                                                    "dx_px": dx, "dy_px": dy}
    return result


def common_diagnostics(endpoints, gold_by_image):
    """Compare identical returned reference points within one reference only."""
    common = sorted(set.intersection(*(set(points) for points in endpoints.values())))
    eligible = {(identifier, index) for identifier, gold in gold_by_image.items()
                for index, obj in enumerate(gold) if all(point[2] == 2 for point in obj["keypoints"])}
    successes = {name: {radius: {key for key in eligible if all(
        (*key, side) in points and points[*key, side]["error_px"] <= radius for side in (0, 1))}
        for radius in PAIR_RADII} for name, points in endpoints.items()}
    pairs = {}
    for new, old in COMPARISONS:
        shared = sorted(set(endpoints[new]) & set(endpoints[old]))
        changes = [endpoints[new][key]["error_px"] - endpoints[old][key]["error_px"] for key in shared]
        pairs[f"{new}_vs_{old}"] = {
            "common_returned_endpoints": len(shared), "error_change_px": statistics(changes),
            "worse_endpoints": sum(value > 0 for value in changes),
            "better_endpoints": sum(value < 0 for value in changes),
            "equal_endpoints": sum(value == 0 for value in changes),
            "returned_only_old": len(set(endpoints[old]) - set(shared)),
            "returned_only_new": len(set(endpoints[new]) - set(shared)),
            "both_corners": [{"radius_px": radius, "eligible_objects": len(eligible),
                "old_success": len(successes[old][radius]), "new_success": len(successes[new][radius]),
                "success_both": len(successes[old][radius] & successes[new][radius]),
                "became_failure": len(successes[old][radius] - successes[new][radius]),
                "became_success": len(successes[new][radius] - successes[old][radius])}
                for radius in PAIR_RADII],
        }
    return {"box_confidence": PRIMARY_CONFIDENCE, "count": len(common),
            "metrics": {name: {field: statistics(points[key][field] for key in common)
                for field in ("error_px", "dx_px", "dy_px")} for name, points in endpoints.items()},
            "paired": pairs,
            "per_image": [{"image_id": identifier,
                "common_returned_endpoints": sum(key[0] == identifier for key in common),
                "median_error_px": {name: statistics(points[key]["error_px"] for key in common
                    if key[0] == identifier)["median"] for name, points in endpoints.items()}}
                for identifier in sorted(gold_by_image)],
            "signed_error": "prediction minus reference; negative Y is upward in the original image",
            "identity_scope": "Reference object indices are local to this reference. Original and revised row indices are never equated.",
            "selection_limit": "The common returned subset is conditional; interpret it alongside misses, missing endpoints, and both-corner success with all eligible objects in its denominator."}


def summarize(root, evidence, candidate, recipe, reference_dir, evaluation_revision):
    from pose_checkpoint import ANNOTATION_RECIPE_IDS
    from verify_pose_annotation_candidate import verify_annotation_candidate

    root, evidence, reference_dir = (Path(path).resolve() for path in (root, evidence, reference_dir))
    code_root = Path(__file__).resolve().parents[1]
    candidate_name = Path(candidate).as_posix()
    _require(Path(candidate_name).parts[:2] == ("data", "candidates")
             and not Path(candidate_name).is_absolute(), "--candidate must be repository-relative under data/candidates")
    candidate_path = _contained_path(root, candidate_name)
    _require(recipe in ANNOTATION_RECIPE_IDS, "this entry point requires a registered annotation-overlay recipe")
    proposal = _read_json(candidate_path)
    _require(proposal.get("kind") == "training_annotation_overlay", "expected a training_annotation_overlay candidate")
    candidate_verified = verify_annotation_candidate(root, candidate_path)
    candidate_hash = file_sha256(candidate_path)
    _require(candidate_hash == candidate_verified["candidate_manifest_sha256"], "candidate changed during verification")
    reference_manifest = _contained_path(reference_dir, "manifest.json")
    _require(file_sha256(reference_manifest) == REFERENCE_MANIFEST_SHA256,
             "--reference-dir must contain the frozen PR30 visible-front manifest")
    source_paths = {"scripts/rescore_annotation_comparison.py", "scripts/rescore_reference_amendment.py",
                    "scripts/summarize_pose_comparison.py", "scripts/diagnose_pose_quality.py",
                    "scripts/package_gpu_results.py", *SOURCE_FILES}
    source_snapshot = {name: file_sha256(_contained_path(code_root, name)) for name in source_paths}

    # This established auditor verifies complete original prediction/quality
    # runs, paired final-checkpoint receipts, and exact replay of saved metrics.
    # Its historical adoption rule is deliberately not carried into this report.
    audit = audit_comparison(root, evidence)
    expected_recipe = recipe_identity(recipe)
    receipts, bundles = {}, {}
    for model, arm in (("m1", "reference"), ("m2", "candidate")):
        prefix = f"checkpoints/{arm}/"
        manifest = _read_json(_contained_path(evidence, prefix + "checkpoint-manifest.json"))
        receipt = _read_json(_contained_path(evidence, prefix + manifest["training_receipt"]["path"]))
        staging = _read_json(_contained_path(evidence, prefix + manifest["staging_manifest"]["path"]))
        _require(manifest["schema_version"] == receipt["schema_version"] == 2,
                 f"{model}: schema-2 completed checkpoint evidence is required")
        for value in (manifest, receipt, audit["models"][model]["checkpoint_identity"]):
            _require(all(_same(value.get(key), expected) for key, expected in expected_recipe.items()),
                     f"{model}: checkpoint/receipt recipe differs from --recipe")
        source = staging["source"]
        _require(source["candidate_id"] == proposal["candidate_id"]
                 and source["candidate_manifest"] == candidate_name
                 and source["candidate_manifest_sha256"] == candidate_hash,
                 f"{model}: staged proposal differs from --candidate")
        _require(re.fullmatch(r"cuda:\d+", receipt["runtime"]["device"] or "") is not None,
                 f"{model}: completed paired GPU training receipts are required")
        hashes = receipt["source"]["files_sha256"]
        _require(set(hashes) == set(SOURCE_FILES), f"{model}: training source inventory differs")
        for path, digest in hashes.items():
            _require(source_snapshot[path] == digest, f"{model}: training code differs from the current audit: {path}")
        receipts[arm], bundles[arm] = receipt, {"source": source, "runtime": receipt["runtime"]}

    original = preflight_dataset(root, require_weights=False)
    _, api = load_api(root)
    manifest, revised, original_lines, revised_lines, reference_files = verify_amendment(
        root, reference_dir, original, api)
    _require(manifest["reference_id"] == REFERENCE_ID, "unexpected revised-reference identity")
    records = {row["id"]: row for row in original["records"]}
    gold = {"original": {identifier: row["gold"] for identifier, row in records.items()}, "revised": revised}
    endpoints = {reference: {name: {} for name in MODEL_ROLES} for reference in gold}
    headers, metrics, per_image = {}, [], []
    for model in MODEL_ROLES:
        checkpoint = (evidence / "checkpoints" / ("reference" if model == "m1" else "candidate")
                      / "checkpoint-manifest.json") if model != "m0" else None
        selected = select_checkpoint(original, checkpoint, require_weights=False) if checkpoint else original
        header, items, thresholds = _read_replay(_contained_path(evidence, f"{model}/predictions.jsonl"), selected, None)
        _require(_same(thresholds, THRESHOLDS), f"{model}: fixed thresholds differ")
        headers[model] = header
        for threshold in THRESHOLDS:
            scores, qualities, views = ({name: [] for name in gold} for _ in range(3))
            for item in sorted((item for item in items if item["threshold"] == threshold), key=lambda row: row["image_id"]):
                identifier, predictions = item["image_id"], item["predictions"]
                record = records[identifier]
                pair = {"model_id": model, "box_confidence": threshold, "image_id": identifier}
                for reference, lines, label_path in (
                    ("original", original_lines[identifier], f"data/{record['label']}"),
                    ("revised", revised_lines[identifier], f"labels/val/{identifier}.txt"),
                ):
                    objects = gold[reference][identifier]
                    public, score, quality = describe_image(objects, predictions, record, lines, label_path, api)
                    pair[reference] = public
                    scores[reference].append(score)
                    qualities[reference].append(quality)
                    views[reference].append(public)
                    if threshold == PRIMARY_CONFIDENCE:
                        endpoints[reference][model].update(returned_endpoints(objects, predictions, score, record))
                pair["revised_minus_original"] = compare(pair["original"], pair["revised"])
                per_image.append(pair)
            row = {"model_id": model, "box_confidence": threshold, "primary": threshold == PRIMARY_CONFIDENCE}
            for reference in gold:
                score = aggregate_scores(scores[reference])
                score.pop("endpoint_errors_px")
                row[reference] = {"score": score, "quality": aggregate_quality(qualities[reference]),
                    **{field: {str(n): sum(view[field][str(n)] for view in views[reference]) for n in range(3)}
                       for field in ("objects_by_locatable_corners", "misses_by_locatable_corners")}}
            row["revised_minus_original"] = compare(row["original"], row["revised"])
            metrics.append(row)
    freshness = verify_fresh_inference(headers, receipts, evaluation_revision)
    indexed = {(row["model_id"], row["box_confidence"]): row for row in metrics}
    comparisons = [{"reference": reference, "box_confidence": threshold,
                    "primary": threshold == PRIMARY_CONFIDENCE,
                    "comparisons": {f"{new}_vs_{old}": compare(indexed[old, threshold][reference], indexed[new, threshold][reference])
                                    for new, old in COMPARISONS}}
                   for reference in gold for threshold in THRESHOLDS]

    for path, digest in audit["input_files_sha256"].items():
        _require(file_sha256(_contained_path(evidence, path)) == digest, f"evidence changed during audit: {path}")
    _require(file_sha256(candidate_path) == candidate_hash, "candidate manifest changed during audit")
    for change in proposal["changes"]:
        _require(file_sha256(_contained_path(candidate_path.parent, change["candidate_label"]))
                 == change["candidate_label_sha256"], f"candidate label changed during audit: {change['image_id']}")
    _require(file_sha256(reference_manifest) == REFERENCE_MANIFEST_SHA256, "revised manifest changed during audit")
    for entry in reference_files:
        _require(file_sha256(_contained_path(reference_dir, entry["label_path"])) == entry["label_sha256"],
                 f"revised label changed during audit: {entry['image_id']}")
    for path, digest in source_snapshot.items():
        _require(file_sha256(_contained_path(code_root, path)) == digest, f"source changed during audit: {path}")
    return {
        "schema_version": 1, "kind": "baomai-annotation-comparison-dual-reference", "status": "verified-descriptive-only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {**audit["source"], "files_sha256": dict(sorted(source_snapshot.items()))},
        "model_roles": MODEL_ROLES, "input_files_sha256": audit["input_files_sha256"],
        "experiment": {**expected_recipe, "candidate_manifest": candidate_name, "candidate_manifest_sha256": candidate_hash,
                       "candidate_verification": candidate_verified, "training_source": receipts["reference"]["source"],
                       "training_runtime": bundles["reference"]["runtime"], "fresh_inference": freshness},
        "original_reference": audit["reference"],
        "revised_reference": {"reference_id": REFERENCE_ID, "manifest_sha256": REFERENCE_MANIFEST_SHA256,
                              "root": str(reference_dir), "files": reference_files},
        "inference": audit["inference"],
        "models": {name: {"model_sha256": header["model_sha256"],
                          "checkpoint_identity": header.get("checkpoint_identity")} for name, header in headers.items()},
        "metrics": metrics, "comparisons": comparisons, "per_image": per_image,
        "common_returned_endpoints": {reference: common_diagnostics(points, gold[reference])
                                      for reference, points in endpoints.items()},
        "decision": "review-required-no-promotion", "automatic_promotion": False,
        "procedure": {"primary_box_confidence": PRIMARY_CONFIDENCE, "box_confidences": THRESHOLDS,
            "matching_iou": 0.50, "endpoint_confidence": 0.50, "radii_px": RADII,
            "priority": "missed objects (fn), assessed against both fresh original-label control and M0",
            "required_companion_metrics": ["fp", "missing_endpoints", "both-corner success at 5 and 10 original pixels",
                "returned_on_unlocatable_reference", "common-returned-endpoint diagnostics"],
            "false_positive_tradeoff": "Unspecified; report the complete tradeoff without selecting or promoting a winner.",
            "other_thresholds_are_diagnostic": True,
            "both_corner_denominator": "All objects with both reference corners locatable, including missed objects; recomputed separately for each reference.",
            "reference_delta": "revised minus original for identical predictions; this is a reference effect, not a training improvement",
            "model_delta": "new minus old within the same reference and threshold",
            "row_numbering": "One-based physical lines within the named reference; revised indices are never treated as original identities.",
            "original_label_path_base": str(root), "revised_label_path_base": str(reference_dir),
            "zero_denominator": "null; deltas involving null are omitted",
            "new_training_or_inference": False, "weights_loaded_or_verified": False},
        "verification_scope": "Complete saved inference and quality runs, paired schema-2 GPU receipts, candidate/recipe binding, original/PR30 hashes, and current source hashes. No model weights are loaded or verified here.",
        "limits": ["One paired training run on repeatedly used development validation is not independent-test evidence.",
            "Related frames and endpoints are not independent trials; this report makes no significance or generalization claim.",
            "Ten pixels is a provisional original-image diagnostic radius, not annotation slack, a physical tolerance, or a deployment acceptance gate.",
            "Outputs on unlocatable reference sides measure availability disagreement; no hidden coordinate error is observable.",
            "Original/revised reference effects must remain distinct from model effects and must not overwrite historical results."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="repository with hydrated original data")
    parser.add_argument("--evidence", type=Path, required=True, help="m0/m1/m2, matching -quality runs, checkpoints/{reference,candidate}")
    parser.add_argument("--candidate", required=True, help="repository-relative annotation-candidate manifest under data/candidates")
    parser.add_argument("--recipe", required=True, help="explicit registered annotation-training recipe ID")
    parser.add_argument("--reference-dir", type=Path, required=True, help="frozen PR30 reference directory")
    parser.add_argument("--evaluation-revision", required=True, help="clean commit used for all three fresh inference/quality sweeps")
    parser.add_argument("--output", type=Path, required=True, help="new comparison JSON; existing output is refused")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        result = summarize(args.root, args.evidence, args.candidate, args.recipe, args.reference_dir, args.evaluation_revision)
        result["command"] = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print(json.dumps({"output": str(args.output.resolve()), "status": result["status"], "decision": result["decision"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError, ImportError) as error:
        print(f"Annotation comparison refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
