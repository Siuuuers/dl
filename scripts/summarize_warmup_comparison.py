#!/usr/bin/env python3
"""Replay and compare M0, a fresh reference control, and lower-bias warmup.

Requires original validation PNGs/labels and small checkpoint provenance bundles,
but no model weights or machine-learning runtime. Existing output is refused.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from evaluate_pose import _source_identity, file_sha256, preflight_dataset
from package_warmup_results import ROLE_RECIPES, check_pair
from pose_checkpoint import _contained_path, _read_json, _require, _same, recipe_identity, select_checkpoint
from summarize_pose_comparison import BASELINE_SHA256, FIELDS, THRESHOLDS, audit_run, dominance, statistics


MODELS = ("m0", "control", "warmup")
COMPARISONS = (("control", "m0"), ("warmup", "m0"), ("warmup", "control"))


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
    _require(preflight["model_sha256"] == BASELINE_SHA256,
             "frozen reference is not the planned epoch173 initialization")
    selected, bundles = {"m0": preflight}, {}
    for role, recipe_id in ROLE_RECIPES.items():
        prefix = f"checkpoints/{role}/"
        manifest_path = input_file(prefix + "checkpoint-manifest.json")
        selected[role] = select_checkpoint(preflight, manifest_path, require_weights=False)
        identity = selected[role]["checkpoint_identity"]
        manifest = _read_json(manifest_path)
        _require(manifest["schema_version"] == 2 and identity["arm"] == "reference"
                 and all(identity.get(key) == value for key, value in recipe_identity(recipe_id).items()),
                 f"{role}: expected a fresh schema-2 reference run with recipe {recipe_id}")
        receipt = _read_json(input_file(prefix + manifest["training_receipt"]["path"]))
        staging = _read_json(input_file(prefix + manifest["staging_manifest"]["path"]))
        for descriptor in (receipt["results"], receipt["log"]):
            input_file(prefix + descriptor["path"])
        bundles[role] = {"receipt": receipt, "staging": staging}
    check_pair(bundles["control"], bundles["warmup"])

    runs = {name: audit_run(name, selected[name], evidence, input_file, code_root) for name in MODELS}
    for name in ("control", "warmup"):
        for key in ("source", "runtime", "images", "inference", "box_confidences", "warmup",
                    "manifest_sha256", "model_card_sha256"):
            _require(_same(runs[name]["header"][key], runs["m0"]["header"][key]),
                     f"{name}: inference differs from M0: {key}")
        for key in ("quality_source", "procedure"):
            _require(_same(runs[name][key], runs["m0"][key]), f"{name}: quality diagnostic differs: {key}")

    comparisons = [{"box_confidence": threshold, "primary": threshold == 0.25,
                    "comparisons": {f"{new}_vs_{old}": dominance(
                        runs[new]["metrics"][threshold], runs[old]["metrics"][threshold])
                        for new, old in COMPARISONS}} for threshold in THRESHOLDS]
    primary = next(row["comparisons"] for row in comparisons if row["primary"])
    baseline_passers = [name for name in ("control", "warmup") if primary[f"{name}_vs_m0"]["passes"]]
    warmup_passes = primary["warmup_vs_m0"]["passes"] and primary["warmup_vs_control"]["passes"]
    eligible = (["control"] if primary["control_vs_m0"]["passes"] else []) + (["warmup"] if warmup_passes else [])

    common = sorted(set.intersection(*(set(run["endpoints"]) for run in runs.values())))
    common_metrics = {name: {field: statistics(run["endpoints"][key][field] for key in common)
                            for field in ("error_px", "dx_px", "dy_px")} for name, run in runs.items()}
    matched = {name: {(image_id, match["gold_index"]) for (image_id, threshold), score in run["scores"].items()
                     if threshold == 0.25 for match in score["matches"]} for name, run in runs.items()}

    def object_ids(identifiers):
        return [{"image_id": image_id, "gold_index": gold_index} for image_id, gold_index in sorted(identifiers)]

    paired = {}
    for new, old in COMPARISONS:
        shared = sorted(set(runs[new]["endpoints"]) & set(runs[old]["endpoints"]))
        changes = [runs[new]["endpoints"][key]["error_px"] - runs[old]["endpoints"][key]["error_px"] for key in shared]
        old_pairs, new_pairs = runs[old]["pair_success"], runs[new]["pair_success"]
        paired[f"{new}_vs_{old}"] = {
            "common_returned_endpoints": len(shared), "error_change_px": statistics(changes),
            "worse_endpoints": sum(value > 0 for value in changes),
            "better_endpoints": sum(value < 0 for value in changes),
            "equal_endpoints": sum(value == 0 for value in changes),
            "returned_only_old": len(set(runs[old]["endpoints"]) - set(shared)),
            "returned_only_new": len(set(runs[new]["endpoints"]) - set(shared)),
            "matched_reference_objects": {"gold_index_is_zero_based": True,
                "old_matched": len(matched[old]), "new_matched": len(matched[new]),
                "matched_both": len(matched[old] & matched[new]),
                "lost": object_ids(matched[old] - matched[new]),
                "gained": object_ids(matched[new] - matched[old])},
            "both_corners_within_5px": {"old_success": len(old_pairs), "new_success": len(new_pairs),
                "success_both": len(old_pairs & new_pairs), "became_failure": len(old_pairs - new_pairs),
                "became_success": len(new_pairs - old_pairs)}}
    per_image = []
    for record in preflight["records"]:
        identifier = record["id"]
        keys = [key for key in common if key[0] == identifier]
        per_image.append({"image_id": identifier, "common_returned_endpoints": len(keys),
            "median_error_px": {name: statistics(run["endpoints"][key]["error_px"] for key in keys)["median"]
                                for name, run in runs.items()},
            "mean_signed_y_error_px": {name: statistics(run["endpoints"][key]["dy_px"] for key in keys)["mean"]
                                       for name, run in runs.items()},
            "both_corners_within_5px": {name: sum(key[0] == identifier for key in run["pair_success"])
                                       for name, run in runs.items()}})

    source = _source_identity(code_root)
    for name in ("summarize_warmup_comparison.py", "summarize_pose_comparison.py",
                 "diagnose_pose_quality.py", "package_warmup_results.py", "package_gpu_results.py"):
        source["files_sha256"][f"scripts/{name}"] = file_sha256(code_root / "scripts" / name)
    for path, digest in input_hashes.items():
        _require(file_sha256(evidence / path) == digest, f"input changed during audit: {path}")
    return {
        "schema_version": 1, "kind": "baomai-bias-warmup-comparison", "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "input_files_sha256": dict(sorted(input_hashes.items())),
        "verification_scope": "Saved-prediction replay, complete-run and matched-recipe evidence consistency; no model weights loaded or verified here.",
        "reference": {key: preflight[key] for key in ("model_sha256", "manifest_sha256", "model_card_sha256",
                                                    "reference_validation_images")},
        "inference": {key: runs["m0"]["header"][key] for key in ("source", "runtime", "inference", "images", "warmup")},
        "models": {name: {"model_sha256": run["header"]["model_sha256"],
            "checkpoint_identity": selected[name].get("checkpoint_identity"),
            "metrics": list(run["metrics"].values()), "quality": list(run["diagnostics"].values())}
            for name, run in runs.items()},
        "decision": "review-eligible-models" if eligible else "retain-M0",
        "baseline_rule_passers": baseline_passers,
        "predeclared_adoption_eligible": eligible,
        "warmup_meets_rule_against_both_controls": warmup_passes,
        "comparison_rule": {"primary_box_confidence": 0.25, "lower_is_better_fields": list(FIELDS),
            "dominance": "All five unrounded metrics no higher, at least one strictly lower; unavailable statistics cannot pass.",
            "required_warmup_comparisons": ["warmup_vs_m0", "warmup_vs_control"],
            "other_thresholds_are_diagnostic": True},
        "comparisons": comparisons,
        "common_returned_endpoints": {"box_confidence": 0.25, "count": len(common), "metrics": common_metrics,
            "signed_error": "prediction minus reference; negative Y is upward in the image", "per_image": per_image},
        "paired_primary_diagnostics": paired,
        "empty_scene_false_detections": {record["id"]: {name: {
            str(threshold): run["scores"][record["id"], threshold]["fp"] for threshold in THRESHOLDS}
            for name, run in runs.items()} for record in preflight["records"] if not record["gold"]},
        "limits": ["Development validation already used for selection; this is not an independent test.",
                   "One training seed; related images and endpoints are not independent trials.",
                   "Pixel radii are illustrative, not acceptance tolerances. Signed residuals do not establish a cause.",
                   "Unlocatable-reference output counts are availability disagreement, not measurable hidden-coordinate errors.",
                   "A matched recipe intervention supports this bounded comparison; it does not establish general causality or repeatability."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="m0/control/warmup, their quality runs, and checkpoints/{control,warmup}")
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
                          "decision": result["decision"],
                          "warmup_meets_rule_against_both_controls": result["warmup_meets_rule_against_both_controls"],
                          "common_returned_endpoints": result["common_returned_endpoints"]["count"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        print(f"Warmup comparison refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
