#!/usr/bin/env python3
"""Replay the fixed box-supervision comparison using the existing audited scorer.

Evidence uses m0/m1/m2 and their matching -quality directories, with checkpoint
bundles under checkpoints/{reference,candidate}. M1 is the fresh original-label
BOX_RECIPE_ID control, and M2 is the corrected-box arm of that same recipe.
No training, inference, or model deserialization occurs here.
"""

import argparse
import json
from pathlib import Path
import sys

from evaluate_pose import file_sha256
from pose_checkpoint import BOX_RECIPE_ID, _contained_path, _read_json, _require, _same, recipe_identity
from summarize_pose_comparison import summarize as audit_comparison
from train_pose_comparison import SOURCE_FILES


CANDIDATE_ID = "box-refinement-2026-11-07"
CANDIDATE_MANIFEST = f"data/candidates/{CANDIDATE_ID}/manifest.json"
MODELS = {"m0": "epoch173 baseline", "m1": "fresh original-label box-recipe reference",
          "m2": "corrected-box box-recipe candidate"}


def summarize(root, evidence):
    """Keep historical replay/scoring unchanged; add this experiment's strict gate."""
    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    result = audit_comparison(root, evidence)
    code_root = Path(__file__).resolve().parents[1]
    expected_recipe = recipe_identity(BOX_RECIPE_ID)

    def audited_json(relative):
        path = _contained_path(evidence, relative)
        _require(result["input_files_sha256"].get(relative) == file_sha256(path),
                 f"box comparison input changed after replay: {relative}")
        return _read_json(path)

    bundles = {}
    for model, arm in (("m1", "reference"), ("m2", "candidate")):
        identity = result["models"][model]["checkpoint_identity"]
        _require(identity["arm"] == arm and all(_same(identity.get(key), value)
                 for key, value in expected_recipe.items()),
                 f"{model}: expected {arm} arm with explicit recipe {BOX_RECIPE_ID}")
        prefix = f"checkpoints/{arm}/"
        manifest = audited_json(prefix + "checkpoint-manifest.json")
        receipt = audited_json(prefix + manifest["training_receipt"]["path"])
        staging = audited_json(prefix + manifest["staging_manifest"]["path"])
        _require(manifest["schema_version"] == receipt["schema_version"] == 2,
                 f"{model}: box comparison requires schema-2 checkpoint evidence")
        proposal = staging["source"]
        _require(proposal["candidate_id"] == CANDIDATE_ID
                 and proposal["candidate_manifest"] == CANDIDATE_MANIFEST
                 and proposal["candidate_manifest_sha256"] == file_sha256(root / CANDIDATE_MANIFEST),
                 f"{model}: expected the predeclared {CANDIDATE_ID} proposal")
        hashes = receipt["source"]["files_sha256"]
        _require(set(hashes) == set(SOURCE_FILES), f"{model}: training source inventory differs")
        for path, digest in hashes.items():
            _require(file_sha256(_contained_path(code_root, path)) == digest,
                     f"{model}: recorded training code differs from the audit implementation: {path}")
        bundles[arm] = {"source": receipt["source"], "runtime": receipt["runtime"],
                        "proposal": proposal}

    # The shared audit already enforces both complete runs, matched training
    # settings/source/runtime, identical inference, and exact score/quality replay.
    # Its legacy baseline-only eligibility list is descriptive here: M1 is a
    # control, and M2 must pass against *both* M0 and M1 to be review-eligible.
    candidate_passes = result["candidate_meets_rule_against_both_controls"]
    result.update(
        kind="baomai-box-supervision-comparison",
        model_roles=MODELS,
        experiment={**expected_recipe, "candidate": bundles["candidate"]["proposal"],
                    "training_source": bundles["reference"]["source"],
                    "training_runtime": bundles["reference"]["runtime"]},
        baseline_rule_passers=result.pop("baseline_replacement_eligible"),
        predeclared_adoption_eligible=["m2"] if candidate_passes else [],
        decision="review-box-candidate" if candidate_passes else "retain-M0",
        verification_scope=("Saved-prediction replay, complete matched box-recipe checkpoint evidence, "
                            "and current training-script hash consistency; no model weights loaded or verified here."),
    )
    result["comparison_rule"].update(
        required_candidate_comparisons=["m2_vs_m0", "m2_vs_m1"],
        reference_arm_is_control_only=True,
    )
    result["limits"].append("The fresh reference is a control for this box intervention; its baseline-rule pass does not authorize its adoption.")
    for name in ("scripts/summarize_box_comparison.py", *SOURCE_FILES):
        result["source"]["files_sha256"][name] = file_sha256(code_root / name)
    for path, digest in result["input_files_sha256"].items():
        _require(file_sha256(_contained_path(evidence, path)) == digest,
                 f"box comparison input changed during audit: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="frozen validation repository")
    parser.add_argument("--evidence", type=Path, required=True,
                        help="m0/m1/m2, their quality runs, checkpoints/{reference,candidate}")
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; existing output is refused")
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
                          "candidate_meets_rule_against_both_controls": result["candidate_meets_rule_against_both_controls"],
                          "common_returned_endpoints": result["common_returned_endpoints"]["count"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        print(f"Box comparison refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
