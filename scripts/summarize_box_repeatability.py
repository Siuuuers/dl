#!/usr/bin/env python3
"""Audit the two fixed prospective box-supervision seeds without model promotion.

Use --evidence DIRECTORY containing seed-43/ and seed-44/. Each seed directory
uses the existing m0/m1/m2, matching -quality, and checkpoints/{reference,candidate}
layout. Saved predictions are replayed by the unchanged three-model auditor.
Historical seed 42 is linked as context only and never enters these summaries.
No training, inference, model loading, or selection of a winning seed occurs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from evaluate_pose import file_sha256
from package_box_repeatability import (CANDIDATE_ID, CANDIDATE_MANIFEST, CANDIDATE_SHA256,
                                      SEEDS, STUDY_ID, check_study_arm, study_settings)
from pose_checkpoint import (BOX_REPEATABILITY_RECIPES, _contained_path, _read_json,
                             _require, _same, recipe_identity)
from summarize_pose_comparison import FIELDS, dominance, summarize as audit_comparison
from train_pose_comparison import SOURCE_FILES


HISTORICAL_REPORT_URL = (
    "https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/"
    "docs/box-supervision-comparison-2026-11-07.md"
)
TIMING_FIELDS = {"stage_ms", "predict_call_wall_ms"}


def _payload_digest(path):
    """Hash every M0 prediction field except the two recorded timing fields."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    _require(len(rows) == 61 and rows[0]["record_type"] == "run",
             "M0 needs its original header and 60 prediction records")
    payload = [{key: value for key, value in row.items() if key not in TIMING_FIELDS} for row in rows[1:]]
    keys = [(row["image_id"], row["threshold"]) for row in payload]
    _require(len(set(keys)) == 60, "M0 contains duplicate image/threshold records")
    payload.sort(key=lambda row: (row["image_id"], row["threshold"]))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _audit_seed(root, evidence, seed):
    """Reuse the scorer, then bind its audit to this seed and frozen box proposal."""
    result = audit_comparison(root, evidence)
    code_root = Path(__file__).resolve().parents[1]
    expected_recipe = recipe_identity(BOX_REPEATABILITY_RECIPES[seed])

    def audited_json(relative):
        path = _contained_path(evidence, relative)
        _require(result["input_files_sha256"].get(relative) == file_sha256(path),
                 f"seed {seed}: input changed after replay: {relative}")
        return _read_json(path)

    bundles = {}
    for model, arm in (("m1", "reference"), ("m2", "candidate")):
        identity = result["models"][model]["checkpoint_identity"]
        _require(identity["arm"] == arm and all(_same(identity.get(key), value)
                 for key, value in expected_recipe.items()),
                 f"seed {seed}: {model} must identify {arm} recipe {expected_recipe['recipe_id']}")
        prefix = f"checkpoints/{arm}/"
        manifest = audited_json(prefix + "checkpoint-manifest.json")
        receipt = audited_json(prefix + manifest["training_receipt"]["path"])
        staging = audited_json(prefix + manifest["staging_manifest"]["path"])
        _require(manifest["schema_version"] == receipt["schema_version"] == 2,
                 f"seed {seed}: schema-2 checkpoint evidence is required")
        check_study_arm({"identity": identity, "receipt": receipt, "staging": staging}, seed)
        for record in (manifest, receipt):
            _require(all(_same(record.get(key), value) for key, value in expected_recipe.items()),
                     f"seed {seed}: checkpoint/receipt recipe identity differs")
        proposal = staging["source"]
        hashes = receipt["source"]["files_sha256"]
        _require(set(hashes) == set(SOURCE_FILES), f"seed {seed}: training source inventory differs")
        for path, digest in hashes.items():
            _require(file_sha256(_contained_path(code_root, path)) == digest,
                     f"seed {seed}: recorded training code differs from audit implementation: {path}")
        settings = study_settings(receipt)
        bundles[arm] = {"source": receipt["source"], "runtime": receipt["runtime"],
                        "proposal": proposal, "nonseed_training_settings": settings}

    # The underlying audit checks both completed runs and their matched settings,
    # source/runtime, staging, inference, scores, and quality diagnostics.
    _require(_same(bundles["reference"], bundles["candidate"]),
             f"seed {seed}: reference/candidate provenance differs")
    quality = audited_json("m0-quality/quality-summary.json")
    primary = {}
    for model in ("m0", "m1", "m2"):
        rows = [row for row in result["models"][model]["metrics"] if row["threshold"] == 0.25]
        _require(len(rows) == 1, f"seed {seed}: missing or duplicate primary metrics for {model}")
        primary[model] = rows[0]
        for field in FIELDS:
            value = rows[0][field]
            _require(type(value) in (int, float) and math.isfinite(value),
                     f"seed {seed}: undefined/nonfinite primary {model} {field}; no complete summary")
    paired = dominance(primary["m2"], primary["m1"])
    controls = dominance(primary["m2"], primary["m0"])
    reported = [row for row in result["comparisons"] if row["primary"]]
    _require(len(reported) == 1 and reported[0]["box_confidence"] == 0.25
             and _same(reported[0]["comparisons"]["m2_vs_m1"], paired)
             and _same(reported[0]["comparisons"]["m2_vs_m0"], controls),
             f"seed {seed}: primary deltas differ from audited metrics")
    prediction_path = _contained_path(evidence, "m0/predictions.jsonl")
    _require(result["input_files_sha256"]["m0/predictions.jsonl"] == file_sha256(prediction_path),
             f"seed {seed}: M0 input changed after replay")
    shared = {**bundles["reference"], "reference": result["reference"],
              "inference": result["inference"], "quality_source": quality["source"],
              "quality_procedure": quality["procedure"],
              "m0_prediction_payload_sha256": _payload_digest(prediction_path)}
    result.update(
        kind="baomai-box-repeatability-seed", seed=seed,
        experiment={**expected_recipe, **bundles["reference"]},
        candidate_minus_reference=paired["new_minus_old"],
        descriptive_five_metric_rule={"candidate_vs_reference": paired["passes"],
            "candidate_vs_m0": controls["passes"],
            "candidate_vs_both": paired["passes"] and controls["passes"],
            "adoption_authorized": False},
        descriptive_baseline_rule_passers=result.pop("baseline_replacement_eligible"),
        decision="diagnostic-only-no-promotion", predeclared_adoption_eligible=[],
    )
    result.pop("candidate_meets_rule_against_both_controls")
    result["comparison_rule"]["descriptive_only"] = True
    result["limits"].append("This prospective seed is descriptive only; it does not reopen the historical seed-42 decision.")
    return result, shared


def summarize(root, evidence):
    """Require both prospective seeds; retain all outcomes and equal seed weights."""
    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    code_root = Path(__file__).resolve().parents[1]
    source_snapshot = {name: file_sha256(code_root / name) for name in
        ("scripts/summarize_box_repeatability.py", "scripts/package_box_repeatability.py", *SOURCE_FILES)}
    _require(set(BOX_REPEATABILITY_RECIPES) == set(SEEDS), "fixed repeatability seed declaration differs")
    _require(evidence.is_dir(), "repeatability evidence directory is missing")
    expected = {f"seed-{seed}" for seed in SEEDS}
    present = {path.name for path in evidence.iterdir() if path.name.startswith("seed-")}
    _require(present == expected,
             f"require exactly seed-43 and seed-44 evidence; missing={sorted(expected-present)}, extra={sorted(present-expected)}")
    seed_paths = []
    for seed in SEEDS:
        path = _contained_path(evidence, f"seed-{seed}", required=False)
        _require(path.is_dir(), f"seed {seed}: evidence directory is missing")
        seed_paths.append(path)
    _require(len({path.resolve() for path in seed_paths}) == len(SEEDS), "duplicate seed evidence directory")
    candidate_sha = file_sha256(root / CANDIDATE_MANIFEST)
    _require(candidate_sha == CANDIDATE_SHA256, "candidate manifest differs from the immutable study proposal")
    results, shared = [], None
    for seed, path in zip(SEEDS, seed_paths):
        result, identity = _audit_seed(root, path, seed)
        if shared is None:
            shared = identity
        else:
            for key in shared:
                _require(_same(shared[key], identity[key]), f"prospective seeds differ: {key}")
        results.append(result)
    aggregates = {}
    for field in FIELDS:
        values = [row["candidate_minus_reference"][field] for row in results]
        aggregates[field] = {"seed_count": len(SEEDS), "mean": math.fsum(values) / len(SEEDS),
            "min": min(values), "max": max(values), "improved": sum(value < 0 for value in values),
            "tied": sum(value == 0 for value in values), "worsened": sum(value > 0 for value in values)}
    source = {**results[0]["source"], "files_sha256": dict(results[0]["source"]["files_sha256"])}
    for path, digest in {**source["files_sha256"], **shared["source"]["files_sha256"]}.items():
        _require(file_sha256(_contained_path(code_root, path)) == digest, f"audit source changed: {path}")
    for name, digest in source_snapshot.items():
        _require(file_sha256(_contained_path(code_root, name)) == digest, f"audit source changed: {name}")
        source["files_sha256"][name] = digest
    inputs = {f"seed-{row['seed']}/{path}": digest for row in results
              for path, digest in row["input_files_sha256"].items()}
    for path, digest in inputs.items():
        _require(file_sha256(_contained_path(evidence, path)) == digest, f"input changed during audit: {path}")
    _require(file_sha256(root / CANDIDATE_MANIFEST) == candidate_sha, "candidate manifest changed during audit")
    return {"schema_version": 1, "kind": "baomai-box-repeatability-summary", "study_id": STUDY_ID,
        "status": "verified-descriptive-only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "input_files_sha256": dict(sorted(inputs.items())),
        "prospective_seeds": list(SEEDS), "prospective_pair_count": len(SEEDS),
        "shared_provenance": shared, "per_seed": results,
        "candidate_minus_reference_summary": aggregates,
        "aggregation": {"primary_box_confidence": 0.25, "lower_is_better_fields": list(FIELDS),
            "seed_weights": {str(seed): 1 / len(SEEDS) for seed in SEEDS},
            "definition": "Equal-weight arithmetic mean/min/max of the two seed-level candidate-minus-reference metric deltas; no rounding before aggregation.",
            "conditional_quantiles": "Each median/p95 is computed within one model and seed on its returned visible endpoints. The mean of seed-level quantile deltas is not a pooled endpoint quantile.",
            "pooled_endpoints": False, "best_seed_selection": False,
            "confidence_intervals": False, "statistical_significance_claim": False},
        "historical_context": {"seed": 42, "report_url": HISTORICAL_REPORT_URL,
            "original_decision": "retain-M0", "included_in_prospective_aggregation": False,
            "reason": "Historical source is different and the earlier runtime/session is not bound as part of this prospective study. Seed 42 is context, not a third matched run."},
        "decision": "retain-M0", "predeclared_adoption_eligible": [],
        "decision_scope": "This repeatability study authorizes no promotion and does not change the original seed-42 decision or acceptance rule.",
        "verification_scope": "Saved-prediction replay and checkpoint-evidence consistency for both fixed seeds; current training/evaluation source hashes, matched source/runtime/settings, and common M0 payloads checked. No model weights loaded or verified here.",
        "limits": ["Only two prospective paired seeds; both outcomes are retained, including regressions or ties.",
            "Repeated development validation is not an independent test. Related images/endpoints are not independent trials.",
            "Conditional endpoint errors must be read with missed objects and missing endpoints; per-seed denominators and common-endpoint diagnostics remain in per_seed.",
            "A repeated trend under these recorded runtimes does not establish unseen-scene generalization, a physical error tolerance, statistical significance, or bitwise GPU determinism.",
            "Missing/failed seeds or undefined primary statistics refuse a complete summary; partial results cannot become a winner."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="frozen validation repository")
    parser.add_argument("--evidence", type=Path, required=True, help="parent containing seed-43 and seed-44")
    parser.add_argument("--output", type=Path, required=True, help="new JSON; existing output is refused")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        result = summarize(args.root, args.evidence)
        result["command"] = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                          "prospective_seeds": result["prospective_seeds"], "decision": result["decision"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError) as error:
        print(f"Box repeatability summary refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
