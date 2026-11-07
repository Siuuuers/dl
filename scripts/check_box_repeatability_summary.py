#!/usr/bin/env python3
"""Focused prospective-study checks; synthetic audited interfaces, no ML runtime.

The existing three-model replay auditor is stubbed at its boundary so these
checks exercise only the new cross-seed/proposal/summary contract. Existing
checkpoint and scoring checks cover that underlying auditor's responsibilities.
"""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluate_pose import file_sha256
from pose_checkpoint import BOX_REPEATABILITY_RECIPES, recipe_identity
import summarize_box_repeatability as study
from summarize_pose_comparison import FIELDS, dominance
from train_pose_comparison import SOURCE_FILES


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


class RepeatabilityChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root, self.evidence = self.base / "root", self.base / "evidence"
        self.code_root = Path(__file__).resolve().parents[1]
        target = self.root / study.CANDIDATE_MANIFEST
        target.parent.mkdir(parents=True)
        target.write_bytes((self.code_root / study.CANDIDATE_MANIFEST).read_bytes())
        self.outputs = {}
        self.source = {"revision": "1" * 40, "working_tree_status": "",
            "files_sha256": {name: file_sha256(self.code_root / name) for name in SOURCE_FILES}}
        self.proposal = {"candidate_id": study.CANDIDATE_ID,
            "candidate_manifest": study.CANDIDATE_MANIFEST, "candidate_manifest_sha256": study.CANDIDATE_SHA256}
        for seed in study.SEEDS:
            self.fixture(seed)
        self.auditor = patch.object(study, "audit_comparison", side_effect=self.fake_audit).start()
        self.addCleanup(patch.stopall)

    def fixture(self, seed):
        path = self.evidence / f"seed-{seed}"
        recipe = recipe_identity(BOX_REPEATABILITY_RECIPES[seed])
        models = {}
        # Seed 43 improves on all five; seed 44 worsens on all five. The full
        # summary must keep both and may never turn the first seed into a winner.
        reference = dict(zip(FIELDS, (10, 20, 30, 1.125, 5.125)))
        delta = (-1, -2, -3, -0.000000000123456, -0.25) if seed == 43 else (2, 3, 4, 0.0625, 0.75)
        candidate = {field: reference[field] + change for field, change in zip(FIELDS, delta)}
        baseline = {field: reference[field] + 10 for field in FIELDS}
        for model, metrics in (("m0", baseline), ("m1", reference), ("m2", candidate)):
            models[model] = {"checkpoint_identity": None,
                "metrics": [{"threshold": threshold, **metrics} for threshold in (.15, .25, .40)]}
        for model, arm in (("m1", "reference"), ("m2", "candidate")):
            arm_root = path / "checkpoints" / arm
            manifest = {"schema_version": 2, **recipe,
                "training_receipt": {"path": "training-receipt.json"},
                "staging_manifest": {"path": "staging-manifest.json"}}
            receipt = {"schema_version": 2, **recipe, "source": self.source,
                "runtime": {"device": "synthetic-fixture", "torch": "synthetic-only"},
                "training_settings": {"seed": seed, "epochs": 30, "batch": 4,
                    "data": str(arm_root / "staged-data"), "project": str(arm_root)}}
            write_json(arm_root / "checkpoint-manifest.json", manifest)
            write_json(arm_root / "training-receipt.json", receipt)
            write_json(arm_root / "staging-manifest.json", {"source": self.proposal})
            models[model]["checkpoint_identity"] = {"arm": arm, **recipe}
        write_json(path / "m0-quality/quality-summary.json", {"source": self.source,
            "procedure": {"radii_px": [2, 5, 10], "matching_iou": .5}})
        predictions = [{"record_type": "run"}]
        predictions += [{"record_type": "prediction", "image_id": f"synthetic-{index}",
            "threshold": threshold, "predictions": [], "stage_ms": {"inference": seed},
            "predict_call_wall_ms": seed + 1} for index in range(20) for threshold in (.15, .25, .40)]
        (path / "m0").mkdir()
        (path / "m0/predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
        comparisons = {"m2_vs_m1": dominance(candidate, reference),
                       "m2_vs_m0": dominance(candidate, baseline)}
        self.outputs[seed] = {"source": copy.deepcopy(self.source), "models": models,
            "reference": {"manifest_sha256": "2" * 64},
            "inference": {"source": self.source, "runtime": {"device": "synthetic-cpu"},
                          "inference": {"iou": .7}, "images": ["synthetic"], "warmup": "synthetic"},
            "comparisons": [{"primary": True, "box_confidence": .25, "comparisons": comparisons}],
            "baseline_replacement_eligible": ["m1", "m2"],
            "candidate_meets_rule_against_both_controls": all(row["passes"] for row in comparisons.values()),
            "comparison_rule": {}, "limits": []}

    def fake_audit(self, root, evidence):
        seed = int(evidence.name.removeprefix("seed-"))
        output = copy.deepcopy(self.outputs[seed])
        output["input_files_sha256"] = {p.relative_to(evidence).as_posix(): file_sha256(p)
            for p in evidence.rglob("*") if p.is_file()}
        return output

    def run_summary(self):
        return study.summarize(self.root, self.evidence)

    def change_receipts(self, seed, scope, key, value):
        for arm in ("reference", "candidate"):
            path = self.evidence / f"seed-{seed}/checkpoints/{arm}/training-receipt.json"
            receipt = json.loads(path.read_text())
            receipt[scope][key] = value
            write_json(path, receipt)

    def test_both_unrounded_outcomes_are_equal_weight_and_never_promoted(self):
        result = self.run_summary()
        self.assertEqual(self.auditor.call_count, 2)
        self.assertEqual([row["seed"] for row in result["per_seed"]], [43, 44])
        self.assertEqual(result["decision"], "retain-M0")
        self.assertEqual(result["predeclared_adoption_eligible"], [])
        self.assertTrue(result["per_seed"][0]["descriptive_five_metric_rule"]["candidate_vs_both"])
        self.assertFalse(result["per_seed"][1]["descriptive_five_metric_rule"]["candidate_vs_reference"])
        self.assertFalse(result["historical_context"]["included_in_prospective_aggregation"])
        self.assertEqual(result["aggregation"]["seed_weights"], {"43": .5, "44": .5})
        self.assertFalse(result["aggregation"]["pooled_endpoints"])
        for field in FIELDS:
            values = [row["candidate_minus_reference"][field] for row in result["per_seed"]]
            summary = result["candidate_minus_reference_summary"][field]
            self.assertEqual(summary["mean"], sum(values) / 2)
            self.assertEqual((summary["min"], summary["max"]), (min(values), max(values)))
            self.assertEqual((summary["improved"], summary["tied"], summary["worsened"]), (1, 0, 1))
        self.assertEqual(result["candidate_minus_reference_summary"]["conditional_p95_error_px"]["mean"], .25)
        self.assertNotEqual(result["per_seed"][0]["candidate_minus_reference"]["conditional_median_error_px"], 0)

    def test_even_two_descriptive_passes_do_not_authorize_promotion(self):
        for row in self.outputs[44]["models"]["m2"]["metrics"]:
            row.update({field: self.outputs[44]["models"]["m1"]["metrics"][1][field] - (1 if field in FIELDS[:3] else 0.25) for field in FIELDS})
        primary = {name: value["metrics"][1] for name, value in self.outputs[44]["models"].items()}
        self.outputs[44]["comparisons"][0]["comparisons"] = {
            "m2_vs_m1": dominance(primary["m2"], primary["m1"]),
            "m2_vs_m0": dominance(primary["m2"], primary["m0"])}
        result = self.run_summary()
        self.assertTrue(all(row["descriptive_five_metric_rule"]["candidate_vs_both"] for row in result["per_seed"]))
        self.assertEqual(result["decision"], "retain-M0")
        self.assertEqual(result["predeclared_adoption_eligible"], [])

    def test_mutated_first_seed_input_is_detected_after_second_audit(self):
        def audit_with_change(root, evidence):
            result = self.fake_audit(root, evidence)
            if evidence.name == "seed-44":
                path = self.evidence / "seed-43/checkpoints/reference/training-receipt.json"
                path.write_text(path.read_text() + "\n")
            return result
        self.auditor.side_effect = audit_with_change
        with self.assertRaisesRegex(ValueError, "input changed during audit"):
            self.run_summary()

    def test_missing_seed_and_extra_seed_attempt_refuse_before_auditing(self):
        path = self.evidence / "seed-44"
        path.rename(self.evidence / "incomplete")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_summary()
        self.auditor.assert_not_called()
        (self.evidence / "incomplete").rename(path)
        (self.evidence / "seed-43-retry").mkdir()
        with self.assertRaisesRegex(ValueError, "extra"):
            self.run_summary()

    def test_duplicate_directory_alias_refuses(self):
        path = self.evidence / "seed-44"
        path.rename(self.evidence / "unused")
        path.symlink_to(self.evidence / "seed-43", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_summary()

    def test_seed43_cannot_masquerade_as_seed44(self):
        for model in ("m1", "m2"):
            self.outputs[44]["models"][model]["checkpoint_identity"].update(recipe_identity(BOX_REPEATABILITY_RECIPES[43]))
        with self.assertRaisesRegex(ValueError, "seed44"):
            self.run_summary()

    def test_resolved_seed_must_match_its_declared_directory(self):
        self.change_receipts(44, "training_settings", "seed", 43)
        with self.assertRaisesRegex(ValueError, "exact repeatability recipe and seed"):
            self.run_summary()

    def test_source_runtime_and_nonseed_settings_cannot_be_pooled(self):
        cases = [("source", "revision", "9" * 40, "source"),
                 ("runtime", "device", "different-fixture-device", "runtime"),
                 ("training_settings", "batch", 8, "nonseed_training_settings")]
        for scope, key, value, message in cases:
            with self.subTest(scope=scope):
                self.fixture_reset_receipts(44)
                self.change_receipts(44, scope, key, value)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_summary()

    def fixture_reset_receipts(self, seed):
        self.change_receipts(seed, "source", "revision", "1" * 40)
        self.change_receipts(seed, "runtime", "device", "synthetic-fixture")
        self.change_receipts(seed, "training_settings", "batch", 4)

    def test_inference_and_quality_sources_must_match_across_seeds(self):
        original = copy.deepcopy(self.outputs[44]["inference"])
        self.outputs[44]["inference"]["runtime"]["device"] = "other-fixture-cpu"
        with self.assertRaisesRegex(ValueError, "inference"):
            self.run_summary()
        self.outputs[44]["inference"] = original
        path = self.evidence / "seed-44/m0-quality/quality-summary.json"
        quality = json.loads(path.read_text())
        quality["source"]["revision"] = "9" * 40
        write_json(path, quality)
        with self.assertRaisesRegex(ValueError, "quality_source"):
            self.run_summary()

    def test_m0_timing_may_change_but_prediction_payload_may_not(self):
        self.run_summary()  # Fixtures already have different timings.
        path = self.evidence / "seed-44/m0/predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["predictions"] = [{"synthetic_changed_prediction": True}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "m0_prediction_payload"):
            self.run_summary()

    def test_changed_proposal_and_stale_training_code_refuse(self):
        proposal = self.root / study.CANDIDATE_MANIFEST
        original = proposal.read_bytes()
        proposal.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "immutable study proposal"):
            self.run_summary()
        proposal.write_bytes(original)
        hashes = {**self.source["files_sha256"], "scripts/train_pose_comparison.py": "0" * 64}
        self.change_receipts(44, "source", "files_sha256", hashes)
        with self.assertRaisesRegex(ValueError, "recorded training code differs"):
            self.run_summary()

    def test_undefined_primary_statistic_refuses_a_complete_summary(self):
        self.outputs[44]["models"]["m2"]["metrics"][1]["conditional_p95_error_px"] = None
        with self.assertRaisesRegex(ValueError, "undefined/nonfinite"):
            self.run_summary()

    def test_failed_second_audit_cannot_publish_first_seed_as_winner(self):
        self.auditor.side_effect = [self.fake_audit(self.root, self.evidence / "seed-43"), ValueError("incomplete seed 44")]
        output = self.base / "summary.json"
        with contextlib.redirect_stderr(io.StringIO()):
            code = study.main(["--root", str(self.root), "--evidence", str(self.evidence), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())

    def test_existing_output_is_preserved_without_auditing(self):
        output = self.base / "keep.json"
        output.write_text("preserve original bytes")
        with contextlib.redirect_stderr(io.StringIO()):
            code = study.main(["--root", str(self.root), "--evidence", str(self.evidence), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(output.read_text(), "preserve original bytes")
        self.auditor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
