#!/usr/bin/env python3
"""Focused composed-pair integration checks; synthetic CPU receipts, never train.

The annotation contract is checked separately by check_pose_composed_candidate.
Here its receipt is mocked around a small, real-PNG 137/20 fixture so routing,
completed-receipt binding, and downstream collection can be tested independently.
Synthetic checkpoints are never deserialized and all outputs are temporary.
"""

import sys
sys.dont_write_bytecode = True

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_evaluate_pose as baseline
from check_pose_checkpoint import bundle_fixture
import evaluate_pose
from package_gpu_results import package_results
from pose_checkpoint import (ANNOTATION_CONFIRMATION_RECIPE_ID, COMPOSED_RECIPE_ID, ENDPOINT_RECIPE_ID,
                             ORIGINAL_RECIPE_ID, PINNED_TRAINER_DEFAULTS, RECIPE_IDS,
                             check_recipe_candidate_kind, get_recipe, recipe_identity, select_checkpoint,
                             training_label_identity, write_checkpoint_manifest)
from stage_pose_training import MANIFEST, stage_datasets, verify_staging
from train_pose_comparison import descriptor, run_parent, worker_training_labels


VISIBLE = "0 .5 .5 .8 .8 .2 .6 2 .7 .6 2\n"
MISSING = "0 .5 .5 .6 .6 0 0 0 0 0 0\n"
CONTROL = "0 .5 .5 .8 .8 .21 .6 2 .7 .6 2\n"
TREATMENT = "0 .5 .5 .7 .7 .2 .6 2 .7 .6 2\n"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def integration_fixture(base):
    root = base / "repository"
    reference = baseline.preflight_fixture(root)
    for index in range(137):
        identifier = f"train_{index:03d}"
        image, label = f"images/train/{identifier}.png", f"labels/train/{identifier}.txt"
        for name, payload in ((image, baseline.png_bytes()), (label, (VISIBLE + MISSING).encode())):
            path = root / "data" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        reference["records"].append({"id": identifier, "split": "train", "image": image, "label": label,
            "original_size": [100, 100], "instances": 2, "visible_endpoints": 2, "missing_endpoints": 2,
            "image_sha256": evaluate_pose.file_sha256(root / "data" / image),
            "label_sha256": evaluate_pose.file_sha256(root / "data" / label)})
    reference["split_summary"]["train"] = {"images": 137, "instances": 274, "visible_endpoints": 274,
                                             "missing_endpoints": 274, "empty_images": 0}
    write_json(root / "data/manifest.json", reference)
    preflight = evaluate_pose.preflight_dataset(root, require_weights=False)
    counts = {split: {("objects" if key == "instances" else key): value for key, value in values.items()}
              for split, values in reference["split_summary"].items()}
    records = {row["id"]: row for row in reference["records"]}

    def proposal(folder, kind, labels, deleted):
        path = root / "data/candidates" / folder / "manifest.json"
        changes = []
        for identifier, payload in labels.items():
            record = records[identifier]
            label = path.parent / record["label"]
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(payload)
            changes.append({"image_id": identifier, "source_label": record["label"],
                "candidate_label": record["label"], "source_label_sha256": record["label_sha256"],
                "candidate_label_sha256": evaluate_pose.file_sha256(label)})
        expected = copy.deepcopy(counts)
        expected["train"]["objects"] -= deleted
        expected["train"]["missing_endpoints"] -= 2 * deleted
        value = {"candidate_id": f"synthetic-{folder}", "kind": kind,
                 "source_manifest_sha256": preflight["manifest_sha256"],
                 "changes": changes, "expected_counts": expected}
        write_json(path, value)
        return path, value

    control_path, control = proposal("control", "training_endpoint_overlay", {"train_000": CONTROL}, 1)
    candidate, composed = proposal("composed", "training_composed_overlay",
                                  {"train_000": CONTROL, "train_001": TREATMENT}, 2)
    composed["control"] = {"manifest": control_path.relative_to(root).as_posix(),
                            "sha256": evaluate_pose.file_sha256(control_path)}
    write_json(candidate, composed)
    receipt = {"candidate_id": composed["candidate_id"], "candidate_manifest_sha256": evaluate_pose.file_sha256(candidate),
               "source_manifest_sha256": preflight["manifest_sha256"], "declared_source_revision": "1" * 40,
               "changed_train_labels": 2, "removed_objects": 2, "counts": composed["expected_counts"],
               "control": {"manifest": composed["control"]["manifest"], "manifest_sha256": composed["control"]["sha256"],
                           "kind": "training_endpoint_overlay", "candidate_id": control["candidate_id"],
                           "counts": control["expected_counts"], "changed_train_labels": 1, "changed_endpoint_slots": 1}}
    frozen = {root / "data/manifest.json": preflight["manifest_sha256"], candidate: receipt["candidate_manifest_sha256"],
              control_path: receipt["control"]["manifest_sha256"]}

    def verified(actual_root, actual_candidate):
        if Path(actual_root).resolve() != root or Path(actual_candidate).resolve() != candidate:
            raise ValueError("synthetic verifier received a different root/proposal")
        if any(evaluate_pose.file_sha256(path) != digest for path, digest in frozen.items()):
            raise ValueError("synthetic verifier frozen proposal/control identity differs")
        return copy.deepcopy(receipt)

    template_dir = base / "template"
    template_dir.mkdir()
    _, _, _, template, _, _ = bundle_fixture(template_dir, "reference", recipe_id=ANNOTATION_CONFIRMATION_RECIPE_ID)
    return SimpleNamespace(base=base, root=root, preflight=preflight, reference=reference, candidate=candidate,
        control_path=control_path, composed=composed, receipt=receipt, verifier=verified, template=template)


def complete_bundle(fixture, arm, name=None):
    """Construct a marked synthetic receipt fixture; no optimizer or GPU is run."""
    run = fixture.base / (name or f"run-{arm}")
    run.mkdir()
    stage_datasets(fixture.root, fixture.candidate, run / "staged-data")
    shutil.copyfile(run / "staged-data" / MANIFEST, run / MANIFEST)
    staging = json.loads((run / MANIFEST).read_text())
    weights = run / "fit/weights/last.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text(f"Synthetic {arm} checkpoint fixture; never deserialize. No training performed.\n")
    results = run / "fit/results.csv"
    results.write_text("epoch,train/box_loss,train/pose_loss,train/kobj_loss,val/box_loss,val/pose_loss,val/kobj_loss\n"
                      + "".join(f"{epoch},1,0.2,0.3,1.1,0.21,0.31\n" for epoch in range(1, 31)))
    log = run / "train.log"
    log.write_text("Synthetic receipt integration check. No training performed.\n")
    receipt = copy.deepcopy(fixture.template)
    receipt.update({**recipe_identity(COMPOSED_RECIPE_ID), "arm": arm,
        "reference_manifest_sha256": fixture.preflight["manifest_sha256"],
        "model_card_sha256": fixture.preflight["model_card_sha256"],
        "initial_checkpoint_sha256": fixture.preflight["model_sha256"],
        "staging_manifest_sha256": evaluate_pose.file_sha256(run / MANIFEST),
        "data_yaml_sha256": staging["variants"][arm]["data_yaml_sha256"],
        "training_labels": training_label_identity(staging, arm),
        "training_settings": {**PINNED_TRAINER_DEFAULTS, **get_recipe(COMPOSED_RECIPE_ID),
            "model": fixture.preflight["weights"], "data": str(run / "staged-data" / arm / "data.yaml"),
            "device": "cpu", "project": str(run), "save_dir": str(run / "fit")},
        "checkpoint": descriptor(weights, run, checkpoint=True), "results": descriptor(results, run),
        "log": descriptor(log, run)})
    receipt["source"]["files_sha256"].update({"scripts/stage_pose_training.py": "4" * 64,
                                              "scripts/verify_pose_composed_candidate.py": "5" * 64})
    write_json(run / "training-receipt.json", receipt)
    manifest = write_checkpoint_manifest(run, fixture.preflight)
    return run, receipt, staging, manifest


def refresh_artifact_hashes(run, receipt, staging=None):
    if staging is not None:
        write_json(run / MANIFEST, staging)
    receipt["staging_manifest_sha256"] = evaluate_pose.file_sha256(run / MANIFEST)
    write_json(run / "training-receipt.json", receipt)
    path = run / "checkpoint-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training_receipt"] = descriptor(run / "training-receipt.json", run)
    manifest["staging_manifest"] = descriptor(run / MANIFEST, run)
    write_json(path, manifest)


class ComposedTrainingChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = integration_fixture(Path(temporary.name))
        mocked = patch("verify_pose_composed_candidate.verify_composed_candidate", side_effect=self.fixture.verifier)
        mocked.start()
        self.addCleanup(mocked.stop)

    def test_fixed_recipe_matches_endpoint_numbers_and_rejects_every_other_kind_pair(self):
        self.assertEqual(get_recipe(COMPOSED_RECIPE_ID), get_recipe(ENDPOINT_RECIPE_ID))
        self.assertEqual(get_recipe(COMPOSED_RECIPE_ID)["seed"], 46)
        self.assertEqual(recipe_identity(COMPOSED_RECIPE_ID)["recipe_sha256"], recipe_identity(ENDPOINT_RECIPE_ID)["recipe_sha256"])
        check_recipe_candidate_kind(COMPOSED_RECIPE_ID, "training_composed_overlay")
        for recipe in RECIPE_IDS:
            if recipe != COMPOSED_RECIPE_ID:
                with self.subTest(recipe=recipe), self.assertRaisesRegex(ValueError, "must be selected together"):
                    check_recipe_candidate_kind(recipe, "training_composed_overlay")
        for kind in ("training_label_overlay", "training_box_overlay", "training_annotation_overlay", "training_endpoint_overlay"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "must be selected together"):
                check_recipe_candidate_kind(COMPOSED_RECIPE_ID, kind)

    def test_staging_uses_control_overlay_and_preserves_validation(self):
        f, output = self.fixture, self.fixture.base / "paired"
        stage_datasets(f.root, f.candidate, output)
        self.assertEqual(verify_staging(f.root, f.candidate, output)["status"], "verified")
        manifest = json.loads((output / MANIFEST).read_text())
        self.assertEqual(manifest["source"]["control"], f.receipt["control"])
        self.assertEqual((output / "reference/labels/train/train_000.txt").read_text(), CONTROL)
        self.assertEqual((output / "candidate/labels/train/train_000.txt").read_text(), CONTROL)
        self.assertEqual((output / "reference/labels/train/train_001.txt").read_text(), VISIBLE + MISSING)
        self.assertEqual((output / "candidate/labels/train/train_001.txt").read_text(), TREATMENT)
        self.assertEqual(manifest["verification"]["removed_training_objects_vs_control"], 1)
        self.assertEqual(manifest["verification"]["removed_training_objects"], 2)
        self.assertEqual(manifest["variants"]["reference"]["counts"], f.receipt["control"]["counts"])
        for path in (output / "reference/labels/val").glob("*.txt"):
            self.assertEqual(path.read_bytes(), (output / "candidate/labels/val" / path.name).read_bytes())

    def test_completed_both_arm_receipts_expose_actual_label_roles(self):
        f = self.fixture
        for arm, role in (("reference", "endpoint_control"), ("candidate", "composed_treatment")):
            with self.subTest(arm=arm):
                _, receipt, _, manifest = complete_bundle(f, arm)
                selected = select_checkpoint(f.preflight, manifest)
                identity = selected["checkpoint_identity"]
                self.assertEqual((identity["arm"], identity["recipe_id"]), (arm, COMPOSED_RECIPE_ID))
                self.assertEqual(identity["training_labels"]["role"], role)
                self.assertEqual(identity["training_labels"], receipt["training_labels"])
                self.assertEqual(identity["training_labels"]["control_manifest_sha256"], f.receipt["control"]["manifest_sha256"])
                self.assertEqual(selected["manifest_sha256"], f.preflight["manifest_sha256"])

    def test_original_reference_substitution_is_rejected_even_after_hash_refresh(self):
        f = self.fixture
        run, receipt, staging, manifest = complete_bundle(f, "reference")
        item = next(x for x in staging["variants"]["reference"]["files"] if x["path"] == "labels/train/train_000.txt")
        digest = evaluate_pose.file_sha256(f.root / "data/labels/train/train_000.txt")
        item["source_sha256"] = item["output_sha256"] = digest
        refresh_artifact_hashes(run, receipt, staging)
        with self.assertRaisesRegex(ValueError, "file inventory differs"):
            select_checkpoint(f.preflight, manifest)

    def test_control_and_label_role_tampering_cannot_hide_behind_new_receipt_hashes(self):
        f = self.fixture
        run, receipt, staging, manifest = complete_bundle(f, "reference")
        for field, value in (("role", "composed_treatment"), ("control_manifest_sha256", "0" * 64),
                             ("label_manifest_sha256", staging["source"]["candidate_manifest_sha256"])):
            with self.subTest(field=field):
                changed = copy.deepcopy(receipt)
                changed["training_labels"][field] = value
                refresh_artifact_hashes(run, changed)
                with self.assertRaisesRegex(ValueError, "control/treatment label identity differs"):
                    select_checkpoint(f.preflight, manifest)
        altered = copy.deepcopy(staging)
        altered["source"]["control"]["manifest_sha256"] = "0" * 64
        refresh_artifact_hashes(run, receipt, altered)
        with self.assertRaisesRegex(ValueError, "staged endpoint control identity/counts differ"):
            select_checkpoint(f.preflight, manifest)

    def test_worker_rejects_arm_swap_or_endpoint_recipe_before_runtime_import(self):
        f = self.fixture
        _, _, staging, _ = complete_bundle(f, "reference")
        config = {"recipe_id": COMPOSED_RECIPE_ID, "arm": "reference",
                  "candidate_manifest": staging["source"]["candidate_manifest"],
                  "training_labels": training_label_identity(staging, "reference")}
        self.assertEqual(worker_training_labels(config, staging)["role"], "endpoint_control")
        with self.assertRaisesRegex(ValueError, "label identity differs"):
            worker_training_labels({**config, "arm": "candidate"}, staging)
        with self.assertRaisesRegex(ValueError, "must be selected together"):
            worker_training_labels({**config, "recipe_id": ENDPOINT_RECIPE_ID}, staging)

    def test_generic_return_collector_and_evaluator_keep_composed_identity(self):
        f = self.fixture
        reference, _, _, manifest = complete_bundle(f, "reference")
        candidate, _, _, _ = complete_bundle(f, "candidate")
        result = package_results(f.root, reference, candidate, f.base / "synthetic-return.zip")
        self.assertEqual(result["arms"]["reference"]["checkpoint_identity"]["training_labels"]["role"], "endpoint_control")
        self.assertEqual(result["arms"]["candidate"]["checkpoint_identity"]["training_labels"]["role"], "composed_treatment")
        selected = select_checkpoint(f.preflight, manifest, require_weights=False)
        fixtures = baseline.ReplayChecks()
        fixtures.evaluator = evaluate_pose
        header, items = fixtures.replay_data(selected, [0.25])
        header["checkpoint_identity"] = copy.deepcopy(selected["checkpoint_identity"])
        replay = f.base / "synthetic-predictions.jsonl"
        fixtures.write_replay(replay, header, items)
        evaluate_pose._read_replay(replay, selected, None)
        header["checkpoint_identity"]["training_labels"]["role"] = "composed_treatment"
        fixtures.write_replay(replay, header, items)
        with self.assertRaisesRegex(ValueError, "checkpoint provenance differs"):
            evaluate_pose._read_replay(replay, selected, None)

    def test_parent_records_control_identity_and_refuses_worker_omission_without_training(self):
        f = self.fixture
        for include_identity in (True, False):
            run = f.base / ("trial-recorded" if include_identity else "trial-omitted")
            args = SimpleNamespace(root=f.root, output=run, threads=1, max_seconds=1, max_rss_mib=128,
                trial_batches=1, device="cpu", recipe=COMPOSED_RECIPE_ID, arm="reference", trial=True,
                candidate=f.candidate.relative_to(f.root).as_posix())

            def no_training_worker(command, output, config):
                (output / "train.log").write_text("Synthetic mocked-worker check. No training performed.\n")
                progress = {**recipe_identity(COMPOSED_RECIPE_ID), "status": "trial-complete"}
                if include_identity:
                    progress["training_labels"] = config["training_labels"]
                write_json(output / "training-progress.json", progress)
                return {"exit_code": 0, "stop_reason": None}

            with patch("train_pose_comparison.__file__", str(f.root / "scripts/train_pose_comparison.py")), \
                    patch("train_pose_comparison.importlib.metadata.version", return_value="8.4.182"), \
                    patch("train_pose_comparison.source_identity", return_value=f.template["source"]), \
                    patch("train_pose_comparison.monitor_worker", side_effect=no_training_worker), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_parent(args), 0 if include_identity else 1)
            receipt = json.loads((run / "training-receipt.json").read_text())
            self.assertEqual(receipt["training_labels"]["role"], "endpoint_control")
            self.assertEqual(receipt["status"], "trial-complete" if include_identity else "failed")
            self.assertFalse((run / "checkpoint-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
