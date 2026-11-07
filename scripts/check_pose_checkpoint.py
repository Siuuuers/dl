#!/usr/bin/env python3
"""Focused provenance checks with synthetic bundles; never train or deserialize weights."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import evaluate_pose
import diagnose_pose_quality
import check_evaluate_pose as baseline_checks
from pose_checkpoint import (ORIGINAL_RECIPE_ID, PINNED_TRAINER_DEFAULTS, RECIPE, RECIPE_IDS,
                             WARMUP_RECIPE_ID, check_loaded_pose_model,
                             get_recipe, recipe_identity, select_checkpoint, write_checkpoint_manifest)
from train_pose_comparison import descriptor


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bundle_fixture(directory, arm="candidate", *, recipe_id=None):
    """Synthetic receipt consistency only: no real model, training, or accuracy claim."""
    root, run = directory / "reference", directory / "run"
    reference = baseline_checks.preflight_fixture(root)
    image_hash = reference["records"][0]["image_sha256"]
    label_hash = hashlib.sha256(b"synthetic unlocatable object\n").hexdigest()
    empty_hash = hashlib.sha256(b"").hexdigest()
    for index in range(137):
        identifier = f"train_{index:03d}"
        reference["records"].append({"id": identifier, "split": "train",
            "image": f"images/train/{identifier}.png", "label": f"labels/train/{identifier}.txt",
            "original_size": [100, 100], "instances": 1, "visible_endpoints": 0,
            "missing_endpoints": 2, "image_sha256": image_hash, "label_sha256": label_hash})
    reference["split_summary"]["train"] = {"images": 137, "instances": 137,
        "visible_endpoints": 0, "missing_endpoints": 274, "empty_images": 0}
    write_json(root / "data/manifest.json", reference)
    preflight = evaluate_pose.preflight_dataset(root, require_weights=False)
    counts = {split: {("objects" if key == "instances" else key): value for key, value in values.items()}
              for split, values in reference["split_summary"].items()}
    candidate_counts = copy.deepcopy(counts)
    candidate_counts["train"].update(objects=136, missing_endpoints=272, empty_images=1)
    proposal = {"candidate_id": "synthetic-provenance-fixture", "source_manifest_sha256": preflight["manifest_sha256"],
        "changes": [{"image_id": "train_000", "source_label": "labels/train/train_000.txt",
                     "source_label_sha256": label_hash, "candidate_label_sha256": empty_hash}],
        "expected_counts": candidate_counts}
    proposal_path = root / "data/candidates/fixture/manifest.json"
    proposal_path.parent.mkdir(parents=True)
    write_json(proposal_path, proposal)
    run.mkdir()
    stage_root = str(run / "staged-data")
    variants = {}
    for variant in ("reference", "candidate"):
        files = []
        for record in reference["records"]:
            for kind in ("image", "label"):
                digest = record[f"{kind}_sha256"]
                if variant == "candidate" and kind == "label" and record["id"] == "train_000":
                    digest = empty_hash
                files.append({"path": record[kind], "source_sha256": digest, "output_sha256": digest})
        variants[variant] = {"data_yaml": f"{variant}/data.yaml",
            "data_yaml_sha256": hashlib.sha256(f"synthetic YAML {variant}".encode()).hexdigest(),
            "counts": counts if variant == "reference" else candidate_counts, "files": files}
    staging = {"schema_version": 1, "status": "complete", "output_root": stage_root,
        "source": {"reference_manifest_sha256": preflight["manifest_sha256"],
            "model_card_sha256": preflight["model_card_sha256"], "initial_checkpoint_sha256": preflight["model_sha256"],
            "candidate_id": proposal["candidate_id"], "candidate_manifest": "data/candidates/fixture/manifest.json",
            "candidate_manifest_sha256": evaluate_pose.file_sha256(proposal_path)},
        "variants": variants, "verification": {"validation_images_and_labels_identical": True},
        "mechanical_geometry_gate": {"passed": True}}
    write_json(run / "staging-manifest.json", staging)
    weights = run / "fit/weights/last.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic final checkpoint; never deserialize\n")
    results = run / "fit/results.csv"
    results.write_text("epoch,train/box_loss,train/pose_loss,train/kobj_loss,val/box_loss,val/pose_loss,val/kobj_loss\n"
        + "".join(f"{epoch},1,0.2,0.3,1.1,0.21,0.31\n" for epoch in range(1, 31)), encoding="utf-8")
    log = run / "train.log"
    log.write_text("Synthetic fixture. No training was performed.\n", encoding="utf-8")
    receipt = {"schema_version": 1, "purpose": "paired-finetune", "status": "complete", "arm": arm,
        "initial_checkpoint_sha256": preflight["model_sha256"], "reference_manifest_sha256": preflight["manifest_sha256"],
        "model_card_sha256": preflight["model_card_sha256"],
        "staging_manifest_sha256": evaluate_pose.file_sha256(run / "staging-manifest.json"),
        "data_yaml_sha256": variants[arm]["data_yaml_sha256"],
        "training_settings": {**RECIPE, "data": str(Path(stage_root) / variants[arm]["data_yaml"])},
        "source": {"revision": "1" * 40, "working_tree_status": "",
                   "files_sha256": {"scripts/pose_checkpoint.py": "2" * 64, "scripts/train_pose_comparison.py": "3" * 64}},
        "runtime": {"python": "3.12.14", "torch": "2.10.0+cpu", "ultralytics": "8.4.182", "device": "cpu"},
        "completed_epochs": list(range(1, 31)), "optimizer_steps": 80,
        "optimizer_parameter_change_observed": True, "validation_executed": True,
        "training_returned": True, "finalization_completed": True, "checkpoint_selection": "final_epoch",
        "execution": {"exit_code": 0, "stop_reason": None},
        "checkpoint": descriptor(weights, run, checkpoint=True),
        "results": descriptor(results, run), "log": descriptor(log, run)}
    if recipe_id is not None:
        receipt.update({"schema_version": 2, **recipe_identity(recipe_id), "inputs_reverified_after_worker": True,
            "training_settings": {**PINNED_TRAINER_DEFAULTS, **get_recipe(recipe_id),
                "model": preflight["weights"], "data": str(Path(stage_root) / variants[arm]["data_yaml"]),
                "device": "cpu", "project": str(run), "save_dir": str(run / "fit")}})
        receipt["runtime"].update({"torch_cuda": None, "threads": 4, "interop_threads": 1,
            "platform": "synthetic CPU fixture; no training performed", "device_name": "cpu",
            "device_capability": None, "packages": {name: "fixture" for name in
                ("torchvision", "numpy", "Pillow", "opencv-python", "psutil")},
            "backend": {"model_channels_last": False, "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_version": None,
                "deterministic_algorithms_enabled": True, "deterministic_warn_only": True,
                "cudnn_deterministic": True, "cudnn_benchmark": False}})
    write_json(run / "training-receipt.json", receipt)
    manifest_path = write_checkpoint_manifest(run, preflight)
    return root, run, preflight, receipt, staging, manifest_path


class CheckpointBundleChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        (self.root, self.run, self.preflight, self.receipt, self.staging,
         self.manifest_path) = bundle_fixture(Path(self.temporary.name))
        self.manifest = json.loads(self.manifest_path.read_text())

    def save_receipt(self, receipt):
        write_json(self.run / "training-receipt.json", receipt)
        manifest = copy.deepcopy(self.manifest)
        manifest["training_receipt"] = descriptor(self.run / "training-receipt.json", self.run)
        write_json(self.manifest_path, manifest)

    def test_completed_bundle_uses_runner_descriptors_and_preserves_reference_identity(self):
        selected = select_checkpoint(self.preflight, self.manifest_path)
        self.assertEqual(set(self.receipt["results"]), {"path", "sha256"})
        self.assertEqual(set(self.receipt["checkpoint"]), {"path", "sha256", "size_bytes"})
        self.assertEqual(selected["reference_model_sha256"], self.preflight["model_sha256"])
        self.assertEqual(selected["model_sha256"], self.receipt["checkpoint"]["sha256"])
        self.assertNotEqual(selected["model_sha256"], self.preflight["model_sha256"])
        self.assertEqual(selected["manifest_sha256"], self.preflight["manifest_sha256"])
        self.assertEqual(selected["model_card_sha256"], self.preflight["model_card_sha256"])
        self.assertEqual(selected["checkpoint_identity"]["arm"], "candidate")
        self.assertEqual(selected["checkpoint_identity"]["completed_epochs"], 30)
        self.assertEqual(set(selected["checkpoint_identity"]), {"kind", "purpose", "arm",
            "checkpoint_sha256", "checkpoint_size_bytes", "training_receipt_sha256",
            "staging_manifest_sha256", "initial_checkpoint_sha256", "completed_epochs", "selection", "manifest_sha256"})
        self.assertTrue(selected["weights_verified"])
        self.assertNotIn("checkpoint_identity", self.preflight)

    def test_reference_arm_is_supported_and_arm_mismatch_is_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["arm"] = "reference"
        self.save_receipt(receipt)
        with self.assertRaisesRegex(ValueError, "arm/YAML"):
            select_checkpoint(self.preflight, self.manifest_path)
        variant = self.staging["variants"]["reference"]
        receipt["data_yaml_sha256"] = variant["data_yaml_sha256"]
        receipt["training_settings"]["data"] = str(Path(self.staging["output_root"]) / variant["data_yaml"])
        self.save_receipt(receipt)
        selected = select_checkpoint(self.preflight, self.manifest_path)
        self.assertEqual(selected["checkpoint_identity"]["arm"], "reference")

    def test_changed_artifact_bytes_are_rejected(self):
        names = ["fit/weights/last.pt", "training-receipt.json", "staging-manifest.json", "fit/results.csv", "train.log"]
        for name in names:
            with self.subTest(artifact=name):
                path = self.run / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                try:
                    with self.assertRaises(ValueError):
                        select_checkpoint(self.preflight, self.manifest_path)
                finally:
                    path.write_bytes(original)

    def test_trial_and_incomplete_receipts_cannot_publish_a_manifest(self):
        self.manifest_path.unlink()
        for receipt in ({"purpose": "resource-trial", "status": "trial-complete"},
                        {"purpose": "paired-finetune", "status": "failed"}):
            with self.subTest(receipt=receipt):
                write_json(self.run / "training-receipt.json", receipt)
                with self.assertRaisesRegex(ValueError, "only a completed"):
                    write_checkpoint_manifest(self.run, self.preflight)
                self.assertFalse(self.manifest_path.exists())

    def test_contradictory_completion_cannot_be_hidden_by_refreshed_hashes(self):
        cases = [("execution", "exit_code", 1), ("execution", "exit_code", False),
                 ("execution", "stop_reason", "RSS limit"),
                 (None, "optimizer_parameter_change_observed", False),
                 (None, "validation_executed", False), (None, "training_returned", False),
                 (None, "finalization_completed", False), (None, "optimizer_steps", True),
                 (None, "input_verification_error", "changed source"),
                 (None, "checkpoint_selection", "best"), (None, "completed_epochs", list(range(1, 30))),
                 ("source", "working_tree_status", " M scripts/train_pose_comparison.py")]
        for scope, key, value in cases:
            with self.subTest(field=f"{scope}.{key}"):
                receipt = copy.deepcopy(self.receipt)
                (receipt[scope] if scope else receipt)[key] = value
                self.save_receipt(receipt)
                with self.assertRaises(ValueError):
                    select_checkpoint(self.preflight, self.manifest_path)

    def test_changed_recipe_or_integer_fraction_is_rejected(self):
        for key, value in (("epochs", 1), ("lr0", 0.001), ("seed", 43), ("fraction", 1), ("resume", True)):
            with self.subTest(setting=key):
                receipt = copy.deepcopy(self.receipt)
                receipt["training_settings"][key] = value
                self.save_receipt(receipt)
                with self.assertRaisesRegex(ValueError, "training setting differs"):
                    select_checkpoint(self.preflight, self.manifest_path)

    def test_results_must_show_every_epoch_and_finite_losses(self):
        path = self.run / "fit/results.csv"
        original = path.read_text()
        lines = original.splitlines()
        variants = ["\n".join(lines[:-1]) + "\n", original.replace("30,1,", "29,1,"),
                    original.replace("30,1,", "30,nan,")]
        for changed in variants:
            with self.subTest(last_line=changed.splitlines()[-1]):
                path.write_text(changed)
                receipt = copy.deepcopy(self.receipt)
                receipt["results"] = descriptor(path, self.run)
                self.save_receipt(receipt)
                with self.assertRaises(ValueError):
                    select_checkpoint(self.preflight, self.manifest_path)

    def test_changed_staged_label_is_rejected_even_with_refreshed_artifact_hashes(self):
        staging = copy.deepcopy(self.staging)
        label = next(row for row in staging["variants"]["candidate"]["files"] if row["path"].startswith("labels/val/"))
        label["source_sha256"] = label["output_sha256"] = "f" * 64
        write_json(self.run / "staging-manifest.json", staging)
        receipt = copy.deepcopy(self.receipt)
        receipt["staging_manifest_sha256"] = evaluate_pose.file_sha256(self.run / "staging-manifest.json")
        self.save_receipt(receipt)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["staging_manifest"] = descriptor(self.run / "staging-manifest.json", self.run)
        write_json(self.manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "file inventory differs"):
            select_checkpoint(self.preflight, self.manifest_path)

    def test_manifest_schema_and_checkpoint_path_cannot_change_task_or_escape(self):
        cases = [("schema_version", True), ("kpt_shape", [5, 3]), ("flip_idx", [0, 1]),
                 ("checkpoint", {**self.manifest["checkpoint"], "path": "../last.pt"}),
                 ("checkpoint", {**self.manifest["checkpoint"], "path": "fit/weights/best.pt"})]
        for key, value in cases:
            with self.subTest(field=key, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest[key] = value
                write_json(self.manifest_path, manifest)
                with self.assertRaises(ValueError):
                    select_checkpoint(self.preflight, self.manifest_path)

    def test_symlinked_checkpoint_is_rejected(self):
        path = self.run / "fit/weights/last.pt"
        other = self.run / "actual.pt"
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            select_checkpoint(self.preflight, self.manifest_path)

    def test_manifest_creation_does_not_replace_existing_evidence(self):
        original = self.manifest_path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_checkpoint_manifest(self.run, self.preflight)
        self.assertEqual(self.manifest_path.read_bytes(), original)

    def test_candidate_replay_requires_explicit_matching_bundle_and_preserves_it(self):
        selected = select_checkpoint(self.preflight, self.manifest_path)
        fixtures = baseline_checks.ReplayChecks()
        fixtures.evaluator = evaluate_pose
        header, items = fixtures.replay_data(selected, [0.25])
        header["checkpoint_identity"] = copy.deepcopy(selected["checkpoint_identity"])
        predictions = self.run / "predictions.jsonl"
        fixtures.write_replay(predictions, header, items)
        with self.assertRaises(ValueError):
            evaluate_pose._read_replay(predictions, self.preflight, None)
        without_provenance = copy.deepcopy(header)
        without_provenance.pop("checkpoint_identity")
        fixtures.write_replay(predictions, without_provenance, items)
        with self.assertRaisesRegex(ValueError, "checkpoint provenance differs"):
            evaluate_pose._read_replay(predictions, selected, None)
        changed_provenance = copy.deepcopy(header)
        changed_provenance["checkpoint_identity"]["arm"] = "reference"
        fixtures.write_replay(predictions, changed_provenance, items)
        with self.assertRaisesRegex(ValueError, "checkpoint provenance differs"):
            evaluate_pose._read_replay(predictions, selected, None)
        fixtures.write_replay(predictions, header, items)
        (self.run / "fit/weights/last.pt").unlink()
        (self.root / "weights/cassette_lower_epoch173.pt").unlink()
        replay_only = select_checkpoint(self.preflight, self.manifest_path, require_weights=False)
        self.assertFalse(replay_only["weights_verified"])
        errors = io.StringIO()
        first, second = self.run / "replay-one", self.run / "replay-two"
        common = ["--root", str(self.root), "--checkpoint-manifest", str(self.manifest_path), "--overlays", "0"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            self.assertEqual(evaluate_pose.main(common + ["--replay", str(predictions), "--output", str(first)]), 0, errors.getvalue())
            self.assertEqual(evaluate_pose.main(common + ["--replay", str(first / "predictions.jsonl"), "--output", str(second)]), 0, errors.getvalue())
            diagnostic = self.run / "diagnostic"
            self.assertEqual(diagnose_pose_quality.main(["--root", str(self.root), "--checkpoint-manifest", str(self.manifest_path),
                "--predictions", str(second / "predictions.jsonl"), "--output", str(diagnostic)]), 0, errors.getvalue())
        replay_header = json.loads((second / "predictions.jsonl").read_text().splitlines()[0])
        self.assertEqual(replay_header["checkpoint_identity"], header["checkpoint_identity"])
        self.assertEqual(replay_header["replay"]["inference_source"], header["source"])
        summary = json.loads((diagnostic / "quality-summary.json").read_text())
        self.assertEqual(summary["reference"]["model_sha256"], self.preflight["model_sha256"])
        self.assertEqual(summary["evaluated_checkpoint"], header["checkpoint_identity"])

    def test_loaded_pose_head_checked_before_any_detection_is_available(self):
        model = SimpleNamespace(task="pose", names={0: "cassette_front"},
            model=SimpleNamespace(model=[SimpleNamespace(kpt_shape=[2, 3])]))
        check_loaded_pose_model(model)
        for shape in ([5, 3], [2, 2], None):
            with self.subTest(shape=shape):
                model.model.model[-1].kpt_shape = shape
                with self.assertRaises(RuntimeError):
                    check_loaded_pose_model(model)


class VersionedRecipeChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        (self.root, self.run, self.preflight, self.receipt, self.staging,
         self.manifest_path) = bundle_fixture(Path(self.temporary.name), "reference", recipe_id=WARMUP_RECIPE_ID)
        self.manifest = json.loads(self.manifest_path.read_text())

    def save(self, receipt, *, manifest=None):
        write_json(self.run / "training-receipt.json", receipt)
        manifest = copy.deepcopy(self.manifest if manifest is None else manifest)
        manifest["training_receipt"] = descriptor(self.run / "training-receipt.json", self.run)
        write_json(self.manifest_path, manifest)

    def test_recipe_identity_is_fixed_and_only_bias_warmup_changes(self):
        original, warmup = get_recipe(ORIGINAL_RECIPE_ID), get_recipe(WARMUP_RECIPE_ID)
        self.assertEqual(original, RECIPE)
        self.assertEqual({key for key in original if original[key] != warmup[key]}, {"warmup_bias_lr"})
        self.assertEqual((original["warmup_bias_lr"], warmup["warmup_bias_lr"]), (0.1, 0.0001))
        self.assertEqual(original["epochs"], warmup["epochs"])
        warmup["epochs"] = 1
        self.assertEqual(get_recipe(WARMUP_RECIPE_ID)["epochs"], 30)
        self.assertEqual(RECIPE["warmup_bias_lr"], 0.1)
        expected = {ORIGINAL_RECIPE_ID: "47d1cf598191fa0b3ff27a075536ae7b1938074f27b26805f933a9c0652032d5",
                    WARMUP_RECIPE_ID: "540a6ad54f1ed104b95bdf0a03e1a77c49de5fffffdab811130198fe0c57498c"}
        self.assertEqual({key: recipe_identity(key)["recipe_sha256"] for key in RECIPE_IDS}, expected)
        with self.assertRaisesRegex(ValueError, "unsupported training recipe"):
            get_recipe("unregistered-experiment")

    def test_versioned_manifest_and_replay_keep_the_explicit_recipe(self):
        selected = select_checkpoint(self.preflight, self.manifest_path)
        for key, value in recipe_identity(WARMUP_RECIPE_ID).items():
            self.assertEqual(selected["checkpoint_identity"][key], value)
            self.assertEqual(self.manifest[key], value)
        self.assertEqual(self.manifest["schema_version"], 2)
        fixtures = baseline_checks.ReplayChecks()
        fixtures.evaluator = evaluate_pose
        header, items = fixtures.replay_data(selected, [0.25])
        header["checkpoint_identity"] = copy.deepcopy(selected["checkpoint_identity"])
        predictions = self.run / "predictions.jsonl"
        fixtures.write_replay(predictions, header, items)
        evaluate_pose._read_replay(predictions, selected, None)
        header["checkpoint_identity"]["recipe_id"] = ORIGINAL_RECIPE_ID
        fixtures.write_replay(predictions, header, items)
        with self.assertRaisesRegex(ValueError, "checkpoint provenance differs"):
            evaluate_pose._read_replay(predictions, selected, None)

    def test_original_versioned_recipe_supports_both_label_arms(self):
        for arm in ("reference", "candidate"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                _, _, preflight, _, _, manifest = bundle_fixture(Path(directory), arm, recipe_id=ORIGINAL_RECIPE_ID)
                identity = select_checkpoint(preflight, manifest)["checkpoint_identity"]
                self.assertEqual(identity["arm"], arm)
                self.assertEqual(identity["recipe_id"], ORIGINAL_RECIPE_ID)

    def test_recipe_receipt_manifest_and_schema_cannot_disagree(self):
        cases = [("receipt", "recipe_id", "unknown"), ("receipt", "recipe_sha256", "0" * 64),
                 ("manifest", "recipe_id", ORIGINAL_RECIPE_ID), ("manifest", "recipe_sha256", "0" * 64),
                 ("receipt", "schema_version", 1), ("manifest", "schema_version", 1)]
        for target, key, value in cases:
            with self.subTest(target=target, key=key):
                receipt, manifest = copy.deepcopy(self.receipt), copy.deepcopy(self.manifest)
                (receipt if target == "receipt" else manifest)[key] = value
                self.save(receipt, manifest=manifest)
                with self.assertRaises(ValueError):
                    select_checkpoint(self.preflight, self.manifest_path)

    def test_fixed_defaults_and_unknown_settings_cannot_change(self):
        cases = [("warmup_bias_lr", 0.1), ("rle", 2.0), ("channels_last", True),
                 ("time", 1.0), ("distill_model", "another.pt"), ("workers", 1),
                 ("future_setting", "undeclared"), ("fraction", 1)]
        for key, value in cases:
            with self.subTest(setting=key):
                receipt = copy.deepcopy(self.receipt)
                receipt["training_settings"][key] = value
                self.save(receipt)
                with self.assertRaisesRegex(ValueError, "training setting"):
                    select_checkpoint(self.preflight, self.manifest_path)
        receipt = copy.deepcopy(self.receipt)
        receipt["training_settings"].pop("rle")
        self.save(receipt)
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            select_checkpoint(self.preflight, self.manifest_path)

    def test_warmup_recipe_requires_reference_labels(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["arm"] = "candidate"
        variant = self.staging["variants"]["candidate"]
        receipt["data_yaml_sha256"] = variant["data_yaml_sha256"]
        receipt["training_settings"]["data"] = str(Path(self.staging["output_root"]) / variant["data_yaml"])
        self.save(receipt)
        with self.assertRaisesRegex(ValueError, "requires the reference label arm"):
            select_checkpoint(self.preflight, self.manifest_path)

    def test_run_paths_and_input_verification_remain_bound_to_staging(self):
        for key, value in (("project", "/different/run"), ("save_dir", "/different/run/fit"),
                           ("model", "/different/initial.pt")):
            with self.subTest(setting=key):
                receipt = copy.deepcopy(self.receipt)
                receipt["training_settings"][key] = value
                self.save(receipt)
                with self.assertRaises(ValueError):
                    select_checkpoint(self.preflight, self.manifest_path)
        receipt = copy.deepcopy(self.receipt)
        receipt["inputs_reverified_after_worker"] = False
        self.save(receipt)
        with self.assertRaisesRegex(ValueError, "not reverified"):
            select_checkpoint(self.preflight, self.manifest_path)

    def test_device_and_backend_observations_are_required(self):
        for key in ("device_name", "device_capability", "backend"):
            with self.subTest(field=key):
                receipt = copy.deepcopy(self.receipt)
                receipt["runtime"].pop(key)
                self.save(receipt)
                with self.assertRaisesRegex(ValueError, "runtime inventory"):
                    select_checkpoint(self.preflight, self.manifest_path)
        receipt = copy.deepcopy(self.receipt)
        receipt["training_settings"]["device"] = "0"
        receipt["runtime"].update(device="cuda:0", device_name="Synthetic GPU fixture",
                                  device_capability=[8, 0], torch_cuda="13.0")
        self.save(receipt)
        select_checkpoint(self.preflight, self.manifest_path)
        for key, value in (("device_name", ""), ("device_capability", [8]),
                           ("device_capability", [True, 0]), ("device", "cpu")):
            with self.subTest(field=key, value=value):
                changed = copy.deepcopy(receipt)
                changed["runtime"][key] = value
                self.save(changed)
                with self.assertRaisesRegex(ValueError, "CUDA device"):
                    select_checkpoint(self.preflight, self.manifest_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
