#!/usr/bin/env python3
"""Focused pairing/recovery checks. Synthetic bundles do not prove GPU execution."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.dont_write_bytecode = True

from check_pose_checkpoint import bundle_fixture, write_json
from evaluate_pose import file_sha256
import package_partial_front_results as packaging
from package_gpu_results import INDEX_NAME
from pose_checkpoint import select_checkpoint, write_checkpoint_manifest
from train_pose_comparison import descriptor


class ProtocolChecks(unittest.TestCase):
    def test_real_protocol_overlay_bytes_and_bad_candidate_hash(self):
        root = Path(__file__).resolve().parents[1]
        context = packaging.validate_protocol(root)
        self.assertEqual(context["delta_train_image_ids"], list(packaging.DELTA_IMAGES))
        self.assertEqual(context["candidate_checks"]["old_reviewed"]["changed_train_labels"], 8)
        self.assertEqual(context["candidate_checks"]["extended_reviewed"]["changed_train_labels"], 13)
        bad = copy.deepcopy(context["document"])
        bad["roles"]["extended_reviewed"]["candidate_manifest_sha256"] = "0" * 64
        original_read = packaging._read_json
        def read(path):
            return bad if Path(path) == root / packaging.DEFAULT_PROTOCOL else original_read(path)
        with mock.patch.object(packaging, "_read_json", side_effect=read):
            with self.assertRaisesRegex(ValueError, "candidate manifest SHA-256 differs"):
                packaging.validate_protocol(root)


class PairPackagingChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.old, self.preflight, receipt, stage, _ = bundle_fixture(
            self.base, arm="candidate", recipe_id=packaging.RECIPE_ID)
        self.extended = self.base / "extended-run"
        shutil.copytree(self.old, self.extended)
        self.delta_ids = tuple(f"train_{i:03d}" for i in range(1, 6))
        # Explicitly synthetic CUDA metadata exercises provenance gates only.
        receipt["runtime"].update(device="cuda:0", device_name="synthetic CUDA fixture; no GPU run",
                                  device_capability=[8, 0], torch_cuda="fixture")
        receipt["training_settings"]["device"] = "0"
        old_proposal = self.root / stage["source"]["candidate_manifest"]
        new_proposal = self.root / "data/candidates/extended-fixture/manifest.json"
        new_proposal.parent.mkdir(parents=True)
        new = json.loads(old_proposal.read_text())
        new["candidate_id"] = "extended-fixture"
        records = {r["id"]: r for r in json.loads((self.root / "data/manifest.json").read_text())["records"]}
        new_hash = hashlib.sha256(b"synthetic extended annotation; no real image review").hexdigest()
        for image_id in self.delta_ids:
            r = records[image_id]
            new["changes"].append({"image_id": image_id, "source_label": r["label"],
                                   "source_label_sha256": r["label_sha256"],
                                   "candidate_label_sha256": new_hash})
        write_json(new_proposal, new)
        self.receipts, self.stages = {}, {}
        for role, path in zip(packaging.ROLES, (self.old, self.extended)):
            s, r = copy.deepcopy(stage), copy.deepcopy(receipt)
            s["output_root"] = str(path / "staged-data")
            if role == "extended_reviewed":
                s["source"].update(candidate_id=new["candidate_id"],
                                   candidate_manifest=new_proposal.relative_to(self.root).as_posix(),
                                   candidate_manifest_sha256=file_sha256(new_proposal))
                for entry in s["variants"]["candidate"]["files"]:
                    if entry["path"] in {records[i]["label"] for i in self.delta_ids}:
                        entry["source_sha256"] = entry["output_sha256"] = new_hash
            r["training_settings"].update(data=str(path / "staged-data/candidate/data.yaml"),
                                          project=str(path), save_dir=str(path / "fit"))
            checkpoint = path / "fit/weights/last.pt"
            checkpoint.write_bytes(f"Synthetic {role} weights; never deserialize.\n".encode())
            r["checkpoint"] = descriptor(checkpoint, path, checkpoint=True)
            write_json(path / "staging-manifest.json", s)
            r["staging_manifest_sha256"] = file_sha256(path / "staging-manifest.json")
            write_json(path / "training-receipt.json", r)
            (path / "checkpoint-manifest.json").unlink()
            write_checkpoint_manifest(path, self.preflight)
            self.receipts[role], self.stages[role] = r, s
        self.context = {"path": "synthetic-protocol.json", "sha256": "c" * 64,
                        "records": records, "delta_train_image_ids": list(self.delta_ids),
                        "document": {"experiment_id": "synthetic-package-check", "roles": {}},
                        "effective_label_sha256": {}}
        for role in packaging.ROLES:
            source = self.stages[role]["source"]
            self.context["document"]["roles"][role] = {k: source[k] for k in (
                "candidate_manifest", "candidate_manifest_sha256", "candidate_id")}
            inventory = {x["path"]: x["output_sha256"]
                         for x in self.stages[role]["variants"]["candidate"]["files"]}
            self.context["effective_label_sha256"][role] = {
                image_id: inventory[r["label"]] for image_id, r in records.items()}
        self.output = self.base / "return.zip"
        for name, value in (("INITIAL_SHA256", self.preflight["model_sha256"]),
                            ("DELTA_IMAGES", self.delta_ids)):
            patch = mock.patch.object(packaging, name, value)
            patch.start(); self.addCleanup(patch.stop)
        for name, value in (("validate_protocol", self.context),
                            ("source_identity", self.receipts["old_reviewed"]["source"])):
            patch = mock.patch.object(packaging, name, return_value=value)
            patch.start(); self.addCleanup(patch.stop)

    def package(self, **changes):
        return packaging.package_results(**dict(root=self.root, old_reviewed=self.old,
            extended_reviewed=self.extended, output=self.output, **changes))

    def save_extended_receipt(self, receipt):
        write_json(self.extended / "training-receipt.json", receipt)
        manifest = json.loads((self.extended / "checkpoint-manifest.json").read_text())
        manifest["training_receipt"] = descriptor(self.extended / "training-receipt.json", self.extended)
        write_json(self.extended / "checkpoint-manifest.json", manifest)

    def test_truthful_candidate_arms_roundtrip_and_existing_zip_is_read_only(self):
        result = self.package()
        before = self.output.read_bytes()
        self.assertEqual(result["files"], 13)
        with zipfile.ZipFile(self.output) as archive:
            index = json.loads(archive.read(INDEX_NAME))
            self.assertEqual(index["kind"], packaging.RESULT_KIND)
            self.assertEqual(set(index["roles"]), set(packaging.ROLES))
            self.assertFalse(any(name.startswith("reference/") for name in archive.namelist()))
            archive.extractall(self.base / "relocated")
        for role in packaging.ROLES:
            selected = select_checkpoint(self.preflight, self.base / "relocated" / role / "checkpoint-manifest.json")
            self.assertEqual(selected["checkpoint_identity"]["arm"], "candidate")
        self.assertEqual(self.package(verify_existing=True)["status"], "verified_existing")
        self.assertEqual(self.output.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            self.package()

    def test_swapped_candidate_roles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "staged candidate_"):
            packaging.package_results(self.root, self.extended, self.old, self.output)
        self.assertFalse(self.output.exists())

    def test_changed_runtime_and_source_are_rejected(self):
        original = self.receipts["extended_reviewed"]
        for field, key, value in (("runtime", "device_name", "different synthetic GPU"),
                                  ("source", "revision", "4" * 40)):
            with self.subTest(field=field):
                receipt = copy.deepcopy(original); receipt[field][key] = value
                self.save_extended_receipt(receipt)
                with self.assertRaises(ValueError):
                    self.package()
                self.assertFalse(self.output.exists())

    def test_incomplete_and_resource_trial_bundles_are_rejected(self):
        for change in ({"completed_epochs": list(range(1, 30))},
                       {"purpose": "resource-trial", "status": "trial-complete"}):
            with self.subTest(change=change):
                self.save_extended_receipt({**copy.deepcopy(self.receipts["extended_reviewed"]), **change})
                with self.assertRaises(ValueError):
                    self.package()
                self.assertFalse(self.output.exists())

    def test_pair_inventory_rejects_validation_or_undeclared_label_changes(self):
        bundles = {role: packaging.collect_reviewed_run(self.preflight, path, role, self.context)
                   for role, path in zip(packaging.ROLES, (self.old, self.extended))}
        for target in ("labels/val/", "labels/train/train_006.txt", "images/train/"):
            with self.subTest(target=target):
                bad = copy.deepcopy(bundles)
                files = bad["extended_reviewed"]["staging"]["variants"]["candidate"]["files"]
                entry = next(x for x in files if x["path"].startswith(target))
                entry["output_sha256"] = "e" * 64
                with self.assertRaisesRegex(ValueError, "inventory differs"):
                    packaging.check_reviewed_pair(bad, self.context)


class NotebookRecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks/baomai_partial_fronts.ipynb"
        cls.notebook = json.loads(path.read_text())
        cls.codes = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
                     for c in cls.notebook["cells"] if c["cell_type"] == "code"]

    def test_all_cells_and_embedded_preflight_compile_and_constraints_have_newlines(self):
        for i, source in enumerate(self.codes):
            compile(source, f"notebook-cell-{i}", "exec")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                    and t.id in ("input_preflight", "probe") for t in node.targets):
                    compile(ast.literal_eval(node.value), "embedded-code", "exec")
        install = ast.parse(self.codes[2])
        node = next(n for n in ast.walk(install) if isinstance(n, ast.JoinedStr))
        self.assertTrue(any(isinstance(v, ast.Constant) and v.value == "\n" for v in node.values))

    def test_unfilled_source_pin_stops_before_subprocess_or_workspace(self):
        module = ast.parse(self.codes[0])
        for node in module.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_REVISION" for t in node.targets):
                node.value = ast.Constant("UNFILLED")
        ast.fix_missing_locations(module)
        with mock.patch.object(subprocess, "run") as external:
            with self.assertRaisesRegex(AssertionError, "source commit pinned"):
                exec(compile(module, "draft-pin-gate", "exec"), {})
            external.assert_not_called()

    def functions(self, index, names, scope):
        nodes = [n for n in ast.parse(self.codes[index]).body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-functions", "exec"), scope)

    def test_completed_backup_restores_only_required_artifacts_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); local = base / "local"; drive = base / "drive"
            local.mkdir(); drive.mkdir(); arm = local / "old_reviewed"; arm.mkdir()
            (arm / "fit/weights").mkdir(parents=True)
            for name in ("fit/weights/last.pt", "fit/results.csv", "train.log", "staging-manifest.json"):
                (arm / name).write_bytes(b"synthetic artifact; no training\n")
            write_json(arm / "training-receipt.json", {"log": {"path": "train.log"}, "results": {"path": "fit/results.csv"}})
            write_json(arm / "checkpoint-manifest.json", {"checkpoint": {"path": "fit/weights/last.pt"}})
            (arm / "not-required-large-staging-copy").write_bytes(b"exclude")
            scope = dict(Path=Path, hashlib=hashlib, json=json, shutil=shutil,
                         RUN_BASE=local, DRIVE_ROOT=drive, workspace_identity={"source_revision":"1"*40,"protocol_sha256":"2"*64})
            self.functions(0, {"sha256", "safe_member", "artifact_inventory", "validate_copy", "save_to_drive", "restore_from_drive"}, scope)
            scope["save_to_drive"](arm)
            shutil.rmtree(arm)
            scope["restore_from_drive"]("old_reviewed")
            self.assertEqual(len([p for p in arm.rglob("*") if p.is_file()]), 6)
            self.assertFalse((arm / "not-required-large-staging-copy").exists())
            shutil.rmtree(arm)
            (drive / "old_reviewed/fit/weights/last.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(AssertionError, "Backup hash mismatch"):
                scope["restore_from_drive"]("old_reviewed")
            self.assertFalse(arm.exists())

    def test_completed_arm_skips_training_and_partial_arm_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); (base / "old_reviewed").mkdir()
            scope = dict(Path=Path, RUN_BASE=base, ROLES=packaging.ROLES,
                         assert_finished=mock.Mock(return_value={"status":"complete"}),
                         save_to_drive=mock.Mock(), subprocess=subprocess)
            self.functions(3, {"train_one"}, scope)
            with mock.patch.object(subprocess, "Popen") as launch:
                scope["train_one"]("old_reviewed")
                launch.assert_not_called()
                scope["assert_finished"].side_effect = ValueError("incomplete")
                with self.assertRaises(FileExistsError):
                    scope["train_one"]("old_reviewed")
                launch.assert_not_called()

    def test_fresh_resource_trials_preserve_separate_drive_backups(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); local = base / "local"; drive = base / "drive"
            local.mkdir(); drive.mkdir(); trial = local / "resource-trial"; trial.mkdir()
            (trial / "train.log").write_text("Synthetic trial; no training executed.\n")
            (trial / "staging-manifest.json").write_text("{}\n")
            receipt = {"log": {"path": "train.log"}, "created_at_utc": "first synthetic session"}
            write_json(trial / "training-receipt.json", receipt)
            first_bytes = (trial / "training-receipt.json").read_bytes()
            scope = dict(Path=Path, hashlib=hashlib, json=json, shutil=shutil,
                         RUN_BASE=local, DRIVE_ROOT=drive,
                         workspace_identity={"source_revision":"1"*40,"protocol_sha256":"2"*64})
            self.functions(0, {"sha256", "safe_member", "artifact_inventory", "validate_copy", "save_to_drive"}, scope)
            scope["save_to_drive"](trial)
            first = next(p for p in (drive / "resource-trials").iterdir() if p.is_dir())
            receipt["created_at_utc"] = "second synthetic session"
            write_json(trial / "training-receipt.json", receipt)
            scope["save_to_drive"](trial)
            self.assertEqual(len([p for p in (drive / "resource-trials").iterdir() if p.is_dir()]), 2)
            self.assertEqual((first / "training-receipt.json").read_bytes(), first_bytes)

    def test_reused_local_trial_runtime_is_checked_before_full_training(self):
        current = {"python": "synthetic", "torch": "fixture", "device": "cuda:0",
                   "device_name": "new synthetic GPU", "cudnn_version": 90000}
        runtime = {**current, "device_name": "old synthetic GPU",
                   "backend": {"cudnn_version": current["cudnn_version"]}}
        runtime.pop("cudnn_version")
        scope = dict(COMPLETED={}, DRIVE_ROOT=None, TRIAL_RUNTIME=None,
                     subprocess=mock.Mock(), sys=sys, json=json,
                     REPO=Path("/synthetic/repo"),
                     train_one=mock.Mock(return_value=(Path("/synthetic/resource-trial"), {"runtime": runtime})))
        scope["subprocess"].check_output.return_value = json.dumps(current)
        self.functions(3, {"assert_current_runtime"}, scope)
        gate = next(n for n in ast.parse(self.codes[3]).body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Call)
                    and isinstance(n.test.left.func, ast.Name)
                    and n.test.left.func.id == "len")
        code = compile(ast.Module(body=[gate], type_ignores=[]), "notebook-pre-full-run-gate", "exec")
        with self.assertRaisesRegex(AssertionError, "differs from resource trial"):
            exec(code, scope)
        scope["train_one"].assert_called_once_with("extended_reviewed", trial=True)
        self.assertIsNone(scope["TRIAL_RUNTIME"])
        runtime["device_name"] = current["device_name"]
        exec(code, scope)
        self.assertEqual(scope["TRIAL_RUNTIME"], runtime)

    def test_unusable_drive_trial_does_not_block_a_fresh_trial(self):
        current = {"python": "synthetic", "torch": "fixture", "device": "cuda:0",
                   "device_name": "synthetic GPU", "cudnn_version": 90000}
        runtime = {**current, "backend": {"cudnn_version": current["cudnn_version"]}}
        runtime.pop("cudnn_version")
        with tempfile.TemporaryDirectory() as temporary:
            drive = Path(temporary)
            (drive / "resource-trials").mkdir()
            (drive / "resource-trials" / "deadbeef.complete").write_text("{not json")
            scope = dict(COMPLETED={}, DRIVE_ROOT=drive, TRIAL_RUNTIME=None,
                         subprocess=mock.Mock(), sys=sys, json=json, REPO=Path("/synthetic/repo"),
                         restore_from_drive=mock.Mock(side_effect=AssertionError(
                             "Drive completion belongs to different source/protocol pins.")),
                         train_one=mock.Mock(return_value=(Path("/synthetic/resource-trial"), {"runtime": runtime})))
            scope["subprocess"].check_output.return_value = json.dumps(current)
            self.functions(3, {"assert_current_runtime"}, scope)
            gate = next(n for n in ast.parse(self.codes[3]).body if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Compare)
                        and isinstance(n.test.left, ast.Call)
                        and isinstance(n.test.left.func, ast.Name)
                        and n.test.left.func.id == "len")
            exec(compile(ast.Module(body=[gate], type_ignores=[]), "notebook-pre-full-run-gate", "exec"), scope)
            scope["restore_from_drive"].assert_called_once_with("resource-trials/deadbeef")
            scope["train_one"].assert_called_once_with("extended_reviewed", trial=True)
            self.assertEqual(scope["TRIAL_RUNTIME"], runtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
