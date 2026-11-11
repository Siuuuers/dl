#!/usr/bin/env python3
"""Check role, inventory and recovery gates with synthetic bundles; no training."""

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
import package_visible_extent_results as packaging
from package_gpu_results import INDEX_NAME
from pose_checkpoint import select_checkpoint, write_checkpoint_manifest
from train_pose_comparison import descriptor


class ProtocolChecks(unittest.TestCase):
    def test_real_candidate_inheritance_delta_and_reviewed_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        context = packaging.validate_protocol(root)
        self.assertEqual(context["delta_train_image_ids"], [
            "new_000061", "new_000079", "new_000082", "new_000094"])
        self.assertEqual(context["document"]["reviewed_unchanged_train_image_ids"], [
            "new_000062", "new_000063"])
        self.assertEqual(context["candidate_checks"]["old_reviewed"]["changed_train_labels"], 8)
        self.assertEqual(context["candidate_checks"]["extended_reviewed"]["changed_train_labels"], 12)
        original_read = packaging._read_json
        for change in ("bad_candidate_hash", "unreviewed_delta", "invented_changed_image", "draft"):
            with self.subTest(change=change):
                bad = copy.deepcopy(context["document"])
                if change == "bad_candidate_hash":
                    bad["roles"]["extended_reviewed"]["candidate_manifest_sha256"] = "0" * 64
                elif change == "unreviewed_delta":
                    bad["delta_train_image_ids"].append("new_000095")
                elif change == "invented_changed_image":
                    bad["delta_train_image_ids"].insert(1, "new_000062")
                    bad["reviewed_unchanged_train_image_ids"] = ["new_000063"]
                else:
                    bad["status"] = "draft"
                def read(path):
                    return bad if Path(path) == root / packaging.DEFAULT_PROTOCOL else original_read(path)
                with mock.patch.object(packaging, "_read_json", side_effect=read):
                    with self.assertRaises(ValueError):
                        packaging.validate_protocol(root)


class PairChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.old, self.preflight, receipt, stage, _ = bundle_fixture(
            self.base, arm="candidate", recipe_id=packaging.RECIPE_ID)
        self.extended = self.base / "extended-run"
        shutil.copytree(self.old, self.extended)
        delta = [f"train_{i:03d}" for i in range(1, 5)]
        receipt["runtime"].update(device="cuda:0", device_name="synthetic CUDA fixture; no GPU run",
                                  device_capability=[8, 0], torch_cuda="fixture")
        receipt["training_settings"]["device"] = "0"
        old_proposal = self.root / stage["source"]["candidate_manifest"]
        new_proposal = self.root / "data/candidates/extent-fixture/manifest.json"
        new_proposal.parent.mkdir(parents=True)
        new = json.loads(old_proposal.read_text())
        new["candidate_id"] = "extent-fixture"
        records = {r["id"]: r for r in json.loads((self.root / "data/manifest.json").read_text())["records"]}
        changed_hash = hashlib.sha256(b"synthetic annotation; no physical review").hexdigest()
        for image_id in delta:
            record = records[image_id]
            new["changes"].append({"image_id": image_id, "source_label": record["label"],
                                   "source_label_sha256": record["label_sha256"],
                                   "candidate_label_sha256": changed_hash})
        write_json(new_proposal, new)
        self.receipts, self.stages = {}, {}
        for role, path in zip(packaging.ROLES, (self.old, self.extended)):
            staged, returned = copy.deepcopy(stage), copy.deepcopy(receipt)
            staged["output_root"] = str(path / "staged-data")
            if role == "extended_reviewed":
                staged["source"].update(candidate_id=new["candidate_id"],
                    candidate_manifest=new_proposal.relative_to(self.root).as_posix(),
                    candidate_manifest_sha256=file_sha256(new_proposal))
                for entry in staged["variants"]["candidate"]["files"]:
                    if entry["path"] in {records[i]["label"] for i in delta}:
                        entry["source_sha256"] = entry["output_sha256"] = changed_hash
            returned["training_settings"].update(data=str(path / "staged-data/candidate/data.yaml"),
                                                   project=str(path), save_dir=str(path / "fit"))
            checkpoint = path / "fit/weights/last.pt"
            checkpoint.write_bytes(f"Synthetic {role} weights; never deserialize.\n".encode())
            returned["checkpoint"] = descriptor(checkpoint, path, checkpoint=True)
            write_json(path / "staging-manifest.json", staged)
            returned["staging_manifest_sha256"] = file_sha256(path / "staging-manifest.json")
            write_json(path / "training-receipt.json", returned)
            (path / "checkpoint-manifest.json").unlink()
            write_checkpoint_manifest(path, self.preflight)
            self.receipts[role], self.stages[role] = returned, staged
        self.context = {"path": "synthetic-protocol.json", "sha256": "c" * 64,
            "records": records, "delta_train_image_ids": delta,
            "document": {"experiment_id": "synthetic-extent-check", "roles": {}},
            "effective_label_sha256": {}}
        for role in packaging.ROLES:
            source = self.stages[role]["source"]
            self.context["document"]["roles"][role] = {k: source[k] for k in (
                "candidate_manifest", "candidate_manifest_sha256", "candidate_id")}
            inventory = {x["path"]: x["output_sha256"]
                         for x in self.stages[role]["variants"]["candidate"]["files"]}
            self.context["effective_label_sha256"][role] = {
                image_id: inventory[record["label"]] for image_id, record in records.items()}
        self.output = self.base / "return.zip"
        for name, value in (("INITIAL_SHA256", self.preflight["model_sha256"]),):
            patch = mock.patch.object(packaging, name, value)
            patch.start(); self.addCleanup(patch.stop)
        for name, value in (("validate_protocol", self.context),
                           ("source_identity", self.receipts["old_reviewed"]["source"])):
            patch = mock.patch.object(packaging, name, return_value=value)
            patch.start(); self.addCleanup(patch.stop)

    def package(self, **changes):
        return packaging.package_results(**dict(root=self.root, old_reviewed=self.old,
            extended_reviewed=self.extended, output=self.output, **changes))

    def save_receipt(self, returned):
        write_json(self.extended / "training-receipt.json", returned)
        manifest = json.loads((self.extended / "checkpoint-manifest.json").read_text())
        manifest["training_receipt"] = descriptor(self.extended / "training-receipt.json", self.extended)
        write_json(self.extended / "checkpoint-manifest.json", manifest)

    def test_both_candidate_roles_roundtrip_and_existing_zip_read_only(self):
        result = self.package(); before = self.output.read_bytes()
        self.assertEqual(result["files"], 13)
        with zipfile.ZipFile(self.output) as archive:
            index = json.loads(archive.read(INDEX_NAME))
            self.assertEqual(index["kind"], packaging.RESULT_KIND)
            self.assertFalse(any(name.startswith("reference/") for name in archive.namelist()))
            archive.extractall(self.base / "relocated")
        for role in packaging.ROLES:
            chosen = select_checkpoint(self.preflight, self.base / "relocated" / role / "checkpoint-manifest.json")
            self.assertEqual(chosen["checkpoint_identity"]["arm"], "candidate")
        self.assertEqual(self.package(verify_existing=True)["status"], "verified_existing")
        self.assertEqual(before, self.output.read_bytes())
        with self.assertRaises(FileExistsError): self.package()

    def test_swapped_roles_and_unmatched_runtime_rejected(self):
        with self.assertRaises(ValueError):
            packaging.package_results(self.root, self.extended, self.old, self.output)
        returned = copy.deepcopy(self.receipts["extended_reviewed"])
        returned["runtime"]["device_name"] = "different synthetic device"
        self.save_receipt(returned)
        with self.assertRaises(ValueError): self.package()
        self.assertFalse(self.output.exists())

    def test_partial_trial_or_undeclared_input_delta_rejected(self):
        bundles = {role: packaging.collect_reviewed_run(self.preflight, path, role, self.context)
                   for role, path in zip(packaging.ROLES, (self.old, self.extended))}
        for target in ("labels/val/", "images/train/", "labels/train/train_006.txt"):
            with self.subTest(target=target):
                bad = copy.deepcopy(bundles)
                item = next(x for x in bad["extended_reviewed"]["staging"]["variants"]["candidate"]["files"]
                            if x["path"].startswith(target))
                item["output_sha256"] = "e" * 64
                with self.assertRaises(ValueError): packaging.check_reviewed_pair(bad, self.context)
        for change in ({"completed_epochs": list(range(1, 30))},
                       {"purpose": "resource-trial", "status": "trial-complete"}):
            self.save_receipt({**copy.deepcopy(self.receipts["extended_reviewed"]), **change})
            with self.assertRaises(ValueError): self.package()
        self.assertFalse(self.output.exists())


class NotebookChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/baomai_visible_extent.ipynb").read_text())
        cls.codes = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                     for c in notebook["cells"] if c["cell_type"] == "code"]

    def test_cells_embedded_preflight_and_unfilled_pin_gate(self):
        for source in self.codes:
            compile(source, "notebook-cell", "exec")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
                    "input_preflight", "probe") for t in node.targets):
                    compile(ast.literal_eval(node.value), "embedded-preflight", "exec")
        module = ast.parse(self.codes[0])
        for node in module.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_REVISION" for t in node.targets):
                node.value = ast.Constant("UNFILLED")
        ast.fix_missing_locations(module)
        with mock.patch.object(subprocess, "run") as launch:
            with self.assertRaisesRegex(AssertionError, "source commit pinned"):
                exec(compile(module, "source-gate", "exec"), {})
            launch.assert_not_called()

    def test_saved_trial_reuse_invalid_fallback_and_partial_local_rejection(self):
        function = next(n for n in ast.parse(self.codes[3]).body if isinstance(n, ast.FunctionDef) and n.name == "recover_trial")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); drive = base / "drive"; (drive / "resource-trials").mkdir(parents=True)
            (drive / "resource-trials/old.complete").write_text("synthetic")
            receipt = {"runtime": {"device_name": "synthetic matching GPU"}}
            scope = dict(RUN_BASE=base / "local", DRIVE_ROOT=drive, COMPLETED={"old_reviewed": receipt},
                restore_from_drive=mock.Mock(return_value=base / "restored"),
                assert_finished=mock.Mock(return_value=receipt), assert_current_runtime=mock.Mock(),
                train_one=mock.Mock(return_value=(base / "fresh", receipt)))
            exec(compile(ast.Module(body=[function], type_ignores=[]), "recovery", "exec"), scope)
            self.assertEqual(scope["recover_trial"]({})[0], base / "restored")
            scope["train_one"].assert_not_called()
            scope["assert_finished"].side_effect = ValueError("bad receipt")
            self.assertEqual(scope["recover_trial"]({})[0], base / "fresh")
            scope["train_one"].assert_called_once_with("extended_reviewed", trial=True)
            (base / "local/resource-trial").mkdir(parents=True)
            scope["train_one"].side_effect = FileExistsError("partial local trial")
            with self.assertRaises(FileExistsError): scope["recover_trial"]({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
