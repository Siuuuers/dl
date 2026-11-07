#!/usr/bin/env python3
"""Compile the delivered notebook and execute setup with safe GPU/install substitutes."""

import ast
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError
from io import StringIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/baomai_box_repeatability.ipynb"


class RepeatabilityNotebookChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check-repeatability-notebook-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        notebook = json.loads(NOTEBOOK.read_text())
        self.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code"}
        self.commands = []

    def test_all_delivered_code_and_inline_preflight_compile_with_fixed_four_run_order(self):
        for identifier, source in self.cells.items():
            with self.subTest(cell=identifier):
                compile(source, identifier, "exec")
        inputs = ast.parse(self.cells["box-repeatability-inputs"])
        inline = next(ast.literal_eval(node.value) for node in inputs.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "input_preflight" for target in node.targets))
        compile(inline, "repeatability-input-preflight", "exec")
        order = []
        for identifier, source in self.cells.items():
            if identifier.startswith("box-repeatability-seed"):
                call = ast.parse(source).body[0].value
                self.assertEqual(call.func.id, "train_one")
                order.append((ast.literal_eval(call.args[0]), ast.literal_eval(call.args[1]),
                              next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "seed")))
        self.assertEqual(order, [("seed-43/reference", "reference", 43), ("seed-43/candidate", "candidate", 43),
                                 ("seed-44/reference", "reference", 44), ("seed-44/candidate", "candidate", 44)])
        workspace = self.cells["box-repeatability-workspace"]
        trial = self.cells["box-repeatability-trial"]
        self.assertIn('SOURCE_REVISION = "f087c35f74bf304eb61998edcaa2172fe1cba49c"', workspace)
        self.assertIn('drive.mount("/content/drive")', workspace)
        self.assertIn("MyDrive/baomai-box-repeatability-20261107", workspace)
        self.assertIn("restore_from_drive(folder)", trial)
        self.assertEqual(trial.count("save_to_drive(output)"), 2)
        self.assertIn('"--max-seconds", "600" if trial else "21600"', trial)
        self.assertIn("save_to_drive(RESULTS_ZIP)", self.cells["box-repeatability-package"])
        self.assertIn('"ultralytics==8.4.182"', self.cells["box-repeatability-install"])

    def execute_workspace(self, revision="a" * 40):
        tree = ast.parse(self.cells["box-repeatability-workspace"])
        workspace = self.base / "study"
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "SOURCE_REVISION":
                    node.value = ast.Constant(revision)
                if node.targets[0].id == "RUN_BASE":
                    node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(workspace))], keywords=[])
        ast.fix_missing_locations(tree)
        def fake_run(command, **kwargs):
            self.commands.append(command)
            if command[:2] == [sys.executable, "-c"]:
                compile(command[2], "repeatability-gpu-probe", "exec")
            else:
                self.assertEqual(command, ["nvidia-smi"])
            return SimpleNamespace(returncode=0)
        scope = {}
        with patch("subprocess.run", side_effect=fake_run), redirect_stdout(StringIO()):
            exec(compile(tree, "repeatability-workspace", "exec"), scope)
        return scope

    def test_actual_workspace_cell_creates_only_a_fresh_folder(self):
        scope = self.execute_workspace()
        self.assertTrue(scope["RUN_BASE"].is_dir())
        self.assertIsNone(scope["DRIVE_ROOT"])
        self.assertEqual(scope["RECIPE_BY_SEED"], {43: "box-supervision-warmup-seed43-v1", 44: "box-supervision-warmup-seed44-v1"})
        marker = scope["RUN_BASE"] / "preserve.txt"
        marker.write_text("preserve completed work")
        with self.assertRaises(FileExistsError):
            self.execute_workspace()
        self.assertEqual(marker.read_text(), "preserve completed work")

    def test_unpinned_workspace_refuses_before_gpu_or_filesystem_actions(self):
        with self.assertRaisesRegex(AssertionError, "full source commit pinned"):
            self.execute_workspace("SOURCE_REVISION_PLACEHOLDER")
        self.assertEqual(self.commands, [])
        self.assertFalse((self.base / "study").exists())

    def execute_install(self, lookup):
        pins = b"torch==2.14.1+cu130\ntorchvision==0.29.1+cu130\n"
        def record_run(command):
            command = list(map(str, command))
            if command[:5] == [sys.executable, "-m", "pip", "install", "-c"]:
                self.assertEqual(Path(command[5]).read_bytes(), pins)
                self.assertEqual(command[6:], ["ultralytics==8.4.182", "psutil"])
            else:
                self.assertEqual(command[:2], [sys.executable, "-c"])
                compile(command[2], "repeatability-ready-probe", "exec")
            self.commands.append(command)
        scope = {"RUN_BASE": self.base, "sys": sys, "run": record_run}
        memory = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=20 * 2**30))
        with patch("importlib.metadata.version", side_effect=lookup), \
                patch.dict(sys.modules, {"psutil": memory}), redirect_stdout(StringIO()):
            exec(compile(self.cells["box-repeatability-install"], "repeatability-install", "exec"), scope)
        return scope

    def test_actual_install_cell_constrains_existing_cuda_versions_before_pip(self):
        scope = self.execute_install({"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130"}.__getitem__)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(scope["HOST_RSS_LIMIT_MIB"], 12000)

    def test_missing_cuda_package_version_stops_before_install_or_partial_pins(self):
        def missing(name):
            if name == "torch":
                return "2.14.1+cu130"
            raise PackageNotFoundError(name)
        with self.assertRaises(PackageNotFoundError):
            self.execute_install(missing)
        self.assertFalse((self.base / "torch-constraints.txt").exists())
        self.assertEqual(self.commands, [])

    def finished_receipt(self, *, trial):
        receipt = {
            "status": "trial-complete" if trial else "complete",
            "optimizer_steps": 4,
            "optimizer_parameter_change_observed": True,
            "arm": "candidate" if trial else "reference",
            "recipe_id": "box-supervision-warmup-seed43-v1",
            "initial_checkpoint_sha256": "abc",
            "source": {"revision": "a" * 40, "working_tree_status": ""},
            "training_settings": {"seed": 43, "epochs": 30, "warmup_bias_lr": 0.0001},
            "batches_completed": 4,
            "completed_epochs": [] if trial else list(range(1, 31)),
            "checkpoint_selection": "none",
        }
        return receipt

    def write_run(self, path, *, trial):
        path.mkdir(parents=True)
        (path / "training-receipt.json").write_text(
            json.dumps(self.finished_receipt(trial=trial)), encoding="utf-8")
        (path / "train.log").write_text("log", encoding="utf-8")
        if not trial:
            (path / "checkpoint-manifest.json").write_text("{}", encoding="utf-8")

    def test_finished_drive_runs_restore_without_training_and_finished_copies_stay(self):
        workspace = self.execute_workspace()
        drive = self.base / "drive"
        workspace["DRIVE_ROOT"] = drive
        run_base = workspace["RUN_BASE"]
        trial_dir = run_base / "resource-trial"
        reference_dir = run_base / "seed-43" / "reference"
        self.write_run(trial_dir, trial=True)
        self.write_run(reference_dir, trial=False)
        with redirect_stdout(StringIO()):
            workspace["save_to_drive"](trial_dir)
            workspace["save_to_drive"](reference_dir)
        self.assertEqual((drive / "resource-trial" / "train.log").read_text(encoding="utf-8"), "log")
        self.assertTrue((self.base / "drive" / "resource-trial.complete").is_file())
        (drive / "resource-trial" / "train.log").write_text("keep", encoding="utf-8")
        with redirect_stdout(StringIO()):
            workspace["save_to_drive"](trial_dir)
        self.assertEqual((drive / "resource-trial" / "train.log").read_text(encoding="utf-8"), "keep")
        archive = run_base / "baomai-box-repeatability-results.zip"
        archive.write_bytes(b"zip-bytes")
        with redirect_stdout(StringIO()):
            workspace["save_to_drive"](archive)
        self.assertEqual((drive / archive.name).read_bytes(), b"zip-bytes")
        archive.write_bytes(b"different")
        with redirect_stdout(StringIO()):
            workspace["save_to_drive"](archive)
        self.assertEqual((drive / archive.name).read_bytes(), b"zip-bytes")
        partial = drive / "seed-44" / "candidate"
        partial.mkdir(parents=True)
        (partial / "training-receipt.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(workspace["restore_from_drive"]("seed-44/candidate"))
        self.assertFalse((run_base / "seed-44" / "candidate").exists())
        half = drive / "seed-44" / "reference"
        half.mkdir(parents=True)
        (half / "train.log").write_text("partial", encoding="utf-8")
        local_half = run_base / "seed-44" / "reference"
        local_half.mkdir(parents=True)
        (local_half / "train.log").write_text("full", encoding="utf-8")
        with redirect_stdout(StringIO()):
            workspace["save_to_drive"](local_half)
        self.assertEqual((half / "train.log").read_text(encoding="utf-8"), "full")
        self.assertTrue(Path(str(half) + ".complete").is_file())
        shutil.rmtree(trial_dir)
        shutil.rmtree(reference_dir)

        class TrainingStarted(RuntimeError):
            pass

        class Subprocess:
            PIPE = object()
            STDOUT = object()

            def Popen(self, *args, **kwargs):
                raise TrainingStarted("training started")

        tree = ast.parse(self.cells["box-repeatability-trial"])
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual([node.name for node in tree.body], ["train_one"])
        scope = {
            "RUN_BASE": run_base,
            "RECIPE_BY_SEED": workspace["RECIPE_BY_SEED"],
            "SOURCE_REVISION": workspace["SOURCE_REVISION"],
            "reference_manifest": {"model_sha256": "abc"},
            "json": json,
            "sys": sys,
            "subprocess": Subprocess(),
            "restore_from_drive": workspace["restore_from_drive"],
            "save_to_drive": workspace["save_to_drive"],
            "CANDIDATE_MANIFEST": "data/candidates/box-refinement-2026-11-07/manifest.json",
            "HOST_RSS_LIMIT_MIB": 12000,
            "REPO": run_base / "dl",
        }
        with redirect_stdout(StringIO()):
            exec(compile(tree, "repeatability-trial", "exec"), scope)
            restored_trial = scope["train_one"]("resource-trial", "candidate", seed=43, trial=True)
            restored_reference = scope["train_one"]("seed-43/reference", "reference", seed=43)
        self.assertEqual(restored_trial, trial_dir)
        self.assertEqual(restored_reference, reference_dir)
        self.assertEqual((trial_dir / "train.log").read_text(encoding="utf-8"), "keep")
        self.assertFalse((trial_dir / "checkpoint-manifest.json").exists())
        self.assertTrue((reference_dir / "checkpoint-manifest.json").is_file())
        bad = run_base / "seed-43" / "candidate"
        bad.mkdir(parents=True)
        (bad / "training-receipt.json").write_text("{}", encoding="utf-8")
        with redirect_stdout(StringIO()):
            with self.assertRaisesRegex(FileExistsError, "new folder name"):
                scope["train_one"]("seed-43/candidate", "candidate", seed=43)
            with self.assertRaisesRegex(TrainingStarted, "training started"):
                scope["train_one"]("seed-44/candidate", "candidate", seed=44)
        workspace["DRIVE_ROOT"] = None
        self.assertIsNone(workspace["save_to_drive"](trial_dir))
        self.assertIsNone(workspace["restore_from_drive"]("resource-trial"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
