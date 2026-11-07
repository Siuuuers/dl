#!/usr/bin/env python3
"""Compile the delivered notebook and execute setup with safe GPU/install substitutes."""

import ast
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError
from io import StringIO
import json
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
