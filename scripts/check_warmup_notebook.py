#!/usr/bin/env python3
"""Execute the notebook's setup cell with safe install/GPU substitutes."""

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


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/baomai_bias_warmup.ipynb"


class WarmupNotebookChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check-warmup-notebook-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cell = next(cell for cell in notebook["cells"] if cell["id"] == "baomai-warmup-06")
        self.source = "".join(cell["source"])
        self.commands = []
        self.expected_pins = b"torch==2.14.1+cu130\ntorchvision==0.29.1+cu130\n"

    def record_run(self, command):
        command = list(map(str, command))
        if command[:5] == [sys.executable, "-m", "pip", "install", "-c"]:
            self.assertEqual(command[5], str(self.base / "torch-constraints.txt"))
            self.assertEqual(Path(command[5]).read_bytes(), self.expected_pins)
            self.assertEqual(command[6:], ["ultralytics==8.4.182", "psutil"])
        else:
            self.assertEqual(command[:2], [sys.executable, "-c"])
            compile(command[2], "notebook-inline-python", "exec")
        self.commands.append(command)

    def execute_setup(self, lookup):
        scope = {"RUN_BASE": self.base, "sys": sys, "run": self.record_run}
        memory = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=20 * 2**30))
        with patch("importlib.metadata.version", side_effect=lookup), \
                patch.dict(sys.modules, {"psutil": memory}), redirect_stdout(StringIO()):
            exec(compile(self.source, "notebook-setup-cell", "exec"), scope)
        return scope

    def test_installed_cuda_pins_exist_before_pip_and_inline_python_compiles(self):
        versions = {"torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130"}
        scope = self.execute_setup(versions.__getitem__)
        self.assertEqual((self.base / "torch-constraints.txt").read_bytes(), self.expected_pins)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[0][1:5], ["-m", "pip", "install", "-c"])
        self.assertEqual(self.commands[1][1], "-c")
        self.assertEqual(scope["HOST_RSS_LIMIT_MIB"], 12000)

    def test_missing_version_stops_before_installation_or_partial_pins(self):
        def missing_torchvision(name):
            if name == "torch":
                return "2.14.1+cu130"
            raise PackageNotFoundError(name)

        with self.assertRaises(PackageNotFoundError):
            self.execute_setup(missing_torchvision)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.base / "torch-constraints.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
