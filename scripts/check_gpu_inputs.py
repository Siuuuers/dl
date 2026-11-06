#!/usr/bin/env python3
"""Focused transfer-integrity and no-overwrite checks; no ML dependencies."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

from package_gpu_inputs import descriptor, package_inputs
from restore_gpu_inputs import (CHECKPOINT, KIND, SOURCE_METADATA, read_package,
                                refresh_verified_assets, relative_name, restore_inputs, run)


class InputPackageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check-gpu-inputs-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.package.mkdir()
        self.output = self.root / "restored"
        self.manifest = {"schema_version": 1, "kind": KIND,
                         "source_revision": "1" * 40, "source_tree": "2" * 40,
                         "source_metadata_sha256": {path: "3" * 64 for path in SOURCE_METADATA},
                         "counts": {}, "assets": []}
        for key, relative in (("source_bundle", "source.bundle"),
                              ("bootstrap", "restore_gpu_inputs.py"), ("guide", "README.md")):
            path = self.package / relative
            path.write_bytes(b"fixture")
            self.manifest[key] = descriptor(path, relative)
        for split, count in (("train", 137), ("val", 20)):
            for index in range(count):
                self.add_asset(f"data/images/{split}/fixture_{index:03d}.png")
        self.add_asset(CHECKPOINT)
        self.save_manifest()

    def add_asset(self, relative):
        path = self.package / "assets" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture payload\r\n")
        self.manifest["assets"].append({**descriptor(path, "assets/" + relative),
                                        "restore_path": relative})

    def save_manifest(self):
        raw = json.dumps(self.manifest, sort_keys=True).encode("utf-8")
        (self.package / "manifest.json").write_bytes(raw)
        (self.package / "manifest.sha256").write_text(
            hashlib.sha256(raw).hexdigest() + "  manifest.json\n", encoding="ascii")

    def test_payload_and_manifest_damage_fail_before_creating_output(self):
        self.assertEqual(read_package(self.package), self.manifest)
        asset = self.package / self.manifest["assets"][0]["path"]
        asset.write_bytes(b"fixture payload\n!")  # Same size, different raw bytes.
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            restore_inputs(self.package, self.output)
        self.assertFalse(self.output.exists())
        asset.write_bytes(b"fixture payload\r\n")
        with (self.package / "manifest.json").open("ab") as target:
            target.write(b" ")
        with self.assertRaisesRegex(ValueError, "manifest.sha256"):
            restore_inputs(self.package, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_or_out_of_scope_asset_is_rejected(self):
        original = copy.deepcopy(self.manifest)
        self.manifest["assets"][1] = copy.deepcopy(self.manifest["assets"][0])
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate restore path"):
            read_package(self.package)
        self.manifest = original
        self.manifest["assets"][0]["restore_path"] = ".git/config"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "unexpected GPU asset"):
            read_package(self.package)

    def test_symlink_and_escaping_paths_are_rejected(self):
        asset = self.package / self.manifest["assets"][0]["path"]
        outside = self.root / "outside.png"
        outside.write_bytes(asset.read_bytes())
        asset.unlink()
        asset.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            read_package(self.package)
        for relative in ("../outside.png", "/outside.png", "assets/../outside.png",
                         "assets\\outside.png", "assets//outside.png", "."):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                relative_name(relative)

    def test_existing_outputs_are_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "user-file.txt"
        sentinel.write_bytes(b"preserve this\r\n")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            restore_inputs(self.package, self.output)
        self.assertEqual(sentinel.read_bytes(), b"preserve this\r\n")
        archive = self.root / "existing.zip"
        archive.write_bytes(b"existing archive")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            package_inputs(archive)
        self.assertEqual(archive.read_bytes(), b"existing archive")

    def test_lfs_payload_overlay_refresh_preserves_exact_source_tree(self):
        source = Path(__file__).resolve().parents[1]
        clone = self.root / "lfs-checkout"
        run(["git", "clone", "--no-checkout", str(source), str(clone)])
        run(["git", "lfs", "install", "--local"], cwd=clone)
        revision = run(["git", "rev-parse", "HEAD"], cwd=source).decode("ascii").strip()
        run(["git", "checkout", "--detach", revision], cwd=clone)
        self.assertEqual(run(["git", "status", "--porcelain=v1"], cwd=clone), b"")
        relative = "data/images/train/new_000001.png"
        self.assertTrue((clone / relative).read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/v1\n"))
        shutil.copyfile(source / relative, clone / relative)
        asset = {**descriptor(source / relative, "assets/" + relative), "restore_path": relative}
        tree = run(["git", "rev-parse", "HEAD^{tree}"], cwd=clone).decode("ascii").strip()
        refresh_verified_assets(clone, {"source_tree": tree, "assets": [asset]})
        self.assertEqual(run(["git", "write-tree"], cwd=clone).decode("ascii").strip(), tree)
        self.assertEqual(run(["git", "status", "--porcelain=v1"], cwd=clone), b"")
        (clone / relative).write_bytes(b"changed payload")
        with self.assertRaisesRegex(ValueError, "byte size mismatch"):
            refresh_verified_assets(clone, {"source_tree": tree, "assets": [asset]})
        self.assertEqual(run(["git", "write-tree"], cwd=clone).decode("ascii").strip(), tree)


if __name__ == "__main__":
    unittest.main()
