#!/usr/bin/env python3
"""Check the real composed protocol and synthetic return transport; no GPU runs."""

import sys
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from evaluate_pose import file_sha256
import package_composed_results as packaging


ROOT = Path(__file__).resolve().parents[1]


class ProtocolChecks(unittest.TestCase):
    def test_real_composed_control_delta_and_frozen_evaluation(self):
        context = packaging.validate_protocol(ROOT)
        roles = context["document"]["roles"]
        self.assertEqual(roles["reference"]["counts"]["train"]["objects"], 3771)
        self.assertEqual(roles["candidate"]["counts"]["train"]["objects"], 3770)
        self.assertEqual(roles["candidate"]["counts"]["train"]["visible_endpoints"], 6842)
        self.assertEqual(roles["reference"]["counts"]["val"], roles["candidate"]["counts"]["val"])
        changed = [p for p, digest in context["inventories"]["reference"].items()
                   if context["inventories"]["candidate"][p] != digest]
        self.assertEqual(sorted(changed), [f"labels/train/new_{n:06}.txt" for n in (23, 31, 45)])
        self.assertEqual(context["document"]["evaluation"]["locatable_endpoint_denominator"], 1227)

    def test_changed_control_role_scope_recipe_reference_and_decision_fail(self):
        original_read = packaging.read_json
        document = original_read(ROOT / packaging.DEFAULT_PROTOCOL)
        mutations = [
            (("control", "sha256"), "0" * 64),
            (("roles", "reference", "arm"), "candidate"),
            (("roles", "reference", "counts", "train", "visible_endpoints"), 6841),
            (("roles", "candidate", "recipe_id"), "reviewed-endpoints-seed46-v1"),
            (("execution", "resume"), True),
            (("execution", "epochs"), 31),
            (("incremental_delta", "preserved_endpoint_slot_edits"), 9),
            (("incremental_delta", "deleted_source_rows", "new_000023"), [36]),
            (("evaluation", "primary_reference_sha256"), "0" * 64),
            (("evaluation", "locatable_endpoint_denominator"), 1228),
            (("evaluation", "primary_box_confidence"), 0.15),
            (("evaluation", "training_validation_unchanged"), False),
            (("evaluation", "decision"), "Promote on any larger aggregate score."),
        ]
        for keys, value in mutations:
            with self.subTest(field=".".join(keys)):
                bad = copy.deepcopy(document)
                target = bad
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                def read(path):
                    return bad if Path(path) == ROOT / packaging.DEFAULT_PROTOCOL else original_read(path)
                with patch.object(packaging, "read_json", side_effect=read), self.assertRaises(ValueError):
                    packaging.validate_protocol(ROOT)


class ReturnTransportChecks(unittest.TestCase):
    """The checkpoint validator is substituted here and exercised by integration checks."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="check-composed-return-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.context = packaging.validate_protocol(ROOT)
        self.bundles = {}
        source = {k: self.context["document"]["roles"]["reference"][k] for k in
                  ("candidate_manifest", "candidate_manifest_sha256", "candidate_id")}
        source["control"] = {"manifest": packaging.CONTROL, "manifest_sha256": packaging.CONTROL_SHA256}
        variants = {arm: {"counts": copy.deepcopy(self.context["document"]["roles"][arm]["counts"]),
                          "files": [{"path": p, "source_sha256": h, "output_sha256": h}
                                    for p, h in self.context["inventories"][arm].items()]}
                    for arm in packaging.ROLES}
        for arm in packaging.ROLES:
            run = self.base / arm
            run.mkdir()
            files = []
            for name in ("checkpoint-manifest.json", "training-receipt.json", "staging-manifest.json",
                         "fit/weights/last.pt", "fit/results.csv", "train.log"):
                path = run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"Synthetic {arm} transport fixture. Never deserialize or claim training.\n")
                files.append({"path": f"{arm}/{name}", "source_path": path,
                              "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
            self.bundles[arm] = {"identity": {"arm": arm, "synthetic_fixture": True}, "files": files,
                "staging": {"source": copy.deepcopy(source), "variants": copy.deepcopy(variants)},
                "receipt": {"arm": arm, "recipe_id": packaging.RECIPE_ID,
                    "recipe_sha256": self.context["document"]["recipe_sha256"],
                    "initial_checkpoint_sha256": self.context["document"]["initial_checkpoint_sha256"],
                    "source": {"revision": "1" * 40, "working_tree_status": ""},
                    "runtime": {"device": "cuda:0", "device_name": "synthetic fixture; no GPU executed"},
                    "training_settings": {"epochs": 30, "seed": 46}}}
        self.output = self.base / "returned.zip"
        for name, value in (("validate_protocol", self.context), ("preflight_dataset", {})):
            mocked = patch.object(packaging, name, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(packaging, "collect_arm", side_effect=lambda preflight, run, arm: self.bundles[Path(run).name])
        mocked.start()
        self.addCleanup(mocked.stop)

    def package(self, **kwargs):
        return packaging.package_results(ROOT, self.base / "reference", self.base / "candidate",
                                         self.output, **kwargs)

    def test_complete_archive_and_readonly_reverification(self):
        result = self.package()
        self.assertEqual(result["files"], 14)
        original = self.output.read_bytes()
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read("protocol.json"), self.context["path"].read_bytes())
            index = json.loads(archive.read(packaging.INDEX_NAME))
            self.assertEqual(set(index["arms"]), {"reference", "candidate"})
            self.assertEqual(index["kind"], "baomai-composed-training-results")
        self.assertEqual(self.package(verify_existing=True)["status"], "verified")
        self.assertEqual(self.output.read_bytes(), original)
        with self.assertRaises(ValueError):
            self.package()
        self.assertEqual(self.output.read_bytes(), original)

    def test_wrong_supervision_and_runtime_are_rejected_before_output(self):
        original = copy.deepcopy(self.bundles)
        changes = (
            lambda b: b["reference"]["staging"]["source"]["control"].update(manifest_sha256="0" * 64),
            lambda b: b["reference"]["staging"]["variants"]["reference"]["files"][0].update(output_sha256="0" * 64),
            lambda b: b["reference"]["receipt"].update(arm="candidate"),
            lambda b: b["candidate"]["receipt"]["runtime"].update(device_name="different synthetic GPU"),
            lambda b: b["candidate"]["receipt"]["source"].update(working_tree_status=" M labels.txt"),
            lambda b: b["candidate"]["receipt"]["runtime"].update(device="cpu"),
        )
        for change in changes:
            self.bundles = copy.deepcopy(original)
            change(self.bundles)
            with self.assertRaises(ValueError):
                self.package()
            self.assertFalse(self.output.exists())

    def test_corrupt_returned_member_is_not_accepted(self):
        self.package()
        with zipfile.ZipFile(self.output) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["candidate/train.log"] += b"corrupted"
        with zipfile.ZipFile(self.output, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        before = self.output.read_bytes()
        with self.assertRaises(ValueError):
            self.package(verify_existing=True)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
