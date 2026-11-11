#!/usr/bin/env python3
"""Check fixed annotation-delta and completed-package gates with synthetic bundles; no training."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.dont_write_bytecode = True

from check_pose_checkpoint import bundle_fixture, write_json
from evaluate_pose import file_sha256
import package_consistency_results as packaging
from package_gpu_results import INDEX_NAME
from pose_checkpoint import select_checkpoint, write_checkpoint_manifest
from train_pose_comparison import descriptor


class ProtocolChecks(unittest.TestCase):
    def test_real_candidate_exact_inheritance_fourteen_boxes_and_unchanged_counts(self):
        root = Path(__file__).resolve().parents[1]
        context = packaging.validate_protocol(root)
        self.assertEqual(context["delta_train_image_ids"], list(packaging.DELTA_ROWS))
        self.assertEqual(context["delta_train_rows_1based"], packaging.DELTA_ROWS)
        self.assertEqual(context["document"]["reviewed_unchanged_train_image_ids"],
                         ["old_000007", "old_000057", "new_000132"])
        checks = context["candidate_checks"]
        self.assertEqual(checks["old_reviewed"]["changed_train_labels"], 8)
        self.assertEqual(checks["consistency_reviewed"]["changed_train_labels"], 11)
        self.assertEqual(checks["old_reviewed"]["counts"], checks["consistency_reviewed"]["counts"])
        self.assertEqual(context["delta_verification"]["new_bbox_replacements"], 14)
        self.assertEqual(context["delta_verification"]["inherited_decision_objects_exact"], 12)
        self.assertTrue(context["delta_verification"]["all_class_and_endpoint_tokens_preserved"])

    def test_wrong_cohort_rows_candidate_reference_or_primary_threshold_rejected(self):
        root = Path(__file__).resolve().parents[1]
        original_read = packaging._read_json
        protocol = original_read(root / packaging.DEFAULT_PROTOCOL)
        for change in ("bad_candidate_hash", "wrong_cohort", "extra_image", "held_row",
                       "primary_threshold", "reference", "draft"):
            with self.subTest(change=change):
                bad = copy.deepcopy(protocol)
                if change == "bad_candidate_hash":
                    bad["roles"]["consistency_reviewed"]["candidate_manifest_sha256"] = "0" * 64
                elif change == "wrong_cohort":
                    bad["review_scope_train_image_ids"][-1] = "new_000133"
                elif change == "extra_image":
                    bad["delta_train_image_ids"].append("old_000007")
                elif change == "held_row":
                    bad["delta_train_rows_1based"]["new_000018"][0] = 23
                elif change == "primary_threshold":
                    bad["evaluation"]["primary_box_confidence"] = 0.15
                elif change == "reference":
                    bad["evaluation"]["primary_reference_sha256"] = "0" * 64
                else:
                    bad["status"] = "draft"
                def read(path):
                    return bad if Path(path) == root / packaging.DEFAULT_PROTOCOL else original_read(path)
                with mock.patch.object(packaging, "_read_json", side_effect=read):
                    with self.assertRaises(ValueError):
                        packaging.validate_protocol(root)


class DeltaChecks(unittest.TestCase):
    """Exercise semantic guards directly, beyond the immutable manifest hashes."""

    def setUp(self):
        source = Path(__file__).resolve().parents[1]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in (packaging.OLD_MANIFEST, packaging.NEW_MANIFEST):
            folder = Path(name).parent
            shutil.copytree(source / folder, self.root / folder)
        reference = json.loads((source / "data/manifest.json").read_text())
        self.records = {item["id"]: item for item in reference["records"]}
        for image_id in packaging.DELTA_ROWS:
            relative = Path("data") / self.records[image_id]["label"]
            (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, self.root / relative)
        self.old = json.loads((self.root / packaging.OLD_MANIFEST).read_text())
        self.new = json.loads((self.root / packaging.NEW_MANIFEST).read_text())
        self.label = (self.root / packaging.NEW_MANIFEST).parent / "labels/train/new_000018.txt"

    def check(self):
        return packaging.check_consistency_delta(self.root, self.old, self.new, self.records)

    def replace_row(self, raw, number, transform):
        lines = raw.splitlines(keepends=True)
        text = lines[number - 1].decode().rstrip("\r\n")
        ending = lines[number - 1][len(text.encode()):]
        tokens = text.split()
        transform(tokens)
        lines[number - 1] = " ".join(tokens).encode() + ending
        self.label.write_bytes(b"".join(lines))

    def test_changed_missing_and_visible_endpoint_tokens_rejected(self):
        raw = self.label.read_bytes()
        # Row24 has a missing left corner and a visible right corner.
        changes = {
            "missing_to_estimated": lambda t: t.__setitem__(slice(5, 8), ["0.65", "0.65", "2"]),
            "zero_token_reformatted": lambda t: t.__setitem__(5, "0"),
            "visible_point_shifted": lambda t: t.__setitem__(8, str(float(t[8]) - 0.001)),
        }
        for name, transform in changes.items():
            with self.subTest(change=name):
                self.replace_row(raw, 24, transform)
                with self.assertRaisesRegex(ValueError, "class/endpoint tokens changed"):
                    self.check()
        self.label.write_bytes(raw)

    def test_deferred_row_cannot_enter_the_delta(self):
        raw = self.label.read_bytes()
        self.replace_row(raw, 23, lambda t: t.__setitem__(1, str(float(t[1]) + 0.001)))
        with self.assertRaisesRegex(ValueError, "held or undeclared row changed"):
            self.check()

    def test_inherited_change_and_accepted_decision_objects_are_exact(self):
        old_id = self.old["changes"][0]["image_id"]
        changed = next(item for item in self.new["changes"] if item["image_id"] == old_id)
        original = copy.deepcopy(changed)
        changed["source_objects"] += 1
        with self.assertRaisesRegex(ValueError, "inherited change object differs"):
            self.check()
        changed.clear()
        changed.update(original)
        review_path = (self.root / packaging.NEW_MANIFEST).parent / "review.json"
        review = json.loads(review_path.read_text())
        review["decisions"][0]["reason"] += " Changed historical justification."
        write_json(review_path, review)
        with self.assertRaisesRegex(ValueError, "inherited accepted decisions differ"):
            self.check()

class PairChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.old, self.preflight, receipt, stage, _ = bundle_fixture(
            self.base, arm="candidate", recipe_id=packaging.RECIPE_ID)
        self.consistency = self.base / "consistency-run"
        shutil.copytree(self.old, self.consistency)
        delta = [f"train_{i:03d}" for i in range(1, 4)]
        receipt["runtime"].update(device="cuda:0", device_name="synthetic CUDA fixture; no GPU run",
                                  device_capability=[8, 0], torch_cuda="fixture")
        receipt["training_settings"]["device"] = "0"
        old_proposal = self.root / stage["source"]["candidate_manifest"]
        new_proposal = self.root / "data/candidates/consistency-fixture/manifest.json"
        new_proposal.parent.mkdir(parents=True)
        new = json.loads(old_proposal.read_text())
        new["candidate_id"] = "consistency-fixture"
        records = {r["id"]: r for r in json.loads((self.root / "data/manifest.json").read_text())["records"]}
        changed_hash = hashlib.sha256(b"synthetic annotation; no physical review").hexdigest()
        for image_id in delta:
            record = records[image_id]
            new["changes"].append({"image_id": image_id, "source_label": record["label"],
                                   "source_label_sha256": record["label_sha256"],
                                   "candidate_label_sha256": changed_hash})
        write_json(new_proposal, new)
        self.receipts, self.stages = {}, {}
        for role, path in zip(packaging.ROLES, (self.old, self.consistency)):
            staged, returned = copy.deepcopy(stage), copy.deepcopy(receipt)
            staged["output_root"] = str(path / "staged-data")
            if role == "consistency_reviewed":
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
            "delta_train_rows_1based": {image_id: [1] for image_id in delta},
            "document": {"experiment_id": "synthetic-consistency-check", "roles": {}},
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
            consistency_reviewed=self.consistency, output=self.output, **changes))

    def save_receipt(self, returned):
        write_json(self.consistency / "training-receipt.json", returned)
        manifest = json.loads((self.consistency / "checkpoint-manifest.json").read_text())
        manifest["training_receipt"] = descriptor(self.consistency / "training-receipt.json", self.consistency)
        write_json(self.consistency / "checkpoint-manifest.json", manifest)

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

    def test_swapped_roles_unmatched_runtime_or_source_and_dirty_checkout_rejected(self):
        with self.assertRaises(ValueError):
            packaging.package_results(self.root, self.consistency, self.old, self.output)
        for field in ("runtime", "source"):
            with self.subTest(field=field):
                returned = copy.deepcopy(self.receipts["consistency_reviewed"])
                if field == "runtime":
                    returned["runtime"]["device_name"] = "different synthetic device"
                else:
                    returned["source"]["revision"] = "9" * 40
                self.save_receipt(returned)
                with self.assertRaises(ValueError):
                    self.package()
        self.save_receipt(self.receipts["consistency_reviewed"])
        dirty = {**self.receipts["old_reviewed"]["source"], "working_tree_status": " M changed.py"}
        with mock.patch.object(packaging, "source_identity", return_value=dirty):
            with self.assertRaisesRegex(ValueError, "clean pinned source"):
                self.package()
        self.assertFalse(self.output.exists())

    def test_partial_trial_or_undeclared_input_delta_rejected(self):
        bundles = {role: packaging.collect_reviewed_run(self.preflight, path, role, self.context)
                   for role, path in zip(packaging.ROLES, (self.old, self.consistency))}
        for target in ("labels/val/", "images/train/", "labels/train/train_006.txt"):
            with self.subTest(target=target):
                bad = copy.deepcopy(bundles)
                item = next(x for x in bad["consistency_reviewed"]["staging"]["variants"]["candidate"]["files"]
                            if x["path"].startswith(target))
                item["output_sha256"] = "e" * 64
                with self.assertRaises(ValueError): packaging.check_reviewed_pair(bad, self.context)
        for change in ({"completed_epochs": list(range(1, 30))},
                       {"purpose": "resource-trial", "status": "trial-complete"},
                       {"finalization_completed": False}):
            self.save_receipt({**copy.deepcopy(self.receipts["consistency_reviewed"]), **change})
            with self.assertRaises(ValueError): self.package()
        self.assertFalse(self.output.exists())

    def test_tampered_return_zip_is_rejected_without_rewriting_it(self):
        self.package()
        with zipfile.ZipFile(self.output) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["consistency_reviewed/train.log"] += b"Tampered archive log.\n"
        with zipfile.ZipFile(self.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "return ZIP content differs"):
            self.package(verify_existing=True)
        self.assertEqual(before, self.output.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
