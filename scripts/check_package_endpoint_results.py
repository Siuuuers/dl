#!/usr/bin/env python3
"""Check the fixed endpoint protocol and returned bundles; no GPU or model loading."""

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
import package_endpoint_results as packaging
from package_gpu_results import INDEX_NAME
from pose_checkpoint import recipe_identity, select_checkpoint, write_checkpoint_manifest
from train_pose_comparison import descriptor


class ProtocolChecks(unittest.TestCase):
    def test_real_candidate_is_the_fixed_ten_slot_intervention(self):
        root = Path(__file__).resolve().parents[1]
        context = packaging.validate_protocol(root)
        self.assertEqual(context["delta_train_image_ids"], sorted(packaging.DELTA_ROWS))
        self.assertEqual(context["delta_train_rows_1based"], packaging.DELTA_ROWS)
        checks = context["candidate_checks"]
        self.assertEqual(checks["old_reviewed"]["changed_train_labels"], 8)
        self.assertEqual(checks["endpoint_reviewed"]["changed_train_labels"], 10)
        self.assertEqual(checks["endpoint_reviewed"]["new_endpoint_edit_counts"], packaging.ENDPOINT_DELTA)
        packaging.check_count_delta(checks["old_reviewed"]["counts"], checks["endpoint_reviewed"]["counts"])
        self.assertEqual(checks["endpoint_reviewed"]["counts"]["train"]["visible_endpoints"], 6842)
        self.assertEqual(checks["endpoint_reviewed"]["counts"]["train"]["missing_endpoints"], 700)
        roles = context["document"]["roles"]
        self.assertNotEqual(roles["old_reviewed"]["recipe_id"], roles["endpoint_reviewed"]["recipe_id"])
        self.assertEqual(roles["old_reviewed"]["recipe_sha256"], roles["endpoint_reviewed"]["recipe_sha256"])
        self.assertEqual(context["document"]["evaluation"]["primary_box_confidence"], 0.13)
        self.assertEqual(context["document"]["evaluation"]["diagnostic_box_confidences"], [])

    def test_protocol_rejects_changed_roles_recipe_inputs_scope_and_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        original_read = packaging._read_json
        protocol = original_read(root / packaging.DEFAULT_PROTOCOL)
        edits = [
            (("roles", "endpoint_reviewed", "candidate_manifest_sha256"), "0" * 64),
            (("roles", "endpoint_reviewed", "candidate_id"), "another-candidate"),
            (("roles", "endpoint_reviewed", "recipe_id"), packaging.ANNOTATION_CONFIRMATION_RECIPE_ID),
            (("execution", "seed"), 47), (("execution", "epochs"), 29),
            (("initial_checkpoint_sha256",), "0" * 64),
            (("candidate_source_revision",), "0" * 40),
            (("evaluation", "primary_box_confidence"), 0.15),
            (("evaluation", "diagnostic_box_confidences"), [0.15]),
            (("evaluation", "primary_reference_sha256"), "0" * 64),
            (("evaluation", "decision"), "Accept a higher aggregate score without identity review."),
            (("delta_train_rows_1based", "new_000023"), [32, 34, 35]),
            (("new_endpoint_edit_counts", "changed_endpoint_slots"), 9),
            (("new_endpoint_edit_counts", "net_visible_endpoint_change"), 0),
            (("status",), "draft"),
        ]
        for keys, value in edits:
            with self.subTest(field=".".join(keys)):
                bad = copy.deepcopy(protocol)
                target = bad
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                def read(path):
                    return bad if Path(path) == root / packaging.DEFAULT_PROTOCOL else original_read(path)
                with mock.patch.object(packaging, "_read_json", side_effect=read):
                    with self.assertRaises(ValueError):
                        packaging.validate_protocol(root)

    def test_rehashed_unlisted_label_change_still_fails_semantic_verification(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("data/labels", str(Path(packaging.OLD_MANIFEST).parent),
                             str(Path(packaging.NEW_MANIFEST).parent)):
                shutil.copytree(source / relative, root / relative)
            for relative in ("data/manifest.json", "docs/annotation-policy.md", packaging.DEFAULT_PROTOCOL,
                             packaging.PRIMARY_REFERENCE, packaging.ADJUDICATION["path"]):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, path)
            manifest = root / packaging.NEW_MANIFEST
            proposal = json.loads(manifest.read_text())
            change = next(row for row in proposal["changes"] if row["image_id"] == "new_000023")
            label = manifest.parent / change["candidate_label"]
            lines = label.read_text().splitlines(keepends=True)
            fields = lines[0].split()
            fields[1] = str(float(fields[1]) + 0.000001)
            lines[0] = " ".join(fields) + "\n"
            label.write_text("".join(lines))
            change["candidate_label_sha256"] = file_sha256(label)
            write_json(manifest, proposal)
            digest = file_sha256(manifest)
            protocol = json.loads((root / packaging.DEFAULT_PROTOCOL).read_text())
            protocol["roles"]["endpoint_reviewed"]["candidate_manifest_sha256"] = digest
            write_json(root / packaging.DEFAULT_PROTOCOL, protocol)
            # Refresh the immutable outer pin only to exercise the independent semantic gate.
            with mock.patch.object(packaging, "NEW_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "unlisted rows/bytes changed"):
                    packaging.validate_protocol(root)


class PairChecks(unittest.TestCase):
    """Synthetic metadata and text weights exercise artifact gates, never accuracy."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root, self.old, self.preflight, receipt, stage, _ = bundle_fixture(
            self.base, arm="candidate", recipe_id=packaging.ANNOTATION_CONFIRMATION_RECIPE_ID)
        self.endpoint = self.base / "endpoint-run"
        shutil.copytree(self.old, self.endpoint)
        delta = ["train_001", "train_002"]
        receipt["runtime"].update(device="cuda:0", device_name="synthetic CUDA fixture; no GPU run",
                                  device_capability=[8, 0], torch_cuda="fixture")
        receipt["training_settings"]["device"] = "0"
        old_proposal = self.root / stage["source"]["candidate_manifest"]
        new_proposal = self.root / "data/candidates/endpoint-fixture/manifest.json"
        new_proposal.parent.mkdir(parents=True)
        new = json.loads(old_proposal.read_text())
        new.update(candidate_id="synthetic-endpoint-fixture", kind="training_endpoint_overlay")
        new["expected_counts"]["train"]["visible_endpoints"] += 1
        new["expected_counts"]["train"]["missing_endpoints"] -= 1
        records = {row["id"]: row for row in json.loads((self.root / "data/manifest.json").read_text())["records"]}
        changed_hash = hashlib.sha256(b"synthetic endpoint annotation; no physical review").hexdigest()
        for image_id in delta:
            record = records[image_id]
            new["changes"].append({"image_id": image_id, "source_label": record["label"],
                "source_label_sha256": record["label_sha256"], "candidate_label_sha256": changed_hash})
        write_json(new_proposal, new)
        self.receipts, self.stages = {}, {}
        for role, path in zip(packaging.ROLES, (self.old, self.endpoint)):
            staged, returned = copy.deepcopy(stage), copy.deepcopy(receipt)
            staged["output_root"] = str(path / "staged-data")
            returned.update(recipe_identity(packaging.ROLE_RECIPE_IDS[role]))
            if role == "endpoint_reviewed":
                staged["source"].update(candidate_id=new["candidate_id"],
                    candidate_manifest=new_proposal.relative_to(self.root).as_posix(),
                    candidate_manifest_sha256=file_sha256(new_proposal))
                staged["variants"]["candidate"]["counts"] = new["expected_counts"]
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
            "document": {"experiment_id": "synthetic-endpoint-check", "roles": {},
                         "new_endpoint_edit_counts": packaging.ENDPOINT_DELTA},
            "effective_label_sha256": {}, "candidate_checks": {}}
        for role in packaging.ROLES:
            source = self.stages[role]["source"]
            self.context["document"]["roles"][role] = {
                **{key: source[key] for key in ("candidate_manifest", "candidate_manifest_sha256", "candidate_id")},
                **recipe_identity(packaging.ROLE_RECIPE_IDS[role])}
            variant = self.stages[role]["variants"]["candidate"]
            inventory = {item["path"]: item["output_sha256"] for item in variant["files"]}
            self.context["effective_label_sha256"][role] = {
                image_id: inventory[record["label"]] for image_id, record in records.items()}
            self.context["candidate_checks"][role] = {"counts": copy.deepcopy(variant["counts"])}
        self.output = self.base / "return.zip"
        for name, value in (("INITIAL_SHA256", self.preflight["model_sha256"]),):
            patch = mock.patch.object(packaging, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        for name, value in (("validate_protocol", self.context),
                           ("source_identity", self.receipts["old_reviewed"]["source"])):
            patch = mock.patch.object(packaging, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)

    def package(self, **changes):
        return packaging.package_results(**dict(root=self.root, old_reviewed=self.old,
            endpoint_reviewed=self.endpoint, output=self.output, **changes))

    def save_receipt(self, returned, *, bind_recipe=False):
        write_json(self.endpoint / "training-receipt.json", returned)
        manifest = json.loads((self.endpoint / "checkpoint-manifest.json").read_text())
        manifest["training_receipt"] = descriptor(self.endpoint / "training-receipt.json", self.endpoint)
        if bind_recipe:
            manifest.update({key: returned[key] for key in ("recipe_id", "recipe_sha256")})
        write_json(self.endpoint / "checkpoint-manifest.json", manifest)

    def test_distinct_recipe_ids_equal_numerics_and_plus_one_endpoint_roundtrip(self):
        result = self.package()
        before = self.output.read_bytes()
        self.assertEqual(result["files"], 13)
        with zipfile.ZipFile(self.output) as archive:
            index = json.loads(archive.read(INDEX_NAME))
            self.assertEqual(index["kind"], packaging.RESULT_KIND)
            self.assertFalse(any(name.startswith("reference/") for name in archive.namelist()))
            archive.extractall(self.base / "relocated")
        old, endpoint = (index["roles"][role] for role in packaging.ROLES)
        self.assertNotEqual(old["recipe_id"], endpoint["recipe_id"])
        self.assertEqual(old["recipe_sha256"], endpoint["recipe_sha256"])
        self.assertEqual(endpoint["training_counts"]["train"]["visible_endpoints"],
                         old["training_counts"]["train"]["visible_endpoints"] + 1)
        for role in packaging.ROLES:
            chosen = select_checkpoint(self.preflight, self.base / "relocated" / role / "checkpoint-manifest.json")
            self.assertEqual(chosen["checkpoint_identity"]["arm"], "candidate")
            self.assertEqual(chosen["checkpoint_identity"]["recipe_id"], packaging.ROLE_RECIPE_IDS[role])
        self.assertEqual(self.package(verify_existing=True)["status"], "verified_existing")
        self.assertEqual(before, self.output.read_bytes())
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(before, self.output.read_bytes())

    def test_wrong_roles_recipe_and_candidate_rejected(self):
        with self.assertRaises(ValueError):
            packaging.package_results(self.root, self.endpoint, self.old, self.output)
        returned = copy.deepcopy(self.receipts["endpoint_reviewed"])
        returned.update(recipe_identity(packaging.ANNOTATION_CONFIRMATION_RECIPE_ID))
        self.save_receipt(returned, bind_recipe=True)
        with self.assertRaisesRegex(ValueError, "recipe and training_endpoint_overlay"):
            self.package()
        self.save_receipt(self.receipts["endpoint_reviewed"], bind_recipe=True)
        # Refresh all outer descriptors so the candidate-ID gate, not a stale hash, rejects it.
        staged = copy.deepcopy(self.stages["endpoint_reviewed"])
        staged["source"]["candidate_id"] = "wrong-endpoint-candidate"
        write_json(self.endpoint / "staging-manifest.json", staged)
        returned = copy.deepcopy(self.receipts["endpoint_reviewed"])
        returned["staging_manifest_sha256"] = file_sha256(self.endpoint / "staging-manifest.json")
        self.save_receipt(returned)
        manifest = json.loads((self.endpoint / "checkpoint-manifest.json").read_text())
        manifest["staging_manifest"] = descriptor(self.endpoint / "staging-manifest.json", self.endpoint)
        write_json(self.endpoint / "checkpoint-manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "candidate reference identity"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_seed_runtime_initialization_source_and_incomplete_runs_rejected(self):
        changes = [
            (("training_settings", "seed"), 47),
            (("runtime", "device_name"), "different synthetic GPU"),
            (("runtime", "device"), "cpu"),
            (("initial_checkpoint_sha256",), "0" * 64),
            (("source", "revision"), "9" * 40),
            (("source", "files_sha256", "scripts/pose_checkpoint.py"), "0" * 64),
            (("purpose",), "resource-trial"),
            (("completed_epochs",), list(range(1, 30))),
            (("finalization_completed",), False),
        ]
        for keys, value in changes:
            with self.subTest(field=".".join(keys)):
                returned = copy.deepcopy(self.receipts["endpoint_reviewed"])
                target = returned
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                self.save_receipt(returned)
                with self.assertRaises(ValueError):
                    self.package()
        self.save_receipt(self.receipts["endpoint_reviewed"])
        dirty = {**self.receipts["old_reviewed"]["source"], "working_tree_status": " M changed.py"}
        with mock.patch.object(packaging, "source_identity", return_value=dirty):
            with self.assertRaisesRegex(ValueError, "clean pinned source"):
                self.package()
        self.assertFalse(self.output.exists())

    def test_complete_inventories_include_validation_images_and_unlisted_labels(self):
        bundles = {role: packaging.collect_reviewed_run(self.preflight, path, role, self.context)
                   for role, path in zip(packaging.ROLES, (self.old, self.endpoint))}
        for target in ("labels/val/", "images/train/", "labels/train/train_006.txt", "duplicate", "missing"):
            with self.subTest(target=target):
                bad = copy.deepcopy(bundles)
                entries = bad["endpoint_reviewed"]["staging"]["variants"]["candidate"]["files"]
                if target == "duplicate":
                    entries.append(copy.deepcopy(entries[0]))
                elif target == "missing":
                    entries.pop()
                else:
                    entry = next(row for row in entries if row["path"].startswith(target))
                    entry["source_sha256"] = entry["output_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "complete candidate inventory"):
                    packaging.check_reviewed_pair(bad, self.context)

    def test_only_the_declared_endpoint_count_delta_is_allowed(self):
        counts = self.context["candidate_checks"]
        old, endpoint = (copy.deepcopy(counts[role]["counts"]) for role in packaging.ROLES)
        packaging.check_count_delta(old, endpoint)
        for split, field in (("train", "visible_endpoints"), ("train", "missing_endpoints"),
                             ("train", "objects"), ("train", "empty_images"),
                             ("val", "objects"), ("val", "visible_endpoints")):
            with self.subTest(split=split, field=field):
                bad = copy.deepcopy(endpoint)
                bad[split][field] += 1
                with self.assertRaisesRegex(ValueError, "counts must differ only"):
                    packaging.check_count_delta(old, bad)

    def test_archive_tamper_is_rejected_without_repair_or_overwrite(self):
        self.package()
        with zipfile.ZipFile(self.output) as archive:
            members = {name: archive.read(name) for name in archive.namelist()}
        members["endpoint_reviewed/train.log"] += b"unrecorded change\n"
        with zipfile.ZipFile(self.output, "w") as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        tampered = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "return ZIP content differs"):
            self.package(verify_existing=True)
        self.assertEqual(tampered, self.output.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
