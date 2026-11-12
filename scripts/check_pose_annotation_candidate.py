#!/usr/bin/env python3
"""Focused mixed-overlay integrity checks; synthetic data, no image adjudication/training."""

import sys
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from check_evaluate_pose import png_bytes, preflight_fixture
from check_pose_checkpoint import bundle_fixture
from evaluate_pose import file_sha256
from pose_checkpoint import (ANNOTATION_CONFIRMATION_RECIPE_ID, ANNOTATION_RECIPE_ID, ANNOTATION_RECIPE_IDS,
                             BOX_RECIPE_ID, COMPOSED_RECIPE_ID, ENDPOINT_RECIPE_ID, ORIGINAL_RECIPE_ID,
                             RECIPE_IDS, check_recipe_candidate_kind, get_recipe,
                             select_checkpoint)
from stage_pose_training import MANIFEST, stage_datasets, verify_staging
from train_pose_comparison import descriptor, run_parent
from verify_pose_annotation_candidate import ANNOTATION_KIND, verify_annotation_candidate
from verify_pose_box_candidate import verify_box_candidate
from verify_pose_candidate import verify_candidate


SOURCE = "0 .5 .5 .8 .8 0 0 0 0 0 0"
DONOR = "0 .5 .5 .8 .8 .2 .6 2 0 0 0"
UNCHANGED = "0 .5 .5 .8 .8 .2 .6 2 .7 .6 2"
REPLACEMENT = "0 .5 .45 .8 .7 .2 .6 2 0 0 0"
ADDITION = "0 .5 .5 .6 .6 0 0 0 0 0 0"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class AnnotationCandidateChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repository"
        reference = preflight_fixture(self.root)
        original = (SOURCE + "\r\n" + DONOR + "\n" + UNCHANGED + "\n").encode()
        for index in range(137):
            identifier = f"train_{index:03d}"
            image, label = f"images/train/{identifier}.png", f"labels/train/{identifier}.txt"
            for relative, data in ((image, png_bytes()), (label, original)):
                path = self.root / "data" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            reference["records"].append({"id": identifier, "split": "train", "image": image, "label": label,
                "original_size": [100, 100], "instances": 3, "visible_endpoints": 3, "missing_endpoints": 3,
                "image_sha256": file_sha256(self.root / "data" / image),
                "label_sha256": file_sha256(self.root / "data" / label)})
        reference["split_summary"]["train"] = {"images": 137, "instances": 411, "visible_endpoints": 411,
                                                "missing_endpoints": 411, "empty_images": 0}
        write_json(self.root / "data/manifest.json", reference)
        self.folder = self.root / "data/candidates/fixture"
        self.manifest = self.folder / "manifest.json"
        self.label = self.folder / "labels/train/train_000.txt"
        self.label.parent.mkdir(parents=True)
        self.label.write_bytes((REPLACEMENT + "\r\n" + UNCHANGED + "\n" + ADDITION + "\n").encode())
        policy = self.root / "docs/annotation-policy.md"
        policy.parent.mkdir()
        policy.write_text("Synthetic policy identity; no physical review performed.\n")
        self.evidence = {"schema_version": 1, "candidate_id": "synthetic-annotation-fixture", "decisions": [
            {"id": "merge", "image_id": "train_000", "review_status": "accepted_incremental",
             "reason": "Synthetic merge and box correction."},
            {"id": "add", "image_id": "train_000", "review_status": "accepted_incremental",
             "reason": "Synthetic omitted-object correction."}]}
        write_json(self.folder / "review.json", self.evidence)
        merge, addition = ({"review_id": item["id"], "review_status": item["review_status"],
                            "reason": item["reason"]} for item in self.evidence["decisions"])
        record = reference["records"][20]
        self.proposal = {"schema_version": 1, "candidate_id": self.evidence["candidate_id"],
            "kind": ANNOTATION_KIND, "status": "proposed", "source_revision": "1" * 40,
            "source_manifest_sha256": file_sha256(self.root / "data/manifest.json"),
            "annotation_policy": {"id": "baomai-visible-front-v1", "path": "docs/annotation-policy.md",
                                  "sha256": file_sha256(policy)},
            "review": {"path": "review.json", "sha256": file_sha256(self.folder / "review.json")},
            "changes": [{"image_id": "train_000", "source_label": record["label"],
                "candidate_label": record["label"], "source_label_sha256": record["label_sha256"],
                "source_image_sha256": record["image_sha256"], "candidate_label_sha256": file_sha256(self.label),
                "source_objects": 3, "candidate_objects": 3,
                "deletions": [{"original_row": 2, "source_row_text": DONOR, **merge}],
                "replacements": [{"original_row": 1, "candidate_row": 1, "source_row_text": SOURCE,
                    "candidate_row_text": REPLACEMENT, "changed_fields": ["bbox", "left_endpoint"],
                    "donor_endpoints": [{"donor_original_row": 2, "donor_endpoint": 0, "candidate_endpoint": 0,
                                         "source_row_text": DONOR, "reason": "Synthetic visible-point transfer."}],
                    **merge}],
                "additions": [{"addition_id": "omitted-front", "candidate_row": 3,
                               "candidate_row_text": ADDITION, **addition}],
                "row_map": [{"original_row": 1, "candidate_row": 1}, {"original_row": 2, "candidate_row": None},
                            {"original_row": 3, "candidate_row": 2}]}],
            "expected_counts": {split: {("objects" if key == "instances" else key): value
                for key, value in counts.items()} for split, counts in reference["split_summary"].items()}}
        self.change = self.proposal["changes"][0]
        self.save()

    def save(self):
        write_json(self.manifest, self.proposal)

    def refresh_label(self, content):
        self.label.write_bytes(content)
        self.change["candidate_label_sha256"] = file_sha256(self.label)
        self.save()

    def test_mixed_operations_count_actual_deletions_even_when_net_objects_unchanged(self):
        result = verify_annotation_candidate(self.root, self.manifest)
        self.assertEqual((result["removed_objects"], result["added_objects"], result["replaced_objects"],
                          result["merged_endpoints"]), (1, 1, 1, 1))
        self.assertEqual(result["counts"]["train"]["objects"], 411)
        self.assertTrue(result["validation_labels_unchanged"])
        self.assertTrue(result["full_row_maps_verified"])
        self.assertFalse(result["visual_correctness_verified"])
        self.assertIn((REPLACEMENT + "\r\n").encode(), self.label.read_bytes())

    def test_actual_staging_preserves_validation_and_routes_only_declared_train_overlay(self):
        output = self.base / "paired-copy"
        stage_datasets(self.root, self.manifest, output)
        self.assertEqual(verify_staging(self.root, self.manifest, output)["status"], "verified")
        staged = json.loads((output / MANIFEST).read_text())
        self.assertEqual(staged["verification"]["removed_training_objects"], 1)
        self.assertIn("scripts/verify_pose_annotation_candidate.py", staged["code_sha256"])
        for path in (output / "reference/labels/val").glob("*.txt"):
            self.assertEqual(path.read_bytes(), (output / "candidate/labels/val" / path.name).read_bytes())
        self.assertEqual((output / "candidate/labels/train/train_000.txt").read_bytes(), self.label.read_bytes())
        self.assertEqual((output / "reference/labels/train/train_000.txt").read_bytes(),
                         (self.root / "data/labels/train/train_000.txt").read_bytes())
        target = output / "candidate/labels/train/train_001.txt"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "staged file SHA-256 mismatch"):
            verify_staging(self.root, self.manifest, output)

    def test_refreshed_label_hash_does_not_hide_unlisted_edits(self):
        self.refresh_label(self.label.read_bytes().replace(UNCHANGED.encode(), DONOR.encode()))
        with self.assertRaisesRegex(ValueError, "unlisted rows/bytes"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_full_row_map_rejects_missing_deleted_or_boolean_entries(self):
        original = copy.deepcopy(self.change["row_map"])
        for mapping in (original[:-1], [{"original_row": 1, "candidate_row": True}, *original[1:]],
                        [original[0], {"original_row": 2, "candidate_row": 2}, original[2]]):
            self.change["row_map"] = mapping
            self.save()
            with self.assertRaisesRegex(ValueError, "row map differs"):
                verify_annotation_candidate(self.root, self.manifest)

    def test_validation_and_wrong_label_mappings_are_rejected(self):
        self.change["image_id"] = "case_00"
        self.save()
        with self.assertRaisesRegex(ValueError, "validation changes"):
            verify_annotation_candidate(self.root, self.manifest)
        self.change["image_id"] = "train_000"
        self.change["candidate_label"] = "labels/train/train_001.txt"
        self.save()
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_source_candidate_and_image_hashes_are_bound(self):
        for key, message in (("source_label_sha256", "source identity"), ("source_image_sha256", "source identity"),
                             ("candidate_label_sha256", "candidate label SHA-256")):
            original = self.change[key]
            self.change[key] = "0" * 64
            self.save()
            with self.assertRaisesRegex(ValueError, message):
                verify_annotation_candidate(self.root, self.manifest)
            self.change[key] = original
        self.save()
        source = self.root / "data/labels/train/train_001.txt"
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source label SHA-256"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_review_hash_and_applied_decision_are_both_bound(self):
        self.evidence["decisions"][0]["review_status"] = "unresolved"
        write_json(self.folder / "review.json", self.evidence)
        with self.assertRaisesRegex(ValueError, "review evidence SHA-256"):
            verify_annotation_candidate(self.root, self.manifest)
        self.proposal["review"]["sha256"] = file_sha256(self.folder / "review.json")
        self.save()
        with self.assertRaisesRegex(ValueError, "accepted incremental review"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_donor_must_be_deleted_and_transfer_exact_visible_endpoint(self):
        donor = self.change["replacements"][0]["donor_endpoints"][0]
        original = copy.deepcopy(donor)
        for mutation in ({"donor_original_row": 3}, {"donor_endpoint": 1}, {"source_row_text": SOURCE}):
            donor.clear()
            donor.update({**original, **mutation})
            self.save()
            with self.assertRaisesRegex(ValueError, "donor"):
                verify_annotation_candidate(self.root, self.manifest)

    def test_omitted_or_repeated_donor_provenance_fails_with_unchanged_labels_and_counts(self):
        replacement = self.change["replacements"][0]
        donors = copy.deepcopy(replacement["donor_endpoints"])
        original_bytes = self.label.read_bytes()
        original_counts = copy.deepcopy(self.proposal["expected_counts"])
        for value, message in (([], "every deleted visible endpoint"),
                               (donors * 2, "donor or recipient endpoint repeated")):
            replacement["donor_endpoints"] = value
            self.save()
            self.assertEqual(self.label.read_bytes(), original_bytes)
            self.assertEqual(self.proposal["expected_counts"], original_counts)
            with self.assertRaisesRegex(ValueError, message):
                verify_annotation_candidate(self.root, self.manifest)

    def test_explicit_retained_visible_point_movement_is_rejected_with_unchanged_counts(self):
        moved = UNCHANGED.replace(".2 .6 2", ".21 .6 2")
        self.change["replacements"].append({**self.change["replacements"][0],
            "original_row": 3, "candidate_row": 2, "source_row_text": UNCHANGED,
            "candidate_row_text": moved, "changed_fields": ["left_endpoint"], "donor_endpoints": []})
        self.refresh_label(self.label.read_bytes().replace(UNCHANGED.encode(), moved.encode()))
        with self.assertRaisesRegex(ValueError, "original visible endpoint tokens must be preserved"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_duplicate_deletion_overlap_or_invented_changed_fields_are_rejected(self):
        original = copy.deepcopy(self.change)
        cases = (("deletions", self.change["deletions"] * 2, "repeated deletion"),
                 ("deletions", [{**self.change["deletions"][0], "original_row": 1, "source_row_text": SOURCE}],
                  "row map differs"))
        for key, value, message in cases:
            self.change.update(copy.deepcopy(original))
            self.change[key] = value
            self.save()
            with self.assertRaisesRegex(ValueError, message):
                verify_annotation_candidate(self.root, self.manifest)
        self.change.update(original)
        self.change["replacements"][0]["changed_fields"] = ["bbox"]
        self.save()
        with self.assertRaisesRegex(ValueError, "field inventory"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_missing_endpoint_state_and_visible_point_containment_are_enforced(self):
        self.change["additions"][0]["candidate_row_text"] = ADDITION.replace("0 0 0 0 0 0", ".3 .6 0 0 0 0")
        self.save()
        with self.assertRaisesRegex(ValueError, "missing endpoint must be 0 0 0"):
            verify_annotation_candidate(self.root, self.manifest)
        self.change["additions"][0]["candidate_row_text"] = ADDITION
        bad = REPLACEMENT.replace(".5 .45 .8 .7", ".5 .25 .8 .2")
        self.change["replacements"][0]["candidate_row_text"] = bad
        self.refresh_label((bad + "\r\n" + UNCHANGED + "\n" + ADDITION + "\n").encode())
        with self.assertRaisesRegex(ValueError, "endpoint outside its box"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_additions_must_append_and_expected_counts_cannot_lie(self):
        self.change["additions"][0]["candidate_row"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "append in declared order"):
            verify_annotation_candidate(self.root, self.manifest)
        self.change["additions"][0]["candidate_row"] = 3
        self.proposal["expected_counts"]["train"]["objects"] -= 1
        self.save()
        with self.assertRaisesRegex(ValueError, "expected_counts"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_extra_labels_and_symlinks_are_rejected(self):
        extra = self.label.parent / "unexpected.txt"
        extra.write_text(SOURCE + "\n")
        with self.assertRaisesRegex(ValueError, "unexpected or missing"):
            verify_annotation_candidate(self.root, self.manifest)
        extra.unlink()
        extra.symlink_to(self.label)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            verify_annotation_candidate(self.root, self.manifest)

    def test_old_verifiers_do_not_accept_new_operation_contract(self):
        for verifier, kind in ((verify_candidate, "training_label_overlay"),
                               (verify_box_candidate, "training_box_overlay")):
            with self.assertRaisesRegex(ValueError, kind):
                verifier(self.root, self.manifest)


class AnnotationRecipeChecks(unittest.TestCase):
    def test_recipe_changes_only_seed_from_established_low_bias_pair(self):
        self.assertEqual(get_recipe(ANNOTATION_RECIPE_ID), {**get_recipe(BOX_RECIPE_ID), "seed": 45})
        self.assertEqual(get_recipe(ANNOTATION_CONFIRMATION_RECIPE_ID),
                         {**get_recipe(ANNOTATION_RECIPE_ID), "seed": 46})
        recipe = get_recipe(ANNOTATION_RECIPE_ID)
        recipe["seed"] = 46
        self.assertEqual(get_recipe(ANNOTATION_RECIPE_ID)["seed"], 45)
        confirmation = get_recipe(ANNOTATION_CONFIRMATION_RECIPE_ID)
        confirmation["seed"] = 45
        self.assertEqual(get_recipe(ANNOTATION_CONFIRMATION_RECIPE_ID)["seed"], 46)
        for recipe_id in RECIPE_IDS:
            for kind in ("training_label_overlay", "training_box_overlay", ANNOTATION_KIND):
                valid = recipe_id not in (ENDPOINT_RECIPE_ID, COMPOSED_RECIPE_ID) and ((recipe_id in ANNOTATION_RECIPE_IDS and kind == ANNOTATION_KIND)
                         or (recipe_id not in ANNOTATION_RECIPE_IDS and kind != ANNOTATION_KIND
                             and (recipe_id.startswith("box-supervision-") == (kind == "training_box_overlay"))))
                if valid:
                    check_recipe_candidate_kind(recipe_id, kind)
                else:
                    with self.assertRaisesRegex(ValueError, "must be selected together"):
                        check_recipe_candidate_kind(recipe_id, kind)

    def test_runner_rejects_wrong_kind_before_training_import_or_output_creation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            proposal = base / "manifest.json"
            for recipe_id, kind in ((ORIGINAL_RECIPE_ID, ANNOTATION_KIND),
                                     *((recipe_id, "training_label_overlay") for recipe_id in ANNOTATION_RECIPE_IDS)):
                write_json(proposal, {"kind": kind})
                output = base / "fresh"
                args = SimpleNamespace(root=root, output=output, threads=1, max_seconds=1,
                    max_rss_mib=128, trial_batches=1, device="cpu", recipe=recipe_id, arm="reference",
                    candidate="data/candidates/fixture/manifest.json")
                with patch("train_pose_comparison._relative_file", return_value=proposal), \
                        patch("train_pose_comparison.importlib.metadata.version") as runtime:
                    with self.assertRaisesRegex(ValueError, "must be selected together"):
                        run_parent(args)
                    runtime.assert_not_called()
                self.assertFalse(output.exists())

    def test_completed_checkpoint_supports_both_arms_and_rejects_refreshed_wrong_seed(self):
        for recipe_id, arm in ((recipe_id, arm) for recipe_id in ANNOTATION_RECIPE_IDS
                               for arm in ("reference", "candidate")):
            with self.subTest(recipe=recipe_id, arm=arm), tempfile.TemporaryDirectory() as directory:
                root, run, preflight, receipt, staging, manifest_path = bundle_fixture(
                    Path(directory), arm, recipe_id=recipe_id)
                identity = select_checkpoint(preflight, manifest_path)["checkpoint_identity"]
                self.assertEqual((identity["arm"], identity["recipe_id"]), (arm, recipe_id))
                receipt["training_settings"]["seed"] = 46 if recipe_id == ANNOTATION_RECIPE_ID else 45
                write_json(run / "training-receipt.json", receipt)
                manifest = json.loads(manifest_path.read_text())
                manifest["training_receipt"] = descriptor(run / "training-receipt.json", run)
                write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "paired recipe: seed"):
                    select_checkpoint(preflight, manifest_path)
                receipt["training_settings"]["seed"] = get_recipe(recipe_id)["seed"]
                candidate_path = root / staging["source"]["candidate_manifest"]
                proposal = json.loads(candidate_path.read_text())
                proposal["kind"] = "training_label_overlay"
                write_json(candidate_path, proposal)
                staging["source"]["candidate_manifest_sha256"] = file_sha256(candidate_path)
                write_json(run / MANIFEST, staging)
                receipt["staging_manifest_sha256"] = file_sha256(run / MANIFEST)
                write_json(run / "training-receipt.json", receipt)
                manifest["training_receipt"] = descriptor(run / "training-receipt.json", run)
                manifest["staging_manifest"] = descriptor(run / MANIFEST, run)
                write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, "must be selected together"):
                    select_checkpoint(preflight, manifest_path)

    def test_same_seed_annotation_pairs_package_and_mixed_seeds_are_rejected(self):
        from package_gpu_results import package_results
        with tempfile.TemporaryDirectory() as directory:
            base, runs = Path(directory), {}
            for recipe_id in ANNOTATION_RECIPE_IDS:
                for arm in ("reference", "candidate"):
                    root, run, _, _, _, _ = bundle_fixture(base / f"{recipe_id}-{arm}", arm, recipe_id=recipe_id)
                    runs[recipe_id, arm] = root, run
                root, reference = runs[recipe_id, "reference"]
                result = package_results(root, reference, runs[recipe_id, "candidate"][1], base / f"{recipe_id}.zip")
                self.assertEqual(result["files"], 13)
                self.assertEqual(result["arms"]["candidate"]["checkpoint_identity"]["recipe_id"], recipe_id)
            root, reference = runs[ANNOTATION_RECIPE_ID, "reference"]
            with self.assertRaisesRegex(ValueError, "shared training settings differ"):
                package_results(root, reference, runs[ANNOTATION_CONFIRMATION_RECIPE_ID, "candidate"][1],
                                base / "mixed-seeds.zip")
            self.assertFalse((base / "mixed-seeds.zip").exists())


if __name__ == "__main__":
    unittest.main()
