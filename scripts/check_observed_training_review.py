#!/usr/bin/env python3
"""Check observed-review release inventory with small synthetic evidence only.

These checks exercise omissions and contradictory status/coverage claims without
model weights, training, GPU access, or changes to any dataset artifact.
"""

import sys
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_observed_training_candidate import (HISTORICAL_REPLAY_CODE, check_annotation_readiness,
                                                check_input_partial_coverage, compose_observed_review,
                                                reconcile_review_inventory,
                                                review_batches, sha)


def fixture():
    input_candidate = {"path": "data/candidates/input/manifest.json", "sha256": "1" * 64}
    parent = {"candidate": copy.deepcopy(input_candidate), "cases": [
        {"case_id": "old-a", "image_id": "a", "status": "held"},
        {"case_id": "old-b", "image_id": "b", "status": "partial"},
        {"case_id": "old-c", "image_id": "c", "status": "resolved"},
    ]}
    images = {
        image_id: {"image_id": image_id, "source_original_image": {"sha256": image_id * 64},
                   "reviewed_partial_rows": rows, "image_review_complete": True,
                   "case_dispositions": [], "new_issues": []}
        for image_id, rows in (("a", [1, 3]), ("b", [2]), ("c", []))
    }
    images["a"]["case_dispositions"] = [{"case_id": "old-a", "status": "resolved"}]
    images["b"]["case_dispositions"] = [{"case_id": "old-b", "status": "partial"}]
    # A missing status means held, including when an export forgets that issue.
    images["a"]["new_issues"] = [{"issue_id": "new-a", "current_rows": [3]}]
    images["c"]["new_issues"] = [{"case_id": "new-c", "current_rows": [], "status": "resolved"}]
    batches = {
        "review/batch-1.json": {"input_candidate": copy.deepcopy(input_candidate), "images": [images["a"]]},
        "review/batch-2.json": {"input_candidate": copy.deepcopy(input_candidate),
                                "images": [images["b"], images["c"]]},
    }
    status = {"cases": copy.deepcopy(parent["cases"])}
    status["cases"][0]["status"] = "resolved"
    status["cases"] += [{"case_id": "new-a", "image_id": "a", "status": "held"},
                        {"case_id": "new-c", "image_id": "c", "status": "resolved"}]
    coverage = {"input_candidate": input_candidate, "images": [
        {"image_id": image_id, "source_image_sha256": image_id * 64,
         "input_partial_rows": images[image_id]["reviewed_partial_rows"].copy(),
         "image_review_complete": True, "review_path": path,
         "unresolved_case_ids": ["new-a"] if image_id == "a" else ["old-b"] if image_id == "b" else []}
        for path, batch in batches.items() for image_id in (image["image_id"] for image in batch["images"])
    ]}
    return parent, batches, status, coverage


class ReviewInventoryChecks(unittest.TestCase):
    def setUp(self):
        self.parent, self.batches, self.status, self.coverage = fixture()

    def check(self):
        return reconcile_review_inventory(self.parent, self.batches, self.status, self.coverage)

    def image(self, image_id):
        return next(image for batch in self.batches.values() for image in batch["images"]
                    if image["image_id"] == image_id)

    def test_mixed_case_history_and_discoveries_are_reconstructed(self):
        self.assertEqual(self.check(), ["old-b", "new-a"])

    def test_all_cases_can_resolve_when_the_evidence_explicitly_resolves_them(self):
        self.image("b")["case_dispositions"][0]["status"] = "resolved"
        self.image("a")["new_issues"][0]["status"] = "resolved"
        for item in self.status["cases"]:
            item["status"] = "resolved"
        for item in self.coverage["images"]:
            item["unresolved_case_ids"] = []
        self.assertEqual(self.check(), [])

    def test_omitted_new_issue_fails_even_with_self_consistent_pending_summary(self):
        self.status["cases"] = [item for item in self.status["cases"] if item["case_id"] != "new-a"]
        self.coverage["images"][0]["unresolved_case_ids"] = []
        self.status["remaining_all_case_groups"] = 1
        with self.assertRaisesRegex(ValueError, "case inventory differs"):
            self.check()

    def test_resolved_history_and_resolved_new_issues_cannot_be_dropped(self):
        for case_id in ("old-c", "new-c"):
            with self.subTest(case_id=case_id):
                parent, batches, status, coverage = fixture()
                status["cases"] = [item for item in status["cases"] if item["case_id"] != case_id]
                with self.assertRaisesRegex(ValueError, "case inventory differs"):
                    reconcile_review_inventory(parent, batches, status, coverage)

    def test_prior_open_case_requires_an_explicit_disposition(self):
        self.image("b")["case_dispositions"] = []
        with self.assertRaisesRegex(ValueError, "prior open cases lack explicit dispositions: old-b"):
            self.check()

    def test_summary_cannot_clear_an_evidence_held_or_partial_case(self):
        for case_id, image_index in (("old-b", 1), ("new-a", 0)):
            with self.subTest(case_id=case_id):
                parent, batches, status, coverage = fixture()
                next(item for item in status["cases"] if item["case_id"] == case_id)["status"] = "resolved"
                coverage["images"][image_index]["unresolved_case_ids"] = []
                with self.assertRaisesRegex(ValueError, "declared case status or image ownership differs"):
                    reconcile_review_inventory(parent, batches, status, coverage)

    def test_resolved_parent_changes_only_with_explicit_evidence(self):
        self.status["cases"][2]["status"] = "held"
        self.coverage["images"][2]["unresolved_case_ids"] = ["old-c"]
        with self.assertRaisesRegex(ValueError, "declared case status or image ownership differs"):
            self.check()
        self.image("c")["case_dispositions"] = [{"case_id": "old-c", "status": "held"}]
        self.assertEqual(self.check(), ["old-b", "old-c", "new-a"])

    def test_case_disposition_cannot_be_moved_to_another_image(self):
        self.image("b")["case_dispositions"] += self.image("a")["case_dispositions"]
        self.image("a")["case_dispositions"] = []
        with self.assertRaisesRegex(ValueError, "different image ownership"):
            self.check()

    def test_status_cannot_reassign_case_ownership(self):
        self.status["cases"][1]["image_id"] = "a"
        with self.assertRaisesRegex(ValueError, "declared case status or image ownership differs"):
            self.check()

    def test_unknown_disposition_and_reused_discovery_id_are_rejected(self):
        self.image("a")["case_dispositions"].append({"case_id": "unknown", "status": "resolved"})
        with self.assertRaisesRegex(ValueError, "unknown parent case"):
            self.check()
        self.image("a")["case_dispositions"].pop()
        self.image("a")["new_issues"][0]["issue_id"] = "old-a"
        with self.assertRaisesRegex(ValueError, "reuses an existing case ID"):
            self.check()

    def test_duplicate_case_or_image_inventory_is_rejected(self):
        self.status["cases"].append(copy.deepcopy(self.status["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate review cases"):
            self.check()
        self.status["cases"].pop()
        self.batches["review/batch-2.json"]["images"].append(copy.deepcopy(self.image("a")))
        with self.assertRaisesRegex(ValueError, "multiple batch reviews"):
            self.check()

    def test_per_image_pending_cannot_omit_duplicate_or_borrow_a_case(self):
        for pending in ([], ["new-a", "new-a"], ["old-b"]):
            with self.subTest(pending=pending):
                self.coverage["images"][0]["unresolved_case_ids"] = pending
                with self.assertRaisesRegex(ValueError, "pending cases"):
                    self.check()

    def test_coverage_claims_must_match_the_evidenced_image_record(self):
        mutations = (("review_path", "review/batch-2.json", "wrong batch review"),
                     ("input_partial_rows", [1], "partial rows differ"),
                     ("source_image_sha256", "0" * 64, "source image differs"),
                     ("image_review_complete", False, "completion differs"))
        for field, value, message in mutations:
            with self.subTest(field=field):
                parent, batches, status, coverage = fixture()
                coverage["images"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    reconcile_review_inventory(parent, batches, status, coverage)

    def test_a_reviewed_image_cannot_disappear_from_coverage(self):
        self.coverage["images"].pop()
        with self.assertRaisesRegex(ValueError, "coverage image inventory differs"):
            self.check()

    def test_mixed_input_candidates_are_rejected(self):
        self.batches["review/batch-1.json"]["input_candidate"]["sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "different input candidate"):
            self.check()

    def test_parent_status_cannot_come_from_another_reviewed_candidate(self):
        self.parent["candidate"]["sha256"] = "8" * 64
        with self.assertRaisesRegex(ValueError, "parent review status belongs to a different input candidate"):
            self.check()

    def test_incomplete_image_is_not_cosmetically_certified(self):
        self.image("a")["image_review_complete"] = False
        with self.assertRaisesRegex(ValueError, "completion differs"):
            self.check()
        self.coverage["images"][0]["image_review_complete"] = False
        self.assertEqual(self.check(), ["old-b", "new-a"])


class ReviewEvidenceChecks(unittest.TestCase):
    def test_batch_path_and_bytes_must_be_bound_by_accepted_evidence(self):
        with tempfile.TemporaryDirectory(prefix="observed-review-check-") as directory:
            root = Path(directory)
            path = root / "batch.json"
            path.write_text(json.dumps({"images": []}))
            descriptor = {"path": "batch.json", "sha256": sha(path)}
            coverage = {"images": [{"review_path": "batch.json"}]}
            review = {"evidence": [descriptor]}
            self.assertEqual(review_batches(root, review, coverage), {"batch.json": {"images": []}})
            with self.assertRaisesRegex(ValueError, "unbound batch review"):
                review_batches(root, {"evidence": []}, coverage)
            path.write_text(json.dumps({"images": [{"image_review_complete": True}]}))
            with self.assertRaises(ValueError):
                review_batches(root, review, coverage)


class InputPartialCoverageChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="observed-input-coverage-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent, self.batches, self.status, self.coverage = fixture()
        full = "0 0.5 0.5 0.4 0.4 0.3 0.7 2 0.7 0.7 2\n"
        partial = "0 0.5 0.5 0.4 0.4 0 0 0 0.7 0.7 2\n"
        missing = "0 0.5 0.5 0.4 0.4 0 0 0 0 0 0\n"
        records = []
        for image_id, split, text in (("a", "train", full * 3),
                                      ("b", "train", full + partial),
                                      ("c", "train", ""), ("v", "val", full)):
            relative = f"labels/{split}/{image_id}.txt"
            path = self.root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            records.append({"id": image_id, "split": split, "label": relative,
                            "label_sha256": sha(path), "image_sha256": image_id * 64})
        source_path = self.root / "data/manifest.json"
        source_path.write_text(json.dumps({"records": records}))
        self.source_sha256 = sha(source_path)
        self.input_path = self.root / "data/candidates/input/manifest.json"
        overlay = self.input_path.parent / records[0]["label"]
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_text(partial + full + missing)
        self.proposal = {"source_manifest_sha256": self.source_sha256, "changes": [{
            "image_id": "a", "source_label": records[0]["label"],
            "candidate_label": records[0]["label"], "source_label_sha256": records[0]["label_sha256"],
            "source_image_sha256": records[0]["image_sha256"], "candidate_label_sha256": sha(overlay)}]}
        self.bind_input_manifest()
        self.coverage.update(reviewed_training_images=3, reviewed_input_partial_rows=3)

    def bind_input_manifest(self):
        self.input_path.write_text(json.dumps(self.proposal))
        descriptor = {"path": self.input_path.relative_to(self.root).as_posix(),
                      "sha256": sha(self.input_path)}
        self.coverage["input_candidate"] = descriptor
        self.parent["candidate"] = copy.deepcopy(descriptor)
        for batch in self.batches.values():
            batch["input_candidate"] = copy.deepcopy(descriptor)

    def check(self):
        return check_input_partial_coverage(self.root, self.coverage, self.source_sha256)

    def test_overlay_source_fallback_and_empty_image_define_exact_coverage(self):
        self.assertEqual(self.check(), {"reviewed_training_images": 3, "reviewed_input_partial_rows": 3})

    def test_matching_batch_and_coverage_cannot_jointly_omit_an_actual_partial_row(self):
        self.coverage["images"][0]["input_partial_rows"] = [1]
        self.batches["review/batch-1.json"]["images"][0]["reviewed_partial_rows"] = [1]
        self.coverage["reviewed_input_partial_rows"] = 2
        self.assertEqual(reconcile_review_inventory(self.parent, self.batches, self.status, self.coverage),
                         ["old-b", "new-a"])
        with self.assertRaisesRegex(ValueError, "partial rows differ from bound input labels: a"):
            self.check()

    def test_input_manifest_and_both_effective_label_sources_are_hash_bound(self):
        for path in (self.input_path,
                     self.input_path.parent / "labels/train/a.txt",
                     self.root / "data/labels/train/b.txt"):
            with self.subTest(path=path.relative_to(self.root)):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "bound file SHA-256 mismatch"):
                    self.check()
                path.write_bytes(original)

    def test_input_cannot_switch_the_frozen_source_manifest(self):
        self.proposal["source_manifest_sha256"] = "9" * 64
        self.bind_input_manifest()
        with self.assertRaisesRegex(ValueError, "different source manifest"):
            self.check()

    def test_coverage_totals_must_equal_actual_input_image_and_partial_counts(self):
        for field, value in (("reviewed_training_images", 2), ("reviewed_input_partial_rows", 2)):
            with self.subTest(field=field):
                self.coverage[field] = value
                with self.assertRaisesRegex(ValueError, field + " differs from bound input labels"):
                    self.check()
                self.coverage[field] = 3


class AnnotationReadinessChecks(unittest.TestCase):
    def setUp(self):
        self.reviewed = {str(number): {"image_review_complete": True} for number in range(137)}
        self.reference = {"remaining_current_rows": [{"image_id": "new_000007", "row": 7,
                                                     "status": "held", "reason": "Unsettled front identity"}]}
        self.protocol = {"training_readiness": "blocked_label_review",
                         "evaluation": {"remaining_reference_rows": 1}}

    def test_settled_training_stays_blocked_by_a_development_reference_row(self):
        self.assertEqual(check_annotation_readiness(self.protocol, self.reviewed, [], self.reference,
                                                   require_ready=False),
                         self.reference["remaining_current_rows"])
        with self.assertRaisesRegex(ValueError, r"1 unresolved development-reference rows \(new_000007:row7\)"):
            check_annotation_readiness(self.protocol, self.reviewed, [], self.reference, require_ready=True)
        self.protocol["training_readiness"] = "ready"
        with self.assertRaisesRegex(ValueError, "declared training readiness differs"):
            check_annotation_readiness(self.protocol, self.reviewed, [], self.reference, require_ready=False)

    def test_protocol_cannot_hide_the_reference_pending_count(self):
        self.protocol["evaluation"]["remaining_reference_rows"] = 0
        with self.assertRaisesRegex(ValueError, "pending count differs from its manifest"):
            check_annotation_readiness(self.protocol, self.reviewed, [], self.reference, require_ready=False)

    def test_readiness_requires_both_training_and_development_review_to_finish(self):
        self.reference["remaining_current_rows"] = []
        self.protocol["evaluation"]["remaining_reference_rows"] = 0
        with self.assertRaisesRegex(ValueError, "1 unresolved training case groups"):
            check_annotation_readiness(self.protocol, self.reviewed, ["old-a"], self.reference,
                                       require_ready=True)
        self.reviewed["0"]["image_review_complete"] = False
        with self.assertRaisesRegex(ValueError, "1 incomplete training images"):
            check_annotation_readiness(self.protocol, self.reviewed, [], self.reference, require_ready=True)
        self.reviewed["0"]["image_review_complete"] = True
        self.protocol["training_readiness"] = "ready"
        self.assertEqual(check_annotation_readiness(self.protocol, self.reviewed, [], self.reference,
                                                   require_ready=True), [])


class HistoricalReplayChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="observed-history-check-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for relative in ("data", "docs", "results/review", "metadata", "weights", "scripts"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        repository = Path(__file__).resolve().parents[1]
        for destination, descriptor in HISTORICAL_REPLAY_CODE.items():
            archive = self.root / descriptor["path"]
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes((repository / descriptor["path"]).read_bytes())
            (self.root / destination).write_text("live recipe extension must survive\n")
        expected = {relative: descriptor["sha256"] for relative, descriptor in HISTORICAL_REPLAY_CODE.items()}
        # A small child composer isolates the snapshot/restore contract; the
        # real frozen chain is checked separately against the existing candidate.
        (self.root / "scripts/compose_observed_pose_review.py").write_text(
            "import hashlib\n"
            "def compose(root, review):\n"
            f"    expected = {expected!r}\n"
            "    for relative, digest in expected.items():\n"
            "        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest\n"
            "    return {'manifest.json': b'{\"candidate_id\":\"isolated-test\"}\\n'}, "
            "{'candidate_id': 'isolated-test'}\n")
        self.review = self.root / "results/review/input.json"
        self.review.write_text("{}\n")

    def test_replay_restores_snapshot_only_and_needs_no_git_history(self):
        before = {relative: (self.root / relative).read_bytes() for relative in HISTORICAL_REPLAY_CODE}
        self.assertFalse((self.root / ".git").exists())
        payloads, manifest = compose_observed_review(self.root, self.review)
        self.assertEqual(manifest, {"candidate_id": "isolated-test"})
        self.assertEqual(json.loads(payloads["manifest.json"]), manifest)
        self.assertEqual({relative: (self.root / relative).read_bytes() for relative in before}, before)

    def test_archived_code_must_match_the_original_pinned_hash(self):
        descriptor = next(iter(HISTORICAL_REPLAY_CODE.values()))
        (self.root / descriptor["path"]).write_text("altered historical code\n")
        with self.assertRaisesRegex(ValueError, "bound file SHA-256 mismatch"):
            compose_observed_review(self.root, self.review)


if __name__ == "__main__":
    unittest.main()
