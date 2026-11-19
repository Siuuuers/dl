#!/usr/bin/env python3
"""Reconstruct r9 from explicit audits and inspect both materialized input arms.

This creates a new review and input candidate. It never trains, changes model
selection, rewrites a prior review, or pins an existing notebook to new labels.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

from evaluate_pose import parse_pose_labels
from front_review_experiment import FrontReviewExperiment, apply_operations
from verify_incremental_pose_candidate import effective_overlay
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_pose_composed_candidate import PARENT as PR35_CONTROL
from verify_reconciled_pose_labels import bound, encoded, sha


SOURCE_REVISION = "817927dacf5cda2232e9286ec83660abe6700caa"
PARENT = {"path": "data/reviews/fronts-2026-11-19-r8/manifest.json",
          "sha256": "07721c3e5ded57ae289650b9e0a96d424a793d595d769f42809c129e95061ca2"}
REVIEW_ID = "fronts-2026-11-19-r9-recovery"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261119-r9-recovery"
CANDIDATE_ID = "front-review-2026-11-19-r9-recovery"
PROTOCOL_PATH = "experiments/front-review-2026-11-19-r9-recovery-inputs.json"
PRIOR_PROTOCOL = {"path": "experiments/front-review-2026-11-19-r8.json",
                  "sha256": "96426b9d0ba7a07b902698e320a82449c051c6790c4bac25dda709d2f181b7d3"}
POLICY = {"path": "docs/annotation-boundaries-r8.md",
          "sha256": "331ecb1579849eafd00ac811d4e763c636395e2e2b9e238d8a8daeb8d12e7907"}
GROUPS = (("box", 1, 5), ("left", 5, 8), ("right", 8, 11))


def descriptor(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), f"Audit/evidence must be inside the repository: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(document))


def source_state(root):
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, original, gold, original_counts = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda item: (item["split"], item["id"]))
    parents = {item["image_id"]: item for item in parent["files"]}
    before = {}
    for record in records:
        iid = record["id"]
        image = root / "data" / record["image"]
        require(sha(image) == record["image_sha256"], f"Original PNG differs: {iid}")
        path = (root / PARENT["path"]).parent / parents[iid]["label_path"]
        require(sha(path) == parents[iid]["label_sha256"], f"Preserved r8 label differs: {iid}")
        before[iid] = path.read_bytes()
    status = json.loads(bound(root, parent["review_status"]).read_text())
    return parent, records, by_id, original, gold, original_counts, before, status


def normalize_audits(root, paths, by_id, before, inherited_status):
    operations, seen, audit_descriptors, inspected = [], {}, [], set()
    cases = {case["case_id"]: copy.deepcopy(case) for case in inherited_status["cases"]}
    updates = []

    def check_case(case):
        iid = case["image_id"]
        require(iid in by_id and case["split"] == by_id[iid]["split"], f"Unknown case image/split: {iid}")
        require(case["status"] in ("resolved", "pending", "held", "partial"), "Unknown case status")
        require(all(type(row) is int and 1 <= row <= len(before[iid].splitlines())
                    for row in case["base_rows"]), f"Case row outside r8: {case['case_id']}")
        require(all(field in ("box", "left", "right") for field in case["fields"]), "Unknown case field")

    def update_case(case_id, changes, evidence):
        require(case_id in cases, f"Unknown inherited case: {case_id}")
        case = cases[case_id]
        old = {key: copy.deepcopy(case.get(key)) for key in ("status", "base_rows", "fields", "reason")}
        if "before_status" in changes:
            require(changes["before_status"] == case["status"], f"Case status drift: {case_id}")
        for key in ("status", "base_rows", "fields", "reason", "training_handling"):
            if key in changes:
                case[key] = copy.deepcopy(changes[key])
        check_case(case)
        note = {"audit": evidence, "before": old, "decision": copy.deepcopy(changes)}
        case.setdefault("r9_recovery_notes", []).append(note)
        updates.append({"case_id": case_id, **note})
        inspected.add(case["image_id"])

    def add_case(value, evidence):
        case = copy.deepcopy(value)
        require(case["case_id"] not in cases, f"Duplicate new case: {case['case_id']}")
        case.setdefault("status", "held")
        case.setdefault("training_handling", "Retain exact PR35 labels on this whole photograph in both arms."
                        if case["split"] == "train" else "Exploratory development review only; frozen validation/scoring stay unchanged.")
        case["audit"] = evidence
        check_case(case)
        cases[case["case_id"]] = case
        inspected.add(case["image_id"])
        updates.append({"case_id": case["case_id"], "audit": evidence, "new_case": True})

    for path in paths:
        audit = json.loads(path.read_text())
        require(audit.get("schema_version") == 1, "Recovery audit schema must be version 1")
        evidence = descriptor(root, path)
        require(evidence not in audit_descriptors, "Repeated audit file")
        audit_descriptors.append(evidence)
        if "source_review" in audit:
            require(audit["source_review"] == "fronts-2026-11-19-r8", "Audit names another base review")
        if "source_revision" in audit:
            require(audit["source_revision"] == SOURCE_REVISION, "Audit names another source revision")
        inspected.update(audit.get("reviewed_image_ids", []))
        inspected.update(audit.get("scope", {}).get("present_images_screened", []))
        for raw in audit.get("proposed_operations", []):
            if raw.get("accepted") is False:
                continue
            op = copy.deepcopy(raw)
            iid = op["image_id"]
            require(iid in by_id and op.get("split", by_id[iid]["split"]) == by_id[iid]["split"], "Operation split differs")
            require(op.get("action", "replace") == "replace", "Recovery only accepts explicit row replacements")
            op["action"] = "replace"
            op["split"] = by_id[iid]["split"]
            op["reason"] = op.get("reason", op.get("rationale", ""))
            require(isinstance(op["reason"], str) and bool(op["reason"].strip()), "Replacement requires a reason")
            row = op["base_row"]
            require(type(row) is int and 1 <= row <= len(before[iid].splitlines()), "Operation row outside r8")
            require(before[iid].decode().splitlines()[row - 1] == op["before_line"], f"Before-line mismatch: {iid}:{row}")
            require(op["before_line"] != op["after_line"], f"No-op replacement: {iid}:{row}")
            if "source_label_sha256" in op:
                require(hashlib.sha256(before[iid]).hexdigest() == op["source_label_sha256"], f"Audit r8 label hash differs: {iid}")
            old, new = op["before_line"].split(), op["after_line"].split()
            require(len(old) == len(new) == 11 and old[0] == new[0], "Replacement changes class/schema")
            changed = [name for name, a, b in GROUPS if old[a:b] != new[a:b]]
            if "changed_groups" in op:
                require(op["changed_groups"] == changed, f"Declared fields differ: {iid}:{row}")
            op["changed_groups"] = changed
            op["audit_sources"] = [evidence]
            key = (iid, row)
            if key in seen:
                previous = seen[key]
                require(previous["before_line"] == op["before_line"] and previous["after_line"] == op["after_line"],
                        f"Conflicting audit edits: {iid}:{row}")
                previous["audit_sources"].append(evidence)
            else:
                seen[key] = op
                operations.append(op)
            inspected.add(iid)
        for item in audit.get("old_cases", []):
            iid = item["image_id"]
            require(iid in by_id and item["split"] == by_id[iid]["split"], "Old-series audit image/split differs")
            require(hashlib.sha256(before[iid]).hexdigest() == item["review_label_sha256"], "Old-series audit label hash differs")
            require(by_id[iid]["image_sha256"] == item["original_image_sha256"], "Old-series image hash differs")
            require(by_id[iid]["label_sha256"] == item["original_label_sha256"], "Old-series original label hash differs")
            require(item["label_action"] == "none", "Old-series audit unexpectedly requests label edits")
            require(item["recommendation"] in ("resolve_blanket_hold_retain_existing_labels", "replace_blanket_hold_with_one_box_case"),
                    "Unknown old-series disposition")
            update_case(item["source_case_id"], {"status": "resolved", "reason": item["reason"],
                        "resolution": item["recommendation"]}, evidence)
            if item.get("replacement_case"):
                add_case(item["replacement_case"], evidence)
        for item in audit.get("case_reviews", []):
            changes = {"reason": item.get("retained_uncertainty", item.get("reason", ""))}
            if item.get("decision") != "retain_hold":
                require(item.get("case_status_after") in ("resolved", "held", "partial", "pending"), "Case review needs explicit disposition")
                changes["status"] = item["case_status_after"]
            if changes["reason"]:
                update_case(item["case_id"], changes, evidence)
        for item in audit.get("late_cases", []):
            require(item["recommendation"] == "retain_existing_hold" and item["label_action"] == "none", "Unexpected late-case mutation")
            require(item["image_id"] in by_id and item["split"] == by_id[item["image_id"]]["split"], "Late-case image/split differs")
            require(by_id[item["image_id"]]["image_sha256"] == item["original_image_sha256"], "Late-case original image hash differs")
            require(hashlib.sha256(before[item["image_id"]]).hexdigest() == item["review_label_sha256"], "Late-case r8 hash differs")
            update_case(item["source_case_id"], {"reason": item["reason"]}, evidence)
        for item in audit.get("status_updates", []):
            require(isinstance(item.get("reason"), str) and item["reason"].strip(), "Status update needs a reason")
            update_case(item["case_id"], item, evidence)
        for case in audit.get("new_cases", []):
            add_case(case, evidence)
    require(inspected <= set(by_id), "Audit coverage contains nonexistent images")
    for case in cases.values():
        check_case(case)
    operations.sort(key=lambda op: (op["image_id"], op["base_row"]))
    return operations, list(cases.values()), updates, audit_descriptors, sorted(inspected)


class RecoveryExperiment(FrontReviewExperiment):
    def compose(self, root):
        payloads, manifest, receipt = super().compose(root)
        manifest["status"] = "pending_review"
        manifest["release_ready"] = False
        payloads["manifest.json"] = encoded(manifest)
        return payloads, manifest, receipt

    def protocol_document(self, root, manifest, source_receipt):
        previous = json.loads(bound(root, PRIOR_PROTOCOL).read_text())
        evaluation = copy.deepcopy(previous["evaluation"])
        evaluation["current_review_reference"] = self.review
        return {"schema_version": 1, "kind": "recovered_annotation_input_protocol",
                "experiment_id": self.experiment_id, "status": "pending_review", "release_ready": False,
                "training_readiness": "annotation_review_only_not_notebook_pinned",
                "full_dataset_certified": False, "training_run": False, "automatic_promotion": False,
                "notebook_pin": None, "prior_protocol": PRIOR_PROTOCOL,
                "candidate": {"path": self.candidate, "sha256": hashlib.sha256(encoded(manifest)).hexdigest()},
                "source_candidate": self.review, "source_review_status": manifest["source_review_status"],
                "control": PR35_CONTROL, "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
                "changed_images_vs_control": manifest["changed_images_vs_control"],
                "unresolved_case_ids": manifest["unresolved_case_ids"],
                "roles": {"reference": {"counts": source_receipt["control"]["counts"]},
                          "candidate": {"counts": manifest["expected_counts"]}},
                "evaluation": evaluation,
                "limits": ["This materializes reviewable inputs only; existing training dispatch and notebooks are unchanged.",
                           "The interrupted r9 bytes were not recovered. This reconstruction uses the saved r8 base and explicitly hashed audits.",
                           "Inherited labels and passed geometry checks do not certify physical boundaries."]}


def experiment(root):
    return RecoveryExperiment(review=descriptor(root, root / REVIEW_PATH), parent=PARENT,
                              candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH,
                              experiment_id="front-review-r9-recovery-inputs", source_revision=SOURCE_REVISION,
                              round_number=9, experiment_version=8, serialization_repairs=(),
                              protocol_limits=(), review_development_labels=True)


def prepare(root, audits):
    parent, records, by_id, original, gold, original_counts, before, inherited = source_state(root)
    ops, cases, updates, evidence, inspected = normalize_audits(root, audits, by_id, before, inherited)
    by_image = defaultdict(list)
    for op in ops:
        by_image[op["image_id"]].append(op)
    labels, maps, parsed, files = {}, {}, {}, []
    unresolved = [case for case in cases if case["status"] != "resolved"]
    for record in records:
        iid = record["id"]
        labels[iid], maps[iid] = apply_operations(before[iid], by_image[iid], iid)
        parsed[iid] = parse_pose_labels(labels[iid].decode(), *record["original_size"], iid)
        check_geometry(parsed[iid], *record["original_size"], iid)
        held_rows = sorted({row for case in unresolved if case["image_id"] == iid for row in case["base_rows"]})
        c = counts(parsed[iid])
        files.append({"image_id": iid, "split": record["split"], "label_path": record["label"],
                      "label_sha256": hashlib.sha256(labels[iid]).hexdigest(), "source_image_sha256": record["image_sha256"],
                      "objects": c["objects"], "locatable_corners": c["visible_endpoints"], "missing_corners": c["missing_endpoints"],
                      "held_current_rows": held_rows, "changed_in_r9_recovery": iid in by_image and bool(by_image[iid]),
                      "fresh_audit_evidence_present": iid in inspected})
    require(labels["old_000013"] == original["old_000013"], "User-confirmed old_000013 bytes changed")
    status = {"schema_version": 1, "review_id": REVIEW_ID, "status": "reconstructed_focused_review_with_explicit_holds",
              "pending_train_photos": sorted({c["image_id"] for c in unresolved if c["split"] == "train"}),
              "unresolved_development_photos": sorted({c["image_id"] for c in unresolved if c["split"] == "val"}),
              "full_dataset_certified": False, "mechanical_verification_is_not_physical_certification": True,
              "inherited_status": parent["review_status"], "reconstruction_audits": evidence, "cases": cases}
    field_counts = dict(Counter(group for op in ops for group in op["changed_groups"]))
    summary = {"review_id": REVIEW_ID, "reconstruction_not_exact_recovery": True,
               "changed_rows": len(ops), "changed_photos": len({o["image_id"] for o in ops}),
               "field_group_changes": field_counts, "unresolved_cases": len(unresolved),
               "held_training_photos": len(status["pending_train_photos"]),
               "unresolved_development_photos": len(status["unresolved_development_photos"]),
               "expected_review_counts": totals(records, parsed),
               "focused_photo_ids": sorted({o["image_id"] for o in ops} | {c["image_id"] for c in unresolved})}
    summary["focused_photos"] = len(summary["focused_photo_ids"])
    summary["omitted_unchanged_resolved_photos"] = len(records) - summary["focused_photos"]
    return parent, records, labels, maps, files, ops, status, updates, evidence, inspected, summary


def build_review(root, audits):
    data = prepare(root, audits)
    parent, records, labels, maps, files, ops, status, updates, evidence, inspected, summary = data
    review_dir, ledger = (root / REVIEW_PATH).parent, root / LEDGER
    generated = [ledger / name for name in ("operations.json", "row-maps.json", "review-status.json", "status-updates.json", "request-coverage.json", "amendment-summary.json", "training-input-inventory.json", "verification.json")]
    require(not review_dir.exists() and not any(p.exists() for p in generated), "Generated r9 artifacts already exist; use --verify-only")
    require(not (root / f"data/candidates/{CANDIDATE_ID}").exists() and not (root / PROTOCOL_PATH).exists(), "r9 candidate already exists")
    for record in records:
        path = review_dir / record["label"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(labels[record["id"]])
    coverage = {"schema_version": 1, "review_id": REVIEW_ID, "base_review": PARENT,
                "source_revision": SOURCE_REVISION, "reconstruction_audits": evidence,
                "targeted_image_ids": inspected, "complete_label_file_count": len(records),
                "scope": "Focused recovery audits plus unchanged inherited r8 rows. This is not an assertion of a new full-row inspection or exact recovery of the interrupted run.",
                "images": [{"image_id": r["id"], "split": r["split"], "source_image_sha256": r["image_sha256"],
                            "has_fresh_audit_evidence": r["id"] in inspected,
                            "changed_base_rows": [o["base_row"] for o in ops if o["image_id"] == r["id"]]}
                           for r in records]}
    for name, value in (("operations.json", ops), ("row-maps.json", maps), ("review-status.json", status),
                        ("status-updates.json", updates), ("request-coverage.json", coverage), ("amendment-summary.json", summary)):
        write_json(ledger / name, value)
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "pending_review", "release_ready": False,
                "date": "2026-11-19", "parent": PARENT, "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": parent["source_manifest_sha256"], "full_dataset_certified": False,
                "original_authoring_bytes_recovered": False,
                "reconstruction": {"basis": "Preserved r8 labels plus explicit original-image recovery audits",
                                   "interrupted_r9_ledger_recovered": False, "prior_reported_counts_used_as_targets": False},
                "annotation_policy": POLICY, "boundary_rubric": POLICY, "review_evidence": evidence,
                "user_references": [], "inherited_row_maps": parent["row_maps"],
                "requested_image_ids": inspected, "complete_atlas_image_ids": sorted(r["id"] for r in records),
                "additional_serialization_repairs": [], "operation_counts": {"replace": len(ops)},
                "field_group_changes": summary["field_group_changes"], "expected_counts": summary["expected_review_counts"],
                "files": files}
    for key, name in (("operations", "operations.json"), ("row_maps", "row-maps.json"),
                      ("review_status", "review-status.json"), ("request_coverage", "request-coverage.json")):
        manifest[key] = descriptor(root, ledger / name)
    write_json(root / REVIEW_PATH, manifest)
    experiment(root).build(root)
    return summary


def frozen_references(root):
    prior = json.loads(bound(root, PRIOR_PROTOCOL).read_text())
    result = {}
    for role, expected_count in (("primary", 666), ("diagnostic", 663)):
        setting = prior["evaluation"]
        desc = {"path": setting[role + "_reference"], "sha256": setting[role + "_reference_sha256"]}
        path = bound(root, desc)
        ref = json.loads(path.read_text())
        require(ref["counts"]["objects"] == expected_count, "Frozen scoring count changed")
        for item in ref["files"]:
            require(sha(path.parent / item["label_path"]) == item["label_sha256"], "Frozen scoring label bytes changed")
        result[role] = {**desc, "counts": ref["counts"], "all_label_hashes_verified": True}
    return result


def materialize_and_verify(root, stage, create):
    exp = experiment(root)
    inspected = exp.verify_candidate(root)
    review, status, records, original, gold, _, _ = exp.read_review(root)
    control_path = bound(root, {"path": PR35_CONTROL["manifest"], "sha256": PR35_CONTROL["sha256"]})
    control = effective_overlay(root, control_path, records, original)
    candidate_path = root / exp.candidate
    candidate = json.loads(candidate_path.read_text())
    changed_original = {c["image_id"]: c for c in candidate["changes"]}
    require(not stage.is_relative_to(root), "Stage actual inputs outside the repository")
    if create:
        require(not stage.exists(), "Choose a fresh staging directory")
        for arm in ("reference", "candidate"):
            for r in records:
                image = stage / arm / r["image"]
                label = stage / arm / r["label"]
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "data" / r["image"], image)
                if arm == "reference":
                    payload = control[r["id"]]["bytes"]
                else:
                    change = changed_original.get(r["id"])
                    payload = (candidate_path.parent / change["candidate_label"]).read_bytes() if change else original[r["id"]]
                label.write_bytes(payload)
            (stage / arm / "data.yaml").write_text(f"path: {stage / arm}\ntrain: images/train\nval: images/val\nnames:\n  0: cassette_front\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n")
    parsed, inventory = {a: {} for a in ("reference", "candidate")}, []
    changed, held_verified, validation_verified = [], [], []
    held = set(candidate["retained_legacy_image_ids"])
    for r in records:
        iid = r["id"]
        require(sha(root / "data" / r["image"]) == r["image_sha256"] and sha(root / "data" / r["label"]) == r["label_sha256"],
                f"Original image or label changed: {iid}")
        arms = {}
        for arm in ("reference", "candidate"):
            image, label = stage / arm / r["image"], stage / arm / r["label"]
            require(image.is_file() and label.is_file() and not image.is_symlink() and not label.is_symlink(), "Staged input is not materialized bytes")
            require(sha(image) == r["image_sha256"], f"Staged original image differs: {arm}/{iid}")
            arms[arm] = label.read_bytes()
            parsed[arm][iid] = parse_pose_labels(arms[arm].decode(), *r["original_size"], str(label))
            check_geometry(parsed[arm][iid], *r["original_size"], str(label))
            if arm == "reference":
                expected = control[iid]["bytes"]
            else:
                c = changed_original.get(iid)
                expected = (candidate_path.parent / c["candidate_label"]).read_bytes() if c else original[iid]
            require(arms[arm] == expected, f"Staged {arm} differs from its exact composed source: {iid}")
            inventory.append({"arm": arm, "image_id": iid, "split": r["split"],
                              "image_sha256": sha(image), "label_sha256": sha(label)})
        if arms["reference"] != arms["candidate"]:
            require(r["split"] == "train" and iid not in held, "Held or validation supervision differs")
            changed.append(iid)
        if iid in held:
            require(arms["reference"] == arms["candidate"] == control[iid]["bytes"], "Held photo is not exact PR35 in both arms")
            held_verified.append(iid)
        if r["split"] == "val":
            require(arms["reference"] == arms["candidate"] == original[iid], "Original validation labels changed")
            validation_verified.append(iid)
    expected_inventory = {r[key] for r in records for key in ("image", "label")}
    for arm in ("reference", "candidate"):
        actual = {p.relative_to(stage / arm).as_posix() for folder in ("images", "labels") for p in (stage / arm / folder).rglob("*") if p.is_file()}
        require(actual == expected_inventory, f"Staged input inventory differs: {arm}")
    require(sorted(changed) == sorted(candidate["changed_images_vs_control"]), "Actual changed-photo count differs from manifest")
    require(sorted(held_verified) == sorted(held) and len(validation_verified) == 20, "Incomplete held/validation verification")
    staged_counts = {arm: totals(records, objects) for arm, objects in parsed.items()}
    protocol = json.loads((root / PROTOCOL_PATH).read_text())
    require(staged_counts["reference"] == protocol["roles"]["reference"]["counts"]
            and staged_counts["candidate"] == candidate["expected_counts"], "Actual staged counts differ")
    require((root / REVIEW_PATH).parent.joinpath("labels/train/old_000013.txt").read_bytes() == original["old_000013"], "old_000013 changed")
    require(all((stage / arm / "labels/train/old_000013.txt").read_bytes() == original["old_000013"]
                for arm in ("reference", "candidate")), "Staged old_000013 differs from original bytes")
    refs = frozen_references(root)
    unresolved = [case for case in status["cases"] if case["status"] != "resolved"]
    report = {"status": "verified", "review_status": "pending_review", "release_ready": False,
              "training_run": False, "automatic_promotion": False,
              "review": descriptor(root, root / REVIEW_PATH), "parent": PARENT,
              "candidate": descriptor(root, candidate_path), "input_protocol": descriptor(root, root / PROTOCOL_PATH),
              "staging_directory": str(stage), "materialized_regular_files_not_symlinks": True,
              "original_images_verified": 157, "all_original_labels_unchanged": True,
              "staged_counts": staged_counts, "original_validation_photos_byte_identical_in_both_arms": 20,
              "unresolved_review_cases": len(unresolved),
              "held_training_cases": len(candidate["unresolved_case_ids"]),
              "held_training_case_ids": candidate["unresolved_case_ids"],
              "unresolved_development_cases": sum(case["split"] == "val" for case in unresolved),
              "held_training_photos_exact_pr35_in_both_arms": len(held_verified),
              "held_training_photo_ids": sorted(held_verified), "changed_training_photos_vs_control": len(changed),
              "changed_training_photo_ids": sorted(changed), "old_000013_review_exact_original_bytes": True,
              "old_000013_staged_exact_original_bytes_in_both_arms": True,
              "frozen_references": refs, "builder_sha256": sha(Path(__file__)),
              "annotation_candidate_verification": inspected,
              "limits": ["Mechanical verification checks saved decisions and actual staged files; it is not physical certification.",
                         "This is an evidence-based reconstruction, not the missing interrupted r9 byte sequence.",
                         "The existing training dispatcher and notebook remain unchanged; these staged inputs are for review."]}
    inventory_path = root / LEDGER / "training-input-inventory.json"
    if create:
        inventory_document = {"schema_version": 1, "files": inventory, "counts": staged_counts,
                              "training_run": False, "review": report["review"], "candidate": report["candidate"]}
        write_json(stage / "staging-manifest.json", inventory_document)
        write_json(inventory_path, inventory_document)
    else:
        staged = json.loads((stage / "staging-manifest.json").read_text())
        require(staged["files"] == inventory and staged["counts"] == staged_counts, "Staging receipt differs from actual bytes")
        require(staged["review"] == report["review"] and staged["candidate"] == report["candidate"], "Staging receipt source identities differ")
        require(inventory_path.read_bytes() == (stage / "staging-manifest.json").read_bytes(),
                "Repository training-input inventory differs from actual staging receipt")
    report["training_input_inventory"] = descriptor(root, inventory_path)
    report["staging_manifest_sha256"] = sha(stage / "staging-manifest.json")
    write_json(root / LEDGER / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit", action="append", type=Path, default=[], help="Audit JSON inside the repo; repeat in intended integration order")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build", action="store_true", help="Create new review, compose candidate, and stage/verify actual files")
    mode.add_argument("--verify-only", action="store_true", help="Inspect existing composed artifacts and materialized staging")
    mode.add_argument("--check-audits", action="store_true", help="Validate/preview supplied edits without writing labels (default)")
    parser.add_argument("--stage", type=Path, help="Fresh directory outside repository for materialized arms")
    args = parser.parse_args()
    root = args.root.resolve()
    audits = [(p if p.is_absolute() else root / p).resolve() for p in args.audit]
    if args.verify_only:
        stage = args.stage
        if stage is None:
            stage = Path(json.loads((root / LEDGER / "verification.json").read_text())["staging_directory"])
        report = materialize_and_verify(root, stage.resolve(), False)
        print(json.dumps({k: report[k] for k in ("status", "original_images_verified", "held_training_photos_exact_pr35_in_both_arms", "changed_training_photos_vs_control", "staged_counts")}, indent=2))
        return
    require(bool(audits), "Supply at least one explicit --audit file")
    if not args.build:
        print(json.dumps(prepare(root, audits)[-1], indent=2))
        return
    stage = (args.stage or root.parent / "r9-recovery-training-inputs").resolve()
    require(not stage.exists() and not stage.is_relative_to(root), "Choose a fresh staging directory outside the repository")
    summary = build_review(root, audits)
    verification = materialize_and_verify(root, stage, True)
    print(json.dumps({"review_summary": summary, "actual_staged_counts": verification["staged_counts"],
                      "actual_changed_training_photos": verification["changed_training_photos_vs_control"],
                      "verification_path": f"{LEDGER}/verification.json"}, indent=2))


if __name__ == "__main__":
    main()
