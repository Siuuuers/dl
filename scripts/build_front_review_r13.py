#!/usr/bin/env python3
"""Replay the requested round 13 amendments and verify the actual paired inputs."""
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

SOURCE_REVISION = "eb8284dcab71074275d9d22fc0fd974839d66863"
PARENT = {"path": "data/reviews/fronts-2026-11-20-r12/manifest.json",
          "sha256": "2ccac69fd2cff2c5748a6063c743271b5dd92c3f1f7493542486d1eddb31602f"}
REVIEW_ID = "fronts-2026-11-21-r13"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261121-r13"
CANDIDATE_ID = "front-review-2026-11-21-r13"
PROTOCOL_PATH = "experiments/front-review-2026-11-21-r13.json"
GROUPS = (("box", 1, 5), ("left", 5, 8), ("right", 8, 11))


def descriptor(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "Evidence must be in the repository")
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def source_state(root):
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, original, gold, original_counts = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    before = {}
    for item in parent["files"]:
        iid = item["image_id"]
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(sha(path) == item["label_sha256"], f"Parent label drift: {iid}")
        require(sha(root / "data" / by_id[iid]["image"]) == by_id[iid]["image_sha256"], f"Original image drift: {iid}")
        before[iid] = path.read_bytes()
    return parent, records, by_id, original, gold, original_counts, before


def normalize_audits(root, paths, by_id, before, inherited):
    cases = {c["case_id"]: copy.deepcopy(c) for c in inherited["cases"]}
    for case in cases.values():
        case["r12_base_rows"] = list(case["base_rows"])
        case["base_rows"] = list(case["current_rows"])
    operations, updates, evidence, inspected, seen, photo_reviews = [], [], [], set(), set(), []
    for path in paths:
        audit = json.loads(path.read_text())
        require(audit.get("schema_version") == 1 and audit.get("source_review") == "fronts-2026-11-20-r12",
                f"Wrong audit source: {path}")
        desc = descriptor(root, path)
        require(desc not in evidence, "Repeated audit")
        evidence.append(desc)
        ids = audit["reviewed_image_ids"]
        require(not inspected.intersection(ids) and all(i in by_id and i.startswith("new_") for i in ids),
                "Primary visual audits must cover disjoint new-photo sets")
        inspected.update(ids)
        coverage = audit.get("photo_reviews", audit.get("image_reviews", audit.get("coverage", [])))
        require(isinstance(coverage, list) and {r["image_id"] for r in coverage} == set(ids)
                and len(coverage) == len(ids), "Audit must include one review record for every assigned photo")
        photo_reviews.extend(coverage)
        for item in audit.get("proposed_operations", []):
            op = copy.deepcopy(item)
            iid = op["image_id"]
            require(iid in ids and iid.startswith("new_") and op.get("split", by_id[iid]["split"]) == by_id[iid]["split"],
                    "Operation outside assigned new-photo scope")
            op["split"] = by_id[iid]["split"]
            action = op.get("action", op.get("operation", "replace"))
            op["action"] = action
            require(action in ("replace", "delete", "add") and isinstance(op.get("reason"), str) and op["reason"].strip(),
                    "Operation needs an action and physical reason")
            if "source_label_sha256" in op:
                require(op["source_label_sha256"] == hashlib.sha256(before[iid]).hexdigest(), "Audit source-label drift")
            if action == "add":
                identity = (iid, "add", op["addition_id"])
                op["changed_groups"] = ["instance", "box", "left", "right"]
            else:
                row = op["base_row"]
                require(type(row) is int and 1 <= row <= len(before[iid].splitlines()), "Invalid immutable base row")
                require(op["before_line"] == before[iid].decode().splitlines()[row - 1], f"Before-line mismatch: {iid}:{row}")
                identity = (iid, row)
                if action == "replace":
                    a, b = op["before_line"].split(), op["after_line"].split()
                    require(len(a) == len(b) == 11 and a[0] == b[0] and a != b, "Invalid/no-op replacement")
                    groups = [name for name, start, stop in GROUPS if a[start:stop] != b[start:stop]]
                    require(op.get("changed_groups", groups) == groups, f"Field declaration mismatch: {iid}:{row}")
                    op["changed_groups"] = groups
                    original_source = op.get("original_source")
                    if original_source:
                        original_path = bound(root, {"path": original_source["path"], "sha256": original_source["sha256"]})
                        original_line = original_path.read_text().splitlines()[original_source["row"] - 1]
                        require(original_line == original_source["line"] and original_source["mapping_visually_verified"] is True,
                                "Original restore lacks verified row identity")
                        restored = original_source["restored_groups"]
                    elif op.get("restored_original_groups"):
                        original_path = bound(root, {"path": op["original_label_path"], "sha256": op["original_label_sha256"]})
                        original_line = original_path.read_text().splitlines()[op["original_row"] - 1]
                        require(original_line == op["original_line"] and op["visually_verified_mapping"]["same_physical_front"] is True,
                                "Original restore lacks verified physical mapping")
                        restored = op["restored_original_groups"]
                    else:
                        restored = []
                    for name, start, stop in GROUPS:
                        if name in restored:
                            require(b[start:stop] == original_line.split()[start:stop], "Restored field is not exact original tokens")
                else:
                    op["changed_groups"] = ["instance"]
            require(identity not in seen, f"Duplicate edit identity: {identity}")
            seen.add(identity)
            op["audit_source"] = desc
            operations.append(op)
        for update in audit.get("status_updates", []):
            cid = update["case_id"]
            require(cid in cases and update.get("status") in ("resolved", "held", "pending", "partial")
                    and update.get("reason"), "Invalid case decision")
            require(cases[cid]["image_id"] in ids or cases[cid]["image_id"].startswith("old_"), "Case decision outside scope")
            prior = {k: copy.deepcopy(cases[cid].get(k)) for k in ("status", "base_rows", "fields", "reason")}
            for key in ("status", "base_rows", "fields", "reason"):
                if key in update:
                    cases[cid][key] = copy.deepcopy(update[key])
            note = {"audit": desc, "before": prior, "decision": update}
            cases[cid].setdefault("r13_notes", []).append(note)
            updates.append({"case_id": cid, **note})
        for value in audit.get("new_cases", []):
            case = copy.deepcopy(value)
            require(case["case_id"] not in cases and case["image_id"] in ids, "New case outside scope/duplicate")
            case.setdefault("status", "held")
            case["audit"] = desc
            cases[case["case_id"]] = case
    for case in cases.values():
        iid = case["image_id"]
        require(case["status"] in ("resolved", "held", "pending", "partial") and case["split"] == by_id[iid]["split"], "Invalid case state")
        require(all(type(r) is int and 1 <= r <= len(before[iid].splitlines()) for r in case["base_rows"]), "Case row is not a round 12 identity")
        require(all(f in ("box", "left", "right") for f in case["fields"]), "Unknown case field")
        if case["status"] != "resolved":
            require(not iid.startswith("old_"), "User-confirmed old photos cannot be unresolved in r13")
    operations.sort(key=lambda o: (o["image_id"], o.get("base_row", o.get("after_base_row", 0)), o["action"], o.get("addition_id", "")))
    return operations, list(cases.values()), updates, evidence, sorted(inspected), photo_reviews


def experiment(root):
    source = json.loads((root / "data/manifest.json").read_text())
    excluded = tuple(sorted(r["id"] for r in source["records"] if r["id"].startswith("old_")))
    return FrontReviewExperiment(review=descriptor(root, root / REVIEW_PATH), parent=PARENT,
        candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH, experiment_id="front-review-r13",
        source_revision=SOURCE_REVISION, round_number=13, experiment_version=12, serialization_repairs=(),
        protocol_limits=("Targeted user-requested amendments remain fallible under blur/occlusion; specific unresolved cases are explicit.",
                        "The accepted old cohort is excluded from the treatment, distinct from unresolved holds.",
                        "No training or automatic promotion occurs while preparing this notebook."),
        review_development_labels=True, excluded_training_image_ids=excluded)


def build_review(root, audits):
    parent, records, by_id, original, gold, original_counts, before = source_state(root)
    inherited = json.loads(bound(root, parent["review_status"]).read_text())
    ops, cases, updates, evidence, inspected, photo_reviews = normalize_audits(root, audits, by_id, before, inherited)
    requests = json.loads((root / LEDGER / "user-reference-index.json").read_text())
    requested = requests["requested_image_ids"]
    require(inspected == requested and len(requested) == 8, "Every requested photo needs one fresh visual audit record")
    review_dir, ledger = (root / REVIEW_PATH).parent, root / LEDGER
    require(not review_dir.exists() and not (ledger / "operations.json").exists(), "Generated r13 already exists; verify instead")
    by_image = defaultdict(list)
    for op in ops:
        by_image[op["image_id"]].append(op)
    labels, maps, parsed, files = {}, {}, {}, []
    unresolved = [c for c in cases if c["status"] != "resolved"]
    for rec in records:
        iid = rec["id"]
        labels[iid], maps[iid] = apply_operations(before[iid], by_image[iid], iid)
        parsed[iid] = parse_pose_labels(labels[iid].decode(), *rec["original_size"], iid)
        check_geometry(parsed[iid], *rec["original_size"], iid)
        if iid.startswith("old_"):
            require(labels[iid] == before[iid], "Old review label changed")
        lookup = {r["base_row"]: r["current_row"] for r in maps[iid]["rows"] if "base_row" in r}
        for case in cases:
            if case["image_id"] == iid:
                case["current_rows"] = [lookup[r] for r in case["base_rows"] if r in lookup]
                if case["status"] != "resolved":
                    require(len(case["current_rows"]) == len(case["base_rows"]), "Unresolved case silently lost its instance")
        c = counts(parsed[iid])
        files.append({"image_id": iid, "split": rec["split"], "label_path": rec["label"],
            "label_sha256": hashlib.sha256(labels[iid]).hexdigest(), "source_image_sha256": rec["image_sha256"],
            "objects": c["objects"], "locatable_corners": c["visible_endpoints"], "missing_corners": c["missing_endpoints"],
            "changed_in_r13": bool(by_image[iid]), "fresh_visual_review": iid in inspected,
            "held_current_rows": sorted({r for case in unresolved if case["image_id"] == iid for r in case["current_rows"]})})
    excluded = sorted(i for i in by_id if i.startswith("old_"))
    status = {"schema_version": 1, "review_id": REVIEW_ID, "status": "targeted_amendment_review_with_explicit_holds",
        "pending_train_photos": sorted({c["image_id"] for c in unresolved if c["split"] == "train"}),
        "unresolved_development_photos": sorted({c["image_id"] for c in unresolved if c["split"] == "val"}),
        "excluded_training_image_ids": excluded, "full_dataset_certified": False,
        "inherited_status": parent["review_status"], "cases": cases}
    summary = {"review_id": REVIEW_ID, "reviewed_new_photos": len(requested), "preserved_old_photos": len(excluded),
        "operation_counts": dict(Counter(o["action"] for o in ops)), "changed_rows": len(ops),
        "changed_photos": len({o["image_id"] for o in ops}),
        "field_group_changes": dict(Counter(g for o in ops if o["action"] == "replace" for g in o["changed_groups"])),
        "unresolved_cases": len(unresolved), "held_training_photos": len(status["pending_train_photos"]),
        "unresolved_development_photos": len(status["unresolved_development_photos"]),
        "expected_review_counts": totals(records, parsed), "changed_image_ids": sorted({o["image_id"] for o in ops}), "pdf_image_ids": sorted(f["image_id"] for f in files if f["image_id"].startswith("new_") and f["objects"] > 0)}
    coverage = {"schema_version": 1, "review_id": REVIEW_ID, "base_review": PARENT, "source_revision": SOURCE_REVISION,
        "reviewed_image_ids": inspected, "excluded_old_image_ids": excluded, "audits": evidence,
        "visual_review_complete_for_requested_photos": True, "full_dataset_certified": False,
        "scope": "Every requested photo inspected in its original scene and r12 overlay with targeted detail crops; unrequested new and all old label bytes inherited unchanged.",
        "images": photo_reviews}
    coverage["request_dispositions"] = []
    for audit_path in audits:
        audit = json.loads(audit_path.read_text())
        requests_for_audit = audit.get("request_dispositions", audit.get("request_coverage", []))
        require(isinstance(requests_for_audit, list) and requests_for_audit,
                "Each audit must record its requested changes and dispositions")
        coverage["request_dispositions"].extend({**item, "audit_source": descriptor(root, audit_path)}
                                               for item in requests_for_audit)
    for rec in records:
        path = review_dir / rec["label"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(labels[rec["id"]])
    for name, value in (("operations.json", ops), ("row-maps.json", maps), ("review-status.json", status),
                        ("status-updates.json", updates), ("request-coverage.json", coverage), ("amendment-summary.json", summary)):
        write_json(ledger / name, value)
    policy = descriptor(root, root / "docs/annotation-boundaries-r13.md")
    references = [descriptor(root, ledger / "user-reference-index.json")] if (ledger / "user-reference-index.json").exists() else []
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
        "status": "visually_reviewed_with_explicit_holds", "date": "2026-11-21", "parent": PARENT,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
        "full_dataset_certified": False, "annotation_policy": policy, "boundary_rubric": policy,
        "review_evidence": evidence, "user_references": references, "inherited_row_maps": parent["row_maps"],
        "requested_image_ids": requested, "changed_image_ids": sorted({o["image_id"] for o in ops}), "pdf_image_ids": sorted(f["image_id"] for f in files if f["image_id"].startswith("new_") and f["objects"] > 0), "excluded_training_image_ids": excluded,
        "additional_serialization_repairs": [], "operation_counts": summary["operation_counts"],
        "field_group_changes": summary["field_group_changes"], "expected_counts": summary["expected_review_counts"], "files": files}
    manifest["secondary_review_evidence"] = [descriptor(root, path) for path in
        sorted(ledger.glob("*-secondary-audit.json"))]
    for key, name in (("operations", "operations.json"), ("row_maps", "row-maps.json"), ("review_status", "review-status.json"), ("request_coverage", "request-coverage.json")):
        manifest[key] = descriptor(root, ledger / name)
    write_json(root / REVIEW_PATH, manifest)
    experiment(root).build(root)
    return summary


def materialize_and_verify(root, stage, create=False):
    exp = experiment(root)
    verification = exp.verify_candidate(root)
    review, status, records, original, gold, original_counts, _ = exp.read_review(root)
    control_path = bound(root, {"path": PR35_CONTROL["manifest"], "sha256": PR35_CONTROL["sha256"]})
    control = effective_overlay(root, control_path, records, original)
    candidate_path = root / exp.candidate
    candidate = json.loads(candidate_path.read_text())
    overlay = {c["image_id"]: c for c in candidate["changes"]}
    held, excluded = set(candidate["retained_legacy_image_ids"]), set(candidate["excluded_training_image_ids"])
    require(not stage.is_relative_to(root), "Stage actual files outside the repository")
    if create:
        require(not stage.exists(), "Choose a fresh input staging directory")
    parsed, inventory, changed = {a: {} for a in ("reference", "candidate")}, [], []
    for rec in records:
        iid = rec["id"]
        require(sha(root / "data" / rec["image"]) == rec["image_sha256"]
                and sha(root / "data" / rec["label"]) == rec["label_sha256"], "Original source changed")
        change = overlay.get(iid)
        expected = {"reference": control[iid]["bytes"], "candidate": (candidate_path.parent / change["candidate_label"]).read_bytes() if change else original[iid]}
        actual = {}
        for arm in ("reference", "candidate"):
            image, label = stage / arm / rec["image"], stage / arm / rec["label"]
            if create:
                image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "data" / rec["image"], image); label.write_bytes(expected[arm])
            require(image.is_file() and label.is_file() and not image.is_symlink() and not label.is_symlink(), "Inputs must be materialized regular files")
            require(sha(image) == rec["image_sha256"] and label.read_bytes() == expected[arm], "Materialized input differs from exact source")
            actual[arm] = label.read_bytes()
            parsed[arm][iid] = parse_pose_labels(actual[arm].decode(), *rec["original_size"], str(label))
            check_geometry(parsed[arm][iid], *rec["original_size"], str(label))
            inventory.append({"arm": arm, "image_id": iid, "split": rec["split"], "image_sha256": sha(image), "label_sha256": sha(label)})
        if actual["reference"] != actual["candidate"]:
            require(rec["split"] == "train" and iid not in held | excluded, "Held, excluded or validation labels differ")
            changed.append(iid)
        if iid in held | excluded:
            require(actual["reference"] == actual["candidate"] == control[iid]["bytes"], "Held/excluded photo not exact PR35 in both arms")
        if rec["split"] == "val":
            require(actual["reference"] == actual["candidate"] == original[iid], "Training validation changed")
    expected_paths = {r[k] for r in records for k in ("image", "label")}
    for arm in parsed:
        actual_paths = {p.relative_to(stage / arm).as_posix() for folder in ("images", "labels") for p in (stage / arm / folder).rglob("*") if p.is_file()}
        require(actual_paths == expected_paths, "Extra/missing materialized training files")
        if create:
            (stage / arm / "data.yaml").write_text(f"path: {stage / arm}\ntrain: images/train\nval: images/val\nnames:\n  0: cassette_front\nkpt_shape: [2, 3]\nflip_idx: [1, 0]\n")
    require(sorted(changed) == sorted(candidate["changed_images_vs_control"]), "Changed-photo inventory differs")
    staged_counts = {arm: totals(records, objects) for arm, objects in parsed.items()}
    protocol = json.loads((root / PROTOCOL_PATH).read_text())
    require(all(staged_counts[a] == protocol["roles"][a]["counts"] for a in parsed), "Actual counts differ from protocol")
    parent_files = {f["image_id"]: f for f in json.loads(bound(root, PARENT).read_text())["files"]}
    for iid in excluded:
        label = parent_files[iid]["label_path"]
        require(((root / REVIEW_PATH).parent / label).read_bytes() == ((root / PARENT["path"]).parent / label).read_bytes(), "Old review bytes changed")
    frozen = {}
    for name, count in (("primary", 666), ("diagnostic", 663)):
        ev = protocol["evaluation"]
        desc = {"path": ev[name + "_reference"], "sha256": ev[name + "_reference_sha256"]}
        path = bound(root, desc); ref = json.loads(path.read_text())
        require(ref["counts"]["objects"] == count, "Frozen reference count changed")
        require(all(sha(path.parent / f["label_path"]) == f["label_sha256"] for f in ref["files"]), "Frozen scoring bytes changed")
        frozen[name] = {**desc, "counts": ref["counts"], "all_label_hashes_verified": True}
    report = {"status": "verified", "training_run": False, "automatic_promotion": False,
        "review": descriptor(root, root / REVIEW_PATH), "candidate": descriptor(root, candidate_path),
        "training_protocol": descriptor(root, root / PROTOCOL_PATH), "staging_directory": str(stage),
        "original_images_verified": len(records), "all_original_labels_unchanged": True,
        "old_review_files_preserved": len(excluded), "held_training_photos_exact_pr35_in_both_arms": len(held),
        "held_training_photo_ids": sorted(held), "excluded_old_photos_exact_pr35_in_both_arms": len(excluded),
        "excluded_training_image_ids": sorted(excluded), "original_validation_photos_byte_identical_in_both_arms": 20,
        "changed_training_photos_vs_control": len(changed), "changed_training_photo_ids": sorted(changed),
        "staged_counts": staged_counts, "frozen_references": frozen, "annotation_candidate_verification": verification,
        "limits": ["Verification proves exact replay and materialized file identity, not infallible physical annotation.", "New training has not been run."]}
    inventory_path = root / LEDGER / "training-input-inventory.json"
    inventory_document = {"schema_version": 1, "review": report["review"], "candidate": report["candidate"], "files": inventory, "counts": staged_counts}
    if create:
        write_json(inventory_path, inventory_document); write_json(stage / "staging-manifest.json", inventory_document)
    else:
        require(json.loads(inventory_path.read_text()) == inventory_document
                and json.loads((stage / "staging-manifest.json").read_text()) == inventory_document, "Saved inventory differs from actual inputs")
    report["training_input_inventory"] = descriptor(root, inventory_path)
    write_json(root / LEDGER / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit", action="append", type=Path, default=[])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.verify_only:
        stage = args.stage or Path(json.loads((root / LEDGER / "verification.json").read_text())["staging_directory"])
        result = materialize_and_verify(root, stage.resolve())
    elif args.build:
        paths = [(p if p.is_absolute() else root / p).resolve() for p in args.audit]
        require(bool(paths), "Supply the explicit audit JSON files")
        summary = build_review(root, paths)
        result = materialize_and_verify(root, (args.stage or root.parent / "r13-training-inputs").resolve(), True)
        result["amendment_summary"] = summary
    else:
        result = experiment(root).verify_candidate(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
