#!/usr/bin/env python3
"""Replay the bounded R31 review after PR109 and compose eligible R31 training labels.

Review coordinates and model experiment roles are separate. The staged reference
is retained R30 supervision; the partial-box-loss runner deliberately uses the
new candidate label arm for BOTH experimental roles.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import apply_operations
from front_review_r27 import FROZEN_REFERENCES, STANDARDS, pixel_geometry, semantic_fields
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree

SOURCE_REVISION = "337a12d8b11f9925b493cba794f29a4ffcf848ed"
PARENT = {"path": "data/reviews/fronts-2026-11-24-r30/manifest.json",
          "sha256": "b3e020fe086a6585b305417c84d7650ea1f7881d86fa3cf77fdc3fbacf4c0f9d"}
CONTROL = {"manifest": "data/candidates/front-review-2026-11-24-r30/manifest.json",
           "sha256": "b74562c7d8f7a66deeb08fdafb3ed16dd6ac00aac7122b43719fa5fa09dd1a80"}
REVIEW_ID = "fronts-2026-11-24-r31"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261124-r31"
AUDIT_PATH = f"{LEDGER}/row-review.json"
AUDIT_SHA256 = "4ac8fdb50529b485cea59eef8b74cd176219bd6e095b45e7412bec2cc5d25173"
CANDIDATE_ID = "front-review-2026-11-24-r31"
CANDIDATE_PATH = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = "experiments/partial-box-loss-2026-11-24-r31.json"
DELTA_PATH = f"{LEDGER}/training-input-delta.json"


def descriptor(path, payload):
    return {"path": path, "sha256": digest(payload)}


def groups(before, after):
    a, b = before.split(), after.split()
    require(len(a) == len(b) == 11, "Expected eleven pose tokens")
    require(all(a[i] == b[i] for i in (0, 7, 10)), "Class/visibility changed")
    for start in (5, 8):
        if float(a[start + 2]) == 0:
            require(a[start:start + 3] == b[start:start + 3]
                    and all(float(v) == 0 for v in b[start:start + 3]), "Missing point changed")
    return [name for name, lo, hi in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
            if a[lo:hi] != b[lo:hi]]


def expected_review(root):
    from front_review_r30 import verify_review as verify_parent
    root = Path(root).resolve()
    verify_parent(root)
    parent = json.loads(bound(root, PARENT).read_bytes())
    audit_ref = {"path": AUDIT_PATH, "sha256": AUDIT_SHA256}
    audit = json.loads(bound(root, audit_ref).read_bytes())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION,
            "Review parent/source identity differs")
    require(audit["full_dataset_certified"] is False and audit["status"] == "accepted",
            "Review scope/acceptance differs")
    for item in audit["independent_evidence"]:
        bound(root, item)
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    parent_files = {r["image_id"]: r for r in parent["files"]}
    before = {}
    for record in records:
        item = parent_files[record["id"]]
        require(item["split"] == record["split"] and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "Source mapping differs")
        require(sha(root / "data" / record["image"]) == record["image_sha256"], "Original pixels differ")
        before[record["id"]] = bound((root / PARENT["path"]).parent,
            {"path": item["label_path"], "sha256": item["label_sha256"]}).read_bytes()
    operations, keys, changed_groups, split_changes = [], set(), Counter(), Counter()
    for row in audit["rows"]:
        image_id, index = row["image_id"], row["row"]
        key = (image_id, index)
        require(key not in keys, "Duplicate review identity")
        keys.add(key)
        record, item = by_id[image_id], parent_files[image_id]
        require(row["before_line"] == before[image_id].decode().splitlines()[index-1], "Before row drift")
        require(row["source_image_sha256"] == record["image_sha256"]
                and row["source_label_sha256"] == item["label_sha256"]
                and row["split"] == record["split"], "Row provenance differs")
        changed = groups(row["before_line"], row["after_line"])
        require(changed == row["changed_groups"] and bool(changed) == (row["action"] == "replace"),
                "Action/group mismatch")
        require(row["reason"] and row["coordinate_uncertainty"]
                and row["original_pixels_inspected"] is True, "Missing review rationale")
        for name, line in (("before_pixels", "before_line"), ("after_pixels", "after_line")):
            require(row[name] == pixel_geometry(row[line], str(key)), "Decoded display differs")
        if not changed:
            require(row["before_line"] == row["after_line"], "Retained row differs")
            continue
        require(image_id not in parent["excluded_training_image_ids"] + ["new_000015"],
                "Held/excluded training photograph changed")
        require(key not in STANDARDS, "Protected standard changed")
        changed_groups.update(changed)
        split_changes[record["split"]] += 1
        operations.append({"image_id": image_id, "split": record["split"], "base_row": index,
                           "action": "replace", "before_line": row["before_line"],
                           "after_line": row["after_line"], "changed_groups": changed,
                           "changed_fields": semantic_fields(row["before_pixels"], row["after_pixels"]),
                           "reason": row["reason"]})
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))
    payloads, files, parsed, maps = {}, [], {}, {}
    for record in records:
        image_id = record["id"]
        ops = [op for op in operations if op["image_id"] == image_id]
        payload, maps[image_id] = apply_operations(before[image_id], ops, image_id)
        require(len(payload.splitlines()) == len(before[image_id].splitlines()), "Row count changed")
        require(not maps[image_id]["deleted_base_rows"]
                and all(x["current_row"] == x["base_row"] for x in maps[image_id]["rows"]), "Row IDs changed")
        parsed[image_id] = parse_pose_labels(payload.decode(), *record["original_size"], image_id)
        check_geometry(parsed[image_id], *record["original_size"], image_id)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        tally = counts(parsed[image_id])
        files.append({"image_id": image_id, "split": record["split"], "label_path": record["label"],
                      "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                      "objects": tally["objects"], "locatable_corners": tally["visible_endpoints"],
                      "missing_corners": tally["missing_endpoints"], "changed_in_r31": bool(ops)})
    review_counts = totals(records, parsed)
    require(review_counts == parent["expected_counts"], "Object/visibility counts changed")
    changed_ids = sorted({x["image_id"] for x in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT, "reviewed_rows": len(keys),
               "replacement_rows": len(operations), "retained_rows": len(keys)-len(operations),
               "changed_photos": len(changed_ids), "changed_image_ids": changed_ids,
               "changed_rows_by_split": dict(split_changes), "field_group_changes": dict(changed_groups),
               "added_rows": 0, "deleted_rows": 0, "expected_review_counts": review_counts,
               "protected_standards_unchanged": len(STANDARDS), "full_dataset_certified": False,
               "training_run": False, "automatic_promotion": False}
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_bytes()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"],
                  current_review_scope="Only the PR109 user requests referring to the exact PR107/R30 report, with bounded same-row corrections.",
                  current_review_rows=[{"image_id": r["image_id"], "row": r["row"], "action": r["action"]}
                                       for r in audit["rows"]], current_row_review=audit_ref,
                  full_dataset_certified=False)
    docs = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
            "review_status": ("review-status.json", status), "amendment_summary": ("amendment-summary.json", summary),
            "request_coverage": ("request-coverage.json", audit["request_coverage"])}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "bounded_original_pixel_review_ready", "date": "2026-11-24", "parent": PARENT,
                "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
                "component": "immutable_annotation_review", "review_only": True, "full_dataset_certified": False,
                "annotation_policy": parent["annotation_policy"], "boundary_rubric": parent["boundary_rubric"],
                "review_method": audit["review_method"], "review_evidence": [audit_ref],
                "independent_review_evidence": audit["independent_evidence"],
                "inherited_user_references": parent["user_references"], "user_references": [],
                "changed_image_ids": changed_ids, "reviewed_image_ids": sorted({i for i, _ in keys}),
                "requested_image_ids": sorted({x["image_id"] for x in audit["request_coverage"]["identities"]}),
                "excluded_training_image_ids": parent["excluded_training_image_ids"],
                "held_training_image_ids": ["new_000015"], "operation_counts": {"replace": len(operations), "add": 0, "delete": 0},
                "field_group_changes": dict(changed_groups), "expected_counts": review_counts,
                "frozen_references": FROZEN_REFERENCES, "files": files,
                "replay_script": {"path": "scripts/front_review_r31.py", "sha256": sha(root / "scripts/front_review_r31.py")}}
    for key, (name, value) in docs.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def verify_review(root):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        require((root/relative).is_file() and not (root/relative).is_symlink()
                and (root/relative).read_bytes() == payload, "R31 replay differs: " + relative)
    label_dir = root / "data/reviews" / REVIEW_ID / "labels"
    expected = {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}
    actual = {p.relative_to(root).as_posix() for p in label_dir.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual == expected and len(actual) == 157, "Unexpected R31 labels")
    return {"status": "verified", "review_manifest": REVIEW_PATH,
            "review_manifest_sha256": sha(root/REVIEW_PATH), "label_files_verified": 157, **summary}


def compose_candidate(root):
    from front_review_r30 import verify_candidate as verify_parent_candidate
    from verify_incremental_pose_candidate import effective_overlay
    root = Path(root).resolve()
    review_receipt = verify_review(root)
    previous = verify_parent_candidate(root, root/CONTROL["manifest"])
    control_path = bound(root, {"path": CONTROL["manifest"], "sha256": CONTROL["sha256"]})
    prior = json.loads(control_path.read_bytes())
    review = json.loads((root/REVIEW_PATH).read_bytes())
    review_ref = {"path": REVIEW_PATH, "sha256": review_receipt["review_manifest_sha256"]}
    _, records, _, originals, gold, _ = read_reference(root, review["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    control = effective_overlay(root, control_path, records, originals)
    held = sorted(prior["retained_legacy_image_ids"])
    excluded = sorted(prior["excluded_training_image_ids"])
    require(held == ["new_000015"] and len(excluded) == 31, "Hold/exclusion drift")
    by_id = {r["image_id"]: r for r in review["files"]}
    payloads, changes, sources, delta_rows, parsed = {}, [], [], [], {}
    for record in records:
        image_id, split = record["id"], record["split"]
        if split == "val":
            require(control[image_id]["bytes"] == originals[image_id], "Trainer validation changed")
            parsed[image_id] = gold[image_id]
            continue
        item = by_id[image_id]
        path = bound((root/REVIEW_PATH).parent, {"path": item["label_path"], "sha256": item["label_sha256"]})
        reviewed = {"path": path.relative_to(root).as_posix(), "sha256": item["label_sha256"], "bytes": path.read_bytes()}
        selected = control[image_id] if image_id in held+excluded else reviewed
        payload = selected["bytes"]
        objects = parse_pose_labels(payload.decode(), *record["original_size"], image_id)
        check_geometry(objects, *record["original_size"], image_id)
        parsed[image_id] = objects
        old_lines, new_lines = control[image_id]["bytes"].decode().splitlines(), payload.decode().splitlines()
        require(len(old_lines) == len(new_lines), "Eligible training row count changed")
        for i, (a, b) in enumerate(zip(old_lines, new_lines), 1):
            changed = groups(a, b)
            if a != b:
                require(changed and image_id not in held+excluded, "Invalid training delta")
                delta_rows.append({"image_id": image_id, "row": i, "before_line": a, "after_line": b,
                                   "changed_groups": changed})
        sources.append({"image_id": image_id, "source_image_sha256": record["image_sha256"],
                        "selection": "retained_held" if image_id in held else "excluded_from_amendment" if image_id in excluded else "accepted_r31_review",
                        "effective_label": {k: selected[k] for k in ("path", "sha256")}})
        if payload != originals[image_id]:
            payloads[record["label"]] = payload
            changes.append({"image_id": image_id, "source_label": record["label"], "candidate_label": record["label"],
                            "source_label_sha256": record["label_sha256"], "candidate_label_sha256": digest(payload),
                            "source_image_sha256": record["image_sha256"], "source_objects": len(gold[image_id]),
                            "candidate_objects": len(objects), "selected_label_source": selected["path"],
                            "selected_label_source_sha256": selected["sha256"]})
    tally = totals(records, parsed)
    require(tally == previous["counts"], "Effective training counts changed")
    delta = {"schema_version": 1, "kind": "r30_to_r31_eligible_training_delta", "reference": CONTROL,
             "accepted_review": review_ref, "rows": delta_rows,
             "summary": {"changed_rows": len(delta_rows), "changed_photos": len({r["image_id"] for r in delta_rows}),
                         "added_rows": 0, "deleted_rows": 0, "classes_visibility_preserved": True},
             "interpretation": "Both R31 loss experiment roles use these same eligible R31 labels. This delta describes preparation versus R30, not an experimental difference between loss roles."}
    manifest = {"schema_version": 1, "kind": "training_incremental_observed_overlay", "candidate_id": CANDIDATE_ID,
                "status": "controlled_incremental_only", "source_revision": SOURCE_REVISION,
                "source_manifest_sha256": review["source_manifest_sha256"], "source_candidate": review_ref,
                "source_review_status": review["review_status"], "control": CONTROL,
                "training_protocol_path": PROTOCOL_PATH, "training_input_delta": descriptor(DELTA_PATH, encoded(delta)),
                "full_dataset_certified": False, "full_source_training_readiness": "review_only",
                "task_scope": prior["task_scope"], "scope": "Eligible R31 supervision; both loss experiment roles select this candidate arm.",
                "unresolved_case_ids": prior["unresolved_case_ids"], "retained_legacy_image_ids": held,
                "excluded_training_image_ids": excluded, "image_sources": sources,
                "changed_images_vs_control": sorted({r["image_id"] for r in delta_rows}), "expected_counts": tally,
                "changes": changes}
    payloads["manifest.json"] = encoded(manifest)
    control_receipt = {"manifest": CONTROL["manifest"], "manifest_sha256": CONTROL["sha256"],
                       "candidate_id": prior["candidate_id"], "kind": prior["kind"],
                       "changed_train_labels": len(prior["changes"]), "counts": previous["counts"],
                       "supervision": "r30_predecessor_control"}
    return payloads, manifest, delta, control_receipt, review_receipt


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    candidate = root/CANDIDATE_PATH if candidate is None else Path(candidate).resolve()
    require(candidate == root/CANDIDATE_PATH, "R31 candidate path differs")
    payloads, manifest, delta, control, review = compose_candidate(root)
    verify_tree(candidate.parent, payloads)
    require((root/DELTA_PATH).read_bytes() == encoded(delta), "Training delta differs")
    protocol = json.loads((root/PROTOCOL_PATH).read_bytes())
    require(protocol["candidate_manifest_sha256"] == sha(candidate)
            and protocol["accepted_review"] == manifest["source_candidate"]
            and protocol["label_arm_for_both_experimental_roles"] == "candidate", "Loss protocol labels differ")
    return {"status": "verified", "candidate_id": CANDIDATE_ID,
            "candidate_manifest_sha256": sha(candidate), "source_manifest_sha256": manifest["source_manifest_sha256"],
            "declared_source_revision": SOURCE_REVISION, "counts": manifest["expected_counts"],
            "changed_train_labels": len(manifest["changes"]),
            "removed_objects": sum(x["source_objects"]-x["candidate_objects"] for x in manifest["changes"]),
            "control": control, "training_protocol": {"path": PROTOCOL_PATH, "sha256": sha(root/PROTOCOL_PATH)},
            "training_input_delta": {**manifest["training_input_delta"], "report": delta},
            "training_readiness": "ready_for_controlled_incremental_experiment", "full_dataset_certified": False,
            "unresolved_case_ids": manifest["unresolved_case_ids"], "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
            "excluded_training_image_ids": manifest["excluded_training_image_ids"],
            "changed_images_vs_control": manifest["changed_images_vs_control"],
            "retained_photo_labels_identical_between_arms": True, "excluded_photo_labels_identical_between_arms": True,
            "review_verification": review, "training_run": False}


def write_new(root, payloads):
    require(all(not (root/p).exists() for p in payloads), "Preserve existing artifacts; verify instead")
    for name, payload in payloads.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-review", action="store_true")
    parser.add_argument("--build-candidate", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.build_review:
        write_new(root, expected_review(root)[0])
    if args.build_candidate:
        payloads, _, delta, _, _ = compose_candidate(root)
        write_new(root, {str(Path(CANDIDATE_PATH).parent/name): b for name, b in payloads.items()})
        write_new(root, {DELTA_PATH: encoded(delta)})
    print(json.dumps(verify_review(root), indent=2))
