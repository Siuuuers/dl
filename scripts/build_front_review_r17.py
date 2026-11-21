#!/usr/bin/env python3
"""Save the requested r17 edits against immutable r16 row identities."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import FrontReviewExperiment, apply_operations
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, encoded, sha

SOURCE_REVISION = "8d72f7b7869aae4e71cbe9066fb28dd54ba7f9a2"
PARENT = {"path": "data/reviews/fronts-2026-11-22-r16/manifest.json",
          "sha256": "1645f999fa8be08e9c8a317ffccd88e54285a81aac181a8e1ec8c9b328b81993"}
REVIEW_ID = "fronts-2026-11-22-r17"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261122-r17"
CANDIDATE_ID = "front-review-2026-11-22-r17"
PROTOCOL_PATH = "experiments/front-review-2026-11-22-r17.json"
REQUESTED = tuple(f"new_{n:06d}" for n in (*range(67, 76), 79, 80))
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


def read_audits(root, paths, by_id, before):
    operations, evidence, photos, dispositions, inspected, seen = [], [], [], [], set(), set()
    for path in paths:
        audit = json.loads(path.read_text())
        require(audit.get("schema_version") == 1 and audit.get("source_review") == "fronts-2026-11-22-r16",
                "Audit must use the accepted r16 source")
        ids = audit["reviewed_image_ids"]
        require(len(ids) == len(set(ids)) and set(ids) <= set(REQUESTED) and not inspected.intersection(ids),
                "Audits must cover disjoint requested photographs")
        require(not audit.get("status_updates") and not audit.get("new_cases"), "This pass inherits existing case decisions")
        desc = descriptor(root, path); evidence.append(desc); inspected.update(ids)
        entries = audit["photo_reviews"]
        require(len(entries) == len(ids) and {p["image_id"] for p in entries} == set(ids), "Missing photo review")
        for item in entries:
            iid = item["image_id"]
            require(item["original_image_sha256"] == by_id[iid]["image_sha256"]
                    and item["source_label_sha256"] == hashlib.sha256(before[iid]).hexdigest()
                    and item["source_row_count"] == len(before[iid].splitlines()), "Audit source identity differs")
        photos.extend(entries)
        for item in audit["proposed_operations"]:
            op = copy.deepcopy(item); iid = op["image_id"]; row = op["base_row"]
            require(iid in ids and by_id[iid]["split"] == "train" and (iid, row) not in seen,
                    "Operation is outside scope or repeats an immutable row")
            require(type(row) is int and 1 <= row <= len(before[iid].splitlines()), "Invalid r16 row")
            require(op["before_line"] == before[iid].decode().splitlines()[row - 1], "Before-line differs")
            require(op["source_image_sha256"] == by_id[iid]["image_sha256"]
                    and op.get("source_label_sha256", hashlib.sha256(before[iid]).hexdigest())
                    == hashlib.sha256(before[iid]).hexdigest(), "Operation source hash differs")
            require(op.get("reason") and op["action"] in ("replace", "delete"), "Unsupported operation")
            if op["action"] == "replace":
                a, b = op["before_line"].split(), op["after_line"].split()
                require(len(a) == len(b) == 11 and a[0] == b[0] and a != b, "Invalid/no-op replacement")
                groups = [name for name, start, stop in GROUPS if a[start:stop] != b[start:stop]]
                require(op.get("changed_groups", groups) == groups, "Changed fields differ")
                op["changed_groups"] = groups
            else:
                require((iid, row) == ("new_000074", 13), "Only requested row074:13 may be deleted")
                op["changed_groups"] = ["instance"]
            op["split"] = "train"; op["audit_source"] = desc
            operations.append(op); seen.add((iid, row))
        requests = audit["request_dispositions"]
        require({d["image_id"] for d in requests} == set(ids)
                and all(d.get("request") and d.get("reason") and d.get("base_rows") for d in requests),
                "Each requested photograph needs explicit request dispositions")
        dispositions.extend({**d, "audit_source": desc} for d in requests)
    require(sorted(inspected) == list(REQUESTED), "Every requested photograph needs one audit")
    operations.sort(key=lambda o: (o["image_id"], o["base_row"]))
    require([(o["image_id"], o["base_row"]) for o in operations if o["action"] == "delete"]
            == [("new_000074", 13)], "Requested row074:13 deletion is missing")
    return operations, evidence, photos, dispositions


def experiment(root):
    records = json.loads((root / "data/manifest.json").read_text())["records"]
    return FrontReviewExperiment(review=descriptor(root, root / REVIEW_PATH), parent=PARENT,
        candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH, experiment_id="front-review-r17",
        source_revision=SOURCE_REVISION, round_number=17, experiment_version=16, serialization_repairs=(),
        protocol_limits=("Targeted user amendments inherit accepted r16 labels; bounded visual judgments remain fallible.",
                         "The old cohort remains excluded from the treatment, separate from the inherited photo015 hold.",
                         "Preparing these inputs does not run training or promote a model."),
        excluded_training_image_ids=tuple(sorted(r["id"] for r in records if r["id"].startswith("old_"))))


def build_review(root, audits):
    parent, records, by_id, _, _, _, before = source_state(root)
    operations, evidence, photos, dispositions = read_audits(root, audits, by_id, before)
    requests_path = root / LEDGER / "user-reference-index.json"
    requests = json.loads(requests_path.read_text())
    require(requests["requested_image_ids"] == list(REQUESTED) and requests["source_revision"] == SOURCE_REVISION
            and requests["source_review"] == "fronts-2026-11-22-r16", "Request identity differs")
    bound(root, requests["text_feedback"])
    for item in requests["references"]:
        bound(root, item)
    review_dir, ledger = (root / REVIEW_PATH).parent, root / LEDGER
    require(not review_dir.exists() and not (ledger / "operations.json").exists(), "r17 already exists; verify instead")
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_text()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"])
    for case in status["cases"]:
        case["r16_base_rows"] = list(case["current_rows"])
        case["base_rows"] = list(case["current_rows"])
    by_image, maps, parsed, files = defaultdict(list), {}, {}, []
    for op in operations:
        by_image[op["image_id"]].append(op)
    for rec in records:
        iid = rec["id"]
        payload, maps[iid] = apply_operations(before[iid], by_image[iid], iid)
        parsed[iid] = parse_pose_labels(payload.decode(), *rec["original_size"], iid)
        check_geometry(parsed[iid], *rec["original_size"], iid)
        lookup = {r["base_row"]: r["current_row"] for r in maps[iid]["rows"]}
        for case in status["cases"]:
            if case["image_id"] == iid:
                case["current_rows"] = [lookup[r] for r in case["base_rows"] if r in lookup]
                require(case["status"] == "resolved" or len(case["current_rows"]) == len(case["base_rows"]),
                        "An unresolved case lost its instance")
        target = review_dir / rec["label"]; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(payload)
        c = counts(parsed[iid])
        files.append({"image_id": iid, "split": rec["split"], "label_path": rec["label"],
            "label_sha256": hashlib.sha256(payload).hexdigest(), "source_image_sha256": rec["image_sha256"],
            "objects": c["objects"], "locatable_corners": c["visible_endpoints"], "missing_corners": c["missing_endpoints"],
            "changed_in_r17": bool(by_image[iid]), "fresh_visual_review": iid in REQUESTED,
            "held_current_rows": sorted({r for case in status["cases"] if case["image_id"] == iid
                                         and case["status"] != "resolved" for r in case["current_rows"]})})
    changed = sorted(by_image.keys() & set(REQUESTED))
    changed = [iid for iid in changed if by_image[iid]]
    summary = {"review_id": REVIEW_ID, "requested_photos": len(REQUESTED), "changed_photos": len(changed),
        "changed_rows": len(operations), "operation_counts": dict(Counter(o["action"] for o in operations)),
        "field_group_changes": dict(Counter(g for o in operations if o["action"] == "replace" for g in o["changed_groups"])),
        "expected_review_counts": totals(records, parsed), "changed_image_ids": changed,
        "pdf_image_ids": [f["image_id"] for f in files if f["image_id"] in changed and f["objects"]],
        "inherited_user_approved_displayed_amber_cases": 17, "training_photos_released_in_this_round": 0,
        "held_training_photo_ids": status["pending_train_photos"], "preserved_old_photos": 31}
    coverage = {"schema_version": 1, "review_id": REVIEW_ID, "base_review": PARENT, "source_revision": SOURCE_REVISION,
        "reviewed_image_ids": list(REQUESTED), "audits": evidence, "images": photos, "request_dispositions": dispositions,
        "inherited_user_approval": parent["user_approval"], "full_dataset_certified": False,
        "scope": "Targeted new amendments only; all accepted r16 annotations outside these edits are inherited unchanged."}
    for name, value in (("operations.json", operations), ("row-maps.json", maps), ("review-status.json", status),
                        ("request-coverage.json", coverage), ("amendment-summary.json", summary)):
        write_json(ledger / name, value)
    policy = descriptor(root, root / "docs/annotation-boundaries-r17.md")
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
        "status": "targeted_amendments_with_inherited_holds", "date": "2026-11-22", "parent": PARENT,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
        "full_dataset_certified": False, "annotation_policy": policy, "boundary_rubric": policy,
        "review_evidence": evidence, "user_references": [descriptor(root, requests_path)],
        "inherited_row_maps": parent["row_maps"], "user_approval": parent["user_approval"],
        "requested_image_ids": list(REQUESTED), "changed_image_ids": changed, "pdf_image_ids": summary["pdf_image_ids"],
        "excluded_training_image_ids": status["excluded_training_image_ids"], "additional_serialization_repairs": [],
        "operation_counts": summary["operation_counts"], "field_group_changes": summary["field_group_changes"],
        "expected_counts": summary["expected_review_counts"], "files": files}
    for key, name in (("operations", "operations.json"), ("row_maps", "row-maps.json"),
                      ("review_status", "review-status.json"), ("request_coverage", "request-coverage.json")):
        manifest[key] = descriptor(root, ledger / name)
    write_json(root / REVIEW_PATH, manifest)
    experiment(root).build(root)
    from verify_front_review_r17 import verify_candidate
    return verify_candidate(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit", action="append", type=Path, default=[])
    args = parser.parse_args(); root = args.root.resolve()
    require(bool(args.audit), "Supply the explicit audit files")
    print(json.dumps(build_review(root, [(p if p.is_absolute() else root / p).resolve() for p in args.audit]), indent=2))


if __name__ == "__main__":
    main()
