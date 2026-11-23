#!/usr/bin/env python3
"""Build or verify the immutable, review-only R27 snapshot from R26 rows.

This command does not create a training candidate, protocol, or notebook. It
replays 21 accepted replacements and preserves every other R26 row exactly.
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
from verify_incremental_pose_candidate import read_reference_manifest
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "cb035d381568ded57917d274385e0fd560c6d6f6"
PARENT = {
    "path": "data/reviews/fronts-2026-11-23-r26/manifest.json",
    "sha256": "b999b9491b18ecd2d55684ec14383d1c98160a6bf080a4a751ea83844fca7468",
}
REVIEW_ID = "fronts-2026-11-23-r27"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261123-r27"
AUDIT = {
    "path": f"{LEDGER}/row-review.json",
    "sha256": "c5598af2c8f71da10f38c074ad7757f3befd5ca50e350a569cbb42afdb24680d",
}
EVIDENCE = (
    {"path": f"{LEDGER}/independent-review-a.json",
     "sha256": "c3d505a84e5424adc38a328baee4927249ee15fe921caee0e9d1932a7a2b557c"},
    {"path": f"{LEDGER}/independent-review-b.json",
     "sha256": "b3ac2b46badc1ac4ca21f306aa89b405abf607832aec1e7f5069748076e90a58"},
    {"path": f"{LEDGER}/independent-color-review.json",
     "sha256": "9043f8673adf4cdbca51564ef45b4efbcca2aa457b92f7ecc9173aa6f110be38"},
)
FROZEN_REFERENCES = {
    "primary": {
        "path": "data/references/sequence-refined-reference-2026-11-18/manifest.json",
        "sha256": "0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96",
    },
    "diagnostic": {
        "path": "data/references/visible-front-completion-2026-11-14/manifest.json",
        "sha256": "bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1",
    },
}
NAMED = {("new_000007", 24), ("new_000007", 41),
         ("new_000008", 33), ("new_000009", 49)}
REVIEWED = {
    "new_000005": (8, 9, 13, 14, 20, 21, 23, 24, 25),
    "new_000006": (50, 51, 52),
    "new_000007": (16, 17, 18, 20, 21, 22, 23, 24, 30, 31, 32, 33,
                   34, 35, 36, 37, 38, 39, 40, 41, 42),
    "new_000008": (28, 29, 30, 31, 32, 33, 34, 35),
    "new_000009": (48, 49, 50, 51, 52),
    "new_000012": (35, 36, 37, 38),
    "new_000013": (8, 9, 18, 19, 23, 24, 26, 27),
    "new_000014": (13, 14, 33, 34, 35),
    "new_000016": (26, 27, 28),
    "new_000037": (7,), "new_000049": (3,), "new_000050": (30,),
    "new_000051": (20,), "new_000056": (12,), "new_000070": (14,),
    "new_000083": (12,), "new_000114": (19,),
}
REPLACEMENTS = {
    "new_000007": (24, 33, 37, 41, 42), "new_000008": (33, 34, 35),
    "new_000009": (48, 49, 50, 51, 52), "new_000013": (27,),
    "new_000014": (35,), "new_000016": (26, 27), "new_000049": (3,),
    "new_000051": (20,), "new_000056": (12,), "new_000083": (12,),
}
STANDARDS = tuple(("new_000008", r) for r in (18, 19, 20, 21, 28, 29, 30, 31, 32)) + (
    ("new_000010", 16), ("new_000016", 24))
FIELD_TOKENS = {"xmin": (1, 3), "xmax": (1, 3), "ymin": (2, 4), "ymax": (2, 4),
                "left.x": (5,), "left.y": (6,), "right.x": (8,), "right.y": (9,)}


def identities(mapping):
    return {(image_id, row) for image_id, rows in mapping.items() for row in rows}


def descriptor(root, path, payload=None):
    return {"path": path, "sha256": sha(root / path) if payload is None else digest(payload)}


def pixel_geometry(line, source):
    objects = parse_pose_labels(line, 1280, 720, source)
    require(len(objects) == 1, f"Expected one row: {source}")
    check_geometry(objects, 1280, 720, source)
    obj = objects[0]
    return {"box_xyxy": obj["bbox_xyxy"], "keypoints": obj["keypoints"]}


def semantic_fields(before, after):
    result = [name for name, a, b in zip(("xmin", "ymin", "xmax", "ymax"),
              before["box_xyxy"], after["box_xyxy"]) if abs(a - b) > 1e-6]
    for side, index in (("left", 0), ("right", 1)):
        for axis, coordinate in (("x", 0), ("y", 1)):
            if abs(before["keypoints"][index][coordinate] -
                   after["keypoints"][index][coordinate]) > 1e-6:
                result.append(f"{side}.{axis}")
    return result


def independent_lines(root):
    """Cross-check the accepted ledger against the unchanged reviewer artifacts."""
    a, b, c = [json.loads(bound(root, item).read_text()) for item in EVIDENCE]
    lines = {}
    for item in a["proposals"]:
        lines[item["image_id"], item["source_row_1based"]] = (
            item["before"]["line"], item["after"]["line"])
    for item in a["retained_rows"]:
        lines[item["image_id"], item["source_row_1based"]] = (item["line"], item["line"])
    for item in b["reviewed_rows"] + c["reviewed_cases"]:
        key = (item["image_id"], item["row"])
        require(key not in lines, f"Independent review identity repeated: {key}")
        lines[key] = (item["before_line"], item["after_line"])
    require(len(lines) == 74 and set(lines) == identities(REVIEWED),
            "Independent review coverage differs from the accepted bounded scope")
    return lines


def expected_review(root):
    root = Path(root).resolve()
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, _, _, original_counts = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    require(original_counts["val"]["objects"] == 664, "Original trainer validation differs")
    frozen = {name: read_reference_manifest(root, desc) for name, desc in FROZEN_REFERENCES.items()}
    require(frozen["primary"]["counts"]["objects"] == 666 and
            frozen["diagnostic"]["counts"]["objects"] == 663, "Frozen scoring counts differ")
    audit = json.loads(bound(root, AUDIT).read_text())
    require(audit["source_revision"] == SOURCE_REVISION and audit["source_review"] == PARENT
            and audit["full_dataset_certified"] is False, "R27 source or certification scope differs")
    require(audit["independent_review_evidence"] == list(EVIDENCE), "Independent evidence inventory differs")
    independent = independent_lines(root)
    row_keys = [(r["image_id"], r["row"]) for r in audit["rows"]]
    require(len(row_keys) == len(set(row_keys)) == 74 and set(row_keys) == identities(REVIEWED),
            "Every examined row must have one final decision")
    require([(r["image_id"], r["row"]) for r in audit["preserved_standards"]] == list(STANDARDS),
            "Accepted standard inventory differs")
    require({(r["image_id"], r["row"]) for r in audit["named_rows"]} == NAMED,
            "The four user-named row identities differ")

    parent_files = {item["image_id"]: item for item in parent["files"]}
    require(len(parent_files) == 157 and set(parent_files) == set(by_id), "Parent label inventory differs")
    before = {}
    for record in records:
        image_id = record["id"]
        item = parent_files[image_id]
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(item["split"] == record["split"] and item["label_path"] == record["label"],
                f"Immutable image/split mapping differs: {image_id}")
        require(sha(path) == item["label_sha256"], f"R26 label drift: {image_id}")
        require(sha(root / "data" / record["image"]) == record["image_sha256"]
                == item["source_image_sha256"], f"Original pixels differ: {image_id}")
        before[image_id] = path.read_bytes()

    operations, groups_count, split_changes, point_splits, box_splits = [], Counter(), Counter(), Counter(), Counter()
    action_counts, scope_counts = Counter(), Counter()
    for item in audit["rows"]:
        image_id, row = item["image_id"], item["row"]
        key = (image_id, row)
        record = by_id[image_id]
        require(type(row) is int and 1 <= row <= len(before[image_id].splitlines()), "Invalid immutable row")
        old_line = before[image_id].decode().splitlines()[row - 1]
        require(item["before_line"] == old_line and
                (item["before_line"], item["after_line"]) == independent[key],
                f"Ledger does not reproduce independent R26-row decision: {key}")
        require(item["source_image_sha256"] == record["image_sha256"] and
                item["source_label_sha256"] == parent_files[image_id]["label_sha256"] and
                item["split"] == record["split"] and record["original_size"] == [1280, 720],
                f"Row provenance differs: {key}")
        require(item["scope"] in ("user_named", "same_photo_analogue", "training_analogue",
                                  "color_screen_sample", "protected_standard") and
                (item["scope"] == "user_named") == (key in NAMED), "Row scope differs")
        require(item.get("reason") and item.get("independent_reason") and item.get("root_review")
                and item.get("coordinate_uncertainty"), "Missing review rationale or uncertainty")
        require(item["independent_evidence"] in EVIDENCE, "Row evidence is outside pinned inputs")
        action = item["action"]
        require(action in ("replace", "retain"), "R27 permits no additions or deletions")
        old, new = old_line.split(), item["after_line"].split()
        require(len(old) == len(new) == 11 and all(old[i] == new[i] for i in (0, 7, 10)),
                "Class and endpoint visibility must be preserved exactly")
        for offset in (5, 8):
            if int(old[offset + 2]) == 0:
                require(old[offset:offset + 3] == new[offset:offset + 3]
                        and all(float(v) == 0 for v in new[offset:offset + 3]),
                        "Unavailable endpoints must retain exact zero tokens")
        pb = pixel_geometry(old_line, f"{image_id}:{row}:before")
        pa = pixel_geometry(item["after_line"], f"{image_id}:{row}:after")
        require(pb == item["before_pixels"] and pa == item["after_pixels"],
                f"Displayed pixel geometry differs from saved YOLO row: {key}")
        fields = semantic_fields(pb, pa)
        changed_indices = [i for i in range(11) if old[i] != new[i]]
        allowed = {index for field in fields for index in FIELD_TOKENS[field]}
        require(fields == item["changed_fields"] and changed_indices == item["changed_token_indices_0based"]
                and set(changed_indices) <= allowed, f"Unchanged token or saved field decision differs: {key}")
        groups = [name for name, start, stop in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
                  if old[start:stop] != new[start:stop]]
        require(groups == item["changed_groups"] and (action == "replace") == bool(groups)
                and (action == "replace") == (key in identities(REPLACEMENTS)), "Replacement scope differs")
        action_counts[action] += 1
        scope_counts[item["scope"]] += 1
        if action == "retain":
            require(item["after_line"] == old_line, "Retained row must be byte-identical")
            continue
        operations.append({"image_id": image_id, "split": record["split"], "base_row": row,
                           "action": "replace", "before_line": old_line, "after_line": item["after_line"],
                           "changed_groups": groups, "changed_fields": fields, "reason": item["reason"]})
        groups_count.update(groups)
        split_changes[record["split"]] += 1
        point_splits[record["split"]] += sum(g in ("left", "right") for g in groups)
        box_splits[record["split"]] += "box" in groups
    require(dict(action_counts) == {"retain": 53, "replace": 21} and
            dict(split_changes) == {"val": 14, "train": 7} and
            dict(groups_count) == {"box": 16, "left": 6, "right": 3}, "Accepted change totals differ")
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))

    payloads, files, maps, parsed = {}, [], {}, {}
    for record in records:
        image_id = record["id"]
        edits = [op for op in operations if op["image_id"] == image_id]
        payload, maps[image_id] = apply_operations(before[image_id], edits, image_id)
        require(not maps[image_id]["deleted_base_rows"] and
                all(r["current_row"] == r["base_row"] for r in maps[image_id]["rows"]),
                "R27 row identities must remain unchanged")
        parsed[image_id] = parse_pose_labels(payload.decode(), *record["original_size"], image_id)
        check_geometry(parsed[image_id], *record["original_size"], image_id)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        stat = counts(parsed[image_id])
        files.append({"image_id": image_id, "split": record["split"], "label_path": record["label"],
                      "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                      "objects": stat["objects"], "locatable_corners": stat["visible_endpoints"],
                      "missing_corners": stat["missing_endpoints"], "changed_in_r27": bool(edits)})
    for item in audit["preserved_standards"]:
        image_id, row = item["image_id"], item["row"]
        current = payloads[f"data/reviews/{REVIEW_ID}/{by_id[image_id]['label']}"]
        require(current.splitlines()[row - 1] == before[image_id].splitlines()[row - 1]
                == item["line"].encode(), f"Protected standard changed: {image_id}:{row}")
    review_counts = totals(records, parsed)
    require(review_counts == parent["expected_counts"], "Replacement-only review changed object/visibility counts")
    changed_ids = sorted({op["image_id"] for op in operations})
    require(len(changed_ids) == 10, "Changed-photo inventory differs")
    summary = {
        "review_id": REVIEW_ID, "parent_review": PARENT, "review_only": True,
        "reviewed_rows": 74, "reviewed_photos": len(REVIEWED), "named_rows_reviewed": 4,
        "replacement_rows": 21, "retained_rows": 53, "deleted_rows": 0, "added_rows": 0,
        "changed_photos": 10, "changed_image_ids": changed_ids, "changed_rows_by_split": dict(split_changes),
        "changed_box_rows": groups_count["box"], "changed_box_rows_by_split": dict(box_splits),
        "point_repairs": groups_count["left"] + groups_count["right"],
        "point_repairs_by_side": {k: groups_count[k] for k in ("left", "right")},
        "point_repairs_by_split": dict(point_splits), "reviewed_rows_by_scope": dict(scope_counts),
        "protected_standard_rows": 11, "unchanged_review_files": 147,
        "expected_review_counts": review_counts, "training_run": False,
        "candidate_created": False, "protocol_created": False, "notebook_created": False,
        "full_dataset_certified": False,
    }
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_text()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"],
                  r27_scope="The enumerated 74 rows only; accepted replacements are review-ready.",
                  inherited_case_status_changes=[], full_dataset_certified=False)
    source_status = {
        "schema_version": 1, "kind": "immutable_review_snapshot_provenance", "review_id": REVIEW_ID,
        "source_revision": SOURCE_REVISION, "source_review": PARENT,
        "source_manifest_sha256": parent["source_manifest_sha256"], "row_review": AUDIT,
        "independent_review_evidence": list(EVIDENCE),
        "original_image_hashes_verified": 157, "parent_label_hashes_verified": 157,
        "original_training_validation_counts": original_counts["val"],
        "frozen_references": {name: {**desc, "counts": frozen[name]["counts"]}
                              for name, desc in FROZEN_REFERENCES.items()},
        "all_splits_preserved": True, "row_identities_preserved": True,
        "classes_visibility_and_missing_tokens_preserved": True, "protected_standard_rows_exact": 11,
        "review_only": True, "training_or_model_promotion": False, "full_dataset_certified": False,
    }
    coverage = {
        "review_id": REVIEW_ID, "source_review": PARENT,
        "requested_rows": [{"image_id": i, "row": r} for i, r in sorted(NAMED)],
        "reviewed_rows_by_photo": {i: list(rows) for i, rows in REVIEWED.items()},
        "row_decisions": AUDIT, "preserved_standards": [{"image_id": i, "row": r} for i, r in STANDARDS],
        "full_dataset_re_review": False, "all_named_photos_are_development": True,
        "color_screened_candidates_automatically_amended": False,
    }
    documents = {
        "operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
        "review_status": ("review-status.json", status), "source_status": ("source-status.json", source_status),
        "request_coverage": ("request-coverage.json", coverage),
        "amendment_summary": ("amendment-summary.json", summary),
    }
    manifest = {
        "schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
        "status": "bounded_original_pixel_review_ready", "date": "2026-11-23", "parent": PARENT,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
        "full_dataset_certified": False, "review_only": True,
        "annotation_policy": parent["annotation_policy"], "boundary_rubric": parent["boundary_rubric"],
        "review_method": audit["review_method"], "review_evidence": [AUDIT],
        "independent_review_evidence": list(EVIDENCE), "user_references": parent["user_references"],
        "current_request_record": AUDIT, "inherited_row_maps": parent["row_maps"],
        "user_approval": parent["user_approval"], "requested_image_ids": sorted({i for i, _ in NAMED}),
        "reviewed_image_ids": sorted(REVIEWED), "changed_image_ids": changed_ids,
        "pdf_image_ids": changed_ids, "excluded_training_image_ids": parent["excluded_training_image_ids"],
        "additional_serialization_repairs": [], "operation_counts": {"replace": 21, "delete": 0, "add": 0},
        "field_group_changes": dict(sorted(groups_count.items())), "expected_counts": review_counts,
        "frozen_references": FROZEN_REFERENCES, "files": files,
        "replay_script": descriptor(root, "scripts/front_review_r27.py"),
    }
    for key, (name, value) in documents.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(root, relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def verify_review(root):
    """Read-only replay verification; safe for saved-prediction review rescoring."""
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"R27 artifact differs from pinned original-pixel decisions: {relative}")
    label_root = root / "data" / "reviews" / REVIEW_ID / "labels"
    actual = {str(p.relative_to(root)) for p in label_root.rglob("*") if p.is_file() or p.is_symlink()}
    expected = {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}
    require(actual == expected and len(actual) == 157, "R27 snapshot must contain exactly157 label files")
    return {
        "schema_version": 1, "kind": "standalone_original_pixel_review_verification", "verified": True,
        "review_manifest": REVIEW_PATH, "review_manifest_sha256": sha(root / REVIEW_PATH),
        "row_review": AUDIT, "source_revision": SOURCE_REVISION, "label_files_verified": 157,
        "original_trainer_validation_objects": 664, "frozen_primary_objects": 666,
        "frozen_diagnostic_objects": 663, "unchanged_tokens_and_row_identities_preserved": True,
        "source_images_parent_labels_and_frozen_references_verified": True, **summary,
    }


def build(root):
    root = Path(root).resolve()
    payloads, _ = expected_review(root)
    require(all(not (root / path).exists() for path in payloads),
            "Preserve existing R27 artifacts; run verification instead of rebuilding")
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return verify_review(root)


def save_receipt(root, path, receipt):
    path = path if path.is_absolute() else root / path
    payload = encoded(receipt)
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                "Preserve an existing differing verification receipt")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true", help="Create new immutable snapshot files; never overwrite")
    parser.add_argument("--receipt", type=Path, help="Optionally save the deterministic verification receipt")
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = build(root) if args.build else verify_review(root)
    if args.receipt is not None:
        save_receipt(root, args.receipt, receipt)
    print(json.dumps(receipt, indent=2))
