#!/usr/bin/env python3
"""Build or verify the immutable R28 annotation-review component from R27.

Replay 23 accepted coordinate replacements and 10 explicit retentions. Preserve
all other rows, class/visibility tokens, missing endpoints, and split identities.
The companion training candidate and notebook are managed separately by
front_review_r28_training.py; this command performs no model run or promotion.
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
from front_review_r27 import (FIELD_TOKENS, FROZEN_REFERENCES, STANDARDS,
    descriptor, identities, pixel_geometry, semantic_fields, verify_review as verify_parent)
from verify_incremental_pose_candidate import read_reference_manifest
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "bfd7d5bd346f92935bd1695e4515e4f62a0a0f85"
PARENT = {"path": "data/reviews/fronts-2026-11-23-r27/manifest.json",
          "sha256": "9e3675c83cb6735242b3b99266e36d5a33248c13e51b8c91bd89a65d2c4b92af"}
REVIEW_ID = "fronts-2026-11-23-r28"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
LEDGER = "results/review/front-review-20261123-r28"
AUDIT = {"path": f"{LEDGER}/row-review.json",
         "sha256": "68c136585e473e92489d14981446e85a553997d6be58a4ccef0af216964662f1"}
EVIDENCE = ({'path': 'results/review/front-review-20261123-r28/independent-review-a.json', 'sha256': 'ed10f2dde192a8eb10b30396b6d1b62c4d31616da151ab7a13407b158e20c221'}, {'path': 'results/review/front-review-20261123-r28/independent-review-b.json', 'sha256': 'a0fde5c9f886c70d4c223cd29865931e55dd604ad14732e5342eadb2d4400850'}, {'path': 'results/review/front-review-20261123-r28/root-pixel-review.json', 'sha256': '8089efa4d22d12a36290ed5c4c233e3a7190fb90ba38faeb5718b4075c5e033c'}, {'path': 'results/review/front-review-20261123-r28/independent-review-root-016.json', 'sha256': 'eb06a5cbd2eb8eae973360ced4e553e692c4d642e90f3ef4820592479a2dc3fd'}, {'path': 'results/review/front-review-20261123-r28/independent-review-root-screen.json', 'sha256': 'dbea331fd5f9a6d3e31d708cefadf9d1afc5fcb4ebdeb945a141c3454802bc78'})
GEOMETRY_SCREEN = {'path': 'results/review/front-review-20261123-r28/geometry-screen.json', 'sha256': '6686df5d06c6ef76f17429d23d9b034d9a0c8f9379b08347fa861368719f7a52'}
REVIEWED = {'new_000006': (56,), 'new_000008': (4, 5, 6, 11, 19, 23, 29), 'new_000009': (55,), 'new_000010': (14, 26, 33), 'new_000011': (49,), 'new_000012': (2,), 'new_000014': (8,), 'new_000016': (26, 27), 'new_000017': (18,), 'new_000018': (1,), 'new_000028': (59,), 'new_000029': (42,), 'new_000055': (23,), 'new_000056': (12,), 'new_000058': (7, 18), 'new_000059': (7, 10), 'new_000087': (53,), 'new_000088': (52,), 'new_000089': (50,), 'new_000090': (44, 45), 'new_000094': (9,)}
REPLACEMENTS = {'new_000008': (5, 11, 23), 'new_000010': (14, 26, 33), 'new_000011': (49,), 'new_000016': (26, 27), 'new_000017': (18,), 'new_000028': (59,), 'new_000055': (23,), 'new_000056': (12,), 'new_000058': (7, 18), 'new_000059': (7, 10), 'new_000087': (53,), 'new_000088': (52,), 'new_000089': (50,), 'new_000090': (44, 45), 'new_000094': (9,)}
NAMED = (('new_000008', 4), ('new_000008', 5), ('new_000008', 6), ('new_000008', 11), ('new_000008', 19), ('new_000008', 23), ('new_000008', 29), ('new_000009', 55), ('new_000010', 14), ('new_000010', 26), ('new_000010', 33), ('new_000011', 49), ('new_000016', 26), ('new_000016', 27), ('new_000055', 23), ('new_000056', 12), ('new_000058', 7), ('new_000058', 18), ('new_000059', 7), ('new_000059', 10), ('new_000087', 53), ('new_000088', 52), ('new_000089', 50), ('new_000090', 44), ('new_000090', 45))
STANDARD_REVISITS = (("new_000008", 19), ("new_000008", 29))


def independent_lines(root):
    """Reconcile exact accepted lines with three primary and two second-pass inputs."""
    a, b, primary_root, second_016, second_screen = [
        json.loads(bound(root, item).read_bytes()) for item in EVIDENCE]
    require(a["source_revision"] == primary_root["source_revision"] == SOURCE_REVISION
            and a["source_review"] == primary_root["source_review"] == PARENT,
            "Independent R28 source identity differs")
    require(b["source_commit"] == SOURCE_REVISION
            and b["source_review"] == "fronts-2026-11-23-r27"
            and b["source_manifest_sha256"] == PARENT["sha256"],
            "Independent B source identity differs")
    lines = {}
    for document, key in ((a, "proposals"), (b, "reviewed_rows"), (primary_root, "rows")):
        for item in document[key]:
            identity = (item["image_id"], item.get("row", item.get("source_row_1based")))
            require(identity not in lines, f"Independent row repeated: {identity}")
            lines[identity] = (item["before_line"], item["after_line"])
    require(len(lines) == 33 and set(lines) == identities(REVIEWED),
            "Independent review coverage differs from the bounded R28 scope")
    require(second_016["image_id"] == "new_000016"
            and second_016["source_review"] == "fronts-2026-11-23-r27"
            and second_016["additional_changes"] == [], "016 second-pass scope differs")
    for item in second_016["decisions"]:
        row = item["source_row_1based"]
        require(row in (26, 27) and item["decision"] == "agree", "016 acceptance differs")
        old, new = [pixel_geometry(line, f"016:{row}:second-pass")
                    for line in lines[("new_000016", row)]]
        if row == 26:
            require(abs(old["box_xyxy"][2] - item["before"]) < 1e-6
                    and abs(new["box_xyxy"][2] - item["proposed"]) < 1e-6,
                    "016:26 second-pass coordinate differs")
        else:
            require(all(abs(x-y) < 1e-6 for x, y in zip(old["keypoints"][1][:2], item["before"]))
                    and all(abs(x-y) < 1e-6 for x, y in zip(new["keypoints"][1][:2], item["proposed"])),
                    "016:27 second-pass coordinate differs")
    require({item["source_row_1based"] for item in second_016["decisions"]} == {26, 27},
            "Both screenshot amendments require independent confirmation")
    require(second_screen["source_revision"] == SOURCE_REVISION
            and second_screen["source_review"] == PARENT
            and second_screen["root_proposal_source"] == EVIDENCE[2],
            "Screen second-pass source differs")
    accepted = set()
    for item in second_screen["decisions"]:
        key = (item["image_id"], item["source_row_1based"])
        require(item["decision"] == "accept" and item["additional_changes"] == []
                and (item["before_line"], item["after_line"]) == lines[key],
                f"Screen second-pass decision differs: {key}")
        accepted.add(key)
    require(accepted == {("new_000028", 59), ("new_000094", 9), ("new_000017", 18)},
            "Screen second-pass replacement scope differs")
    return lines


def validate_evidence(root, audit, parent_files):
    require(audit["independent_review_evidence"] == list(EVIDENCE),
            "Independent evidence inventory differs")
    provenance = json.loads(bound(root, audit["evidence_provenance"]).read_bytes())
    require(provenance["source_review"] == PARENT
            and [a["copied_evidence"] for a in provenance["artifacts"]] == list(EVIDENCE),
            "Copied evidence provenance differs")
    acceptance = json.loads(bound(root, audit["root_acceptance"]).read_bytes())
    require(acceptance["source_revision"] == SOURCE_REVISION
            and acceptance["source_review"] == PARENT and acceptance["status"] == "accepted"
            and acceptance["accepted_replacements"] == 23
            and acceptance["accepted_retentions"] == 10
            and acceptance["additional_coordinate_edits"] == []
            and acceptance["protected_standard_revisions"] == 0,
            "Final coordinating review acceptance differs")
    require(acceptance["accepted_identities"] == [
        {"image_id": r["image_id"], "row": r["row"], "action": r["action"]}
        for r in audit["rows"]], "Acceptance identity inventory differs")
    for item in audit["user_references"]:
        bound(root, item)
    bound(root, audit["source_prediction_evidence"])
    require(audit["geometry_screen"] == GEOMETRY_SCREEN, "Geometry screen identity differs")
    screen = json.loads(bound(root, GEOMETRY_SCREEN).read_bytes())
    require(screen["parent_review"] == PARENT
            and sha(root / "scripts/rank_front_geometry_candidates.py") == screen["script_sha256"],
            "Geometry screen source or implementation differs")
    selected = {(r["image_id"], r["row"]) for r in screen["selected_training_review"]}
    require(len(selected) == 8 and selected == identities(REVIEWED) - set(NAMED),
            "Visual screen sample differs from the preselected eight rows")
    require({(r["image_id"], r["row"]) for r in screen["named_rows"]} == set(NAMED),
            "Screen named-row inventory differs")
    require(len(screen["sources"]) == 157, "Screen source label inventory differs")
    for item in screen["sources"]:
        source = parent_files[item["image_id"]]
        require(item["label_sha256"] == source["label_sha256"]
                and item["source_image_sha256"] == source["source_image_sha256"],
                "Screen did not use the immutable parent labels/pixels")
    return screen


def expected_review(root):
    root = Path(root).resolve()
    parent = json.loads(bound(root, PARENT).read_bytes())
    bound(root, parent["replay_script"])
    parent_receipt = verify_parent(root)
    require(parent_receipt["review_manifest_sha256"] == PARENT["sha256"],
            "R27 standalone replay identity differs")
    _, records, by_id, _, _, original_counts = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    require(original_counts["val"]["objects"] == 664, "Original trainer validation differs")
    frozen = {name: read_reference_manifest(root, item) for name, item in FROZEN_REFERENCES.items()}
    require(frozen["primary"]["counts"]["objects"] == 666
            and frozen["diagnostic"]["counts"]["objects"] == 663,
            "Frozen scoring-reference counts differ")
    audit = json.loads(bound(root, AUDIT).read_bytes())
    require(audit["source_revision"] == SOURCE_REVISION and audit["source_review"] == PARENT
            and audit["full_dataset_certified"] is False,
            "R28 source or certification scope differs")
    independent = independent_lines(root)
    row_keys = [(r["image_id"], r["row"]) for r in audit["rows"]]
    require(len(row_keys) == len(set(row_keys)) == 33 and set(row_keys) == identities(REVIEWED),
            "Each examined row requires exactly one final decision")
    require([(r["image_id"], r["row"]) for r in audit["preserved_standards"]] == list(STANDARDS),
            "Preserved standard inventory differs")
    require({(r["image_id"], r["row"]) for r in audit["named_rows"]} == set(NAMED),
            "The25 user-named/screenshot identities differ")
    parent_files = {item["image_id"]: item for item in parent["files"]}
    require(len(parent_files) == 157 and set(parent_files) == set(by_id), "Parent label inventory differs")
    screen = validate_evidence(root, audit, parent_files)
    before = {}
    for record in records:
        image_id = record["id"]
        item = parent_files[image_id]
        require(item["split"] == record["split"] and item["label_path"] == record["label"],
                f"Immutable image/split mapping differs: {image_id}")
        path = bound((root / PARENT["path"]).parent,
                     {"path": item["label_path"], "sha256": item["label_sha256"]})
        require(sha(root / "data" / record["image"]) == record["image_sha256"]
                == item["source_image_sha256"], f"Original pixels differ: {image_id}")
        before[image_id] = path.read_bytes()

    operations, groups_count, split_changes, point_splits, box_splits = [], Counter(), Counter(), Counter(), Counter()
    actions, scopes = Counter(), Counter()
    for item in audit["rows"]:
        image_id, row = item["image_id"], item["row"]
        key = (image_id, row)
        record = by_id[image_id]
        require(type(row) is int and 1 <= row <= len(before[image_id].splitlines()), "Invalid R27 row")
        old_line = before[image_id].decode().splitlines()[row - 1]
        require(item["before_line"] == old_line
                and (item["before_line"], item["after_line"]) == independent[key],
                f"Ledger does not reproduce independent R27-row decision: {key}")
        require(item["source_image_sha256"] == record["image_sha256"]
                and item["source_label_sha256"] == parent_files[image_id]["label_sha256"]
                and item["split"] == record["split"] and record["original_size"] == [1280, 720],
                f"Row provenance differs: {key}")
        require(item["scope"] in ("user_named", "user_screenshot", "geometry_screen_sample")
                and (item["scope"] != "geometry_screen_sample") == (key in NAMED),
                f"Row review scope differs: {key}")
        require(item.get("reason") and item.get("independent_reason")
                and item.get("coordinate_uncertainty")
                and item["root_review"]["status"] == "accepted"
                and item["root_review"]["acceptance"] == audit["root_acceptance"]
                and item["root_review"]["original_and_comparison_panels_inspected"] is True,
                "Missing accepted rationale, inspection or uncertainty")
        require(item["independent_evidence"] in EVIDENCE[:3], "Row evidence outside primary pinned inputs")
        for extra in item["supplementary_independent_evidence"]:
            require(extra in EVIDENCE[3:], "Unknown independent second-pass evidence")
        if key in (("new_000010", 14), ("new_000010", 33)):
            require(item.get("ownership_inference"), "Fragment-union ownership inference must be explicit")
        require(item["standard_revisit_explicitly_requested"] == (key in STANDARD_REVISITS),
                "Explicit standard-revisit inventory differs")
        action = item["action"]
        require(action in ("replace", "retain"), "R28 permits no additions or deletions")
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
                and set(changed_indices) <= allowed, f"Unchanged token or semantic field differs: {key}")
        groups = [name for name, start, stop in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
                  if old[start:stop] != new[start:stop]]
        require(groups == item["changed_groups"] and (action == "replace") == bool(groups)
                and (action == "replace") == (key in identities(REPLACEMENTS)),
                "Replacement scope differs")
        actions[action] += 1
        scopes[item["scope"]] += 1
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
    require(dict(actions) == {"retain": 10, "replace": 23}
            and dict(split_changes) == {"val": 18, "train": 5}
            and dict(groups_count) == {"left": 9, "box": 14, "right": 7}
            and dict(scopes) == {"geometry_screen_sample": 8, "user_named": 23, "user_screenshot": 2},
            "Accepted change totals differ")
    operations.sort(key=lambda r: (r["image_id"], r["base_row"]))
    payloads, files, maps, parsed = {}, [], {}, {}
    for record in records:
        image_id = record["id"]
        edits = [op for op in operations if op["image_id"] == image_id]
        payload, maps[image_id] = apply_operations(before[image_id], edits, image_id)
        require(not maps[image_id]["deleted_base_rows"]
                and all(r["current_row"] == r["base_row"] for r in maps[image_id]["rows"]),
                "R28 row identities must remain unchanged")
        require(len(payload.splitlines()) == len(before[image_id].splitlines()),
                "R28 row count differs from the parent")
        parsed[image_id] = parse_pose_labels(payload.decode(), *record["original_size"], image_id)
        check_geometry(parsed[image_id], *record["original_size"], image_id)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        stat = counts(parsed[image_id])
        files.append({"image_id": image_id, "split": record["split"], "label_path": record["label"],
                      "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                      "objects": stat["objects"], "locatable_corners": stat["visible_endpoints"],
                      "missing_corners": stat["missing_endpoints"], "changed_in_r28": bool(edits)})
    for item in audit["preserved_standards"]:
        image_id, row = item["image_id"], item["row"]
        current = payloads[f"data/reviews/{REVIEW_ID}/{by_id[image_id]['label']}"]
        require(current.splitlines()[row - 1] == before[image_id].splitlines()[row - 1]
                == item["line"].encode(), f"Protected standard changed: {image_id}:{row}")
    excluded = parent["excluded_training_image_ids"]
    require(len(excluded) == 31 and all(i.startswith("old_") for i in excluded),
            "Old-series training exclusion inventory differs")
    for image_id in ["new_000015", *excluded]:
        require(payloads[f"data/reviews/{REVIEW_ID}/{by_id[image_id]['label']}"] == before[image_id],
                f"Held or excluded photograph changed: {image_id}")
    review_counts = totals(records, parsed)
    require(review_counts == parent["expected_counts"], "Review changed object/visibility counts")
    changed_ids = sorted({op["image_id"] for op in operations})
    require(len(changed_ids) == 15, "Changed-photo inventory differs")
    summary = {
        "review_id": REVIEW_ID, "parent_review": PARENT,
        "component": "immutable_annotation_review", "review_only": True,
        "reviewed_rows": 33, "reviewed_photos": len(REVIEWED), "named_rows_reviewed": 25,
        "user_text_rows_reviewed": 23, "user_screenshot_rows_reviewed": 2,
        "geometry_screen_rows_reviewed": 8, "geometry_screen_sampling": "nonrandom priority sample",
        "geometry_screen_replacements": 3, "geometry_screen_retentions": 5,
        "replacement_rows": 23, "retained_rows": 10, "deleted_rows": 0, "added_rows": 0,
        "changed_photos": 15, "changed_image_ids": changed_ids,
        "changed_rows_by_split": dict(split_changes), "changed_box_rows": groups_count["box"],
        "changed_box_rows_by_split": dict(box_splits),
        "point_repairs": groups_count["left"] + groups_count["right"],
        "point_repairs_by_side": {k: groups_count[k] for k in ("left", "right")},
        "point_repairs_by_split": dict(point_splits), "reviewed_rows_by_scope": dict(scopes),
        "protected_standard_rows": 11, "explicit_standard_revisits": 2, "standard_revisions": 0,
        "unchanged_review_files": 142, "expected_review_counts": review_counts,
        "held_training_photos": ["new_000015"], "excluded_old_training_photos": 31,
        "training_run_by_this_component": False, "model_promotion": False,
        "companion_training_artifacts_managed_separately": True, "full_dataset_certified": False,
    }
    parent_status = json.loads(bound(root, parent["review_status"]).read_bytes())
    status = copy.deepcopy(parent_status)
    inherited_scopes = {key: status.pop(key) for key in list(status) if key.endswith("_scope")}
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"],
                  inherited_scope_notes=inherited_scopes,
                  current_review_scope="The enumerated33 rows:25 user-named/screenshot rows and eight nonrandom geometry-screen samples.23 coordinate replacements accepted;10 rows retained.",
                  current_review_rows=[{"image_id": r["image_id"], "row": r["row"], "action": r["action"]}
                                       for r in audit["rows"]],
                  current_row_review=AUDIT, inherited_case_status_changes=[], full_dataset_certified=False)
    require(status["pending_train_photos"] == ["new_000015"]
            and status["excluded_training_image_ids"] == excluded
            and status["cases"] == parent_status["cases"],
            "Inherited hold/exclusion/case status changed")
    source_status = {
        "schema_version": 1, "kind": "immutable_review_snapshot_provenance", "review_id": REVIEW_ID,
        "source_revision": SOURCE_REVISION, "source_review": PARENT,
        "source_manifest_sha256": parent["source_manifest_sha256"], "row_review": AUDIT,
        "independent_review_evidence": list(EVIDENCE), "geometry_screen": GEOMETRY_SCREEN,
        "root_acceptance": audit["root_acceptance"], "user_references": audit["user_references"],
        "original_image_hashes_verified": 157, "parent_label_hashes_verified": 157,
        "parent_review_replay_verified": True,
        "original_training_validation_counts": original_counts["val"],
        "frozen_references": {name: {**item, "counts": frozen[name]["counts"]}
                              for name, item in FROZEN_REFERENCES.items()},
        "all_splits_preserved": True, "row_identities_preserved": True,
        "classes_visibility_and_missing_tokens_preserved": True, "protected_standard_rows_exact": 11,
        "held_training_photos_unchanged": ["new_000015"], "excluded_old_training_photos_unchanged": 31,
        "component": "immutable_annotation_review", "review_only": True,
        "training_or_model_promotion_by_this_component": False, "full_dataset_certified": False,
    }
    coverage = {
        "review_id": REVIEW_ID, "source_review": PARENT,
        "requested_rows": audit["named_rows"],
        "reviewed_rows_by_photo": {i: list(rows) for i, rows in REVIEWED.items()},
        "row_decisions": AUDIT, "geometry_screen": GEOMETRY_SCREEN,
        "screen_selected_rows": [{"image_id": r["image_id"], "row": r["row"]}
                                 for r in screen["selected_training_review"]],
        "preserved_standards": [{"image_id": i, "row": r} for i, r in STANDARDS],
        "explicit_standard_revisits_retained": [{"image_id": i, "row": r} for i, r in STANDARD_REVISITS],
        "full_dataset_re_review": False,
        "all_user_text_rows_are_development": True, "user_screenshot_rows_are_training": True,
        "geometry_or_color_candidates_automatically_amended": False,
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
        "component": "immutable_annotation_review", "full_dataset_certified": False, "review_only": True,
        "annotation_policy": parent["annotation_policy"], "boundary_rubric": parent["boundary_rubric"],
        "review_method": audit["review_method"], "review_evidence": [AUDIT],
        "independent_review_evidence": list(EVIDENCE), "geometry_screen": GEOMETRY_SCREEN,
        "root_acceptance": audit["root_acceptance"], "user_references": audit["user_references"],
        "inherited_user_references": parent["user_references"],
        "current_request_record": AUDIT, "inherited_row_maps": parent["row_maps"],
        "user_approval": parent["user_approval"], "requested_image_ids": sorted({i for i, _ in NAMED}),
        "reviewed_image_ids": sorted(REVIEWED), "changed_image_ids": changed_ids,
        "pdf_image_ids": changed_ids, "excluded_training_image_ids": excluded,
        "held_training_image_ids": ["new_000015"],
        "additional_serialization_repairs": [], "operation_counts": {"replace": 23, "delete": 0, "add": 0},
        "field_group_changes": dict(sorted(groups_count.items())), "expected_counts": review_counts,
        "frozen_references": FROZEN_REFERENCES, "files": files,
        "companion_training_component": {"builder": "scripts/front_review_r28_training.py",
            "candidate_id": "front-review-2026-11-23-r28",
            "scope": "Managed separately; this review replay neither runs training nor establishes a model improvement."},
        "replay_script": descriptor(root, "scripts/front_review_r28.py"),
    }
    for key, (name, value) in documents.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(root, relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def verify_review(root):
    """Read-only deterministic replay; raise ValueError on any differing artifact."""
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"R28 artifact differs from accepted original-pixel decisions: {relative}")
    label_root = root / "data" / "reviews" / REVIEW_ID / "labels"
    actual = {str(p.relative_to(root)) for p in label_root.rglob("*") if p.is_file() or p.is_symlink()}
    expected = {p for p in payloads if p.startswith(f"data/reviews/{REVIEW_ID}/labels/")}
    require(actual == expected and len(actual) == 157, "R28 must contain exactly157 label files")
    return {
        "schema_version": 1, "kind": "standalone_original_pixel_review_verification",
        "verified": True, "status": "verified", "review_manifest": REVIEW_PATH,
        "review_manifest_sha256": sha(root / REVIEW_PATH), "row_review": AUDIT,
        "source_revision": SOURCE_REVISION, "label_files_verified": 157,
        "original_trainer_validation_objects": 664, "frozen_primary_objects": 666,
        "frozen_diagnostic_objects": 663, "unchanged_tokens_and_row_identities_preserved": True,
        "source_images_parent_labels_and_frozen_references_verified": True,
        "root_acceptance_and_independent_evidence_verified": True, **summary,
    }


def build(root):
    root = Path(root).resolve()
    payloads, _ = expected_review(root)
    require(all(not (root / path).exists() for path in payloads),
            "Preserve existing R28 artifacts; run verification instead of rebuilding")
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
    parser.add_argument("--build", action="store_true", help="Create immutable snapshot files; never overwrite")
    parser.add_argument("--receipt", type=Path, help="Save the deterministic verification receipt")
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = build(root) if args.build else verify_review(root)
    if args.receipt is not None:
        save_receipt(root, args.receipt, receipt)
    print(json.dumps(receipt, indent=2))
