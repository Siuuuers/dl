#!/usr/bin/env python3
"""Replay r22 colour-assisted contour review and independently adjudicated landmarks."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import FrontReviewExperiment, apply_operations
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "41dd2be835f5b0e30b10e15431c6795cb00f7388"
PARENT = {"path": "data/reviews/fronts-2026-11-22-r21/manifest.json",
          "sha256": "7e66bfb6f5889b81a7ba6fadc0d66e28979c5ac79dae9995b4f17eeea281fb49"}
REVIEW_ID = "fronts-2026-11-22-r22"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
CANDIDATE_ID = "front-review-2026-11-22-r22"
PROTOCOL_PATH = "experiments/front-review-2026-11-22-r22.json"
LEDGER = "results/review/front-review-20261122-r22"
AUDIT_PATH = f"{LEDGER}/visual-audit.json"
POLICY_PATH = "docs/annotation-boundaries-r22.md"
FEEDBACK_PATH = f"{LEDGER}/user-feedback.txt"
ABSENT_NUMBERS = (21, 22, 46, 47, 48, 52, 60, 76, 77, 78, 91, 92, 93, 104, 105, 106, 107, 108, 109, 110)
REQUESTED = tuple(f"new_{n:06d}" for n in range(1, 131) if n not in ABSENT_NUMBERS)
ABSENT = tuple(f"new_{n:06d}" for n in ABSENT_NUMBERS)
PROTECTED = {("new_000086", 3), ("new_000087", 5), ("new_000094", 31), ("new_000094", 32)}
VERTICAL_ALLOWED = {("new_000094", 23)}
POINT_DELETIONS = {("new_000096", 37, "right"), ("new_000096", 38, "right")}
NAMED_EDGES = {
    "new_000082": {"xmin": (4, 11)},
    "new_000096": {"xmax": (*range(5, 10), *range(16, 20), *range(25, 34), 35, 36)},
    "new_000100": {"xmax": tuple(range(34, 49))},
}
EPSILON_PX = 1e-6


def descriptor(root, relative, payload=None):
    return {"path": relative, "sha256": sha(root / relative) if payload is None else digest(payload)}


def source_state(root):
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    require(len(parent["files"]) == 157 and {f["image_id"] for f in parent["files"]} == set(by_id),
            "The parent must contain all 157 original photos")
    before = {}
    for item in parent["files"]:
        iid = item["image_id"]
        require(item["label_path"] == by_id[iid]["label"], f"Parent mapping differs: {iid}")
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(sha(path) == item["label_sha256"], f"Parent label drift: {iid}")
        require(sha(root / "data" / by_id[iid]["image"]) == by_id[iid]["image_sha256"],
                f"Original image drift: {iid}")
        before[iid] = path.read_bytes()
    require(len(REQUESTED) == 110 and all(i in by_id and before[i].strip() for i in REQUESTED)
            and all(i not in by_id for i in ABSENT), "Prepared review inventory changed")
    require(sum(by_id[i]["split"] == "train" for i in REQUESTED) == 93,
            "The 93 training / 17 development review split changed")
    return parent, records, by_id, before


def point_values(tokens, index, dimensions):
    return [float(tokens[index]) * dimensions[0], float(tokens[index + 1]) * dimensions[1], int(tokens[index + 2])]


def close_list(given, expected):
    return isinstance(given, list) and len(given) == len(expected) and all(
        type(a) in (int, float) and math.isfinite(a) and abs(a - b) <= EPSILON_PX
        for a, b in zip(given, expected))


def read_audit(root, by_id, before):
    evidence = descriptor(root, AUDIT_PATH)
    audit = json.loads(bound(root, evidence).read_text())
    require(audit.get("schema_version") == 1 and audit.get("source_review") == PARENT
            and audit.get("source_revision") == SOURCE_REVISION, "Audit source identity differs")
    photos = audit["photos"]
    require(len(photos) == len(REQUESTED) and {p["image_id"] for p in photos} == set(REQUESTED),
            "Audit must inspect all 110 supplied new-series photos exactly once")
    for photo in photos:
        iid = photo["image_id"]
        require(photo["source_image_sha256"] == by_id[iid]["image_sha256"]
                and photo["source_label_sha256"] == digest(before[iid])
                and type(photo["source_rows"]) is int
                and photo["source_rows"] == len(before[iid].splitlines()), f"Audit source differs: {iid}")
        require(isinstance(photo.get("inspection"), str) and photo["inspection"].strip(),
                f"Missing photo inspection: {iid}")
    operations, seen = [], set()
    for proposal in audit["proposals"]:
        iid, row = proposal["image_id"], proposal["row"]
        require(iid in REQUESTED and type(row) is int and 1 <= row <= len(before[iid].splitlines()),
                "Invalid amendment image or immutable row")
        require((iid, row) not in seen and (iid, row) not in PROTECTED,
                f"Repeated or protected reference row: {iid}:{row}")
        old_line, new_line = proposal["before_line"], proposal["after_line"]
        require(old_line == before[iid].decode().splitlines()[row - 1]
                and isinstance(new_line, str) and "\n" not in new_line and "\r" not in new_line,
                f"Before-line or replacement differs: {iid}:{row}")
        a, b = old_line.split(), new_line.split()
        require(len(a) == len(b) == 11 and a != b and a[0] == b[0],
                f"Class or pose-row structure differs: {iid}:{row}")
        require(isinstance(proposal.get("reason"), str) and proposal["reason"].strip(),
                f"Missing physical reason: {iid}:{row}")
        dimensions = by_id[iid]["original_size"]
        old = parse_pose_labels(old_line, *dimensions, iid)[0]["bbox_xyxy"]
        objects = parse_pose_labels(new_line, *dimensions, iid)
        check_geometry(objects, *dimensions, iid)
        new = objects[0]["bbox_xyxy"]
        require(close_list(proposal.get("bounds_xyxy"), new),
                f"Pixel bounds differ from serialized row: {iid}:{row}")
        vertical = any(abs(new[i] - old[i]) > EPSILON_PX for i in (1, 3))
        if vertical:
            adjudication = proposal.get("vertical_review", {})
            require((iid, row) in VERTICAL_ALLOWED and all(
                isinstance(adjudication.get(key), str) and adjudication[key].strip()
                for key in ("reason", "independent_review", "coordinate_interpretation")),
                f"Vertical change lacks specific request/adjudication: {iid}:{row}")
            require(abs(new[1] - 605) <= EPSILON_PX and abs(new[3] - old[3]) <= EPSILON_PX,
                    "094:23 must restore the visible upper face to y=605 and preserve its bottom")
        else:
            require(a[2] == b[2] and a[4] == b[4], f"Unchanged vertical tokens were rewritten: {iid}:{row}")
        changed_points = {side for side, index in (("left", 5), ("right", 8)) if a[index:index+3] != b[index:index+3]}
        repairs = proposal.get("point_repairs", [])
        require(isinstance(repairs, list) and len(repairs) == len(changed_points)
                and {item.get("side") for item in repairs} == changed_points,
                f"Point change lacks separate visual adjudication: {iid}:{row}")
        for item in repairs:
            index = 5 if item["side"] == "left" else 8
            previous, current = (point_values(t, index, dimensions) for t in (a, b))
            require(previous[2] in (0, 2) and current[2] in (0, 2)
                    and (current[2] != 0 or b[index:index+3] == ["0", "0", "0"]),
                    f"Missing points must be canonical zero triples: {iid}:{row}")
            require(any(abs(x-y) > EPSILON_PX for x,y in zip(previous,current)),
                    f"Point repair only changes formatting: {iid}:{row}")
            require(close_list(item.get("before"), previous) and close_list(item.get("after"), current),
                    f"Point evidence differs from coordinates: {iid}:{row}")
            require(all(isinstance(item.get(key), str) and item[key].strip()
                        for key in ("reason", "independent_review")),
                    f"Missing point rationale or independent review: {iid}:{row}")
            region = item.get("evidence_region_xyxy", [])
            require(len(region) == 4 and 0 <= region[0] < region[2] <= dimensions[0]
                    and 0 <= region[1] < region[3] <= dimensions[1], f"Invalid point evidence region: {iid}:{row}")
        groups = (["box"] if a[1:5] != b[1:5] else []) + sorted(changed_points)
        if "box" in groups:
            require(any(abs(x-y) > EPSILON_PX for x,y in zip(old,new)),
                    f"Box change only rewrites formatting: {iid}:{row}")
        require(groups, f"Replacement has no changed annotation fields: {iid}:{row}")
        operations.append({"image_id": iid, "split": by_id[iid]["split"], "base_row": row,
            "action": "replace", "before_line": old_line, "after_line": new_line,
            "bounds_xyxy": new, "reason": proposal["reason"],
            "source_image_sha256": by_id[iid]["image_sha256"], "source_label_sha256": digest(before[iid]),
            "changed_groups": groups, "point_repairs": repairs,
            "vertical_review": proposal.get("vertical_review"), "audit_source": evidence})
        seen.add((iid, row))
    return sorted(operations, key=lambda op: (op["image_id"], op["base_row"])), evidence


def expected_review(root):
    parent, records, by_id, before = source_state(root)
    operations, audit = read_audit(root, by_id, before)
    policy, feedback = descriptor(root, POLICY_PATH), descriptor(root, FEEDBACK_PATH)
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_text()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"])
    excluded = sorted(r["id"] for r in records if r["id"].startswith("old_"))
    require(len(excluded) == 31 and status["excluded_training_image_ids"] == excluded
            and parent["excluded_training_image_ids"] == excluded
            and status["pending_train_photos"] == ["new_000015"]
            and status["unresolved_development_photos"] == [], "Inherited decisions or exclusions changed")
    approval = json.loads(bound(root, parent["user_approval"]).read_text())
    require(len(approval["decisions"]) == 17, "Inherited displayed-case approvals changed")
    payloads, by_image, maps, parsed, files = {}, defaultdict(list), {}, {}, []
    for op in operations:
        by_image[op["image_id"]].append(op)
    for rec in records:
        iid = rec["id"]
        payload, maps[iid] = apply_operations(before[iid], by_image.get(iid, []), iid)
        parsed[iid] = parse_pose_labels(payload.decode(), *rec["original_size"], iid)
        check_geometry(parsed[iid], *rec["original_size"], iid)
        payloads[f"data/reviews/{REVIEW_ID}/{rec['label']}"] = payload
        c = counts(parsed[iid])
        files.append({"image_id": iid, "split": rec["split"], "label_path": rec["label"],
            "label_sha256": digest(payload), "source_image_sha256": rec["image_sha256"],
            "objects": c["objects"], "locatable_corners": c["visible_endpoints"],
            "missing_corners": c["missing_endpoints"], "changed_in_r22": bool(by_image.get(iid)),
            "fresh_visual_review": iid in REQUESTED,
            "held_current_rows": sorted({row for case in status["cases"] if case["image_id"] == iid
                                         and case["status"] != "resolved" for row in case["current_rows"]})})
    review_counts = totals(records, parsed)
    point_transitions, split_point_delta, groups, directions = Counter(), Counter(), Counter(), Counter()
    changes_by_split = {split: Counter() for split in ("train", "val")}
    vertical_rows, point_x, point_y = [], 0, 0
    for op in operations:
        groups.update(op["changed_groups"])
        split = op["split"]
        changes_by_split[split]["rows"] += 1
        changes_by_split[split]["boxes"] += "box" in op["changed_groups"]
        for repair in op["point_repairs"]:
            old, new = repair["before"], repair["after"]
            kind = "reposition" if old[2] == new[2] else "remove" if new[2] == 0 else "restore"
            point_transitions[kind] += 1
            split_point_delta[split] += (new[2] != 0) - (old[2] != 0)
            changes_by_split[split]["point_repairs"] += 1
            point_x += old[2] == new[2] == 2 and abs(old[0] - new[0]) > EPSILON_PX
            point_y += old[2] == new[2] == 2 and abs(old[1] - new[1]) > EPSILON_PX
        old, new = (parse_pose_labels(op[key], *by_id[op["image_id"]]["original_size"])[0]["bbox_xyxy"]
                    for key in ("before_line", "after_line"))
        for side, index, inward_sign in (("xmin", 0, 1), ("xmax", 2, -1)):
            delta = new[index] - old[index]
            if abs(delta) > EPSILON_PX:
                directions["inward" if delta * inward_sign > 0 else "outward"] += 1
        if any(abs(new[i]-old[i]) > EPSILON_PX for i in (1,3)):
            vertical_rows.append({"image_id": op["image_id"], "base_row": op["base_row"],
                                  "before_xyxy": old, "after_xyxy": new, "adjudication": op["vertical_review"]})
    for split, previous in parent["expected_counts"].items():
        expected = dict(previous)
        expected["visible_endpoints"] += split_point_delta[split]
        expected["missing_endpoints"] -= split_point_delta[split]
        require(review_counts[split] == expected, f"Unexplained count or cohort change: {split}")
    operation_index = {(op["image_id"], op["base_row"]): op for op in operations}
    named_coverage = []
    for iid, requested in NAMED_EDGES.items():
        for side, rows in requested.items():
            index, sign = (0, 1) if side == "xmin" else (2, -1)
            for row in rows:
                require((iid, row) in operation_index, f"Named correction missing: {iid}:{row}")
                op = operation_index[iid, row]
                old, new = (parse_pose_labels(op[key], *by_id[iid]["original_size"])[0]["bbox_xyxy"]
                            for key in ("before_line", "after_line"))
                require((new[index]-old[index])*sign > EPSILON_PX,
                        f"Named rack edge did not move inward: {iid}:{row}:{side}")
                named_coverage.append({"image_id": iid, "base_row": row, "field": side,
                    "before_px": old[index], "after_px": new[index], "disposition": "corrected"})
    require(len(named_coverage) == 37, "Named horizontal feedback inventory differs")
    removed = set()
    for op in operations:
        for repair in op["point_repairs"]:
            if repair["before"][2] == 2 and repair["after"][2] == 0:
                removed.add((op["image_id"], op["base_row"], repair["side"]))
    require(POINT_DELETIONS <= removed, "Both explicitly requested right points must be missing")
    require({(r["image_id"],r["base_row"]) for r in vertical_rows} == VERTICAL_ALLOWED,
            "The specifically requested vertical extent must be adjudicated")
    changed = sorted(by_image)
    actual_changed = sorted(rec["id"] for rec in records
                            if payloads[f"data/reviews/{REVIEW_ID}/{rec['label']}"] != before[rec["id"]])
    require(changed == actual_changed and set(changed) <= set(REQUESTED),
            "Changed-photo inventory must match actual label-byte differences within the reviewed scope")
    summary = {"review_id": REVIEW_ID, "requested_photos": len(REQUESTED),
        "absent_requested_range_image_ids": list(ABSENT),
        "reviewed_training_photos": 93, "reviewed_development_photos": 17,
        "reviewed_rows": sum(len(before[i].splitlines()) for i in REQUESTED),
        "changed_photos": len(changed), "changed_rows": len(operations), "changed_box_rows": groups["box"],
        "operation_counts": {"replace": len(operations)}, "field_group_changes": dict(groups),
        "horizontal_edge_direction_counts": dict(directions), "vertical_box_changes": vertical_rows,
        "named_horizontal_edges": len(named_coverage), "named_horizontal_edges_corrected": len(named_coverage),
        "requested_right_point_deletions": len(POINT_DELETIONS),
        "point_repairs": sum(point_transitions.values()), "point_transitions": dict(point_transitions),
        "visible_point_x_changes": point_x, "visible_point_y_changes": point_y,
        "changes_by_split": {s:dict(c) for s,c in changes_by_split.items()},
        "expected_review_counts": review_counts, "changed_image_ids": changed, "pdf_image_ids": changed,
        "full_nonempty_pdf_photos": sum(f["objects"] > 0 for f in files),
        "full_nonempty_pdf_instances": sum(f["objects"] for f in files),
        "instance_counts_preserved": True, "confirmed_reference_rows_preserved": True,
        "unchanged_review_files": 157-len(changed), "unrequested_review_files_identical_to_r21": 47,
        "inherited_displayed_case_approvals": 17, "new_cases": 0, "new_holds": 0,
        "held_training_photo_ids": status["pending_train_photos"], "old_review_files_preserved": 31}
    coverage = {"schema_version": 1, "review_id": REVIEW_ID, "source_review": PARENT,
        "source_revision": SOURCE_REVISION, "reviewed_image_ids": list(REQUESTED), "audit": audit,
        "annotation_policy": policy, "text_feedback": feedback,
        "inherited_user_approval": parent["user_approval"], "full_dataset_certified": False,
        "named_edge_dispositions": named_coverage, "vertical_request_adjudication": vertical_rows,
        "requested_missing_points": [{"image_id":i,"base_row":r,"side":s,"disposition":"canonical_missing"}
                                     for i,r,s in sorted(POINT_DELETIONS)],
        "preserved_standard_rows": [{"image_id":i,"row":r} for i,r in sorted(PROTECTED)]}
    for name, value in (("operations.json",operations),("row-maps.json",maps),("review-status.json",status),
                        ("request-coverage.json",coverage),("amendment-summary.json",summary)):
        payloads[f"{LEDGER}/{name}"] = encoded(value)
    manifest = {"schema_version":1,"kind":"original_pixel_front_review","review_id":REVIEW_ID,
        "status":"visually_reviewed_contours_and_landmarks","date":"2026-11-22","parent":PARENT,
        "source_revision":SOURCE_REVISION,"source_manifest_sha256":parent["source_manifest_sha256"],
        "full_dataset_certified":False,"annotation_policy":policy,"boundary_rubric":policy,
        "review_evidence":[audit],"user_references":[feedback],"inherited_row_maps":parent["row_maps"],
        "user_approval":parent["user_approval"],"requested_image_ids":list(REQUESTED),
        "changed_image_ids":changed,"pdf_image_ids":changed,"excluded_training_image_ids":excluded,
        "additional_serialization_repairs":[],"operation_counts":summary["operation_counts"],
        "field_group_changes":summary["field_group_changes"],"expected_counts":review_counts,"files":files}
    for key,name in (("operations","operations.json"),("row_maps","row-maps.json"),
                     ("review_status","review-status.json"),("request_coverage","request-coverage.json"),
                     ("amendment_summary","amendment-summary.json")):
        relative = f"{LEDGER}/{name}"
        manifest[key] = descriptor(root,relative,payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads,summary


def experiment(root):
    records = json.loads((root/"data/manifest.json").read_text())["records"]
    return FrontReviewExperiment(review=descriptor(root,REVIEW_PATH),parent=PARENT,
        candidate_id=CANDIDATE_ID,protocol_path=PROTOCOL_PATH,experiment_id="front-review-r22",
        source_revision=SOURCE_REVISION,round_number=22,experiment_version=21,serialization_repairs=(),
        review_development_labels=True,
        protocol_limits=("Colour is a review cue, never an automatic material classifier. Correct visible front contours and independently adjudicated actual lower-corner landmarks against immutable r21. Preserve object identities and unrelated fields; canonical missing points are 0 0 0.",
                         "Photo 015 and the 31 old-series treatment exclusions keep exact PR35 supervision in both arms.",
                         "Preparation does not run training or promote a model."),
        excluded_training_image_ids=tuple(sorted(r["id"] for r in records if r["id"].startswith("old_"))))


def verify_candidate(root,candidate=None):
    root=Path(root).resolve()
    payloads,summary=expected_review(root)
    for relative,payload in payloads.items():
        path=root/relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes()==payload,
                f"r22 artifact differs from its exact visual audit: {relative}")
    configured=experiment(root)
    receipt=configured.verify_candidate(root,candidate)
    require(receipt["counts"]["train"]["objects"]==3784,"Object-preserving review changed candidate count")
    receipt.update(summary)
    receipt["excluded_training_image_ids"]=list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"]=True
    return receipt


def build_review(root):
    root=Path(root).resolve()
    require(not(root/REVIEW_PATH).parent.exists() and not(root/f"data/candidates/{CANDIDATE_ID}").exists()
            and not(root/PROTOCOL_PATH).exists(),"r22 already exists; verify instead of overwriting")
    payloads,_=expected_review(root)
    require(all(not(root/p).exists() and not(root/p).is_symlink() for p in payloads),
            "Generated r22 evidence exists; preserve completed evidence")
    for relative,payload in payloads.items():
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:stream.write(payload)
    experiment(root).build(root)
    return verify_candidate(root)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    print(json.dumps(build_review(args.root) if args.build else verify_candidate(args.root),indent=2))
