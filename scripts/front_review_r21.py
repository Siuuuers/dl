#!/usr/bin/env python3
"""Replay r21 silhouette corrections and explicitly reviewed point-X repairs."""
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

SOURCE_REVISION = "bf13f8c293c316b62ede247324ef1e4e45c979a7"
PARENT = {"path": "data/reviews/fronts-2026-11-22-r20/manifest.json",
          "sha256": "0aac0accef28e13fab3db1939b8d9d559859b4c9e187bd0b0ee3158c31dd6eb7"}
REVIEW_ID = "fronts-2026-11-22-r21"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
CANDIDATE_ID = "front-review-2026-11-22-r21"
PROTOCOL_PATH = "experiments/front-review-2026-11-22-r21.json"
LEDGER = "results/review/front-review-20261122-r21"
AUDIT_PATH = f"{LEDGER}/horizontal-audit.json"
POLICY_PATH = "docs/annotation-boundaries-r21.md"
FEEDBACK_PATH = f"{LEDGER}/user-feedback.txt"
REQUESTED = tuple(f"new_{n:06d}" for n in (*range(81, 91), *range(94, 104), *range(111, 131)))
ABSENT = tuple(f"new_{n:06d}" for n in (*range(91, 94), *range(104, 111)))
PROTECTED = {("new_000087", 5), ("new_000086", 3),
             ("new_000094", 31), ("new_000094", 32)}
PRESERVED_TOKENS = (0, 2, 4, 6, 7, 9, 10)
EPSILON_PX = 1e-6
NAMED_EDGES = {
    "new_000084": {"rack_xmax": (20, 29)},
    "new_000085": {"rack_xmax": (8, 20, 21, 22, 23, 32)},
    "new_000086": {"rack_xmax": (*range(10, 15), *range(16, 25),
                                  *range(36, 42), 46, 49, 50, 51),
                   "rack_xmin": (53, 54, 55, 56)},
    "new_000087": {"clipped_xmax": (3, 53, 54, 55)},
    "new_000089": {"clipped_xmin": (39, 44)},
    "new_000090": {"clipped_xmin": (39,), "clipped_xmax": (44, 45, 46)},
}


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
        require(item["label_path"] == by_id[iid]["label"], f"Parent label mapping differs: {iid}")
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(sha(path) == item["label_sha256"], f"Parent label drift: {iid}")
        require(sha(root / "data" / by_id[iid]["image"]) == by_id[iid]["image_sha256"],
                f"Original image drift: {iid}")
        before[iid] = path.read_bytes()
    require(all(i in by_id and before[i].strip() for i in REQUESTED)
            and all(i not in by_id for i in ABSENT), "Prepared source inventory changed")
    require(sum(by_id[i]["split"] == "train" for i in REQUESTED) == 35
            and {i for i in REQUESTED if by_id[i]["split"] == "val"}
            == {f"new_{i:06d}" for i in range(86, 91)}, "Requested split changed")
    return parent, records, by_id, before


def read_audit(root, by_id, before):
    evidence = descriptor(root, AUDIT_PATH)
    audit = json.loads(bound(root, evidence).read_text())
    require(audit.get("schema_version") == 1 and audit.get("source_review") == PARENT
            and audit.get("source_revision") == SOURCE_REVISION, "Audit source identity differs")
    photos = audit["photos"]
    require(len(photos) == len(REQUESTED) and {p["image_id"] for p in photos} == set(REQUESTED),
            "Audit must inspect each of the 40 supplied photos exactly once")
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
        require(len(a) == len(b) == 11 and a != b
                and all(a[i] == b[i] for i in PRESERVED_TOKENS),
                f"Class, all Y coordinates and visibility must be preserved: {iid}:{row}")
        require(isinstance(proposal.get("reason"), str) and proposal["reason"].strip(),
                f"Missing physical reason: {iid}:{row}")
        dimensions = by_id[iid]["original_size"]
        old = parse_pose_labels(old_line, *dimensions, iid)[0]["bbox_xyxy"]
        objects = parse_pose_labels(new_line, *dimensions, iid)
        check_geometry(objects, *dimensions, iid)
        new = objects[0]["bbox_xyxy"]
        desired = [proposal["xmin"], proposal["xmax"]]
        require(all(type(v) in (int, float) and math.isfinite(v) for v in desired)
                and all(abs(v - actual) <= EPSILON_PX for v, actual in zip(desired, (new[0], new[2]))),
                f"Pixel bounds differ from serialized row: {iid}:{row}")
        require(abs(new[0] - old[0]) > EPSILON_PX or abs(new[2] - old[2]) > EPSILON_PX,
                f"Each correction must change a horizontal box boundary: {iid}:{row}")
        changed_points = {side for side, index in (("left", 5), ("right", 8)) if a[index] != b[index]}
        repairs = proposal.get("point_repairs", [])
        require(isinstance(repairs, list) and len(repairs) == len(changed_points)
                and {item.get("side") for item in repairs} == changed_points,
                f"Point change lacks separate visual adjudication: {iid}:{row}")
        for item in repairs:
            index = 5 if item["side"] == "left" else 8
            require(a[index + 2] == b[index + 2] == "2",
                    f"Only existing visible points may be repositioned: {iid}:{row}")
            xy = [[float(tokens[index]) * dimensions[0], float(tokens[index + 1]) * dimensions[1]]
                  for tokens in (a, b)]
            require(all(isinstance(item.get(key), str) and item[key].strip()
                        for key in ("reason", "independent_review")),
                    f"Missing visual point rationale/review: {iid}:{row}")
            require(all(len(item.get(key, [])) == 2
                        and all(abs(given - actual) <= EPSILON_PX for given, actual in zip(item[key], expected))
                        for key, expected in zip(("before_xy", "after_xy"), xy)),
                    f"Point evidence differs from serialized coordinates: {iid}:{row}")
            region = item.get("evidence_region_xyxy", [])
            require(len(region) == 4 and 0 <= region[0] < region[2] <= dimensions[0]
                    and 0 <= region[1] < region[3] <= dimensions[1],
                    f"Invalid original-image evidence region: {iid}:{row}")
        operations.append({"image_id": iid, "split": by_id[iid]["split"], "base_row": row,
            "action": "replace", "before_line": old_line, "after_line": new_line,
            "xmin": proposal["xmin"], "xmax": proposal["xmax"], "reason": proposal["reason"],
            "source_image_sha256": by_id[iid]["image_sha256"], "source_label_sha256": digest(before[iid]),
            "changed_groups": ["box", *sorted(changed_points)],
            "point_repairs": repairs, "audit_source": evidence})
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
        payload, maps[iid] = apply_operations(before[iid], by_image[iid], iid)
        parsed[iid] = parse_pose_labels(payload.decode(), *rec["original_size"], iid)
        check_geometry(parsed[iid], *rec["original_size"], iid)
        payloads[f"data/reviews/{REVIEW_ID}/{rec['label']}"] = payload
        c = counts(parsed[iid])
        files.append({"image_id": iid, "split": rec["split"], "label_path": rec["label"],
            "label_sha256": digest(payload), "source_image_sha256": rec["image_sha256"],
            "objects": c["objects"], "locatable_corners": c["visible_endpoints"],
            "missing_corners": c["missing_endpoints"], "changed_in_r21": bool(by_image[iid]),
            "fresh_visual_review": iid in REQUESTED,
            "held_current_rows": sorted({row for case in status["cases"] if case["image_id"] == iid
                                         and case["status"] != "resolved" for row in case["current_rows"]})})
    review_counts = totals(records, parsed)
    require(review_counts == parent["expected_counts"] and review_counts["train"]["objects"] == 3786
            and review_counts["train"]["visible_endpoints"] == 6838
            and review_counts["val"]["objects"] == 662, "Horizontal trims changed review counts")
    changed = sorted({op["image_id"] for op in operations})
    groups = Counter(group for op in operations for group in op["changed_groups"])
    edge_directions = Counter()
    named_coverage = []
    operation_index = {(op["image_id"], op["base_row"]): op for op in operations}
    for op in operations:
        old, new = (parse_pose_labels(op[key], *by_id[op["image_id"]]["original_size"])[0]["bbox_xyxy"]
                    for key in ("before_line", "after_line"))
        for side, index, inward_sign in (("xmin", 0, 1), ("xmax", 2, -1)):
            delta = new[index] - old[index]
            if abs(delta) > EPSILON_PX:
                edge_directions["inward" if delta * inward_sign > 0 else "outward"] += 1
    for iid, requested in NAMED_EDGES.items():
        for problem, rows in requested.items():
            for row in rows:
                require((iid, row) in operation_index, f"Named correction missing: {iid}:{row}")
                op = operation_index[iid, row]
                old, new = (parse_pose_labels(op[key], *by_id[iid]["original_size"])[0]["bbox_xyxy"]
                            for key in ("before_line", "after_line"))
                index = 0 if problem.endswith("xmin") else 2
                expected_sign = 1 if problem in ("rack_xmin", "clipped_xmax") else -1
                require((new[index] - old[index]) * expected_sign > EPSILON_PX,
                        f"Named edge did not move in the requested correction direction: {iid}:{row}:{problem}")
                named_coverage.append({"image_id": iid, "base_row": row, "problem": problem,
                    "before_px": old[index], "after_px": new[index], "disposition": "corrected",
                    "point_repairs": [item["side"] for item in op["point_repairs"]]})
    require(len(named_coverage) == 46, "Named feedback inventory differs")
    summary = {"review_id": REVIEW_ID, "requested_photos": 40,
        "absent_requested_range_image_ids": list(ABSENT),
        "reviewed_training_photos": 35, "reviewed_development_photos": 5,
        "changed_photos": len(changed), "changed_rows": len(operations),
        "operation_counts": {"replace": len(operations)} if operations else {},
        "field_group_changes": dict(groups), "edge_direction_counts": dict(edge_directions),
        "named_requested_edges": len(named_coverage), "named_requested_edges_corrected": len(named_coverage),
        "expected_review_counts": review_counts, "changed_image_ids": changed, "pdf_image_ids": changed,
        "point_x_repairs": sum(len(op["point_repairs"]) for op in operations),
        "rows_with_point_x_repairs": sum(bool(op["point_repairs"]) for op in operations),
        "all_other_point_tokens_preserved": True, "vertical_box_fields_preserved": True,
        "all_y_coordinates_preserved": True, "visibility_preserved": True,
        "horizontal_bounds_may_shrink_or_expand": True, "instance_counts_preserved": True,
        "confirmed_reference_rows_preserved": True,
        "unchanged_review_files": 157 - len(changed), "unrequested_review_files_identical_to_r20": 117,
        "inherited_displayed_case_approvals": 17, "new_cases": 0, "new_holds": 0,
        "held_training_photo_ids": status["pending_train_photos"], "old_review_files_preserved": 31}
    coverage = {"schema_version": 1, "review_id": REVIEW_ID, "source_review": PARENT,
        "source_revision": SOURCE_REVISION, "reviewed_image_ids": list(REQUESTED), "audit": audit,
        "annotation_policy": policy, "text_feedback": feedback,
        "inherited_user_approval": parent["user_approval"], "full_dataset_certified": False,
        "named_edge_dispositions": named_coverage,
        "preserved_standard_rows": [{"image_id": iid, "row": row} for iid, row in sorted(PROTECTED)]}
    for name, value in (("operations.json", operations), ("row-maps.json", maps),
                        ("review-status.json", status), ("request-coverage.json", coverage),
                        ("amendment-summary.json", summary)):
        payloads[f"{LEDGER}/{name}"] = encoded(value)
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
        "status": "visually_reviewed_silhouette_amendments", "date": "2026-11-22", "parent": PARENT,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
        "full_dataset_certified": False, "annotation_policy": policy, "boundary_rubric": policy,
        "review_evidence": [audit], "user_references": [feedback],
        "inherited_row_maps": parent["row_maps"], "user_approval": parent["user_approval"],
        "requested_image_ids": list(REQUESTED), "changed_image_ids": changed, "pdf_image_ids": changed,
        "excluded_training_image_ids": excluded, "additional_serialization_repairs": [],
        "operation_counts": summary["operation_counts"], "field_group_changes": summary["field_group_changes"],
        "expected_counts": review_counts, "files": files}
    for key, name in (("operations", "operations.json"), ("row_maps", "row-maps.json"),
                      ("review_status", "review-status.json"), ("request_coverage", "request-coverage.json"),
                      ("amendment_summary", "amendment-summary.json")):
        relative = f"{LEDGER}/{name}"
        manifest[key] = descriptor(root, relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def experiment(root):
    records = json.loads((root / "data/manifest.json").read_text())["records"]
    return FrontReviewExperiment(review=descriptor(root, REVIEW_PATH), parent=PARENT,
        candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH, experiment_id="front-review-r21",
        source_revision=SOURCE_REVISION, round_number=21, experiment_version=20, serialization_repairs=(),
        review_development_labels=True,
        protocol_limits=("Correct visually supported horizontal front silhouettes against r20, allowing rack trims and restoration of clipped visible body. Endpoint-X repairs require separate original-pixel adjudication; preserve every Y coordinate, visibility value and instance.",
                         "Photo 015 and the 31 old-series treatment exclusions keep exact PR35 supervision in both arms.",
                         "Preparation does not run training or promote a model."),
        excluded_training_image_ids=tuple(sorted(r["id"] for r in records if r["id"].startswith("old_"))))


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"r21 artifact differs from its exact horizontal audit: {relative}")
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    require(receipt["counts"]["train"]["objects"] == 3784
            and receipt["counts"]["train"]["visible_endpoints"] == 6838,
            "Horizontal trims changed candidate counts")
    receipt.update(summary)
    receipt["excluded_training_image_ids"] = list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"] = True
    return receipt


def build_review(root):
    root = Path(root).resolve()
    require(not (root / REVIEW_PATH).parent.exists()
            and not (root / f"data/candidates/{CANDIDATE_ID}").exists()
            and not (root / PROTOCOL_PATH).exists(), "r21 already exists; verify instead of overwriting")
    payloads, _ = expected_review(root)
    require(all(not (root / path).exists() and not (root / path).is_symlink() for path in payloads),
            "Generated r21 evidence already exists; preserve it and verify instead")
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    experiment(root).build(root)
    return verify_candidate(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_review(args.root) if args.build else verify_candidate(args.root), indent=2))
