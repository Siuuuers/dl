#!/usr/bin/env python3
"""Replay the bounded R26 original-pixel decisions on immutable R25 labels."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import FrontReviewExperiment, apply_operations
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "ace99d4aa71e5c20d7c5d2445f18fdab58d63496"
PARENT = {"path": "data/reviews/fronts-2026-11-23-r25/manifest.json",
          "sha256": "de80c21ed4b148efe6446e66b8a15f75c3b6363daf9c03d0b88a29e187483517"}
REVIEW_ID = "fronts-2026-11-23-r26"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
CANDIDATE_ID = "front-review-2026-11-23-r26"
PROTOCOL_PATH = "experiments/front-review-2026-11-23-r26.json"
LEDGER = "results/review/front-review-20261123-r26"
REQUESTED = {"new_000007": (4, 8, 9, 14, 15), "new_000008": (5, 36),
             "new_000009": (48, 49, 50, 51, 52), "new_000010": (24,),
             "new_000011": (54,), "new_000053": (1,), "new_000089": (48, 49),
             "new_000090": (39, 40, 41, 42, 43)}
TRAIN_ANALOGUES = (("new_000013", 29), ("new_000013", 30), ("new_000013", 31))
STANDARDS = tuple(("new_000008", r) for r in (18, 19, 20, 21, 28, 29, 30, 31, 32)) + (
    ("new_000010", 16), ("new_000016", 24))


def descriptor(root, path, payload=None):
    return {"path": path, "sha256": sha(root / path) if payload is None else digest(payload)}


def expected_review(root):
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda record: (record["split"], record["id"]))
    evidence = descriptor(root, f"{LEDGER}/row-review.json")
    audit = json.loads(bound(root, evidence).read_text())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION,
            "R26 decisions do not identify the immutable R25 source")
    for item in audit["independent_review_evidence"]:
        bound(root, item)
    named = {(image_id, row) for image_id, rows in REQUESTED.items() for row in rows}
    reviewed = named | set(TRAIN_ANALOGUES)
    require(len(named) == 22 and len(audit["rows"]) == 25 and
            {(r["image_id"], r["row"]) for r in audit["rows"]} == reviewed,
            "Every named row and each declared training analogue needs one decision")
    require([(r["image_id"], r["row"]) for r in audit["preserved_standards"]] == list(STANDARDS),
            "Accepted standard inventory differs")
    before = {}
    for item in parent["files"]:
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(sha(path) == item["label_sha256"], f"R25 label drift: {item['image_id']}")
        before[item["image_id"]] = path.read_bytes()
    operations, group_counts, split_changes = [], Counter(), Counter()
    point_splits, box_splits, action_counts = Counter(), Counter(), Counter()
    for item in audit["rows"]:
        image_id, row = item["image_id"], item["row"]
        record = by_id[image_id]
        require(item["scope"] == ("user_named" if (image_id, row) in named else "training_analogue"),
                "Named/analogue scope attribution differs")
        require(item["split"] == record["split"] and record["original_size"] == [1280, 720],
                "Row image or split changed")
        require(sha(root / "data" / record["image"]) == item["source_image_sha256"] == record["image_sha256"],
                f"Original pixels differ: {image_id}")
        old_line = before[image_id].decode().splitlines()[row - 1]
        require(item["before_line"] == old_line, f"Immutable R25 row differs: {image_id}:{row}")
        require(item.get("reason") and item.get("root_review"), "Missing visual decision rationale")
        action = item["action"]
        action_counts[action] += 1
        if action == "delete":
            require((image_id, row) == ("new_000008", 36) and item["after_line"] is None,
                    "Only the explicitly reviewed non-front patch may be deleted")
            groups = ["object"]
        else:
            old, new = old_line.split(), item["after_line"].split()
            require(action in ("retain", "replace") and len(new) == 11 and old[0] == new[0]
                    and old[7] == new[7] and old[10] == new[10],
                    "Class and endpoint visibility must be preserved")
            for offset in (5, 8):
                if int(old[offset + 2]) == 0:
                    require(old[offset:offset + 3] == new[offset:offset + 3]
                            and all(float(v) == 0 for v in new[offset:offset + 3]),
                            "Unavailable endpoints must retain exact zero tokens")
            groups = [name for name, start, stop in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
                      if old[start:stop] != new[start:stop]]
            require((action == "replace") == bool(groups) and groups == item["changed_groups"],
                    f"Saved field decisions differ: {image_id}:{row}")
            check_geometry(parse_pose_labels(item["after_line"], 1280, 720), 1280, 720, image_id)
            if action == "retain":
                require(item["after_line"] == old_line, "Retained row must be byte-identical")
                continue
            point_splits[record["split"]] += sum(g in ("left", "right") for g in groups)
            box_splits[record["split"]] += "box" in groups
        operation = {"image_id": image_id, "split": record["split"], "base_row": row,
                     "action": action, "before_line": old_line, "changed_groups": groups,
                     "reason": item["reason"]}
        if action != "delete":
            operation["after_line"] = item["after_line"]
        operations.append(operation)
        group_counts.update(groups)
        split_changes[record["split"]] += 1
    require(dict(action_counts) == {"replace": 19, "delete": 1, "retain": 5}, "R26 operation scope changed")
    require(dict(point_splits) == {"val": 6, "train": 3} and dict(box_splits) == {"val": 16, "train": 2},
            "R26 training/development intervention changed")
    operations.sort(key=lambda item: (item["image_id"], item["base_row"]))
    status = copy.deepcopy(json.loads(bound(root, parent["review_status"]).read_text()))
    status.update(review_id=REVIEW_ID, inherited_status=parent["review_status"])
    payloads, files, maps, parsed = {}, [], {}, {}
    for record in records:
        image_id = record["id"]
        edits = [item for item in operations if item["image_id"] == image_id]
        payload, maps[image_id] = apply_operations(before[image_id], edits, image_id)
        parsed[image_id] = parse_pose_labels(payload.decode(), *record["original_size"], image_id)
        check_geometry(parsed[image_id], *record["original_size"], image_id)
        payloads[f"data/reviews/{REVIEW_ID}/{record['label']}"] = payload
        statistics = counts(parsed[image_id])
        files.append({"image_id": image_id, "split": record["split"], "label_path": record["label"],
                      "label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                      "objects": statistics["objects"], "locatable_corners": statistics["visible_endpoints"],
                      "missing_corners": statistics["missing_endpoints"], "changed_in_r26": bool(edits)})
    for item in audit["preserved_standards"]:
        image_id, row = item["image_id"], item["row"]
        path = f"data/reviews/{REVIEW_ID}/{by_id[image_id]['label']}"
        require(payloads[path].splitlines()[row - 1] == before[image_id].splitlines()[row - 1]
                == item["line"].encode(), f"Accepted standard changed: {image_id}:{row}")
    review_counts = totals(records, parsed)
    expected_counts = copy.deepcopy(parent["expected_counts"])
    expected_counts["val"]["objects"] -= 1
    expected_counts["val"]["missing_endpoints"] -= 2
    require(review_counts == expected_counts, "Only the reviewed zero-corner development row may change counts")
    changed_images = sorted({op["image_id"] for op in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT, "named_rows_reviewed": len(named),
               "training_analogue_rows_reviewed": len(TRAIN_ANALOGUES), "protected_standard_rows": len(STANDARDS),
               "changed_rows": len(operations), "replacement_rows": action_counts["replace"],
               "deleted_rows": action_counts["delete"], "retained_named_rows": action_counts["retain"],
               "changed_photos": len(changed_images), "changed_image_ids": changed_images,
               "changed_box_rows": group_counts["box"], "changed_box_rows_by_split": dict(box_splits),
               "point_repairs": group_counts["left"] + group_counts["right"],
               "point_repairs_by_side": {k: group_counts[k] for k in ("left", "right")},
               "point_repairs_by_split": dict(point_splits), "changed_rows_by_split": dict(split_changes),
               "unchanged_review_files": len(records) - len(changed_images),
               "confirmed_reference_rows_preserved": True, "classes_visibility_preserved": True,
               "development_review_labels_changed": True, "training_validation_preserved": True,
               "expected_review_counts": review_counts, "training_run": False}
    policy = descriptor(root, "docs/annotation-boundaries-r26.md")
    feedback = descriptor(root, f"{LEDGER}/user-feedback.txt")
    coverage = {"review_id": REVIEW_ID, "source_review": PARENT,
                "requested_rows": [{"image_id": i, "rows": list(rows)} for i, rows in REQUESTED.items()],
                "additional_training_rows": [{"image_id": i, "row": row} for i, row in TRAIN_ANALOGUES],
                "row_decisions": evidence, "preserved_standards": [{"image_id": i, "row": r} for i, r in STANDARDS],
                "full_dataset_re_review": False, "all_named_photos_are_development": True}
    documents = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
                 "review_status": ("review-status.json", status), "request_coverage": ("request-coverage.json", coverage),
                 "amendment_summary": ("amendment-summary.json", summary)}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "visually_reviewed_front_face_boundaries", "date": "2026-11-23", "parent": PARENT,
                "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
                "full_dataset_certified": False, "annotation_policy": policy, "boundary_rubric": policy,
                "review_evidence": [evidence], "user_references": [feedback],
                "inherited_row_maps": parent["row_maps"], "user_approval": parent["user_approval"],
                "requested_image_ids": sorted(set(REQUESTED) | {i for i, _ in TRAIN_ANALOGUES}),
                "changed_image_ids": changed_images, "pdf_image_ids": [],
                "excluded_training_image_ids": parent["excluded_training_image_ids"],
                "additional_serialization_repairs": [], "operation_counts": {"replace": 19, "delete": 1},
                "field_group_changes": dict(sorted(group_counts.items())), "expected_counts": review_counts, "files": files}
    for key, (name, value) in documents.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(root, relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def experiment(root):
    review = json.loads((root / REVIEW_PATH).read_text())
    return FrontReviewExperiment(review=descriptor(root, REVIEW_PATH), parent=PARENT,
        candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH, experiment_id="front-review-r26",
        source_revision=SOURCE_REVISION, round_number=26, experiment_version=25, serialization_repairs=(),
        review_development_labels=True, excluded_training_image_ids=tuple(review["excluded_training_image_ids"]),
        protocol_limits=("R26 reviews all22 named development rows and3 analogous training rows. Original pixels support19 replacements and one non-front deletion; five named rows are retained. Missing-point tokens and11 accepted standard rows remain exact. No universal neighbor bound or size-based deletion is applied.",
                         "Only013:29-31 changes training relative to R25: three left corners and two xmin trims. The eight named photographs remain development-only; their16 replacements and one deletion enter only the separately versioned661-front exploratory review. The original664-front trainer validation and frozen666/663 scoring references are unchanged.",
                         "PR101 R25 results did not support replacing PR35. The accumulated candidate is still compared with PR35 supervision; it does not isolate the three latest training rows. Use the returned R25 candidate as the saved predecessor comparator under the same references. This small supervised correction is an experiment, not an established remedy for partial-front misses or proof of new-session reliability. Preserve the fixed recipe, operating point and final-epoch rule; no training or promotion during preparation."))


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"R26 artifact differs from saved original-pixel decisions: {relative}")
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(summary)
    receipt["excluded_training_image_ids"] = list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"] = True
    return receipt


def build(root):
    root = Path(root).resolve()
    payloads, _ = expected_review(root)
    require(all(not (root / path).exists() for path in payloads), "Preserve existing R26 artifacts; verify instead")
    require(not (root / f"data/candidates/{CANDIDATE_ID}").exists() and not (root / PROTOCOL_PATH).exists(),
            "The R26 candidate already exists")
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
    print(json.dumps(build(args.root) if args.build else verify_candidate(args.root), indent=2))
