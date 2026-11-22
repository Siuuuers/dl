#!/usr/bin/env python3
"""Replay the focused r24 corner review on immutable r23 labels."""
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

SOURCE_REVISION = "ee2ca5f260af96621aed4bf5803680b9b6403349"
PARENT = {"path": "data/reviews/fronts-2026-11-22-r23/manifest.json",
          "sha256": "d93be83707ee3090d05808333410c924c8a32df82df87f4579f0c3aead4176d0"}
REVIEW_ID = "fronts-2026-11-23-r24"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
CANDIDATE_ID = "front-review-2026-11-23-r24"
PROTOCOL_PATH = "experiments/front-review-2026-11-23-r24.json"
LEDGER = "results/review/front-review-20261123-r24"
REQUESTED = {"new_000010": (15, 17, 18, 19), "new_000012": (4, 6, 7, 8, 9),
             "new_000014": (21, 22, 23, 24, 25, 31, 32),
             "new_000016": (18, 19, 25, 26, 27)}
ADJACENT = (("new_000012", 5),)
STANDARDS = (("new_000010", 16), ("new_000016", 24))


def descriptor(root, path, payload=None):
    return {"path": path, "sha256": sha(root / path) if payload is None else digest(payload)}


def expected_review(root):
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda record: (record["split"], record["id"]))
    evidence = descriptor(root, f"{LEDGER}/point-review.json")
    audit = json.loads(bound(root, evidence).read_text())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION,
            "Corner review does not identify the frozen r23 source")
    requested = {(image_id, row) for image_id, rows in REQUESTED.items() for row in rows}
    reviewed_rows = requested | set(ADJACENT)
    require(len(requested) == 21 and len(audit["rows"]) == len(reviewed_rows) == 22 and
            {(item["image_id"], item["row"]) for item in audit["rows"]} == reviewed_rows,
            "Review both endpoints of all 21 named rows and the single declared adjacent row")
    require([(item["image_id"], item["row"]) for item in audit["preserved_standards"]] == list(STANDARDS),
            "Accepted standard inventory differs")
    before = {}
    for item in parent["files"]:
        image_id = item["image_id"]
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(sha(path) == item["label_sha256"], f"r23 label drift: {image_id}")
        before[image_id] = path.read_bytes()
    require(set(before) == set(by_id), "r23 image inventory differs")
    operations, point_changes, split_changes = [], Counter(), Counter()
    for item in audit["rows"]:
        image_id, row = item["image_id"], item["row"]
        require(item["scope"] == ("user_named" if (image_id, row) in requested else "adjacent_context"),
                "Named/adjacent scope attribution differs")
        record = by_id[image_id]
        require(item["split"] == record["split"] and record["original_size"] == [1280, 720],
                "The requested points must retain their original image/split")
        require(sha(root / "data" / record["image"]) == item["source_image_sha256"] == record["image_sha256"],
                f"Original pixels differ: {image_id}")
        old_line = before[image_id].decode().splitlines()[row - 1]
        require(item["before_line"] == old_line, f"r23 row identity differs: {image_id}:{row}")
        old, new = old_line.split(), item["after_line"].split()
        require(len(old) == len(new) == 11 and old[:5] == new[:5] and
                old[7] == new[7] == old[10] == new[10] == "2",
                f"Only visible corner XY may change: {image_id}:{row}")
        require(set(item["point_reviews"]) == {"left", "right"}, "Both endpoint decisions are required")
        groups = []
        for side, offset in (("left", 5), ("right", 8)):
            review = item["point_reviews"][side]
            changed = old[offset:offset + 2] != new[offset:offset + 2]
            require(review["decision"] == ("corrected" if changed else "retained"),
                    f"Decision differs from saved coordinates: {image_id}:{row}:{side}")
            require(all(isinstance(review.get(key), str) and review[key].strip()
                        for key in ("reason", "independent_review")), "Missing visual decision")
            for name, tokens in (("before_pixel_xy", old), ("after_pixel_xy", new)):
                xy = review[name]
                require(len(xy) == 2 and all(abs(float(tokens[offset + axis]) * scale - xy[axis]) < 1e-6
                        for axis, scale in enumerate((1280, 720))),
                        f"Pixel decision differs from serialized point: {image_id}:{row}:{side}")
            if changed:
                groups.append(side)
                point_changes[side] += 1
        require(float(new[5]) < float(new[8]), "Image-X point ordering changed")
        check_geometry(parse_pose_labels(item["after_line"], 1280, 720), 1280, 720, image_id)
        if groups:
            split_changes[record["split"]] += 1
            operations.append({"image_id": image_id, "split": record["split"], "base_row": row,
                               "action": "replace", "before_line": old_line, "after_line": item["after_line"],
                               "changed_groups": groups,
                               "reason": " ".join(item["point_reviews"][side]["reason"] for side in groups)})
        else:
            require(old_line == item["after_line"], "Retained row must be byte-identical")
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
                      "missing_corners": statistics["missing_endpoints"], "changed_in_r24": bool(edits)})
    for item in audit["preserved_standards"]:
        image_id, row = item["image_id"], item["row"]
        path = f"data/reviews/{REVIEW_ID}/{by_id[image_id]['label']}"
        require(payloads[path].splitlines()[row - 1] == before[image_id].splitlines()[row - 1]
                == item["line"].encode(), f"Accepted standard changed: {image_id}:{row}")
    review_counts = totals(records, parsed)
    require(review_counts == parent["expected_counts"], "Point-only amendment changed counts")
    changed_images = sorted({op["image_id"] for op in operations})
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT,
               "named_rows_reviewed": len(requested), "adjacent_rows_reviewed": len(ADJACENT),
               "endpoints_reviewed": 2 * len(reviewed_rows),
               "changed_rows": len(operations), "changed_photos": len(changed_images),
               "changed_image_ids": changed_images, "changed_box_rows": 0,
               "point_repairs": sum(point_changes.values()), "point_repairs_by_side": dict(sorted(point_changes.items())),
               "changed_rows_by_split": dict(sorted(split_changes.items())),
               "unchanged_review_files": len(records) - len(changed_images),
               "confirmed_reference_rows_preserved": True, "instance_counts_preserved": True,
               "classes_boxes_visibility_preserved": True,
               "development_review_labels_changed": bool(split_changes["val"]),
               "training_validation_preserved": True, "expected_review_counts": review_counts}
    policy = descriptor(root, "docs/annotation-boundaries-r24.md")
    feedback = descriptor(root, f"{LEDGER}/user-feedback.txt")
    coverage = {"review_id": REVIEW_ID, "source_review": PARENT,
                "requested_rows": [{"image_id": i, "rows": list(rows)} for i, rows in REQUESTED.items()],
                "additional_adjacent_rows": [{"image_id": i, "row": row} for i, row in ADJACENT],
                "endpoint_decisions": evidence,
                "preserved_standards": [{"image_id": i, "row": r} for i, r in STANDARDS],
                "scope": "Both endpoints of 21 named rows and one adjacent row (012:5), plus two accepted standards.",
                "full_dataset_re_review": False}
    documents = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
                 "review_status": ("review-status.json", status),
                 "request_coverage": ("request-coverage.json", coverage),
                 "amendment_summary": ("amendment-summary.json", summary)}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "visually_reviewed_front_face_landmarks", "date": "2026-11-23", "parent": PARENT,
                "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
                "full_dataset_certified": False, "annotation_policy": policy, "boundary_rubric": policy,
                "review_evidence": [evidence], "user_references": [feedback],
                "inherited_row_maps": parent["row_maps"], "user_approval": parent["user_approval"],
                "requested_image_ids": list(REQUESTED), "changed_image_ids": changed_images,
                "pdf_image_ids": changed_images,
                "excluded_training_image_ids": parent["excluded_training_image_ids"],
                "additional_serialization_repairs": [], "operation_counts": {"replace": len(operations)},
                "field_group_changes": dict(sorted(point_changes.items())),
                "expected_counts": review_counts, "files": files}
    for key, (name, value) in documents.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(root, relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def experiment(root):
    review = json.loads((root / REVIEW_PATH).read_text())
    return FrontReviewExperiment(review=descriptor(root, REVIEW_PATH), parent=PARENT,
        candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH, experiment_id="front-review-r24",
        source_revision=SOURCE_REVISION, round_number=24, experiment_version=23, serialization_repairs=(),
        review_development_labels=True, excluded_training_image_ids=tuple(review["excluded_training_image_ids"]),
        protocol_limits=("Use the actual lower-front face corners. Shadow and brightness support the local geometry; they are not a fixed threshold or an offset. Both endpoints of 21 named r23 rows and adjacent row 012:5 were reviewed; two accepted standards are preserved.",
                         "Photo 010 is development-only. Its new point decisions enter only the separately identified r24 exploratory review; the original trainer validation and frozen scoring references are unchanged. Photo 015 and 31 old-series photos retain exact PR35 training supervision in both arms.",
                         "The accumulated candidate is compared with PR35 supervision; this does not isolate the latest point edits. PR97 did not support promotion on the frozen references. Keep the recipe, operating point and final-epoch rule fixed; no training or promotion during preparation."))


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"r24 artifact differs from the saved corner decisions: {relative}")
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(summary)
    receipt["excluded_training_image_ids"] = list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"] = True
    return receipt


def build(root):
    root = Path(root).resolve()
    payloads, _ = expected_review(root)
    require(all(not (root / path).exists() for path in payloads), "Preserve existing r24 artifacts; verify instead")
    require(not (root / f"data/candidates/{CANDIDATE_ID}").exists() and not (root / PROTOCOL_PATH).exists(),
            "The r24 candidate already exists")
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
