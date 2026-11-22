#!/usr/bin/env python3
"""Apply four front-face corner corrections to the completed r22 review."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import FrontReviewExperiment, apply_operations
from verify_pose_candidate import check_geometry, counts, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha

SOURCE_REVISION = "c1d68bc6ed388077f238c09b74f6a8e15a1375a5"
PARENT = {"path": "data/reviews/fronts-2026-11-22-r22/manifest.json",
          "sha256": "a2af88f2e9e6a6e94db123d2d20f6d057f59062eaae294edbdf0ad94267e13ae"}
REVIEW_ID = "fronts-2026-11-22-r23"
REVIEW_PATH = f"data/reviews/{REVIEW_ID}/manifest.json"
CANDIDATE_ID = "front-review-2026-11-22-r23"
PROTOCOL_PATH = "experiments/front-review-2026-11-22-r23.json"
LEDGER = "results/review/front-review-20261122-r23"
TARGETS = {("new_000001", 27): (488, 518), ("new_000002", 25): (426, 543),
           ("new_000002", 26): (430, 571), ("new_000002", 27): (432, 600)}
STANDARDS = (("new_000001", 1), ("new_000001", 19),
             ("new_000001", 25), ("new_000002", 28))


def descriptor(root, path, payload=None):
    return {"path": path, "sha256": sha(root / path) if payload is None else digest(payload)}


def expected_review(root):
    parent = json.loads(bound(root, PARENT).read_text())
    _, records, by_id, _, _, _ = read_reference(root, parent["source_manifest_sha256"])
    records = sorted(records, key=lambda record: (record["split"], record["id"]))
    evidence = descriptor(root, f"{LEDGER}/point-review.json")
    audit = json.loads(bound(root, evidence).read_text())
    require(audit["source_review"] == PARENT and audit["source_revision"] == SOURCE_REVISION,
            "Corner review does not identify the frozen r22 source")
    require(len(audit["changes"]) == 4 and
            {(item["image_id"], item["row"]) for item in audit["changes"]} == set(TARGETS),
            "The corner amendment must contain exactly the four requested rows")
    before = {}
    for item in parent["files"]:
        image_id = item["image_id"]
        path = (root / PARENT["path"]).parent / item["label_path"]
        require(sha(path) == item["label_sha256"], f"r22 label drift: {image_id}")
        before[image_id] = path.read_bytes()
    require(set(before) == set(by_id), "r22 image inventory differs")
    operations = []
    for item in audit["changes"]:
        image_id, row = item["image_id"], item["row"]
        record = by_id[image_id]
        require(record["split"] == "train" and record["original_size"] == [1280, 720],
                "The requested points must remain in their original training images")
        require(sha(root / "data" / record["image"]) == item["source_image_sha256"] == record["image_sha256"],
                f"Original pixels differ: {image_id}")
        old_line = before[image_id].decode().splitlines()[row - 1]
        require(item["before_line"] == old_line, f"r22 row identity differs: {image_id}:{row}")
        old, new = old_line.split(), item["after_line"].split()
        require(len(old) == len(new) == 11 and old[:8] == new[:8] and old[10] == new[10] == "2",
                f"Only a visible right point may change: {image_id}:{row}")
        x, y = TARGETS[image_id, row]
        require(abs(float(new[8]) * 1280 - x) < 1e-6 and abs(float(new[9]) * 720 - y) < 1e-6,
                f"Saved corner differs from the visual decision: {image_id}:{row}")
        require(float(new[5]) < float(new[8]), "Image-X point ordering changed")
        require(all(isinstance(item.get(key), str) and item[key].strip()
                    for key in ("reason", "independent_review")), "Missing visual decision")
        check_geometry(parse_pose_labels(item["after_line"], 1280, 720), 1280, 720, image_id)
        operations.append({"image_id": image_id, "split": "train", "base_row": row,
                           "action": "replace", "before_line": old_line, "after_line": item["after_line"],
                           "changed_groups": ["right"], "reason": item["reason"]})
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
                      "missing_corners": statistics["missing_endpoints"], "changed_in_r23": bool(edits)})
    for image_id, row in STANDARDS:
        path = f"data/reviews/{REVIEW_ID}/{by_id[image_id]['label']}"
        require(payloads[path].splitlines()[row - 1] == before[image_id].splitlines()[row - 1],
                f"Accepted standard changed: {image_id}:{row}")
    review_counts = totals(records, parsed)
    require(review_counts == parent["expected_counts"], "Point-only amendment changed counts")
    summary = {"review_id": REVIEW_ID, "parent_review": PARENT, "changed_rows": 4,
               "changed_photos": 2, "changed_image_ids": ["new_000001", "new_000002"],
               "changed_box_rows": 0, "point_repairs": 4, "unchanged_review_files": 155,
               "confirmed_reference_rows_preserved": True, "instance_counts_preserved": True,
               "classes_boxes_left_points_visibility_preserved": True,
               "development_labels_preserved": True, "expected_review_counts": review_counts}
    policy = descriptor(root, "docs/annotation-boundaries-r23.md")
    feedback = descriptor(root, f"{LEDGER}/user-feedback.txt")
    coverage = {"review_id": REVIEW_ID, "source_review": PARENT,
                "corrected_right_points": [{"image_id": i, "row": r, "pixel_xy": list(xy)}
                                            for (i, r), xy in TARGETS.items()],
                "preserved_standards": [{"image_id": i, "row": r} for i, r in STANDARDS],
                "scope": "Four named right points; nearby context and accepted standards inspected.",
                "full_dataset_re_review": False}
    documents = {"operations": ("operations.json", operations), "row_maps": ("row-maps.json", maps),
                 "review_status": ("review-status.json", status),
                 "request_coverage": ("request-coverage.json", coverage),
                 "amendment_summary": ("amendment-summary.json", summary)}
    manifest = {"schema_version": 1, "kind": "original_pixel_front_review", "review_id": REVIEW_ID,
                "status": "visually_reviewed_front_face_landmarks", "date": "2026-11-22", "parent": PARENT,
                "source_revision": SOURCE_REVISION, "source_manifest_sha256": parent["source_manifest_sha256"],
                "full_dataset_certified": False, "annotation_policy": policy, "boundary_rubric": policy,
                "review_evidence": [evidence], "user_references": [feedback],
                "inherited_row_maps": parent["row_maps"], "user_approval": parent["user_approval"],
                "requested_image_ids": summary["changed_image_ids"], "changed_image_ids": summary["changed_image_ids"],
                "pdf_image_ids": summary["changed_image_ids"],
                "excluded_training_image_ids": parent["excluded_training_image_ids"],
                "additional_serialization_repairs": [], "operation_counts": {"replace": 4},
                "field_group_changes": {"right": 4}, "expected_counts": review_counts, "files": files}
    for key, (name, value) in documents.items():
        relative = f"{LEDGER}/{name}"
        payloads[relative] = encoded(value)
        manifest[key] = descriptor(root, relative, payloads[relative])
    payloads[REVIEW_PATH] = encoded(manifest)
    return payloads, summary


def experiment(root):
    review = json.loads((root / REVIEW_PATH).read_text())
    return FrontReviewExperiment(review=descriptor(root, REVIEW_PATH), parent=PARENT,
        candidate_id=CANDIDATE_ID, protocol_path=PROTOCOL_PATH, experiment_id="front-review-r23",
        source_revision=SOURCE_REVISION, round_number=23, experiment_version=22, serialization_repairs=(),
        review_development_labels=True, excluded_training_image_ids=tuple(review["excluded_training_image_ids"]),
        protocol_limits=("Use the actual lower-front face corners, not the outer side silhouette. Four right-point corrections are applied to frozen r22; other annotation fields are unchanged.",
                         "Photo 015 and 31 old-series photos retain PR35 supervision in both training arms. Original trainer validation and fixed scoring references remain unchanged.",
                         "The accumulated candidate is compared with PR35 supervision; this does not isolate the effect of four points. No training or model promotion during preparation."))


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    payloads, summary = expected_review(root)
    for relative, payload in payloads.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"r23 artifact differs from the saved corner decisions: {relative}")
    configured = experiment(root)
    receipt = configured.verify_candidate(root, candidate)
    receipt.update(summary)
    receipt["excluded_training_image_ids"] = list(configured.excluded_training_image_ids)
    receipt["excluded_photo_labels_identical_between_arms"] = True
    return receipt


def build(root):
    root = Path(root).resolve()
    payloads, _ = expected_review(root)
    require(all(not (root / path).exists() for path in payloads), "Preserve existing r23 artifacts; verify instead")
    require(not (root / f"data/candidates/{CANDIDATE_ID}").exists() and not (root / PROTOCOL_PATH).exists(),
            "The r23 candidate already exists")
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
