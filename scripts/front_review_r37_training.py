#!/usr/bin/env python3
"""Build the fixed R36-supervision versus R37-supervision training experiment."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_r36_training import verify_candidate as verify_r36_candidate
from front_review_r27 import pixel_geometry, semantic_fields
from verify_incremental_pose_candidate import effective_overlay
from verify_pose_candidate import check_geometry, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree

SOURCE_REVISION = "a0db36122c4c03a7fa3a9115525113dc2a21636b"
REVIEW_PATH = "data/reviews/fronts-2026-11-26-r37/manifest.json"
REVIEW_SHA256 = "35a2c6b3ddc1691eaae4b7e416144b8f68f0ab960cbd3ca8b65edd647b619471"
CANDIDATE_ID = "front-review-2026-11-26-r37"
CANDIDATE_PATH = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = "experiments/front-review-2026-11-26-r37.json"
DELTA_PATH = "results/review/front-review-20261126-r37/training-input-delta.json"
CONTROL = {
    "manifest": "data/candidates/front-review-2026-11-26-r36/manifest.json",
    "sha256": "3ea3c07c9735cba6b407d702ca20f00877ac39069eb0f7f81995de48597b90cb",
}
BASE_PROTOCOL = {
    "path": "experiments/front-review-2026-11-26-r36.json",
    "sha256": "aa0f50570ac06b8b9da95938955ee53413f25087e53f53779fb3fc5f317c970b",
}
PARENT_REVIEW = {"path": "data/reviews/fronts-2026-11-26-r36/manifest.json",
                 "sha256": "d0cdb9d56c043175ca12e538951cb28b1a8cd897039cce7a971e8f046407d416"}
ALLOWED_FIELDS = {("new_000006", 53): ["xmax"],
                  ("new_000034", 35): ["left.x", "left.y"]}


def changed_groups(before, after, image_id, row):
    a, b = before.split(), after.split()
    require(len(a) == len(b) == 11 and a[0] == b[0], "R37 class or row schema differs")
    for start in (5, 8):
        require(a[start+2] == b[start+2], f"R37 visibility changed: {image_id}:{row}")
        if int(float(a[start+2])) == 0:
            require(a[start:start+3] == b[start:start+3], "Unavailable-point tokens changed")
    groups = [name for name, start, end in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
              if a[start:end] != b[start:end]]
    if before != after:
        key = (image_id, row)
        require(key in ALLOWED_FIELDS, f"Unrequested R37 training change: {image_id}:{row}")
        fields = semantic_fields(pixel_geometry(before, str(key)), pixel_geometry(after, str(key)))
        require(fields == ALLOWED_FIELDS[key], f"R37 changed fields differ: {image_id}:{row}")
    return groups


def pending_fields_preserved(before, after, fields, size, source, preserve_row=False):
    require(isinstance(fields, list) and fields and len(fields) == len(set(fields)),
            "Pending fields must be explicit and unique")
    if preserve_row:
        require(before == after, f"Fully held row changed: {source}")
    def values(line):
        objects = parse_pose_labels(line, *size, source)
        require(len(objects) == 1, "A pending request must name one retained row")
        obj = objects[0]
        result = {name: (value,) for name, value in
                  zip(("xmin", "ymin", "xmax", "ymax"), obj["bbox_xyxy"])}
        result["box"] = tuple(obj["bbox_xyxy"])
        for side, point in zip(("left", "right"), obj["keypoints"]):
            result[side] = tuple(point)
            result.update({side+"."+name: (value,) for name, value in zip(("x", "y", "visibility"), point)})
        return result
    a, b = values(before), values(after)
    for field in fields:
        require(field in a, f"Unsupported pending field: {field}")
        require(all(abs(x-y) <= 1e-6 for x, y in zip(a[field], b[field])),
                f"Pending field changed: {source} {field}")


def pending_requests(root, review, by_id):
    audit_descriptors = [item for item in review["review_evidence"]
                         if Path(item["path"]).name == "row-review.json"]
    require(len(audit_descriptors) == 1, "R37 requires one hash-bound row-review audit")
    audit = json.loads(bound(root, audit_descriptors[0]).read_bytes())
    require(isinstance(audit.get("held_request_rows"), list), "Pending request inventory is missing")
    rows = {(row["image_id"], row["row"]): row for row in audit["rows"]}
    pending, keys = [], set()
    for item in audit["held_request_rows"]:
        key = (item["image_id"], item["row"])
        require(key in rows and key not in keys and item["image_id"] in by_id and item["reason"],
                "Pending request identity is duplicate, unknown or unexplained")
        keys.add(key)
        row, record = rows[key], by_id[item["image_id"]]
        require(row["disposition"] in ("partially_verified", "pending")
                and row["pending_fields"] == item["pending_fields"],
                "Pending fields differ from the accepted audit")
        preserve = item.get("preserve_row", False)
        require(isinstance(preserve, bool) and (row["disposition"] != "pending" or preserve),
                "A fully pending row must preserve its exact R36 annotation")
        pending_fields_preserved(row["before_line"], row["after_line"], item["pending_fields"],
                                 record["original_size"], f"{key[0]}:{key[1]}", preserve)
        pending.append({**copy.deepcopy(item), "split": record["split"]})
    require(keys == {key for key, row in rows.items()
                     if row.get("disposition") in ("partially_verified", "pending")},
            "Pending request inventory omits an unresolved audit row")
    return pending


def compose(root):
    from front_review_r37 import verify_review
    root = Path(root).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", REVIEW_SHA256), "Pin the accepted R37 review before building")
    review_receipt = verify_review(root)
    review_descriptor = {"path": REVIEW_PATH, "sha256": REVIEW_SHA256}
    review_path = bound(root, review_descriptor)
    review = json.loads(review_path.read_bytes())
    require(review["review_id"] == "fronts-2026-11-26-r37" and review["full_dataset_certified"] is False,
            "Unexpected R37 review identity or certification")
    predecessor_receipt = verify_r36_candidate(root, root / CONTROL["manifest"])
    control_path = bound(root, {"path": CONTROL["manifest"], "sha256": CONTROL["sha256"]})
    control_manifest = json.loads(control_path.read_bytes())
    base_protocol = json.loads(bound(root, BASE_PROTOCOL).read_bytes())
    _, records, by_id, source_bytes, gold, original_counts = read_reference(root, review["source_manifest_sha256"])
    pending = pending_requests(root, review, by_id)
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    control = effective_overlay(root, control_path, records, source_bytes)
    held = sorted(control_manifest["retained_legacy_image_ids"])
    excluded = sorted(control_manifest["excluded_training_image_ids"])
    require(held == ["new_000015"] and len(excluded) == 31
            and excluded == sorted(review["excluded_training_image_ids"]),
            "R36/R37 held or excluded training photos differ")
    status = json.loads(bound(root, review["review_status"]).read_bytes())
    unresolved = sorted(case["case_id"] for case in status["cases"]
                        if case["split"] == "train" and case["status"] != "resolved")
    require(unresolved == sorted(control_manifest["unresolved_case_ids"]),
            "The new experiment cannot silently change unresolved training scope")
    files = {item["image_id"]: item for item in review["files"]}
    require(len(files) == 157 and set(files) == set(by_id), "Exact original photo inventory required")
    parent_review = json.loads(bound(root, PARENT_REVIEW).read_bytes())
    parent_files = {item["image_id"]: item for item in parent_review["files"]}
    require(all(files[r["id"]]["label_sha256"] == parent_files[r["id"]]["label_sha256"]
                for r in records if r["split"] == "val"), "R37 development labels must remain identical to R36")
    control_receipt = {
        "manifest": CONTROL["manifest"], "manifest_sha256": CONTROL["sha256"],
        "candidate_id": control_manifest["candidate_id"], "kind": control_manifest["kind"],
        "changed_train_labels": len(control_manifest["changes"]),
        "counts": predecessor_receipt["counts"], "supervision": "r36_predecessor_control",
    }
    payloads, changes, sources, parsed, delta_rows, changed_images = {}, [], [], {}, [], []
    field_counts = Counter()
    for record in records:
        image_id, split = record["id"], record["split"]
        if split == "val":
            require(control[image_id]["bytes"] == source_bytes[image_id], "R36 trainer validation differs")
            parsed[image_id] = gold[image_id]
            continue
        item = files[image_id]
        require(item["split"] == split and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "R37 label/image mapping differs")
        reviewed_path = bound(review_path.parent, {"path": item["label_path"], "sha256": item["label_sha256"]})
        reviewed = {"path": reviewed_path.relative_to(root).as_posix(),
                    "sha256": item["label_sha256"], "bytes": reviewed_path.read_bytes()}
        selected = control[image_id] if image_id in held or image_id in excluded else reviewed
        payload = selected["bytes"]
        objects = parse_pose_labels(payload.decode(), *record["original_size"], selected["path"])
        check_geometry(objects, *record["original_size"], selected["path"])
        parsed[image_id] = objects
        old_lines, new_lines = control[image_id]["bytes"].decode().splitlines(), payload.decode().splitlines()
        require(len(old_lines) == len(new_lines), "This experiment permits coordinate replacements only")
        for row, (before, after) in enumerate(zip(old_lines, new_lines), 1):
            groups = changed_groups(before, after, image_id, row)
            if before != after:
                require(groups and image_id not in held and image_id not in excluded,
                        "Changed row is outside the eligible training treatment")
                delta_rows.append({"image_id": image_id, "row": row, "split": "train",
                                   "before_line": before, "after_line": after,
                                   "changed_groups": groups,
                                   "source_image_sha256": record["image_sha256"],
                                   "reference_label_sha256": control[image_id]["sha256"],
                                   "candidate_label_sha256": digest(payload)})
                field_counts.update(groups)
        if payload != control[image_id]["bytes"]:
            changed_images.append(image_id)
        sources.append({
            "image_id": image_id, "source_image_sha256": record["image_sha256"],
            "selection": "r36_retained_held" if image_id in held else (
                "r36_excluded_from_treatment" if image_id in excluded else "accepted_r37_review"),
            "effective_label": {key: selected[key] for key in ("path", "sha256")},
            "control_label": {key: control[image_id][key] for key in ("path", "sha256")},
            "reviewed_label": {key: reviewed[key] for key in ("path", "sha256")},
        })
        if payload != source_bytes[image_id]:
            relative = record["label"]
            payloads[relative] = payload
            changes.append({"image_id": image_id, "source_label": relative, "candidate_label": relative,
                            "source_label_sha256": record["label_sha256"], "candidate_label_sha256": digest(payload),
                            "source_image_sha256": record["image_sha256"], "source_objects": len(gold[image_id]),
                            "candidate_objects": len(objects), "selected_label_source": selected["path"],
                            "selected_label_source_sha256": selected["sha256"]})
    counts = totals(records, parsed)
    require(counts["val"] == original_counts["val"] == control_receipt["counts"]["val"]
            and counts["train"]["images"] == 137 and counts["train"]["empty_images"] == 13
            and counts["train"]["objects"] == control_receipt["counts"]["train"]["objects"]
            and counts["val"]["objects"] == 664,
            "R36/R37 cohort, object inventory or original trainer validation changed")
    require(counts == control_receipt["counts"], "R37 training visibility totals changed")
    require(delta_rows, "Do not prepare an identical-label training comparison")
    require({(row["image_id"], row["row"]) for row in delta_rows} == set(ALLOWED_FIELDS)
            and dict(field_counts) == {"box": 1, "left": 1},
            "R37 requires the exact two marked training corrections")
    delta = {
        "schema_version": 1, "kind": "r36_to_r37_eligible_training_input_delta",
        "reference": {"path": CONTROL["manifest"], "sha256": CONTROL["sha256"]},
        "candidate_id": CANDIDATE_ID, "accepted_review": review_descriptor,
        "rows": delta_rows, "changed_image_ids": changed_images,
        "held_request_rows": pending,
        "summary": {"changed_rows": len(delta_rows), "changed_photos": len(changed_images),
                    "field_group_changes": dict(sorted(field_counts.items())),
                    "added_rows": 0, "deleted_rows": 0, "classes_preserved": True, "visibility_changes": 0,
                    "classes_visibility_preserved": True},
        "retained_legacy_image_ids": held, "excluded_training_image_ids": excluded,
        "reference_counts": control_receipt["counts"], "candidate_counts": counts,
        "interpretation": "Only these eligible training-coordinate replacements differ between the two fresh arms. R37 development amendments are excluded from trainer supervision. Both arms independently start from original epoch 173 under the same fixed recipe and recorded runtime.",
    }
    delta_descriptor = {"path": DELTA_PATH, "sha256": digest(encoded(delta))}
    manifest = {
        "schema_version": 1, "kind": "training_incremental_observed_overlay",
        "candidate_id": CANDIDATE_ID, "status": "controlled_incremental_only", "experiment_version": 37,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": review["source_manifest_sha256"],
        "source_candidate": review_descriptor, "source_review_status": review["review_status"],
        "control": CONTROL, "training_protocol_path": PROTOCOL_PATH,
        "training_input_delta": delta_descriptor,
        "held_request_rows": pending,
        "full_dataset_certified": False, "full_source_training_readiness": "review_only",
        "task_scope": control_manifest["task_scope"],
        "scope": "Compare verified R36 supervision with the eligible R37 review. Retain exact R36 labels on held photo 015 and the 31 excluded old-series photos in both arms. All original photos, negative scenes and trainer-validation labels are unchanged.",
        "unresolved_case_ids": unresolved, "retained_legacy_image_ids": held,
        "excluded_training_image_ids": excluded, "image_sources": sources,
        "changed_images_vs_control": changed_images, "expected_counts": counts, "changes": changes,
    }
    payloads["manifest.json"] = encoded(manifest)
    protocol = copy.deepcopy(base_protocol)
    protocol.update({
        "kind": "baomai-r37-fronts-training-protocol", "experiment_id": "front-review-r37",
        "candidate": {"path": CANDIDATE_PATH, "sha256": digest(payloads["manifest.json"])},
        "source_candidate": review_descriptor, "source_review_status": review["review_status"],
        "control": CONTROL, "training_input_delta": delta_descriptor,
        "held_request_rows": pending,
        "request_completion": review["status"],
        "current_marked_requests_completed": True,
        "development_labels_identical_to_r36": True,
        "changed_images_vs_control": changed_images, "complete_review_counts": review["expected_counts"],
        "unresolved_case_ids": unresolved, "retained_legacy_image_ids": held,
        "excluded_training_image_ids": excluded,
        "limits": [
            "This is a regression experiment for two user-marked annotation changes: new_000006:53 xmax and new_000034:35 left-corner coordinates. The reference uses eligible R36 supervision and the candidate uses eligible R37 supervision. Both independently start from original epoch 173 under the same recipe. No general cure for missed fragments is promised.",
            "All 20 development label files remain byte-identical to R36. Original trainer validation remains 664 fronts; frozen 666/663 references are unchanged. The current review remains exploratory evidence. Object, class and corner-availability inventories are unchanged. The inherited new_000010:24 point request remains pending and its entire row is preserved. Earlier R35 development deletions remain outside both training arms and original trainer validation.",
            "Use L4 or A100. Both fresh arms must have the same recorded runtime. Preserve seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and final-epoch selection. The unmodified native Pose26 loss is used in both arms. PR113 failed the prespecified clean-improvement screen; partial-box factor2 is discontinued. No threshold/seed/epoch search and no automatic promotion.",
        ],
    })
    for arm in ("reference", "candidate"):
        protocol["roles"][arm].update({
            "supervision": "r36_predecessor_control" if arm == "reference" else "accepted_fronts_treatment",
            "display_name": "R36 supervision" if arm == "reference" else "R37 supervision",
            "candidate_manifest": CANDIDATE_PATH,
            "candidate_manifest_sha256": protocol["candidate"]["sha256"], "candidate_id": CANDIDATE_ID,
            "counts": control_receipt["counts"] if arm == "reference" else counts,
        })
    protocol["execution"].update({"recommended_gpu": "NVIDIA L4 or A100", "historical_runtime_match_required": False,
                                  "same_runtime_required": True, "expected_batches": 1050,
                                  "expected_images": 4110, "expected_optimizer_steps": 78,
                                  "loss": "unmodified_native_Pose26", "partial_box_factor": 1})
    protocol["created_date"] = "2026-11-26"
    protocol["timezone"] = "Asia/Hong_Kong"
    protocol["hypothesis"] = "What detection and corner-localization regressions or gains follow from the two marked R37 annotation corrections under the unchanged native seed46 recipe?"
    protocol["prediction_semantics"] = "Hidden-corner estimates remain allowed at inference. A missing annotation is unscorable; it is not proof that a predicted point is physically visible or a false positive."
    protocol["prior_decision"] = copy.deepcopy(base_protocol["prior_decision"])
    protocol["prior_decision"].update({
        "immediate_predecessor_result_pr": "https://github.com/Siuuuers/dl/pull/121",
        "r37_design": "Two marked annotation corrections only: eligible R36 versus R37 supervision. No model, initializer, native loss, budget, sampling or primary inference change.",
        "pr121_r36_reference_control_candidate": {
            "misses": [15, 15], "unmatched": [93, 94], "duplicates": [31, 31],
            "pair5": [570, 570], "pair10": [582, 582], "pairs": 583,
        },
        "pr121_frozen666_control_candidate": {
            "misses": [36, 36], "unmatched": [107, 108], "duplicates": [29, 29],
            "pair5": [517, 518], "pair10": [573, 573], "pairs": 584,
        },
        "pr121_interpretation": "The R36 candidate preserves current-review misses, duplicates and both-corner successes but adds one unmatched output. Frozen 666 gains one 5-pixel pair and adds one unmatched output. The fresh control reproduces PR119 candidate predictions exactly. No model is promoted.",
        "pr121_same_checkpoint_one2one_probe": {
            "scope": "Exploratory candidate-only native head comparison at unchanged confidence 0.13",
            "r36_misses": [15, 16], "r36_unmatched": [94, 67], "r36_duplicates": [31, 37],
            "r36_pair5": [570, 572], "r36_pair10": [582, 581], "pairs": 583,
            "invalid_output_count": 1, "invalid_output_counted_as_unmatched": True,
            "decision": "Keep one-to-many plus NMS. The alternative loses recall, increases duplicates and emits a zero-area box despite reducing total unmatched outputs.",
        },
    })
    protocol["evaluation"]["current_review_reference"] = review_descriptor
    protocol["evaluation"]["current_review_counts"] = review["expected_counts"]["val"]
    protocol["evaluation"]["preserved_r36_reference"] = PARENT_REVIEW
    protocol["evaluation"]["decision"] = (
        "Compare the fresh R36-supervision and R37-supervision arms and retained PR35 on unchanged frozen 666-front primary and 663-front historical references. Report misses, unmatched outputs, duplicates, empty scenes, zero/one/two-locatable-corner groups and both-corner successes at 5/10 original-image pixels with eligible misses retained in denominators. Report the separately identified current R37 review as exploratory only. Inspect gained/lost identities and the exact training delta. Conflicting reference rankings remain inconclusive; no automatic promotion or new-session reliability claim follows from this run.")
    return payloads, manifest, protocol, delta, control_receipt, review_receipt


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    candidate = (root / CANDIDATE_PATH) if candidate is None else Path(candidate).resolve()
    require(candidate == root / CANDIDATE_PATH, "R37 candidate path differs")
    payloads, manifest, protocol, delta, control, review_receipt = compose(root)
    require(candidate.read_bytes() == encoded(manifest), "R37 candidate differs from exact eligible composition")
    verify_tree(candidate.parent, payloads)
    require((root / PROTOCOL_PATH).read_bytes() == encoded(protocol), "R37 protocol differs from fixed paired design")
    require((root / DELTA_PATH).read_bytes() == encoded(delta), "R36-to-R37 training input delta differs")
    return {
        "status": "verified", "candidate_id": CANDIDATE_ID, "candidate_manifest_sha256": sha(candidate),
        "source_manifest_sha256": manifest["source_manifest_sha256"], "declared_source_revision": SOURCE_REVISION,
        "counts": manifest["expected_counts"], "changed_train_labels": len(manifest["changes"]),
        "removed_objects": sum(x["source_objects"] - x["candidate_objects"] for x in manifest["changes"]),
        "control": control, "training_protocol": {"path": PROTOCOL_PATH, "sha256": sha(root / PROTOCOL_PATH)},
        "training_input_delta": {**manifest["training_input_delta"], "report": delta},
        "training_readiness": protocol["training_readiness"], "full_dataset_certified": False,
        "unresolved_case_ids": manifest["unresolved_case_ids"],
        "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
        "excluded_training_image_ids": manifest["excluded_training_image_ids"],
        "changed_images_vs_control": manifest["changed_images_vs_control"],
        "unresolved_reference_rows": protocol["evaluation"]["unresolved_reference_rows"],
        "diagnostic_unresolved_reference_rows": protocol["evaluation"]["diagnostic_unresolved_reference_rows"],
        "retained_photo_labels_identical_between_arms": True,
        "excluded_photo_labels_identical_between_arms": True,
        "training_changes_vs_r36": delta["summary"], "review_verification": review_receipt,
        "training_run": False,
    }


def build(root):
    root = Path(root).resolve()
    payloads, manifest, protocol, delta, _, _ = compose(root)
    candidate = root / CANDIDATE_PATH
    require(not candidate.parent.exists() and not (root / PROTOCOL_PATH).exists()
            and not (root / DELTA_PATH).exists(), "Preserve existing R37 experiment artifacts; verify instead")
    for relative, payload in payloads.items():
        path = candidate.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for relative, value in ((PROTOCOL_PATH, protocol), (DELTA_PATH, delta)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(value))
    return verify_candidate(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = build(args.root) if args.build else verify_candidate(args.root)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(encoded(result))
    print(json.dumps({key: result[key] for key in ("status", "candidate_id", "candidate_manifest_sha256",
                     "control", "counts", "training_changes_vs_r36", "training_run")}, indent=2))
