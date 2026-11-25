#!/usr/bin/env python3
"""Build the fixed R34-supervision versus R35-supervision training experiment."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_r34_training import verify_candidate as verify_r34_candidate
from verify_incremental_pose_candidate import effective_overlay
from verify_pose_candidate import check_geometry, read_reference, require, totals
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree

SOURCE_REVISION = "9b69f1f4fd4e6dc2ad1ca1f41d7667fa9504151a"
REVIEW_PATH = "data/reviews/fronts-2026-11-25-r35/manifest.json"
REVIEW_SHA256 = "1ec7e4826d3e14c61d83e5267fdb798255348858c834e64ec2d66347076f9019"
CANDIDATE_ID = "front-review-2026-11-25-r35"
CANDIDATE_PATH = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = "experiments/front-review-2026-11-25-r35.json"
DELTA_PATH = "results/review/front-review-20261125-r35/training-input-delta.json"
CONTROL = {
    "manifest": "data/candidates/front-review-2026-11-25-r34/manifest.json",
    "sha256": "e1af61c80d86a15dce445c546d7c4b2b7b7f8c235e6ca03e9e45eea530b9318d",
}
BASE_PROTOCOL = {
    "path": "experiments/front-review-2026-11-25-r34.json",
    "sha256": "05a23cbef0fab6a2b5be1e1b7c8aafc4a261ab32f13cde26d13d68bbff3e3277",
}


def changed_groups(before, after, image_id, row):
    a, b = before.split(), after.split()
    require(len(a) == len(b) == 11 and a[0] == b[0], "R35 class or row schema differs")
    for start in (5, 8):
        require(a[start+2] == b[start+2], f"R35 visibility changed: {image_id}:{row}")
        if int(float(a[start+2])) == 0:
            require(a[start:start+3] == b[start:start+3], "Unavailable-point tokens changed")
    return [name for name, start, end in (("box", 1, 5), ("left", 5, 7), ("right", 8, 10))
            if a[start:end] != b[start:end]]


def compose(root):
    from front_review_r35 import verify_review
    root = Path(root).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", REVIEW_SHA256), "Pin the accepted R35 review before building")
    review_receipt = verify_review(root)
    review_descriptor = {"path": REVIEW_PATH, "sha256": REVIEW_SHA256}
    review_path = bound(root, review_descriptor)
    review = json.loads(review_path.read_bytes())
    require(review["review_id"] == "fronts-2026-11-25-r35" and review["full_dataset_certified"] is False,
            "Unexpected R35 review identity or certification")
    predecessor_receipt = verify_r34_candidate(root, root / CONTROL["manifest"])
    control_path = bound(root, {"path": CONTROL["manifest"], "sha256": CONTROL["sha256"]})
    control_manifest = json.loads(control_path.read_bytes())
    base_protocol = json.loads(bound(root, BASE_PROTOCOL).read_bytes())
    _, records, by_id, source_bytes, gold, original_counts = read_reference(root, review["source_manifest_sha256"])
    records = sorted(records, key=lambda r: (r["split"], r["id"]))
    control = effective_overlay(root, control_path, records, source_bytes)
    held = sorted(control_manifest["retained_legacy_image_ids"])
    excluded = sorted(control_manifest["excluded_training_image_ids"])
    require(held == ["new_000015"] and len(excluded) == 31
            and excluded == sorted(review["excluded_training_image_ids"]),
            "R34/R35 held or excluded training photos differ")
    status = json.loads(bound(root, review["review_status"]).read_bytes())
    unresolved = sorted(case["case_id"] for case in status["cases"]
                        if case["split"] == "train" and case["status"] != "resolved")
    require(unresolved == sorted(control_manifest["unresolved_case_ids"]),
            "The new experiment cannot silently change unresolved training scope")
    files = {item["image_id"]: item for item in review["files"]}
    require(len(files) == 157 and set(files) == set(by_id), "Exact original photo inventory required")
    control_receipt = {
        "manifest": CONTROL["manifest"], "manifest_sha256": CONTROL["sha256"],
        "candidate_id": control_manifest["candidate_id"], "kind": control_manifest["kind"],
        "changed_train_labels": len(control_manifest["changes"]),
        "counts": predecessor_receipt["counts"], "supervision": "r34_predecessor_control",
    }
    payloads, changes, sources, parsed, delta_rows, changed_images = {}, [], [], {}, [], []
    field_counts = Counter()
    for record in records:
        image_id, split = record["id"], record["split"]
        if split == "val":
            require(control[image_id]["bytes"] == source_bytes[image_id], "R34 trainer validation differs")
            parsed[image_id] = gold[image_id]
            continue
        item = files[image_id]
        require(item["split"] == split and item["label_path"] == record["label"]
                and item["source_image_sha256"] == record["image_sha256"], "R35 label/image mapping differs")
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
            "selection": "r34_retained_held" if image_id in held else (
                "r34_excluded_from_treatment" if image_id in excluded else "accepted_r35_review"),
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
            "R34/R35 cohort, object inventory or original trainer validation changed")
    require(counts == control_receipt["counts"], "R35 training visibility totals changed")
    require(delta_rows, "Do not prepare an identical-label training comparison")
    delta = {
        "schema_version": 1, "kind": "r34_to_r35_eligible_training_input_delta",
        "reference": {"path": CONTROL["manifest"], "sha256": CONTROL["sha256"]},
        "candidate_id": CANDIDATE_ID, "accepted_review": review_descriptor,
        "rows": delta_rows, "changed_image_ids": changed_images,
        "summary": {"changed_rows": len(delta_rows), "changed_photos": len(changed_images),
                    "field_group_changes": dict(sorted(field_counts.items())),
                    "added_rows": 0, "deleted_rows": 0, "classes_preserved": True, "visibility_changes": 0,
                    "classes_visibility_preserved": True},
        "retained_legacy_image_ids": held, "excluded_training_image_ids": excluded,
        "reference_counts": control_receipt["counts"], "candidate_counts": counts,
        "interpretation": "Only these eligible training-coordinate replacements differ between the two fresh arms. R35 development amendments are excluded from trainer supervision. Both arms independently start from original epoch 173 under the same fixed recipe and recorded runtime.",
    }
    delta_descriptor = {"path": DELTA_PATH, "sha256": digest(encoded(delta))}
    manifest = {
        "schema_version": 1, "kind": "training_incremental_observed_overlay",
        "candidate_id": CANDIDATE_ID, "status": "controlled_incremental_only", "experiment_version": 35,
        "source_revision": SOURCE_REVISION, "source_manifest_sha256": review["source_manifest_sha256"],
        "source_candidate": review_descriptor, "source_review_status": review["review_status"],
        "control": CONTROL, "training_protocol_path": PROTOCOL_PATH,
        "training_input_delta": delta_descriptor,
        "full_dataset_certified": False, "full_source_training_readiness": "review_only",
        "task_scope": control_manifest["task_scope"],
        "scope": "Compare verified R34 supervision with the eligible R35 review. Retain exact R34 labels on held photo 015 and the 31 excluded old-series photos in both arms. All original photos, negative scenes and trainer-validation labels are unchanged.",
        "unresolved_case_ids": unresolved, "retained_legacy_image_ids": held,
        "excluded_training_image_ids": excluded, "image_sources": sources,
        "changed_images_vs_control": changed_images, "expected_counts": counts, "changes": changes,
    }
    payloads["manifest.json"] = encoded(manifest)
    protocol = copy.deepcopy(base_protocol)
    protocol.update({
        "kind": "baomai-r35-fronts-training-protocol", "experiment_id": "front-review-r35",
        "candidate": {"path": CANDIDATE_PATH, "sha256": digest(payloads["manifest.json"])},
        "source_candidate": review_descriptor, "source_review_status": review["review_status"],
        "control": CONTROL, "training_input_delta": delta_descriptor,
        "changed_images_vs_control": changed_images, "complete_review_counts": review["expected_counts"],
        "unresolved_case_ids": unresolved, "retained_legacy_image_ids": held,
        "excluded_training_image_ids": excluded,
        "limits": [
            "The reference arm uses R34 training supervision; the candidate uses eligible R35 supervision. Both are freshly trained together from the original epoch-173 initializer. The comparison isolates the exact hash-bound training-input delta, after the accepted user-named follow-up review in original pixels.",
            "Development amendments do not enter trainer supervision. Original trainer validation remains 664 fronts; frozen 666/663 references are unchanged. The current R35 review is separately identified exploratory evidence only. Training changes are original-pixel boundary/corner corrections with unchanged object, class and visibility inventories. Development-only deletions of R34 new_000009 rows 56 and 57 do not enter either training arm or trainer validation. Audit row identities remain R34 identities; current TXT positions are separately mapped.",
            "Use L4 or A100. Both fresh arms must have the same recorded runtime. Preserve seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and final-epoch selection. The unmodified native Pose26 loss is used in both arms. PR113 failed the prespecified clean-improvement screen; partial-box factor2 is discontinued. No threshold/seed/epoch search and no automatic promotion.",
        ],
    })
    for arm in ("reference", "candidate"):
        protocol["roles"][arm].update({
            "supervision": "r34_predecessor_control" if arm == "reference" else "accepted_fronts_treatment",
            "display_name": "R34 supervision" if arm == "reference" else "R35 supervision",
            "candidate_manifest": CANDIDATE_PATH,
            "candidate_manifest_sha256": protocol["candidate"]["sha256"], "candidate_id": CANDIDATE_ID,
            "counts": control_receipt["counts"] if arm == "reference" else counts,
        })
    protocol["execution"].update({"recommended_gpu": "NVIDIA L4 or A100", "historical_runtime_match_required": False,
                                  "same_runtime_required": True, "expected_batches": 1050,
                                  "expected_images": 4110, "expected_optimizer_steps": 78,
                                  "loss": "unmodified_native_Pose26", "partial_box_factor": 1})
    protocol["created_date"] = "2026-11-25"
    protocol["timezone"] = "Asia/Hong_Kong"
    protocol["hypothesis"] = "Does the exact accepted R35 training-annotation delta improve observed-front localization when both arms use the unchanged native loss and the same fixed seed46 recipe?"
    protocol["prediction_semantics"] = "Hidden-corner estimates remain allowed at inference. A missing annotation is unscorable; it is not proof that a predicted point is physically visible or a false positive."
    protocol["prior_decision"] = copy.deepcopy(base_protocol["prior_decision"])
    protocol["prior_decision"].update({
        "immediate_predecessor_result_pr": "https://github.com/Siuuuers/dl/pull/117",
        "r35_design": "Accepted R34 versus eligible R35 supervision only; native loss, original sampling, architecture and inference defaults are unchanged.",
        "pr117_r34_reference_control_candidate": {"misses": [16, 17], "unmatched": [91, 92],
            "duplicates": [30, 30], "pair5": [570, 569], "pair10": [582, 582], "pairs": 583},
        "pr117_frozen666_control_candidate": {"misses": [35, 35], "unmatched": [105, 105],
            "pair5": [516, 515], "pair10": [573, 573], "pairs": 584},
        "pr117_interpretation": "The R34-supervision candidate loses one match and one 5-pixel corner-pair success under the preserved R34 reference; frozen-666 misses are unchanged and one 5-pixel pair is lost. No model is promoted. R35 isolates the next accepted annotation delta under the same native recipe and fixed primary inference settings.",
    })
    protocol["evaluation"]["current_review_reference"] = review_descriptor
    protocol["evaluation"]["current_review_counts"] = review["expected_counts"]["val"]
    protocol["evaluation"]["preserved_r34_reference"] = {
        "path": "data/reviews/fronts-2026-11-25-r34/manifest.json",
        "sha256": "79e41ce201771959c66b7a7d5f841e97b42f27690c6e1b962e8a4c99eccfc948"}
    protocol["evaluation"]["decision"] = (
        "Compare the fresh R34-supervision and R35-supervision arms and retained PR35 on unchanged frozen 666-front primary and 663-front historical references. Report misses, unmatched outputs, duplicates, empty scenes, zero/one/two-locatable-corner groups and both-corner successes at 5/10 original-image pixels with eligible misses retained in denominators. Report the separately identified current R35 review as exploratory only. Inspect gained/lost identities and the exact training delta. Conflicting reference rankings remain inconclusive; no automatic promotion or new-session reliability claim follows from this run.")
    return payloads, manifest, protocol, delta, control_receipt, review_receipt


def verify_candidate(root, candidate=None):
    root = Path(root).resolve()
    candidate = (root / CANDIDATE_PATH) if candidate is None else Path(candidate).resolve()
    require(candidate == root / CANDIDATE_PATH, "R35 candidate path differs")
    payloads, manifest, protocol, delta, control, review_receipt = compose(root)
    require(candidate.read_bytes() == encoded(manifest), "R35 candidate differs from exact eligible composition")
    verify_tree(candidate.parent, payloads)
    require((root / PROTOCOL_PATH).read_bytes() == encoded(protocol), "R35 protocol differs from fixed paired design")
    require((root / DELTA_PATH).read_bytes() == encoded(delta), "R34-to-R35 training input delta differs")
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
        "training_changes_vs_r34": delta["summary"], "review_verification": review_receipt,
        "training_run": False,
    }


def build(root):
    root = Path(root).resolve()
    payloads, manifest, protocol, delta, _, _ = compose(root)
    candidate = root / CANDIDATE_PATH
    require(not candidate.parent.exists() and not (root / PROTOCOL_PATH).exists()
            and not (root / DELTA_PATH).exists(), "Preserve existing R35 experiment artifacts; verify instead")
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
                     "control", "counts", "training_changes_vs_r34", "training_run")}, indent=2))
