#!/usr/bin/env python3
"""Verify a bounded correction experiment while retaining unresolved legacy photos.

The full observed-front composition keeps its original blocked release gate.
This separate experiment changes labels only on photos with no remaining case,
and uses the exact PR35 labels on every unresolved photo in both arms.
"""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, parse_pose_labels
from verify_observed_training_candidate import verify_observed_training_candidate
from verify_pose_candidate import check_geometry, read_reference, require, totals
from verify_pose_composed_candidate import PARENT as PR35_CONTROL
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree


KIND = "training_incremental_observed_overlay"
CANDIDATE_ID = "accepted-front-corrections-2026-11-14"
DEFAULT_CANDIDATE = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = "experiments/accepted-front-corrections-2026-11-14.json"
RECIPE_ID = "reviewed-incremental-fronts-seed46-v1"
SOURCE_REVISION = "bcf6cce55a94ecca2a859bc927dc7122382d31dc"
SOURCE_CANDIDATE = {
    "path": "data/candidates/complete-visible-fronts-2026-11-14/manifest.json",
    "sha256": "9d138930b523fc8d124e116a58aa584a23c30a81d874551fcb5625a1978639a8",
}
SOURCE_PROTOCOL = {
    "path": "experiments/observed-fronts-2026-11-14.json",
    "sha256": "a501eddce8fcf9b0361583e64b1a308a85ddef223f72d9f6788927ecba0278ef",
}
SOURCE_STATUS = {
    "path": "results/review/complete-labels-20261114/review-status.json",
    "sha256": "257d18c2cbe69471fc5a0070748747a92534cda88d5065b14520b98ea37cc761",
}
ADJUDICATION = {
    "path": "results/review/physical-adjudication-20261114/decisions.json",
    "sha256": "07f545434877c4d9d44eb90b8e1c0ae4aa1c13882a500ccd81bfd2b54f9dbbc6",
}
CLOSED_CASES = {"new_000037-100": "new_000037", "new_000017-075": "new_000017"}
PRIMARY_REFERENCE = {
    "path": "data/references/visible-front-adjudication-2026-11-12/manifest.json",
    "sha256": "75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e",
}
DIAGNOSTIC_REFERENCE = {
    "path": "data/references/visible-front-completion-2026-11-14/manifest.json",
    "sha256": "bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1",
}

# Separate v2 inputs. The v1 constants, reconstruction and default CLI selection
# remain unchanged, so its original notebook can still replay its own contract.
SEQUENCE_CANDIDATE_ID = "sequence-front-corrections-2026-11-18"
SEQUENCE_CANDIDATE = f"data/candidates/{SEQUENCE_CANDIDATE_ID}/manifest.json"
SEQUENCE_PROTOCOL_PATH = "experiments/sequence-front-corrections-2026-11-18.json"
SEQUENCE_SOURCE_REVISION = "cff28170b7fc82b859850607f06ac8d690dc5d03"
SEQUENCE_REVIEW = {
    "path": "data/candidates/sequence-refined-fronts-2026-11-18-r2/manifest.json",
    "sha256": "de3cd6e5605ec7bd1250b53b4eb52c3a688e2431a19f4016bbfa08f2e09b6849",
}
SEQUENCE_STATUS = {
    "path": "results/review/front-refinement-20261118-r2/review-status.json",
    "sha256": "b3a2a3adc5b0f589e9f64e9fa96570fea6f557be913d3d575691be32109ef54b",
}
SEQUENCE_PRIMARY_REFERENCE = {
    "path": "data/references/sequence-refined-reference-2026-11-18/manifest.json",
    "sha256": "0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96",
}


def verify_adjudication(root, adjudication_path=None):
    """Lightweight verification for the atlas and the incremental experiment.

Return the parsed additive document; never alter the frozen case ledger.
Only the two explicitly bound non-target decisions can close source cases.
"""
    root = Path(root).resolve()
    path = bound(root, ADJUDICATION)
    if adjudication_path is not None:
        selected = Path(adjudication_path)
        selected = selected if selected.is_absolute() else root / selected
        require(selected.resolve() == path, "another physical adjudication was selected")
    document = json.loads(path.read_text())
    require(document.get("schema_version") == 1 and document.get("kind") == "physical_adjudication",
            "unsupported physical adjudication")
    require(document.get("source_candidate") == SOURCE_CANDIDATE
            and document.get("source_review_status") == SOURCE_STATUS,
            "physical adjudication names different source evidence")
    source = json.loads(bound(root, SOURCE_CANDIDATE).read_text())
    status = json.loads(bound(root, SOURCE_STATUS).read_text())
    require(status["candidate"] == SOURCE_CANDIDATE, "physical adjudication source case ledger differs")
    reference_path = _relative_file(root, "data/manifest.json")
    require(sha(reference_path) == source["source_manifest_sha256"], "physical adjudication reference differs")
    records = {row["id"]: row for row in json.loads(reference_path.read_text())["records"]}
    cases = {item["case_id"]: item for item in status["cases"]}
    decisions = document.get("decisions")
    require(isinstance(decisions, list) and len(decisions) == len(CLOSED_CASES)
            and {item.get("case_id") for item in decisions} == set(CLOSED_CASES),
            "physical adjudication must contain the two exact non-target closures")
    require(document.get("label_operations") == [], "physical adjudication cannot change label rows")
    for item in decisions:
        case_id, image_id = item["case_id"], item["image_id"]
        require(image_id == CLOSED_CASES[case_id] and case_id in cases,
                "physical adjudication case/image mismatch")
        case, record = cases[case_id], records[image_id]
        require(case["image_id"] == image_id and case["split"] == record["split"] == "train"
                and case["status"] in ("held", "partial"), "physical adjudication does not close a pending train case")
        require(item.get("status") == "resolved" and item.get("classification") == "non_target"
                and isinstance(item.get("explanation"), str) and item["explanation"].strip(),
                "physical adjudication is not an explained non-target closure")
        expected_basis = ("user_physical_identification" if image_id == "new_000037"
                          else "independent_original_image_review")
        require(item.get("evidence_basis") == expected_basis, "physical adjudication evidence attribution differs")
        expected_image = {"path": f"data/{record['image']}", "sha256": record["image_sha256"]}
        require(item.get("source_image") == expected_image, "physical adjudication image identity differs")
        bound(root, expected_image)
    # Clarifications are evidence, not extra closures. Check every bound image.
    for item in document.get("clarifications", []):
        for descriptor in ([item["source_image"]] if "source_image" in item else []) + item.get("source_images", []):
            bound(root, descriptor)
    return document


def effective_overlay(root, manifest_path, records, source_bytes):
    """Resolve a verified overlay to exact per-image bytes and source paths."""
    proposal = json.loads(manifest_path.read_text())
    changes = {item["image_id"]: item for item in proposal["changes"]}
    result = {}
    for record in records:
        image_id = record["id"]
        change = changes.get(image_id)
        if change:
            path = _relative_file(manifest_path.parent, change["candidate_label"])
            require(sha(path) == change["candidate_label_sha256"], f"overlay label drift: {image_id}")
            payload = path.read_bytes()
        else:
            path, payload = root / "data" / record["label"], source_bytes[image_id]
        result[image_id] = {"path": path.relative_to(root).as_posix(), "sha256": digest(payload), "bytes": payload}
    return result


def compose(root):
    """Reconstruct the limited candidate from verified, immutable source bytes."""
    root = Path(root).resolve()
    source_path = bound(root, SOURCE_CANDIDATE)
    bound(root, SOURCE_PROTOCOL)
    source_receipt = verify_observed_training_candidate(root, source_path, require_ready=False)
    require(source_receipt["training_readiness"] == "blocked_label_review",
            "this bounded experiment requires its unchanged blocked source composition")
    status = json.loads(bound(root, SOURCE_STATUS).read_text())
    adjudication = verify_adjudication(root)
    pending = source_receipt["unresolved_case_ids"]
    case_map = {item["case_id"]: item for item in status["cases"]}
    closed = {item["case_id"] for item in adjudication["decisions"]}
    require(closed <= set(pending), "physical closure is absent from verified source pending inventory")
    remaining = [case_id for case_id in pending if case_id not in closed]
    retained = sorted({case_map[case_id]["image_id"] for case_id in remaining})
    _, records, _, source_bytes, gold, _ = read_reference(root, source_receipt["source_manifest_sha256"])
    records = sorted(records, key=lambda item: (item["split"], item["id"]))
    train_ids = {row["id"] for row in records if row["split"] == "train"}
    require(set(retained) <= train_ids and source_receipt["coverage_images"] == len(train_ids) == 137,
            "incremental experiment requires all 137 reviewed training photos")
    control_path = bound(root, {"path": PR35_CONTROL["manifest"], "sha256": PR35_CONTROL["sha256"]})
    control = effective_overlay(root, control_path, records, source_bytes)
    reviewed = effective_overlay(root, source_path, records, source_bytes)
    payloads, changes, image_sources, parsed = {}, [], [], {}
    changed_vs_control = []
    for record in records:
        image_id = record["id"]
        use_legacy = image_id in retained
        chosen = control[image_id] if use_legacy else reviewed[image_id]
        payload = chosen["bytes"]
        objects = parse_pose_labels(payload.decode("utf-8"), *record["original_size"], chosen["path"])
        check_geometry(objects, *record["original_size"], chosen["path"])
        parsed[image_id] = objects
        if record["split"] == "val":
            require(payload == source_bytes[image_id] == control[image_id]["bytes"], "validation label changed")
            continue
        if payload != control[image_id]["bytes"]:
            require(not use_legacy, "retained legacy photo has a treatment difference")
            changed_vs_control.append(image_id)
        image_sources.append({
            "image_id": image_id, "source_image_sha256": record["image_sha256"],
            "selection": "pr35_retained_legacy" if use_legacy else "accepted_observed_source",
            "effective_label": {"path": chosen["path"], "sha256": chosen["sha256"]},
            "control_label": {"path": control[image_id]["path"], "sha256": control[image_id]["sha256"]},
            "unresolved_case_ids": [case_id for case_id in remaining if case_map[case_id]["image_id"] == image_id],
        })
        if payload != source_bytes[image_id]:
            relative = record["label"]
            payloads[relative] = payload
            changes.append({"image_id": image_id, "source_label": relative, "candidate_label": relative,
                            "source_label_sha256": record["label_sha256"],
                            "candidate_label_sha256": digest(payload), "source_image_sha256": record["image_sha256"],
                            "source_objects": len(gold[image_id]), "candidate_objects": len(objects),
                            "selected_label_source": chosen["path"], "selected_label_source_sha256": chosen["sha256"]})
    counts = totals(records, parsed)
    require(counts["train"]["images"] == 137 and counts["train"]["empty_images"] == 13,
            "incremental experiment cannot remove training or negative photos")
    manifest = {
        "schema_version": 1, "kind": KIND, "candidate_id": CANDIDATE_ID,
        "status": "controlled_incremental_only", "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": source_receipt["source_manifest_sha256"],
        "source_candidate": SOURCE_CANDIDATE, "source_training_protocol": SOURCE_PROTOCOL,
        "source_review_status": SOURCE_STATUS, "physical_adjudication": ADJUDICATION,
        "control": PR35_CONTROL, "training_protocol_path": PROTOCOL_PATH,
        "full_dataset_certified": False, "full_source_training_readiness": "blocked_label_review",
        "scope": "Accepted source corrections only on photos without a remaining case; exact PR35 label bytes on every unresolved photo in both arms. No image, pixel or uncertain object is removed for this experiment.",
        "closed_non_target_case_ids": sorted(closed), "unresolved_case_ids": remaining,
        "retained_legacy_image_ids": retained, "image_sources": image_sources,
        "changed_images_vs_control": changed_vs_control, "expected_counts": counts, "changes": changes,
    }
    payloads["manifest.json"] = encoded(manifest)
    return payloads, manifest, source_receipt


def read_reference_manifest(root, descriptor):
    path = bound(root, descriptor)
    document = json.loads(path.read_text())
    for item in document["files"]:
        label = _relative_file(path.parent, item["label_path"])
        require(sha(label) == item["label_sha256"], "development-reference label bytes differ")
    return document


def protocol_document(root, manifest, source_receipt):
    from pose_checkpoint import recipe_identity
    recipe = recipe_identity(RECIPE_ID)
    primary = read_reference_manifest(root, PRIMARY_REFERENCE)
    diagnostic = read_reference_manifest(root, DIAGNOSTIC_REFERENCE)
    descriptor = {"path": DEFAULT_CANDIDATE, "sha256": digest(encoded(manifest))}
    return {
        "schema_version": 1, "kind": "baomai-accepted-fronts-training-protocol",
        "experiment_id": "accepted-front-corrections-seed46-v1", "status": "prospectively_prepared",
        "training_readiness": "ready_for_controlled_incremental_experiment", "full_dataset_certified": False,
        "candidate": descriptor, "source_candidate": SOURCE_CANDIDATE, "source_review_status": SOURCE_STATUS,
        "physical_adjudication": ADJUDICATION, "control": PR35_CONTROL, **recipe,
        "initial_checkpoint_sha256": json.loads((root / "metadata/model_card.json").read_text())["sha256"],
        "unresolved_case_ids": manifest["unresolved_case_ids"],
        "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
        "changed_images_vs_control": manifest["changed_images_vs_control"],
        "roles": {arm: {
            "arm": arm, "supervision": "pr35_annotation_control" if arm == "reference" else "accepted_fronts_treatment",
            "candidate_manifest": DEFAULT_CANDIDATE, "candidate_manifest_sha256": descriptor["sha256"],
            "candidate_id": CANDIDATE_ID, **recipe,
            "counts": source_receipt["control"]["counts"] if arm == "reference" else manifest["expected_counts"],
        } for arm in ("reference", "candidate")},
        "execution": {"seed": 46, "epochs": 30, "initialization": "original_epoch173_independently_for_each_arm",
                      "checkpoint_selection": "final_epoch", "checkpoint_path": "fit/weights/last.pt",
                      "resource_trial": {"role": "candidate", "batches": 4, "initializes_full_runs": False},
                      "same_runtime_required": True, "automatic_promotion": False},
        "evaluation": {
            "primary_box_confidence": 0.13,
            "endpoint_confidence": 0.5, "imgsz": 1280, "nms_iou": 0.7, "matching_iou": 0.5, "radii_px": [5, 10],
            "primary_reference": PRIMARY_REFERENCE["path"], "primary_reference_sha256": PRIMARY_REFERENCE["sha256"],
            "primary_counts": primary["counts"],
            "diagnostic_reference": DIAGNOSTIC_REFERENCE["path"], "diagnostic_reference_sha256": DIAGNOSTIC_REFERENCE["sha256"],
            "diagnostic_counts": diagnostic["counts"], "diagnostic_reference_required": True,
            "unresolved_reference_rows": diagnostic["remaining_current_rows"],
            "training_validation_unchanged": True, "references_reported_separately": True,
            "PR35_checkpoint_sha256": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
            "decision": "Rescore both fresh arms and retained PR35 on both fixed references. Report misses, extras/duplicates, zero/one/two-locatable-corner coverage and both-corner successes within 5/10 original pixels, retaining misses in denominators. Inspect gained/lost identities and empty scenes. Conflicting reference rankings are inconclusive; no threshold, epoch or seed search, and no automatic promotion.",
        },
        "limits": [
            "This tests a limited correction scope under retained legacy uncertainty, not full migration to the observed-front policy.",
            "Every unresolved training photo has identical exact PR35 label bytes in both arms; no image or region is quarantined or made an empty negative.",
            "The full observed-front candidate and its gate remain unchanged and blocked; this separate protocol authorizes only its own incremental candidate.",
            "All 13 unresolved development rows remain disclosed and unchanged across the two references; an older reference does not remove their uncertainty.",
            "The development images have already informed this project. Performance on them is not independent deployment validation.",
        ],
    }


def compose_sequence(root):
    """Bind the complete r2 review, then select reviewed or exact PR35 bytes.

    A remaining question holds the entire photograph. In particular, approved
    new_000034 endpoint corrections are not part of this treatment while its
    separate row-15 question remains open. No ambiguous region is erased.
    """
    import re
    from verify_pose_composed_candidate import verify_historical_control

    root = Path(root).resolve()
    require(re.fullmatch(r"[0-9a-f]{40}", SEQUENCE_SOURCE_REVISION),
            "sequence experiment requires its final reviewed source commit")
    source_path = bound(root, SEQUENCE_REVIEW)
    source = json.loads(source_path.read_text())
    status = json.loads(bound(root, SEQUENCE_STATUS).read_text())
    require(source.get("schema_version") == 1
            and source.get("kind") == "review_sequence_front_refinement"
            and source.get("candidate_id") == "sequence-refined-fronts-2026-11-18-r2",
            "sequence source is not the frozen complete r2 review")
    require(source.get("review_status") == SEQUENCE_STATUS
            and status.get("candidate_id") == source["candidate_id"],
            "sequence review and case ledger identities differ")
    require(source.get("release_ready") is False and source.get("full_dataset_certified") is False
            and source.get("training_recipe_id") is None,
            "the full sequence review must keep its separate unreleased status")
    require(source.get("development_reference") == SEQUENCE_PRIMARY_REFERENCE
            and source.get("development_reference_is_separate") is True,
            "sequence review must keep its development reference separate")
    for key in ("review", "annotation_policy", "parent"):
        bound(root, source[key])
    _, records, by_id, source_bytes, gold, original_counts = read_reference(
        root, source["source_manifest_sha256"])
    records = sorted(records, key=lambda item: (item["split"], item["id"]))
    train_ids = {row["id"] for row in records if row["split"] == "train"}
    require(len(train_ids) == 137, "sequence experiment needs the original training cohort")
    cases = status.get("cases")
    require(isinstance(cases, list)
            and len({item["case_id"] for item in cases}) == len(cases)
            and all(item["status"] in ("resolved", "pending", "held", "partial") for item in cases),
            "sequence case ledger has an unknown state or repeated case")
    pending = [item for item in cases if item["status"] != "resolved"]
    require(all(item["split"] == "train" and item["image_id"] in train_ids for item in pending),
            "remaining sequence case is not an original training photo")
    remaining = [item["case_id"] for item in pending]
    retained = sorted({item["image_id"] for item in pending})
    require(len(remaining) == status["remaining_all_case_groups"] == source["remaining_case_groups"]
            and len(retained) == status["remaining_all_training_images"] == source["remaining_training_images"],
            "sequence pending-case or whole-photo inventory differs")
    require("new_000034" in retained, "new034 must retain PR35 labels while row15 is unresolved")

    # The strict historical validator uses an isolated old-policy snapshot;
    # neither the current policy nor the historical PR35 artifacts are altered.
    verify_historical_control(root, records, source_bytes)
    control_path = bound(root, {"path": PR35_CONTROL["manifest"], "sha256": PR35_CONTROL["sha256"]})
    control_manifest = json.loads(control_path.read_text())
    control = effective_overlay(root, control_path, records, source_bytes)
    control_receipt = {
        "manifest": PR35_CONTROL["manifest"], "manifest_sha256": PR35_CONTROL["sha256"],
        "candidate_id": control_manifest["candidate_id"], "kind": control_manifest["kind"],
        "changed_train_labels": len(control_manifest["changes"]), "counts": control_manifest["expected_counts"],
    }
    reviewed, reviewed_objects, listed = {}, dict(gold), set()
    files = source.get("files")
    require(isinstance(files, list) and len(files) == len(train_ids),
            "sequence source must explicitly contain every training label")
    for item in files:
        image_id = item["image_id"]
        require(image_id in train_ids and image_id not in reviewed, "duplicate or non-training reviewed label")
        record = by_id[image_id]
        relative = record["label"]
        require(item["split"] == "train" and item["candidate_label"] == item["source_label"] == relative
                and item["source_label_sha256"] == record["label_sha256"]
                and item["source_image_sha256"] == record["image_sha256"]
                and item["source_objects"] == len(gold[image_id]),
                f"sequence source mapping or identity differs: {image_id}")
        path = _relative_file(source_path.parent, relative)
        payload = path.read_bytes()
        require(digest(payload) == item["candidate_label_sha256"], f"sequence reviewed label drift: {image_id}")
        objects = parse_pose_labels(payload.decode("utf-8"), *record["original_size"], str(path))
        check_geometry(objects, *record["original_size"], str(path))
        require(len(objects) == item["candidate_objects"], f"sequence reviewed count differs: {image_id}")
        reviewed_objects[image_id] = objects
        reviewed[image_id] = {"path": path.relative_to(root).as_posix(), "sha256": digest(payload), "bytes": payload}
        listed.add(relative)
    actual = {path.relative_to(source_path.parent).as_posix()
              for path in (source_path.parent / "labels").rglob("*") if path.is_file() or path.is_symlink()}
    require(actual == listed and set(reviewed) == train_ids, "sequence review label inventory differs")
    require(totals(records, reviewed_objects) == source["expected_counts"], "complete sequence review counts differ")

    payloads, changes, image_sources, parsed, changed_vs_control = {}, [], [], {}, []
    for record in records:
        image_id = record["id"]
        if record["split"] == "val":
            require(control[image_id]["bytes"] == source_bytes[image_id], "PR35 validation bytes differ")
            parsed[image_id] = gold[image_id]
            continue
        use_legacy = image_id in retained
        chosen = control[image_id] if use_legacy else reviewed[image_id]
        payload = chosen["bytes"]
        objects = parse_pose_labels(payload.decode("utf-8"), *record["original_size"], chosen["path"])
        check_geometry(objects, *record["original_size"], chosen["path"])
        parsed[image_id] = objects
        if payload != control[image_id]["bytes"]:
            require(not use_legacy, "held sequence photo has a treatment difference")
            changed_vs_control.append(image_id)
        image_sources.append({
            "image_id": image_id, "source_image_sha256": record["image_sha256"],
            "selection": "pr35_retained_legacy" if use_legacy else "accepted_sequence_review",
            "effective_label": {key: chosen[key] for key in ("path", "sha256")},
            "control_label": {key: control[image_id][key] for key in ("path", "sha256")},
            "reviewed_label": {key: reviewed[image_id][key] for key in ("path", "sha256")},
            "unresolved_case_ids": [item["case_id"] for item in pending if item["image_id"] == image_id],
        })
        if payload != source_bytes[image_id]:
            relative = record["label"]
            payloads[relative] = payload
            changes.append({"image_id": image_id, "source_label": relative, "candidate_label": relative,
                            "source_label_sha256": record["label_sha256"], "candidate_label_sha256": digest(payload),
                            "source_image_sha256": record["image_sha256"], "source_objects": len(gold[image_id]),
                            "candidate_objects": len(objects), "selected_label_source": chosen["path"],
                            "selected_label_source_sha256": chosen["sha256"]})
    counts = totals(records, parsed)
    require(counts["train"]["images"] == 137 and counts["train"]["empty_images"] == 13
            and counts["val"] == original_counts["val"], "sequence experiment changed the frozen cohort or validation")
    manifest = {
        "schema_version": 1, "kind": KIND, "candidate_id": SEQUENCE_CANDIDATE_ID,
        "status": "controlled_incremental_only", "experiment_version": 2,
        "source_revision": SEQUENCE_SOURCE_REVISION, "source_manifest_sha256": source["source_manifest_sha256"],
        "source_candidate": SEQUENCE_REVIEW, "source_review_status": SEQUENCE_STATUS,
        "control": PR35_CONTROL, "training_protocol_path": SEQUENCE_PROTOCOL_PATH,
        "full_dataset_certified": False, "full_source_training_readiness": "blocked_label_review",
        "task_scope": "Detect currently visible cassette front faces and locatable actual lower-front corners; no hidden-object tracking or inferred hidden corner targets.",
        "scope": "Use exact r2 reviewed labels only on photographs without a remaining case. Keep exact PR35 labels on every whole held photograph in both arms; keep all original images, negatives and training-validation labels.",
        "unresolved_case_ids": remaining, "retained_legacy_image_ids": retained,
        "image_sources": image_sources, "changed_images_vs_control": changed_vs_control,
        "expected_counts": counts, "changes": changes,
        "new034_scope": "Its row15 question holds the whole photo, so its reviewed endpoint/box corrections are excluded from this treatment and PR35 bytes are identical in both arms.",
        "new030_new041_scope": "Reopened upper-front ownership questions hold both whole photos, so their supported lower-front corrections also remain outside this treatment.",
    }
    payloads["manifest.json"] = encoded(manifest)
    return payloads, manifest, {"control": control_receipt, "complete_review_counts": source["expected_counts"]}


def sequence_protocol_document(root, manifest, source_receipt):
    from pose_checkpoint import recipe_identity
    recipe = recipe_identity(RECIPE_ID)
    primary = read_reference_manifest(root, SEQUENCE_PRIMARY_REFERENCE)
    diagnostic = read_reference_manifest(root, DIAGNOSTIC_REFERENCE)
    descriptor = {"path": SEQUENCE_CANDIDATE, "sha256": digest(encoded(manifest))}
    return {
        "schema_version": 1, "kind": "baomai-sequence-fronts-training-protocol",
        "experiment_id": "sequence-front-corrections-seed46-v2", "status": "prospectively_prepared",
        "training_readiness": "ready_for_controlled_incremental_experiment", "full_dataset_certified": False,
        "candidate": descriptor, "source_candidate": SEQUENCE_REVIEW, "source_review_status": SEQUENCE_STATUS,
        "control": PR35_CONTROL, **recipe,
        "initial_checkpoint_sha256": json.loads((root / "metadata/model_card.json").read_text())["sha256"],
        "task_scope": manifest["task_scope"], "new034_scope": manifest["new034_scope"],
        "new030_new041_scope": manifest["new030_new041_scope"],
        "unresolved_case_ids": manifest["unresolved_case_ids"],
        "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
        "changed_images_vs_control": manifest["changed_images_vs_control"],
        "complete_review_counts": source_receipt["complete_review_counts"],
        "roles": {arm: {
            "arm": arm, "supervision": "pr35_annotation_control" if arm == "reference" else "accepted_fronts_treatment",
            "candidate_manifest": SEQUENCE_CANDIDATE, "candidate_manifest_sha256": descriptor["sha256"],
            "candidate_id": SEQUENCE_CANDIDATE_ID, **recipe,
            "counts": source_receipt["control"]["counts"] if arm == "reference" else manifest["expected_counts"],
        } for arm in ("reference", "candidate")},
        "execution": {"seed": 46, "epochs": 30, "initialization": "original_epoch173_independently_for_each_arm",
                      "checkpoint_selection": "final_epoch", "checkpoint_path": "fit/weights/last.pt",
                      "resource_trial": {"role": "candidate", "batches": 4, "initializes_full_runs": False},
                      "same_runtime_required": True, "automatic_promotion": False},
        "evaluation": {
            "primary_box_confidence": 0.13, "endpoint_confidence": 0.5, "imgsz": 1280,
            "nms_iou": 0.7, "matching_iou": 0.5, "radii_px": [5, 10],
            "primary_reference": SEQUENCE_PRIMARY_REFERENCE["path"],
            "primary_reference_sha256": SEQUENCE_PRIMARY_REFERENCE["sha256"], "primary_counts": primary["counts"],
            "diagnostic_reference": DIAGNOSTIC_REFERENCE["path"],
            "diagnostic_reference_sha256": DIAGNOSTIC_REFERENCE["sha256"], "diagnostic_counts": diagnostic["counts"],
            "diagnostic_reference_required": True, "unresolved_reference_rows": primary["remaining_current_rows"],
            "diagnostic_unresolved_reference_rows": diagnostic["remaining_current_rows"],
            "training_validation_unchanged": True, "references_reported_separately": True,
            "PR35_checkpoint_sha256": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
            "decision": "Evaluate both fresh final-epoch arms and retained PR35 on the frozen 18 November reference, plus the historical 14 November completion reference separately. Report misses, extras/duplicates, empty-scene outputs, zero/one/two-locatable-corner coverage, and both-corner successes within 5/10 original pixels with misses retained in denominators. Inspect gained/lost identities. Conflicting reference rankings are inconclusive; no threshold, epoch or seed search and no automatic promotion.",
        },
        "limits": [
            "This is one new, bounded supervision experiment, not a rerun of the failed v1 candidate or a claim of model improvement.",
            "The task concerns currently visible front material. Sequence context informs annotation identity, not invisible-object targets or tracking state.",
            "Every unresolved training photo retains identical exact PR35 labels in both arms, including any legacy uncertainties; no photo or uncertain region is removed or made an empty negative.",
            "Whole-photo retention excludes supported new034 endpoint corrections as well as its unresolved row15, and supported lower-front corrections on new030/new041 while their upper-front ownership remains unresolved. The full r2 review remains unreleased.",
            "Current and historical development references retain their own disclosed uncertain rows. Neither is used as training-validation supervision or silently substituted for the other.",
            "These development photos have already informed review and model decisions. A favorable result would not establish reliability on unseen camera scenes.",
        ],
    }


def verify_sequence_candidate(root, candidate):
    payloads, expected, source_receipt = compose_sequence(root)
    require(json.loads(candidate.read_text()) == expected,
            "sequence manifest differs from exact unresolved-photo composition")
    verify_tree(candidate.parent, payloads)
    protocol_path = _relative_file(root, SEQUENCE_PROTOCOL_PATH)
    protocol = sequence_protocol_document(root, expected, source_receipt)
    require(json.loads(protocol_path.read_text()) == protocol,
            "sequence protocol differs from its fixed experiment and evaluation scope")
    return {
        "status": "verified", "candidate_id": SEQUENCE_CANDIDATE_ID, "candidate_manifest_sha256": sha(candidate),
        "source_manifest_sha256": expected["source_manifest_sha256"], "declared_source_revision": SEQUENCE_SOURCE_REVISION,
        "counts": expected["expected_counts"], "changed_train_labels": len(expected["changes"]),
        "removed_objects": sum(item["source_objects"] - item["candidate_objects"] for item in expected["changes"]),
        "control": source_receipt["control"],
        "training_protocol": {"path": SEQUENCE_PROTOCOL_PATH, "sha256": sha(protocol_path)},
        "training_readiness": protocol["training_readiness"], "full_dataset_certified": False,
        "unresolved_case_ids": expected["unresolved_case_ids"], "retained_legacy_image_ids": expected["retained_legacy_image_ids"],
        "changed_images_vs_control": expected["changed_images_vs_control"],
        "unresolved_reference_rows": protocol["evaluation"]["unresolved_reference_rows"],
        "diagnostic_unresolved_reference_rows": protocol["evaluation"]["diagnostic_unresolved_reference_rows"],
        "retained_photo_labels_identical_between_arms": True, "training_run": False,
    }


def verify_incremental_candidate(root, candidate=None):
    root = Path(root).resolve()
    candidate = Path(DEFAULT_CANDIDATE) if candidate is None else Path(candidate)
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if candidate == root / "data/candidates/front-review-2026-11-24-r30/manifest.json":
        from front_review_r30 import verify_candidate as verify_r30_candidate
        return verify_r30_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-23-r28/manifest.json":
        from front_review_r28_training import verify_candidate as verify_r28_candidate
        return verify_r28_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-23-r26/manifest.json":
        from front_review_r26 import verify_candidate as verify_r26_candidate
        return verify_r26_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-23-r25/manifest.json":
        from front_review_r25 import verify_candidate as verify_r25_candidate
        return verify_r25_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-23-r24/manifest.json":
        from front_review_r24 import verify_candidate as verify_r24_candidate
        return verify_r24_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r23/manifest.json":
        from front_review_r23 import verify_candidate as verify_r23_candidate
        return verify_r23_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r22/manifest.json":
        from front_review_r22 import verify_candidate as verify_r22_candidate
        return verify_r22_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r21/manifest.json":
        from front_review_r21 import verify_candidate as verify_r21_candidate
        return verify_r21_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r20/manifest.json":
        from front_review_r20 import verify_candidate as verify_r20_candidate
        return verify_r20_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r19/manifest.json":
        from verify_front_review_r19 import verify_candidate as verify_r19_candidate
        return verify_r19_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r18/manifest.json":
        from verify_front_review_r18 import verify_candidate as verify_r18_candidate
        return verify_r18_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r17/manifest.json":
        from verify_front_review_r17 import verify_candidate as verify_r17_candidate
        return verify_r17_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-22-r16/manifest.json":
        from verify_front_review_r16 import verify_candidate as verify_r16_candidate
        return verify_r16_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-21-r15/manifest.json":
        from verify_front_review_r15 import verify_candidate as verify_r15_candidate
        return verify_r15_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-21-r14/manifest.json":
        from verify_front_review_r14 import verify_candidate as verify_r14_candidate
        return verify_r14_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-21-r13/manifest.json":
        from verify_front_review_r13 import verify_candidate as verify_r13_candidate
        return verify_r13_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-20-r12/manifest.json":
        from verify_front_review_r12 import verify_candidate as verify_r12_candidate
        return verify_r12_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-19-r11/manifest.json":
        from verify_front_review_r11 import verify_candidate as verify_r11_candidate
        return verify_r11_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-19-r10/manifest.json":
        from verify_front_review_r10 import verify_candidate as verify_r10_candidate
        return verify_r10_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-19-r8/manifest.json":
        from verify_front_review_r8 import verify_candidate as verify_r8_candidate
        return verify_r8_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-19-r7/manifest.json":
        from verify_front_review_r7 import verify_candidate as verify_r7_candidate
        return verify_r7_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-19-r6/manifest.json":
        from verify_front_review_r6 import verify_candidate as verify_r6_candidate
        return verify_r6_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-19-r5/manifest.json":
        from verify_front_review_r5 import verify_candidate as verify_r5_candidate
        return verify_r5_candidate(root, candidate)
    if candidate == root / "data/candidates/front-review-2026-11-18-r4/manifest.json":
        from verify_front_review_r4 import verify_candidate as verify_r4_candidate
        return verify_r4_candidate(root, candidate)
    if candidate == root / SEQUENCE_CANDIDATE:
        return verify_sequence_candidate(root, candidate)
    require(candidate == root / DEFAULT_CANDIDATE, "incremental candidate path differs")
    payloads, expected, source_receipt = compose(root)
    actual = json.loads(candidate.read_text())
    require(actual == expected, "incremental manifest differs from exact unresolved-photo composition")
    verify_tree(candidate.parent, payloads)
    protocol_path = _relative_file(root, PROTOCOL_PATH)
    protocol = protocol_document(root, expected, source_receipt)
    require(json.loads(protocol_path.read_text()) == protocol,
            "incremental protocol differs from its fixed experiment and evaluation scope")
    return {
        "status": "verified", "candidate_id": CANDIDATE_ID, "candidate_manifest_sha256": sha(candidate),
        "source_manifest_sha256": expected["source_manifest_sha256"], "declared_source_revision": SOURCE_REVISION,
        "counts": expected["expected_counts"], "changed_train_labels": len(expected["changes"]),
        "removed_objects": sum(item["source_objects"] - item["candidate_objects"] for item in expected["changes"]),
        "control": source_receipt["control"], "training_protocol": {"path": PROTOCOL_PATH, "sha256": sha(protocol_path)},
        "training_readiness": protocol["training_readiness"], "full_dataset_certified": False,
        "unresolved_case_ids": expected["unresolved_case_ids"], "retained_legacy_image_ids": expected["retained_legacy_image_ids"],
        "changed_images_vs_control": expected["changed_images_vs_control"],
        "unresolved_reference_rows": protocol["evaluation"]["unresolved_reference_rows"],
        "retained_photo_labels_identical_between_arms": True, "training_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_incremental_candidate(args.root, args.candidate), indent=2))


if __name__ == "__main__":
    main()
