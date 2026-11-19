#!/usr/bin/env python3
"""Shared exact replay, label composition and fixed experiment verification.

Each round supplies a small immutable configuration. Earlier review bytes and
protocols remain independently pinned; a new round never rewrites an old review.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, parse_pose_labels
from verify_pose_candidate import check_geometry, read_reference, require, totals
from verify_pose_composed_candidate import PARENT as PR35_CONTROL, verify_historical_control
from verify_reconciled_pose_labels import bound, digest, encoded, sha, verify_tree


def apply_operations(before, operations, image_id):
    """Return exact new bytes and stable parent-to-current row identities."""
    lines = before.decode("utf-8").splitlines(keepends=True)
    edits, additions, seen_additions = {}, {}, set()
    for op in operations:
        require(op["image_id"] == image_id and isinstance(op.get("reason"), str)
                and op["reason"].strip(), f"operation lacks image/reason: {image_id}")
        action = op["action"]
        if action == "add":
            anchor, identity = op["after_base_row"], op["addition_id"]
            require(type(anchor) is int and 0 <= anchor <= len(lines)
                    and isinstance(identity, str) and identity and identity not in seen_additions,
                    f"invalid addition identity/anchor: {image_id}")
            seen_additions.add(identity)
            additions.setdefault(anchor, []).append(op)
        else:
            row = op["base_row"]
            require(action in ("replace", "delete") and type(row) is int
                    and 1 <= row <= len(lines) and row not in edits,
                    f"invalid/repeated immutable base row: {image_id}")
            require(op["before_line"] == lines[row - 1].rstrip("\r\n"),
                    f"operation before text differs: {image_id}:{row}")
            if action == "replace":
                require(op["after_line"] != op["before_line"], f"no-op replacement: {image_id}:{row}")
            edits[row] = op
        if action != "delete":
            require("\n" not in op["after_line"] and "\r" not in op["after_line"]
                    and len(op["after_line"].split()) == 11, f"operation is not one pose row: {image_id}")
    output, identities, deleted = [], [], []
    for row in range(len(lines) + 1):
        if row:
            op = edits.get(row)
            if op and op["action"] == "delete":
                deleted.append(row)
            else:
                output.append((op["after_line"] + "\n") if op else lines[row - 1])
                identities.append({"current_row": len(output), "base_row": row,
                                   "changed": op is not None})
        for op in additions.get(row, []):
            output.append(op["after_line"] + "\n")
            identities.append({"current_row": len(output), "addition_id": op["addition_id"],
                               "after_base_row": row, "changed": True})
    return "".join(output).encode(), {"rows": identities, "deleted_base_rows": deleted}


@dataclass(frozen=True)
class FrontReviewExperiment:
    review: dict
    parent: dict
    candidate_id: str
    protocol_path: str
    experiment_id: str
    source_revision: str
    round_number: int
    experiment_version: int
    serialization_repairs: tuple
    protocol_limits: tuple
    review_development_labels: bool = False
    excluded_training_image_ids: tuple = ()

    @property
    def candidate(self):
        return f"data/candidates/{self.candidate_id}/manifest.json"

    def read_review(self, root):
        root = Path(root).resolve()
        path = bound(root, self.review)
        review = json.loads(path.read_text())
        require(review.get("schema_version") == 1 and review.get("kind") == "original_pixel_front_review"
                and review.get("parent") == self.parent and review.get("full_dataset_certified") is False,
                f"r{self.round_number} review identity or scope differs")
        parent = json.loads(bound(root, self.parent).read_text())
        _, records, by_id, source_bytes, gold, original_counts = read_reference(root, review["source_manifest_sha256"])
        records = sorted(records, key=lambda row: (row["split"], row["id"]))
        operations = json.loads(bound(root, review["operations"]).read_text())
        status = json.loads(bound(root, review["review_status"]).read_text())
        bound(root, review["request_coverage"])
        bound(root, review["annotation_policy"])
        for descriptor in review["user_references"]:
            bound(root, descriptor)
        for descriptor in review.get("review_evidence", []):
            bound(root, descriptor)
        require(status.get("review_id") == review["review_id"], f"r{self.round_number} status is for another review")
        cases = status["cases"]
        require(len({case["case_id"] for case in cases}) == len(cases)
                and all(case["status"] in ("resolved", "pending", "held", "partial")
                        and case["image_id"] in by_id and case["split"] == by_id[case["image_id"]]["split"]
                        for case in cases), f"r{self.round_number} case ledger has an invalid identity or state")
        op_map = {}
        for op in operations:
            require(op["image_id"] in by_id and (self.review_development_labels
                    or by_id[op["image_id"]]["split"] == "train"),
                    f"r{self.round_number} operation is outside its permitted review split")
            op_map.setdefault(op["image_id"], []).append(op)
        repairs = set(self.serialization_repairs)
        require({(item["image_id"], item["base_row"]) for item in review["additional_serialization_repairs"]} == repairs,
                f"r{self.round_number} inherited rounding-repair inventory differs")
        for image_id, items in op_map.items():
            if image_id in review["requested_image_ids"]:
                continue
            for op in items:
                require(op.get("change_kind") == "serialization_repair" and op["action"] == "replace"
                        and (image_id, op["base_row"]) in repairs, f"r{self.round_number} operation is outside its declared scope")
                before, after = op["before_line"].split(), op["after_line"].split()
                require(all(before[i] == after[i] for i in (0, 2, 4, 5, 6, 7, 8, 9, 10)),
                        "rounding repair changed a point or vertical box token")
                boxes = [parse_pose_labels(line, *by_id[image_id]["original_size"])[0]["bbox_xyxy"]
                         for line in (op["before_line"], op["after_line"])]
                require(0 < boxes[0][0] - boxes[1][0] <= .001
                        and max(abs(a-b) for a,b in zip(boxes[0][1:],boxes[1][1:])) < .000001,
                        "rounding repair exceeds its tiny left-bound-only scope")
        parents = {item["image_id"]: item for item in parent["files"]}
        files = review["files"]
        require(len(files) == len(records) == 157 and {item["image_id"] for item in files} == set(by_id),
                f"r{self.round_number} review must cover the exact 157 original photos")
        parsed, reviewed, maps, listed = {}, {}, {}, set()
        for item in files:
            image_id = item["image_id"]
            record, old = by_id[image_id], parents[image_id]
            require(item["split"] == record["split"] and item["label_path"] == record["label"]
                    and item["source_image_sha256"] == record["image_sha256"] == old["source_image_sha256"],
                    f"r{self.round_number} source image/label mapping differs: {image_id}")
            before_path = _relative_file((root / self.parent["path"]).parent, old["label_path"])
            require(sha(before_path) == old["label_sha256"], f"parent label drift: {image_id}")
            expected, maps[image_id] = apply_operations(before_path.read_bytes(), op_map.get(image_id, []), image_id)
            label = _relative_file(path.parent, item["label_path"])
            require(label.read_bytes() == expected and sha(label) == item["label_sha256"],
                    f"r{self.round_number} label differs from exact base-row operations: {image_id}")
            objects = parse_pose_labels(expected.decode(), *record["original_size"], str(label))
            check_geometry(objects, *record["original_size"], str(label))
            require(len(objects) == item["objects"], f"r{self.round_number} per-photo count differs: {image_id}")
            if record["split"] == "val" and not self.review_development_labels:
                require(expected == before_path.read_bytes(), f"r{self.round_number} changed development review bytes")
            parsed[image_id] = objects
            reviewed[image_id] = {"path": label.relative_to(root).as_posix(), "sha256": digest(expected), "bytes": expected}
            listed.add(item["label_path"])
        actual = {p.relative_to(path.parent).as_posix() for p in (path.parent / "labels").rglob("*")
                  if p.is_file() or p.is_symlink()}
        require(actual == listed and totals(records, parsed) == review["expected_counts"],
                f"r{self.round_number} label inventory or split counts differ")
        require(json.loads(bound(root, review["row_maps"]).read_text()) == maps, f"r{self.round_number} base/current row map differs")
        return review, status, records, source_bytes, gold, original_counts, reviewed


    def compose(self, root):
        from verify_incremental_pose_candidate import KIND, effective_overlay
        root = Path(root).resolve()
        review, status, records, source_bytes, gold, original_counts, reviewed = self.read_review(root)
        pending = [case for case in status["cases"] if case["split"] == "train" and case["status"] != "resolved"]
        remaining = sorted(case["case_id"] for case in pending)
        retained = sorted({case["image_id"] for case in pending})
        excluded = sorted(self.excluded_training_image_ids)
        require(len(excluded) == len(set(excluded))
                and set(excluded) <= {r["id"] for r in records if r["split"] == "train"}
                and not set(excluded).intersection(retained),
                "Excluded treatment photos must be distinct resolved training photos")
        if excluded:
            require(review.get("excluded_training_image_ids") == excluded,
                    "Review and experiment treatment exclusions differ")
        verify_historical_control(root, records, source_bytes)
        control_path = bound(root, {"path": PR35_CONTROL["manifest"], "sha256": PR35_CONTROL["sha256"]})
        control_manifest = json.loads(control_path.read_text())
        control = effective_overlay(root, control_path, records, source_bytes)
        control_receipt = {"manifest": PR35_CONTROL["manifest"], "manifest_sha256": PR35_CONTROL["sha256"],
                           "candidate_id": control_manifest["candidate_id"], "kind": control_manifest["kind"],
                           "changed_train_labels": len(control_manifest["changes"]), "counts": control_manifest["expected_counts"]}
        payloads, changes, sources, parsed, changed = {}, [], [], {}, []
        for record in records:
            image_id = record["id"]
            if record["split"] == "val":
                require(control[image_id]["bytes"] == source_bytes[image_id], "PR35 validation bytes differ")
                parsed[image_id] = gold[image_id]
                continue
            held = image_id in retained
            outside_treatment = image_id in excluded
            chosen = control[image_id] if held or outside_treatment else reviewed[image_id]
            payload = chosen["bytes"]
            objects = parse_pose_labels(payload.decode(), *record["original_size"], chosen["path"])
            check_geometry(objects, *record["original_size"], chosen["path"])
            parsed[image_id] = objects
            if payload != control[image_id]["bytes"]:
                require(not held and not outside_treatment, f"held/excluded r{self.round_number} photo differs between arms")
                changed.append(image_id)
            sources.append({"image_id": image_id, "source_image_sha256": record["image_sha256"],
                            "selection": ("pr35_retained_legacy" if held else "pr35_excluded_from_treatment"
                                          if outside_treatment else f"accepted_r{self.round_number}_review"),
                            "effective_label": {key: chosen[key] for key in ("path", "sha256")},
                            "control_label": {key: control[image_id][key] for key in ("path", "sha256")},
                            "reviewed_label": {key: reviewed[image_id][key] for key in ("path", "sha256")},
                            "unresolved_case_ids": [case["case_id"] for case in pending if case["image_id"] == image_id]})
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
                and counts["val"] == original_counts["val"], f"r{self.round_number} experiment changed cohort or original validation")
        manifest = {"schema_version": 1, "kind": KIND, "candidate_id": self.candidate_id,
                    "status": "controlled_incremental_only", "experiment_version": self.experiment_version,
                    "source_revision": self.source_revision, "source_manifest_sha256": review["source_manifest_sha256"],
                    "source_candidate": self.review, "source_review_status": review["review_status"],
                    "control": PR35_CONTROL, "training_protocol_path": self.protocol_path,
                    "full_dataset_certified": False, "full_source_training_readiness": "review_only",
                    "task_scope": "Visible cassette front material and locatable actual lower-front corners; no inferred hidden targets.",
                    "scope": f"Use the current r{self.round_number} review on resolved training photos; retain exact PR35 labels on any whole held training photo in both arms. Keep all original photos, negatives and training-validation labels.",
                    "unresolved_case_ids": remaining, "retained_legacy_image_ids": retained,
                    "image_sources": sources, "changed_images_vs_control": changed,
                    "expected_counts": counts, "changes": changes}
        if excluded:
            manifest["excluded_training_image_ids"] = excluded
            manifest["scope"] += " The user-confirmed old cohort is excluded from this treatment and uses exact PR35 labels in both arms; exclusion is not an unresolved hold."
        payloads["manifest.json"] = encoded(manifest)
        return payloads, manifest, {"control": control_receipt, "complete_review_counts": review["expected_counts"]}


    def protocol_document(self, root, manifest, source_receipt):
        from pose_checkpoint import recipe_identity
        from verify_incremental_pose_candidate import RECIPE_ID, SEQUENCE_PRIMARY_REFERENCE, DIAGNOSTIC_REFERENCE, read_reference_manifest
        recipe = recipe_identity(RECIPE_ID)
        primary = read_reference_manifest(root, SEQUENCE_PRIMARY_REFERENCE)
        diagnostic = read_reference_manifest(root, DIAGNOSTIC_REFERENCE)
        descriptor = {"path": self.candidate, "sha256": digest(encoded(manifest))}
        document = {
            "schema_version": 1, "kind": f"baomai-r{self.round_number}-fronts-training-protocol",
            "experiment_id": self.experiment_id, "status": "prospectively_prepared",
            "training_readiness": "ready_for_controlled_incremental_experiment", "full_dataset_certified": False,
            "candidate": descriptor, "source_candidate": self.review, "source_review_status": manifest["source_review_status"],
            "control": PR35_CONTROL, **recipe,
            "initial_checkpoint_sha256": json.loads((root / "metadata/model_card.json").read_text())["sha256"],
            "task_scope": manifest["task_scope"], "unresolved_case_ids": manifest["unresolved_case_ids"],
            "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
            "changed_images_vs_control": manifest["changed_images_vs_control"],
            "complete_review_counts": source_receipt["complete_review_counts"],
            "roles": {arm: {"arm": arm, "supervision": "pr35_annotation_control" if arm == "reference" else "accepted_fronts_treatment",
                             "candidate_manifest": self.candidate, "candidate_manifest_sha256": descriptor["sha256"],
                             "candidate_id": self.candidate_id, **recipe,
                             "counts": source_receipt["control"]["counts"] if arm == "reference" else manifest["expected_counts"]}
                      for arm in ("reference", "candidate")},
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
                "current_review_reference": self.review if self.review_development_labels else self.parent,
                "current_review_reference_is_exploratory": True,
                "PR35_checkpoint_sha256": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
                "decision": ("Compare both fresh final-epoch arms and retained PR35 on the unchanged 666-front primary and 663-front historical diagnostic references. Report misses first, then extras/duplicates, empty scenes, zero/one/two-locatable-corner coverage and both-corner successes at 5/10 original pixels with misses retained in denominators. Inspect gained/lost identities. Conflicting rankings are inconclusive. The separately versioned current development review may be replayed as exploratory evidence only; report its own hash, counts and unresolved rows, and never substitute it silently. No threshold, seed or epoch search and no automatic promotion."
                             if self.review_development_labels else
                             "Compare both fresh final-epoch arms and retained PR35 on the unchanged 666-front primary and 663-front historical diagnostic references. Report misses first, then extras/duplicates, empty scenes, zero/one/two-locatable-corner coverage and both-corner successes at 5/10 original pixels with misses retained in denominators. Inspect gained/lost identities. Conflicting rankings are inconclusive. The separate 662-front review may be replayed as exploratory evidence only; never substitute it silently. No threshold, seed or epoch search and no automatic promotion."),
            },
            "limits": list(self.protocol_limits),
        }
        if self.excluded_training_image_ids:
            document["excluded_training_image_ids"] = list(sorted(self.excluded_training_image_ids))
        return document


    def verify_candidate(self, root, candidate=None):
        root = Path(root).resolve()
        candidate = (root / self.candidate) if candidate is None else Path(candidate).resolve()
        require(candidate == root / self.candidate, f"r{self.round_number} candidate path differs")
        payloads, manifest, receipt = self.compose(root)
        require(json.loads(candidate.read_text()) == manifest, f"r{self.round_number} candidate differs from exact reviewed/held composition")
        verify_tree(candidate.parent, payloads)
        protocol = self.protocol_document(root, manifest, receipt)
        protocol_path = _relative_file(root, self.protocol_path)
        require(json.loads(protocol_path.read_text()) == protocol, f"r{self.round_number} protocol differs from fixed experiment scope")
        return {"status": "verified", "candidate_id": self.candidate_id, "candidate_manifest_sha256": sha(candidate),
                "source_manifest_sha256": manifest["source_manifest_sha256"], "declared_source_revision": self.source_revision,
                "counts": manifest["expected_counts"], "changed_train_labels": len(manifest["changes"]),
                "removed_objects": sum(item["source_objects"] - item["candidate_objects"] for item in manifest["changes"]),
                "control": receipt["control"], "training_protocol": {"path": self.protocol_path, "sha256": sha(protocol_path)},
                "training_readiness": protocol["training_readiness"], "full_dataset_certified": False,
                "unresolved_case_ids": manifest["unresolved_case_ids"], "retained_legacy_image_ids": manifest["retained_legacy_image_ids"],
                "changed_images_vs_control": manifest["changed_images_vs_control"],
                "unresolved_reference_rows": protocol["evaluation"]["unresolved_reference_rows"],
                "diagnostic_unresolved_reference_rows": protocol["evaluation"]["diagnostic_unresolved_reference_rows"],
                "retained_photo_labels_identical_between_arms": True, "training_run": False}


    def build(self, root):
        root = Path(root).resolve()
        candidate, protocol = root / self.candidate, root / self.protocol_path
        require(not candidate.parent.exists() and not protocol.exists(), f"r{self.round_number} artifacts already exist; verify instead of overwriting")
        payloads, manifest, receipt = self.compose(root)
        for relative, payload in payloads.items():
            path = candidate.parent / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_bytes(encoded(self.protocol_document(root, manifest, receipt)))
        return self.verify_candidate(root)


