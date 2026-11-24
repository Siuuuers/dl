#!/usr/bin/env python3
"""One fixed partial-target detection-loss experiment with ordinary image exposure."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, file_sha256
from package_gpu_results import INDEX_NAME, check_pair, collect_arm, copy_member, shared_settings, verify_archive
from pose_checkpoint import _same, recipe_identity
from predict_pose_folder import BASELINE_SHA256, _reference_model
from restore_gpu_inputs import read_json, require


DEFAULT_PROTOCOL = "experiments/partial-target-loss-2026-11-24-r30.json"
PROTOCOL_SHA256 = "2324cd24fbd50acfbce2545d4ae374a83fe3ede978dac3c9414725f363a8a9b0"
ROLES = ("control", "candidate")
NATIVE_SOURCE_HASHES = {
    "utils/loss.py": "f8db64a3093ef310b23cb83d587dd5c8aa3db20a27da94d95a62a734a6c055d2",
    "utils/tal.py": "7fc3127e4b094247ccbbcf8af9829972295f96fc53027313ef51b076d27274cd",
    "engine/trainer.py": "7016131a47b61a8b9b9ea51ef5a9e59b0fb896a328eed4901c89ac7a83169655",
    "data/augment.py": "802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3",
}
SPEC = {
    "control_factor": 1, "candidate_factor": 2,
    "partial_definition": "post_augmentation_fewer_than_two_nonzero_keypoint_visibilities",
    "normalizer": "original_unweighted_target_scores_sum",
    "classification": "entire_soft_target_bce_on_assigned_partial_anchors",
    "box": "ciou_and_native_box_auxiliary_on_assigned_partial_anchors",
    "native_box_auxiliary": "normalized_l1", "native_reg_max": 1, "native_model_head": "Pose26",
    "native_pose_criterion": "E2ELoss(PoseLoss26,PoseLoss26)",
    "sampling": "ordinary_once_per_image_per_epoch",
    "background_unchanged": True, "keypoint_losses_unchanged": True,
}
COUNT_FIELDS = ("targets", "partial_targets", "full_targets", "assigned_partial_targets",
                "unassigned_partial_targets", "assigned_full_targets", "unassigned_full_targets",
                "partial_foreground_anchors", "foreground_anchors", "background_anchors")
BRANCHES = ("one2many", "one2one")
CRITERION_CONTRACT = {"model_head": "Pose26", "criterion": "E2ELoss", "branches": ["PoseLoss26", "PoseLoss26"],
                      "reg_max": 1, "box_auxiliary": "normalized_l1", "rle_unchanged": True}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_protocol(root, path=DEFAULT_PROTOCOL):
    root, path = Path(root).resolve(), Path(path)
    path = path.resolve() if path.is_absolute() else _relative_file(root, str(path))
    require(path == root / DEFAULT_PROTOCOL, "partial-target protocol path differs")
    sha = file_sha256(path)
    require(PROTOCOL_SHA256 is None or sha == PROTOCOL_SHA256, "partial-target protocol differs")
    document = read_json(path)
    require(document["schema_version"] == 1 and document["kind"] == "baomai-partial-target-loss-protocol"
            and document["experiment_id"] == "partial-target-loss-r30", "wrong partial-target experiment")
    require(document["candidate_manifest"] == "data/candidates/front-review-2026-11-24-r30/manifest.json"
            and document["label_arm_for_both_experimental_roles"] == "candidate", "both roles require R30 labels")
    require(all(document[key] == value for key, value in recipe_identity(document["recipe_id"]).items()),
            "base numerical recipe differs")
    require(document["initial_checkpoint_sha256"] == BASELINE_SHA256, "epoch173 initializer differs")
    for relative, expected in (("data/manifest.json", document["source_manifest_sha256"]),
                               (document["candidate_manifest"], document["candidate_manifest_sha256"])):
        require(file_sha256(_relative_file(root, relative)) == expected, "source identity differs: " + relative)
    accepted = document["accepted_review"]
    require(file_sha256(_relative_file(root, accepted["path"])) == accepted["sha256"], "accepted R30 review differs")
    require(all(document["loss_weighting"].get(key) == value for key, value in SPEC.items()), "fixed loss treatment differs")
    execution = document["execution"]
    require(execution["epochs"] == 30 and execution["resource_trial_batches"] == 4
            and execution["expected_batches"] == 1050 and execution["expected_images"] == 4110
            and execution["expected_optimizer_steps"] == 78 and execution["same_runtime_required"] is True
            and execution["automatic_promotion"] is False, "fixed experiment budget differs")
    evaluation = document["evaluation"]
    for prefix in ("primary", "diagnostic"):
        require(file_sha256(_relative_file(root, evaluation[prefix + "_reference"]))
                == evaluation[prefix + "_reference_sha256"], "frozen evaluation reference differs")
    return {"document": document, "sha256": sha, "path": path}


def verify_native_sources():
    require(importlib.metadata.version("ultralytics") == "8.4.182", "pinned Ultralytics required")
    package = Path(importlib.util.find_spec("ultralytics").origin).parent
    for relative, expected in NATIVE_SOURCE_HASHES.items():
        require(file_sha256(package / relative) == expected, "native training source differs: " + relative)
    return dict(NATIVE_SOURCE_HASHES)


def source_identity(root):
    from train_pose_comparison import source_identity as base_source
    root = Path(root).resolve()
    result = base_source(root)
    for name in ("scripts/partial_target_loss.py", "scripts/train_pose_partial_loss.py", DEFAULT_PROTOCOL):
        result["files_sha256"][name] = file_sha256(_relative_file(root, name))
    return result


def experiment_identity(context, role):
    require(role in ROLES, "unknown partial-target role")
    document = context["document"]
    return {"id": document["experiment_id"], "role": role, "protocol_sha256": context["sha256"],
            "base_recipe_id": document["recipe_id"], "base_recipe_sha256": document["recipe_sha256"],
            "label_arm": "candidate", "label_manifest": document["candidate_manifest"],
            "label_manifest_sha256": document["candidate_manifest_sha256"],
            "loss_factor": document["loss_weighting"][role + "_factor"],
            "loss_weighting_spec_sha256": digest_json(document["loss_weighting"])}


def training_inventory(root, context=None):
    """Source-label counts for audit; the treatment uses actual post-augmentation targets."""
    root = Path(root).resolve()
    context = context or load_protocol(root)
    original = read_json(root / "data/manifest.json")
    candidate_path = root / context["document"]["candidate_manifest"]
    changes = {item["image_id"]: item for item in read_json(candidate_path)["changes"]}
    images = []
    for record in sorted(original["records"], key=lambda item: item["id"]):
        if record["split"] != "train":
            continue
        change = changes.get(record["id"])
        label = (_relative_file(candidate_path.parent, change["candidate_label"]) if change
                 else _relative_file(root / "data", record["label"]))
        expected = change["candidate_label_sha256"] if change else record["label_sha256"]
        require(file_sha256(label) == expected, "effective R30 training label differs: " + record["id"])
        groups = Counter()
        for line in label.read_text().splitlines():
            fields = line.split()
            require(len(fields) == 11 and all(float(fields[index]) in (0, 2) for index in (7, 10)),
                    "unexpected accepted visibility encoding")
            groups[sum(float(fields[index]) > 0 for index in (7, 10))] += 1
        images.append({"image_id": record["id"], "label_path": label.relative_to(root).as_posix(),
                       "label_sha256": expected, "source_image_sha256": record["image_sha256"],
                       "objects": sum(groups.values()), "zero_corner_fronts": groups[0],
                       "one_corner_fronts": groups[1], "two_corner_fronts": groups[2]})
    counts = {"images": len(images), "objects": sum(row["objects"] for row in images),
              "empty_images": sum(row["objects"] == 0 for row in images),
              **{name: sum(row[name] for row in images) for name in
                 ("zero_corner_fronts", "one_corner_fronts", "two_corner_fronts")}}
    require(counts["images"] == 137 and counts["empty_images"] == 13, "ordinary training inventory differs")
    result = {"image_ids": [row["image_id"] for row in images], "images": images, "counts": counts}
    result["sha256"] = digest_json(result)
    return result


def make_criterion(model, factor, observer=None):
    """Reuse the pinned pose loss; alter only assigned partial-target detection contributions."""
    verify_native_sources()
    import torch
    from types import MethodType
    from ultralytics.utils.loss import E2ELoss, PoseLoss26
    from ultralytics.utils.tal import make_anchors

    require(factor in (1, 2), "loss factor must be one or two")
    native = model.init_criterion()
    require(type(native) is E2ELoss and type(native.one2many) is type(native.one2one) is PoseLoss26
            and type(model.model[-1]).__name__ == "Pose26" and model.model[-1].reg_max == 1
            and list(model.model[-1].kpt_shape) == [2, 3]
            and native.one2many.rle_loss is not None and native.one2one.rle_loss is not None,
            "fixed Pose26 / E2ELoss / normalized-L1 / RLE architecture differs")

    def get_assigned_targets_and_loss(self, preds, batch, **assign_kwargs):
        # The ordinary implementation is pinned above. Its targets, assigner,
        # gain multipliers and denominator are retained verbatim.
        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (preds["boxes"].permute(0, 2, 1).contiguous(),
                                    preds["scores"].permute(0, 2, 1).contiguous())
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype, batch_size = pred_scores.dtype, pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt, **assign_kwargs)
        target_scores_sum = target_scores.sum().clamp_(min=1)

        # Preserve native packed-target order. Padding and background anchors
        # must never inherit a partial-target label from target index zero.
        partial_gt = torch.zeros(gt_bboxes.shape[:2], dtype=torch.bool, device=self.device)
        batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
        keypoints = batch["keypoints"].to(self.device)
        require(keypoints.shape == (len(batch_idx), 2, 3), "unexpected post-augmentation keypoint shape")
        require(not len(batch_idx) or bool(torch.all(batch_idx[1:] >= batch_idx[:-1])),
                "native target packing requires grouped batch indices")
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within = torch.arange(len(batch_idx), device=self.device) - offsets[batch_idx]
        partial_gt[batch_idx, within] = (keypoints[..., 2] > 0).sum(-1) < 2
        partial_gt &= mask_gt.squeeze(-1).bool()
        partial_anchor = (fg_mask & partial_gt.gather(1, target_gt_idx)) if partial_gt.shape[1] else torch.zeros_like(fg_mask)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        weighted_scores = target_scores
        if factor != 1:
            anchor_weight = 1 + (factor - 1) * partial_anchor.unsqueeze(-1).to(dtype)
            bce_loss = bce_loss * anchor_weight
            weighted_scores = target_scores * anchor_weight
        loss[1] = bce_loss.sum() / target_scores_sum
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            weighted_scores, target_scores_sum, fg_mask, imgsz, stride_tensor)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        if observer is not None:
            assigned = torch.zeros_like(partial_gt, dtype=torch.long)
            if partial_gt.shape[1]:
                assigned.scatter_add_(1, target_gt_idx, fg_mask.long())
            full_gt = mask_gt.squeeze(-1).bool() & ~partial_gt
            statistics = {
                "branch": self._partial_loss_branch,
                "targets": int(mask_gt.sum()), "partial_targets": int(partial_gt.sum()),
                "full_targets": int(full_gt.sum()),
                "assigned_partial_targets": int((partial_gt & assigned.gt(0)).sum()),
                "unassigned_partial_targets": int((partial_gt & assigned.eq(0)).sum()),
                "assigned_full_targets": int((full_gt & assigned.gt(0)).sum()),
                "unassigned_full_targets": int((full_gt & assigned.eq(0)).sum()),
                "partial_foreground_anchors": int(partial_anchor.sum()),
                "foreground_anchors": int(fg_mask.sum()), "background_anchors": int((~fg_mask).sum()),
                "original_target_score_sum": float(target_scores.sum().detach()),
                "normalizer": float(target_scores_sum.detach()),
                "partial_target_score_sum": float(target_scores[partial_anchor].sum().detach()),
                "loss_factor": factor,
            }
            observer(statistics)
        return ((fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss,
                dict(zip(self.loss_names, loss.detach())))

    for name in ("one2many", "one2one"):
        branch = getattr(native, name)
        branch._partial_loss_branch = name
        branch.get_assigned_targets_and_loss = MethodType(get_assigned_targets_and_loss, branch)
    return native


class PartialTargetLoss:
    """Install only on the training model after EMA creation; record consumed batches."""

    def __init__(self, root, context, role):
        self.root, self.context, self.role = Path(root).resolve(), context, role
        self.inventory = training_inventory(root, context)
        self.factor = experiment_identity(context, role)["loss_factor"]
        self.native_sources = verify_native_sources()
        self.epochs, self.pending, self.criterion = [], None, None
        self.exposures = Counter({image_id: 0 for image_id in self.inventory["image_ids"]})

    def start_epoch(self, model, ema_model, epoch):
        require(epoch == len(self.epochs) + 1 and self.pending is None, "loss observer epoch sequence differs")
        require(model is not ema_model, "loss treatment must not be installed on EMA")
        if self.criterion is None:
            self.criterion = make_criterion(model, self.factor, self.record_loss)
            model.criterion = self.criterion
        require(model.criterion is self.criterion and getattr(ema_model, "criterion", None) is not self.criterion,
                "training-only criterion routing differs")
        self.epochs.append({"epoch": epoch, "observed_image_ids": [], "batches": []})

    def prepare_batch(self, batch, epoch):
        require(self.pending is None and self.epochs[-1]["epoch"] == epoch, "loss observer batch order differs")
        self.pending = {"image_ids": [Path(name).stem for name in batch["im_file"]],
                        "objects": len(batch["cls"]), "statistics": {}}

    def record_loss(self, statistics):
        branch = statistics["branch"]
        require(self.pending is not None and branch in ("one2many", "one2one")
                and branch not in self.pending["statistics"], "unexpected repeated loss branch call")
        require(statistics["targets"] == self.pending["objects"], "loss target count differs from consumed batch")
        self.pending["statistics"][branch] = statistics

    def finish_batch(self, image_ids, epoch):
        require(self.pending is not None and set(self.pending["statistics"]) == {"one2many", "one2one"}
                and self.pending["image_ids"] == image_ids and self.epochs[-1]["epoch"] == epoch,
                "loss instrumentation did not observe the completed batch")
        require(all(image_id in self.exposures for image_id in image_ids), "non-training image entered loss treatment")
        record = self.epochs[-1]
        record["observed_image_ids"].extend(image_ids)
        branches = self.pending["statistics"]
        require(all(branches["one2many"][key] == branches["one2one"][key]
                    for key in ("targets", "partial_targets", "full_targets")), "native branches saw different targets")
        record["batches"].append({"batch": len(record["batches"]) + 1, "image_ids": list(image_ids),
                                  **{key: branches["one2many"][key] for key in ("targets", "partial_targets", "full_targets")},
                                  "branches": branches,
                                  "native_branch_schedule": {"updates": self.criterion.updates,
                                                             "one2many": self.criterion.o2m, "one2one": self.criterion.o2o}})
        self.exposures.update(image_ids)
        self.pending = None

    def finish_epoch(self, epoch):
        require(self.pending is None and self.epochs[-1]["epoch"] == epoch, "incomplete loss epoch")
        observed = self.epochs[-1]["observed_image_ids"]
        require(len(observed) == len(set(observed)) == 137 and set(observed) == set(self.exposures),
                "ordinary per-epoch image exposure changed")

    def state(self):
        batches = [batch for epoch in self.epochs for batch in epoch["batches"]]
        return {"schema_version": 1, "role": self.role, "protocol_sha256": self.context["sha256"],
                "inventory_sha256": self.inventory["sha256"], "native_source_hashes": self.native_sources,
                "loss_weighting_spec_sha256": digest_json(self.context["document"]["loss_weighting"]),
                "loss_factor": self.factor, "epochs": self.epochs, "images_seen": sum(self.exposures.values()),
                "batches_seen": len(batches), "per_image": dict(sorted(self.exposures.items())),
                "assignment_totals": {branch: {name: sum(batch["branches"][branch][name] for batch in batches)
                                                   for name in COUNT_FIELDS} for branch in ("one2many", "one2one")},
                "criterion_contract": dict(CRITERION_CONTRACT),
                "criterion_scope": "training_model_only_after_ema_creation"}


def validate_state(state, root, context, role, *, complete=False):
    inventory = training_inventory(root, context)
    require(isinstance(state, dict) and state["schema_version"] == 1 and state["role"] == role
            and state["protocol_sha256"] == context["sha256"] and state["inventory_sha256"] == inventory["sha256"]
            and state["native_source_hashes"] == NATIVE_SOURCE_HASHES
            and state["loss_weighting_spec_sha256"] == digest_json(context["document"]["loss_weighting"])
            and state["loss_factor"] == experiment_identity(context, role)["loss_factor"]
            and state["criterion_contract"] == CRITERION_CONTRACT
            and state["criterion_scope"] == "training_model_only_after_ema_creation", "loss receipt identity differs")
    exposures = Counter({image_id: 0 for image_id in inventory["image_ids"]})
    totals, batch_count = {branch: Counter({name: 0 for name in COUNT_FIELDS}) for branch in BRANCHES}, 0
    require(1 <= len(state["epochs"]) <= 30, "missing loss exposure")
    for epoch, record in enumerate(state["epochs"], 1):
        require(record["epoch"] == epoch and 1 <= len(record["batches"]) <= 35, "loss epoch sequence differs")
        observed = []
        for index, batch in enumerate(record["batches"], 1):
            ids = batch["image_ids"]
            require(batch["batch"] == index and len(ids) == (1 if index == 35 else 4)
                    and set(ids) <= set(exposures), "loss batch identity or shape differs")
            require(set(batch["branches"]) == set(BRANCHES), "native loss branch inventory differs")
            for branch, stats in batch["branches"].items():
                require(stats["branch"] == branch and all(type(stats[name]) is int and stats[name] >= 0 for name in COUNT_FIELDS),
                        "invalid assignment count")
                require(all(batch[key] == stats[key] for key in ("targets", "partial_targets", "full_targets"))
                        and stats["targets"] == stats["partial_targets"] + stats["full_targets"]
                        and stats["partial_targets"] == stats["assigned_partial_targets"] + stats["unassigned_partial_targets"]
                        and stats["full_targets"] == stats["assigned_full_targets"] + stats["unassigned_full_targets"]
                        and stats["assigned_partial_targets"] <= stats["partial_foreground_anchors"] <= stats["foreground_anchors"]
                        and stats["assigned_full_targets"] <= stats["foreground_anchors"] - stats["partial_foreground_anchors"],
                        "inconsistent foreground assignment counts")
                score_sum, partial_sum = stats["original_target_score_sum"], stats["partial_target_score_sum"]
                require(all(isinstance(value, (int, float)) and math.isfinite(value) for value in
                            (score_sum, partial_sum, stats["normalizer"])) and 0 <= partial_sum <= score_sum + 1e-6 * max(score_sum, 1)
                        and stats["normalizer"] == max(score_sum, 1) and stats["loss_factor"] == state["loss_factor"],
                        "loss target weights or original normalizer differ")
                totals[branch].update({name: stats[name] for name in COUNT_FIELDS})
            schedule = batch["native_branch_schedule"]
            one2many = 0.8 if epoch == 1 else max(1 - (epoch - 1) / 29, 0) * (0.8 - 0.1) + 0.1
            require(schedule == {"updates": epoch - 1, "one2many": one2many, "one2one": max(1 - one2many, 0)},
                    "native E2E branch schedule changed")
            observed.extend(ids)
            batch_count += 1
        require(observed == record["observed_image_ids"] and len(observed) == len(set(observed)),
                "ordinary image exposure repeated or differs from completed batches")
        if complete or epoch < len(state["epochs"]):
            require(len(observed) == 137 and set(observed) == set(exposures), "incomplete ordinary image epoch")
        exposures.update(observed)
    require(state["images_seen"] == sum(exposures.values()) and state["batches_seen"] == batch_count
            and state["per_image"] == dict(sorted(exposures.items()))
            and state["assignment_totals"] == {branch: dict(value) for branch, value in totals.items()},
            "loss receipt totals differ")
    if complete:
        require(len(state["epochs"]) == 30 and batch_count == 1050 and state["images_seen"] == 4110
                and set(exposures.values()) == {30}, "completed loss exposure differs from fixed budget")
    return {"images_seen": state["images_seen"], "batches_seen": batch_count,
            "epochs_seen": len(state["epochs"]), "assignment_totals": {branch: dict(value) for branch, value in totals.items()}}


def metadata_preflight(root):
    root = Path(root).resolve()
    weights, card_hash = _reference_model(root)
    return {"root": str(root), "weights": str(weights), "model_sha256": BASELINE_SHA256,
            "model_card_sha256": card_hash, "manifest_sha256": file_sha256(root / "data/manifest.json")}


def verify_trial(root, run, role, context=None):
    from pose_checkpoint import (_check_runtime_v2, _check_staging, _contained_path,
                                 check_training_settings, training_label_identity)
    from stage_pose_training import MANIFEST
    root, run = Path(root).resolve(), Path(run).resolve()
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(receipt["purpose"] == "resource-trial" and receipt["status"] == "trial-complete"
            and receipt["arm"] == "candidate" and receipt["checkpoint_selection"] == "none", "not a completed loss resource trial")
    require(_same(receipt["experiment"], experiment_identity(context, role)), "trial experiment differs")
    validate_state(receipt["partial_target_loss"], root, context, role, complete=False)
    state, document = receipt["partial_target_loss"], context["document"]
    expected = document["execution"]["resource_trial_batches"]
    require(receipt["batches_completed"] == state["batches_seen"] == expected
            and receipt["images_processed"] == state["images_seen"] == expected * 4
            and receipt["optimizer_steps"] > 0 and receipt["optimizer_parameter_change_observed"] is True,
            "trial did not establish real training exposure and updates")
    require(receipt["execution"]["exit_code"] == 0 and receipt["execution"]["stop_reason"] is None
            and receipt["inputs_reverified_after_worker"] is True, "trial stopped or lost verified inputs")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"], "trial source differs")
    require(all(receipt[key] == value for key, value in recipe_identity(document["recipe_id"]).items())
            and receipt["initial_checkpoint_sha256"] == document["initial_checkpoint_sha256"], "trial base recipe differs")
    staging_path = _contained_path(run, MANIFEST)
    require(file_sha256(staging_path) == receipt["staging_manifest_sha256"], "trial staging hash differs")
    staging, preflight = read_json(staging_path), metadata_preflight(root)
    variants = _check_staging(preflight, staging)
    require(receipt["reference_manifest_sha256"] == preflight["manifest_sha256"]
            and receipt["model_card_sha256"] == preflight["model_card_sha256"], "trial source metadata differs")
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"]
            and _same(receipt["training_labels"], training_label_identity(staging, "candidate")), "trial label arm differs")
    variant = variants["candidate"]
    require(receipt["data_yaml_sha256"] == variant["data_yaml_sha256"], "trial selected another dataset")
    stage_root = Path(staging["output_root"])
    require(stage_root.is_absolute() and ".." not in stage_root.parts and stage_root.name == "staged-data",
            "trial staging location differs")
    settings, model = receipt["training_settings"], Path(receipt["training_settings"]["model"])
    relative = Path(preflight["weights"]).relative_to(root)
    require(model.is_absolute() and ".." not in model.parts and model.parts[-len(relative.parts):] == relative.parts,
            "trial initializer location differs")
    check_training_settings(settings, document["recipe_id"], model=model,
                            data=stage_root / variant["data_yaml"], device=settings["device"], project=stage_root.parent)
    _check_runtime_v2(receipt["runtime"], settings)
    log = _contained_path(run, receipt["log"]["path"])
    require(log.stat().st_size > 0 and file_sha256(log) == receipt["log"]["sha256"], "trial log hash differs")
    require(len(receipt["batch_samples"]) == expected and len(state["epochs"]) == 1, "trial batch observations differ")
    for index, sample in enumerate(receipt["batch_samples"]):
        shape = sample["shape"]
        require(len(shape) == 4 and shape[:2] == [4, 3] and min(shape[2:]) > 0 and max(shape[2:]) == 1280
                and sample["epoch"] == 1 and sample["batch"] == index + 1
                and sample["image_ids"] == state["epochs"][0]["batches"][index]["image_ids"], "trial batch geometry differs")
    return receipt


def verify_run(root, run, role, context=None, require_weights=True):
    root, run = Path(root).resolve(), Path(run).resolve()
    require(require_weights is True, "returned loss runs require actual checkpoint bytes")
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(_same(receipt.get("experiment"), experiment_identity(context, role)), "completed loss role differs")
    state = receipt["partial_target_loss"]
    validate_state(state, root, context, role, complete=True)
    require(state["images_seen"] == receipt["images_processed"] and state["batches_seen"] == receipt["batches_completed"]
            and receipt["optimizer_steps"] == context["document"]["execution"]["expected_optimizer_steps"], "loss run budget differs")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"], "completed loss implementation differs")
    bundle = collect_arm(metadata_preflight(root), run, "candidate")
    staging, document = bundle["staging"], context["document"]
    require(receipt["training_settings"]["device"].isdecimal() and receipt["runtime"]["device"].startswith("cuda:")
            and receipt["runtime"]["threads"] == 4, "full loss comparison requires fixed CUDA/four-thread runtime")
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"], "completed R30 labels differ")
    # The pinned native saver serializes EMA and removes its criterion. Verify the
    # returned pickle has no dependency on either experiment module.
    with zipfile.ZipFile(run / "fit/weights/last.pt") as checkpoint:
        pickles = [checkpoint.read(name) for name in checkpoint.namelist() if name.endswith("data.pkl")]
    require(len(pickles) == 1 and all(token not in pickles[0] for token in
            (b"partial_target_loss", b"train_pose_partial_loss", b"PartialTargetPoseLoss")), "custom loss leaked into checkpoint")
    for entry in bundle["files"]:
        entry["path"] = role + "/" + entry["path"].split("/", 1)[1]
    return bundle


def package_pair(root, control, candidate, output, verify_existing=False, protocol=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    context = load_protocol(root, protocol)
    runs = {"control": Path(control).resolve(), "candidate": Path(candidate).resolve()}
    require(all(path.is_dir() and not Path(raw).is_symlink() for raw, path in
                ((control, runs["control"]), (candidate, runs["candidate"]))), "run directories must be regular")
    require(not runs["control"].is_relative_to(runs["candidate"]) and not runs["candidate"].is_relative_to(runs["control"]),
            "run directories overlap")
    bundles = {role: verify_run(root, run, role, context) for role, run in runs.items()}
    check_pair(bundles["control"], bundles["candidate"])
    receipts = {role: bundle["receipt"] for role, bundle in bundles.items()}
    require(receipts["control"]["training_labels"] == receipts["candidate"]["training_labels"], "paired loss labels differ")
    for a, b in zip(receipts["control"]["partial_target_loss"]["epochs"], receipts["candidate"]["partial_target_loss"]["epochs"]):
        require(a["observed_image_ids"] == b["observed_image_ids"], "ordinary paired image order differs")
        require(all(x["targets"] == y["targets"] and x["partial_targets"] == y["partial_targets"]
                    for x, y in zip(a["batches"], b["batches"])), "paired post-augmentation target cohorts differ")
    rows = [entry for bundle in bundles.values() for entry in bundle["files"]]
    rows.append({"path": "experiment-protocol.json", "sha256": context["sha256"],
                 "size_bytes": context["path"].stat().st_size, "source_path": context["path"]})
    rows.sort(key=lambda row: row["path"])
    receipt = receipts["control"]
    index = {"schema_version": 1, "kind": "baomai-partial-target-loss-results", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": context["sha256"], "source": receipt["source"], "runtime": receipt["runtime"],
             "shared_base_settings": shared_settings(receipt),
             "roles": {role: {"checkpoint_manifest": f"{role}/checkpoint-manifest.json", "experiment": bundle["receipt"]["experiment"],
                              "partial_target_loss": bundle["receipt"]["partial_target_loss"], "checkpoint_identity": bundle["identity"]}
                       for role, bundle in bundles.items()},
             "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
             "scope": "Completed fixed-budget partial-target loss comparison only; no accuracy approval or model promotion."}
    output = Path(output).absolute()
    require(not output.is_symlink() and output.suffix.lower() == ".zip", "output must be a regular ZIP path")
    output = output.resolve()
    require(all(not output.is_relative_to(path) for path in (root, *runs.values())), "ZIP overlaps source/run directories")
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            previous = json.loads(archive.read(INDEX_NAME))
        index["created_at_utc"] = previous["created_at_utc"]
        require(_same(previous, index), "existing loss return index differs")
    else:
        require(not output.exists(), "return ZIP already exists")
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode()
    if verify_existing:
        verify_archive(output, index_bytes, rows)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".partial-loss-return-", dir=output.parent) as temporary:
            staged = Path(temporary) / "results.zip"
            with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, index_bytes)
                for entry in rows:
                    copy_member(archive, entry)
            verify_archive(staged, index_bytes, rows)
            os.link(staged, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "status": "verified", "accuracy_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "package"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
        item.add_argument("--protocol", default=DEFAULT_PROTOCOL)
        if name == "package":
            for option in ("control", "candidate", "output"):
                item.add_argument("--" + option, type=Path, required=True)
            item.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        context = load_protocol(args.root, args.protocol)
        result = {"status": "verified", "protocol_sha256": context["sha256"],
                  "inventory": training_inventory(args.root, context), "training_executed": False}
    else:
        result = package_pair(args.root, args.control, args.candidate, args.output, args.verify_existing, args.protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
