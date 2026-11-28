#!/usr/bin/env python3
"""Run one frozen crop-context comparison with complete pose-record lineage."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
import diagnose_current_proposals_r69 as current
from analyze_duplicate_audit_r66 import digest, encoded, need
from analyze_partial_detection_r68 import REFERENCES, known_corner_outcomes
from analyze_proposal_stages_r63 import endpoint_outcomes, score_identity
from evaluate_pose import INFERENCE, _prediction_records
from pose_checkpoint import check_loaded_pose_model
from predict_pose_folder import postclip_nms
from replay_partial_inclusion_r67 import matrix_row
from run_corner_average_r58 import report
from run_reference_matrix_r60 import verified_zip


PROTOCOL_PATH = "docs/thesis/evidence/crop-context-r71/protocol.json"
CONTROL_DIRECTORY = "docs/thesis/evidence/current-proposals-r69"
CONTROL_FILES = {
    "native-predictions.json.gz": {"bytes": 272772, "sha256": "66ffc3da1ade6d2522e098ce22bc3f0ae182c242dfa22259218283b33fae016c"},
    "predictions.json.gz": {"bytes": 694543, "sha256": "b069e017cb382f73edbf12a164b410766bf8e4051ded93217be221887523133c"},
    "per-image.json.gz": {"bytes": 925601, "sha256": "fbd184c0992bf4744ae7b2d7e2d7060840382c39d5adfdef8ab9abbfe6784b3f"},
    "summary.json": {"bytes": 77635, "sha256": "973627f75a593f43ff3d35ca6f5924b8bcb125c0e7241b7446b4d8cf0bc379e7"},
}
CROPS = (("left", (0, 0, 768, 720)), ("right", (512, 0, 1280, 720)))
CROP_SIZES = (768, 1280)
VARIANTS = ("baseline", "crop_context_768", "crop_context_scale_1280")
RESULT_NAME = "baomai-crop-context-r71-results.zip"
SOURCES = tuple(dict.fromkeys(("analyze_crop_context_r71.py", *current.SOURCES)))
IMPLEMENTATION_CONTRACT = {
    "checkpoint_sha256": current.MODEL_SHA,
    "photographs": 20,
    "image_size_wh": [1280, 720],
    "baseline": "unchanged original/reflected two-view corner averaging at input 1280",
    "crops_xyxy": {name: list(box) for name, box in CROPS},
    "crop_input_sizes": list(CROP_SIZES),
    "crop_reflection": False,
    "confidence_strictly_above": .25,
    "native_nms_iou": .70,
    "native_max_det_per_call": 300,
    "corner_confidence_inclusive": .50,
    "matching_iou_inclusive": .50,
    "artificial_crop_edge_rejection_margin_px": 1.,
    "point_validity_before_mapping": "finite unclipped crop coordinates in [0,width) x [0,height)",
    "invalid_crop_point": [None, None, "original confidence"],
    "merge": "union baseline then left crop then right crop, whole-record postclip NMS at IoU strictly above 0.70",
    "merge_score_ties": "earlier pool index, baseline first then left/right native output order",
    "pooled_output_cap": None,
    "references": list(REFERENCES),
    "primary_reference": "r67",
    "control": "R69 fresh CPU, exact per-image score identities and known-corner threshold outcomes",
    "training": False,
    "optimizer_updates": 0,
    "labels_changed": False,
    "threshold_search": False,
    "automatic_promotion": False,
}


def protocol(root):
    data = (root / PROTOCOL_PATH).read_bytes()
    value = json.loads(data)
    need(value.get("implementation_contract") == IMPLEMENTATION_CONTRACT,
         "R71 fixed protocol and implementation contract differ")
    return value, digest(data)


def prepare(root, directory):
    protocol(root)
    plan = current.prepare(root, directory)
    for name, spec in CONTROL_FILES.items():
        current.checked(root / CONTROL_DIRECTORY / name, spec["sha256"], spec["bytes"])
    return plan


def load_control(root):
    result = {}
    for name, spec in CONTROL_FILES.items():
        data = current.checked(root / CONTROL_DIRECTORY / name, spec["sha256"], spec["bytes"])
        result[name.split(".")[0]] = json.loads(gzip.decompress(data) if name.endswith(".gz") else data)
    return result


def letterbox_geometry(input_hw, original_hw):
    """Use the pinned native scale_coords rounding, without its final clamp."""
    input_h, input_w = map(int, input_hw)
    height, width = map(int, original_hw)
    gain = min(input_h / height, input_w / width)
    resized_h, resized_w = round(height * gain), round(width * gain)
    left = round((input_w - resized_w) / 2 - .1)
    top = round((input_h - resized_h) / 2 - .1)
    return {"original_size_wh": [width, height], "input_size_wh": [input_w, input_h],
            "resized_size_wh": [resized_w, resized_h],
            "gain_xy": [resized_w / width, resized_h / height],
            "pad_left_top_right_bottom": [left, top, input_w - resized_w - left, input_h - resized_h - top]}


def finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


def map_crop_prediction(prediction, unclipped_points, crop_name, crop_box, native_index):
    """Translate one complete record, recording seam rejection and point validity."""
    left, top, right, bottom = crop_box
    width, height = right - left, bottom - top
    box = prediction["bbox_xyxy"]
    need(len(box) == 4 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in box),
         "A crop output has a nonfinite box")
    need(0 <= box[0] <= box[2] <= width and 0 <= box[1] <= box[3] <= height,
         "A native clipped crop box is outside its crop")
    need(prediction["class_id"] == 0 and .25 < prediction["confidence"] <= 1.,
         "A native crop output violates the fixed confidence/class contract")
    need(len(unclipped_points) == len(prediction["keypoints"]) == 2, "Crop keypoint shape differs")
    reasons = []
    if left > 0 and box[0] <= 1.:
        reasons.append("box_touches_artificial_left_edge")
    if right < 1280 and box[2] >= width - 1.:
        reasons.append("box_touches_artificial_right_edge")
    mapped = deepcopy(prediction)
    mapped["bbox_xyxy"] = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
    point_rows = []
    for index, (unclipped, native) in enumerate(zip(unclipped_points, prediction["keypoints"])):
        x, y, score = unclipped
        need(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1
             and score == native[2], "A crop keypoint confidence differs")
        invalid = []
        finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y))
        if not finite:
            invalid.append("nonfinite_unclipped_crop_coordinate")
        elif not (0 <= x < width and 0 <= y < height):
            invalid.append("unclipped_coordinate_outside_crop")
        if invalid:
            point = [None, None, score]
        else:
            need(native[:2] == [x, y], "Valid crop point differs from the pre-clamp native point")
            point = [native[0] + left, native[1] + top, score]
            need(0 <= point[0] < 1280 and 0 <= point[1] < 720, "Mapped valid point is outside the photograph")
        mapped["keypoints"][index] = point
        point_rows.append({"corner_index": index, "unclipped_crop_xy": [x, y],
                           "native_clipped_crop_xy": native[:2], "confidence": score,
                           "invalid_reasons": invalid, "mapped_keypoint": point})
    return mapped, {"crop": crop_name, "crop_xyxy": list(crop_box), "native_index": native_index,
                    "native_prediction": prediction, "mapped_prediction": mapped,
                    "boundary_rejection_reasons": reasons, "retained_for_pool": not reasons,
                    "point_mapping": point_rows}


def initialize_predictor(weights):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PosePredictor
    from ultralytics.utils import ops

    runtime = current.native_runtime()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    class TracePredictor(PosePredictor):
        def construct_result(self, pred, img, orig_img, img_path):
            shape = tuple(self.model.kpt_shape)
            need(shape == (2, 3), "Expected two native three-channel keypoints")
            unscaled = pred[:, 6:].detach().clone().view(pred.shape[0], *shape)
            transform = letterbox_geometry(img.shape[2:], orig_img.shape[:2])
            points = unscaled.clone()
            for axis in (0, 1):
                points[..., axis] -= transform["pad_left_top_right_bottom"][axis]
                points[..., axis] /= transform["gain_xy"][axis]
            expected = ops.clip_coords(points.clone(), orig_img.shape)
            result = super().construct_result(pred, img, orig_img, img_path)
            need(torch.allclose(expected, result.keypoints.data, rtol=0., atol=0., equal_nan=True),
                 "Independent pre-clamp point reconstruction differs from native coordinates")
            need(torch.isfinite(points[..., 2]).all().item(), "Nonfinite native point confidence")
            self.r71_trace = {"transform": transform,
                "input_tensor_shape": list(img.shape),
                "input_tensor_sha256": digest(img.detach().cpu().contiguous().numpy().tobytes()),
                "unclipped_keypoints": [[[finite_or_none(v) for v in p] for p in row]
                                         for row in points.detach().cpu().tolist()],
                "native_point_reconstruction_exact": True}
            return result

    model = YOLO(str(weights))
    check_loaded_pose_model(model)
    model.model.requires_grad_(False)
    with torch.inference_mode():
        model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), predictor=TracePredictor,
                      device="cpu", conf=.25, **INFERENCE)
    need(not model.predictor.model.end2end and not model.predictor.model.fp16
         and str(model.predictor.model.device) == "cpu", "Expected ordinary FP32 CPU pose inference")
    need(not any(module.training for module in model.model.modules()), "Inference model is not fully in eval mode")
    return model, runtime, current.state_hash(model.model)


def predict(model, raster, image_size, call_id):
    import torch
    started = time.perf_counter()
    with torch.inference_mode():
        result = model.predict(source=raster, device="cpu", conf=.25, **{**INFERENCE, "imgsz": image_size})[0]
    seconds = time.perf_counter() - started
    need(tuple(result.orig_shape) == tuple(raster.shape[:2]), "Native crop dimensions differ")
    records = _prediction_records(result)
    trace = deepcopy(model.predictor.r71_trace)
    need(len(records) == len(trace["unclipped_keypoints"]), "Native point trace count differs")
    need(len(records) <= 300 and all(.25 < p["confidence"] <= 1 for p in records), "Native detection contract differs")
    trace.update(call_id=call_id, image_size=image_size, view_bgr_pixel_sha256=digest(raster.tobytes()),
                 native_output_count=len(records), native_output_cap_reached=len(records) == 300,
                 forward_seconds=seconds)
    return records, trace


def check_control(root, references, native, outputs, output):
    archived = load_control(root)
    before_native = archived["native-predictions"]["fresh_cpu"]
    before_output = archived["predictions"]["fresh_cpu"]["two_view"]
    comparisons, gates, idempotence = [], [], []
    for view in ("original", "reflected"):
        for image in sorted(native[view]):
            comparisons.append({"scope": "native_" + view, "image": image,
                **current.record_difference(before_native[view][image], native[view][image])})
    for image in sorted(outputs):
        comparisons.append({"scope": "two_view", "image": image,
                            **current.record_difference(before_output[image], outputs[image])})
        selected, selection = postclip_nms(outputs[image])
        before_records = [{k: v for k, v in p.items() if k != "primary_prediction_id"} for p in outputs[image]]
        after_records = [{k: v for k, v in p.items() if k != "primary_prediction_id"} for p in selected]
        kept_indices = [i - 1 for i in selection["retained_primary_prediction_ids"]]
        idempotence.append({"image": image, "pose_records_exact": before_records == after_records,
                            "kept_pool_indices": kept_indices,
                            "all_indices_in_original_order": kept_indices == list(range(len(outputs[image])))})
    for reference in REFERENCES:
        _, scores, _ = report(references[reference], outputs)
        old_scores = archived["per-image"]["fresh_cpu"]["two_view"][reference]
        for image in sorted(outputs):
            gates.append({"reference": reference, "image": image,
                "matching_and_count_identities_equal": score_identity(old_scores[image]) == score_identity(scores[image]),
                "known_corner_outcomes_equal": endpoint_outcomes(references[reference][image], before_output[image], old_scores[image])
                    == endpoint_outcomes(references[reference][image], outputs[image], scores[image]),
                "exact_score_object_equal": old_scores[image] == scores[image]})
    passed = all(r["matching_and_count_identities_equal"] and r["known_corner_outcomes_equal"] for r in gates)
    passed = passed and all(r["same_count_and_classes_in_record_order"] for r in comparisons)
    passed = passed and all(r["pose_records_exact"] and r["all_indices_in_original_order"] for r in idempotence)
    receipt = {"passed": passed, "reference": "R69 fresh CPU", "input_files": CONTROL_FILES,
               "per_image_gates": gates, "record_comparisons": comparisons,
               "baseline_only_union_idempotence": idempotence,
               "all_records_exact": all(r["exact_record_equality"] for r in comparisons),
               "crop_inference_started": False}
    if not passed:
        current.frozen(output / "failed-control-gate.json", encoded(receipt))
        raise RuntimeError("R69 fresh-CPU baseline gate failed. Crop inference was not started.")
    return receipt


def merge_records(baseline, crops, variant, image):
    pool = deepcopy(baseline)
    lineage = [{"source_id": f"baseline/{image}/P{i + 1:03d}", "source": "baseline",
                "source_index": i, "crop": None} for i in range(len(baseline))]
    for crop_name, _ in CROPS:
        for row in crops[crop_name]:
            if row["retained_for_pool"]:
                pool.append(deepcopy(row["mapped_prediction"]))
                lineage.append({"source_id": f"{variant}/{crop_name}/{image}/P{row['native_index'] + 1:03d}",
                                "source": "crop", "source_index": row["native_index"], "crop": crop_name})
    selected, audit = postclip_nms(pool)
    indices = [i - 1 for i in audit["retained_primary_prediction_ids"]]
    need(len(indices) == len(selected), "Union selection count differs")
    for pi, value in zip(indices, selected):
        need({k: v for k, v in value.items() if k != "primary_prediction_id"}
             == {k: v for k, v in pool[pi].items() if k != "primary_prediction_id"},
             "Union suppression changed a surviving pose record")
    return selected, {"pool_predictions": pool, "pool_lineage": lineage,
                       "kept_pool_indices": indices, "kept_source_ids": [lineage[i]["source_id"] for i in indices],
                       "suppression": audit}


def evaluate(references, outputs, merges, lineage):
    matrix, measurements, scores, states = [], {}, {}, {}
    stable = {(r["image"], r["candidate_index"]): r["stable_id"] for r in lineage}
    for variant, predictions in outputs.items():
        measurements[variant], scores[variant], states[variant] = {}, {}, {}
        for reference in REFERENCES:
            measurement, images, cases = report(references[reference], predictions)
            measurements[variant][reference], scores[variant][reference], states[variant][reference] = measurement, images, cases
            matrix.append(matrix_row(reference, variant, measurement, cases, references[reference], predictions))
    target_rows, output_rows, known_totals, empty = [], [], {}, []
    for variant in VARIANTS[1:]:
        known_totals[variant] = {}
        for reference in REFERENCES:
            known = {str(k): Counter({name: 0 for name in ("fronts", "reference_corners", "before_available", "after_available",
                "before_within5", "after_within5", "before_within10", "after_within10")}) for k in (0, 1, 2)}
            old = states["baseline"][reference]
            new = states[variant][reference]
            need(len(old) == len(new), "Paired target count differs")
            for before, after in zip(old, new):
                image, gi = before["image"], before["reference_index"]
                need((image, gi) == (after["image"], after["reference_index"]), "Paired target order differs")
                bi, ai = before["prediction_index"], after["prediction_index"]
                bp = outputs["baseline"][image][bi] if bi is not None else None
                ap = outputs[variant][image][ai] if ai is not None else None
                bid = f"baseline/{image}/P{bi + 1:03d}" if bi is not None else None
                aid = merges[variant][image]["kept_source_ids"][ai] if ai is not None else None
                target = references[reference][image][gi]
                bc, ac = known_corner_outcomes(target, bp), known_corner_outcomes(target, ap)
                number = sum(point[2] == 2 for point in target["keypoints"])
                k = known[str(number)]
                k.update(fronts=1, reference_corners=number)
                for prefix, corners in (("before", bc), ("after", ac)):
                    for metric in ("available", "within5", "within10"):
                        k[prefix + "_" + metric] += sum(c[metric] for c in corners)
                target_rows.append({"variant": variant, "reference": reference, "image": image,
                    "reference_index": gi, "reference_id": f"{reference}/{image}/R{gi + 1:03d}",
                    "stable_reference_id": stable[(image, gi)] if reference == "r67" else f"{reference}/{image}/R{gi + 1:03d}",
                    "known_corners": number, "source_changed": bid != aid,
                    "detection_gain": not before["detected"] and after["detected"],
                    "detection_loss": before["detected"] and not after["detected"],
                    "pair_within5_gain": not before["within5"] and after["within5"],
                    "pair_within5_loss": before["within5"] and not after["within5"],
                    "pair_within10_gain": not before["within10"] and after["within10"],
                    "pair_within10_loss": before["within10"] and not after["within10"],
                    "before": {**before, "source_id": bid, "known_corner_outcomes": bc},
                    "after": {**after, "source_id": aid, "known_corner_outcomes": ac}})
            known_totals[variant][reference] = {k: dict(v) for k, v in known.items()}
    for variant in VARIANTS:
        for reference in REFERENCES:
            for image in sorted(outputs[variant]):
                score = scores[variant][reference][image]
                matches = {m["prediction_index"]: m["gold_index"] for m in score["matches"]}
                duplicates = set(score["duplicate_prediction_indices"])
                for pi, prediction in enumerate(outputs[variant][image]):
                    gi = matches.get(pi)
                    output_rows.append({"variant": variant, "reference": reference, "image": image,
                        "prediction_index": pi,
                        "source_id": f"baseline/{image}/P{pi + 1:03d}" if variant == "baseline"
                            else merges[variant][image]["kept_source_ids"][pi],
                        "status": "matched" if gi is not None else "duplicate" if pi in duplicates else "other_extra",
                        "reference_index": gi})
                if not references[reference][image]:
                    empty.append({"variant": variant, "reference": reference, "image": image,
                                  "outputs": len(outputs[variant][image]), "predictions": outputs[variant][image]})
    paired = {}
    for variant in VARIANTS[1:]:
        paired[variant] = {}
        for reference in REFERENCES:
            rows = [r for r in target_rows if r["variant"] == variant and r["reference"] == reference]
            names = ("detection_gain", "detection_loss", "pair_within5_gain", "pair_within5_loss",
                     "pair_within10_gain", "pair_within10_loss", "source_changed")
            paired[variant][reference] = {key: sum(r[key] for r in rows) for key in names}
            paired[variant][reference]["detection_gain_ids"] = [r["stable_reference_id"] for r in rows if r["detection_gain"]]
            paired[variant][reference]["detection_loss_ids"] = [r["stable_reference_id"] for r in rows if r["detection_loss"]]
    return matrix, paired, known_totals, {"measurements": measurements, "per-image": scores,
        "target-states": states, "paired-targets": target_rows, "output-states": output_rows, "empty-scenes": empty}


def analyze(root, directory, output, plan):
    import cv2
    root, directory, output = Path(root), Path(directory), Path(output)
    _, protocol_sha = protocol(root)
    need(not output.exists() or not any(output.iterdir()), "Existing R71 outputs preserved. Choose a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    references, reference_text, lineage, hashes, _, _ = current.reference_inputs(root, directory)
    weights = directory / plan["checkpoint"]["filename"]
    current.checked(weights, current.MODEL_SHA, current.MODEL_BYTES)
    model, runtime, before_state = initialize_predictor(weights)
    images = {}
    for row in plan["images"]:
        path = directory / "images" / (row["id"] + ".png")
        current.checked(path, row["image_sha256"])
        raster = cv2.imread(str(path))
        need(raster is not None and raster.shape == (720, 1280, 3), "Photograph decoding differs")
        images[row["id"]] = raster
    native = {"original": {}, "reflected": {}}
    calls = []
    for image, raster in images.items():
        for view, value in (("original", raster), ("reflected", np.ascontiguousarray(raster[:, ::-1, :]))):
            records, trace = predict(model, value, 1280, f"baseline/{view}/{image}")
            need(trace["input_tensor_shape"] == [1, 3, 736, 1280], "Baseline tensor shape differs")
            native[view][image] = records
            calls.append({"image": image, "variant": "baseline", "view": view, **trace})
    baseline_stages, associations = current.completed_outputs(native)
    baseline = baseline_stages["two_view"]
    gate = check_control(root, references, native, baseline, output)
    outputs, merges, crop_rows = {"baseline": baseline}, {}, {}
    for size, variant in zip(CROP_SIZES, VARIANTS[1:]):
        outputs[variant], merges[variant], crop_rows[variant] = {}, {}, {}
        for image, raster in images.items():
            crop_rows[variant][image] = {}
            for crop_name, box in CROPS:
                x0, y0, x1, y1 = box
                crop = np.ascontiguousarray(raster[y0:y1, x0:x1])
                records, trace = predict(model, crop, size, f"{variant}/{crop_name}/{image}")
                expected = [1, 3, 736, 768] if size == 768 else [1, 3, 1216, 1280]
                need(trace["input_tensor_shape"] == expected, "Fixed crop tensor shape differs")
                gain = 1. if size == 768 else 5. / 3.
                need(trace["transform"]["gain_xy"] == [gain, gain], "Fixed crop resize gain differs")
                rows = []
                for i, prediction in enumerate(records):
                    _, mapped = map_crop_prediction(prediction, trace["unclipped_keypoints"][i], crop_name, box, i)
                    rows.append(mapped)
                crop_rows[variant][image][crop_name] = rows
                calls.append({"image": image, "variant": variant, "crop": crop_name, "crop_xyxy": list(box), **trace})
            outputs[variant][image], merges[variant][image] = merge_records(baseline[image], crop_rows[variant][image], variant, image)
    after_state = current.state_hash(model.model)
    need(before_state == after_state, "Frozen model state changed")
    current.checked(weights, current.MODEL_SHA, current.MODEL_BYTES)
    runtime.update(frozen_fused_state_sha256=before_state, frozen_fused_state_unchanged=True,
                   checkpoint_file_unchanged=True, optimizer_updates=0)
    matrix, paired, known, evaluated = evaluate(references, outputs, merges, lineage)
    crop_counts = {}
    for variant in VARIANTS[1:]:
        rows = [row for image in crop_rows[variant].values() for values in image.values() for row in values]
        crop_counts[variant] = {"native_outputs": len(rows),
            "boundary_rejected": sum(not r["retained_for_pool"] for r in rows),
            "pooled_crop_outputs": sum(r["retained_for_pool"] for r in rows),
            "invalidated_points_all_native_outputs": sum(bool(p["invalid_reasons"]) for r in rows for p in r["point_mapping"]),
            "invalidated_points_pooled_outputs": sum(bool(p["invalid_reasons"]) for r in rows if r["retained_for_pool"] for p in r["point_mapping"]),
            "retained_baseline_outputs": sum(sum(merges[variant][i]["pool_lineage"][k]["source"] == "baseline"
                for k in merges[variant][i]["kept_pool_indices"]) for i in images),
            "retained_crop_outputs": sum(sum(merges[variant][i]["pool_lineage"][k]["source"] == "crop"
                for k in merges[variant][i]["kept_pool_indices"]) for i in images)}
    summary = {"schema": "cassette-crop-context-r71-v1", "status": "completed",
        "protocol_path": PROTOCOL_PATH, "protocol_sha256": protocol_sha,
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "analysis_source_sha256": {"scripts/" + name: digest((root / "scripts" / name).read_bytes()) for name in SOURCES},
        "checkpoint": plan["checkpoint"], "source_archives": current.INPUTS,
        "image_inputs": [{"id": r["id"], "image_sha256": r["image_sha256"]} for r in plan["images"]],
        "reference_input_sha256": hashes, "control_input_files": CONTROL_FILES,
        "runtime": runtime, "control_gate_passed": gate["passed"], "control_records_exact": gate["all_records_exact"],
        "matrix": matrix, "paired_changes": paired, "known_corner_totals": known, "crop_counts": crop_counts,
        "record_counts": {"forward_calls": len(calls), "baseline_forward_calls": 40, "crop_forward_calls": 80,
                          "matrix_rows": len(matrix), "paired_targets": len(evaluated["paired-targets"])},
        "fresh_inference": True, "training": False, "optimizer_updates": 0, "labels_changed": False,
        "threshold_search": False, "automatic_promotion": False, "development_only": True,
        "interpretation": "The 768 arm changes context and adds candidate exposure. The 1280 arm also enlarges those crops. Both are fixed development probes."}
    records = {"native-baseline": native, "baseline-associations": associations, "control-gate": gate,
        "inference-calls": calls, "crop-lineage": crop_rows, "merge-lineage": merges,
        "predictions": outputs, "reference-labels": reference_text, "reference-lineage": lineage, **evaluated}
    return summary, records


def markdown(summary):
    lines = ["# Fixed crop context and scale comparison", "",
        "The frozen reviewed-label model was run on the same 20 development photographs. "
        "The fresh full-image control passed the saved CPU matching and corner-outcome checks.", "",
        "Each candidate adds the same two overlapping full-height crops. The first keeps the original image scale. "
        "The second enlarges each crop by 5/3. Artificial-edge boxes are rejected before whole-record suppression.", "",
        "The first candidate changes context and adds candidate exposure. The second adds enlargement to the same crop geometry.", "",
        "| Reference | Method | Missed | Extras | Duplicates | Both corners within 5 px | Both within 10 px | Empty-scene outputs |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["matrix"]:
        lines.append(f"| {row['reference']} | {row['method']} | {row['missed']} | {row['extras']} | {row['duplicates']} | "
                     f"{row['within5']}/{row['eligible_pairs']} | {row['within10']}/{row['eligible_pairs']} | {row['empty_scene_outputs']} |")
    lines += ["", "## Paired changes under r67", ""]
    for variant in VARIANTS[1:]:
        row = summary["paired_changes"][variant]["r67"]
        lines.append(f"- {variant}: {row['detection_gain']} recovered fronts and {row['detection_loss']} newly missed fronts. "
                     f"Five-pixel pair gains/losses: {row['pair_within5_gain']}/{row['pair_within5_loss']}. "
                     f"Ten-pixel pair gains/losses: {row['pair_within10_gain']}/{row['pair_within10_loss']}.")
    lines += ["", "The package contains every native crop record, pre-clamp corner coordinate, boundary decision, "
        "mapped point, pooled source, suppression decision and paired target outcome. "
        "All four references use fresh matching. This comparison does not train a model or select a deployment default.", ""]
    return "\n".join(lines)


def write_outputs(root, output, summary, records):
    output = Path(output)
    payload = {"summary.json": encoded(summary), "results.md": markdown(summary).encode(),
               "protocol.json": (Path(root) / PROTOCOL_PATH).read_bytes()}
    payload.update({name + ".json.gz": gzip.compress(encoded(value), mtime=0) for name, value in records.items()})
    table = io.StringIO(newline="")
    rows = []
    for row in summary["matrix"]:
        flat = {k: v for k, v in row.items() if k != "known_corner_groups"}
        for known, values in row["known_corner_groups"].items():
            flat.update({f"known{known}_{key}": value for key, value in values.items()})
        rows.append(flat)
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payload["matrix.csv"] = table.getvalue().encode()
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()})
    for name, data in payload.items():
        current.frozen(output / name, data)


def package_results(output, source_commit):
    need(re.fullmatch("[0-9a-f]{40}", source_commit) is not None, "A complete source commit is required")
    output = Path(output)
    inventory = json.loads((output / "inventory.json").read_bytes())
    payload = {name: (output / name).read_bytes() for name in inventory}
    for name, spec in inventory.items():
        need(spec == {"bytes": len(payload[name]), "sha256": digest(payload[name])}, "Saved result differs: " + name)
    summary = json.loads(payload["summary.json"])
    payload["source.json"] = encoded({"source_commit": source_commit,
        "analysis_source_sha256": summary["analysis_source_sha256"], "protocol_sha256": summary["protocol_sha256"],
        "checkpoint": summary["checkpoint"], "fresh_inference": True, "training": False, "optimizer_updates": 0})
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()})
    path = output / RESULT_NAME
    current.fixed_zip(path, payload)
    need(verified_zip(path) == payload, "Final result package verification differs")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output.parent / "runtime-config"))
    plan = prepare(root, args.inputs)
    summary, records = analyze(root, args.inputs, args.output, plan)
    write_outputs(root, args.output, summary, records)
    if args.source_commit:
        package_results(args.output, args.source_commit)
    print(json.dumps({"status": summary["status"], "control_gate_passed": summary["control_gate_passed"],
                      "record_counts": summary["record_counts"], "crop_counts": summary["crop_counts"]}, indent=2))


if __name__ == "__main__":
    main()
