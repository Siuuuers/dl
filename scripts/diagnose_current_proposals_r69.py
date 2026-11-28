#!/usr/bin/env python3
"""Collect and diagnose the frozen PR158 reviewed checkpoint's complete proposals."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED

import numpy as np

sys.dont_write_bytecode = True
from analyze_partial_detection_r68 import (
    SOURCES as PREVIOUS_SOURCES, FOCUS_IDS, REFERENCES, geometry, known_corner_outcomes,
)
import analyze_proposal_stages_r63 as stages
from analyze_duplicate_audit_r66 import digest, encoded, need, read
from evaluate_pose import INFERENCE, _prediction_records, parse_pose_labels
from pose_checkpoint import check_loaded_pose_model
from pose_scoring import _box_iou
from predict_pose_folder import postclip_nms
from replay_partial_inclusion_r67 import PROVENANCE_PATH, REFERENCE_PATH, candidate_reference, matrix_row
from run_corner_average_r58 import average_predictions, report
from run_flip_inference_r56 import inverse_horizontal_predictions
from run_reference_matrix_r60 import verified_zip
from run_training_label_comparison_r61 import score_native


PROTOCOL_PATH = "docs/thesis/evidence/current-proposals-r69/protocol.json"
PROTOCOL_SHA = "86a1808314962c2ced91ea6d0af757dac6adb6ff05b02149dae3f6bd0a3108f8"
R68_PATH = "docs/thesis/evidence/partial-detection-r68/baomai-partial-detection-r68-results.zip"
COMPANION_SCRIPT = "analyze_corner_suppression_r69.py"
COMPANION_PROTOCOL = "docs/thesis/evidence/corner-suppression-r69/protocol.json"
RETURN_COMMIT = "6bde3b7a759604d605965d468e6102bf99bf7946"
RETURN_DIRECTORY = "results/l4/baomai-training-label-comparison-r61-l4-001/"
IMAGE_COMMIT = "ea4ac955f2dfaf1add24c243599715f802eb41a7"
MODEL_SHA = "c749fcff2126467cbc770f75a71fd79da51ebd08308ba909a448c3dd8cb5b372"
MODEL_BYTES = 21704814
INPUTS = {
    "pr158": {"filename": "training-labels-pr158.zip", "bytes": 2038950,
        "sha256": "33dbf73b0d8e145fb0aa74184b16a306d148f0b177787349b13465e7b77c3974",
        "commit": RETURN_COMMIT, "repository_path": RETURN_DIRECTORY + "baomai-training-label-comparison-r61-results.zip"},
    "r68": {"filename": "partial-detection-r68.zip", "bytes": 2289832,
        "sha256": "befe5da298de4432e74e90c8fbac0867bdf001493475679f40efad3f8ec8ad32",
        "repository_path": R68_PATH},
    "return_manifest": {"filename": "pr158-return-manifest.json", "bytes": 2378,
        "sha256": "fdba6b57af71eac70484a08dfb7654dd3823261e2165cbd1c8b54a1f7fc59204",
        "commit": RETURN_COMMIT, "repository_path": RETURN_DIRECTORY + "return-manifest.json"},
}
NATIVE_SOURCES = {
    "nn/modules/head.py": "b471bb8a39cc1e38672c20feaa8c28bdf43739adb8eafd14226807b34b31f3ee",
    "nn/tasks.py": "1048ab8db7504e7c57a77d05bfbd855e9de0ef477e15a5aae8cdb7e8a958246a",
    "models/yolo/pose/predict.py": "8f0ce1afb26df2d30c38c5e7a550377cfa23c334cb0effba6927672779335c3e",
    "models/yolo/detect/predict.py": "1711eaacf6f577d5277e0933a5437881e2c3ba833d13358d3dc4244424b5edc2",
    "engine/predictor.py": "9b29a5322293167a705a6d6054ef653a62d27f6e4bff589b242946df89b5b62f",
    "utils/nms.py": "386cd0e3e38fddef5d434fbcfd58264bedfa37295534e9a2b7174a3436d4db62",
    "utils/ops.py": "499672757c9463187ce2b354585a9ada28325e57178f4bc3307d9548bd552089",
    "utils/tal.py": "7fc3127e4b094247ccbbcf8af9829972295f96fc53027313ef51b076d27274cd",
    "nn/autobackend.py": "62e7d83beea56911691d58dacca3c9f20319920fbb2e2b07309c963946b707d6",
    "data/augment.py": "802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3",
    "cfg/default.yaml": "b91e4622e293270595e6e58d5bc3f3cdcb4a1d4f8b28c9035ec883bb2835081f",
}
SOURCES = ("diagnose_current_proposals_r69.py", *PREVIOUS_SOURCES, COMPANION_SCRIPT)
VIEWS = ("original", "reflected")
OUTPUT_STAGES = ("native_original", "native_reflected", "postclip_original", "postclip_reflected", "two_view")
RESULT_NAME = "baomai-current-proposals-r69-results.zip"
RAW_NAME = "decoded-proposals.npz"


def checked(path, sha256, size=None):
    data = path.read_bytes()
    need(digest(data) == sha256 and (size is None or len(data) == size), "Input differs: " + str(path))
    return data


def frozen(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    need(not path.exists() or path.read_bytes() == data, "Different existing output preserved: " + str(path))
    path.write_bytes(data)


def obtain(url, path, sha256, size=None):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        need(digest(data) == sha256 and (size is None or len(data) == size), "Downloaded input differs: " + path.name)
        frozen(path, data)
    return checked(path, sha256, size)


def prepare(root, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, spec in INPUTS.items():
        path = directory / spec["filename"]
        if name == "r68" and not path.exists():
            frozen(path, checked(root / R68_PATH, spec["sha256"], spec["bytes"]))
        elif name != "r68":
            obtain("https://raw.githubusercontent.com/Siuuuers/dl/" + spec["commit"] + "/" + spec["repository_path"],
                   path, spec["sha256"], spec["bytes"])
        checked(path, spec["sha256"], spec["bytes"])
    manifest = json.loads((directory / INPUTS["return_manifest"]["filename"]).read_bytes())
    model = manifest["models"]["reviewed_labels"]
    need(model["model_bytes"] == MODEL_BYTES and model["model_sha256"] == MODEL_SHA, "Returned checkpoint differs")
    weights = directory / model["model_file"]
    if not weights.exists():
        parts = []
        for spec in model["parts"]:
            wrapper = obtain("https://raw.githubusercontent.com/Siuuuers/dl/" + RETURN_COMMIT + "/" + RETURN_DIRECTORY + spec["file"],
                directory / spec["file"], spec["sha256"], spec["bytes"])
            with ZipFile(io.BytesIO(wrapper)) as archive:
                need(archive.namelist() == [spec["member"]] and archive.testzip() is None, "Checkpoint part members/CRC differ")
                data = archive.read(spec["member"])
            need(len(data) == spec["member_bytes"] and digest(data) == spec["member_sha256"], "Checkpoint part differs")
            parts.append(data)
        data = b"".join(parts)
        need(len(data) == MODEL_BYTES and digest(data) == MODEL_SHA, "Reconstructed checkpoint differs")
        frozen(weights, data)
    checked(weights, MODEL_SHA, MODEL_BYTES)
    archive = verified_zip(directory / INPUTS["pr158"]["filename"], INPUTS["pr158"]["sha256"], INPUTS["pr158"]["bytes"])
    plan = read(archive, "plan.json")
    images = sorted((row for row in plan["roles"]["reviewed_labels"] if row["split"] == "val"), key=lambda row: row["id"])
    need(len(images) == 20, "Evaluation photograph count differs")
    def image(row):
        return obtain("https://media.githubusercontent.com/media/Siuuuers/dl/" + IMAGE_COMMIT + "/" + row["image_source"],
                      directory / "images" / (row["id"] + ".png"), row["image_sha256"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(image, images))
    return {"checkpoint": {"filename": model["model_file"], "bytes": MODEL_BYTES, "sha256": MODEL_SHA},
            "archives": INPUTS, "images": images, "model_return_manifest": manifest}


def reference_inputs(root, directory):
    archives = {name: verified_zip(directory / INPUTS[name]["filename"], INPUTS[name]["sha256"], INPUTS[name]["bytes"])
                for name in ("pr158", "r68")}
    text = read(archives["r68"], "reference-labels.json.gz")
    references = {name: {image: parse_pose_labels(value, 1280, 720, name + "/" + image)
                         for image, value in labels.items()} for name, labels in text.items()}
    candidate, candidate_text, lineage, _, hashes = candidate_reference(root, text["r50"])
    need(candidate == references["r67"] and candidate_text == text["r67"], "r67 archived labels differ")
    need(set(references) == set(REFERENCES), "Reference inventory differs")
    plan = read(archives["pr158"], "plan.json")
    for name in ("r50", "supplied664", "frozen666"):
        need(all(plan["references"][name]["labels"][image]["sha256"] == digest(value.encode())
                 for image, value in text[name].items()), "PR158 reference bytes differ: " + name)
    native = read(archives["pr158"], "native-predictions.json.gz")
    measurement, details = score_native({k: references[k] for k in ("r50", "supplied664", "frozen666")}, native, "original_labels")
    need(measurement == read(archives["pr158"], "summary.json")["measurement"], "PR158 archived measurement differs")
    for name, value in details.items():
        need(value == read(archives["pr158"], name + ".json.gz"), "PR158 archived replay differs: " + name)
    prior = read(archives["r68"], "predictions.json.gz")["pr158_reviewed_labels"]["unchanged"]
    need(prior == details["predictions"]["reviewed_labels"]["two_view"], "r68 completed baseline differs")
    prior_scores = read(archives["r68"], "per-image.json.gz")["pr158_reviewed_labels"]
    for name in REFERENCES:
        _, scores, _ = report(references[name], prior)
        need(scores == prior_scores[name]["unchanged"], "r68 baseline scores differ: " + name)
    return references, text, lineage, hashes, native["reviewed_labels"], read(archives["pr158"], "summary.json")


def native_runtime():
    import torch, torchvision, ultralytics
    need(ultralytics.__version__ == "8.4.182", "Pinned Ultralytics8.4.182 required")
    package = Path(ultralytics.__file__).parent
    for name, sha256 in NATIVE_SOURCES.items():
        checked(package / name, sha256)
    tv = Path(torchvision.__file__).parent
    return {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
        "packages": {name: version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "Pillow")},
        "native_sources_sha256": NATIVE_SOURCES,
        "torchvision_source_sha256": {name: digest((tv / name).read_bytes()) for name in ("ops/boxes.py", "_C.so")},
        "cpu_threads": 2, "cpu_interop_threads": 1, "deterministic_algorithms": True,
        "tf32_matmul": False, "tf32_cudnn": False, "native_nms_operator": "torchvision.ops.nms",
        "inference_fusion": "normal YOLO predict AutoBackend setup; frozen fused state checked after warmup"}


def state_hash(model):
    import hashlib
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes())
    return result.hexdigest()


def mapped_records(raw, anchors, view):
    values = stages.prediction_records(raw, stages.input_boxes(raw), anchors)
    return values if view == "original" else inverse_horizontal_predictions(values)


def mapped_boxes(raw, view):
    boxes = stages.original_boxes(stages.input_boxes(raw)).astype(np.float64)
    if view == "reflected":
        left = 1280. - boxes[:, 2]
        boxes[:, 2] = 1280. - boxes[:, 0]
        boxes[:, 0] = left
    return boxes


def fixed_zip(path, payload, stored_names=()):
    output = io.BytesIO()
    with ZipFile(output, "w") as archive:
        for name, data in sorted(payload.items()):
            entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_STORED if name in stored_names else ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    frozen(path, output.getvalue())


def native_rank_trace(boxes, scores, returned_indices):
    """Validate native greedy choices, preserving the operator's observed score ties."""
    active = np.flatnonzero(scores > .25)
    need(len(active) < 30000 and len(returned_indices) < 300, "A reached native cap requires explicit additional attribution")
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    suppressed, ties = {}, []
    for position, chosen in enumerate(returned_indices):
        need(chosen in active and scores[chosen] == np.max(scores[active]), "Native selection is not a current maximum score")
        alternatives = active[scores[active] == scores[chosen]].tolist()
        if len(alternatives) > 1:
            ties.append({"native_position": position, "chosen_anchor_index": chosen,
                         "equal_maximum_score": float(scores[chosen]), "tied_active_anchor_indices": alternatives})
        rest = active[active != chosen]
        intersection = np.prod(np.maximum(np.minimum(boxes[rest, 2:], boxes[chosen, 2:]) -
            np.maximum(boxes[rest, :2], boxes[chosen, :2]), 0), axis=1)
        union = area[chosen] + area[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        for index, value in zip(rest, overlap):
            if value > .70:
                suppressed[int(index)] = {"suppressor": chosen, "pair_iou_fp32": float(value)}
        active = rest[overlap <= .70]
    need(len(active) == 0, "Native suppression witness replay leaves an unexplained survivor")
    return suppressed, ties


def collect(directory, output, plan):
    import cv2
    import torch
    import torchvision
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PosePredictor
    from ultralytics.utils import nms
    runtime = native_runtime()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = directory / plan["checkpoint"]["filename"]
    checked(weights, MODEL_SHA, MODEL_BYTES)

    class TracePredictor(PosePredictor):
        def postprocess(self, preds, image_tensor, original_images, **kwargs):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            self.decoded_trace = (raw.detach().clone(), image_tensor, original_images)
            return super().postprocess(preds, image_tensor, original_images, **kwargs)

    model = YOLO(str(weights))
    check_loaded_pose_model(model)
    model.model.requires_grad_(False)
    with torch.inference_mode():
        model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), predictor=TracePredictor,
                      device="cpu", conf=.25, **INFERENCE)
    predictor = model.predictor
    need(not predictor.model.end2end and not predictor.model.fp16 and str(predictor.model.device) == "cpu",
         "Expected ordinary FP32 CPU pose prediction")
    need(not any(module.training for module in model.model.modules()), "Inference model is not fully in eval mode")
    before_state = state_hash(model.model)
    raw_arrays, raw_members, native, traces = {}, {}, {view: {} for view in VIEWS}, {view: {} for view in VIEWS}
    for row in plan["images"]:
        identity = row["id"]
        path = directory / "images" / (identity + ".png")
        checked(path, row["image_sha256"])
        image = cv2.imread(str(path))
        need(image is not None and image.shape == (720, 1280, 3), "Photograph decoding differs")
        for view, raster in (("original", image), ("reflected", np.ascontiguousarray(image[:, ::-1, :]))):
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.predict(source=raster, device="cpu", conf=.25, **INFERENCE)[0]
            elapsed = time.perf_counter() - started
            raw, tensor, originals = predictor.decoded_trace
            array = raw.cpu().numpy().copy()
            need(array.dtype == np.float32 and array.shape == (1, 11, 19320)
                 and list(tensor.shape) == [1, 3, 736, 1280] and np.isfinite(array).all(), "Decoded/input tensor differs")
            need(np.all((array[0, 4] >= 0) & (array[0, 4] <= 1)), "Decoded score range differs")
            detections, kept = nms.non_max_suppression(raw.clone(), conf_thres=.25, iou_thres=.70,
                classes=None, agnostic=False, max_det=300, nc=1, end2end=False, rotated=False, return_idxs=True)
            anchors = [int(i) for i in kept[0].tolist()]
            primary = _prediction_records(result)
            replayed = _prediction_records(predictor.construct_results([d.clone() for d in detections], tensor, originals)[0])
            need(replayed == primary, "Actual native NMS index reconstruction differs")
            suppressed, ties = native_rank_trace(stages.input_boxes(array), array[0, 4], anchors)
            need(stages.prediction_records(array, stages.input_boxes(array), anchors) == primary,
                 "Independent decoder/coordinate reconstruction differs")
            mapped = primary if view == "original" else inverse_horizontal_predictions(primary)
            final, final_audit = postclip_nms(mapped)
            final_anchors = [anchors[i - 1] for i in final_audit["retained_primary_prediction_ids"]]
            above = np.flatnonzero(array[0, 4] > .25)
            ranked = above[np.argsort(-array[0, 4, above], kind="stable")].tolist()
            stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
            member = view + "/" + identity + ".npy"
            raw_members[member] = stream.getvalue(); raw_arrays[(view, identity)] = array
            native[view][identity] = primary
            traces[view][identity] = {
                "image_sha256": row["image_sha256"], "view_bgr_pixel_sha256": digest(raster.tobytes()),
                "input_tensor_sha256": digest(tensor.cpu().numpy().tobytes()), "input_shape": list(tensor.shape),
                "preprocess": {"source_wh": [1280, 720], "network_wh": [1280, 736], "scale": 1.,
                    "pad_left_top_right_bottom": [0, 8, 0, 8], "channel_order": "BGR raster to RGB tensor", "normalization_divisor": 255},
                "decoded_member": member, "decoded_member_sha256": digest(raw_members[member]),
                "decoded_tensor_sha256": digest(array.tobytes()), "decoded_shape": list(array.shape), "decoded_dtype": str(array.dtype),
                "candidate_id_rule": view + "/" + identity + "/A{zero_based_anchor_index:05d}",
                "above_confidence_count": len(above), "descending_confident_anchor_indices": ranked,
                "descending_order_ties": "ascending anchor index for display only; native operator tie choices are recorded separately",
                "equal_score_candidate_groups": [np.flatnonzero(array[0, 4] == value).tolist()
                    for value, count in zip(*np.unique(array[0, 4, above], return_counts=True)) if count > 1],
                "native_equal_maximum_choices": ties,
                "native_anchor_indices": anchors, "native_suppression": {str(i): value for i, value in suppressed.items()},
                "native_record_reconstruction_exact": True, "independent_native_choice_and_suppression_witness_validation": True,
                "native_candidate_cap_reached": len(above) >= 30000, "native_output_cap_reached": len(anchors) >= 300,
                "postclip_anchor_indices": final_anchors, "postclip_audit": final_audit,
                "forward_seconds": elapsed,
            }
    after_state = state_hash(model.model)
    need(before_state == after_state, "Frozen inference state changed after warmup")
    checked(weights, MODEL_SHA, MODEL_BYTES)
    runtime.update(frozen_fused_state_sha256=before_state, frozen_fused_state_unchanged=True,
                   optimizer_updates=0, checkpoint_file_unchanged=True)
    import ultralytics
    package = Path(ultralytics.__file__).parent
    runtime["imported_ultralytics_python_sha256"] = {
        str(Path(module.__file__).relative_to(package)): digest(Path(module.__file__).read_bytes())
        for module in sys.modules.values() if getattr(module, "__file__", None)
        and str(module.__file__).endswith(".py") and Path(module.__file__).is_relative_to(package)}
    fixed_zip(output / RAW_NAME, raw_members)
    return native, traces, raw_arrays, runtime


def completed_outputs(native):
    outputs = {name: {} for name in OUTPUT_STAGES}
    links = {}
    for image in native["original"]:
        for view in VIEWS:
            values = native[view][image] if view == "original" else inverse_horizontal_predictions(native[view][image])
            outputs["native_" + view][image] = values
            outputs["postclip_" + view][image] = postclip_nms(values)[0]
        outputs["two_view"][image], links[image] = average_predictions(
            outputs["postclip_original"][image], outputs["postclip_reflected"][image])
    return outputs, links


def candidate_record(raw, trace, view, image, index, target):
    if index is None:
        return None
    p = mapped_records(raw, [index], view)[0]
    native_index = trace["native_anchor_indices"].index(index) if index in trace["native_anchor_indices"] else None
    final_index = trace["postclip_anchor_indices"].index(index) if index in trace["postclip_anchor_indices"] else None
    ranked = trace["descending_confident_anchor_indices"]
    return {"candidate_id": f"{view}/{image}/A{index:05d}", "anchor_index": index,
        "prediction": p, "reference_iou": _box_iou(p["bbox_xyxy"], target["bbox_xyxy"]),
        "confidence_rank_above025_with_anchor_tie_break": ranked.index(index) + 1 if index in ranked else None,
        "native_prediction_index": native_index, "postclip_prediction_index": final_index,
        "known_corner_outcomes": known_corner_outcomes(target, p), **geometry(p, target)}


def target_stages(references, arrays, traces, fresh_outputs):
    rows, graph_rows = [], []
    for view in VIEWS:
        for image in sorted(fresh_outputs["postclip_" + view]):
            raw, trace = arrays[(view, image)], traces[view][image]
            boxes = mapped_boxes(raw, view)
            native_ids, final_ids = trace["native_anchor_indices"], trace["postclip_anchor_indices"]
            native_set, final_set = set(native_ids), set(final_ids)
            for reference in REFERENCES:
                gold = references[reference][image]
                overlaps = stages.iou_matrix(boxes, gold)
                _, scores, _ = report({image: gold}, {image: fresh_outputs["postclip_" + view][image]})
                matches = {m["gold_index"]: m for m in scores[image]["matches"]}
                for gi, target in enumerate(gold):
                    eligible = np.flatnonzero(overlaps[:, gi] >= .5).tolist()
                    above = [i for i in eligible if raw[0, 4, i] > .25]
                    native_eligible = [i for i in eligible if i in native_set]
                    final_eligible = [i for i in eligible if i in final_set]
                    best = int(np.argmax(overlaps[:, gi]))
                    best_iou = float(overlaps[best, gi])
                    if best_iou > 0:
                        tied = np.flatnonzero(overlaps[:, gi] == best_iou).tolist()
                        best = max(tied, key=lambda i: (float(raw[0, 4, i]), -i))
                    else:
                        best = None
                    highest = max(eligible, key=lambda i: (float(raw[0, 4, i]), -i)) if eligible else None
                    stage = "matched" if gi in matches else "no_iou_eligible_decoded_box" if not eligible else \
                        "confidence_gate" if not above else "native_nms" if not native_eligible else \
                        "postclip_suppression" if not final_eligible else "retained_assignment_contention"
                    match = matches.get(gi)
                    matched_anchor = final_ids[match["prediction_index"]] if match else None
                    rows.append({"reference": reference, "image": image, "view": view,
                        "reference_id": f"{reference}/{image}/R{gi + 1:03d}", "reference_index": gi,
                        "known_corners": sum(point[2] == 2 for point in target["keypoints"]), "target": target,
                        "stage": stage, "eligible_decoded_count": len(eligible), "eligible_above_confidence_count": len(above),
                        "eligible_native_count": len(native_eligible), "eligible_postclip_count": len(final_eligible),
                        "eligible_decoded_anchor_indices": eligible, "eligible_above_confidence_anchor_indices": above,
                        "eligible_native_anchor_indices": native_eligible, "eligible_postclip_anchor_indices": final_eligible,
                        "best_iou_candidate": candidate_record(raw, trace, view, image, best, target),
                        "highest_score_eligible_candidate": candidate_record(raw, trace, view, image, highest, target),
                        "matched_candidate": candidate_record(raw, trace, view, image, matched_anchor, target)})
                for stage, indices in (("decoded", range(raw.shape[2])),
                    ("above_confidence", trace["descending_confident_anchor_indices"]),
                    ("native_nms", native_ids), ("postclip", final_ids)):
                    graph_rows.append({"reference": reference, "image": image, "view": view, "stage": stage,
                                       **stages.eligibility_graph(overlaps, indices)})
    return rows, graph_rows


def duplicate_competitions(references, outputs, traces):
    rows = []
    for stage, images in outputs.items():
        view = "reflected" if stage.endswith("reflected") else "original"
        map_name = "native_anchor_indices" if stage.startswith("native") else "postclip_anchor_indices"
        for reference in REFERENCES:
            _, scores, _ = report(references[reference], images)
            for image, score in scores.items():
                values = images[image]
                for pi in score["duplicate_prediction_indices"]:
                    duplicate = values[pi]
                    competitors = []
                    for match in score["matches"]:
                        gi, mi = match["gold_index"], match["prediction_index"]
                        target = references[reference][image][gi]
                        duplicate_iou = _box_iou(duplicate["bbox_xyxy"], target["bbox_xyxy"])
                        if duplicate_iou < .5:
                            continue
                        partner = values[mi]
                        competitors.append({"reference_id": f"{reference}/{image}/R{gi + 1:03d}",
                            "matched_prediction_index": mi, "matched_anchor_index": traces[view][image][map_name][mi],
                            "matched_prediction": partner, "duplicate_reference_iou": duplicate_iou,
                            "matched_reference_iou": match["iou"],
                            "prediction_pair_iou": _box_iou(duplicate["bbox_xyxy"], partner["bbox_xyxy"]),
                            "duplicate_score_minus_matched_score": duplicate["confidence"] - partner["confidence"],
                            "duplicate_known_corner_outcomes": known_corner_outcomes(target, duplicate),
                            "matched_known_corner_outcomes": known_corner_outcomes(target, partner)})
                    need(competitors, "Scored duplicate lacks an eligible matched-reference competitor")
                    rows.append({"reference": reference, "stage": stage, "image": image, "source_view": view,
                        "prediction_index": pi, "anchor_index": traces[view][image][map_name][pi],
                        "prediction": duplicate, "competitors": competitors})
    return rows


def record_difference(before, after):
    same_shape = len(before) == len(after) and all(a["class_id"] == b["class_id"] for a, b in zip(before, after))
    value = {"archived_records": len(before), "fresh_records": len(after), "exact_record_equality": before == after,
             "same_count_and_classes_in_record_order": same_shape,
             "alignment": "output index only; this does not establish physical correspondence if ranking changes"}
    if same_shape:
        coordinate = score = 0.
        changed = []
        for i, (a, b) in enumerate(zip(before, after)):
            if a != b:
                changed.append(i)
            coordinate = max(coordinate, *(abs(x - y) for x, y in zip(a["bbox_xyxy"], b["bbox_xyxy"])))
            score = max(score, abs(a["confidence"] - b["confidence"]))
            for x, y in zip(a["keypoints"], b["keypoints"]):
                coordinate = max(coordinate, abs(x[0] - y[0]), abs(x[1] - y[1]))
                score = max(score, abs(x[2] - y[2]))
        value.update(maximum_coordinate_difference_px=coordinate, maximum_score_difference=score,
                     changed_output_indices=changed)
    return value


def change_flags(before, after, old_index, new_index, old_corners, new_corners, old_prediction, new_prediction):
    flags = {"matched_output_index_changed": old_index != new_index,
        "detection_gain": not before["detected"] and after["detected"],
        "detection_loss": before["detected"] and not after["detected"],
        "pair_within5_changed": before["within5"] != after["within5"],
        "pair_within10_changed": before["within10"] != after["within10"],
        "known_corner_availability_changed": any(a["available"] != b["available"] for a, b in zip(old_corners, new_corners)),
        "known_corner_within5_changed": any(a["within5"] != b["within5"] for a, b in zip(old_corners, new_corners)),
        "known_corner_within10_changed": any(a["within10"] != b["within10"] for a, b in zip(old_corners, new_corners))}
    if old_prediction is not None and new_prediction is not None:
        flags["matched_box_pair_iou"] = _box_iou(old_prediction["bbox_xyxy"], new_prediction["bbox_xyxy"])
        flags["maximum_matched_box_coordinate_difference_px"] = max(
            abs(a - b) for a, b in zip(old_prediction["bbox_xyxy"], new_prediction["bbox_xyxy"]))
    return flags


def decompose_changes(rows):
    keys = ("matched_output_index_changed", "detection_gain", "detection_loss", "pair_within5_changed",
            "pair_within10_changed", "known_corner_availability_changed", "known_corner_within5_changed", "known_corner_within10_changed")
    result = {key: sum(row["change_flags"][key] for row in rows) for key in keys}
    result["index_changes_without_scored_outcome_change"] = sum(row["change_flags"][keys[0]] and
        not any(row["change_flags"][key] for key in keys[1:]) for row in rows)
    paired = [row["change_flags"] for row in rows if "matched_box_pair_iou" in row["change_flags"]]
    result["minimum_matched_box_pair_iou"] = min((row["matched_box_pair_iou"] for row in paired), default=None)
    result["maximum_matched_box_coordinate_difference_px"] = max(
        (row["maximum_matched_box_coordinate_difference_px"] for row in paired), default=None)
    result["scope"] = "Changed index/outcome records only; unchanged-index matched outputs are excluded from these geometric bounds. The same physical front can appear in several stage/reference records."
    return result


def compare_outputs(references, archived, fresh):
    matrix, per_image, measurements, changes, record_rows, comparison_rows = [], {}, {}, [], [], []
    states = {}
    for source, outputs in (("archived_l4", archived), ("fresh_cpu", fresh)):
        per_image[source], measurements[source], states[source] = {}, {}, {}
        for stage, values in outputs.items():
            per_image[source][stage], measurements[source][stage], states[source][stage] = {}, {}, {}
            for reference in REFERENCES:
                measurement, scores, target_states = report(references[reference], values)
                matrix.append({**matrix_row(reference, source + "/" + stage, measurement, target_states, references[reference], values),
                               "source": source, "stage": stage})
                per_image[source][stage][reference] = scores
                measurements[source][stage][reference] = measurement
                states[source][stage][reference] = target_states
    for stage in OUTPUT_STAGES:
        for image in archived[stage]:
            record_rows.append({"stage": stage, "image": image, **record_difference(archived[stage][image], fresh[stage][image])})
        for reference in REFERENCES:
            for b, a in zip(states["archived_l4"][stage][reference], states["fresh_cpu"][stage][reference]):
                image, gi = b["image"], b["reference_index"]
                need((image, gi) == (a["image"], a["reference_index"]), "Cross-runtime reference order differs")
                target = references[reference][image][gi]
                bp = archived[stage][image][b["prediction_index"]] if b["prediction_index"] is not None else None
                ap = fresh[stage][image][a["prediction_index"]] if a["prediction_index"] is not None else None
                bc, ac = known_corner_outcomes(target, bp), known_corner_outcomes(target, ap)
                discrete_before = {k: v for k, v in b.items() if k != "maximum_corner_error_px"}
                discrete_after = {k: v for k, v in a.items() if k != "maximum_corner_error_px"}
                corner_before = [{k: c[k] for k in ("corner_index", "available", "within5", "within10")} for c in bc]
                corner_after = [{k: c[k] for k in ("corner_index", "available", "within5", "within10")} for c in ac]
                if discrete_before != discrete_after or corner_before != corner_after:
                    changes.append({"stage": stage, "reference": reference, "image": image,
                        "reference_id": f"{reference}/{image}/R{gi + 1:03d}", "target": target,
                        "change_flags": change_flags(b, a, b["prediction_index"], a["prediction_index"], bc, ac, bp, ap),
                        "before": {**b, "known_corner_outcomes": bc}, "after": {**a, "known_corner_outcomes": ac}})
            before_scores, after_scores = per_image["archived_l4"][stage][reference], per_image["fresh_cpu"][stage][reference]
            comparison_rows.append({"stage": stage, "reference": reference,
                "matching_and_discrete_score_identities_equal": all(stages.score_identity(before_scores[i]) == stages.score_identity(after_scores[i]) for i in before_scores),
                "exact_score_objects_equal": before_scores == after_scores,
                "target_discrete_changes": sum(r["stage"] == stage and r["reference"] == reference for r in changes)})
    return matrix, per_image, measurements, changes, record_rows, comparison_rows


def duplicate_boundary_changes(references, archived, fresh, per_image, arrays, traces):
    """Explain category-count differences using explicitly diagnostic cross-runtime box associations."""
    rows = []
    for stage in OUTPUT_STAGES:
        view = "reflected" if stage.endswith("reflected") else "original"
        map_name = "native_anchor_indices" if stage.startswith("native") else "postclip_anchor_indices"
        for reference in REFERENCES:
            for image, old_score in per_image["archived_l4"][stage][reference].items():
                new_score = per_image["fresh_cpu"][stage][reference][image]
                if old_score["duplicate_fp"] == new_score["duplicate_fp"]:
                    continue
                old_values, new_values = archived[stage][image], fresh[stage][image]
                old_extras, new_extras = old_score["unmatched_prediction_indices"], new_score["unmatched_prediction_indices"]
                old_duplicates, new_duplicates = set(old_score["duplicate_prediction_indices"]), set(new_score["duplicate_prediction_indices"])
                for ni in new_extras:
                    if not old_extras:
                        continue
                    oi = max(old_extras, key=lambda i: (_box_iou(old_values[i]["bbox_xyxy"], new_values[ni]["bbox_xyxy"]), -i))
                    reverse = max(new_extras, key=lambda i: (_box_iou(old_values[oi]["bbox_xyxy"], new_values[i]["bbox_xyxy"]), -i))
                    if reverse != ni or (oi in old_duplicates) == (ni in new_duplicates):
                        continue
                    old_p, new_p = old_values[oi], new_values[ni]
                    cross = []
                    for gi, target in enumerate(references[reference][image]):
                        before = _box_iou(old_p["bbox_xyxy"], target["bbox_xyxy"])
                        after = _box_iou(new_p["bbox_xyxy"], target["bbox_xyxy"])
                        if (before >= .5) != (after >= .5):
                            cross.append({"reference_id": f"{reference}/{image}/R{gi + 1:03d}", "reference_index": gi,
                                "target": target, "archived_reference_iou": before, "fresh_reference_iou": after})
                    raw, trace = arrays[(view, image)], traces[view][image]
                    overlaps = stages.iou_matrix(mapped_boxes(raw, view), [old_p])[:, 0]
                    closest = int(np.argmax(overlaps))
                    current_anchor = trace[map_name][ni]
                    witness = trace["native_suppression"].get(str(closest))
                    rows.append({"reference": reference, "stage": stage, "image": image, "view": view,
                        "association": "Mutual greatest-IoU association among unmatched outputs; diagnostic, not an archived anchor identity",
                        "archived_anchor_identity_known": False, "archived_output_index": oi, "fresh_output_index": ni,
                        "archived_prediction": old_p, "fresh_prediction": new_p,
                        "archived_category": "duplicate" if oi in old_duplicates else "other_extra",
                        "fresh_category": "duplicate" if ni in new_duplicates else "other_extra",
                        "cross_runtime_box_iou": _box_iou(old_p["bbox_xyxy"], new_p["bbox_xyxy"]),
                        "reference_iou_boundary_crossings": cross, "fresh_selected_anchor_index": current_anchor,
                        "fresh_selected_raw_score": float(raw[0, 4, current_anchor]),
                        "closest_fresh_raw_to_archived_box": {"anchor_index": closest,
                            "box_iou_to_archived_output": float(overlaps[closest]),
                            "prediction": mapped_records(raw, [closest], view)[0], "native_suppression_witness": witness}})
    return rows


def qualified_rule_continuity(root, references, archived, fresh):
    """Apply the companion's unchanged fixed selector to this checkpoint in both runtimes."""
    import analyze_corner_suppression_r69 as rule
    selected, removed, output, per_image, target_states, matrix = {}, {}, {}, {}, {}, []
    variant = "corner_qualified_nms_060"
    for source, values in (("archived_l4", archived["two_view"]), ("fresh_cpu", fresh["two_view"])):
        selected[source], removed[source], output[source], per_image[source], target_states[source] = {}, {}, {}, {}, {}
        for image, predictions in values.items():
            selected[source][image], removed[source][image] = rule.select(predictions, variant)
            output[source][image] = [predictions[i] for i in selected[source][image]]
        for reference in REFERENCES:
            measurement, scores, states = report(references[reference], output[source])
            per_image[source][reference], target_states[source][reference] = scores, states
            matrix.append({**matrix_row(reference, source + "/" + variant, measurement, states, references[reference], output[source]),
                           "source": source, "variant": variant})
    differences, baseline_changes = [], []
    for reference in REFERENCES:
        for b, a in zip(target_states["archived_l4"][reference], target_states["fresh_cpu"][reference]):
            image, gi = b["image"], b["reference_index"]
            bi = selected["archived_l4"][image][b["prediction_index"]] if b["prediction_index"] is not None else None
            ai = selected["fresh_cpu"][image][a["prediction_index"]] if a["prediction_index"] is not None else None
            target = references[reference][image][gi]
            bc = known_corner_outcomes(target, archived["two_view"][image][bi] if bi is not None else None)
            ac = known_corner_outcomes(target, fresh["two_view"][image][ai] if ai is not None else None)
            signature = lambda state, index, corners: (index, state["detected"], state["within5"], state["within10"],
                [(c["available"], c["within5"], c["within10"]) for c in corners])
            if signature(b, bi, bc) != signature(a, ai, ac):
                differences.append({"reference": reference, "image": image, "reference_index": gi,
                    "change_flags": change_flags(b, a, bi, ai, bc, ac,
                        archived["two_view"][image][bi] if bi is not None else None,
                        fresh["two_view"][image][ai] if ai is not None else None),
                    "before": {**b, "original_prediction_index": bi, "known_corner_outcomes": bc},
                    "after": {**a, "original_prediction_index": ai, "known_corner_outcomes": ac}})
        _, _, baseline = report(references[reference], fresh["two_view"])
        for b, a in zip(baseline, target_states["fresh_cpu"][reference]):
            image = b["image"]
            ai = selected["fresh_cpu"][image][a["prediction_index"]] if a["prediction_index"] is not None else None
            if b["prediction_index"] != ai or any(b[k] != a[k] for k in ("detected", "within5", "within10", "maximum_corner_error_px")):
                baseline_changes.append({"reference": reference, "image": image, "reference_index": b["reference_index"],
                    "before": b, "after": {**a, "original_prediction_index": ai}})
    return {"protocol_path": COMPANION_PROTOCOL, "protocol_sha256": digest((root / COMPANION_PROTOCOL).read_bytes()),
        "source_path": "scripts/" + COMPANION_SCRIPT, "source_sha256": digest((root / "scripts" / COMPANION_SCRIPT).read_bytes()),
        "variant": variant, "thresholds_changed": False, "matrix": matrix, "per-image": per_image,
        "selected_indices": selected, "removed": removed,
        "selection_comparisons": [{"image": image, "same_kept_output_indices": selected["archived_l4"][image] == selected["fresh_cpu"][image]}
                                 for image in selected["fresh_cpu"]],
        "cross_runtime_discrete_target_changes": differences, "fresh_baseline_to_rule_target_changes": baseline_changes}


def verify_companion_continuity(records, companion):
    saved = json.loads(gzip.decompress((companion / "per-image.json.gz").read_bytes()))
    for reference in REFERENCES:
        need(records["per-image"]["archived_l4"][reference] ==
             saved[reference]["pr158_reviewed_labels"]["corner_qualified_nms_060"],
             "Companion archived-rule score replay differs")
    return True


def analyze(root, directory, output, plan):
    protocol = json.loads(checked(root / PROTOCOL_PATH, PROTOCOL_SHA))
    references, text, lineage, ref_hashes, archived_native, archived_summary = reference_inputs(root, directory)
    native, traces, arrays, runtime = collect(directory, output, plan)
    archived, archived_links = completed_outputs(archived_native)
    fresh, links = completed_outputs(native)
    stage_rows, graph_rows = target_stages(references, arrays, traces, fresh)
    duplicates = duplicate_competitions(references, fresh, traces)
    matrix, per_image, measurements, changes, numerical, comparisons = compare_outputs(references, archived, fresh)
    boundary_changes = duplicate_boundary_changes(references, archived, fresh, per_image, arrays, traces)
    continuity = qualified_rule_continuity(root, references, archived, fresh)
    focused = {(row["image"], row["candidate_index"]): row for row in lineage if row["stable_id"] in FOCUS_IDS}
    focus_rows = [{**row, "historical_stable_id": focused[(row["image"], row["reference_index"])]["stable_id"]}
                  for row in stage_rows if row["reference"] == "r67" and (row["image"], row["reference_index"]) in focused]
    stage_counts = {reference: {view: dict(Counter(row["stage"] for row in stage_rows
                    if row["reference"] == reference and row["view"] == view)) for view in VIEWS} for reference in REFERENCES}
    both_views = []
    indexed = {(r["reference"], r["image"], r["reference_index"], r["view"]): r for r in stage_rows}
    for reference in REFERENCES:
        _, _, completed_states = report(references[reference], fresh["two_view"])
        for state in completed_states:
            image, gi = state["image"], state["reference_index"]
            original, reflected = (indexed[(reference, image, gi, view)] for view in VIEWS)
            both_views.append({"reference": reference, "reference_id": f"{reference}/{image}/R{gi + 1:03d}",
                "image": image, "reference_index": gi, "known_corners": original["known_corners"],
                "original_stage": original["stage"], "reflected_stage": reflected["stage"],
                "raw_support_in_either_view": original["eligible_decoded_count"] > 0 or reflected["eligible_decoded_count"] > 0,
                "above025_support_in_either_view": original["eligible_above_confidence_count"] > 0 or reflected["eligible_above_confidence_count"] > 0,
                "completed_two_view_state": state})
    raw_path = output / RAW_NAME
    verify_raw_archive(raw_path, traces)
    need(len(stage_rows) == 2 * sum(sum(map(len, ref.values())) for ref in references.values()), "Per-target stage count differs")
    need(len(matrix) == 40 and len(focus_rows) == 6, "Comparison/focused scope differs")
    summary = {"schema": "current-proposals-r69-v1", "status": "completed", "fresh_inference": True,
        "training": False, "optimizer_updates": 0, "labels_changed": False, "threshold_search": False,
        "automatic_promotion": False, "development_only": True, "checkpoint": plan["checkpoint"],
        "source_archives": INPUTS, "image_inputs": [{"id": r["id"], "sha256": r["image_sha256"],
            "source_path": r["image_source"], "source_commit": IMAGE_COMMIT} for r in plan["images"]],
        "protocol_path": PROTOCOL_PATH, "protocol_sha256": PROTOCOL_SHA,
        "analysis_source_sha256": {"scripts/" + name: digest((root / "scripts" / name).read_bytes()) for name in SOURCES},
        "companion_protocol_sha256": continuity["protocol_sha256"],
        "reference_input_sha256": ref_hashes, "runtime": runtime, "archived_runtime": archived_summary["runtime"],
        "decoded_archive": {"filename": RAW_NAME, "bytes": raw_path.stat().st_size, "sha256": digest(raw_path.read_bytes()),
            "arrays": 40, "uncompressed_tensor_bytes": sum(a.nbytes for a in arrays.values()), "quantization": "none; FP32 preserved"},
        "matrix": matrix, "stage_counts": stage_counts, "cross_runtime_comparisons": comparisons,
        "cross_runtime_change_decomposition": decompose_changes(changes),
        "qualified_rule_runtime_check": {"matrix": continuity["matrix"],
            "same_selected_indices_all_images": all(r["same_kept_output_indices"] for r in continuity["selection_comparisons"]),
            "cross_runtime_discrete_target_changes": len(continuity["cross_runtime_discrete_target_changes"]),
            "cross_runtime_change_decomposition": decompose_changes(continuity["cross_runtime_discrete_target_changes"]),
            "fresh_baseline_to_rule_target_changes": len(continuity["fresh_baseline_to_rule_target_changes"])},
        "record_counts": {"forward_views": 40, "per_target_view_stages": len(stage_rows),
            "stage_coverage_graphs": len(graph_rows), "duplicate_output_records": len(duplicates),
            "cross_runtime_discrete_target_changes": len(changes), "cross_runtime_record_comparisons": len(numerical),
            "cross_runtime_duplicate_boundary_records": len(boundary_changes),
            "per_image_score_objects": 800, "combined_view_reference_records": len(both_views)},
        "limitations": [
            "These are 20 repeatedly inspected development photographs, not an independent test set.",
            "Raw tensors describe only the frozen PR158 reviewed-label checkpoint and the recorded fresh CPU runtime. Other checkpoints remain outside this full-stage collection.",
            "The archived L4 and fresh CPU runtimes differ. Numerical records and scored outcomes are compared explicitly; prediction byte identity is not presumed.",
            "Equal confident scores are preserved. Native tied choices are observed and checked against the active maximum score; display ranking uses anchor index only to make tied rows legible, without changing native inference.",
            "A best-overlap candidate is selected using annotations for diagnosis. It may depict an adjacent physical front and is not a deployable ranking or matching rule.",
            "Known-corner errors are recorded only where the annotation supplies that corner; predicted endpoint availability is a separate confidence test.",
            "Fixed-pool support and maximum one-to-one matching are upper bounds under the existing IoU relation. They do not establish recoverable accuracy, a confidence cutoff, or an optimal suppression rule.",
            "First-lost stages explain this computation, not the training cause of a low score or poor box. No training, loss change, label change or optimizer update is performed.",
            "Original and reflected postclip outputs are separately scored to expose view support; the completed two-view method still preserves original boxes and averages only eligible associated corners.",
            "Collection timings are local resource observations, not a detector speed benchmark or cross-hardware comparison.",
            "The notebook's separately packaged corner-suppression companion uses saved R68 predictions and its own fixed protocol; it does not modify these freshly collected tensors.",
        ]}
    records = {"traces": traces, "native-predictions": {"archived_l4": archived_native, "fresh_cpu": native},
        "predictions": {"archived_l4": archived, "fresh_cpu": fresh},
        "associations": {"archived_l4": archived_links, "fresh_cpu": links},
        "reference-labels": text, "reference-lineage": lineage, "target-stages": stage_rows,
        "stage-coverage": graph_rows, "duplicate-competitions": duplicates, "focused-targets": focus_rows,
        "combined-view-support": both_views, "per-image": per_image, "measurements": measurements,
        "cross-runtime-target-changes": changes, "cross-runtime-record-differences": numerical,
        "cross-runtime-duplicate-boundaries": boundary_changes,
        "qualified-rule-runtime-continuity": continuity}
    return summary, records


def verify_raw_archive(path, traces):
    expected = {trace["decoded_member"]: trace for rows in traces.values() for trace in rows.values()}
    with ZipFile(path) as archive:
        need(len(archive.namelist()) == len(expected) == 40 and set(archive.namelist()) == set(expected)
             and archive.testzip() is None, "Decoded archive member inventory/CRC differs")
        for name, trace in expected.items():
            data = archive.read(name)
            need(digest(data) == trace["decoded_member_sha256"], "Encoded raw-array member differs")
            raw = np.load(io.BytesIO(data), allow_pickle=False)
            need(raw.dtype == np.float32 and list(raw.shape) == trace["decoded_shape"]
                 and digest(raw.tobytes()) == trace["decoded_tensor_sha256"], "Saved decoded tensor differs")


def markdown(summary, records):
    lines = ["# Current reviewed-checkpoint proposal stages", "",
        "Fresh CPU inference collected every decoded FP32 proposal from the frozen PR158 reviewed-label checkpoint "
        "for all 20 development photographs in both original and reflected views. The confidence, NMS, matching "
        "and corner thresholds remain unchanged. No optimizer, training or label editing is used.", "",
        "## r67 measurements and runtime comparison", "",
        "| Runtime | Output stage | Misses | Extras | Duplicate subset | Pair ≤5 px | Pair ≤10 px |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["matrix"]:
        if row["reference"] == "r67":
            lines.append(f"| {row['source']} | {row['stage']} | {row['missed']} | {row['extras']} | {row['duplicates']} | "
                         f"{row['within5']}/{row['eligible_pairs']} | {row['within10']}/{row['eligible_pairs']} |")
    decomposition = summary["cross_runtime_change_decomposition"]
    lines += ["", f"The archive/fresh comparison records {summary['record_counts']['cross_runtime_discrete_target_changes']} "
        "target-level index/outcome differences across five output stages and four references. "
        f"Of these, {decomposition['index_changes_without_scored_outcome_change']} are matched-output index changes with unchanged "
        f"detection, known-corner availability and 5/10-pixel statuses. Detection gains/losses are "
        f"{decomposition['detection_gain']}/{decomposition['detection_loss']}; complete-pair five/ten-pixel status changes are "
        f"{decomposition['pair_within5_changed']}/{decomposition['pair_within10_changed']}. "
        "Full numerical differences and per-image scores are retained. A changed record index does not by itself establish a changed physical match.", "",
        "The duplicate-category comparison is separate. Its ledger preserves any unmatched-output reference-IoU boundary crossing "
        "and the corresponding fresh native suppression witness. A closest fresh raw box to an archived output is explicitly "
        "a diagnostic association; archived full decoded anchors remain unavailable.", "", "## Fresh r67 first-lost stages", "",
        "| View | Stage | Fronts |", "|---|---|---:|"]
    for view, counts in summary["stage_counts"]["r67"].items():
        for stage, count in sorted(counts.items()):
            lines.append(f"| {view} | {stage} | {count} |")
    lines += ["", "## Three restored focus targets", "",
        "| Target | View | Stage | Best raw IoU | Highest IoU-eligible score | Eligible raw boxes |",
        "|---|---|---|---:|---:|---:|"]
    for row in records["focused-targets"]:
        best, highest = row["best_iou_candidate"], row["highest_score_eligible_candidate"]
        best_text = f"{best['reference_iou']:.6f}" if best else "No positive overlap"
        score_text = f"{highest['prediction']['confidence']:.8f}" if highest else "None"
        lines.append(f"| {row['reference_id']} | {row['view']} | {row['stage']} | {best_text} | {score_text} | {row['eligible_decoded_count']} |")
    lines += ["", "The best-overlap and highest-scoring eligible raw boxes are separate records. Each includes exact geometry, "
        "candidate identity, stage survival and confidence-qualified errors at known corners. A geometrically eligible proposal "
        "does not automatically supply a useful corner estimate.", "",
        "The duplicate-competition ledger compares each scored duplicate with the matched outputs sharing its eligible reference: "
        "their mutual box overlap, confidence order, individual reference overlaps and known-corner errors remain visible. "
        "Matching IoU 0.50 to the same reference does not imply that their mutual overlap exceeds native NMS 0.70.", "",
        "## Fixed corner-rule runtime continuity", "",
        "The separately frozen corner-qualified suppression rule is also applied unchanged to this checkpoint's "
        "fresh completed two-view outputs. This is a runtime continuity check, not a new threshold comparison.", "",
        "| Runtime | r67 misses | Extras | Duplicate subset | Pair ≤5 px | Pair ≤10 px |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in summary["qualified_rule_runtime_check"]["matrix"]:
        if row["reference"] == "r67":
            lines.append(f"| {row['source']} | {row['missed']} | {row['extras']} | {row['duplicates']} | "
                         f"{row['within5']}/{row['eligible_pairs']} | {row['within10']}/{row['eligible_pairs']} |")
    continuity = summary["qualified_rule_runtime_check"]
    lines += ["", f"All-image selected-index agreement: {continuity['same_selected_indices_all_images']}. "
        f"Cross-runtime discrete target changes after the rule: {continuity['cross_runtime_discrete_target_changes']}. "
        f"Fresh baseline-to-rule target changes across four references: {continuity['fresh_baseline_to_rule_target_changes']}.", "",
        "## Scope and evidence", "",
        "The compressed raw archive contains 40 complete FP32 tensors, without quantization. The trace records preprocessing, "
        "native candidate rank, native suppression witnesses, native returned indices, and postclip lineage. Native outputs are "
        "reconstructed exactly from captured tensors and independently replayed before stage results are interpreted.", ""]
    lines += ["- " + value for value in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_outputs(root, output, summary, records):
    payload = {"summary.json": encoded(summary), "results.md": markdown(summary, records).encode(),
               "protocol.json": (root / PROTOCOL_PATH).read_bytes()}
    for name, value in records.items():
        suffix = ".json" if name == "focused-targets" else ".json.gz"
        payload[name + suffix] = encoded(value) if suffix == ".json" else gzip.compress(encoded(value), mtime=0)
    table = io.StringIO(newline="")
    rows = []
    for row in summary["matrix"]:
        flat = {k: v for k, v in row.items() if k != "known_corner_groups"}
        for known, counts in row["known_corner_groups"].items():
            flat.update({f"known{known}_{key}": value for key, value in counts.items()})
        rows.append(flat)
    writer = csv.DictWriter(table, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload["matrix.csv"] = table.getvalue().encode()
    inventory = {name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()}
    raw = output / RAW_NAME
    inventory[RAW_NAME] = {"bytes": raw.stat().st_size, "sha256": digest(raw.read_bytes())}
    payload["inventory.json"] = encoded(inventory)
    for name, data in payload.items():
        frozen(output / name, data)


def package_results(output, source_commit, companion):
    inventory = json.loads((output / "inventory.json").read_bytes())
    payload = {name: (output / name).read_bytes() for name in [*inventory, "inventory.json"]}
    for name, spec in inventory.items():
        need(spec == {"bytes": len(payload[name]), "sha256": digest(payload[name])}, "Saved raw-diagnostic output differs: " + name)
    companion_manifest = json.loads((companion / "inventory.json").read_bytes())
    for name, spec in companion_manifest["files"].items():
        data = (companion / name).read_bytes()
        need(spec == {"bytes": len(data), "sha256": digest(data)}, "Saved companion output differs: " + name)
        payload["corner-suppression/" + name] = data
    payload["corner-suppression/inventory.json"] = (companion / "inventory.json").read_bytes()
    summary = json.loads(payload["summary.json"])
    payload["source.json"] = encoded({"source_commit": source_commit,
        "raw_diagnostic_source_sha256": summary["analysis_source_sha256"],
        "reference_input_sha256": summary["reference_input_sha256"], "protocol_sha256": PROTOCOL_SHA,
        "companion_inventory_sha256": digest(payload["corner-suppression/inventory.json"]),
        "fresh_inference": True, "training": False, "optimizer_updates": 0})
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)}
        for name, data in payload.items() if name != "inventory.json"})
    path = output / RESULT_NAME
    fixed_zip(path, payload, stored_names=(RAW_NAME,))
    need(verified_zip(path) == payload, "Final package verification differs")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output.parent / "runtime-config"))
    plan = prepare(root, args.inputs)
    summary, records = analyze(root, args.inputs, args.output, plan)
    write_outputs(root, args.output, summary, records)
    print(json.dumps({"status": summary["status"], "record_counts": summary["record_counts"],
                      "stage_counts_r67": summary["stage_counts"]["r67"], "decoded_archive": summary["decoded_archive"]}, indent=2))


if __name__ == "__main__":
    main()
