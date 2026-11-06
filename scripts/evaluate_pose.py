#!/usr/bin/env python3
"""Evaluate the baomai baseline or an explicit completed training bundle; replay saved predictions."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pose_scoring import aggregate_scores, score_image
from pose_checkpoint import check_loaded_pose_model, select_checkpoint


ULTRALYTICS_VERSION = "8.4.182"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COUNT_FIELDS = ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp",
                "visible_endpoints", "returned_endpoints", "missing_endpoints")
TABLE_FIELDS = (*COUNT_FIELDS, "endpoint_coverage", "conditional_median_error_px", "conditional_p95_error_px")
INFERENCE = {"imgsz": 1280, "rect": True, "batch": 1, "nms": True, "iou": 0.70,
             "max_det": 300, "quantize": 32, "save": False, "save_txt": False,
             "save_crop": False, "show": False, "verbose": False}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pose_labels(text, width, height, source="labels"):
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive integers")
    objects = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        location = f"{source}:{line_number}"
        fields = line.split()
        if len(fields) != 11:
            raise ValueError(f"{location}: expected 11 fields")
        try:
            row = [float(value) for value in fields]
        except ValueError as error:
            raise ValueError(f"{location}: every field must be numeric") from error
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{location}: every field must be finite")
        if row[0] != 0 or row[7] not in (0, 2) or row[10] not in (0, 2):
            raise ValueError(f"{location}: class must be 0 and visibility must be 0 or 2")
        if any(not 0 <= row[index] <= 1 for index in (1, 2, 3, 4, 5, 6, 8, 9)):
            raise ValueError(f"{location}: normalized fields must be in [0, 1]")
        if row[3] <= 0 or row[4] <= 0:
            raise ValueError(f"{location}: box dimensions must be positive")
        for start in (5, 8):
            if row[start + 2] == 0 and (row[start] != 0 or row[start + 1] != 0):
                raise ValueError(f"{location}: missing endpoint must be 0 0 0")
        if row[7] == row[10] == 2 and row[5] > row[8]:
            raise ValueError(f"{location}: visible endpoints must follow image-X order")
        cx, cy, box_width, box_height = row[1:5]
        objects.append({
            "class_id": 0,
            "bbox_xyxy": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                          (cx + box_width / 2) * width, (cy + box_height / 2) * height],
            "keypoints": [[row[5] * width, row[6] * height, int(row[7])],
                          [row[8] * width, row[9] * height, int(row[10])]],
        })
    return objects


def _relative_file(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"expected a relative reference path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"reference path escapes its directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"missing reference file: {path}")
    with path.open("rb") as source:
        if source.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer, not downloaded bytes: {path}")
    return path


def preflight_dataset(root, selected_ids=None, require_weights=True):
    root = Path(root).resolve()
    data_root = root / "data"
    card_path = root / "metadata/model_card.json"
    manifest_path = data_root / "manifest.json"
    card = json.loads(card_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (card["task"] != "pose" or card["class_names"] != ["cassette_front"]
            or card["kpt_shape"] != [2, 3] or card["flip_idx"] != [1, 0]
            or manifest["kpt_shape"] != [2, 3] or manifest["model_sha256"] != card["sha256"]):
        raise ValueError("model card and manifest do not describe the same baomai two-corner baseline")
    records = [record for record in manifest["records"] if record["split"] == "val"]
    expected = manifest["split_summary"]["val"]
    if len(records) != 20 or expected["images"] != 20 or card["validation_images"] != 20:
        raise ValueError("the reference baseline requires exactly 20 validation records")
    for key in ("id", "image", "label"):
        if len({record[key] for record in records}) != len(records):
            raise ValueError(f"duplicate validation {key}")
    for key in ("instances", "visible_endpoints", "missing_endpoints"):
        if sum(record[key] for record in records) != expected[key]:
            raise ValueError(f"validation manifest summary mismatch: {key}")
    if sum(record["instances"] == 0 for record in records) != expected["empty_images"]:
        raise ValueError("validation manifest empty-image count mismatch")
    selected = set(selected_ids) if selected_ids is not None else {record["id"] for record in records}
    if not selected or selected - {record["id"] for record in records}:
        raise ValueError("selected image IDs must be a nonempty subset of the validation manifest")
    verified = []
    for record in sorted(records, key=lambda item: item["id"]):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", record["id"]):
            raise ValueError("invalid image ID")
        if record["id"] not in selected:
            continue
        if (Path(record["image"]).parts[:2] != ("images", "val")
                or Path(record["label"]).parts[:2] != ("labels", "val")
                or Path(record["image"]).stem != record["id"] or Path(record["label"]).stem != record["id"]):
            raise ValueError(f"image/label split or stem mismatch: {record['id']}")
        image = _relative_file(data_root, record["image"])
        label = _relative_file(data_root, record["label"])
        for path, key in ((image, "image_sha256"), (label, "label_sha256")):
            if file_sha256(path) != record[key]:
                raise ValueError(f"reference SHA-256 mismatch: {path}")
        with image.open("rb") as source:
            header = source.read(24)
        if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
            raise ValueError(f"invalid original PNG header: {image}")
        width, height = struct.unpack(">II", header[16:24])
        if [width, height] != record["original_size"] or min(width, height) <= 0:
            raise ValueError(f"original image dimensions differ from manifest: {image}")
        gold = parse_pose_labels(label.read_text(), width, height, str(label))
        visible = sum(point[2] == 2 for item in gold for point in item["keypoints"])
        if (len(gold) != record["instances"] or visible != record["visible_endpoints"]
                or 2 * len(gold) - visible != record["missing_endpoints"]):
            raise ValueError(f"label counts differ from manifest: {label}")
        verified.append({**record, "image_path": str(image), "label_path": str(label),
                         "width": width, "height": height, "gold": gold})
    weights = root / card["checkpoint"]
    if require_weights:
        weights = _relative_file(root, card["checkpoint"])
        if weights.stat().st_size != card["size_bytes"] or file_sha256(weights) != card["sha256"]:
            raise ValueError("checkpoint size or SHA-256 differs from the model card")
    return {"root": str(root), "records": verified, "weights": str(weights),
            "model_sha256": card["sha256"], "manifest_sha256": file_sha256(manifest_path),
            "model_card_sha256": file_sha256(card_path), "weights_verified": require_weights,
            "reference_validation_images": len(records)}


def create_output_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _source_identity(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"revision": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
            "files_sha256": {path: file_sha256(root / path) for path in
                             ("scripts/evaluate_pose.py", "scripts/pose_scoring.py", "scripts/pose_checkpoint.py")}}


def _runtime_and_model(preflight, device, threads):
    try:
        version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"install ultralytics=={ULTRALYTICS_VERSION} in an isolated inference environment") from error
    if version != ULTRALYTICS_VERSION:
        raise RuntimeError(f"expected ultralytics=={ULTRALYTICS_VERSION}, found {version}")
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(threads)
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("requested accelerator is unavailable; select --device cpu explicitly")
    model = YOLO(preflight["weights"])
    check_loaded_pose_model(model)
    runtime = {"python": platform.python_version(), "platform": platform.platform(), "device_requested": device,
               "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "torch_threads_requested": threads, "ultralytics": version, "loaded_kpt_shape": [2, 3],
               "packages": {name: importlib.metadata.version(name) for name in
                            ("torchvision", "numpy", "Pillow", "opencv-python")}}
    return model, runtime


def _prediction_records(result):
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    if result.keypoints is None or tuple(result.keypoints.data.shape) != (len(boxes), 2, 3):
        raise RuntimeError("loaded model did not return the expected N x 2 x 3 keypoints")
    points = result.keypoints.data.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    classes = result.boxes.cls.cpu().tolist()
    detections = []
    for box, confidence, class_id, keypoints in zip(boxes, confidences, classes, points):
        if class_id != 0:
            raise RuntimeError("unexpected predicted class")
        keypoints = [[x if math.isfinite(x) else None, y if math.isfinite(y) else None, probability]
                     for x, y, probability in keypoints]
        detections.append({"class_id": int(class_id), "bbox_xyxy": box,
                           "confidence": confidence, "keypoints": keypoints})
    return detections


def _run_inference(preflight, thresholds, device, threads, output, metadata):
    import numpy as np
    import torch
    model, runtime = _runtime_and_model(preflight, device, threads)
    first = preflight["records"][0]
    warmup = np.zeros((first["height"], first["width"], 3), dtype=np.uint8)
    model.predict(source=warmup, device=device, conf=0.25, **INFERENCE)
    backend = model.predictor.model
    if backend.end2end or backend.fp16:
        raise RuntimeError("expected ordinary NMS and FP32, but the loaded predictor differs")
    # Ultralytics selects the device during initial predict and resets CPU threads.
    # Restore the requested value after setup, before any measured prediction.
    torch.set_num_threads(threads)
    runtime.update(device_actual=str(backend.device), backend_end2end=bool(backend.end2end),
                   backend_fp16=bool(backend.fp16), torch_threads=torch.get_num_threads())
    metadata.update(runtime=runtime, warmup="one explicit zero-image prediction excluded from all timings")
    batches = []
    with (output / "predictions.jsonl").open("w") as stream:
        stream.write(json.dumps({"record_type": "run", **metadata}, allow_nan=False) + "\n")
        stream.flush()
        for threshold in thresholds:
            for record in preflight["records"]:
                started = time.perf_counter()
                result = model.predict(source=record["image_path"], device=device, conf=threshold, **INFERENCE)[0]
                wall_ms = (time.perf_counter() - started) * 1000
                if tuple(result.orig_shape) != (record["height"], record["width"]):
                    raise RuntimeError(f"prediction dimensions changed: {record['id']}")
                predictions = _prediction_records(result)
                item = {"record_type": "prediction", "image_id": record["id"], "threshold": threshold,
                        "image_sha256": record["image_sha256"], "label_sha256": record["label_sha256"],
                        "width": record["width"], "height": record["height"], "predictions": predictions,
                        "stage_ms": {key: float(value) for key, value in result.speed.items()},
                        "predict_call_wall_ms": wall_ms}
                stream.write(json.dumps(item, allow_nan=False) + "\n")
                stream.flush()
                batches.append(item)
                print(f"conf={threshold:g} {record['id']}: {len(predictions)} predictions", flush=True)
    return batches


def _read_replay(path, preflight, requested_thresholds):
    def reject_constant(value):
        raise ValueError(f"nonstandard JSON constant in replay: {value}")
    lines = [json.loads(line, parse_constant=reject_constant) for line in Path(path).read_text().splitlines() if line.strip()]
    if (not lines or not isinstance(lines[0], dict) or lines[0].get("record_type") != "run"
            or lines[0].get("schema_version") != 1):
        raise ValueError("replay requires a schema-version-1 run header")
    header, predictions = lines[0], lines[1:]
    for field in ("model_sha256", "manifest_sha256", "model_card_sha256"):
        if header[field] != preflight[field]:
            raise ValueError("replay model/dataset identity differs from the reference")
    if (json.dumps(header.get("checkpoint_identity"), sort_keys=True, allow_nan=False)
            != json.dumps(preflight.get("checkpoint_identity"), sort_keys=True, allow_nan=False)):
        raise ValueError("replay checkpoint provenance differs; select the matching explicit checkpoint manifest")
    if header["inference"] != INFERENCE:
        raise ValueError("replay inference configuration differs from the baseline contract")
    inferred_thresholds = header["box_confidences"]
    if (not isinstance(inferred_thresholds, list) or not inferred_thresholds
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in inferred_thresholds)
            or len(set(inferred_thresholds)) != len(inferred_thresholds)):
        raise ValueError("replay box confidences must be unique finite numbers in [0, 1]")
    thresholds = requested_thresholds if requested_thresholds is not None else inferred_thresholds
    if not thresholds or not set(thresholds).issubset(inferred_thresholds):
        raise ValueError("requested threshold was not actually inferred in the replay")
    records = {record["id"]: record for record in preflight["records"]}
    chosen = []
    seen = set()
    for item in predictions:
        if not isinstance(item, dict) or item.get("record_type") != "prediction":
            raise ValueError("unexpected replay record type")
        key = (item["threshold"], item["image_id"])
        if key[0] not in thresholds or key[1] not in records:
            continue
        if key in seen:
            raise ValueError("duplicate image/threshold in replay")
        record = records[key[1]]
        for field in ("image_sha256", "label_sha256", "width", "height"):
            if item[field] != record[field]:
                raise ValueError(f"replay {field} differs: {key[1]}")
        for prediction in item["predictions"]:
            if not isinstance(prediction, dict) or prediction.get("class_id") != 0:
                raise ValueError("replay contains an unexpected predicted class/object")
            confidence = prediction.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not key[0] < confidence <= 1):
                raise ValueError(f"replay detection confidence is incompatible with its threshold: {key}")
        times = [item["stage_ms"][stage] for stage in ("preprocess", "inference", "postprocess")]
        times.append(item["predict_call_wall_ms"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in times):
            raise ValueError("replay timings must be finite nonnegative milliseconds")
        seen.add(key)
        chosen.append(item)
    if seen != {(threshold, identifier) for threshold in thresholds for identifier in records}:
        raise ValueError("replay is missing an image/threshold, including any empty scenes")
    return header, chosen, thresholds


def _draw_overlay(record, item, score, path, endpoint_conf):
    from PIL import Image, ImageDraw, ImageFont
    original = Image.open(record["image_path"]).convert("RGB")
    image = Image.new("RGB", (original.width, original.height + 88), "#101820")
    image.paste(original)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    missing = set(score["unmatched_gold_indices"])
    extras = set(score["unmatched_prediction_indices"])
    for index, gold in enumerate(record["gold"]):
        color = "#ff4040" if index in missing else "#42e090"
        draw.rectangle(gold["bbox_xyxy"], outline=color, width=2 if index in missing else 1)
        for x, y, visibility in gold["keypoints"]:
            if visibility == 2:
                draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
        if index in missing:
            draw.text(tuple(gold["bbox_xyxy"][:2]), f"miss g{index+1}", fill=color, font=font, stroke_width=1, stroke_fill="black")
    for index, prediction in enumerate(item["predictions"]):
        color = "#ffad33" if index in extras else "#f05bff"
        draw.rectangle(prediction["bbox_xyxy"], outline=color, width=2 if index in extras else 1)
        for x, y, confidence in prediction["keypoints"]:
            if x is not None and y is not None and confidence >= endpoint_conf and 0 <= x < original.width and 0 <= y < original.height:
                draw.line((x-3, y, x+3, y), fill=color, width=2)
                draw.line((x, y-3, x, y+3), fill=color, width=2)
        if index in extras:
            draw.text(tuple(prediction["bbox_xyxy"][:2]), f"extra p{index+1}", fill=color, font=font, stroke_width=1, stroke_fill="black")
    coverage = score["endpoint_coverage"]
    details = f"coverage={coverage:.1%}" if coverage is not None else "coverage=n/a (no visible reference endpoints)"
    lines = [f"{record['id']} | box confidence {item['threshold']:g} | FN {score['fn']} | FP {score['fp']} | {details}",
             "Green circles: reference | Magenta crosses: predictions | Red boxes: missed reference | Orange: extra prediction",
             "Diagnostic overlay only. Corner errors are conditional on matched, returned visible endpoints."]
    for line, text in enumerate(lines):
        draw.text((10, original.height + 8 + line * 24), text, fill="white", font=font)
    image.save(path)


def _write_reports(preflight, items, metadata, output, matching_iou, endpoint_conf, overlay_count):
    records = {record["id"]: record for record in preflight["records"]}
    per_image = []
    for item in items:
        record = records[item["image_id"]]
        score = score_image(record["gold"], item["predictions"], record["width"], record["height"], matching_iou, endpoint_conf)
        per_image.append({"image_id": item["image_id"], "threshold": item["threshold"], **score,
                          "stage_ms": item["stage_ms"], "predict_call_wall_ms": item["predict_call_wall_ms"]})
    summaries = []
    for threshold in metadata["box_confidences"]:
        selected = [row for row in per_image if row["threshold"] == threshold]
        aggregate = aggregate_scores(selected)
        aggregate.pop("endpoint_errors_px")
        aggregate.update(threshold=threshold, mean_stage_ms={
            key: sum(row["stage_ms"][key] for row in selected) / len(selected)
            for key in ("preprocess", "inference", "postprocess")},
            mean_predict_call_wall_ms=sum(row["predict_call_wall_ms"] for row in selected) / len(selected))
        summaries.append(aggregate)
    summary = {**metadata, "matching_iou": matching_iou, "endpoint_confidence": endpoint_conf,
               "quantile_method": "linear interpolation at q * (n - 1) over pooled returned endpoint errors",
               "statistics_scope": "development validation; pixel error conditional on matched returned gold-visible endpoints",
               "timing_scope": "original inference timings; batch=1; explicit warm-up excluded; stage timers exclude file decoding, serialization, overlays and model setup",
               "prediction_scope": "Ultralytics postprocessed original-pixel boxes/keypoints, including framework coordinate clipping",
               "metrics": summaries}
    _json_write(output / "summary.json", summary)
    _json_write(output / "per_image.json", per_image)
    with (output / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_id", "threshold", *TABLE_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_image)
    if overlay_count:
        reference_threshold = 0.25 if 0.25 in metadata["box_confidences"] else metadata["box_confidences"][0]
        ranked = sorted((row for row in per_image if row["threshold"] == reference_threshold),
                        key=lambda row: (-(row["fn"] + row["fp"]), -row["missing_endpoints"],
                                         -(row["conditional_p95_error_px"] or 0), row["image_id"]))
        selected_ids = {row["image_id"] for row in ranked[:overlay_count]}
        overlay_dir = output / "overlays"
        overlay_dir.mkdir()
        for item, row in zip(items, per_image):
            if item["image_id"] in selected_ids:
                path = overlay_dir / f"{item['image_id']}_conf{item['threshold']:g}.png"
                _draw_overlay(records[item["image_id"]], item, row, path, endpoint_conf)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="new output directory; existing paths are refused")
    parser.add_argument("--device", default="cpu", help="cpu or explicit CUDA device index")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads, recorded in the run")
    parser.add_argument("--conf", type=float, nargs="+", help="box confidence values; default 0.15 0.25 0.40")
    parser.add_argument("--endpoint-conf", type=float, default=0.50)
    parser.add_argument("--matching-iou", type=float, default=0.50)
    parser.add_argument("--image", action="append", help="validation image ID for a smoke test; repeatable")
    parser.add_argument("--overlays", type=int, default=5, help="number of representative images per threshold; 0 skips overlays")
    parser.add_argument("--preflight-only", action="store_true", help="verify reference files without importing the model runtime")
    parser.add_argument("--replay", type=Path, help="rescore saved predictions.jsonl without model inference")
    parser.add_argument("--checkpoint-manifest", type=Path,
                        help="verified completed paired-training bundle; omitted uses the frozen epoch173 baseline")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.overlays < 0:
        parser.error("threads must be positive and overlays nonnegative")
    if not math.isfinite(args.matching_iou) or not 0 < args.matching_iou <= 1:
        parser.error("matching IoU must be in (0, 1]")
    if not math.isfinite(args.endpoint_conf) or not 0 <= args.endpoint_conf <= 1:
        parser.error("endpoint confidence must be in [0, 1]")
    if args.conf and (len(set(args.conf)) != len(args.conf) or any(not math.isfinite(c) or not 0 <= c <= 1 for c in args.conf)):
        parser.error("box confidences must be unique finite numbers in [0, 1]")
    if not args.preflight_only and args.output is None:
        parser.error("--output is required except with --preflight-only")
    output = None
    try:
        preflight = preflight_dataset(args.root, args.image,
                                      require_weights=not bool(args.replay or args.checkpoint_manifest))
        if args.checkpoint_manifest:
            preflight = select_checkpoint(preflight, args.checkpoint_manifest, require_weights=not bool(args.replay))
        if args.preflight_only:
            print(json.dumps({key: value for key, value in preflight.items() if key != "records"} |
                             {"verified_images": len(preflight["records"]),
                              "objects": sum(len(record["gold"]) for record in preflight["records"])}, indent=2))
            return 0
        metadata = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": _source_identity(Path(__file__).resolve().parents[1]), "model_sha256": preflight["model_sha256"],
                    "manifest_sha256": preflight["manifest_sha256"], "model_card_sha256": preflight["model_card_sha256"],
                    "inference": INFERENCE, "box_confidences": args.conf or [0.15, 0.25, 0.40],
                    "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                    "images": [{key: record[key] for key in ("id", "image", "label", "image_sha256", "label_sha256", "width", "height")}
                               for record in preflight["records"]]}
        if "checkpoint_identity" in preflight:
            metadata["checkpoint_identity"] = preflight["checkpoint_identity"]
        if args.replay:
            original, items, thresholds = _read_replay(args.replay, preflight, args.conf)
            origin = original.get("replay", {})
            metadata.update(box_confidences=thresholds, runtime=original["runtime"], warmup=original["warmup"],
                            replay={"path": str(args.replay.resolve()), "sha256": file_sha256(args.replay),
                                    "inference_source": origin.get("inference_source", original["source"]),
                                    "inference_created_at_utc": origin.get("inference_created_at_utc", original["created_at_utc"])})
            output = create_output_directory(args.output)
            with (output / "predictions.jsonl").open("w") as stream:
                for item in ({"record_type": "run", **metadata}, *items):
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
        else:
            output = create_output_directory(args.output)
            items = _run_inference(preflight, metadata["box_confidences"], args.device, args.threads, output, metadata)
        summary = _write_reports(preflight, items, metadata, output, args.matching_iou, args.endpoint_conf, args.overlays)
        _json_write(output / "run_status.json", {"status": "complete", "exit_code": 0, "scored_image_threshold_pairs": len(items)})
        print(json.dumps({"output": str(output.resolve()), "metrics": summary["metrics"]}, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        if output is not None:
            _json_write(output / "run_status.json", {"status": "failed", "exit_code": 1,
                                                     "error_type": type(error).__name__, "error": str(error)})
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
