#!/usr/bin/env python3
"""Find cassette fronts and their two lower corners in a folder of still images."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

from evaluate_pose import INFERENCE, _prediction_records, _runtime_and_model, _source_identity, file_sha256
from pose_checkpoint import POSE_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"
BASELINE_BYTES = 106838542
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CSV_FIELDS = ("image", "status", "width", "height", "prediction_count", "displayed_endpoints", "overlay", "error")


def discover_images(input_dir, output_dir):
    requested_output = Path(output_dir).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f"output already exists: {requested_output}")
    source, output = Path(input_dir).expanduser().resolve(), requested_output.resolve()
    if not source.is_dir():
        raise ValueError(f"input is not a directory: {source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input and output directories must not overlap")
    images = sorted((path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                    key=lambda path: path.relative_to(source).as_posix())
    if not images:
        raise ValueError(f"no PNG or JPEG images found: {source}")
    return source, output, images


def decode_image(path):
    from PIL import Image, ImageOps
    contents = Path(path).read_bytes()
    with Image.open(io.BytesIO(contents)) as image:
        if image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("expected a single-frame PNG or JPEG")
        if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
            raise ValueError(f"unsupported high-bit-depth image mode {image.mode}; use an 8-bit camera image")
        source_width, source_height = image.size
        orientation = int(image.getexif().get(274, 1))
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.load()
    return rgb, {"image_sha256": hashlib.sha256(contents).hexdigest(),
                 "source_width": source_width, "source_height": source_height,
                 "exif_orientation": orientation, "width": rgb.width, "height": rgb.height}


def verify_baseline(root):
    root = Path(root).resolve()
    card_path = root / "metadata/model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if (any(card.get(key) != value for key, value in POSE_CONTRACT.items())
            or card.get("epoch") != 173 or card.get("sha256") != BASELINE_SHA256
            or card.get("size_bytes") != BASELINE_BYTES):
        raise ValueError("model card does not identify the frozen epoch173 cassette model")
    weights = (root / card["checkpoint"]).resolve()
    if not weights.is_relative_to(root) or not weights.is_file():
        raise ValueError(f"missing repository checkpoint: {weights}")
    if weights.stat().st_size != BASELINE_BYTES or file_sha256(weights) != BASELINE_SHA256:
        raise ValueError(f"epoch173 size or SHA-256 mismatch; obtain the original Git LFS bytes: {weights}")
    return {"weights": str(weights), "model_sha256": BASELINE_SHA256,
            "model_card_sha256": file_sha256(card_path), "epoch": 173}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def describe_predictions(raw, width, height, endpoint_conf):
    described = []
    for prediction in raw:
        box, confidence, points = prediction["bbox_xyxy"], prediction["confidence"], prediction["keypoints"]
        if (prediction["class_id"] != 0 or len(box) != 4 or not all(_finite(value) for value in box)
                or box[2] <= box[0] or box[3] <= box[1] or not _finite(confidence)
                or not 0 <= confidence <= 1 or len(points) != 2):
            raise RuntimeError("invalid cassette prediction")
        statuses = []
        for point in points:
            if len(point) != 3 or not _finite(point[2]) or not 0 <= point[2] <= 1:
                raise RuntimeError("invalid predicted endpoint confidence")
            x, y, probability = point
            in_image = _finite(x) and _finite(y) and 0 <= x < width and 0 <= y < height
            passes = probability >= endpoint_conf
            statuses.append({"passes_display_threshold": passes, "in_image": in_image,
                             "displayed": passes and in_image})
        described.append({**prediction, "endpoint_status": statuses})
    return described


def prepare_predictor(weights, device, threads, first_rgb, conf=0.25):
    import numpy as np
    import torch
    model, runtime = _runtime_and_model({"weights": str(weights)}, device, threads)
    model.predict(source=np.zeros((first_rgb.height, first_rgb.width, 3), dtype=np.uint8),
                  device=device, conf=conf, **INFERENCE)
    backend = model.predictor.model
    if backend.end2end or backend.fp16:
        raise RuntimeError("expected ordinary NMS and FP32")
    torch.set_num_threads(threads)
    runtime.update(device_actual=str(backend.device), backend_end2end=bool(backend.end2end),
                   backend_fp16=bool(backend.fp16), torch_threads=torch.get_num_threads())

    def predict(rgb):
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        started = time.perf_counter()
        results = model.predict(source=bgr, device=device, conf=conf, **INFERENCE)
        duration = (time.perf_counter() - started) * 1000
        if len(results) != 1 or tuple(results[0].orig_shape) != (rgb.height, rgb.width):
            raise RuntimeError("prediction image count or dimensions changed")
        result = results[0]
        return {"predictions": _prediction_records(result),
                "stage_ms": {key: float(value) for key, value in result.speed.items()},
                "predict_call_wall_ms": duration}

    return predict, runtime


def _save_overlay(rgb, predictions, path):
    from PIL import ImageDraw, ImageFont
    canvas = rgb.copy()
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=11)
    for number, prediction in enumerate(predictions, 1):
        box = prediction["bbox_xyxy"]
        draw.rectangle(box, outline="#4ce2ff", width=1)
        draw.text((max(0, box[0]), max(0, box[1] - 12)), str(number),
                  font=font, fill="white", stroke_width=1, stroke_fill="black")
        if all(status["displayed"] for status in prediction["endpoint_status"]):
            draw.line((*prediction["keypoints"][0][:2], *prediction["keypoints"][1][:2]), fill="#ffb74c", width=2)
        for point, status, color in zip(prediction["keypoints"], prediction["endpoint_status"], ("#77ed7e", "#ff77cc")):
            if status["displayed"]:
                x, y = point[:2]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        canvas.save(stream, format="PNG")


def _write_status(path, summary):
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if any(not _finite(value) or not 0 <= value <= 1 for value in (args.conf, args.endpoint_conf)):
        raise ValueError("confidence thresholds must be finite numbers in [0, 1]")
    source, output, images = discover_images(args.input, args.output)
    baseline = verify_baseline(ROOT)
    identity = _source_identity(ROOT)
    identity["files_sha256"]["scripts/predict_pose_folder.py"] = file_sha256(Path(__file__))
    summary = {"schema_version": 1, "status": "running", "exit_code": None,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "input_directory": str(source), "output_directory": str(output), **baseline,
               "selected_images": len(images), "succeeded_images": 0, "failed_images": 0,
               "unprocessed_images": 0, "prediction_count": 0, "displayed_endpoints": 0,
               "ground_truth_available": False, "source": identity, "runtime": None,
               "inference": {**INFERENCE, "conf": args.conf}, "endpoint_confidence": args.endpoint_conf,
               "coordinate_system": "Pixels of the EXIF-oriented decoded image, before model resizing; overlays use these same pixels.",
               "prediction_scope": "Ultralytics postprocessed boxes/keypoints, including framework clipping; endpoint confidence is not reference visibility.",
               "warmup": None,
               "timing_scope": "predict call excludes file decoding, overlay rendering and serialization"}
    output.mkdir(parents=True, exist_ok=False)
    _write_status(output / "run_status.json", summary)
    predictor, fatal_error = None, None
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream, \
            (output / "summary.csv").open("x", encoding="utf-8", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for path in images:
            relative = path.relative_to(source)
            item = {"record_type": "prediction", "image": relative.as_posix(), "status": "failed",
                    "attempted": fatal_error is None, "predictions": None, "prediction_count": None,
                    "displayed_endpoints": None}
            rgb = None
            if fatal_error is not None:
                item.update(error_type="RunAborted", error=f"not processed after fatal error: {fatal_error}")
                summary["unprocessed_images"] += 1
            else:
                try:
                    rgb, metadata = decode_image(path)
                    item.update(metadata)
                except Exception as error:
                    item.update(error_type=type(error).__name__, error=f"{relative.as_posix()}: {error}", error_stage="decode")
            if rgb is not None:
                stage = "inference"
                try:
                    if predictor is None:
                        predictor, summary["runtime"] = prepare_predictor(baseline["weights"], args.device, args.threads,
                                                                        rgb, conf=args.conf)
                        summary["warmup"] = "one explicit zero-image prediction excluded from timings"
                    result = predictor(rgb)
                    predictions = describe_predictions(result["predictions"], rgb.width, rgb.height, args.endpoint_conf)
                    overlay = Path("overlays") / relative.with_name(relative.name + ".prediction.png")
                    candidate = {**item, **result, "predictions": predictions, "status": "complete",
                                 "prediction_count": len(predictions), "overlay": overlay.as_posix(),
                                 "displayed_endpoints": sum(status["displayed"] for prediction in predictions
                                                            for status in prediction["endpoint_status"])}
                    json.dumps(candidate, allow_nan=False)
                    stage = "overlay"
                    _save_overlay(rgb, predictions, output / overlay)
                    item = candidate
                except Exception as error:
                    fatal_error = f"{relative.as_posix()}: {error}"
                    item.update(error_type=type(error).__name__, error=fatal_error, error_stage=stage)
            if item["status"] == "complete":
                summary["succeeded_images"] += 1
                summary["prediction_count"] += item["prediction_count"]
                summary["displayed_endpoints"] += item["displayed_endpoints"]
            else:
                summary["failed_images"] += 1
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            writer.writerow(item)
            table.flush()
            print(f"{item['status']}: {item['image']}" +
                  (f" ({item['prediction_count']} predictions)" if item["status"] == "complete" else f" - {item['error']}"), flush=True)
    status = "complete" if summary["failed_images"] == 0 else "partial" if summary["succeeded_images"] else "failed"
    if fatal_error is not None:
        status = "failed"
        summary["fatal_error"] = fatal_error
    summary.update(status=status, exit_code=0 if status == "complete" else 1,
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
    _write_status(output / "run_status.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="PNG/JPEG folder, searched recursively; labels are not read")
    parser.add_argument("--output", required=True, type=Path, help="new output directory outside the input tree")
    parser.add_argument("--device", default="cpu", help="cpu or explicit CUDA device index")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25, help="box confidence threshold")
    parser.add_argument("--endpoint-conf", type=float, default=0.50, help="endpoint display threshold, not physical visibility")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
        print(json.dumps({key: summary[key] for key in ("status", "output_directory", "selected_images", "succeeded_images",
                                                       "failed_images", "prediction_count")}, indent=2))
        return summary["exit_code"]
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
