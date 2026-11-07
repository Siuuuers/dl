#!/usr/bin/env python3
"""Compare a fixed epoch173 NMS/confidence grid on the frozen development reference."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

from evaluate_pose import (
    INFERENCE, TABLE_FIELDS, _prediction_records, _runtime_and_model,
    _source_identity, create_output_directory, file_sha256, preflight_dataset,
)
from diagnose_pose_quality import aggregate_quality, quality_for_image
from pose_scoring import aggregate_scores, score_image


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "baomai-fixed-postprocess-v1"
IDENTITY = {
    "model_sha256": "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e",
    "manifest_sha256": "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926",
    "model_card_sha256": "636f0952e9f36e7e26b091bd0ab9b857f841317673507611e998af1ada719566",
}
GATE_FIELDS = ("fn", "fp", "missing_endpoints", "conditional_median_error_px", "conditional_p95_error_px")
PLAN = {
    "nms_ious": [0.70, 0.60, 0.68, 0.50],
    "box_confidences": [0.15, 0.25, 0.40],
    "reference_images": 20,
    "expected_prediction_records": 240,
    "baseline": {"nms_iou": 0.70, "threshold": 0.25},
    "primary_candidate": {"nms_iou": 0.60, "threshold": 0.25},
    "secondary_candidates": [{"nms_iou": 0.68, "threshold": 0.25}, {"nms_iou": 0.50, "threshold": 0.25}],
    "matching_iou": 0.50,
    "endpoint_confidence": 0.50,
    "illustrative_radii_px": [2, 5, 10],
    "gate_metrics": list(GATE_FIELDS),
    "gate": "All five unrounded metrics finite and no worse than baseline, with at least one strictly better.",
    "scope": "Development comparison; other box confidences are sensitivity analyses. No automatic adoption.",
    "nms_068_reason": "Added before fresh inference after a saved-baseline overlap audit identified a possible lost true positive at 0.60.",
    "fresh_inference": "Every image/setting pair is inferred separately; never filter already-suppressed predictions to construct another cell.",
}
IMAGE_FIELDS = ("id", "image", "label", "image_sha256", "label_sha256", "width", "height")
CSV_FIELDS = ("nms_iou", "threshold", *TABLE_FIELDS, "precision", "recall", "f1", "fn_plus_fp")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_identity():
    source = _source_identity(ROOT)
    for relative in ("scripts/experiment_pose_postprocess.py", "scripts/diagnose_pose_quality.py"):
        source["files_sha256"][relative] = file_sha256(ROOT / relative)
    return source


def reference_images(preflight):
    return [{field: row[field] for field in IMAGE_FIELDS} for row in preflight["records"]]


def fixed_preflight(root, require_weights):
    preflight = preflight_dataset(root, require_weights=require_weights)
    if any(preflight[field] != expected for field, expected in IDENTITY.items()):
        raise ValueError("experiment requires the exact frozen epoch173 model card and development manifest")
    if len(preflight["records"]) != PLAN["reference_images"]:
        raise ValueError("experiment requires all 20 development images")
    return preflight


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_prediction(item, records):
    if not isinstance(item, dict) or item.get("record_type") != "prediction":
        raise ValueError("expected a prediction record")
    nms, threshold, identifier = item["nms_iou"], item["threshold"], item["image_id"]
    if (not finite_number(nms) or nms not in PLAN["nms_ious"]
            or not finite_number(threshold) or threshold not in PLAN["box_confidences"]
            or identifier not in records):
        raise ValueError("prediction is outside the declared image/settings grid")
    record = records[identifier]
    for field in ("image_sha256", "label_sha256", "width", "height"):
        if item[field] != record[field] or isinstance(item[field], bool):
            raise ValueError(f"prediction {field} differs from reference: {identifier}")
    if item.get("inference") != {**INFERENCE, "iou": nms, "conf": threshold}:
        raise ValueError("per-record inference settings differ from the declared cell")
    if not isinstance(item["predictions"], list):
        raise ValueError("predictions must be a list, including for empty scenes")
    for predicted in item["predictions"]:
        if not isinstance(predicted, dict) or predicted.get("class_id") != 0:
            raise ValueError("unexpected predicted class/object")
        confidence = predicted.get("confidence")
        if not finite_number(confidence) or not threshold < confidence <= 1:
            raise ValueError("predicted confidence is incompatible with its cell threshold")
    times = [item["stage_ms"][stage] for stage in ("preprocess", "inference", "postprocess")]
    times.append(item["predict_call_wall_ms"])
    if any(not finite_number(value) or value < 0 for value in times):
        raise ValueError("timings must be finite nonnegative milliseconds")
    score_image(record["gold"], item["predictions"], record["width"], record["height"],
                PLAN["matching_iou"], PLAN["endpoint_confidence"])
    return nms, threshold, identifier


def read_replay(path, preflight):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard JSON value: {value}")

    lines = [json.loads(line, object_pairs_hook=unique_keys, parse_constant=reject_constant)
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2 or not all(isinstance(row, dict) for row in lines):
        raise ValueError("replay requires a run header, predictions and completion footer")
    header, items, footer = lines[0], lines[1:-1], lines[-1]
    if (header.get("record_type") != "run" or header.get("schema") != SCHEMA
            or header.get("schema_version") != 1 or header.get("plan") != PLAN
            or header.get("inference_base") != INFERENCE):
        raise ValueError("replay schema or fixed experiment plan differs")
    for field, expected in IDENTITY.items():
        if header.get(field) != expected or preflight[field] != expected:
            raise ValueError(f"replay {field} differs from the fixed reference")
    if header.get("images") != reference_images(preflight):
        raise ValueError("replay reference image inventory differs")
    current = source_identity()
    original = header.get("source", {})
    if (not isinstance(original, dict) or original.get("files_sha256") != current["files_sha256"]
            or not isinstance(original.get("revision"), str) or len(original["revision"]) != 40):
        raise ValueError("replay requires the recorded exact inference/scoring source bytes and revision")
    runtime = header.get("runtime", {})
    if (not isinstance(runtime, dict) or runtime.get("backend_end2end") is not False
            or runtime.get("backend_fp16") is not False or runtime.get("loaded_kpt_shape") != [2, 3]
            or runtime.get("ultralytics") != "8.4.182"):
        raise ValueError("replay runtime does not record ordinary-NMS FP32 two-corner inference")
    if header.get("warmup") != "one explicit zero-image prediction excluded from all timings":
        raise ValueError("replay warmup declaration differs")
    expected_count = PLAN["expected_prediction_records"]
    if (footer.get("record_type") != "completion" or footer.get("status") != "complete"
            or footer.get("prediction_records") != expected_count or len(items) != expected_count):
        raise ValueError("replay is not a completed full-grid inference")
    records = {row["id"]: row for row in preflight["records"]}
    seen = set()
    for item in items:
        key = validate_prediction(item, records)
        if key in seen:
            raise ValueError("duplicate image/settings pair in replay")
        seen.add(key)
    expected = {(nms, threshold, identifier) for nms in PLAN["nms_ious"]
                for threshold in PLAN["box_confidences"] for identifier in records}
    if seen != expected:
        raise ValueError("replay does not cover the complete image/settings grid")
    return header, items, footer


def infer(preflight, device, threads, output, header):
    import numpy as np
    import torch

    model, runtime = _runtime_and_model(preflight, device, threads)
    first = preflight["records"][0]
    model.predict(source=np.zeros((first["height"], first["width"], 3), dtype=np.uint8),
                  device=device, conf=0.25, **INFERENCE)
    backend = model.predictor.model
    if backend.end2end or backend.fp16:
        raise RuntimeError("experiment requires ordinary NMS and FP32")
    torch.set_num_threads(threads)
    runtime.update(device_actual=str(backend.device), backend_end2end=bool(backend.end2end),
                   backend_fp16=bool(backend.fp16), torch_threads=torch.get_num_threads())
    header.update(runtime=runtime, warmup="one explicit zero-image prediction excluded from all timings")
    records = {row["id"]: row for row in preflight["records"]}
    items = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(header, allow_nan=False) + "\n")
        stream.flush()
        for nms in PLAN["nms_ious"]:
            for threshold in PLAN["box_confidences"]:
                cell_count = 0
                for record in preflight["records"]:
                    settings = {**INFERENCE, "iou": nms, "conf": threshold}
                    started = time.perf_counter()
                    results = model.predict(source=record["image_path"], device=device, **settings)
                    duration = (time.perf_counter() - started) * 1000
                    if len(results) != 1 or tuple(results[0].orig_shape) != (record["height"], record["width"]):
                        raise RuntimeError(f"prediction image count or dimensions changed: {record['id']}")
                    actual = vars(model.predictor.args)
                    if any(actual.get(name) != value for name, value in settings.items()):
                        raise RuntimeError("actual predictor settings differ from the requested cell")
                    result = results[0]
                    item = {"record_type": "prediction", "nms_iou": nms, "threshold": threshold,
                            "image_id": record["id"], "inference": settings,
                            **{field: record[field] for field in ("image_sha256", "label_sha256", "width", "height")},
                            "predictions": _prediction_records(result),
                            "stage_ms": {key: float(value) for key, value in result.speed.items()},
                            "predict_call_wall_ms": duration}
                    validate_prediction(item, records)
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
                    stream.flush()
                    items.append(item)
                    cell_count += len(item["predictions"])
                print(f"NMS IoU={nms:g}, box confidence={threshold:g}: {len(records)} images, {cell_count} predictions", flush=True)
        stream.write(json.dumps({"record_type": "completion", "status": "complete", "prediction_records": len(items)}) + "\n")
    return items


def gate_result(candidate, baseline):
    comparisons = []
    for field in GATE_FIELDS:
        value, reference = candidate[field], baseline[field]
        finite = finite_number(value) and finite_number(reference)
        comparisons.append({"metric": field, "baseline": reference, "candidate": value,
                            "finite": finite, "no_worse": finite and value <= reference,
                            "strictly_better": finite and value < reference})
    passed = all(row["no_worse"] for row in comparisons) and any(row["strictly_better"] for row in comparisons)
    return {"passes_declared_gate": passed, "comparisons": comparisons, "automatic_adoption": False}


def reports(preflight, items, header, output):
    records = {record["id"]: record for record in preflight["records"]}
    per_image = []
    for item in items:
        record = records[item["image_id"]]
        score = score_image(record["gold"], item["predictions"], record["width"], record["height"],
                            PLAN["matching_iou"], PLAN["endpoint_confidence"])
        quality = quality_for_image(record["gold"], item["predictions"], record["width"], record["height"],
                                    PLAN["illustrative_radii_px"])
        per_image.append({"image_id": item["image_id"], "nms_iou": item["nms_iou"], "threshold": item["threshold"],
                          **score, "quality": quality})
    metrics = []
    for nms in PLAN["nms_ious"]:
        for threshold in PLAN["box_confidences"]:
            rows = [row for row in per_image if row["nms_iou"] == nms and row["threshold"] == threshold]
            metric = aggregate_scores(rows)
            metric.pop("endpoint_errors_px")
            tp, fp, fn = metric["tp"], metric["fp"], metric["fn"]
            metric.update(nms_iou=nms, threshold=threshold,
                          precision=tp / (tp + fp) if tp + fp else None,
                          recall=tp / (tp + fn) if tp + fn else None,
                          f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                          fn_plus_fp=fn + fp, quality=aggregate_quality([row["quality"] for row in rows]))
            metrics.append(metric)
    by_cell = {(row["nms_iou"], row["threshold"]): row for row in metrics}
    baseline = by_cell[(PLAN["baseline"]["nms_iou"], PLAN["baseline"]["threshold"])]
    comparisons = [{"role": role, **candidate,
                    **gate_result(by_cell[(candidate["nms_iou"], candidate["threshold"])], baseline)}
                   for role, candidate in [("primary", PLAN["primary_candidate"]),
                                            *(("secondary", candidate) for candidate in PLAN["secondary_candidates"])]]
    summary = {"schema": SCHEMA, "schema_version": 1, "plan": PLAN, "run": header,
               "metrics": metrics, "candidate_comparisons": comparisons,
               "prediction_file_sha256": file_sha256(output / "predictions.jsonl"),
               "interpretation": "Same development reference already used for model selection; no independent-test or physical-accuracy claim. FN+FP is descriptive equal-cost error, not the adoption gate."}
    write_json(output / "summary.json", summary)
    write_json(output / "per_image.json", per_image)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--replay", type=Path, help="completed predictions.jsonl; no machine-learning runtime required")
    args = parser.parse_args(argv)
    output = None
    try:
        if args.threads <= 0:
            raise ValueError("threads must be positive")
        preflight = fixed_preflight(args.root, require_weights=not bool(args.replay))
        if args.replay:
            header, items, footer = read_replay(args.replay, preflight)
        else:
            header = {"record_type": "run", "schema": SCHEMA, "schema_version": 1,
                      "created_at_utc": datetime.now(timezone.utc).isoformat(),
                      "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
                      "source": source_identity(), "plan": PLAN, "inference_base": INFERENCE,
                      **IDENTITY, "images": reference_images(preflight)}
        output = create_output_directory(args.output)
        write_json(output / "run_status.json", {"status": "running", "exit_code": None, "plan": PLAN})
        if args.replay:
            with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row in (header, *items, footer):
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            write_json(output / "replay.json", {"source_path": str(args.replay.resolve()),
                       "source_sha256": file_sha256(args.replay), "replay_source": source_identity(),
                       "created_at_utc": datetime.now(timezone.utc).isoformat()})
        else:
            items = infer(preflight, args.device, args.threads, output, header)
        summary = reports(preflight, items, header, output)
        write_json(output / "run_status.json", {"status": "complete", "exit_code": 0,
                   "scored_image_settings_pairs": len(items), "model_sha256": IDENTITY["model_sha256"]})
        print(json.dumps({"status": "complete", "output": str(output.resolve()),
                          "candidate_comparisons": summary["candidate_comparisons"]}, indent=2, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if output is not None:
            write_json(output / "run_status.json", {"status": "failed", "exit_code": 1,
                       "error_type": type(error).__name__, "error": str(error)})
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
