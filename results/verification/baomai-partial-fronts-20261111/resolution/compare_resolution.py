#!/usr/bin/env python3
"""Fixed, offline seed-46 M2 inference comparison: 1280 versus 1920, all 20 images.

This draft does not train, edit labels, search thresholds, or select a production
default. It imports the repository's existing inference and scoring helpers.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

sys.dont_write_bytecode = True

PROTOCOL = {
    "schema_version": 1,
    "experiment_id": "seed46-m2-fixed-resolution-1280-1920-v1",
    "status": "prospectively_declared_single_comparison",
    "model_sha256": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
    "checkpoint_manifest_sha256": "c5f07c4445a7b33a679fd4fc896aaf9643d81b4f72d8ca8e4e7179cb0bc691e8",
    "reference": "data/references/visible-front-2026-11-11",
    "reference_manifest_sha256": "2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c",
    "capsule": "results/evaluation/baomai-annotations-seed46-20261111/evaluation.zip",
    "capsule_sha256": "3dfe3415841b47a605e2fd4b9e490b2c12b348b0014c6c2a8c78c06f323ffc1a",
    "archived_prediction_member": "evidence/seed-46/m2/predictions.jsonl",
    "archived_prediction_sha256": "25b5eabecbffa13c1d01d62194dfbdebdc0fcdf914f0661123c23347c4060ee9",
    "reference_replay": "results/verification/baomai-low-confidence-20261111/reference-replay.json.gz",
    "reference_replay_sha256": "770eea983e325b5d0ac85390b9de17c437228fc4810ae4e0e8d4ac7e3936c32f",
    "image_count": 20, "reference_objects": 663, "visible_reference_endpoints": 1228,
    "sizes_in_fixed_order": [1280, 1920],
    "fresh_model_load_for_each_arm": True,
    "image_order": "ascending image_id in both arms",
    "box_confidence": 0.15, "nms_iou": 0.70, "matching_iou": 0.50,
    "endpoint_confidence": 0.50, "radii_px": [2, 5, 10],
    "device": "cpu", "torch_threads": 4, "precision": "FP32", "ordinary_nms": True,
    "warmup": "one zero-image prediction at that arm's size and confidence, excluded from image timings",
    "coordinates": "original 1280 x 720 image pixels returned by the existing prediction helper",
    "focus_object_ids": ["new_000008:33", "new_000009:57"],
    "focus_hypothesis": "Upsampling may change the network representation of two reviewed partial fronts: a 68.16 x 46.56 px right-clipped target and an 18 x 31 px fragment beside a jaw. It adds no optical information.",
    "primary_comparison": "fresh 1920 against fresh 1280; independently compare fresh 1280 exactly with archived M2 at confidence 0.15",
    "required_outcomes": ["all-object FN/FP and duplicate/other extras", "every gained/lost reference row", "both focus objects", "returned corners and 2/5/10 px success", "each empty image and pooled nonempty/empty results", "per-image stage and wall timings"],
    "decision": "Report gains, losses, corner regressions, and review burden. No automatic promotion or default change. A gain confined to unresolved legacy rows does not establish the stated partial-front hypothesis.",
    "network_controls": "YOLO_OFFLINE=true, YOLO_AUTOINSTALL=false, Python audit rejection of DNS/connect/sendto events; all assets must already exist locally",
    "internal_deadline_seconds": 590,
    "outer_timeout_recommendation": "timeout --kill-after=5s 595s (600 seconds maximum including forced termination)",
    "forbidden_changes": ["labels", "matcher", "endpoint gate", "other resolutions", "threshold/NMS search", "training", "production defaults"],
    "limits": ["Repeated use of the same 20 images is exploratory validation, not independent generalization evidence.", "CPU timing is descriptive and is not a deployment latency limit.", "Fixed arm order and one timing sample per image do not support a rigorous hardware benchmark.", "Eighteen of the 21 archived misses retain documented reference uncertainty; official denominators stay unchanged.", "The prior interrupted size trial has no verified recovered results and is not evidence for this comparison."],
}

SOURCE_HASHES = {
    "scripts/evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "scripts/pose_checkpoint.py": "8c92a452f632aec92bfd51bc8fbf9fc745a09978d840cdb294af0a37dea8696f",
    "scripts/pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "scripts/diagnose_pose_quality.py": "a3baa1285d1580ee44547f7c3d606f524f2f40616fed568885746d2014159953",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def pinned(path, expected):
    require(Path(path).is_file() and sha256(path) == expected, f"SHA-256 mismatch or missing file: {path}")


def disable_network():
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    blocked = []

    def reject(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise PermissionError(f"offline experiment forbids {event}")

    sys.addaudithook(reject)
    return blocked


def load_inputs(root, checkpoint_manifest):
    for relative, expected in SOURCE_HASHES.items():
        pinned(root / relative, expected)
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import (INFERENCE, _prediction_records, _runtime_and_model,
                               _source_identity, parse_pose_labels, preflight_dataset)
    from pose_checkpoint import select_checkpoint
    from pose_scoring import _box_iou, _percentile, aggregate_scores, score_image
    from diagnose_pose_quality import _returned, aggregate_quality, quality_for_image

    api = dict(INFERENCE=INFERENCE, prediction_records=_prediction_records,
               runtime_and_model=_runtime_and_model, source_identity=_source_identity,
               box_iou=_box_iou, percentile=_percentile, score_image=score_image,
               aggregate_scores=aggregate_scores, returned=_returned,
               aggregate_quality=aggregate_quality, quality_for_image=quality_for_image)
    pinned(checkpoint_manifest, PROTOCOL["checkpoint_manifest_sha256"])
    original = preflight_dataset(root, require_weights=False)
    selected = select_checkpoint(original, checkpoint_manifest, require_weights=True)
    require(selected["model_sha256"] == PROTOCOL["model_sha256"], "unexpected checkpoint")
    reference_root = root / PROTOCOL["reference"]
    pinned(reference_root / "manifest.json", PROTOCOL["reference_manifest_sha256"])
    reference = json.loads((reference_root / "manifest.json").read_text())
    files = {item["image_id"]: item for item in reference["files"]}
    require(len(files) == len(reference["files"]) == 20, "reference must contain 20 unique files")
    require(set(files) == {r["id"] for r in original["records"]}, "reference image set differs")
    records = []
    for source in original["records"]:
        item = files[source["id"]]
        require(item["image_sha256"] == source["image_sha256"], "reference image hash differs")
        require([item["width"], item["height"]] == [source["width"], source["height"]] == [1280, 720], "image dimensions differ")
        require(item["label_path"] == f"labels/val/{source['id']}.txt", "reference label mapping differs")
        label = reference_root / item["label_path"]
        pinned(label, item["label_sha256"])
        label_text = label.read_text()
        require(all(line.strip() for line in label_text.splitlines()), "blank physical reference row")
        gold = parse_pose_labels(label_text, source["width"], source["height"], str(label))
        require(len(gold) == item["row_count"], "reference row count differs")
        records.append({**source, "source_label_sha256": source["label_sha256"],
                        "label_path": str(label), "label_sha256": item["label_sha256"], "gold": gold})
    require(sum(len(r["gold"]) for r in records) == 663, "reference object count differs")
    require(sum(p[2] == 2 for r in records for g in r["gold"] for p in g["keypoints"]) == 1228, "reference endpoint count differs")
    require(sum(not r["gold"] for r in records) == 3, "empty image count differs")
    selected["records"] = records
    capsule = root / PROTOCOL["capsule"]
    pinned(capsule, PROTOCOL["capsule_sha256"])
    with zipfile.ZipFile(capsule) as archive:
        member = archive.read(PROTOCOL["archived_prediction_member"])
    require(hashlib.sha256(member).hexdigest() == PROTOCOL["archived_prediction_sha256"], "archived predictions changed")
    saved = [json.loads(line) for line in member.splitlines()]
    header = saved[0]
    require(header["model_sha256"] == selected["model_sha256"], "archive model differs")
    require(header["checkpoint_identity"] == selected["checkpoint_identity"], "archive checkpoint provenance differs")
    require(header["inference"] == INFERENCE, "production 1280 inference settings differ")
    archived = [item for item in saved[1:] if item["threshold"] == PROTOCOL["box_confidence"]]
    require(len(archived) == 20 and {item["image_id"] for item in archived} == set(files), "archived image set differs")
    original_by_id = {record["id"]: record for record in original["records"]}
    for item in archived:
        record = original_by_id[item["image_id"]]
        for key in ("image_sha256", "label_sha256", "width", "height"):
            require(item[key] == record[key], f"archive {key} differs for {item['image_id']}")
    replay_path = root / PROTOCOL["reference_replay"]
    pinned(replay_path, PROTOCOL["reference_replay_sha256"])
    replay = json.loads(gzip.decompress(replay_path.read_bytes()))
    require(replay["references"]["amended"]["manifest_sha256"] == PROTOCOL["reference_manifest_sha256"], "replay reference differs")
    return api, selected, archived, header, replay, sha256(replay_path)


def object_states(gold, predictions, score, api):
    matches = {pair["gold_index"]: pair for pair in score["matches"]}
    states = []
    for index, item in enumerate(gold):
        pair = matches.get(index)
        distances = [None, None]
        returned = [False, False]
        if pair is not None:
            for side, (point, target) in enumerate(zip(predictions[pair["prediction_index"]]["keypoints"], item["keypoints"])):
                returned[side] = api["returned"](point, 1280, 720)
                if target[2] == 2 and returned[side]:
                    distances[side] = math.hypot(point[0] - target[0], point[1] - target[1])
        both = all(point[2] == 2 for point in item["keypoints"])
        states.append({"reference_row_one_based": index + 1, "locatable_corners": sum(point[2] == 2 for point in item["keypoints"]),
                       "matched": pair is not None,
                       "prediction_index_zero_based": pair["prediction_index"] if pair else None,
                       "prediction_index_one_based": pair["prediction_index"] + 1 if pair else None,
                       "matching_iou": pair["iou"] if pair else None,
                       "endpoint_returned": returned, "visible_endpoint_errors_px": distances,
                       "both_visible_within_radius": {str(radius): both and all(d is not None and d <= radius for d in distances) for radius in PROTOCOL["radii_px"]}})
    return states


def score_records(records, items, api):
    by_id = {item["image_id"]: item for item in items}
    require(len(items) == len(by_id) == 20, "expected exactly 20 image predictions")
    per_image = []
    for record in records:
        predictions = by_id[record["id"]]["predictions"]
        score = api["score_image"](record["gold"], predictions, 1280, 720, PROTOCOL["matching_iou"], PROTOCOL["endpoint_confidence"])
        quality = api["quality_for_image"](record["gold"], predictions, 1280, 720, PROTOCOL["radii_px"])
        per_image.append({"image_id": record["id"], "scene": "nonempty" if record["gold"] else "empty",
                          "score": score, "quality": quality,
                          "objects": object_states(record["gold"], predictions, score, api)})

    def aggregate(rows):
        return {"score": api["aggregate_scores"]([row["score"] for row in rows]),
                "quality": api["aggregate_quality"]([row["quality"] for row in rows])}

    result = {**aggregate(per_image), "per_image": per_image,
              "by_scene": {name: aggregate([r for r in per_image if r["scene"] == name]) for name in ("empty", "nonempty")}}
    # These are filename ranges, not claims of independent scenes or captures.
    result["by_filename_range"] = {}
    for lo, hi in ((7, 11), (53, 59), (86, 90), (146, 148)):
        chosen = [r for r in per_image if lo <= int(r["image_id"].split("_")[-1]) <= hi]
        result["by_filename_range"][f"{lo:03d}_{hi:03d}"] = aggregate(chosen)
    result["focus"] = {}
    for object_id in PROTOCOL["focus_object_ids"]:
        image_id, row_text = object_id.split(":")
        index = int(row_text) - 1
        record = next(r for r in records if r["id"] == image_id)
        image_score = next(r for r in per_image if r["image_id"] == image_id)
        proposals = by_id[image_id]["predictions"]
        overlaps = [(api["box_iou"](record["gold"][index]["bbox_xyxy"], p["bbox_xyxy"]), pi) for pi, p in enumerate(proposals)]
        best_iou, pi = max(overlaps, default=(0, None))
        result["focus"][object_id] = {**image_score["objects"][index], "reference_geometry": record["gold"][index],
                                       "max_proposal_iou": best_iou,
                                       "max_overlap_prediction_index_one_based": pi + 1 if best_iou > 0 else None,
                                       "max_overlap_raw_prediction": proposals[pi] if best_iou > 0 else None}
    return result


def numeric_delta(before, after):
    if isinstance(before, dict) and isinstance(after, dict):
        if "radius_px" in before:
            require(before["radius_px"] == after.get("radius_px"), "comparison radii differ")
        return {key: before[key] if key == "radius_px" else numeric_delta(before[key], after[key])
                for key in before if key in after and key != "endpoint_errors_px"}
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [numeric_delta(a, b) for a, b in zip(before, after)]
    if isinstance(before, (int, float)) and isinstance(after, (int, float)) and not isinstance(before, bool) and not isinstance(after, bool):
        return after - before
    return {"before": before, "after": after} if before != after else None


def compare_reports(before, after):
    def score_delta(first, second):
        # Arrays of match indexes/errors are evidence, not subtractable metrics.
        scalar = lambda score: {key: value for key, value in score.items() if value is None or isinstance(value, (int, float))}
        return numeric_delta(scalar(first), scalar(second))

    older = {row["image_id"]: row for row in before["per_image"]}
    changed = []
    for current in after["per_image"]:
        image_id = current["image_id"]
        old = older[image_id]
        gained = [f"{image_id}:{b['reference_row_one_based']}" for a, b in zip(old["objects"], current["objects"]) if not a["matched"] and b["matched"]]
        lost = [f"{image_id}:{a['reference_row_one_based']}" for a, b in zip(old["objects"], current["objects"]) if a["matched"] and not b["matched"]]
        gained10 = [f"{image_id}:{b['reference_row_one_based']}" for a, b in zip(old["objects"], current["objects"]) if not a["both_visible_within_radius"]["10"] and b["both_visible_within_radius"]["10"]]
        lost10 = [f"{image_id}:{a['reference_row_one_based']}" for a, b in zip(old["objects"], current["objects"]) if a["both_visible_within_radius"]["10"] and not b["both_visible_within_radius"]["10"]]
        changed.append({"image_id": image_id, "scene": current["scene"], "gained_object_ids": gained, "lost_object_ids": lost,
                        "gained_both_visible_within_10_px": gained10, "lost_both_visible_within_10_px": lost10,
                        "score_delta": score_delta(old["score"], current["score"]),
                        "quality_delta": numeric_delta(old["quality"], current["quality"])})
    return {"score_delta": score_delta(before["score"], after["score"]), "quality_delta": numeric_delta(before["quality"], after["quality"]),
            "by_scene_delta": numeric_delta(before["by_scene"], after["by_scene"]), "per_image": changed,
            "gained_object_ids": [x for row in changed for x in row["gained_object_ids"]],
            "lost_object_ids": [x for row in changed for x in row["lost_object_ids"]],
            "focus_before": before["focus"], "focus_after": after["focus"]}


def compare_raw(before, after):
    old = {item["image_id"]: item["predictions"] for item in before}
    rows = []
    for item in after:
        a, b = old[item["image_id"]], item["predictions"]
        differences = []

        def walk(x, y, path):
            if x == y:
                return
            if isinstance(x, dict) and isinstance(y, dict) and x.keys() == y.keys():
                for key in x:
                    walk(x[key], y[key], f"{path}.{key}")
            elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                for index, (xx, yy) in enumerate(zip(x, y)):
                    walk(xx, yy, f"{path}[{index}]")
            else:
                differences.append({"path": path, "archived": x, "fresh": y})

        walk(a, b, "predictions")
        rows.append({"image_id": item["image_id"], "ordered_raw_exact": a == b,
                     "archived_count": len(a), "fresh_count": len(b), "differences": differences})
    return {"all_images_ordered_raw_exact": all(row["ordered_raw_exact"] for row in rows),
            "changed_images": [row["image_id"] for row in rows if not row["ordered_raw_exact"]],
            "index_convention": "raw array indexes in difference paths are zero-based; no cross-run physical identity is assumed", "per_image": rows}


def infer_arm(preflight, size, output, api, header):
    import numpy as np
    import torch
    model, runtime = api["runtime_and_model"](preflight, "cpu", 4)
    inference = {**api["INFERENCE"], "imgsz": size}
    started = time.perf_counter()
    model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), device="cpu", conf=0.15, **inference)
    warmup_ms = (time.perf_counter() - started) * 1000
    backend = model.predictor.model
    require(not backend.end2end and not backend.fp16 and str(backend.device) == "cpu", "expected ordinary NMS, FP32, and CPU")
    torch.set_num_threads(4)  # Restore after Ultralytics device setup, as in evaluate_pose.
    require(torch.get_num_threads() == 4, "torch thread count differs")
    require(not model.predictor.args.augment, "test-time augmentation must be disabled")
    runtime.update(device_actual=str(backend.device), backend_end2end=bool(backend.end2end),
                   backend_fp16=bool(backend.fp16), torch_threads=torch.get_num_threads(),
                   torch_interop_threads=torch.get_num_interop_threads(),
                   resolved_prediction_args={key: getattr(model.predictor.args, key) for key in
                       ("imgsz", "conf", "iou", "rect", "batch", "nms", "max_det", "quantize", "augment")})
    arm_header = {"record_type": "run", **header, "imgsz": size, "inference": inference,
                  "runtime": runtime, "warmup_wall_ms_excluded": warmup_ms}
    items = []
    with (output / f"predictions-{size}.jsonl").open("w") as stream:
        stream.write(json.dumps(arm_header, allow_nan=False) + "\n")
        stream.flush()
        for record in preflight["records"]:
            started = time.perf_counter()
            result = model.predict(source=record["image_path"], device="cpu", conf=0.15, **inference)[0]
            wall_ms = (time.perf_counter() - started) * 1000
            require(tuple(result.orig_shape) == (720, 1280), f"original-pixel shape changed: {record['id']}")
            require(not model.predictor.model.end2end and not model.predictor.model.fp16 and torch.get_num_threads() == 4, "inference contract changed during arm")
            item = {"record_type": "prediction", "image_id": record["id"], "threshold": 0.15, "imgsz": size,
                    "image_sha256": record["image_sha256"], "label_sha256": record["label_sha256"],
                    "width": 1280, "height": 720, "predictions": api["prediction_records"](result),
                    "stage_ms": {key: float(value) for key, value in result.speed.items()}, "predict_call_wall_ms": wall_ms}
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            items.append(item)
            print(f"imgsz={size} {record['id']}: {len(item['predictions'])} proposals, {wall_ms:.1f} ms", flush=True)
    timings = {}
    for name in ("predict_call_wall_ms", "preprocess", "inference", "postprocess"):
        values = sorted(item[name] if name == "predict_call_wall_ms" else item["stage_ms"][name] for item in items)
        timings[name] = {"count": len(values), "sum_ms": sum(values), "mean_ms": sum(values) / len(values),
                         "median_ms": api["percentile"](values, 0.5), "p95_ms": api["percentile"](values, 0.95),
                         "min_ms": min(values), "max_ms": max(values)}
    del model
    return items, {"runtime": runtime, "timings": timings, "warmup_wall_ms_excluded": warmup_ms}


def run(args):
    root = args.root.expanduser().resolve()
    checkpoint_manifest = args.checkpoint_manifest.expanduser()
    if not checkpoint_manifest.is_absolute():
        checkpoint_manifest = root / checkpoint_manifest
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime-config")
    blocked = disable_network()
    json_write(output / "protocol.json", PROTOCOL)
    try:
        api, preflight, archived, archived_header, replay, replay_hash = load_inputs(root, checkpoint_manifest)
        identities = {"source": api["source_identity"](root), "source_helpers_sha256": SOURCE_HASHES,
                      "script_sha256": sha256(__file__), "protocol_sha256": sha256(output / "protocol.json"),
                      "reference_replay_sha256": replay_hash, "checkpoint_identity": preflight["checkpoint_identity"],
                      "image_reference_inventory": [{key: r[key] for key in ("id", "image_sha256", "source_label_sha256", "label_sha256", "width", "height")} for r in preflight["records"]]}
        header = {"schema_version": 1, "experiment_id": PROTOCOL["experiment_id"],
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": preflight["model_sha256"],
                  "reference_manifest_sha256": PROTOCOL["reference_manifest_sha256"], "identities": identities}
        json_write(output / "identity.json", identities)
        archive_report = score_records(preflight["records"], archived, api)
        expected = next(m["amended"] for m in replay["metrics"] if m["model_id"] == "m2" and m["box_confidence"] == 0.15)
        require({key: archive_report["score"][key] for key in expected["score"]} == expected["score"], "archived official summary did not reproduce exactly")
        require(archive_report["quality"] == expected["quality"], "archived quality did not reproduce exactly")
        for row in archive_report["per_image"]:
            old = next(item["amended"] for item in replay["per_image"] if item["model_id"] == "m2" and item["box_confidence"] == 0.15 and item["image_id"] == row["image_id"])
            require({key: row["score"][key] for key in old["score"]} == old["score"] and row["quality"] == old["quality"], f"archived image score did not reproduce: {row['image_id']}")
        json_write(output / "scoring-archived-1280.json", archive_report)
        with (output / "predictions-archived-1280.jsonl").open("w") as stream:
            stream.write(json.dumps({"record_type": "archive_subset", "selected_confidence": 0.15, "original_run_header": archived_header}, allow_nan=False) + "\n")
            for item in archived:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        arms = {}
        for size in PROTOCOL["sizes_in_fixed_order"]:
            items, runtime = infer_arm(preflight, size, output, api, header)
            report = score_records(preflight["records"], items, api)
            json_write(output / f"scoring-{size}.json", report)
            arms[size] = {"items": items, "report": report, **runtime}
            if size == 1280:
                json_write(output / "fresh-1280-vs-archive.json", {"raw": compare_raw(archived, items), "scoring": compare_reports(archive_report, report)})
        comparison = compare_reports(arms[1280]["report"], arms[1920]["report"])
        json_write(output / "comparison.json", {"status": "complete", "experiment_id": PROTOCOL["experiment_id"],
                    "baseline": "fresh 1280", "candidate": "fresh 1920", "comparison": comparison,
                    "arms": {str(size): {"score": arm["report"]["score"], "quality": arm["report"]["quality"],
                                         "focus": arm["report"]["focus"], "runtime": arm["runtime"], "timings": arm["timings"],
                                         "warmup_wall_ms_excluded": arm["warmup_wall_ms_excluded"]} for size, arm in arms.items()},
                    "network_events_blocked": blocked, "total_wall_seconds": time.perf_counter() - started,
                    "default_changed": False, "limits": PROTOCOL["limits"]})
        inventory = {p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(output.glob("*.json*"))}
        json_write(output / "inventory.json", inventory)
        json_write(output / "status.json", {"status": "complete", "images_per_arm": 20, "sizes": [1280, 1920], "total_wall_seconds": time.perf_counter() - started})
        print(json.dumps({"status": "complete", "output": str(output), "score_delta": comparison["score_delta"], "gained": comparison["gained_object_ids"], "lost": comparison["lost_object_ids"]}, allow_nan=False))
        return 0
    except Exception as error:
        json_write(output / "status.json", {"status": "incomplete", "error_type": type(error).__name__, "error": str(error), "total_wall_seconds": time.perf_counter() - started, "network_events_blocked": blocked})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="repository root")
    parser.add_argument("--checkpoint-manifest", required=True, type=Path, help="complete seed46 candidate bundle manifest; relative paths resolve from --root")
    parser.add_argument("--output", required=True, type=Path, help="new, nonexisting output directory")
    args = parser.parse_args()

    def deadline(signum, frame):
        raise TimeoutError("590-second experiment deadline reached; incomplete results must not be promoted")

    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, PROTOCOL["internal_deadline_seconds"])
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
