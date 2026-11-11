#!/usr/bin/env python3
"""Evaluate the fixed PR43 visible-extent pair and the current checkpoint; replay without inference.

All matching and quality calculations use the unchanged repository scorers.
The current checkpoint is an operational comparator, not a third training arm.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True

PROTOCOL_PATH = "experiments/visible-extent-2026-11-11.json"
PROTOCOL_SHA256 = "42aa275a5d2ed1ccabb786c07b537a8f0d515b27b7804fec2b71e942d6c9022b"
REFERENCE_PATH = "data/references/visible-front-2026-11-11/manifest.json"
REFERENCE_SHA256 = "2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c"
ORIGINAL_SHA256 = "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926"
TRAINING_REVISION = "d43a58b9456a782ace14e20f0bf7ed2da3572ca1"
ROLES = ("old_reviewed", "extended_reviewed", "current")
THRESHOLDS = (0.13, 0.15)
RADII = (2, 5, 10)
FOCUS = ("new_000008:33", "new_000009:57")
MODEL_SHA256 = {
    "current": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
    "old_reviewed": "db0e8c625aa0261d32f8af0f48b5362ab66e5c51a06d2b6f45b046798bd4184f",
    "extended_reviewed": "d7cc62189743e7c2750035fc2b98e8b499e9ca160489407dc11e0002c8f01fd0"
}
MANIFEST_SHA256 = {
    "current": "c5f07c4445a7b33a679fd4fc896aaf9643d81b4f72d8ca8e4e7179cb0bc691e8",
    "old_reviewed": "5db89fa828b51446f651e4e7c064ba9f4738658cff3bdad3971d1a1215705c87",
    "extended_reviewed": "f0b229fb6649a3c0c116163d425c1f5992ef2d360a1d8d3d1346d581bfebbf45"
}
SOURCE_SHA256 = {
    "scripts/evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "scripts/pose_checkpoint.py": "8c92a452f632aec92bfd51bc8fbf9fc745a09978d840cdb294af0a37dea8696f",
    "scripts/pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "scripts/diagnose_pose_quality.py": "a3baa1285d1580ee44547f7c3d606f524f2f40616fed568885746d2014159953",
    "scripts/package_visible_extent_results.py": "a5faf45a58dbf31c5e038149f1d60f8b4aea48371e8ebbf0f4c7afd7738816f2",
    "scripts/package_gpu_results.py": "120b1a3d22f9cb8896af67a84b7d3da686d89e41b63d5889eb7fb106bdce477c",
    "scripts/verify_pose_annotation_candidate.py": "0dc3d263b4aad594e1019de8f8318dcd963d9c0c2f4604ad2dee408c83af7979",
    "scripts/verify_pose_candidate.py": "03395b9a82a4c120647a1cac51c89fd214d8cf9b15c04e2196a40f39077d8491",
    "scripts/verify_pose_box_candidate.py": "f8613b9b310f25930a9d47169650473d3fbc0a1032868dd80222d956d26e1ea5",
    "scripts/stage_pose_training.py": "e28a4f1b705086e4461876df97627d57111932a7455e20ed116d35b4f4d38eae",
    "scripts/train_pose_comparison.py": "d4ab5031029c150039843383d0d42d0a72a4062b97212be5633dd39993309abe",
    "docs/low-confidence-review-2026-11-11.md": "f11e3442acc89cadebc9074dea45583517d6bccf45c710c9071e1dd7087c41c5",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned(path, expected):
    require(Path(path).is_file() and sha256(path) == expected, f"file SHA-256 mismatch: {path}")


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def disable_network():
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    blocked = []
    def reject(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise PermissionError(f"offline evaluation forbids {event}")
    sys.addaudithook(reject)
    return blocked


def load_inputs(root, returned, current_manifest):
    for name, expected in SOURCE_SHA256.items():
        pinned(root / name, expected)
    pinned(root / PROTOCOL_PATH, PROTOCOL_SHA256)
    pinned(root / REFERENCE_PATH, REFERENCE_SHA256)
    pinned(root / "data/manifest.json", ORIGINAL_SHA256)
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import (INFERENCE, _prediction_records, _runtime_and_model,
                               _source_identity, parse_pose_labels, preflight_dataset)
    from pose_checkpoint import select_checkpoint
    from pose_scoring import _box_iou, _percentile, aggregate_scores, score_image
    from diagnose_pose_quality import _returned, aggregate_quality, quality_for_image
    from package_visible_extent_results import (check_reviewed_pair, collect_reviewed_run,
                                              validate_protocol)
    api = dict(inference=INFERENCE, predictions=_prediction_records, model=_runtime_and_model,
               source=_source_identity, iou=_box_iou, percentile=_percentile, score=score_image,
               aggregate=aggregate_scores, returned=_returned, quality=quality_for_image,
               aggregate_quality=aggregate_quality)
    context = validate_protocol(root)
    protocol = context["document"]
    settings = protocol["evaluation"]
    require([settings["primary_box_confidence"], *settings["diagnostic_box_confidences"]]
            == list(THRESHOLDS), "fixed confidence thresholds differ")
    require(settings["radii_px"] == list(RADII) and settings["imgsz"] == 1280
            and settings["nms_iou"] == .7 and settings["matching_iou"] == .5
            and settings["endpoint_confidence"] == .5, "fixed scoring/inference settings differ")
    original = preflight_dataset(root, require_weights=False)
    bundles = {role: collect_reviewed_run(original, returned / role, role, context)
               for role in ROLES[:2]}
    check_reviewed_pair(bundles, context)
    for role, bundle in bundles.items():
        require(bundle["receipt"]["source"]["revision"] == TRAINING_REVISION,
                f"unexpected returned training revision: {role}")
        for name, expected in bundle["receipt"]["source"]["files_sha256"].items():
            pinned(root / name, expected)
    selected = {}
    for role in ROLES:
        manifest = current_manifest if role == "current" else returned / role / "checkpoint-manifest.json"
        pinned(manifest, MANIFEST_SHA256[role])
        selected[role] = select_checkpoint(original, manifest, require_weights=True)
        require(selected[role]["model_sha256"] == MODEL_SHA256[role], f"unexpected model: {role}")
    manifest = read_json(root / REFERENCE_PATH)
    files = {row["image_id"]: row for row in manifest["files"]}
    require(len(files) == len(manifest["files"]) == 20 and
            set(files) == {row["id"] for row in original["records"]}, "primary image set differs")
    primary, inventory = [], []
    for source in original["records"]:
        entry = files[source["id"]]
        require(entry["image_sha256"] == source["image_sha256"] and
                [entry["width"], entry["height"]] == [source["width"], source["height"]] == [1280, 720],
                "primary image identity differs")
        require(entry["label_path"] == f"labels/val/{source['id']}.txt", "primary label mapping differs")
        label = (root / REFERENCE_PATH).parent / entry["label_path"]
        pinned(label, entry["label_sha256"])
        text = label.read_text()
        require(all(line.strip() for line in text.splitlines()), "blank physical reference row")
        gold = parse_pose_labels(text, 1280, 720, str(label))
        require(len(gold) == entry["row_count"], "primary row count differs")
        primary.append({**source, "label_path": str(label), "label_sha256": entry["label_sha256"], "gold": gold})
        inventory.append({"image_id": source["id"], "image_sha256": source["image_sha256"],
                          "original_label_sha256": source["label_sha256"],
                          "primary_label_sha256": entry["label_sha256"], "width": 1280, "height": 720})
    references = {"primary": primary, "original": original["records"]}
    for name, expected in (("primary", (663, 1228, 3)), ("original", (664, 1227, 3))):
        rows = references[name]
        actual = (sum(len(r["gold"]) for r in rows),
                  sum(p[2] == 2 for r in rows for g in r["gold"] for p in g["keypoints"]),
                  sum(not r["gold"] for r in rows))
        require(actual == expected, f"{name} reference counts differ")
    notes = {}
    for row in manifest["queue_dispositions"]:
        mapped = files[row["image_id"]]["retained_source_to_revised_row_one_based"][row["source_row_one_based"] - 1]
        if mapped is not None:
            notes.setdefault(f"{row['image_id']}:{mapped}", {})["queue_disposition"] = row
    for row in manifest["edits"]:
        if row.get("revised_row_one_based") is not None:
            notes.setdefault(f"{row['image_id']}:{row['revised_row_one_based']}", {}).setdefault("accepted_edits", []).append(row)
    notes.setdefault("new_000011:61", {})["additional_caveat"] = (
        "Earlier original-pixel review retains boundary/identity uncertainty; "
        "see the pinned docs/low-confidence-review-2026-11-11.md. No new adjudication is made here.")
    identities = {"protocol_sha256": PROTOCOL_SHA256, "primary_reference_sha256": REFERENCE_SHA256,
                  "original_reference_sha256": ORIGINAL_SHA256, "source_helpers_sha256": SOURCE_SHA256,
                  "image_reference_inventory": inventory,
                  "checkpoints": {role: data["checkpoint_identity"] for role, data in selected.items()},
                  "returned_training_source": bundles["old_reviewed"]["receipt"]["source"],
                  "returned_training_runtime": bundles["old_reviewed"]["receipt"]["runtime"]}
    return api, selected, references, identities, notes


def object_states(gold, predictions, score, api):
    matches = {pair["gold_index"]: pair for pair in score["matches"]}
    result = []
    for index, target in enumerate(gold):
        pair = matches.get(index)
        visible = [p[2] == 2 for p in target["keypoints"]]
        returned, errors = [False, False], [None, None]
        if pair is not None:
            points = predictions[pair["prediction_index"]]["keypoints"]
            for side, point in enumerate(points):
                returned[side] = api["returned"](point, 1280, 720)
                if visible[side] and returned[side]:
                    errors[side] = math.hypot(point[0] - target["keypoints"][side][0],
                                              point[1] - target["keypoints"][side][1])
        result.append({"reference_row_one_based": index + 1, "locatable_corners": sum(visible),
                       "reference_endpoint_visible": visible, "matched": pair is not None,
                       "prediction_index_zero_based": pair["prediction_index"] if pair else None,
                       "matching_iou": pair["iou"] if pair else None,
                       "endpoint_returned": returned, "visible_endpoint_errors_px": errors,
                       "both_visible_within_radius": {str(r): all(visible) and all(e is not None and e <= r for e in errors) for r in RADII}})
    return result


def group_summary(states, api):
    result = {}
    for count in range(3):
        rows = [s for s in states if s["locatable_corners"] == count]
        errors = sorted(e for s in rows for e in s["visible_endpoint_errors_px"] if e is not None)
        result[str(count)] = {
            "reference_objects": len(rows), "matched_objects": sum(s["matched"] for s in rows),
            "missed_objects": sum(not s["matched"] for s in rows),
            "visible_endpoints": count * len(rows), "returned_visible_endpoints": len(errors),
            "missing_visible_endpoints": count * len(rows) - len(errors),
            "unlocatable_on_matched_objects": (2 - count) * sum(s["matched"] for s in rows),
            "returned_on_unlocatable_reference": sum(not visible and returned for s in rows
                for visible, returned in zip(s["reference_endpoint_visible"], s["endpoint_returned"])),
            "conditional_median_error_px": api["percentile"](errors, .5),
            "conditional_p95_error_px": api["percentile"](errors, .95),
            "radii": [{"radius_px": r, "visible_endpoints_within_radius": sum(e <= r for e in errors),
                       "both_visible_objects_within_radius": sum(s["both_visible_within_radius"][str(r)] for s in rows)} for r in RADII],
        }
    return result


def score_records(records, items, api, primary):
    by_id = {item["image_id"]: item for item in items}
    require(len(items) == len(by_id) == 20 and set(by_id) == {r["id"] for r in records}, "prediction image set differs")
    per_image = []
    for record in records:
        predictions = by_id[record["id"]]["predictions"]
        score = api["score"](record["gold"], predictions, 1280, 720, .5, .5)
        quality = api["quality"](record["gold"], predictions, 1280, 720, RADII)
        per_image.append({"image_id": record["id"], "scene": "nonempty" if record["gold"] else "empty",
                          "score": score, "quality": quality,
                          "objects": object_states(record["gold"], predictions, score, api)})
    def pooled(rows):
        return {"score": api["aggregate"]([r["score"] for r in rows]),
                "quality": api["aggregate_quality"]([r["quality"] for r in rows])}
    result = {**pooled(per_image), "per_image": per_image,
              "by_scene": {name: pooled([r for r in per_image if r["scene"] == name]) for name in ("empty", "nonempty")},
              "groups_by_locatable_corners": group_summary([s for r in per_image for s in r["objects"]], api),
              "group_matching_note": "Groups partition the single full-image matching. Unmatched predictions have no gold corner group; they remain in overall FP counts.",
              "miss_free_nonempty_image_ids": [r["image_id"] for r in per_image if r["scene"] == "nonempty" and r["score"]["fn"] == 0],
              "empty_image_predictions": {r["image_id"]: r["score"]["prediction_count"] for r in per_image if r["scene"] == "empty"}}
    result["by_filename_range"] = {f"{lo:03d}_{hi:03d}": pooled(
        [r for r in per_image if lo <= int(r["image_id"].split("_")[-1]) <= hi])
        for lo, hi in ((7, 11), (53, 59), (86, 90), (146, 148))}
    result["focus"] = {}
    if primary:
        for object_id in FOCUS:
            image_id, row = object_id.split(":")
            index = int(row) - 1
            gold = next(r["gold"] for r in records if r["id"] == image_id)[index]
            states = next(r["objects"] for r in per_image if r["image_id"] == image_id)
            proposals = by_id[image_id]["predictions"]
            overlaps = [(api["iou"](gold["bbox_xyxy"], p["bbox_xyxy"]), i) for i, p in enumerate(proposals)]
            overlap, pi = max(overlaps, default=(0, None))
            result["focus"][object_id] = {**states[index], "reference_geometry": gold,
                "max_proposal_iou": overlap, "max_overlap_prediction_index_zero_based": pi if overlap > 0 else None,
                "max_overlap_raw_prediction": proposals[pi] if overlap > 0 else None}
    return result


def numeric_delta(before, after):
    if isinstance(before, dict) and isinstance(after, dict):
        if "radius_px" in before:
            require(before["radius_px"] == after.get("radius_px"), "radii differ")
        return {k: before[k] if k == "radius_px" else numeric_delta(before[k], after[k])
                for k in before if k in after and k != "endpoint_errors_px"}
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [numeric_delta(a, b) for a, b in zip(before, after)]
    if (isinstance(before, (int, float)) and isinstance(after, (int, float))
            and not isinstance(before, bool) and not isinstance(after, bool)):
        return after - before
    return None if before == after else {"before": before, "after": after}


def compare_reports(before, after, notes):
    scalar = lambda score: {k: v for k, v in score.items() if v is None or isinstance(v, (int, float))}
    old = {row["image_id"]: row for row in before["per_image"]}
    changes, corner_changes, per_image = [], {str(r): {"gained": [], "lost": []} for r in RADII}, []
    for row in after["per_image"]:
        prior = old[row["image_id"]]
        require(len(prior["objects"]) == len(row["objects"]), "reference row counts differ")
        gained, lost = [], []
        for a, b in zip(prior["objects"], row["objects"]):
            require(a["reference_row_one_based"] == b["reference_row_one_based"], "reference row identities differ")
            object_id = f"{row['image_id']}:{b['reference_row_one_based']}"
            if a["matched"] != b["matched"]:
                (gained if b["matched"] else lost).append(object_id)
                changes.append({"object_id": object_id, "change": "gained" if b["matched"] else "lost",
                                "before": a, "after": b,
                                "prior_review": notes.get(object_id, {"status": "No specific assessment in the frozen review map; not a certification."})})
            for radius in RADII:
                key = str(radius)
                if a["both_visible_within_radius"][key] != b["both_visible_within_radius"][key]:
                    corner_changes[key]["gained" if b["both_visible_within_radius"][key] else "lost"].append(object_id)
        per_image.append({"image_id": row["image_id"], "gained_object_ids": gained, "lost_object_ids": lost,
                          "score_delta": numeric_delta(scalar(prior["score"]), scalar(row["score"])),
                          "quality_delta": numeric_delta(prior["quality"], row["quality"])})
    return {"score_delta": numeric_delta(scalar(before["score"]), scalar(after["score"])),
            "quality_delta": numeric_delta(before["quality"], after["quality"]),
            "groups_delta": numeric_delta(before["groups_by_locatable_corners"], after["groups_by_locatable_corners"]),
            "gained_object_ids": [r["object_id"] for r in changes if r["change"] == "gained"],
            "lost_object_ids": [r["object_id"] for r in changes if r["change"] == "lost"],
            "object_changes": changes, "both_corner_changes": corner_changes, "per_image": per_image,
            "focus_before": before["focus"], "focus_after": after["focus"]}


def state_identity(model):
    import torch
    rows = []
    for name, tensor in sorted(model.model.state_dict().items()):
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        rows.append({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                     "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"stage": "after YOLO loading, before prediction/fusion; loader dtype conversion is included",
            "sha256": hashlib.sha256(canonical).hexdigest(), "tensor_count": len(rows),
            "tensor_bytes": sum(r["bytes"] for r in rows), "tensors": rows}


def infer_role(role, preflight, output, api, identities):
    import numpy as np
    import torch
    model, runtime = api["model"](preflight, "cpu", 4)
    state = state_identity(model)
    started = time.perf_counter()
    model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), device="cpu", conf=THRESHOLDS[0], **api["inference"])
    warmup_ms = 1000 * (time.perf_counter() - started)
    backend = model.predictor.model
    require(not backend.end2end and not backend.fp16 and str(backend.device) == "cpu", "ordinary NMS/FP32/CPU contract differs")
    torch.set_num_threads(4)
    require(torch.get_num_threads() == 4 and not model.predictor.args.augment, "thread or augmentation setting differs")
    runtime.update(device_actual=str(backend.device), backend_end2end=bool(backend.end2end),
                   backend_fp16=bool(backend.fp16), torch_threads=torch.get_num_threads(),
                   torch_interop_threads=torch.get_num_interop_threads(),
                   mkldnn_enabled=torch.backends.mkldnn.enabled,
                   deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                   float32_matmul_precision=torch.get_float32_matmul_precision(),
                   resolved_prediction_args_after_warmup={k: getattr(model.predictor.args, k) for k in
                       ("imgsz", "conf", "iou", "rect", "batch", "nms", "max_det", "quantize", "augment")})
    header = {"record_type": "run", "role": role, "model_sha256": MODEL_SHA256[role],
              "checkpoint_identity": preflight["checkpoint_identity"], "identities": identities,
              "thresholds": list(THRESHOLDS), "inference": api["inference"], "runtime": runtime,
              "model_state_before_prediction": state, "warmup_wall_ms_excluded": warmup_ms}
    items = []
    with (output / f"predictions-{role}.jsonl").open("x") as stream:
        stream.write(json.dumps(header, allow_nan=False) + "\n"); stream.flush()
        for threshold in THRESHOLDS:
            for record in preflight["records"]:
                started = time.perf_counter()
                result = model.predict(source=record["image_path"], device="cpu", conf=threshold, **api["inference"])[0]
                wall_ms = 1000 * (time.perf_counter() - started)
                require(tuple(result.orig_shape) == (720, 1280), "original image coordinates changed")
                require(not model.predictor.model.end2end and not model.predictor.model.fp16
                        and torch.get_num_threads() == 4 and model.predictor.args.conf == threshold,
                        "inference contract changed during predictions")
                item = {"record_type": "prediction", "role": role, "image_id": record["id"],
                        "threshold": threshold, "image_sha256": record["image_sha256"],
                        "original_label_sha256": record["label_sha256"], "width": 1280, "height": 720,
                        "predictions": api["predictions"](result),
                        "stage_ms": {k: float(v) for k, v in result.speed.items()}, "predict_call_wall_ms": wall_ms}
                stream.write(json.dumps(item, allow_nan=False) + "\n"); stream.flush()
                items.append(item)
                print(f"{role} conf={threshold:.2f} {record['id']}: {len(item['predictions'])} predictions", flush=True)
    del model
    return header, items


def replay_role(path, role, preflight, api, identities):
    inventory = read_json(path.parent / "inventory.json")[path.name]
    pinned(path, inventory["sha256"])
    require(path.stat().st_size == inventory["size_bytes"], "replay file size differs")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    header, items = lines[0], lines[1:]
    require(header["record_type"] == "run" and header["role"] == role
            and header["model_sha256"] == MODEL_SHA256[role]
            and header["checkpoint_identity"] == preflight["checkpoint_identity"]
            and header["thresholds"] == list(THRESHOLDS)
            and header["inference"] == api["inference"], "replay model/settings identity differs")
    require(header["identities"]["driver_sha256"] == sha256(__file__), "replay inference driver differs")
    for key in ("protocol_sha256", "primary_reference_sha256", "original_reference_sha256",
                "source_helpers_sha256", "image_reference_inventory", "checkpoints"):
        require(header["identities"][key] == identities[key], f"replay {key} differs")
    require(len(items) == len(THRESHOLDS) * len(preflight["records"]), "replay real-image prediction count differs")
    records = {r["id"]: r for r in preflight["records"]}
    expected = {(t, i) for t in THRESHOLDS for i in records}
    actual = {(r["threshold"], r["image_id"]) for r in items}
    require(actual == expected, "replay threshold/image pairs differ")
    for item in items:
        source = records[item["image_id"]]
        require(item["record_type"] == "prediction" and item["role"] == role
                and item["image_sha256"] == source["image_sha256"]
                and item["original_label_sha256"] == source["label_sha256"]
                and [item["width"], item["height"]] == [1280, 720], "replay image identity differs")
    return header, items


def run(args):
    root, returned = args.root.resolve(), args.returned_root.resolve()
    output = args.output.resolve()
    require(not output.is_relative_to(root), "write run evidence outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    os.environ["YOLO_CONFIG_DIR"] = str(output / "runtime-config")
    blocked = disable_network()
    try:
        api, selected, references, identities, notes = load_inputs(root, returned, args.current_manifest.resolve())
        source = api["source"](root)
        require(args.replay_directory is not None or source["working_tree_status"] == "", "live inference requires committed clean source")
        identities.update(source=source, driver_sha256=sha256(__file__),
                          command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                          created_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "identity.json", identities)
        (output / "protocol.json").write_bytes((root / PROTOCOL_PATH).read_bytes())
        write_json(output / "reference-review-notes.json", notes)
        if args.replay_directory is None:
            import torch
            torch.set_num_interop_threads(1)
        reports, headers = {}, {}
        for role in ROLES:
            if args.replay_directory is None:
                header, items = infer_role(role, selected[role], output, api, identities)
            else:
                path = args.replay_directory.resolve() / f"predictions-{role}.jsonl"
                header, items = replay_role(path, role, selected[role], api, identities)
            headers[role] = {"runtime": header["runtime"], "model_state_before_prediction": header["model_state_before_prediction"],
                             "warmup_wall_ms_excluded": header["warmup_wall_ms_excluded"]}
            reports[role] = {}
            for reference, records in references.items():
                by_threshold = {str(t): score_records(records, [r for r in items if r["threshold"] == t], api, reference == "primary") for t in THRESHOLDS}
                reports[role][reference] = by_threshold
                write_json(output / f"scoring-{role}-{reference}.json", by_threshold)
        require(all(headers[role]["runtime"] == headers[ROLES[0]]["runtime"] for role in ROLES), "inference runtimes differ between models")
        comparison = {}
        for reference in references:
            comparison[reference] = {}
            for before, after in (("old_reviewed", "extended_reviewed"), ("current", "old_reviewed"), ("current", "extended_reviewed")):
                comparison[reference][f"{after}-vs-{before}"] = {str(t): compare_reports(
                    reports[before][reference][str(t)], reports[after][reference][str(t)], notes if reference == "primary" else {}) for t in THRESHOLDS}
        require(source == api["source"](root), "source changed during evaluation")
        for name, expected in SOURCE_SHA256.items():
            pinned(root / name, expected)
        summary = {"status": "complete", "experiment_id": "visible-extent-seed46-v1",
                   "mode": "replay" if args.replay_directory else "fresh_inference",
                   "primary_reference": "primary", "primary_confidence": THRESHOLDS[0], "thresholds": list(THRESHOLDS),
                   "roles": {role: {"checkpoint_identity": selected[role]["checkpoint_identity"], **headers[role]} for role in ROLES},
                   "metrics": {role: {reference: {str(t): {k: reports[role][reference][str(t)][k] for k in
                       ("score", "quality", "groups_by_locatable_corners", "miss_free_nonempty_image_ids", "empty_image_predictions", "focus")}
                       for t in THRESHOLDS} for reference in references} for role in ROLES},
                   "comparisons": comparison, "network_events_blocked": blocked,
                   "fresh_image_prediction_calls": 0 if args.replay_directory else len(ROLES) * len(THRESHOLDS) * len(references["primary"]),
                   "warmup_calls_excluded": 0 if args.replay_directory else 3,
                   "total_wall_seconds": time.perf_counter() - started,
                   "limits": ["The current checkpoint is an operational comparator, not a third training arm.",
                       "Primary and original references are scored separately. References are not changed after observing results.",
                       "Filename ranges are descriptive groups, not independent scenes or captures.",
                       "Corner availability on reference-v0 sides is a disagreement diagnostic; no hidden-coordinate error is computed.",
                       "Repeated development images do not establish unseen-scene generalization.",
                       "Single observations in a fixed order are descriptive CPU timings, not deployment benchmarks."]}
        write_json(output / "comparison.json", summary)
        write_json(output / "status.json", {"status": "complete", "mode": summary["mode"],
                   "fresh_image_prediction_calls": summary["fresh_image_prediction_calls"],
                   "scored_image_reference_threshold_pairs": len(ROLES) * len(THRESHOLDS) * sum(len(rows) for rows in references.values()), "total_wall_seconds": time.perf_counter() - started})
        write_json(output / "inventory.json", {p.name: {"sha256": sha256(p), "size_bytes": p.stat().st_size}
                   for p in sorted(output.glob("*.json*"))})
        print(json.dumps({"status": "complete", "output": str(output), "primary_metrics": {
            role: {k: reports[role]["primary"][str(THRESHOLDS[0])]["score"][k] for k in ("tp", "fn", "fp")} for role in ROLES}}, allow_nan=False))
        return 0
    except Exception as error:
        write_json(output / "status.json", {"status": "incomplete", "error_type": type(error).__name__,
                   "error": str(error), "network_events_blocked": blocked, "total_wall_seconds": time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--returned-root", type=Path, required=True, help="verified extracted PR43 bundle containing both reviewed roles")
    parser.add_argument("--current-manifest", type=Path, required=True, help="completed historical seed46 candidate manifest")
    parser.add_argument("--output", type=Path, required=True, help="new directory outside the source checkout")
    parser.add_argument("--replay-directory", type=Path, help="read saved role predictions and score without model loading/inference")
    args = parser.parse_args()
    def deadline(signum, frame):
        raise TimeoutError("900-second bounded evaluation deadline reached")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 900)
    try:
        return run(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
