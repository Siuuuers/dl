#!/usr/bin/env python3
"""Prepare, run, or replay the fixed development-only input-pixel occlusion probe."""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from evaluate_pose import file_sha256, parse_pose_labels
from pose_scoring import _box_iou, aggregate_scores, score_image


FROZEN = ("protocol.json", "prepared-cases.json", "reference-objects.json", "expected-baseline.json")
RUN_FILES = (*FROZEN, "preparation.json", "predictions.jsonl", "scores.json", "summary.json")
CONDITIONS = ("intact", "left", "right", "center")
SETTINGS = {"box_confidence": .13, "endpoint_confidence": .5, "nms_iou": .7,
            "matching_iou": .5, "imgsz": 1280, "threads": 4, "device": "cpu"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(root, name):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root), "invalid repository-relative path")
    return path


def verify_sources(root, protocol):
    require(all(protocol["settings"].get(k) == v for k, v in SETTINGS.items()), "unexpected inference settings")
    for name, digest in protocol["source_files"].items():
        require(file_sha256(relative(root, name)) == digest, "source helper hash mismatch")
    for item, path_key, hash_key in ((protocol["cohort"], "path", "sha256"),
                                    (protocol["reference"], "manifest_path", "manifest_sha256")):
        require(file_sha256(relative(root, item[path_key])) == item[hash_key], "reference provenance hash mismatch")


def checkpoint(root, manifest, protocol):
    from predict_pose_folder import verify_trained_checkpoint
    selected = verify_trained_checkpoint(root, manifest)
    require(selected["weights_verified"] and selected["model_sha256"] == protocol["checkpoint_sha256"],
            "checkpoint identity mismatch")
    require(Path(selected["weights"]).stat().st_size == protocol["checkpoint_size_bytes"], "checkpoint size mismatch")
    return selected


def point_state(point, rect, radius):
    x, y, visibility = point
    covered = rect is not None and rect[0] <= x < rect[2] and rect[1] <= y < rect[3]
    intersects = rect is not None and x + radius >= rect[0] and x - radius < rect[2] and y + radius >= rect[1] and y - radius < rect[3]
    interior = min(x - rect[0], y - rect[1], rect[2] - x, rect[3] - y) if covered else None
    return {"source_visibility": visibility, "coordinate_known_before_mask": visibility == 2,
            "observed_in_masked_input": visibility == 2 and not covered,
            "center_covered": covered, "radius_neighborhood_intersects_mask": intersects,
            "interior_margin_px": interior}


def mask_rect(box, condition, size, masks):
    x0, y0, x1, y1 = box
    width, cx = x1 - x0, (x0 + x1) / 2
    side, outside = masks["side_fraction"], masks["outside_fraction"]
    lo, hi = {"left": (x0 - outside * width, x0 + side * width),
              "right": (x1 - side * width, x1 + outside * width),
              "center": (cx - (side + outside) * width / 2, cx + (side + outside) * width / 2)}[condition]
    rect = [max(0, math.floor(lo)), max(0, math.floor(y0 - masks["pad_y"])),
            min(size[0], math.ceil(hi)), min(size[1], math.ceil(y1 + masks["pad_y"]))]
    require(rect[0] < rect[2] and rect[1] < rect[3], "empty mask")
    return rect


def altered(rgb, rect, fill):
    if rect is None:
        return rgb
    from PIL import ImageChops
    result = rgb.copy()
    result.paste(tuple(fill), tuple(rect))
    difference = ImageChops.difference(rgb, result)
    difference.paste((0, 0, 0), tuple(rect))
    require(difference.getbbox() is None, "pixels changed outside mask")
    require(result.crop(tuple(rect)).getextrema() == tuple((v, v) for v in fill), "mask fill mismatch")
    return result


def build(root, image_root, protocol):
    from PIL import Image
    images, references, cases = {}, {}, []
    radius = protocol["masks"]["neighborhood_radius_px"]
    require(protocol["masks"]["conditions"] == list(CONDITIONS[1:]), "unexpected mask conditions")
    for item in protocol["images"]:
        image_id = item["id"]
        require(re.fullmatch(r"[A-Za-z0-9_-]+", image_id) and image_id not in images, "invalid image identity")
        image_path, label_path = image_root / (image_id + ".png"), relative(root, item["label_path"])
        require(file_sha256(image_path) == item["image_sha256"], "original image hash mismatch")
        require(file_sha256(label_path) == item["label_sha256"], "original reference label hash mismatch")
        with Image.open(image_path) as source:
            require(source.format == "PNG" and source.getexif().get(274, 1) == 1, "unexpected image encoding")
            rgb = source.convert("RGB")
        require(rgb.size == (item["width"], item["height"]), "original image dimensions mismatch")
        gold = parse_pose_labels(label_path.read_text(), rgb.width, rgb.height, image_id)
        images[image_id] = rgb
        references[image_id] = {**item, "objects": gold}
        cases.append({"case_id": image_id + "-intact", "image_id": image_id, "condition": "intact",
                      "target_row": None, "mask_rect_xyxy": None, "input_rgb_sha256": sha(rgb.tobytes()),
                      "target_point_states": None, "collateral_endpoints": []})
    identities = [(item["image_id"], item["row"]) for item in protocol["targets"]]
    require(len(images) == 4 and len(identities) == len(set(identities)) == 30, "cohort cardinality mismatch")
    for image_id, row in identities:
        rgb, gold = images[image_id], references[image_id]["objects"]
        require(isinstance(row, int) and 1 <= row <= len(gold), "invalid 1-based target row")
        target = gold[row - 1]
        require(all(point[2] == 2 for point in target["keypoints"]), "target coordinates must be originally observed")
        for condition in CONDITIONS[1:]:
            rect = mask_rect(target["bbox_xyxy"], condition, rgb.size, protocol["masks"])
            states = [point_state(point, rect, radius) for point in target["keypoints"]]
            for index, state in enumerate(states):
                intended = condition == ("left", "right")[index]
                require((state["center_covered"] and state["interior_margin_px"] >= radius) if intended
                        else not state["radius_neighborhood_intersects_mask"], "target mask geometry failed")
            collateral = [{"row": index + 1, "endpoint": side, **point_state(point, rect, radius)}
                          for index, obj in enumerate(gold) if index + 1 != row
                          for side, point in zip(("left", "right"), obj["keypoints"])
                          if point[2] == 2 and point_state(point, rect, radius)["radius_neighborhood_intersects_mask"]]
            cases.append({"case_id": f"{image_id}-row{row:03d}-{condition}", "image_id": image_id,
                          "condition": condition, "target_row": row, "mask_rect_xyxy": rect,
                          "input_rgb_sha256": sha(altered(rgb, rect, protocol["masks"]["fill_rgb"]).tobytes()),
                          "target_point_states": states, "collateral_endpoints": collateral})
    require(len(cases) == 94, "prediction count mismatch")
    require(sum(any(p["center_covered"] for p in c["collateral_endpoints"]) for c in cases) ==
            protocol["expected"]["masks_intersecting_other_reference_endpoint_centers"], "collateral inventory mismatch")
    return cases, references, images


def expected_baseline(root, protocol):
    baseline = protocol["baseline"]
    archive = relative(root, baseline["archive_path"])
    require(file_sha256(archive) == baseline["archive_sha256"], "baseline archive hash mismatch")
    with zipfile.ZipFile(archive) as stream:
        rows = [json.loads(line) for line in stream.read(baseline["predictions_member"]).splitlines() if line]
    require(rows[0]["record_type"] == "run" and rows[0]["model_sha256"] == protocol["checkpoint_sha256"], "baseline model mismatch")
    expected = {}
    for item in protocol["images"]:
        matches = [row for row in rows if row.get("record_type") == "prediction" and
                   row.get("image_id") == item["id"] and row.get("threshold") == SETTINGS["box_confidence"]]
        require(len(matches) == 1 and matches[0]["image_sha256"] == item["image_sha256"] and
                (matches[0]["width"], matches[0]["height"]) == (item["width"], item["height"]), "baseline image mismatch")
        expected[item["id"]] = matches[0]["predictions"]
    return expected


def frozen_load(directory, protocol_path, root):
    preparation = read(directory / "preparation.json")
    require(preparation["status"] == "prepared", "preparation is incomplete")
    require(file_sha256(protocol_path) == preparation["protocol_sha256"], "protocol differs from preparation")
    require(file_sha256(Path(__file__)) == preparation["runner_sha256"], "runner differs from preparation")
    for name in FROZEN:
        require(file_sha256(directory / name) == preparation["files"][name], "frozen artifact hash mismatch")
    protocol = read(directory / "protocol.json")
    require(preparation["source_files"] == protocol["source_files"], "preparation source identity mismatch")
    verify_sources(root, protocol)
    return protocol, read(directory / "prepared-cases.json"), read(directory / "reference-objects.json"), read(directory / "expected-baseline.json")


def endpoint(point, truth, width, height):
    finite = point is not None and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point[:2])
    in_image = finite and 0 <= point[0] < width and 0 <= point[1] < height
    available = bool(in_image and point[2] >= SETTINGS["endpoint_confidence"])
    error = math.hypot(point[0] - truth[0], point[1] - truth[1]) if finite else None
    return {"reference_xy": truth[:2], "raw_prediction": point, "available": available, "error_px": error,
            "within_5px": available and error <= 5, "within_10px": available and error <= 10}


def score_all(protocol, cases, references, records):
    require(len(records) == len(cases) == 94, "raw prediction count mismatch")
    full, targets, by_id = [], [], {}
    for case, record in zip(cases, records):
        require(record.get("record_type") == "prediction" and all(record.get(k) == case[k] for k in
                ("case_id", "image_id", "condition", "input_rgb_sha256")), "raw case binding mismatch")
        ref, predictions = references[case["image_id"]], record["predictions"]
        require(all(p["confidence"] >= SETTINGS["box_confidence"] for p in predictions), "unfiltered prediction")
        result = score_image(ref["objects"], predictions, ref["width"], ref["height"],
                             SETTINGS["matching_iou"], SETTINGS["endpoint_confidence"])
        full.append({"case_id": case["case_id"], "image_id": case["image_id"], "condition": case["condition"], "score": result})
        by_id[case["case_id"]] = result
        rows = [t["row"] for t in protocol["targets"] if t["image_id"] == case["image_id"]] if case["condition"] == "intact" else [case["target_row"]]
        matches = {m["gold_index"]: m for m in result["matches"]}
        for row in rows:
            truth, match = ref["objects"][row - 1], matches.get(row - 1)
            prediction = predictions[match["prediction_index"]] if match else None
            points = [endpoint(prediction["keypoints"][i] if prediction else None, truth["keypoints"][i], ref["width"], ref["height"]) for i in range(2)]
            diagnostic = None
            if not match and predictions:
                candidates = [i for i, p in enumerate(predictions) if p["class_id"] == truth["class_id"]]
                if candidates:
                    index = min(candidates, key=lambda i: (-_box_iou(truth["bbox_xyxy"], predictions[i]["bbox_xyxy"]), i))
                    diagnostic = {"diagnostic_only": True, "prediction_index": index,
                                  "box_iou": _box_iou(truth["bbox_xyxy"], predictions[index]["bbox_xyxy"]),
                                  "assigned_gold_row": next((m["gold_index"] + 1 for m in result["matches"] if m["prediction_index"] == index), None),
                                  "prediction": predictions[index],
                                  "endpoints": [endpoint(predictions[index]["keypoints"][i], truth["keypoints"][i], ref["width"], ref["height"]) for i in range(2)]}
            targets.append({"case_id": case["case_id"], "image_id": case["image_id"], "row": row,
                            "condition": case["condition"], "matched": match is not None, "match": match,
                            "matched_box_xyxy": prediction["bbox_xyxy"] if prediction else None,
                            "point_states": case["target_point_states"] or [point_state(p, None, 3) for p in truth["keypoints"]],
                            "left": points[0], "right": points[1], "pair_available": all(p["available"] for p in points),
                            "pair_within_5px": all(p["within_5px"] for p in points),
                            "pair_within_10px": all(p["within_10px"] for p in points), "best_iou_unmatched_diagnostic": diagnostic})
    controls = {(t["image_id"], t["row"]): t for t in targets if t["condition"] == "intact"}
    def flag(target, key):
        if key in target:
            return target[key]
        side, field = key.split("_", 1)
        return target[side][field]

    summaries = {}
    for condition in CONDITIONS:
        group = [t for t in targets if t["condition"] == condition]
        require(len(group) == 30, "target denominator changed")
        counts = {key: sum(t[key] for t in group) for key in ("matched", "pair_available", "pair_within_5px", "pair_within_10px")}
        counts.update({f"{side}_{key}": sum(t[side][key] for t in group)
                       for side in ("left", "right") for key in ("available", "within_5px", "within_10px")})
        counts.update({f"{side}_available_error_over_10px": sum(t[side]["available"] and t[side]["error_px"] > 10 for t in group) for side in ("left", "right")})
        paired = {}
        keys = ("matched", "pair_available", "pair_within_5px", "pair_within_10px") + tuple(
            f"{side}_{field}" for side in ("left", "right") for field in ("available", "within_5px", "within_10px"))
        for key in keys:
            gained = [f'{t["image_id"]}:{t["row"]}' for t in group if flag(t, key) and not flag(controls[t["image_id"], t["row"]], key)]
            lost = [f'{t["image_id"]}:{t["row"]}' for t in group if not flag(t, key) and flag(controls[t["image_id"], t["row"]], key)]
            paired[key] = {"gained": len(gained), "lost": len(lost), "gained_targets": gained, "lost_targets": lost}
        confidence_deltas = {}
        for side in ("left", "right"):
            deltas = [t[side]["raw_prediction"][2] - controls[t["image_id"], t["row"]][side]["raw_prediction"][2]
                      for t in group if t[side]["raw_prediction"] is not None and
                      controls[t["image_id"], t["row"]][side]["raw_prediction"] is not None]
            confidence_deltas[side] = {"conditional_denominator": len(deltas),
                                       "condition": "Canonical target match and raw endpoint present in both images; no display-confidence filtering.",
                                       "mean": statistics.mean(deltas) if deltas else None,
                                       "median": statistics.median(deltas) if deltas else None}
        summaries[condition] = {"denominator": 30, "counts": counts, "paired_to_same_targets_intact": paired,
                                "endpoint_confidence_delta_from_intact": confidence_deltas}
    all_images = {}
    for condition in CONDITIONS:
        group = [f for f in full if f["condition"] == condition]
        paired_base = [by_id[f["image_id"] + "-intact"] for f in group]
        totals, base = aggregate_scores([f["score"] for f in group]), aggregate_scores(paired_base)
        all_images[condition] = {"aggregate": totals, "paired_intact_aggregate": base,
                                 "paired_count_delta": {k: totals[k] - base[k] for k in ("tp", "fn", "fp", "prediction_count", "returned_endpoints")}}
    summary = {"experiment_id": protocol["experiment_id"], "target_conditions": summaries,
               "full_image_conditions": all_images, "image_predictions_excluding_warmup": 94,
               "interpretation": "Canonical frozen-box associations and original known-coordinate errors after synthetic pixel removal; not independently measured physical accuracy or a visibility calibration.",
               "full_image_reference_semantics": "The scorer's visible_endpoints field counts known original v2 coordinates, including coordinates subsequently hidden by the mask.",
               "decision_limits": protocol["decision_limits"],
               "collateral_masks": {"other_endpoint_centers": sum(any(p["center_covered"] for p in c["collateral_endpoints"]) for c in cases),
                                    "other_endpoint_radius_neighborhoods": sum(bool(c["collateral_endpoints"]) for c in cases)}}
    return {"full_image_scores": full, "target_scores": targets}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-prepared", type=Path)
    mode.add_argument("--replay", type=Path)
    args = parser.parse_args()
    root, output, protocol_path = args.repo_root.resolve(), args.output.resolve(), args.protocol.resolve()
    inputs = [root] + [p.resolve() for p in (args.image_root, args.run_prepared, args.replay) if p]
    require(not output.exists() and not args.output.is_symlink(), "output must be fresh")
    require(all(output != p and p not in output.parents and output not in p.parents for p in inputs), "output overlaps source inputs")
    output.mkdir(parents=True)
    stage, completed = "validation", 0
    try:
        protocol = read(protocol_path)
        verify_sources(root, protocol)
        if args.prepare_only:
            require(args.image_root and args.checkpoint_manifest, "preparation requires image root and checkpoint manifest")
            checkpoint(root, args.checkpoint_manifest, protocol)
            cases, references, _ = build(root, args.image_root.resolve(), protocol)
            expected = expected_baseline(root, protocol)
            (output / "protocol.json").write_bytes(protocol_path.read_bytes())
            for name, value in zip(FROZEN[1:], (cases, references, expected)):
                write(output / name, value)
            preparation = {"status": "prepared", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                           "protocol_sha256": file_sha256(protocol_path), "runner_sha256": file_sha256(Path(__file__)),
                           "source_files": protocol["source_files"], "files": {n: file_sha256(output / n) for n in FROZEN},
                           "prediction_count": len(cases), "inference_executed": False}
            write(output / "preparation.json", preparation)
            print(json.dumps({k: preparation[k] for k in ("status", "protocol_sha256", "runner_sha256", "prediction_count")}))
            return
        source = (args.run_prepared or args.replay).resolve()
        protocol, cases, references, expected = frozen_load(source, protocol_path, root)
        require(expected_baseline(root, protocol) == expected, "prepared baseline differs")
        for name in (*FROZEN, "preparation.json"):
            (output / name).write_bytes((source / name).read_bytes())
        if args.replay:
            stage = "replay"
            status = read(source / "status.json")
            require(status["status"] == "complete" and status["completed_predictions"] == 94 and
                    status["protocol_sha256"] == file_sha256(protocol_path) and
                    status["intact_arrays_match_published_baseline"] is True, "cannot replay incomplete or differently bound inference")
            require(set(status["files"]) == set(RUN_FILES), "run artifact inventory mismatch")
            for name, digest in status["files"].items():
                require(Path(name).name == name and file_sha256(source / name) == digest, "run artifact hash mismatch")
            lines = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()]
            expected_header = {"record_type": "run", "protocol_sha256": file_sha256(protocol_path),
                               "preparation_sha256": file_sha256(source / "preparation.json"),
                               "model_sha256": protocol["checkpoint_sha256"], "runner_sha256": file_sha256(Path(__file__)),
                               "source_files": protocol["source_files"]}
            require(all(lines[0].get(k) == v for k, v in expected_header.items()), "raw provenance binding mismatch")
            records = lines[1:]
            (output / "predictions.jsonl").write_bytes((source / "predictions.jsonl").read_bytes())
        else:
            stage = "preinference"
            require(args.image_root and args.checkpoint_manifest, "run requires image root and checkpoint manifest")
            rebuilt_cases, rebuilt_refs, images = build(root, args.image_root.resolve(), protocol)
            require(rebuilt_cases == cases and rebuilt_refs == references, "prepared inputs cannot be reproduced")
            selected = checkpoint(root, args.checkpoint_manifest, protocol)
            write(output / "status.json", {"status": "running", "completed_predictions": 0})
            from predict_pose_folder import prepare_predictor
            predictor, runtime = prepare_predictor(selected["weights"], "cpu", 4, images[cases[0]["image_id"]], .13, .70)
            header = {"record_type": "run", "protocol_sha256": file_sha256(protocol_path),
                      "preparation_sha256": file_sha256(output / "preparation.json"), "model_sha256": selected["model_sha256"],
                      "runner_sha256": file_sha256(Path(__file__)), "source_files": protocol["source_files"],
                      "warmup": "One blank image, excluded from the 94 prediction records.",
                      "runtime": {k: v for k, v in runtime.items() if k in ("python", "torch", "ultralytics", "packages", "device_actual", "backend_end2end", "backend_fp16", "torch_threads")}}
            records = []
            with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(header, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                for case in cases:
                    stage = "intact_control" if case["condition"] == "intact" else "masked_prediction"
                    rgb = altered(images[case["image_id"]], case["mask_rect_xyxy"], protocol["masks"]["fill_rgb"])
                    require(sha(rgb.tobytes()) == case["input_rgb_sha256"], "runtime pixel hash mismatch")
                    record = {k: case[k] for k in ("case_id", "image_id", "condition", "input_rgb_sha256")}
                    record.update(record_type="prediction", **predictor(rgb))
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    completed += 1
                    records.append(record)
                    if case["condition"] == "intact":
                        require(record["predictions"] == expected[case["image_id"]], "intact raw arrays differ from published baseline")
                    print(json.dumps({"case_id": case["case_id"], "completed_predictions": completed}), flush=True)
        stage = "scoring"
        for case, record in zip(cases[:4], records[:4]):
            require(record["predictions"] == expected[case["image_id"]], "intact control replay mismatch")
        scores, summary = score_all(protocol, cases, references, records)
        write(output / "scores.json", scores)
        write(output / "summary.json", summary)
        if args.replay:
            require(scores == read(source / "scores.json") and summary == read(source / "summary.json"), "replay score mismatch")
        write(output / "status.json", {"status": "complete", "mode": "replay" if args.replay else "inference",
                                       "completed_predictions": len(records), "intact_arrays_match_published_baseline": True,
                                       "protocol_sha256": file_sha256(protocol_path), "files": {n: file_sha256(output / n) for n in RUN_FILES}})
        print(json.dumps({"status": "complete", "prediction_count": len(records), "replayed": bool(args.replay)}))
    except Exception as error:
        write(output / "status.json", {"status": "failed", "stage": stage, "completed_predictions": completed,
                                       "error_type": type(error).__name__})
        print(json.dumps({"status": "failed", "stage": stage, "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
