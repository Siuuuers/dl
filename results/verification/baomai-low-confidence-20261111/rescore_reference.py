#!/usr/bin/env python3
"""Replay the frozen seed-46 trio on PR30 and the append-only 2026-11-11 reference.

No model loading, inference, training, threshold search, or label modification.
Run from any directory; --check verifies the committed deterministic JSON.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.dont_write_bytecode = True

CAPSULE = "results/evaluation/baomai-annotations-seed46-20261111/evaluation.zip"
CAPSULE_SHA256 = "3dfe3415841b47a605e2fd4b9e490b2c12b348b0014c6c2a8c78c06f323ffc1a"
COMPARISON_MEMBER = "evidence/seed-46/comparison-original-and-pr30.json"
COMPARISON_SHA256 = "a0d74aebe68fb0c5b1e7864df53ddfa8894dd1d2484fbd77d5333b4af0afb651"
OLD_REFERENCE = "data/references/visible-front-2026-11-10"
NEW_REFERENCE = "data/references/visible-front-2026-11-11"
OLD_MANIFEST_SHA256 = "edf2d01624b96d0a4e652b1fa1fdfbc861c453f825343c840972493a908e6ed2"
# Filled only after the reference is frozen, before the first new-reference score.
NEW_MANIFEST_SHA256 = "2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c"
THRESHOLDS = [0.15, 0.25, 0.4]
RADII = [2, 5, 10]
MODEL_HASHES = {
    "m0": "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e",
    "m1": "ad99cff1bd8f252f7851fec371a9d5878244ca66a3b1865e98ae16687c22ec33",
    "m2": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
}
PREDICTION_HASHES = {
    "m0": "72786d83b433029c806e259784be1e2ba945174264c444fe1b167cf340a5fe3a",
    "m1": "dcc31c1cf4ab46ff9e07d2f1884647dff11f4496ba67cb7b4898a3a47f025ea2",
    "m2": "25b5eabecbffa13c1d01d62194dfbdebdc0fcdf914f0661123c23347c4060ee9",
}
SCORING_SOURCES = [
    "scripts/evaluate_pose.py", "scripts/verify_pose_candidate.py",
    "scripts/pose_scoring.py", "scripts/diagnose_pose_quality.py",
]
INFERENCE = {
    "imgsz": 1280, "rect": True, "batch": 1, "nms": True, "iou": 0.70,
    "max_det": 300, "quantize": 32, "save": False, "save_txt": False,
    "save_crop": False, "show": False, "verbose": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def exact(actual, expected, context):
    """No float tolerance: the frozen official results must reproduce exactly."""
    require(actual == expected, f"exact baseline/identity comparison failed: {context}")


def load_capsule(root):
    path = root / CAPSULE
    raw = path.read_bytes()
    require(digest(raw) == CAPSULE_SHA256, "pinned evaluation capsule SHA-256 mismatch")
    sidecar = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    exact(sidecar, CAPSULE_SHA256, "capsule checksum sidecar")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        require(len(names) == len(set(names)), "duplicate archive members")
        require(len(names) <= 1000 and sum(item.file_size for item in infos) <= 64 * 1024**2,
                "unexpected capsule size")
        for name in names:
            parts = PurePosixPath(name)
            require(not parts.is_absolute() and ".." not in parts.parts and "\\" not in name,
                    f"unsafe capsule member: {name}")
        inventory = strict_json(archive.read("inventory.json"))
        exact(set(names), set(inventory["payload_files"]) | {"inventory.json"}, "inventory members")
        members = {}
        for name, metadata in inventory["payload_files"].items():
            content = archive.read(name)
            exact(len(content), metadata["size_bytes"], f"member size {name}")
            exact(digest(content), metadata["sha256"], f"member SHA-256 {name}")
            members[name] = content
    exact(digest(members[COMPARISON_MEMBER]), COMPARISON_SHA256, "comparison member SHA-256")
    comparison = strict_json(members[COMPARISON_MEMBER])
    exact(comparison["procedure"]["box_confidences"], THRESHOLDS, "fixed thresholds")
    exact(comparison["procedure"]["radii_px"], RADII, "fixed radii")
    for key in ("matching_iou", "endpoint_confidence"):
        exact(comparison["procedure"][key], 0.5, key)
    exact(comparison["inference"]["inference"], INFERENCE, "fixed inference settings")
    source_hashes = {}
    for name in SCORING_SOURCES:
        source_hashes[name] = digest((root / name).read_bytes())
        exact(source_hashes[name], comparison["source"]["files_sha256"][name], name)
    return members, comparison, {
        "path": CAPSULE, "sha256": CAPSULE_SHA256, "size_bytes": len(raw),
        "inventory_verified_payload_members": len(inventory["payload_files"]),
        "comparison_member": COMPARISON_MEMBER, "comparison_sha256": COMPARISON_SHA256,
        "scoring_sources_sha256": source_hashes,
    }


def reference_counts(gold):
    objects = [obj for values in gold.values() for obj in values]
    visible = sum(point[2] == 2 for obj in objects for point in obj["keypoints"])
    return {
        "images": len(gold), "objects": len(objects), "visible_endpoints": visible,
        "missing_endpoints": 2 * len(objects) - visible,
        "empty_images": sum(not values for values in gold.values()),
    }


def eligibility(objects):
    counts = Counter(sum(point[2] == 2 for point in obj["keypoints"]) for obj in objects)
    return {str(count): counts[count] for count in range(3)}


def load_references(root, comparison, api):
    require(len(NEW_MANIFEST_SHA256) == 64, "new reference has not been frozen/pinned")
    manifests, gold, identities, label_bytes = {}, {}, {}, {}
    for name, directory, expected_hash in (
        ("pr30", OLD_REFERENCE, OLD_MANIFEST_SHA256),
        ("amended", NEW_REFERENCE, NEW_MANIFEST_SHA256),
    ):
        base = root / directory
        raw = (base / "manifest.json").read_bytes()
        exact(digest(raw), expected_hash, f"{name} manifest SHA-256")
        manifest = strict_json(raw)
        files = manifest["files"]
        require(len(files) == 20 and len({item["image_id"] for item in files}) == 20,
                f"{name} must contain 20 unique images")
        manifests[name], gold[name], label_bytes[name] = manifest, {}, {}
        identities[name] = {"path": directory, "manifest_sha256": expected_hash, "files": []}
        for item in files:
            image_id = item["image_id"]
            exact(item["label_path"], f"labels/val/{image_id}.txt", f"label mapping {image_id}")
            exact([item["width"], item["height"]], [1280, 720], f"dimensions {image_id}")
            data = (base / item["label_path"]).read_bytes()
            exact(digest(data), item["label_sha256"], f"{name} label SHA-256 {image_id}")
            text = data.decode("utf-8")
            require(all(line.strip() for line in text.splitlines()), f"blank label row: {image_id}")
            objects = api["parse_pose_labels"](text, 1280, 720, f"{directory}/{item['label_path']}")
            api["check_geometry"](objects, 1280, 720, f"{directory}/{item['label_path']}")
            exact(len(objects), item["row_count"], f"{name} row count {image_id}")
            gold[name][image_id], label_bytes[name][image_id] = objects, data
            identities[name]["files"].append({key: item[key] for key in
                ("image_id", "label_path", "label_sha256", "image_sha256", "width", "height", "row_count")})
        exact(reference_counts(gold[name]), manifest["counts"], f"{name} counts")
        exact(eligibility([obj for values in gold[name].values() for obj in values]),
              manifest["objects_by_locatable_corners"], f"{name} corner eligibility")
        identities[name]["counts"] = manifest["counts"]
        identities[name]["objects_by_locatable_corners"] = manifest["objects_by_locatable_corners"]

    exact(manifests["pr30"]["files"], comparison["revised_reference"]["files"], "frozen PR30 file manifest")
    exact(comparison["revised_reference"]["manifest_sha256"], OLD_MANIFEST_SHA256, "frozen PR30 identity")
    exact(set(gold["pr30"]), set(gold["amended"]), "reference image set")
    old_files = {item["image_id"]: item for item in manifests["pr30"]["files"]}
    new_files = {item["image_id"]: item for item in manifests["amended"]["files"]}
    for image_id in sorted(gold["pr30"]):
        exact(new_files[image_id]["image_sha256"], old_files[image_id]["image_sha256"],
              f"unchanged image identity {image_id}")
        before, after = label_bytes["pr30"][image_id], label_bytes["amended"][image_id]
        if image_id != "new_000090":
            exact(after, before, f"unchanged label bytes {image_id}")
        else:
            require(before.endswith(b"\n") and after.startswith(before), "090 retained label bytes changed")
            exact(len(gold["pr30"][image_id]), 47, "090 old rows")
            exact(len(gold["amended"][image_id]), 49, "090 amended rows")
            exact(len(after[len(before):].splitlines()), 2, "090 appended lines")
            for row in gold["amended"][image_id][47:]:
                exact(row["keypoints"], [[0.0, 0.0, 0], [0.0, 0.0, 0]], "appended missing corner sentinels")
    exact(reference_counts(gold["pr30"]),
          {"images": 20, "objects": 661, "visible_endpoints": 1228, "missing_endpoints": 94, "empty_images": 3},
          "PR30 fixed denominators")
    exact(reference_counts(gold["amended"]),
          {"images": 20, "objects": 663, "visible_endpoints": 1228, "missing_endpoints": 98, "empty_images": 3},
          "amended fixed denominators")
    return gold, identities


def load_predictions(root, members, comparison, reference):
    records, models = {}, {}
    frames = {item["id"]: item for item in comparison["inference"]["images"]}
    exact(set(frames), set(reference), "inference/reference image identities")
    for item in comparison["revised_reference"]["files"]:
        for field in ("image_sha256", "width", "height"):
            exact(item[field], frames[item["image_id"]][field], f"reference/inference {item['image_id']}/{field}")
    exact(digest((root / "data/manifest.json").read_bytes()),
          comparison["original_reference"]["manifest_sha256"], "original data manifest SHA-256")
    for image_id, item in frames.items():
        exact(digest((root / "data" / item["label"]).read_bytes()), item["label_sha256"],
              f"original inference-label SHA-256 {image_id}")
    for model, model_hash in MODEL_HASHES.items():
        member = f"evidence/seed-46/{model}/predictions.jsonl"
        data = members[member]
        exact(digest(data), PREDICTION_HASHES[model], f"{model} predictions SHA-256")
        lines = [strict_json(line) for line in data.splitlines()]
        header, items = lines[0], lines[1:]
        exact(header["record_type"], "run", f"{model} header type")
        exact(header["schema_version"], 1, f"{model} header schema")
        exact(header["model_sha256"], model_hash, f"{model} checkpoint SHA-256")
        exact(header["model_sha256"], comparison["models"][model]["model_sha256"], f"{model} official identity")
        exact(header.get("checkpoint_identity"), comparison["models"][model]["checkpoint_identity"],
              f"{model} checkpoint metadata")
        exact(header["inference"], INFERENCE, f"{model} inference settings")
        exact(header["box_confidences"], THRESHOLDS, f"{model} box thresholds")
        exact(header["source"], comparison["inference"]["source"], f"{model} inference source")
        exact(header["runtime"], comparison["inference"]["runtime"], f"{model} inference runtime")
        exact(header["images"], comparison["inference"]["images"], f"{model} inference image manifest")
        exact(header["manifest_sha256"], comparison["original_reference"]["manifest_sha256"],
              f"{model} original data manifest")
        require(len(items) == 60, f"{model} must contain all 60 image/threshold records")
        for item in items:
            exact(item["record_type"], "prediction", f"{model} prediction type")
            image_id, threshold = item["image_id"], item["threshold"]
            require(image_id in frames and threshold in THRESHOLDS, "unexpected prediction record")
            key = model, threshold, image_id
            require(key not in records, f"duplicate prediction record: {key}")
            frame = frames[image_id]
            for field in ("image_sha256", "label_sha256", "width", "height"):
                exact(item[field], frame[field], f"{key}/{field}")
            require(len(item["predictions"]) <= INFERENCE["max_det"], "max_det exceeded")
            require(all(obj["class_id"] == 0 and obj["confidence"] >= threshold for obj in item["predictions"]),
                    f"unexpected class or confidence below stored threshold: {key}")
            records[key] = item["predictions"]
        models[model] = {
            **comparison["models"][model], "prediction_member": member,
            "prediction_member_sha256": PREDICTION_HASHES[model],
            "image_threshold_records": len(items),
        }
    exact(set(records), {(model, threshold, image_id) for model in MODEL_HASHES
          for threshold in THRESHOLDS for image_id in frames}, "all prediction combinations")
    return records, models, frames


def public_score(score, include_prediction_indices=False):
    return {key: value for key, value in score.items()
            if key not in ("endpoint_errors_px", "matches", "unmatched_gold_indices")
            and (include_prediction_indices or not key.endswith("_prediction_indices"))}


def image_view(gold, predictions, api):
    score = api["score_image"](gold, predictions, 1280, 720, 0.5, 0.5)
    quality = api["quality_for_image"](gold, predictions, 1280, 720, RADII)
    return {
        "score": score, "quality": quality, "objects_by_locatable_corners": eligibility(gold),
        "misses_by_locatable_corners": eligibility([gold[index] for index in score["unmatched_gold_indices"]]),
        "missed_label_rows_one_based": [index + 1 for index in score["unmatched_gold_indices"]],
    }


def aggregate(views, api):
    return {
        "score": public_score(api["aggregate_scores"]([view["score"] for view in views])),
        "quality": api["aggregate_quality"]([view["quality"] for view in views]),
        **{field: {str(corners): sum(view[field][str(corners)] for view in views) for corners in range(3)}
           for field in ("objects_by_locatable_corners", "misses_by_locatable_corners")},
    }


def scene_group(views, api):
    result = aggregate(views, api)
    result["scenes"] = {
        "image_count": len(views),
        "with_any_missed_object": sum(view["score"]["fn"] > 0 for view in views),
        "with_any_extra_proposal": sum(view["score"]["fp"] > 0 for view in views),
        "with_neither_missed_nor_extra": sum(view["score"]["fn"] == 0 and view["score"]["fp"] == 0 for view in views),
        "with_missing_visible_endpoints": sum(view["score"]["missing_endpoints"] > 0 for view in views),
    }
    return result


def numeric_delta(before, after):
    return {key: after[key] - value for key, value in before.items()
            if type(value) in (int, float) and type(after.get(key)) in (int, float)}


def view_delta(before, after):
    return {
        "score": numeric_delta(before["score"], after["score"]),
        "quality": numeric_delta(before["quality"], after["quality"]),
        "radii": [{"radius_px": first["radius_px"], **numeric_delta(
            {key: value for key, value in first.items() if key != "radius_px"}, second)}
            for first, second in zip(before["quality"]["radii"], after["quality"]["radii"])],
        "misses_by_locatable_corners": numeric_delta(before["misses_by_locatable_corners"],
                                                       after["misses_by_locatable_corners"]),
    }


def returned(point):
    x, y, confidence = point
    return (confidence >= 0.5 and x is not None and y is not None
            and math.isfinite(x) and math.isfinite(y) and 0 <= x < 1280 and 0 <= y < 720)


def match_details(pair, predictions):
    if pair is None:
        return None
    index = pair["prediction_index"]
    prediction = predictions[index]
    return {
        "prediction_index_zero_based": index, "prediction_number_one_based": index + 1,
        "confidence": prediction["confidence"], "iou": pair["iou"],
        "endpoint_confidences": [point[2] for point in prediction["keypoints"]],
        "returned_endpoint_slots_zero_based": [slot for slot, point in enumerate(prediction["keypoints"]) if returned(point)],
    }


def endpoint_state(obj, pair, predictions, slot):
    state = {"reference_visibility": obj["keypoints"][slot][2], "returned": False,
             "prediction_index_zero_based": None, "confidence": None, "scored_error_px": None}
    if pair is not None:
        index = pair["prediction_index"]
        point = predictions[index]["keypoints"][slot]
        state.update(prediction_index_zero_based=index, confidence=point[2], returned=returned(point))
        if state["returned"] and state["reference_visibility"] == 2:
            gx, gy, _ = obj["keypoints"][slot]
            state["scored_error_px"] = math.hypot(point[0] - gx, point[1] - gy)
    return state


def match_changes(old_gold, new_gold, predictions, before, after):
    old = {pair["gold_index"]: pair for pair in before["score"]["matches"]}
    new = {pair["gold_index"]: pair for pair in after["score"]["matches"]}
    changes = {"retained_rows_lost": [], "retained_rows_gained": [],
               "retained_rows_reassigned": [], "retained_endpoint_changes": [], "added_rows": []}
    for row, obj in enumerate(old_gold):
        first, second = old.get(row), new.get(row)
        if first is None and second is not None:
            changes["retained_rows_gained"].append(row + 1)
        elif first is not None and second is None:
            changes["retained_rows_lost"].append(row + 1)
        elif first is not None and second is not None and first["prediction_index"] != second["prediction_index"]:
            changes["retained_rows_reassigned"].append({
                "row_one_based": row + 1, "before": match_details(first, predictions),
                "after": match_details(second, predictions)})
        for slot in range(2):
            previous = endpoint_state(obj, first, predictions, slot)
            current = endpoint_state(obj, second, predictions, slot)
            if previous != current:
                changes["retained_endpoint_changes"].append({"row_one_based": row + 1,
                    "slot_zero_based": slot, "before": previous, "after": current})
    for row in range(len(old_gold), len(new_gold)):
        changes["added_rows"].append({"row_one_based": row + 1,
            "bbox_xyxy": new_gold[row]["bbox_xyxy"], "reference_locatable_corners": 0,
            "match": match_details(new.get(row), predictions)})
    return changes


def run(root):
    members, comparison, capsule = load_capsule(root)
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels
    from verify_pose_candidate import check_geometry
    from pose_scoring import score_image, aggregate_scores
    from diagnose_pose_quality import quality_for_image, aggregate_quality
    api = {function.__name__: function for function in (
        parse_pose_labels, check_geometry, score_image, aggregate_scores,
        quality_for_image, aggregate_quality)}
    gold, references = load_references(root, comparison, api)
    records, models, frames = load_predictions(root, members, comparison, gold["pr30"])
    image_ids = sorted(frames)
    official = {(row["model_id"], row["box_confidence"], row["image_id"]): row["revised"]
                for row in comparison["per_image"]}
    exact(len(official), 180, "official per-image combinations")
    exact(set(official), set(records), "official per-image keys")
    official_aggregates = {(row["model_id"], row["box_confidence"]): row["revised"]
                          for row in comparison["metrics"]}
    exact(len(official_aggregates), 9, "official aggregate combinations")

    # Complete all old-reference exact comparisons before any new-reference scoring.
    baseline = {}
    for key, predictions in records.items():
        view = image_view(gold["pr30"][key[2]], predictions, api)
        expected = official[key]
        exact(public_score(view["score"], True), expected["score"], f"score {key}")
        for field in ("quality", "objects_by_locatable_corners", "misses_by_locatable_corners",
                      "missed_label_rows_one_based"):
            exact(view[field], expected[field], f"{field} {key}")
        baseline[key] = view
    for model in MODEL_HASHES:
        for threshold in THRESHOLDS:
            pooled = aggregate([baseline[(model, threshold, image_id)] for image_id in image_ids], api)
            exact(pooled, official_aggregates[(model, threshold)], f"aggregate {model}/{threshold}")

    amended, per_image, metrics = {}, [], []
    for model in MODEL_HASHES:
        for threshold in THRESHOLDS:
            for image_id in image_ids:
                key = model, threshold, image_id
                predictions, before = records[key], baseline[key]
                after = image_view(gold["amended"][image_id], predictions, api)
                amended[key] = after
                changes = match_changes(gold["pr30"][image_id], gold["amended"][image_id], predictions, before, after)
                per_image.append({
                    "model_id": model, "box_confidence": threshold, "image_id": image_id,
                    "scene": "nonempty" if gold["pr30"][image_id] else "empty",
                    "pr30": {**before, "score": public_score(before["score"])},
                    "amended": {**after, "score": public_score(after["score"])},
                    "amended_minus_pr30": view_delta(before, after), "match_changes": changes,
                })
            old_views = [baseline[(model, threshold, image_id)] for image_id in image_ids]
            new_views = [amended[(model, threshold, image_id)] for image_id in image_ids]
            old_aggregate, new_aggregate = aggregate(old_views, api), aggregate(new_views, api)
            grouped = {}
            for scene, nonempty in (("nonempty", True), ("empty", False)):
                ids = [image_id for image_id in image_ids if bool(gold["pr30"][image_id]) == nonempty]
                grouped[scene] = {"image_ids": ids,
                    "pr30": scene_group([baseline[(model, threshold, image_id)] for image_id in ids], api),
                    "amended": scene_group([amended[(model, threshold, image_id)] for image_id in ids], api)}
            metrics.append({"model_id": model, "box_confidence": threshold, "primary": threshold == 0.25,
                "pr30": old_aggregate, "amended": new_aggregate,
                "amended_minus_pr30": view_delta(old_aggregate, new_aggregate), "by_scene": grouped})

    changed = [row for row in per_image if row["image_id"] == "new_000090"]
    return {
        "schema_version": 1, "kind": "frozen-seed46-append-only-reference-replay", "status": "verified",
        "capsule": capsule, "models": models, "references": references,
        "generator": {"path": str(Path(__file__).resolve().relative_to(root)),
                      "sha256": digest(Path(__file__).read_bytes())},
        "procedure": {
            "models": list(MODEL_HASHES), "box_confidences": THRESHOLDS, "primary_box_confidence": 0.25,
            "matching_iou": 0.5, "endpoint_confidence": 0.5, "radii_px": RADII,
            "inference_settings": INFERENCE, "new_inference": False, "new_training": False,
            "weights_loaded": False, "prediction_coordinates_modified": False,
            "prediction_order_modified": False, "threshold_tuning": False,
            "reference_change": "Only new_000090 rows 48 and 49 appended; all prior label bytes preserved.",
            "row_numbering": "One-based physical label lines; prediction indices explicitly zero- and one-based.",
            "comparison_direction": "Amended reference minus PR30 on identical stored predictions.",
        },
        "verification": {
            "all_capsule_inventory_members_verified": True, "all_20_labels_in_each_reference_verified": True,
            "all_geometry_checks_passed": True, "retained_label_bytes_exact": True,
            "old_reference_per_image_scores_exact": 180, "old_reference_per_image_quality_exact": 180,
            "old_reference_aggregate_scores_and_quality_exact": 9,
            "image_identity_check": "All frozen inference/header/reference hashes agree; PNGs are not needed or reloaded for replay.",
        },
        "summary": {
            "retained_row_matches_lost": sum(len(row["match_changes"]["retained_rows_lost"]) for row in per_image),
            "retained_row_matches_gained": sum(len(row["match_changes"]["retained_rows_gained"]) for row in per_image),
            "retained_row_match_reassignments": sum(len(row["match_changes"]["retained_rows_reassigned"]) for row in per_image),
            "retained_endpoint_changes": sum(len(row["match_changes"]["retained_endpoint_changes"]) for row in per_image),
            "added_row_matches_by_combination": [{"model_id": row["model_id"],
                "box_confidence": row["box_confidence"], "rows": row["match_changes"]["added_rows"]} for row in changed],
        },
        "metrics": metrics, "per_image": per_image,
        "limits": [
            "Reference-driven metric changes do not mean that any model learned or improved.",
            "These are the same 20 validation images; scene dependence and prior reference uncertainties remain.",
            "The two new fronts have no locatable lower corners; detected boxes cannot establish useful corner poses.",
            "The three stored thresholds are fixed comparisons, not new threshold selection or tuning.",
            "No independent scene test, production acceptance, or model promotion is implied.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().with_name("reference-replay.json.gz"),
                        help="result path; .gz uses deterministic gzip, other suffixes use plain JSON")
    parser.add_argument("--check", action="store_true", help="compare with existing result without writing")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    try:
        result = run(root)
        serialized = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        compressed = output.suffix == ".gz"
        payload = serialized
        if compressed:
            buffer = io.BytesIO()
            with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
                stream.write(serialized)
            payload = buffer.getvalue()
        if args.check:
            existing = output.read_bytes()
            decoded = gzip.decompress(existing) if compressed else existing
            exact(decoded, serialized, "complete reference replay content")
            exact(existing, payload, "deterministic reference replay artifact bytes")
        else:
            output.write_bytes(payload)
        print(json.dumps({"status": "verified", "check_only": args.check,
            "output": str(output.relative_to(root)) if output.is_relative_to(root) else str(output),
            "model_threshold_combinations": len(result["metrics"]),
            "per_image_combinations": len(result["per_image"]), "gzip": compressed,
            "size_bytes": len(payload), "sha256": digest(payload), "content_sha256": digest(serialized)}))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"reference replay failed: {error}\n")


if __name__ == "__main__":
    main()
