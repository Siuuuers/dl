#!/usr/bin/env python3
"""Replay five saved models against frozen and separately amended references.

This diagnostic performs no inference or training and never edits its inputs.
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

THRESHOLDS = [0.15, 0.25, 0.40]
RADII = [2, 5, 10]
MODELS = (
    ("m0", 43, "m0", None),
    ("seed43_reference", 43, "m1", "reference"),
    ("seed43_candidate", 43, "m2", "candidate"),
    ("seed44_reference", 44, "m1", "reference"),
    ("seed44_candidate", 44, "m2", "candidate"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"nonstandard JSON constant in {path}: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant, object_pairs_hook=unique_object)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_api(root):
    # When installed in scripts/, use adjacent evaluator modules. A standalone
    # copy uses the explicitly supplied original repository's scripts directory.
    adjacent = Path(__file__).resolve().parent
    code_dir = adjacent if (adjacent / "evaluate_pose.py").is_file() else root / "scripts"
    names = ("evaluate_pose.py", "pose_checkpoint.py", "pose_scoring.py",
             "diagnose_pose_quality.py", "verify_pose_candidate.py")
    require(all((code_dir / name).is_file() for name in names),
            f"missing existing evaluator modules in {code_dir}")
    sys.path.insert(0, str(code_dir))
    from evaluate_pose import _read_replay, _relative_file, parse_pose_labels, preflight_dataset
    from pose_checkpoint import select_checkpoint
    from pose_scoring import aggregate_scores, score_image
    from diagnose_pose_quality import aggregate_quality, quality_for_image
    from verify_pose_candidate import check_geometry
    return code_dir, locals()


def line_numbers(text):
    return [number for number, line in enumerate(text.splitlines(), 1) if line.strip()]


def verify_amendment(root, reference_dir, original, api):
    manifest_path = reference_dir / "manifest.json"
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "reference manifest must be an object")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
            "unsupported reference schema_version")
    reference_id = manifest.get("reference_id")
    require(isinstance(reference_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", reference_id),
            "reference_id must be a nonempty safe identifier")
    require(manifest.get("source_manifest_sha256") == original["manifest_sha256"],
            "amended reference names a different source manifest")
    entries = manifest.get("files")
    require(isinstance(entries, list) and len(entries) == 20,
            "reference manifest must contain all 20 validation label files")
    records = {record["id"]: record for record in original["records"]}
    amended, source_lines, amended_lines, provenance = {}, {}, {}, []
    listed_paths = set()
    for entry in entries:
        require(isinstance(entry, dict), "each reference file entry must be an object")
        identifier = entry.get("image_id")
        require(isinstance(identifier, str) and identifier in records and identifier not in amended,
                f"unknown or duplicate reference image: {identifier}")
        record = records[identifier]
        source_path = f"data/{record['label']}"
        label_path = f"labels/val/{identifier}.txt"
        require(entry.get("source_label_path") == source_path and entry.get("label_path") == label_path,
                f"incorrect source/amended label path: {identifier}")
        for field in ("source_label_sha256", "label_sha256", "image_sha256"):
            value = entry.get(field)
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
                    f"invalid {field}: {identifier}")
        require(entry["source_label_sha256"] == record["label_sha256"]
                and entry["image_sha256"] == record["image_sha256"],
                f"source label/image identity differs: {identifier}")
        for field in ("width", "height"):
            require(type(entry.get(field)) is int and entry[field] == record[field],
                    f"original {field} differs: {identifier}")
        for field, expected in (("source_row_count", len(record["gold"])),):
            require(type(entry.get(field)) is int and entry[field] == expected,
                    f"{field} differs: {identifier}")
        original_path = api["_relative_file"](root, source_path)
        path = api["_relative_file"](reference_dir, label_path)
        require(sha256(original_path) == entry["source_label_sha256"],
                f"source label changed after preflight: {identifier}")
        require(sha256(path) == entry["label_sha256"], f"amended label SHA-256 mismatch: {identifier}")
        text = path.read_text(encoding="utf-8")
        gold = api["parse_pose_labels"](text, record["width"], record["height"], str(path))
        api["check_geometry"](gold, record["width"], record["height"], str(path))
        require(type(entry.get("row_count")) is int and entry["row_count"] == len(gold),
                f"amended row_count differs: {identifier}")
        amended[identifier] = gold
        source_lines[identifier] = line_numbers(original_path.read_text(encoding="utf-8"))
        amended_lines[identifier] = line_numbers(text)
        listed_paths.add(label_path)
        provenance.append(dict(entry))
    require(set(amended) == set(records), "reference image set differs from all 20 validation images")
    labels_dir = reference_dir / "labels"
    require(not labels_dir.is_symlink(), "amended labels directory must not be a symlink")
    actual_paths = {path.relative_to(reference_dir).as_posix() for path in labels_dir.rglob("*")
                    if path.is_file() or path.is_symlink()}
    require(actual_paths == listed_paths, "unexpected or missing files in amended labels directory")
    return manifest, amended, source_lines, amended_lines, sorted(provenance, key=lambda row: row["image_id"])


def eligibility(gold):
    counts = {str(n): 0 for n in range(3)}
    for obj in gold:
        counts[str(sum(point[2] == 2 for point in obj["keypoints"]))] += 1
    return counts


def describe_image(gold, predictions, record, physical_lines, label_path, api):
    score = api["score_image"](gold, predictions, record["width"], record["height"], 0.50, 0.50)
    quality = api["quality_for_image"](gold, predictions, record["width"], record["height"], RADII)
    missed = score["unmatched_gold_indices"]
    visible = eligibility(gold)
    missed_visible = eligibility([gold[index] for index in missed])
    public = {
        "label_path": label_path,
        "score": {key: value for key, value in score.items()
                  if key not in ("endpoint_errors_px", "matches", "unmatched_gold_indices")},
        "quality": quality,
        "objects_by_locatable_corners": visible,
        "misses_by_locatable_corners": missed_visible,
        "missed_label_rows_one_based": [physical_lines[index] for index in missed],
    }
    return public, score, quality


def numeric_delta(before, after):
    return {key: after[key] - value for key, value in before.items()
            if type(value) in (int, float) and type(after.get(key)) in (int, float)}


def compare(before, after):
    return {
        "score": numeric_delta(before["score"], after["score"]),
        "quality": numeric_delta(before["quality"], after["quality"]),
        "radii": [{"radius_px": first["radius_px"],
                   **{key: value for key, value in numeric_delta(first, second).items() if key != "radius_px"}}
                  for first, second in zip(before["quality"]["radii"], after["quality"]["radii"])],
        "misses_by_locatable_corners": numeric_delta(before["misses_by_locatable_corners"],
                                                      after["misses_by_locatable_corners"]),
    }


def run(args):
    root, reference_dir, evidence = args.root.resolve(), args.reference_dir.resolve(), args.evidence.resolve()
    require(not args.output.exists(), f"output already exists: {args.output}")
    code_dir, api = load_api(root)
    original = api["preflight_dataset"](root, require_weights=False)
    if (evidence / "evidence").is_dir():
        evidence = evidence / "evidence"
    # Validate every ORIGINAL replay before applying the amended gold. No replay
    # header or original label identity is ever replaced to admit an amendment.
    verified = []
    for model_id, seed, folder, arm in MODELS:
        base = evidence / f"seed-{seed}"
        checkpoint = base / "checkpoints" / arm / "checkpoint-manifest.json" if arm else None
        selected = original if checkpoint is None else api["select_checkpoint"](
            original, checkpoint, require_weights=False)
        path = base / folder / "predictions.jsonl"
        header, items, thresholds = api["_read_replay"](path, selected, None)
        require(set(thresholds) == set(THRESHOLDS) and len(thresholds) == 3 and len(items) == 60,
                f"expected all 20 images at 0.15, 0.25 and 0.40: {model_id}")
        identity = {"model_id": model_id, "model_sha256": header["model_sha256"],
                    "predictions_path": str(path), "predictions_sha256": sha256(path),
                    "checkpoint_identity": header.get("checkpoint_identity"),
                    "inference": header["inference"], "original_inference_source": header["source"]}
        if checkpoint:
            identity["checkpoint_manifest_sha256"] = sha256(checkpoint)
        verified.append((identity, items))
    manifest, amended, source_lines, amended_lines, files = verify_amendment(root, reference_dir, original, api)
    records = {record["id"]: record for record in original["records"]}
    aggregates, per_image = [], []
    for identity, items in verified:
        for threshold in THRESHOLDS:
            scores, qualities, views = ({name: [] for name in ("original", "amended")} for _ in range(3))
            chosen = sorted((item for item in items if item["threshold"] == threshold), key=lambda row: row["image_id"])
            for item in chosen:
                identifier, predictions = item["image_id"], item["predictions"]
                record = records[identifier]
                pair = {"model_id": identity["model_id"], "box_confidence": threshold, "image_id": identifier}
                for name, gold, lines, path in (
                    ("original", record["gold"], source_lines[identifier], f"data/{record['label']}"),
                    ("amended", amended[identifier], amended_lines[identifier], f"labels/val/{identifier}.txt"),
                ):
                    public, score, quality = describe_image(gold, predictions, record, lines, path, api)
                    pair[name] = public
                    scores[name].append(score)
                    qualities[name].append(quality)
                    views[name].append(public)
                pair["amended_minus_original"] = compare(pair["original"], pair["amended"])
                per_image.append(pair)
            result = {"model_id": identity["model_id"], "box_confidence": threshold}
            for name in ("original", "amended"):
                score = api["aggregate_scores"](scores[name])
                score.pop("endpoint_errors_px")
                result[name] = {"score": score, "quality": api["aggregate_quality"](qualities[name]),
                    **{field: {str(n): sum(view[field][str(n)] for view in views[name]) for n in range(3)}
                       for field in ("objects_by_locatable_corners", "misses_by_locatable_corners")}}
            result["amended_minus_original"] = compare(result["original"], result["amended"])
            aggregates.append(result)
    return {
        "schema_version": 1, "kind": "baomai-reference-amendment-replay", "status": "complete",
        "diagnostic_only": True, "new_inference": False, "new_training": False,
        "weights_loaded_or_verified": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"script_sha256": sha256(Path(__file__)), "evaluator_code_directory": str(code_dir),
                   "files_sha256": {path.name: sha256(path) for path in (code_dir / name for name in
                       ("evaluate_pose.py", "pose_scoring.py", "pose_checkpoint.py", "diagnose_pose_quality.py", "verify_pose_candidate.py"))}},
        "original_reference": {"root": str(root), "manifest_sha256": original["manifest_sha256"],
                               "model_card_sha256": original["model_card_sha256"]},
        "amended_reference": {"reference_id": manifest["reference_id"], "root": str(reference_dir),
                              "manifest_sha256": sha256(reference_dir / "manifest.json"), "files": files},
        "procedure": {"box_confidences": THRESHOLDS, "matching_iou": 0.50, "endpoint_confidence": 0.50,
                      "radii_px": RADII, "new_acceptance_decision": False,
                      "row_numbering": "One-based physical lines in the separately named original or amended label file; never treat amended row numbers as original row identities.",
                      "original_label_path_base": str(root), "amended_label_path_base": str(reference_dir),
                      "delta_direction": "amended minus original; a reference effect, not a model improvement",
                      "both_corner_denominator": "Every object with two locatable reference corners, including misses; recomputed separately for each reference.",
                      "zero_denominator": "null; a delta involving null is omitted",
                      "geometry_epsilon_px": 0.0001,
                      "geometry_epsilon_purpose": "Serialization arithmetic only; not annotation accuracy."},
        "verification": {"original_images_and_labels_hash_verified": True,
                         "original_replay_identity_verified_before_amendment": True,
                         "complete_amended_reference_and_geometry_verified": True,
                         "unique_models": 5, "images": 20, "image_threshold_records": len(per_image),
                         "model_threshold_aggregates": len(aggregates)},
        "limits": ["Development validation already used for model selection; this is not an independent test.",
                   "Amended-label effects must not be attributed to new training or rewritten into historical benchmark results.",
                   "Coordinate radii are measured in original image pixels; no physical-unit guarantee follows.",
                   "Geometry checks establish file consistency, not correctness of visual annotation judgments."],
        "models": [identity for identity, _ in verified], "metrics": aggregates, "per_image": per_image,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original repository with all 20 hydrated validation images")
    parser.add_argument("--reference-dir", type=Path, required=True, help="versioned manifest.json plus all labels/val files")
    parser.add_argument("--evidence", type=Path, required=True, help="extracted capsule, or its evidence directory")
    parser.add_argument("--output", type=Path, required=True, help="new diagnostic JSON file; existing file refused")
    args = parser.parse_args(argv)
    try:
        result = run(args)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print(json.dumps({"status": "complete", "output": str(args.output.resolve()),
                          "reference_id": result["amended_reference"]["reference_id"],
                          **result["verification"]}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, ImportError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
