#!/usr/bin/env python3
"""Probe one additional NMS pass on final clipped PR123 boxes; no inference.

The fixed extra pass uses IoU > 0.70 and confidence-descending, stable ordering.
It is an exploratory output-stage comparison, not an adopted detector setting.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CACHE = "results/evaluation/pr123-r37-20261126/predictions/candidate.jsonl"
COMPARISON = "results/evaluation/pr123-r37-20261126/comparison.json"
DIRECTORY = "results/evaluation/pr123-r39-diagnosis-20261126"
TRACE = f"{DIRECTORY}/diagnosis.json"
REFERENCES = {
    "r38": "data/reviews/fronts-2026-11-26-r38/manifest.json",
    "frozen666": "data/references/sequence-refined-reference-2026-11-18/manifest.json",
}
SCORERS = ("scripts/evaluate_pose.py", "scripts/pose_scoring.py",
           "scripts/pose_checkpoint.py", "scripts/diagnose_pose_quality.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl_bytes(rows):
    return "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()


def clipped_nms(predictions):
    """Return primary zero-based kept indices and exact suppression pairs."""
    from pose_scoring import _box_iou
    kept = []; dropped = []
    for i in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i)):
        current = predictions[i]
        suppressor = next((j for j in kept
                           if predictions[j]["class_id"] == current["class_id"]
                           and _box_iou(predictions[j]["bbox_xyxy"], current["bbox_xyxy"]) > .70), None)
        if suppressor is None:
            kept.append(i)
        else:
            winner = predictions[suppressor]
            dropped.append({
                "primary_prediction_id": i + 1,
                "suppressor_primary_prediction_id": suppressor + 1,
                "class_id": current["class_id"],
                "confidence": current["confidence"],
                "suppressor_confidence": winner["confidence"],
                "bbox_xyxy": current["bbox_xyxy"],
                "suppressor_bbox_xyxy": winner["bbox_xyxy"],
                "postclip_prediction_iou": _box_iou(winner["bbox_xyxy"], current["bbox_xyxy"]),
            })
    return kept, dropped


def build_probe(root):
    """Regenerate fixed post-clip NMS and score against two separate references."""
    from evaluate_pose import INFERENCE, parse_pose_labels
    from pose_scoring import _box_iou, aggregate_scores, score_image
    from diagnose_pose_quality import aggregate_quality, quality_for_image
    root = Path(root).resolve()
    comparison = read_json(root / COMPARISON)
    if sha(root / CACHE) != comparison["models"]["candidate"]["predictions_sha256"]:
        raise ValueError("Accepted PR123 candidate cache changed")
    header, *rows = [json.loads(line) for line in (root / CACHE).read_text().splitlines()]
    if header["inference"] != INFERENCE or header["box_confidences"] != [.13]:
        raise ValueError("Expected fixed confidence .13 and ordinary NMS .70 input cache")
    if len(rows) != 20 or len({row["image_id"] for row in rows}) != 20:
        raise ValueError("Expected twenty distinct accepted development images")
    trace = read_json(root / TRACE)
    if trace["provenance"]["accepted_cache_sha256"] != sha(root / CACHE):
        raise ValueError("Raw trace belongs to another primary prediction cache")
    captures = {row["image_id"]: row for row in trace["per_image"]}
    per_image = []; probe_rows = []; suppressed = []
    primary_rows = {row["image_id"]: row for row in rows}
    probe_lookup = {}
    for row in rows:
        name = row["image_id"]; predictions = row["predictions"]
        kept, dropped = clipped_nms(predictions)
        input_boxes = captures[name]["baseline_retained_input_boxes_xyxy"]
        if len(input_boxes) != len(predictions):
            raise ValueError("Input-space boxes do not align with primary outputs")
        for item in dropped:
            a = input_boxes[item["primary_prediction_id"] - 1]
            b = input_boxes[item["suppressor_primary_prediction_id"] - 1]
            item.update(image_id=name, input_bbox_xyxy=a, suppressor_input_bbox_xyxy=b,
                        preclip_prediction_iou=_box_iou(a, b))
        suppressed.extend(dropped)
        updated = {"record_type": "prediction", "image_id": name,
                   "image_sha256": row["image_sha256"], "width": row["width"], "height": row["height"],
                   "threshold": .13, "primary_prediction_ids": [i + 1 for i in kept],
                   "predictions": [predictions[i] for i in kept]}
        probe_rows.append(updated); probe_lookup[name] = updated
        per_image.append({"image_id": name, "image_sha256": row["image_sha256"],
                          "primary_outputs": len(predictions), "probe_outputs": len(kept),
                          "retained_primary_prediction_ids": updated["primary_prediction_ids"],
                          "dropped_primary_prediction_ids": [item["primary_prediction_id"] for item in dropped]})
    references = {}; all_label_hashes = {}
    for name, relative in REFERENCES.items():
        manifest_path = root / relative
        manifest = read_json(manifest_path)
        if sha(manifest_path) != comparison["references"][name]["manifest_sha256"]:
            raise ValueError(f"Reference manifest changed: {name}")
        files = [f for f in manifest["files"] if f.get("split", "val") == "val"]
        if len(files) != 20 or {f["image_id"] for f in files} != set(primary_rows):
            raise ValueError(f"Reference image identities differ: {name}")
        scores = {"baseline": [], "probe": []}; qualities = {"baseline": [], "probe": []}
        matched = {"baseline": {}, "probe": {}}; miss_ids = {"baseline": [], "probe": []}
        empty = {"baseline": [], "probe": []}; label_hashes = {}
        for item in files:
            image_id = item["image_id"]; original = primary_rows[image_id]
            image_hash = item.get("source_image_sha256", item.get("image_sha256"))
            if original["image_sha256"] != image_hash:
                raise ValueError(f"Reference/cached image identity differs: {image_id}")
            label = manifest_path.parent / item["label_path"]
            if sha(label) != item["label_sha256"]:
                raise ValueError(f"Reference label changed: {label}")
            label_hashes[str(label.relative_to(root))] = item["label_sha256"]
            gold = parse_pose_labels(label.read_text(), original["width"], original["height"])
            for role, selected in (("baseline", original), ("probe", probe_lookup[image_id])):
                predictions = selected["predictions"]
                score = score_image(gold, predictions, original["width"], original["height"], .5, .5)
                scores[role].append(score)
                qualities[role].append(quality_for_image(gold, predictions, original["width"], original["height"], [5, 10]))
                ids = (list(range(1, len(predictions) + 1)) if role == "baseline"
                       else selected["primary_prediction_ids"])
                for pair in score["matches"]:
                    matched[role][f"{image_id}:{pair['gold_index'] + 1}"] = ids[pair["prediction_index"]]
                miss_ids[role].extend(f"{image_id}:{i + 1}" for i in score["unmatched_gold_indices"])
                if not gold:
                    empty[role].append({"image_id": image_id, "outputs": len(predictions)})
        reference = {"manifest": relative, "manifest_sha256": sha(manifest_path), "label_files_sha256": label_hashes}
        all_label_hashes.update(label_hashes)
        for role in ("baseline", "probe"):
            detection = aggregate_scores(scores[role]); detection.pop("endpoint_errors_px")
            reference[role] = {"detection": detection, "corner_quality": aggregate_quality(qualities[role]),
                               "missed_reference_ids": sorted(miss_ids[role]), "empty_scenes": empty[role]}
        if reference["baseline"]["detection"] != comparison["references"][name]["models"]["candidate"]["detection"]:
            raise ValueError(f"Rescored primary metrics differ: {name}")
        old_ids, new_ids = set(matched["baseline"]), set(matched["probe"])
        reference["identity_delta"] = {
            "lost_matches": sorted(old_ids - new_ids), "gained_matches": sorted(new_ids - old_ids),
            "changed_primary_prediction_assignments": [
                {"reference_id": identifier, "before_primary_prediction_id": matched["baseline"][identifier],
                 "after_primary_prediction_id": matched["probe"][identifier]}
                for identifier in sorted(old_ids & new_ids)
                if matched["baseline"][identifier] != matched["probe"][identifier]],
            "matched_primary_prediction_associations_exact": matched["baseline"] == matched["probe"],
        }
        references[name] = reference
    probe_header = {"record_type": "run", "schema_version": 1,
                    "kind": "fixed-extra-nms-on-final-clipped-boxes", "primary_cache_sha256": sha(root / CACHE),
                    "checkpoint_sha256": header["model_sha256"], "box_confidence": .13,
                    "primary_nms_iou": .70, "extra_nms_iou": .70, "training": False, "inference": False}
    output_rows = [probe_header, *probe_rows]
    encoded = jsonl_bytes(output_rows)
    summary = {
        "primary_outputs": sum(len(row["predictions"]) for row in rows),
        "probe_outputs": sum(len(row["predictions"]) for row in probe_rows),
        "dropped_outputs": len(suppressed),
        "references": {name: {"baseline": r["baseline"], "probe": r["probe"], "identity_delta": r["identity_delta"]}
                       for name, r in references.items()},
        "selected": False,
    }
    evidence = {
        "schema_version": 1, "kind": "r39-fixed-postclip-nms-probe", "training": False,
        "inference": False, "automatic_model_promotion": False,
        "scope": "One exploratory extra NMS pass at the existing IoU .70 on already-clipped original-image rectangles. The primary detector and references remain unchanged.",
        "settings": {"box_confidence": .13, "primary_input_space_nms_iou": .70,
                     "extra_final_clipped_nms_iou": .70, "suppression_comparison": "strictly greater than",
                     "ordering": "descending confidence; equal confidence uses original primary array order",
                     "same_class_only": True, "matching_iou": .50, "endpoint_confidence": .50},
        "provenance": {"primary_cache": CACHE, "primary_cache_sha256": sha(root / CACHE),
                       "comparison_sha256": sha(root / COMPARISON), "decoded_trace": TRACE,
                       "decoded_trace_sha256": sha(root / TRACE), "checkpoint_sha256": header["model_sha256"],
                       "script_sha256": sha(__file__),
                       "scorer_files_sha256": {relative: sha(root / relative) for relative in SCORERS}},
        "references": references, "per_image": per_image, "suppressed_outputs": suppressed,
        "predictions_file": {"path": "clipped-nms-predictions.jsonl", "sha256": hashlib.sha256(encoded).hexdigest()},
        "summary": summary,
        "limits": ["Development comparison, not independent unseen-session validation.",
                   "Overlap and matching outcomes do not alone prove two boxes depict the same physical object.",
                   "An extra suppression step can remove a nearby real target in other scenes; no deployment default is changed."]}
    return evidence, output_rows


def replay_probe(root, evidence_path):
    """Recompute suppression pairs, prediction bytes and all reference scores."""
    evidence_path = Path(evidence_path).resolve()
    saved = read_json(evidence_path)
    recomputed, rows = build_probe(root)
    if recomputed != saved:
        raise ValueError("Recomputed clipped-NMS evidence differs from the saved probe")
    path = evidence_path.parent / saved["predictions_file"]["path"]
    if path.read_bytes() != jsonl_bytes(rows):
        raise ValueError("Recomputed clipped-NMS prediction arrays differ")
    return recomputed["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        summary = replay_probe(args.root, args.replay)
    else:
        evidence, rows = build_probe(args.root)
        directory = args.output or args.root / DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "clip-nms-probe.json"
        if path.exists() or (directory / "clipped-nms-predictions.jsonl").exists():
            raise ValueError("Preserve existing probe results or choose a new output directory")
        path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
        (directory / "clipped-nms-predictions.jsonl").write_bytes(jsonl_bytes(rows))
        summary = evidence["summary"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
