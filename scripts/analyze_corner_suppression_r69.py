#!/usr/bin/env python3
"""Compare one fixed corner-qualified suppression rule on archived outputs."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

sys.dont_write_bytecode = True
from analyze_duplicate_audit_r66 import SOURCES as COMMON_SOURCES, select_outputs
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou
from run_corner_average_r58 import report

ARCHIVE_SHA256 = "befe5da298de4432e74e90c8fbac0867bdf001493475679f40efad3f8ec8ad32"
ARCHIVE_BYTES = 2289832
PROTOCOL_PATH = "docs/thesis/evidence/corner-suppression-r69/protocol.json"
SOURCES = tuple(dict.fromkeys(("analyze_corner_suppression_r69.py", *COMMON_SOURCES)))
VARIANTS = ("unchanged", "additional_nms_060", "corner_qualified_nms_060")
REFERENCES = ("r50", "r67", "supplied664", "frozen666")


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def accepted(point):
    return (len(point) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
            and 0 <= point[0] < 1280 and 0 <= point[1] < 720 and point[2] >= .50)


def corner_agreement(first, second):
    available = all(accepted(p) for p in first["keypoints"] + second["keypoints"])
    distances = [math.hypot(a[0] - b[0], a[1] - b[1])
                 for a, b in zip(first["keypoints"], second["keypoints"])] if available else None
    return {"both_pairs_accepted": available, "corresponding_distances_px": distances,
            "both_distances_within5": available and max(distances) <= 5.0}


def select(values, variant):
    if variant == "unchanged":
        return list(range(len(values))), []
    kept, removed = [], []
    for index in sorted(range(len(values)), key=lambda i: (-values[i]["confidence"], i)):
        for winner in kept:
            if values[index]["class_id"] != values[winner]["class_id"]:
                continue
            overlap = _box_iou(values[index]["bbox_xyxy"], values[winner]["bbox_xyxy"])
            if overlap <= .60:
                continue
            agreement = corner_agreement(values[index], values[winner])
            if variant == "corner_qualified_nms_060" and not agreement["both_distances_within5"]:
                continue
            removed.append({"original_index": index, "suppressor_original_index": winner,
                            "prediction_pair_iou": overlap, **agreement})
            break
        else:
            kept.append(index)
    return sorted(kept), removed


def known_corners(target, prediction):
    rows = []
    for slot, point in enumerate(target["keypoints"]):
        if point[2] != 2:
            continue
        current = prediction["keypoints"][slot] if prediction is not None else None
        valid = current is not None and accepted(current)
        distance = math.hypot(current[0] - point[0], current[1] - point[1]) if valid else None
        rows.append({"slot": slot, "available": valid, "error_px": distance,
                     "within5": valid and distance <= 5, "within10": valid and distance <= 10})
    return rows


def matrix_row(reference, method, variant, measurement, states, gold, output):
    det, pairs = measurement["detection"], measurement["pairs"]
    row = {"reference": reference, "method": method, "variant": variant,
           "fronts": det["gold_count"], "outputs": det["prediction_count"],
           "matched": det["tp"], "missed": det["fn"], "extras": det["fp"],
           "duplicates": det["duplicate_fp"], "other_extras": det["other_fp"],
           "eligible_pairs": pairs["eligible"], "within5": pairs["within5"], "within10": pairs["within10"],
           "complete_pair_missed_boxes": pairs["missed_box"],
           "matched_pair_unavailable": pairs["matched_pair_unavailable"],
           "empty_scene_outputs": sum(len(output[image]) for image in gold if not gold[image])}
    for n in range(3):
        group = [s for s in states if sum(p[2] == 2 for p in gold[s["image"]][s["reference_index"]]["keypoints"]) == n]
        row.update({f"known{n}_fronts": len(group), f"known{n}_matched": sum(s["detected"] for s in group),
                    f"known{n}_missed": sum(not s["detected"] for s in group)})
    return row


def output_status(score, index):
    match = next((m["gold_index"] for m in score["matches"] if m["prediction_index"] == index), None)
    return {"status": "matched" if match is not None else "duplicate" if index in score["duplicate_prediction_indices"] else "other_extra",
            "reference_index": match}


def analyze(root, archive):
    raw = archive.read_bytes()
    if len(raw) != ARCHIVE_BYTES or digest(raw) != ARCHIVE_SHA256:
        raise ValueError("The R68 archive differs from its declared immutable bytes")
    with ZipFile(io.BytesIO(raw)) as z:
        predictions = json.loads(gzip.decompress(z.read("predictions.json.gz")))
        labels = json.loads(gzip.decompress(z.read("reference-labels.json.gz")))
        expected = list(csv.DictReader(io.StringIO(z.read("matrix.csv").decode())))
    gold = {ref: {image: parse_pose_labels(text, 1280, 720, ref + "/" + image)
                  for image, text in labels[ref].items()} for ref in REFERENCES}
    original_digest = digest(encoded(predictions))
    matrices, changes, removals, comparisons = [], [], [], []
    selections, per_image, kept_outputs = {}, {}, {}
    for method, variants in predictions.items():
        values = variants["unchanged"]
        selected, removed = {}, {}
        for variant in VARIANTS:
            selected[variant], removed[variant] = {}, {}
            for image, records in values.items():
                selected[variant][image], removed[variant][image] = select(records, variant)
                if variant == "additional_nms_060":
                    old, _ = select_outputs(records, [{"original_index": i} for i in range(len(records))], variant)
                    assert old == selected[variant][image], "Box-only control differs from R66"
        selections[method] = selected
        kept_outputs[method] = {v: {image: [values[image][i] for i in ids] for image, ids in selected[v].items()} for v in VARIANTS}
        for image, records in values.items():
            for first in range(len(records)):
                for second in range(first + 1, len(records)):
                    a, b = records[first], records[second]
                    overlap = _box_iou(a["bbox_xyxy"], b["bbox_xyxy"])
                    if a["class_id"] == b["class_id"] and overlap > .60:
                        comparisons.append({"method": method, "image": image, "first_original_index": first,
                            "second_original_index": second, "prediction_pair_iou": overlap,
                            "first_box_confidence": a["confidence"], "second_box_confidence": b["confidence"],
                            **corner_agreement(a, b)})
        for reference in REFERENCES:
            targets = gold[reference]
            _, baseline_scores, baseline_states = report(targets, values)
            for variant in VARIANTS:
                output = kept_outputs[method][variant]
                measurement, scores, states = report(targets, output)
                per_image.setdefault(reference, {}).setdefault(method, {})[variant] = scores
                row = matrix_row(reference, method, variant, measurement, states, targets, output)
                matrices.append(row)
                if variant == "unchanged":
                    old = next(r for r in expected if r["reference"] == reference and r["method"] == method and r["variant"] == variant)
                    assert all(str(row[key]) == value for key, value in old.items()), "R68 baseline matrix differs"
                    continue
                for before, after in zip(baseline_states, states):
                    image, gi = before["image"], before["reference_index"]
                    assert (image, gi) == (after["image"], after["reference_index"])
                    bi = before["prediction_index"]
                    ai = selected[variant][image][after["prediction_index"]] if after["prediction_index"] is not None else None
                    if bi == ai and all(before[k] == after[k] for k in ("detected", "within5", "within10", "maximum_corner_error_px")):
                        continue
                    target = targets[image][gi]
                    changes.append({"reference": reference, "method": method, "variant": variant, "image": image,
                        "reference_index": gi, "reference_id": f"{reference}/{image}/R{gi+1:03d}",
                        "known_corners": sum(p[2] == 2 for p in target["keypoints"]),
                        "detection_gain": not before["detected"] and after["detected"],
                        "detection_loss": before["detected"] and not after["detected"],
                        "rematched": before["detected"] and after["detected"] and bi != ai,
                        "before": {**before, "original_prediction_index": bi,
                                   "known_corner_outcomes": known_corners(target, values[image][bi] if bi is not None else None)},
                        "after": {**after, "original_prediction_index": ai,
                                  "known_corner_outcomes": known_corners(target, values[image][ai] if ai is not None else None)}})
                for image, records in removed[variant].items():
                    for removed_record in records:
                        index, winner = removed_record["original_index"], removed_record["suppressor_original_index"]
                        removals.append({"reference": reference, "method": method, "variant": variant, "image": image,
                            **removed_record, "removed_baseline": output_status(baseline_scores[image], index),
                            "suppressor_baseline": output_status(baseline_scores[image], winner),
                            "prediction": values[image][index], "suppressor": values[image][winner]})
    assert digest(encoded(predictions)) == original_digest, "Input records were modified"
    summary = {"schema": "cassette-corner-suppression-r69-v1", "status": "completed",
        "protocol": json.loads((root / PROTOCOL_PATH).read_bytes()),
        "protocol_sha256": digest((root / PROTOCOL_PATH).read_bytes()),
        "input_archive_sha256": ARCHIVE_SHA256, "input_archive_bytes": ARCHIVE_BYTES,
        "preserved_input_predictions_sha256": original_digest,
        "source_sha256": {"scripts/"+name: digest((root/"scripts"/name).read_bytes()) for name in SOURCES},
        "baseline_rows_reproduced": 20, "matrix": matrices,
        "record_counts": {"matrix_rows": len(matrices), "per_image_scores": len(matrices)*20,
                          "target_changes": len(changes), "removed_output_reference_records": len(removals),
                          "overlapping_prediction_pairs": len(comparisons)},
        "fresh_inference": False, "training": False, "labels_changed": False, "automatic_promotion": False}
    return summary, {"per-image.json.gz": per_image, "target-changes.json.gz": changes,
        "removed-outputs.json.gz": removals, "overlapping-pairs.json.gz": comparisons,
        "selected-indices.json.gz": selections, "predictions.json.gz": kept_outputs}


def markdown(summary):
    rows = ["# Corner agreement as a condition for additional suppression", "",
        "This fixed saved-output experiment compares unchanged detections, the existing additional box-only IoU 0.60 control, and the same suppression qualified by two accepted corresponding corners within five pixels. All selection uses predictions alone. The R68 reflected-output union is excluded.", "",
        "| Method | Variant | r67 misses | Extras | Duplicate subset | Pair within 5 px | Pair within 10 px |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["matrix"]:
        if row["reference"] == "r67":
            rows.append(f"| {row['method']} | {row['variant']} | {row['missed']} | {row['extras']} | {row['duplicates']} | {row['within5']}/{row['eligible_pairs']} | {row['within10']}/{row['eligible_pairs']} |")
    rows += ["", "The full matrix retains five methods and four references. Fresh matching can change which surviving prediction represents a target. The ledgers preserve removed outputs, their suppressors, target gains, losses, rematches and confidence-qualified known-corner errors. Outputs keep their original coordinates and scores.", "",
        "The two thresholds were fixed before the first candidate outcome measurement. This is an exploratory two-corner rule inspired by pose comparison in RMPE, with no learned pose-distance parameters or coordinate merging. Agreement between predictions can be shared error. The same development photographs support all reference comparisons.", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r68-result-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary, records = analyze(root, args.r68_result_zip)
    args.output.mkdir(parents=True, exist_ok=True)
    files = {"summary.json": encoded(summary), "results.md": markdown(summary).encode(),
             "protocol.json": (root/PROTOCOL_PATH).read_bytes()}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(summary["matrix"][0]))
    writer.writeheader()
    writer.writerows(summary["matrix"])
    files["matrix.csv"] = stream.getvalue().encode()
    files.update({name: gzip.compress(encoded(value), mtime=0) for name, value in records.items()})
    for name, data in files.items():
        (args.output/name).write_bytes(data)
    inventory = {"schema": "cassette-corner-suppression-r69-inventory-v1",
                 "files": {name: {"bytes": len(data), "sha256": digest(data)} for name, data in files.items()}}
    (args.output/"inventory.json").write_bytes(encoded(inventory))
    print(json.dumps({"status": "completed", **summary["record_counts"]}))


if __name__ == "__main__":
    main()
