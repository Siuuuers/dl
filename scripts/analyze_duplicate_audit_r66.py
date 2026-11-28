#!/usr/bin/env python3
"""Replay fixed output-removal probes without inference, training, or new labels."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou
from run_corner_average_r58 import report
from run_original_comparison_r59 import measure
from run_reference_matrix_r60 import verified_zip
from run_training_label_comparison_r61 import score_native


HISTORICAL_PATH = "docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip"
INPUTS = {
    "historical": {
        "filename": "historical-r60.zip", "bytes": 972658,
        "sha256": "ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601",
        "repository_path": HISTORICAL_PATH,
    },
    "qfl": {
        "filename": "quality-focal-pr157.zip", "bytes": 7528011,
        "sha256": "cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a",
        "commit": "f971e880e8aff2859ce4d8ae2923efea826eddb4",
        "repository_path": "results/a100/baomai-quality-focal-r61-a100-001/baomai-quality-focal-r61-results.zip",
    },
    "labels": {
        "filename": "training-labels-pr158.zip", "bytes": 2038950,
        "sha256": "33dbf73b0d8e145fb0aa74184b16a306d148f0b177787349b13465e7b77c3974",
        "commit": "6bde3b7a759604d605965d468e6102bf99bf7946",
        "repository_path": "results/l4/baomai-training-label-comparison-r61-l4-001/baomai-training-label-comparison-r61-results.zip",
    },
}
REFERENCES = ("r50", "supplied664", "frozen666")
VARIANTS = ("unchanged", "mutual_view_support", "additional_nms_060", "additional_nms_050")
METHOD_NAMES = {
    "historical_retained": "Historical retained method",
    "pr157_native": "PR157 native control",
    "pr157_qfl": "PR157 QFL candidate",
    "pr158_original_labels": "PR158 original-label arm",
    "pr158_reviewed_labels": "PR158 reviewed-label arm",
}
SOURCES = (
    "analyze_duplicate_audit_r66.py", "evaluate_pose.py", "pose_scoring.py", "pose_checkpoint.py",
    "audit_reflection_return.py", "predict_pose_folder.py", "run_corner_average_r58.py",
    "run_original_comparison_r59.py", "run_reference_matrix_r60.py",
    "run_training_label_comparison_r61.py", "run_flip_inference_r56.py",
    "diagnose_fronts_r50.py", "run_inference_ablation_r53.py",
)
RESULT_NAME = "baomai-duplicate-audit-r66-results.zip"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(members, name):
    data = members[name]
    return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def stage_inputs(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for key, spec in INPUTS.items():
        path = destination / spec["filename"]
        if not path.exists():
            if key == "historical":
                data = (root / HISTORICAL_PATH).read_bytes()
            else:
                url = "https://raw.githubusercontent.com/Siuuuers/dl/" + spec["commit"] + "/" + spec["repository_path"]
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            need(len(data) == spec["bytes"] and digest(data) == spec["sha256"], "Downloaded input differs: " + key)
            path.write_bytes(data)
        verified_zip(path, spec["sha256"], spec["bytes"])
    return destination


def load_inputs(directory):
    archives = {key: verified_zip(directory / spec["filename"], spec["sha256"], spec["bytes"])
                for key, spec in INPUTS.items()}
    old, labels, qfl = archives["historical"], archives["labels"], archives["qfl"]
    text = read(old, "reference-labels.json.gz")
    references = {ref: {image: parse_pose_labels(value, 1280, 720, ref + "/" + image)
                        for image, value in text[ref].items()} for ref in REFERENCES}
    label_plan = read(labels, "plan.json")
    for ref in REFERENCES:
        for image, value in text[ref].items():
            need(qfl[f"references/{ref}/labels/{image}.txt"] == value.encode(), "PR157 reference differs")
            need(label_plan["references"][ref]["labels"][image]["sha256"] == digest(value.encode()), "PR158 reference differs")

    historical_measurement, historical = measure(references, read(old, "native-predictions.json.gz"))
    for name, value in historical.items():
        need(value == read(old, name + ".json.gz"), "Historical replay differs: " + name)
    old_summary = read(old, "summary.json")
    for row in old_summary["matrix"]:
        actual = historical_measurement["references"][row["reference"]][row["method"]]
        need(row["matched"] == actual["detection"]["tp"] and row["within5"] == actual["pairs"]["within5"],
             "Historical headline differs")
    qfl_summary = read(qfl, "evaluation/summary.json")
    qfl_saved = read(qfl, "evaluation/details.json.gz")
    qfl_measurement, qfl_replay = score_native(references, qfl_saved["native-predictions"], baseline="reference")
    need(qfl_measurement == qfl_summary["measurement"] and qfl_replay == qfl_saved, "PR157 full output replay differs")
    labels_summary = read(labels, "summary.json")
    labels_measurement, labels_replay = score_native(references, read(labels, "native-predictions.json.gz"), baseline="original_labels")
    need(labels_measurement == labels_summary["measurement"], "PR158 full measurement replay differs")
    for name, value in labels_replay.items():
        need(value == read(labels, name + ".json.gz"), "PR158 full output replay differs: " + name)

    methods = {"historical_retained": {
        "predictions": historical["predictions"]["proposed"], "associations": historical["associations"],
        "model_sha256": old_summary["contract"]["adapted_model_sha256"], "input": "historical"}}
    for name, role in (("pr157_native", "reference"), ("pr157_qfl", "candidate")):
        methods[name] = {"predictions": qfl_replay["predictions"][role]["two_view"],
                         "associations": qfl_replay["associations"][role],
                         "model_sha256": qfl_summary["model_sha256"][role], "input": "qfl"}
    for role in ("original_labels", "reviewed_labels"):
        methods["pr158_" + role] = {"predictions": labels_replay["predictions"][role]["two_view"],
                                   "associations": labels_replay["associations"][role],
                                   "model_sha256": labels_summary["model_sha256"][role], "input": "labels"}
    return references, methods, {"source_archives": INPUTS,
        "reference_labels_sha256": {ref: {image: digest(value.encode()) for image, value in images.items()}
                                     for ref, images in text.items()},
        "source_photographs": old_summary["contract"]["photographs"],
        "replay": "All archived historical and PR157/PR158 prediction, association, case and per-image objects reproduce exactly.",
        "replayed_image_scores": 20 * 3 * (4 + 3 * 2 + 2 * 2),
        "model_bytes_loaded": False, "photograph_bytes_loaded": False}


def select_outputs(predictions, links, variant):
    need(len(predictions) == len(links), "Association count differs")
    need(all(row["original_index"] == i for i, row in enumerate(links)), "Association order differs")
    if variant == "unchanged":
        return list(range(len(predictions))), []
    if variant == "mutual_view_support":
        kept = [i for i, row in enumerate(links) if row["reflected_index"] is not None]
        return kept, [{"original_index": i, "reason": "no_mutual_reflected_box"}
                      for i, row in enumerate(links) if row["reflected_index"] is None]
    threshold = {"additional_nms_060": .60, "additional_nms_050": .50}[variant]
    kept, removed = [], []
    for i in sorted(range(len(predictions)), key=lambda j: (-predictions[j]["confidence"], j)):
        suppressor = next((j for j in kept if predictions[i]["class_id"] == predictions[j]["class_id"]
                           and _box_iou(predictions[i]["bbox_xyxy"], predictions[j]["bbox_xyxy"]) > threshold), None)
        if suppressor is None:
            kept.append(i)
        else:
            removed.append({"original_index": i, "reason": "additional_post_output_suppression",
                            "suppressor_original_index": suppressor,
                            "prediction_pair_iou": _box_iou(predictions[i]["bbox_xyxy"], predictions[suppressor]["bbox_xyxy"])})
    return sorted(kept), removed


def matching_bound(gold, predictions):
    neighbors = {i: [j for j, target in enumerate(gold) if prediction["class_id"] == target["class_id"]
                     and _box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"]) >= .50]
                 for i, prediction in enumerate(predictions)}
    assigned = {}

    def augment(candidate, seen):
        for target in neighbors[candidate]:
            if target in seen:
                continue
            seen.add(target)
            if target not in assigned or augment(assigned[target], seen):
                assigned[target] = candidate
                return True
        return False

    for i in neighbors:
        augment(i, set())
    return {"maximum_one_to_one_matches": len(assigned),
            "predictions_eligible_for_multiple_references": sum(len(v) > 1 for v in neighbors.values())}


def prediction_id(method, image, index):
    return method + "/" + image + f"/P{index + 1:03d}"


def analyze(root, input_directory):
    references, methods, provenance = load_inputs(input_directory)
    matrices, changes, removals, bounds, selections, baseline_cases = [], [], [], [], {}, []
    for method, data in methods.items():
        predictions, links = data["predictions"], data["associations"]
        selected, removed = {}, {}
        for variant in VARIANTS:
            selected[variant], removed[variant] = {}, {}
            for image, values in predictions.items():
                selected[variant][image], removed[variant][image] = select_outputs(values, links[image], variant)
        selections[method] = selected
        for ref in REFERENCES:
            gold = references[ref]
            baseline, baseline_scores, baseline_states = report(gold, predictions)
            for image, score in baseline_scores.items():
                duplicates = set(score["duplicate_prediction_indices"])
                for i in score["unmatched_prediction_indices"]:
                    p = predictions[image][i]
                    overlaps = [(j, _box_iou(p["bbox_xyxy"], g["bbox_xyxy"])) for j, g in enumerate(gold[image])
                                if p["class_id"] == g["class_id"]]
                    nearest, maximum = max(overlaps, key=lambda x: (x[1], -x[0])) if overlaps else (None, 0.)
                    baseline_cases.append({"reference": ref, "method": method, "image": image,
                        "id": prediction_id(method, image, i), "kind": "extra", "prediction": p,
                        "duplicate": i in duplicates, "maximum_reference_iou": maximum,
                        "nearest_reference_id": ref + "/" + image + f"/R{nearest + 1:03d}" if nearest is not None and maximum > 0 else None})
                for j in score["unmatched_gold_indices"]:
                    baseline_cases.append({"reference": ref, "method": method, "image": image,
                        "id": ref + "/" + image + f"/R{j + 1:03d}", "kind": "miss", "target": gold[image][j],
                        "known_corners": sum(p[2] == 2 for p in gold[image][j]["keypoints"])})
            for variant in VARIANTS:
                output = {image: [predictions[image][i] for i in indexes] for image, indexes in selected[variant].items()}
                measurement, scores, states = report(gold, output)
                det, pairs = measurement["detection"], measurement["pairs"]
                availability = {str(n): {"fronts": 0, "matched": 0, "missed": 0} for n in range(3)}
                for state in states:
                    n = str(sum(p[2] == 2 for p in gold[state["image"]][state["reference_index"]]["keypoints"]))
                    availability[n]["fronts"] += 1
                    availability[n]["matched" if state["detected"] else "missed"] += 1
                maximum = 0
                for image, score in scores.items():
                    bound = matching_bound(gold[image], output[image])
                    need(bound["maximum_one_to_one_matches"] >= score["tp"], "Matching bound below valid greedy matches")
                    maximum += bound["maximum_one_to_one_matches"]
                    bounds.append({"reference": ref, "method": method, "variant": variant, "image": image,
                                   "greedy_matches": score["tp"], **bound,
                                   "gap": bound["maximum_one_to_one_matches"] - score["tp"]})
                row = {"reference": ref, "method": method, "variant": variant,
                       "fronts": det["gold_count"], "outputs": det["prediction_count"], "matched": det["tp"],
                       "missed": det["fn"], "extras": det["fp"], "duplicates": det["duplicate_fp"],
                       "other_extras": det["other_fp"], "eligible_pairs": pairs["eligible"],
                       "within5": pairs["within5"], "within10": pairs["within10"],
                       "complete_pair_missed_boxes": pairs["missed_box"], "known_corner_groups": availability,
                       "maximum_one_to_one_matches": maximum, "matching_gap": maximum - det["tp"],
                       "empty_scene_outputs": sum(len(output[i]) for i in gold if not gold[i])}
                need(sum(v["missed"] for v in availability.values()) == row["missed"], "Availability partition differs")
                need(row["extras"] == row["duplicates"] + row["other_extras"], "Extra partition differs")
                matrices.append(row)
                if variant == "unchanged":
                    continue
                for before, after in zip(baseline_states, states):
                    image, gi = before["image"], before["reference_index"]
                    need((image, gi) == (after["image"], after["reference_index"]), "Reference identity order changed")
                    before_i = before["prediction_index"]
                    after_i = selected[variant][image][after["prediction_index"]] if after["prediction_index"] is not None else None
                    fields = ("detected", "within5", "within10")
                    if before_i == after_i and all(before[f] == after[f] for f in fields):
                        continue
                    changes.append({"reference": ref, "method": method, "variant": variant,
                        "id": ref + "/" + image + f"/R{gi + 1:03d}", "image": image, "reference_index": gi,
                        "known_corners": sum(p[2] == 2 for p in gold[image][gi]["keypoints"]),
                        "target": gold[image][gi], "before": before, "after": after,
                        "before_prediction_id": prediction_id(method, image, before_i) if before_i is not None else None,
                        "after_prediction_id": prediction_id(method, image, after_i) if after_i is not None else None})
                for image, values in removed[variant].items():
                    old_score = baseline_scores[image]
                    matches = {m["prediction_index"]: m["gold_index"] for m in old_score["matches"]}
                    for value in values:
                        pi = value["original_index"]
                        gi = matches.get(pi)
                        status = "matched" if gi is not None else "duplicate" if pi in old_score["duplicate_prediction_indices"] else "other_extra"
                        removals.append({"reference": ref, "method": method, "variant": variant, "image": image,
                            "id": prediction_id(method, image, pi), "baseline_status": status, **value,
                            "prediction": predictions[image][pi], "association": links[image][pi],
                            "baseline_reference_id": ref + "/" + image + f"/R{gi + 1:03d}" if gi is not None else None,
                            "known_corners": sum(p[2] == 2 for p in gold[image][gi]["keypoints"]) if gi is not None else None})

    source_hashes = {"scripts/" + name: digest((root / "scripts" / name).read_bytes()) for name in SOURCES}
    summary = {"schema": "cassette-duplicate-audit-r66-v1", "status": "completed", "fresh_inference": False,
        "training": False, "labels_modified": False, "automatic_promotion": False, "development_only": True,
        "provenance": provenance, "analysis_source_sha256": source_hashes,
        "methods": {name: {"name": METHOD_NAMES[name], "model_sha256": data["model_sha256"], "input": data["input"]}
                    for name, data in methods.items()},
        "protocol": {"box_confidence_strictly_greater": .25, "matching_iou_inclusive": .5,
            "endpoint_confidence_inclusive": .5, "native_nms_unchanged": .7,
            "existing_postclip_suppression_unchanged": .7, "mutual_association_iou_inclusive": .5,
            "additional_nms_iou_strictly_greater": [.60, .50], "coordinates_and_scores_unchanged": True,
            "cross_view_associations_recomputed_after_removal": False,
            "evaluation_matches_recomputed_after_removal": True,
            "equal_confidence_tie_break": "Original completed-output index, ascending",
            "variant_combination": "Separate probes; the support gate and suppression thresholds are not combined.",
            "choice_status": "Two illustrative fixed suppression settings; no threshold is selected or promoted."},
        "matrix": matrices,
        "record_counts": {"paired_changes": len(changes), "removed_output_reference_records": len(removals),
                          "matching_bounds": len(bounds), "baseline_error_records": len(baseline_cases)},
        "limitations": [
            "All three references describe the same 20 repeatedly examined development photographs.",
            "An evaluator extra or duplicate is not an independent physical background or object-identity verdict.",
            "Historically excluded visible partial fronts remain excluded; the evaluator has no ignore-region protocol.",
            "Extra suppression acts on already filtered completed two-view outputs. It does not replace or replay native NMS and cannot recover discarded proposals.",
            "Retained boxes, scores and averaged coordinates are unchanged. Fresh evaluation matching can select a different retained record for a reference.",
            "The reflected image is another network response to the same photograph, not independent camera evidence.",
            "Maximum one-to-one matching uses reference geometry only as a cardinality bound; it changes neither the primary scorer nor inference.",
            "A decrease in extras with unchanged complete-pair success can still lose incomplete-corner fronts.",
            "No label changes, training, model inference, threshold selection, or deployment promotion are performed."]}
    return summary, {"paired-changes": changes, "removed-outputs": removals, "matching-bounds": bounds,
                     "baseline-cases": baseline_cases, "selected-original-indices": selections}


def markdown(summary):
    lines = ["# Duplicate and unmatched output audit", "",
        "The saved outputs reproduce exactly before three separate output-removal probes. "
        "The probes reveal tradeoffs; no model or threshold is promoted.", "",
        "`mutual_view_support` requires the existing mutual original/reflected box link. "
        "`additional_nms_060` and `additional_nms_050` suppress completed outputs at prediction-pair IoU strictly above 0.60 or 0.50. "
        "They do not replace native NMS. All retained coordinates and confidence values remain unchanged.", "",
        "## Reviewed-reference results", "",
        "| Method | Probe | Misses | Extras | Duplicate subset | Both within 5 px | Both within 10 px |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["matrix"]:
        if r["reference"] == "r50":
            lines.append(f"| {METHOD_NAMES[r['method']]} | {r['variant']} | {r['missed']} | {r['extras']} | {r['duplicates']} | {r['within5']}/{r['eligible_pairs']} | {r['within10']}/{r['eligible_pairs']} |")
    lines += ["", "The maximum-cardinality reference-matching bound is reported separately in the machine-readable matrix. "
              "It is a diagnostic upper bound, not a replacement scorer or a usable inference rule.", "",
              "The full matrix retains all three reference versions. Paired changes identify every lost/gained reference and reassigned prediction; "
              "availability groups distinguish zero, one and two known corners. The baseline error ledger does not label unmatched regions as physical negatives.", "",
              "## Interpretation limits", ""]
    lines += ["- " + value for value in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_outputs(directory, summary, records):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"summary.json": encoded(summary), "results.md": markdown(summary).encode()}
    payload.update({name + ".json.gz": gzip.compress(encoded(value), mtime=0) for name, value in records.items()})
    fields = [k for k in summary["matrix"][0] if k != "known_corner_groups"]
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["matrix"])
    payload["matrix.csv"] = table.getvalue().encode()
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()})
    for name, data in payload.items():
        (directory / name).write_bytes(data)
    return payload


def package_results(directory, source_commit):
    inventory = json.loads((directory / "inventory.json").read_bytes())
    payload = {name: (directory / name).read_bytes() for name in [*inventory, "inventory.json"]}
    for name, spec in inventory.items():
        need(spec == {"bytes": len(payload[name]), "sha256": digest(payload[name])}, "Saved result differs: " + name)
    payload["source.json"] = encoded({"source_commit": source_commit,
        "source_sha256": json.loads(payload["summary.json"])["analysis_source_sha256"],
        "no_new_inference_or_training": True})
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)}
                                        for name, data in payload.items() if name != "inventory.json"})
    result = directory / RESULT_NAME
    with ZipFile(result, "w", ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    need(verified_zip(result) == payload, "Saved result package differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-inputs", action="store_true")
    args = parser.parse_args()
    if args.stage_inputs:
        stage_inputs(args.root, args.inputs)
    summary, records = analyze(args.root, args.inputs)
    write_outputs(args.output, summary, records)
    print(json.dumps({"status": summary["status"], "matrix_rows": len(summary["matrix"]),
                      "baseline_replay": summary["provenance"]["replay"],
                      "records": summary["record_counts"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
