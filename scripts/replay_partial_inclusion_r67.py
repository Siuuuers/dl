#!/usr/bin/env python3
"""Rescore six unchanged saved methods after four reviewed partial-front restorations."""

import argparse
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

sys.dont_write_bytecode = True
from analyze_duplicate_audit_r66 import (
    HISTORICAL_PATH, INPUTS, METHOD_NAMES as PRIOR_METHOD_NAMES, SOURCES as PRIOR_SOURCES,
    VARIANTS, digest, encoded, load_inputs, need, prediction_id, read, select_outputs, stage_inputs,
)
from evaluate_pose import parse_pose_labels
from run_corner_average_r58 import report
from run_reference_matrix_r60 import verified_zip


REFERENCE_PATH = "data/reviews/fronts-2026-11-28-r67/manifest.json"
REFERENCE_SHA256 = "07c8c566f41a04bd1336dd833b47857af8c0eaeecbea04dc6a0e125af5802911"
PROVENANCE_PATH = "docs/thesis/evidence/partial-inclusion-r67/excluded-front-provenance.json"
RESTORED_IDS = ("r34/new_000009/R056", "r37/new_000009/R052", "r37/new_000010/R030", "r37/new_000053/R001")
HELD_IDS = ("r34/new_000009/R057", "r37/new_000086/R057", "r37/new_000088/R057")
SOURCES = ("replay_partial_inclusion_r67.py", *PRIOR_SOURCES)
METHOD_NAMES = {"supplied_original": "Supplied original model", **PRIOR_METHOD_NAMES}
RESULT_NAME = "baomai-partial-inclusion-r67-results.zip"


def candidate_reference(root, old_text):
    """Verify the append-only candidate against archived r50 bytes and row provenance."""
    path = root / REFERENCE_PATH
    data = path.read_bytes()
    need(digest(data) == REFERENCE_SHA256, "Candidate reference manifest differs")
    manifest = json.loads(data)
    provenance_bytes = (root / PROVENANCE_PATH).read_bytes()
    need(manifest["source_provenance"] == {"path": PROVENANCE_PATH, "sha256": digest(provenance_bytes)},
         "Historical row provenance differs")
    historical = {row["historical_identity"]: row for row in json.loads(provenance_bytes)["cases"]}
    additions = manifest["restored_rows"]
    need(tuple(sorted(row["stable_id"] for row in additions)) == RESTORED_IDS, "Restored identities differ")
    need(tuple(sorted(row["stable_id"] for row in manifest["held_cases"])) == HELD_IDS, "Held identities differ")
    need(set(manifest["files"]) == set(old_text) and len(old_text) == 20, "Photograph inventory differs")
    text, parsed, lineage, hashes = {}, {}, [], {REFERENCE_PATH: digest(data), PROVENANCE_PATH: digest(provenance_bytes)}
    groups = Counter()
    for image, before in sorted(old_text.items()):
        spec = manifest["files"][image]
        relative = str(Path(REFERENCE_PATH).parent / spec["path"])
        payload = (root / relative).read_bytes()
        old = before.encode()
        need(not old or old.endswith(b"\n"), "Parent label lacks a final newline")
        selected = sorted((r for r in additions if r["image_id"] == image), key=lambda r: r["stable_id"])
        expected = old + b"".join((r["canonical_line"] + "\n").encode() for r in selected)
        need(payload == expected and digest(payload) == spec["sha256"] and len(payload) == spec["bytes"],
             "Candidate changed parent rows or restoration tokens: " + image)
        need(spec["parent_sha256"] == digest(old) and spec["parent_rows"] == len(before.splitlines()),
             "Parent label identity differs: " + image)
        need(spec["rows"] == spec["parent_rows"] + len(selected) and spec["appended_rows"] == len(selected),
             "Appended row count differs")
        for index, line in enumerate(before.splitlines()):
            lineage.append({"image": image, "candidate_index": index,
                "candidate_reference_id": f"r67/{image}/R{index + 1:03d}",
                "stable_id": f"r50/{image}/R{index + 1:03d}", "origin": "retained_r50",
                "parent_reference_index": index, "line_sha256": digest(line.encode())})
        for offset, row in enumerate(selected, start=spec["parent_rows"] + 1):
            source = historical[row["stable_id"]]
            need(row["canonical_line"] == source["historical_label"]["canonical_line"]
                 and row["line_sha256"] == digest(row["canonical_line"].encode())
                 and row["candidate_row"] == offset
                 and row["candidate_reference_id"] == f"r67/{image}/R{offset:03d}"
                 and row["known_corner_count"] == source["known_corner_count"] <= 1,
                 "Restored historical row differs")
            lineage.append({"image": image, "candidate_index": offset - 1,
                "candidate_reference_id": row["candidate_reference_id"], "stable_id": row["stable_id"],
                "origin": "restored_historical", "parent_reference_index": None,
                "line_sha256": row["line_sha256"]})
        text[image] = payload.decode()
        parsed[image] = parse_pose_labels(text[image], 1280, 720, "r67/" + image)
        groups.update(str(sum(point[2] == 2 for point in target["keypoints"])) for target in parsed[image])
        hashes[relative] = digest(payload)
    need(sum(map(len, parsed.values())) == 658 and groups == {"0": 18, "1": 57, "2": 583},
         "Candidate object or corner-availability population differs")
    need(manifest["expected_counts"] == {"images": 20, "fronts": 658, "complete_pairs": 583,
         "known_corner_groups": dict(groups), "restored_fronts": 4, "held_candidates": 3}, "Manifest counts differ")
    return parsed, text, lineage, manifest, hashes


def matrix_row(reference, method, measurement, states, gold, predictions):
    det, pairs = measurement["detection"], measurement["pairs"]
    availability = {str(n): {"fronts": 0, "matched": 0, "missed": 0} for n in range(3)}
    for state in states:
        target = gold[state["image"]][state["reference_index"]]
        group = availability[str(sum(point[2] == 2 for point in target["keypoints"]))]
        group["fronts"] += 1
        group["matched" if state["detected"] else "missed"] += 1
    need(sum(g["missed"] for g in availability.values()) == det["fn"], "Miss strata differ")
    need(det["fp"] == det["duplicate_fp"] + det["other_fp"], "Extra partition differs")
    return {"reference": reference, "method": method, "fronts": det["gold_count"],
        "outputs": det["prediction_count"], "matched": det["tp"], "missed": det["fn"],
        "extras": det["fp"], "duplicates": det["duplicate_fp"], "other_extras": det["other_fp"],
        "eligible_pairs": pairs["eligible"], "within5": pairs["within5"], "within10": pairs["within10"],
        "complete_pair_missed_boxes": pairs["missed_box"],
        "matched_pair_unavailable": pairs["matched_pair_unavailable"],
        "known_corner_groups": availability,
        "empty_scene_outputs": sum(len(predictions[image]) for image in gold if not gold[image])}


def output_states(scores, method, predictions, identities):
    rows = {}
    for image, score in scores.items():
        matches = {m["prediction_index"]: m["gold_index"] for m in score["matches"]}
        duplicates = set(score["duplicate_prediction_indices"])
        for index in range(len(predictions[image])):
            target = matches.get(index)
            rows[(image, index)] = {"status": "matched" if target is not None else
                "duplicate" if index in duplicates else "other_extra",
                "stable_reference_id": identities[(image, target)] if target is not None else None}
    return rows


def suppression_sensitivity(references, method_source, lineage, baseline_restored):
    """One-arm supplement: repeat the three already fixed r66 selectors under each reference."""
    method = "pr158_reviewed_labels"
    predictions, links = method_source["predictions"], method_source["associations"]
    original_restored = {r["stable_id"]: r for r in baseline_restored if r["method"] == method}
    added = [row for row in lineage if row["origin"] == "restored_historical"]
    rows, outcomes = [], []
    for variant in VARIANTS:
        selected = {image: select_outputs(values, links[image], variant)[0]
                    for image, values in predictions.items()}
        output = {image: [predictions[image][i] for i in indices] for image, indices in selected.items()}
        for reference, gold in references.items():
            measurement, _, states = report(gold, output)
            states_by_id = {(r["image"], r["reference_index"]): r for r in states}
            row = {**matrix_row(reference, method, measurement, states, gold, output), "variant": variant,
                   "restored_targets_in_reference": 4 if reference == "r67" else 0,
                   "restored_target_matches": 0 if reference == "r67" else None,
                   "unchanged_r67_supporting_outputs_kept": 0}
            for target in added:
                identity, image = target["stable_id"], target["image"]
                before = original_restored[identity]["after"]
                original_pi = before["prediction_index"]
                present = original_pi in selected[image] if original_pi is not None else None
                row["unchanged_r67_supporting_outputs_kept"] += int(present is True)
                case = states_by_id[(image, target["candidate_index"])] if reference == "r67" else None
                pi = selected[image][case["prediction_index"]] if case is not None and case["prediction_index"] is not None else None
                if case is not None:
                    row["restored_target_matches"] += int(case["detected"])
                outcomes.append({"method": method, "reference": reference, "variant": variant,
                    "stable_id": identity, "candidate_reference_id": target["candidate_reference_id"],
                    "annotated_in_reference": reference == "r67",
                    "detected": case["detected"] if case is not None else None,
                    "prediction_id": prediction_id(method, image, pi) if pi is not None else None,
                    "unchanged_r67_prediction_id": original_restored[identity]["after_prediction_id"],
                    "unchanged_r67_supporting_output_retained": present})
            rows.append(row)
    return rows, outcomes


def analyze(root, input_directory):
    references, methods, provenance = load_inputs(input_directory)
    archives = {name: verified_zip(input_directory / spec["filename"], spec["sha256"], spec["bytes"])
                for name, spec in INPUTS.items()}
    old = archives["historical"]
    old_text = read(old, "reference-labels.json.gz")["r50"]
    candidate, candidate_text, lineage, manifest, reference_hashes = candidate_reference(root, old_text)
    supplied = {"predictions": read(old, "predictions.json.gz")["original"],
        "model_sha256": read(old, "summary.json")["contract"]["original_model_sha256"], "input": "historical"}
    methods = {"supplied_original": supplied, **methods}
    references = {"r50": references["r50"], "r67": candidate}
    old_scores = read(old, "per-image.json.gz")["r50"]
    qfl_scores = read(archives["qfl"], "evaluation/details.json.gz")["per-image"]["r50"]
    labels_scores = read(archives["labels"], "per-image.json.gz")["r50"]
    expected = {"supplied_original": old_scores["original"], "historical_retained": old_scores["proposed"],
        "pr157_native": qfl_scores["reference"]["two_view"], "pr157_qfl": qfl_scores["candidate"]["two_view"],
        "pr158_original_labels": labels_scores["original_labels"]["two_view"],
        "pr158_reviewed_labels": labels_scores["reviewed_labels"]["two_view"]}
    stable = {"r67": {(r["image"], r["candidate_index"]): r["stable_id"] for r in lineage},
              "r50": {(image, i): f"r50/{image}/R{i + 1:03d}" for image, rows in references["r50"].items()
                      for i in range(len(rows))}}
    matrices, deltas, target_changes, restored, transitions, all_targets = [], [], [], [], [], []
    measurements, per_image, predictions = {}, {}, {}
    for method, source in methods.items():
        values = source["predictions"]
        predictions[method] = values
        measurements[method], per_image[method], states, statuses, rows = {}, {}, {}, {}, {}
        for reference, gold in references.items():
            measurement, scores, cases = report(gold, values)
            if reference == "r50":
                need(scores == expected[method], "Primary baseline replay differs: " + method)
            measurements[method][reference], per_image[method][reference] = measurement, scores
            states[reference] = {(r["image"], r["reference_index"]): r for r in cases}
            statuses[reference] = output_states(scores, method, values, stable[reference])
            rows[reference] = matrix_row(reference, method, measurement, cases, gold, values)
            matrices.append(rows[reference])
            for case in cases:
                image, gi, pi = case["image"], case["reference_index"], case["prediction_index"]
                all_targets.append({"method": method, "reference": reference,
                    "stable_id": stable[reference][(image, gi)],
                    "known_corners": sum(p[2] == 2 for p in gold[image][gi]["keypoints"]),
                    "prediction_id": prediction_id(method, image, pi) if pi is not None else None, **case})
        fields = ("fronts", "outputs", "matched", "missed", "extras", "duplicates", "other_extras",
                  "eligible_pairs", "within5", "within10", "complete_pair_missed_boxes", "matched_pair_unavailable")
        delta = {field: rows["r67"][field] - rows["r50"][field] for field in fields}
        need(delta["fronts"] == 4 and delta["outputs"] == delta["eligible_pairs"] == 0,
             "Reference-only population contract differs")
        deltas.append({"method": method, "r67_minus_r50": delta})
        for item in lineage:
            image, gi = item["image"], item["candidate_index"]
            after = states["r67"][(image, gi)]
            before = states["r50"].get((image, gi)) if item["origin"] == "retained_r50" else None
            record = {"method": method, "stable_id": item["stable_id"],
                "candidate_reference_id": item["candidate_reference_id"], "image": image,
                "known_corners": sum(p[2] == 2 for p in candidate[image][gi]["keypoints"]),
                "before": before, "after": after,
                "before_prediction_id": prediction_id(method, image, before["prediction_index"])
                    if before is not None and before["prediction_index"] is not None else None,
                "after_prediction_id": prediction_id(method, image, after["prediction_index"])
                    if after["prediction_index"] is not None else None}
            if before is None:
                record["prediction_status_under_r50"] = statuses["r50"][(image, after["prediction_index"])] \
                    if after["prediction_index"] is not None else None
                restored.append(record)
            elif before != after:
                target_changes.append(record)
        for (image, pi), before in statuses["r50"].items():
            after = statuses["r67"][(image, pi)]
            if before != after:
                transitions.append({"method": method, "image": image, "prediction_index": pi,
                    "prediction_id": prediction_id(method, image, pi), "before": before, "after": after,
                    "prediction": values[image][pi]})
    sensitivity, survival = suppression_sensitivity(references, methods["pr158_reviewed_labels"], lineage, restored)
    source_hashes = {"scripts/" + name: digest((root / "scripts" / name).read_bytes()) for name in SOURCES}
    summary = {"schema": "cassette-partial-inclusion-replay-r67-v1", "status": "completed",
        "fresh_inference": False, "training": False, "predictions_modified": False,
        "new_evaluation_reference": True, "historical_references_modified": False,
        "training_labels_modified": False, "primary_scorer_modified": False,
        "automatic_promotion": False, "development_only": True,
        "provenance": provenance, "analysis_source_sha256": source_hashes,
        "reference_input_sha256": reference_hashes,
        "candidate_reference": {"path": REFERENCE_PATH, "sha256": REFERENCE_SHA256,
            "counts": manifest["expected_counts"], "restored_ids": RESTORED_IDS, "held_ids": HELD_IDS},
        "methods": {method: {"name": METHOD_NAMES[method], "model_sha256": source["model_sha256"],
            "input": source["input"], "completed_predictions_sha256": digest(encoded(source["predictions"]))}
            for method, source in methods.items()},
        "protocol": {"box_confidence_strictly_greater": .25, "matching_iou_inclusive": .5,
            "endpoint_confidence_inclusive": .5, "native_nms_unchanged": .7,
            "postclip_nms_unchanged": .7, "completed_outputs_and_coordinates_unchanged": True,
            "cross_view_associations_unchanged": True, "fresh_evaluation_matching_per_reference": True,
            "restoration_rule": "Four exact historical lines appended; retained r50 rows preserve bytes and indices."},
        "matrix": matrices, "reference_only_deltas": deltas,
        "suppression_sensitivity_scope": {"method": "pr158_reviewed_labels", "rows": len(sensitivity),
            "variants": VARIANTS, "purpose": "Check the same previously fixed r66 restrictions after reference restoration.",
            "replacement_native_nms": False, "threshold_selection": False,
            "retained_prediction_coordinates_and_associations_unchanged": True},
        "record_counts": {"retained_target_changes": len(target_changes), "restored_target_outcomes": len(restored),
            "prediction_category_or_assignment_changes": len(transitions), "target_states": len(all_targets),
            "per_image_score_objects": 6 * 2 * 20, "baseline_image_scores_rechecked": 6 * 20,
            "supplemental_filter_rows": len(sensitivity), "supplemental_restored_target_states": len(survival),
            "supplemental_per_image_score_objects": len(sensitivity) * 20},
        "limitations": [
            "This changes the evaluated reference population, not learned recognition or model outputs.",
            "All 20 photographs are repeatedly examined development images, not independent test captures.",
            "Three other excluded candidates remain held for ownership or corner uncertainty; the evaluator has no ignore-region mechanism.",
            "The corrected inventory is a bounded four-case inclusion decision, not certification that every visible front is annotated.",
            "Restored coordinates retain historical nominal precision; soft image edges do not establish subpixel physical accuracy.",
            "Complete-pair success omits incomplete-corner fronts; report detection strata and restored-front outcomes alongside it.",
            "Evaluation matches are recomputed from unchanged predictions. Reference changes can reassign an output or change its duplicate category.",
            "No training labels, historical references, scorer rules, confidence thresholds or suppression settings are changed."]}
    records = {"measurements": measurements, "per-image": per_image, "target-states": all_targets,
        "retained-target-changes": target_changes, "restored-target-outcomes": restored,
        "prediction-transitions": transitions, "reference-lineage": lineage,
        "reference-labels": {"r50": old_text, "r67": candidate_text}, "predictions": predictions,
        "candidate-reference-manifest": manifest, "suppression-sensitivity": sensitivity,
        "suppression-restored-target-survival": survival}
    return summary, records


def markdown(summary, records):
    lines = ["# Four restored partial fronts: reference-only replay", "",
        "Four reviewed partial-front rows are appended to a separately versioned development reference. "
        "The six methods retain exactly the same saved outputs. This is a change in measured agreement with the reference, not a model improvement.", "",
        "| Method | Reference | Fronts | Misses | Extras | Duplicate subset | Other extras | Both within 5 px | Both within 10 px |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["matrix"]:
        lines.append(f"| {METHOD_NAMES[row['method']]} | {row['reference']} | {row['fronts']} | {row['missed']} | "
            f"{row['extras']} | {row['duplicates']} | {row['other_extras']} | "
            f"{row['within5']}/{row['eligible_pairs']} | {row['within10']}/{row['eligible_pairs']} |")
    lines += ["", "## Evidence and interpretation", "",
        "The primary scorer replays all saved r50 image scores exactly. New evaluation matching is then performed against r67. "
        "The reference grows from 654 to 658 fronts; its known-corner strata are 18 with zero, 57 with one and 583 with two. "
        "Every inherited r50 row remains byte-identical and at the same row position.", "",
        "The machine-readable matrix retains zero/one/two-known-corner detection outcomes. Separate ledgers identify all restored-front outcomes, "
        "any reassigned retained target, and any unchanged prediction whose matched/duplicate/other-extra status changes. "
        "An unchanged complete-pair count does not establish that the restored partial fronts were detected.", "",
        "## Existing fixed filters after reference restoration", "",
        "This separate eight-row sensitivity check uses only the PR158 reviewed-label arm. "
        "It reuses the three fixed r66 restrictions, independently: require its existing mutual reflected-view link, "
        "or add suppression at prediction-pair IoU strictly above 0.60 or 0.50. This does not replace native NMS, "
        "change retained coordinates, recompute view associations, or select a threshold.", "",
        "| Reference | Fixed restriction | Misses | Extras | Duplicate subset | Partial fronts matched | Restored fronts matched |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in records["suppression-sensitivity"]:
        groups = row["known_corner_groups"]
        partial = sum(groups[k]["fronts"] for k in ("0", "1"))
        matched = sum(groups[k]["matched"] for k in ("0", "1"))
        restored = str(row["restored_target_matches"]) + "/4" if row["reference"] == "r67" else "Not in reference"
        lines.append(f"| {row['reference']} | {row['variant']} | {row['missed']} | {row['extras']} | "
                     f"{row['duplicates']} | {matched}/{partial} | {restored} |")
    lines += ["", "The survival ledger separately records whether each restored front's unchanged supporting output remains after filtering. "
              "A reference-only match change must not be described as a newly generated detection.", "", "## Limits", ""]
    lines += ["- " + value for value in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_outputs(directory, summary, records):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"summary.json": encoded(summary), "results.md": markdown(summary, records).encode()}
    payload.update({name + ".json.gz": gzip.compress(encoded(value), mtime=0) for name, value in records.items()})
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=[k for k in summary["matrix"][0] if k != "known_corner_groups"], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary["matrix"])
    payload["matrix.csv"] = table.getvalue().encode()
    supplemental = []
    for row in records["suppression-sensitivity"]:
        flattened = {k: v for k, v in row.items() if k != "known_corner_groups"}
        for group, values in row["known_corner_groups"].items():
            flattened.update({f"known{group}_{field}": value for field, value in values.items()})
        supplemental.append(flattened)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(supplemental[0]))
    writer.writeheader()
    writer.writerows(supplemental)
    payload["suppression-sensitivity.csv"] = table.getvalue().encode()
    payload["suppression-sensitivity.json"] = encoded(records["suppression-sensitivity"])
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items()})
    for name, data in payload.items():
        (directory / name).write_bytes(data)
    return payload


def package_results(directory, source_commit):
    inventory = json.loads((directory / "inventory.json").read_bytes())
    payload = {name: (directory / name).read_bytes() for name in [*inventory, "inventory.json"]}
    for name, spec in inventory.items():
        need(spec == {"bytes": len(payload[name]), "sha256": digest(payload[name])}, "Saved output differs: " + name)
    summary = json.loads(payload["summary.json"])
    payload["source.json"] = encoded({"source_commit": source_commit,
        "source_sha256": summary["analysis_source_sha256"], "reference_input_sha256": summary["reference_input_sha256"],
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
    need(verified_zip(result) == payload, "Result package verification differs")
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
        "counts": summary["candidate_reference"]["counts"], "records": summary["record_counts"],
        "deltas": summary["reference_only_deltas"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
