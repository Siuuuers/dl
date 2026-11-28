#!/usr/bin/env python3
"""Recheck PR158 bytes, paired-training receipts and saved native predictions.

This CPU-only audit never deserializes a checkpoint, runs neural inference,
trains a model, modifies labels, or changes Git state. The evaluator is imported
only after all archived source bytes have been checked against the pinned Git
commit. A second, independently written matcher checks every scored image.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

RESULT_COMMIT = "6bde3b7a759604d605965d468e6102bf99bf7946"
SOURCE_COMMIT = "acaf5d4b4043583fdd81f5ea848b3a6a68ada728"
RESULT_PATH = "results/l4/baomai-training-label-comparison-r61-l4-001/"
EVIDENCE_SHA = "33dbf73b0d8e145fb0aa74184b16a306d148f0b177787349b13465e7b77c3974"
ARMS = ("original_labels", "reviewed_labels")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def overlap(a, b):
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        return 0.0
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def usable(point):
    return all(isinstance(x, (float, int)) and math.isfinite(x) for x in point) and \
        0 <= point[0] < 1280 and 0 <= point[1] < 720 and point[2] >= .5


def independent_image(targets, predictions):
    """Separate confidence-ordered one-to-one matching and case calculation."""
    remaining = set(range(len(targets)))
    links = []
    for pi in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["confidence"], i)):
        p = predictions[pi]
        eligible = [(overlap(p["bbox_xyxy"], targets[gi]["bbox_xyxy"]), gi)
                    for gi in remaining if p["class_id"] == targets[gi]["class_id"]]
        eligible = [(v, gi) for v, gi in eligible if v >= .5]
        if eligible:
            value, gi = min(eligible, key=lambda x: (-x[0], x[1]))
            links.append({"gold_index": gi, "prediction_index": pi, "iou": value})
            remaining.remove(gi)
    by_gold = {r["gold_index"]: r["prediction_index"] for r in links}
    matched = set(by_gold.values())
    extras = [pi for pi in range(len(predictions)) if pi not in matched]
    duplicates = [pi for pi in extras if any(
        predictions[pi]["class_id"] == targets[gi]["class_id"] and
        overlap(predictions[pi]["bbox_xyxy"], targets[gi]["bbox_xyxy"]) >= .5 for gi in by_gold)]
    cases = []
    for gi, target in enumerate(targets):
        pi = by_gold.get(gi)
        complete = all(k[2] == 2 for k in target["keypoints"])
        error = None
        if complete and pi is not None and all(usable(p) for p in predictions[pi]["keypoints"]):
            error = max(math.hypot(p[0] - g[0], p[1] - g[1])
                        for p, g in zip(predictions[pi]["keypoints"], target["keypoints"]))
        cases.append({"reference_index": gi, "prediction_index": pi, "complete_reference": complete,
                      "detected": pi is not None, "maximum_corner_error_px": error,
                      "within5": error is not None and error <= 5,
                      "within10": error is not None and error <= 10})
    return {"matches": links, "unmatched_gold_indices": sorted(remaining),
            "unmatched_prediction_indices": extras, "duplicate_prediction_indices": duplicates,
            "tp": len(links), "fn": len(remaining), "fp": len(extras),
            "duplicate_fp": len(duplicates), "other_fp": len(extras) - len(duplicates)}, cases


def case_descriptor(reference, image, index, target):
    return {"reference": reference, "image": image, "reference_index_0based": index,
            "reference_row_1based": index + 1, "id": f"{image}/R{index + 1:03d}",
            "available_corners": sum(k[2] == 2 for k in target["keypoints"]),
            "bbox_xyxy": target["bbox_xyxy"], "keypoints": target["keypoints"]}


def audit(repo, returned, output):
    repo, returned, output = map(lambda p: Path(p).resolve(), (repo, returned, output))
    manifest_bytes = (returned / "return-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest["status"] == "completed", "Incomplete return")
    evidence = (returned / manifest["evidence_file"]).read_bytes()
    require(len(evidence) == manifest["evidence_bytes"] and sha(evidence) == manifest["evidence_sha256"] == EVIDENCE_SHA,
            "Compact evidence identity differs")
    return_files = {"return-manifest.json": manifest_bytes, manifest["evidence_file"]: evidence}
    verified_models = {}
    for arm, model in manifest["models"].items():
        accumulator, size = hashlib.sha256(), 0
        for part in model["parts"]:
            wrapper = (returned / part["file"]).read_bytes()
            require(sha(wrapper) == part["sha256"] and len(wrapper) == part["bytes"], "Checkpoint wrapper differs")
            with ZipFile(io.BytesIO(wrapper)) as archive:
                require(archive.namelist() == [part["member"]] and archive.testzip() is None, "Checkpoint wrapper membership differs")
                payload = archive.read(part["member"])
            require(sha(payload) == part["member_sha256"] and len(payload) == part["member_bytes"], "Checkpoint member differs")
            accumulator.update(payload)
            size += len(payload)
            return_files[part["file"]] = wrapper
        require(accumulator.hexdigest() == model["model_sha256"] and size == model["model_bytes"], "Assembled checkpoint differs")
        verified_models[arm] = {"sha256": accumulator.hexdigest(), "bytes": size, "parts": len(model["parts"])}
    for name, data in return_files.items():
        git_bytes = subprocess.check_output(["git", "show", f"{RESULT_COMMIT}:{RESULT_PATH}{name}"], cwd=repo)
        require(data == git_bytes, "Result commit differs: " + name)
    with ZipFile(io.BytesIO(evidence)) as archive:
        names = archive.namelist()
        inventory = json.loads(archive.read("inventory.json"))
        require(len(names) == len(set(names)) and set(names) == set(inventory) | {"inventory.json"}
                and archive.testzip() is None, "Compact archive membership or CRC differs")
        members = {name: archive.read(name) for name in names}
    for name, entry in inventory.items():
        require(entry == {"bytes": len(members[name]), "sha256": sha(members[name])}, "Archive member differs: " + name)
    source_paths = [name for name in members if name.startswith("source/")]
    for name in source_paths:
        git_bytes = subprocess.check_output(["git", "show", SOURCE_COMMIT + ":" + name[7:]], cwd=repo)
        require(members[name] == git_bytes, "Archived source differs from pinned commit: " + name)
    read = lambda name: json.loads(members[name])
    source, plan, preflight, summary = map(read, ("source.json", "plan.json", "preflight.json", "summary.json"))
    require(source["commit"] == SOURCE_COMMIT, "Execution source differs")
    for name, digest in source["files_sha256"].items():
        require(sha(members["source/" + name]) == digest, "Execution source closure differs: " + name)
    for name, digest in plan["source_files"].items():
        require(sha(members["source/" + name]) == digest, "Frozen input differs: " + name)
    require(source["driver_sha256"] == sha(members["source/scripts/run_training_label_comparison_r61.py"]), "Driver identity differs")
    require(summary["status"] == "completed" and summary["training"] and summary["fresh_inference"], "Run was not complete")
    require(summary["plan_sha256"] == preflight["plan_sha256"] == sha(members["plan.json"])
            and summary["source_sha256"] == sha(members["source.json"]), "Summary identity differs")
    require(preflight["status"] == "passed" and preflight["images_and_initializer_verified"], "Input preflight failed")
    receipts = {arm: read(f"training/{arm}/receipt.json") for arm in ARMS}
    observations = []
    budget = plan["budget_per_arm"]
    for arm, receipt in receipts.items():
        require(receipt["status"] == "complete" and receipt["role"] == arm and not receipt["trial"] and receipt["updated"], "Arm incomplete")
        require(receipt["source_sha256"] == sha(members["source.json"]) and receipt["plan_sha256"] == sha(members["plan.json"]), "Arm identity differs")
        require(receipt["epochs"] == list(range(1, 31)) and all(receipt[k] == budget[k] for k in ("batches", "images", "steps", "optimizer_batch_indices")), "Training exposure differs")
        require(receipt["runtime"] == summary["runtime"], "Training/inference runtime differs")
        require(receipt["artifacts"]["fit/weights/last.pt"] == verified_models[arm]["sha256"] == summary["model_sha256"][arm], "Selected checkpoint differs")
        results = members[f"training/{arm}/fit/results.csv"]
        require(sha(results) == receipt["artifacts"]["fit/results.csv"], "Training CSV differs")
        csv_rows = list(csv.DictReader(io.StringIO(results.decode())))
        require([int(r["epoch"]) for r in csv_rows] == list(range(1, 31)), "Training CSV epoch sequence differs")
        require(all(math.isfinite(float(v)) for r in csv_rows for v in r.values()), "Training CSV has nonfinite entries")
        progress_bytes = members[f"training/{arm}/progress.json"]
        receipt_bytes = members[f"training/{arm}/receipt.json"]
        if progress_bytes != receipt_bytes:
            require(arm == "original_labels" and progress_bytes == b"arm\n" + receipt_bytes,
                    "Unexpected progress-log anomaly")
            observations.append({"member": f"training/{arm}/progress.json",
                                 "issue": "Four-byte arm-newline prefix makes this auxiliary member invalid JSON.",
                                 "prefix_hex": "61726d0a", "remaining_bytes_exactly_equal_valid_final_receipt": True,
                                 "original_bytes_preserved": True,
                                 "effect": "Final receipt and all saved-output replays remain verifiable; producer verifier reads receipt.json."})
        else:
            require(json.loads(progress_bytes) == receipt, "Final progress and receipt differ")
    a, b = (receipts[arm] for arm in ARMS)
    paired_keys = ("runtime", "initial_state_sha256", "epoch_sample_order_sha256", "epochs", "batches", "images", "steps", "optimizer_batch_indices")
    require(all(a[k] == b[k] for k in paired_keys) and len(a["epoch_sample_order_sha256"]) == 30, "Paired training conditions differ")
    settings_diff = {k: [a["settings"].get(k), b["settings"].get(k)] for k in set(a["settings"]) | set(b["settings"])
                     if a["settings"].get(k) != b["settings"].get(k)}
    require(set(settings_diff) == {"data", "project", "save_dir"}, "Non-path training setting differs")
    trial = read("training/resource-trial/receipt.json")
    require(trial["status"] == "trial-complete" and trial["trial"] and trial["batches"] == trial["steps"] == 4
            and trial["optimizer_batch_indices"] == [0, 1, 2, 3]
            and trial["initial_state_sha256"] == a["initial_state_sha256"], "Trial receipt differs")

    with tempfile.TemporaryDirectory(prefix="pr158-replay-") as temporary:
        root = Path(temporary)
        for name in source_paths:
            destination = root / name[7:]
            require(destination.resolve().is_relative_to(root), "Unsafe archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(members[name])
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "scripts"))
        import run_training_label_comparison_r61 as driver
        require(driver.label_plan(root) == plan, "Rebuilt frozen label plan differs")
        references, provenance = driver.read_references(root, json.loads(members["source/data/manifest.json"])["records"])
        saved = {key: json.loads(gzip.decompress(members[key + ".json.gz"]))
                 for key in ("native-predictions", "predictions", "suppression", "associations", "per-image", "cases")}
        measurement, details = driver.score_native(references, saved["native-predictions"], "original_labels")
        require(measurement == summary["measurement"], "Replayed measurement differs")
        require(all(details[k] == saved[k] for k in saved), "Replayed detailed evidence differs")
    matrix, per_image_rows, groups, paired_detection, duplicate_rows, clipped_rows = [], [], {}, {}, [], []
    independent_count = 0
    for reference, targets_by_image in references.items():
        groups[reference] = {}
        for arm in ARMS:
            groups[reference][arm] = {}
            for output_name in ("original", "two_view"):
                result = measurement["references"][reference][arm][output_name]
                detection, pairs = result["detection"], result["pairs"]
                counts = {str(n): {"total": 0, "detected": 0, "missed": 0} for n in range(3)}
                states = {(r["image"], r["reference_index"]): r for r in details["cases"][reference][arm][output_name]}
                for image, targets in targets_by_image.items():
                    predictions = details["predictions"][arm][output_name][image]
                    independent, cases = independent_image(targets, predictions)
                    recorded = details["per-image"][reference][arm][output_name][image]
                    for key in ("tp", "fn", "fp", "duplicate_fp", "other_fp", "unmatched_gold_indices", "unmatched_prediction_indices", "duplicate_prediction_indices"):
                        require(independent[key] == recorded[key], "Independent matching differs: " + str((reference, arm, output_name, image, key)))
                    require([(r["gold_index"], r["prediction_index"]) for r in independent["matches"]] ==
                            [(r["gold_index"], r["prediction_index"]) for r in recorded["matches"]], "Independent match identities differ")
                    require(all(abs(x["iou"] - y["iou"]) < 1e-12 for x, y in zip(independent["matches"], recorded["matches"])), "Independent IoU differs")
                    for case in cases:
                        require({"image": image, **case} == states[(image, case["reference_index"])], "Independent corner case differs")
                        n = str(sum(k[2] == 2 for k in targets[case["reference_index"]]["keypoints"]))
                        counts[n]["total"] += 1
                        counts[n]["detected" if case["detected"] else "missed"] += 1
                    independent_count += 1
                    per_image_rows.append({"reference": reference, "arm": arm, "output": output_name, "image": image,
                                           **{k: independent[k] for k in ("tp", "fn", "fp", "duplicate_fp", "other_fp")},
                                           "complete_pairs": sum(r["complete_reference"] for r in cases),
                                           "within5": sum(r["within5"] for r in cases), "within10": sum(r["within10"] for r in cases)})
                    if reference == "r50" and output_name == "original":
                        for pi, prediction in enumerate(predictions):
                            box = prediction["bbox_xyxy"]
                            if box[2] <= box[0] or box[3] <= box[1]:
                                clipped_rows.append({"arm": arm, "image": image, "prediction_index": pi})
                        for pi in independent["duplicate_prediction_indices"]:
                            prediction = predictions[pi]
                            eligible = [(overlap(prediction["bbox_xyxy"], targets[r["gold_index"]]["bbox_xyxy"]), r)
                                        for r in independent["matches"]
                                        if prediction["class_id"] == targets[r["gold_index"]]["class_id"]]
                            value, link = max(eligible, key=lambda item: item[0])
                            target = targets[link["gold_index"]]
                            duplicate_rows.append({"arm": arm, "image": image, "prediction_index_0based": pi,
                                "prediction": prediction, "best_overlapping_matched_reference": case_descriptor(reference, image, link["gold_index"], target),
                                "reference_iou": value, "matched_prediction_index_0based": link["prediction_index"],
                                "prediction_to_matched_prediction_iou": overlap(prediction["bbox_xyxy"], predictions[link["prediction_index"]]["bbox_xyxy"]),
                                "eligible_matched_reference_rows_1based": [r["gold_index"] + 1 for v, r in eligible if v >= .5]})
                groups[reference][arm][output_name] = counts
                errors = [r["maximum_corner_error_px"] for r in states.values() if r["maximum_corner_error_px"] is not None]
                matrix.append({"reference": reference, "arm": arm, "output": output_name,
                    **{k: detection[k] for k in ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp")},
                    **pairs, "conditional_mean_maximum_corner_error_px": sum(errors) / len(errors) if errors else None,
                    "conditional_error_denominator": len(errors)})
        old, new = (details["cases"][reference][arm]["two_view"] for arm in ARMS)
        recovered, newly_missed = [], []
        gain_breakdown = Counter()
        for x, y in zip(old, new):
            target = targets_by_image[x["image"]][x["reference_index"]]
            descriptor = {**case_descriptor(reference, x["image"], x["reference_index"], target),
                          "old_state": x, "new_state": y}
            if not x["detected"] and y["detected"]:
                recovered.append(descriptor)
            if x["detected"] and not y["detected"]:
                newly_missed.append(descriptor)
            if x["complete_reference"]:
                for tolerance in (5, 10):
                    key = "within" + str(tolerance)
                    if not x[key] and y[key]:
                        gain_breakdown[f"within{tolerance}_gain_" + ("already_detected" if x["detected"] else "newly_detected")] += 1
                    if x[key] and not y[key]:
                        gain_breakdown[f"within{tolerance}_lost"] += 1
        paired_detection[reference] = {"recovered_count": len(recovered), "newly_missed_count": len(newly_missed),
            "recovered_by_available_corners": dict(sorted(Counter(str(r["available_corners"]) for r in recovered).items())),
            "newly_missed_by_available_corners": dict(sorted(Counter(str(r["available_corners"]) for r in newly_missed).items())),
            "corner_gain_breakdown": dict(sorted(gain_breakdown.items())), "recovered": recovered, "newly_missed": newly_missed}
    require(independent_count == 240 and len(matrix) == 12, "Incomplete matrix replay")
    audit_result = {
        "schema": "training-label-return-independent-audit-v1", "status": "passed_with_auxiliary_log_observation",
        "result_commit": RESULT_COMMIT, "result_path": RESULT_PATH, "source_commit": SOURCE_COMMIT,
        "compact_archive": {"bytes": len(evidence), "sha256": sha(evidence), "members": len(members)},
        "manifest_sha256": sha(manifest_bytes), "verified_result_files": len(return_files),
        "verified_source_files_against_git": len(source_paths), "verified_execution_source_files": len(source["files_sha256"]),
        "verified_frozen_input_files": len(plan["source_files"]), "models": verified_models,
        "plan_sha256": sha(members["plan.json"]), "source_record_sha256": sha(members["source.json"]),
        "paired_training": {"settings_differences_only_paths": sorted(settings_diff), "seed": a["settings"]["seed"],
            "initial_state_sha256": a["initial_state_sha256"], "epoch_sample_order_hashes_equal": True,
            "epoch_sample_order_hashes_sha256": sha(encode(a["epoch_sample_order_sha256"])),
            "optimizer_update_indices_equal": True, "budget_per_arm": budget, "runtime_equal": True,
            "runtime": a["runtime"], "frozen_label_plan_rebuilt_exactly": True,
            "counts": plan["counts"], "numeric_changed_training_files": sum(r["numeric_content_changed"] for r in plan["changed_training_files"]),
            "same_validation_labels": plan["trainer_validation_labels_equal"], "final_epoch_selection": plan["final_epoch_selection"],
            "discarded_resource_trial_steps": trial["steps"], "additional_training_photographs": plan["additional_training_photographs"]},
        "exact_pinned_replay": {"measurement": True, "details": {key: True for key in details}, "matrix_cells": len(matrix)},
        "independent_matcher": {"images_scored": independent_count, "all_match_indices_counts_and_corner_states_equal": True},
        "clipped_nonpositive_original_predictions": clipped_rows, "matrix": matrix, "availability_groups": groups,
        "paired_detection_totals": {ref: {k: v for k, v in data.items() if k not in ("recovered", "newly_missed")} for ref, data in paired_detection.items()},
        "paired_corner_changes": measurement["paired_changes"],
        "observations": observations,
        "reviewed_duplicate_best_reference_availability": {arm: dict(sorted(Counter(str(r["best_overlapping_matched_reference"]["available_corners"])
            for r in duplicate_rows if r["arm"] == arm).items())) for arm in ARMS},
        "limits": ["No neural training or inference rerun; this is a byte, receipt and saved-output audit.",
                   "Epoch sample order and seed are recorded; per-sample augmentation parameter draws are not archived.",
                   "Source image and initializer byte checks are recorded in training preflight, not repeated here because those assets are not in the compact return.",
                   "The outer upload ZIP is not committed and is not verified here; its committed members are verified individually.",
                   "One paired seed and repeatedly used development photographs do not establish generalization to an independent capture session.",
                   "The comparison is conditional on the same already-trained supplied initializer, not a from-scratch annotation intervention."]}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("audit.json", audit_result), ("paired-detection-cases.json", paired_detection),
                          ("reviewed-duplicate-cases.json", duplicate_rows), ("measurement-replayed.json", measurement)):
        (output / name).write_bytes(encode(content))
    with (output / "per-image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(json.dumps({"status": audit_result["status"], "matrix_cells": len(matrix), "independent_image_scores": independent_count,
                      "archive_members": len(members), "source_files": len(source_paths), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Git repository with both pinned commits fetched")
    parser.add_argument("--return-dir", type=Path, required=True, help="Directory with compact ZIP, manifest and four wrapped checkpoint parts")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo, args.return_dir, args.output_dir)
