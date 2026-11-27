#!/usr/bin/env python3
"""Replay fixed PR137 caches against R38 and proposed R48 labels on CPU.

This is reference sensitivity, not a new model evaluation or training result.
The default mode requires every generated R48 development label to agree
byte-for-byte with the independently replayed operations in the audit ledger.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

from audit_additional_front_pilot_r45_results import REFERENCES, HELPERS, corner_groups
from audit_additional_front_pilot_r47_results import (
    ARCHIVE_SHA, ARCHIVE_BYTES, PR_HEAD, SOURCE, MODEL_HASHES,
)
from build_front_review_r44 import changed_fields, geometry
from build_front_review_r46 import amend_line
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, aggregate_scores, score_image


ROOT = Path(__file__).resolve().parents[1]
REVIEW = Path("data/reviews/fronts-2026-11-27-r48")
OUTPUT = Path("results/evaluation/front-errors-r48-20261127/reference-sensitivity.json")
ARMS = ("reference", "candidate")
VIEWS = ("postclip", "raw")
DENOMINATORS = {"0": 17, "1": 54, "2": 583}
MATCHING_IOU = ENDPOINT_CONFIDENCE = 0.5


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = Path(path).read_bytes()
    need(sha(data) == expected, f"Hash mismatch: {path}")
    return data


def parse(data, identity):
    return parse_pose_labels(data.decode(), 1280, 720, identity)


def score(gold, predictions):
    return score_image(gold, predictions, 1280, 720, MATCHING_IOU, ENDPOINT_CONFIDENCE)


def compact_detection(result):
    return {key: value for key, value in result.items()
            if key not in ("endpoint_errors_px", "matches", "unmatched_gold_indices",
                           "unmatched_prediction_indices", "duplicate_prediction_indices",
                           "other_fp_prediction_indices")}


def difference(before, after):
    return {key: None if before[key] is None or after[key] is None else after[key] - before[key]
            for key in before}


def evaluate(gold, predictions):
    per_image = {}
    groups = {str(n): Counter(eligible=0, matched=0, within5=0, within10=0) for n in (0, 1, 2)}
    for identity, targets in gold.items():
        per_image[identity] = score(targets, predictions[identity])
        for key, values in corner_groups(targets, predictions[identity], per_image[identity]).items():
            groups[key].update(values)
    need({key: value["eligible"] for key, value in groups.items()} == DENOMINATORS,
         "Locatable-corner denominators changed")
    return dict(detection=aggregate_scores(list(per_image.values())),
                corners_by_locatable_count=groups, per_image=per_image)


def compact_evaluation(result):
    return dict(detection=compact_detection(result["detection"]),
                corners_by_locatable_count=result["corners_by_locatable_count"],
                missed_rows={identity: [index + 1 for index in item["unmatched_gold_indices"]]
                             for identity, item in result["per_image"].items() if item["fn"]})


def prediction_id(predictions, index):
    if index is None:
        return None
    return dict(prediction_number=index + 1,
                primary_prediction_id=predictions[index].get("primary_prediction_id", index + 1))


def match_map(result):
    return {match["gold_index"]: match for match in result["matches"]}


def target_assignments(before_gold, before, after, predictions, before_name="r38", after_name="r48"):
    old, new = match_map(before), match_map(after)
    rows = []
    for index, target in enumerate(before_gold):
        a, b = old.get(index), new.get(index)
        if a and b:
            status = "matched_retained" if a["prediction_index"] == b["prediction_index"] else "matched_reassigned"
        elif a:
            status = "matched_to_missed"
        elif b:
            status = "missed_to_matched"
        else:
            status = "missed_retained"
        item = dict(reference_row=index + 1,
                    visible_corners=sum(point[2] == 2 for point in target["keypoints"]),
                    transition=status)
        item.update({before_name + "_prediction": prediction_id(predictions, a["prediction_index"]) if a else None,
                     after_name + "_prediction": prediction_id(predictions, b["prediction_index"]) if b else None,
                     before_name + "_iou": a["iou"] if a else None,
                     after_name + "_iou": b["iou"] if b else None})
        rows.append(item)
    return rows


def best_overlap(prediction, gold):
    candidates = [(_box_iou(prediction["bbox_xyxy"], target["bbox_xyxy"]), index)
                  for index, target in enumerate(gold) if target["class_id"] == prediction["class_id"]]
    if not candidates:
        return None
    value, index = min(candidates, key=lambda item: (-item[0], item[1]))
    return dict(reference_row=index + 1, iou=value)


def prediction_transitions(before_gold, after_gold, predictions, before, after,
                           before_name="r38", after_name="r48"):
    old_matches = {item["prediction_index"]: item["gold_index"] for item in before["matches"]}
    new_matches = {item["prediction_index"]: item["gold_index"] for item in after["matches"]}
    old_duplicates, new_duplicates = (set(item["duplicate_prediction_indices"]) for item in (before, after))
    changes = []
    for index, prediction in enumerate(predictions):
        old = "matched" if index in old_matches else "duplicate" if index in old_duplicates else "other_unmatched"
        new = "matched" if index in new_matches else "duplicate" if index in new_duplicates else "other_unmatched"
        if old == new and old_matches.get(index) == new_matches.get(index):
            continue
        evidence = dict(**prediction_id(predictions, index), before=old, after=new,
                        confidence=prediction["confidence"], bbox_xyxy=prediction["bbox_xyxy"])
        evidence.update({before_name + "_matched_row": old_matches[index] + 1 if index in old_matches else None,
                         after_name + "_matched_row": new_matches[index] + 1 if index in new_matches else None,
                         "best_" + before_name + "_overlap": best_overlap(prediction, before_gold),
                         "best_" + after_name + "_overlap": best_overlap(prediction, after_gold)})
        for label, gold, result in ((before_name, before_gold, before), (after_name, after_gold, after)):
            evidence[label + "_overlaps_with_matched_targets"] = [
                dict(reference_row=pair["gold_index"] + 1, iou=overlap)
                for pair in result["matches"]
                if (overlap := _box_iou(prediction["bbox_xyxy"], gold[pair["gold_index"]]["bbox_xyxy"])) >= MATCHING_IOU
            ]
        changes.append(evidence)
    return changes


def target_details(target, index, predictions, result):
    pair = match_map(result).get(index)
    details = dict(matched=pair is not None, prediction=None, iou=None, endpoints=[])
    if pair is None:
        return details
    pi = pair["prediction_index"]
    details.update(prediction=prediction_id(predictions, pi), iou=pair["iou"])
    for side, point, proposed in zip(("left", "right"), target["keypoints"], predictions[pi]["keypoints"]):
        x, y, confidence = proposed
        finite = x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
        accepted = point[2] == 2 and finite and 0 <= x < 1280 and 0 <= y < 720 and confidence >= ENDPOINT_CONFIDENCE
        details["endpoints"].append(dict(side=side, eligible=point[2] == 2, accepted=accepted,
                                         error_px=math.hypot(x - point[0], y - point[1]) if accepted else None))
    eligible = [item for item in details["endpoints"] if item["eligible"]]
    for threshold in (5, 10):
        details[f"all_locatable_within{threshold}"] = bool(eligible) and all(
            item["accepted"] and item["error_px"] <= threshold for item in eligible)
    return details


def replay_labels(root, old_payloads, source_records):
    path = root / REVIEW / "audit-ledger.json"
    ledger_data = path.read_bytes()
    ledger = json.loads(ledger_data)
    need(ledger["round"] == 48 and ledger["parent_manifest_sha256"] == REFERENCES["r38"][1],
         "R48 ledger parent differs")
    need(ledger["human_review_performed"] is False, "R48 proposal review status differs")
    operations = ledger["rows"]
    need(len(operations) == 10 and len({(op["image_id"], op["reference_row"]) for op in operations}) == 10,
         "Expected ten distinct R48 development targets")
    payloads = dict(old_payloads)
    holds, amended = [], 0
    for identity, original in old_payloads.items():
        lines = original.decode().splitlines()
        for operation in operations:
            if operation["image_id"] != identity:
                continue
            index = operation["reference_row"] - 1
            need(0 <= index < len(lines) and lines[index] == operation["before_line"],
                 f"Exact R38 predecessor row differs: {operation['stable_id']}")
            need(operation["source_image_sha256"] == source_records[identity]["source_image_sha256"],
                 "Source photograph identity differs")
            after = amend_line(operation["before_line"], operation["updates_px"])
            need(after == operation["after_line"], "R48 line differs from numeric operation replay")
            need(changed_fields(operation["before_line"], after) == operation["changed_fields"],
                 "R48 changed-field inventory differs")
            old_tokens, new_tokens = operation["before_line"].split(), after.split()
            for offset in (5, 8):
                need(old_tokens[offset + 2] == new_tokens[offset + 2], "Corner visibility changed")
                if old_tokens[offset + 2] == "0":
                    need(old_tokens[offset:offset + 3] == new_tokens[offset:offset + 3],
                         "Unavailable endpoint tokens changed")
            amended += after != operation["before_line"]
            for held in operation["held_fields"]:
                need(held["field"] not in operation["updates_px"], "Held field was numerically amended")
                need(abs(geometry(after)[held["field"]] - geometry(operation["before_line"])[held["field"]]) < 1e-4,
                     "Held boundary changed during replay")
                holds.append(dict(stable_id=operation["stable_id"], **held))
            lines[index] = after
        payload = ("\n".join(lines) + "\n").encode() if lines else b""
        need(len(parse(payload, identity)) == source_records[identity]["objects"], "Reference object count changed")
        payloads[identity] = payload
    need(amended == 9, "Expected nine amended development rows")
    need({(item["stable_id"], item["field"]) for item in holds} == {
        ("new_000010:R38:029", "ymax"), ("new_000089:R38:041", "xmax")}, "Development hold inventory differs")
    return payloads, operations, holds, sha(ledger_data)


def verify_generated_labels(root, payloads, provisional):
    directory = root / REVIEW
    required = [directory / "manifest.json"] + [directory / "labels" / (identity + ".txt") for identity in payloads]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        need(provisional, "Generated R48 labels are required; missing " + ", ".join(missing))
        return dict(verified=False, missing=missing, status="provisional_unbuilt_labels")
    manifest_data = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    need(manifest["parent_manifest_sha256"] == REFERENCES["r38"][1]
         and manifest["development_only"] and not manifest["full_dataset_certified"]
         and manifest["changed_training_rows"] == 0, "Generated reference scope differs")
    entries = {item["image_id"]: item for item in manifest["files"]}
    need(len(manifest["files"]) == 20 and set(entries) == set(payloads), "Generated label membership differs")
    need({path.name for path in (directory / "labels").glob("*.txt")} == {identity + ".txt" for identity in payloads},
         "Unexpected or missing generated label")
    hashes = {}
    for identity, payload in payloads.items():
        entry = entries[identity]
        need(entry["label_path"] == f"labels/{identity}.txt", "Unexpected generated label path")
        data = checked(directory / entry["label_path"], entry["label_sha256"])
        need(data == payload, f"Committed R48 label differs from independent ledger replay: {identity}")
        hashes[identity] = sha(data)
    return dict(verified=True, label_count=20, all_bytes_equal=True,
                manifest_sha256=sha(manifest_data), label_sha256=hashes)


def operation_attribution(operations, old_gold, new_gold, predictions, old_scores, new_scores):
    evidence = []
    for operation in operations:
        if not operation["updates_px"]:
            continue
        identity, index = operation["image_id"], operation["reference_row"] - 1
        old, new = old_gold[identity], new_gold[identity]
        single, reverted = deepcopy(old), deepcopy(new)
        single[index] = deepcopy(new[index])
        reverted[index] = deepcopy(old[index])
        saved = predictions[identity]
        only_score, reverted_score = score(single, saved), score(reverted, saved)
        before, after = old_scores[identity], new_scores[identity]
        evidence.append(dict(
            stable_id=operation["stable_id"], image_id=identity, reference_row=index + 1,
            updates_px=operation["updates_px"], held_fields=operation["held_fields"],
            only_this_operation_from_r38=dict(
                image_metric_delta=difference(compact_detection(before), compact_detection(only_score)),
                changed_target_assignments=[item for item in target_assignments(old, before, only_score, saved, "r38", "single_operation")
                                            if item["transition"] not in ("matched_retained", "missed_retained")],
                changed_prediction_categories=prediction_transitions(old, single, saved, before, only_score, "r38", "single_operation")),
            revert_this_operation_from_full_r48=dict(
                reverted_minus_full_r48=difference(compact_detection(after), compact_detection(reverted_score)),
                changed_target_assignments=[item for item in target_assignments(new, after, reverted_score, saved, "full_r48", "reverted")
                                            if item["transition"] not in ("matched_retained", "missed_retained")],
                changed_prediction_categories=prediction_transitions(new, reverted, saved, after, reverted_score, "full_r48", "reverted"))))
    return evidence


def run(root, archive_path, provisional=False):
    root = Path(root).resolve()
    archive_data = checked(archive_path, ARCHIVE_SHA)
    need(len(archive_data) == ARCHIVE_BYTES, "Pinned PR137 archive size differs")
    manifest_path, manifest_sha, object_count = REFERENCES["r38"]
    manifest_data = checked(root / manifest_path, manifest_sha)
    manifest = json.loads(manifest_data)
    entries = [item for item in manifest["files"] if item.get("split", "val") == "val"]
    need(len(entries) == 20, "Expected twenty development photographs")
    records = {item["image_id"]: item for item in entries}
    need(len(records) == 20, "Duplicate R38 photograph identity")
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        need(archive.read("references/r38/manifest.json") == manifest_data, "Archived R38 manifest differs")
        for helper in ("pose_scoring.py", "evaluate_pose.py"):
            data = checked(root / "scripts" / helper, HELPERS[helper])
            need(archive.read("source/scripts/" + helper) == data, "Scoring/parser source differs from saved run")
        prediction_data = archive.read("evaluation/predictions.json")
        summary_data = archive.read("evaluation/summary.json")
        predictions, summary = json.loads(prediction_data), json.loads(summary_data)
        need(set(predictions) == set(ARMS), "Prediction arm membership differs")
        need(summary["source"] == SOURCE and summary["model_sha256"] == MODEL_HASHES,
             "Saved evaluation provenance differs")
        need(summary["inference"]["matching_iou"] == MATCHING_IOU
             and summary["inference"]["endpoint_conf"] == ENDPOINT_CONFIDENCE
             and summary["inference"]["conf"] == 0.13, "Fixed operating point differs")
        need(summary["references"]["r38"]["manifest_sha256"] == manifest_sha, "Saved R38 reference differs")
        old_payloads = {}
        for identity, entry in records.items():
            data = checked((root / manifest_path).parent / entry["label_path"], entry["label_sha256"])
            need(archive.read("references/r38/" + entry["label_path"]) == data, "Archived R38 label differs")
            old_payloads[identity] = data
    for arm in ARMS:
        need(set(predictions[arm]) == set(records), "Cached photograph membership differs")
    new_payloads, operations, holds, ledger_sha = replay_labels(root, old_payloads, records)
    generated = verify_generated_labels(root, new_payloads, provisional)
    old_gold = {identity: parse(data, identity) for identity, data in old_payloads.items()}
    new_gold = {identity: parse(data, identity) for identity, data in new_payloads.items()}
    need(sum(map(len, old_gold.values())) == sum(map(len, new_gold.values())) == object_count == 654,
         "Reference object denominator differs")
    for identity in old_gold:
        need([[p[2] for p in row["keypoints"]] for row in old_gold[identity]] ==
             [[p[2] for p in row["keypoints"]] for row in new_gold[identity]], "Visibility population changed")
    result = dict(
        schema="front-review-r48-fixed-cache-reference-sensitivity-v1",
        status="passed" if generated["verified"] else "provisional_unbuilt_labels",
        interpretation="Post-hoc development-reference sensitivity with predictions, models and operating point held fixed. Score differences are annotation/assignment effects, not model improvement or an independent generalization result.",
        fresh_inference=False, fresh_training=False, predictions_modified=False,
        provenance=dict(pr=137, pr_head=PR_HEAD, source=SOURCE, archive_sha256=ARCHIVE_SHA,
                        archive_bytes=ARCHIVE_BYTES, prediction_cache_sha256=sha(prediction_data),
                        saved_summary_sha256=sha(summary_data), r38_manifest_sha256=manifest_sha,
                        r48_audit_ledger_sha256=ledger_sha, model_sha256=MODEL_HASHES,
                        scoring_source_sha256=HELPERS["pose_scoring.py"]),
        operating_point=dict(matching_iou=MATCHING_IOU, endpoint_confidence=ENDPOINT_CONFIDENCE,
                             cached_inference=summary["inference"], primary_view="postclip"),
        population=dict(images=20, objects=654, visible_endpoints=1220,
                        locatable_corner_object_counts=DENOMINATORS,
                        reviewed_targets=10, amended_rows=9, added_rows=0, deleted_rows=0,
                        visibility_changes=0, human_review_performed=False),
        preserved_development_holds=holds, generated_labels_verification=generated,
        r38_exact_reproduction={}, references={"r38_verified": {}, "r48_proposed_diagnostic": {}},
        comparisons={},
        attribution_method="Each amended target is applied alone to R38 and separately reverted from full R48 in the same image. Full confidence-ranked greedy matching is rerun each time. These conditional effects need not add when assignments interact. Prediction IDs are one-based within their arm and view; primary IDs bind postclip entries to the same arm's raw cache.")
    for view in VIEWS:
        result["comparisons"][view] = {}
        result["r38_exact_reproduction"][view] = {}
        for reference in result["references"].values():
            reference[view] = {}
        for arm in ARMS:
            saved = {identity: predictions[arm][identity][view] for identity in old_gold}
            old, new = evaluate(old_gold, saved), evaluate(new_gold, saved)
            need(old == summary["references"]["r38"][view][arm], f"R38 exact score reproduction failed: {view}/{arm}")
            result["r38_exact_reproduction"][view][arm] = dict(passed=True, exact_full_score_dict_equal=True)
            result["references"]["r38_verified"][view][arm] = compact_evaluation(old)
            result["references"]["r48_proposed_diagnostic"][view][arm] = compact_evaluation(new)
            per_image, reviewed = {}, []
            for identity in old_gold:
                before, after = old["per_image"][identity], new["per_image"][identity]
                assignments = target_assignments(old_gold[identity], before, after, saved[identity])
                per_image[identity] = dict(
                    r38=compact_detection(before), r48=compact_detection(after),
                    delta_r48_minus_r38=difference(compact_detection(before), compact_detection(after)),
                    target_assignments=assignments,
                    changed_prediction_categories=prediction_transitions(old_gold[identity], new_gold[identity], saved[identity], before, after))
            for operation in operations:
                identity, index = operation["image_id"], operation["reference_row"] - 1
                reviewed.append(dict(stable_id=operation["stable_id"], updates_px=operation["updates_px"],
                                     held_fields=operation["held_fields"],
                                     r38=target_details(old_gold[identity][index], index, saved[identity], old["per_image"][identity]),
                                     r48=target_details(new_gold[identity][index], index, saved[identity], new["per_image"][identity])))
            result["comparisons"][view][arm] = dict(
                detection_delta_r48_minus_r38=difference(compact_detection(old["detection"]), compact_detection(new["detection"])),
                corner_group_delta={key: difference(old["corners_by_locatable_count"][key], new["corners_by_locatable_count"][key])
                                    for key in DENOMINATORS},
                per_image=per_image, reviewed_targets=reviewed,
                operation_attribution=operation_attribution(operations, old_gold, new_gold, saved,
                                                            old["per_image"], new["per_image"]))
    findings = {}
    for arm in ARMS:
        comparison = result["comparisons"]["postclip"][arm]
        assignment_counts = Counter(item["transition"] for image in comparison["per_image"].values()
                                    for item in image["target_assignments"])
        category_changes = [dict(image_id=identity, **item)
                            for identity, image in comparison["per_image"].items()
                            for item in image["changed_prediction_categories"]]
        findings[arm] = dict(
            target_transition_counts=dict(assignment_counts),
            total_unmatched_change=comparison["detection_delta_r48_minus_r38"]["fp"],
            miss_change=comparison["detection_delta_r48_minus_r38"]["fn"],
            duplicate_change=comparison["detection_delta_r48_minus_r38"]["duplicate_fp"],
            other_unmatched_change=comparison["detection_delta_r48_minus_r38"]["other_fp"],
            changed_prediction_categories=category_changes,
            two_corner_within5_change=comparison["corner_group_delta"]["2"]["within5"])
    result["primary_findings"] = findings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Exact reassembled PR137 R47 results ZIP")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help=f"Default: <root>/{OUTPUT}")
    parser.add_argument("--allow-unbuilt-labels", action="store_true",
                        help="Scratch-only provisional replay before the R48 builder exports labels")
    args = parser.parse_args()
    need(not args.allow_unbuilt_labels or args.output is not None,
         "Provisional replay requires an explicit scratch output path")
    output = args.output or args.root / OUTPUT
    if args.allow_unbuilt_labels:
        need(not output.resolve().is_relative_to((args.root / "results").resolve()),
             "Provisional output must not enter committed results")
    result = run(args.root, args.archive, args.allow_unbuilt_labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(output),
                          generated_labels_verified=result["generated_labels_verification"]["verified"],
                          postclip={arm: result["comparisons"]["postclip"][arm]["detection_delta_r48_minus_r38"]
                                    for arm in ARMS}), indent=2))


if __name__ == "__main__":
    main()
