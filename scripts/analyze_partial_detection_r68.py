#!/usr/bin/env python3
"""Diagnose three restored partial fronts and replay one fixed reflected-output union."""

import argparse
from collections import Counter
import csv
import gzip
import io
import json
import math
from pathlib import Path
import platform
import sys
import urllib.request
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import numpy as np

sys.dont_write_bytecode = True
import analyze_proposal_stages_r63 as raw_audit
from analyze_duplicate_audit_r66 import (
    INPUTS as PRIOR_INPUTS, HISTORICAL_PATH, digest, encoded, load_inputs, need, read,
    stage_inputs as stage_prior_inputs,
)
from audit_reflection_return import valid_point
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image
from predict_pose_folder import postclip_nms
from replay_partial_inclusion_r67 import (
    PROVENANCE_PATH, REFERENCE_PATH, candidate_reference, matrix_row,
)
from run_corner_average_r58 import average_predictions, report
from run_flip_inference_r56 import inverse_horizontal_predictions
from run_reference_matrix_r60 import verified_zip


PROTOCOL_PATH = "docs/thesis/evidence/partial-detection-r68/protocol.json"
PROTOCOL_SHA = "14fe3559bb22a584c17e29364e2ce7a481a7b0136c884a16ec93385da2a1ae71"
R67_PATH = "docs/thesis/evidence/partial-inclusion-r67/baomai-partial-inclusion-r67-results.zip"
INPUTS = {**PRIOR_INPUTS,
    "decoded": {"filename": raw_audit.R49_NAME, "bytes": raw_audit.R49_BYTES,
        "sha256": raw_audit.R49_SHA, "commit": raw_audit.R49_HEAD,
        "repository_path": "results/t4/baomai-r49-diagnosis-t4-001/" + raw_audit.R49_NAME},
    "r67": {"filename": "partial-inclusion-r67.zip", "bytes": 1093811,
        "sha256": "18c0e5491bb59b2b242853fc3df2c0ef3655099e5c98a08194846814923665cd",
        "repository_path": R67_PATH},
}
FOCUS_IDS = ("r34/new_000009/R056", "r37/new_000009/R052", "r37/new_000053/R001")
REFERENCES = ("r50", "r67", "supplied664", "frozen666")
VARIANTS = ("unchanged", "unpaired_reflection_union")
RESULT_NAME = "baomai-partial-detection-r68-results.zip"
SOURCES = (
    "analyze_partial_detection_r68.py", "analyze_duplicate_audit_r66.py",
    "analyze_proposal_stages_r63.py", "audit_additional_front_pilot_r45_results.py",
    "audit_additional_front_pilot_r47_results.py", "audit_front_diagnosis_r49_results.py",
    "audit_reflection_return.py", "diagnose_fronts_r50.py", "evaluate_pose.py",
    "export_front_error_review_r48.py", "pose_checkpoint.py", "pose_scoring.py",
    "predict_pose_folder.py", "replay_front_confidence_r51.py", "replay_partial_inclusion_r67.py",
    "run_corner_average_r58.py", "run_flip_inference_r56.py", "run_inference_ablation_r53.py",
    "run_original_comparison_r59.py", "run_reference_matrix_r60.py",
    "run_training_label_comparison_r61.py",
)


def stage_inputs(root, destination):
    stage_prior_inputs(root, destination)
    for name in ("decoded", "r67"):
        spec = INPUTS[name]
        path = destination / spec["filename"]
        if not path.exists():
            if name == "r67":
                data = (root / R67_PATH).read_bytes()
            else:
                url = "https://raw.githubusercontent.com/Siuuuers/dl/" + spec["commit"] + "/" + spec["repository_path"]
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
            need(len(data) == spec["bytes"] and digest(data) == spec["sha256"], "Downloaded archive differs: " + name)
            path.write_bytes(data)
        verified_zip(path, spec["sha256"], spec["bytes"])


def dimensions(box):
    width, height = box[2] - box[0], box[3] - box[1]
    return {"width_px": width, "height_px": height, "area_px2": width * height}


def geometry(prediction, target):
    a, b = target["bbox_xyxy"], prediction["bbox_xyxy"]
    before, after = dimensions(a), dimensions(b)
    return {"target_box": before, "candidate_box": after,
        "edge_offsets_ltrb_px": [b[i] - a[i] for i in range(4)],
        "width_ratio": after["width_px"] / before["width_px"],
        "height_ratio": after["height_px"] / before["height_px"],
        "area_ratio": after["area_px2"] / before["area_px2"]}


def best_record(values, target):
    if not values:
        return None
    index = max(range(len(values)), key=lambda i: (
        _box_iou(values[i]["bbox_xyxy"], target["bbox_xyxy"]), values[i]["confidence"], -i))
    p = values[index]
    overlap = _box_iou(p["bbox_xyxy"], target["bbox_xyxy"])
    if overlap <= 0:
        return None
    return {"index": index, "iou": overlap, "prediction": p, **geometry(p, target)}


def known_corner_outcomes(target, prediction):
    rows = []
    for index, point in enumerate(target["keypoints"]):
        if point[2] != 2:
            continue
        proposed = prediction["keypoints"][index] if prediction is not None else None
        available = proposed is not None and valid_point(proposed)
        error = math.hypot(proposed[0] - point[0], proposed[1] - point[1]) if available else None
        rows.append({"corner_index": index, "reference_xy": point[:2], "prediction": proposed,
                     "available": available, "error_px": error,
                     "within5": error is not None and error <= 5,
                     "within10": error is not None and error <= 10})
    return rows


def view_outputs(archives, methods):
    historical = read(archives["historical"], "native-predictions.json.gz")
    qfl = read(archives["qfl"], "evaluation/details.json.gz")["native-predictions"]
    labels = read(archives["labels"], "native-predictions.json.gz")
    result = {}
    for method, data in methods.items():
        if method == "historical_retained":
            native = {"original": historical["trained"], "reflected": historical["reflected"]}
        else:
            source = qfl[{"pr157_native": "reference", "pr157_qfl": "candidate"}[method]] \
                if method.startswith("pr157") else labels[method[6:]]
            native = {"original": source["original"],
                      "reflected": {image: inverse_horizontal_predictions(values)
                                    for image, values in source["reflected"].items()}}
        result[method] = {view: {image: postclip_nms([p for p in values if p["confidence"] > .25])[0]
                                for image, values in images.items()} for view, images in native.items()}
        for image in data["predictions"]:
            completed, links = average_predictions(result[method]["original"][image], result[method]["reflected"][image])
            need(completed == data["predictions"][image] and links == data["associations"][image],
                 "Completed view outputs or associations differ: " + method + "/" + image)
    return result


def source_id(method, image, view, index):
    return f"{method}/{image}/{'O' if view == 'original' else 'F'}{index + 1:03d}"


def union_outputs(method, data, views):
    output, identities, selected, counts = {}, {}, {}, Counter()
    for image, original in data["predictions"].items():
        reflected, links = views["reflected"][image], data["associations"][image]
        used = {link["reflected_index"] for link in links if link["reflected_index"] is not None}
        added = [i for i in range(len(reflected)) if i not in used]
        pool = original + [reflected[i] for i in added]
        pool_ids = [source_id(method, image, "original", i) for i in range(len(original))] + \
                   [source_id(method, image, "reflected", i) for i in added]
        _, audit = postclip_nms(pool)
        kept = [i - 1 for i in audit["retained_primary_prediction_ids"]]
        suppressed = {r["primary_prediction_id"] - 1: r for r in audit["suppressed_predictions"]}
        output[image] = [pool[i] for i in kept]
        identities[image] = [pool_ids[i] for i in kept]
        rows = []
        for i, value in enumerate(pool):
            view, vi = ("original", i) if i < len(original) else ("reflected", added[i - len(original)])
            loss = suppressed.get(i)
            rows.append({"pool_index": i, "source_id": pool_ids[i], "source_view": view, "source_index": vi,
                "prior_primary_prediction_id": value.get("primary_prediction_id"), "prediction": value,
                "retained": i in kept, "union_output_index": kept.index(i) if i in kept else None,
                "suppressor_source_id": pool_ids[loss["suppressor_primary_prediction_id"] - 1] if loss else None,
                "suppression_iou": loss["postclip_prediction_iou"] if loss else None})
        selected[image] = {"original_count": len(original), "reflected_count": len(reflected),
            "used_reflected_indices": sorted(used), "added_reflected_indices": added,
            "union_source_ids": identities[image], "rows": rows}
        counts.update(original_outputs=len(original), reflected_outputs=len(reflected), added_reflected=len(added),
            retained_reflected=sum(r["retained"] and r["source_view"] == "reflected" for r in rows),
            suppressed_original=sum(not r["retained"] and r["source_view"] == "original" for r in rows),
            union_outputs=len(kept))
    return output, identities, selected, dict(counts)


def support_gate(method, gold, data, views, selection):
    _, scores, _ = report(gold, data["predictions"])
    rows = []
    for image, score in scores.items():
        candidates = [r for r in selection[image]["rows"] if r["source_view"] == "reflected"]
        for gi in score["unmatched_gold_indices"]:
            eligible = [r for r in candidates if _box_iou(r["prediction"]["bbox_xyxy"], gold[image][gi]["bbox_xyxy"]) >= .5]
            if eligible:
                rows.append({"method": method, "image": image, "reference_id": f"r67/{image}/R{gi + 1:03d}",
                    "reference_index": gi, "known_corners": sum(p[2] == 2 for p in gold[image][gi]["keypoints"]),
                    "eligible_unpaired_source_ids": [r["source_id"] for r in eligible],
                    "surviving_eligible_unpaired_source_ids": [r["source_id"] for r in eligible if r["retained"]]})
    return {"missed_targets_with_unpaired_support": len(rows),
            "supported_after_fixed_nms": sum(bool(r["surviving_eligible_unpaired_source_ids"]) for r in rows)}, rows


def decoded_diagnostic(root, archives, input_directory, gold, focused, final):
    # Reuse the completed r63 bridge and its historical identity guards unchanged.
    prior, _ = raw_audit.analyze(input_directory / INPUTS["decoded"]["filename"],
        input_directory / INPUTS["historical"]["filename"], root)
    old = archives["decoded"]
    traces, saved = read(old, "trace.json"), read(old, "predictions.json")
    native = read(archives["historical"], "native-predictions.json.gz")
    cases, bounds, bridges = [], [], []
    for image in sorted({row["image"] for row in focused}):
        raw = np.load(io.BytesIO(old["decoded/" + image + ".npy"]), allow_pickle=False)
        trace = traces[image]
        need(raw.dtype == np.float32 and list(raw.shape) == trace["shape"]
             and digest(raw.tobytes()) == trace["raw_tensor_sha256"], "Focused tensor differs")
        boxes = raw_audit.input_boxes(raw)
        old_indices, _ = raw_audit.native_select(boxes, raw[0, 4], .13)
        old_final, old_audit = postclip_nms(raw_audit.prediction_records(raw, boxes, old_indices))
        need(old_final == saved[image]["hard"] and old_audit == saved[image]["hard_audit"], "Historical focused replay differs")
        old_final_indices = [old_indices[i - 1] for i in old_audit["retained_primary_prediction_ids"]]
        indices, suppressed = raw_audit.native_select(boxes, raw[0, 4], .25)
        records = raw_audit.prediction_records(raw, boxes, indices)
        difference = raw_audit.record_difference(records, native["trained"][image])
        current, audit = postclip_nms(records)
        final_indices = [indices[i - 1] for i in audit["retained_primary_prediction_ids"]]
        score = score_image(gold[image], current, 1280, 720, .5, .5)
        need(raw_audit.score_identity(score) == raw_audit.score_identity(
            score_image(gold[image], final[image], 1280, 720, .5, .5)), "r67 focused matching identity differs")
        overlap = raw_audit.iou_matrix(raw_audit.original_boxes(boxes), gold[image])
        all_cases, _ = raw_audit.describe_cases(image, gold[image], raw, boxes, overlap, indices, suppressed,
            current, audit, final_indices, old_final, old_final_indices)
        selected = {row["candidate_index"]: row for row in focused if row["image"] == image}
        for case in all_cases:
            gi = case["reference_index"]
            if gi not in selected:
                continue
            case.update(reference="r67", stable_id=selected[gi]["stable_id"],
                        candidate_reference_id=selected[gi]["candidate_reference_id"])
            for name in ("best_decoded", "highest_confidence_eligible"):
                if case[name] is not None:
                    case[name].update(geometry(case[name], case["target"]))
            cases.append(case)
            highest = case["highest_confidence_eligible"]
            bounds.append({"stable_id": case["stable_id"], "image": image,
                "finite_decoded_candidates": raw.shape[2], "iou_eligible_candidates": case["eligible_decoded_count"],
                "highest_eligible_score": highest["confidence"] if highest else None,
                "at_current_cutoff_eligible_candidates": case["eligible_above_confidence_count"],
                "minimum_decoded_candidates_admitted_by_cutoff_below_highest_eligible_score":
                    int(np.count_nonzero(raw[0, 4] >= highest["confidence"])) if highest else None,
                "interpretation": "A confidence cutoff can only discard fixed boxes; support before NMS is an upper bound, not recovered detections."})
        bridges.append({"image": image, "historical013_exact": True, "r67_matching_identity_exact": True, **difference})
    need(len(cases) == len(FOCUS_IDS), "Focused miss identity differs")
    return {"cases": cases, "threshold_support_bound": bounds, "focused_r67_bridge": bridges,
            "prior_r63_continuity": prior["verification"], "raw_runtime_comparison": prior["runtime_comparison"],
            "fixed_pool_supported_focus_targets": sum(row["iou_eligible_candidates"] > 0 for row in bounds),
            "fixed_pool_focus_targets": len(bounds)}


def saved_stage_support(archives, methods, gold, focused, all_final):
    old = read(archives["historical"], "native-predictions.json.gz")
    qfl = read(archives["qfl"], "evaluation/details.json.gz")["native-predictions"]
    labels = read(archives["labels"], "native-predictions.json.gz")
    qsummary = read(archives["qfl"], "evaluation/summary.json")
    native_rows, final_rows = [], []
    for row in focused:
        image, gi = row["image"], row["candidate_index"]
        target = gold[image][gi]
        for method, images in all_final.items():
            values = images[image]
            _, scores, _ = report({image: gold[image]}, {image: values})
            need(gi in scores[image]["unmatched_gold_indices"], "A focus target was not a saved final miss")
            best = best_record(values, target)
            final_rows.append({"stable_id": row["stable_id"], "candidate_reference_id": row["candidate_reference_id"],
                "method": method, "image": image, "target": target, "output_count": len(values),
                "best_positive_overlap_final": best, "no_positive_overlap_output": best is None})
        native_sets = [("supplied_original", "original", "r60_native_conf025", old["original"][image]),
            ("historical_retained", "original", "r60_native_conf025", old["trained"][image]),
            ("historical_retained", "reflected", "r60_native_conf025", old["reflected"][image])]
        for role, method in (("supplied", "supplied_original"), ("reference", "pr157_native"), ("candidate", "pr157_qfl")):
            if method in methods:
                need(qsummary["model_sha256"][role] == methods[method]["model_sha256"], "Current model identity differs")
            for view, images in qfl[role].items():
                values = images[image] if view == "original" else inverse_horizontal_predictions(images[image])
                native_sets.append((method, view, "pr157_native_conf0001", values))
        for role, native in labels.items():
            for view, images in native.items():
                values = images[image] if view == "original" else inverse_horizontal_predictions(images[image])
                native_sets.append(("pr158_" + role, view, "pr158_native_conf025", values))
        for method, view, scope, values in native_sets:
            eligible = [i for i, p in enumerate(values) if _box_iou(p["bbox_xyxy"], target["bbox_xyxy"]) >= .5]
            above = [p for p in values if p["confidence"] > .25]
            kept, _ = postclip_nms(above)
            best = best_record(values, target)
            native_rows.append({"stable_id": row["stable_id"], "candidate_reference_id": row["candidate_reference_id"],
                "method": method, "view": view, "cache_stage": scope, "saved_native_count": len(values),
                "output_cap_300_reached": len(values) == 300, "eligible_native_count": len(eligible),
                "highest_eligible_score": max((values[i]["confidence"] for i in eligible), default=None),
                "eligible_above025_count": sum(values[i]["confidence"] > .25 for i in eligible),
                "eligible_after_postclip_count": sum(_box_iou(p["bbox_xyxy"], target["bbox_xyxy"]) >= .5 for p in kept),
                "eligible_saved_native_records": [{"index": i, "prediction": values[i],
                    "iou": _box_iou(values[i]["bbox_xyxy"], target["bbox_xyxy"]), **geometry(values[i], target)} for i in eligible],
                "best_positive_overlap_saved_native": best, "no_positive_overlap_native": best is None})
    return native_rows, final_rows


def training_support(archive):
    staging = read(archive, "staging.json")
    train = [r for r in staging["roles"]["reference"] if r["split"] == "train"]
    need(train == [r for r in staging["roles"]["candidate"] if r["split"] == "train"], "Paired training inventories differ")
    all_rows, labels = [], {}
    for source in train:
        data = archive["datasets/reference/" + source["label"]]
        need(digest(data) == source["label_sha256"] and data == archive["datasets/candidate/" + source["label"]], "Frozen training label differs")
        labels[source["id"]] = digest(data)
        for index, target in enumerate(parse_pose_labels(data.decode(), 1280, 720, source["id"])):
            all_rows.append({"image_id": source["id"], "row": index + 1, "source_kind": source["source_kind"],
                "known_corners": sum(p[2] == 2 for p in target["keypoints"]), **dimensions(target["bbox_xyxy"])})
    need(len(train) == 142 and len(all_rows) == 3999, "Frozen training population differs")
    one = [r for r in all_rows if r["known_corners"] == 1]
    strata = []
    for identity, width, height in ((FOCUS_IDS[0], 18, 31), (FOCUS_IDS[1], 15, 14), (FOCUS_IDS[2], 47, 10)):
        selected = [r for r in one if r["width_px"] <= width and r["height_px"] <= height]
        tolerant = [r for r in one if r["width_px"] <= width + .001 and r["height_px"] <= height + .001]
        need(selected == tolerant, "Size-boundary serialization sensitivity changed focused count")
        strata.append({"focus_stable_id": identity, "width_lte_px": width, "height_lte_px": height,
            "known_corners": 1, "count": len(selected), "images": len({r["image_id"] for r in selected}),
            "same_count_with_0p001px_boundary_tolerance": True, "rows": selected})
    return {"source": "PR157 frozen paired training labels", "images": len(train), "fronts": len(all_rows),
        "one_known_corner_fronts": len(one), "nominal_image_size": [1280, 720], "label_sha256": labels,
        "strata": strata, "interpretation": "Descriptive size bins chosen from the three development targets. Equal box size does not establish similar appearance or post-augmentation training support; these counts do not diagnose a loss defect."}


def paired_counterexample(methods, views, gold, focused):
    target_row = next(row for row in focused if row["stable_id"] == FOCUS_IDS[1])
    method, image = "pr157_qfl", target_row["image"]
    target = gold[image][target_row["candidate_index"]]
    reflected = views[method]["reflected"][image]
    eligible = [i for i, p in enumerate(reflected) if _box_iou(p["bbox_xyxy"], target["bbox_xyxy"]) >= .5]
    need(len(eligible) == 1, "Paired counterexample identity differs")
    j = eligible[0]
    link = next((link for link in methods[method]["associations"][image] if link["reflected_index"] == j), None)
    need(link is not None, "Eligible reflected counterexample is no longer paired")
    original = methods[method]["predictions"][image][link["original_index"]]
    overlap = _box_iou(original["bbox_xyxy"], reflected[j]["bbox_xyxy"])
    need(overlap > .7 and original["confidence"] > reflected[j]["confidence"], "Counterexample suppression relation differs")
    return {"stable_id": target_row["stable_id"], "method": method, "association": link,
        "original_source_id": source_id(method, image, "original", link["original_index"]),
        "reflected_source_id": source_id(method, image, "reflected", j), "original": original, "reflected": reflected[j],
        "original_target_iou": _box_iou(original["bbox_xyxy"], target["bbox_xyxy"]),
        "reflected_target_iou": _box_iou(reflected[j]["bbox_xyxy"], target["bbox_xyxy"]),
        "prediction_pair_iou": overlap, "excluded_by_unpaired_rule": True,
        "would_lose_to_original_in_full_union_nms070": True,
        "interpretation": "Corner-pair unavailability does not undo an existing mutual box link. This eligible reflected box cannot motivate recovery by an unpaired-only union."}


def evaluation_ledgers(reference, method, gold, before_output, after_output, after_ids, selection, lineage_ids):
    before, bscores, bstates = report(gold, before_output)
    after, ascores, astates = report(gold, after_output)
    rows = [{**matrix_row(reference, method, measurement, states, gold, output), "variant": variant}
            for variant, measurement, states, output in (
                ("unchanged", before, bstates, before_output), ("unpaired_reflection_union", after, astates, after_output))]
    changes, output_changes = [], []
    for b, a in zip(bstates, astates):
        image, gi = b["image"], b["reference_index"]
        need((image, gi) == (a["image"], a["reference_index"]), "Target case order differs")
        bi, ai = b["prediction_index"], a["prediction_index"]
        bid = source_id(method, image, "original", bi) if bi is not None else None
        aid = after_ids[image][ai] if ai is not None else None
        if bid == aid and all(b[k] == a[k] for k in ("detected", "within5", "within10", "maximum_corner_error_px")):
            continue
        target = gold[image][gi]
        changes.append({"reference": reference, "method": method, "image": image,
            "reference_id": f"{reference}/{image}/R{gi + 1:03d}",
            "stable_reference_id": lineage_ids[(image, gi)] if reference == "r67" else f"{reference}/{image}/R{gi + 1:03d}",
            "known_corners": sum(p[2] == 2 for p in target["keypoints"]), "target": target,
            "detection_gain": not b["detected"] and a["detected"], "detection_loss": b["detected"] and not a["detected"],
            "rematched": b["detected"] and a["detected"] and bid != aid,
            "within5_gain": not b["within5"] and a["within5"], "within5_loss": b["within5"] and not a["within5"],
            "within10_gain": not b["within10"] and a["within10"], "within10_loss": b["within10"] and not a["within10"],
            "before": {**b, "source_id": bid, "known_corner_outcomes": known_corner_outcomes(target, before_output[image][bi] if bi is not None else None)},
            "after": {**a, "source_id": aid, "known_corner_outcomes": known_corner_outcomes(target, after_output[image][ai] if ai is not None else None)}})
    for image in gold:
        def states(score, ids):
            matches = {m["prediction_index"]: m["gold_index"] for m in score["matches"]}
            duplicates = set(score["duplicate_prediction_indices"])
            return {identity: {"status": "matched" if i in matches else "duplicate" if i in duplicates else "other_extra",
                "reference_id": f"{reference}/{image}/R{matches[i] + 1:03d}" if i in matches else None}
                for i, identity in enumerate(ids)}
        bmap = states(bscores[image], [source_id(method, image, "original", i) for i in range(len(before_output[image]))])
        amap = states(ascores[image], after_ids[image])
        for candidate in selection[image]["rows"]:
            identity = candidate["source_id"]
            b = bmap.get(identity, {"status": "not_in_baseline", "reference_id": None})
            a = amap.get(identity, {"status": "suppressed", "reference_id": None})
            if b != a:
                output_changes.append({"reference": reference, "method": method, "image": image,
                    "source_id": identity, "source_view": candidate["source_view"], "before": b, "after": a,
                    "suppressor_source_id": candidate["suppressor_source_id"], "suppression_iou": candidate["suppression_iou"]})
    return rows, changes, output_changes, {"unchanged": bscores, "unpaired_reflection_union": ascores}


def analyze(root, input_directory):
    protocol_bytes = (root / PROTOCOL_PATH).read_bytes()
    need(digest(protocol_bytes) == PROTOCOL_SHA, "Fixed protocol changed")
    protocol = json.loads(protocol_bytes)
    references, methods, provenance = load_inputs(input_directory)
    archives = {name: verified_zip(input_directory / spec["filename"], spec["sha256"], spec["bytes"])
                for name, spec in INPUTS.items()}
    text = read(archives["historical"], "reference-labels.json.gz")
    candidate, candidate_text, lineage, manifest, reference_hashes = candidate_reference(root, text["r50"])
    references["r67"], text["r67"] = candidate, candidate_text
    focused = [row for row in lineage if row["stable_id"] in FOCUS_IDS]
    need(len(focused) == 3, "Focused lineage differs")
    all_final = read(archives["r67"], "predictions.json.gz")
    saved_r67_scores = read(archives["r67"], "per-image.json.gz")
    need(read(archives["r67"], "reference-labels.json.gz") == {r: text[r] for r in ("r50", "r67")}, "r67 archived references differ")
    for method, images in all_final.items():
        if method in methods:
            need(images == methods[method]["predictions"], "r67 archived completed predictions differ")
        for reference in ("r50", "r67"):
            _, scores, _ = report(references[reference], images)
            need(scores == saved_r67_scores[method][reference], "r67 original score replay differs")
    views = view_outputs(archives, methods)
    unions, ids, selections, selection_counts, gate_counts, gate_cases = {}, {}, {}, {}, {}, []
    for method, data in methods.items():
        unions[method], ids[method], selections[method], selection_counts[method] = union_outputs(method, data, views[method])
        gate_counts[method], cases = support_gate(method, candidate, data, views[method], selections[method])
        gate_cases.extend(cases)
    need(gate_counts == protocol["support_gate"], "Pre-global support gate does not reproduce")
    need(any(row["supported_after_fixed_nms"] for row in gate_counts.values()), "No support justifies the fixed union probe")
    focused_reference_ids = {row["candidate_reference_id"] for row in focused}
    need(not any(row["reference_id"] in focused_reference_ids for row in gate_cases), "Union focus scope changed")
    diagnostic = decoded_diagnostic(root, archives, input_directory, candidate, focused, all_final["historical_retained"])
    native_support, final_geometry = saved_stage_support(archives, methods, candidate, focused, all_final)
    training = training_support(archives["qfl"])
    counterexample = paired_counterexample(methods, views, candidate, focused)
    matrix, changes, output_changes, per_image = [], [], [], {}
    lineage_ids = {(row["image"], row["candidate_index"]): row["stable_id"] for row in lineage}
    for method, data in methods.items():
        per_image[method] = {}
        for reference in REFERENCES:
            rows, targets, outputs, scores = evaluation_ledgers(reference, method, references[reference],
                data["predictions"], unions[method], ids[method], selections[method], lineage_ids)
            matrix.extend(rows); changes.extend(targets); output_changes.extend(outputs)
            per_image[method][reference] = scores
    need(len(matrix) == 40, "Fixed comparison matrix size differs")
    for row in focused:
        for method in methods:
            score = per_image[method]["r67"]["unpaired_reflection_union"][row["image"]]
            need(row["candidate_index"] in score["unmatched_gold_indices"], "Unexpected focus recovery requires explanation")
    source_hashes = {"scripts/" + name: digest((root / "scripts" / name).read_bytes()) for name in SOURCES}
    summary = {"schema": "partial-detection-r68-v1", "status": "completed", "fresh_inference": False,
        "training": False, "labels_changed": False, "primary_scorer_modified": False, "automatic_promotion": False,
        "development_only": True, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu"},
        "protocol_path": PROTOCOL_PATH, "protocol_sha256": PROTOCOL_SHA, "source_archives": INPUTS,
        "analysis_source_sha256": source_hashes, "reference_input_sha256": reference_hashes,
        "methods": {name: {k: v for k, v in data.items() if k not in ("predictions", "associations")} for name, data in methods.items()},
        "provenance": provenance, "r67_archived_image_scores_replayed": 240,
        "focus": [{**row, "target": candidate[row["image"]][row["candidate_index"]]} for row in focused],
        "support_gate": gate_counts, "selection_counts": selection_counts, "matrix": matrix,
        "focused_stage_counts": dict(Counter(case["stage"] for case in diagnostic["cases"])),
        "record_counts": {"union_target_changes": len(changes), "union_output_changes": len(output_changes),
            "union_score_objects": 800, "native_stage_focus_rows": len(native_support), "final_geometry_rows": len(final_geometry)},
        "limitations": [
            "All comparisons use repeatedly inspected development photographs. No model or threshold is promoted.",
            "Complete decoded pre-confidence/pre-NMS tensors exist only for the historical adapted original view. Newer caches are already native-NMS filtered; PR157 keeps scores >0.001 with a 300-output cap, while PR158 and r60 native caches keep scores >0.25.",
            "A missing current cached candidate cannot be attributed to raw confidence, geometry, native NMS or the cap without missing raw tensors. A cap-hit flag does not prove a relevant candidate was truncated.",
            "Raw historical tensors and common-runtime final records differ numerically; r63 continuity and focused r67 matching are checked without asserting byte-identical inference.",
            "The greatest-overlap saved box can describe a neighboring front. It is a comparison of boxes, not proof of the network's physical ownership intent.",
            "The union adds all unpaired reflected outputs using prediction-only associations. Reflected corners remain unaveraged, scores are unchanged, and added outputs may suppress original outputs or change evaluation matching.",
            "Native views each had max_det=300; the existing postclip union pass introduces no additional cap. Source-view identities distinguish additions, suppression and rematching.",
            "Finite-pool threshold support is an upper bound before suppression and matching, not a trained result, usable threshold recommendation or estimate of output precision.",
            "Training size bins are descriptive, chosen from the three development targets, and count nominal original-image labels before augmentation. They do not establish visual similarity, effective training exposure or an optimal loss.",
            "The three restored focus targets remain missed by this fixed union. No labels, detector tensors, boxes, scores or existing averaged corner coordinates are modified.",
        ]}
    records = {"stage-diagnostic": diagnostic, "available-native-support": native_support,
        "final-focused-geometry": final_geometry, "training-support": training, "paired-counterexample": counterexample,
        "union-support-gate": {"counts": gate_counts, "cases": gate_cases}, "union-selection": selections,
        "union-target-changes": changes, "union-output-changes": output_changes, "per-image": per_image,
        "predictions": {name: {"unchanged": methods[name]["predictions"], "unpaired_reflection_union": unions[name]} for name in methods},
        "reference-labels": text, "reference-lineage": lineage}
    return summary, records


def markdown(summary, records):
    lines = ["# Partial-front stages and one reflected-output union", "",
        "This completed CPU replay uses immutable saved outputs. It runs no model or training and changes no annotation. "
        "The three focused restorations remain missed. The separate union experiment asks whether other missed fronts "
        "already have unused reflected-view detections under the existing thresholds.", "",
        "## Three restored misses", "",
        "Only the historical adapted original-view checkpoint has a complete decoded pool. Its focused stage trace is:", "",
        "| Historical target | Stage | Best decoded IoU | Best-box confidence | Eligible decoded boxes |",
        "|---|---|---:|---:|---:|"]
    for row in records["stage-diagnostic"]["cases"]:
        best = row["best_decoded"]
        lines.append(f"| {row['stable_id']} | {row['stage']} | {best['target_iou']:.6f} | {best['confidence']:.8f} | {row['eligible_decoded_count']} |")
    lines += ["", "Two targets have eligible boxes below the confidence cutoff. The lower009 target has no IoU-eligible box "
        "anywhere in this fixed original-view tensor. Lowering a confidence cutoff cannot create a missing box geometry. "
        "The bound is limited to this checkpoint and view. Newer post-native-NMS caches do not provide equivalent stage attribution.", "",
        "The QFL reflected view does contain an eligible lower009 box above 0.25. It is already mutually paired to a higher-scoring "
        "original box. Its corner-pair-unavailable status does not make it unpaired, and the original would also suppress it in "
        "an unrestricted union at 0.70. This is a counterexample to treating every additional view response as recoverable output.", "",
        "## Fixed union, all fronts", "",
        "Retain the completed original-view detections and their existing averaged corners. Append every reflected output unused "
        "by any existing mutual box link, then apply the same score-ranked postclip suppression at IoU strictly above 0.70. "
        "No reference is used to select additions; no coordinates, scores, weights, thresholds or caps are tuned. "
        "The preliminary r67 support gate was fixed before measuring this full comparison.", "",
        "| Method | Variant | r67 misses | Extras | Duplicate subset | Pair ≤5 px | Pair ≤10 px |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["matrix"]:
        if row["reference"] == "r67":
            lines.append(f"| {row['method']} | {row['variant']} | {row['missed']} | {row['extras']} | {row['duplicates']} | "
                         f"{row['within5']}/{row['eligible_pairs']} | {row['within10']}/{row['eligible_pairs']} |")
    lines += ["", "The complete 40-row matrix also includes frozen r50, supplied664 and frozen666 sensitivity references. "
        "The per-target ledger identifies gains, losses and rematches, with confidence-qualified errors for every annotated known corner. "
        "The output ledger distinguishes added reflected detections from original outputs removed by the new pooled suppression pass. "
        "A better miss count alone does not establish a preferable operating method.", "",
        "## Training-label support", "",
        f"The unchanged PR157 training labels contain {records['training-support']['fronts']} fronts, "
        f"including {records['training-support']['one_known_corner_fronts']} with exactly one known corner. "
        "At nominal 1280×720 image coordinates, the one-known-corner counts within the focused width/height bounds are:", "",
        "| Width and height bound | Fronts | Images |", "|---|---:|---:|"]
    for row in records["training-support"]["strata"]:
        lines.append(f"| ≤{row['width_lte_px']} × ≤{row['height_lte_px']} px | {row['count']} | {row['images']} |")
    lines += ["", "These descriptive counts identify a possible data-support issue for a later controlled experiment. "
        "They do not show that a particular loss or augmentation will improve these targets. "
        "The restored development labels are not training-label additions.", "", "## Limits", ""]
    lines += ["- " + line for line in summary["limitations"]]
    return "\n".join(lines) + "\n"


def write_outputs(directory, summary, records, root):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"summary.json": encoded(summary), "results.md": markdown(summary, records).encode(),
               "protocol.json": (root / PROTOCOL_PATH).read_bytes()}
    small = {"stage-diagnostic", "available-native-support", "final-focused-geometry", "training-support", "paired-counterexample", "union-support-gate"}
    for name, value in records.items():
        payload[name + (".json" if name in small else ".json.gz")] = encoded(value) if name in small else gzip.compress(encoded(value), mtime=0)
    rows = []
    for row in summary["matrix"]:
        value = {k: v for k, v in row.items() if k != "known_corner_groups"}
        for group, counts in row["known_corner_groups"].items():
            value.update({f"known{group}_{key}": count for key, count in counts.items()})
        rows.append(value)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload["matrix.csv"] = table.getvalue().encode()
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
    payload["source.json"] = encoded({"source_commit": source_commit, "source_sha256": summary["analysis_source_sha256"],
        "reference_input_sha256": summary["reference_input_sha256"], "protocol_sha256": PROTOCOL_SHA,
        "no_new_inference_or_training": True})
    payload["inventory.json"] = encoded({name: {"bytes": len(data), "sha256": digest(data)} for name, data in payload.items() if name != "inventory.json"})
    path = directory / RESULT_NAME
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)); entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16; archive.writestr(entry, data)
    need(verified_zip(path) == payload, "Result archive differs")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage-inputs", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage_inputs:
        stage_inputs(root, args.inputs)
    summary, records = analyze(root, args.inputs)
    write_outputs(args.output, summary, records, root)
    print(json.dumps({"status": summary["status"], "matrix_rows": len(summary["matrix"]), "record_counts": summary["record_counts"]}, indent=2))


if __name__ == "__main__":
    main()
