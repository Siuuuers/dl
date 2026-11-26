#!/usr/bin/env python3
"""Trace the eleven accepted R38 misses through unchanged PR123 inference.

Inference needs the pinned CPU environment, original images and PR123 candidate
checkpoint. ``--summarize`` needs only Python and the compact saved evidence.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REVIEW = "data/reviews/fronts-2026-11-26-r38/manifest.json"
COMPARISON = "results/evaluation/pr123-r37-20261126/comparison.json"
CACHE = "results/evaluation/pr123-r37-20261126/predictions/candidate.jsonl"
OUT = "results/evaluation/pr123-r39-diagnosis-20261126"
BASE_CONF = .13
PROBE_CONF = .05
MATCH_IOU = .5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cause(case):
    eligible = case["iou_eligible_proposals"]
    above = [p for p in eligible if p["confidence"] > BASE_CONF]
    kept = [p for p in above if p["retained_baseline"]]
    if not eligible:
        return "no_iou_eligible_decoded_box"
    if not above:
        return "confidence_gate"
    if not kept:
        if not case["max_nms_reached"] and not case["max_det_reached"]:
            if all(p.get("primary_suppressor") for p in above):
                return "nms_suppression"
        return "nms_or_output_limit_unresolved"
    return "retained_assignment"


def summarize_evidence(evidence):
    """Rebuild mechanism totals from the compact, exhaustive eligible lists."""
    cases = evidence["missed_cases"]
    if not evidence["baseline_validation"]["all_twenty_prediction_arrays_exact"]:
        raise ValueError("Unverified instrumentation cannot support a diagnostic")
    if len(cases) != 11 or len({c["stable_id"] for c in cases}) != 11:
        raise ValueError("Expected the eleven distinct accepted R38 misses")
    counts = Counter()
    for case in cases:
        eligible = case["iou_eligible_proposals"]
        if len(eligible) != case["iou_eligible_count"]:
            raise ValueError("Compact eligible-proposal list is incomplete")
        if any(p["iou"] < MATCH_IOU for p in eligible):
            raise ValueError("Nonqualifying proposal in eligible list")
        if len({p["proposal_index"] for p in eligible}) != len(eligible):
            raise ValueError("Repeated raw-proposal index")
        inferred = cause(case)
        if inferred != case["mechanism"]:
            raise ValueError("Stored mechanism differs from compact evidence")
        counts[inferred] += 1
    return {
        "baseline": evidence["baseline"],
        "counterfactual": evidence["counterfactual"],
        "mechanism_counts": {name: counts[name] for name in (
            "no_iou_eligible_decoded_box", "confidence_gate",
            "nms_suppression", "nms_or_output_limit_unresolved", "retained_assignment")},
        "remaining_miss_ids": [c["stable_id"] for c in cases],
        "interpretation": (
            "A stage diagnosis for this checkpoint and accepted R38 reference; "
            "not a training result or model promotion. Raw proposals with useful "
            "geometry can still have negligible confidence and are not detections."
        ),
        "replay_limit": (
            "All IoU-eligible proposals for the eleven misses are retained. The "
            "full raw tensors are fingerprinted, not included. Completeness and "
            "zero-eligible claims require rerunning inference to independently "
            "reconstruct; a tensor hash alone cannot demonstrate absence."
        ),
    }


def replay_evidence(root, evidence_path):
    """Rescore both cached arrays against hashed R38 labels without ML packages."""
    from evaluate_pose import parse_pose_labels
    from pose_scoring import aggregate_scores, score_image, _box_iou
    from diagnose_pose_quality import aggregate_quality, quality_for_image
    root = Path(root).resolve(); evidence_path = Path(evidence_path).resolve()
    evidence = read_json(evidence_path)
    summary = summarize_evidence(evidence)
    provenance = evidence["provenance"]
    if sha(root / REVIEW) != provenance["reference_manifest_sha256"]:
        raise ValueError("R38 manifest changed")
    if sha(root / CACHE) != provenance["accepted_cache_sha256"]:
        raise ValueError("Accepted primary cache changed")
    review = read_json(root / REVIEW)
    records = {r["image_id"]: r for r in review["files"] if r["split"] == "val"}
    rowmap = root / review["row_maps"]["path"]
    if sha(rowmap) != review["row_maps"]["sha256"]:
        raise ValueError("R38 row identity map changed")
    maps = read_json(rowmap)
    counter_path = evidence_path.parent / evidence["counterfactual_predictions"]["path"]
    if sha(counter_path) != evidence["counterfactual_predictions"]["sha256"]:
        raise ValueError("Counterfactual predictions changed")
    score_by_role = {}
    predictions_by_role = {}
    for role, path in (("baseline", root / CACHE), ("counterfactual", counter_path)):
        header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 20 or {r["image_id"] for r in rows} != set(records):
            raise ValueError("Replay must include exactly twenty images")
        scores = []; qualities = []; by_image = {}; by_predictions = {}
        for row in rows:
            name = row["image_id"]; record = records[name]
            label = (root / REVIEW).parent / record["label_path"]
            if sha(label) != record["label_sha256"] or row["image_sha256"] != record["source_image_sha256"]:
                raise ValueError(f"Replay image/label identity differs: {name}")
            if row["threshold"] != evidence[role]["box_confidence"]:
                raise ValueError("Replay threshold differs")
            gold = parse_pose_labels(label.read_text(), row["width"], row["height"])
            score = score_image(gold, row["predictions"], row["width"], row["height"], MATCH_IOU, .5)
            scores.append(score)
            qualities.append(quality_for_image(gold, row["predictions"], row["width"], row["height"], [5, 10]))
            by_image[name] = score
            by_predictions[name] = row["predictions"]
        detection = aggregate_scores(scores); detection.pop("endpoint_errors_px")
        if detection != evidence[role]["detection"] or aggregate_quality(qualities) != evidence[role]["corner_quality"]:
            raise ValueError(f"Replayed {role} scores differ")
        score_by_role[role] = by_image
        predictions_by_role[role] = by_predictions
    baseline_misses = set(); probe_misses = set()
    for name in records:
        inverse = {r["current_row"]: r for r in maps[name]["rows"]}
        for role, dest in (("baseline", baseline_misses), ("counterfactual", probe_misses)):
            dest.update(inverse[i+1]["stable_id"] for i in score_by_role[role][name]["unmatched_gold_indices"])
    if baseline_misses != {c["stable_id"] for c in evidence["missed_cases"]}:
        raise ValueError("Trace does not cover exactly the primary misses")
    probe = evidence["counterfactual"]
    if baseline_misses - probe_misses != {r["stable_id"] for r in probe["recovered"]}:
        raise ValueError("Recovered identities differ")
    if probe_misses - baseline_misses != {r["stable_id"] for r in probe["new_misses"]}:
        raise ValueError("Newly missed identities differ")
    if probe_misses != {r["stable_id"] for r in probe["remaining_misses"]}:
        raise ValueError("Remaining miss identities differ")
    captured_images = {row["image_id"]: row for row in evidence["per_image"]}
    if set(captured_images) != set(records):
        raise ValueError("Captured proposal trace image identities differ")
    for case in evidence["missed_cases"]:
        captured = captured_images[case["image_id"]]
        kept = captured["baseline_retained_proposal_indices"]
        input_boxes = captured["baseline_retained_input_boxes_xyxy"]
        if len(kept) != len(input_boxes) or len(kept) != len(predictions_by_role["baseline"][case["image_id"]]):
            raise ValueError("Captured primary input-box inventory differs")
        for proposal in case["iou_eligible_proposals"]:
            overlap = _box_iou(proposal["bbox_xyxy"], case["gold"]["bbox_xyxy"])
            if overlap != proposal["iou"]:
                raise ValueError("Raw eligible-proposal geometry differs")
            for role, key in (("baseline", "baseline_prediction_id"), ("counterfactual", "probe_prediction_id")):
                pid = proposal[key]
                if pid is not None:
                    pred = predictions_by_role[role][case["image_id"]][pid - 1]
                    if pred["bbox_xyxy"] != proposal["bbox_xyxy"] or pred["confidence"] != proposal["confidence"]:
                        raise ValueError("Raw-to-output identity differs")
            suppressor = proposal.get("primary_suppressor")
            if suppressor is not None:
                pid = suppressor["baseline_prediction_id"]
                pred = predictions_by_role["baseline"][case["image_id"]][pid - 1]
                if (proposal["proposal_index"] in kept or proposal["confidence"] <= BASE_CONF
                        or kept[pid - 1] != suppressor["proposal_index"]
                        or input_boxes[pid - 1] != suppressor["input_bbox_xyxy"]
                        or pred["bbox_xyxy"] != suppressor["bbox_xyxy"]
                        or pred["confidence"] != suppressor["confidence"]
                        or suppressor["confidence"] < proposal["confidence"]):
                    raise ValueError("NMS suppressor does not identify a preceding retained primary output")
                pair_iou = _box_iou(suppressor["suppressed_input_bbox_xyxy"], suppressor["input_bbox_xyxy"])
                if pair_iou != suppressor["iou_between_input_boxes"] or pair_iou <= .70:
                    raise ValueError("NMS suppression lacks above-threshold input-space overlap")
                if _box_iou(suppressor["bbox_xyxy"], case["gold"]["bbox_xyxy"]) != suppressor["suppressor_iou_with_reference"]:
                    raise ValueError("Suppressor-to-reference geometry differs")
                if case["max_nms_reached"] or case["max_det_reached"]:
                    raise ValueError("NMS-only diagnosis cannot ignore a reached output cap")
    return summary


def run(root, image_root, weights, output):
    from evaluate_pose import INFERENCE, _prediction_records, parse_pose_labels
    from pose_checkpoint import check_loaded_pose_model
    from pose_scoring import _box_iou, aggregate_scores, score_image
    from diagnose_pose_quality import aggregate_quality, quality_for_image

    if importlib.metadata.version("ultralytics") != "8.4.182":
        raise ValueError("Use pinned ultralytics==8.4.182")
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.utils import nms, ops

    if torch.__version__ != "2.10.0+cpu":
        raise ValueError("Use pinned torch==2.10.0+cpu for exact baseline comparison")
    comparison = read_json(root / COMPARISON)
    previous = comparison["references"]["r38"]["models"]["candidate"]
    if sha(weights) != comparison["models"]["candidate"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from returned PR123 candidate")
    if sha(root / CACHE) != comparison["models"]["candidate"]["predictions_sha256"]:
        raise ValueError("Accepted prediction cache changed")
    header, *cache_rows = [json.loads(line) for line in (root / CACHE).read_text().splitlines()]
    cache = {r["image_id"]: r for r in cache_rows}
    if header["inference"] != INFERENCE or header["box_confidences"] != [BASE_CONF]:
        raise ValueError("Accepted inference settings differ")
    if sha(root / REVIEW) != comparison["references"]["r38"]["manifest_sha256"]:
        raise ValueError("Accepted R38 reference changed")
    review = read_json(root / REVIEW)
    rowmap_path = root / review["row_maps"]["path"]
    if sha(rowmap_path) != review["row_maps"]["sha256"]:
        raise ValueError("R38 row lineage changed")
    rowmaps = read_json(rowmap_path)
    records = sorted((f for f in review["files"] if f["split"] == "val"), key=lambda f: f["image_id"])
    if len(records) != 20 or set(cache) != {r["image_id"] for r in records}:
        raise ValueError("Expected the same twenty development photographs")

    class TracePredictor(PosePredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            self.trace = {
                "raw": raw.detach().clone(),
                "input_shape": list(img.shape),
                "originals": orig_imgs,
                "image_tensor": img,
            }
            return super().postprocess(preds, img, orig_imgs, **kwargs)

    torch.set_num_threads(4)
    model = YOLO(str(weights))
    check_loaded_pose_model(model)
    model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), predictor=TracePredictor,
                  device="cpu", conf=.25, **INFERENCE)
    predictor = model.predictor
    if predictor.model.end2end or predictor.model.fp16:
        raise ValueError("Expected unchanged one-to-many head and CPU FP32")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    runtime = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "ultralytics": importlib.metadata.version("ultralytics"),
        "torchvision": importlib.metadata.version("torchvision"), "threads": torch.get_num_threads(),
        "device": str(predictor.model.device), "backend_end2end": bool(predictor.model.end2end),
        "backend_fp16": bool(predictor.model.fp16),
    }
    baseline_scores = []; probe_scores = []; baseline_quality = []; probe_quality = []
    missed_cases = []; image_evidence = []; probe_rows = []; baseline_exact = []; probe_misses = []
    recovered = []; new_misses = []; groups = {str(k): {"gold": 0, "baseline_missed": 0, "probe_missed": 0} for k in range(3)}
    for record in records:
        name = record["image_id"]
        stored = cache[name]
        width, height = stored["width"], stored["height"]
        image = image_root / "val" / f"{name}.png"
        label = (root / REVIEW).parent / record["label_path"]
        if sha(image) != record["source_image_sha256"] or sha(label) != record["label_sha256"]:
            raise ValueError(f"Image or accepted label changed: {name}")
        gold = parse_pose_labels(label.read_text(), width, height)
        result = model.predict(source=str(image), device="cpu", conf=BASE_CONF, **INFERENCE)[0]
        predictions = _prediction_records(result)
        if predictions != stored["predictions"]:
            raise ValueError(f"Instrumented output does not exactly reproduce accepted cache: {name}")
        baseline_exact.append(name)
        trace = predictor.trace
        raw = trace["raw"]
        if tuple(raw.shape[:2]) != (1, 11):
            raise ValueError(f"Unexpected decoded one-class/two-corner shape: {tuple(raw.shape)}")
        raw_hash = hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()
        original_shape = result.orig_img.shape
        input_shape = trace["input_shape"][2:]
        input_boxes = ops.xywh2xyxy(raw[0, :4].T.clone())
        input_box_values = input_boxes.cpu().tolist()
        boxes = ops.scale_boxes(input_shape, input_boxes.clone(), original_shape)
        confidences = raw[0, 4].cpu().tolist()
        box_values = boxes.cpu().tolist()
        def select(threshold):
            detections, indices = nms.non_max_suppression(
                raw.clone(), conf_thres=threshold, iou_thres=.70,
                classes=None, agnostic=False, max_det=300, nc=1,
                end2end=False, rotated=False, return_idxs=True)
            constructed = predictor.construct_results(
                [d.clone() for d in detections], trace["image_tensor"], trace["originals"])[0]
            return _prediction_records(constructed), [int(i) for i in indices[0].tolist()]
        baseline_again, keep = select(BASE_CONF)
        probe, probe_keep = select(PROBE_CONF)
        if baseline_again != predictions:
            raise ValueError(f"return_idxs reconstruction altered prediction arrays: {name}")
        if any(i not in probe_keep for i in keep):
            raise ValueError(f"Low-confidence probe removed a primary output: {name}")
        baseline_by_raw = dict(zip(keep, predictions)); probe_by_raw = dict(zip(probe_keep, probe))
        if any(probe_by_raw[i] != p for i, p in baseline_by_raw.items()):
            raise ValueError(f"Low-confidence probe changed an existing output: {name}")
        baseline_score = score_image(gold, predictions, width, height, MATCH_IOU, .5)
        probe_score = score_image(gold, probe, width, height, MATCH_IOU, .5)
        baseline_scores.append(baseline_score); probe_scores.append(probe_score)
        baseline_quality.append(quality_for_image(gold, predictions, width, height, [5, 10]))
        probe_quality.append(quality_for_image(gold, probe, width, height, [5, 10]))
        original_rows = {r["current_row"]: r for r in rowmaps[name]["rows"]}
        old_misses = set(baseline_score["unmatched_gold_indices"])
        low_misses = set(probe_score["unmatched_gold_indices"])
        for i, target in enumerate(gold):
            n = str(sum(p[2] == 2 for p in target["keypoints"]))
            groups[n]["gold"] += 1
            groups[n]["baseline_missed"] += int(i in old_misses)
            groups[n]["probe_missed"] += int(i in low_misses)
        def identity(i):
            mapped = original_rows[i + 1]
            return {"image_id": name, "r38_row": i + 1, "r37_row": mapped["base_row"], "stable_id": mapped["stable_id"]}
        recovered.extend(identity(i) for i in sorted(old_misses - low_misses))
        new_misses.extend(identity(i) for i in sorted(low_misses - old_misses))
        probe_misses.extend(identity(i) for i in sorted(low_misses))
        above_count = sum(c > BASE_CONF for c in confidences)
        keep_set = set(keep); low_keep_set = set(probe_keep)
        for i in sorted(old_misses):
            target = gold[i]
            overlaps = [_box_iou(b, target["bbox_xyxy"]) for b in box_values]
            def proposal(j):
                value = {
                    "proposal_index": j, "bbox_xyxy": box_values[j], "iou": overlaps[j],
                    "confidence": confidences[j], "retained_baseline": j in keep_set,
                    "retained_probe": j in low_keep_set,
                    "baseline_prediction_id": keep.index(j) + 1 if j in keep_set else None,
                    "probe_prediction_id": probe_keep.index(j) + 1 if j in low_keep_set else None,
                }
                if confidences[j] > BASE_CONF and j not in keep_set and above_count <= 30000 and len(keep) < 300:
                    suppressors = [k for k in keep if confidences[k] >= confidences[j]
                                   and _box_iou(input_box_values[j], input_box_values[k]) > .70]
                    if suppressors:
                        k = suppressors[0]
                        value["primary_suppressor"] = {
                            "proposal_index": k, "baseline_prediction_id": keep.index(k) + 1,
                            "confidence": confidences[k], "bbox_xyxy": box_values[k],
                            "input_bbox_xyxy": input_box_values[k], "suppressed_input_bbox_xyxy": input_box_values[j],
                            "iou_between_input_boxes": _box_iou(input_box_values[j], input_box_values[k]),
                            "suppressor_iou_with_reference": overlaps[k],
                        }
                return value
            eligible = [j for j, overlap in enumerate(overlaps) if overlap >= MATCH_IOU]
            best = max(range(len(overlaps)), key=lambda j: (overlaps[j], confidences[j], -j))
            kept_best = max(keep, key=lambda j: (overlaps[j], confidences[j], -j)) if keep else None
            case = {
                **identity(i), "gold": target,
                "locatable_corners": sum(p[2] == 2 for p in target["keypoints"]),
                "decoded_proposal_count": len(overlaps), "decoded_tensor_sha256": raw_hash,
                "best_iou_proposal": proposal(best),
                "best_retained_proposal": proposal(kept_best) if kept_best is not None else None,
                "iou_eligible_count": len(eligible),
                "iou_eligible_proposals": [proposal(j) for j in sorted(eligible, key=lambda j: (-confidences[j], j))],
                "above_confidence_proposal_count": above_count,
                "max_nms_reached": above_count > 30000,
                "max_det_reached": len(keep) == 300,
                "probe_recovers": i not in low_misses,
            }
            case["mechanism"] = cause(case)
            missed_cases.append(case)
        image_evidence.append({
            "image_id": name, "image_sha256": sha(image), "reference_label_sha256": sha(label),
            "input_shape": trace["input_shape"], "raw_shape": list(raw.shape), "raw_dtype": str(raw.dtype),
            "decoded_tensor_sha256": raw_hash, "decoded_proposal_count": len(confidences),
            "confidence_above_primary": above_count,
            "confidence_above_probe": sum(c > PROBE_CONF for c in confidences),
            "baseline_retained_proposal_indices": keep, "probe_retained_proposal_indices": probe_keep,
            "baseline_retained_input_boxes_xyxy": [input_box_values[i] for i in keep],
            "baseline_prediction_count": len(predictions), "probe_prediction_count": len(probe),
            "baseline_arrays_exact": True, "primary_arrays_preserved_in_probe": True,
            "baseline_misses": len(old_misses), "probe_misses": len(low_misses),
            "empty_scene": not gold,
        })
        probe_rows.append({"record_type": "prediction", "image_id": name, "threshold": PROBE_CONF,
                           "image_sha256": sha(image), "width": width, "height": height,
                           "predictions": probe, "decoded_proposal_indices": probe_keep})
        print(f"{name}: exact baseline {len(predictions)}; probe {len(probe)}; misses {len(old_misses)} -> {len(low_misses)}", flush=True)
    baseline = aggregate_scores(baseline_scores); baseline.pop("endpoint_errors_px")
    counterfactual = aggregate_scores(probe_scores); counterfactual.pop("endpoint_errors_px")
    if baseline != previous["detection"]:
        raise ValueError("Baseline R38 aggregate differs despite exact prediction arrays")
    baseline = {"box_confidence": BASE_CONF, "nms_iou": .7, "detection": baseline,
                "corner_quality": aggregate_quality(baseline_quality)}
    counterfactual = {"box_confidence": PROBE_CONF, "nms_iou": .7, "detection": counterfactual,
                      "corner_quality": aggregate_quality(probe_quality),
                      "recovered": recovered, "new_misses": new_misses, "remaining_misses": probe_misses,
                      "by_locatable_corners": groups,
                      "scope": "One fixed exploratory confidence counterfactual applied to the same decoded tensors; unchanged weights and NMS. Not selected for deployment."}
    evidence = {
        "schema_version": 1, "kind": "r39-pr123-decoded-proposal-diagnosis", "training": False,
        "automatic_model_promotion": False, "runtime": runtime,
        "provenance": {"checkpoint_sha256": sha(weights), "reference_manifest": REVIEW,
                       "reference_manifest_sha256": sha(root / REVIEW), "accepted_cache": CACHE,
                       "accepted_cache_sha256": sha(root / CACHE), "comparison_sha256": sha(root / COMPARISON),
                       "script_sha256": sha(__file__),
                       "ultralytics_nms_sha256": sha(nms.__file__),
                       "torchvision_imported_at_nms": "torchvision" in sys.modules},
        "settings": {**INFERENCE, "primary_confidence": BASE_CONF, "probe_confidence": PROBE_CONF,
                     "matching_iou": MATCH_IOU, "endpoint_confidence": .5, "max_nms": 30000,
                     "confidence_comparison": "strictly greater than", "proposal_indices": "zero-based decoded anchor indices; prediction IDs are one-based within each setting"},
        "baseline_validation": {"all_twenty_prediction_arrays_exact": len(baseline_exact) == 20,
                                "exact_image_ids": baseline_exact,
                                "ordinary_nms_and_return_idxs_outputs_exact": True,
                                "primary_arrays_preserved_in_probe": True},
        "baseline": baseline, "counterfactual": counterfactual, "per_image": image_evidence,
        "missed_cases": missed_cases,
    }
    probe_header = {"record_type": "run", "schema_version": 1,
                    "kind": "same-decoded-tensor-confidence-counterfactual", "model_sha256": sha(weights),
                    "inference": INFERENCE, "box_confidences": [PROBE_CONF], "runtime": runtime,
                    "reference_manifest_sha256": sha(root / REVIEW)}
    predictions_path = output / "counterfactual-predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in [probe_header, *probe_rows]))
    evidence["counterfactual_predictions"] = {"path": "counterfactual-predictions.jsonl", "sha256": sha(predictions_path)}
    write_json(output / "diagnosis.json", evidence)
    summary = summarize_evidence(evidence)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summarize", type=Path, help="Read saved diagnosis.json with standard-library Python only")
    parser.add_argument("--replay", type=Path, help="Rescore saved predictions/trace with standard-library Python only")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay_evidence(args.root, args.replay), indent=2))
        return
    if args.summarize:
        print(json.dumps(summarize_evidence(read_json(args.summarize)), indent=2))
        return
    if args.weights is None:
        parser.error("--weights is required for inference")
    root = args.root.resolve()
    run(root, (args.image_root or root / "data/images").resolve(), args.weights.resolve(),
        (args.output or root / OUT).resolve())


if __name__ == "__main__":
    main()
