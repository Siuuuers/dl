"""Prespecified single-class AP50 beside the fixed-confidence detection counts."""

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from pose_scoring import score_image


CONFIDENCE_FLOOR = 0.001
MAX_DETECTIONS_PER_IMAGE = 300
MATCHING_IOU = 0.50
RECALL_POINTS = 101
METRIC_NAME = (
    "Cassette-front AP50 (101 recall points; confidence > 0.001; "
    "max 300 detections/image; postclip NMS)"
)


def score_detection_ranking(reference_by_image, predictions_by_image, *, width=1280, height=720):
    """Score native records already filtered and suppressed by the caller.

    Image dictionaries must have identical keys, including empty scenes. All
    reference objects enter the recall denominator. The supplied prediction-list
    position is its original index for both image matching and global score ties.
    AP is undefined (None) if the complete reference contains no target objects.
    This single-IoU, single-class metric is not COCO mAP and selects no threshold.
    """
    if not isinstance(reference_by_image, Mapping) or not isinstance(predictions_by_image, Mapping):
        raise ValueError("reference and predictions must be image-ID mappings")
    for values in (reference_by_image, predictions_by_image):
        if any(not isinstance(image_id, str) or not image_id for image_id in values):
            raise ValueError("image IDs must be nonempty strings")
    if set(reference_by_image) != set(predictions_by_image):
        raise ValueError("reference and prediction image IDs must match, including empty scenes")
    score_image([], [], width, height, MATCHING_IOU, 0.50)
    positives = 0
    ranked = []
    for image_id in sorted(reference_by_image):
        gold, predictions = reference_by_image[image_id], predictions_by_image[image_id]
        for records in (gold, predictions):
            if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
                raise ValueError(f"records must be sequences: {image_id}")
            if any(not isinstance(record, Mapping) or record.get("class_id") != 0 for record in records):
                raise ValueError(f"expected cassette-front class 0 records: {image_id}")
        if len(predictions) > MAX_DETECTIONS_PER_IMAGE:
            raise ValueError(f"more than 300 predictions in image: {image_id}")
        for prediction in predictions:
            confidence = prediction.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, Real)
                    or not math.isfinite(confidence) or not CONFIDENCE_FLOOR < confidence <= 1):
                raise ValueError(f"prediction confidence must be finite and strictly > 0.001: {image_id}")
        scored = score_image(gold, predictions, width, height, MATCHING_IOU, 0.50)
        positives += scored["gold_count"]
        matches = {pair["prediction_index"]: pair for pair in scored["matches"]}
        duplicates = set(scored["duplicate_prediction_indices"])
        for index, prediction in enumerate(predictions):
            pair = matches.get(index)
            ranked.append({
                "image_id": image_id,
                "prediction_index": index,
                "confidence": float(prediction["confidence"]),
                "tp": int(pair is not None),
                "fp": int(pair is None),
                "duplicate_fp": index in duplicates,
                "matched_gold_index": pair["gold_index"] if pair else None,
                "matched_iou": pair["iou"] if pair else None,
            })
    ranked.sort(key=lambda row: (-row["confidence"], row["image_id"], row["prediction_index"]))
    cumulative_tp = cumulative_fp = 0
    for rank, row in enumerate(ranked, 1):
        cumulative_tp += row["tp"]
        cumulative_fp += row["fp"]
        row.update(rank=rank, cumulative_tp=cumulative_tp, cumulative_fp=cumulative_fp,
                   precision=cumulative_tp / rank, recall=cumulative_tp / positives if positives else None)

    interpolated = None
    ap50 = None
    if positives:
        precision_envelope = [row["precision"] for row in ranked]
        for index in range(len(precision_envelope) - 2, -1, -1):
            precision_envelope[index] = max(precision_envelope[index], precision_envelope[index + 1])
        interpolated = []
        index = 0
        for recall_step in range(RECALL_POINTS):
            while index < len(ranked) and 100 * ranked[index]["cumulative_tp"] < recall_step * positives:
                index += 1
            precision = precision_envelope[index] if index < len(ranked) else 0.0
            interpolated.append({"recall": recall_step / 100, "precision": precision})
        ap50 = math.fsum(point["precision"] for point in interpolated) / RECALL_POINTS

    return {
        "schema": "cassette-detection-ranking-r61-v1",
        "metric_name": METRIC_NAME,
        "status": "defined" if positives else "undefined_no_reference_targets",
        "ap50": ap50,
        "image_count": len(reference_by_image),
        "positive_count": positives,
        "prediction_count": len(ranked),
        "tp": cumulative_tp,
        "fp": cumulative_fp,
        "fn": positives - cumulative_tp,
        "maximum_recall": cumulative_tp / positives if positives else None,
        "protocol": {
            "matching_iou": MATCHING_IOU,
            "confidence_floor_exclusive": CONFIDENCE_FLOOR,
            "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
            "postclip_nms_applied_by_caller": True,
            "matching": "pose_scoring.score_image score-ordered one-to-one greatest-IoU matching",
            "global_tie_break": ["image_id", "original_prediction_index"],
            "original_prediction_index": "zero-based position in the caller-supplied image prediction list",
            "interpolation": "maximum precision at recall >= each of 101 equally spaced levels from 0 to 1",
            "unattained_recall_precision": 0.0,
            "scope": "Single-class AP at IoU 0.50 over supplied low-floor outputs; not COCO mAP; no threshold selection.",
        },
        "curve": ranked,
        "interpolated_precision": interpolated,
    }
