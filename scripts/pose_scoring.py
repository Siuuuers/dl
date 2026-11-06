"""Deterministic two-corner pose scoring in original-image pixel coordinates."""

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real


def _number(value, name, finite=True):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if finite and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value, name):
    value = _number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _sequence(value, name, length=None):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return value


def _objects(values, name, predicted):
    normalized = []
    for index, item in enumerate(_sequence(values, name)):
        prefix = f"{name}[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{prefix} must be an object")
        class_id = _integer(item.get("class_id"), f"{prefix}.class_id")
        box = tuple(
            _number(value, f"{prefix}.bbox_xyxy[{axis}]")
            for axis, value in enumerate(_sequence(item.get("bbox_xyxy"), f"{prefix}.bbox_xyxy", 4))
        )
        area = (box[2] - box[0]) * (box[3] - box[1])
        if box[2] <= box[0] or box[3] <= box[1] or not math.isfinite(area) or area <= 0:
            raise ValueError(f"{prefix}.bbox_xyxy must have finite positive area")
        points = []
        for point_index, point in enumerate(_sequence(item.get("keypoints"), f"{prefix}.keypoints", 2)):
            point_name = f"{prefix}.keypoints[{point_index}]"
            point = _sequence(point, point_name, 3)
            coordinates = tuple(
                math.nan if predicted and value is None else _number(value, f"{point_name}[{axis}]", finite=not predicted)
                for axis, value in enumerate(point[:2])
            )
            if predicted:
                status = _probability(point[2], f"{point_name}.confidence")
            else:
                status = _number(point[2], f"{point_name}.visibility")
                if status not in (0, 2):
                    raise ValueError(f"{point_name}.visibility must be 0 or 2")
            points.append((*coordinates, status))
        result = {"class_id": class_id, "bbox_xyxy": box, "keypoints": points}
        if predicted:
            result["confidence"] = _probability(item.get("confidence"), f"{prefix}.confidence")
        normalized.append(result)
    return normalized


def _box_iou(first, second):
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    scale = max(first_area, second_area)
    return (intersection / scale) / (first_area / scale + second_area / scale - intersection / scale)


def _percentile(ordered, quantile):
    if not ordered:
        return None
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metrics(gold_count, prediction_count, tp, duplicate_fp, visible_endpoints, errors):
    ordered = sorted(errors)
    returned = len(errors)
    fp = prediction_count - tp
    return {
        "gold_count": gold_count,
        "prediction_count": prediction_count,
        "tp": tp,
        "fn": gold_count - tp,
        "fp": fp,
        "duplicate_fp": duplicate_fp,
        "other_fp": fp - duplicate_fp,
        "visible_endpoints": visible_endpoints,
        "returned_endpoints": returned,
        "missing_endpoints": visible_endpoints - returned,
        "endpoint_coverage": returned / visible_endpoints if visible_endpoints else None,
        "conditional_median_error_px": _percentile(ordered, 0.50),
        "conditional_p95_error_px": _percentile(ordered, 0.95),
        "endpoint_errors_px": list(errors),
    }


def score_image(gold, predictions, width, height, matching_iou=0.5, endpoint_conf=0.5):
    """Score already box-confidence-filtered predictions; preserve original indices."""
    width = _integer(width, "width", minimum=1)
    height = _integer(height, "height", minimum=1)
    matching_iou = _probability(matching_iou, "matching_iou")
    if matching_iou == 0:
        raise ValueError("matching_iou must be greater than 0")
    endpoint_conf = _probability(endpoint_conf, "endpoint_conf")
    gold = _objects(gold, "gold", predicted=False)
    predictions = _objects(predictions, "predictions", predicted=True)
    overlaps = [[_box_iou(prediction["bbox_xyxy"], item["bbox_xyxy"]) for item in gold] for prediction in predictions]
    unmatched_gold = set(range(len(gold)))
    matched_predictions = set()
    matches = []

    for prediction_index in sorted(range(len(predictions)), key=lambda index: (-predictions[index]["confidence"], index)):
        candidates = [
            gold_index for gold_index in unmatched_gold
            if gold[gold_index]["class_id"] == predictions[prediction_index]["class_id"]
            and overlaps[prediction_index][gold_index] >= matching_iou
        ]
        if not candidates:
            continue
        gold_index = min(candidates, key=lambda index: (-overlaps[prediction_index][index], index))
        unmatched_gold.remove(gold_index)
        matched_predictions.add(prediction_index)
        matches.append({"gold_index": gold_index, "prediction_index": prediction_index, "iou": overlaps[prediction_index][gold_index]})

    unmatched_predictions = [index for index in range(len(predictions)) if index not in matched_predictions]
    matched_gold = set(range(len(gold))) - unmatched_gold
    duplicate_predictions = [
        prediction_index for prediction_index in unmatched_predictions
        if any(
            gold[gold_index]["class_id"] == predictions[prediction_index]["class_id"]
            and overlaps[prediction_index][gold_index] >= matching_iou
            for gold_index in matched_gold
        )
    ]
    duplicate_set = set(duplicate_predictions)
    errors = []
    for pair in matches:
        gold_points = gold[pair["gold_index"]]["keypoints"]
        predicted_points = predictions[pair["prediction_index"]]["keypoints"]
        for (gold_x, gold_y, visibility), (x, y, confidence) in zip(gold_points, predicted_points):
            if visibility != 2 or confidence < endpoint_conf or not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not (0 <= x < width and 0 <= y < height):
                continue
            error = math.hypot(x - gold_x, y - gold_y)
            if not math.isfinite(error):
                raise ValueError("endpoint distance exceeds the finite numeric range")
            errors.append(error)
    visible = sum(point[2] == 2 for item in gold for point in item["keypoints"])
    result = _metrics(len(gold), len(predictions), len(matches), len(duplicate_predictions), visible, errors)
    result.update(
        matches=matches,
        unmatched_gold_indices=sorted(unmatched_gold),
        unmatched_prediction_indices=unmatched_predictions,
        duplicate_prediction_indices=duplicate_predictions,
        other_fp_prediction_indices=[index for index in unmatched_predictions if index not in duplicate_set],
    )
    return result


def aggregate_scores(scores):
    """Sum image counts and pool endpoint errors; never average image quantiles."""
    scores = _sequence(scores, "scores")
    count_names = (
        "gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp",
        "visible_endpoints", "returned_endpoints", "missing_endpoints",
    )
    totals = dict.fromkeys(count_names, 0)
    errors = []
    for index, score in enumerate(scores):
        if not isinstance(score, Mapping):
            raise ValueError(f"scores[{index}] must be an object")
        counts = {name: _integer(score.get(name), f"scores[{index}].{name}") for name in count_names}
        distances = [
            _number(value, f"scores[{index}].endpoint_errors_px[{point}]")
            for point, value in enumerate(_sequence(score.get("endpoint_errors_px"), f"scores[{index}].endpoint_errors_px"))
        ]
        if any(distance < 0 for distance in distances):
            raise ValueError(f"scores[{index}].endpoint_errors_px must be nonnegative")
        if (
            counts["tp"] + counts["fn"] != counts["gold_count"]
            or counts["tp"] + counts["fp"] != counts["prediction_count"]
            or counts["duplicate_fp"] + counts["other_fp"] != counts["fp"]
            or counts["returned_endpoints"] != len(distances)
            or counts["returned_endpoints"] + counts["missing_endpoints"] != counts["visible_endpoints"]
        ):
            raise ValueError(f"scores[{index}] has inconsistent counts")
        for name, value in counts.items():
            totals[name] += value
        errors.extend(distances)
    result = _metrics(
        totals["gold_count"], totals["prediction_count"], totals["tp"],
        totals["duplicate_fp"], totals["visible_endpoints"], errors,
    )
    result["image_count"] = len(scores)
    return result
