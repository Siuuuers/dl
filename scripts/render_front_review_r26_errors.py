#!/usr/bin/env python3
"""Render matched PR35 geometry errors against a specified reviewed label set.

The scorer is reused unchanged. This report does not run inference, amend labels,
evaluate unlocatable reference corners, or select missed/unmatched objects.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from evaluate_pose import parse_pose_labels
from pose_scoring import score_image


PR35_SHA256 = "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1"
BOX_CONFIDENCE = 0.13
POINT_CONFIDENCE = 0.50
MATCHING_IOU = 0.50
CORNER_ERROR_PX = 5.0
BOX_ERROR_IOU = 0.70
CORNER_LIMIT = 16
BOX_LIMIT = 8
SIDES = ("left", "right")
INK = colors.HexColor("#172C36")
MUTED = colors.HexColor("#52636C")
RULE = colors.HexColor("#DCE3E5")
PALE = colors.HexColor("#F2F5F5")
GREEN = colors.HexColor("#087B54")
ORANGE = colors.HexColor("#DF6600")
WIDTH, HEIGHT = A4
MARGIN = 36


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def describe_path(path, root):
    path = Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def source_path(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def available(point, width, height):
    return (point[2] >= POINT_CONFIDENCE
            and all(value is not None and math.isfinite(value) for value in point[:2])
            and 0 <= point[0] < width and 0 <= point[1] < height)


def point_record(reference, prediction, width, height):
    locatable = reference[2] == 2
    returned = available(prediction, width, height)
    error = (math.hypot(prediction[0] - reference[0], prediction[1] - reference[1])
             if locatable and returned else None)
    return {
        "reference_visibility": reference[2],
        "reference_xy_px": reference[:2] if locatable else None,
        "prediction_xy_px": prediction[:2],
        "prediction_confidence": prediction[2],
        "returned_at_fixed_threshold": returned,
        "error_px": error,
        "status": ("measured" if error is not None else
                   "reference_unlocatable" if not locatable else "prediction_not_returned"),
    }


def matched_records(image_id, gold, predictions, width, height, row_map=None):
    """Only scorer matches enter the inventory; unmatched rows never get records."""
    score = score_image(gold, predictions, width, height, MATCHING_IOU, POINT_CONFIDENCE)
    result = []
    for pair in score["matches"]:
        row = pair["gold_index"] + 1
        reference, prediction = gold[pair["gold_index"]], predictions[pair["prediction_index"]]
        points = {side: point_record(ref, pred, width, height) for side, ref, pred in
                  zip(SIDES, reference["keypoints"], prediction["keypoints"])}
        measured = [point["error_px"] for point in points.values() if point["error_px"] is not None]
        result.append({
            "identity": f"{image_id}:{row}", "image_id": image_id, "current_row": row,
            "matched": True, "prediction_index_0based": pair["prediction_index"],
            "box_iou": pair["iou"], "box_confidence": prediction["confidence"],
            "reference_box_xyxy_px": reference["bbox_xyxy"],
            "prediction_box_xyxy_px": prediction["bbox_xyxy"], "corners": points,
            "maximum_measured_corner_error_px": max(measured) if measured else None,
            "measured_corner_count": len(measured),
            "row_lineage": (row_map or {}).get(row),
            "selected_for_pdf": False, "selection_category": None, "selection_rank": None,
        })
    result.sort(key=lambda item: item["current_row"])
    require(len(result) == score["tp"], "Matched inventory count differs from scorer")
    for item in result:
        require(item["box_iou"] >= MATCHING_IOU, "Nonmatch entered inventory")
        require(item["current_row"] - 1 not in score["unmatched_gold_indices"], "Miss entered inventory")
        require(item["prediction_index_0based"] not in score["unmatched_prediction_indices"],
                "Unmatched output entered inventory")
    return result, score


def select_cases(records):
    """Rank measurable returned-corner error, then distinct low-overlap matches."""
    corner_pool = [item for item in records if item["maximum_measured_corner_error_px"] is not None
                   and item["maximum_measured_corner_error_px"] > CORNER_ERROR_PX]
    corner_pool.sort(key=lambda item: (-item["maximum_measured_corner_error_px"],
                                      item["image_id"], item["current_row"]))
    corner_cases = corner_pool[:CORNER_LIMIT]
    used = {item["identity"] for item in corner_cases}
    box_pool = [item for item in records if item["box_iou"] < BOX_ERROR_IOU and item["identity"] not in used]
    box_pool.sort(key=lambda item: (item["box_iou"], item["image_id"], item["current_row"]))
    box_cases = box_pool[:BOX_LIMIT]
    for category, cases in (("returned_corner_error", corner_cases), ("matched_box_overlap", box_cases)):
        for rank, item in enumerate(cases, 1):
            item.update(selected_for_pdf=True, selection_category=category, selection_rank=rank)
    selected = corner_cases + box_cases
    require(len({item["identity"] for item in selected}) == len(selected), "Duplicate PDF case")
    require(len(selected) <= 24 and all(item["matched"] for item in selected), "Selection scope changed")
    return selected, {"corner_candidates_over_5px": len(corner_pool),
                      "distinct_box_candidates_below_0_70": len(box_pool),
                      "selected_corner_cases": len(corner_cases), "selected_box_cases": len(box_cases)}


def verify_selection_contract():
    """Small adversarial check for the consequential exclusions and ordering."""
    def ref(box, left=(10, 20, 2), right=(30, 20, 2)):
        return {"class_id": 0, "bbox_xyxy": list(box), "keypoints": [list(left), list(right)]}
    def pred(box, left=(25, 20, .8), right=(30, 20, .8), confidence=.9):
        return {**ref(box, left, right), "confidence": confidence}
    gold = [ref((10, 10, 30, 30)), ref((60, 10, 80, 30)),
            ref((10, 60, 30, 80), left=(0, 0, 0), right=(30, 75, 2))]
    predictions = [pred((10, 10, 30, 30)), pred((150, 150, 170, 170)),
                   pred((10, 60, 30, 80), left=(90, 90, .99), right=(90, 90, .49))]
    records, score = matched_records("synthetic", gold, predictions, 200, 200)
    require({item["current_row"] for item in records} == {1, 3}, "Miss-exclusion contract failed")
    require(score["unmatched_prediction_indices"] == [1], "Synthetic unmatched-output setup failed")
    require(records[1]["maximum_measured_corner_error_px"] is None,
            "Unlocatable or below-threshold corner fabricated a measured error")
    selected, _ = select_cases(records)
    require([item["identity"] for item in selected] == ["synthetic:1"], "Corner eligibility contract failed")
    ranks = []
    for number in range(30):
        ranks.append({"identity": f"order:{number}", "image_id": "order", "current_row": number,
                      "matched": True, "maximum_measured_corner_error_px": 6 + number,
                      "box_iou": .50 + number / 1000})
    chosen, _ = select_cases(ranks)
    require([item["current_row"] for item in chosen[:16]] == list(range(29, 13, -1)),
            "Descending corner-error ranking failed")
    require([item["current_row"] for item in chosen[16:]] == list(range(8)),
            "Distinct ascending-IoU ranking failed")
    return {"miss_exclusion": True, "unmatched_output_exclusion": True,
            "unlocatable_corner_exclusion": True, "point_threshold_enforced": True,
            "corner_sort_descending": True, "box_sort_ascending": True,
            "selection_disjoint_and_capped": True}


def gather(args):
    root = args.root.resolve()
    manifest_path = source_path(root, args.review_manifest)
    predictions_path = source_path(root, args.predictions)
    manifest = read(manifest_path)
    lines = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    runs = [line for line in lines if line.get("record_type") == "run"]
    require(len(runs) == 1, "Expected one inference run record")
    run = runs[0]
    require(run["model_sha256"] == PR35_SHA256, "The supplied predictions are not retained PR35")
    prediction_records = [line for line in lines if line.get("record_type") == "prediction"
                          and line.get("threshold") == BOX_CONFIDENCE]
    predictions = {line["image_id"]: line for line in prediction_records}
    require(len(predictions) == len(prediction_records), "Duplicate image/threshold prediction record")
    file_records = [item for item in manifest["files"] if item["split"] == "val"]
    require(set(predictions) == {item["image_id"] for item in file_records},
            "Prediction and review development image sets differ")
    row_maps = {}
    row_map_source = None
    if manifest.get("row_maps"):
        row_map_path = source_path(root, manifest["row_maps"]["path"])
        require(sha(row_map_path) == manifest["row_maps"]["sha256"], "Row-map hash differs")
        row_maps = read(row_map_path)
        row_map_source = {"path": describe_path(row_map_path, root), "sha256": sha(row_map_path)}
    images, records, sources = {}, [], []
    excluded_gold = excluded_predictions = 0
    for spec in sorted(file_records, key=lambda item: item["image_id"]):
        image_id = spec["image_id"]
        prediction = predictions[image_id]
        image_path = source_path(root, args.image_root) / (image_id + ".png")
        label_path = manifest_path.parent / spec["label_path"]
        image_hash, label_hash = sha(image_path), sha(label_path)
        require(image_hash == spec["source_image_sha256"] == prediction["image_sha256"],
                "Original image identity differs: " + image_id)
        require(label_hash == spec["label_sha256"], "Review label hash differs: " + image_id)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        require((width, height) == (prediction["width"], prediction["height"]), "Image dimensions differ")
        raw_text = label_path.read_text()
        require(all(line.strip() for line in raw_text.splitlines()), "Blank lines would make row IDs ambiguous")
        gold = parse_pose_labels(raw_text, width, height, str(label_path))
        require(len(gold) == spec["objects"], "Manifest object count differs")
        raw_predictions = prediction["predictions"]
        require(all(item["confidence"] >= BOX_CONFIDENCE for item in raw_predictions),
                "A saved prediction is below the fixed box confidence")
        lineage = {item["current_row"]: item for item in row_maps.get(image_id, {}).get("rows", [])}
        matched, score = matched_records(image_id, gold, raw_predictions, width, height, lineage)
        records.extend(matched)
        excluded_gold += len(score["unmatched_gold_indices"])
        excluded_predictions += len(score["unmatched_prediction_indices"])
        images[image_id] = image
        sources.append({"image_id": image_id, "image_path": describe_path(image_path, root),
                        "image_sha256": image_hash, "label_path": describe_path(label_path, root),
                        "label_sha256": label_hash, "width": width, "height": height,
                        "reference_objects": len(gold), "matched_objects": len(matched)})
    selected, counts = select_cases(records)
    checks = verify_selection_contract()
    require(sum(item["matched_objects"] for item in sources) == len(records), "Inventory total differs")
    inventory = {
        "schema_version": 1, "kind": "retained_pr35_matched_prediction_geometry_review",
        "review_id": manifest["review_id"],
        "review_manifest": {"path": describe_path(manifest_path, root), "sha256": sha(manifest_path)},
        "row_maps": row_map_source,
        "prediction_source": {"path": describe_path(predictions_path, root), "sha256": sha(predictions_path),
                              "model_sha256": run["model_sha256"],
                              "inference_created_at_utc": run.get("created_at_utc"),
                              "source_revision": run.get("source", {}).get("revision"),
                              "inference": run.get("inference")},
        "thresholds": {"box_confidence": BOX_CONFIDENCE, "point_confidence": POINT_CONFIDENCE,
                       "matching_iou": MATCHING_IOU},
        "matching": "Unchanged pose_scoring.score_image: descending prediction confidence; best remaining same-class IoU; deterministic index ties.",
        "selection": {"first": "Up to16 matched targets with maximum returned, locatable-corner error >5px, descending error.",
                      "then": "Up to8 other matched targets with box IoU <0.70, ascending IoU.",
                      "ties": "image_id then current one-based row", "maximum_cases": 24,
                      "no_unmatched_objects_or_outputs": True, "no_inferred_unlocatable_reference_corners": True},
        "scope": "Development diagnostics against current reviewed labels, not an independent test or a model replacement decision. All distances are original-image Euclidean pixels. The PDF rounds numeric displays; this inventory retains full precision.",
        "summary": {"images": len(sources), "matched_targets": len(records),
                    "matched_targets_with_measured_corners": sum(item["measured_corner_count"] > 0 for item in records),
                    **counts, "selected_total": len(selected),
                    "unmatched_reference_targets_excluded": excluded_gold,
                    "unmatched_outputs_excluded": excluded_predictions},
        "verification": {**checks, "images_and_review_labels_hash_verified": True,
                         "every_inventory_entry_is_a_scorer_match": True,
                         "scorer_sha256": sha(root / "scripts/pose_scoring.py"),
                         "renderer_sha256": sha(Path(__file__))},
        "source_images": sources, "selected_identities": [item["identity"] for item in selected],
        "matched_targets": records,
    }
    return inventory, images, selected


def crop_bounds(item, size):
    width, height = size
    ref, pred = item["reference_box_xyxy_px"], item["prediction_box_xyxy_px"]
    xs, ys = [ref[0], ref[2], pred[0], pred[2]], [ref[1], ref[3], pred[1], pred[3]]
    for point in item["corners"].values():
        if point["reference_xy_px"] is not None:
            xs.append(point["reference_xy_px"][0]); ys.append(point["reference_xy_px"][1])
        if point["error_px"] is not None:
            xs.append(point["prediction_xy_px"][0]); ys.append(point["prediction_xy_px"][1])
    box_width, box_height = max(xs) - min(xs), max(ys) - min(ys)
    mx = max(14, min(32, box_width * .18))
    my = max(22, min(48, box_height * .85))
    left, right = max(0, math.floor(min(xs) - mx)), min(width, math.ceil(max(xs) + mx))
    top, bottom = max(0, math.floor(min(ys) - my)), min(height, math.ceil(max(ys) + my))
    require(right > left and bottom > top, "Invalid context crop")
    return [left, top, right, bottom]


def register_fonts():
    regular = Path("/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf")
    bold = Path("/usr/share/fonts/truetype/open-sans/OpenSans-Semibold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Review", str(regular)))
        pdfmetrics.registerFont(TTFont("ReviewBold", str(bold)))
    else:
        pdfmetrics.registerFont(pdfmetrics.Font("Review", "Helvetica", "WinAnsiEncoding"))
        pdfmetrics.registerFont(pdfmetrics.Font("ReviewBold", "Helvetica-Bold", "WinAnsiEncoding"))


def text(pdf, value, x, y, size=9, color=INK, bold=False):
    pdf.setFillColor(color)
    pdf.setFont("ReviewBold" if bold else "Review", size)
    pdf.drawString(x, y, value)


def paragraph(pdf, value, x, top, width, size=10, leading=15, color=INK):
    style = ParagraphStyle("body", fontName="Review", fontSize=size, leading=leading, textColor=color)
    block = Paragraph(value, style)
    _, height = block.wrap(width, HEIGHT)
    block.drawOn(pdf, x, top - height)
    return top - height


def footer(pdf, page, pages, version):
    pdf.setStrokeColor(RULE); pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH - MARGIN, 31)
    text(pdf, f"RETAINED PR35  /  {version} DEVELOPMENT REVIEW", MARGIN, 18, 7, MUTED)
    pdf.setFont("Review", 7); pdf.setFillColor(MUTED)
    pdf.drawRightString(WIDTH - MARGIN, 18, f"{page} / {pages}")


def symbol(pdf, x, y, prediction=False, radius=3.4):
    pdf.setStrokeColor(ORANGE if prediction else GREEN)
    pdf.setLineWidth(1.35)
    if prediction:
        pdf.line(x - radius, y - radius, x + radius, y + radius)
        pdf.line(x - radius, y + radius, x + radius, y - radius)
    else:
        pdf.circle(x, y, radius, stroke=1, fill=0)


def legend(pdf, x, y, compact=False):
    symbol(pdf, x + 4, y + 3)
    text(pdf, "Reviewed reference: ring + solid box", x + 15, y, 8 if compact else 9, GREEN)
    offset = 245 if compact else 0
    if not compact:
        y -= 18
    symbol(pdf, x + offset + 4, y + 3, True)
    text(pdf, "PR35 prediction: cross + dashed box", x + offset + 15, y, 8 if compact else 9, ORANGE)


def intro(pdf, inventory, total_pages, version):
    summary = inventory["summary"]
    text(pdf, "MATCHED PREDICTIONS  /  RETAINED MODEL", MARGIN, HEIGHT - 51, 9, MUTED, True)
    text(pdf, "PR35: geometry review", MARGIN, HEIGHT - 91, 27, INK, True)
    y = paragraph(pdf, f"The least accurate returned predictions, compared with the current {version} reviewed development labels. Each case shows the original context beside the saved reference and model geometry.",
                  MARGIN, HEIGHT - 115, WIDTH - 2 * MARGIN, 11, 17)
    card_y = y - 85
    card_width = (WIDTH - 2 * MARGIN - 20) / 3
    for index, (number, label) in enumerate(((summary["matched_targets"], "matched targets scored"),
                                            (summary["selected_corner_cases"], "largest corner-error cases"),
                                            (summary["selected_box_cases"], "additional box-error cases"))):
        x = MARGIN + index * (card_width + 10)
        pdf.setFillColor(PALE); pdf.roundRect(x, card_y, card_width, 63, 4, fill=1, stroke=0)
        text(pdf, str(number), x + 13, card_y + 30, 22, INK, True)
        text(pdf, label, x + 13, card_y + 13, 8, MUTED)
    y = card_y - 31
    text(pdf, "Selection, in reading order", MARGIN, y, 12, INK, True)
    y = paragraph(pdf, "1. Up to 16 matched targets with a returned, locatable-corner error greater than 5 px, ranked by the largest such error.<br/>2. Up to 8 additional matched targets with box IoU below 0.70, ranked from lowest overlap. Cases do not repeat.",
                  MARGIN, y - 14, WIDTH - 2 * MARGIN, 10, 16)
    y -= 27
    text(pdf, "Fixed measurement rules", MARGIN, y, 12, INK, True)
    y = paragraph(pdf, "Box confidence: 0.13. Corner confidence: 0.50. Matching IoU: 0.50. Matching uses the existing deterministic scorer. Errors are Euclidean distances in original-image pixels; the PDF rounds them to two decimals. Current label rows are one-based.",
                  MARGIN, y - 14, WIDTH - 2 * MARGIN, 10, 16)
    y -= 27
    text(pdf, "How to read the crops", MARGIN, y, 12, INK, True)
    legend(pdf, MARGIN + 3, y - 21)
    y = paragraph(pdf, "Both panels use the same crop and scale. Reference rings mark only locatable corners; prediction crosses mark comparable corners returned above the threshold. Thin connectors show their displacement. Neighboring cassettes provide context without extra annotation.",
                  MARGIN, y - 57, WIDTH - 2 * MARGIN, 10, 16)
    y -= 24
    text(pdf, "Scope", MARGIN, y, 12, INK, True)
    y = paragraph(pdf, "Missed objects and unmatched outputs are excluded. Missing reference corners receive no invented location or distance. This deliberately selected development report does not represent overall accuracy or an independent test, and does not establish a replacement for PR35.",
                  MARGIN, y - 14, WIDTH - 2 * MARGIN, 10, 16)
    y -= 25
    text(pdf, "Evidence", MARGIN, y, 10, INK, True)
    source = inventory["prediction_source"]
    y = paragraph(pdf, f"Reviewed labels: {inventory['review_id']}<br/>Fresh inference: {source['inference_created_at_utc']}<br/>PR35 checkpoint SHA-256: {source['model_sha256'][:16]}...<br/>The companion prediction-error-inventory.json contains every matched target, full-precision measurements, source hashes and the selection checks.",
                  MARGIN, y - 13, WIDTH - 2 * MARGIN, 8, 12, MUTED)
    require(y > 44, "Introduction overflows footer")
    footer(pdf, 1, total_pages, version)
    pdf.showPage()


def draw_crop(pdf, crop, bounds, item, x, top, panel_width, panel_height, overlay=False):
    scale = min(panel_width / crop.width, panel_height / crop.height)
    dw, dh = crop.width * scale, crop.height * scale
    ox = x + (panel_width - dw) / 2
    oy = top - dh
    pdf.drawImage(ImageReader(crop), ox, oy, dw, dh)
    pdf.setStrokeColor(RULE); pdf.setLineWidth(.5); pdf.rect(ox, oy, dw, dh, stroke=1, fill=0)
    if not overlay:
        return
    left, crop_top = bounds[:2]
    def transform(point):
        return ox + (point[0] - left) * scale, top - (point[1] - crop_top) * scale
    for box, color, dashed in ((item["reference_box_xyxy_px"], GREEN, False),
                                (item["prediction_box_xyxy_px"], ORANGE, True)):
        a, b = transform(box[:2]), transform(box[2:])
        pdf.saveState(); pdf.setStrokeColor(color); pdf.setLineWidth(.95)
        if dashed:
            pdf.setDash(3, 2)
        pdf.rect(a[0], b[1], b[0] - a[0], a[1] - b[1], stroke=1, fill=0)
        pdf.restoreState()
    for point in item["corners"].values():
        ref = point["reference_xy_px"]
        if ref is None:
            continue
        a = transform(ref)
        if point["error_px"] is not None:
            b = transform(point["prediction_xy_px"])
            pdf.setStrokeColor(colors.HexColor("#E8A563")); pdf.setLineWidth(.65)
            pdf.line(a[0], a[1], b[0], b[1])
            symbol(pdf, *b, prediction=True)
        symbol(pdf, *a)


def pair_text(values):
    return f"({values[0]:.2f}, {values[1]:.2f})"


def case_card(pdf, item, image, index, top, crop_dir):
    category = "CORNER ERROR" if item["selection_category"] == "returned_corner_error" else "BOX OVERLAP"
    text(pdf, f"{index:02d}   {item['image_id']}  /  row {item['current_row']}", MARGIN, top, 12, INK, True)
    pdf.setFont("ReviewBold", 8); pdf.setFillColor(MUTED)
    pdf.drawRightString(WIDTH - MARGIN, top + 1, f"{category}  #{item['selection_rank']:02d}")
    error = item["maximum_measured_corner_error_px"]
    metric = f"IoU {item['box_iou']:.3f}   |   Box confidence {item['box_confidence']:.3f}"
    if error is not None:
        metric += f"   |   Largest measured corner error {error:.2f} px"
    text(pdf, metric, MARGIN, top - 18, 8.5, MUTED)
    gap = 15
    panel_width = (WIDTH - 2 * MARGIN - gap) / 2
    text(pdf, "ORIGINAL / UNMARKED", MARGIN, top - 39, 8, MUTED, True)
    text(pdf, "REVIEWED REFERENCE + PR35", MARGIN + panel_width + gap, top - 39, 8, MUTED, True)
    bounds = crop_bounds(item, image.size)
    crop = image.crop(bounds)
    crop_name = f"{index:02d}-{item['image_id']}-row-{item['current_row']}-original.png"
    crop_path = crop_dir / crop_name
    crop.save(crop_path)
    require(Image.open(crop_path).convert("RGB").tobytes() == image.crop(bounds).tobytes(),
            "Unmarked crop pixels changed")
    item["crop_xyxy_px"] = bounds
    item["pdf_case_number"] = index
    item["crop_rgb_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
    image_top = top - 47
    image_height = 180
    draw_crop(pdf, crop, bounds, item, MARGIN, image_top, panel_width, image_height)
    draw_crop(pdf, crop, bounds, item, MARGIN + panel_width + gap, image_top, panel_width, image_height, True)
    y = top - 240
    pdf.setStrokeColor(RULE); pdf.setLineWidth(.5); pdf.line(MARGIN, y + 10, WIDTH - MARGIN, y + 10)
    columns = [MARGIN, MARGIN + 54, MARGIN + 188, MARGIN + 331, MARGIN + 430]
    for label, x in zip(("Corner", "Reference (x, y)", "Prediction (x, y)", "Point conf.", "Error (px)"), columns):
        text(pdf, label, x, y, 8, MUTED, True)
    for side in SIDES:
        y -= 17
        point = item["corners"][side]
        ref = pair_text(point["reference_xy_px"]) if point["reference_xy_px"] is not None else "unlocatable"
        pred = (pair_text(point["prediction_xy_px"]) if point["returned_at_fixed_threshold"] else "not returned")
        err = f"{point['error_px']:.2f}" if point["error_px"] is not None else "not scored"
        for value, x in zip((side.capitalize(), ref, pred, f"{point['prediction_confidence']:.3f}", err), columns):
            text(pdf, value, x, y, 8.2, INK)
    text(pdf, f"Crop x={bounds[0]}:{bounds[2]}, y={bounds[1]}:{bounds[3]}  |  Image {image.width} x {image.height} px",
         MARGIN, y - 18, 7, MUTED)
    return y - 18


def render(inventory, images, selected, output, crop_dir):
    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    version = inventory["review_id"].rsplit("-", 1)[-1].upper()
    pages = 1 + math.ceil(len(selected) / 2)
    pdf = canvas.Canvas(str(output), pagesize=A4, pageCompression=1, invariant=1)
    pdf.setTitle(f"Retained PR35 inaccurate matched predictions - {version}")
    pdf.setAuthor("Embedding vision annotation review")
    pdf.setSubject("Original-pixel matched-corner and bounding-box diagnostic review")
    intro(pdf, inventory, pages, version)
    for start in range(0, len(selected), 2):
        text(pdf, f"PR35 / {version} geometry review", MARGIN, HEIGHT - 35, 11, INK, True)
        legend(pdf, MARGIN, HEIGHT - 55, compact=True)
        for slot, item in enumerate(selected[start:start + 2]):
            top = HEIGHT - 89 - slot * 350
            bottom = case_card(pdf, item, images[item["image_id"]], start + slot + 1, top, crop_dir)
            require(bottom > 36, "Case card overflows footer")
            if slot == 0:
                pdf.setStrokeColor(RULE); pdf.setLineWidth(.6)
                pdf.line(MARGIN, top - 323, WIDTH - MARGIN, top - 323)
        footer(pdf, 2 + start // 2, pages, version)
        pdf.showPage()
    pdf.save()
    inventory["pdf"] = {"filename": output.name, "sha256": sha(output), "pages": pages,
                        "cases_per_page": 2, "intro_pages": 1,
                        "unmarked_crops_verified_pixel_identical": True,
                        "numeric_display": "coordinates and errors: 2 decimals; confidence and IoU: 3 decimals"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--review-manifest", type=Path,
                        default=Path("data/reviews/fronts-2026-11-23-r26/manifest.json"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("results/evaluation/pr101-r25-20261123/inference/eval-pr35/predictions.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("data/images/val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path,
                        default=Path("results/review/front-review-20261123-r26/prediction-error-inventory.json"))
    parser.add_argument("--crop-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory, images, selected = gather(args)
    render(inventory, images, selected, args.output.resolve(), args.crop_dir.resolve())
    inventory_path = source_path(args.root.resolve(), args.inventory)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pdf": str(args.output.resolve()), "inventory": str(inventory_path),
                      "pages": inventory["pdf"]["pages"], **inventory["summary"]}, indent=2))


if __name__ == "__main__":
    main()
