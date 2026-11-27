#!/usr/bin/env python3
"""Render a complete R47 review of the fixed PR133 current-R38 error inventory.

This is a CPU audit of saved predictions, not training or fresh inference.
Only the PDF/JSON outputs are created; reference labels are never changed.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request
import zipfile

from audit_additional_front_pilot_r45_results import audit, REFERENCES, SOURCE, PR_HEAD, ARCHIVE_SHA
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou

PR134_HEAD = "d49a55d7c2240bbb18565e643cf01e691d39a7a3"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def best_overlap(box, objects):
    values = [(_box_iou(box, value["bbox_xyxy"]), i) for i, value in enumerate(objects)]
    if not values:
        return None, 0.0
    overlap, index = min(values, key=lambda item: (-item[0], item[1]))
    return (index if overlap > 0 else None), overlap


def corner_details(target, prediction):
    result = []
    for side, gold, pred in zip(("L", "R"), target["keypoints"], prediction["keypoints"]):
        if gold[2] != 2:
            result.append(dict(side=side, eligible=False, error_px=None, accepted=False))
            continue
        x, y, confidence = pred
        finite = x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
        accepted = finite and 0 <= x < 1280 and 0 <= y < 720 and confidence >= 0.5
        reason = "accepted" if accepted else "endpoint confidence below 0.50" if finite and confidence < 0.5 else "invalid or outside image"
        result.append(dict(side=side, eligible=True, accepted=accepted,
                           error_px=math.hypot(x - gold[0], y - gold[1]) if accepted else None,
                           confidence=confidence, reason=reason))
    return result


def collect(root, archive_path):
    evidence = audit(archive_path, root)
    with zipfile.ZipFile(archive_path) as archive:
        summary = json.loads(archive.read("evaluation/summary.json"))
        predictions = json.loads(archive.read("evaluation/predictions.json"))
    manifest_path = root / REFERENCES["r38"][0]
    gold = {item["image_id"]: parse_pose_labels((manifest_path.parent / item["label_path"]).read_text(), 1280, 720)
            for item in json.loads(manifest_path.read_text())["files"] if item.get("split", "val") == "val"}
    scores = summary["references"]["r38"]["postclip"]

    def target_case(kind, identity, row, title):
        target = gold[identity][row]
        case = dict(kind=kind, image=identity, row=row + 1, title=title,
                    target=target, visible_corners=sum(point[2] == 2 for point in target["keypoints"]), arms={})
        for role in ("reference", "candidate"):
            s = scores[role]["per_image"][identity]
            ps = predictions[role][identity]["postclip"]
            pair = next((m for m in s["matches"] if m["gold_index"] == row), None)
            index, overlap = (pair["prediction_index"], pair["iou"]) if pair else best_overlap(target["bbox_xyxy"], ps)
            pred = ps[index] if index is not None else None
            case["arms"][role] = dict(matched=pair is not None, prediction_index=index,
                                       prediction_number=index + 1 if index is not None else None,
                                       prediction=pred, iou=overlap,
                                       corners=corner_details(target, pred) if pred else [])
        return case

    misses, recovered, corners, fragile, unmatched = [], [], [], [], []
    per_image, categories = [], Counter()
    for identity in sorted(gold):
        candidate = scores["candidate"]["per_image"][identity]
        reference = scores["reference"]["per_image"][identity]
        ps = predictions["candidate"][identity]["postclip"]
        per_image.append(dict(image=identity, gold=len(gold[identity]),
                              reference={key: reference[key] for key in ("fn", "fp", "duplicate_fp", "other_fp")},
                              candidate={key: candidate[key] for key in ("fn", "fp", "duplicate_fp", "other_fp")}))
        for row in candidate["unmatched_gold_indices"]:
            title = "Newly missed" if row not in reference["unmatched_gold_indices"] else "Missed by both"
            misses.append(target_case("miss", identity, row, title))
        for row in sorted(set(reference["unmatched_gold_indices"]) - set(candidate["unmatched_gold_indices"])):
            recovered.append(target_case("recovery", identity, row, "Recovered by candidate"))
        assigned = {m["gold_index"]: m["prediction_index"] for m in candidate["matches"]}
        reference_assigned = {m["gold_index"]: m["prediction_index"] for m in reference["matches"]}
        for pair in candidate["matches"]:
            row, index = pair["gold_index"], pair["prediction_index"]
            endpoints = corner_details(gold[identity][row], ps[index])
            failures = [point for point in endpoints if point["eligible"] and
                        (not point["accepted"] or point["error_px"] > 5)]
            if failures:
                case = target_case("corner", identity, row, "Matched front; corner check fails 5px")
                case["failing_corners"] = failures
                corners.append(case)
            if 0.5 <= pair["iou"] < 0.55:
                fragile.append(target_case("fragile", identity, row, "Matched; box IoU within 0.05 of gate"))
        for index in candidate["unmatched_prediction_indices"]:
            pred = ps[index]
            row, overlap = best_overlap(pred["bbox_xyxy"], gold[identity])
            duplicate = index in candidate["duplicate_prediction_indices"]
            kind = "duplicate" if duplicate else "near_gate" if overlap >= 0.4 else "positive_overlap" if overlap > 0 else "zero_overlap"
            categories[kind] += 1
            duplicate_rows = [r for r in assigned if _box_iou(pred["bbox_xyxy"], gold[identity][r]["bbox_xyxy"]) >= 0.5]
            duplicate_rows.sort(key=lambda r: (-_box_iou(pred["bbox_xyxy"], gold[identity][r]["bbox_xyxy"]), r))
            if duplicate:
                assert duplicate_rows
            # Duplicate owner is explicit even if another, missed gold row has larger overlap.
            shown_row = duplicate_rows[0] if duplicate else row
            unmatched.append(dict(kind=kind, image=identity, prediction_index=index, prediction_number=index + 1,
                                  primary_prediction_id=pred["primary_prediction_id"], prediction=pred,
                                  best_row=row + 1 if row is not None else None, best_iou=overlap,
                                  shown_row=shown_row + 1 if shown_row is not None else None,
                                  target=gold[identity][shown_row] if shown_row is not None else None,
                                  best_row_candidate_matched=row in assigned if row is not None else None,
                                  best_row_reference_matched=row in reference_assigned if row is not None else None,
                                  owner_prediction_number=assigned[shown_row] + 1 if shown_row in assigned else None,
                                  duplicate_owner_rows=[r + 1 for r in duplicate_rows]))
    assert len(misses) == 13 and len(recovered) == 1 and len(corners) == 18 and len(fragile) == 1
    assert len(unmatched) == 90 and categories == dict(duplicate=18, near_gate=17, positive_overlap=44, zero_overlap=11)
    inventory = dict(schema="baomai-r47-fixed-prediction-error-review-v1", source_pr=133, source_pr_head=PR_HEAD,
                     review_predecessor_pr=134, review_predecessor_head=PR134_HEAD,
                     archive_sha256=ARCHIVE_SHA, reference="r38", reference_manifest_sha256=REFERENCES["r38"][1],
                     fresh_inference=False, fresh_training=False, reference_labels_modified=False,
                     postclip=True, box_confidence=0.13, ordinary_nms_iou=0.7, extra_postclip_iou_strictly_greater_than=0.7,
                     matching_iou=0.5, endpoint_confidence=0.5,
                     counts=dict(misses=13, recoveries=1, matched_corner_failures=18, fragile_matches=1,
                                 unmatched=90, **categories),
                     scope="Complete candidate current-R38 inventory at the fixed PR133 operating point. Other unmatched is an assignment category, not a physical background verdict.",
                     per_image=per_image, misses=misses, recoveries=recovered, corner_failures=corners,
                     fragile_matches=fragile, unmatched_predictions=unmatched)
    return inventory, evidence, summary, gold


def originals_for(inventory, root, images_dir, download):
    from PIL import Image
    records = {item["id"]: item for item in json.loads((root / "data/manifest.json").read_text())["records"]}
    identities = sorted({case["image"] for key in ("misses", "recoveries", "corner_failures", "fragile_matches", "unmatched_predictions") for case in inventory[key]})
    images_dir.mkdir(parents=True, exist_ok=True)
    def obtain(identity):
        record, path = records[identity], images_dir / (identity + ".png")
        if not path.exists() and download:
            url = f"https://media.githubusercontent.com/media/Siuuuers/dl/{SOURCE}/data/{record['image']}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if sha(data) != record["image_sha256"]:
                raise ValueError("Downloaded original differs: " + identity)
            path.write_bytes(data)
        data = path.read_bytes()
        if sha(data) != record["image_sha256"]:
            raise ValueError("Original differs: " + identity)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        if image.size != (1280, 720):
            raise ValueError("Original dimensions differ: " + identity)
        return identity, image, record["image_sha256"]
    with ThreadPoolExecutor(max_workers=5) as executor:
        items = list(executor.map(obtain, identities))
    return {i: image for i, image, _ in items}, {i: digest for i, _, digest in items}


def render(inventory, evidence, originals, archive_path, output):
    from PIL import Image
    import reportlab
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    # Bundle-owned Vera avoids platform-specific font substitution.
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("ErrorReview", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("ErrorReviewBold", str(fonts / "VeraBd.ttf")))
    regular, strong = "ErrorReview", "ErrorReviewBold"
    W, H = landscape(A4)
    M, BW = 30, W - 60
    ink, muted, teal, line, amber, cyan = map(HexColor, ("#173043", "#526877", "#087f85", "#d6e1e6", "#ffad32", "#00d9ee"))
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(W, H), invariant=1, pageCompression=1)
    pdf.setTitle("R47 - complete missed, unmatched and corner-error review")
    pdf.setAuthor("Siuuuers/dl project review")
    page, case_pages = 0, []

    def text(x, y, value, size=9, bold=False, color=None):
        pdf.setFillColor(color or ink)
        pdf.setFont(strong if bold else regular, size)
        pdf.drawString(x, y, str(value))

    def paragraph(value, x, y, width=BW, size=9, leading=13, color=None):
        current = ""
        for word in str(value).split():
            trial = (current + " " + word).strip()
            if current and pdfmetrics.stringWidth(trial, regular, size) > width:
                text(x, y, current, size, color=color)
                y -= leading
                current = word
            else:
                current = trial
        if current:
            text(x, y, current, size, color=color)
            y -= leading
        return y

    def start(section, title, subtitle):
        nonlocal page
        page += 1
        pdf.setFillColor(teal)
        pdf.rect(0, H - 7, W, 7, stroke=0, fill=1)
        text(M, H - 31, section.upper(), 8, True, teal)
        text(M, H - 59, title, 20, True)
        paragraph(subtitle, M, H - 78, size=9, leading=12, color=muted)

    def end():
        pdf.setStrokeColor(line)
        pdf.line(M, 31, W - M, 31)
        text(M, 17, "R47 review | 27 Nov 2026 HKT | PR133 cached predictions | Current R38 labels unchanged", 7, color=muted)
        text(W - M - 22, 17, f"{page:02d}", 9, True, teal)
        pdf.showPage()

    def table(headers, rows, x, y, widths, row_height=25, size=9):
        for number, row in enumerate([headers] + rows):
            pdf.setFillColor(HexColor("#e8f2f3") if number == 0 else HexColor("#f4f7f9") if number % 2 else white)
            pdf.rect(x, y - row_height, sum(widths), row_height, stroke=0, fill=1)
            xx = x
            for value, width in zip(row, widths):
                text(xx + 7, y - row_height + 8, value, size, number == 0)
                xx += width
            y -= row_height
        return y - 15

    start("Result status", "PR134 is a review, not a new training run.", "The model result remains PR133 / reconstructed R45. This report expands its error coverage.")
    y = paragraph("Both R45 arms trained the original 137 photographs; the candidate additionally trained frame_0383 (43 fronts). PR134's R46 notebook verifies those saved results on CPU. It adds no new model result. All counts below use the same frozen cached predictions.", M, 484, size=11, leading=16)
    y = table(["Current R38 / 654 reference fronts", "Reference", "Candidate", "Change"], [
        ("Missed fronts", "11", "13", "+2"), ("Unmatched predictions (all)", "86", "90", "+4"),
        ("Formal duplicate subset", "22", "18", "-4"), ("Other unmatched subset", "64", "72", "+8"),
        ("Two-corner fronts within 5px / eligible", "571 / 583", "569 / 583", "-2"),
        ("Two-corner fronts within 10px / eligible", "582 / 583", "581 / 583", "-1")], M, y - 12, [358, 135, 135, BW - 628])
    y = paragraph("Unmatched is not a diagnosis of physical background. A visible cassette can remain unmatched because its box misses the IoU gate, another output already claimed the target, or the reference definition differs. The two unmatched subgroups sum to the total: 18 + 72 = 90.", M, y, size=10, leading=15)
    y = paragraph("This complete candidate review includes all 13 missed targets, all 90 unmatched outputs and all 18 matched fronts that fail the 5px corner check. It also shows one recovered target and one matched target with box IoU below 0.55. A missed target and unmatched output can describe the same physical cassette; do not add those counts as distinct objects.", M, y - 8, size=10, leading=15)
    paragraph("Decision remains: keep PR35 selected. Improve the physical annotations and qualify changed whole photographs before attributing any benefit to another training run.", M, y - 10, size=10, leading=15, color=teal)
    end()

    start("How to read the evidence", "Prioritize ownership and geometry, then suppression.", "All thresholds are unchanged. Diagnostic bins below do not alter the scoring rule.")
    y = table(["Candidate unmatched category", "Count", "What the number establishes"], [
        ("Formal duplicate", "18", "IoU >= 0.50 with an already matched reference front"),
        ("Other: best IoU 0.40 to below 0.50", "17", "Existing overlap is near the fixed match gate"),
        ("Other: best IoU above 0 to below 0.40", "44", "Some reference overlap; geometry or identity needs review"),
        ("Other: zero overlap with every R38 box", "11", "No R38 box overlap; physical background is unproven")], M, 485, [300, 60, BW - 360], size=8.5)
    for value in [
        "R### is a one-based row in the current R38 development label file. P### is a one-based index in the candidate's postclip list for that photograph. Primary # is the cached output index before the extra postclip filter; ordinary model NMS has already happened.",
        "Cyan boxes/circles are the named R38 target and its locatable corners. Amber boxes/crosses are a model output and accepted predicted corners. Dotted amber boxes identify the actual matched prediction in a duplicate card. The original crop has no overlay.",
        "For misses, the reference/candidate panels show the assigned prediction if matched, otherwise the highest-IoU overlapping retained output. That diagnostic local overlap does not replace full-image confidence-ranked greedy matching.",
        "All original pixels are hash checked. Crops use nearest-neighbor enlargement. Up decreases image y. Missing reference corners are never invented; rejected predicted endpoints are explained numerically.",
        "The three new R38 misses still have retained overlapping boxes: new_000009/R051 IoU 0.4916, new_000011/R059 0.4702, and new_000089/R048 0.4973. Lowering confidence does not fix those existing boxes.",
        "The cache does not expose pre-confidence or pre-NMS model output. It cannot diagnose why a different prediction is absent. An 'other unmatched' output can also be a second physical hypothesis below the formal duplicate-IoU gate. Review identity before choosing a suppression rule."]:
        y = paragraph(value, M, y - 9, size=9, leading=13)
    end()

    start("Coverage by photograph", "A small set of scenes carries most of the work.", "Every nonempty or empty development scene is included in the totals; zero-error scenes are shown too.")
    rows = []
    for item in inventory["per_image"]:
        a, b = item["reference"], item["candidate"]
        rows.append((item["image"], item["gold"], f"{a['fn']} / {b['fn']}", f"{a['fp']} / {b['fp']}", f"{a['duplicate_fp']} / {b['duplicate_fp']}", f"{a['other_fp']} / {b['other_fp']}"))
    table(["Photograph", "Fronts", "Miss ref / cand", "Unmatched ref / cand", "Duplicate ref / cand", "Other ref / cand"], rows, M, 485,
          [145, 66, 130, 160, 150, BW - 651], row_height=18, size=8)
    paragraph("new_000009, new_000011, new_000089 and new_000090 contribute 54 of 90 unmatched predictions and 9 of 13 misses. Review their crowded/partial-front regions before choosing a global confidence or NMS change.", M, 88, size=9, leading=13)
    end()

    def crop_bounds(boxes):
        x0, y0 = min(b[0] for b in boxes) - 18, min(b[1] for b in boxes) - 16
        x1, y1 = max(b[2] for b in boxes) + 18, max(b[3] for b in boxes) + 16
        width, height = max(92, x1 - x0), max(68, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        width, height = min(1280, width), min(720, height)
        x0, y0 = min(max(0, cx - width / 2), 1280 - width), min(max(0, cy - height / 2), 720 - height)
        return (math.floor(x0), math.floor(y0), math.ceil(x0 + width), math.ceil(y0 + height))

    def panel(image, bounds, x, y, width, height, target=None, pred=None, duplicate_owner=None):
        crop = image.crop(bounds)
        enlarged = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
        scale = min(width / crop.width, height / crop.height)
        iw, ih = crop.width * scale, crop.height * scale
        ox, oy = x + (width - iw) / 2, y + (height - ih) / 2
        pdf.drawImage(ImageReader(enlarged), ox, oy, width=iw, height=ih)
        def coord(px, py):
            return ox + (px - bounds[0]) * scale, oy + ih - (py - bounds[1]) * scale
        pdf.saveState()
        clip = pdf.beginPath()
        clip.rect(ox, oy, iw, ih)
        pdf.clipPath(clip, stroke=0)
        def draw(obj, color, predicted=False, dashed=False):
            bx, by, ex, ey = obj["bbox_xyxy"]
            left, bottom = coord(bx, ey)
            right, top = coord(ex, by)
            pdf.setStrokeColor(color)
            pdf.setLineWidth(0.85)
            pdf.setDash(2, 2) if dashed else pdf.setDash()
            pdf.rect(left, bottom, right - left, top - bottom, stroke=1, fill=0)
            pdf.setDash()
            if dashed:
                return
            for px, py, status in obj["keypoints"]:
                valid = px is not None and py is not None and math.isfinite(px) and math.isfinite(py)
                if not valid or (predicted and (status < .5 or not 0 <= px < 1280 or not 0 <= py < 720)) or (not predicted and status != 2):
                    continue
                xx, yy = coord(px, py)
                radius = 2.1
                if predicted:
                    pdf.line(xx - radius, yy, xx + radius, yy)
                    pdf.line(xx, yy - radius, xx, yy + radius)
                else:
                    pdf.circle(xx, yy, radius, stroke=1, fill=0)
        if duplicate_owner:
            draw(duplicate_owner, amber, True, True)
        if target:
            draw(target, cyan)
        if pred:
            draw(pred, amber, True)
        pdf.restoreState()

    def corner_label(points):
        parts = []
        for point in points:
            if not point["eligible"]:
                parts.append(point["side"] + " unavailable in reference")
            elif not point["accepted"]:
                parts.append(f"{point['side']} rejected (c {point['confidence']:.3f})")
            else:
                parts.append(f"{point['side']} {point['error_px']:.2f}px")
        return "; ".join(parts)

    def target_card(case, x, y, width, height):
        pdf.setStrokeColor(line)
        pdf.roundRect(x, y, width, height, 5, stroke=1, fill=0)
        text(x + 10, y + height - 18, f"{case['image']} / R{case['row']:03d}   {case['title']}", 10, True)
        text(x + 10, y + height - 33, f"{case['visible_corners']} locatable corner(s) | Current R38 reference", 8, color=muted)
        arms = case["arms"]
        boxes = [case["target"]["bbox_xyxy"]] + [a["prediction"]["bbox_xyxy"] for a in arms.values() if a["prediction"]]
        bounds = crop_bounds(boxes)
        inset, gap = 10, 12
        pw = (width - 2 * inset - 2 * gap) / 3
        image_y, image_h = y + 51, height - 103
        panel(originals[case["image"]], bounds, x + inset, image_y, pw, image_h)
        text(x + inset, image_y - 12, "Original pixels", 8, True)
        for offset, role in enumerate(("reference", "candidate"), 1):
            arm = arms[role]
            xx = x + inset + offset * (pw + gap)
            panel(originals[case["image"]], bounds, xx, image_y, pw, image_h, case["target"], arm["prediction"])
            label = role.title() + (" matched" if arm["matched"] else " missed")
            if arm["prediction"]:
                label += f" | IoU {arm['iou']:.4f} | c {arm['prediction']['confidence']:.3f}"
            else:
                label += " | no overlapping retained output"
            text(xx, image_y - 12, label, 7.3, True)
        if case["kind"] == "corner":
            note = "Candidate: " + corner_label(arms["candidate"]["corners"]) + "."
        elif case["kind"] == "fragile":
            note = "This remains a match at the unchanged gate. Inspect extent before drawing a robust recovery conclusion."
        elif case["kind"] == "recovery":
            note = "Recovered under current R38; assess alongside all three newly missed targets, not in isolation."
        elif case["title"] == "Newly missed":
            note = "A retained box exists below IoU 0.50. Prioritize visible-front extent and ownership, not lower confidence."
        else:
            note = "Unmatched in both arms. The highest-overlap view is diagnostic; absent pre-filter alternatives are unknown."
        paragraph(note, x + 10, y + 22, width - 20, size=7.6, leading=10)
        case_pages.append(dict(kind=case["kind"], image=case["image"], row=case["row"], page=page))

    target_groups = [
        ("Missed targets and changed matches", "13 candidate misses + 1 recovery + 1 fragile match", inventory["misses"] + inventory["recoveries"] + inventory["fragile_matches"]),
        ("Corner localization errors", "All 18 matched fronts that fail the 5px check", inventory["corner_failures"]),
    ]
    for section, title, cases in target_groups:
        for offset in range(0, len(cases), 2):
            start(section, title, f"Cases {offset + 1}-{min(offset + 2, len(cases))} of {len(cases)} | Original and both model arms use the same crop.")
            for index, case in enumerate(cases[offset:offset + 2]):
                target_card(case, M, 271 if index == 0 else 42, BW, 217)
            end()

    # Duplicate overlay also shows the output that actually received the gold assignment.
    with zipfile.ZipFile(archive_path) as archive:
        cached_predictions = json.loads(archive.read("evaluation/predictions.json"))["candidate"]

    def unmatched_card(case, x, y, width, height):
        pdf.setStrokeColor(line)
        pdf.roundRect(x, y, width, height, 5, stroke=1, fill=0)
        text(x + 8, y + height - 17, f"{case['image']} / P{case['prediction_number']:03d}  (primary #{case['primary_prediction_id']})", 9, True)
        names = dict(duplicate="Formal duplicate", near_gate="Other / near gate", positive_overlap="Other / positive overlap", zero_overlap="Other / zero R38 overlap")
        best = f"R{case['best_row']:03d}" if case["best_row"] else "none"
        text(x + 8, y + height - 31, f"{names[case['kind']]} | c {case['prediction']['confidence']:.3f}", 7.8, color=teal)
        text(x + 8, y + height - 44, f"Best R38 overlap: {best}, IoU {case['best_iou']:.4f}", 7.8, color=muted)
        target = case["target"]
        boxes = [case["prediction"]["bbox_xyxy"]] + ([target["bbox_xyxy"]] if target else [])
        bounds = crop_bounds(boxes)
        gap, pw = 10, (width - 26) / 2
        owner = cached_predictions[case["image"]]["postclip"][case["owner_prediction_number"] - 1] if case["kind"] == "duplicate" else None
        panel(originals[case["image"]], bounds, x + 8, y + 53, pw, height - 108)
        panel(originals[case["image"]], bounds, x + 8 + pw + gap, y + 53, pw, height - 108, target, case["prediction"], owner)
        text(x + 8, y + 41, "Original pixels", 7.5, True)
        text(x + 8 + pw + gap, y + 41, "R38 / candidate", 7.5, True)
        if case["kind"] == "duplicate":
            note = f"R{case['shown_row']:03d} was assigned to P{case['owner_prediction_number']:03d} (dotted). This extra output meets the duplicate rule."
        elif case["best_row"] is None:
            note = "No overlapping R38 box. Inspect physical identity; this count does not prove rack or background."
        else:
            cand = "matched" if case["best_row_candidate_matched"] else "missed"
            ref = "matched" if case["best_row_reference_matched"] else "missed"
            note = f"Best-overlap R{case['best_row']:03d}: candidate {cand}, reference {ref}. This output fails the 0.50 box-IoU gate."
        paragraph(note, x + 8, y + 26, width - 16, size=7.4, leading=10)
        case_pages.append(dict(kind=case["kind"], image=case["image"], prediction_number=case["prediction_number"], page=page))

    unmatched_groups = [
        ("Other unmatched outputs", "All 72 other unmatched predictions", [c for c in inventory["unmatched_predictions"] if c["kind"] != "duplicate"]),
        ("Formal duplicate outputs", "All 18 formal duplicate predictions", [c for c in inventory["unmatched_predictions"] if c["kind"] == "duplicate"]),
    ]
    for section, title, cases in unmatched_groups:
        for offset in range(0, len(cases), 4):
            start(section, title, f"Cases {offset + 1}-{min(offset + 4, len(cases))} of {len(cases)} | P### identifies a model output, not a source annotation row.")
            for index, case in enumerate(cases[offset:offset + 4]):
                unmatched_card(case, M + (index % 2) * ((BW + 12) / 2), 271 if index < 2 else 42, (BW - 12) / 2, 217)
            end()

    start("Sources and limits", "Complete coverage of this fixed operating point.", "The companion JSON preserves full coordinates, model-output identities and exact page membership.")
    y = paragraph("The archive CPU audit passed all 334 inventory entries, all 40 image/role postclip replays and all 12 reference/view/arm score reports. Checkpoint hashes bind recorded files; they do not independently establish model-to-prediction generation without fresh inference.", M, 484, size=10, leading=15)
    y = table(["Alternative reference / same development photographs", "Misses ref / cand", "Unmatched ref / cand", "Duplicates ref / cand"], [
        ("Frozen666 historical primary", "36 / 33", "99 / 98", "25 / 22"),
        ("Historical663 diagnostic", "32 / 30", "98 / 98", "25 / 22"),
        ("R38 current654 / cards in this report", "11 / 13", "86 / 90", "22 / 18")], M, y - 12, [350, 130, 160, BW - 640], size=8)
    y = paragraph("These reference versions describe the same 20 development photographs, not three independent test sets. The previous R45 PDF contains the additional historical changed-identity cards. R47 concentrates the complete error inventory on current R38 so a case cannot silently switch definitions.", M, y, size=10, leading=15)
    sources = [
        "Training result: https://github.com/Siuuuers/dl/pull/133",
        "Preceding review: https://github.com/Siuuuers/dl/pull/134",
        "PR133 result revision: " + PR_HEAD,
        "R45 recorded source revision: " + SOURCE,
        "Archive SHA-256: " + ARCHIVE_SHA,
        "R38 manifest SHA-256: " + REFERENCES["r38"][1],
        "Scorer: scripts/pose_scoring.py; audit: scripts/audit_additional_front_pilot_r45_results.py",
        "Renderer: scripts/export_front_error_review_r47.py; no model loading or reference-label writes.",
    ]
    for value in sources:
        y = paragraph(value, M, y - 8, size=8, leading=12)
    paragraph("Evidence-supported next step: inspect the named physical boundaries, preserve whole-image annotation gates, then test a genuinely changed training treatment against the unchanged reference arm. A lower duplicate count alone is not a sufficient success criterion.", M, y - 15, size=10, leading=15, color=teal)
    end()
    pdf.save()
    return page, case_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory, evidence, summary, gold = collect(root, args.archive)
    originals, hashes = originals_for(inventory, root, args.images_dir, args.download_images)
    inventory["original_image_sha256"] = hashes
    pages, case_pages = render(inventory, evidence, originals, args.archive, args.output)
    inventory["case_pages"] = case_pages
    index = args.index or args.output.with_suffix(".index.json")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n")
    verification = dict(schema="baomai-r47-error-pdf-verification-v1", pages=pages, pdf_bytes=args.output.stat().st_size,
                        pdf_sha256=sha(args.output.read_bytes()), index_sha256=sha(index.read_bytes()),
                        case_count=len(case_pages), counts=inventory["counts"],
                        original_image_sha256=hashes, cpu_audit_status=evidence["status"],
                        archive_sha256=ARCHIVE_SHA, fresh_inference=False, fresh_training=False,
                        reference_labels_modified=False)
    args.output.with_suffix(".verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(args.output), pages=pages, counts=inventory["counts"]), indent=2))


if __name__ == "__main__":
    main()
