#!/usr/bin/env python3
"""Render the combined R48 annotation review without changing any labels.

The supplied original pixels are shown with vector overlays. Legacy prediction
cues mean PR133 candidate / PR137 reference, never the new PR137 candidate.
Optional contours remain diagnostic traces, not segmentation or training masks.
Every development request and every current supplementary identity is indexed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import render_front_review_r46 as base

W, H, M, WIDTH = base.PAGE_W, base.PAGE_H, base.MARGIN, base.WIDTH
FONT, BOLD = "R46Sans", "R46Sans-Bold"
INK, MUTED, LIGHT, PALE = base.INK, base.MUTED, base.LIGHT, base.PALE
OLD_PREDICTION = colors.HexColor("#b26a00")
OLD_PREDICTION_NAME = "PR133 candidate = PR137 reference"
BOUND_NAMES = {"xmin": "left / xmin", "ymin": "top / ymin",
               "xmax": "right / xmax", "ymax": "bottom / ymax",
               "left_point": "left point", "right_point": "right point"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def e(value):
    return escape(str(value))


def number(value):
    """Display nominal source pixels, not misleading serialization precision."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def value_text(value):
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(number(v) for v in value) + ")"
    return number(value)


def field_name(value):
    return value.removeprefix("bbox.").removeprefix("box.")


def held_fields(row):
    values = list(row.get("unresolved_fields", []))
    values += [item["field"] for item in row.get("held_fields", [])]
    return list(dict.fromkeys(values))


def status(row):
    words = []
    if row.get("changed_fields"):
        words.append("AMEND")
    if held_fields(row):
        words.append("HOLD")
    if row.get("hold_resolutions"):
        words.append("RESOLVED")
    return " + ".join(words) or "RETAIN"


def flags(row):
    parts = []
    for condition, label in ((row.get("changed_fields"), "A"),
                             (row.get("requested"), "Q"),
                             (held_fields(row), "H"),
                             (row.get("hold_resolutions"), "R")):
        if condition:
            parts.append(label)
    return "/".join(parts) or "-"


def selected(row):
    return bool(row.get("changed_fields") or row.get("requested") or
                held_fields(row) or row.get("hold_resolutions"))


def source_name(row):
    if row.get("source_row") is None:
        return "A001"
    return f"S{row['source_row']:03d}"


def uncertainty(row):
    v = row.get("nominal_visual_uncertainty_px")
    if v is None:
        return "not quantified"
    if isinstance(v, (list, tuple)):
        return "-".join(number(x) for x in v) + " px"
    return "+/- " + number(v) + " px"


def delta(row):
    before, after = base.geometry(row["before_line"]), base.geometry(row["after_line"])
    changes = row.get("changed_fields", [])
    if not changes:
        return "Coordinates retained; see the recorded ownership hold below." if held_fields(row) else "Coordinates retained after original-pixel review."
    return "; ".join(f"{BOUND_NAMES.get(k, k)}: {value_text(before[k])} -> {value_text(after[k])}"
                     for k in changes)


def fit(c, text, x, top, width, height, size=8.3, minimum=7.0, color=INK):
    return base.fit_para(c, text, x, top, width, height, size=size,
                         minimum=minimum, color=color)


def header(c, title, subtitle):
    # Both lines are measured; identifiers must never run into the margin.
    base.para(c, e(title), M, H - 20, WIDTH, 18, leading=21,
              max_height=23, bold=True)
    fit(c, e(subtitle), M, H - 48, WIDTH, 12, size=8.2, minimum=7.4, color=MUTED)
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.6)
    c.line(M, H - 64, W - M, H - 64)


def footer(c, text="R48 assistant review proposals | Original pixels | Frozen model outputs and references preserved"):
    base.footer(c, text)


def legend(c, x, y, width=WIDTH, size=7.5):
    parts = [(base.CYAN, "retained box"), (base.MAGENTA, "R48 edit"),
             (base.LEFT, "left point"), (base.RIGHT, "right point"),
             (base.AMBER, "hold / supplied trace")]
    needed = sum(c.stringWidth(label, FONT, size) + 29 for _, label in parts)
    if needed > width:
        raise ValueError("Legend exceeds its allocated width")
    for color, label in parts:
        c.setFillColor(color)
        c.rect(x, y - 1, 8, 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont(FONT, size)
        c.drawString(x + 12, y - 2, label)
        x += c.stringWidth(label, FONT, size) + 29


def outlines(path):
    if not Path(path).is_file():
        return {}
    payload = read(path)
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("items", payload.get("outlines", []))
    if isinstance(items, dict):
        items = [{"stable_id": k, **v} for k, v in items.items()]
    result = {}
    for item in items:
        vertices = item.get("vertices_px", [])
        if len(vertices) >= 2:
            result[item["stable_id"]] = item
    return result


def draw_held(c, row, region, x, y, scale, overview=False):
    """Only existing, explicitly held edges are marked; no point is invented."""
    g = base.geometry(row["after_line"])
    if not g:
        return
    def p(px, py):
        return x + (px - region[0]) * scale, y + (region[3] - py) * scale
    edges = {"xmin": ((g["xmin"], g["ymin"]), (g["xmin"], g["ymax"])),
             "xmax": ((g["xmax"], g["ymin"]), (g["xmax"], g["ymax"])),
             "ymin": ((g["xmin"], g["ymin"]), (g["xmax"], g["ymin"])),
             "ymax": ((g["xmin"], g["ymax"]), (g["xmax"], g["ymax"]))}
    c.saveState()
    c.setStrokeColor(base.AMBER)
    c.setLineWidth(0.85 if overview else 1.1)
    c.setDash(2.1, 1.6)
    for field in held_fields(row):
        key = field_name(field)
        if key in edges:
            a, b = edges[key]
            c.line(*p(*a), *p(*b))
    c.restoreState()


def region_for(row, outline=None, include_cues=False):
    region = base.crop_region([row["before_line"], row["after_line"]], outline, margin=13)
    xs, ys = [region[0], region[2]], [region[1], region[3]]
    if include_cues:
        for cue in row.get("prediction_cues", []):
            p = cue["prediction"]
            a, b, d, f = p["bbox_xyxy"]
            xs.extend((a - 13, d + 13))
            ys.extend((b - 13, f + 13))
            for kp in p.get("keypoints", []):
                if len(kp) >= 2 and 0 <= kp[0] <= 1280 and 0 <= kp[1] <= 720:
                    xs.append(kp[0])
                    ys.append(kp[1])
    # Preserve surrounding ownership evidence even for a narrow partial front.
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = min(x0, cx - 53), max(x1, cx + 53)
    y0, y1 = min(y0, cy - 26), max(y1, cy + 26)
    return max(0, math.floor(x0)), max(0, math.floor(y0)), min(1280, math.ceil(x1)), min(720, math.ceil(y1))


def panel(c, image, region, x, top, width, height, label, line=None,
          row=None, outline=None, prediction=None, context=False):
    fit(c, e(label), x, top + 6, width, 12, size=7.8, minimum=7.0, color=MUTED)
    rw, rh = region[2] - region[0], region[3] - region[1]
    scale = min(width / rw, (height - 15) / rh)
    iw, ih = rw * scale, rh * scale
    left, bottom = x + (width - iw) / 2, top - 12 - ih
    # The originals are never sharpened, resampled with smoothing, or enhanced.
    raster = base.nearest_image(image, region, scale=1 if context else 5)
    c.drawImage(ImageReader(raster), left, bottom, iw, ih)
    c.saveState()
    clip = c.beginPath()
    clip.rect(left, bottom, iw, ih)
    c.clipPath(clip, stroke=0)
    base.draw_labels(c, line, region, left, bottom, scale,
                     changes=row.get("changed_fields", []) if row else (), outline=outline)
    if row:
        draw_held(c, row, region, left, bottom, scale)
    if prediction:
        def p(px, py):
            return left + (px - region[0]) * scale, bottom + (region[3] - py) * scale
        a, b, d, f = prediction["bbox_xyxy"]
        c.setStrokeColor(OLD_PREDICTION)
        c.setLineWidth(0.95)
        c.setDash(3, 2)
        c.rect(*p(a, f), (d - a) * scale, (f - b) * scale, stroke=1, fill=0)
        c.setDash()
        # These crosses display the recorded raw point cues, including clipped
        # image-edge coordinates; they make no availability or acceptance claim.
        for kp in prediction.get("keypoints", []):
            if len(kp) < 2:
                continue
            px, py = p(kp[0], kp[1])
            c.line(px - 2.2, py - 2.2, px + 2.2, py + 2.2)
            c.line(px - 2.2, py + 2.2, px + 2.2, py - 2.2)
    c.restoreState()
    return left, bottom, scale, iw, ih


def summary_page(c, manifest, plan):
    supp, dev = manifest["supplementary"], manifest["development"]
    rows = [r for f in supp["frames"] for r in f["rows"]]
    changed = sum(bool(r.get("changed_fields")) for r in rows)
    held = sum(bool(held_fields(r)) for r in rows)
    devheld = sum(bool(held_fields(r)) for r in dev["rows"])
    header(c, "R48 | Visible-front amendment review", "Original-pixel decisions following PR137 | 27 November 2026 | Assistant proposals for review")
    y = H - 85
    y -= base.para(c, "<b>Place each bound on the target cassette's own visible material.</b> "
                  "This report separates supported edits from unresolved ownership, and preserves the identity of every earlier label and prediction cue.",
                  M, y, WIDTH, 11.8, leading=15.5, max_height=50) + 16
    stats = [
        ("Development hints / distinct reviewed targets", f"{len(dev['requests'])} / {len(dev['rows'])}"),
        ("Development targets amended / targets with an explicit hold", f"{sum(bool(r.get('changed_fields')) for r in dev['rows'])} / {devheld}"),
        ("Supplementary original photos / current fronts inspected", f"{len(supp['frames'])} / {len(rows)}"),
        ("Supplementary rows amended / rows with unresolved fields", f"{changed} / {held}"),
        ("Whole-photo annotation candidates / photos still held", f"{len(supp['annotation_complete_ids'])} / {len(supp['held_image_ids'])}"),
        ("Development additions / deletions / visibility changes", "0 / 0 / 0"),
    ]
    for label, value in stats:
        c.setFillColor(PALE)
        c.rect(M, y - 24, WIDTH, 23, fill=1, stroke=0)
        base.para(c, e(label), M + 10, y - 5, WIDTH - 150, 9.5, max_height=15)
        c.setFillColor(INK)
        c.setFont(BOLD, 10.4)
        c.drawRightString(W - M - 11, y - 16, value)
        y -= 27
    y -= 16
    notes = [
        "<b>Prediction identity:</b> every amber prediction panel is an old <b>PR133 candidate</b> cue, reproduced by the <b>PR137 reference</b>. The new PR137 candidate is not drawn in this amendment report. A cue locates an issue; it does not establish physical truth.",
        "<b>Reference boundary:</b> development proposals are a separate R48 reference. The published R38 reference and PR137 model scores stay intact. This amendment package includes no new training or model promotion.",
        f"<b>Navigation:</b> coordinate and ownership rules on page 2; all development requests on page 3; target reviews on pages {plan['development_start']}-{plan['development_end']}; supplementary overviews on pages {plan['overview_start']}-{plan['overview_end']}; detailed comparisons from page {plan['cards_start']}.",
        "<b>Review status:</b> these are assistant visual proposals, with recorded nominal blur-limited uncertainty. They are not independent human sign-off or calibrated physical measurements. Only whole photographs without an open field are annotation-complete candidates; training split assignment remains separate.",
    ]
    for text in notes:
        y -= base.para(c, text, M, y, WIDTH, 8.8, leading=11.5, max_height=43) + 11
    legend(c, M, max(44, y - 2))
    footer(c)
    c.showPage()


def rules_page(c, manifest):
    header(c, "Coordinates, ownership and shadows", "One physical target, one consistent boundary convention; context may extend beyond the target")
    y = H - 84
    y -= base.para(c, "<b>Original image coordinates:</b> 1280 x 720 pixels. x increases to the right; y increases downward. Bounds are stored as xmin / ymin / xmax / ymax. The user's physical top/bottom wording is translated before any edit.",
                  M, y, WIDTH, 10.2, leading=13.4, max_height=42) + 15
    headers = ["User's physical wording", "Stored field", "Motion in image pixels"]
    table = [("Top / user's ymax", "ymin", "Move top down: increase ymin"),
             ("Bottom / user's ymin", "ymax", "Move bottom up: decrease ymax"),
             ("Left boundary", "xmin", "Move left edge right: increase xmin"),
             ("Right boundary", "xmax", "Move right edge left: decrease xmax")]
    xs, widths = [M + 9, M + 280, M + 405], [260, 110, WIDTH - 415]
    c.setFillColor(INK)
    c.rect(M, y - 23, WIDTH, 23, fill=1, stroke=0)
    for x, width, text in zip(xs, widths, headers):
        base.para(c, e(text), x, y - 6, width, 8.8, color=colors.white, bold=True, max_height=15)
    y -= 25
    for idx, values in enumerate(table):
        c.setFillColor(PALE if idx % 2 == 0 else colors.white)
        c.rect(M, y - 23, WIDTH, 23, fill=1, stroke=0)
        for x, width, text in zip(xs, widths, values):
            base.para(c, e(text), x, y - 5, width, 9, max_height=16)
        y -= 24
    y -= 18
    rules = [
        ("Follow material ownership", "Include the cassette's own visible gray molded rim, cap and side turn. Exclude a continuing rack strip, a gripper surface, and a neighboring cassette. Brightness alone cannot identify ownership."),
        ("Small objects keep the same target definition", "Do not add an upper or lower cassette's shadow line just because this front is small. A shadow on the target's own material may belong inside its visible extent; another object's surface or background does not. Extra context can remain in the input crop without expanding the target."),
        ("Use lighting as a cue", "A bright corner with a darker underside and a soft lateral gray transition can support a contour judgment. Illumination, reflections, blur and exposure can reverse that appearance. Confirm the continuous side-to-bottom turn and neighboring structure."),
        ("Preserve missing points and explicit holds", "Do not infer a hidden lower corner from a jaw, neighboring cassette, rectangle, or prediction. Amber dashed edges mark unresolved retained bounds; solid supplied traces are explanatory only. An ownership hold prevents whole-photo candidate export."),
        ("Read the evidence at its actual precision", "All comparison panels use the same original crop and nearest-neighbor enlargement, with vector overlays. Displayed coordinates are rounded to two decimals; the TXT lines remain exact. Nominal uncertainty is an annotation judgment, not a confidence interval."),
    ]
    for title, body in rules:
        y -= base.para(c, f"<b>{e(title)}.</b> {e(body)}", M, y, WIDTH, 9.0,
                       leading=11.8, max_height=38) + 12
    footer(c, "Diagnostic traces are optional, partial explanatory curves; never inferred segmentation polygons")
    c.showPage()


def request_map_page(c, dev, target_pages):
    header(c, "Development requests | stable target map", "Old P numbers identify PR133 candidate / PR137 reference predictions; R numbers identify R38 label rows")
    rows_by_key = {(r["image_id"], r["reference_row"]): r for r in dev["rows"]}
    y = H - 84
    cols = [(M + 7, 35), (M + 49, 140), (M + 197, 59),
            (M + 264, 77), (M + 350, 160), (M + 520, 185), (W - M - 40, 33)]
    heads = ["ID", "Original image", "Old cue", "R38 row", "Fields requested", "Disposition", "Page"]
    c.setFillColor(INK)
    c.rect(M, y - 25, WIDTH, 25, fill=1, stroke=0)
    for (x, width), label in zip(cols, heads):
        base.para(c, e(label), x, y - 7, width, 8.2, color=colors.white, bold=True, max_height=15)
    y -= 27
    for idx, request in enumerate(dev["requests"]):
        row = rows_by_key[(request["image_id"], request["reference_row"])]
        vals = [request["request_id"], request["image_id"], f"{request['kind']}{request['number']:03d}",
                f"R{request['reference_row']:03d}", ", ".join(request.get("requested_fields", [])),
                status(row), str(target_pages[row["stable_id"]])]
        c.setFillColor(PALE if idx % 2 == 0 else colors.white)
        c.rect(M, y - 22, WIDTH, 22, fill=1, stroke=0)
        for (x, width), value in zip(cols, vals):
            fit(c, e(value), x, y - 5, width, 17, size=8.2, minimum=7.2)
        y -= 23
    y -= 15
    notes = [
        "Several old predictions can map to the same existing physical front. They do not create extra annotation instances. Each target page shows all recorded cue panels separately from the reviewed labels.",
        "D01 / new_000010 R029 retains its clipped bottom with an ownership hold. D09 / new_000089 R041 amends the top but retains the unresolved right extent. A recorded hold is a review result, not a silently accepted prediction.",
        "Every request's physical hint, full reason, exact before/after TXT line, supplied old prediction geometry and original image hash are retained in the combined manifest and audit ledger.",
    ]
    for text in notes:
        y -= base.para(c, e(text), M, y, WIDTH, 8.6, leading=11.2, max_height=26) + 10
    footer(c)
    c.showPage()


def development_page(c, row, requests, image, outline):
    title = f"{row['image_id']} / R38 row {row['reference_row']:03d}"
    header(c, title, status(row) + " | Original / R38 / R48 comparisons | Stable identity: " + row["stable_id"])
    hints = [f"{r['request_id']} ({r['kind']}{r['number']:03d}): {r.get('physical_hint', '')}" for r in requests]
    fit(c, "<b>Request:</b> " + e("  |  ".join(hints)), M, H - 78, WIDTH, 29, size=8.4, minimum=7.4)
    region = region_for(row, outline, include_cues=True)
    gap = 12
    pw = (WIDTH - 2 * gap) / 3
    for idx, (label, line, current) in enumerate((("Original pixels", None, False),
                                               ("R38 frozen label", row["before_line"], False),
                                               ("R48 proposal / hold", row["after_line"], True))):
        panel(c, image, region, M + idx * (pw + gap), H - 114, pw, 125, label,
              line, row=row if current else None, outline=outline if current else None)
    legend(c, M, H - 248, size=7.3)
    base.para(c, "<b>Old prediction cues - " + OLD_PREDICTION_NAME + "</b>",
              M, H - 263, WIDTH, 9.4, max_height=15)
    base.para(c, "Amber dashed boxes and crosses are recorded model geometry; crosses do not certify corner availability.",
              M, H - 279, WIDTH, 7.7, color=MUTED, max_height=12)
    cues = row.get("prediction_cues", [])
    if len(cues) > 4:
        raise ValueError(f"{row['stable_id']}: more than four cue panels require an additional page")
    if cues:
        cw = (WIDTH - gap * (len(cues) - 1)) / len(cues)
        for idx, cue in enumerate(cues):
            p = cue["prediction"]
            role = "requested" if cue.get("role") == "requested" else "context / assigned"
            x = M + idx * (cw + gap)
            panel(c, image, region, x, H - 304, cw, 103,
                  f"Old P{cue['prediction_number']:03d} | {role}", prediction=p)
            fit(c, e(f"Model score {p.get('confidence', 0):.3f} | raw point cues"),
                x, H - 413, cw, 12, size=7.3, minimum=6.8, color=MUTED)
    else:
        base.para(c, "No frozen prediction cue is supplied for this target.", M, H - 320, WIDTH, 9)
    text_w = WIDTH - 178
    fit(c, "<b>Nominal edit:</b> " + e(delta(row)), M, 167, text_w, 27, size=8.5, minimum=7.5)
    fit(c, "<b>Original-pixel decision:</b> " + e(row["reason"]), M, 140, text_w, 58, size=8.4, minimum=7.3)
    holds = row.get("held_fields", [])
    if holds:
        msg = "  ".join(f"{h['field']}: {h['reason']}" for h in holds)
        fit(c, "<b>Explicit HOLD:</b> " + e(msg), M, 77, text_w, 33, size=8.0, minimum=7.0, color=base.AMBER)
    else:
        fit(c, "<b>Point slots:</b> availability is unchanged; no additional physical front is introduced.",
            M, 73, text_w, 21, size=8.1, minimum=7.4, color=MUTED)
    # Whole-frame context anchors the enlarged crop without adding another target.
    left, bottom, scale, _, _ = panel(c, image, (0, 0, 1280, 720), W - M - 160,
                                     160, 160, 101, "Full photo / crop location", context=True)
    c.setStrokeColor(base.MAGENTA)
    c.setLineWidth(0.8)
    c.rect(left + region[0] * scale, bottom + (720 - region[3]) * scale,
           (region[2] - region[0]) * scale, (region[3] - region[1]) * scale, stroke=1, fill=0)
    fit(c, e("Nominal uncertainty: " + uncertainty(row)), W - M - 160, 54, 160, 16,
        size=7.4, minimum=6.9, color=MUTED)
    footer(c, "Old cues are PR133 candidate = PR137 reference | R48 proposals do not replace published R38 scores")
    c.showPage()


def overview_labels(c, rows, x, y, scale, outlines_by_id, image_width, image_height):
    region = (0, 0, 1280, 720)
    for row in rows:
        base.draw_labels(c, row["after_line"], region, x, y, scale,
                         changes=row.get("changed_fields", []),
                         outline=outlines_by_id.get(row["stable_id"]), overview=True)
        draw_held(c, row, region, x, y, scale, overview=True)
    occupied = []
    for row in rows:
        g = base.geometry(row["after_line"])
        label = source_name(row)
        fs = 5.1
        lw = c.stringWidth(label, BOLD, fs) + 3
        lh = 6.6
        lx = x + g["xmin"] * scale
        rx = x + g["xmax"] * scale
        bottom = y + (720 - g["ymax"]) * scale
        top = y + (720 - g["ymin"]) * scale
        mid = (bottom + top - lh) / 2
        choices = [(lx + 1, mid), (rx - lw - 1, mid),
                   (lx - lw - 1, mid), (rx + 1, mid),
                   (lx + 1, top + 1), (rx - lw, bottom - lh - 1)]
        choices += [(cx, cy + offset) for offset in (-7, 7, -14, 14)
                    for cx, cy in choices[:4]]
        chosen = None
        for cx, cy in choices:
            box = (cx, cy, cx + lw, cy + lh)
            if cx < x or cy < y or cx + lw > x + image_width or cy + lh > y + image_height:
                continue
            if not any(base.overlap(box, old) for old in occupied):
                chosen = box
                break
        if chosen is None:
            raise ValueError("No non-overlapping full-frame label placement for " + row["stable_id"])
        occupied.append(chosen)
        c.setFillColor(colors.white)
        c.rect(chosen[0], chosen[1], lw, lh, fill=1, stroke=0)
        c.setFillColor(base.AMBER if held_fields(row) else INK)
        c.setFont(BOLD, fs)
        c.drawString(chosen[0] + 1.5, chosen[1] + 1.3, label)


def overview_page(c, frame, image, outlines_by_id, card_pages):
    rows = frame["rows"]
    title = frame["image_id"] + " | Full-frame coverage"
    gate = "ANNOTATION-COMPLETE CANDIDATE" if frame["annotation_complete"] else "HELD PHOTOGRAPH"
    header(c, title, gate + f" | All {len(rows)} current identities shown | Training split: unassigned")
    left_w = 552
    index_x = M + left_w + 17
    index_w = W - M - index_x
    top = H - 98
    scale = left_w / 1280
    ih = 720 * scale
    bottom = top - ih
    c.drawImage(ImageReader(image), M, bottom, left_w, ih)
    c.saveState()
    clip = c.beginPath()
    clip.rect(M, bottom, left_w, ih)
    c.clipPath(clip, stroke=0)
    overview_labels(c, rows, M, bottom, scale, outlines_by_id, left_w, ih)
    c.restoreState()
    c.setFillColor(INK)
    c.setFont(BOLD, 8.0)
    c.drawString(M, H - 83, "R48 proposal | Every current front and its stable source ID")
    cols = [(index_x + 2, 40), (index_x + 47, 27), (index_x + 80, 55), (index_x + 142, index_w - 144)]
    for (x, _), label in zip(cols, ["Source", "TXT", "Flags", "Detail p."]):
        c.setFont(BOLD, 7.2)
        c.drawString(x, H - 83, label)
    row_height = min(9.45, (H - 150) / max(1, len(rows)))
    y = H - 95
    for idx, row in enumerate(rows):
        if idx % 2 == 0:
            c.setFillColor(PALE)
            c.rect(index_x, y - row_height + 1, index_w, row_height, fill=1, stroke=0)
        vals = [source_name(row), f"{row['output_row']:03d}", flags(row),
                str(card_pages.get(row["stable_id"], "-"))]
        for (x, width), val in zip(cols, vals):
            fit(c, e(val), x, y - 0.8, width, row_height - 0.5,
                size=6.9, minimum=6.3, color=base.AMBER if held_fields(row) else INK)
        y -= row_height
    y = bottom - 21
    legend(c, M, y, width=left_w, size=6.9)
    y -= 21
    summary = (f"{sum(bool(r.get('changed_fields')) for r in rows)} rows amended; "
               f"{sum(bool(held_fields(r)) for r in rows)} rows held; "
               f"{sum(len(r.get('hold_resolutions', [])) for r in rows)} previous fields resolved. "
               "Corner coordinates and availability remain unchanged.")
    y -= base.para(c, e(summary), M, y, left_w, 8.7, leading=11.5, max_height=30) + 12
    y -= base.para(c, "<b>Index:</b> S = original source row; A001 = historical added identity. "
                  "TXT is the current output line and may differ from the source row. "
                  "A = amended; Q = explicitly requested; H = unresolved hold; R = resolved field. "
                  "A dash in Detail p. means the exact row is retained and indexed here.",
                  M, y, left_w, 8.2, leading=10.7, max_height=47) + 9
    base.para(c, "All exact row decisions are in the manifest. Solid amber traces are optional diagnostic curves; dashed amber box edges are held. Neither is a segmentation mask.",
              M, y, left_w, 7.8, leading=10.1, color=MUTED, max_height=33)
    footer(c, "Every current supplementary identity is indexed | Whole-photo export requires every field to be resolved")
    c.showPage()


def supplementary_card(c, row, image, outline, top, height=234):
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.65)
    c.setFillColor(colors.white)
    c.roundRect(M, top - height, WIDTH, height, 5, stroke=1, fill=1)
    name = f"{row['image_id']} / " + (f"source row {row['source_row']:03d}" if row.get("source_row") is not None else "historical added row A001")
    base.para(c, e(name), M + 9, top - 8, WIDTH - 260, 10.2, bold=True, max_height=15)
    label = status(row) + (" | REQUESTED" if row.get("requested") else "") + f" | TXT {row['output_row']:03d}"
    c.setFont(BOLD, 7.5)
    c.setFillColor(base.AMBER if held_fields(row) else (base.MAGENTA if row.get("changed_fields") else MUTED))
    c.drawRightString(W - M - 9, top - 18, label)
    region = region_for(row, outline)
    gap, pw = 10, (WIDTH - 38) / 3
    for idx, (label, line, current) in enumerate((("Original pixels", None, False),
                                               ("R47 previous label", row["before_line"], False),
                                               ("R48 proposal / hold", row["after_line"], True))):
        panel(c, image, region, M + 9 + idx * (pw + gap), top - 34, pw, 79,
              label, line, row=row if current else None, outline=outline if current else None)
    fit(c, "<b>Nominal edit:</b> " + e(delta(row)), M + 9, top - 119, WIDTH - 18, 22,
        size=8.0, minimum=7.0)
    fit(c, e(row["reason"]), M + 9, top - 146, WIDTH - 18, 57, size=8.0, minimum=7.0)
    holds = ", ".join(held_fields(row)) or "none"
    resolved = ", ".join(item["field"] for item in row.get("hold_resolutions", [])) or "none"
    meta = f"Open fields: {holds} | Resolved: {resolved} | Nominal uncertainty: {uncertainty(row)}"
    if outline:
        meta += " | Solid amber: diagnostic trace only"
    fit(c, e(meta), M + 9, top - 209, WIDTH - 18, 19, size=7.2, minimum=6.5,
        color=base.AMBER if held_fields(row) else MUTED)


def make_plan(manifest):
    dev = manifest["development"]["rows"]
    frames = manifest["supplementary"]["frames"]
    selected_rows = [r for f in frames for r in f["rows"] if selected(r)]
    dev_start = 4
    target_pages = {r["stable_id"]: dev_start + i for i, r in enumerate(dev)}
    overview_start = dev_start + len(dev)
    overview_pages = {f["image_id"]: overview_start + i for i, f in enumerate(frames)}
    cards_start = overview_start + len(frames)
    card_pages = {r["stable_id"]: cards_start + i // 2 for i, r in enumerate(selected_rows)}
    return dict(development_start=dev_start, development_end=overview_start - 1,
                overview_start=overview_start, overview_end=cards_start - 1,
                cards_start=cards_start, pages=cards_start - 1 + math.ceil(len(selected_rows) / 2),
                target_pages=target_pages, overview_pages=overview_pages,
                card_pages=card_pages, selected_rows=selected_rows)


def render(folder: Path, pdf: Path):
    """Render a combined builder folder and a deterministic page index beside PDF."""
    folder, pdf = Path(folder), Path(pdf)
    manifest = read(folder / "manifest.json")
    if manifest.get("schema") != "baomai-r48-visible-front-amendments-v1":
        raise ValueError("Expected the combined R48 builder manifest")
    dev, supp = manifest["development"], manifest["supplementary"]
    plan = make_plan(manifest)
    dev_outlines = outlines(folder / "development/outlines.json")
    supp_outlines = outlines(folder / "supplementary/outlines.json")
    images = {}
    expected = {f["image_id"]: f["source_image_sha256"] for f in supp["frames"]}
    expected.update({r["image_id"]: r["source_image_sha256"] for r in dev["rows"]})
    for identity, digest in expected.items():
        path = folder / "originals" / (identity + (".png" if identity.startswith("new_") else ".jpg"))
        if sha(path) != digest:
            raise ValueError("Original image hash differs: " + identity)
        with Image.open(path) as source:
            if source.size != (1280, 720):
                raise ValueError("Expected original 1280 x 720 geometry: " + identity)
            images[identity] = source.convert("RGB")
    all_rows = [r for f in supp["frames"] for r in f["rows"]]
    if len({r["stable_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Duplicate supplementary stable identities")
    request_ids = {r["request_id"] for r in dev["requests"]}
    row_request_ids = [i for r in dev["rows"] for i in r["request_ids"]]
    if set(row_request_ids) != request_ids or len(row_request_ids) != len(request_ids):
        raise ValueError("Every development request must map to exactly one reviewed target")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf), pagesize=(W, H), invariant=1, pageCompression=1)
    c.setTitle("R48 visible-front annotation amendments and ownership holds")
    c.setAuthor("Siuuuers/dl annotation review")
    c.setSubject("Original-pixel review; old prediction cues are PR133 candidate = PR137 reference")
    summary_page(c, manifest, plan)
    rules_page(c, manifest)
    request_map_page(c, dev, plan["target_pages"])
    requests_by_id = {r["request_id"]: r for r in dev["requests"]}
    page_records = [dict(page=1, section="summary"), dict(page=2, section="coordinate_ownership_rules"),
                    dict(page=3, section="development_request_map", request_ids=sorted(request_ids))]
    for row in dev["rows"]:
        if c.getPageNumber() != plan["target_pages"][row["stable_id"]]:
            raise AssertionError("Development pagination drift")
        page_records.append(dict(page=c.getPageNumber(), section="development_target",
                                 stable_ids=[row["stable_id"]], request_ids=row["request_ids"],
                                 old_prediction_numbers=[q["prediction_number"] for q in row.get("prediction_cues", [])]))
        development_page(c, row, [requests_by_id[i] for i in row["request_ids"]], images[row["image_id"]],
                         dev_outlines.get(row["stable_id"]))
    for frame in supp["frames"]:
        if c.getPageNumber() != plan["overview_pages"][frame["image_id"]]:
            raise AssertionError("Supplementary overview pagination drift")
        page_records.append(dict(page=c.getPageNumber(), section="supplementary_overview_and_index",
                                 image_id=frame["image_id"], stable_ids=[r["stable_id"] for r in frame["rows"]]))
        overview_page(c, frame, images[frame["image_id"]], supp_outlines, plan["card_pages"])
    cards = plan["selected_rows"]
    for start in range(0, len(cards), 2):
        batch = cards[start:start + 2]
        header(c, "Supplementary fronts | detailed comparisons", "Same original crop | R47 previous label | R48 proposal | Magenta edits; dashed amber holds; solid amber diagnostic traces")
        page_records.append(dict(page=c.getPageNumber(), section="supplementary_comparison_cards",
                                 stable_ids=[r["stable_id"] for r in batch]))
        for idx, row in enumerate(batch):
            if c.getPageNumber() != plan["card_pages"][row["stable_id"]]:
                raise AssertionError("Supplementary card pagination drift")
            supplementary_card(c, row, images[row["image_id"]], supp_outlines.get(row["stable_id"]),
                               H - 78 - idx * 244)
        footer(c)
        c.showPage()
    page_count = c.getPageNumber() - 1
    if page_count != plan["pages"]:
        raise AssertionError("Final pagination differs from the page index")
    c.save()
    index = dict(
        schema="baomai-r48-amendment-pdf-page-index-v1", pdf_name=pdf.name, pages=page_count,
        pdf_sha256=sha(pdf), manifest_sha256=sha(folder / "manifest.json"), renderer_sha256=sha(__file__),
        prediction_cue_identity=OLD_PREDICTION_NAME, original_geometry=[1280, 720],
        coordinate_display="Rounded to two source-pixel decimals; exact lines remain in manifest",
        diagnostic_outlines={"development": len(dev_outlines), "supplementary": len(supp_outlines), "segmentation": False},
        coverage=dict(development_requests=len(request_ids), development_targets=len(dev["rows"]),
                      supplementary_rows=len(all_rows), supplementary_frames=len(supp["frames"]),
                      supplementary_detail_cards=len(cards)),
        page_records=page_records,
        development=[dict(stable_id=r["stable_id"], image_id=r["image_id"], reference_row=r["reference_row"],
                          page=plan["target_pages"][r["stable_id"]], request_ids=r["request_ids"],
                          changed_fields=r["changed_fields"], held_fields=r.get("held_fields", []),
                          prediction_numbers=[q["prediction_number"] for q in r.get("prediction_cues", [])],
                          crop_xyxy=region_for(r, dev_outlines.get(r["stable_id"]), include_cues=True),
                          before_line=r["before_line"], after_line=r["after_line"])
                     for r in dev["rows"]],
        supplementary=[dict(stable_id=r["stable_id"], image_id=r["image_id"], source_row=r["source_row"],
                            output_row=r["output_row"], overview_page=plan["overview_pages"][r["image_id"]],
                            detail_page=plan["card_pages"].get(r["stable_id"]), flags=flags(r),
                            changed_fields=r["changed_fields"], unresolved_fields=held_fields(r),
                            requested=bool(r.get("requested")), before_line=r["before_line"], after_line=r["after_line"])
                       for r in all_rows])
    index_path = pdf.with_suffix(".index.json")
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"pages": page_count, "pdf": str(pdf), "page_index": str(index_path), "coverage": index["coverage"]}


def render_pdf(folder: Path, pdf: Path):
    """Builder-compatible alias."""
    return render(folder, pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True, help="Combined R48 builder output folder")
    parser.add_argument("--output", type=Path, required=True, help="Destination PDF; page index is written beside it")
    args = parser.parse_args()
    print(json.dumps(render(args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
