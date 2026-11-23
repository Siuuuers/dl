#!/usr/bin/env python3
"""Render the compact, four-page R25 amendment PDF from saved label bytes."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

from render_front_review_r24 import decode, descriptor, digest, font, require


REVIEWS = {
    "r24": "data/reviews/fronts-2026-11-23-r24/manifest.json",
    "r25": "data/reviews/fronts-2026-11-23-r25/manifest.json",
}
LEDGER = "results/review/front-review-20261123-r25"
PHOTO_ORDER = ("new_000008", "new_000010", "new_000012")
VISIBLE_STANDARDS = {("new_000008", row) for row in (18, 19, 20, 21, 28, 29, 30, 31, 32)} | {
    ("new_000010", 16)
}
ALL_STANDARDS = VISIBLE_STANDARDS | {("new_000016", 24)}
GROUPS = {
    "new_000008": [
        {"rows": (18, 19, 20, 21), "scale": 2, "title": "Accepted standards: rows 18-21"},
        {"rows": (28, 29, 30, 31, 32), "scale": 2, "title": "Accepted standards: rows 28-32"},
        {"rows": (35,), "scale": 2, "title": "Amended row 35; right corner remains outside the image"},
    ],
    "new_000010": [
        {"rows": (16, 18, 19, 20), "scale": 2, "title": "Accepted row 16 and amended rows 18-20"},
        {"rows": (18, 19, 20), "scale": 4, "crop": (772, 250, 825, 369),
         "title": "Closer view of the three amended left corners"},
    ],
    "new_000012": [
        {"rows": (1, 2, 3), "scale": 2, "title": "Rows 1-3: each lower front-face turn was reviewed separately"},
        {"rows": (28,), "scale": 3, "title": "Row 28: corrected left corner, retained soft right boundary"},
        {"rows": (1, 2, 3), "scale": 4, "crop": (98, 395, 153, 490),
         "title": "Closer view of the three left front-face corners"},
    ],
}
PALETTE = {
    "ink": "#102637", "muted": "#526675", "line": "#cad5dd",
    "page": "#ffffff", "panel": "#f3f6f8", "old": "#f12cb1",
    "changed": "#00bc79", "retained": "#00b8d2", "amber": "#fff0d5",
}
CANVAS_SIZE = (2400, 1680)


def text_width(draw, value, face):
    return draw.textbbox((0, 0), value, font=face)[2]


def wrapped(draw, value, xy, width, size=24, fill=None, line_gap=7):
    """Draw bounded text and return its lower edge."""
    x, y = xy
    face = font(size)
    for paragraph in value.split("\n"):
        words, line = paragraph.split(), ""
        if not words:
            y += size + line_gap
            continue
        for word in words:
            proposed = (line + " " + word).strip()
            if line and text_width(draw, proposed, face) > width:
                draw.text((x, y), line, font=face, fill=fill or PALETTE["ink"])
                y += size + line_gap
                line = word
            else:
                line = proposed
        draw.text((x, y), line, font=face, fill=fill or PALETTE["ink"])
        y += size + line_gap
    return y


def pixel_xy(values):
    return f"({values[0]:.2f}, {values[1]:.2f})"


def source_bundle(root):
    manifests = {rev: json.loads((root / path).read_text()) for rev, path in REVIEWS.items()}
    require(manifests["r25"]["parent"] == descriptor(root, root / REVIEWS["r24"]), "R25 parent changed")
    data_path = root / "data/manifest.json"
    require(all(m["source_manifest_sha256"] == digest(data_path) for m in manifests.values()),
            "Dataset manifest changed")
    records = {r["id"]: r for r in json.loads(data_path.read_text())["records"]}
    index = {rev: {r["image_id"]: r for r in m["files"]} for rev, m in manifests.items()}
    audit_path = root / LEDGER / "point-review.json"
    require(descriptor(root, audit_path) in manifests["r25"]["review_evidence"], "Point ledger changed")
    audit = json.loads(audit_path.read_text())
    decisions = {(r["image_id"], r["row"]): r for r in audit["rows"]}
    standards = {(r["image_id"], r["row"]): r for r in audit["preserved_standards"]}
    require(set(standards) == ALL_STANDARDS, "Protected standards differ")
    require(len(decisions) == 8, "Expected exactly eight changed rows")
    require(set(manifests["r25"]["changed_image_ids"]) == set(PHOTO_ORDER), "Changed-photo scope differs")
    bundle = {}
    standard_receipts = []
    for iid in (*PHOTO_ORDER, "new_000016"):
        record = records[iid]
        image_path = root / "data" / record["image"]
        require(digest(image_path) == record["image_sha256"], f"Image hash changed: {iid}")
        original = Image.open(image_path).convert("RGB")
        require(original.size == (1280, 720), f"Image dimensions differ: {iid}")
        item = {"original": original, "record": record, "image_source": descriptor(root, image_path),
                "labels": {}, "label_sources": {}, "points": {}}
        for rev in REVIEWS:
            entry = index[rev][iid]
            require(entry["source_image_sha256"] == digest(image_path), f"Review image hash changed: {iid}")
            require(entry["split"] == record["split"], f"Image split changed: {iid}")
            path = (root / REVIEWS[rev]).parent / entry["label_path"]
            require(digest(path) == entry["label_sha256"], f"Saved label hash changed: {rev}/{iid}")
            rows = path.read_text().splitlines()
            require(all(line.strip() for line in rows), "Unexpected blank line in labels")
            item["labels"][rev] = rows
            item["label_sources"][rev] = descriptor(root, path)
            item["points"][rev] = {i + 1: decode(line, original.size) for i, line in enumerate(rows)}
        old_lines, new_lines = item["labels"]["r24"], item["labels"]["r25"]
        require(len(old_lines) == len(new_lines), f"Row count changed: {iid}")
        for row, (old_line, new_line) in enumerate(zip(old_lines, new_lines), 1):
            key = (iid, row)
            old, new = old_line.split(), new_line.split()
            if key in decisions:
                decision = decisions[key]
                require(old_line == decision["before_line"] and new_line == decision["after_line"],
                        f"Saved/audited line mismatch: {key}")
                require(decision["source_image_sha256"] == digest(image_path), f"Audit image mismatch: {key}")
                require(old[:5] == new[:5] and old[7:] == new[7:], f"Non-left-XY change: {key}")
                require(old[5:7] != new[5:7], f"Missing expected left-point change: {key}")
                for side in ("left", "right"):
                    for rev, field in (("r24", "before_pixel_xy"), ("r25", "after_pixel_xy")):
                        observed = item["points"][rev][row][side]
                        expected = decision["point_reviews"][side][field]
                        require(all(abs(a - b) < 1e-6 for a, b in zip(observed[:2], expected)),
                                f"Saved/audited XY mismatch: {key}/{side}")
                        require(observed[2] == decision["point_reviews"][side]["visibility"],
                                f"Visibility mismatch: {key}/{side}")
            else:
                require(old_line == new_line, f"Undeclared changed row: {key}")
            if key in standards:
                require(old_line == new_line == standards[key]["line"], f"Protected row changed: {key}")
                standard_receipts.append({"image_id": iid, "row": row,
                                          "exact_line_sha256_utf8_no_newline": sha256(old_line.encode()).hexdigest(),
                                          "label_sources": item["label_sources"],
                                          "visually_in_comparison": key in VISIBLE_STANDARDS})
        bundle[iid] = item
    require(len(standard_receipts) == 11, "Incomplete preservation receipt")
    return manifests, audit, decisions, bundle, standard_receipts


def automatic_crop(item, rows):
    boxes = [item["points"]["r24"][row]["box"] for row in rows]
    return (max(0, math.floor(min(b[0] for b in boxes)) - 12),
            max(0, math.floor(min(b[1] for b in boxes)) - 12),
            min(1280, math.ceil(max(b[2] for b in boxes)) + 12),
            min(720, math.ceil(max(b[3] for b in boxes)) + 12))


def draw_marker(draw, center, label, color, side, panel_box):
    cx, cy = center
    radius = 6
    draw.ellipse((cx-radius-1, cy-radius-1, cx+radius+1, cy+radius+1), outline="#16313e", width=2)
    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=3)
    draw.line((cx-3, cy, cx+3, cy), fill=color, width=1)
    draw.line((cx, cy-3, cx, cy+3), fill=color, width=1)
    face = font(22)
    label_w = text_width(draw, label, face)
    label_x = cx + 11 if side == "left" else cx - label_w - 11
    label_x = max(panel_box[0]+3, min(label_x, panel_box[2]-label_w-3))
    label_y = max(panel_box[1]+2, cy-28)
    draw.text((label_x, label_y), label, font=face, fill=color, stroke_width=2, stroke_fill="#122938")


def sidebar(canvas, item, iid, decisions):
    draw = ImageDraw.Draw(canvas)
    x, width, y = 1872, 460, 215
    thumb = item["original"].resize((width, round(width*720/1280)), Image.Resampling.LANCZOS)
    draw.text((x, y-31), "Full original context", font=font(23), fill=PALETTE["ink"])
    canvas.paste(thumb, (x, y))
    y += thumb.height + 28
    split = "TRAINING IMAGE" if item["record"]["split"] == "train" else "DEVELOPMENT REVIEW ONLY"
    draw.rounded_rectangle((x, y, x+width, y+42), radius=7, fill="#e5eef4")
    draw.text((x+12, y+7), split, font=font(23), fill=PALETTE["ink"])
    y += 65
    changed = sorted((r for (im, r) in decisions if im == iid))
    draw.text((x, y), f"{len(changed)} left point{'s' if len(changed)>1 else ''} changed", font=font(29), fill=PALETTE["ink"])
    y += 46
    for row in changed:
        d = decisions[(iid, row)]
        suffix = " / adjacent" if d["scope"] == "adjacent_context" else " / named"
        draw.text((x, y), f"Row {row} L{suffix}", font=font(24), fill=PALETTE["ink"])
        y += 32
        old, new = item["points"]["r24"][row]["left"], item["points"]["r25"][row]["left"]
        draw.text((x, y), "R24 " + pixel_xy(old), font=font(23), fill="#b31c80")
        y += 30
        draw.text((x, y), "R25 " + pixel_xy(new), font=font(23), fill="#08794f")
        y += 47
    if iid == "new_000008":
        y = wrapped(draw, "Nine accepted standards\nRows 18-21 and 28-32: every saved token is identical.",
                    (x, y), width, 24) + 23
        y = wrapped(draw, "Row 35 right: still missing. The physical corner lies outside the image; no marker is drawn.",
                    (x, y), width, 23) + 23
        wrapped(draw, "This one correction belongs to the review reference only. It does not enter training.",
                (x, y), width, 23, PALETTE["muted"])
    elif iid == "new_000010":
        y = wrapped(draw, "Accepted standard\nRow 16 retains its entire saved line, including both endpoints.",
                    (x, y), width, 24) + 22
        wrapped(draw, "These three corrections belong to the review reference only. No trainer-validation or frozen scoring label changes.",
                (x, y), width, 23, PALETTE["muted"])
    else:
        note = "28R! is retained at (330.02, 705.32). The shadowed boundary is soft; retention is not a claim of exact placement."
        note_bottom = y+166
        draw.rounded_rectangle((x-12, y-8, x+width+12, note_bottom), radius=9, fill=PALETTE["amber"])
        note_text_bottom = wrapped(draw, note, (x, y), width, 23)
        y = max(note_text_bottom+32, note_bottom+24)
        wrapped(draw, "These four left-point corrections enter the eligible training candidate. Every right-point token is retained.",
                (x, y), width, 23, PALETTE["muted"])
    return {"thumbnail_source": item["image_source"], "thumbnail_unmarked": True}


def comparison_image(root, iid, item, decisions, page_number):
    canvas = Image.new("RGB", CANVAS_SIZE, PALETTE["page"])
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 2400, 13), fill=PALETTE["ink"])
    draw.text((56, 39), f"R25 / {iid}", font=font(45), fill=PALETTE["ink"])
    draw.text((56, 99), "Original pixels and the exact saved corner coordinates", font=font(27), fill=PALETTE["muted"])
    draw.text((56, 143), "Magenta: saved R24    Green: changed R25    Cyan: retained R25    * accepted standard    + adjacent amendment",
              font=font(24), fill=PALETTE["ink"])
    draw.line((56, 183, 2344, 183), fill=PALETTE["line"], width=2)
    left, column_w, gap = 56, 560, 31
    for col, title in enumerate(("ORIGINAL / unmarked", "R24 / previous saved", "R25 / current saved")):
        draw.text((left+col*(column_w+gap), 200), title, font=font(25), fill=PALETTE["ink"])
    top, crops, points_drawn = 248, [], []
    for group in GROUPS[iid]:
        rows, scale = group["rows"], group["scale"]
        box = group.get("crop") or automatic_crop(item, rows)
        crop = item["original"].crop(box)
        enlarged = crop.resize((crop.width*scale, crop.height*scale), Image.Resampling.NEAREST)
        require(enlarged.width <= column_w, f"Crop wider than its column: {iid}/{rows}")
        require(top+enlarged.height+70 < 1535, f"Crop group exceeds page area: {iid}/{rows}")
        draw.text((left, top), group["title"], font=font(24), fill=PALETTE["ink"])
        crop_y = top+37
        record = {"rows": list(rows), "source_crop_xyxy": list(box), "integer_enlargement": scale,
                  "title": group["title"], "source_crop_rgb_sha256": sha256(crop.tobytes()).hexdigest(),
                  "panels": [], "original_column_unmarked_pixel_match": True}
        for col, rev in enumerate((None, "r24", "r25")):
            panel_x = left+col*(column_w+gap)
            draw.rectangle((panel_x, crop_y, panel_x+column_w, crop_y+enlarged.height), fill=PALETTE["panel"])
            offset_x = panel_x+(column_w-enlarged.width)//2
            canvas.paste(enlarged, (offset_x, crop_y))
            pixel_box = (offset_x, crop_y, offset_x+enlarged.width, crop_y+enlarged.height)
            if rev is None:
                require(canvas.crop(pixel_box).tobytes() == enlarged.tobytes(), "Original panel pixels were altered")
            else:
                for row in rows:
                    for side, short in (("left", "L"), ("right", "R")):
                        x, y, visibility = item["points"][rev][row][side]
                        if visibility == 0:
                            continue
                        if not (box[0] <= x < box[2] and box[1] <= y < box[3]):
                            continue
                        old = item["points"]["r24"][row][side]
                        new = item["points"]["r25"][row][side]
                        changed = old != new
                        color = PALETTE["old" if rev == "r24" else "changed" if changed else "retained"]
                        suffix = "*" if (iid, row) in VISIBLE_STANDARDS else "+" if decisions[(iid, row)]["scope"] == "adjacent_context" else ""
                        if iid == "new_000012" and row == 28 and side == "right":
                            suffix += "!"
                        label = f"{row}{short}{suffix}"
                        draw_marker(draw, (offset_x+(x-box[0])*scale, crop_y+(y-box[1])*scale),
                                    label, color, side, pixel_box)
                        points_drawn.append({"revision": rev, "row": row, "side": side, "xy": [x, y], "label": label})
            record["panels"].append({"revision": rev or "original", "canvas_xyxy": list(pixel_box)})
        crop_bottom = crop_y+enlarged.height
        draw.text((left, crop_bottom+7), f"Source crop {tuple(box)}  |  original pixels enlarged {scale}x; no generated detail",
                  font=font(18), fill=PALETTE["muted"])
        crops.append(record)
        top = crop_bottom+45
    side_receipt = sidebar(canvas, item, iid, decisions)
    draw.line((56, 1550, 2344, 1550), fill=PALETTE["line"], width=2)
    footer = "Colour records edit status, not certainty. Boxes, classes, visibility, row identity and split remain unchanged."
    draw.text((56, 1574), footer, font=font(23), fill=PALETTE["ink"])
    note = ("012:28 right remains uncertain; no sharper coordinate was invented. Left placement ambiguity is about 2-3 px."
            if iid == "new_000012" else "Rounded front-face turns retain about 1-2 original pixels of qualitative placement ambiguity.")
    draw.text((56, 1614), note, font=font(22), fill=PALETTE["muted"])
    draw.text((2233, 1614), f"{page_number} / 4", font=font(22), fill=PALETTE["muted"])
    for record in crops:
        expected = item["original"].crop(record["source_crop_xyxy"])
        scale = record["integer_enlargement"]
        expected = expected.resize((expected.width*scale, expected.height*scale), Image.Resampling.NEAREST)
        require(canvas.crop(record["panels"][0]["canvas_xyxy"]).tobytes() == expected.tobytes(),
                "Final original column differs from its unmarked source crop")
    path = root / LEDGER / f"{iid}-front-face-comparison.png"
    canvas.save(path, optimize=True)
    require(Image.open(path).convert("RGB").tobytes() == canvas.tobytes(), "PNG pixels changed on readback")
    return {**descriptor(root, path), "image_id": iid, "pdf_page": page_number,
            "image_size": list(canvas.size), "source_image": item["image_source"],
            "label_sources": item["label_sources"], "crop_groups": crops,
            "markers": points_drawn, "context_thumbnail": side_receipt,
            "png_rgb_sha256": sha256(canvas.tobytes()).hexdigest()}


def pdf_paragraph(pdf, text, x, y, width, size=12, leading=17, color="#526675"):
    pdf.setFillColor(colors.HexColor(color))
    pdf.setFont("ReviewSans", size)
    line = ""
    for word in text.split():
        proposed = (line+" "+word).strip()
        if line and pdfmetrics.stringWidth(proposed, "ReviewSans", size) > width:
            pdf.drawString(x, y, line)
            y -= leading
            line = word
        else:
            line = proposed
    if line:
        pdf.drawString(x, y, line)
    return y-leading


def create_pdf(root, pdf_path, manifests, decisions, bundle, standard_receipts, artifacts):
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("ReviewSans", regular))
    pdfmetrics.registerFont(TTFont("ReviewBold", bold))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = landscape(A3)
    pdf = pdf_canvas.Canvas(str(pdf_path), pagesize=(width, height), pageCompression=1, invariant=1)
    pdf.setTitle("R25 - focused front-face corner amendments")
    pdf.setAuthor("Cassette vision project")
    pdf.setSubject("Eight saved left-corner amendments with R24/R25 original-pixel comparisons")
    pdf.setFillColor(colors.HexColor(PALETTE["ink"]))
    pdf.rect(0, height-9, width, 9, fill=1, stroke=0)
    x, top, content_w = 44, height-65, width-88
    pdf.setFont("ReviewBold", 34)
    pdf.drawString(x, top, "R25 | front-face corner amendments")
    pdf.setFont("ReviewSans", 14)
    pdf.setFillColor(colors.HexColor(PALETTE["muted"]))
    pdf.drawString(x, top-31, "23 November 2026  /  saved R24 to saved R25  /  compact visual amendment record")
    card_y, card_h, card_gap = top-143, 80, 18
    card_w = (content_w-card_gap*2)/3
    cards = [("8 left points", "5 named rows + 3 related adjacent rows"),
             ("3 photographs", "4 training changes + 4 development changes"),
             ("11 standards", "Complete accepted rows preserved exactly")]
    for col, (title, subtitle) in enumerate(cards):
        cx = x+col*(card_w+card_gap)
        pdf.setFillColor(colors.HexColor("#eef3f6"))
        pdf.roundRect(cx, card_y, card_w, card_h, 8, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor(PALETTE["ink"]))
        pdf.setFont("ReviewBold", 24)
        pdf.drawString(cx+17, card_y+45, title)
        pdf.setFont("ReviewSans", 11)
        pdf.drawString(cx+17, card_y+20, subtitle)
    y = card_y-40
    pdf.setFillColor(colors.HexColor(PALETTE["ink"]))
    pdf.setFont("ReviewBold", 16)
    pdf.drawString(x, y, "Complete amendment inventory")
    y -= 30
    columns = [x, x+220, x+395, x+645, x+880]
    headings = ["Photograph / row", "Review scope", "R24 left (x, y) px", "R25 left (x, y) px", "Data role"]
    pdf.setFillColor(colors.HexColor(PALETTE["ink"]))
    pdf.rect(x, y-13, content_w, 36, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("ReviewBold", 12)
    for cx, title in zip(columns, headings):
        pdf.drawString(cx+9, y, title)
    y -= 47
    for index, ((iid, row), decision) in enumerate(sorted(decisions.items())):
        if index % 2 == 0:
            pdf.setFillColor(colors.HexColor("#f3f6f8"))
            pdf.rect(x, y-10, content_w, 34, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor(PALETTE["ink"]))
        pdf.setFont("ReviewSans", 12)
        old, new = bundle[iid]["points"]["r24"][row]["left"], bundle[iid]["points"]["r25"][row]["left"]
        values = [f"{iid} / {row}", "Adjacent context" if decision["scope"] == "adjacent_context" else "User named",
                  pixel_xy(old), pixel_xy(new), "Training" if decision["split"] == "train" else "Development only"]
        for cx, value in zip(columns, values):
            pdf.drawString(cx+9, y, value)
        y -= 34
    y -= 18
    y = pdf_paragraph(pdf, "Protected complete rows: new_000008 rows 18-21 and 28-32; new_000010 row 16; new_000016 row 24. The first ten appear on the comparison pages. The remaining row is verified by exact line and label hashes below and in the render receipt.",
                      x, y, content_w, 12)
    standard = next(r for r in standard_receipts if r["image_id"] == "new_000016")
    y = pdf_paragraph(pdf, "016:24 exact-line SHA-256 (UTF-8, no newline): "+standard["exact_line_sha256_utf8_no_newline"],
                      x, y-5, content_w, 9.5, 14)
    y -= 10
    pdf.setFillColor(colors.HexColor(PALETTE["amber"]))
    pdf.roundRect(x, y-50, content_w, 71, 6, fill=1, stroke=0)
    pdf_paragraph(pdf, "Retained uncertainty: 012:28 right stays at (330.02, 705.32). Shadow and the adjacent cassette layer merge into a soft boundary; no better-supported exact replacement was established. 008:35 right stays missing because it is outside the frame.",
                  x+13, y+3, content_w-26, 12, 18, PALETTE["ink"])
    y -= 78
    pdf_paragraph(pdf, "Read the next three pages left to right: unmarked original, saved R24, saved R25. Magenta marks previous coordinates; green marks changes; cyan marks retained points. All original trainer-validation and frozen 666/663 reference labels remain unchanged. These annotation amendments do not establish a model-performance gain.",
                  x, y, content_w, 11.5, 17)
    pdf.setFont("ReviewSans", 9)
    pdf.setFillColor(colors.HexColor(PALETTE["muted"]))
    pdf.drawString(x, 26, "Source images and saved label bytes verified against their manifests; no model predictions used as targets.")
    pdf.drawRightString(width-x, 26, "1 / 4")
    pdf.showPage()
    for artifact in artifacts:
        image = Image.open(root / artifact["path"]).convert("RGB")
        factor = min((width-36)/image.width, (height-28)/image.height)
        draw_w, draw_h = image.width*factor, image.height*factor
        pdf.drawImage(ImageReader(image), (width-draw_w)/2, (height-draw_h)/2,
                      width=draw_w, height=draw_h, mask=None)
        pdf.showPage()
    pdf.save()
    require(pdf_path.stat().st_size < 50_000_000, "PDF exceeds the requested 50 MB bound")


def verify_pdf_pixels(pdf_path, root, artifacts):
    import fitz
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    require(len(reader.pages) == 4, "Expected four PDF pages")
    text = reader.pages[0].extract_text()
    require("Complete amendment inventory" in text and "330.02, 705.32" in text, "Cover text missing")
    doc = fitz.open(pdf_path)
    checks = []
    for page_no, artifact in enumerate(artifacts, 1):
        candidates = [im for im in doc[page_no].get_images(full=True) if im[2:4] == CANVAS_SIZE]
        require(len(candidates) == 1, f"Expected one whole comparison image on page {page_no+1}")
        pixmap = fitz.Pixmap(doc, candidates[0][0])
        require(pixmap.n == 3 and not pixmap.alpha, "Unexpected PDF image colour conversion")
        require(sha256(pixmap.samples).hexdigest() == artifact["png_rgb_sha256"], "PDF changed comparison image pixels")
        checks.append({"page": page_no+1, "comparison_rgb_exact": True,
                       "rgb_sha256": artifact["png_rgb_sha256"]})
    doc.close()
    return {"page_count": 4, "cover_text_checked": True, "comparison_image_checks": checks,
            "all_comparison_pixels_lossless_in_pdf": True}


def main(root, pdf_path):
    root, pdf_path = root.resolve(), pdf_path.resolve()
    manifests, audit, decisions, bundle, standards = source_bundle(root)
    out_dir = root / LEDGER
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [comparison_image(root, iid, bundle[iid], decisions, page)
                 for page, iid in enumerate(PHOTO_ORDER, 2)]
    displayed = {(a["image_id"], m["row"]) for a in artifacts for m in a["markers"] if m["revision"] == "r25"}
    require(set(decisions) | VISIBLE_STANDARDS <= displayed, "Incomplete changed-row or standard coverage")
    require(not any(a["image_id"] == "new_000008" and m["row"] == 35 and m["side"] == "right"
                    for a in artifacts for m in a["markers"]), "Missing right endpoint was drawn")
    create_pdf(root, pdf_path, manifests, decisions, bundle, standards, artifacts)
    pdf_check = verify_pdf_pixels(pdf_path, root, artifacts)
    receipt = {
        "schema_version": 1, "kind": "r25_compact_amendment_pdf_and_comparisons",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_id": manifests["r25"]["review_id"],
        "renderer": descriptor(root, Path(__file__).resolve()),
        "shared_helpers": descriptor(root, root/"scripts/render_front_review_r24.py"),
        "verification_scope": "Three affected photographs, their full R24/R25 label files, and the exact protected 016:24 line.",
        "source_manifests": {rev: descriptor(root, root/path) for rev, path in REVIEWS.items()},
        "point_review": descriptor(root, root/LEDGER/"point-review.json"),
        "policy": descriptor(root, root/"docs/annotation-boundaries-r25.md"),
        "source_and_label_hashes_verified": True,
        "saved_coordinates_verified": True,
        "unchanged_fields_and_all_right_tokens_verified": True,
        "all_eight_changes_visually_included": True,
        "original_panels_unmarked_and_pixel_exact": True,
        "missing_points_not_drawn": True,
        "accepted_standards": standards,
        "artifacts": artifacts,
        "pdf": {"path": str(pdf_path), "sha256": digest(pdf_path), "size_bytes": pdf_path.stat().st_size,
                "page_count": 4, "page_size": "A3 landscape", **pdf_check},
        "visual_review": {"status": "awaiting_poppler_render_and_visual_inspection", "all_four_pages_required": True},
    }
    receipt_path = out_dir / "rendering-manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"pdf": str(pdf_path), "size_bytes": pdf_path.stat().st_size,
                      "render_receipt": str(receipt_path), "pngs": [a["path"] for a in artifacts]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.pdf)
