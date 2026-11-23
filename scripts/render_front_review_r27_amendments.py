#!/usr/bin/env python3
"""Render every accepted R27 amendment from hash-bound original image crops."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

WIDTH, HEIGHT = landscape(A4)
MARGIN = 34
INK = colors.HexColor("#142C3A")
MUTED = colors.HexColor("#536674")
RULE = colors.HexColor("#D8E2E6")
PALE = colors.HexColor("#F0F5F6")
BEFORE = colors.HexColor("#B44278")
AFTER = colors.HexColor("#007F72")
GROUPS = [
    ("new_000007", [24]), ("new_000007", [33]),
    ("new_000007", [37]), ("new_000007", [41, 42]),
    ("new_000008", [33, 34, 35]), ("new_000009", [48, 49, 50, 51, 52]),
    ("new_000013", [27]), ("new_000014", [35]), ("new_000016", [26, 27]),
    ("new_000049", [3]), ("new_000051", [20]), ("new_000056", [12]),
    ("new_000083", [12]),
]
REASONS = {
    ("new_000007", 24): "Remove the grey rail strip at the left. Move the left point onto the visible front turn, above and inside the previous shadow-side point. The model only indicated where to inspect.",
    ("new_000007", 33): "The previous left edge includes adjacent material, and its lower edge extends into the next cassette. Tighten the box around this front; both existing points remain supported.",
    ("new_000007", 37): "The previous box includes the left separator and part of the next front below. The two saved points are retained.",
    ("new_000007", 41): "Row 41: extend xmax to the outer visible rim at 926. A farther model edge would include more dark rack. Row 42: remove the left separator and move the left point onto its own rounded front turn. An outer box edge and a lower-front corner need not coincide.",
    ("new_000008", 33): "Rows 33-35: remove the excessive left rack/neighbor region. Move the left points in rows 33 and 34 to their visible turns, and trim their lower boxes at the local separator. Keep xmax at 1280: these fronts continue beyond the image edge. A smaller predicted xmax is not supported by the visible clipping.",
    ("new_000009", 48): "Use each front's own dark separator to place the vertical box bounds. The old lower edges extend into the next cassette. These are local visible boundaries, not a fixed spacing rule. The one-pixel top change in row 49 is less certain than the larger lower-edge cleanup.",
    ("new_000013", 27): "Remove a clearly excessive left separator strip. Keep the existing left point and the earlier R26 corrections to rows 29-31.",
    ("new_000014", 35): "Move the left point from the dark exterior onto the rounded lower-front turn. Keep xmin: the higher, shadowed cassette side is real material and must not be discarded by a white-only rule.",
    ("new_000016", 26): "Tighten both left box edges to the sloping cassette outline. Preserve shadowed material inside that outline. The existing corner points and protected standard row 24 remain unchanged.",
    ("new_000049", 3): "The right point is below and outside the visible lower-front turn, in the rail/shadow gap. Move only that point onto the turn; preserve the box and other endpoint.",
    ("new_000051", 20): "The right point lies in the dark rack junction. Move it to the visible front-to-side turn just above and inside that strip. No global inward offset is applied.",
    ("new_000056", 12): "The left point lies on the dark rail beside the lower front edge. Move it to the visible turn. The gripper-hidden right point remains unavailable, with its original zero tokens.",
    ("new_000083", 12): "The right point lies in the rail/lower-shadow junction. Move it onto the front's visible turn. Local brightness contrast flagged this case but did not determine the new coordinate.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register_fonts():
    base = Path("/usr/share/fonts/truetype/open-sans")
    for name, filename, fallback in (("Amend", "OpenSans-Regular.ttf", "Helvetica"),
                                     ("AmendBold", "OpenSans-Semibold.ttf", "Helvetica-Bold")):
        if (base / filename).exists():
            pdfmetrics.registerFont(TTFont(name, str(base / filename)))
        else:
            pdfmetrics.registerFont(pdfmetrics.Font(name, fallback, "WinAnsiEncoding"))


def text(pdf, value, x, y, size=9, color=INK, bold=False):
    pdf.setFillColor(color)
    pdf.setFont("AmendBold" if bold else "Amend", size)
    pdf.drawString(x, y, value)


def para(pdf, value, x, top, width, size=9, leading=13, color=INK):
    block = Paragraph(value, ParagraphStyle("body", fontName="Amend", fontSize=size,
                                           leading=leading, textColor=color))
    _, height = block.wrap(width, HEIGHT)
    block.drawOn(pdf, x, top - height)
    return top - height


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH - MARGIN, 31)
    text(pdf, "R26 TO R27  /  ORIGINAL-PIXEL ANNOTATION REVIEW  /  23 NOV 2026", MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7)
    pdf.drawRightString(WIDTH - MARGIN, 18, f"{page} / {pages}")


def geometry(line):
    token = list(map(float, line.split()))
    require(len(token) == 11, "Expected class + box + two pose triplets")
    cx, cy, w, h = token[1:5]
    return {"box": [(cx - w / 2) * 1280, (cy - h / 2) * 720,
                    (cx + w / 2) * 1280, (cy + h / 2) * 720],
            "points": [[token[5] * 1280, token[6] * 720, int(token[7])],
                       [token[8] * 1280, token[9] * 720, int(token[10])]]}


def display(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def field_changes(row):
    old, new = geometry(row["before_line"]), geometry(row["after_line"])
    result = []
    for i, name in enumerate(("xmin", "ymin", "xmax", "ymax")):
        if abs(old["box"][i] - new["box"][i]) > 1e-5:
            result.append((row["row"], name, display(old["box"][i]), display(new["box"][i])))
    for i, name in enumerate(("left point", "right point")):
        if max(abs(old["points"][i][k] - new["points"][i][k]) for k in (0, 1)) > 1e-5:
            before = "(" + ", ".join(display(v) for v in old["points"][i][:2]) + ")"
            after = "(" + ", ".join(display(v) for v in new["points"][i][:2]) + ")"
            result.append((row["row"], name, before, after))
    return result


def bounds_for(rows):
    boxes = [geometry(row[key])["box"] for row in rows for key in ("before_line", "after_line")]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    margin_x = max(16, min(36, (x1 - x0) * .16))
    margin_y = max(20, min(42, (y1 - y0) * .55))
    return [max(0, math.floor(x0 - margin_x)), max(0, math.floor(y0 - margin_y)),
            min(1280, math.ceil(x1 + margin_x)), min(720, math.ceil(y1 + margin_y))]


def draw_panel(pdf, crop, bounds, rows, mode, x, top, panel_width, max_height):
    scale = min(panel_width / crop.width, max_height / crop.height)
    dw, dh = crop.width * scale, crop.height * scale
    ox, oy = x + (panel_width - dw) / 2, top - dh
    # Enlarged raster pixels remain the original observations; overlays are vectors.
    enlarged = crop.resize((max(crop.width, math.ceil(dw * 3)), max(crop.height, math.ceil(dh * 3))),
                           Image.Resampling.NEAREST)
    pdf.drawImage(ImageReader(enlarged), ox, oy, dw, dh)
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.5)
    pdf.rect(ox, oy, dw, dh, stroke=1, fill=0)
    if mode:
        color = BEFORE if mode == "before_line" else AFTER
        for row in rows:
            geo = geometry(row[mode])
            a, b, c, d = geo["box"]
            px, py = ox + (a - bounds[0]) * scale, oy + (bounds[3] - d) * scale
            pdf.setStrokeColor(color)
            pdf.setLineWidth(.95)
            pdf.rect(px, py, (c - a) * scale, (d - b) * scale, stroke=1, fill=0)
            for qx, qy, visible in geo["points"]:
                if visible:
                    xx, yy = ox + (qx - bounds[0]) * scale, oy + (bounds[3] - qy) * scale
                    pdf.circle(xx, yy, 2.6, stroke=1, fill=0)
            label = str(row["row"])
            lx, ly = px + 2, oy + (bounds[3] - b) * scale - 10
            pdf.setFillColor(color)
            pdf.rect(lx - 1, ly - 2, pdfmetrics.stringWidth(label, "AmendBold", 8) + 4, 11,
                     stroke=0, fill=1)
            text(pdf, label, lx + 1, ly, 8, colors.white, True)
    return dh


def cover(pdf, inventory, pages):
    text(pdf, "ANNOTATION EVIDENCE  /  COMPLETE ACCEPTED CHANGE SET", MARGIN, HEIGHT - 40, 9, MUTED, True)
    text(pdf, "R27 | Annotation amendments", MARGIN, HEIGHT - 79, 27, INK, True)
    y = para(pdf, "Every changed row is shown with the same original crop, the saved R26 annotation, and the accepted R27 amendment. Neighboring fronts and rack pixels provide context. Retained-only photographs are omitted.",
             MARGIN, HEIGHT - 98, WIDTH - 2 * MARGIN, 11, 16)
    cards = [(21, "changed rows"), (10, "photographs"), (7, "training rows"), (14, "development rows")]
    card_width = (WIDTH - 2 * MARGIN - 36) / 4
    for i, (number, label) in enumerate(cards):
        x = MARGIN + i * (card_width + 12)
        pdf.setFillColor(PALE)
        pdf.roundRect(x, y - 80, card_width, 62, 4, fill=1, stroke=0)
        text(pdf, str(number), x + 12, y - 47, 23, INK, True)
        text(pdf, label, x + 12, y - 67, 9, MUTED)
    y -= 106
    left_w = 370
    text(pdf, "How the coordinates were chosen", MARGIN, y, 12, INK, True)
    yy = para(pdf, "Original pixels decide the physical front contour. Predictions and local color contrast locate questions; neither supplies annotation truth. Include visible cassette material, including its shadowed side, and exclude the rack, gripper, and neighboring front. The lower-front turn may differ from the outermost box edge.",
              MARGIN, y - 15, left_w, 9.5, 14)
    text(pdf, "What the color check established", MARGIN, yy - 25, 12, INK, True)
    yy = para(pdf, "All 157 images and 8,062 locatable endpoints were screened. 8,050 had complete local patch comparisons; 12 were at image borders. Exactly eight additional, high-ranked cases on distinct photos were visually inspected: four amendments, two valid dark-edge points retained, and two uncertain points retained. This selected sample does not estimate dataset error rate or cue precision.",
              MARGIN, yy - 40, left_w, 9.5, 14)
    right_x = 438
    text(pdf, "Changed photographs and rows", right_x, y, 12, INK, True)
    by_image = {}
    for item in inventory["changed_rows"]:
        by_image.setdefault((item["image_id"], item["split"]), []).append(item["row"])
    tab_y = y - 23
    for (image_id, split), rows in sorted(by_image.items()):
        text(pdf, image_id.replace("new_000", ""), right_x, tab_y, 9, INK, True)
        text(pdf, "train" if split == "train" else "dev", right_x + 45, tab_y, 8.5, MUTED)
        text(pdf, ", ".join(map(str, rows)), right_x + 91, tab_y, 9, INK)
        tab_y -= 18
    yy = min(yy, tab_y) - 20
    yy = para(pdf, "<b>Preserved:</b> every object identity, the 11 protected standards, class and visibility tokens, unavailable-point zero triplets, image splits, original trainer validation, and frozen evaluation references. No row is deleted. Coordinates are original 1280 x 720 pixels; typical visual uncertainty is roughly 1-2 px, with softer edge cases noted in the ledger. Displayed values are rounded; exact saved lines and source hashes are in the companion inventory and ledger.",
              MARGIN, yy, WIDTH - 2 * MARGIN, 9, 13)
    require(yy > 43, "Cover overlaps footer")
    footer(pdf, 1, pages)
    pdf.showPage()


def case_page(pdf, group, rows, image, page, pages, crop_dir):
    image_id, ids = group
    name = "row " + str(ids[0]) if len(ids) == 1 else "rows " + ", ".join(map(str, ids))
    split = "TRAINING" if rows[0]["split"] == "train" else "DEVELOPMENT"
    text(pdf, f"{split}  /  R26 TO R27", MARGIN, HEIGHT - 33, 8, MUTED, True)
    text(pdf, f"{image_id} | {name}", MARGIN, HEIGHT - 62, 21, INK, True)
    bounds = bounds_for(rows)
    crop = image.crop(bounds)
    crop_name = image_id + "-rows-" + "-".join(map(str, ids)) + ".png"
    crop.save(crop_dir / crop_name)
    changes = [item for row in rows for item in field_changes(row)]
    panel_height = min(233, 360 - 17 * len(changes))
    panel_w = (WIDTH - 2 * MARGIN - 24) / 3
    panel_top = HEIGHT - 104
    label_y = panel_top + 10
    heights = []
    for i, (title, mode, color) in enumerate((("Original pixels", None, MUTED),
                                             ("R26 | previous annotation", "before_line", BEFORE),
                                             ("R27 | accepted amendment", "after_line", AFTER))):
        x = MARGIN + i * (panel_w + 12)
        text(pdf, title, x, label_y, 9, color, True)
        heights.append(draw_panel(pdf, crop, bounds, rows, mode, x, panel_top, panel_w, panel_height))
    y = panel_top - max(heights) - 23
    text(pdf, "Changed coordinates | original pixels", MARGIN, y, 10, INK, True)
    text(pdf, "Visible basis for the amendment", 446, y, 10, INK, True)
    ty = y - 20
    columns = [MARGIN + 4, MARGIN + 40, MARGIN + 130, MARGIN + 258]
    pdf.setFillColor(PALE)
    pdf.rect(MARGIN, ty - 4, 389, 17, fill=1, stroke=0)
    for x, label in zip(columns, ("Row", "Field", "R26", "R27")):
        text(pdf, label, x, ty + 1, 8, MUTED, True)
    ty -= 17
    for row, field, before, after in changes:
        for x, label in zip(columns, (str(row), field, before, after)):
            text(pdf, label, x, ty, 8.4, INK)
        pdf.setStrokeColor(RULE)
        pdf.setLineWidth(.35)
        pdf.line(MARGIN, ty - 5, MARGIN + 389, ty - 5)
        ty -= 17
    reason = escape(REASONS[(image_id, ids[0])])
    py = para(pdf, reason, 446, y - 13, WIDTH - MARGIN - 446, 9, 13)
    py = para(pdf, "Rings mark available lower-front corners; boxes enclose the visible front. No location is drawn for an unavailable corner. Typical coordinate uncertainty is about 1-2 original pixels.",
              446, py - 12, WIDTH - MARGIN - 446, 8, 11.5, MUTED)
    end_y = min(ty, py) - 14
    text(pdf, "Same crop and scale in all panels. Context crop: " + str(bounds) + " (x0, y0, x1, y1).",
         MARGIN, end_y, 7.5, MUTED)
    require(end_y > 43, f"Case page overlaps footer: {image_id}:{ids}")
    footer(pdf, page, pages)
    pdf.showPage()
    return {"page": page, "image_id": image_id, "rows": ids, "crop_xyxy": bounds,
            "original_crop_file": crop_name, "original_crop_sha256": sha(crop_dir / crop_name),
            "changed_coordinates_display": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    ledger_path = root / "results/review/front-review-20261123-r27/row-review.json"
    manifest_path = root / "data/reviews/fronts-2026-11-23-r27/manifest.json"
    ledger = json.loads(ledger_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    rows = [item for item in ledger["rows"] if item["action"] == "replace"]
    by_id = {(r["image_id"], r["row"]): r for r in rows}
    requested = [(image_id, row) for image_id, ids in GROUPS for row in ids]
    require(len(by_id) == len(rows) == len(requested) == 21 and set(by_id) == set(requested),
            "Every accepted amendment must appear exactly once, without retained rows")
    require(Counter(r["split"] for r in rows) == {"train": 7, "val": 14}, "Intervention scope differs")
    files = {item["image_id"]: item for item in manifest["files"]}
    images = {}
    for row in rows:
        image_id = row["image_id"]
        path = root / "data/images" / row["split"] / (image_id + ".png")
        require(sha(path) == row["source_image_sha256"] == files[image_id]["source_image_sha256"],
                "Source image drift: " + image_id)
        label_path = manifest_path.parent / files[image_id]["label_path"]
        require(sha(label_path) == files[image_id]["label_sha256"], "R27 label drift")
        require(label_path.read_text().splitlines()[row["row"] - 1] == row["after_line"], "R27 row drift")
        image = Image.open(path).convert("RGB")
        require(image.size == (1280, 720), "Source image dimensions differ")
        images[image_id] = image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.crop_dir.mkdir(parents=True, exist_ok=True)
    register_fonts()
    inventory = {"schema_version": 1, "kind": "complete_r27_annotation_amendment_pdf",
                 "reference_before": "fronts-2026-11-23-r26", "reference_after": manifest["review_id"],
                 "ledger": {"path": str(ledger_path.relative_to(root)), "sha256": sha(ledger_path)},
                 "review_manifest": {"path": str(manifest_path.relative_to(root)), "sha256": sha(manifest_path)},
                 "renderer": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha(__file__)},
                 "summary": {"changed_rows": 21, "changed_photos": len(images), "training_rows": 7,
                             "development_rows": 14, "deleted_rows": 0, "case_pages": len(GROUPS)},
                 "changed_rows": rows, "pages": [],
                 "verification": {"original_images_hash_verified": True, "every_changed_row_exactly_once": True,
                                  "saved_r27_lines_verified": True, "unavailable_corners_not_drawn": True,
                                  "predictions_not_drawn": True}}
    pdf = canvas.Canvas(str(args.output), pagesize=(WIDTH, HEIGHT), pageCompression=1, invariant=1)
    pdf.setTitle("R27 Annotation Amendments - Original Pixel Review")
    pdf.setAuthor("Siuuuers/dl project review")
    pages = len(GROUPS) + 1
    cover(pdf, inventory, pages)
    for page, group in enumerate(GROUPS, start=2):
        image_id, ids = group
        group_rows = [by_id[(image_id, row)] for row in ids]
        inventory["pages"].append(case_page(pdf, group, group_rows, images[image_id], page, pages, args.crop_dir))
    pdf.save()
    inventory["pdf"] = {"filename": args.output.name, "sha256": sha(args.output), "pages": pages,
                        "bytes": args.output.stat().st_size}
    inventory_path = args.inventory or root / "results/review/front-review-20261123-r27/amendment-pdf-inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"pdf": str(args.output), **inventory["pdf"], "inventory": str(inventory_path)}, indent=2))


if __name__ == "__main__":
    main()
