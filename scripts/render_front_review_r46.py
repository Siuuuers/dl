#!/usr/bin/env python3
"""Render the R46 recorded front review as a compact, offline PDF.

Original JPEG pixels are enlarged with nearest-neighbor sampling. All boxes,
corner markers and supplied diagnostic outlines are drawn as PDF vectors.
No polygon is inferred from a rectangle. This renderer makes no training or
physical-accuracy claim; authoritative decisions live in the review manifest.
"""
import argparse
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

FONT_PAIRS = [
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    (Path(reportlab.__file__).parent / "fonts/Vera.ttf", Path(reportlab.__file__).parent / "fonts/VeraBd.ttf"),
]
for REGULAR_FONT, BOLD_FONT in FONT_PAIRS:
    if REGULAR_FONT.is_file() and BOLD_FONT.is_file():
        break
else:
    raise RuntimeError("Install an embeddable TrueType font pair (DejaVu Sans or ReportLab Vera)")
pdfmetrics.registerFont(TTFont("R46Sans", str(REGULAR_FONT)))
pdfmetrics.registerFont(TTFont("R46Sans-Bold", str(BOLD_FONT)))
pdfmetrics.registerFontFamily("R46Sans", normal="R46Sans", bold="R46Sans-Bold", italic="R46Sans", boldItalic="R46Sans-Bold")

PAGE_W, PAGE_H = landscape(A4)
MARGIN = 30.0
WIDTH = PAGE_W - 2 * MARGIN
INK = colors.HexColor("#19383f")
MUTED = colors.HexColor("#53676c")
LIGHT = colors.HexColor("#e3ebed")
PALE = colors.HexColor("#f4f8f8")
CYAN = colors.HexColor("#08a5bd")
MAGENTA = colors.HexColor("#d21d9e")
LEFT = colors.HexColor("#09b442")
RIGHT = colors.HexColor("#0875ed")
AMBER = colors.HexColor("#a65d00")
FIELDS = ("xmin", "ymin", "xmax", "ymax", "left_point", "right_point")
REQUESTS = (
    "frame_0001:source:001", "frame_0001:source:008", "frame_0001:source:015",
    "frame_0001:source:023", "frame_0001:source:030", "frame_0160:source:005",
    "frame_0160:source:039", "frame_0038:source:008", "frame_0038:source:009",
    "frame_0074:source:027", "frame_0074:source:043", "frame_0128:source:029",
    "frame_0128:source:039", "frame_0128:source:041", "frame_0146:source:025",
    "frame_0178:source:006", "frame_0225:source:003",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_by_id(manifest):
    return {r["stable_id"]: r for f in manifest["frames"]
            for r in f["rows"] + f.get("additions", [])}


def geometry(line):
    if line is None:
        return None
    a = [float(x) for x in line.split()]
    if len(a) != 11:
        raise ValueError("Expected one class, one box and two XYV slots")
    return dict(xmin=(a[1] - a[3] / 2) * 1280,
                ymin=(a[2] - a[4] / 2) * 720,
                xmax=(a[1] + a[3] / 2) * 1280,
                ymax=(a[2] + a[4] / 2) * 720,
                left_point=[a[5] * 1280, a[6] * 720],
                right_point=[a[8] * 1280, a[9] * 720],
                visibility=[int(a[7]), int(a[10])])


def number(value):
    # Eight-decimal normalized labels introduce subpixel numerical rounding.
    # Show six source-pixel decimals; this is serialization, not claimed accuracy.
    if abs(value - round(value)) < 0.0001:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def value_text(value):
    if isinstance(value, list):
        return "(" + ", ".join(number(x) for x in value) + ")"
    return number(value)


def para(c, text, x, top, width, size=9, leading=None, color=INK,
         max_height=None, bold=False):
    """Draw top-aligned, measured text; fail instead of clipping silently."""
    style = ParagraphStyle("r46", fontName="R46Sans-Bold" if bold else "R46Sans",
                           fontSize=size, leading=leading or size * 1.28,
                           textColor=color, spaceAfter=0, spaceBefore=0,
                           splitLongWords=True)
    p = Paragraph(text, style)
    _, height = p.wrap(width, 10000)
    if max_height is not None and height > max_height + 0.1:
        raise ValueError(f"PDF text does not fit: {height:.1f}>{max_height:.1f}: {text[:80]}")
    p.drawOn(c, x, top - height)
    return height


def fit_para(c, text, x, top, width, max_height, size=8.4, minimum=7.0, color=INK):
    for i in range(15):
        actual = max(minimum, size - i * 0.1)
        style = ParagraphStyle("fit", fontName="R46Sans", fontSize=actual,
                               leading=actual * 1.19, textColor=color)
        p = Paragraph(text, style)
        _, height = p.wrap(width, 10000)
        if height <= max_height:
            p.drawOn(c, x, top - height)
            return height
        if actual <= minimum:
            break
    raise ValueError("Annotation text exceeds the measured card space: " + text[:120])


def page_header(c, title, subtitle):
    c.setFillColor(INK)
    c.setFont("R46Sans-Bold", 19)
    c.drawString(MARGIN, PAGE_H - 37, title)
    c.setFont("R46Sans", 8.4)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, PAGE_H - 53, subtitle)
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.6)
    c.line(MARGIN, PAGE_H - 61, PAGE_W - MARGIN, PAGE_H - 61)


def footer(c, text="Assistant-reviewed proposals | Original pixels | No model promotion"):
    c.setStrokeColor(LIGHT)
    c.line(MARGIN, 25, PAGE_W - MARGIN, 25)
    c.setFillColor(MUTED)
    c.setFont("R46Sans", 7)
    c.drawString(MARGIN, 14, text)
    c.drawRightString(PAGE_W - MARGIN, 14, str(c.getPageNumber()))


def draw_legend(c, x, y, small=False):
    font = 7 if small else 8.3
    parts = [(CYAN, "box / retained"), (LEFT, "left corner"), (RIGHT, "right corner"),
             (MAGENTA, "R46 change"), (AMBER, "supplied review trace")]
    for color, label in parts:
        c.setFillColor(color)
        c.rect(x, y - 1, 8, 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("R46Sans", font)
        c.drawString(x + 12, y - 2, label)
        x += c.stringWidth(label, "R46Sans", font) + 29


def crop_region(lines, outline=None, margin=10):
    points = []
    for line in lines:
        g = geometry(line)
        if g:
            points += [(g["xmin"], g["ymin"]), (g["xmax"], g["ymax"])]
            for idx, field in enumerate(("left_point", "right_point")):
                if g["visibility"][idx]:
                    points.append(g[field])
    if outline:
        points += outline["vertices_px"]
    return (max(0, math.floor(min(p[0] for p in points)) - margin),
            max(0, math.floor(min(p[1] for p in points)) - margin),
            min(1280, math.ceil(max(p[0] for p in points)) + margin),
            min(720, math.ceil(max(p[1] for p in points)) + margin))


def nearest_image(im, region, scale=6):
    cut = im.crop(region).convert("RGB")
    return cut.resize((cut.width * scale, cut.height * scale), Image.Resampling.NEAREST)


def draw_labels(c, line, region, x, y, scale, changes=(), outline=None, overview=False):
    """Transform original image-X/Y into PDF coordinates; markers remain vectors."""
    def p(px, py):
        return x + (px - region[0]) * scale, y + (region[3] - py) * scale
    g = geometry(line)
    if outline:
        path = c.beginPath()
        for idx, (px, py) in enumerate(outline["vertices_px"]):
            (path.moveTo if idx == 0 else path.lineTo)(*p(px, py))
        if outline.get("closed"):
            path.close()
        c.setStrokeColor(AMBER)
        c.setLineWidth(0.75 if not overview else 0.6)
        c.drawPath(path)
    if g is None:
        return
    edges = [("xmin", (g["xmin"], g["ymin"]), (g["xmin"], g["ymax"])),
             ("ymin", (g["xmin"], g["ymin"]), (g["xmax"], g["ymin"])),
             ("xmax", (g["xmax"], g["ymin"]), (g["xmax"], g["ymax"])),
             ("ymax", (g["xmin"], g["ymax"]), (g["xmax"], g["ymax"]))]
    for name, start, end in edges:
        c.setStrokeColor(MAGENTA if name in changes else CYAN)
        c.setLineWidth(0.65 if overview else 0.8)
        c.line(*p(*start), *p(*end))
    for idx, (field, color) in enumerate((("left_point", LEFT), ("right_point", RIGHT))):
        if not g["visibility"][idx]:
            continue
        px, py = p(*g[field])
        radius = 1.0 if overview else 1.65
        c.setStrokeColor(color)
        c.setLineWidth(0.7)
        c.circle(px, py, radius, stroke=1, fill=0)
        if field in changes:
            c.setStrokeColor(MAGENTA)
            c.circle(px, py, radius + (0.85 if overview else 1.0), stroke=1, fill=0)


def image_panel(c, im, region, x, top, width, height, label, line=None,
                changes=(), outline=None):
    c.setFillColor(MUTED)
    c.setFont("R46Sans-Bold", 8)
    c.drawString(x, top, label)
    region_w, region_h = region[2] - region[0], region[3] - region[1]
    scale = min(width / region_w, (height - 13) / region_h)
    iw, ih = region_w * scale, region_h * scale
    left = x + (width - iw) / 2
    bottom = top - 12 - ih
    c.drawImage(ImageReader(nearest_image(im, region)), left, bottom, iw, ih)
    c.saveState()
    # Clip vectors to the crop's exact image rectangle.
    path = c.beginPath()
    path.rect(left, bottom, iw, ih)
    c.clipPath(path, stroke=0)
    draw_labels(c, line, region, left, bottom, scale, changes, outline)
    c.restoreState()


def title_for(row):
    fid = row["stable_id"].split(":")[0]
    source = "added identity" if row["source_row"] is None else f"source row {row['source_row']:03d}"
    return f"{fid} / {source}"


def change_text(row):
    before, after = geometry(row["r44_line"]), geometry(row["after_line"])
    fields = row["latest_update"].get("changed_fields", [])
    if not fields:
        return "<b>R44 to R46:</b> retained. No new coordinate edit is justified by this review."
    chunks = [f"<b>{escape(name)}</b> {value_text(before[name])} to {value_text(after[name])}"
              for name in fields]
    return "<b>R44 to R46 (original pixels):</b> " + " &nbsp; | &nbsp; ".join(chunks)


def row_card(c, row, old43, outline, im, top, height=242):
    x = MARGIN
    c.setFillColor(colors.white)
    c.setStrokeColor(LIGHT)
    c.setLineWidth(0.6)
    c.roundRect(x, top - height, WIDTH, height, 6, fill=1, stroke=1)
    c.setFillColor(INK)
    c.setFont("R46Sans-Bold", 11.4)
    c.drawString(x + 10, top - 17, title_for(row))
    changed = row["latest_update"].get("changed_fields", [])
    status = "AMENDED" if changed else "RETAINED REQUEST"
    c.setFillColor(MAGENTA if changed else MUTED)
    c.setFont("R46Sans-Bold", 8)
    c.drawRightString(x + WIDTH - 10, top - 16, f"{status} | TXT row {row.get('output_row')}")
    gap = 10
    panel_w = (WIDTH - 20 - 3 * gap) / 4
    historical = old43.get(row["stable_id"], {}).get("after_line")
    lines = (None, historical, row["r44_line"], row["after_line"])
    region = crop_region(lines[1:], outline, margin=9)
    labels = ("Original pixels", "Previous R43", "R44", "R46 / current")
    for idx, (label, line) in enumerate(zip(labels, lines)):
        image_panel(c, im, region, x + 10 + idx * (panel_w + gap), top - 32,
                    panel_w, 105, label, line, changed if idx == 3 else (),
                    outline if idx == 3 else None)
    coord_h = fit_para(c, change_text(row), x + 10, top - 142, WIDTH - 20,
                       30, size=8.1, minimum=7.1)
    reason_top = top - 146 - coord_h
    reason = escape(row["latest_update"]["reason"])
    hold = row.get("unresolved_fields", [])
    hold_text = ("<b>Still held:</b> " + escape(", ".join(hold)) + "." if hold
                 else "<b>Open field holds:</b> none recorded for this row.")
    unc = row.get("nominal_visual_uncertainty_px", [])
    uncertainty = "-".join(number(x) for x in unc) if isinstance(unc, list) else number(unc)
    bottom_note = hold_text + " &nbsp; Nominal visual uncertainty: " + uncertainty + " px; not calibrated confidence."
    note_style = ParagraphStyle("note", fontName="R46Sans", fontSize=7.2, leading=8.6, textColor=AMBER if hold else MUTED)
    note = Paragraph(bottom_note, note_style)
    _, note_h = note.wrap(WIDTH - 20, 1000)
    note_top = top - height + 9 + note_h
    reason_room = reason_top - note_top - 5
    fit_para(c, reason, x + 10, reason_top, WIDTH - 20, reason_room, size=8.2, minimum=7.0)
    note.drawOn(c, x + 10, top - height + 9)


def summary_page(c, manifest, selected, outline_count):
    page_header(c, "R46 | Fronts, corners, ownership", "27 November 2026 (Asia/Hong_Kong) | Review of 12 original photographs")
    count = manifest["latest_counts"]
    y = PAGE_H - 83
    y -= para(c, "<b>The target remains one identifiable visible cassette front.</b> This review records local corrections and keeps unresolved physical ownership explicit. It does not train or select a model.", MARGIN, y, WIDTH, 12, leading=16) + 18
    stats = [("Current fronts reviewed", str(count["reviewed_rows"])),
             ("Rows amended / retained", f"{count['amended_rows']} / {count['retained_rows']}"),
             ("Corner positions / box boundaries changed", f"{count['corner_coordinate_slots_changed']} / {count['box_boundaries_changed']}"),
             ("Explicit user-requested source rows", str(len(REQUESTS))),
             ("Added / deleted / visibility changes", f"{count['added_rows']} / {count['deleted_rows']} / {count['visibility_changes']}")]
    for label, value in stats:
        c.setFillColor(PALE)
        c.rect(MARGIN, y - 24, WIDTH, 23, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("R46Sans", 10)
        c.drawString(MARGIN + 9, y - 16, label)
        c.setFont("R46Sans-Bold", 11)
        c.drawRightString(PAGE_W - MARGIN - 10, y - 16, value)
        y -= 27
    y -= 14
    text = ("<b>Whole-image gate.</b> Complete annotation candidates: "
            + escape(", ".join(manifest["annotation_complete_ids"]) or "none")
            + ". Held photographs: " + str(len(manifest["held_image_ids"]))
            + ". A useful correction to one row does not release an incomplete photograph. Proposal TXT files include held rows; only the separate complete-candidates export passes this gate.")
    y -= para(c, text, MARGIN, y, WIDTH, 10, leading=13) + 14
    y -= para(c, "<b>Training context.</b> PR133 actually trained frame_0383 together with the original 137 training photographs. The other eleven reviewed additional frames were held. This R46 annotation PDF is separate from that completed run; annotations and model performance must be assessed separately.", MARGIN, y, WIDTH, 10, leading=13) + 14
    y -= para(c, f"<b>How to read this report.</b> The next page indexes all 17 requests. The comparison section includes {len(selected)} amended or explicitly requested rows, two cards per page. A full-frame appendix shows every current annotation in all 12 photographs. {outline_count} supplied traces explain physical contours; none is a training mask.", MARGIN, y, WIDTH, 9.5, leading=12.5) + 13
    draw_legend(c, MARGIN, y - 3)
    para(c, "The existing HTML remains the historical R44 snapshot. This PDF, its manifest and the R46 TXT proposals carry the newer decisions. Decimal serialization does not imply subpixel annotation accuracy.", MARGIN, y - 24, WIDTH, 8.3, leading=10.8, color=MUTED)
    footer(c)
    c.showPage()


def index_page(c, rows, pages):
    page_header(c, "Requested-row index", "Stable source IDs identify the physical annotation history; TXT row numbers identify current exported line order")
    y = PAGE_H - 78
    columns = [MARGIN + 6, MARGIN + 250, MARGIN + 305, MARGIN + 540, MARGIN + WIDTH - 35]
    c.setFillColor(INK)
    c.setFont("R46Sans-Bold", 8.6)
    for x, label in zip(columns, ("Frame / source row", "TXT", "R46 disposition", "Open fields", "Page")):
        c.drawString(x, y, label)
    y -= 8
    for idx, sid in enumerate(REQUESTS):
        row = rows[sid]
        if idx % 2 == 0:
            c.setFillColor(PALE)
            c.rect(MARGIN, y - 18, WIDTH, 18, fill=1, stroke=0)
        fields = row["latest_update"].get("changed_fields", [])
        display = "amend " + ", ".join(fields) if fields else "retain / uncertainty preserved"
        c.setFont("R46Sans", 8.0)
        c.setFillColor(INK)
        c.drawString(columns[0], y - 13, sid.replace(":source:", " / "))
        c.drawString(columns[1], y - 13, str(row.get("output_row")))
        fit_para(c, escape(display), columns[2], y - 4, 224, 17, size=8, minimum=7)
        fit_para(c, escape(", ".join(row.get("unresolved_fields", [])) or "none"), columns[3], y - 4, 190, 17, size=7.6, minimum=7)
        c.drawString(columns[4], y - 13, str(pages[sid]))
        y -= 19
    y -= 20
    para(c, "<b>Coordinate direction:</b> x increases rightward; y increases downward. Moving a point upward decreases y. Top is ymin; bottom is ymax. Original photos are 1280 x 720 pixels.", MARGIN, y, WIDTH, 10, leading=13)
    y -= 45
    para(c, "<b>The bright-edge clue:</b> a lower-front corner often has a bright turn, darker underside and lateral gray/shadow. Use this as supporting evidence after identifying the front's own side and lower rim. A highlight, rack seam or shadow cast onto another object is not automatically the corner.", MARGIN, y, WIDTH, 10, leading=13)
    y -= 57
    para(c, "A box follows visible extrema and can include internal background beside a sloping surface. A physical lower-front corner need not equal the box intersection. Keep each unlocatable slot exactly 0 0 0; do not draw or invent it.", MARGIN, y, WIDTH, 9, leading=11.7, color=MUTED)
    footer(c)
    c.showPage()


def outline_page(c, items, rows, images):
    page_header(c, "Contour interpretation", "Only explicitly supplied manual traces are shown; no polygon is inferred from a box")
    heights = [226, 226]
    for idx, item in enumerate(items):
        top = PAGE_H - 80 - idx * 237
        row = rows[item["stable_id"]]
        fid = item["stable_id"].split(":")[0]
        c.setFillColor(INK)
        c.setFont("R46Sans-Bold", 11)
        c.drawString(MARGIN, top, title_for(row))
        region = crop_region([row["after_line"]], item, margin=9)
        image_panel(c, images[fid], region, MARGIN, top - 17, 350, 130,
                    "Original pixels", None)
        image_panel(c, images[fid], region, MARGIN + 390, top - 17, 350, 130,
                    "R46 labels + supplied diagnostic trace", row["after_line"],
                    row["latest_update"].get("changed_fields", []), item)
        fit_para(c, escape(item["reason"]), MARGIN, top - 157, WIDTH, 52, size=9.3, minimum=8)
        c.setStrokeColor(LIGHT)
        c.line(MARGIN, top - 218, PAGE_W - MARGIN, top - 218)
    footer(c, "Amber traces are nominal review aids, not segmentation labels or training masks")
    c.showPage()


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def overview_page(c, frame, im, outlines):
    fid = frame["image_id"]
    current = [r for r in frame["rows"] + frame.get("additions", []) if r["after_line"] is not None]
    held = sum(bool(r.get("unresolved_fields")) for r in current)
    page_header(c, fid + " | all current fronts",
                f"{len(current)} current identities | {held} rows with open fields | "
                + ("complete annotation candidate" if frame["eligible_for_candidate_export"] else "whole photograph held"))
    draw_legend(c, MARGIN, PAGE_H - 75, small=True)
    scale = min(WIDTH / 1280, 420 / 720)
    iw, ih = 1280 * scale, 720 * scale
    x = (PAGE_W - iw) / 2
    y = 82
    c.drawImage(ImageReader(im.info.get("r46_original_path", im)), x, y, iw, ih)
    region = (0, 0, 1280, 720)
    for row in current:
        draw_labels(c, row["after_line"], region, x, y, scale,
                    row["latest_update"].get("changed_fields", []),
                    outlines.get(row["stable_id"]), overview=True)
    occupied = []
    new_no = 0
    for row in sorted(current, key=lambda r: (geometry(r["after_line"])["ymin"], geometry(r["after_line"])["xmin"])):
        g = geometry(row["after_line"])
        if row["source_row"] is None:
            new_no += 1
            label = "NEW" + str(new_no)
        else:
            label = "S" + str(row["source_row"])
        font = 5.8
        tw = c.stringWidth(label, "R46Sans-Bold", font) + 3
        th = 7.4
        bx0, bx1 = x + g["xmin"] * scale, x + g["xmax"] * scale
        bymid = y + (720 - (g["ymin"] + g["ymax"]) / 2) * scale
        bytop = y + (720 - g["ymin"]) * scale
        candidates = [(bx0 - tw - 2, bymid - th / 2),
                      (bx1 + 2, bymid - th / 2),
                      (bx0, bytop + 1), (bx1 - tw, bytop + 1)]
        chosen = None
        for tx, ty in candidates:
            rect = (tx, ty, tx + tw, ty + th)
            if x <= tx and tx + tw <= x + iw and y <= ty and ty + th <= y + ih and not any(overlap(rect, p) for p in occupied):
                chosen = rect
                break
        if chosen is None:
            # A bounded outward search is preferable to silently overlapping labels.
            for dy in (-8, 8, -16, 16, -24, 24):
                for tx, ty in candidates:
                    rect = (max(x, min(x + iw - tw, tx)), max(y, min(y + ih - th, ty + dy)), 0, 0)
                    rect = (*rect[:2], rect[0] + tw, rect[1] + th)
                    if not any(overlap(rect, p) for p in occupied):
                        chosen = rect
                        break
                if chosen:
                    break
        if chosen is None:
            raise ValueError("Cannot place source ID without overlap: " + row["stable_id"])
        occupied.append(chosen)
        tx, ty = chosen[:2]
        c.setFillColor(colors.white)
        c.rect(tx, ty, tw, th, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("R46Sans-Bold", font)
        c.drawString(tx + 1.5, ty + 1.3, label)
    caption = "S numbers are stable source rows, not new object counts. NEW labels are added identities. Missing corner slots have no dot. Box interiors can contain background beside sloping material; the outer limits still follow the identifiable front."
    para(c, caption, MARGIN, 65, WIDTH, 8.2, leading=10.4, color=MUTED)
    added = [r for r in current if r["source_row"] is None]
    mapping = " | ".join(f"NEW{i + 1}: {r['stable_id']} = TXT row {r.get('output_row')}" for i, r in enumerate(added))
    if mapping:
        para(c, escape(mapping), MARGIN, 40, WIDTH, 7.7, leading=9.5, color=INK)
    footer(c, "Every current row shown | Historical deleted identities remain in the manifest, not drawn as current targets")
    c.showPage()


def render_pdf(folder, pdf):
    folder, pdf = Path(folder), Path(pdf)
    manifest = read(folder / "manifest.json")
    rows = rows_by_id(manifest)
    old43 = rows_by_id(read(folder / "history/r43-manifest.json"))
    outline_data = read(folder / "outlines.json")
    if outline_data.get("training_mask") is not False:
        raise ValueError("Diagnostic trace scope must be explicit")
    outlines = {item["stable_id"]: item for item in outline_data["items"]}
    for sid in REQUESTS:
        if sid not in rows or rows[sid]["after_line"] is None:
            raise ValueError("Missing requested current identity: " + sid)
    selected = [r for f in manifest["frames"] for r in f["rows"] + f.get("additions", [])
                if r["after_line"] is not None and
                (r["latest_update"].get("changed_fields") or r["stable_id"] in REQUESTS)]
    pages = {r["stable_id"]: 3 + i // 2 for i, r in enumerate(selected)}
    images = {}
    for frame in manifest["frames"]:
        fid = frame["image_id"]
        source = folder / "originals" / (fid + ".jpg")
        images[fid] = Image.open(source).convert("RGB")
        images[fid].info["r46_original_path"] = str(source)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf), pagesize=(PAGE_W, PAGE_H), pageCompression=1, invariant=1)
    c.setTitle("R46 cassette-front annotation amendments")
    c.setAuthor("Cassette-front review")
    c.setSubject("Original / R43 / R44 / R46 comparisons and complete frame annotation overviews")
    summary_page(c, manifest, selected, len(outlines))
    index_page(c, rows, pages)
    for start in range(0, len(selected), 2):
        page_header(c, "Annotation comparisons", "Same original crop in all four panels | Vector labels | Changed points retain L/R color with a magenta ring")
        for idx, row in enumerate(selected[start:start + 2]):
            row_card(c, row, old43, outlines.get(row["stable_id"]),
                     images[row["stable_id"].split(":")[0]], PAGE_H - 73 - idx * 250)
        footer(c)
        c.showPage()
    outline_items = list(outlines.values())
    for start in range(0, len(outline_items), 2):
        outline_page(c, outline_items[start:start + 2], rows, images)
    for frame in manifest["frames"]:
        overview_page(c, frame, images[frame["image_id"]], outlines)
    c.save()
    return {"path": str(pdf), "comparison_rows": len(selected), "requested_rows": len(REQUESTS),
            "overview_frames": len(manifest["frames"]), "supplied_traces": len(outlines),
            "pages": 2 + math.ceil(len(selected) / 2) + math.ceil(len(outlines) / 2) + len(manifest["frames"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render_pdf(a.review, a.output), indent=2))


if __name__ == "__main__":
    main()
