#!/usr/bin/env python3
"""Render R47 original/R46/current evidence, resolutions and all-frame coverage."""
import argparse
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
para, fit, geometry = base.para, base.fit_para, base.geometry


def legend(c, x, y, small=False):
    font = 7 if small else 8.2
    for color, label in [(base.CYAN, "current box"), (base.LEFT, "left point"),
                         (base.RIGHT, "right point"), (base.MAGENTA, "R47 coordinate change"),
                         (base.AMBER, "manual contour")]:
        c.setFillColor(color)
        c.rect(x, y - 1, 8, 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont(FONT, font)
        c.drawString(x + 12, y - 2, label)
        x += c.stringWidth(label, FONT, font) + 28


def header(c, title, subtitle):
    base.page_header(c, title, subtitle)


def footer(c, note="Original photographs | R47 assistant proposals | No new model result"):
    base.footer(c, note)


def delta(row):
    before, after = geometry(row["r46_line"]), geometry(row["after_line"])
    changes = row["latest_update"]["changed_fields"]
    if not changes:
        return "Coordinates retained."
    return "; ".join(f"{k}: {base.value_text(before[k])} -> {base.value_text(after[k])}" for k in changes)


def summary(c, manifest, selected):
    header(c, "R47 | Follow the cassette's own contour", "27 November 2026 (Asia/Hong_Kong) | Amendment of the PR134 / R46 review")
    y = H - 83
    y -= para(c, "<b>Neighboring cassettes help identify the repeated physical front.</b> Each bound is still placed on this cassette's visible material. A right-hand cassette can need its <b>left</b> boundary moved right to exclude the intervening rack.", M, y, WIDTH, 12, leading=16) + 17
    n = manifest["latest_counts"]
    stats = [("Original photographs / current fronts inspected", f"12 / {n['reviewed_rows']}"),
             ("Rows with coordinate amendments / retained coordinates", f"{n['amended_rows']} / {n['retained_rows']}"),
             ("Box boundaries / corner positions changed", f"{n['box_boundaries_changed']} / {n['corner_coordinate_slots_changed']}"),
             ("Field holds resolved / newly recorded", f"{n['resolved_fields']} / {n['newly_held_fields']}"),
             ("Complete photographs / still held photographs", f"{len(manifest['annotation_complete_ids'])} / {len(manifest['held_image_ids'])}"),
             ("Added rows / deleted rows / changed visibility slots", "0 / 0 / 0")]
    for label, value in stats:
        c.setFillColor(PALE)
        c.rect(M, y - 23, WIDTH, 22, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(FONT, 10)
        c.drawString(M + 9, y - 15, label)
        c.setFont(BOLD, 11)
        c.drawRightString(W - M - 10, y - 15, value)
        y -= 26
    y -= 14
    for text in [
        "<b>Complete annotation candidates:</b> " + escape(", ".join(manifest["annotation_complete_ids"]) or "none") + ". The complete-candidates folder includes whole photos only. All other TXT files are review proposals, including unresolved rows.",
        "<b>How to use this report:</b> the next page shows photograph status. Comparisons show identical original crops without labels, with R46 labels, and with R47 labels. Later pages show manual contours, all 507 current identities, and a complete row-to-page index.",
        "<b>Result boundary:</b> PR134 is a CPU annotation/replay package. PR133 remains the latest measured R45 pilot and trained frame_0383 only in addition to 137 original photos. These newer labels have not produced a new measured model result.",
    ]:
        y -= para(c, text, M, y, WIDTH, 9.5, leading=12.2) + 13
    legend(c, M, y - 1)
    para(c, "Physical interpretations are reviewable proposals. Pixel uncertainty reflects blur and contour judgment, not a calibrated confidence interval. Historical labels and missing-point tokens are preserved in the manifest.", M, y - 21, WIDTH, 8.1, color=MUTED)
    footer(c)
    c.showPage()


def frame_status(c, manifest, prior):
    header(c, "Whole-photograph status and guidance", "A resolved field is recorded separately from a coordinate change; neither alone is a model improvement")
    y = H - 82
    xs = [M + 6, M + 145, M + 220, M + 310, M + 430, M + 560]
    c.setFillColor(INK)
    c.setFont(BOLD, 9)
    for x, title in zip(xs, ["Photograph", "Fronts", "Amended", "Holds: R46 -> R47", "Fields resolved", "Current gate"]):
        c.drawString(x, y, title)
    y -= 12
    old = {f["image_id"]: f for f in prior["frames"]}
    for idx, frame in enumerate(manifest["frames"]):
        rows = [r for r in frame["rows"] + frame["additions"] if r["after_line"] is not None]
        before = len(old[frame["image_id"]]["unresolved_stable_ids"])
        after = len(frame["unresolved_stable_ids"])
        if idx % 2 == 0:
            c.setFillColor(PALE); c.rect(M, y - 21, WIDTH, 21, fill=1, stroke=0)
        values = [frame["image_id"], str(len(rows)), str(sum(bool(r["latest_update"]["changed_fields"]) for r in rows)),
                  f"{before} -> {after} rows", str(sum(len(r["latest_update"]["hold_resolutions"]) for r in rows)),
                  "complete candidate" if frame["eligible_for_candidate_export"] else "held"]
        c.setFont(FONT, 8.8); c.setFillColor(INK)
        for x, value in zip(xs, values): c.drawString(x, y - 14, value)
        y -= 23
    y -= 20
    for text in [
        "<b>Neighbor comparison:</b> compare the same molded front/rim and the side-to-bottom turn, allowing for perspective and pose. Locate the target's visible seam independently. A neighboring corner is contextual evidence, not a coordinate to copy. Entirely hidden extents and missing corners stay unlabelled.",
        "<b>Rack exclusion:</b> a continuing vertical strip or lattice belongs to the rack even when it touches a white cassette. Move whichever bound is on that strip toward the target's own edge. Keep gray material when its contour is continuous with the cassette front. Do not crop every box to a bright stripe.",
        "<b>Coordinates:</b> x increases right; y increases down. A left edge moved right increases xmin; a top moved down increases ymin. Bright lower turns, darker undersides and lateral shadow support the contour judgment but are not lighting-invariant rules.",
    ]:
        y -= para(c, text, M, y, WIDTH, 9.3, leading=12.1) + 12
    footer(c)
    c.showPage()


def card(c, row, outline, image, top, height=157):
    c.setStrokeColor(LIGHT); c.setFillColor(colors.white)
    c.roundRect(M, top - height, WIDTH, height, 5, fill=1, stroke=1)
    c.setFillColor(INK); c.setFont(BOLD, 10)
    c.drawString(M + 8, top - 14, base.title_for(row))
    changed = row["latest_update"]["changed_fields"]
    resolved = [x["field"] for x in row["latest_update"]["hold_resolutions"]]
    status = "AMEND" if changed else ("RESOLVE" if resolved else "RETAIN")
    c.setFont(BOLD, 7.6); c.setFillColor(base.MAGENTA if changed else MUTED)
    c.drawRightString(W - M - 8, top - 13, f"{status} | TXT {row.get('output_row')}")
    region = base.crop_region([row["r46_line"], row["after_line"]], outline, margin=7)
    gap, pw = 10, (WIDTH - 36) / 3
    for index, (name, line) in enumerate((("Original pixels", None), ("R46", row["r46_line"]), ("R47", row["after_line"]))):
        base.image_panel(c, image, region, M + 8 + index * (pw + gap), top - 27,
                         pw, 62, name, line, changed if index == 2 else (), outline if index == 2 else None)
    fit(c, escape(delta(row)), M + 8, top - 94, WIDTH - 16, 18, size=7.6, minimum=6.8)
    # The recorded concise decision omits repeated general guidance. Full local
    # field reasons remain alongside it in the audit ledger.
    fit(c, escape(row["latest_update"].get("summary_reason", row["latest_update"]["reason"])),
        M + 8, top - 114, WIDTH - 16, 27, size=7.5, minimum=6.6)
    note = "Resolved: " + (", ".join(resolved) or "none") + " | Open: " + (", ".join(row["unresolved_fields"]) or "none")
    uncertainty = row.get("nominal_visual_uncertainty_px", [1, 3])
    note += " | Nominal uncertainty: " + ("-".join(str(x) for x in uncertainty) if isinstance(uncertainty, list) else str(uncertainty)) + " px"
    fit(c, escape(note), M + 8, top - 144, WIDTH - 16, 10, size=6.9, minimum=6.3, color=base.AMBER if row["unresolved_fields"] else MUTED)


def trace_page(c, items, rows, images):
    header(c, "Local contour interpretation", "Supplied traces explain material ownership; they are not bounding-box polygons or segmentation labels")
    for index, item in enumerate(items):
        row = rows[item["stable_id"]]
        top = H - 80 - index * 237
        c.setFillColor(INK); c.setFont(BOLD, 11)
        c.drawString(M, top, base.title_for(row))
        anchors = item.get("context_anchors", [])
        context_extent = dict(item, vertices_px=item["vertices_px"] + [a["xy_px"] for a in anchors])
        region = base.crop_region([row["after_line"]], context_extent, margin=12)
        im = images[item["stable_id"].split(":")[0]]
        base.image_panel(c, im, region, M, top - 17, 350, 130, "Original pixels")
        base.image_panel(c, im, region, M + 390, top - 17, 350, 130,
                         "R47 + manual contour", row["after_line"], row["latest_update"]["changed_fields"], item)
        scale = min(350 / (region[2] - region[0]), 117 / (region[3] - region[1]))
        left = M + 390 + (350 - (region[2] - region[0]) * scale) / 2
        bottom = top - 29 - (region[3] - region[1]) * scale
        for anchor in anchors:
            px = left + (anchor["xy_px"][0] - region[0]) * scale
            py = bottom + (region[3] - anchor["xy_px"][1]) * scale
            c.setStrokeColor(base.LEFT); c.setLineWidth(1.1)
            c.circle(px, py, 2.6, stroke=1, fill=0)
            label = "S" + anchor["stable_id"].split(":")[-1] + (" L" if anchor["point"] == "left_point" else " R") + ": context"
            c.setFillColor(INK); c.setFont(BOLD, 6.8)
            c.drawRightString(px - 4, py + 6, label)
        fit(c, escape(item["reason"]), M, top - 157, WIDTH, 52, size=9.2, minimum=7.5)
        c.setStrokeColor(LIGHT); c.line(M, top - 218, W - M, top - 218)
    footer(c, "Amber traces are diagnostic aids; a missing upper contour is never closed automatically")
    c.showPage()


def overview(c, frame, image, outlines):
    # Reuse the proven collision-aware source-ID placement from R46 while using
    # the current legend. The imported function reads its module's legend.
    original_legend = base.draw_legend
    try:
        base.draw_legend = legend
        base.overview_page(c, frame, image, outlines)
    finally:
        base.draw_legend = original_legend


def index_pages(c, frame, pages):
    current = [r for r in frame["rows"] + frame["additions"] if r["after_line"] is not None]
    header(c, frame["image_id"] + " | complete audit index", "Every current identity is listed; source identity and current TXT line are separate")
    y = H - 79
    xs = [M + 4, M + 80, M + 125, M + 343, M + 500, M + 733]
    c.setFillColor(INK); c.setFont(BOLD, 8)
    for x, title in zip(xs, ["Source", "TXT", "Coordinate edits", "Resolved fields", "Still open", "Page"]): c.drawString(x, y, title)
    y -= 7
    for idx, row in enumerate(current):
        if idx % 2 == 0:
            c.setFillColor(PALE); c.rect(M, y - 10, WIDTH, 10, fill=1, stroke=0)
        source = f"S{row['source_row']:03d}" if row["source_row"] is not None else "ADDED"
        values = [source, str(row.get("output_row")), ", ".join(row["latest_update"]["changed_fields"]) or "retained",
                  ", ".join(x["field"] for x in row["latest_update"]["hold_resolutions"]) or "-",
                  ", ".join(row["unresolved_fields"]) or "none", str(pages.get(row["stable_id"], "-"))]
        widths = [73, 39, 211, 150, 226, 39]
        for x, width, value in zip(xs, widths, values):
            fit(c, escape(value), x, y - 1, width, 9.5, size=6.6, minimum=5.8)
        y -= 10.1
    y -= 12
    para(c, "A dash in Page means no separate comparison card; its annotation is visible on the full-frame overview. Retained coordinates can have a resolved field when local contour evidence supports the existing nominal bound. Exact reasons, before/after TXT and original SHA-256 are in audit-ledger.json.", M, y, WIDTH, 8.1, leading=10.5, color=MUTED)
    footer(c, "Stable history includes one earlier deletion; deleted rows are not counted as current targets")
    c.showPage()


def render_pdf(folder, output):
    folder, output = Path(folder), Path(output)
    manifest = base.read(folder / "manifest.json")
    prior = base.read(folder / "history/r46-manifest.json")
    rows = base.rows_by_id(manifest)
    outline_data = base.read(folder / "outlines.json")
    if outline_data["training_mask"] is not False:
        raise ValueError("Only diagnostic outlines are supported")
    outlines = {x["stable_id"]: x for x in outline_data["items"]}
    selected = [r for f in manifest["frames"] for r in f["rows"] + f["additions"]
                if r["after_line"] is not None and
                (r["latest_update"]["changed_fields"] or r["latest_update"]["hold_resolutions"]
                 or r["latest_update"]["requested_fields"] or r["unresolved_fields"])]
    # The two directly named examples come first; the remaining frame order is
    # stable, so finding a comparison does not depend on a severity heuristic.
    direct = ["frame_0160:source:001", "frame_0160:source:037"]
    selected.sort(key=lambda r: (0, direct.index(r["stable_id"])) if r["stable_id"] in direct else (1, list(rows).index(r["stable_id"])))
    pages = {r["stable_id"]: 3 + i // 3 for i, r in enumerate(selected)}
    images = {}
    for frame in manifest["frames"]:
        fid = frame["image_id"]
        path = folder / "originals" / (fid + ".jpg")
        images[fid] = Image.open(path).convert("RGB")
        images[fid].info["r46_original_path"] = str(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1, invariant=1)
    c.setTitle("R47 cassette-front amendments and complete audit")
    c.setAuthor("Cassette-front review")
    c.setSubject("Original/R46/R47 comparisons, locally resolved fields, source identities and original photographs")
    summary(c, manifest, selected)
    frame_status(c, manifest, prior)
    for start in range(0, len(selected), 3):
        header(c, "Original / R46 / R47", "Identical original crop in each panel | Coordinates in original pixels | Magenta marks only this round's changes")
        for idx, row in enumerate(selected[start:start + 3]):
            card(c, row, outlines.get(row["stable_id"]), images[row["stable_id"].split(":")[0]], H - 72 - idx * 164)
        footer(c); c.showPage()
    items = list(outlines.values())
    for start in range(0, len(items), 2): trace_page(c, items[start:start + 2], rows, images)
    for frame in manifest["frames"]: overview(c, frame, images[frame["image_id"]], outlines)
    for frame in manifest["frames"]: index_pages(c, frame, pages)
    c.save()
    report = {"path": str(output), "pages": 2 + math.ceil(len(selected) / 3) + math.ceil(len(items) / 2) + 24,
              "comparison_rows": len(selected), "manual_traces": len(items), "overview_frames": 12,
              "audit_index_rows": 507, "comparison_pages_by_stable_id": pages}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(render_pdf(a.review, a.output), indent=2))


if __name__ == "__main__": main()
