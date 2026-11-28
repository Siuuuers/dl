#!/usr/bin/env python3
"""Render the three changed R67 photographs from verified pixels and label rows.

This creates a review PDF, page PNGs, and a machine verification receipt. It
never modifies labels or turns rendering into a claim of visual inspection.
The receipt's visual-review section must be completed after viewing every page.
Requires Pillow, reportlab, pypdf, and Poppler's pdftoppm.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

from PIL import Image
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


PAGES = [
    ("new_000009", [
        ("A", "r37/new_000009/R052", [550, 450, 670, 535]),
        ("B", "r34/new_000009/R056", [765, 455, 885, 540]),
    ]),
    ("new_000010", [("A", "r37/new_000010/R030", [290, 585, 490, 720])]),
    ("new_000053", [("A", "r37/new_000053/R001", [605, 0, 725, 100])]),
]
SHORT_REASONS = {
    "r37/new_000009/R052": "Distinct lower band beside the left jaw; its right corner remains hidden.",
    "r34/new_000009/R056": "Distinct visible front behind the right jaw; its left corner remains hidden.",
    "r37/new_000010/R030": "A distinct lower band reaches the bottom image boundary. Neither actual lower corner is visible; both remain unavailable. Overlapping axis-aligned boxes can enclose separate slanted fronts.",
    "r37/new_000053/R001": "A separate top-edge wedge has its own visible lower contour above the next broad front. Its left lower turn is marked; the right lower corner is outside the photograph.",
}
NAVY = colors.HexColor("#173D4B")
MUTED = colors.HexColor("#52616B")
RULE = colors.HexColor("#D4DCE1")
OLD = colors.HexColor("#4169D9")
ADDED = colors.HexColor("#E64B32")
POINT = colors.HexColor("#17D9CD")
LOCATOR = colors.HexColor("#F5C84C")
PW, PH = A4
MARGIN = 36
CONTENT_W = PW - 2 * MARGIN


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def geometry(line: str, width: int, height: int) -> dict:
    v = list(map(float, line.split()))
    require(len(v) == 11 and v[0] == 0, "Expected one-class two-point row")
    box = [(v[1] - v[3] / 2) * width, (v[2] - v[4] / 2) * height,
           (v[1] + v[3] / 2) * width, (v[2] + v[4] / 2) * height]
    points = [[v[5] * width, v[6] * height, int(v[7])],
              [v[8] * width, v[9] * height, int(v[10])]]
    for point in points:
        require(point[2] in (0, 2), "Unexpected availability value")
        require(point[2] != 0 or point[:2] == [0, 0], "Missing point is not canonical 0 0 0")
    return {"bbox_xyxy": box, "keypoints": points}


def fmt(value: float) -> str:
    if abs(value) < 0.005:
        value = 0
    return f"{value:.2f}".rstrip("0").rstrip(".")


def point_text(point: list) -> str:
    if point[2] == 0:
        return "unavailable, v=0 (stored 0 0 0)"
    return f"({fmt(point[0])}, {fmt(point[1])}), v=2"


def text(c, x, top, string, size=9, bold=False, color=NAVY):
    c.setFont("DV-Bold" if bold else "DV", size)
    c.setFillColor(color)
    c.drawString(x, PH - top, string)


def paragraph(c, x, top, width, string, size=9, leading=13, color=MUTED):
    style = ParagraphStyle("body", fontName="DV", fontSize=size,
                           leading=leading, textColor=color, spaceAfter=0)
    p = Paragraph(string, style)
    _, height = p.wrap(width, PH)
    p.drawOn(c, x, PH - top - height)
    return height


def image_bytes(image: Image.Image) -> ImageReader:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return ImageReader(buf)


def draw_overlay(c, shape, roi, x, top, width, color, added=False):
    left, upper, right, lower = roi
    scale = width / (right - left)
    height = (lower - upper) * scale
    bottom = PH - top - height
    x1, y1, x2, y2 = shape["bbox_xyxy"]
    if x2 <= left or x1 >= right or y2 <= upper or y1 >= lower:
        return
    c.saveState()
    clip = c.beginPath()
    clip.rect(x, bottom, width, height)
    c.clipPath(clip, stroke=0, fill=0)
    c.setStrokeColor(color)
    c.setLineWidth(1.55 if added else 0.65)
    c.rect(x + (x1 - left) * scale, bottom + (lower - y2) * scale,
           (x2 - x1) * scale, (y2 - y1) * scale, stroke=1, fill=0)
    for px, py, availability in shape["keypoints"]:
        if availability == 2 and left <= px <= right and upper <= py <= lower:
            c.setFillColor(POINT if added else color)
            c.setStrokeColor(NAVY if added else colors.white)
            c.setLineWidth(0.55 if added else 0.35)
            c.circle(x + (px - left) * scale, bottom + (lower - py) * scale,
                     2.7 if added else 1.6, stroke=1, fill=1)
    c.restoreState()


def crop_panel(c, image, roi, x, top, width, rows, added_rows=()):
    cropped = image.crop(roi)
    height = width * cropped.height / cropped.width
    c.drawImage(image_bytes(cropped), x, PH - top - height, width, height)
    for row in rows:
        draw_overlay(c, row, roi, x, top, width, OLD)
    for row in added_rows:
        draw_overlay(c, row, roi, x, top, width, ADDED, added=True)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.45)
    c.rect(x, PH - top - height, width, height, stroke=1, fill=0)
    return height


def full_context(c, image, crops):
    x, top, width = MARGIN, 105, 314
    height = width * image.height / image.width
    c.drawImage(image_bytes(image), x, PH - top - height, width, height)
    scale = width / image.width
    for letter, _, (l, u, r, b) in crops:
        c.setStrokeColor(LOCATOR)
        c.setLineWidth(1)
        c.setDash(3, 2)
        c.rect(x + l * scale, PH - top - b * scale, (r-l)*scale, (b-u)*scale,
               stroke=1, fill=0)
        c.setDash()
        bx = x + l * scale
        by = PH - top - u * scale
        # Keep locator letter on the full photograph, including frame-edge crops.
        by = min(PH - top - 11, by + 2)
        c.setFillColor(NAVY)
        c.rect(bx, by, 12, 12, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("DV-Bold", 8)
        c.drawCentredString(bx + 6, by + 3, letter)
    text(c, x, 296, "Original photograph; dashed frames locate the crops.", 8, color=MUTED)


def legend(c):
    y = PH - 315
    c.setStrokeColor(OLD); c.setLineWidth(1)
    c.line(MARGIN, y, MARGIN + 14, y)
    text(c, MARGIN + 19, 318, "Retained R50 labels", 8)
    c.setStrokeColor(ADDED); c.setLineWidth(1.55)
    c.line(208, y, 222, y)
    text(c, 227, 318, "Restored front box", 8)
    c.setFillColor(POINT); c.setStrokeColor(NAVY); c.setLineWidth(.55)
    c.circle(396, y, 2.7, stroke=1, fill=1)
    text(c, 405, 318, "Restored visible corner", 8)


def crop_group(c, image, old_shapes, case, letter, roi, start):
    text(c, MARGIN, start, f"{letter}  |  Restored front R{case['candidate_row']:03d}", 10, True)
    text(c, MARGIN, start + 15,
         f"{case['stable_id']}  ->  {case['candidate_reference_id']}", 8.2, color=MUTED)
    gap = 10
    col = (CONTENT_W - 2 * gap) / 3
    panel_top = start + 41
    shape = geometry(case["canonical_line"], image.width, image.height)
    for i, label in enumerate(["RAW CROP", "BEFORE | R50", "AFTER | R67"]):
        x = MARGIN + i * (col + gap)
        text(c, x, start + 33, label, 8, True)
        height = crop_panel(c, image, roi, x, panel_top, col,
                            old_shapes if i else [], [shape] if i == 2 else [])
    y = panel_top + height + 14
    box = ", ".join(fmt(v) for v in shape["bbox_xyxy"])
    text(c, MARGIN, y, f"Nominal box (x1, y1, x2, y2): ({box}) px", 8.4)
    text(c, MARGIN, y + 13, "Left: " + point_text(shape["keypoints"][0]) +
         "   |   Right: " + point_text(shape["keypoints"][1]), 8.1)
    text(c, MARGIN, y + 26,
         f"Identical crop in all panels: x [{roi[0]}, {roi[2]}), y [{roi[1]}, {roi[3]}) px.",
         7.8, color=MUTED)
    return y + 26


def main():
    p = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[1]
    p.add_argument("--repo", type=Path, default=repo_default)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--decisions", type=Path)
    p.add_argument("--parent-label-root", type=Path, required=True,
                   help="Directory containing the original three R50 label files")
    p.add_argument("--image-root", type=Path,
                   help="Optional raw image root; otherwise use verified provenance paths")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--render-dir", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    ev = repo / "docs/thesis/evidence/partial-inclusion-r67"
    provenance_path = args.provenance or ev / "excluded-front-provenance.json"
    manifest_path = args.manifest or repo / "data/reviews/fronts-2026-11-28-r67/manifest.json"
    decisions_path = args.decisions or ev / "review-decisions.json"
    receipt_path = args.receipt or ev / "pdf-validation.json"
    provenance, manifest, decisions = map(read_json, [provenance_path, manifest_path, decisions_path])
    expected_ids = {i for _, crops in PAGES for _, i, _ in crops}
    restored = {row["stable_id"]: row for row in manifest["restored_rows"]}
    accepted = {row["historical_identity"]: row for row in decisions["cases"]
                if row["decision"] == "restore_exact_historical_row"}
    source_cases = {row["historical_identity"]: row for row in provenance["cases"]}
    require(set(restored) == expected_ids == set(accepted), "Accepted row set differs from fixed PDF coverage")
    require(decisions["changed_photos"] == [p[0] for p in PAGES], "Changed photograph scope differs")
    require(digest(provenance_path) == manifest["source_provenance"]["sha256"], "Provenance hash differs")
    all_label_checks = []
    for image_id, record in manifest["files"].items():
        path = manifest_path.parent / record["path"]
        content = path.read_bytes()
        lines = content.splitlines(keepends=True)
        require(digest(path) == record["sha256"], f"New label hash mismatch: {image_id}")
        require(len(lines) == record["rows"], f"New row count mismatch: {image_id}")
        prefix = b"".join(lines[:record["parent_rows"]])
        require(hashlib.sha256(prefix).hexdigest() == record["parent_sha256"],
                f"R50 byte prefix changed: {image_id}")
        all_label_checks.append({"image_id": image_id, **file_record(path),
                                 "parent_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                                 "parent_rows_preserved": record["parent_rows"],
                                 "appended_rows": record["appended_rows"]})
    require(sum(x["appended_rows"] for x in all_label_checks) == 4, "Unexpected number of appended rows")
    for stable_id, row in restored.items():
        old = source_cases[stable_id]["historical_label"]["canonical_line"]
        new_lines = (manifest_path.parent / manifest["files"][row["image_id"]]["path"]).read_text().splitlines()
        require(old == row["canonical_line"] == new_lines[row["candidate_row"] - 1]
                == accepted[stable_id]["proposed_canonical_line"], f"Historical row changed: {stable_id}")
        require(accepted[stable_id]["coordinate_changes"] == [], f"Unexpected coordinate edit: {stable_id}")
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DV", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.render_dir.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=A4, pageCompression=1, invariant=1)
    c.setTitle("Cassette partial-front amendments - R67")
    c.setAuthor("Cassette vision project")
    c.setSubject("Four exact historical rows restored; only three changed development photographs")
    page_records = []
    for page_number, (image_id, crops) in enumerate(PAGES, 1):
        source = source_cases[crops[0][1]]["full_image"]
        image_path = (args.image_root / f"{image_id}.png") if args.image_root else Path(source["local_path"])
        require(digest(image_path) == source["sha256"] == manifest["files"][image_id]["image_sha256"],
                f"Raw image hash differs: {image_id}")
        image = Image.open(image_path).convert("RGB")
        require(list(image.size) == source["size"], "Raw image dimensions differ")
        old_path = args.parent_label_root / f"{image_id}.txt"
        require(digest(old_path) == manifest["files"][image_id]["parent_sha256"], "R50 label hash differs")
        old_shapes = [geometry(line, *image.size) for line in old_path.read_text().splitlines() if line.strip()]
        text(c, MARGIN, 44, "Partial-front amendments", 18, True)
        text(c, MARGIN, 67, f"{image_id}  |  R50 to R67", 11, True)
        c.setFont("DV", 9); c.setFillColor(MUTED)
        c.drawRightString(PW-MARGIN, PH-43, "28 November 2026")
        c.drawRightString(PW-MARGIN, PH-66, f"Changed photograph {page_number} / 3")
        c.setStrokeColor(RULE); c.setLineWidth(.7)
        c.line(MARGIN, PH-82, PW-MARGIN, PH-82)
        full_context(c, image, crops)
        sx, sw = 370, PW-MARGIN-370
        before_count = manifest["files"][image_id]["parent_rows"]
        text(c, sx, 114, f"+{len(crops)} restored front{'s' if len(crops)>1 else ''}", 12, True)
        text(c, sx, 134, f"{before_count} to {before_count+len(crops)} reference rows", 9)
        paragraph(c, sx, 148, sw,
                  "Inclusion only. Historical rows are restored exactly; all existing R50 rows retain their bytes and positions.", 9, 13)
        paragraph(c, sx, 211, sw,
                  "Original image: 1280 x 720 px.<br/>Coordinates start at the upper left. Missing corners are not drawn.<br/>Their stored value is 0 0 0.", 8.5, 12)
        text(c, sx, 284, "Annotation overlays only.", 8.5, color=MUTED)
        legend(c)
        for index, (letter, stable_id, roi) in enumerate(crops):
            ending = crop_group(c, image, old_shapes, restored[stable_id], letter, roi,
                                342 if index == 0 else 566)
            if len(crops) == 1:
                text(c, MARGIN, ending + 37, "Why this row is included", 10, True)
                h = paragraph(c, MARGIN, ending + 49, CONTENT_W, SHORT_REASONS[stable_id], 9.5, 14)
                text(c, MARGIN, ending + 49 + h + 29, "How to read the amendment", 10, True)
                paragraph(c, MARGIN, ending + 49 + h + 41, CONTENT_W,
                          "The highlighted box encloses the recorded visible front extent. A turquoise point marks only an available lower corner. A box can remain in the reference with zero known corners. Restoring this row changes the evaluation population; it adds no new camera observation or model prediction.", 9, 13)
        c.setStrokeColor(RULE); c.line(MARGIN, 51, PW-MARGIN, 51)
        text(c, MARGIN, PH-37, "Nominal pixel coordinates; decimal storage does not establish subpixel physical accuracy.", 7.5, color=MUTED)
        text(c, MARGIN, PH-24, "Development reference | 4 restored fronts | 3 changed photographs | 0 new complete pairs", 7.4, color=MUTED)
        c.showPage()
        page_records.append({"page": page_number, "image_id": image_id,
                             "raw_image": file_record(image_path), "before_labels": file_record(old_path),
                             "crops": [{"letter": letter, "historical_id": stable_id,
                                        "new_id": restored[stable_id]["candidate_reference_id"],
                                        "crop_xyxy_half_open": roi,
                                        "geometry_original_pixels": geometry(restored[stable_id]["canonical_line"], *image.size),
                                        "canonical_line": restored[stable_id]["canonical_line"]}
                                       for letter, stable_id, roi in crops]})
    c.save()
    reader = PdfReader(args.output)
    require(len(reader.pages) == 3, "PDF must have exactly three pages")
    for page, expected in zip(reader.pages, PAGES):
        extracted = page.extract_text()
        require(expected[0] in extracted, "Expected image ID missing from page")
        require("new_000086" not in extracted and "new_000088" not in extracted,
                "Unchanged photograph unexpectedly present")
        for _, stable_id, _ in expected[1]:
            require(stable_id in extracted and restored[stable_id]["candidate_reference_id"] in extracted,
                    "Historical or new row ID missing from page")
    subprocess.run(["pdftoppm", "-r", "150", "-png", str(args.output), str(args.render_dir / "page")], check=True)
    pngs = sorted(args.render_dir.glob("page-*.png"))
    require(len(pngs) == 3, "Expected exactly three rendered page images")
    for record, png in zip(page_records, pngs):
        record["render"] = file_record(png)
    receipt = {
        "schema": "cassette-partial-inclusion-pdf-validation-r67-v1",
        "status": "rendered_pending_visual_inspection",
        "date": "2026-11-28",
        "generator": file_record(Path(__file__)),
        "inputs": {"manifest": file_record(manifest_path), "provenance": file_record(provenance_path),
                   "review_decisions": file_record(decisions_path), "candidate_labels": all_label_checks},
        "checks": {"exactly_three_changed_photographs": True, "four_exact_historical_rows_restored": True,
                   "all_20_r50_label_prefix_hashes_preserved": True, "same_crop_extents_across_raw_before_after": True,
                   "all_missing_points_canonical_and_not_drawn": True, "predictions_loaded": False,
                   "image_pixels_enhanced_or_generated": False, "coordinates_reestimated": False,
                   "all_historical_and_new_ids_present_in_extracted_pdf_text": True,
                   "new_complete_pairs": 0},
        "pdf": file_record(args.output), "page_count": len(reader.pages),
        "pages": page_records,
        "visual_review": {"status": "pending", "instructions": "Inspect each final page PNG at readable size; record actual findings and inspected hashes."},
        "precision_note": "Printed coordinates rounded to two decimal places for reading; exact historical normalized row tokens are retained in source data and this receipt. No subpixel physical-accuracy claim.",
        "commands": {"render": ["pdftoppm", "-r", "150", "-png", str(args.output), str(args.render_dir / "page")]},
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(args.output), "sha256": digest(args.output), "pages": 3,
                      "renders": [str(p) for p in pngs], "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
