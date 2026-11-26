#!/usr/bin/env python3
"""Render the R38 development amendments with stable R37 row identities."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from front_review_r38 import verify_review
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE, BEFORE, AFTER,
    require, sha, register_fonts, text, para, geometry, display,
)
from render_front_review_r36_amendments import panel, pixel_detail

LEDGER = "results/review/front-review-20261126-r38"
TOTAL = 8
AMBER = colors.HexColor("#9B5E16")


def footer(pdf, page):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, "R37 TO R38  /  DEVELOPMENT REFERENCE ONLY  /  26 NOV 2026", MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f"{page} / {TOTAL}")


def title(pdf, kicker, heading, size=25):
    text(pdf, kicker, MARGIN, HEIGHT-35, 8.5, MUTED, True)
    text(pdf, heading, MARGIN, HEIGHT-72, size, INK, True)


def entry(row, page, kind):
    return {"page": page, "kind": kind, "image_id": row["image_id"], "r37_row": row["row"],
            "stable_id": row["stable_id"], "crop_xyxy": row["crop_xyxy"],
            "source_image_sha256": row["source_image_sha256"], "action": row["action"],
            "changed_fields": row["changed_fields"]}


def triptych(pdf, row, image, top, max_height=190):
    width = (WIDTH-2*MARGIN-24)/3
    after = ("R38 | accepted amendment", "after_line", AFTER) if row["after_line"] else (
        "R38 | selected target excluded", None, AMBER)
    heights = []
    for i, (label, mode, color) in enumerate((("Original pixels", None, MUTED),
            ("R37 | saved annotation", "before_line", BEFORE), after)):
        x = MARGIN+i*(width+12)
        text(pdf, label, x, top+11, 9, color, True)
        heights.append(panel(pdf, image, row["crop_xyxy"], row, mode, x, top, width, max_height))
    return top-max(heights)


def table(pdf, entries, x, top, width, headers=("Field", "R37", "R38")):
    cols = [x+4, x+width*.43, x+width*.74]
    pdf.setFillColor(PALE)
    pdf.rect(x, top-15, width, 18, fill=1, stroke=0)
    for xx, label in zip(cols, headers):
        text(pdf, label, xx, top-9, 8, MUTED, True)
    y = top-31
    for values in entries:
        for xx, value in zip(cols, values):
            text(pdf, str(value), xx, y, 9, INK)
        pdf.setStrokeColor(RULE)
        pdf.line(x, y-7, x+width, y-7)
        y -= 23
    return y


def cover(pdf):
    title(pdf, "ANNOTATION REVIEW  /  EIGHT CURRENT REQUESTS", "R38 | Front boundaries and explicit exclusions", 25)
    y = para(pdf, "Three original-pixel coordinate corrections and five case-specific exclusions. Every request is in the development split. All row numbers in this report identify the R37 reference before deletion.",
             MARGIN, HEIGHT-96, WIDTH-2*MARGIN, 11, 15.5)
    cards = [(3, "coordinate-amended rows"), (5, "explicit target exclusions"),
             (0, "training rows changed"), (1, "inherited pending request")]
    cw = (WIDTH-2*MARGIN-36)/4
    for i, (number, label) in enumerate(cards):
        x = MARGIN+i*(cw+12)
        pdf.setFillColor(PALE)
        pdf.roundRect(x, y-81, cw, 64, 4, fill=1, stroke=0)
        text(pdf, str(number), x+12, y-47, 24, INK, True)
        text(pdf, label, x+12, y-67, 8.4, MUTED)
    y -= 111
    sections = [
        ("Follow the physical front, then place its observed corner",
         "new_000010 row 29 moves top to 666 and L to (325,714). new_000008 row 7 tightens bottom to 23. new_000007 row 47 moves top to 385. Other fields remain exact; unavailable corners stay unavailable."),
        ("Five removals change the reference population",
         "Exclude R37 rows 086:57, 088:57, 053:1, 010:30 and 009:52 as specifically requested. These are not five proven duplicate annotations or an automatic small-size rule. A deleted target is not a model recovery."),
        ("Keep the next step small",
         "All 137 reviewed training files remain byte-identical to R37. Reuse the existing eligible R37 training manifest and replay fixed predictions against the new reference. No new paired GPU run is justified by a zero training delta."),
    ]
    for heading, body in sections:
        text(pdf, heading, MARGIN, y, 12, INK, True)
        y = para(pdf, body, MARGIN, y-17, WIDTH-2*MARGIN, 10, 14)-23
    y = para(pdf, "<b>Development inventory:</b> 659 to 654 fronts; 1,224 to 1,220 available endpoints; 94 to 88 unavailable endpoints. The 583 two-corner objects remain. The inherited new_000010 row 24 question stays open. Coordinate ambiguity is about 2 original pixels.",
             MARGIN, y+3, WIDTH-2*MARGIN, 9, 13)
    require(y > 43, "R38 cover overlaps footer")
    footer(pdf, 1)
    pdf.showPage()
    return {"page": 1, "kind": "scope_and_summary"}


def corner_page(pdf, row, image):
    title(pdf, "DEVELOPMENT  /  R37 ROW IDENTITY PRESERVED", "new_000010 | row 29", 24)
    bottom = triptych(pdf, row, image, HEIGHT-109, 175)
    y = bottom-24
    text(pdf, "Coordinates | original pixels", MARGIN, y, 10, INK, True)
    ty = table(pdf, [("Top / stored ymin", "679.912", "666"),
                    ("Left point x", "320.91", "325"), ("Left point y", "717.27", "714")],
               MARGIN, y-17, 395)
    ty = para(pdf, "The old top cuts frontal material. The light upper rim continues near y666, below the broad upper plane. The observed lower-left turn lies around (325,714), above and right of the old point in the dark supporting region.",
              MARGIN, ty-9, 395, 9.2, 13)
    ty = para(pdf, "<b>Held:</b> xmin 316.585, xmax 457.043, bottom 720; L visibility 2; right point unavailable. The photo clips the bottom extent. No predicted right point is added.",
              MARGIN, ty-10, 395, 8.8, 12.5, MUTED)
    dx = 490
    text(pdf, "Lower-left turn | original pixel grid", dx, y, 9.5, INK, True)
    h = pixel_detail(pdf, image, (314,700,340,720), dx, y-25, 303, 150,
        points=((320.9100032,717.2699976,BEFORE,"B"), (325,714,AFTER,"C")))
    py = para(pdf, "B: saved R37 point. C: accepted R38 point.<br/>PR121 candidate p28 and separate PR123 p28 place top near 665.8 and L near (324.6,713.7). Their agreement supports the direction; original contour ownership determines the accepted label.",
              dx, y-25-h-16, WIDTH-MARGIN-dx, 8.6, 12)
    yy = min(ty, py)-18
    text(pdf, "Same native crop in all panels. Integer adjudications have approximately 2 px visual ambiguity.", MARGIN, yy, 7.5, MUTED)
    require(yy > 43, "R38 corner page overlaps footer")
    footer(pdf, 2)
    pdf.showPage()
    result = entry(row, 2, "coordinate_amendment")
    result["detail_crop"] = [314,700,340,720]
    return result


def bottom_page(pdf, row, image):
    title(pdf, "DEVELOPMENT  /  EXCLUDE THE BROADER LOWER SEPARATOR", "new_000008 | row 7", 24)
    bottom = triptych(pdf, row, image, HEIGHT-109, 190)
    y = bottom-25
    text(pdf, "Coordinates | original pixels", MARGIN, y, 10, INK, True)
    ty = table(pdf, [("Bottom / stored ymax", "33", "23")], MARGIN, y-17, 395)
    ty = para(pdf, "<b>Held:</b> the other box edges; observed L=(1221,20), visibility 2; right point unavailable. The image clips the front at its top and right boundaries.",
              MARGIN, ty-9, 395, 9.1, 13)
    ty = para(pdf, "The box does not have to end at L.y=20. Its bottom encloses the sloping and fading owned extent across the whole visible front, reaching about y23 near the right border.",
              MARGIN, ty-12, 395, 9.1, 13, MUTED)
    text(pdf, "Original contour takes precedence", 448, y, 10, INK, True)
    py = para(pdf, "The saved bottom at y33 includes the broad grey separator below the selected front. Move it upward to y23 while retaining the front's own soft lower edge. Approximate visual ambiguity is 2 pixels.",
              448, y-17, WIDTH-MARGIN-448, 9.4, 13.2)
    py = para(pdf, "PR121 candidate p36 and PR123 candidate p36 both extend to about y34. Their taller boxes are not adopted. The left point is also preserved instead of being replaced by a predicted point.",
              448, py-12, WIDTH-MARGIN-448, 9.1, 13, MUTED)
    yy = min(ty, py)-20
    text(pdf, "User physical ymin / bottom corresponds to stored ymax because image y increases downward.", MARGIN, yy, 7.5, MUTED)
    require(yy > 43, "R38 bottom page overlaps footer")
    footer(pdf, 3)
    pdf.showPage()
    return entry(row, 3, "coordinate_amendment")


def top_page(pdf, row, image):
    title(pdf, "DEVELOPMENT  /  EXPLICIT PR121 p57 CUE", "new_000007 | row 47", 24)
    bottom = triptych(pdf, row, image, HEIGHT-109, 190)
    y = bottom-25
    text(pdf, "Coordinates | original pixels", MARGIN, y, 10, INK, True)
    ty = table(pdf, [("Top / stored ymin", "388", "385")], MARGIN, y-17, 395)
    ty = para(pdf, "<b>Held edges:</b> xmin 726, xmax 789, bottom 403.<br/><b>Held points:</b> L=(729,403), R=(785.5,397), both visibility 2.",
              MARGIN, ty-9, 395, 9.1, 13)
    ty = para(pdf, "The own shaded upper edge reaches about y385 near the right side. The old top at y388 cuts that sloping extent. Include the supporting front edge while excluding unrelated rack and neighboring material.",
              MARGIN, ty-12, 395, 9.1, 13, MUTED)
    text(pdf, "Prediction IDs remain cache-specific", 448, y, 10, INK, True)
    py = para(pdf, "The requested PR121 candidate p57 has top 384.77. The separate PR123 matched p45 and duplicate p57 have similar top coordinates. All three support integer top 385 within about 2 px visual ambiguity.",
              448, y-17, WIDTH-MARGIN-448, 9.4, 13.2)
    py = para(pdf, "Only the top direction is accepted. PR121 p57 is narrower and places its left point farther right. Keep the existing xmin and both observed corners; do not copy the entire prediction.",
              448, py-12, WIDTH-MARGIN-448, 9.1, 13, MUTED)
    yy = min(ty, py)-20
    text(pdf, "User physical ymax / top corresponds to stored ymin. R37 row 47 remains R38 row 47.", MARGIN, yy, 7.5, MUTED)
    require(yy > 43, "R38 top page overlaps footer")
    footer(pdf, 4)
    pdf.showPage()
    return entry(row, 4, "coordinate_amendment")


def narrow_deletions_page(pdf, rows, images):
    title(pdf, "DEVELOPMENT  /  TWO EXPLICIT CASE-SPECIFIC EXCLUSIONS", "Narrow regions beside the finger", 25)
    y = para(pdf, "Remove these two named targets as requested. Their size and appearance do not create a general rule for deleting other partial fronts. The after panels hide only the selected row; other annotations are unchanged.",
             MARGIN, HEIGHT-96, WIDTH-2*MARGIN, 10, 14)
    half = (WIDTH-2*MARGIN-28)/2
    top = y-47
    page_entries = []
    ends = []
    for i, key in enumerate((("new_000086",57),("new_000088",57))):
        row, image = rows[key], images[key]
        x = MARGIN+i*(half+28)
        text(pdf, f"{row['image_id']} | R37 row 57", x, top+29, 11, INK, True)
        pw = (half-12)/2
        heights = []
        for j, (label, mode, col) in enumerate((("R37 | selected row", "before_line", BEFORE),
                                              ("R38 | excluded", None, AMBER))):
            xx = x+j*(pw+12)
            text(pdf, label, xx, top+11, 8.5, col, True)
            heights.append(panel(pdf, image, row["crop_xyxy"], row, mode, xx, top, pw, 225))
        by = top-max(heights)-22
        by = para(pdf, "<b>Action:</b> exclude original R37 row 57. It has no R38 row destination. All earlier rows in this image retain their identities and exact tokens.",
                  x, by, half, 9.3, 13)
        by = para(pdf, "This is an explicit reference exclusion, not proof of background, a confirmed duplicate annotation, or a recovered detection.",
                  x, by-12, half, 9, 12.5, MUTED)
        ends.append(by)
        page_entries.append(entry(row, 5, "user_directed_exclusion"))
    require(min(ends) > 43, "R38 narrow exclusion page overlaps footer")
    footer(pdf, 5)
    pdf.showPage()
    return {"page": 5, "kind": "paired_user_directed_exclusions", "cases": page_entries}


def wide_deletion_block(pdf, row, image, top, height, note):
    text(pdf, f"{row['image_id']} | R37 row {row['row']}", MARGIN, top+29, 11, INK, True)
    width = (WIDTH-2*MARGIN-24)/2
    heights = []
    for i, (label, mode, col) in enumerate((("R37 | selected row", "before_line", BEFORE),
                                           ("R38 | selected target excluded", None, AMBER))):
        x = MARGIN+i*(width+24)
        text(pdf, label, x, top+11, 8.5, col, True)
        heights.append(panel(pdf, image, row["crop_xyxy"], row, mode, x, top, width, height))
    y = top-max(heights)-13
    return para(pdf, note, MARGIN, y, WIDTH-2*MARGIN, 8.8, 12)


def border_deletions_page(pdf, rows, images):
    title(pdf, "DEVELOPMENT  /  EXCLUSIONS AT IMAGE BOUNDARIES", "Containment alone does not prove duplication", 24)
    y = wide_deletion_block(pdf, rows['new_000053',1], images['new_000053',1], HEIGHT-125, 130,
        "Exclude R37 row 1 as requested. The neighboring R37 row 2 becomes R38 row 1. High containment is a physical-review flag; it does not independently establish that both labels describe one cassette.")
    y = wide_deletion_block(pdf, rows['new_000010',30], images['new_000010',30], y-49, 115,
        "Exclude the clipped R37 row 30 as requested. R37 row 29 receives the separate amendment shown on page 2; old rows 31-34 shift to R38 rows 30-33. The after panel hides only the selected row 30.")
    require(y > 43, "R38 border exclusion page overlaps footer")
    footer(pdf, 6)
    pdf.showPage()
    return {"page": 6, "kind": "paired_user_directed_exclusions",
            "cases": [entry(rows[k],6,"user_directed_exclusion") for k in (("new_000053",1),("new_000010",30))]}


def partial_exclusion_page(pdf, row, image):
    title(pdf, "DEVELOPMENT  /  EXPLICIT TARGET EXCLUSION", "new_000009 | R37 row 52", 24)
    bottom = triptych(pdf, row, image, HEIGHT-109, 190)
    y = bottom-25
    text(pdf, "A visible fragment can still be explicitly excluded", MARGIN, y, 11, INK, True)
    y = para(pdf, "The original pixels visibly support a small partial front. The user specifically selects this row for removal, so R38 excludes the target. That action does not establish that the pixels are background, prove an annotation duplicate, or demonstrate a model recovery.",
             MARGIN, y-17, WIDTH-2*MARGIN, 10, 14)
    y -= 22
    text(pdf, "Preserve the old identity when row numbers shift", MARGIN, y, 10.5, INK, True)
    y = table(pdf, [("Deleted target", "row 52", "no row"),
                    ("Retained next rows", "rows 53-55", "rows 52-54")],
              MARGIN, y-18, WIDTH-2*MARGIN, headers=("Meaning", "R37", "R38"))
    y = para(pdf, "Stable identity: fronts-2026-11-26-r37/new_000009:52. The row map binds exact file hashes and records a null destination for this target. Earlier R35 exclusions remain excluded. Other retained row contents are unchanged.",
             MARGIN, y-7, WIDTH-2*MARGIN, 9, 12.5, MUTED)
    require(y > 43, "R38 partial-front exclusion page overlaps footer")
    footer(pdf, 7)
    pdf.showPage()
    return entry(row, 7, "user_directed_exclusion")


def inherited_page(pdf, row, image):
    title(pdf, "INHERITED REQUEST  /  RETAINED IN FULL", "new_000010 | row 24 stays unchanged", 24)
    y = para(pdf, "This earlier landmark question is separate from the eight current requests. Its original R37 row remains R38 row 24, with exactly the same tokens.",
             MARGIN, HEIGHT-96, WIDTH-2*MARGIN, 10.5, 14.5)
    top = y-31
    width = (WIDTH-2*MARGIN-24)/2
    heights = []
    for i, (label, mode, col) in enumerate((("Original pixels", None, MUTED),
                ("R38 | exact R37 row retained", "after_line", AFTER))):
        x = MARGIN+i*(width+24)
        text(pdf, label, x, top+12, 9.5, col, True)
        heights.append(panel(pdf, image, row["crop_xyxy"], row, mode, x, top, width, 150))
    y = top-max(heights)-24
    text(pdf, "Preserve the actual landmark meaning", MARGIN, y, 11.5, INK, True)
    y = para(pdf, "The left end remains hidden by the jaw and its point stays unavailable. The observed right point stays (909,491), visibility 2; the box stays (810,464,913,495). No current request identifies a replacement, so the entire row remains held.",
             MARGIN, y-17, WIDTH-2*MARGIN, 9.8, 13.7)
    y -= 20
    text(pdf, "Context is already present in the image", MARGIN, y, 11.5, INK, True)
    y = para(pdf, "Keep front boxes tied to owned visible material. Expanding them into rack or environment would change the geometry being labeled; it is not adopted as a remedy for missed fronts. A later context region would need its own explicit meaning.",
             MARGIN, y-17, WIDTH-2*MARGIN, 9.6, 13.4)
    y = para(pdf, "<b>Next step:</b> CPU replay of the same cached predictions against clearly named references. All 137 reviewed training files and the eligible R37 training manifest remain unchanged. No new checkpoint is trained, and a reference-population change is never presented as a learned improvement.",
             MARGIN, y-13, WIDTH-2*MARGIN, 9.3, 13, AMBER)
    require(y > 43, "R38 inherited page overlaps footer")
    footer(pdf, 8)
    pdf.showPage()
    return entry(row, 8, "inherited_pending_request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    receipt = verify_review(root)
    audit_path = root/LEDGER/"row-review.json"
    audit = json.loads(audit_path.read_text())
    rows = {(r["image_id"],r["row"]): r for r in audit["rows"]}
    images = {}
    for key, row in rows.items():
        path = root/"data/images"/row["split"]/(row["image_id"]+".png")
        require(sha(path) == row["source_image_sha256"], "Original PNG changed")
        images[key] = Image.open(path).convert("RGB")
    register_fonts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.output), pagesize=(WIDTH,HEIGHT), pageCompression=1)
    pdf.setTitle("R38 development annotation amendments and explicit exclusions")
    pdf.setAuthor("Siuuuers/dl annotation review")
    pages = [cover(pdf), corner_page(pdf,rows['new_000010',29],images['new_000010',29]),
        bottom_page(pdf,rows['new_000008',7],images['new_000008',7]),
        top_page(pdf,rows['new_000007',47],images['new_000007',47]),
        narrow_deletions_page(pdf,rows,images), border_deletions_page(pdf,rows,images),
        partial_exclusion_page(pdf,rows['new_000009',52],images['new_000009',52]),
        inherited_page(pdf,rows['new_000010',24],images['new_000010',24])]
    pdf.save()
    inventory = {"schema_version": 1, "pdf_pages": TOTAL, "current_requested_rows": 8,
        "coordinate_amended_rows": 3, "changed_box_edges": 3, "changed_observed_points": 1,
        "inherited_pending_rows": 1, "added_rows": 0, "deleted_rows": 5, "changed_training_rows": 0,
        "point_visibility_changes_on_retained_rows": 0,
        "review_manifest_sha256": receipt["review_manifest_sha256"], "row_audit_sha256": sha(audit_path),
        "renderer_sha256": sha(Path(__file__)), "pdf_sha256": sha(args.output),
        "prediction_evidence": audit["prediction_evidence"], "row_identity_semantics": audit["row_identity_semantics"],
        "pages": pages}
    if args.inventory:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"pages":TOTAL,"bytes":args.output.stat().st_size,
                      "sha256":sha(args.output),"review_manifest_sha256":receipt["review_manifest_sha256"]}))


if __name__ == "__main__":
    main()
