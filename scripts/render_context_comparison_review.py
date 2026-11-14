#!/usr/bin/env python3
"""Render the four-page context review from hashed evidence and saved labels.

Only original PNG crops and PDF vector overlays are used. No label, candidate,
protocol, notebook or existing atlas is modified. The blue top guide is an
explicitly unapplied review proposal, never a replacement saved annotation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


PW, PH = landscape(A3)
MARGIN = 30
INK = colors.HexColor("#16232e")
MUTED = colors.HexColor("#556675")
PAPER = colors.HexColor("#f2f5f8")
CYAN = colors.HexColor("#16d9ec")
AMBER = colors.HexColor("#ffb632")
GREEN = colors.HexColor("#48f293")
MAGENTA = colors.HexColor("#ff69dd")
BLUE = colors.HexColor("#1767d2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decoded_label(text, size):
    values = list(map(float, text.split()))
    require(len(values) == 11 and all(math.isfinite(x) for x in values),
            "invalid saved pose row")
    require(values[0] == 0 and values[7] in (0, 2) and values[10] in (0, 2),
            "unexpected class or corner visibility")
    iw, ih = size
    _, cx, cy, width, height, lx, ly, lv, rx, ry, rv = values
    return ([(cx - width / 2) * iw, (cy - height / 2) * ih,
             (cx + width / 2) * iw, (cy + height / 2) * ih],
            [[lx * iw, ly * ih, int(lv)], [rx * iw, ry * ih, int(rv)]])


def verify_evidence(root, evidence):
    verified = {}

    def visit(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = (root / value["path"]).resolve()
                require(path.is_relative_to(root), "evidence path escapes repository")
                require(sha(path) == value["sha256"], "source hash differs: " + value["path"])
                verified[value["path"]] = value["sha256"]
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(evidence)
    require(evidence["kind"] == "context_comparison_review", "wrong evidence kind")
    require(evidence["label_operations_applied"] == [] and evidence["closed_case_ids"] == [],
            "this addendum must not apply labels or close unresolved cases")
    require(evidence["training_handoff"]["training_readiness"] ==
            "ready_for_controlled_incremental_experiment", "unexpected handoff state")
    images = {item["image_id"]: item for item in evidence["images"]}
    require(set(images) == {"new_000030", "new_000038", "new_000117", "new_000128"},
            "four expected source images are required")
    row_count = 0
    for iid, item in images.items():
        with Image.open(root / item["source_image"]["path"]) as image:
            require(list(image.size) == item["original_size"] == [1280, 720],
                    "original image size differs: " + iid)
        lines = (root / item["full_review_labels"]["path"]).read_text().splitlines()
        require(len({row["row"] for row in item["rows"]}) == len(item["rows"]),
                "duplicate selected row: " + iid)
        for row in item["rows"]:
            require(lines[row["row"] - 1] == row["label_text"], "saved row differs: " + iid)
            box, points = decoded_label(row["label_text"], item["original_size"])
            require(all(abs(x - y) < 0.00002 for x, y in zip(box, row["bbox_xyxy_px"])),
                    "decoded box differs: " + iid)
            require(all(abs(x - y) < 0.00002 for p, q in zip(points, row["keypoints_px"])
                        for x, y in zip(p, q)), "decoded corners differ: " + iid)
            row_count += 1
    limited = read(root / evidence["limited_candidate"]["path"])
    source_map = {item["image_id"]: item for item in limited["image_sources"]}
    for iid, item in images.items():
        source = source_map[iid]
        require(source["effective_label"]["sha256"] == item["limited_treatment_labels"]["sha256"]
                and sha(root / source["effective_label"]["path"]) == source["effective_label"]["sha256"],
                "limited treatment label differs: " + iid)
        if iid == "new_000038":
            require(item["full_review_labels"]["sha256"] ==
                    item["limited_treatment_labels"]["sha256"],
                    "three-front labels must already be in the limited candidate")
        else:
            require(source["selection"] == "pr35_retained_legacy" and
                    source["effective_label"] == source["control_label"] == item["limited_treatment_labels"],
                    "both arms must retain identical legacy labels: " + iid)
    proposal, = evidence["prospective_field_corrections"]
    require(proposal["image_id"] == "new_000030" and proposal["current_row"] == 12
            and proposal["field"] == "bbox_ymin_px" and proposal["proposed_px"] == 443
            and proposal["status"] == "review_proposal_not_applied"
            and proposal["corner_changes"] == [], "unexpected proposal")
    return images, verified, row_count


def text(c, value, x, y, width, size=12, leading=17, color=INK, min_y=65):
    c.setFont("Review", size)
    c.setFillColor(color)
    for paragraph in value.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = (line + " " + word).strip()
            if line and pdfmetrics.stringWidth(trial, "Review", size) > width:
                require(y >= min_y, "text overflows its page")
                c.drawString(x, y, line)
                y -= leading
                line = word
            else:
                line = trial
        if line:
            require(y >= min_y, "text overflows its page")
            c.drawString(x, y, line)
        y -= leading
    return y


def heading(c, title, iid, status, page, evidence_hash):
    c.setFillColor(colors.white)
    c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("ReviewBold", 23)
    c.drawString(MARGIN, PH - 39, title)
    c.setFont("Review", 11)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, PH - 61, iid + " | Original PNG: 1280 x 720 | Coordinates are original pixels")
    c.drawRightString(PW - MARGIN, PH - 39, status)
    c.setStrokeColor(colors.HexColor("#dce3e9"))
    c.line(MARGIN, 37, PW - MARGIN, 37)
    c.setFont("Review", 8)
    c.drawString(MARGIN, 22, "Cassette context review | 14 November 2026 | Evidence " + evidence_hash[:12])
    c.drawRightString(PW - MARGIN, 22, str(page) + " / 4")


def overlay(c, rows, crop, left, bottom, width, height, held):
    x0, y0, x1, y1 = crop
    sx, sy = width / (x1 - x0), height / (y1 - y0)

    def xy(x, y):
        return left + (x - x0) * sx, bottom + height - (y - y0) * sy

    c.saveState()
    clip = c.beginPath()
    clip.rect(left, bottom, width, height)
    c.clipPath(clip, stroke=0)
    label_bounds = []
    for row in rows:
        bx0, by0, bx1, by1 = row["bbox_xyxy_px"]
        if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
            continue
        tint = AMBER if row["row"] in held else CYAN
        c.setStrokeColor(tint)
        c.setLineWidth(1.15)
        c.setDash(3, 2) if row["row"] in held else c.setDash()
        c.rect(*xy(bx0, by1), (bx1 - bx0) * sx, (by1 - by0) * sy, fill=0, stroke=1)
        c.setDash()
        points = row["keypoints_px"]
        if all(point[2] == 2 for point in points):
            c.setStrokeColor(GREEN)
            c.setLineWidth(0.8)
            c.line(*xy(*points[0][:2]), *xy(*points[1][:2]))
        for point, point_tint in zip(points, (GREEN, MAGENTA)):
            if point[2] == 2:
                c.setFillColor(point_tint)
                c.setStrokeColor(INK)
                c.setLineWidth(0.5)
                c.circle(*xy(*point[:2]), 2.6, stroke=1, fill=1)
        label = str(row["row"])
        tx, ty = xy(max(x0, bx0) + 1.5 / sx, max(y0, by0) + 11 / sy)
        tw = pdfmetrics.stringWidth(label, "ReviewBold", 9)
        while any(tx - 1 < x1 + 2 and tx + tw + 1 > x0 - 2
                  and ty - 1.5 < y1 + 2 and ty + 9.5 > y0 - 2
                  for x0, y0, x1, y1 in label_bounds):
            tx += tw + 5
        label_bounds.append((tx - 1, ty - 1.5, tx + tw + 1, ty + 9.5))
        c.setFillColor(INK)
        c.rect(tx - 1, ty - 1.5, tw + 2, 11, fill=1, stroke=0)
        c.setFont("ReviewBold", 9)
        c.setFillColor(tint)
        c.drawString(tx, ty, label)
    c.restoreState()


def panel(c, root, image, title, crop, x, top, max_w, max_h,
          rows=(), held=(), proposal=False):
    iw, ih = crop[2] - crop[0], crop[3] - crop[1]
    scale = min(max_w / iw, max_h / ih)
    width, height = iw * scale, ih * scale
    left, bottom = x + (max_w - width) / 2, top - height
    c.setFont("ReviewBold", 11)
    c.setFillColor(INK)
    c.drawString(x, top + 17, title)
    with Image.open(root / image["source_image"]["path"]) as original:
        raw = original.convert("RGB").crop(crop)
        c.drawImage(ImageReader(raw), left, bottom, width, height)
    selected = [row for row in image["rows"] if row["row"] in rows]
    require(len(selected) == len(rows), "panel selects an unbound row")
    overlay(c, selected, crop, left, bottom, width, height, held)
    if proposal:
        row = next(row for row in selected if row["row"] == 12)
        y = bottom + height - (443 - crop[1]) * scale
        x0 = left + (max(crop[0], row["bbox_xyxy_px"][0]) - crop[0]) * scale
        x1 = left + (min(crop[2], row["bbox_xyxy_px"][2]) - crop[0]) * scale
        c.setStrokeColor(BLUE)
        c.setLineWidth(1.6)
        c.setDash(7, 3)
        c.line(x0, y, x1, y)
        c.setDash()
    text(c, "Source crop " + str(crop), x, bottom - 15, max_w, 9, 12, MUTED)
    return {"title": title, "crop_xyxy_px": crop, "selected_rows": list(rows),
            "held_rows": list(held), "draw_rect_pdf_pt": [left, bottom, width, height],
            "unapplied_top_443_guide": proposal}


def scope_note(c):
    text(c, "Display: selected full-review labels. The limited notebook keeps legacy labels for new030, new117 and new128 identically in both arms.",
         MARGIN, 52, PW - 2 * MARGIN, 9, 12, MUTED, min_y=45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review", type=Path,
                        default=Path("results/review/context-comparison-20261114/review.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path,
                        default=Path("results/review/context-comparison-20261114/render-audit.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    review_path = args.review if args.review.is_absolute() else root / args.review
    audit_path = args.audit if args.audit.is_absolute() else root / args.audit
    evidence = read(review_path)
    images, verified, checked_rows = verify_evidence(root, evidence)
    evidence_hash = sha(review_path)
    for name, filename in (("Review", "DejaVuSans.ttf"), ("ReviewBold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(Path("/usr/share/fonts/truetype/dejavu") / filename)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(args.output), pagesize=(PW, PH), pageCompression=1, invariant=1)
    c.setTitle("Cassette context review - 14 November 2026")
    c.setAuthor("Siuuuers/dl - assisted annotation review")
    c.setSubject("Four original-image comparisons; existing labels and explicitly unapplied proposal")
    pages = []

    iid = "new_000038"
    image = images[iid]
    heading(c, "Three thumb-side fronts are already labelled", iid, "EXISTING COUNT CONFIRMED", 1, evidence_hash)
    c.setFillColor(PAPER)
    c.roundRect(MARGIN, 714, PW - 2 * MARGIN, 42, 5, fill=1, stroke=0)
    text(c, "READY FOR THE CONTROLLED EXPERIMENT | Current pinned notebook: 137 training photos, 3,788 objects. No training run is claimed here.",
         43, 731, PW - 86, 12, 17, BLUE)
    panels = [panel(c, root, image, "Original pixels - the second attachment matches this image",
                    [470, 560, 640, 665], 30, 675, 545, 337),
              panel(c, root, image, "Current saved rows 1, 16 and 17",
                    [470, 560, 640, 665], 615, 675, 545, 337, (1, 16, 17))]
    y = text(c, "Attachment 02-426184.jpg matches new_000038: the nail crease, thumb outline, gripper edge, rack pattern and three stepped contours agree. This is a different image from the earlier two-front new_000037 example.",
             30, 298, PW - 60, 14, 21)
    y = text(c, "Decision: the three fronts are already separate rows in both the full review and the notebook's limited candidate. Green points show the three saved left lower corners; all three right lower corners remain missing. Your confirmation supports count and identity, not new pixel coordinates.",
             30, y - 10, PW - 60, 14, 21)
    text(c, "No extra object or label edit is needed. Run the existing pinned notebook for the controlled comparison.",
         30, y - 10, PW - 60, 14, 21, BLUE)
    url = evidence["notebook"]["colab_url"]
    text(c, "Open the pinned Colab notebook:", 30, 111, PW - 60, 11, 15, BLUE)
    text(c, url, 30, 92, PW - 60, 9, 13, BLUE)
    c.linkURL(url, (30, 82, PW - 30, 126), relative=0, thickness=0)
    scope_note(c)
    pages.append({"page": 1, "image_id": iid, "decision_id": "new038-three-thumb-fronts", "panels": panels})
    c.showPage()

    iid = "new_000030"
    image = images[iid]
    heading(c, "The upper patch has a credible owner", iid, "PARTIAL ASSOCIATION; CASE OPEN", 2, evidence_hash)
    panels = [panel(c, root, image, "Neighboring tiers and the gripper", [580, 305, 1000, 535],
                    30, 730, 650, 356, (9, 10, 11, 12, 20, 21, 22), (12,), True),
              panel(c, root, image, "Current row 12 and the proposed top guide", [620, 425, 805, 525],
                    715, 730, 445, 300, (12,), (12,), True)]
    text(c, "Amber dashed box: saved row 12.\nBlue dashed line: nominal y443 review proposal.\nThe blue guide is NOT an active label.\nBoth actual lower corners remain missing.",
         715, 451, 445, 12, 18)
    y = text(c, "The marked right shoulder lies below row 11 and beside the divider. Its tier and upper extent agree with the upper gray material in the opening; the gripper explains the gap between the visible pieces. This is stronger ownership evidence than material identity alone.",
             30, 320, PW - 60, 14, 21)
    y = text(c, "The lower dark triangle continues to about y500-502. Neighboring seams make a following front plausible, but do not establish another instance. The orange stroke's endpoint is a hand-drawn cue, not a measured physical bottom edge.",
             30, y - 10, PW - 60, 14, 21)
    text(c, "Decision: keep the owner/count question open. Top y455.72 -> nominal y443 is recorded only as an unapplied review proposal (about 2-3 px edge uncertainty). The inherited horizontal and lower bounds are not newly certified. No box, corner or training input changes.",
         30, y - 10, PW - 60, 14, 21)
    scope_note(c)
    pages.append({"page": 2, "image_id": iid, "decision_id": "new030-upper-patch-association", "panels": panels})
    c.showPage()

    iid = "new_000117"
    image = images[iid]
    heading(c, "A thin band can have competing owners", iid, "HOLD: FRONT OR MOLDED TRANSITION", 3, evidence_hash)
    panels = [panel(c, root, image, "Original pixels around the hand", [735, 350, 990, 535],
                    30, 730, 395, 300),
              panel(c, root, image, "Saved broad front 23 and disputed band 24", [735, 350, 990, 535],
                    445, 730, 395, 300, (23, 24), (24,)),
              panel(c, root, image, "Same-image tiers 20 -> 21 -> 22", [580, 350, 750, 480],
                    870, 730, 290, 222, (20, 21, 22)),
              panel(c, root, image, "Terminal front 25 and its lower rim", [915, 470, 1075, 540],
                    870, 459, 290, 127, (25,))]
    y = text(c, "The right-hand stack makes a further hand-covered tier plausible. However, the same photo also shows narrow molded or roof transitions between rows 20, 21 and 22, and at the terminal lower rim of row 25. A narrow band alone does not uniquely identify another target front.",
             30, 298, PW - 60, 14, 21)
    y = text(c, "Decision: retain the ownership hold on row 24. Its saved full-review box is [815,491,902,503], with both lower corners missing. No small-object exclusion, extra row, deletion or inferred corner is adopted.",
             30, y - 10, PW - 60, 14, 21)
    text(c, "The separate top-strip question for row 11 also remains open. The current limited experiment retains this image's PR35 legacy geometry identically in both arms.",
         30, y - 10, PW - 60, 13, 20, MUTED)
    scope_note(c)
    pages.append({"page": 3, "image_id": iid, "decision_id": "new117-thin-band-comparison", "panels": panels})
    c.showPage()

    iid = "new_000128"
    image = images[iid]
    heading(c, "Clipped strips: compare the visible transitions", iid, "HOLD: OWNERSHIP, NOT SIZE", 4, evidence_hash)
    panels = [panel(c, root, image, "Full top context - amber rows 1, 6, 9; cyan fronts 2, 7, 10", [340, 0, 1280, 190],
                    30, 730, PW - 60, 230, (1, 2, 6, 7, 9, 10), (1, 6, 9))]
    for x, left, right, title in ((30, 370, 650, "Left column: rows 1 / 2"),
                                 (415, 820, 1080, "Middle column: rows 6 / 7"),
                                 (800, 1040, 1280, "Right column: rows 9 / 10")):
        panels.append(panel(c, root, image, title + " | y0-35", [left, 0, right, 35],
                            x, 442, 360, 54))
        panels.append(panel(c, root, image, "Same column, lower transition | y48-85", [left, 48, right, 85],
                            x, 333, 360, 56))
    y = text(c, "The top strips resemble molded or roof transitions visible lower in the same columns. Clipping removes the region where a preceding broad front would be visible, so a preceding-front remnant remains a viable alternative. The three columns are not equally exposed.",
             30, 228, PW - 60, 14, 21)
    text(c, "Decision: keep rows 1, 6 and 9 pending. Their roughly five-pixel height is not a deletion rule. Keep all missing corners missing; add no object and make no group deletion from neighboring spacing alone.",
         30, y - 10, PW - 60, 14, 21)
    scope_note(c)
    pages.append({"page": 4, "image_id": iid, "decision_id": "new128-clipped-strips-comparison", "panels": panels})
    c.showPage()
    c.save()

    audit = {"schema_version": 1, "kind": "context_comparison_render",
             "source_review": {"path": review_path.relative_to(root).as_posix(), "sha256": evidence_hash},
             "generator": {"path": Path(__file__).resolve().relative_to(root).as_posix(), "sha256": sha(Path(__file__))},
             "pdf_filename": args.output.name, "pdf_sha256": sha(args.output),
             "pdf_bytes": args.output.stat().st_size, "page_count": 4,
             "verified_file_descriptors": [{"path": p, "sha256": h} for p, h in sorted(verified.items())],
             "verified_saved_rows": checked_rows, "source_images": 4,
             "rendering": {"photos": "lossless crops of original PNG pixels", "annotations": "PDF vectors",
                           "proposal": "blue dashed top guide only; no changed saved label",
                           "model_inference": False, "source_files_modified": False},
             "limited_candidate_legacy_equal_both_arms": ["new_000030", "new_000117", "new_000128"],
             "new038_three_fronts_already_in_limited_candidate": True,
             "colab_url": url, "pages": pages}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("pdf_filename", "pdf_sha256", "pdf_bytes", "page_count", "verified_saved_rows")}, indent=2))


if __name__ == "__main__":
    main()
