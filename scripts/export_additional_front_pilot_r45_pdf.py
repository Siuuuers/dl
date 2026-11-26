#!/usr/bin/env python3
"""Render the verified PR133 A/B result and every R38 candidate miss to PDF."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = audit(args.archive, root)
    with zipfile.ZipFile(args.archive) as archive:
        summary = json.loads(archive.read("evaluation/summary.json"))
        predictions = json.loads(archive.read("evaluation/predictions.json"))
    gold = {}
    for name, (relative, _, _) in REFERENCES.items():
        path = root / relative
        records = json.loads(path.read_text())["files"]
        gold[name] = {e["image_id"]: parse_pose_labels((path.parent / e["label_path"]).read_text(), 1280, 720) for e in records if e.get("split", "val") == "val"}
    cases = []
    r38 = summary["references"]["r38"]["postclip"]
    for image, result in r38["candidate"]["per_image"].items():
        for index in result["unmatched_gold_indices"]:
            newly = index not in r38["reference"]["per_image"][image]["unmatched_gold_indices"]
            cases.append(("r38", image, index + 1, "Newly missed" if newly else "Missed by both"))
    for image, delta in r38["identity_delta"].items():
        cases.extend(("r38", image, row, "Recovered") for row in delta["recovered_rows"])
    frozen = summary["references"]["frozen666"]["postclip"]["identity_delta"]
    for image, delta in frozen.items():
        cases.extend(("frozen666", image, row, "Recovered under frozen666") for row in delta["recovered_rows"])
    historical = summary["references"]["historical663"]["postclip"]["identity_delta"]
    for image, delta in historical.items():
        cases.extend(("historical663", image, row, "Newly missed under historical663") for row in delta["newly_missed_rows"])
    assert len(cases) == 18 and sum(c[0] == "r38" and c[3] != "Recovered" for c in cases) == 13
    records = {e["id"]: e for e in json.loads((root / "data/manifest.json").read_text())["records"]}
    args.images_dir.mkdir(parents=True, exist_ok=True)
    originals, image_hashes = {}, {}
    from PIL import Image, ImageDraw
    for identity in sorted({case[1] for case in cases}):
        record = records[identity]
        path = args.images_dir / (identity + ".png")
        if not path.exists() and args.download_images:
            url = f"https://media.githubusercontent.com/media/Siuuuers/dl/{SOURCE}/data/{record['image']}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != record["image_sha256"]:
                raise ValueError("Downloaded original differs: " + identity)
            path.write_bytes(data)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["image_sha256"]:
            raise ValueError("Original differs: " + identity)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        assert im.size == (1280, 720)
        originals[identity] = im
        image_hashes[identity] = record["image_sha256"]

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if font.exists():
        pdfmetrics.registerFont(TTFont("Report", str(font)))
        pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
        regular, strong = "Report", "ReportBold"
    else:
        regular, strong = "Helvetica", "Helvetica-Bold"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.output), pagesize=A4, invariant=1)
    pdf.setTitle("R45 pilot results and missed-front review")
    pdf.setAuthor("Siuuuers/dl project review")
    W, H = A4
    M, WIDTH = 36, W - 72
    ink, muted, teal, line = map(HexColor, ("#172b3a", "#546775", "#007f88", "#d8e3e7"))
    page = 0

    def text(x, y, value, size=10, bold=False, color=None):
        pdf.setFillColor(color or ink)
        pdf.setFont(strong if bold else regular, size)
        pdf.drawString(x, y, str(value))

    def paragraph(value, x, y, width=WIDTH, size=10, leading=15, color=None):
        words = str(value).split()
        current = ""
        for word in words:
            candidate = (current + " " + word).strip()
            if pdfmetrics.stringWidth(candidate, regular, size) > width and current:
                text(x, y, current, size, color=color)
                y -= leading
                current = word
            else:
                current = candidate
        if current:
            text(x, y, current, size, color=color)
            y -= leading
        return y

    def start(section, title, subtitle):
        nonlocal page
        page += 1
        pdf.setFillColor(teal)
        pdf.rect(0, H - 8, W, 8, fill=1, stroke=0)
        text(M, H - 37, section.upper(), 9, bold=True, color=teal)
        text(M, H - 69, title, 22, bold=True)
        paragraph(subtitle, M, H - 90, size=10, color=muted)

    def end():
        pdf.setStrokeColor(line)
        pdf.line(M, 38, W - M, 38)
        text(M, 24, "PR133 | 27 Nov 2026 HKT | Fixed predictions; no model promotion", 8, color=muted)
        text(W - M - 20, 24, f"{page:02d}", 9, bold=True, color=teal)
        pdf.showPage()

    def table(headers, rows, x, y, widths, row_height=27, size=9):
        pdf.setFillColor(HexColor("#e8f2f3"))
        pdf.rect(x, y - row_height, sum(widths), row_height, fill=1, stroke=0)
        xx = x
        for head, width in zip(headers, widths):
            text(xx + 7, y - 18, head, size, bold=True)
            xx += width
        y -= row_height
        for n, row in enumerate(rows):
            if n % 2 == 0:
                pdf.setFillColor(HexColor("#f5f8fa"))
                pdf.rect(x, y - row_height, sum(widths), row_height, fill=1, stroke=0)
            xx = x
            for item, width in zip(row, widths):
                text(xx + 7, y - 18, item, size)
                xx += width
            y -= row_height
        return y - 18

    start("Verified training result", "One added photo. A mixed result.", "R45 reconstructed supplementary-front pilot | Reference versus candidate")
    y = paragraph("Yes: the candidate now trained frame_0383. Earlier documented runs used the original new_* / old_* photos; the supplementary frame_* pool was reviewed but had not entered this recorded training lineage.", M, 715, size=11, leading=17)
    y -= 10
    y = table(["Measured training item", "Reference", "Candidate"], [
        ("Photographs", "137", "138"), ("Fronts", "3,784", "3,827"),
        ("Added supplementary photo", "None", "frame_0383"), ("Locatable corner slots", "6,837", "6,923"),
        ("Missing corner slots", "731", "731"), ("Epochs / optimizer updates", "30 / 78", "30 / 78"),
        ("Image presentations", "4,110", "4,140")], M, y, [269, 119, WIDTH - 388])
    y = paragraph("Both runs used the same original 137-photo labels, fresh epoch173 initialization, seed46 recipe and T4 runtime. The 20 original development images stayed outside training. The other eleven reviewed supplementary photos stayed held.", M, y - 4, size=10)
    y -= 14
    text(M, y, "Fixed postclip result: reference to candidate", 12, bold=True)
    y = table(["Reference", "Misses", "Unmatched", "Duplicates", "Other"], [
        ("Frozen666", "36 to 33", "99 to 98", "25 to 22", "74 to 76"),
        ("Historical663", "32 to 30", "98 to 98", "25 to 22", "73 to 76"),
        ("R38 current654", "11 to 13", "86 to 90", "22 to 18", "64 to 72")], M, y - 14, [139, 84, 105, 99, WIDTH - 427])
    y = paragraph("Duplicates are part of the unmatched total. R38's candidate has 90 unmatched = 18 duplicates + 72 other unmatched. All three references describe the same development photos using different target definitions.", M, y, size=10)
    paragraph("Decision: keep PR35 selected. This pilot supports further annotation work; it does not establish a general accuracy improvement or independent-session reliability.", M, y - 12, size=10, color=teal)
    end()

    start("Why geometry matters", "Fewer duplicates can hide more misses.", "All ratios include every eligible reference front, including a missed front.")
    y = table(["Reference", "Both corners within 5px", "Both within 10px"], [
        ("Frozen666", "517 to 518 / 584", "573 to 575 / 584"),
        ("Historical663", "517 to 518 / 584", "573 to 575 / 584"),
        ("R38 current654", "571 to 569 / 583", "582 to 581 / 583")], M, 715, [139, 202, WIDTH - 341])
    y = paragraph("R38: both models detect all 583 fronts with two locatable corners. The one-corner group drops from 46/54 detected to 44/54; its within-5px success drops from 41/54 to 40/54. The zero-corner group stays 14/17 detected.", M, y - 4, size=11, leading=17)
    y -= 14
    text(M, y, "Changed R38 missed-target identities", 12, bold=True)
    y = table(["Image / row", "Candidate outcome", "Best box IoU"], [
        ("new_000009 / 051", "Newly missed", "0.5381 to 0.4916"),
        ("new_000010 / 030", "Recovered", "0.3780 to 0.6145"),
        ("new_000011 / 059", "Newly missed", "0.7860 to 0.4702"),
        ("new_000089 / 048", "Newly missed", "0.5342 to 0.4973")], M, y - 15, [190, 164, WIDTH - 354])
    y = paragraph("These four targets each have one locatable corner. All three new misses still have an overlapping retained box; its IoU is below the unchanged 0.50 match criterion. In new_000089 / 048, confidence is 0.834. Lower confidence would not repair that existing box's geometry.", M, y - 4, size=10)
    y = paragraph("The existing postclip filter removes eight outputs in each arm without changing matches or corner scores. Both arms return zero predictions on the three empty scenes. Do not replace this fresh reference arm with counts from PR123/R40.", M, y - 12, size=10)
    y = paragraph("The added photo has 43 fronts with all 86 corners visible. It is one capture condition and does not add missing-corner supervision. The next useful step is to qualify more complete photos and freeze their corrected labels before another treatment comparison.", M, y - 12, size=10)
    paragraph("The following pages show all thirteen candidate R38 misses, its one recovery, and the four additional changed identities under the historical references. Row numbers belong to their named reference version.", M, y - 12, size=10, color=teal)
    end()

    start("Evidence and reading guide", "Verified bytes. Replayed scores.", "This review uses the returned prediction cache; no fresh GPU inference or training was performed.")
    y = paragraph("The reassembled 41,067,308-byte archive contains 335 members. All 334 inventory entries passed byte-size and SHA-256 checks. Training receipts, saved final checkpoint bytes, shared label lineage and source/runtime records were checked. All 40 image-role postclip results and all twelve score reports were reproduced exactly.", M, 715, size=11, leading=17)
    y = paragraph("Checkpoint hashes confirm the recorded files. They do not independently prove prediction generation without rerunning model inference. The archive's code and receipts provide that recorded provenance; this CPU audit verifies its consistency.", M, y - 12, size=10)
    y -= 15
    text(M, y, "How to read a case page", 12, bold=True)
    for value in [
        "Original crop: unaltered pixels, enlarged with nearest-neighbor sampling. The full-image locator marks the inspected front.",
        "Cyan: reference box and its locatable corners. Amber: one model prediction and its confident corner estimates. Missing reference corner slots are not invented.",
        "For a matched front, the panel uses the actual matched prediction. For a missed front, it uses the highest-IoU retained prediction if any overlaps. Confidence and overlap are shown separately.",
        "The match result uses greedy confidence-ranked matching across the full image, with box IoU at least 0.50. A local best-overlap view is a diagnostic aid, not a replacement evaluation rule.",
        "A missed target with no retained overlapping box cannot be classified as confidence loss, NMS loss or absent raw detection using the final cache alone."]:
        y = paragraph(value, M, y - 24, size=10)
    y -= 12
    text(M, y, "Sources", 12, bold=True)
    y = paragraph("github.com/Siuuuers/dl/pull/133", M, y - 22, size=9)
    y = paragraph("Source revision: " + SOURCE, M, y - 4, size=9)
    y = paragraph("Result revision: " + PR_HEAD, M, y - 4, size=9)
    y = paragraph("Archive SHA-256:", M, y - 4, size=9)
    y = paragraph(ARCHIVE_SHA, M, y - 2, size=8)
    y = paragraph("Repository evidence: results/evaluation/pr133-r45-20261127/cpu-audit.json; docs/additional-front-pilot-r45-results.md.", M, y - 12, size=9)
    paragraph("Reproduction: scripts/audit_additional_front_pilot_r45_results.py (CPU, Python standard library). This PDF additionally requires Pillow and ReportLab.", M, y - 6, size=9)
    end()

    def fitted(image, x, y, width, height):
        scale = min(width / image.width, height / image.height)
        w, h = image.width * scale, image.height * scale
        pdf.drawImage(ImageReader(image), x + (width - w) / 2, y + (height - h) / 2, width=w, height=h)

    def candidate_for(reference, identity, row, role):
        ps = predictions[role][identity]["postclip"]
        score = summary["references"][reference]["postclip"][role]["per_image"][identity]
        matched = next((m for m in score["matches"] if m["gold_index"] == row - 1), None)
        if matched:
            return ps[matched["prediction_index"]], matched["iou"], True
        box = gold[reference][identity][row - 1]["bbox_xyxy"]
        best = max(((_box_iou(p["bbox_xyxy"], box), n) for n, p in enumerate(ps)), default=(0, 0))
        return (ps[best[1]] if best[0] > 0 else None), best[0], False

    def crop_bounds(box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        width, height = max(66, box[2] - box[0] + 32), max(50, box[3] - box[1] + 28)
        width = max(width, height * 1.25)
        height = max(height, width / 1.45)
        x0 = min(max(0, cx - width / 2), 1280 - width)
        y0 = min(max(0, cy - height / 2), 720 - height)
        return tuple(map(int, (math.floor(x0), math.floor(y0), math.ceil(x0 + width), math.ceil(y0 + height))))

    def overlay(crop, bounds, target, pred):
        image = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        def point(x, y):
            return ((x - bounds[0]) * 4, (y - bounds[1]) * 4)
        def box(item, color):
            b = item["bbox_xyxy"]
            draw.rectangle((*point(b[0], b[1]), *point(b[2], b[3])), outline=color, width=3)
        box(target, "#00e5f2")
        for x, y, visible in target["keypoints"]:
            if visible == 2:
                px, py = point(x, y)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), outline="#00e5f2", width=3)
        if pred:
            box(pred, "#ffba45")
            for x, y, confidence in pred["keypoints"]:
                if x is not None and y is not None and confidence >= .5 and 0 <= x < 1280 and 0 <= y < 720:
                    px, py = point(x, y)
                    draw.line((px - 5, py, px + 5, py), fill="#ffba45", width=3)
                    draw.line((px, py - 5, px, py + 5), fill="#ffba45", width=3)
        return image

    for reference, identity, row, outcome in cases:
        target = gold[reference][identity][row - 1]
        visible = sum(point[2] == 2 for point in target["keypoints"])
        start("Missed-front detail / " + reference, f"{identity} / row {row:03d}", f"{outcome} | {visible} locatable reference corner(s) | Reference version: {reference}")
        bounds = crop_bounds(target["bbox_xyxy"])
        crop = originals[identity].crop(bounds)
        locator = originals[identity].copy()
        draw = ImageDraw.Draw(locator)
        b = target["bbox_xyxy"]
        draw.rectangle((max(0, b[0] - 8), max(0, b[1] - 8), min(1279, b[2] + 8), min(719, b[3] + 8)), outline="#ff2859", width=6)
        locator = locator.resize((512, 288), Image.Resampling.LANCZOS)
        gap, panel = 16, (WIDTH - 16) / 2
        text(M, 716, "Original crop", 10, bold=True)
        text(M + panel + gap, 716, "Full-image locator", 10, bold=True)
        fitted(crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST), M, 492, panel, 210)
        fitted(locator, M + panel + gap, 532, panel, 150)
        paragraph(f"Crop pixels: x {bounds[0]}..{bounds[2]}, y {bounds[1]}..{bounds[3]}. Image coordinates increase downward in y.", M + panel + gap, 518, panel, size=9, leading=13)
        details = []
        for offset, role in enumerate(("reference", "candidate")):
            pred, overlap, matched = candidate_for(reference, identity, row, role)
            details.append((pred, overlap, matched))
            x = M + offset * (panel + gap)
            text(x, 470, role.title() + (" / matched" if matched else " / missed"), 11, bold=True)
            fitted(overlay(crop, bounds, target, pred), x, 245, panel, 210)
            label = f"IoU {overlap:.4f} | conf {pred['confidence']:.3f}" if pred else "No retained overlapping prediction"
            text(x, 230, label, 9, color=muted)
        y = paragraph("Cyan = named reference. Amber = selected model prediction. Circles mark locatable reference corners; crosses mark confident predicted corners. All panels show the same original crop.", M, 204, size=9, leading=13)
        if outcome == "Newly missed":
            note = "The candidate still predicts an overlapping front, but its box is below the 0.50 matching threshold. Inspect front ownership and box extent; a confidence-only adjustment does not correct that existing box."
        elif outcome == "Missed by both":
            note = "This front remains unmatched in both arms at the fixed operating point. The displayed overlap is diagnostic; final cached outputs alone cannot establish whether an absent alternative was lost to confidence, NMS, or raw detection."
        elif "Recovered" in outcome:
            note = "The candidate recovers this identity under the named reference. This local recovery must be considered alongside newly missed fronts and other unmatched predictions; it is not a global improvement claim."
        else:
            note = "This identity changes under the historical663 definition. Keep its row and geometry separate from R38 and frozen666; differing reference versions are not independent test samples."
        y = paragraph(note, M, y - 9, size=10, leading=14)
        paragraph("Original SHA-256: " + image_hashes[identity][:24] + "... | Full hashes and replay results are in cpu-audit.json and the image source manifest.", M, 68, size=8, leading=11, color=muted)
        end()
    pdf.save()
    verification = {"pages_expected": 3 + len(cases), "candidate_r38_misses": 13, "r38_recoveries": 1,
                    "additional_historical_changed_cases": 4, "case_inventory": [dict(reference=n, image=i, row=r, outcome=o) for n, i, r, o in cases],
                    "original_image_sha256": image_hashes, "archive_sha256": ARCHIVE_SHA,
                    "pdf_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "pdf_bytes": args.output.stat().st_size,
                    "cpu_audit_status": evidence["status"], "fresh_inference": False}
    args.output.with_suffix(".verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(args.output, "pages", 3 + len(cases))


if __name__ == "__main__":
    main()
