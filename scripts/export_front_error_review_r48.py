#!/usr/bin/env python3
"""Render a complete R48 review of the fixed PR137 current-R38 error inventory.

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
import statistics
import urllib.request
import zipfile

from audit_additional_front_pilot_r47_results import audit, REFERENCES, SOURCE, INPUT_SOURCE, PR_HEAD, ARCHIVE_SHA
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image, aggregate_scores
from audit_additional_front_pilot_r45_results import corner_groups

PR136_HEAD = "d0136ae8b2283d38da953e8f109e542029682678"


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
    assert len(misses) == scores["candidate"]["detection"]["fn"]
    assert len(unmatched) == scores["candidate"]["detection"]["fp"]
    inventory = dict(schema="baomai-r48-fixed-prediction-error-review-v1", source_pr=137, source_pr_head=PR_HEAD,
                     review_predecessor_pr=136, review_predecessor_head=PR136_HEAD,
                     archive_sha256=ARCHIVE_SHA, reference="r38", reference_manifest_sha256=REFERENCES["r38"][1],
                     fresh_inference=False, fresh_training=False, reference_labels_modified=False,
                     postclip=True, box_confidence=0.13, ordinary_nms_iou=0.7, extra_postclip_iou_strictly_greater_than=0.7,
                     matching_iou=0.5, endpoint_confidence=0.5,
                     counts=dict(misses=len(misses), recoveries=len(recovered), matched_corner_failures=len(corners), fragile_matches=len(fragile),
                                 unmatched=len(unmatched), **categories),
                     scope="Complete candidate current-R38 inventory at the fixed PR137 operating point. Other unmatched is an assignment category, not a physical background verdict.",
                     per_image=per_image, misses=misses, recoveries=recovered, corner_failures=corners,
                     fragile_matches=fragile, unmatched_predictions=unmatched)
    inventory["diagnostics"] = diagnostics(inventory, summary, predictions, gold, archive_path)
    return inventory, evidence, summary, gold


def diagnostics(inventory, summary, predictions, gold, archive_path):
    visibility, extra_visibility = {}, {}
    with zipfile.ZipFile(archive_path) as archive:
        plan = json.loads(archive.read("staging.json"))
        for role, entries in plan["roles"].items():
            groups = Counter({str(n): 0 for n in (0, 1, 2)})
            for entry in entries:
                if entry["split"] != "train":
                    continue
                rows = parse_pose_labels(archive.read(f"datasets/{role}/{entry['label']}").decode(), 1280, 720)
                local = Counter(str(sum(p[2] == 2 for p in row["keypoints"])) for row in rows)
                groups.update(local)
                if role == "candidate" and entry["id"] in plan["treatment"]["candidate_extra_ids"]:
                    extra_visibility[entry["id"]] = {str(n): local[str(n)] for n in (0, 1, 2)}
            visibility[role] = dict(groups)
    scores = summary["references"]["r38"]["postclip"]
    target_groups = {str(n): Counter(targets=0, reference_misses=0, candidate_misses=0) for n in (0, 1, 2)}
    heights = {name: [] for name in ("missed_partial", "matched_partial", "matched_two_corner")}
    height_bins = {name: Counter(targets=0, reference_misses=0, candidate_misses=0) for name in ("under8", "8_to_under16", "16_to_under32", "32_or_more")}
    missed_geometry = []
    for identity, targets in gold.items():
        a, b = (scores[role]["per_image"][identity] for role in ("reference", "candidate"))
        for row, target in enumerate(targets):
            count = str(sum(p[2] == 2 for p in target["keypoints"]))
            ref_missed, cand_missed = row in a["unmatched_gold_indices"], row in b["unmatched_gold_indices"]
            group = target_groups[count]
            group["targets"] += 1
            group["reference_misses"] += ref_missed
            group["candidate_misses"] += cand_missed
            x0, y0, x1, y1 = target["bbox_xyxy"]
            width, height = x1 - x0, y1 - y0
            key = "missed_partial" if cand_missed and count != "2" else "matched_partial" if count != "2" else "matched_two_corner"
            heights[key].append(height)
            rounded = round(height, 3)
            name = "under8" if rounded < 8 else "8_to_under16" if rounded < 16 else "16_to_under32" if rounded < 32 else "32_or_more"
            group = height_bins[name]
            group["targets"] += 1
            group["reference_misses"] += ref_missed
            group["candidate_misses"] += cand_missed
            if cand_missed:
                missed_geometry.append(dict(image=identity, row=row + 1, width_px=width, height_px=height,
                                            area_px2=width * height, locatable_corners=int(count)))
    nearest_visibility = Counter({"0": 0, "1": 0, "2": 0, "no_reference_overlap": 0})
    for case in inventory["unmatched_predictions"]:
        row = case["best_row"]
        key = str(sum(p[2] == 2 for p in gold[case["image"]][row - 1]["keypoints"])) if row else "no_reference_overlap"
        nearest_visibility[key] += 1
    sweep = []
    for confidence in (.13, .20, .25, .30, .40, .50, .60):
        point = dict(confidence=confidence)
        for role in ("reference", "candidate"):
            per_image, groups = [], {str(n): Counter() for n in (0, 1, 2)}
            for identity, targets in gold.items():
                filtered = [p for p in predictions[role][identity]["postclip"] if p["confidence"] >= confidence]
                result = score_image(targets, filtered, 1280, 720, .5, .5)
                per_image.append(result)
                for key, value in corner_groups(targets, filtered, result).items():
                    groups[key].update(value)
            aggregate = aggregate_scores(per_image)
            point[role] = dict(detection={k: aggregate[k] for k in ("tp", "fn", "fp", "duplicate_fp", "other_fp")},
                               corners_by_locatable_count=groups)
        sweep.append(point)
    height_summary = {name: dict(count=len(values), min_px=min(values), median_px=statistics.median(values), max_px=max(values))
                      for name, values in heights.items()}
    failure_types = Counter("coordinate" if all(p["accepted"] for p in case["failing_corners"]) else "endpoint_rejected"
                            for case in inventory["corner_failures"])
    return dict(training_visibility=visibility, supplementary_visibility=extra_visibility,
                current_r38_target_visibility=target_groups, target_height_bins=height_bins,
                height_bin_note="Original-image vertical extent, rounded to 0.001px only to remove TXT floating-point boundary noise.",
                candidate_height_summary=height_summary, candidate_missed_geometry=missed_geometry,
                unmatched_best_target_visibility=nearest_visibility,
                unmatched_visibility_note="Highest-IoU reference association; no physical ownership label is inferred.",
                corner_failure_types=failure_types, confidence_replay=sweep,
                confidence_replay_scope="Exploratory post-hoc filtering of both arms' retained PR137 postclip predictions on the same development set, then full scorer replay. It cannot recover outputs removed before caching, tests no image enhancement, and does not select a new deployment threshold.",
                postclip_effect={role: {view: {key: summary['references']['r38'][view][role]['detection'][key]
                                               for key in ('prediction_count','fn','fp','duplicate_fp','other_fp')}
                                       for view in ('raw','postclip')} for role in ('reference','candidate')})


def originals_for(inventory, root, images_dir, download):
    from PIL import Image
    records = {item["id"]: item for item in json.loads((root / "data/manifest.json").read_text())["records"]}
    identities = sorted({case["image"] for key in ("misses", "recoveries", "corner_failures", "fragile_matches", "unmatched_predictions") for case in inventory[key]})
    images_dir.mkdir(parents=True, exist_ok=True)
    def obtain(identity):
        record, path = records[identity], images_dir / (identity + ".png")
        if not path.exists() and download:
            url = f"https://media.githubusercontent.com/media/Siuuuers/dl/{INPUT_SOURCE}/data/{record['image']}"
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
    pdf.setTitle("R48 - PR137 missed, unmatched and corner-error review")
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
        text(M, 17, "R48 review | 27 Nov 2026 HKT | PR137 cached predictions | Frozen R38 labels unchanged", 7, color=muted)
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

    start("Result status", "R47 training completed; the outcome is mixed.", "R48 review of PR137: the 138-photo reference versus the 142-photo supplementary-data candidate.")
    y = paragraph("Both arms started independently from epoch173 and completed 30 epochs on the same Tesla T4 runtime. Reference: 1,050 batches / 78 optimizer updates. Candidate: 1,080 batches / 80 updates. The candidate adds four newly complete photos; extra exposure is part of this equal-epoch treatment.", M, 484, size=11, leading=16)
    y = table(["Current R38 / 654 reference fronts", "Reference", "Candidate", "Change"], [
        ("Missed fronts", "13", "15", "+2"), ("Unmatched predictions (all)", "90", "84", "-6"),
        ("Formal duplicate subset", "18", "14", "-4"), ("Other unmatched subset", "72", "70", "-2"),
        ("Two-corner fronts within 5px / eligible", "569 / 583", "566 / 583", "-3"),
        ("Two-corner fronts within 10px / eligible", "581 / 583", "582 / 583", "+1")], M, y - 12, [358, 135, 135, BW - 628])
    y = paragraph("The reduction in unmatched outputs comes with two additional misses and three fewer two-corner fronts within 5px. The frozen666 result is also mixed. These results do not justify model promotion; PR35 remains selected.", M, y, size=10, leading=15)
    y = paragraph("This report includes all 15 candidate misses, all 84 unmatched outputs and all 20 matched fronts failing the 5px corner check. There are no recovered current-R38 targets and no candidate matches in the diagnostic IoU interval [0.50, 0.55). A miss and unmatched output can describe the same physical cassette.", M, y - 8, size=10, leading=15)
    paragraph("The returned run is real training evidence. This report itself replays saved evidence on CPU; it performs no fresh inference or additional training.", M, y - 10, size=10, leading=15, color=teal)
    end()

    start("How to read the evidence", "Unmatched is an assignment outcome.", "All thresholds and R38 labels remain fixed; diagnostic bins do not change the matching rule.")
    y = table(["Candidate unmatched category", "Count", "What the number establishes"], [
        ("Formal duplicate", "14", "IoU >= 0.50 with an already matched reference front"),
        ("Other: best IoU 0.40 to below 0.50", "21", "Retained output overlaps near the fixed match gate"),
        ("Other: best IoU above 0 to below 0.40", "37", "Some reference overlap; inspect geometry and identity"),
        ("Other: zero overlap with every R38 box", "12", "No R38 box overlap; physical background is unproven")], M, 485, [300, 60, BW - 360], size=8.5)
    for value in [
        "R### is a one-based row in the frozen R38 development label file. P### IN THIS PDF is the PR137 candidate's postclip output index. Earlier R47 PDF P### identities refer to PR133 and cannot be reused by number alone. The source archive and primary output ID are retained in the JSON index.",
        "Cyan boxes/circles show the named R38 target and locatable corners. Amber boxes/crosses show the model output and accepted endpoints. A dotted amber box identifies the output actually assigned to that target. Each case preserves an unannotated original crop.",
        "For misses, each arm shows the assigned output if matched, otherwise the highest-IoU overlapping retained output. This diagnostic association does not replace confidence-ranked full-image greedy matching. No overlap is not proof of physical background.",
        "All original pixels are hash checked. Crops use nearest-neighbor enlargement. In stored image coordinates, ymin is TOP and ymax is BOTTOM. Missing reference corners are never invented. Diagnostic contours and predictions do not authorize including neighboring cassette material.",
        "The two newly missed R38 targets are new_000008/R001 (best retained box IoU 0.3906) and new_000010/R030 (0.3296). They both have one locatable corner; all 583 two-corner R38 targets remain matched.",
        "The cache contains outputs after ordinary NMS and the original confidence gate. It cannot reveal suppressed alternatives or diagnose an image filter without new inference. Some other-unmatched boxes may repeat a physical cassette but fall below the formal duplicate-IoU gate."]:
        y = paragraph(value, M, y - 8, size=9, leading=13)
    end()

    start("Coverage by photograph", "Crowded partial fronts dominate the remaining errors.", "The table includes all 20 development photographs; three empty scenes remain free of predictions.")
    rows = []
    for item in inventory["per_image"]:
        a, b = item["reference"], item["candidate"]
        rows.append((item["image"], item["gold"], f"{a['fn']} / {b['fn']}", f"{a['fp']} / {b['fp']}", f"{a['duplicate_fp']} / {b['duplicate_fp']}", f"{a['other_fp']} / {b['other_fp']}"))
    table(["Photograph", "Fronts", "Miss ref / cand", "Unmatched ref / cand", "Duplicate ref / cand", "Other ref / cand"], rows, M, 485,
          [145, 66, 130, 160, 150, BW - 651], row_height=18, size=8)
    paragraph("new_000009, new_000011, new_000089 and new_000090 contain 55 of 84 unmatched outputs and 9 of 15 misses. These are fixed-reference assignment counts; the original crops must establish each physical cause.", M, 88, size=9, leading=13)
    end()

    start("Visibility and size", "The added data mostly covers fully locatable fronts.", "The treatment adds 172 fronts: 164 with two locatable corners, eight with one, and none with zero.")
    y = table(["Locatable reference corners", "Training ref / cand", "R38 targets", "Miss ref / cand"], [
        ("Two corners", "3,233 / 3,397", "583", "0 / 0"),
        ("One corner", "457 / 465", "54", "10 / 12"),
        ("Zero corners", "137 / 137", "17", "3 / 3")], M, 485, [250, 200, 140, BW - 590])
    y = paragraph("frame_0146, frame_0160 and frame_0371 each add 43 two-corner fronts. frame_0178 adds 35 two-corner and eight one-corner fronts. Shared frame_0383 is unchanged. The observed failure stratum therefore received comparatively little new supervision. This association is not causal proof of why a model changed.", M, y, size=10, leading=15)
    y = table(["Candidate / original-image box height", "Targets", "Min", "Median", "Max"], [
        ("Missed partial fronts", "15", "10.00px", "23.00px", "45.56px"),
        ("Matched partial fronts", "56", "11.00px", "26.18px", "60.00px"),
        ("Matched two-corner fronts", "583", "9.50px", "29.00px", "67.00px")], M, y - 6, [330, 80, 110, 110, BW - 630], size=9)
    y = paragraph("Short visible height makes box overlap sensitive to small boundary displacement. But missed partial fronts span 10 to 45.56 pixels, and some fully locatable fronts match at 9.5 pixels. These results do not establish a single size or dimness cause. Diagnose occlusion, boundary definition and resolution separately.", M, y, size=10, leading=15)
    paragraph("Of 84 unmatched outputs, the highest-overlap reference has one locatable corner for 56 outputs, zero for four, and two for twelve. Twelve outputs have no reference overlap. These are diagnostic associations, not validated physical identities.", M, y - 10, size=10, leading=15)
    end()

    start("Filter trade-off", "Fewer outputs alone is not a better detector.", "Exploratory post-hoc replay on the development set; both arms, unchanged matching and endpoint gates.")
    sweep_rows = []
    for point in inventory["diagnostics"]["confidence_replay"]:
        a, b = point["reference"], point["candidate"]
        da, db = a["detection"], b["detection"]
        ca, cb = a["corners_by_locatable_count"], b["corners_by_locatable_count"]
        sweep_rows.append((f"{point['confidence']:.2f}", f"{da['fn']} / {db['fn']}", f"{da['fp']} / {db['fp']}",
                           f"{da['duplicate_fp']} / {db['duplicate_fp']}", f"{ca['2']['within5']} / {cb['2']['within5']}",
                           f"{ca['1']['within5']} / {cb['1']['within5']}"))
    y = table(["Confidence", "Miss R / C", "Unmatched R / C", "Duplicate R / C", "2-corner 5px R / C", "1-corner 5px R / C"],
              sweep_rows, M, 485, [90, 110, 145, 140, 150, BW - 635], size=8)
    y = paragraph("R / C means reference / candidate. The 5px two-corner success denominator is 583 in each arm; the one-corner denominator is 54. All within10px counts and zero-corner detection denominators are in the JSON. The deployed confidence remains 0.13.", M, y, size=9, leading=13)
    y = paragraph("The existing extra postclip NMS already removes nine candidate outputs: raw 732 becomes 723, unmatched 93 becomes 84, formal duplicates 22 becomes 14, and misses remain 15. This demonstrates a useful existing filter at this operating point.", M, y, size=10, leading=15)
    y = paragraph("Filtering retained outputs at higher confidence reduces unmatched outputs while also losing true matches. At 0.25, unmatched drops by 33 but misses increase by three. The cache cannot test lower confidence, earlier NMS alternatives, image enhancement or higher-resolution inference because those outputs were never saved.", M, y - 10, size=10, leading=15)
    paragraph("For the next experiment, first settle front ownership and focus training evidence on the observed partial-front cases. If testing tiling, enhancement or a learned filter, change one factor and retain recall, endpoint success and original-image geometry as joint criteria. Tune on development data and reserve a separate capture session for final testing.", M, y - 10, size=10, leading=15, color=teal)
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
            note = "Recovered under current R38; assess alongside every newly missed target, not in isolation."
        elif case["title"] == "Newly missed":
            note = "A retained box exists below IoU 0.50. Prioritize visible-front extent and ownership, not lower confidence."
        else:
            note = "Unmatched in both arms. The highest-overlap view is diagnostic; absent pre-filter alternatives are unknown."
        paragraph(note, x + 10, y + 22, width - 20, size=7.6, leading=10)
        case_pages.append(dict(kind=case["kind"], image=case["image"], row=case["row"], page=page))

    target_groups = [
        ("Missed targets", "All 15 candidate misses; no recovered or fragile matches", inventory["misses"] + inventory["recoveries"] + inventory["fragile_matches"]),
        ("Corner localization errors", "All 20 matched fronts that fail the 5px check", inventory["corner_failures"]),
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
        text(x + 8 + pw + gap, y + 41, "R38 / PR137 candidate", 7.5, True)
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
        ("Other unmatched outputs", "All 70 other unmatched predictions", [c for c in inventory["unmatched_predictions"] if c["kind"] != "duplicate"]),
        ("Formal duplicate outputs", "All 14 formal duplicate predictions", [c for c in inventory["unmatched_predictions"] if c["kind"] == "duplicate"]),
    ]
    for section, title, cases in unmatched_groups:
        for offset in range(0, len(cases), 4):
            start(section, title, f"Cases {offset + 1}-{min(offset + 4, len(cases))} of {len(cases)} | P### identifies a model output, not a source annotation row.")
            for index, case in enumerate(cases[offset:offset + 4]):
                unmatched_card(case, M + (index % 2) * ((BW + 12) / 2), 271 if index < 2 else 42, (BW - 12) / 2, 217)
            end()

    start("Sources and limits", "A complete audit of this fixed operating point.", "The JSON index retains exact coordinates, model-output identities, diagnostic tables and every case page.")
    y = paragraph("The CPU audit passed all 499 inventory entries, all 81 archived source files, all 40 image/role postclip replays and all 12 reference/view/arm score reports. Exact source, labels, whole-photo eligibility, runtime and measured optimizer-update indices agree. Checkpoint hashes bind returned files; this audit does not independently regenerate their predictions.", M, 484, size=10, leading=15)
    y = table(["Reference version / same 20 development photos", "Miss ref / cand", "Unmatched ref / cand", "Duplicates ref / cand"], [
        ("Frozen666 historical primary", "33 / 34", "98 / 91", "22 / 17"),
        ("Historical663 diagnostic", "30 / 30", "98 / 90", "22 / 17"),
        ("R38 current654 / cards in this report", "13 / 15", "90 / 84", "18 / 14")], M, y - 12, [350, 130, 160, BW - 640], size=8)
    y = paragraph("These label versions describe the same photographs, not three independent test sets. R48 amendments are reviewed separately and do not overwrite any frozen reference used here. A diagnostic rescore under a later annotation version would be a label-version comparison, not a model-training improvement.", M, y, size=10, leading=15)
    for value in [
        "Training return: https://github.com/Siuuuers/dl/pull/137",
        "Prepared source / notebook: https://github.com/Siuuuers/dl/pull/136",
        "PR137 return revision: " + PR_HEAD,
        "R47 recorded source revision: " + SOURCE,
        "Archive SHA-256: " + ARCHIVE_SHA,
        "R38 manifest SHA-256: " + REFERENCES["r38"][1],
        "Scorer: scripts/pose_scoring.py; audit: scripts/audit_additional_front_pilot_r47_results.py",
        "Renderer: scripts/export_front_error_review_r48.py; no model loading or reference-label writes."]:
        y = paragraph(value, M, y - 7, size=8, leading=12)
    paragraph("Keep PR35 selected. Prioritize consistent partial-front supervision and measure miss/unmatched/corner trade-offs together. The confidence replay is exploratory; additional image processing has not been tested by this cached-prediction audit.", M, y - 12, size=10, leading=15, color=teal)
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
    verification = dict(schema="baomai-r48-error-pdf-verification-v1", pages=pages, pdf_bytes=args.output.stat().st_size,
                        pdf_sha256=sha(args.output.read_bytes()), index_sha256=sha(index.read_bytes()),
                        case_count=len(case_pages), counts=inventory["counts"],
                        original_image_sha256=hashes, cpu_audit_status=evidence["status"],
                        archive_sha256=ARCHIVE_SHA, fresh_inference=False, fresh_training=False,
                        reference_labels_modified=False)
    args.output.with_suffix(".verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(args.output), pages=pages, counts=inventory["counts"]), indent=2))


if __name__ == "__main__":
    main()
