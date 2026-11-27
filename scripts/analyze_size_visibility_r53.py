#!/usr/bin/env python3
"""Describe R50 target geometry versus saved PR146 outcomes; never infer anew.

The bins are descriptive original-pixel heights, not COCO area classes. The
same target can be small, occluded, clipped and low contrast; this analysis
does not identify a cause and uses no image-appearance surrogate as truth.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import run_partial_classification_r51 as r51
from audit_partial_classification_r51_results import ARCHIVE_SHA, ARCHIVE_BYTES
from pose_scoring import score_image, aggregate_scores

HEIGHT_BINS = ("<8", "8–<16", "16–<32", ">=32")


def height_bin(value):
    return HEIGHT_BINS[0 if value < 8 else 1 if value < 16 else 2 if value < 32 else 3]


def extent_summary(rows):
    return {
        "count": len(rows),
        **{key: ({"min": min(values), "median": statistics.median(values), "max": max(values)}
                 if (values := [row[key] for row in rows]) else None)
           for key in ("width_px", "height_px", "area_px2")},
    }


def tally(rows):
    result = {"targets": len(rows)}
    for role in ("reference", "candidate"):
        missed = sum(not row[role]["matched"] for row in rows)
        result[role] = {"matched": len(rows) - missed, "missed": missed,
                        "miss_rate": missed / len(rows) if rows else None}
    return result


def analyze(root, archive):
    root, archive = Path(root).resolve(), Path(archive)
    payload = archive.read_bytes()
    r51.need(len(payload) == ARCHIVE_BYTES and hashlib.sha256(payload).hexdigest() == ARCHIVE_SHA,
             "PR146 archive differs from the independently audited return")
    with zipfile.ZipFile(archive) as bundle:
        staging = json.loads(bundle.read("staging.json"))
        predictions = json.loads(bundle.read("evaluation/predictions.json"))
        reports = json.loads(bundle.read("evaluation/summary.json"))
    refs = r51.read_refs(root, staging["roles"]["reference"])
    manifest_sha, targets = refs["r50"]
    rows, image_counts = [], {}
    replay = {role: {} for role in ("reference", "candidate")}
    for identity, gold in sorted(targets.items()):
        maps = {}
        for role in replay:
            outputs = predictions[role][identity]["postclip"]
            score = score_image(gold, outputs, 1280, 720, .5, .5)
            r51.need(score == reports["references"]["r50"]["postclip"][role]["per_image"][identity],
                     "Saved per-image scores do not replay: " + role + "/" + identity)
            replay[role][identity] = score
            maps[role] = {match["gold_index"]: match["prediction_index"] for match in score["matches"]}
        image_counts[identity] = len(gold)
        for index, target in enumerate(gold):
            xmin, ymin, xmax, ymax = target["bbox_xyxy"]
            width, height = xmax - xmin, ymax - ymin
            r51.need(width > 0 and height > 0, "Nonpositive target extent")
            row = dict(image=identity, r50_row=index + 1,
                       stable_id=f"{identity} / R50 row{index + 1:03d}",
                       width_px=width, height_px=height, area_px2=width * height,
                       locatable_corners=sum(point[2] == 2 for point in target["keypoints"]),
                       height_bin_px=height_bin(height),
                       touches_image_boundary=bool(xmin <= 1e-5 or ymin <= 1e-5 or
                                                   xmax >= 1280 - 1e-5 or ymax >= 720 - 1e-5))
            for role in replay:
                position = maps[role].get(index)
                pred = predictions[role][identity]["postclip"][position] if position is not None else None
                row[role] = dict(matched=position is not None,
                                 prediction_number=position + 1 if position is not None else None,
                                 primary_prediction_id=pred.get("primary_prediction_id") if pred else None)
            rows.append(row)
    totals = {role: aggregate_scores(list(scores.values())) for role, scores in replay.items()}
    for role in totals:
        r51.need(totals[role] == reports["references"]["r50"]["postclip"][role]["detection"],
                 "Aggregate score replay differs")
    r51.need(len(rows) == 654 and len(image_counts) == 20, "R50 membership differs")
    r51.need((totals["reference"]["fn"], totals["candidate"]["fn"]) == (15, 11), "Audited misses differ")
    crossed = {str(n): {bucket: tally([row for row in rows if row["locatable_corners"] == n and
                                                 row["height_bin_px"] == bucket])
                       for bucket in HEIGHT_BINS} for n in (0, 1, 2)}
    result = dict(schema_version=1, experiment="r53-descriptive-size-visibility",
                  evidence="Saved PR146 inference replay only; no new inference or training",
                  source_pr=146, archive_sha256=ARCHIVE_SHA, reference="r50",
                  reference_manifest_sha256=manifest_sha, prediction_stage="postclip",
                  original_dimensions=[1280, 720], matching_iou=.5, endpoint_confidence=.5,
                  row_numbering="One-based rows in the immutable R50 reference; not source rows or P-IDs",
                  bin_policy={"measure": "bbox height in original pixels", "bins": list(HEIGHT_BINS),
                              "purpose": "descriptive; not a cause test, optimization or COCO size class"},
                  roles={"reference": "PR146 reference; prediction values equal PR137 candidate",
                         "candidate": "PR146 partial-target classification factor2"},
                  population=tally(rows), per_image_target_counts=image_counts,
                  by_locatable_corners={str(n): tally([row for row in rows if row["locatable_corners"] == n])
                                        for n in (0, 1, 2)},
                  by_height={bucket: tally([row for row in rows if row["height_bin_px"] == bucket])
                             for bucket in HEIGHT_BINS},
                  by_visibility_and_height=crossed,
                  extents={"all": extent_summary(rows),
                           **{role + "_missed": extent_summary([row for row in rows if not row[role]["matched"]])
                              for role in replay}},
                  missed_rows={role: [row for row in rows if not row[role]["matched"]] for role in replay},
                  rows=rows,
                  limitations=["Repeatedly inspected development images; no independent generalization test.",
                               "Locatable corner count is a label state, not an occlusion percentage.",
                               "Height, clipping, contrast and visibility can be confounded.",
                               "Multiple targets in the same photograph are correlated.",
                               "Upscaling existing images adds samples, not new optical information.",
                               "The descriptive bins were chosen after prior model outcomes were known."])
    r51.need(sum(item["targets"] for item in result["by_height"].values()) == 654,
             "Height strata do not partition targets")
    r51.need(sum(item["targets"] for item in result["by_locatable_corners"].values()) == 654,
             "Visibility strata do not partition targets")
    return result


def report(value):
    lines = ["# R53: is target size enough to explain the misses?", "",
             "27 November 2026. Descriptive CPU analysis of verified PR146 saved predictions; no new model run.", "",
             "The R50 reference contains 654 fronts on the same 20 development photographs. "
             "The two arms have 15 and 11 misses after the existing postclip step. "
             "Every per-image assignment and aggregate score was recomputed and checked against the return.", "",
             "## Corner availability", "",
             "| Locatable lower-front corners | Targets | Reference misses | Candidate misses |",
             "| --- | ---: | ---: | ---: |"]
    for n, row in value["by_locatable_corners"].items():
        lines.append(f"| {n} | {row['targets']} | {row['reference']['missed']} | {row['candidate']['missed']} |")
    lines += ["", "Corner availability is not a measured occlusion fraction. "
              "It is nevertheless necessary to stratify it: a size-only explanation could hide "
              "the fact that all misses occur in fronts with unavailable corners.", "",
              "## Visible-front box height", "",
              "These fixed descriptive bands use original-image pixels, not input-tensor pixels or "
              "COCO area categories. They were selected after previous outcomes were known; "
              "do not treat the table as a preregistered causal test.", "",
              "| Height (px) | Targets | Reference misses | Candidate misses |",
              "| --- | ---: | ---: | ---: |"]
    for bucket, row in value["by_height"].items():
        lines.append(f"| {bucket} | {row['targets']} | {row['reference']['missed']} | {row['candidate']['missed']} |")
    lines += ["", "### Cross the bands with corner availability", "",
              "| Corners | Height (px) | Targets | Reference misses | Candidate misses |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for n, bands in value["by_visibility_and_height"].items():
        for bucket, row in bands.items():
            lines.append(f"| {n} | {bucket} | {row['targets']} | {row['reference']['missed']} | {row['candidate']['missed']} |")
    lines += ["", "## What this does and does not suggest", "",
              "A single increase from inference size 1280 to 1600 is a reasonable diagnostic "
              "on the frozen PR137 checkpoint. It tests whether a changed sampling scale helps "
              "these specific instances. It does not prove that size caused the misses, "
              "reveal a hidden corner, or add information that was absent in the capture. "
              "Compare exact recovered and newly missed rows, unmatched outputs, corner error "
              "in original pixels, latency and memory. Keep the CLAHE contrast test separate.", "",
              "The current data already make corner availability an essential reporting stratum. "
              "Do not select a resolution solely from the pooled recall or call 654 related "
              "targets 654 independent trials. New capture sessions remain necessary for "
              "external validation.", "",
              "## Reproduce", "", "```bash",
              "python scripts/analyze_size_visibility_r53.py \\",
              "  --archive /path/to/baomai-partial-classification-r51-results.zip \\",
              "  --output /tmp/r53-size-visibility.json \\",
              "  --markdown /tmp/r53-size-visibility.md", "```", "",
              "The script verifies the exact archived bytes, all four reference manifests and "
              "label hashes, then replays R50 assignments. It makes no network calls. "
              "The full JSON retains every target's geometry and both match identities.", "",
              "[Machine-readable analysis](../results/review/thesis-r53-20261127/size-visibility.json) · "
              "[Prior result audit](partial-classification-r51-results.md) · [Thesis](thesis.md)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.root, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(report(value))
    print(json.dumps({"population": value["population"], "by_corners": value["by_locatable_corners"],
                      "by_height": value["by_height"]}, indent=2))


if __name__ == "__main__":
    main()
