#!/usr/bin/env python3
"""Rank R27 TXT-label geometry for bounded visual review; never edit labels."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PARENT = "data/reviews/fronts-2026-11-23-r27/manifest.json"
PARENT_SHA = "9e3675c83cb6735242b3b99266e36d5a33248c13e51b8c91bd89a65d2c4b92af"
STANDARDS = {("new_000008", n) for n in (18, 19, 20, 21, 28, 29, 30, 31, 32)} | {
    ("new_000010", 16), ("new_000016", 24)}
EXPLICIT_REVISITS = {("new_000008", 19), ("new_000008", 29)}
NAMED = {"new_000008": [4, 5, 6, 11, 19, 23, 29], "new_000009": [55],
         "new_000010": [14, 26, 33], "new_000011": [49], "new_000016": [26, 27],
         "new_000055": [23], "new_000056": [12], "new_000058": [7, 18],
         "new_000059": [7, 10], "new_000087": [53], "new_000088": [52],
         "new_000089": [50], "new_000090": [44, 45]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def distribution(values):
    return {"n": len(values), "min": min(values) if values else None,
            "median": quantile(values, .5), "p90": quantile(values, .9),
            "p95": quantile(values, .95), "max": max(values) if values else None}


def parse(text):
    rows = []
    for row, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        require(len(tokens) == 11, "Expected one object and two pose triplets per row")
        values = list(map(float, tokens))
        require(all(math.isfinite(v) for v in values), "Nonfinite label coordinate")
        cx, cy, width, height = values[1:5]
        box = [(cx - width / 2) * 1280, (cy - height / 2) * 720,
               (cx + width / 2) * 1280, (cy + height / 2) * 720]
        points = []
        for side, offset in (("left", 5), ("right", 8)):
            if values[offset + 2] > 0:
                points.append({"side": side, "x": values[offset] * 1280,
                               "y": values[offset + 1] * 720})
        rows.append({"row": row, "box_xyxy_px": box, "points": points})
    return rows


def run(root):
    parent_path = root / PARENT
    require(sha(parent_path) == PARENT_SHA, "Immutable R27 manifest changed")
    parent = json.loads(parent_path.read_bytes())
    excluded = set(parent["excluded_training_image_ids"])
    require(len(excluded) == 31, "Old-series treatment exclusion inventory changed")
    records, sources = [], []
    counts = Counter()
    all_top_gaps, all_bottom_gaps = [], []
    for spec in sorted(parent["files"], key=lambda item: item["image_id"]):
        image_id, split = spec["image_id"], spec["split"]
        path = parent_path.parent / spec["label_path"]
        require(sha(path) == spec["label_sha256"], "Parent label drift: " + image_id)
        rows = parse(path.read_text())
        counts["label_files"] += 1
        counts["objects"] += len(rows)
        sources.append({"image_id": image_id, "split": split,
                        "label_path": str(path.relative_to(root)), "label_sha256": sha(path),
                        "source_image_sha256": spec["source_image_sha256"], "objects": len(rows)})
        for item in rows:
            identity = (image_id, item["row"])
            x0, y0, x1, y1 = item["box_xyxy_px"]
            height = y1 - y0
            ys = [p["y"] for p in item["points"]]
            counts["locatable_endpoints"] += len(ys)
            bottom_gap = y1 - max(ys) if ys else None
            top_gap = min(ys) - y0 if ys else None
            threshold = max(5.0, .25 * height)
            foreign = []
            for other in rows:
                if other["row"] == item["row"]:
                    continue
                for point in other["points"]:
                    # A two-pixel inset avoids shared-border/rounding contacts.
                    if x0 + 2 < point["x"] < x1 - 2 and y0 + 2 < point["y"] < y1 - 2:
                        relationship = "no_own_locatable_corner"
                        if ys:
                            relationship = "above" if point["y"] < min(ys) - 2 else (
                                "below" if point["y"] > max(ys) + 2 else "near_corner_level")
                        foreign.append({"row": other["row"], **point, "relationship": relationship})
            reasons = []
            if bottom_gap is not None:
                all_top_gaps.append(top_gap)
                all_bottom_gaps.append(bottom_gap)
                if bottom_gap > threshold:
                    reasons.append("large_lower_edge_gap")
            else:
                counts["objects_without_locatable_corner"] += 1
            if foreign:
                reasons.append("other_object_corner_strictly_inside_box")
            counts["large_lower_edge_gap"] += "large_lower_edge_gap" in reasons
            counts["boxes_containing_other_corner"] += bool(foreign)
            counts["flagged_objects"] += bool(reasons)
            records.append({"identity": f"{image_id}:{item['row']}", "image_id": image_id,
                            "row": item["row"], "split": split, "box_xyxy_px": item["box_xyxy_px"],
                            "locatable_points": item["points"], "box_height_px": height,
                            "ymin_to_nearest_lower_corner_px": top_gap,
                            "ymax_minus_lowest_locatable_corner_px": bottom_gap,
                            "bottom_gap_review_threshold_px": threshold,
                            "bottom_gap_threshold_ratio": bottom_gap / threshold if bottom_gap is not None else None,
                            "foreign_corners": foreign, "flags": reasons,
                            "tags": {"protected_in_r27": identity in STANDARDS,
                                     "user_authorized_standard_revisit": identity in EXPLICIT_REVISITS,
                                     "user_named": item["row"] in NAMED.get(image_id, []),
                                     "held_photo": image_id == "new_000015",
                                     "excluded_training_photo": image_id in excluded},
                            "source_label_sha256": spec["label_sha256"],
                            "source_image_sha256": spec["source_image_sha256"]})
    eligible = [r for r in records if r["split"] == "train" and r["image_id"] != "new_000016"
                and not r["tags"]["held_photo"] and not r["tags"]["excluded_training_photo"]
                and not r["tags"]["protected_in_r27"]]
    gap_order = sorted((r for r in records if "large_lower_edge_gap" in r["flags"]),
                       key=lambda r: (-r["bottom_gap_threshold_ratio"], r["image_id"], r["row"]))
    overlap_order = sorted((r for r in records if r["foreign_corners"]),
                           key=lambda r: (-len(r["foreign_corners"]), r["image_id"], r["row"]))
    eligible_ids = {r["identity"] for r in eligible}
    selected, used_photos = [], set()
    for cue, order, cap in (("large_lower_edge_gap", gap_order, 4),
                            ("other_object_corner_strictly_inside_box", overlap_order, 8)):
        for item in order:
            if len(selected) >= cap:
                break
            if item["identity"] not in eligible_ids or item["image_id"] in used_photos:
                continue
            selected.append({**item, "selected_by": cue})
            used_photos.add(item["image_id"])
    require(len(selected) == 8, "Expected eight eligible photo-diverse visual candidates")
    named = [r for r in records if r["tags"]["user_named"]]
    require(len(named) == 25, "Every textual/screenshot request must have one identity")
    return {"schema_version": 1, "kind": "txt_geometry_review_priority_screen",
            "parent_review": {"path": PARENT, "sha256": PARENT_SHA},
            "script_sha256": sha(Path(__file__)), "coordinate_system": "Original1280x720; x right, y down.",
            "method": {"inputs": "157 label TXT files only; source image identities are inherited from the hash-bound parent manifest. No image pixels or predictions are read by this screen.",
                       "lower_edge": "ymax - max(y of locatable lower-front keypoints); flag if greater than max(5px,0.25*box_height).",
                       "upper_edge": "min(y of locatable lower-front keypoints) - ymin is recorded as front height context, not treated as a near-zero constraint.",
                       "overlap": "Flag a box containing another object's locatable keypoint more than2px inside all four edges. Record every such foreign point and its vertical relation.",
                       "selection": "Four largest qualifying lower-gap ratios, then four highest foreign-corner counts, one candidate per eligible new-series training photo; exclude held015, old-series treatment exclusions, protected standards and already-requested016.",
                       "threshold_status": "Fixed inspection priorities chosen before visual review; not physical laws, automatic corrections, or tuned evaluation thresholds."},
            "counts": dict(counts),
            "distributions_px": {"ymin_to_nearest_lower_corner": distribution(all_top_gaps),
                                  "ymax_to_lowest_lower_corner": distribution(all_bottom_gaps)},
            "limits": ["Lower-front keypoints naturally lie near ymax, while their distance from ymin includes front height.",
                       "Perspective, rim thickness, occlusion and axis-aligned rectangles can create legitimate gaps and neighboring-point overlaps.",
                       "Unavailable corners supply no guessed location; zero-corner rows have no corner-to-bottom gap score.",
                       "Every amendment requires original-pixel adjudication. This screen never edits labels or copies predictions.",
                       "The eight selected cases are nonrandom review priorities, not a precision or dataset-error-rate sample."],
            "sources": sources, "named_rows": named, "top_lower_gap_candidates": gap_order[:40],
            "top_overlap_candidates": overlap_order[:40], "selected_training_review": selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root)
    output = args.output or root / "results/review/front-review-20261123-r28/geometry-screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": result["counts"],
                      "distributions_px": result["distributions_px"],
                      "selected_training_review": [{"identity": r["identity"], "cue": r["selected_by"],
                          "lower_gap": r["ymax_minus_lowest_locatable_corner_px"],
                          "threshold": r["bottom_gap_review_threshold_px"],
                          "foreign_corner_count": len(r["foreign_corners"])} for r in result["selected_training_review"]]}, indent=2))
