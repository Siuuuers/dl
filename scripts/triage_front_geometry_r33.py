#!/usr/bin/env python3
"""Read-only numerical triage for every R32 new-series annotation.

Flags request original-pixel inspection. They never rewrite coordinates or infer
occluded points. Color and physical surface ownership are not inferred from TXT.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from front_review_r27 import pixel_geometry
from pose_scoring import _box_iou
from verify_reconciled_pose_labels import bound, encoded, sha

SOURCE = {"path": "data/reviews/fronts-2026-11-25-r32/manifest.json",
          "sha256": "bc2e14ffeb1608be55ae39c5082ac8bed5d769b7bfc098a2592684fdb2ed654f"}


def audit(root):
    root = Path(root).resolve()
    source = bound(root, SOURCE)
    manifest = json.loads(source.read_bytes())
    rows, photos, count = [], [], Counter()
    for item in sorted(manifest["files"], key=lambda r: r["image_id"]):
        iid = item["image_id"]
        if not iid.startswith("new_"):
            continue
        label = bound(source.parent, {"path": item["label_path"], "sha256": item["label_sha256"]})
        image = root/"data/images"/item["split"]/(iid+".png")
        if sha(image) != item["source_image_sha256"]:
            raise ValueError("Original image differs: " + iid)
        geometry = [pixel_geometry(line, iid) for line in label.read_text().splitlines()]
        photos.append({"image_id": iid, "split": item["split"], "objects": len(geometry),
                       "source_image_sha256": item["source_image_sha256"],
                       "source_label_sha256": item["label_sha256"]})
        for n, g in enumerate(geometry, 1):
            box, points = g["box_xyxy"], g["keypoints"]
            x0, y0, x1, y1 = box
            flags, hard = [], []
            if not all(math.isfinite(x) for x in box + [v for p in points for v in p]):
                hard.append("nonfinite")
            if x1 <= x0 or y1 <= y0:
                hard.append("nonpositive_box")
            if x0 < -1e-4 or y0 < -1e-4 or x1 > 1280.0001 or y1 > 720.0001:
                hard.append("out_of_image_box")
            visible = [p for p in points if p[2] == 2]
            for p in points:
                if p[2] == 0 and (p[0] != 0 or p[1] != 0):
                    hard.append("nonzero_unavailable_point")
                if p[2] == 2 and not (x0-1 <= p[0] <= x1+1 and y0-1 <= p[1] <= y1+1):
                    hard.append("visible_point_outside_box_over1px")
            gap = y1-max(p[1] for p in visible) if visible else None
            if gap is not None and gap > 5:
                flags.append("lower_gap_over5px")
            if gap is not None and gap > max(6, .25*(y1-y0)):
                flags.append("large_lower_gap")
            if x1-x0 < 12 or y1-y0 < 8:
                flags.append("small_observed_fragment")
            if y1 > y0 and (x1-x0)/(y1-y0) < 1.5:
                flags.append("tall_observed_fragment")
            if any(min(p[0], p[1], 1280-p[0], 720-p[1]) <= 2 for p in visible):
                flags.append("landmark_near_image_clip")
            inside, overlap = [], []
            for j, other in enumerate(geometry, 1):
                if j == n:
                    continue
                if _box_iou(box, other["box_xyxy"]) > .7:
                    overlap.append(j)
                for side, p in zip(("left", "right"), other["keypoints"]):
                    if p[2] == 2 and x0+1 < p[0] < x1-1 and y0+1 < p[1] < y1-1:
                        inside.append({"row": j, "side": side})
            if inside:
                flags.append("neighbor_point_inside")
            if overlap:
                flags.append("high_box_overlap")
            count.update(flags)
            rows.append({"image_id": iid, "row": n, "split": item["split"],
                         "hard_errors": sorted(set(hard)), "flags": flags,
                         "lower_gap_px": gap, "visible_corners": len(visible),
                         "neighbor_points_inside": inside, "overlap_rows": overlap})
    return {"schema_version": 1, "kind": "read_only_geometry_triage", "source_review": SOURCE,
            "automatic_corrections": False,
            "interpretation": "Flags are review prompts, not errors. One-corner fronts can have large lower gaps under slant or occlusion; AABBs can contain neighbors or an internal gripper while retaining owned extrema.",
            "thresholds": {"lower_gap_over5px": 5, "large_lower_gap": "max(6px, 0.25*box_height)",
                           "high_box_overlap_iou": .7, "neighbor_point_inset_px": 1},
            "summary": {"photos": len(photos), "rows": len(rows),
                        "empty_photos": sum(p["objects"] == 0 for p in photos),
                        "rows_with_hard_errors": sum(bool(r["hard_errors"]) for r in rows),
                        "flagged_rows": sum(bool(r["flags"]) for r in rows),
                        "flag_counts": dict(sorted(count.items()))},
            "photos": photos, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded(result))
    print(json.dumps(result["summary"], indent=2))
