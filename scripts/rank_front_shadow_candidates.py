#!/usr/bin/env python3
"""Rank saved endpoints for bounded original-pixel review, without changing labels.

The cue is local median luminance contrast, not a correctness probability. Real
front corners often lie on a dark material edge, so high scores can be false
flags. Comparison-patch centers are sampling locations, never proposed labels.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image


PATCH_RADIUS = 1
INWARD_PIXELS = 4.0
UPWARD_PIXELS = 4.0
BLOCKED_PHOTOS = {f"new_{number:06d}" for number in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def relative(path, root):
    return str(path.resolve().relative_to(root.resolve()))


def patch(image, xy):
    """Nearest integer center, floor(v + .5); require the complete 3x3 patch."""
    cx, cy = (math.floor(value + .5) for value in xy)
    bounds = [cx - PATCH_RADIUS, cy - PATCH_RADIUS, cx + PATCH_RADIUS + 1, cy + PATCH_RADIUS + 1]
    if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > image.width or bounds[3] > image.height:
        return {"bounds_xyxy": bounds, "median_luminance": None, "complete": False}
    values = [image.getpixel((x, y)) for y in range(bounds[1], bounds[3])
              for x in range(bounds[0], bounds[2])]
    return {"bounds_xyxy": bounds, "median_luminance": median(values), "complete": True}


def rank_key(item):
    return (-item["priority_score"], item["image_id"], item["row"], item["side"])


def scan(root, manifest_path, protected_path):
    manifest = read(manifest_path)
    status_path = root / manifest["review_status"]["path"]
    require(digest(status_path) == manifest["review_status"]["sha256"], "Review-status hash differs")
    status = read(status_path)
    excluded = set(manifest["excluded_training_image_ids"])
    held = set(status.get("pending_train_photos", [])) | set(status.get("unresolved_development_photos", []))
    held.update(case["image_id"] for case in status.get("cases", []) if case.get("status") == "held")
    protected = {(item["image_id"], item["row"]): item for item in read(protected_path)["preserved_standards"]}
    records, sources = [], []
    totals = Counter(images=0, locatable_endpoints=0, scored_endpoints=0, skipped_border_endpoints=0)
    files = sorted(manifest["files"], key=lambda item: item["image_id"])
    require(len(files) == 157, "This bounded screen expects all 157 original manifest images")
    for spec in files:
        image_id, split = spec["image_id"], spec["split"]
        image_path = root / "data/images" / split / (image_id + ".png")
        label_path = manifest_path.parent / spec["label_path"]
        image_hash, label_hash = digest(image_path), digest(label_path)
        require(image_hash == spec["source_image_sha256"], "Image hash differs: " + image_id)
        require(label_hash == spec["label_sha256"], "Label hash differs: " + image_id)
        rgb = Image.open(image_path).convert("RGB")
        gray = rgb.convert("L")
        width, height = rgb.size
        lines = label_path.read_text().splitlines()
        require(len(lines) == spec["objects"] and all(line.strip() for line in lines), "Row identity differs")
        totals["images"] += 1
        totals["images_" + split] += 1
        for row, line in enumerate(lines, 1):
            values = list(map(float, line.split()))
            require(len(values) == 11, "Expected 11-field pose row")
            require(all(math.isfinite(value) for value in values), "Nonfinite source coordinate")
            points = [[values[5] * width, values[6] * height, int(values[7])],
                      [values[8] * width, values[9] * height, int(values[10])]]
            is_protected = (image_id, row) in protected
            if is_protected:
                require(line == protected[(image_id, row)]["line"], "Protected source row changed")
            both = all(point[2] == 2 for point in points)
            dx, dy = (points[1][0] - points[0][0], points[1][1] - points[0][1]) if both else (1.0, 0.0)
            length = math.hypot(dx, dy)
            if length < 1e-9:
                dx, dy, length = 1.0, 0.0, 1.0
                method = "horizontal_fallback_coincident_points"
            else:
                method = "two_locatable_corners" if both else "horizontal_fallback_one_locatable_corner"
            tangent = [dx / length, dy / length]
            upward = [tangent[1], -tangent[0]]
            if upward[1] > 0:
                upward = [-value for value in upward]
            for side_index, (x, y, visibility) in enumerate(points):
                if visibility != 2:
                    continue
                totals["locatable_endpoints"] += 1
                inward = [value * (1 if side_index == 0 else -1) for value in tangent]
                sample = [x + INWARD_PIXELS * inward[0] + UPWARD_PIXELS * upward[0],
                          y + INWARD_PIXELS * inward[1] + UPWARD_PIXELS * upward[1]]
                point_patch, inward_patch = patch(gray, (x, y)), patch(gray, sample)
                complete = point_patch["complete"] and inward_patch["complete"]
                contrast = inward_patch["median_luminance"] - point_patch["median_luminance"] if complete else None
                tags = {"protected_standard": is_protected, "held_photo": image_id in held,
                        "excluded_training_photo": image_id in excluded,
                        "already_in_current_bounded_review": image_id in BLOCKED_PHOTOS,
                        "new_series": image_id.startswith("new_")}
                record = {"identity": f"{image_id}:{row}:{('left', 'right')[side_index]}",
                          "image_id": image_id, "row": row, "side": ("left", "right")[side_index],
                          "split": split, "saved_point_xy_px": [x, y], "tangent_method": method,
                          "comparison_sample_xy_px": sample, "saved_point_patch": point_patch,
                          "inward_up_patch": inward_patch, "signed_contrast_luminance": contrast,
                          "priority_score": max(0, contrast) if contrast is not None else None,
                          "tags": tags, "source_image_sha256": image_hash, "source_label_sha256": label_hash}
                records.append(record)
                totals["scored_endpoints" if complete else "skipped_border_endpoints"] += 1
                for tag in ("protected_standard", "held_photo", "excluded_training_photo"):
                    totals["locatable_endpoints_" + tag] += int(tags[tag])
        sources.append({"image_id": image_id, "split": split, "width": width, "height": height,
                        "image_path": relative(image_path, root), "image_sha256": image_hash,
                        "label_path": relative(label_path, root), "label_sha256": label_hash,
                        "objects": len(lines)})
    scored = sorted((item for item in records if item["priority_score"] is not None), key=rank_key)
    for rank, item in enumerate(scored, 1):
        item["screen_rank"] = rank
    selected, seen = [], set()
    for item in scored:
        tags = item["tags"]
        if (not tags["new_series"] or tags["already_in_current_bounded_review"]
                or tags["protected_standard"] or tags["held_photo"] or tags["excluded_training_photo"]
                or item["image_id"] in seen):
            continue
        selected.append(item)
        seen.add(item["image_id"])
        if len(selected) == 8:
            break
    require(len(selected) == 8 and len(seen) == 8, "Expected eight diverse eligible cases")
    require(totals["locatable_endpoints"] == totals["scored_endpoints"] + totals["skipped_border_endpoints"],
            "Screen accounting differs")
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1, "kind": "local_luminance_review_priority_screen",
        "reference": {"review_id": manifest["review_id"], "path": relative(manifest_path, root),
                      "sha256": digest(manifest_path)},
        "review_status": {"path": relative(status_path, root), "sha256": digest(status_path)},
        "protected_standards": {"path": relative(protected_path, root), "sha256": digest(protected_path),
                                "identities": [f"{iid}:{row}" for iid, row in sorted(protected)]},
        "screen_script": {"path": relative(Path(__file__), root), "sha256": digest(Path(__file__))},
        "method": {
            "purpose": "Review priority only; no label edits, coordinate proposals, model predictions, or white-pixel threshold.",
            "grayscale": "Pillow RGB.convert('L'), 8-bit luminance",
            "patch_size_px": [3, 3], "rounding": "floor(coordinate +0.5)",
            "sampling_offset": {"inward_along_front_px": INWARD_PIXELS, "upward_normal_px": UPWARD_PIXELS},
            "tangent": "Left-to-right line between two locatable saved corners; horizontal fallback when only one is locatable.",
            "score": "max(0, median(inward/up patch) - median(saved-point patch)), in grayscale levels",
            "border_policy": "Require both full 3x3 patches. Otherwise record the endpoint as unscored; do not pad or infer pixels.",
            "sort": "Score descending, then image_id, current one-based row, side alphabetical",
            "limitation": "A valid material corner is often darker than the face. Brightness, curvature, shadows, blur, and occlusion can make false flags. The score is not a probability or an annotation-accuracy measure.",
        },
        "totals": dict(totals), "top_candidates": scored[:40],
        "additional_review_selection": {
            "rule": "Top 8 scored eligible endpoints on distinct new-series photos; skip protected rows, held/excluded photos and all currently reviewed photos. No manual replacement of selected cases.",
            "blocked_photo_ids": sorted(BLOCKED_PHOTOS), "selected": selected,
        },
        "source_images": sources, "complete_score_table_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_preservation": {"original_images_hash_verified": 157,
                                "review_label_files_hash_verified": 157,
                                "source_files_written": 0},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference", type=Path,
                        default=Path("data/reviews/fronts-2026-11-23-r26/manifest.json"))
    parser.add_argument("--protected-ledger", type=Path,
                        default=Path("results/review/front-review-20261123-r26/row-review.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/review/front-review-20261123-r27/color-screen.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    result = scan(root, root / args.reference, root / args.protected_ledger)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "totals": result["totals"],
                      "additional_review_cases": [{"identity": item["identity"], "score": item["priority_score"],
                                                   "screen_rank": item["screen_rank"]}
                                                  for item in result["additional_review_selection"]["selected"]]}, indent=2))


if __name__ == "__main__":
    main()
