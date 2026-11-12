#!/usr/bin/env python3
"""Reconstruct the explicitly reviewed label files from pinned source text.

This bounded recovery utility is not a training-stager extension. It checks
file identity and exact edits; it does not repeat the original-image review.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BOXES = {
    "new_000023": {12: [463, 413, 630, 452], 13: [469, 449, 625, 484],
        14: [473, 482, 623, 520], 29: [348, 629, 494, 665],
        30: [352, 660, 467, 691], 31: [359, 686, 451, 714], 36: [640, 397, 814, 435]},
    "new_000031": {26: [754, 427, 916, 467], 27: [818, 461, 906, 498],
        28: [835, 493, 897, 527], 37: [428, 545, 481, 578],
        38: [432, 574, 475, 601], 39: [435, 598, 470, 627],
        40: [439, 623, 463, 650], 41: [440, 645, 451, 670]},
    "new_000045": {1: [606, 345, 776, 405], 15: [994, 631, 1110, 683]},
}
POINTS = {
    "new_000031": {22: (None, [742, 458]), 23: (None, [737, 491]), 24: (None, [732, 518])},
    "new_000023": {33: ([482, 550], None), 34: ([486, 580], None), 35: ([490, 607], None)},
}
RANGES = {
    "new_000031:22": [[739, 744], [455, 461]],
    "new_000031:23": [[734, 739], [488, 494]],
    "new_000031:24": [[729, 735], [515, 521]],
    "new_000023:33": [[480, 485], [548, 553]],
    "new_000023:34": [[484, 489], [577, 583]],
    "new_000023:35": [[488, 493], [605, 610]],
}
RECOVERED_PARENT_HASHES = {
    "new_000023": "13057ffcdc5339ac1b5d723c0b4f44c5672a9cd7ce47fe2b93d7380697d6a54b",
    "new_000031": "bad3f71a1d577b5cfa2c0333e5b4944ca85af4b08b1a25bbf0b697c0a2f82ded",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_or_check(path, payload, check):
    if check:
        if path.read_bytes() != payload:
            raise ValueError(f"File differs from exact reviewed reconstruction: {path.relative_to(ROOT)}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def geometry(line):
    t = list(map(float, line.split()))
    assert len(t) == 11 and all(math.isfinite(v) for v in t)
    assert t[0] == 0 and t[3] > 0 and t[4] > 0
    bounds = (t[1] - t[3] / 2, t[2] - t[4] / 2, t[1] + t[3] / 2, t[2] + t[4] / 2)
    ex, ey = 0.0001 / 1280, 0.0001 / 720
    assert -ex <= bounds[0] <= 1 + ex and -ex <= bounds[2] <= 1 + ex
    assert -ey <= bounds[1] <= 1 + ey and -ey <= bounds[3] <= 1 + ey
    for i in (5, 8):
        assert t[i + 2] in (0, 2)
        if t[i + 2] == 0:
            assert t[i:i+3] == [0, 0, 0]
        else:
            assert 0 <= t[i] < 1 and 0 <= t[i+1] < 1
            assert bounds[0] - ex <= t[i] <= bounds[2] + ex
            assert bounds[1] - ey <= t[i+1] <= bounds[3] + ey
    if t[7] == t[10] == 2:
        assert t[5] <= t[8]


def run(check):
    meta = json.loads((ROOT / "source/metadata.json").read_text())
    evidence = {"source_revision": meta["source_revision"], "source_label_hashes_verified": [],
        "recovered_parent_labels_byte_identical": [], "box_edits": [], "deletions": [],
        "endpoint_rows": [], "outputs": {}, "new_inference": False, "new_training": False,
        "actual_pngs_rechecked_after_reset": False,
        "historical_image_review": "Both original images and crops inspected, hashes verified before executor reset.",
        "status": "reconstructed_from_pinned_labels_and_preserved_review_decisions"}
    for image_id, boxes in BOXES.items():
        source = (ROOT / "source" / f"{image_id}.txt").read_bytes()
        record = next(r for r in meta["records"] if r["id"] == image_id)
        assert sha(source) == record["label_sha256"], f"Source SHA mismatch: {image_id}"
        evidence["source_label_hashes_verified"].append(image_id)
        original = source.decode().splitlines(keepends=True)
        lines = list(original)
        for row, (left, top, right, bottom) in boxes.items():
            tokens = lines[row - 1].split()
            tokens[1:5] = [f"{v:.8f}" for v in ((left+right)/2560, (top+bottom)/1440,
                                                   (right-left)/1280, (bottom-top)/720)]
            lines[row - 1] = " ".join(tokens) + "\n"
            assert tokens[0] == original[row-1].split()[0]
            assert tokens[5:] == original[row-1].split()[5:]
            evidence["box_edits"].append({"image_id": image_id, "original_row": row,
                "bbox_xyxy_original_pixels": [left, top, right, bottom]})
        if image_id == "new_000023":
            assert list(map(float, original[21].split()[5:])) == [0]*6
            lines.pop(21)
            evidence["deletions"].append({"image_id": image_id, "original_row": 22,
                "reason": "Confirmed duplicate of retained original row36; both removed slots missing."})
        parent = "".join(lines).encode()
        if image_id in RECOVERED_PARENT_HASHES:
            assert sha(parent) == RECOVERED_PARENT_HASHES[image_id], f"Recovered parent bytes differ: {image_id}"
            evidence["recovered_parent_labels_byte_identical"].append(image_id)
        name = f"box_review/labels/train/{image_id}.txt"
        save_or_check(ROOT / name, parent, check)
        evidence["outputs"][name] = sha(parent)
        if image_id not in POINTS:
            continue
        amended = list(lines)
        for original_row, pair in POINTS[image_id].items():
            parent_row = original_row - (1 if image_id == "new_000023" else 0)
            before = lines[parent_row - 1].split()
            after = list(before)
            for side, point in enumerate(pair):
                i = 5 + 3 * side
                if point is None:
                    if float(before[i+2]) == 2:
                        after[i:i+3] = ["0", "0", "0"]
                else:
                    after[i:i+3] = [f"{point[0]/1280:.8f}", f"{point[1]/720:.8f}", "2"]
            assert after[:5] == before[:5]
            amended[parent_row - 1] = " ".join(after) + "\n"
            evidence["endpoint_rows"].append({"image_id": image_id, "original_row": original_row,
                "parent_row": parent_row, "before_tokens": before[5:], "after_tokens": after[5:],
                "left_original_pixels": pair[0], "right_original_pixels": pair[1],
                "visible_point_descriptive_ranges_xy": RANGES[f"{image_id}:{original_row}"]})
        for row in amended:
            geometry(row)
        payload = "".join(amended).encode()
        name = f"endpoint_review/labels/train/{image_id}.txt"
        save_or_check(ROOT / name, payload, check)
        evidence["outputs"][name] = sha(payload)
    evidence["changed_endpoint_slots"] = sum(
        r["before_tokens"][i:i+3] != r["after_tokens"][i:i+3]
        for r in evidence["endpoint_rows"] for i in (0, 3))
    assert len(evidence["box_edits"]) == 17 and len(evidence["deletions"]) == 1
    assert evidence["changed_endpoint_slots"] == 9
    evidence["unchanged_missing_slots_in_selected_rows"] = 3
    evidence["total_visible_slot_delta"] = sum(
        (int(r["after_tokens"][i+2]) == 2) - (int(r["before_tokens"][i+2]) == 2)
        for r in evidence["endpoint_rows"] for i in (0, 3))
    assert evidence["total_visible_slot_delta"] == 0
    payload = (json.dumps(evidence, indent=2) + "\n").encode()
    save_or_check(ROOT / "correction-record.json", payload, check)
    print(json.dumps({"status": "verified" if check else "rebuilt", "box_edits": 17,
        "deleted_duplicates": 1, "changed_endpoint_slots": 9,
        "recovered_parent_labels_byte_identical": evidence["recovered_parent_labels_byte_identical"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    run(parser.parse_args().check)
