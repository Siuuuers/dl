#!/usr/bin/env python3
"""Recover this exact saved PDF review, without releasing it for training."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import fitz

ATLAS_SHA = "e5a34c8f8670b2b1ca67c770fca5ad21782942fb212f04384e09ac154b4c4a6f"
CHANGES_SHA = "53d913f6e62bf8ca5eb573a088f4f9c2857dc0744e0ede1b5c949ffe50297ca3"
PARENTS = {
    "data/manifest.json": "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926",
    "data/candidates/sequence-refined-fronts-2026-11-18-r2/manifest.json": "de3cd6e5605ec7bd1250b53b4eb52c3a688e2431a19f4016bbfa08f2e09b6849",
    "data/references/sequence-refined-reference-2026-11-18/manifest.json": "0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96",
}
PALETTES = {
    "atlas": {"box": (22, 217, 236), "hold": (255, 182, 50), "left": (72, 242, 147), "right": (255, 105, 221)},
    "changes": {"box": (18, 212, 231), "hold": (255, 181, 48), "left": (62, 229, 138), "right": (245, 107, 221)},
}
TOKEN_TOLERANCE_PX = 0.001
PANEL_TOLERANCE_PX = 0.002


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def row_list(value):
    value = value.strip()
    if value == "none":
        return []
    require(re.fullmatch(r"[\d,\s-]+", value), "Unexpected PDF row-list syntax: " + value)
    result = []
    for part in value.split(","):
        ends = [int(x) for x in part.strip().split("-")]
        result.extend(range(ends[0], ends[-1] + 1))
    require(result == sorted(set(result)), "Nonunique/unsorted PDF row identity")
    return result


def color_is(value, expected):
    return value is not None and max(abs(a - b / 255) for a, b in zip(value, expected)) < 0.000002


def parse_tokens(tokens):
    require(len(tokens) == 11, "Expected 11-field pose row")
    v = list(map(float, tokens))
    require(all(math.isfinite(x) for x in v), "Nonfinite pose value")
    require(v[0] == 0 and v[7] in (0, 2) and v[10] in (0, 2), "Unexpected class/visibility")
    require(all(0 <= v[i] <= 1 for i in (1, 2, 3, 4, 5, 6, 8, 9)), "Normalized value outside [0,1]")
    require(v[3] > 0 and v[4] > 0, "Nonpositive box")
    for i in (5, 8):
        require(v[i + 2] != 0 or v[i:i + 3] == [0, 0, 0], "Noncanonical missing corner")
    require(not (v[7] == v[10] == 2) or v[5] <= v[8], "Reversed lower-corner order")
    cx, cy, w, h = v[1:5]
    return {"box": [(cx - w / 2) * 1280, (cy - h / 2) * 720, (cx + w / 2) * 1280, (cy + h / 2) * 720],
            "points": [[v[5] * 1280, v[6] * 720, int(v[7])], [v[8] * 1280, v[9] * 720, int(v[10])]]}


def geometry_error(a, b):
    require([p[2] for p in a["points"]] == [p[2] for p in b["points"]], "Visibility disagreement between PDF views")
    values = [abs(x - y) for x, y in zip(a["box"], b["box"])]
    for pa, pb in zip(a["points"], b["points"]):
        if pa[2] == 2:
            values.extend(abs(x - y) for x, y in zip(pa[:2], pb[:2]))
    return max(values)


def extract_rows(page, palette_name, image_info, crop, seq_start=0, seq_end=10**9):
    palette = PALETTES[palette_name]
    bx0, by0, bx1, by1 = image_info["bbox"]
    x0, y0, x1, y1 = crop
    sx, sy = (bx1 - bx0) / (x1 - x0), (by1 - by0) / (y1 - y0)
    require(image_info["width"] == x1 - x0 and image_info["height"] == y1 - y0, "Image/crop size differs")
    require(abs(sx - sy) < 0.000002 and sx > 0, "Unexpected PDF image transform")
    require(abs(image_info["transform"][1]) < 1e-8 and abs(image_info["transform"][2]) < 1e-8, "Rotated image transform")
    def xy(x, y):
        return [x0 + (x - bx0) / sx, y0 + (y - by0) / sy]
    draws = [d for d in page.get_drawings() if seq_start < d["seqno"] < seq_end]
    rows = []
    ignored_leaders = 0
    for draw in draws:
        stroke, fill = draw.get("color"), draw.get("fill")
        kinds = [item[0] for item in draw["items"]]
        box_kind = next((key for key in ("box", "hold") if color_is(stroke, palette[key])), None)
        point_kind = next((key for key in ("left", "right") if color_is(fill, palette[key])), None)
        if box_kind:
            if kinds == ["l"] and abs(draw["width"] - 0.35) < 1e-5:
                ignored_leaders += 1
                continue
            require(kinds == ["re"] and draw["type"] == "s", "Unexpected colored box primitive")
            require(bool(draw["dashes"] != "[] 0") == (box_kind == "hold"), "Box color/dash hold disagreement")
            rect = draw["items"][0][1]
            rows.append({"box": xy(rect.x0, rect.y0) + xy(rect.x1, rect.y1),
                         "points": [[0, 0, 0], [0, 0, 0]], "held": box_kind == "hold",
                         "box_seqno": draw["seqno"], "point_seqnos": [None, None], "line": None})
        elif point_kind:
            require(rows and kinds == ["c"] * 4 and draw["type"] == "fs", "Unexpected corner primitive")
            side = 0 if point_kind == "left" else 1
            require(rows[-1]["points"][side][2] == 0, "Multiple same-side corners after one box")
            r = draw["rect"]
            expected_diameter = 3.6 if palette_name == "atlas" else 3.2
            require(abs(r.width - expected_diameter) < 0.001 and abs(r.height - expected_diameter) < 0.001, "Unexpected corner-circle size")
            rows[-1]["points"][side] = xy((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2) + [2]
            rows[-1]["point_seqnos"][side] = draw["seqno"]
        elif color_is(stroke, palette["left"]):
            require(rows and kinds == ["l"] and rows[-1]["line"] is None, "Unexpected connecting line")
            rows[-1]["line"] = [xy(*draw["items"][0][1]), xy(*draw["items"][0][2])]
    for row in rows:
        both = all(point[2] == 2 for point in row["points"])
        require((row["line"] is not None) == both, "Line and corner availability disagree")
        if both:
            require(max(abs(a - b) for p, q in zip(row["line"], row["points"]) for a, b in zip(p, q[:2])) < PANEL_TOLERANCE_PX,
                    "Connecting line does not terminate at its row's circles")
    return rows, ignored_leaders


def recover(root, atlas_path, changes_path, output):
    require(not output.exists(), "Output already exists; choose a new recovery directory")
    require(sha(atlas_path) == ATLAS_SHA and sha(changes_path) == CHANGES_SHA, "Source PDF checksum mismatch")
    for path, digest in PARENTS.items():
        require(sha(root / path) == digest, "Pinned parent metadata mismatch: " + path)
    source = read_json(root / "data/manifest.json")
    records = {r["id"]: r for r in source["records"]}
    require(len(records) == 157, "Frozen image inventory differs")
    parent_rows, parent_bytes, parent_paths, parent_meta = {}, {}, {}, {}
    for manifest_path in list(PARENTS)[1:]:
        directory = (root / manifest_path).parent
        manifest = read_json(root / manifest_path)
        for item in manifest["files"]:
            iid = item["image_id"]
            label = directory / item.get("candidate_label", item.get("label_path"))
            require(sha(label) == item.get("candidate_label_sha256", item.get("label_sha256")), "Parent label checksum mismatch: " + iid)
            raw = label.read_bytes()
            lines = raw.decode().splitlines(keepends=True)
            require(all(line.strip() for line in lines), "Blank line in nonempty parent")
            parent_rows[iid] = [parse_tokens(line.split()) for line in lines]
            parent_bytes[iid], parent_paths[iid], parent_meta[iid] = raw, label.relative_to(root).as_posix(), item
    require(set(parent_rows) == set(records), "Parent image coverage differs")
    changes = fitz.open(changes_path)
    require(len(changes) == 74, "Change-PDF page count differs")
    descriptions, deletion_map, expected_replace, panel_specs = {}, {}, {}, []
    for page_no in range(5, 75):
        page = changes[page_no - 1]
        text = page.get_text()
        iid = re.search(r"new_\d{6}", text).group()
        match = re.search(r"Parent rows:\s*(.*?)\.\s*New rows:\s*(.*?)\.", text, re.S)
        require(match is not None, "Missing explicit row list")
        before, after = row_list(match[1]), row_list(match[2])
        crop = list(map(int, re.search(r"Crop \[(\d+), (\d+), (\d+), (\d+)\]", text).groups()))
        panels = sorted(page.get_image_info(), key=lambda item: item["bbox"][1])
        require(len(panels) == 2, "Expected exactly two comparison-panel images")
        image_seq = [i for i, item in enumerate(page.get_bboxlog()) if item[0] == "fill-image"]
        require(len(image_seq) == 2, "Expected exactly two image paint operations")
        has_before = "Before | parent current row numbers" in text
        panel_specs.append((page_no, iid, crop, before if has_before else [], after, panels, image_seq))
        if page_no >= 71:
            continue
        require(iid not in descriptions, "Duplicate primary change-page identity")
        descriptions[iid] = {"pdf_page": page_no, "text": text[match.start():].strip()}
        mapping_pairs = re.findall(r"(\d+(?:-\d+)?)\s*->\s*(deleted|\d+(?:-\d+)?)", text)
        mapping = {}
        if mapping_pairs:
            for left, right in mapping_pairs:
                old = row_list(left)
                new = [None] * len(old) if right == "deleted" else row_list(right)
                require(len(old) == len(new), "Invalid row renumbering range")
                for a, b in zip(old, new):
                    require(a not in mapping, "Duplicate parent row in mapping")
                    mapping[a] = b
            require(set(mapping) == set(range(1, len(parent_rows[iid]) + 1)), "Renumbering does not cover parent")
            deletion_map[iid] = [row for row, new in mapping.items() if new is None]
            require([x for x in mapping.values() if x is not None] == list(range(1, len(parent_rows[iid]) - len(deletion_map[iid]) + 1)), "Nonmonotonic output mapping")
        context = re.search(r"Retained context only: parent (.*?); new (.*?)\.", text, re.S)
        context_before = row_list(context[1]) if context else []
        if "/ CHANGE" in text:
            expected_replace[iid] = set(before) - set(context_before) - set(deletion_map.get(iid, []))
        else:
            require("/ HOLD" in text or "/ DECISION" in text, "Unexpected primary page status")
            expected_replace[iid] = set()
    require(sum(map(len, deletion_map.values())) == 12, "Expected twelve explicit deletions")
    require(sum(map(len, expected_replace.values())) == 888, "Expected 888 explicit replacements")
    atlas = fitz.open(atlas_path)
    require(len(atlas) == 165, "Atlas page count differs")
    recovered, receipts, operations, preserved_errors = {}, {}, [], []
    inherited_containment = []
    derived_errors = []
    counts = {"train": Counter(), "val": Counter()}
    actual_replace = {}
    for page_no in range(9, 166):
        page = atlas[page_no - 1]
        text = page.get_text()
        lines = text.splitlines()
        iid = lines[0]
        require(iid in records and iid not in recovered, "Missing/duplicate atlas image identity")
        record = records[iid]
        require(record["original_size"] == [1280, 720], "Original image geometry changed")
        summary = re.search(r"(\d+) objects \| (\d+) locatable corners \| (\d+) missing corners", text)
        require(summary is not None, "Missing atlas numeric header")
        n, visible, missing = map(int, summary.groups())
        row_numbers = [int(x) for line in lines[5:next(i for i, line in enumerate(lines) if line.startswith("Review rows:"))] for x in line.split()]
        require(row_numbers == list(range(1, n + 1)), "Atlas printed row sequence differs")
        images = page.get_image_info()
        require(len(images) == 1, "Expected one full-resolution atlas photo")
        draws, leaders = extract_rows(page, "atlas", images[0], [0, 0, 1280, 720])
        require(len(draws) == n, "Atlas rectangle/header count differs")
        require(sum(p[2] == 2 for obj in draws for p in obj["points"]) == visible and 2 * n - visible == missing, "Atlas corner/header count differs")
        held = [i for i, row in enumerate(draws, 1) if row["held"]]
        footer = re.search(r"Review rows: ([^\n]+)", text)[1]
        require(held == ([] if footer == "none recorded" else row_list(footer)), "Atlas amber/footer row set differs")
        require(held == ([7, 14] if iid == "new_000007" else []), "Unexpected remaining amber row")
        old_lines = parent_bytes[iid].decode().splitlines(keepends=True)
        deleted = set(deletion_map.get(iid, []))
        mapping = [i for i in range(1, len(old_lines) + 1) if i not in deleted]
        require(len(mapping) == n, "Atlas count and explicit deletion mapping disagree")
        output_lines, row_map = [], []
        actual_replace[iid] = set()
        for new_row, (old_row, obj) in enumerate(zip(mapping, draws), 1):
            old = parent_rows[iid][old_row - 1]
            tokens = old_lines[old_row - 1].split()
            groups = []
            box_error = max(abs(x - y) for x, y in zip(old["box"], obj["box"]))
            if box_error <= TOKEN_TOLERANCE_PX:
                preserved_errors.append(box_error)
            else:
                b = obj["box"].copy()
                for i, edge in enumerate(b):
                    limit = 1280 if i % 2 == 0 else 720
                    if abs(edge) <= TOKEN_TOLERANCE_PX:
                        b[i] = 0.0
                    elif abs(edge - limit) <= TOKEN_TOLERANCE_PX:
                        b[i] = float(limit)
                values = [(b[0] + b[2]) / 2560, (b[1] + b[3]) / 1440, (b[2] - b[0]) / 1280, (b[3] - b[1]) / 720]
                tokens[1:5] = [f"{value:.9f}" for value in values]
                groups.append("box")
            for side, start in enumerate((5, 8)):
                a, b = old["points"][side], obj["points"][side]
                error = max(abs(x - y) for x, y in zip(a[:2], b[:2])) if a[2] == b[2] else math.inf
                if error <= TOKEN_TOLERANCE_PX:
                    preserved_errors.append(error)
                else:
                    tokens[start:start + 3] = ["0", "0", "0"] if b[2] == 0 else [f"{b[0] / 1280:.9f}", f"{b[1] / 720:.9f}", "2"]
                    groups.append("left_corner" if side == 0 else "right_corner")
            line = " ".join(tokens) + "\n" if groups else old_lines[old_row - 1]
            geometry = parse_tokens(line.split())
            error = geometry_error(geometry, obj)
            require(error <= TOKEN_TOLERANCE_PX, "Normalized recovery exceeds PDF round-trip bound")
            derived_errors.append(error)
            require(all(-TOKEN_TOLERANCE_PX <= x <= (1280 if i % 2 == 0 else 720) + TOKEN_TOLERANCE_PX for i, x in enumerate(geometry["box"])), "Box extends beyond original image")
            for side, point in enumerate(geometry["points"]):
                if point[2] != 2:
                    continue
                b = geometry["box"]
                outside = max(b[0] - point[0], b[1] - point[1], point[0] - b[2], point[1] - b[3], 0)
                if outside > TOKEN_TOLERANCE_PX:
                    inherited_containment.append({"image_id": iid, "current_row": new_row, "parent_row": old_row, "side": side, "outside_px": outside, "changed_groups": groups})
            if groups:
                actual_replace[iid].add(old_row)
                operations.append({"image_id": iid, "split": record["split"], "operation": "replace", "parent_row": old_row, "current_row": new_row,
                                   "changed_groups": groups, "parent_tokens": old_lines[old_row - 1].split(), "recovered_tokens": line.split(), "atlas_page": page_no,
                                   "pdf_vector_seqnos": {"box": obj["box_seqno"], "corners": obj["point_seqnos"]}, "cross_pdf_page": descriptions.get(iid, {}).get("pdf_page")})
            row_map.append({"parent_row": old_row, "current_row": new_row, "parent_identity": parent_meta[iid]["row_map"][old_row - 1], "changed_groups": groups})
            output_lines.append(line)
        for old_row in sorted(deleted):
            operations.append({"image_id": iid, "split": record["split"], "operation": "delete", "parent_row": old_row, "current_row": None,
                               "parent_tokens": old_lines[old_row - 1].split(), "cross_pdf_page": descriptions[iid]["pdf_page"]})
        require(actual_replace[iid] == expected_replace.get(iid, set()), f"Unexplained recovery modifications for {iid}: actual={sorted(actual_replace[iid])}, PDF={sorted(expected_replace.get(iid,set()))}")
        recovered[iid] = {"geometry": draws, "bytes": "".join(output_lines).encode(), "row_map": row_map, "held": held}
        counts[record["split"]].update(images=1, objects=n, locatable_corners=visible, missing_corners=missing, empty_images=int(n == 0), changed_images=int(bool(actual_replace[iid] or deleted)))
        receipts[iid] = {"image_id": iid, "split": record["split"], "source_image_path": "data/" + record["image"], "source_image_sha256": record["image_sha256"],
                         "source_image_hash_status": "Declared by the pinned original manifest; image bytes are not read by this vector recovery.",
                         "parent_label_path": parent_paths[iid], "parent_label_sha256": hashlib.sha256(parent_bytes[iid]).hexdigest(),
                         "label_path": f"labels/{record['split']}/{iid}.txt", "recovered_label_sha256": hashlib.sha256(recovered[iid]["bytes"]).hexdigest(),
                         "atlas_page": page_no, "objects": n, "locatable_corners": visible, "missing_corners": missing, "held_current_rows": held,
                         "ignored_label_leader_lines": leaders, "row_map": row_map, "deleted_parent_rows": sorted(deleted),
                         "pdf_footer_excerpt": text[text.index("Review rows:"):].strip()}
    require(set(recovered) == set(records), "Incomplete atlas recovery")
    require(dict(counts["train"]) == {"images": 137, "objects": 3789, "locatable_corners": 6856, "missing_corners": 722, "empty_images": 13, "changed_images": 59}, "Unexpected recovered train totals")
    require(dict(counts["val"]) == {"images": 20, "objects": 662, "locatable_corners": 1226, "missing_corners": 98, "empty_images": 3, "changed_images": 5}, "Unexpected recovered development totals")
    panel_checks, panel_errors = [], []
    for page_no, iid, crop, before, after, images, image_seq in panel_specs:
        page = changes[page_no - 1]
        for side, ids in enumerate((before, after)):
            panel_rows, leaders = extract_rows(page, "changes", images[side], crop, image_seq[side], image_seq[side + 1] if side == 0 else 10**9)
            require(len(panel_rows) == len(ids), f"Change-PDF row count differs: page {page_no} panel {side}")
            max_error = 0
            for row_number, geometry in zip(ids, panel_rows):
                target = parent_rows[iid][row_number - 1] if side == 0 else recovered[iid]["geometry"][row_number - 1]
                error = geometry_error(geometry, target)
                require(error <= PANEL_TOLERANCE_PX, f"Cross-PDF geometry disagreement at {iid} row {row_number}, page {page_no} panel {side}: {error}")
                panel_errors.append(error)
                max_error = max(max_error, error)
            panel_checks.append({"pdf_page": page_no, "image_id": iid, "panel": "parent" if side == 0 else "current", "rows": ids, "max_error_px": max_error, "ignored_label_leader_lines": leaders})
    require(not inherited_containment, "Containment discrepancies require human inspection: " + json.dumps(inherited_containment))
    runtime = {"python": sys.version.split()[0], "pymupdf": fitz.VersionBind}
    verification = {"schema_version": 1, "status": "verified_vector_recovery_only", "release_ready": False, "training_recipe_id": None, "runtime": runtime,
                    "original_authoring_bytes_recovered": False, "physical_correctness_certified": False,
                    "counts": {k: dict(v) for k, v in counts.items()}, "operation_counts": dict(Counter(x["operation"] for x in operations)),
                    "atlas_pages": len(atlas), "full_photo_pages": 157, "change_pdf_pages": len(changes),
                    "compared_panel_rows": sum(len(x["rows"]) for x in panel_checks), "panel_checks": panel_checks,
                    "token_preservation_tolerance_px": TOKEN_TOLERANCE_PX, "cross_pdf_tolerance_px": PANEL_TOLERANCE_PX,
                    "max_preserved_parent_group_error_px": max(preserved_errors), "max_final_label_to_atlas_error_px": max(derived_errors),
                    "max_cross_pdf_geometry_error_px": max(panel_errors), "containment_discrepancies": inherited_containment,
                    "new_000101_parent_row52": "No r3 operation is present in the PDF. Preserved from parent; user-request completeness is a separate open review item."}
    output.mkdir(parents=True)
    for iid, value in recovered.items():
        destination = output / receipts[iid]["label_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value["bytes"])
    common = {"schema_version": 1, "kind": "pdf_recovered_pose_review", "recovery_id": "sequence-refined-fronts-2026-11-18-r3-pdf-recovered",
              "status": "review_only_not_released_for_training", "release_ready": False, "training_recipe_id": None, "full_dataset_certified": False,
              "pdf_review_before_merge": True, "original_authoring_bytes_recovered": False, "recovery_script_sha256": sha(Path(__file__)), "runtime": runtime,
              "parent_source_commit": "9304d4feb8aac481638ff08829dfb6e86e437c95", "pinned_parent_manifests": PARENTS,
              "source_pdfs": [{"filename": atlas_path.name, "sha256": ATLAS_SHA}, {"filename": changes_path.name, "sha256": CHANGES_SHA}],
              "format": {"task": "pose", "class_names": ["cassette_front"], "kpt_shape": [2, 3], "flip_idx": [1, 0], "row_fields": 11},
              "provenance": "Coordinates recovered from saved vector rectangles and colored circle centers. Unchanged parent field groups keep exact tokens; changed groups are newly serialized. This is not a byte-identical restoration of the missing authoring files.",
              "limits": ["PDF geometric consistency is not independent physical annotation review.", "Review notes may be abridged; the missing original per-field ledger is not reconstructed as historical fact.", "Original photo hashes are retained from the pinned source manifest; this recovery does not read original image pixels.", "The PR67 training data, predefined evaluation references and selected model are unchanged.", "Row new_000101:52 remains an explicitly open user-request completeness check."],
              "counts": verification["counts"], "operation_counts": verification["operation_counts"], "files": list(receipts.values())}
    write_json(output / "manifest.json", common)
    write_json(output / "operations.json", operations)
    write_json(output / "pdf-review-notes.json", descriptions)
    write_json(output / "verification.json", verification)
    return {k: verification[k] for k in ("status", "counts", "operation_counts", "compared_panel_rows", "max_preserved_parent_group_error_px", "max_final_label_to_atlas_error_px", "max_cross_pdf_geometry_error_px")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--changes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.atlas, args.changes, args.output), indent=2))


if __name__ == "__main__":
    main()
