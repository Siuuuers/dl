#!/usr/bin/env python3
"""Replay the published PR35/PR52 outputs against a corner-only reference amendment.

This narrow diagnostic accepts coordinate edits in new_000010 only. It does not
choose coordinates, change labels, load weights, infer, train, or access a network.
Historical scorers and evidence contracts remain unchanged.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys

sys.dont_write_bytecode = True

ORIGINAL_SHA256 = "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926"
PARENT_SHA256 = "2e60b3e81195448cc565a1000b4305e3215baba2bf0ee0c976e6551c2e64937c"
PARENT_ID = "baomai-visible-front-2026-11-11"
PROTOCOL_SHA256 = "0e257cc036c86ea6859b6ed3b86f10768a87994e8f3b8b82b7e553a1cc1f8f60"
EVIDENCE_COMMIT = "23b8aaac264179a18b0738d3e0b2e44ce0a53e56"
EVIDENCE_PATH = "results/verification/baomai-endpoint-results-20261112/evidence.zip"
THRESHOLD = 0.13
RADII = (5, 10)
TARGET_ID = "new_000010"
ROLES = ("current", "old_reviewed", "endpoint_reviewed")
MODEL_SHA256 = {
    "current": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
    "old_reviewed": "c480ff0881e9bc8feca94b2478e6cbca76cdb8de5ed5bbf82c5de47041920b29",
    "endpoint_reviewed": "890ff20f4d7c377435ee29872de18739a6571fece2f8672a19802e24dbba418c",
}
PREDICTION_SHA256 = {
    "current": "45587e9bd1c194fe44730a5b5404408856aa80bfb4e582632acaefb19e95c700",
    "old_reviewed": "21b800f89cdd21d5da174074139b9bced6f4a2f2551aa618c9414f37d2df83f7",
    "endpoint_reviewed": "7a2fd7b8d66032f58dbfac1db5db7fcac7467d63a0895367589782b5a5d143e3",
}
SAVED_SCORING_SHA256 = {
    "current": "4b97981807f0d7baddc84e9cba8e8e4b79c33bc0686aa316ac45c899b2ec4344",
    "old_reviewed": "481b14e1fdfed3b60ee0f15b93e8331f4f96041127cbdb094a1b820395775fca",
    "endpoint_reviewed": "019604424046ba8a0623fdb9fc9b38cbd673d6c24ac67586e18d92fd6626565a",
}
HELPER_SHA256 = {
    "rescore_reference_amendment.py": "a4321d0354345ca89f7055ef8bacc611cd342b7d047bbc0e9df48aaa4815c2de",
    "evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "pose_checkpoint.py": "88ec2b4d19616470718707b90a6b3d982c7a5f3470bab3e161f14068db11591a",
    "pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "diagnose_pose_quality.py": "a3baa1285d1580ee44547f7c3d606f524f2f40616fed568885746d2014159953",
    "verify_pose_candidate.py": "03395b9a82a4c120647a1cac51c89fd214d8cf9b15c04e2196a40f39077d8491",
}
MATCH_KEYS = (
    "matches", "unmatched_gold_indices", "unmatched_prediction_indices",
    "duplicate_prediction_indices", "other_fp_prediction_indices",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned(path, expected):
    require(Path(path).is_file() and sha256(path) == expected, f"SHA256 mismatch: {path}")


def json_read(text, location):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {location}: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"non-finite JSON constant in {location}: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)


def read_json(path):
    return json_read(Path(path).read_text(encoding="utf-8"), str(path))


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def reconcile(actual, expected, location, audit):
    """Compare replayed values with independently saved outputs, preserving counts."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"keys differ: {location}")
        for key in expected:
            reconcile(actual[key], expected[key], f"{location}.{key}", audit)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"list differs: {location}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            reconcile(a, e, f"{location}[{index}]", audit)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual), f"non-finite value: {location}")
        delta = abs(actual - expected)
        require(delta <= 1e-10, f"numeric disagreement at {location}: {actual} versus {expected}")
        audit["largest_absolute_difference"] = max(audit["largest_absolute_difference"], delta)
        audit["scalar_checks"] += 1
    else:
        require(type(actual) is type(expected) and actual == expected, f"exact disagreement: {location}")
        audit["scalar_checks"] += 1


def reference_counts(gold):
    eligible = {str(n): 0 for n in range(3)}
    for objects in gold.values():
        for obj in objects:
            eligible[str(sum(p[2] == 2 for p in obj["keypoints"]))] += 1
    visible = eligible["1"] + 2 * eligible["2"]
    objects = sum(eligible.values())
    return ({"images": len(gold), "objects": objects, "visible_endpoints": visible,
             "missing_endpoints": 2 * objects - visible,
             "empty_images": sum(not x for x in gold.values())}, eligible)


def validate_delta(parent_dir, new_dir, parent_manifest, new_manifest, parent_gold, new_gold):
    require(new_manifest["reference_id"] != PARENT_ID, "amendment must have a new reference ID")
    require(new_manifest.get("policy_id") == parent_manifest["policy_id"], "policy ID changed")
    parent_link = new_manifest.get("parent_reference", {})
    require(parent_link.get("reference_id") == PARENT_ID and parent_link.get("sha256") == PARENT_SHA256,
            "new reference must name the pinned parent reference")
    require(parent_link.get("path") == "data/references/visible-front-2026-11-11/manifest.json",
            "parent reference path differs")
    old_entries = {item["image_id"]: item for item in parent_manifest["files"]}
    new_entries = {item["image_id"]: item for item in new_manifest["files"]}
    changes, changed_paths = [], []
    for identifier in sorted(old_entries):
        old_entry, new_entry = old_entries[identifier], new_entries[identifier]
        for key in ("source_label_path", "label_path", "source_label_sha256", "image_sha256",
                    "width", "height", "source_row_count", "row_count",
                    "retained_source_to_revised_row_one_based", "added_revised_rows_one_based"):
            require(new_entry.get(key) == old_entry.get(key), f"reference identity/mapping changed: {identifier}:{key}")
        before = (parent_dir / old_entry["label_path"]).read_bytes()
        after = (new_dir / new_entry["label_path"]).read_bytes()
        if identifier != TARGET_ID:
            require(before == after, f"only {TARGET_ID} may change: {identifier}")
            continue
        if before == after:
            continue
        changed_paths.append(old_entry["label_path"])
        old_lines, new_lines = before.decode().splitlines(keepends=True), after.decode().splitlines(keepends=True)
        require(len(old_lines) == len(new_lines) == 32, "new_000010 row count must remain 32")
        for row, (old_line, new_line) in enumerate(zip(old_lines, new_lines), 1):
            if old_line == new_line:
                continue
            old_tokens, new_tokens = re.findall(r"\S+", old_line), re.findall(r"\S+", new_line)
            require(len(old_tokens) == len(new_tokens) == 11, f"expected 11 tokens at row {row}")
            require(re.sub(r"\S+", "#", old_line) == re.sub(r"\S+", "#", new_line),
                    f"only coordinate tokens may change, including whitespace preservation: row {row}")
            changed_fields = [i for i in range(11) if old_tokens[i] != new_tokens[i]]
            require(changed_fields and set(changed_fields) <= {5, 6, 8, 9}, f"non-coordinate edit: row {row}")
            changed_sides = [side for side, start in enumerate((5, 8))
                             if set(changed_fields) & {start, start + 1}]
            for side in changed_sides:
                require(parent_gold[identifier][row - 1]["keypoints"][side][2] == 2,
                        f"missing endpoint cannot be edited: row {row}, side {side}")
                require(parent_gold[identifier][row - 1]["keypoints"][side][:2] !=
                        new_gold[identifier][row - 1]["keypoints"][side][:2],
                        f"format-only endpoint edit: row {row}, side {side}")
            changes.append({"image_id": identifier, "row_one_based": row,
                            "changed_sides": ["left" if x == 0 else "right" for x in changed_sides],
                            "changed_token_indices_zero_based": changed_fields,
                            "parent_row_text": old_line.rstrip("\r\n"),
                            "amended_row_text": new_line.rstrip("\r\n")})
    require(changes and len(changed_paths) == 1, "expected at least one supported coordinate edit in new_000010")
    counts, eligibility = reference_counts(parent_gold)
    require(reference_counts(new_gold) == (counts, eligibility), "coordinate amendment changed eligibility")
    for manifest in (parent_manifest, new_manifest):
        require(manifest.get("counts") == counts, "manifest counts differ from label bytes")
        require(manifest.get("objects_by_locatable_corners") == eligibility, "manifest eligibility differs")
    return changes, counts, eligibility


def load_predictions(prediction_dir, role, records, parent_entries):
    path = prediction_dir / f"predictions-{role}.jsonl"
    score_path = prediction_dir / f"scoring-{role}-primary.json"
    pinned(path, PREDICTION_SHA256[role])
    pinned(score_path, SAVED_SCORING_SHA256[role])
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 21 and all(line.strip() for line in lines), f"expected run header plus 20 images: {role}")
    header, *items = [json_read(line, f"{path}:{i}") for i, line in enumerate(lines, 1)]
    require(header["record_type"] == "run" and header["role"] == role, f"incorrect run role: {role}")
    require(header["model_sha256"] == MODEL_SHA256[role]
            and header["checkpoint_identity"]["checkpoint_sha256"] == MODEL_SHA256[role],
            f"incorrect checkpoint identity: {role}")
    require(header["thresholds"] == [THRESHOLD], f"incorrect box threshold: {role}")
    identities = header["identities"]
    require(identities["primary_reference_sha256"] == PARENT_SHA256
            and identities["original_reference_sha256"] == ORIGINAL_SHA256
            and identities["protocol_sha256"] == PROTOCOL_SHA256, f"incorrect reference/protocol identity: {role}")
    expected_inventory = [{"image_id": identifier, "image_sha256": record["image_sha256"],
                           "original_label_sha256": record["label_sha256"],
                           "primary_label_sha256": parent_entries[identifier]["label_sha256"],
                           "width": record["width"], "height": record["height"]}
                          for identifier, record in sorted(records.items())]
    require(identities["image_reference_inventory"] == expected_inventory, f"input inventory differs: {role}")
    for name in ("evaluate_pose.py", "pose_scoring.py", "pose_checkpoint.py", "diagnose_pose_quality.py", "verify_pose_candidate.py"):
        require(identities["source_helpers_sha256"][f"scripts/{name}"] == HELPER_SHA256[name],
                f"saved helper identity differs: {role}:{name}")
    inference = header["inference"]
    for key, expected in {"imgsz": 1280, "rect": True, "batch": 1, "nms": True,
                          "iou": 0.7, "max_det": 300}.items():
        require(inference.get(key) == expected, f"inference setting differs: {role}:{key}")
    by_id = {}
    for item in items:
        identifier = item["image_id"]
        require(identifier in records and identifier not in by_id, f"unknown/duplicate prediction image: {role}:{identifier}")
        record = records[identifier]
        require(item["record_type"] == "prediction" and item["role"] == role
                and item["threshold"] == THRESHOLD, f"prediction header differs: {role}:{identifier}")
        for key, source_key in (("image_sha256", "image_sha256"), ("original_label_sha256", "label_sha256"),
                                ("width", "width"), ("height", "height")):
            require(item[key] == record[source_key], f"prediction input differs: {role}:{identifier}:{key}")
        require(all(p["confidence"] >= THRESHOLD for p in item["predictions"]), f"below-threshold prediction: {role}:{identifier}")
        by_id[identifier] = item["predictions"]
    require(set(by_id) == set(records), f"prediction image coverage differs: {role}")
    saved = read_json(score_path)
    require(set(saved) == {str(THRESHOLD)}, f"saved scoring threshold differs: {role}")
    return header, by_id, saved[str(THRESHOLD)], path, score_path


def compact_score(score):
    return {key: value for key, value in score.items() if key != "endpoint_errors_px" and key not in MATCH_KEYS}


def affected_object(change, before, after, predictions, score, width, height, returned):
    index = change["row_one_based"] - 1
    match = next((m for m in score["matches"] if m["gold_index"] == index), None)
    pred = predictions[match["prediction_index"]] if match else None
    endpoints = []
    for side, (old_point, new_point) in enumerate(zip(before[index]["keypoints"], after[index]["keypoints"])):
        point = pred["keypoints"][side] if pred else None
        available = bool(point and returned(point, width, height))
        visible = old_point[2] == 2
        old_error = math.hypot(point[0] - old_point[0], point[1] - old_point[1]) if available and visible else None
        new_error = math.hypot(point[0] - new_point[0], point[1] - new_point[1]) if available and visible else None
        endpoints.append({"side": ("left", "right")[side], "reference_visible": visible,
                          "changed": old_point != new_point, "parent_reference_xy": old_point[:2] if visible else None,
                          "amended_reference_xy": new_point[:2] if visible else None,
                          "prediction_xy": point[:2] if point else None,
                          "prediction_confidence": point[2] if point else None, "returned": available,
                          "parent_error_px": old_error, "amended_error_px": new_error,
                          "parent_within_radius": {str(r): old_error is not None and old_error <= r for r in RADII},
                          "amended_within_radius": {str(r): new_error is not None and new_error <= r for r in RADII}})
    return {"image_id": TARGET_ID, "row_one_based": index + 1, "matched": match is not None,
            "prediction_index_zero_based": match["prediction_index"] if match else None,
            "matching_iou": match["iou"] if match else None,
            "box_confidence": pred["confidence"] if pred else None,
            "both_reference_corners_visible": all(e["reference_visible"] for e in endpoints),
            "endpoints": endpoints,
            "parent_both_within_radius": {str(r): all(e["reference_visible"] and e["parent_within_radius"][str(r)] for e in endpoints) for r in RADII},
            "amended_both_within_radius": {str(r): all(e["reference_visible"] and e["amended_within_radius"][str(r)] for e in endpoints) for r in RADII}}


def run(args):
    root, parent_dir, new_dir, prediction_dir = [p.resolve() for p in
        (args.repo_root, args.parent_reference, args.reference_dir, args.prediction_dir)]
    output = args.output.resolve()
    require(not output.exists(), f"output already exists: {output}")
    require(not any(output.is_relative_to(p) for p in (root / "data", parent_dir, new_dir, prediction_dir)),
            "output must be outside source data, reference and prediction directories")
    require(new_dir != parent_dir, "new reference must be a separate directory")
    protected = {root / "data/manifest.json": ORIGINAL_SHA256, parent_dir / "manifest.json": PARENT_SHA256,
                 Path(__file__).resolve(): sha256(Path(__file__))}
    protected.update({root / "scripts" / name: digest for name, digest in HELPER_SHA256.items()})
    for path, expected in protected.items():
        pinned(path, expected)
    sys.path.insert(0, str(root / "scripts"))
    from rescore_reference_amendment import load_api, verify_amendment
    from diagnose_pose_quality import _returned
    _, api = load_api(root)
    original = api["preflight_dataset"](root, require_weights=False)
    protected[root / "metadata/model_card.json"] = original["model_card_sha256"]
    for record in original["records"]:
        protected[Path(record["image_path"])] = record["image_sha256"]
        protected[Path(record["label_path"])] = record["label_sha256"]
    parent_manifest, parent_gold, _, _, parent_files = verify_amendment(root, parent_dir, original, api)
    new_manifest, new_gold, _, _, new_files = verify_amendment(root, new_dir, original, api)
    require(parent_manifest["reference_id"] == PARENT_ID, "parent reference ID differs")
    protected[new_dir / "manifest.json"] = sha256(new_dir / "manifest.json")
    for directory, files in ((parent_dir, parent_files), (new_dir, new_files)):
        protected.update({directory / f["label_path"]: f["label_sha256"] for f in files})
    changes, counts, eligibility = validate_delta(parent_dir, new_dir, parent_manifest, new_manifest, parent_gold, new_gold)
    records = {r["id"]: r for r in original["records"]}
    parent_entries = {r["image_id"]: r for r in parent_files}
    results = {}
    total_audit = {"scalar_checks": 0, "largest_absolute_difference": 0.0}
    for role in ROLES:
        header, predicted, saved, pred_path, score_path = load_predictions(prediction_dir, role, records, parent_entries)
        protected[pred_path] = PREDICTION_SHA256[role]
        protected[score_path] = SAVED_SCORING_SHA256[role]
        saved_images = {x["image_id"]: x for x in saved["per_image"]}
        require(len(saved["per_image"]) == 20 and set(saved_images) == set(records), f"saved scorer coverage differs: {role}")
        scores, qualities = {"parent": [], "amended": []}, {"parent": [], "amended": []}
        matching_trace, affected = [], []
        audit = {"scalar_checks": 0, "largest_absolute_difference": 0.0}
        changed_image = {}
        for identifier, record in sorted(records.items()):
            pair = {}
            for name, gold in (("parent", parent_gold), ("amended", new_gold)):
                score = api["score_image"](gold[identifier], predicted[identifier], record["width"], record["height"], 0.5, 0.5)
                quality = api["quality_for_image"](gold[identifier], predicted[identifier], record["width"], record["height"], RADII)
                scores[name].append(score)
                qualities[name].append(quality)
                pair[name] = score
                if name == "parent":
                    reconcile(score, saved_images[identifier]["score"], f"{role}:{identifier}.score", audit)
                    reconcile(quality, saved_images[identifier]["quality"], f"{role}:{identifier}.quality", audit)
                if identifier == TARGET_ID:
                    changed_image[name] = {"score": compact_score(score), "quality": quality}
            old_matches = {key: pair["parent"][key] for key in MATCH_KEYS}
            new_matches = {key: pair["amended"][key] for key in MATCH_KEYS}
            require(old_matches == new_matches, f"coordinate amendment changed matching: {role}:{identifier}")
            matching_trace.append({"image_id": identifier, **old_matches})
            if identifier == TARGET_ID:
                affected = [affected_object(change, parent_gold[identifier], new_gold[identifier], predicted[identifier],
                                            pair["parent"], record["width"], record["height"], _returned) for change in changes]
        aggregate = {}
        for name in ("parent", "amended"):
            score = api["aggregate_scores"](scores[name])
            quality = api["aggregate_quality"](qualities[name])
            if name == "parent":
                reconcile(score, saved["score"], f"{role}.aggregate.score", audit)
                reconcile(quality, saved["quality"], f"{role}.aggregate.quality", audit)
            aggregate[name] = {"score": compact_score(score), "quality": quality}
        total_audit["scalar_checks"] += audit["scalar_checks"]
        total_audit["largest_absolute_difference"] = max(total_audit["largest_absolute_difference"], audit["largest_absolute_difference"])
        results[role] = {"model_sha256": MODEL_SHA256[role],
                         "prediction_file": {"path": str(pred_path), "sha256": PREDICTION_SHA256[role], "bytes": pred_path.stat().st_size},
                         "saved_parent_scoring": {"path": str(score_path), "sha256": SAVED_SCORING_SHA256[role], "bytes": score_path.stat().st_size},
                         "checkpoint_identity": header["checkpoint_identity"], "inference": header["inference"],
                         "summaries": aggregate, "new_000010_summaries": changed_image,
                         "affected_rows": affected,
                         "parent_saved_scoring_reconciliation": audit,
                         "matching_identities_unchanged": True,
                         "matching_identity_canonical_sha256_both_references": canonical_sha256(matching_trace)}
    for path, expected in protected.items():
        pinned(path, expected)
    return {"schema_version": 1, "kind": "corner-reference-saved-prediction-replay",
            "status": "verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "new_inference": False, "new_training": False, "weights_loaded_or_verified": False,
            "source": {"script_sha256": sha256(Path(__file__)), "python": platform.python_version(),
                       "source_helpers_sha256": {f"scripts/{k}": v for k, v in HELPER_SHA256.items()},
                       "published_evidence_commit": EVIDENCE_COMMIT, "published_evidence_path": EVIDENCE_PATH,
                       "original_manifest_sha256": ORIGINAL_SHA256},
            "parent_reference": {"id": PARENT_ID, "manifest_sha256": PARENT_SHA256, "root": str(parent_dir)},
            "amended_reference": {"id": new_manifest["reference_id"], "manifest_sha256": protected[new_dir / "manifest.json"],
                                  "root": str(new_dir), "counts": counts, "objects_by_locatable_corners": eligibility},
            "procedure": {"box_confidence": THRESHOLD, "matching_iou": 0.5, "endpoint_confidence": 0.5,
                          "radii_px": list(RADII), "image_coordinate_system": [1280, 720],
                          "denominators_include_misses": True, "case_coordinates_selected_by_replay": False},
            "label_changes": changes,
            "verification": {"original_images_verified": 20, "references_verified": 2,
                             "label_files_verified_per_reference": 20, "unchanged_other_image_labels": 19,
                             "boxes_and_visibility_and_row_identities_unchanged": True,
                             "matching_identities_unchanged_for_all_models": True,
                             "input_hashes_reverified_before_output": True,
                             "saved_parent_scoring_reconciliation": total_audit},
            "models": results,
            "limits": ["Coordinates were selected by image review outside this replay; this script does not certify their visual correctness.",
                       "Reference revisions do not represent new model learning. Historical scores remain attached to their original reference bytes.",
                       "All 20 images are reused development images, not an independent final test.",
                       "Missing reference endpoints have no measurable coordinate error; confidence does not establish visibility."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="source checkout with all 20 actual original validation PNGs")
    parser.add_argument("--parent-reference", type=Path, required=True, help="pinned 2026-11-11 reference directory")
    parser.add_argument("--reference-dir", type=Path, required=True, help="separate complete reference containing reviewed coordinate amendments")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="PR53 capsule evaluation directory with the six pinned files")
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON path outside all input data directories")
    args = parser.parse_args(argv)

    def deny_network(event, unused):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise PermissionError(f"offline replay forbids {event}")

    sys.addaudithook(deny_network)
    try:
        result = run(args)
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "changed_rows": len(result["label_changes"]),
                          "parent_reconciliation": result["verification"]["saved_parent_scoring_reconciliation"]}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
