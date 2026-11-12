#!/usr/bin/env python3
"""Verify a frozen two-endpoint adjudication and replay the same saved outputs.

Only new_000010 row 28 L -> (923,492,2) and row 32 L -> canonical (0,0,0)
are admitted. Coordinates and visibility are chosen outside this program. The
unchanged historical replay verifies its original saved scores; this program
then reconciles that replay to frozen PR56 evidence and scores the adjudication.
No inference, training, weight loading, source edits, or network access occurs.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys

sys.dont_write_bytecode = True

DRIVER_PATH = "scripts/replay_corner_reference.py"
DRIVER_SHA256 = "f8f8371531274d45901e5715f863481916398fed9c6c005178c7f534a3588925"
PARENT_PATH = "data/references/visible-front-corners-2026-11-12"
PARENT_ID = "baomai-visible-front-corners-2026-11-12"
PARENT_SHA256 = "28ba06fe7183ebbfa6f295639194a67c4b473c17a8d67c28a9733eb359bdba92"
PARENT_REPLAY_PATH = "results/review/visible-corners-20261112/replay.json"
PARENT_REPLAY_SHA256 = "f6776b8cfc5fe2417bf412a285230e77f48d536e99129935dc33a40c1df6aa7d"
HISTORICAL_PATH = "data/references/visible-front-2026-11-11"
NEW_PATH = "data/references/visible-front-adjudication-2026-11-12"
NEW_ID = "baomai-visible-front-adjudication-2026-11-12"
NEW_MANIFEST_SHA256 = "75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e"
REVIEW_PATH = "results/review/corner-adjudication-20261112/review.json"
REVIEW_SHA256 = "73dce68d0a701e67c63a51eccfe4b5bf7164b126d8cc69e7ba38d474309adf82"
TARGET_LABEL_SHA256 = "65a81914d32d9153963d6bbb05bd4e57012e7aee9126d1388e3f6cbb3a3eac9d"
TARGET = "new_000010"
EXPECTED_TOKENS = {
    28: {5: "0.72109375", 6: "0.68333333"},
    32: {5: "0.00000000", 6: "0.00000000", 7: "0"},
}
PARENT_COUNTS = {"images": 20, "objects": 663, "visible_endpoints": 1228,
                 "missing_endpoints": 98, "empty_images": 3}
NEW_COUNTS = {**PARENT_COUNTS, "visible_endpoints": 1227, "missing_endpoints": 99}
PARENT_ELIGIBILITY = {"0": 19, "1": 60, "2": 584}
NEW_ELIGIBILITY = {"0": 20, "1": 59, "2": 584}
DETECTION_FIELDS = ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pinned(path, expected):
    require(Path(path).is_file() and sha256(path) == expected, f"SHA256 mismatch: {path}")


def relative_input(root, value):
    require(isinstance(value, str) and value, "input path must be a nonempty string")
    rel = Path(value)
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe input path: {value}")
    path = root / rel
    require(path.resolve().is_relative_to(root), f"input escapes its root: {value}")
    require(not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)),
            f"symlink input is not supported: {value}")
    require(path.is_file(), f"missing input: {path}")
    return path


def fresh_output(path, forbidden):
    raw = path.absolute()
    require(not raw.exists() and not raw.is_symlink(), f"output already exists: {raw}")
    require(not any(p.is_symlink() for p in raw.parents), "output parents must not be symlinks")
    output = raw.resolve()
    require(output.suffix == ".json", "output must be a fresh JSON file")
    require(not any(output.is_relative_to(p) for p in forbidden),
            "output must be outside source data/code, reference, prediction, and frozen evidence directories")
    return output


def load_driver(review_root):
    path = relative_input(review_root, DRIVER_PATH)
    pinned(path, DRIVER_SHA256)
    spec = importlib.util.spec_from_file_location("frozen_corner_reference_replay", path)
    require(spec is not None and spec.loader is not None, "cannot load frozen replay driver")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    require(driver.THRESHOLD == 0.13 and driver.RADII == (5, 10), "frozen replay procedure differs")
    return driver, path


def parse_timestamp(value, location):
    require(isinstance(value, str), f"missing freeze timestamp: {location}")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None and stamp.utcoffset() is not None,
            f"freeze timestamp must include a timezone: {location}")
    return stamp.astimezone(timezone.utc)


def verify_freeze(driver, review_root, manifest, parent_manifest, changes, started_at):
    """This gate runs before any saved prediction or replay body is opened."""
    link = manifest.get("review", {})
    require(link.get("path") == REVIEW_PATH and link.get("sha256") == REVIEW_SHA256,
            "new reference must name the exact frozen adjudication review")
    review_path = relative_input(review_root, link.get("path"))
    digest = link.get("sha256")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid frozen review hash")
    pinned(review_path, digest)
    review = driver.read_json(review_path)
    frozen_at = link.get("coordinates_and_states_frozen_at_utc")
    stamp = parse_timestamp(frozen_at, "manifest.review")
    require(stamp <= started_at, "review freeze is after replay start")
    require(stamp > parse_timestamp(parent_manifest["review"]["coordinates_frozen_at_utc"], "parent.review"),
            "adjudication freeze must follow the parent review")
    require(review.get("frozen_at_utc") == frozen_at, "review/manifest freeze timestamps differ")
    entry = next(item for item in manifest["files"] if item["image_id"] == TARGET)
    require(review.get("new_reference") == {"reference_id": NEW_ID, "path": f"{NEW_PATH}/manifest.json",
                                            "target_label_sha256": TARGET_LABEL_SHA256},
            "frozen review names a different new reference")
    require(entry["label_sha256"] == TARGET_LABEL_SHA256,
            "frozen review must bind the final new_000010 label bytes")
    require(link.get("model_coordinates_copied") is False, "review must record that coordinates were not copied from models")
    accepted = review.get("accepted_changes")
    require(isinstance(accepted, list) and len(accepted) == 2, "review must contain exactly the two allowed changes")
    by_row = {item["row_one_based"]: item for item in accepted}
    require(set(by_row) == set(EXPECTED_TOKENS), "review change rows differ")
    for change in changes:
        item = by_row[change["row_one_based"]]
        require(item.get("image_id") == TARGET and item.get("side") == "left", "review endpoint identity differs")
        for key, recorded in (("before_row_text", "parent_row_text"), ("after_row_text", "adjudicated_row_text"),
                              ("changed_token_indices_zero_based", "changed_token_indices_zero_based"),
                              ("selected_pixel_triplet", "intended_point_px")):
            require(item.get(key) == change[recorded], f"review/label change differs: row {change['row_one_based']}:{key}")
        require(item.get("before_normalized_tokens") == change["parent_row_text"].split()[5:8]
                and item.get("after_normalized_tokens") == change["adjudicated_row_text"].split()[5:8],
                "review endpoint tokens differ from label bytes")
    evidence = review.get("review_evidence")
    require(isinstance(evidence, list) and len(evidence) == 4
            and len({item["path"] for item in evidence}) == 4, "expected four distinct frozen reviewer evidence files")
    evidence_pins = {}
    for item in evidence:
        path = relative_input(review_root, item["path"])
        pinned(path, item["sha256"])
        evidence_pins[path] = item["sha256"]
    require(bool(review.get("visibility_dissent")), "the frozen review must preserve visibility dissent")
    return review_path, digest, review, evidence_pins, {
        "path": link["path"], "sha256": digest,
        "coordinates_and_states_frozen_at_utc": frozen_at,
        "final_label_sha256": entry["label_sha256"],
        "reviewer_evidence": evidence,
        "verified_before_loading_predictions": True,
    }


def replace_tokens(line, replacements):
    index = -1

    def replace(match):
        nonlocal index
        index += 1
        return replacements.get(index, match.group(0))

    result = re.sub(r"\S+", replace, line)
    require(index == 10, "expected exactly 11 tokens in a changed row")
    return result


def verify_delta(driver, parent_dir, new_dir, parent, new, parent_gold, new_gold):
    require(parent["reference_id"] == PARENT_ID and new["reference_id"] == NEW_ID, "reference IDs differ")
    require(new.get("policy_id") == parent["policy_id"], "reference policy changed")
    require(new.get("parent_reference") == {
        "reference_id": PARENT_ID, "path": f"{PARENT_PATH}/manifest.json", "sha256": PARENT_SHA256},
        "adjudication must name the exact frozen parent")
    old_entries = {x["image_id"]: x for x in parent["files"]}
    new_entries = {x["image_id"]: x for x in new["files"]}
    require(set(old_entries) == set(new_entries) and len(old_entries) == 20, "reference coverage differs")
    changes = []
    for identifier in sorted(old_entries):
        old_entry, new_entry = old_entries[identifier], new_entries[identifier]
        for key in ("source_label_path", "label_path", "source_label_sha256", "image_sha256", "width", "height",
                    "source_row_count", "row_count", "retained_source_to_revised_row_one_based", "added_revised_rows_one_based"):
            require(new_entry.get(key) == old_entry.get(key), f"identity/mapping changed: {identifier}:{key}")
        before = (parent_dir / old_entry["label_path"]).read_bytes()
        after = (new_dir / new_entry["label_path"]).read_bytes()
        if identifier != TARGET:
            require(before == after and old_entry["label_sha256"] == new_entry["label_sha256"],
                    f"unapproved label change: {identifier}")
            continue
        old_lines = before.decode("utf-8").splitlines(keepends=True)
        require(len(old_lines) == 32, "new_000010 must retain exactly 32 physical rows")
        expected_lines = list(old_lines)
        for row, replacements in EXPECTED_TOKENS.items():
            expected_lines[row - 1] = replace_tokens(old_lines[row - 1], replacements)
            require(expected_lines[row - 1] != old_lines[row - 1], f"expected a real change at row {row}")
            changes.append({"image_id": TARGET, "row_one_based": row, "side": "left",
                            "changed_token_indices_zero_based": sorted(replacements),
                            "parent_row_text": old_lines[row - 1].rstrip("\r\n"),
                            "adjudicated_row_text": expected_lines[row - 1].rstrip("\r\n"),
                            "parent_point_px": parent_gold[TARGET][row - 1]["keypoints"][0],
                            "adjudicated_point_px": new_gold[TARGET][row - 1]["keypoints"][0],
                            "intended_point_px": [923, 492, 2] if row == 28 else [0, 0, 0]})
        require(after == "".join(expected_lines).encode("utf-8"),
                "only the exact row 28 L coordinate edit and row 32 L canonical 000 edit are allowed")
        require(all(parent_gold[TARGET][row - 1]["keypoints"][0][2] == 2 for row in EXPECTED_TOKENS),
                "expected both edited parent endpoints to be visible")
    for manifest, gold, counts, eligibility in (
        (parent, parent_gold, PARENT_COUNTS, PARENT_ELIGIBILITY),
        (new, new_gold, NEW_COUNTS, NEW_ELIGIBILITY),
    ):
        require(driver.reference_counts(gold) == (counts, eligibility), "unexpected visibility-dependent counts")
        require(manifest.get("counts") == counts and manifest.get("objects_by_locatable_corners") == eligibility,
                "manifest denominator counts disagree with label bytes")
    return changes


def replay_projection(result):
    """Exclude only runtime time/version and absolute installation locations."""
    result = deepcopy(result)
    result.pop("created_at_utc")
    result["source"].pop("python")
    result["parent_reference"].pop("root")
    result["amended_reference"].pop("root")
    for model in result["models"].values():
        model["prediction_file"].pop("path")
        model["saved_parent_scoring"].pop("path")
    return result


def describe_changed_row(row, parent_gold, new_gold, predictions, score, width, height, returned, radii):
    index = row - 1
    match = next((m for m in score["matches"] if m["gold_index"] == index), None)
    prediction = predictions[match["prediction_index"]] if match else None
    endpoints = []
    for side, (before, after) in enumerate(zip(parent_gold[index]["keypoints"], new_gold[index]["keypoints"])):
        point = prediction["keypoints"][side] if prediction else None
        available = bool(point is not None and returned(point, width, height))
        old_error = math.hypot(point[0] - before[0], point[1] - before[1]) if available and before[2] == 2 else None
        new_error = math.hypot(point[0] - after[0], point[1] - after[1]) if available and after[2] == 2 else None
        endpoints.append({
            "side": ("left", "right")[side], "changed": before != after,
            "parent_reference_visibility": before[2], "adjudicated_reference_visibility": after[2],
            "parent_reference_xy": before[:2] if before[2] == 2 else None,
            "adjudicated_reference_xy": after[:2] if after[2] == 2 else None,
            "prediction_xy": point[:2] if point else None,
            "prediction_confidence": point[2] if point else None, "returned": available,
            "parent_error_px": old_error, "adjudicated_error_px": new_error,
            "parent_within_radius": {str(r): (old_error is not None and old_error <= r) if before[2] == 2 else None for r in radii},
            "adjudicated_within_radius": {str(r): (new_error is not None and new_error <= r) if after[2] == 2 else None for r in radii},
            "displayed_on_adjudicated_unlocatable_reference": after[2] == 0 and available,
        })
    result = {"image_id": TARGET, "row_one_based": row, "matched": match is not None,
              "prediction_index_zero_based": match["prediction_index"] if match else None,
              "matching_iou": match["iou"] if match else None,
              "box_confidence": prediction["confidence"] if prediction else None, "endpoints": endpoints}
    for name in ("parent", "adjudicated"):
        eligible = all(e[f"{name}_reference_visibility"] == 2 for e in endpoints)
        result[f"{name}_both_corners_eligible"] = eligible
        result[f"{name}_both_within_radius"] = {
            str(r): all(e[f"{name}_within_radius"][str(r)] for e in endpoints) if eligible else None for r in radii}
    return result


def unlocatable_displayed(identifier, gold, predictions, score, width, height, returned):
    rows = []
    for match in score["matches"]:
        prediction = predictions[match["prediction_index"]]
        for side, (reference, point) in enumerate(zip(gold[match["gold_index"]]["keypoints"], prediction["keypoints"])):
            if reference[2] == 0 and returned(point, width, height):
                rows.append({"image_id": identifier, "row_one_based": match["gold_index"] + 1,
                             "side": ("left", "right")[side],
                             "prediction_index_zero_based": match["prediction_index"],
                             "prediction_xy": point[:2], "endpoint_confidence": point[2],
                             "coordinate_error_px": None})
    return rows


def denominator_view(quality):
    return {name: quality[name] for name in (
        "gold_count", "visible_endpoints", "returned_visible_endpoints", "both_visible_objects",
        "both_visible_objects_with_both_returned", "unlocatable_reference_endpoints",
        "unlocatable_on_matched_objects", "returned_on_unlocatable_reference")}


def change_attribution(affected, aggregates, radii):
    by_row = {item["row_one_based"]: item for item in affected}
    moved, removed = by_row[28]["endpoints"][0], by_row[32]["endpoints"][0]
    before, after = (aggregates[name]["quality"] for name in ("parent", "adjudicated"))
    require(after["visible_endpoints"] - before["visible_endpoints"] == -1, "visible denominator delta differs")
    require(after["both_visible_objects"] == before["both_visible_objects"], "both-corner denominator changed")
    require(after["returned_visible_endpoints"] - before["returned_visible_endpoints"] == -int(removed["returned"]),
            "returned-visible count delta is not explained by row 32 L")
    matched = by_row[32]["matched"]
    require(after["unlocatable_on_matched_objects"] - before["unlocatable_on_matched_objects"] == int(matched),
            "unlocatable matched denominator delta differs")
    require(after["returned_on_unlocatable_reference"] - before["returned_on_unlocatable_reference"] == int(removed["returned"]),
            "displayed-on-unlocatable delta differs")
    radius_effects = []
    for i, radius in enumerate(radii):
        key = str(radius)
        coordinate_effect = int(moved["adjudicated_within_radius"][key]) - int(moved["parent_within_radius"][key])
        visibility_effect = -int(removed["parent_within_radius"][key])
        total = after["radii"][i]["visible_endpoints_within_radius"] - before["radii"][i]["visible_endpoints_within_radius"]
        require(total == coordinate_effect + visibility_effect, "endpoint success-count delta does not reconcile")
        both_effect = int(by_row[28]["adjudicated_both_within_radius"][key]) - int(by_row[28]["parent_both_within_radius"][key])
        require(after["radii"][i]["both_visible_objects_within_radius"] - before["radii"][i]["both_visible_objects_within_radius"] == both_effect,
                "both-corner success-count delta does not reconcile")
        radius_effects.append({"radius_px": radius, "row28_coordinate_edit_visible_success_count_delta": coordinate_effect,
                               "row32_unlocatable_edit_visible_success_count_delta": visibility_effect,
                               "total_visible_success_count_delta": total,
                               "both_visible_object_success_count_delta": both_effect})
    return {"visible_endpoint_denominator_delta": -1, "both_visible_object_denominator_delta": 0,
            "row32_removed_from_visible_endpoint_errors_if_returned": removed["returned"],
            "row32_added_to_unlocatable_matched_denominator": matched,
            "row32_displayed_on_unlocatable_reference": removed["returned"], "radii": radius_effects,
            "interpretation": "These are effects of reference changes on fixed predictions. Removing an unlocatable reference endpoint is not improved model localization."}


def run(args):
    started_at = datetime.now(timezone.utc)
    root, review_root, prediction_dir = (p.resolve() for p in (args.repo_root, args.review_root, args.prediction_dir))
    parent_dir, new_dir, historical_dir = review_root / PARENT_PATH, review_root / NEW_PATH, root / HISTORICAL_PATH
    output = fresh_output(args.output, (root / "data", root / "scripts", root / "metadata",
                                       review_root / "data", review_root / "scripts", prediction_dir,
                                       (review_root / PARENT_REPLAY_PATH).parent))
    driver, driver_path = load_driver(review_root)
    protected = {driver_path: DRIVER_SHA256, review_root / PARENT_REPLAY_PATH: PARENT_REPLAY_SHA256,
                 parent_dir / "manifest.json": PARENT_SHA256, historical_dir / "manifest.json": driver.PARENT_SHA256,
                 new_dir / "manifest.json": NEW_MANIFEST_SHA256, review_root / REVIEW_PATH: REVIEW_SHA256,
                 root / "data/manifest.json": driver.ORIGINAL_SHA256,
                 Path(__file__).resolve(): sha256(Path(__file__))}
    protected.update({root / "scripts" / name: digest for name, digest in driver.HELPER_SHA256.items()})
    # Hashing frozen replay bytes does not parse or expose any prediction coordinates.
    for path, digest in protected.items():
        pinned(path, digest)
    sys.path.insert(0, str(root / "scripts"))
    from rescore_reference_amendment import load_api, verify_amendment
    from diagnose_pose_quality import _returned, MATCHING_IOU, ENDPOINT_CONFIDENCE
    require((MATCHING_IOU, ENDPOINT_CONFIDENCE) == (0.5, 0.5), "scoring constants differ")
    _, api = load_api(root)
    original = api["preflight_dataset"](root, require_weights=False)
    references = {}
    for name, directory in (("historical", historical_dir), ("parent", parent_dir), ("adjudicated", new_dir)):
        manifest, gold, _, _, files = verify_amendment(root, directory, original, api)
        references[name] = {"manifest": manifest, "gold": gold, "files": files}
        protected[directory / "manifest.json"] = sha256(directory / "manifest.json")
        protected.update({directory / item["label_path"]: item["label_sha256"] for item in files})
    parent, amended = references["parent"], references["adjudicated"]
    changes = verify_delta(driver, parent_dir, new_dir, parent["manifest"], amended["manifest"], parent["gold"], amended["gold"])
    review_path, review_sha, review, evidence_pins, freeze = verify_freeze(
        driver, review_root, amended["manifest"], parent["manifest"], changes, started_at)
    protected[review_path] = review_sha
    protected.update(evidence_pins)
    require(output != review_path.resolve(), "output must not replace the frozen review")
    protected[root / "metadata/model_card.json"] = original["model_card_sha256"]
    for record in original["records"]:
        protected[Path(record["image_path"])] = record["image_sha256"]
        protected[Path(record["label_path"])] = record["label_sha256"]

    # Prediction access starts only after every new-reference and freeze check above.
    prediction_access_at = datetime.now(timezone.utc).isoformat()
    historical_replay = driver.run(argparse.Namespace(repo_root=root, parent_reference=historical_dir,
        reference_dir=parent_dir, prediction_dir=prediction_dir, output=output))
    historical_audit = historical_replay["verification"]["saved_parent_scoring_reconciliation"]
    require(historical_audit["scalar_checks"] == 15591 and historical_audit["largest_absolute_difference"] <= 1e-10,
            "historical saved-score reconciliation must reproduce all 15,591 checks")
    frozen_parent_replay = driver.read_json(review_root / PARENT_REPLAY_PATH)
    parent_replay_audit = {"scalar_checks": 0, "largest_absolute_difference": 0.0}
    driver.reconcile(replay_projection(historical_replay), replay_projection(frozen_parent_replay),
                     "frozen_PR56_replay", parent_replay_audit)

    records = {item["id"]: item for item in original["records"]}
    historical_entries = {item["image_id"]: item for item in references["historical"]["files"]}
    models = {}
    for role in driver.ROLES:
        header, predicted, _, pred_path, score_path = driver.load_predictions(prediction_dir, role, records, historical_entries)
        protected[pred_path] = driver.PREDICTION_SHA256[role]
        protected[score_path] = driver.SAVED_SCORING_SHA256[role]
        scores = {name: [] for name in ("parent", "adjudicated")}
        qualities = {name: [] for name in scores}
        per_image, matching_trace, displayed, affected = [], [], [], []
        for identifier, record in sorted(records.items()):
            pair = {}
            public = {"image_id": identifier}
            for name, reference in (("parent", parent), ("adjudicated", amended)):
                gold = reference["gold"][identifier]
                score = api["score_image"](gold, predicted[identifier], record["width"], record["height"], 0.5, 0.5)
                quality = api["quality_for_image"](gold, predicted[identifier], record["width"], record["height"], driver.RADII)
                scores[name].append(score)
                qualities[name].append(quality)
                pair[name] = score
                public[name] = {"score": driver.compact_score(score), "quality": quality}
            matches = {key: pair["parent"][key] for key in driver.MATCH_KEYS}
            require(matches == {key: pair["adjudicated"][key] for key in driver.MATCH_KEYS}, f"matching changed: {role}:{identifier}")
            require(all(pair["parent"][key] == pair["adjudicated"][key] for key in DETECTION_FIELDS),
                    f"object detection changed: {role}:{identifier}")
            matching_trace.append({"image_id": identifier, **matches})
            displayed.extend(unlocatable_displayed(identifier, amended["gold"][identifier], predicted[identifier],
                pair["adjudicated"], record["width"], record["height"], _returned))
            if identifier == TARGET:
                affected = [describe_changed_row(row, parent["gold"][identifier], amended["gold"][identifier],
                    predicted[identifier], pair["parent"], record["width"], record["height"], _returned, driver.RADII)
                    for row in EXPECTED_TOKENS]
            per_image.append(public)
        aggregates = {name: {"score": driver.compact_score(api["aggregate_scores"](scores[name])),
                             "quality": api["aggregate_quality"](qualities[name])} for name in scores}
        audit = {"scalar_checks": 0, "largest_absolute_difference": 0.0}
        frozen_model = frozen_parent_replay["models"][role]
        driver.reconcile(aggregates["parent"], frozen_model["summaries"]["amended"], f"{role}.parent_aggregate", audit)
        target_summary = next(x for x in per_image if x["image_id"] == TARGET)
        driver.reconcile(target_summary["parent"], frozen_model["new_000010_summaries"]["amended"], f"{role}.parent_target", audit)
        matching_hash = driver.canonical_sha256(matching_trace)
        require(matching_hash == frozen_model["matching_identity_canonical_sha256_both_references"],
                f"matching differs from frozen PR56: {role}")
        require(len(displayed) == aggregates["adjudicated"]["quality"]["returned_on_unlocatable_reference"],
                f"unlocatable displayed-endpoint inventory differs: {role}")
        models[role] = {
            "model_sha256": driver.MODEL_SHA256[role], "checkpoint_identity": header["checkpoint_identity"],
            "inference": header["inference"],
            "prediction_file": {"path": str(pred_path), "sha256": driver.PREDICTION_SHA256[role], "bytes": pred_path.stat().st_size},
            "summaries": aggregates, "new_000010_summaries": {name: target_summary[name] for name in scores},
            "per_image": per_image, "affected_rows": affected,
            "visibility_dependent_denominators": {name: denominator_view(aggregates[name]["quality"]) for name in scores},
            "reference_unlocatable_displayed_endpoints": displayed,
            "reference_change_attribution": change_attribution(affected, aggregates, driver.RADII),
            "parent_PR56_reconciliation": audit, "matching_identities_unchanged": True,
            "matching_identity_canonical_sha256_both_references": matching_hash,
        }
    for path, digest in protected.items():
        pinned(path, digest)
    return {
        "schema_version": 1, "kind": "corner-adjudication-saved-prediction-replay", "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "started_at_utc": started_at.isoformat(),
        "new_inference": False, "new_training": False, "weights_loaded_or_verified": False,
        "source": {"script_sha256": protected[Path(__file__).resolve()], "python": platform.python_version(),
                   "frozen_replay_driver_sha256": DRIVER_SHA256,
                   "source_helpers_sha256": {f"scripts/{name}": digest for name, digest in driver.HELPER_SHA256.items()},
                   "original_manifest_sha256": driver.ORIGINAL_SHA256,
                   "published_evidence_commit": driver.EVIDENCE_COMMIT, "published_evidence_path": driver.EVIDENCE_PATH},
        "review_freeze": {**freeze, "prediction_access_started_at_utc": prediction_access_at},
        "visibility_dissent": review["visibility_dissent"],
        "historical_reference": {"id": driver.PARENT_ID, "manifest_sha256": driver.PARENT_SHA256, "preserved": True},
        "parent_reference": {"id": PARENT_ID, "manifest_sha256": PARENT_SHA256, "preserved": True,
                             "counts": PARENT_COUNTS, "objects_by_locatable_corners": PARENT_ELIGIBILITY},
        "adjudicated_reference": {"id": NEW_ID, "manifest_sha256": protected[new_dir / "manifest.json"],
                                  "counts": NEW_COUNTS, "objects_by_locatable_corners": NEW_ELIGIBILITY},
        "procedure": {"box_confidence": 0.13, "matching_iou": 0.5, "endpoint_confidence": 0.5,
                      "radii_px": list(driver.RADII), "image_coordinate_system": [1280, 720],
                      "denominators_include_misses": True, "case_coordinates_selected_by_replay": False,
                      "unlocatable_displayed_endpoint_definition": "On a matched object, reference v0 but a finite in-bounds model coordinate is returned at endpoint confidence >=0.5. It has no measured coordinate error and is not automatically a physical false positive."},
        "label_changes": changes,
        "verification": {"original_images_verified": 20, "references_verified": 3,
                         "label_files_verified_per_reference": 20, "unchanged_other_image_labels": 19,
                         "exact_two_endpoint_delta_verified": True,
                         "boxes_classes_row_identity_and_right_endpoints_unchanged": True,
                         "visibility_changes_exactly_new_000010_row32_left_2_to_0": True,
                         "matching_identities_unchanged_for_all_models": True,
                         "input_hashes_reverified_before_output": True,
                         "historical_saved_scoring_reconciliation": historical_audit,
                         "frozen_PR56_replay_sha256": PARENT_REPLAY_SHA256,
                         "frozen_PR56_replay_reconciliation": parent_replay_audit},
        "models": models,
        "limits": [
            "Image review selected the changes before this replay. Hash and delta verification do not independently certify visual correctness.",
            "The row 32 tip has reviewer disagreement. The frozen review records the decision and dissent; no hidden physical coordinate is scored.",
            "Predictions, matching, and weights are unchanged. Any metric movement comes from changed reference coordinates or eligibility, not model improvement.",
            "Visible-endpoint success uses 1228 parent or 1227 adjudicated endpoints. Both-corner success uses 584 eligible objects in both references, including missed objects.",
            "Reference-v0 displayed coordinates are a visibility disagreement diagnostic, not proof of a physical false positive. Their coordinate accuracy is unmeasurable against this reference.",
            "These 20 images are reused development images, not an independent final test. Historical scores remain attached to their original reference bytes.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="source checkout with original images, labels, frozen historical reference and helpers")
    parser.add_argument("--review-root", type=Path, required=True, help="full current overlay with parent/new references, frozen review, driver and PR56 replay")
    parser.add_argument("--prediction-dir", type=Path, required=True, help="directory containing the six original pinned prediction/scoring files")
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON path outside protected inputs")
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
                          "historical_reconciliation": result["verification"]["historical_saved_scoring_reconciliation"],
                          "frozen_PR56_reconciliation": result["verification"]["frozen_PR56_replay_reconciliation"]}, indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError, StopIteration) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
