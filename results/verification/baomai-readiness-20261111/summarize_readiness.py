import argparse
import hashlib
import json
from pathlib import Path
import zipfile


CAPSULE = "results/evaluation/baomai-annotations-seed46-20261111/evaluation.zip"
CAPSULE_SHA256 = "3dfe3415841b47a605e2fd4b9e490b2c12b348b0014c6c2a8c78c06f323ffc1a"
MEMBER = "evidence/seed-46/comparison-original-and-pr30.json"
OFFICIAL = "docs/annotation-confirmation-2026-11-11.json"
REFERENCE = "data/references/visible-front-2026-11-10/manifest.json"
OUTPUT = "results/verification/baomai-readiness-20261111/readiness.json"
SETTINGS = (("m0", 0.25), ("m1", 0.25), ("m2", 0.25), ("m2", 0.15))
SCORE_COUNTS = ("gold_count", "prediction_count", "tp", "fn", "fp", "duplicate_fp", "other_fp", "visible_endpoints", "returned_endpoints", "missing_endpoints")
QUALITY_COUNTS = ("both_visible_objects", "both_visible_objects_with_both_returned", "unlocatable_reference_endpoints", "unlocatable_on_matched_objects", "returned_on_unlocatable_reference")


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_setting(comparison, official, model, confidence):
    rows = sorted((row for row in comparison["per_image"] if (row["model_id"], row["box_confidence"]) == (model, confidence)), key=lambda row: row["image_id"])
    require(len(rows) == len({row["image_id"] for row in rows}) == 20, "Expected 20 unique image records")
    views = [row["revised"] for row in rows]
    aggregate = next(row["revised"] for row in comparison["metrics"] if (row["model_id"], row["box_confidence"]) == (model, confidence))
    counts = {name: sum(view["score"][name] for view in views) for name in SCORE_COUNTS}
    require(all(counts[name] == aggregate["score"][name] for name in SCORE_COUNTS), "Per-image detection totals differ from frozen aggregate")
    quality = {name: sum(view["quality"][name] for view in views) for name in QUALITY_COUNTS}
    require(all(quality[name] == aggregate["quality"][name] for name in QUALITY_COUNTS), "Per-image quality totals differ from frozen aggregate")
    strata = []
    for corners in (0, 1, 2):
        key = str(corners)
        objects = sum(view["objects_by_locatable_corners"][key] for view in views)
        misses = sum(view["misses_by_locatable_corners"][key] for view in views)
        require(objects == aggregate["objects_by_locatable_corners"][key] and misses == aggregate["misses_by_locatable_corners"][key], "Visibility stratum differs from frozen aggregate")
        strata.append({"locatable_corners": corners, "objects": objects, "misses": misses, "miss_rate": misses / objects})
    require(sum(row["objects"] for row in strata) == counts["gold_count"] and sum(row["misses"] for row in strata) == counts["fn"], "Strata do not cover all objects and misses")
    radii = []
    for radius in (5, 10):
        pair_count = sum(next(row["both_visible_objects_within_radius"] for row in view["quality"]["radii"] if row["radius_px"] == radius) for view in views)
        expected = next(row for row in aggregate["quality"]["radii"] if row["radius_px"] == radius)
        require(pair_count == expected["both_visible_objects_within_radius"], "Pair-success count differs from frozen aggregate")
        radii.append({"radius_px": radius, "successful_objects": pair_count, "eligible_objects": quality["both_visible_objects"], "success_fraction": pair_count / quality["both_visible_objects"]})
    expected_official = next(row["revised"] for row in official["metrics"] if (row["model_id"], row["box_confidence"]) == (model, confidence))
    checks = {
        "missed_objects": counts["fn"], "extra_detections": counts["fp"],
        "missing_visible_endpoints": counts["missing_endpoints"],
        "both_corners_within5px": radii[0]["successful_objects"],
        "both_corners_within10px": radii[1]["successful_objects"],
        "both_corner_eligible_objects": quality["both_visible_objects"],
        "objects_by_locatable_corners": {str(row["locatable_corners"]): row["objects"] for row in strata},
        "misses_by_locatable_corners": {str(row["locatable_corners"]): row["misses"] for row in strata},
        "returned_on_unlocatable_sides": quality["returned_on_unlocatable_reference"],
        "unlocatable_sides_on_matched_objects": quality["unlocatable_on_matched_objects"],
    }
    require(all(value == expected_official[name] for name, value in checks.items()), "Summary differs from official seed46 report")
    nonempty = [row for row in rows if row["revised"]["score"]["gold_count"] > 0]
    empty = [row for row in rows if row["revised"]["score"]["gold_count"] == 0]
    miss_free = [row["image_id"] for row in nonempty if row["revised"]["score"]["fn"] == 0]
    detection_error_free = [row["image_id"] for row in nonempty if row["revised"]["score"]["fn"] == row["revised"]["score"]["fp"] == 0]
    worst = {}
    for key in ("fn", "fp"):
        maximum = max(row["revised"]["score"][key] for row in nonempty)
        worst[key] = {"count": maximum, "image_ids": [row["image_id"] for row in nonempty if row["revised"]["score"][key] == maximum]}
    return {
        "model_id": model, "box_confidence": confidence,
        "counts": counts, "object_recall": counts["tp"] / counts["gold_count"],
        "object_precision": counts["tp"] / counts["prediction_count"],
        "visibility_strata": strata, "both_corner_success": radii, "endpoint_availability": quality,
        "scenes": {"nonempty_images": len(nonempty), "miss_free_nonempty_count": len(miss_free), "miss_free_nonempty_ids": miss_free, "detection_error_free_nonempty_count": len(detection_error_free), "detection_error_free_nonempty_ids": detection_error_free, "empty_image_ids": [row["image_id"] for row in empty], "extras_on_empty_images": sum(row["revised"]["score"]["fp"] for row in empty), "worst_nonempty": worst},
        "per_image": [{"image_id": row["image_id"], **{name: row["revised"]["score"][name] for name in ("gold_count", "tp", "fn", "fp")}} for row in rows],
    }


def summarize(root):
    capsule = (root / CAPSULE).read_bytes()
    require(sha256(capsule) == CAPSULE_SHA256, "Frozen evaluation capsule hash mismatch")
    checksum = (root / (CAPSULE + ".sha256")).read_bytes()
    require(checksum.decode().split()[0] == CAPSULE_SHA256, "Published capsule checksum mismatch")
    with zipfile.ZipFile(root / CAPSULE) as archive:
        member = archive.read(MEMBER)
    comparison = json.loads(member)
    official_bytes = (root / OFFICIAL).read_bytes()
    official = json.loads(official_bytes)
    require(sha256(member) == official["complete_summary_sha256"]["seed46"], "Comparison member differs from published report")
    reference = (root / REFERENCE).read_bytes()
    require(sha256(reference) == comparison["revised_reference"]["manifest_sha256"], "Revised reference manifest mismatch")
    seed46 = next(row for row in official["per_seed"] if row["seed"] == 46)
    summaries = [summarize_setting(comparison, seed46, model, confidence) for model, confidence in SETTINGS]
    primary, lower = summaries[2:]
    gained_matches = primary["counts"]["fn"] - lower["counts"]["fn"]
    extra_cost = lower["counts"]["fp"] - primary["counts"]["fp"]
    return {
        "schema_version": 1, "kind": "baomai-saved-score-readiness-summary", "report_date": "2026-11-11", "timezone": "Asia/Hong_Kong",
        "sources": {"published_evidence_revision": "5e4950a0071c9960d430a84b1ba16c586e775714", "capsule": {"path": CAPSULE, "sha256": CAPSULE_SHA256}, "checksum_file_sha256": sha256(checksum), "comparison": {"member": MEMBER, "sha256": sha256(member)}, "official_summary": {"path": OFFICIAL, "sha256": sha256(official_bytes)}, "reference_manifest": {"path": REFERENCE, "sha256": sha256(reference)}, "generator_sha256": sha256(Path(__file__).read_bytes())},
        "procedure": {"reference_view": "revised", "reference_id": comparison["revised_reference"]["reference_id"], "inference_settings": comparison["inference"]["inference"], "matching_iou": comparison["procedure"]["matching_iou"], "endpoint_confidence": comparison["procedure"]["endpoint_confidence"], "coordinates": "Euclidean distance in original 1280x720 pixels", "ten_pixels": "provisional per-corner diagnostic ceiling; not physical acceptance or annotation slack", "scene_success_definition": "Miss-free means FN=0; detection-error-free means FN=FP=0. Neither establishes accurate corners or complete task success.", "empty_scenes": "Reported separately; excluded from nonempty scene fractions", "new_inference_or_training": False, "verification": "Per-image revised counts sum exactly to the frozen revised aggregates and the official seed46 report."},
        "models": {model: {"role": comparison["model_roles"][model], "checkpoint_sha256": comparison["models"][model]["model_sha256"]} for model in ("m0", "m1", "m2")},
        "settings": summaries,
        "candidate_confidence_015_vs_025": {"recovered_matches": gained_matches, "additional_extras": extra_cost, "additional_extras_per_recovered_match": extra_cost / gained_matches, "both_corner_success_count_delta": {str(row["radius_px"]): row["successful_objects"] - primary["both_corner_success"][index]["successful_objects"] for index, row in enumerate(lower["both_corner_success"])}, "miss_free_nonempty_count_delta": lower["scenes"]["miss_free_nonempty_count"] - primary["scenes"]["miss_free_nonempty_count"]},
        "limits": ["Twenty development images already used for model selection and repeated review; not independent unseen-scene testing.", "Objects, endpoints and related frames are correlated; no independent-trial confidence interval or significance claim.", "Unresolved reference cases remain included under the existing policy; they are neither removed nor treated as established label errors.", "Returned coordinates on unlocatable reference sides measure availability disagreement, not observable hidden-coordinate error.", "This summary verifies saved-score arithmetic and provenance; it performs no image adjudication, checkpoint loading, inference or training."]
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize the frozen seed46 revised-reference scores without inference.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output or args.root / OUTPUT
    encoded = (json.dumps(summarize(args.root), indent=2, sort_keys=True) + "\n").encode()
    if args.check:
        require(output.read_bytes() == encoded, "Stored readiness summary differs from deterministic regeneration")
        print("Readiness summary matches deterministic regeneration exactly.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
