#!/usr/bin/env python3
"""Replay immutable PR66 predictions on the 14 and 18 November references.

No inference, training, checkpoint loading or model promotion is performed.
Source JSON is extracted only into a temporary directory under <root>/tmp.
Existing output files must reproduce byte for byte; use a fresh output directory
for a deliberately different experiment instead of overwriting evidence.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

ARTIFACT_SHA = "a60c9404d8ca07a1ccbb724f999e7d7aacb73d373a1f2769edef1e6e9bacc915"
SOURCE_HELPER_SHA = "01fcb42ecce388b63302f569c63f75c3525d1451aa5a20b1fa4f00cf543bda71"
MODEL_SHA = {
    "pr35": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
    "reference": "92ab01f57fa5029976d3929ff20af0c37d132aa9abe2b433943d58998e47bb5b",
    "candidate": "2461f93dea1bec63754c08a2918f9077780820d164a1b330334b862d9653654d",
}
PREDICTION_SHA = {
    "pr35": "97debc8ca8f26d47bd3fc4f3171f206533d42ad8a8c66e9f63b8018b4e47cfd8",
    "reference": "fdfb31d3c18c2c1a0f1a75753409ecf9707e62f0fca8ab6bbb7ffe155efa24eb",
    "candidate": "0614037f598a1418ff23ab52a2afbcdbf5f46524dc3edae53ea7293d7128d776",
}
HELPERS = {
    "pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "diagnose_pose_quality.py": "a3baa1285d1580ee44547f7c3d606f524f2f40616fed568885746d2014159953",
    "evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "rescore_reference_amendment.py": "a4321d0354345ca89f7055ef8bacc611cd342b7d047bbc0e9df48aaa4815c2de",
    "pose_checkpoint.py": "e0f85a76fd5234a98e66783d3408e22175ba334c6209f91aca8680221a2f8d5d",
    "verify_pose_candidate.py": "03395b9a82a4c120647a1cac51c89fd214d8cf9b15c04e2196a40f39077d8491",
}
HISTORICAL_REFERENCE = Path("data/references/visible-front-completion-2026-11-14/manifest.json")
HISTORICAL_SHA = "bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1"
REVIEWED_REFERENCE = Path("data/references/sequence-refined-reference-2026-11-18/manifest.json")
REVIEWED_SHA = "0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96"
DEFAULT_OUTPUT = Path("results/review/sequence-refinement-20261118/model-replay")
ARTIFACT_MEMBERS = {
    "comparison.json", "training-audit.json",
    "eval-pr35/predictions.jsonl", "eval-reference/predictions.jsonl",
    "eval-candidate/predictions.jsonl",
}
HISTORICAL_KEY = "historical_2026_11_14"
REVIEWED_KEY = "reviewed_2026_11_18"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON constant in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def resolve(root, path):
    return (path if path.is_absolute() else root / path).resolve()


def relative_name(root, path):
    require(path.is_relative_to(root), f"Reproducible input/output must be within --root: {path}")
    return path.relative_to(root).as_posix()


def preserve_output(path, payload, verify_only):
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                f"Existing replay output differs; preserve it and choose a fresh directory: {path}")
    else:
        require(not verify_only, f"Missing expected replay output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def extract_source(archive, destination):
    """Read only the five pinned members; never extract arbitrary archive paths."""
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        require(len(names) == len(ARTIFACT_MEMBERS) and set(names) == ARTIFACT_MEMBERS,
                "Prediction artifact contains unexpected, duplicate or missing members")
        for name in names:
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts
                    and "\\" not in name and relative.as_posix() == name,
                    f"Unsafe archive member: {name}")
            target = (destination / relative).resolve()
            require(target.is_relative_to(destination.resolve()), "Archive member escapes temporary directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))


def reconcile(actual, expected, location, audit):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f"Keys differ: {location}")
        for key in expected:
            reconcile(actual[key], expected[key], location + "." + key, audit)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Length differs: {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            reconcile(left, right, f"{location}[{index}]", audit)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual), f"Non-finite value: {location}")
        difference = abs(actual - expected)
        require(difference <= 1e-10, f"Historical score differs: {location}")
        audit["max_absolute_difference"] = max(audit["max_absolute_difference"], difference)
        audit["scalar_checks"] += 1
    else:
        require(type(actual) is type(expected) and actual == expected, f"Historical value differs: {location}")
        audit["scalar_checks"] += 1


def reference_counts(gold):
    groups = {str(count): 0 for count in range(3)}
    for objects in gold.values():
        for obj in objects:
            groups[str(sum(point[2] == 2 for point in obj["keypoints"]))] += 1
    objects = sum(groups.values())
    visible = groups["1"] + 2 * groups["2"]
    return ({"images": len(gold), "objects": objects, "visible_endpoints": visible,
             "missing_endpoints": 2 * objects - visible,
             "empty_images": sum(not items for items in gold.values())}, groups)


def load_source_helper(path):
    specification = importlib.util.spec_from_file_location("immutable_pr66_measurement", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def affected_rows(manifest, report, models, box_iou):
    rows = []
    for operation in manifest["operations"]:
        image_id, row = operation["image_id"], operation["output_row"]
        require(row is not None, "This pinned replay expects replacements/additions, not deletions")
        result = {"image_id": image_id, "row": row, "operation": operation["operation"], "models": {}}
        for role, model in models.items():
            trace = report["changed_image_matching"][REVIEWED_KEY][role][image_id]
            truth = trace["gold"][row - 1]
            match = next((item for item in trace["matches"] if item["gold_index"] == row - 1), None)
            predictions = model["predictions"][image_id]
            best = max(enumerate(predictions), key=lambda item: box_iou(truth["bbox_xyxy"], item[1]["bbox_xyxy"])) if predictions else None
            result["models"][role] = {
                "matched": match is not None,
                "matched_prediction_index": match["prediction_index"] if match else None,
                "matching_iou": match["iou"] if match else None,
                "best_prediction_index": best[0] if best else None,
                "best_prediction_iou": box_iou(truth["bbox_xyxy"], best[1]["bbox_xyxy"]) if best else None,
                "best_prediction_bbox": best[1]["bbox_xyxy"] if best else None,
            }
            if operation["operation"] == "replace":
                previous = report["changed_image_matching"][HISTORICAL_KEY][role][image_id]
                result["models"][role]["historical_match"] = next(
                    (item for item in previous["matches"] if item["gold_index"] == operation["base_row"] - 1), None)
        rows.append(result)
    return rows


def replay(root, sources, source_helper, protected):
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import preflight_dataset, _read_replay
    from rescore_reference_amendment import load_api, verify_amendment
    from pose_scoring import score_image, aggregate_scores, _box_iou
    from diagnose_pose_quality import quality_for_image, aggregate_quality

    case_success = load_source_helper(source_helper).case_success
    preflight = preflight_dataset(root, require_weights=False)
    _, api = load_api(root)
    records = {record["id"]: record for record in preflight["records"]}
    saved = read(sources / "comparison.json")
    for name in ARTIFACT_MEMBERS:
        protected[sources / name] = sha(sources / name)
    for record in records.values():
        protected[Path(record["image_path"])] = record["image_sha256"]
        protected[Path(record["label_path"])] = record["label_sha256"]
    models = {}
    for role in MODEL_SHA:
        path = sources / f"eval-{role}" / "predictions.jsonl"
        require(sha(path) == PREDICTION_SHA[role], f"Prediction bytes differ: {role}")
        header = json.loads(path.read_text().splitlines()[0])
        require(header["model_sha256"] == header["checkpoint_identity"]["checkpoint_sha256"] == MODEL_SHA[role],
                f"Checkpoint identity differs: {role}")
        require(saved["models"][role] == {"predictions_sha256": PREDICTION_SHA[role], "checkpoint_sha256": MODEL_SHA[role]},
                f"Saved comparison identifies different predictions: {role}")
        require(header["source"]["revision"] == "aaa2643232c8dc8370ad92ed007c7cc77e2cdd6a"
                and not header["source"]["working_tree_status"], "Prediction source revision differs")
        require(header["source"]["files_sha256"]["scripts/pose_scoring.py"] == HELPERS["pose_scoring.py"]
                and header["runtime"]["ultralytics"] == "8.4.182", "Prediction evaluator differs")
        selected = {**preflight, "model_sha256": MODEL_SHA[role], "checkpoint_identity": header["checkpoint_identity"]}
        _, predictions, thresholds = _read_replay(path, selected, [0.13])
        require(len(predictions) == 20 and thresholds == [0.13], "Prediction inventory or threshold differs")
        models[role] = {"predictions": {item["image_id"]: item["predictions"] for item in predictions}}

    report = {
        "schema_version": 1, "kind": "saved-prediction-development-reference-diagnostic",
        "status": "provisional_development_diagnostic", "training_run": False,
        "inference_run": False, "model_promotion": False,
        "models_trained_on_current_review": False,
        "model_roles": {
            "pr35": "Retained PR35 checkpoint",
            "reference": "Previously trained PR64 matched control using PR35 labels",
            "candidate": "Previously trained PR64 candidate using the 3,788-object accepted-fronts training labels from 14 November",
        },
        "thresholds": {"box_confidence": 0.13, "endpoint_confidence": 0.5, "nms_iou": 0.7,
                       "matching_iou": 0.5, "imgsz": 1280, "radii_px": [5, 10]},
        "provenance": {
            "artifact_id": 14162004655, "workflow_run_id": 43717526943, "artifact_sha256": ARTIFACT_SHA,
            "workflow_source_revision": "aaa2643232c8dc8370ad92ed007c7cc77e2cdd6a",
            "report_revision": "d71303a2a3e2c6abed278fe91ba0f0a910fea63d",
            "pr64_revision": "8df5a4b611ac793892e589b3a3ee2789fa5506f7",
            "replay_script_sha256": sha(__file__), "source_helpers_sha256": HELPERS,
            "immutable_pr66_helper_sha256": SOURCE_HELPER_SHA, "models": saved["models"],
            "checkpoint_bytes_reloaded": False,
            "checkpoint_identity_source": "Pinned GitHub Actions prediction artifact and its training audit; this replay does not load weights.",
        },
        "references": {},
        "limits": [
            "The saved predictions are unchanged. Score changes between references are annotation effects, not model learning.",
            "These development photos have already informed labels, thresholds and models; they are not an independent camera-session test.",
            "Unmatched proposals are fixed-association outcomes, not individually adjudicated background hallucinations.",
            "Unresolved reference rows retain their stated uncertainty. This diagnostic authorizes no release or model promotion.",
            "The historical comparison here uses the 14 November completion reference, separately from PR66's 12 November primary adjudication reference.",
        ],
    }
    audit = {"scalar_checks": 0, "max_absolute_difference": 0.0}
    all_traces, manifests = {}, {}
    for reference_name, relative in ((HISTORICAL_KEY, HISTORICAL_REFERENCE), (REVIEWED_KEY, REVIEWED_REFERENCE)):
        path = root / relative
        manifest, gold, *_ = verify_amendment(root, path.parent, preflight, api)
        counts, groups = reference_counts(gold)
        require(counts == manifest["counts"] and groups == manifest["objects_by_locatable_corners"],
                "Reference counts differ from parsed labels")
        manifests[reference_name] = manifest
        for entry in manifest["files"]:
            protected[path.parent / entry["label_path"]] = entry["label_sha256"]
        per_model, traces, grouped = {}, {}, {}
        for role, model in models.items():
            scores, qualities, scenes = [], [], []
            successes = {5: set(), 10: set()}
            role_trace = {}
            group_counts = {str(count): {"gold": 0, "matched": 0, "misses": 0} for count in range(3)}
            for image_id, record in sorted(records.items()):
                truth, predicted = gold[image_id], model["predictions"][image_id]
                score = score_image(truth, predicted, record["width"], record["height"], 0.5, 0.5)
                quality = quality_for_image(truth, predicted, record["width"], record["height"], [5, 10])
                scores.append(score)
                qualities.append(quality)
                matched = {item["gold_index"] for item in score["matches"]}
                for index, obj in enumerate(truth):
                    group = group_counts[str(sum(point[2] == 2 for point in obj["keypoints"]))]
                    group["gold"] += 1
                    group["matched"] += int(index in matched)
                    group["misses"] += int(index not in matched)
                for radius in successes:
                    for row in case_success(truth, predicted, record["width"], record["height"], radius):
                        successes[radius].add(f"{image_id}:{row}")
                scenes.append({"image_id": image_id, "gold": len(truth), "predictions": len(predicted),
                               "matched": score["tp"], "misses": score["fn"], "extras": score["fp"],
                               "duplicates": score["duplicate_fp"],
                               "missed_rows": [index + 1 for index in score["unmatched_gold_indices"]]})
                role_trace[image_id] = {key: score[key] for key in
                                       ("matches", "unmatched_gold_indices", "unmatched_prediction_indices")}
                role_trace[image_id]["gold"] = truth
            aggregate = aggregate_scores(scores)
            aggregate.pop("endpoint_errors_px", None)
            per_model[role] = {
                "detection": aggregate, "corner_quality": aggregate_quality(qualities), "scenes": scenes,
                "success_identities": {str(radius): sorted(values) for radius, values in successes.items()},
                "empty_scene_outputs": sum(scene["predictions"] for scene in scenes if scene["gold"] == 0),
                "miss_free_nonempty_scenes": sum(scene["misses"] == 0 for scene in scenes if scene["gold"] > 0),
                "error_free_nonempty_scenes": sum(scene["misses"] == 0 and scene["extras"] == 0
                                                  for scene in scenes if scene["gold"] > 0),
            }
            traces[role], grouped[role] = role_trace, group_counts
            if reference_name == HISTORICAL_KEY:
                reconcile(per_model[role], saved["references"]["diagnostic_2026_11_14"]["models"][role], role, audit)
        differences = {
            baseline: {key: per_model["candidate"]["detection"][key] - per_model[baseline]["detection"][key]
                       for key in ("tp", "fn", "fp", "duplicate_fp", "other_fp", "returned_endpoints")}
            for baseline in ("reference", "pr35")
        }
        report["references"][reference_name] = {
            "manifest_path": relative.as_posix(), "manifest_sha256": sha(path),
            "reference_id": manifest["reference_id"], "counts": counts,
            "objects_by_locatable_corners": groups, "remaining_current_rows": manifest.get("remaining_current_rows", []),
            "models": per_model, "detection_by_locatable_corner_group": grouped,
            "candidate_minus_baseline": differences,
        }
        all_traces[reference_name] = traces
    old_entries = {entry["image_id"]: entry for entry in manifests[HISTORICAL_KEY]["files"]}
    new_entries = {entry["image_id"]: entry for entry in manifests[REVIEWED_KEY]["files"]}
    changed = [image_id for image_id in sorted(old_entries)
               if old_entries[image_id]["label_sha256"] != new_entries[image_id]["label_sha256"]]
    report["historical_reconciliation"] = audit
    report["changed_reference_images"] = changed
    report["changed_image_matching"] = {
        name: {role: {image_id: trace[image_id] for image_id in changed} for role, trace in traces.items()}
        for name, traces in all_traces.items()
    }
    report["reference_effects"] = {
        role: {key: report["references"][REVIEWED_KEY]["models"][role]["detection"][key]
                    - report["references"][HISTORICAL_KEY]["models"][role]["detection"][key]
               for key in ("gold_count", "tp", "fn", "fp", "duplicate_fp", "other_fp", "returned_endpoints")}
        for role in MODEL_SHA
    }
    statuses = affected_rows(manifests[REVIEWED_KEY], report, models, _box_iou)
    for path, expected in protected.items():
        require(sha(path) == expected, f"Input changed during replay: {path}")
    report["protected_inputs_verified_unchanged"] = len(protected)
    return report, statuses


def run(args):
    root = args.root.resolve()
    archive, source_helper, output = [resolve(root, path) for path in
                                      (args.source_archive, args.source_helper, args.output_directory)]
    for path in (archive, source_helper, output):
        relative_name(root, path)
    require(not output.is_relative_to(root / "data"), "Replay output cannot modify source data")
    protected = {archive: ARTIFACT_SHA, source_helper: SOURCE_HELPER_SHA,
                 root / HISTORICAL_REFERENCE: HISTORICAL_SHA, root / REVIEWED_REFERENCE: REVIEWED_SHA}
    protected.update({root / "scripts" / name: digest for name, digest in HELPERS.items()})
    for relative in ("data/manifest.json", "metadata/model_card.json"):
        protected[root / relative] = sha(root / relative)
    protected[Path(__file__).resolve()] = sha(__file__)
    for path, expected in protected.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == expected, f"Pinned input differs: {path}")
    temporary_root = root / "tmp"
    require(not temporary_root.is_symlink(), "Temporary workspace cannot be a symlink")
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sequence-reference-replay-", dir=temporary_root) as directory:
        sources = Path(directory).resolve()
        extract_source(archive, sources)
        report, statuses = replay(root, sources, source_helper, protected)
        report_bytes = encoded(report)
        affected = {
            "reference_manifest_sha256": REVIEWED_SHA,
            "replay_report_sha256": hashlib.sha256(report_bytes).hexdigest(), "matching_iou": 0.5,
            "note": "Geometric association is not physical presence adjudication. Prediction arrays and thresholds are unchanged.",
            "rows": statuses,
        }
        affected_bytes = encoded(affected)
        payloads = {"sequence-reference-replay.json": report_bytes, "affected-row-status.json": affected_bytes}
        inventory = {
            "status": "verified_read_only_development_replay",
            "github": {
                "repository": "Siuuuers/dl", "artifact_id": 14162004655,
                "artifact_url": "https://github.com/Siuuuers/dl/actions/runs/43717526943/artifacts/14162004655",
                "workflow_run_id": 43717526943, "workflow_head_sha": "aaa2643232c8dc8370ad92ed007c7cc77e2cdd6a",
                "report_revision": "d71303a2a3e2c6abed278fe91ba0f0a910fea63d",
                "pr64_revision": "8df5a4b611ac793892e589b3a3ee2789fa5506f7",
                "artifact_created_at": "2026-11-15T13:11:28Z", "artifact_expires_at": "2026-12-15T13:11:27Z",
                "artifact_digest": "sha256:" + ARTIFACT_SHA,
            },
            "references": {"historical_completion_2026_11_14": HISTORICAL_SHA, "reviewed_2026_11_18": REVIEWED_SHA},
            "inputs": [{"path": relative_name(root, path), "size_bytes": path.stat().st_size, "sha256": sha(path)}
                       for path in (archive, source_helper, Path(__file__).resolve())],
            "archive_members": [{"path": name, "size_bytes": (sources / name).stat().st_size, "sha256": sha(sources / name)}
                                for name in sorted(ARTIFACT_MEMBERS)],
            "outputs": [{"path": relative_name(root, output / name), "size_bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest()} for name, payload in payloads.items()],
            "reproduction": "Run python scripts/replay_sequence_reference.py --root . --verify-only from the repository. "
                            "The preserved ZIP is sufficient after GitHub artifact expiration; no weights, network, inference or training are required.",
        }
        payloads["source-inventory.json"] = encoded(inventory)
        for name, payload in payloads.items():
            preserve_output(output / name, payload, args.verify_only)
    for name, reference in report["references"].items():
        print(name, reference["counts"])
        for role, model in reference["models"].items():
            detection = model["detection"]
            radii = {int(item["radius_px"]): item for item in model["corner_quality"]["radii"]}
            print(role, {**{key: detection[key] for key in ("tp", "fn", "fp", "duplicate_fp")},
                         "both5": radii[5]["both_visible_objects_within_radius"],
                         "both10": radii[10]["both_visible_objects_within_radius"],
                         "empty": model["empty_scene_outputs"]})
    print("Historical reconciliation:", report["historical_reconciliation"])
    print("Verified outputs:" if args.verify_only else "Saved outputs:", relative_name(root, output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_OUTPUT / "pr66-evaluation-receipts.zip")
    parser.add_argument("--source-helper", type=Path, default=DEFAULT_OUTPUT / "measure_accepted_fronts_pr64.py")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
