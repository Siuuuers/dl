#!/usr/bin/env python3
"""Verify and replay R38 development-only amendments using existing predictions.

This module and its scoring dependencies use only the Python standard library.
It never loads image bytes, model weights, Torch or a training implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = "experiments/front-review-2026-11-26-r38-replay.json"
SOURCE_FILES = (
    "scripts/front_review_r38_replay.py", "scripts/score_pr123_review.py",
    "scripts/evaluate_pose.py", "scripts/pose_scoring.py",
    "scripts/diagnose_pose_quality.py", "scripts/pose_checkpoint.py",
)
DATA_MANIFEST = "data/manifest.json"
R37_REVIEW = "data/reviews/fronts-2026-11-26-r37/manifest.json"
R38_REVIEW = "data/reviews/fronts-2026-11-26-r38/manifest.json"
ELIGIBLE_R37 = "data/candidates/front-review-2026-11-26-r37/manifest.json"
EXPECTED_REQUESTS = {
    ("new_000010", 29): "replace", ("new_000008", 7): "replace",
    ("new_000007", 47): "replace", ("new_000086", 57): "delete",
    ("new_000088", 57): "delete", ("new_000053", 1): "delete",
    ("new_000010", 30): "delete", ("new_000009", 52): "delete",
}
EXPECTED_REVIEW_COUNTS = {"r37": (659, 1224, 94, 583), "r38": (654, 1220, 88, 583)}
RESULT_NAME = "baomai-front-review-r38-replay-results.zip"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def contained(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            "Replay paths must be relative to the source checkout")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(),
            f"Missing or escaping replay input: {relative}")
    require(not path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/v1"),
            f"Replay input is an LFS pointer: {relative}")
    return path


def descriptor(root, relative):
    return {"path": relative, "sha256": sha(contained(root, relative))}


def _relative_label(manifest_path, item):
    return (Path(manifest_path).parent / item["label_path"]).as_posix()


def make_contract(root):
    """Build a review replay contract after annotations and cache scoring freeze."""
    from score_pr123_review import REFERENCES, default_prediction_paths
    root = Path(root).resolve()
    references = {**REFERENCES, "r38": R38_REVIEW}
    files = {}

    def bind(relative, expected=None):
        item = descriptor(root, relative)
        require(expected is None or item["sha256"] == expected, f"Declared input hash differs: {relative}")
        files[relative] = item["sha256"]
        return item

    original = read_json(root / DATA_MANIFEST)
    bind(DATA_MANIFEST)
    for item in original["records"]:
        bind((Path("data") / item["label"]).as_posix(), item["label_sha256"])
    reviews = {}
    for name, relative in references.items():
        reviews[name] = bind(relative)
        manifest = read_json(root / relative)
        for item in manifest["files"]:
            if name in ("r37", "r38") or item.get("split", "val") == "val":
                bind(_relative_label(relative, item), item["label_sha256"])
    current = read_json(root / R38_REVIEW)
    evidence = {}
    for key in ("operations", "row_maps", "review_status", "amendment_summary", "training_unchanged_audit", "replay_script"):
        value = current.get(key)
        if isinstance(value, dict) and "path" in value and "sha256" in value:
            evidence[key] = bind(value["path"], value["sha256"])
    for i, value in enumerate(current.get("review_evidence", [])):
        evidence[f"review_evidence_{i}"] = bind(value["path"], value["sha256"])
    evidence["input_context_audit"] = bind("results/review/front-review-20261126-r38/input-context-audit.json")
    require({"operations", "row_maps"} <= evidence.keys(), "R38 operations and row maps are required")
    eligible = read_json(root / ELIGIBLE_R37)
    eligible_descriptor = bind(ELIGIBLE_R37)
    overlays = {item["image_id"]: item for item in eligible["changes"]}
    records = {item["id"]: item for item in original["records"]}
    effective = []
    for item in eligible["image_sources"]:
        name = item["image_id"]
        selected = item["effective_label"]
        bind(selected["path"], selected["sha256"])
        relative = ((Path(ELIGIBLE_R37).parent / overlays[name]["candidate_label"]).as_posix()
                    if name in overlays else (Path("data") / records[name]["label"]).as_posix())
        actual = bind(relative, selected["sha256"])
        effective.append({"image_id": name, "selection": item["selection"],
                          "source_image_sha256": item["source_image_sha256"],
                          "effective_label": actual, "selected_source_label": selected})
    caches = {}
    for role, relative in default_prediction_paths().items():
        item = bind(relative)
        header = json.loads((root / relative).read_text().splitlines()[0])
        caches[role] = {**item, "checkpoint_sha256": header["model_sha256"]}
    expected = {name: bind(f"results/evaluation/pr123-r37-20261126/{filename}")
                for name, filename in (("comparison", "comparison.json"),
                                       ("reference_delta", "r38-reference-delta.json"))}
    return {
        "schema_version": 1, "kind": "baomai-r38-cached-review-replay",
        "review_round": "r38", "training": False, "inference": False,
        "automatic_model_promotion": False, "retained_selected_model": "PR35",
        "scope": "Existing predictions replayed on separate frozen, historical, R37 and R38 references. R38 changes development labels only; no new model is trained.",
        "original_data_manifest": descriptor(root, DATA_MANIFEST),
        "review_manifests": {name: reviews[name] for name in ("r37", "r38")},
        "reference_manifests": reviews, "review_evidence": evidence,
        "unchanged_eligible_training_manifest": eligible_descriptor,
        "unchanged_effective_training_labels": effective,
        "prediction_caches": caches, "expected_replay": expected,
        "scoring": {"box_confidence": .13, "matching_iou": .5, "endpoint_confidence": .5,
                    "radii_px": [5, 10], "nms_iou": .7},
        "source_files_sha256": {name: sha(root / name) for name in SOURCE_FILES},
        "input_files_sha256": dict(sorted(files.items())),
        "limits": ["Cached replay does not rerun inference or verify present GPU performance.",
                   "Deleting a reference row is an evaluation exclusion, not a detector recovery or proof of a duplicate.",
                   "R37/R38 are repeatedly inspected development references, not independent new capture sessions."]
    }


def _stats(root, relative, records, all_splits=False):
    from evaluate_pose import parse_pose_labels
    manifest = read_json(root / relative)
    items = [item for item in manifest["files"] if all_splits or item.get("split", "val") == "val"]
    require(len({item["image_id"] for item in items}) == len(items), "Duplicate reference image ID")
    result = {}; by_id = {}
    for item in items:
        name = item["image_id"]; original = records[name]; split = item.get("split", "val")
        require(split == original["split"], f"Split changed: {name}")
        require(item.get("source_image_sha256", item.get("image_sha256")) == original["image_sha256"],
                f"Reference source image changed: {name}")
        label = contained(root, _relative_label(relative, item))
        require(sha(label) == item["label_sha256"], f"Reference label hash differs: {name}")
        objects = parse_pose_labels(label.read_text(), *original["original_size"], str(label))
        count = result.setdefault(split, {"images": 0, "objects": 0, "visible_endpoints": 0,
                                          "missing_endpoints": 0, "empty_images": 0, "both_visible_objects": 0})
        visible = sum(point[2] == 2 for obj in objects for point in obj["keypoints"])
        count["images"] += 1; count["objects"] += len(objects); count["visible_endpoints"] += visible
        count["missing_endpoints"] += 2 * len(objects) - visible; count["empty_images"] += not objects
        count["both_visible_objects"] += sum(all(point[2] == 2 for point in obj["keypoints"]) for obj in objects)
        by_id[name] = (item, label, objects)
    return result, by_id


def _verify_lineage(root, old, new, current, records):
    operations = read_json(contained(root, current["operations"]["path"]))
    row_maps = read_json(contained(root, current["row_maps"]["path"]))
    found = {(item["image_id"], item["base_row"]): item["action"] for item in operations}
    require(len(found) == len(operations) and found == EXPECTED_REQUESTS,
            "R38 operation identities differ from the eight reviewed requests")
    require(set(row_maps) == set(records), "R38 row maps do not cover exactly 157 source images")
    lookup = {(item["image_id"], item["base_row"]): item for item in operations}
    for item in operations:
        require(item["split"] == records[item["image_id"]]["split"] == "val",
                "R38 replay must not contain a training-label operation")
    unchanged_train = []; lineage = []
    for name, original in records.items():
        before = old[name][1].read_text().splitlines(); after = new[name][1].read_text().splitlines()
        generated = []; retained = []; deleted = []
        for base_row, line in enumerate(before, 1):
            op = lookup.get((name, base_row))
            if op is not None:
                require(op["before_line"] == line, f"Operation uses a different R37 row: {name}:{base_row}")
            if op is not None and op["action"] == "delete":
                deleted.append(base_row)
                lineage.append({"image_id": name, "r37_row": base_row, "r38_row": None, "action": "delete"})
                continue
            generated.append(op["after_line"] if op is not None else line)
            retained.append({"base_row": base_row, "current_row": len(generated), "changed": op is not None})
            if op is not None:
                lineage.append({"image_id": name, "r37_row": base_row, "r38_row": len(generated), "action": "replace"})
        require(generated == after, f"R38 labels do not exactly replay the audited operations: {name}")
        actual_map = row_maps[name]
        require(actual_map["base_label_sha256"] == sha(old[name][1])
                and actual_map["current_label_sha256"] == sha(new[name][1])
                and actual_map["base_row_count"] == len(before) and actual_map["current_row_count"] == len(after),
                f"Row-map file bindings differ: {name}")
        simple_rows = [{key: row[key] for key in ("base_row", "current_row", "changed")} for row in actual_map["rows"]]
        require(simple_rows == retained and actual_map.get("deleted_base_rows", []) == deleted,
                f"R38 stable-row lineage differs: {name}")
        if "deleted_rows" in actual_map:
            require([row["base_row"] for row in actual_map["deleted_rows"]] == deleted
                    and all(row["current_row"] is None for row in actual_map["deleted_rows"]),
                    f"Deleted rows lost their original identity: {name}")
            for row in actual_map["deleted_rows"]:
                require(row["before_line_sha256"] == hashlib.sha256(before[row["base_row"] - 1].encode()).hexdigest()
                        and row["stable_id"] == f"fronts-2026-11-26-r37/{name}:{row['base_row']}",
                        f"Deleted row evidence differs: {name}")
        require(all(row["stable_id"] == f"fronts-2026-11-26-r37/{name}:{row['base_row']}" for row in actual_map["rows"]),
                f"Retained original-row identity differs: {name}")
        if original["split"] == "train":
            require(old[name][1].read_bytes() == new[name][1].read_bytes(), f"R38 train labels changed: {name}")
            unchanged_train.append({"image_id": name, "r37_sha256": sha(old[name][1]), "r38_sha256": sha(new[name][1])})
    require(len(unchanged_train) == 137, "Training comparison did not cover all 137 images")
    for held in current.get("held_request_rows", []):
        if held.get("preserve_row"):
            name = held["image_id"]; base = held["row"]
            mapped = next(row["current_row"] for row in row_maps[name]["rows"] if row["base_row"] == base)
            require(old[name][1].read_text().splitlines()[base - 1] == new[name][1].read_text().splitlines()[mapped - 1],
                    f"Inherited pending row changed: {name}:{base}")
    return unchanged_train, lineage


def verify_inputs(root, contract_path=CONTRACT_PATH, expected_contract_sha256=None):
    """Verify actual source/label/cache bytes without requiring image or weight bytes."""
    from evaluate_pose import INFERENCE, parse_pose_labels
    root = Path(root).resolve(); path = contained(root, str(contract_path)); contract = read_json(path)
    require(expected_contract_sha256 is None or sha(path) == expected_contract_sha256, "Replay contract hash differs")
    require(contract["schema_version"] == 1 and contract["kind"] == "baomai-r38-cached-review-replay"
            and contract["training"] is False and contract["inference"] is False
            and contract["automatic_model_promotion"] is False, "R38 replay scope changed")
    require(set(contract["source_files_sha256"]) == set(SOURCE_FILES), "Replay source inventory differs")
    for group in ("source_files_sha256", "input_files_sha256"):
        for relative, expected in contract[group].items():
            require(sha(contained(root, relative)) == expected, f"Replay input hash differs: {relative}")
    original = read_json(root / DATA_MANIFEST); records = {item["id"]: item for item in original["records"]}
    require(len(records) == 157 and sum(r["split"] == "train" for r in records.values()) == 137,
            "Original train/development split differs")
    review_stats = {}; reviews = {}
    for name, relative in (("r37", R37_REVIEW), ("r38", R38_REVIEW)):
        require(contract["review_manifests"][name] == descriptor(root, relative), "Review manifest binding differs")
        stats, items = _stats(root, relative, records, all_splits=True)
        require(set(items) == set(records), "Review snapshot must cover all 157 images")
        manifest = read_json(root / relative)
        for split, declared in manifest["expected_counts"].items():
            require(all(stats[split][key] == value for key, value in declared.items()), f"Review counts differ: {name}/{split}")
        val = stats["val"]
        require(tuple(val[key] for key in ("objects", "visible_endpoints", "missing_endpoints", "both_visible_objects"))
                == EXPECTED_REVIEW_COUNTS[name], f"Unexpected {name} development population")
        reviews[name] = items; review_stats[name] = stats
    current = read_json(root / R38_REVIEW)
    unchanged_train, lineage = _verify_lineage(root, reviews["r37"], reviews["r38"], current, records)
    eligible = read_json(root / ELIGIBLE_R37)
    require(contract["unchanged_eligible_training_manifest"] == descriptor(root, ELIGIBLE_R37),
            "Eligible R37 training manifest changed")
    effective = contract["unchanged_effective_training_labels"]
    require(len(effective) == 137 and {item["image_id"] for item in effective}
            == {name for name, item in records.items() if item["split"] == "train"}, "Eligible train inventory differs")
    declared = {item["image_id"]: item for item in eligible["image_sources"]}
    totals = {"images": 137, "objects": 0, "visible_endpoints": 0, "missing_endpoints": 0, "empty_images": 0}
    for item in effective:
        name = item["image_id"]; selection = declared[name]
        require(item["selection"] == selection["selection"] and item["selected_source_label"] == selection["effective_label"]
                and item["source_image_sha256"] == records[name]["image_sha256"], "Eligible supervision selection differs")
        for key in ("effective_label", "selected_source_label"):
            bound = item[key]
            require(sha(contained(root, bound["path"])) == bound["sha256"], "Eligible label bytes changed")
        require(item["effective_label"]["sha256"] == item["selected_source_label"]["sha256"], "Eligible overlay no longer equals its source")
        objects = parse_pose_labels((root / item["effective_label"]["path"]).read_text(), *records[name]["original_size"])
        visible = sum(p[2] == 2 for o in objects for p in o["keypoints"])
        totals["objects"] += len(objects); totals["visible_endpoints"] += visible
        totals["missing_endpoints"] += 2 * len(objects) - visible; totals["empty_images"] += not objects
    require(totals == eligible["expected_counts"]["train"], "Eligible training counts changed")
    val_records = [r for r in original["records"] if r["split"] == "val"]
    original_val = [o for r in val_records for o in parse_pose_labels((root / "data" / r["label"]).read_text(), *r["original_size"])]
    require(len(val_records) == 20 and len(original_val) == 664, "Original trainer validation changed")
    cache_info = {}; expected_ids = [r["id"] for r in val_records]
    for role, item in contract["prediction_caches"].items():
        header, *rows = [json.loads(line) for line in (root / item["path"]).read_text().splitlines()]
        require(header["model_sha256"] == item["checkpoint_sha256"] and header["inference"] == INFERENCE
                and header["box_confidences"] == [.13], "Prediction cache identity/settings changed")
        require([row["image_id"] for row in rows] == expected_ids, "Prediction cache image order changed")
        for row in rows:
            source = records[row["image_id"]]
            require(row["image_sha256"] == source["image_sha256"] and row["label_sha256"] == source["label_sha256"]
                    and [row["width"], row["height"]] == source["original_size"] and row["threshold"] == .13,
                    "Prediction cache no longer refers to the original image inventory")
        cache_info[role] = {**item, "images": len(rows), "predictions": sum(len(row["predictions"]) for row in rows)}
    return {"schema_version": 1, "status": "verified", "contract_sha256": sha(path),
            "source_files_verified": len(SOURCE_FILES), "input_files_verified": len(contract["input_files_sha256"]),
            "training_executed": False, "inference_executed": False, "selected_model_changed": False,
            "review_counts": review_stats, "unchanged_review_training_files": unchanged_train,
            "eligible_r37_training_counts": totals, "eligible_training_changed_images": [],
            "original_trainer_validation": {"images": 20, "objects": 664},
            "review_operation_counts": {"replace": 3, "delete": 5, "train": 0, "val": 8},
            "original_request_lineage": lineage, "held_request_rows": current.get("held_request_rows", []),
            "prediction_caches": cache_info}


def verify_source(root, expected_revision):
    root = Path(root).resolve()
    git = ["git", "-c", "filter.lfs.process=", "-c", "filter.lfs.clean=", "-c", "filter.lfs.required=false", "-C", str(root)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    require(revision == expected_revision, "Source revision differs from the pinned replay source")
    status = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=all"], text=True)
    require(not status.strip(), "Source checkout is dirty; preserve it and use a clean checkout")
    tree = subprocess.check_output([*git, "rev-parse", "HEAD^{tree}"], text=True).strip()
    return {"revision": revision, "tree": tree, "working_tree_status": ""}


def run_replay(root, output, archive, expected_revision, expected_contract_sha256):
    """Replay once; exact completed results may be reused, altered outputs are refused."""
    from score_pr123_review import replay
    root = Path(root).resolve(); output = Path(output).resolve(); archive = Path(archive).resolve()
    require(not output.is_relative_to(root) and not archive.is_relative_to(root), "Replay outputs must be outside the source checkout")
    require(archive.name == RESULT_NAME and not archive.is_relative_to(output), "Replay ZIP destination differs")
    source = verify_source(root, expected_revision)
    audit = verify_inputs(root, expected_contract_sha256=expected_contract_sha256)
    contract = read_json(root / CONTRACT_PATH)
    result = replay(root, {role: item["path"] for role, item in contract["prediction_caches"].items()},
                    {name: item["path"] for name, item in contract["reference_manifests"].items()})
    for key in ("comparison", "reference_delta"):
        require(result[key] == read_json(root / contract["expected_replay"][key]["path"]),
                f"Replay differs from the preserved audited result: {key}")
    audit["source"] = source; audit["preserved_replay_exactly_reproduced"] = True
    current = read_json(root / R38_REVIEW)
    payload = {
        "comparison.json": json_bytes(result["comparison"]),
        "r38-reference-delta.json": json_bytes(result["reference_delta"]),
        "input-audit.json": json_bytes(audit),
        "replay-contract.json": (root / CONTRACT_PATH).read_bytes(),
        "row-maps.json": (root / current["row_maps"]["path"]).read_bytes(),
        "operations.json": (root / current["operations"]["path"]).read_bytes(),
        "input-context-audit.json": (root / contract["review_evidence"]["input_context_audit"]["path"]).read_bytes(),
        "README.md": ("# R38 cached review replay\n\n"
                       "These are existing PR123/PR121/PR35 predictions scored on separate references. "
                       "No training or inference ran, and PR35 remains selected.\n\n"
                       "R38 changes development labels only: three replacements and five explicit exclusions. "
                       "All 137 reviewed training files and eligible R37 supervision remain unchanged. "
                       "Use r38-reference-delta.json and row-maps.json to preserve original R37 row identities. "
                       "A removed missed label is an evaluation exclusion, not a detector recovery.\n").encode(),
    }
    index = {"schema_version": 1, "kind": "baomai-r38-cached-review-replay-results",
             "source": source, "contract_sha256": expected_contract_sha256,
             "training_executed": False, "inference_executed": False,
             "files": [{"path": name, "sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
                       for name, value in sorted(payload.items())]}
    payload["manifest.json"] = json_bytes(index)
    if output.exists():
        require(output.is_dir() and {p.name for p in output.iterdir()} == set(payload), "Existing replay output inventory differs")
        require(all((output / name).read_bytes() == value for name, value in payload.items()), "Existing replay output was changed; refusing to overwrite it")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".r38-replay-", dir=output.parent) as temporary:
            stage = Path(temporary) / "result"; stage.mkdir()
            for name, value in payload.items():
                (stage / name).write_bytes(value)
            stage.replace(output)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".r38-zip-", dir=archive.parent) as temporary:
        stage = Path(temporary) / RESULT_NAME
        with zipfile.ZipFile(stage, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for name, value in sorted(payload.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        if archive.exists():
            require(archive.read_bytes() == stage.read_bytes(), "Existing replay ZIP was changed; refusing to overwrite it")
        else:
            stage.replace(archive)
    verify_source(root, expected_revision)
    return {"status": "verified", "output": str(output), "archive": str(archive),
            "archive_sha256": sha(archive), "archive_size_bytes": archive.stat().st_size,
            "archive_members": len(payload), "comparison": result["comparison"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--build-contract", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--source-revision")
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.build_contract:
        path = root / CONTRACT_PATH; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json_bytes(make_contract(root)))
        report = verify_inputs(root)
        print(json.dumps({"contract": CONTRACT_PATH, "sha256": sha(path), "input_files_verified": report["input_files_verified"]}, indent=2))
    elif args.verify_only:
        print(json.dumps(verify_inputs(root, expected_contract_sha256=args.contract_sha256), indent=2))
    else:
        require(all((args.source_revision, args.contract_sha256, args.output, args.archive)),
                "Replay needs --source-revision, --contract-sha256, --output and --archive")
        result = run_replay(root, args.output, args.archive, args.source_revision, args.contract_sha256)
        print(json.dumps({key: value for key, value in result.items() if key != "comparison"}, indent=2))


if __name__ == "__main__":
    main()
