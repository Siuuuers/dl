#!/usr/bin/env python3
"""Verify PR133 bytes and replay its saved scores on CPU; never train or infer."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile

PR_HEAD = "22a49e59adb2aeb9007c3fdac4c70ede2fb67802"
SOURCE = "ea4ac955f2dfaf1add24c243599715f802eb41a7"
NAME = "baomai-additional-front-pilot-r45-rebuilt-results.zip"
ARCHIVE_SHA = "0babe043f8358f7ceed3e4e0e570421585690ebf79496c689ba84c1b5b25869a"
ARCHIVE_BYTES = 41067308
BASE_URL = f"https://raw.githubusercontent.com/Siuuuers/dl/{PR_HEAD}/results/t4/baomai-additional-front-pilot-r45-rebuilt-t4-001/"
CHUNK_HASHES = (
    "48965fdbd8175424c109e83fbac0c7e1c42bbf8bdbe802d556063d4ebf1d7bc8",
    "18ba14049d01318701fc86b025ae7a8d8c8ed64775e8195cfb8479a3e99984c1",
)
SOURCE_SHA = "e6a61f4bd6aa792c0bb74ee6a6e4c07259c72494211f40cb90402354a2ab2507"
INIT_SHA = "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"
RECIPE_SHA = "f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8"
STAGING_SHA = "3935dc0d359a0ffe503fc71e0521e500294b3bbbe75ae296634277f0b9aa9bec"
MODEL_HASHES = {
    "reference": "dfd68ac6d180c3adff2604a69ea5769421f8780f98774f831d08f21acce52002",
    "candidate": "e311400a4db6d912714166e839b136b49bbb5790d3d1d633d484961ac2d0a095",
}
HELPERS = {
    "pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "predict_pose_folder.py": "2942864e86303a67911826fc0dbfe547b4b4fc2ed3860e74c9d0210c635520a6",
    "evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "pose_checkpoint.py": "c1311307fd49b37e60f2939c9b084bc36690c1d13b6b31b9e4c51ea9353562d4",
}
REFERENCES = {
    "frozen666": ("data/references/sequence-refined-reference-2026-11-18/manifest.json", "0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96", 666),
    "historical663": ("data/references/visible-front-completion-2026-11-14/manifest.json", "bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1", 663),
    "r38": ("data/reviews/fronts-2026-11-26-r38/manifest.json", "0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5", 654),
}
COUNTS = {"reference": (137, 3784, 6837, 731, 13), "candidate": (138, 3827, 6923, 731, 13)}
VAL_COUNTS = (20, 664, 1227, 101, 3)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = Path(path).read_bytes()
    need(digest(data) == expected, f"Hash mismatch: {path}")
    return data


def obtain_archive(directory):
    """Download exact PR133 chunks and reassemble; preserve incompatible files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / NAME
    if target.exists():
        checked(target, ARCHIVE_SHA)
        return target
    parts = []
    for number, expected in enumerate(CHUNK_HASHES):
        name = f"{NAME}.part{number:02d}.zip"
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                data = response.read()
            need(digest(data) == expected, f"Downloaded chunk differs: {name}")
            path.write_bytes(data)
        data = checked(path, expected)
        with zipfile.ZipFile(io.BytesIO(data)) as wrapped:
            need(len(wrapped.infolist()) == 1 and wrapped.testzip() is None, "Invalid chunk wrap")
            parts.append(wrapped.read(wrapped.namelist()[0]))
    data = b"".join(parts)
    need(len(data) == ARCHIVE_BYTES and digest(data) == ARCHIVE_SHA, "Reassembled archive differs")
    target.write_bytes(data)
    return target


def corner_groups(gold, predictions, scores):
    matches = {x["gold_index"]: x["prediction_index"] for x in scores["matches"]}
    groups = {str(n): Counter(eligible=0, matched=0, within5=0, within10=0) for n in (0, 1, 2)}
    for index, item in enumerate(gold):
        visible = [j for j, point in enumerate(item["keypoints"]) if point[2] == 2]
        group = groups[str(len(visible))]
        group["eligible"] += 1
        if index not in matches:
            continue
        group["matched"] += 1
        if not visible:
            continue
        errors = []
        for j in visible:
            x, y, confidence = predictions[matches[index]]["keypoints"][j]
            gx, gy, _ = item["keypoints"][j]
            valid = x is not None and y is not None and 0 <= x < 1280 and 0 <= y < 720 and confidence >= 0.5
            errors.append(math.hypot(x - gx, y - gy) if valid else math.inf)
        for threshold in (5, 10):
            group[f"within{threshold}"] += int(max(errors) <= threshold)
    return groups


def audit(archive, root):
    """Return deterministic, compact evidence; imports only standard-library helpers."""
    root = Path(root).resolve()
    for name, expected in HELPERS.items():
        checked(root / "scripts" / name, expected)
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels
    from pose_scoring import score_image, aggregate_scores
    from predict_pose_folder import postclip_nms

    def parse(data, source):
        return parse_pose_labels(data.decode(), 1280, 720, source)

    def counts(groups):
        objects = sum(map(len, groups))
        visible = sum(p[2] == 2 for g in groups for row in g for p in row["keypoints"])
        return len(groups), objects, visible, 2 * objects - visible, sum(not g for g in groups)

    archive_data = checked(archive, ARCHIVE_SHA)
    need(len(archive_data) == ARCHIVE_BYTES, "Archive size mismatch")
    with zipfile.ZipFile(io.BytesIO(archive_data)) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) == 335 and z.testzip() is None, "Archive membership or CRC failure")
        need(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names), "Unsafe member path")
        inventory = json.loads(z.read("inventory.json"))
        need(set(inventory) == set(names) - {"inventory.json"} and len(inventory) == 334, "Inventory membership mismatch")
        for name, expected in inventory.items():
            data = z.read(name)
            need(len(data) == expected["bytes"] and digest(data) == expected["sha256"], f"Inventory differs: {name}")
        need(digest(z.read("source/rebuild_additional_front_pilot_r45.py")) == SOURCE_SHA, "Archived training source differs")
        need(digest(z.read("staging.json")) == STAGING_SHA, "Staging differs")
        staging = json.loads(z.read("staging.json"))
        need(staging["source"] == SOURCE and staging["initializer"] == INIT_SHA, "Staging provenance differs")
        need(staging["recipe"] == {"recipe_id": "reviewed-incremental-fronts-seed46-v1", "recipe_sha256": RECIPE_SHA}, "Recipe differs")
        original = json.loads(checked(root / "data/manifest.json", "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926"))
        overlay_path = root / "data/candidates/front-review-2026-11-26-r37/manifest.json"
        overlay = json.loads(checked(overlay_path, "ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3"))
        sources = {x["image_id"]: x for x in overlay["image_sources"]}
        originals = {x["id"]: x for x in original["records"]}
        roles = staging["roles"]
        need(roles["candidate"][:-1] == roles["reference"], "Shared treatment differs")
        extra = roles["candidate"][-1]
        need(extra == {"id": "frame_0383", "image": "images/train/frame_0383.jpg", "image_sha256": "cad520edcb4299210090da752b4bdc7bd8e857651cdd2351dfe58ac9a534cbd6", "label": "labels/train/frame_0383.txt", "label_sha256": "1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0", "split": "train"}, "Extra image differs")
        need({e["id"] for e in roles["reference"]} == set(originals), "Original membership differs")
        for e in roles["reference"]:
            old = originals[e["id"]]
            expected_label = sources[e["id"]]["effective_label"]["sha256"] if e["split"] == "train" else old["label_sha256"]
            need(e["split"] == old["split"] and e["image_sha256"] == old["image_sha256"] and e["label_sha256"] == expected_label, "Original provenance differs")
        receipts = {}
        for role, entries in roles.items():
            grouped = {split: [] for split in ("train", "val")}
            need(len(entries) == len({e["id"] for e in entries}), "Repeated image id")
            for e in entries:
                data = z.read(f"datasets/{role}/{e['label']}")
                need(digest(data) == e["label_sha256"], "Staged label differs")
                grouped[e["split"]].append(parse(data, e["id"]))
            need(counts(grouped["train"]) == COUNTS[role] and counts(grouped["val"]) == VAL_COUNTS, "Dataset counts differ")
            receipt = json.loads(z.read(f"{role}/receipt.json"))
            need(receipt == json.loads(z.read(f"{role}/progress.json")), "Final progress differs")
            need(receipt["source"] == SOURCE and receipt["staging_sha256"] == STAGING_SHA and receipt["role"] == role, "Receipt provenance differs")
            need(receipt["status"] == "complete" and receipt["epochs"] == list(range(1, 31)) and receipt["batches"] == 1050 and receipt["steps"] == 78 and receipt["images"] == COUNTS[role][0] * 30 and receipt["updated"] and not receipt["trial"], "Completion budget differs")
            need(set(receipt["artifacts"]) == {"fit/weights/last.pt", "fit/results.csv"}, "Receipt artifacts differ")
            for name, expected in receipt["artifacts"].items():
                need(digest(z.read(f"{role}/{name}")) == expected, "Receipt artifact hash differs")
            need(receipt["artifacts"]["fit/weights/last.pt"] == MODEL_HASHES[role], "Final checkpoint differs")
            settings = receipt["settings"]
            for key, value in {"amp": False, "batch": 4, "imgsz": 1280, "optimizer": "AdamW", "seed": 46, "epochs": 30, "resume": False, "device": "0"}.items():
                need(settings[key] == value, f"Training setting differs: {key}")
            rows = list(csv.DictReader(io.StringIO(z.read(f"{role}/fit/results.csv").decode())))
            need([int(x["epoch"]) for x in rows] == list(range(1, 31)), "CSV epochs differ")
            receipts[role] = receipt
        need(receipts["reference"]["runtime"] == receipts["candidate"]["runtime"], "A/B runtime differs")
        trial = json.loads(z.read("resource-trial/receipt.json"))
        need(trial["status"] == "trial-complete" and trial["trial"] and trial["role"] == "candidate" and trial["batches"] == trial["steps"] == 4 and trial["images"] == 16 and trial["updated"], "Trial receipt differs")
        need(trial["source"] == SOURCE and trial["staging_sha256"] == STAGING_SHA and trial["runtime"] == receipts["reference"]["runtime"], "Trial provenance differs")
        summary = json.loads(z.read("evaluation/summary.json"))
        predictions = json.loads(z.read("evaluation/predictions.json"))
        need(summary["source"] == SOURCE and summary["runtime"] == receipts["reference"]["runtime"], "Evaluation provenance differs")
        need(summary["inference"]["conf"] == .13 and summary["inference"]["iou"] == .7 and summary["inference"]["matching_iou"] == summary["inference"]["endpoint_conf"] == .5, "Evaluation thresholds differ")
        validation = {e["id"]: e for e in roles["reference"] if e["split"] == "val"}
        for role in ("reference", "candidate"):
            need(set(predictions[role]) == set(validation), "Prediction image membership differs")
            for record in predictions[role].values():
                kept, filtered = postclip_nms(record["raw"])
                need(kept == record["postclip"] and filtered == record["filter"], "Postclip replay differs")
        compact = {}
        for name, (relative, expected, count) in REFERENCES.items():
            path = root / relative
            manifest = json.loads(checked(path, expected))
            records = [e for e in manifest["files"] if e.get("split", "val") == "val"]
            need({e["image_id"] for e in records} == set(validation) and len(records) == 20, "Reference image membership differs")
            gold = {}
            for e in records:
                identity = e["image_id"]
                need(e.get("image_sha256", e.get("source_image_sha256")) == validation[identity]["image_sha256"], "Reference image hash differs")
                gold[identity] = parse(checked(path.parent / e["label_path"], e["label_sha256"]), identity)
            need(sum(map(len, gold.values())) == count, "Reference object count differs")
            need(summary["references"][name]["manifest_sha256"] == expected, "Evaluation reference hash differs")
            compact[name] = {"manifest_sha256": expected}
            for view in ("raw", "postclip"):
                compact[name][view] = {}
                for role in ("reference", "candidate"):
                    scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
                    for identity, labels in gold.items():
                        pred = predictions[role][identity][view]
                        scores[identity] = score_image(labels, pred, 1280, 720, .5, .5)
                        for key, values in corner_groups(labels, pred, scores[identity]).items():
                            groups[key].update(values)
                    result = {"detection": aggregate_scores(list(scores.values())), "corners_by_locatable_count": groups, "per_image": scores}
                    need(result == summary["references"][name][view][role], f"Score replay differs: {name}/{view}/{role}")
                    compact[name][view][role] = {
                        "detection": {k: v for k, v in result["detection"].items() if k != "endpoint_errors_px"},
                        "corners_by_locatable_count": groups,
                        "missed_rows": {i: [x + 1 for x in s["unmatched_gold_indices"]] for i, s in scores.items() if s["fn"]},
                        "empty_scene_predictions": {i: scores[i]["prediction_count"] for i, labels in gold.items() if not labels},
                    }
                delta = summary["references"][name][view]["identity_delta"]
                for identity in gold:
                    a, b = (set(compact[name][view][r]["missed_rows"].get(identity, [])) for r in ("reference", "candidate"))
                    need(delta[identity] == {"recovered_rows": sorted(a - b), "newly_missed_rows": sorted(b - a)}, "Miss identity delta differs")
                compact[name][view]["identity_delta"] = {i: d for i, d in delta.items() if any(d.values())}
        return {"schema": "baomai-r45-return-cpu-audit-v1", "status": "passed", "pr": 133, "pr_head": PR_HEAD,
                "source": SOURCE, "archive_sha256": ARCHIVE_SHA, "archive_bytes": ARCHIVE_BYTES,
                "verified_inventory_entries": 334, "replayed_image_role_filters": 40, "replayed_score_reports": 12,
                "scope": "Verified returned bytes, label lineage, completion receipts, source/runtime consistency and cached scoring. No fresh inference or training; checkpoint hashes do not independently prove prediction generation.",
                "fresh_inference": False, "fresh_training": False, "automatic_promotion": False,
                "model_sha256": MODEL_HASHES, "helper_sha256": HELPERS,
                "training_counts_fields": ["photos", "fronts", "visible_corners", "missing_corners", "empty_photos"],
                "training_counts": COUNTS, "validation_counts": VAL_COUNTS,
                "supplementary_train_images": {"reference": [], "candidate": ["frame_0383"]},
                "completed_epochs": 30, "batches_per_arm": 1050, "optimizer_steps_per_arm": 78,
                "image_presentations": {"reference": 4110, "candidate": 4140}, "reports": compact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--download-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive or obtain_archive(args.download_dir)
    result = audit(archive, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("PR133 CPU audit passed: 334 inventory entries, 40 filters, 12 score reports.")
    print(args.output)


if __name__ == "__main__":
    main()
