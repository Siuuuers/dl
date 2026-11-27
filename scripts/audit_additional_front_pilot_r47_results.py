#!/usr/bin/env python3
"""Verify the PR137 R47 return and replay cached scores on CPU."""
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile

from audit_additional_front_pilot_r45_results import (
    REFERENCES, HELPERS, INIT_SHA, RECIPE_SHA, VAL_COUNTS,
    checked, corner_groups, digest, need,
)

PR_HEAD = "dcb43d7ad061fc3a43f1392398bb8794d9d7168d"
SOURCE = "96b05f0d2149ab71c9211c81f35be67f79344b81"
INPUT_SOURCE = "ea4ac955f2dfaf1add24c243599715f802eb41a7"
NAME = "baomai-additional-front-pilot-r47-results.zip"
ARCHIVE_SHA = "fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023"
ARCHIVE_BYTES = 42395650
BASE_URL = f"https://raw.githubusercontent.com/Siuuuers/dl/{PR_HEAD}/results/t4/baomai-additional-front-pilot-r47-t4-001/"
CHUNK_HASHES = (
    "b03259100c57f0e5bcff785c0019676286b0cb2e68986a662e1239ef05dc17ce",
    "7d9e01ac7d071ebec844ff142c8055aee41f18122a56bf55f3c2919efbcf558a",
    "ffe74b25373eb3283e82b004b142643a9876a0d22c3b8ae5f816007825ab061a",
)
STAGING_SHA = "6440e7e75976ac7b8f9bd94651c8c9b4a9443a972f4bc2d68f8573871954ae29"
RELEASE_SHA = "1d4f3ac4cb3efc97c6e05ef392b83654d4eed2039f3f167340a425ff837641a7"
RUNNER_SHA = "8475e28cb113a59893f44c8420fdefeac722275f4a71f7b63a0dfbd93fe0879e"
REVIEW = "data/reviews/additional-fronts-2026-11-27-r47"
MODEL_HASHES = {
    "reference": "605f067d6fc43ed263f956190132371921228ac93efb0c169f22ff14941c0308",
    "candidate": "222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9",
}
COUNTS = {"reference": (138, 3827, 6923, 731, 13), "candidate": (142, 3999, 7259, 739, 13)}
EXTRAS = ["frame_0146", "frame_0160", "frame_0178", "frame_0371", "frame_0383"]


def obtain_archive(directory):
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
            need(digest(data) == expected, "Downloaded chunk differs: " + name)
            path.write_bytes(data)
        with zipfile.ZipFile(io.BytesIO(checked(path, expected))) as wrapped:
            need(wrapped.namelist() == [name[:-4]] and wrapped.testzip() is None, "Invalid chunk wrap")
            parts.append(wrapped.read(name[:-4]))
    data = b"".join(parts)
    need(len(data) == ARCHIVE_BYTES and digest(data) == ARCHIVE_SHA, "Reassembled archive differs")
    target.write_bytes(data)
    return target


def audit(archive, root):
    root = Path(root).resolve()
    for name, expected in HELPERS.items():
        checked(root / "scripts" / name, expected)
    checked(root / "scripts/run_additional_front_pilot_r47.py", RUNNER_SHA)
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import parse_pose_labels, INFERENCE
    from pose_scoring import score_image, aggregate_scores
    from predict_pose_folder import postclip_nms
    from pose_checkpoint import check_training_settings
    from partial_box_loss_r31 import NATIVE_SOURCE_HASHES
    from run_additional_front_pilot_r47 import expected_budget, local_source_files

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
        need(len(names) == len(set(names)) == 500 and z.testzip() is None, "Archive membership or CRC failure")
        need(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names), "Unsafe member path")
        inventory = json.loads(z.read("inventory.json"))
        need(set(inventory) == set(names) - {"inventory.json"} and len(inventory) == 499, "Inventory membership differs")
        for name, expected in inventory.items():
            data = z.read(name)
            need(expected == dict(bytes=len(data), sha256=digest(data)), "Inventory differs: " + name)
        need(digest(z.read("staging.json")) == STAGING_SHA, "Staging differs")
        staging = json.loads(z.read("staging.json"))
        need(staging["source"] == SOURCE and staging["initializer"] == INIT_SHA, "Staging provenance differs")
        need(staging["recipe"] == dict(recipe_id="reviewed-incremental-fronts-seed46-v1", recipe_sha256=RECIPE_SHA), "Recipe differs")
        need(staging["source_files_sha256"] == local_source_files(root), "Local source closure differs")
        for relative, expected in staging["source_files_sha256"].items():
            need(digest(z.read("source/" + relative)) == expected, "Archived source differs: " + relative)
        release_data = checked(root / REVIEW / "release-status.json", RELEASE_SHA)
        need(z.read("release-status.json") == release_data and staging["release_sha256"] == RELEASE_SHA, "Release differs")
        release = json.loads(release_data)
        need(sorted(release["annotation_complete_ids"]) == EXTRAS, "Whole-photo eligibility differs")
        need(digest(z.read("review/manifest.json")) == release["generated_manifest_sha256"] == staging["review_manifest_sha256"], "Review manifest differs")
        for name, key in (("audit-ledger.json", "audit_sha256"), ("outlines.json", "outlines_sha256")):
            need(digest(z.read("review/" + name)) == release[key], "Review evidence differs")
        frames = {f["image_id"]: f for f in json.loads(z.read("review/manifest.json"))["frames"]}
        for identity in EXTRAS:
            frame = frames[identity]
            need(frame["annotation_complete"] and frame["eligible_for_candidate_export"] and frame["training_split"] == "unassigned", "Supplementary photo held: " + identity)
            need(not any(row["unresolved_fields"] for row in frame["rows"] + frame["additions"]), "Unresolved exported front")
            relative = "complete-candidates/labels/" + identity + ".txt"
            need(digest(z.read("review/" + relative)) == release["files"][relative], "Exported label differs")
        original_data = checked(root / "data/manifest.json", "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926")
        overlay_path = root / "data/candidates/front-review-2026-11-26-r37/manifest.json"
        overlay_data = checked(overlay_path, "ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3")
        need(z.read("inputs/source-manifest.json") == original_data and z.read("inputs/r37-overlay-manifest.json") == overlay_data, "Original lineage differs")
        originals = {r["id"]: r for r in json.loads(original_data)["records"]}
        sources = {r["image_id"]: r for r in json.loads(overlay_data)["image_sources"]}
        roles = staging["roles"]
        need(set(roles) == {"reference", "candidate"}, "Unexpected roles")
        budgets = {role: expected_budget(COUNTS[role][0]) for role in roles}
        need(staging["budgets"] == budgets, "Native optimizer schedule differs")
        need(staging["counts"] == {r: dict(train=list(COUNTS[r]), val=list(VAL_COUNTS)) for r in roles}, "Declared inventory differs")
        treatment = staging["treatment"]
        need(treatment["reference_extra_ids"] == ["frame_0383"] and treatment["candidate_extra_ids"] == EXTRAS and
             treatment["added_photo_ids"] == EXTRAS[:-1] and not treatment["shared_extra_label_changes"] and
             treatment["whole_photo_gate"] and treatment["equal_epochs"] and not treatment["equal_optimizer_updates"] and
             treatment["extra_exposure_is_part_of_treatment"] and not treatment["isolated_data_from_compute"] and
             not treatment["automatic_promotion"], "Treatment differs")
        need(set(treatment["held_photo_ids"]) == set(release["held_image_ids"]), "Held photograph set differs")
        receipts = {}
        for role, entries in roles.items():
            grouped = {split: [] for split in ("train", "val")}
            by_id = {entry["id"]: entry for entry in entries}
            extras = ["frame_0383"] if role == "reference" else EXTRAS
            need(len(entries) == len(by_id) and set(by_id) == set(originals) | set(extras), "Role membership differs")
            need(set(by_id).isdisjoint(release["held_image_ids"]), "Held photograph trained")
            for e in entries:
                identity = e["id"]
                data = z.read(f"datasets/{role}/{e['label']}")
                need(digest(data) == e["label_sha256"], "Staged label differs")
                if identity in originals:
                    old = originals[identity]
                    expected_label = sources[identity]["effective_label"]["sha256"] if old["split"] == "train" else old["label_sha256"]
                    need(e["split"] == old["split"] and e["image_sha256"] == old["image_sha256"] and e["label_sha256"] == expected_label, "Original labels/split differ")
                else:
                    need(e["split"] == "train" and e["image_sha256"] == frames[identity]["source_image_sha256"], "Supplementary image differs")
                    need(data == z.read("review/complete-candidates/labels/" + identity + ".txt"), "Supplementary training labels differ")
                grouped[e["split"]].append(parse(data, identity))
            need(counts(grouped["train"]) == COUNTS[role] and counts(grouped["val"]) == VAL_COUNTS, "Actual label inventory differs")
            receipt = json.loads(z.read(role + "/receipt.json"))
            need(receipt == json.loads(z.read(role + "/progress.json")), "Final progress differs")
            need(receipt["source"] == SOURCE and receipt["staging_sha256"] == STAGING_SHA and receipt["role"] == role, "Receipt identity differs")
            need(receipt["status"] == "complete" and receipt["epochs"] == list(range(1, 31)) and receipt["updated"] and not receipt["trial"], "Run incomplete")
            need(all(receipt[k] == budgets[role][k] for k in ("batches", "images", "steps", "optimizer_batch_indices")), "Measured exposure differs")
            need(set(receipt["artifacts"]) == {"fit/weights/last.pt", "fit/results.csv"}, "Receipt artifact membership differs")
            for name, expected in receipt["artifacts"].items():
                need(digest(z.read(role + "/" + name)) == expected, "Completed artifact differs")
            need(receipt["artifacts"]["fit/weights/last.pt"] == MODEL_HASHES[role], "Checkpoint differs")
            settings = receipt["settings"]
            check_training_settings(settings, staging["recipe"]["recipe_id"], model=settings["model"], data=settings["data"], device="0", project=settings["project"])
            rows = list(csv.DictReader(io.StringIO(z.read(role + "/fit/results.csv").decode())))
            need([int(r["epoch"]) for r in rows] == list(range(1, 31)), "CSV epochs differ")
            need(all(math.isfinite(float(value)) for row in rows for value in row.values() if value.strip()), "Nonfinite CSV value")
            receipts[role] = receipt
        old_extra = z.read("inputs/pr133-frame0383-labels.txt")
        need(digest(old_extra) == "1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0", "Prior frame0383 label differs")
        need(all(z.read(f"datasets/{role}/labels/train/frame_0383.txt") == old_extra for role in roles), "Shared frame0383 changed")
        runtime = receipts["reference"]["runtime"]
        need(runtime == receipts["candidate"]["runtime"] and runtime["gpu"] == "Tesla T4" and runtime["native"] == NATIVE_SOURCE_HASHES and runtime["packages"]["ultralytics"] == "8.4.182", "Runtime/native source differs")
        trial = json.loads(z.read("resource-trial/receipt.json"))
        need(trial == json.loads(z.read("resource-trial/progress.json")), "Trial progress differs")
        need(trial["status"] == "trial-complete" and trial["trial"] and trial["role"] == "candidate" and trial["batches"] == trial["steps"] == 4 and trial["images"] == 16 and trial["optimizer_batch_indices"] == [0, 1, 2, 3] and trial["updated"], "Trial completion differs")
        need(trial["source"] == SOURCE and trial["staging_sha256"] == STAGING_SHA and trial["runtime"] == runtime, "Trial provenance differs")
        summary = json.loads(z.read("evaluation/summary.json"))
        predictions = json.loads(z.read("evaluation/predictions.json"))
        need(summary["source"] == SOURCE and summary["staging_sha256"] == STAGING_SHA and summary["runtime"] == runtime and summary["model_sha256"] == MODEL_HASHES, "Evaluation provenance differs")
        need(summary["budgets"] == budgets and summary["treatment"] == treatment and summary["experiment"] == "additional-front-pilot-r47", "Evaluation treatment differs")
        need(summary["inference"] == {**INFERENCE, "conf": .13, "matching_iou": .5, "endpoint_conf": .5}, "Inference settings differ")
        validation = {e["id"]: e for e in roles["reference"] if e["split"] == "val"}
        need(set(predictions) == set(roles) and set(summary["references"]) == set(REFERENCES), "Evaluation membership differs")
        for role in roles:
            need(set(predictions[role]) == set(validation), "Prediction photograph membership differs")
            for record in predictions[role].values():
                kept, filtered = postclip_nms(record["raw"])
                need(kept == record["postclip"] and filtered == record["filter"], "Postclip replay differs")
        compact = {}
        for name, (relative, expected, object_count) in REFERENCES.items():
            path = root / relative
            manifest_data = checked(path, expected)
            need(z.read("references/" + name + "/manifest.json") == manifest_data, "Archived reference manifest differs")
            records = [e for e in json.loads(manifest_data)["files"] if e.get("split", "val") == "val"]
            need(len(records) == 20 and {e["image_id"] for e in records} == set(validation), "Reference photograph membership differs")
            gold = {}
            for e in records:
                identity = e["image_id"]
                need(e.get("image_sha256", e.get("source_image_sha256")) == validation[identity]["image_sha256"], "Reference image differs")
                data = checked(path.parent / e["label_path"], e["label_sha256"])
                need(z.read("references/" + name + "/" + e["label_path"]) == data, "Archived reference label differs")
                gold[identity] = parse(data, identity)
            need(sum(map(len, gold.values())) == object_count, "Reference front count differs")
            need(summary["references"][name]["manifest_sha256"] == expected, "Scored reference differs")
            compact[name] = dict(manifest_sha256=expected)
            for view in ("raw", "postclip"):
                compact[name][view] = {}
                for role in roles:
                    scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
                    for identity, targets in gold.items():
                        pred = predictions[role][identity][view]
                        scores[identity] = score_image(targets, pred, 1280, 720, .5, .5)
                        for key, values in corner_groups(targets, pred, scores[identity]).items():
                            groups[key].update(values)
                    result = dict(detection=aggregate_scores(list(scores.values())), corners_by_locatable_count=groups, per_image=scores)
                    need(result == summary["references"][name][view][role], f"Score replay differs: {name}/{view}/{role}")
                    compact[name][view][role] = dict(
                        detection={k: v for k, v in result["detection"].items() if k != "endpoint_errors_px"},
                        corners_by_locatable_count=groups,
                        missed_rows={i: [n + 1 for n in s["unmatched_gold_indices"]] for i, s in scores.items() if s["fn"]},
                        empty_scene_predictions={i: scores[i]["prediction_count"] for i, targets in gold.items() if not targets})
                delta = summary["references"][name][view]["identity_delta"]
                for identity in gold:
                    a, b = (set(compact[name][view][role]["missed_rows"].get(identity, [])) for role in ("reference", "candidate"))
                    need(delta[identity] == dict(recovered_rows=sorted(a - b), newly_missed_rows=sorted(b - a)), "Miss identity delta differs")
                compact[name][view]["identity_delta"] = {i: d for i, d in delta.items() if any(d.values())}
        return dict(schema="baomai-r47-return-cpu-audit-v1", status="passed", pr=137, pr_head=PR_HEAD,
                    source=SOURCE, archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES,
                    verified_inventory_entries=499, verified_source_files=len(staging["source_files_sha256"]),
                    replayed_image_role_filters=40, replayed_score_reports=12,
                    source_files_sha256=staging["source_files_sha256"], model_sha256=MODEL_HASHES,
                    training_counts_fields=["photos", "fronts", "locatable_corner_slots", "unavailable_corner_slots", "empty_photos"],
                    training_counts=COUNTS, validation_counts=VAL_COUNTS, budgets=budgets, treatment=treatment,
                    runtime=runtime, completed_epochs=30, reports=compact,
                    fresh_inference=False, fresh_training=False, automatic_promotion=False,
                    scope="Return integrity, exact source/label lineage, measured completion receipts and saved-score replay. Archived code checks actual loader coordinates and epoch exposure during the reported run. This CPU audit does not rerun the loader/training or independently regenerate predictions from checkpoints.")


def compare_prior_predictions(prior_archive, current_archive):
    """Bind old PR133 candidate cue numbers to the exact PR137 reference cache."""
    from audit_additional_front_pilot_r45_results import ARCHIVE_SHA as prior_sha
    with zipfile.ZipFile(io.BytesIO(checked(prior_archive, prior_sha))) as old:
        prior = json.loads(old.read("evaluation/predictions.json"))["candidate"]
    with zipfile.ZipFile(io.BytesIO(checked(current_archive, ARCHIVE_SHA))) as new:
        current = json.loads(new.read("evaluation/predictions.json"))["reference"]
    need(prior == current and len(current) == 20, "Prior candidate / current reference predictions differ")
    need(all(set(record) == {"raw", "postclip", "filter"} for record in current.values()), "Prediction views differ")
    canonical = (json.dumps(current, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return dict(schema="baomai-r47-prior-reference-identity-v1", prior_pr=133, prior_arm="candidate",
                current_pr=137, current_arm="reference", prior_archive_sha256=prior_sha,
                current_archive_sha256=ARCHIVE_SHA, photographs=20, views=["raw", "postclip", "filter"],
                same_prediction_values=True, canonical_prediction_sha256=digest(canonical),
                canonicalization="UTF-8 json.dumps(sort_keys=True,separators=(',',':'),allow_nan=False) plus LF",
                note="Exact parsed JSON values agree for every photograph and view. Checkpoint files have different hashes; no claim of byte-identical checkpoints is made.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--download-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-archive", type=Path, help="Optional exact PR133 archive for old cue identity comparison")
    parser.add_argument("--identity-output", type=Path, help="Write the optional PR133-candidate / PR137-reference comparison")
    args = parser.parse_args()
    if bool(args.prior_archive) != bool(args.identity_output):
        parser.error("--prior-archive and --identity-output must be supplied together")
    archive = args.archive or obtain_archive(args.download_dir)
    result = audit(archive, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.prior_archive:
        identity = compare_prior_predictions(args.prior_archive, archive)
        args.identity_output.parent.mkdir(parents=True, exist_ok=True)
        args.identity_output.write_text(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print("Exact PR133 candidate / PR137 reference prediction identity passed for all 20 photographs.")
    print("PR137 CPU audit passed: 499 inventory entries, 81 source files, 40 filters, 12 score reports.")
    print(args.output)


if __name__ == "__main__":
    main()
