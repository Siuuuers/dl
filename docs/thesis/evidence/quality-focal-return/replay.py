#!/usr/bin/env python3
"""Replay PR157's exact frozen scorer and derive compact, deterministic evidence.

Read an already extracted evidence pack. No inference, training or threshold search.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
SOURCE = "acaf5d4b4043583fdd81f5ea848b3a6a68ada728"
RETURN = "f971e880e8aff2859ce4d8ae2923efea826eddb4"
ARCHIVE_SHA = "cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def quantiles(values):
    values = sorted(values)
    result = {"count": len(values), "mean": math.fsum(values) / len(values) if values else None}
    for name, q in (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1)):
        pos = q * (len(values) - 1)
        low, high = math.floor(pos), math.ceil(pos)
        result[name] = values[low] + (values[high] - values[low]) * (pos - low) if values else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack, out = args.pack.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert sha(args.archive.read_bytes()) == ARCHIVE_SHA
    inventory = read(pack / "inventory.json")
    for name, expected in inventory.items():
        data = (pack / name).read_bytes()
        assert len(data) == expected["bytes"] and sha(data) == expected["sha256"], name
    actual = {p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file()}
    assert actual == set(inventory) | {"inventory.json"}
    tree = {}
    for line in subprocess.check_output(["git", "ls-tree", "-r", SOURCE], cwd=args.repo, text=True).splitlines():
        metadata, name = line.split("\t")
        tree[name] = metadata.split()[2]
    for file in sorted((pack / "source/scripts").glob("*.py")):
        data = file.read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == tree["scripts/" + file.name]
    sys.path.insert(0, str(pack / "source/scripts"))
    import run_training_label_comparison_r61 as scoring
    from run_quality_focal_r61 import detection_ranking
    from quality_focal_r61 import validate_state
    from run_corner_average_r58 import clipped_prediction_indices
    from predict_pose_folder import postclip_nms

    plan, summary = read(pack / "staging.json"), read(pack / "evaluation/summary.json")
    details = json.loads(gzip.decompress((pack / "evaluation/details.json.gz").read_bytes()))
    assert plan["source"] == summary["source"] == SOURCE
    assert plan["roles"]["reference"] == plan["roles"]["candidate"]
    staging_sha = sha((pack / "staging.json").read_bytes())
    assert summary["staging_sha256"] == staging_sha
    refs = {}
    for name, spec in plan["references"].items():
        assert sha((pack / "references" / name / "manifest.json").read_bytes()) == spec["manifest_sha256"]
        refs[name] = {}
        for identity, specrow in spec["labels"].items():
            data = (pack / "references" / name / "labels" / (identity + ".txt")).read_bytes()
            assert sha(data) == specrow["sha256"]
            assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == tree[specrow["path"]]
            refs[name][identity] = scoring.parse_pose_labels(data.decode(), 1280, 720, identity)
        assert sum(map(len, refs[name].values())) == spec["objects"]
    measurement, replayed = scoring.score_native(refs, details["native-predictions"], baseline="reference")
    assert measurement == summary["measurement"] and replayed == details
    ranking = detection_ranking(refs, details["native-predictions"])
    assert ranking == summary["detection_ranking"]

    receipts = {role: read(pack / role / "receipt.json") for role in ("reference", "candidate")}
    a, b = receipts.values()
    assert a["runtime"] == b["runtime"] == summary["runtime"]
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["optimizer_batch_indices"] == b["optimizer_batch_indices"]
    changed_settings = sorted(k for k in a["settings"] if a["settings"][k] != b["settings"][k])
    assert changed_settings == ["data", "project", "save_dir"]
    batch_lists = {}
    training, epoch_rows = {}, []
    for role, receipt in receipts.items():
        assert receipt["status"] == "complete" and not receipt["trial"] and receipt["updated"]
        budget = plan["budgets"][role]
        assert all(receipt[key] == budget[key] for key in ("batches", "images", "steps", "optimizer_batch_indices"))
        assert receipt["epochs"] == list(range(1, 31))
        assert receipt["source"] == SOURCE and receipt["staging_sha256"] == staging_sha
        assert receipt["artifacts"]["fit/weights/last.pt"] == summary["model_sha256"][role]
        assert sha((pack / role / "fit/results.csv").read_bytes()) == receipt["artifacts"]["fit/results.csv"]
        for row in plan["roles"][role]:
            assert sha((pack / "datasets" / role / row["label"]).read_bytes()) == row["label_sha256"]
        ids = sorted(r["id"] for r in plan["roles"][role] if r["split"] == "train")
        state = receipt["quality_focal_controller"]
        validate_state(state, ids, 0 if role == "reference" else 2,
                       role=role, protocol_sha256=staging_sha, complete=True)
        batch_lists[role] = [batch for epoch in state["epochs"] for batch in epoch["batches"]]
        training[role] = {}
        for branch in ("one2many", "one2one"):
            def aggregate(batches):
                values = [batch["branches"][branch] for batch in batches]
                total = lambda field: math.fsum(row[field] for row in values)
                result = {"batches": len(batches), "partial_targets": int(total("partial_targets")),
                          "unassigned_partial_targets": int(total("unassigned_partial_targets")),
                          "unassigned_full_targets": int(total("unassigned_full_targets")),
                          "foreground_anchors": int(total("foreground_anchors")),
                          "foreground_zero_target_terms": int(total("foreground_anchors") - total("positive_classification_terms")),
                          "partial_foreground_anchors": int(total("partial_foreground_anchors")),
                          "partial_mean_q_per_foreground_anchor": total("partial_target_score_sum") / total("partial_foreground_anchors"),
                          "full_mean_q_per_foreground_anchor": (total("original_target_score_sum") - total("partial_target_score_sum")) / (total("foreground_anchors") - total("partial_foreground_anchors"))}
                for objective in ("native", "focal"):
                    contributions = {group: math.fsum(
                        batch["branches"][branch][objective + "_" + group + "_bce_numerator"] /
                        batch["branches"][branch]["normalizer"] * batch["branches"][branch]["classification_gain"] *
                        batch["native_branch_schedule"][branch] for batch in batches)
                        for group in ("foreground", "background")}
                    result[objective + "_normalized_weighted_sum"] = sum(contributions.values())
                    result[objective + "_background_fraction"] = contributions["background"] / sum(contributions.values())
                    result[objective + "_foreground_sum"] = contributions["foreground"]
                    result[objective + "_background_sum"] = contributions["background"]
                return result
            training[role][branch] = aggregate(batch_lists[role])
            training[role][branch]["assignment_totals"] = state["assignment_totals"][branch]
            for epoch in state["epochs"]:
                epoch_rows.append(dict(role=role, branch=branch, epoch=epoch["epoch"], **aggregate(epoch["batches"])))
    assert [v["image_ids"] for v in batch_lists["reference"]] == [v["image_ids"] for v in batch_lists["candidate"]]
    for old, new in zip(batch_lists["reference"], batch_lists["candidate"]):
        for branch in ("one2many", "one2one"):
            for name in ("targets", "partial_targets", "full_targets"):
                assert old["branches"][branch][name] == new["branches"][branch][name]

    metrics, paired, distributions, support = [], {}, {}, {}
    for ref, arms in measurement["references"].items():
        paired[ref] = {"corner_changes": measurement["paired_changes"][ref]["candidate"]}
        ca, cb = (details["cases"][ref][role]["two_view"] for role in ("reference", "candidate"))
        paired[ref]["detection_gains"], paired[ref]["detection_losses"] = [], []
        for old, new in zip(ca, cb):
            identity = old["image"] + "/R" + str(old["reference_index"] + 1).zfill(3)
            known = sum(k[2] > 0 for k in refs[ref][old["image"]][old["reference_index"]]["keypoints"])
            if old["detected"] != new["detected"]:
                paired[ref]["detection_gains" if new["detected"] else "detection_losses"].append(
                    {"id": identity, "known_corners": known, "control": old, "candidate": new})
        for role, views in arms.items():
            for view, value in views.items():
                detection, pairs = value["detection"], value["pairs"]
                cases = details["cases"][ref][role][view]
                errors = [v["maximum_corner_error_px"] for v in cases if v["complete_reference"] and v["maximum_corner_error_px"] is not None]
                metrics.append(dict(reference=ref, role=role, view=view, missed=detection["fn"],
                                    extras=detection["fp"], duplicates=detection["duplicate_fp"],
                                    other_extras=detection["other_fp"], matched=detection["tp"],
                                    predictions=detection["prediction_count"], precision=detection["tp"]/detection["prediction_count"],
                                    recall=detection["tp"]/detection["gold_count"],
                                    eligible_pairs=pairs["eligible"], within5=pairs["within5"], within10=pairs["within10"],
                                    mean_maximum_corner_error=math.fsum(errors)/len(errors)))
        distributions[ref], support[ref] = {}, {}
        for role in ("supplied", "reference", "candidate"):
            distributions[ref][role] = {}
            for label, values in (("fixed_cutoff", details["predictions"][role]["original"]),
                                  ("low_floor", {im: postclip_nms([row for row in rows if row["confidence"] > .001])[0]
                                                 for im, rows in details["native-predictions"][role]["original"].items()})):
                groups = {k: [] for k in ("matched_complete", "matched_incomplete", "duplicate_extra", "other_extra")}
                invalid, empty = 0, 0
                from run_corner_average_r58 import score_image_counting_clipped
                for im, preds in values.items():
                    scored = score_image_counting_clipped(refs[ref][im], preds, 1280, 720, .5, .5)
                    matches = {v["prediction_index"]: v["gold_index"] for v in scored["matches"]}
                    for index, prediction in enumerate(preds):
                        if index in matches:
                            known = sum(k[2] > 0 for k in refs[ref][im][matches[index]]["keypoints"])
                            group = "matched_complete" if known == 2 else "matched_incomplete"
                        else:
                            group = "duplicate_extra" if index in scored["duplicate_prediction_indices"] else "other_extra"
                        groups[group].append(prediction["confidence"])
                    invalid += len(clipped_prediction_indices(preds))
                    if not refs[ref][im]:
                        empty += len(preds)
                distributions[ref][role][label] = {"groups": {k: quantiles(v) for k, v in groups.items()},
                                                   "clipped_nonpositive_outputs": invalid, "empty_scene_outputs": empty}
            counts = Counter()
            for im, preds in details["predictions"][role]["two_view"].items():
                scored = details["per-image"][ref][role]["two_view"][im]
                match = {v["prediction_index"]: v["gold_index"] for v in scored["matches"]}
                cases = {v["reference_index"]: v for v in details["cases"][ref][role]["two_view"] if v["image"] == im}
                for index, pred in enumerate(preds):
                    valid = all(k[2] >= .5 and k[0] is not None and k[1] is not None and math.isfinite(k[0]) and math.isfinite(k[1]) and 0 <= k[0] < 1280 and 0 <= k[1] < 720 for k in pred["keypoints"])
                    if not valid:
                        counts["not_returned_as_complete_pair"] += 1
                    elif index not in match:
                        counts["emitted_unmatched"] += 1
                    elif not cases[match[index]]["complete_reference"]:
                        counts["emitted_incomplete_reference"] += 1
                    elif cases[match[index]]["within5"]:
                        counts["emitted_supported_within5"] += 1
                    else:
                        counts["emitted_complete_over5"] += 1
            support[ref][role] = dict(counts)

    rank_compact = {ref: {role: {k: v for k, v in obj.items() if k not in ("curve", "interpolated_precision")}
                          for role, obj in arms.items()} for ref, arms in ranking.items()}
    verification = {"schema": "quality-focal-return-independent-replay-r65-v1", "source_commit": SOURCE,
                    "return_commit": RETURN, "archive_sha256": ARCHIVE_SHA, "inventory_members_verified": len(inventory),
                    "source_files_verified_against_git": len(plan["source_files_sha256"]),
                    "fixed_metric_groups_exact": 18, "ap50_rankings_exact": 9,
                    "all_reconstructed_details_exact": True, "all_reference_labels_match_git": True,
                    "paired_training_receipts_validated": True, "identical_initial_state_sha256": a["initial_state_sha256"],
                    "training_settings_difference_keys": changed_settings,
                    "identical_image_order_and_per_batch_target_counts": True,
                    "model_sha256_recorded": summary["model_sha256"],
                    "checkpoint_bytes_independently_verified_by_this_script": False,
                    "fresh_inference": False, "training": False, "development_only": True,
                    "runtime_recorded": summary["runtime"], "input_hashes": {name: sha((pack / name).read_bytes()) for name in
                        ("staging.json", "evaluation/summary.json", "evaluation/details.json.gz", "reference/receipt.json", "candidate/receipt.json")}}
    write(out / "verification.json", verification)
    write(out / "paired-cases.json", paired)
    write(out / "ranking.json", rank_compact)
    write(out / "score-distributions.json", distributions)
    write(out / "emitted-pair-support.json", support)
    write(out / "training-diagnostics.json", {"roles": training, "notes": [
        "q means are weighted by foreground anchors, not by targets or photographs.",
        "Normalized sums divide each batch numerator by its original target-score normalizer, multiply by classification gain and native branch weight, and sum; before batch-size factor.",
        "Native values in candidate traces are BCE evaluated on candidate predictions, not a separate control run.",
        "Loss-value shares are not gradient norms and do not identify a training cause.",
        "Partial combines zero and one known corner after augmentation; p and individual q distributions are not recorded."]})
    for name, rows in (("metrics.csv", metrics), ("training-by-epoch.csv", epoch_rows)):
        with (out / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": "passed", "metric_groups": 18, "ap_rankings": 9,
                      "reviewed_detection_gains": len(paired["r50"]["detection_gains"]),
                      "reviewed_detection_losses": len(paired["r50"]["detection_losses"])}))


if __name__ == "__main__":
    main()
