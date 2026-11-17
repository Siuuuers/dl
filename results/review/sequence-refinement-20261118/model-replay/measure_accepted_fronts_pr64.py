#!/usr/bin/env python3
"""Measure the returned PR64 models against PR35 using frozen baomai references.

No training or model selection is performed. Use the repository's original scoring
and reference-verification code. All predictions must come from the pinned evaluator.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def inspect_archive(path, expected_sha):
    path = Path(path)
    assert digest(path.read_bytes()) == expected_sha, "PR64 archive checksum differs"
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, "Archive CRC failure"
        index = json.loads(archive.read("gpu-results-manifest.json"))
        assert index["kind"] == "baomai-paired-training-results"
        members = {x["path"]: x for x in index["files"]}
        assert set(archive.namelist()) == {"gpu-results-manifest.json", *members}
        for name, descriptor in members.items():
            payload = archive.read(name)
            assert len(payload) == descriptor["size_bytes"] and digest(payload) == descriptor["sha256"], name
        result = {"archive_sha256": expected_sha, "size_bytes": path.stat().st_size,
                  "source": index["source"], "runtime": index["runtime"], "models": {}}
        for arm in ("reference", "candidate"):
            receipt = json.loads(archive.read(f"{arm}/training-receipt.json"))
            assert receipt["status"] == "complete" and receipt["arm"] == arm
            assert receipt["completed_epochs"] == list(range(1,31))
            assert receipt["optimizer_parameter_change_observed"] is True
            assert receipt["validation_executed"] is True
            assert receipt["execution"]["exit_code"] == 0
            rows = list(csv.DictReader(io.StringIO(archive.read(f"{arm}/fit/results.csv").decode())))
            assert len(rows) == 30 and [int(row["epoch"]) for row in rows] == list(range(1,31))
            numeric = {k: [float(row[k]) for row in rows] for k in rows[0] if k != "epoch"}
            assert all(math.isfinite(v) for values in numeric.values() for v in values)
            result["models"][arm] = {
                "checkpoint_sha256": index["arms"][arm]["checkpoint_identity"]["checkpoint_sha256"],
                "completed_epochs": len(rows),
                "optimizer_steps": receipt["optimizer_steps"],
                "last_epoch": {k: v[-1] for k,v in numeric.items()},
                "last_five_mean": {k: sum(v[-5:])/5 for k,v in numeric.items()}}
    return result


def audit(args):
    report = inspect_archive(args.archive, args.sha256)
    save(args.output, report)
    for name, arm in report["models"].items():
        print("TRAINING", name, "epochs", arm["completed_epochs"],
              "optimizer steps", arm["optimizer_steps"],
              "checkpoint SHA256", arm["checkpoint_sha256"])
        for column, value in arm["last_five_mean"].items():
            if ("loss" in column) and ("val/" in column or "train/" in column):
                print(" ", column, "last5 mean", round(value,6))
    print("Training losses are NOT cassette-detection accuracy.")


def case_success(gold, predicted, width, height, radius):
    from pose_scoring import score_image
    score = score_image(gold, predicted, width, height, 0.5, 0.5)
    successes = set()
    for pair in score["matches"]:
        i = pair["gold_index"]
        gt = gold[i]["keypoints"]
        if not all(p[2] == 2 for p in gt):
            continue
        pred = predicted[pair["prediction_index"]]["keypoints"]
        ok = True
        for (x,y,_), (px,py,conf) in zip(gt,pred):
            ok = ok and conf >= .5 and px is not None and py is not None
            if ok:
                ok = (0 <= px < width and 0 <= py < height and math.hypot(px-x,py-y) <= radius)
        if ok:
            successes.add(i+1)
    return successes


def score_all(args):
    from evaluate_pose import preflight_dataset, _read_replay
    from pose_checkpoint import select_checkpoint
    from rescore_reference_amendment import load_api, verify_amendment
    from pose_scoring import score_image, aggregate_scores
    from diagnose_pose_quality import quality_for_image, aggregate_quality
    root = Path(args.root).resolve()
    preflight = preflight_dataset(root, require_weights=False)
    _, api = load_api(root)
    references = {}
    for name, rel, expected in (
        ("primary_2026_11_12", "data/references/visible-front-adjudication-2026-11-12",
         "75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e"),
        ("diagnostic_2026_11_14", "data/references/visible-front-completion-2026-11-14",
         "bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1")):
        folder = root / rel
        assert digest((folder / "manifest.json").read_bytes()) == expected
        manifest, amended, *_ = verify_amendment(root, folder, preflight, api)
        references[name] = {"manifest":manifest, "gold":amended}
    records = {r["id"]:r for r in preflight["records"]}
    models = {}
    for name, prediction_path, manifest_path in (
        ("pr35", args.pr35, args.pr35_manifest),
        ("reference", args.reference, args.reference_manifest),
        ("candidate", args.candidate, args.candidate_manifest)):
        selected = select_checkpoint(preflight, manifest_path, require_weights=False)
        header, rows, thresholds = _read_replay(prediction_path, selected, [0.13])
        assert len(rows) == 20 and thresholds == [0.13]
        predictions = {r["image_id"]:r["predictions"] for r in rows}
        assert set(predictions) == set(records)
        models[name] = {"selected":selected, "predictions":predictions,
                        "provenance":{"predictions_sha256":digest(Path(prediction_path).read_bytes()),
                        "checkpoint_sha256":selected["model_sha256"]}}
    report = {"status":"measured", "models":{name:obj["provenance"] for name,obj in models.items()},
              "threshold":0.13,"matching_iou":0.5,"endpoint_confidence":0.5,"radii_px":[5,10],
              "references":{}}
    for ref_name, ref in references.items():
        per_model = {}
        for name, model in models.items():
            scored, quality, scenes = [],[],[]
            pairs = {5:{},10:{}}
            for image_id in sorted(records):
                record = records[image_id]; truth = ref["gold"][image_id]
                pred = model["predictions"][image_id]
                s = score_image(truth,pred,record["width"],record["height"],0.5,0.5)
                q = quality_for_image(truth,pred,record["width"],record["height"],[5,10])
                scored.append(s);quality.append(q)
                for radius in [5,10]:
                    for i in case_success(truth,pred,record["width"],record["height"],radius):
                        pairs[radius][f"{image_id}:{i}"] = True
                scenes.append({"image_id":image_id,"gold":len(truth),"predictions":len(pred),
                               "matched":s["tp"],"misses":s["fn"],"extras":s["fp"],
                               "duplicates":s["duplicate_fp"],
                               "missed_rows":[i+1 for i in s["unmatched_gold_indices"]]})
            aggregate=aggregate_scores(scored); aggregate.pop("endpoint_errors_px",None)
            per_model[name]={"detection":aggregate,"corner_quality":aggregate_quality(quality),
                             "scenes":scenes,"success_identities":{str(k):sorted(v) for k,v in pairs.items()},
                             "empty_scene_outputs":sum(s["predictions"] for s in scenes if s["gold"]==0),
                             "miss_free_nonempty_scenes":sum(s["misses"]==0 for s in scenes if s["gold"]>0),
                             "error_free_nonempty_scenes":sum(s["misses"]==0 and s["extras"]==0 for s in scenes if s["gold"]>0)}
        differences={}
        for base_name in ("reference","pr35"):
            baseline = per_model[base_name];candidate = per_model["candidate"]
            differences[base_name] = {
                "candidate_minus_baseline_counts":{
                    metric:candidate["detection"][metric]-baseline["detection"][metric]
                    for metric in ("tp","fn","fp","duplicate_fp","other_fp","returned_endpoints")},
                "corner_identity_changes":{str(radius):{
                    "gained":sorted(set(candidate["success_identities"][str(radius)])-set(baseline["success_identities"][str(radius)])),
                    "lost":sorted(set(baseline["success_identities"][str(radius)])-set(candidate["success_identities"][str(radius)]))
                    } for radius in (5,10)}}
        report["references"][ref_name]={"reference_id":ref["manifest"]["reference_id"],
                                      "counts":ref["manifest"]["counts"],"models":per_model,
                                      "candidate_deltas":differences}
    save(args.output,report)
    for ref_name, ref in report["references"].items():
        print("\nREFERENCE:", ref_name, "objects",ref["counts"].get("objects"))
        print("MODEL          MATCH    MISS   EXTRA   DUP   BOTH5   BOTH10 EMPTY")
        for name in ("pr35","reference","candidate"):
            m=ref["models"][name];d=m["detection"];q=m["corner_quality"]
            radii={int(r["radius_px"]):r for r in q["radii"]}
            print(f"{name:12s} {d['tp']:6d} {d['fn']:7d} {d['fp']:7d} {d['duplicate_fp']:5d}"
                  f" {radii[5]['both_visible_objects_within_radius']:7d}"
                  f" {radii[10]['both_visible_objects_within_radius']:8d}"
                  f" {m['empty_scene_outputs']:5d}")
        for against,delta in ref["candidate_deltas"].items():
            print("candidate versus",against, "counts",delta["candidate_minus_baseline_counts"],
                  "identity changes",{r:{key:len(arr) for key,arr in obj.items()}
                   for r,obj in delta["corner_identity_changes"].items()})
    print("Development references have been used for model development; no unseen-scene claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="action", required=True)
    a = cmd.add_parser("audit")
    a.add_argument("--archive",required=True)
    a.add_argument("--sha256",required=True)
    a.add_argument("--output",required=True)
    s = cmd.add_parser("score")
    for name in ("root","pr35","reference","candidate","pr35-manifest","reference-manifest","candidate-manifest","output"):
        s.add_argument("--"+name,required=True)
    args = parser.parse_args()
    (audit if args.action=="audit" else score_all)(args)


if __name__ == "__main__":
    main()
