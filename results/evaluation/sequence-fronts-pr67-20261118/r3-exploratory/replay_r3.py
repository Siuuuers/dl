#!/usr/bin/env python3
"""Exploratory replay of unchanged PR67 predictions on the PDF-recovered r3 reference."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

RECOVERY_SHA = "6e70cb78b7ae6a71d9a61021c42a94bc7c24ccb2eaeeba673ba326c2434aa3d7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(args):
    root = args.root.resolve()
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(args.prediction_root.resolve()))
    from evaluate_pose import preflight_dataset, parse_pose_labels, _read_replay
    from pose_scoring import aggregate_scores, score_image
    from diagnose_pose_quality import quality_for_image, aggregate_quality
    from verify_pose_candidate import check_geometry
    from measure_sequence_fronts_pr67 import MODEL_SHA, SCORER_SHA, success_rows

    require(sha(args.manifest) == RECOVERY_SHA, "Recovery manifest identity differs")
    for name, expected in SCORER_SHA.items():
        require(sha(root / "scripts" / name) == expected, f"Scoring helper differs: {name}")
    recovery = json.loads(args.manifest.read_text())
    preflight = preflight_dataset(root, require_weights=False)
    records = {r["id"]:r for r in preflight["records"]}
    files = {f["image_id"]:f for f in recovery["files"] if f["split"] == "val"}
    require(set(files) == set(records), "The recovered validation inventory differs")
    gold, row_maps = {}, {}
    for image_id, descriptor in files.items():
        path = args.manifest.parent / descriptor["label_path"]
        require(sha(path) == descriptor["recovered_label_sha256"], f"Recovered label changed: {image_id}")
        record = records[image_id]
        require(record["image_sha256"] == descriptor["source_image_sha256"], f"Image identity differs: {image_id}")
        gold[image_id] = parse_pose_labels(path.read_text(), record["width"], record["height"], str(path))
        check_geometry(gold[image_id], record["width"], record["height"], str(path))
        row_maps[image_id] = {row["current_row"]:row["parent_row"] for row in descriptor["row_map"]}
        require(set(row_maps[image_id]) == set(range(1, len(gold[image_id])+1)), f"Incomplete row map: {image_id}")
    groups = {str(k):sum(sum(point[2] == 2 for point in obj["keypoints"]) == k
                         for objects in gold.values() for obj in objects) for k in range(3)}
    counts = {"images":len(gold),"objects":sum(map(len,gold.values())),
              "locatable_corners":groups["1"]+2*groups["2"],
              "missing_corners":2*groups["0"]+groups["1"],
              "empty_images":sum(not objects for objects in gold.values())}
    require(counts == {k:recovery["counts"]["val"][k] for k in counts}, "Recovered counts differ")
    frozen_path = args.prediction_root / "comparison.json"
    frozen = json.loads(frozen_path.read_text())
    report = {
        "schema_version":1,"kind":"exploratory-r3-reference-replay","status":"exploratory_only",
        "inference_run":False,"training_run":False,"model_promotion":False,
        "replaces_preregistered_comparison":False,
        "recovery_manifest_sha256":RECOVERY_SHA,"fixed_comparison_sha256":sha(frozen_path),
        "script_sha256":sha(__file__),"scoring_helpers_sha256":SCORER_SHA,
        "counts":counts,"objects_by_locatable_corners":groups,
        "reference_provenance":{key:recovery[key] for key in
            ("recovery_id","status","release_ready","original_authoring_bytes_recovered","source_pdfs","provenance")},
        "thresholds":frozen["thresholds"],"models":{},"candidate_deltas":{},
        "held_rows":[{"image_id":image_id,"current_rows":f["held_current_rows"]}
                     for image_id,f in files.items() if f["held_current_rows"]],
        "limits":[
            "This revised-reference replay is exploratory and does not replace the preregistered 666/663-reference comparison.",
            "Predictions are unchanged; score changes reflect reference edits, not model learning.",
            "The labels were recovered from PDF vector geometry, not restored byte-identically from original authoring files.",
            "The r3 review is not released or fully certified. Physical ownership ambiguity remains, including new_000007 rows7 and14.",
            "The reused video frames are development data and do not establish unseen-session reliability.",
        ],
    }
    for role, expected_model in MODEL_SHA.items():
        path = args.prediction_root / f"eval-{role}/predictions.jsonl"
        expected = frozen["models"][role]
        require(sha(path) == expected["predictions_sha256"], f"Saved predictions changed: {role}")
        selected = {**preflight,"model_sha256":expected_model,
                    "checkpoint_identity":expected["metadata"]["checkpoint_identity"]}
        header, items, thresholds = _read_replay(path, selected, [.13])
        require(thresholds == [.13] and len(items) == 20, "Prediction scope differs")
        scores,qualities,scenes=[],[],[]
        matches,success=set(),{5:set(),10:set()}
        group_counts={str(k):{"gold":0,"matched":0,"misses":0} for k in range(3)}
        for item in items:
            image_id=item["image_id"];record=records[image_id]
            target,predicted=gold[image_id],item["predictions"]
            width,height=record["width"],record["height"]
            score=score_image(target,predicted,width,height,.5,.5)
            scores.append(score);qualities.append(quality_for_image(target,predicted,width,height,[5,10]))
            matched={p["gold_index"] for p in score["matches"]}
            for index,obj in enumerate(target):
                group=group_counts[str(sum(point[2]==2 for point in obj["keypoints"]))]
                group["gold"]+=1;group["matched"]+=int(index in matched);group["misses"]+=int(index not in matched)
            matches.update(f"{image_id}:{i+1}" for i in matched)
            for radius in success:
                success[radius].update(f"{image_id}:{row}" for row in success_rows(target,predicted,score,width,height,radius))
            scenes.append({"image_id":image_id,"gold":len(target),"predictions":len(predicted),
                "matched":score["tp"],"misses":score["fn"],"extras":score["fp"],"duplicates":score["duplicate_fp"],
                "missed_current_rows":[i+1 for i in score["unmatched_gold_indices"]],
                "missed_primary_parent_rows":[row_maps[image_id][i+1] for i in score["unmatched_gold_indices"]]})
        aggregate=aggregate_scores(scores);aggregate.pop("endpoint_errors_px",None)
        report["models"][role]={"model_sha256":expected_model,"predictions_sha256":sha(path),
            "detection":aggregate,"corner_quality":aggregate_quality(qualities),"corner_groups":group_counts,
            "scenes":scenes,"matched_identities":sorted(matches),
            "success_identities":{str(k):sorted(v) for k,v in success.items()},
            "empty_scene_outputs":sum(s["predictions"] for s in scenes if s["gold"]==0),
            "miss_free_nonempty_scenes":sum(s["misses"]==0 for s in scenes if s["gold"]>0),
            "error_free_nonempty_scenes":sum(s["misses"]==s["extras"]==0 for s in scenes if s["gold"]>0)}
    for baseline in ("reference","pr35"):
        a,b=report["models"][baseline],report["models"]["candidate"]
        report["candidate_deltas"][baseline]={
            "counts":{key:b["detection"][key]-a["detection"][key] for key in ("tp","fn","fp","duplicate_fp","other_fp")},
            "gained_current_matches":sorted(set(b["matched_identities"])-set(a["matched_identities"])),
            "lost_current_matches":sorted(set(a["matched_identities"])-set(b["matched_identities"])),
            "corner_identity_changes":{str(radius):{
                "gained":sorted(set(b["success_identities"][str(radius)])-set(a["success_identities"][str(radius)])),
                "lost":sorted(set(a["success_identities"][str(radius)])-set(b["success_identities"][str(radius)]))}
                for radius in [5,10]}}
    require(not args.output.exists(), "Choose a fresh output path")
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"counts":counts,"groups":groups,"models":{role:{
        "detection":model["detection"],"corner_quality":model["corner_quality"],"corner_groups":model["corner_groups"],
        "miss_free_nonempty_scenes":model["miss_free_nonempty_scenes"]} for role,model in report["models"].items()},
        "candidate_deltas":report["candidate_deltas"]},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","manifest","prediction-root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
