#!/usr/bin/env python3
"""Score the fixed PR79 r10 comparison from saved predictions, without training."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PROTOCOL_SHA = "0b73d423fcaff8f7b7d72da0e6bc33acb5814f2d8435fffa0e163b3efd1bd48f"
MODEL_SHA = {
    "pr35": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
    "reference": "0d927417d7a6017a5976b76cb712229ae2aad3828776f69c4f070d55bee93399",
    "candidate": "7e3fe8e0616ff85a01e29f5783294396b8f0a692bfb5e38888a7a5e30d786480",
}
SCORER_SHA = {
    "evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "diagnose_pose_quality.py": "a3baa1285d1580ee44547f7c3d606f524f2f40616fed568885746d2014159953",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def success_rows(gold, predictions, score, width, height, radius):
    successes = []
    for pair in score["matches"]:
        index = pair["gold_index"]
        target = gold[index]["keypoints"]
        predicted = predictions[pair["prediction_index"]]["keypoints"]
        if not all(point[2] == 2 for point in target):
            continue
        if all(conf >= .5 and x is not None and y is not None
               and 0 <= x < width and 0 <= y < height
               and math.hypot(x-gx, y-gy) <= radius
               for (gx, gy, _), (x, y, conf) in zip(target, predicted)):
            successes.append(index+1)
    return successes


def run(args):
    root = args.root.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from evaluate_pose import preflight_dataset, _read_replay
    from pose_checkpoint import select_checkpoint
    from pose_scoring import aggregate_scores, score_image, _box_iou
    from diagnose_pose_quality import aggregate_quality, quality_for_image
    from rescore_reference_amendment import load_api, verify_amendment

    protocol_path = root / "experiments/front-review-2026-11-19-r10.json"
    require(sha(protocol_path) == PROTOCOL_SHA, "The preregistered protocol changed")
    protocol = json.loads(protocol_path.read_text())
    for name, expected in SCORER_SHA.items():
        require(sha(root / "scripts" / name) == expected, f"Scorer changed: {name}")
    preflight = preflight_dataset(root, require_weights=False)
    records = {record["id"]:record for record in preflight["records"]}
    _, api = load_api(root)
    models = {}
    for role in MODEL_SHA:
        pred_path = getattr(args,role)
        checkpoint_path = getattr(args,role+"_manifest")
        selected = select_checkpoint(preflight,checkpoint_path,require_weights=False)
        require(selected["model_sha256"] == MODEL_SHA[role], f"Wrong model: {role}")
        header, rows, thresholds = _read_replay(pred_path,selected,[.13])
        require(thresholds == [.13] and len(rows) == 20, f"Incomplete inference: {role}")
        require(header["runtime"]["ultralytics"] == "8.4.182", f"Ultralytics changed: {role}")
        require(all(header["source"]["files_sha256"].get("scripts/"+name) == expected
                    for name,expected in SCORER_SHA.items() if name != "diagnose_pose_quality.py"),
                f"Prediction evaluator provenance differs: {role}")
        models[role] = {"predictions":{row["image_id"]:row["predictions"] for row in rows},
                        "metadata":{key:header[key] for key in
                          ("source","model_sha256","runtime","inference","box_confidences","checkpoint_identity")},
                        "predictions_sha256":sha(pred_path)}
    runtimes = [model["metadata"]["runtime"] for model in models.values()]
    require(all(runtime == runtimes[0] for runtime in runtimes), "Inference runtimes differ")
    report = {
        "schema_version":1,"kind":"pr79-r10-fixed-development-comparison","status":"measured",
        "protocol":{"path":str(protocol_path.relative_to(root)),"sha256":PROTOCOL_SHA},
        "measurement_script_sha256":sha(__file__), "scorers_sha256":SCORER_SHA,
        "models":{role:{key:value for key,value in model.items() if key != "predictions"}
                  for role,model in models.items()},
        "thresholds":{"box_confidence":.13,"endpoint_confidence":.5,"nms_iou":.7,
                      "matching_iou":.5,"imgsz":1280,"radii_px":[5,10]},
        "references":{},
        "limits":[
            "The reused development frames do not establish reliability in unseen capture sessions.",
            "Unmatched outputs are matching outcomes, not all independently adjudicated background false positives.",
            "Unlocatable reference corners have no verified coordinate; their predicted coordinates cannot be scored for accuracy.",
            "Confidence is not a calibrated probability of pixel accuracy or physical visibility.",
            "Training validation uses 664 original objects; primary and historical development references use 666 and 663 separately.",
            "No seed, epoch, confidence or NMS search is performed. No automatic model promotion follows.",
        ],
    }
    for label in ("primary","diagnostic"):
        setting = protocol["evaluation"]
        path = root / setting[label+"_reference"]
        require(sha(path) == setting[label+"_reference_sha256"], f"Reference changed: {label}")
        manifest,gold,*_ = verify_amendment(root,path.parent,preflight,api)
        per_model = {}
        traces = {}
        for role,model in models.items():
            scores,qualities,scenes=[],[],[]
            success={5:set(),10:set()}
            matches=set()
            groups={str(k):{"gold":0,"matched":0,"misses":0} for k in range(3)}
            traces[role]={}
            for image_id,record in sorted(records.items()):
                target,predicted=gold[image_id],model["predictions"][image_id]
                width,height=record["width"],record["height"]
                score=score_image(target,predicted,width,height,.5,.5)
                quality=quality_for_image(target,predicted,width,height,[5,10])
                scores.append(score);qualities.append(quality)
                matched={pair["gold_index"] for pair in score["matches"]}
                for index,obj in enumerate(target):
                    group=groups[str(sum(point[2]==2 for point in obj["keypoints"]))]
                    group["gold"]+=1;group["matched"]+=int(index in matched);group["misses"]+=int(index not in matched)
                matches.update(f"{image_id}:{index+1}" for index in matched)
                for radius in success:
                    success[radius].update(f"{image_id}:{row}" for row in
                                          success_rows(target,predicted,score,width,height,radius))
                scenes.append({"image_id":image_id,"gold":len(target),"predictions":len(predicted),
                               "matched":score["tp"],"misses":score["fn"],"extras":score["fp"],
                               "duplicates":score["duplicate_fp"],
                               "missed_rows":[i+1 for i in score["unmatched_gold_indices"]]})
                traces[role][image_id]={key:score[key] for key in
                    ("matches","unmatched_gold_indices","unmatched_prediction_indices","duplicate_prediction_indices")}
            aggregate=aggregate_scores(scores);aggregate.pop("endpoint_errors_px",None)
            per_model[role]={"detection":aggregate,"corner_quality":aggregate_quality(qualities),
                             "locatable_corner_groups":groups,"scenes":scenes,
                             "matched_identities":sorted(matches),
                             "success_identities":{str(k):sorted(v) for k,v in success.items()},
                             "empty_scene_outputs":sum(s["predictions"] for s in scenes if s["gold"]==0),
                             "miss_free_nonempty_scenes":sum(s["misses"]==0 for s in scenes if s["gold"]>0),
                             "error_free_nonempty_scenes":sum(s["misses"]==s["extras"]==0 for s in scenes if s["gold"]>0)}
        deltas={}
        for baseline in ("reference","pr35"):
            a,b=per_model[baseline],per_model["candidate"]
            gained=sorted(set(b["matched_identities"])-set(a["matched_identities"]))
            lost=sorted(set(a["matched_identities"])-set(b["matched_identities"]))
            changed=[]
            for identity in gained+lost:
                image_id,row=identity.rsplit(":",1);index=int(row)-1
                item={"identity":identity,"change":"gained" if identity in gained else "lost","models":{}}
                for role in (baseline,"candidate"):
                    predicted=models[role]["predictions"][image_id]
                    best=max(enumerate(predicted),key=lambda pair:_box_iou(gold[image_id][index]["bbox_xyxy"],pair[1]["bbox_xyxy"])) if predicted else None
                    pair=next((p for p in traces[role][image_id]["matches"] if p["gold_index"]==index),None)
                    item["models"][role]={"matched":pair,"best_prediction_index":best[0] if best else None,
                        "best_iou":_box_iou(gold[image_id][index]["bbox_xyxy"],best[1]["bbox_xyxy"]) if best else None,
                        "best_prediction":best[1] if best else None}
                changed.append(item)
            deltas[baseline]={"counts":{key:b["detection"][key]-a["detection"][key] for key in
                               ("tp","fn","fp","duplicate_fp","other_fp","returned_endpoints")},
                              "gained_matches":gained,"lost_matches":lost,"changed_match_details":changed,
                              "corner_identity_changes":{str(radius):{
                                  "gained":sorted(set(b["success_identities"][str(radius)])-set(a["success_identities"][str(radius)])),
                                  "lost":sorted(set(a["success_identities"][str(radius)])-set(b["success_identities"][str(radius)]))}
                                  for radius in [5,10]}}
        report["references"][label]={"reference_id":manifest["reference_id"],"manifest_sha256":sha(path),
            "counts":manifest["counts"],"remaining_uncertain_rows":setting[("unresolved_reference_rows" if label=="primary" else "diagnostic_unresolved_reference_rows")],
            "models":per_model,"candidate_deltas":deltas,"matching_traces":traces}
    require(not args.output.exists(), "Choose a fresh output; previous evidence is preserved")
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for label,ref in report["references"].items():
        print(label,ref["counts"])
        for role,model in ref["models"].items():
            d=model["detection"];q=model["corner_quality"]
            print(role,{key:d[key] for key in ["tp","fn","fp","duplicate_fp","other_fp"]},
                  "both-corner",{str(r["radius_px"]):r["both_visible_objects_within_radius"] for r in q["radii"]},
                  "groups",model["locatable_corner_groups"],"miss-free",model["miss_free_nonempty_scenes"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    for role in MODEL_SHA:
        parser.add_argument("--"+role,type=Path,required=True)
        parser.add_argument("--"+role+"-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
