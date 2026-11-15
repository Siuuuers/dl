#!/usr/bin/env python3
"""Fair PR35-versus-original-epoch173 comparison, at identical settings/references."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from evaluate_pose import _read_replay, preflight_dataset
from pose_checkpoint import select_checkpoint
from pose_scoring import score_image, aggregate_scores
from diagnose_pose_quality import quality_for_image, aggregate_quality
from rescore_reference_amendment import load_api,verify_amendment


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while part:=f.read(1024*1024):h.update(part)
    return h.hexdigest()

def run(root,original,selected,manifest,out):
    preflight=preflight_dataset(root,require_weights=False)
    pr35=select_checkpoint(preflight,manifest,require_weights=False)
    loaded={}
    for label,path,checkpoint,expected in [
        ("original_epoch173",original,preflight,"e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"),
        ("pr35_seed46",selected,pr35,"420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1")]:
        header,items,thresholds=_read_replay(path,checkpoint,[0.13,0.15,0.25])
        assert thresholds==[0.13,0.15,0.25] and len(items)==60 and checkpoint["model_sha256"]==expected
        assert all(x["threshold"] in thresholds for x in items)
        loaded[label]={t:{r["image_id"]:r["predictions"] for r in items if r["threshold"]==t} for t in thresholds}
        assert all(len(loaded[label][t])==20 for t in thresholds)
    records={r["id"]:r for r in preflight["records"]}
    _,api=load_api(root)
    refs={}
    for key,dirname,ref_sha in [
        ("2026-11-12-primary","visible-front-adjudication-2026-11-12","75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e"),
        ("2026-11-14-diagnostic","visible-front-completion-2026-11-14","bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1")]:
        folder=root/"data"/"references"/dirname
        assert sha(folder/"manifest.json")==ref_sha
        parsed, gold,*_=verify_amendment(root,folder,preflight,api)
        refs[key]={"manifest":parsed,"gold":gold,"sha256":ref_sha}
    report={"kind":"pr35-versus-original-epoch173","no_training":True,
            "score_thresholds":[0.13,0.15,0.25],"nms_iou":0.7,
            "matching_iou":0.5,"endpoint_confidence":0.5,"radii_px":[5,10],
            "original_model_sha256":"e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e",
            "pr35_model_sha256":"420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
            "original_predictions_sha256":sha(original),"pr35_predictions_sha256":sha(selected),
            "references":{}}
    for refname, ref in refs.items():
        tables={}
        for threshold in [0.13,0.15,0.25]:
            models={}
            for label,pred_by_threshold in loaded.items():
                scored=[];quality=[];per=[]
                for image_id in sorted(records):
                    rec=records[image_id]
                    truth=ref["gold"][image_id]
                    p=pred_by_threshold[threshold][image_id]
                    s=score_image(truth,p,rec["width"],rec["height"],0.5,0.5)
                    q=quality_for_image(truth,p,rec["width"],rec["height"],[5,10])
                    scored.append(s); quality.append(q)
                    per.append({"image_id":image_id,"gold":len(truth),"pred":len(p),"tp":s["tp"],"fn":s["fn"],"fp":s["fp"],
                                "duplicates":s["duplicate_fp"],"missing_row_numbers":[i+1 for i in s["unmatched_gold_indices"]]})
                counts=aggregate_scores(scored)
                counts.pop("endpoint_errors_px",None)
                q=aggregate_quality(quality)
                models[label]={"score":counts,"quality":q,"empty_outputs":sum(r["pred"] for r in per if r["gold"]==0),
                               "no_miss_nonempty":sum(r["fn"]==0 for r in per if r["gold"]>0),
                               "all_correct_nonempty":sum(r["fn"]==0 and r["fp"]==0 for r in per if r["gold"]>0),
                               "per_image":per}
            tables[str(threshold)]={"models":models,"candidate_minus_original": {
                k:models["pr35_seed46"]["score"][k]-models["original_epoch173"]["score"][k]
                for k in ("tp","fn","fp","duplicate_fp","other_fp","returned_endpoints")}}
        report["references"][refname]={"reference_sha256":ref["sha256"],"counts":ref["manifest"]["counts"],"thresholds":tables}
    benchmark=report["references"]["2026-11-12-primary"]["thresholds"]["0.13"]["models"]["pr35_seed46"]
    radius={int(x["radius_px"]):x["both_visible_objects_within_radius"] for x in benchmark["quality"]["radii"]}
    assert (benchmark["score"]["tp"],benchmark["score"]["fn"],benchmark["score"]["fp"],radius[5],radius[10])==(643,20,85,529,577), "PR35 reproduction mismatch"
    out.mkdir(parents=True,exist_ok=False)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    body=[
        "# PR35 compared with the original baomai epoch173 model",
        "",
        "Original baomai and PR35 are compared using **identical inference thresholds** and the *same versioned development reference*. At 0.13 PR35 is the selected assistant-reviewed operating point; 0.15 and 0.25 are additional fixed historical diagnostics. Original frames in this development reference are not independent from model development. No hidden-coordinate accuracy is inferred.",
        "",
        "## Main 2026-11-12 reference: 663 objects, 584 two-corner-eligible, three empty images",
        "",
        "| Conf | Model | Matched | Misses ↓ | Extras ↓ | Duplicates ↓ | 5px pairs ↑ | 10px pairs ↑ | Empty outputs ↓ |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for refname in ["2026-11-12-primary","2026-11-14-diagnostic"]:
        if refname=="2026-11-14-diagnostic":
            body+=["","## Secondary 2026-11-14 reference (box refinements; compare within this section only)","","| Conf | Model | Matched | Misses ↓ | Extras ↓ | Duplicates ↓ | 5px pairs ↑ | 10px pairs ↑ | Empty outputs ↓ |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for conf in [0.13,0.15,0.25]:
            for label in ["original_epoch173","pr35_seed46"]:
                data=report["references"][refname]["thresholds"][str(conf)]["models"][label]
                s=data["score"];q=data["quality"]
                radii={int(r["radius_px"]):r for r in q["radii"]}
                body.append(f"| {conf} | {label} | {s['tp']} | {s['fn']} | {s['fp']} | {s['duplicate_fp']} | {radii[5]['both_visible_objects_within_radius']}/584 | {radii[10]['both_visible_objects_within_radius']}/584 | {data['empty_outputs']} |")
    body+=["","**Interpretation:** PR35's advantage over original epoch173 includes 30 extra training epochs and PR35-specific label corrections. That baseline contrast does not isolate the causal effect of annotation edits alone; for that, use PR35's matched-control comparison in its seed46 study.",
           "","**Provenance:** historical original baomai commit 351eb3cf960ccc34829e2bc72e1fa4645acc6295; PR35 selected weight SHA-256 420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1.",
           "","**Limit:** 663-object reference was repeatedly used for development. Nonempty scenes with complete, correction-free results must be checked; pair success excludes objects whose actual lower corners cannot be located.",""]
    (out/"README.md").write_text("\n".join(body))
    print("\n".join(body))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ["root","original","selected","manifest","output"]:
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    run(a.root.resolve(),a.original.resolve(),a.selected.resolve(),a.manifest.resolve(),a.output.resolve())
