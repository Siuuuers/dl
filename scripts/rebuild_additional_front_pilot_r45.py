#!/usr/bin/env python3
"""Reconstructed R45 supplementary-photo experiment. No unpublished dependencies."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile
sys.dont_write_bytecode = True

from evaluate_pose import parse_pose_labels, INFERENCE, _prediction_records
from pose_checkpoint import get_recipe, recipe_identity, check_training_settings, check_loaded_pose_model
from pose_scoring import score_image, aggregate_scores
from predict_pose_folder import postclip_nms

RECIPE = "reviewed-incremental-fronts-seed46-v1"
RECIPE_HASH = "f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8"
INIT = "e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e"
OVERLAY = "data/candidates/front-review-2026-11-26-r37/manifest.json"
OVERLAY_HASH = "ccfe09cae8e890b80896dac5d025f981573e5da3f49bb9f40175f2d0685de2f3"
EXTRA = "frame_0383"
ROLES = ("reference","candidate")
COUNTS = {"reference": (137,3784,6837,731,13), "candidate": (138,3827,6923,731,13)}
VAL = (20,664,1227,101,3)
REFS = {
    "r38": ("data/reviews/fronts-2026-11-26-r38/manifest.json",None,654),
    "frozen666": ("data/references/sequence-refined-reference-2026-11-18/manifest.json","0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96",666),
    "historical663": ("data/references/visible-front-completion-2026-11-14/manifest.json","bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1",663),
}

def need(ok,msg):
    if not ok: raise RuntimeError(msg)

def digest(b): return hashlib.sha256(b).hexdigest()
def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def checked(p,h):
    p=Path(p); need(p.is_file() and sha(p)==h,"Input hash differs: "+str(p)); return p
def encoded(obj): return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def put(p,b):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): need(sha(p)==digest(b),"Preserved incompatible file: "+str(p))
    else: p.write_bytes(b)
def save(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(".tmp"); t.write_bytes(encoded(obj)); t.replace(p)
def read(p): return json.loads(Path(p).read_text())
def rev(root): return subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
def labels(b,name): return parse_pose_labels(b.decode(),1280,720,name)
def counts(groups):
    n=sum(map(len,groups)); v=sum(p[2]==2 for g in groups for r in g for p in r["keypoints"])
    return len(groups),n,v,2*n-v,sum(not g for g in groups)

def extra(root):
    p=checked(root/"results/review/front-mining-r44/review.zip","100bfaceb9281bcf8c37c48eaaacf5159df14b0d04dd0ec37d5dcd4fc89c672d")
    with zipfile.ZipFile(p) as z:
        m=z.read("manifest.json")
        need(digest(m)=="6a52279ac5c53452cf7e6a3dbd6ea8f76d4aeae287bd2d2a23ae5b61a4c231bf","R44 manifest differs")
        fs=[f for f in json.loads(m)["frames"] if f["eligible_for_candidate_export"]]
        need([f["image_id"] for f in fs]==[EXTRA],"Unexpected complete-image export")
        f=fs[0]
        need(f["candidate_rows"]==43 and f["training_split"]=="unassigned" and not any(r["unresolved_fields"] for r in f["rows"]+f["additions"]),"Held extra frame")
        lab=z.read("complete-candidates/labels/"+EXTRA+".txt")
        old=z.read("history/pr131-r43.zip")
    need(digest(old)=="ae3d7310fd6b2d3a623bd7b8760dc9d6ffc75e6df1be28bbf0a2555fdb380d6e","R43 archive differs")
    with zipfile.ZipFile(io.BytesIO(old)) as z: im=z.read("originals/"+EXTRA+".jpg")
    need(digest(im)==f["source_image_sha256"] and digest(lab)==f["candidate_label_sha256"],"Extra bytes differ")
    need(counts([labels(lab,EXTRA)])==(1,43,86,0,0),"Extra labels differ")
    return im,lab

def stage(root,work):
    from PIL import Image
    m=read(root/"data/manifest.json"); ov=read(checked(root/OVERLAY,OVERLAY_HASH))
    need(m["model_epoch"]==173 and m["model_sha256"]==INIT,"Initializer declaration differs")
    checked(root/m["model_checkpoint"],INIT)
    need(recipe_identity(RECIPE)["recipe_sha256"]==RECIPE_HASH,"Recipe differs")
    rows=m["records"]; ts={r["id"] for r in rows if r["split"]=="train"}; vs={r["id"] for r in rows if r["split"]=="val"}
    need(len(rows)==157 and len(ts)==137 and len(vs)==20 and not ts&vs and EXTRA not in ts|vs,"Split differs")
    changes={r["image_id"]:r for r in ov["changes"]}; sources={r["image_id"]:r for r in ov["image_sources"]}
    need(set(sources)==ts and len(changes)==92 and set(changes)<=ts,"Overlay inventory differs")
    im,lab=extra(root)
    need(digest(im) not in {r["image_sha256"] for r in rows},"Extra image duplicates original")
    with Image.open(io.BytesIO(im)) as x: need(x.size==(1280,720),"Extra dimensions")
    common=[]
    for r in rows:
        i,s=r["id"],r["split"]
        p=checked(root/"data"/r["image"],r["image_sha256"])
        with Image.open(p) as x: need(x.size==(1280,720),"Original dimensions: "+i)
        q=checked(root/"data"/r["label"],r["label_sha256"])
        if i in changes:
            c=changes[i]
            need(c["source_label_sha256"]==r["label_sha256"] and c["source_image_sha256"]==r["image_sha256"],"Overlay provenance differs")
            q=checked((root/OVERLAY).parent/c["candidate_label"],c["candidate_label_sha256"])
        if s=="train": need(sha(q)==sources[i]["effective_label"]["sha256"] and r["image_sha256"]==sources[i]["source_image_sha256"],"Effective R37 bytes differ")
        common.append({"id":i,"split":s,"image":"images/"+s+"/"+i+".png","label":"labels/"+s+"/"+i+".txt","image_sha256":r["image_sha256"],"label_sha256":sha(q),"image_source":str(p.relative_to(root)),"label_source":str(q.relative_to(root))})
    e={"id":EXTRA,"split":"train","image":"images/train/"+EXTRA+".jpg","label":"labels/train/"+EXTRA+".txt","image_sha256":digest(im),"label_sha256":digest(lab)}
    manifest={"source":rev(root),"recipe":recipe_identity(RECIPE),"initializer":INIT,"roles":{"reference":common,"candidate":common+[e]}}
    for role in ROLES:
        dest=work/"datasets"/role
        for item in manifest["roles"][role]:
            for kind in ("image","label"):
                target=dest/item[kind]
                if item["id"]==EXTRA: put(target,im if kind=="image" else lab)
                else:
                    src=root/item[kind+"_source"]
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists(): shutil.copyfile(src,target)
                    checked(target,item[kind+"_sha256"])
        put(dest/"data.yaml",encoded({"path":str(dest),"train":"images/train","val":"images/val","names":["cassette_front"],"kpt_shape":[2,3],"flip_idx":[1,0]}))
    put(work/"staging.json",encoded(manifest))
    verify_stage(work)
    # Fail before expensive training if any reference is incomplete.
    read_refs(root,common)
    print("Input preflight passed:",COUNTS,"val",VAL,flush=True)

def verify_stage(work):
    m=read(work/"staging.json")
    need(m["roles"]["candidate"][:-1]==m["roles"]["reference"] and m["roles"]["candidate"][-1]["id"]==EXTRA,"Treatment differs")
    for role in ROLES:
        d=work/"datasets"/role; es=m["roles"][role]; groups={"train":[],"val":[]}
        for e in es:
            checked(d/e["image"],e["image_sha256"])
            groups[e["split"]].append(labels(checked(d/e["label"],e["label_sha256"]).read_bytes(),e["id"]))
        for kind,sub,suffix in (("image","images",{".png",".jpg",".jpeg"}),("label","labels",{".txt"})):
            actual={p.relative_to(d).as_posix() for p in (d/sub).rglob("*") if p.is_file() and p.suffix.lower() in suffix}
            need(actual=={e[kind] for e in es},"Unexpected "+role+" "+kind+" files")
        need(counts(groups["train"])==COUNTS[role] and counts(groups["val"])==VAL,"Staged inventory differs")
    return m

def runtime():
    import torch, ultralytics
    from importlib.metadata import version
    from partial_box_loss_r31 import verify_native_sources
    need(torch.cuda.is_available() and ultralytics.__version__=="8.4.182","Pinned GPU environment required")
    native=verify_native_sources()
    return {"python":platform.python_version(),"torch":str(torch.__version__),"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
            "gpu":torch.cuda.get_device_name(0),"native":native,"packages":{n:version(n) for n in ("ultralytics","torchvision","numpy","Pillow","opencv-python","psutil")}}

def verify_arm(work,role,current=None):
    out=work/role; r=read(out/"receipt.json")
    need(r["status"]=="complete" and r["role"]==role and r["staging_sha256"]==sha(work/"staging.json"),"Arm identity differs")
    need(r["epochs"]==list(range(1,31)) and r["batches"]==1050 and r["images"]==COUNTS[role][0]*30 and r["steps"]==78 and r["updated"],"Arm budget differs")
    if current is not None: need(r["runtime"]==current,"Runtime mismatch")
    need(set(r["artifacts"])=={"fit/weights/last.pt","fit/results.csv"},"Arm artifacts differ")
    for p,h in r["artifacts"].items(): checked(out/p,h)
    return r

def train(root,work,role,trial=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import SETTINGS
    m=verify_stage(work); rt=runtime(); need(m["source"]==rev(root),"Source differs")
    folder=work/"datasets"/role; out=work/("resource-trial" if trial else role)
    if not trial and (out/"receipt.json").exists():
        verify_arm(work,role,rt); print("Reusing",role); return
    if trial and (out/"receipt.json").exists():
        r=read(out/"receipt.json")
        need(r["status"]=="trial-complete" and r["batches"]==4 and r["updated"] and r["runtime"]==rt and r["staging_sha256"]==sha(work/"staging.json"),"Trial identity differs")
        print("Reusing verified trial"); return
    if not trial and role=="candidate": verify_arm(work,"reference",rt)
    need(not out.exists(),"Preserved incomplete run; rename before retry: "+str(out)); out.mkdir()
    init=checked(root/"weights/cassette_lower_epoch173.pt",INIT)
    progress={"role":role,"trial":trial,"source":rev(root),"runtime":rt,"staging_sha256":sha(work/"staging.json"),"status":"starting","epochs":[],"batches":0,"images":0,"steps":0,"updated":False}
    def record(): save(out/"progress.json",progress)
    class TrialDone(Exception): pass
    class Trainer(PoseTrainer):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            need(self.world_size in (0,1) and not self.ddp,"Single GPU only")
            for event,callback in (("on_train_epoch_start",self.start),("on_train_batch_end",self.batch_done),("on_train_epoch_end",self.epoch_done),("on_model_save",self.saved)):
                self.add_callback(event,callback)
        def _build_train_pipeline(self):
            super()._build_train_pipeline()
            need(self.batch_size==4 and len(self.train_loader)==35,"Actual batch budget differs")
            for split,loader in (("train",self.train_loader),("val",self.test_loader)):
                es={e["image"]:e for e in m["roles"][role] if e["split"]==split}; seen=set()
                ls=loader.dataset.labels
                n=sum(len(x["cls"]) for x in ls); v=sum(int((x["keypoints"][:,:,2]==2).sum()) for x in ls)
                observed=(len(loader.dataset),n,v,2*n-v,sum(not len(x["cls"]) for x in ls))
                need(observed==(COUNTS[role] if split=="train" else VAL),"Actual loader inventory differs: "+str(observed))
                for image,label in zip(loader.dataset.im_files,img2label_paths(loader.dataset.im_files)):
                    key=Path(image).resolve().relative_to(folder).as_posix(); e=es[key]
                    need(Path(label).resolve()==(folder/e["label"]).resolve(),"Loader label path differs")
                    checked(image,e["image_sha256"]); checked(label,e["label_sha256"]); seen.add(key)
                need(seen==set(es),"Loader image paths differ")
            check_training_settings(dict(vars(self.args)),RECIPE,model=init,data=folder/"data.yaml",device="0",project=out)
            progress["settings"]=dict(vars(self.args)); record()
        def start(self,t): self.seen=Counter(); self.nb=0
        def preprocess_batch(self,batch):
            b=super().preprocess_batch(batch); self.ids=[Path(p).stem for p in b["im_file"]]
            need(len(self.ids)==b["img"].shape[0],"Batch image count differs"); return b
        def optimizer_step(self):
            need(bool(torch.isfinite(self.loss).all()),"Nonfinite loss")
            gs=[p.grad for p in self.model.parameters() if p.grad is not None]
            need(gs and all(bool(torch.isfinite(g).all()) for g in gs),"Nonfinite or missing gradients")
            p=next((p for g in self.optimizer.param_groups if g["lr"]>0 for p in g["params"] if p.grad is not None and bool(torch.count_nonzero(p.grad))),None)
            need(p is not None,"No nonzero gradient at positive learning rate")
            before=p.detach().clone(); super().optimizer_step()
            need(bool(torch.isfinite(p).all()),"Nonfinite parameter"); progress["updated"]|=bool(torch.any(before!=p.detach())); progress["steps"]+=1
        def batch_done(self,t):
            need(bool(torch.isfinite(self.loss).all()),"Nonfinite batch")
            self.seen.update(self.ids); self.nb+=1; progress["batches"]+=1; progress["images"]+=len(self.ids); record()
            if trial and progress["batches"]==4:
                need(progress["updated"] and progress["steps"]>0,"No actual trial update"); raise TrialDone()
        def epoch_done(self,t):
            need(self.nb==35 and self.seen==Counter(e["id"] for e in m["roles"][role] if e["split"]=="train"),"Epoch exposure differs")
        def saved(self,t):
            need(self.last.is_file() and self.epoch+1==len(progress["epochs"])+1,"Noncontiguous saved epoch")
            progress["epochs"].append(self.epoch+1); record()
    SETTINGS.update({k:False for k in ("sync","clearml","comet","dvc","mlflow","raytune","tensorboard","wandb")})
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    model=YOLO(str(init)); check_loaded_pose_model(model)
    try:
        model.train(trainer=Trainer,**get_recipe(RECIPE),data=str(folder/"data.yaml"),device="0",workers=0,plots=False,verbose=False,project=str(out),name="fit",exist_ok=False)
        need(not trial,"Trial did not stop")
        need(progress["epochs"]==list(range(1,31)) and progress["batches"]==1050 and progress["images"]==COUNTS[role][0]*30 and progress["steps"]==78 and progress["updated"],"Full measured budget differs")
        with (out/"fit/results.csv").open() as f: need(len(list(csv.DictReader(f)))==30,"Results epoch count differs")
        verify_stage(work)
        progress["artifacts"]={p:sha(out/p) for p in ("fit/weights/last.pt","fit/results.csv")}
        progress["status"]="complete"
    except TrialDone: progress["status"]="trial-complete"
    except Exception as e:
        progress["status"]="failed"; progress["error"]=str(e); record(); raise
    record(); save(out/"receipt.json",progress)
    print(role,progress["status"],flush=True)

def read_refs(root,entries):
    vs={e["id"]:e for e in entries if e["split"]=="val"}; result={}
    for name,(rel,h,n) in REFS.items():
        p=root/rel
        if h: checked(p,h)
        rs=[x for x in read(p)["files"] if x.get("split","val")=="val"]
        need(len(rs)==20 and {x["image_id"] for x in rs}==set(vs),"Reference image membership differs: "+name)
        gold={}
        for x in rs:
            i=x["image_id"]
            need(x.get("image_sha256",x.get("source_image_sha256"))==vs[i]["image_sha256"],"Reference image differs")
            gold[i]=labels(checked(p.parent/x["label_path"],x["label_sha256"]).read_bytes(),name+"/"+i)
        need(sum(map(len,gold.values()))==n,"Reference count differs: "+name)
        result[name]=(sha(p),gold)
    return result

def corners(gold,pred,score):
    matched={x["gold_index"]:x["prediction_index"] for x in score["matches"]}
    groups={str(n):Counter(eligible=0,matched=0,within5=0,within10=0) for n in (0,1,2)}
    for i,g in enumerate(gold):
        js=[j for j,k in enumerate(g["keypoints"]) if k[2]==2]; d=groups[str(len(js))]; d["eligible"]+=1
        if i not in matched: continue
        d["matched"]+=1
        if not js: continue
        errors=[]
        for j in js:
            x,y,c=pred[matched[i]]["keypoints"][j]; gx,gy,_=g["keypoints"][j]
            errors.append(math.hypot(x-gx,y-gy) if x is not None and y is not None and 0<=x<1280 and 0<=y<720 and c>=.5 else math.inf)
        for radius in (5,10): d["within"+str(radius)]+=int(max(errors)<=radius)
    return groups

def evaluate(root,work):
    import numpy as np
    import torch
    from ultralytics import YOLO
    m=verify_stage(work); rt=runtime()
    for role in ROLES: verify_arm(work,role,rt)
    entries=m["roles"]["reference"]; refs=read_refs(root,entries)
    detail={}; summary={"source":rev(root),"runtime":rt,"inference":{**INFERENCE,"conf":.13,"matching_iou":.5,"endpoint_conf":.5},"references":{}}
    for role in ROLES:
        model=YOLO(str(work/role/"fit/weights/last.pt")); check_loaded_pose_model(model)
        model.predict(source=np.zeros((720,1280,3),dtype=np.uint8),device="0",conf=.13,**INFERENCE)
        need(not model.predictor.model.end2end and not model.predictor.model.fp16,"Expected ordinary NMS and FP32")
        detail[role]={}
        for e in entries:
            if e["split"]!="val":continue
            r=model.predict(source=str(work/"datasets/reference"/e["image"]),device="0",conf=.13,**INFERENCE)[0]
            need(tuple(r.orig_shape)==(720,1280),"Prediction dimensions differ")
            raw=_prediction_records(r); kept,audit=postclip_nms(raw)
            detail[role][e["id"]]={"raw":raw,"postclip":kept,"filter":audit}
        del model; torch.cuda.empty_cache()
    for name,(h,gs) in refs.items():
        reports={"manifest_sha256":h}
        for view in ("raw","postclip"):
            reports[view]={}
            for role in ROLES:
                scores={}; groups={str(i):Counter() for i in (0,1,2)}
                for i,gold in gs.items():
                    pred=detail[role][i][view]; s=score_image(gold,pred,1280,720,matching_iou=.5,endpoint_conf=.5); scores[i]=s
                    for k,d in corners(gold,pred,s).items(): groups[k].update(d)
                reports[view][role]={"detection":aggregate_scores(list(scores.values())),"corners_by_locatable_count":groups,"per_image":scores}
                print(name,view,role,{k:reports[view][role]["detection"][k] for k in ("tp","fn","fp","duplicate_fp","other_fp")},flush=True)
            a,b=(reports[view][role]["per_image"] for role in ROLES)
            reports[view]["identity_delta"]={i:{"recovered_rows":[x+1 for x in sorted(set(a[i]["unmatched_gold_indices"])-set(b[i]["unmatched_gold_indices"]))],"newly_missed_rows":[x+1 for x in sorted(set(b[i]["unmatched_gold_indices"])-set(a[i]["unmatched_gold_indices"]))]} for i in a}
        summary["references"][name]=reports
    save(work/"evaluation/predictions.json",detail); save(work/"evaluation/summary.json",summary)

def package(root,work,output):
    m=verify_stage(work); a,b=(verify_arm(work,r) for r in ROLES)
    need(a["runtime"]==b["runtime"],"Runtime mismatch")
    need(not output.exists(),"Existing return preserved")
    members={"staging.json":work/"staging.json","evaluation/summary.json":work/"evaluation/summary.json","evaluation/predictions.json":work/"evaluation/predictions.json","source/rebuild_additional_front_pilot_r45.py":Path(__file__).resolve()}
    for role in ("resource-trial",*ROLES):
        for rel in ("receipt.json","progress.json","fit/results.csv","fit/weights/last.pt"):
            p=work/role/rel
            if p.is_file() and not(role=="resource-trial" and rel.startswith("fit/")): members[role+"/"+rel]=p
        p=work/(role+".log")
        if p.is_file():members[p.name]=p
    for role in ROLES:
        for e in m["roles"][role]:members["datasets/"+role+"/"+e["label"]]=work/"datasets"/role/e["label"]
        members["datasets/"+role+"/data.yaml"]=work/"datasets"/role/"data.yaml"
    inventory={n:{"sha256":sha(p),"bytes":p.stat().st_size} for n,p in members.items()}
    t=output.with_suffix(".tmp")
    with zipfile.ZipFile(t,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n,p in sorted(members.items()):z.write(p,n)
        z.writestr("inventory.json",encoded(inventory))
    with zipfile.ZipFile(t) as z:need(z.testzip() is None,"ZIP CRC failure")
    t.replace(output); print("Return",output,sha(output),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=("prepare","trial","train","evaluate","package","verify"))
    p.add_argument("--root",type=Path,required=True);p.add_argument("--work",type=Path,required=True);p.add_argument("--role",choices=ROLES,default="candidate");p.add_argument("--output",type=Path)
    a=p.parse_args(); root=a.root.resolve(); work=a.work.resolve()
    if a.command=="prepare":stage(root,work)
    elif a.command=="verify":verify_stage(work);print("Stage verified")
    elif a.command in ("trial","train"):train(root,work,a.role,a.command=="trial")
    elif a.command=="evaluate":evaluate(root,work)
    else:need(a.output is not None,"Specify output");package(root,work,a.output.resolve())

if __name__=="__main__": main()
