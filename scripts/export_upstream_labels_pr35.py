#!/usr/bin/env python3
"""Build an additive, hash-checked upstream commit ZIP: reviewed labels + PR35 model.

This must NOT overwrite the frozen epoch173 label snapshot. The 92 changed
labels are review data; PR35 was trained on the older 8-file annotation overlay.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REVIEW="data/candidates/complete-visible-fronts-2026-11-14"
REVIEW_CANDIDATE_SHA="9d138930b523fc8d124e116a58aa584a23c30a81d874551fcb5625a1978639a8"
PR35_SHA="420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1"
PR35_BUNDLE_SHA="df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91"
BASE_COMMIT="351eb3cf960ccc34829e2bc72e1fa4645acc6295"
REVIEW_COMMIT="bc34ffe28c43e6d2d3ef49f5563eaa1e6ca2af08"
REF_REVIEW="https://github.com/Siuuuers/dl/tree/"+REVIEW_COMMIT+"/"+REVIEW
REF_MODEL="https://github.com/Siuuuers/dl/pull/35"

def sha(b):
    return hashlib.sha256(b).hexdigest()

def check(condition, message):
    if not condition: raise ValueError(message)

def label_counts(payload, path):
    try: s=payload.decode("utf-8")
    except UnicodeDecodeError as error: raise ValueError(f"invalid label encoding {path}") from error
    n=visible=0
    for i,line in enumerate(s.splitlines(),1):
        if not line.strip(): continue
        fields=line.split()
        check(len(fields)==11,f"11 YOLO pose fields required: {path}:{i}")
        v=list(map(float,fields))
        check(all(__import__('math').isfinite(x) for x in v) and v[0]==0, f"invalid numeric label {path}:{i}")
        check(all(0<=x<=1 for x in v[1:5]), f"out of range bbox {path}:{i}")
        check(v[3]>0 and v[4]>0 and v[7] in (0,2) and v[10] in (0,2), f"invalid box/visibility {path}:{i}")
        for j in (5,8):
            check(v[j+2]==2 or v[j:j+2]==[0.0,0.0], f"noncanonical hidden corner {path}:{i}")
        n+=1; visible+=(v[7]==2)+(v[10]==2)
    return n,visible

def build(repo, archive_path, output):
    check(not archive_path.samefile(output) if output.exists() else True, "do not overwrite input")
    check(not output.exists(), "destination ZIP already exists")
    manifest_path=repo/REVIEW/"manifest.json"
    review_bytes=manifest_path.read_bytes()
    check(sha(review_bytes)==REVIEW_CANDIDATE_SHA,"review manifest differs")
    review=json.loads(review_bytes)
    base=json.loads((repo/"data/manifest.json").read_bytes())
    check(review["kind"]=="training_observed_review_composition" and not review["full_dataset_certified"] and not review["release_ready"],"review status misidentified")
    check(len(review["changes"])==92,"expected 92 changed files")
    check(review["expected_counts"]["train"]==dict(images=137,objects=3791,visible_endpoints=6861,missing_endpoints=721,empty_images=13),"full review counts changed")
    original={x["id"]:x for x in base["records"] if x["split"]=="train"}
    check(len(original)==137 and base["split_summary"]["train"]["instances"]==3772,"original frozen labels changed")
    entries={}; summary=[]; actual_total=actual_visible=0; changed=[]
    for item in sorted(review["changes"],key=lambda a:a["image_id"]):
        iid=item["image_id"]
        check(iid in original and re.fullmatch(r"(?:new|old)_[0-9]{6}",iid),"bad image ID")
        check(item["candidate_label"]==f"labels/train/{iid}.txt","unexpected candidate relative path")
        source=repo/REVIEW/item["candidate_label"]
        payload=source.read_bytes()
        check(sha(payload)==item["candidate_label_sha256"], f"candidate hash mismatch: {iid}")
        check(sha((repo/"data"/original[iid]["label"]).read_bytes())==original[iid]["label_sha256"],f"original hash mismatch {iid}")
        num,vis=label_counts(payload,iid)
        relative=f"{REVIEW}/labels/train/{iid}.txt"
        entries[relative]=payload
        summary.append({"image_id":iid,"path":relative,"sha256":sha(payload),"objects":num,"locatable_corners":vis})
        changed.append(iid)
    all_ids=set(changed)
    for iid,item in original.items():
        if iid in all_ids:
            target=entries[f"{REVIEW}/labels/train/{iid}.txt"]
        else:
            target=(repo/"data"/item["label"]).read_bytes()
        n,v=label_counts(target,iid)
        actual_total+=n; actual_visible+=v
    check((actual_total,actual_visible)==(3791,6861),"composed full review inventory mismatch")
    check(137 in (len(original),) and len(all_ids)==92, "changed photo inventory mismatch")
    model_archive=archive_path.read_bytes()
    check(sha(model_archive)==PR35_BUNDLE_SHA,"PR35 archive checksum differs")
    with zipfile.ZipFile(archive_path) as bundle:
        check(bundle.testzip() is None, "PR35 ZIP CRC error")
        ckpt_name="candidate/fit/weights/last.pt"
        check(ckpt_name in bundle.namelist(), "PR35 candidate weight not found")
        weights=bundle.read(ckpt_name)
        check(sha(weights)==PR35_SHA,"PR35 selected weight SHA mismatch")
        checkpoint=json.loads(bundle.read("candidate/checkpoint-manifest.json"))
        check(checkpoint["checkpoint"]["sha256"]==PR35_SHA,"PR35 checkpoint manifest differs")
    target_weight="weights/cassette_lower_pr35_seed46.pt"
    entries[target_weight]=weights
    ext={"schema_version":1,"kind":"baomai-reviewed-labels-and-selected-weight-export",
         "original_baomai_commit":BASE_COMMIT,"review_commit":REVIEW_COMMIT,
         "original_training_label_count":3772,
         "reviewed_training_label_count":3791,
         "changed_label_files":92,
         "review_release_ready":False,
         "training_images":137,"source_images_included":False,
         "model":{"kind":"trained_pose_checkpoint","selection":"PR35 seed46 candidate final epoch30",
                  "path":target_weight,"sha256":PR35_SHA,"source_pr":REF_MODEL,
                  "trained_on":"2026-11-10 PR35 eight-label overlay; NOT the 92-file full review"},
         "labels":{"overlay_directory":REVIEW+"/labels/train",
                   "manifest_source_sha256":REVIEW_CANDIDATE_SHA,
                   "source_url":REF_REVIEW,"changed":summary},
         "safety":"Overlay is review-only; 24 unresolved training groups and 13 held development-reference rows. PR65 provisional edits excluded. Do not overwrite historical data/labels/train or old data/manifest.json."}
    entries[f"{REVIEW}/UPSTREAM_EXPORT_MANIFEST.json"]=(json.dumps(ext,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    readme=f"""# Baomai reviewed labels + selected PR35 pose checkpoint

**Additive upstream commit, not a replacement of the historical training snapshot.**

Copy the contents of this ZIP into the *root* of the original
\`an94-doll/embedding_vision\` repository checkout. Then:
\`\`\`bash
git lfs install
git add data/candidates/complete-visible-fronts-2026-11-14/weights-do-not-use
\`\`\`
(The path in that placeholder command is deliberately invalid to prevent a mistaken copy-paste!)
Use these actual paths:
\`\`\`bash
git add data/candidates/complete-visible-fronts-2026-11-14/labels/train
git add data/candidates/complete-visible-fronts-2026-11-14/UPSTREAM_EXPORT_MANIFEST.json
git add weights/cassette_lower_pr35_seed46.pt
git add BAOMAI_REVIEW_EXPORT_README.md
git commit -m "Add reviewed cassette labels and selected PR35 checkpoint"
git lfs push origin HEAD
git push origin HEAD
\`\`\`

- Original baomai commit: \`{BASE_COMMIT}\`
- New label overlay: **92 revised .txt files** (complete reviewed source at \`{REVIEW_COMMIT}\`).
- Original frozen training count: **3,772 objects** in 137 images.
- Full reviewed overlay count: **3,791 objects** in the same 137 images. All
  unchanged label .txt files are inherited from the original baseline.
- Selected trained model: \`{target_weight}\`, SHA-256 \`{PR35_SHA}\`,
  YOLO one-class, two actual lower-front corners, epoch-30 seed-46 candidate.
- PR35 actually trained on the older **eight-file** correction set, **not**
  this new 92-file overlay. Do not state that PR35 was trained on 3,791 objects.
- **24 unresolved physical groups across 21 training photos** and **13 held
  development-reference rows** remain; this full reviewed snapshot is not
  certified. The newer, still-provisional PR65 two-file suggestions are omitted.
- Original \`data/images/*\`, \`data/labels/*\`, \`data/annotated/*\`,
  \`data/manifest.json\`, \`data/SHA256SUMS.txt\` and \`metadata/model_card.json\`
  are **untouched**. These fields still accurately describe the historical
  epoch173 dataset rather than falsely claiming its bytes have been changed.
- Original model \`weights/cassette_lower_epoch173.pt\` is also untouched.
- For inference use \`weights/cassette_lower_pr35_seed46.pt\` explicitly.
  Reviewed workflow: \`imgsz=1280\`, box \`conf=0.13\`, NMS \`iou=0.70\`;
  draw each reported corner only with endpoint confidence >= 0.50.
  Changing those inference settings does not retrain the model.

The overlay is intended for review or constructing a new *versioned* training
snapshot with its own consistent manifest and previews. It is not a complete
drop-in training \`data.yaml\` substitution. Do not overwrite the original
\`data/labels/train\` files without generating a fully new dataset manifest
and appropriate annotation previews.
Source review: {REF_REVIEW}
Source PR35: {REF_MODEL}
"""
    readme=readme.replace("""git lfs install
git add data/candidates/complete-visible-fronts-2026-11-14/weights-do-not-use
\`\`\`
(The path in that placeholder command is deliberately invalid to prevent a mistaken copy-paste!)
Use these actual paths:
\`\`\`bash
""","""git lfs install
""")
    entries["BAOMAI_REVIEW_EXPORT_README.md"]=readme.encode()
    file_hashes="\n".join(f"{sha(payload)}  {path}" for path,payload in sorted(entries.items()))+"\n"
    entries["BAOMAI_REVIEW_EXPORT_SHA256SUMS.txt"]=file_hashes.encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path,payload in sorted(entries.items()):
            z.writestr(path,payload)
    with zipfile.ZipFile(output) as z:
        check(z.testzip() is None,"export ZIP CRC error")
        check(set(z.namelist())==set(entries),"archive inventory differs")
        for name,data in entries.items():
            check(sha(z.read(name))==sha(data),f"export round trip mismatch {name}")
    report={"status":"verified","zip_sha256":sha(output.read_bytes()),"zip_bytes":output.stat().st_size,
            "weight_sha256":PR35_SHA,"changed_label_files":len(changed),
            "reviewed_count":actual_total,"reviewed_visible_points":actual_visible,
            "zip_members":len(entries),"untouched_original_snapshot":True}
    output_report=output.with_suffix(".receipt.json")
    output_report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--root",type=Path,default=ROOT)
    a.add_argument("--pr35-zip",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    opts=a.parse_args()
    build(opts.root.resolve(),opts.pr35_zip.resolve(),opts.output.resolve())
