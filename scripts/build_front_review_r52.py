#!/usr/bin/env python3
"""Replay the R52 original-pixel side amendments, protected slots and photo gates.

CPU-only provenance and geometry verification. This does not train a model,
establish physical ground truth, or substitute for independent human review.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
import audit_front_geometry_r52 as audit
from build_front_review_r46 import amend_line
from build_front_review_r44 import changed_fields
from evaluate_pose import parse_pose_labels
from verify_pose_candidate import check_geometry

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "data/reviews/additional-fronts-2026-11-27-r52"
PARENT_RELEASE_SHA = "10f07fe62ad8d72e9de54fbc4ece163dd890a027dc20758c190f085ef2bd565e"
PARENT_MANIFEST_SHA = "184abbe67d6c8095b709f69d988665435e0a34680c5827bc6c20dc4947fe06ab"
R44_URL = "https://raw.githubusercontent.com/Siuuuers/dl/61c04d6a6dcb7e9f01ca99c603148b7f1506766b/results/review/front-mining-r44/review.zip"
R44_SHA = "100bfaceb9281bcf8c37c48eaaacf5159df14b0d04dd0ec37d5dcd4fc89c672d"
R44_BYTES = 6599840
EXPECTED_EDITS = {
    "frame_0160:source:015": {"xmax": 600},
    "frame_0160:source:016": {"xmax": 600},
    "frame_0160:source:018": {"xmax": 601},
    "frame_0160:source:019": {"xmax": 538},
    "frame_0160:source:020": {"xmax": 601},
    "frame_0160:source:022": {"xmax": 601},
    "frame_0160:source:024": {"xmin": 416},
    "frame_0160:source:025": {"xmin": 356},
    "frame_0160:source:029": {"xmin": 412.5},
    "frame_0160:source:030": {"xmin": 353.5, "xmax": 415},
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def protect(before, after):
    """An x-side edit may change only YOLO cx/width, never protected fields."""
    old, new = before.split(), after.split()
    need(len(old) == len(new) == 11, "Two-corner schema changed")
    for i in (0, 2, 4, 5, 6, 7, 8, 9, 10):
        need(old[i] == new[i], f"Protected class/y/corner token changed at {i}")


def restore_originals(capsule, destination, rows):
    """Extract only the 12 hash-bound JPEG members; never extract arbitrary paths."""
    raw = Path(capsule).read_bytes()
    need(len(raw) == R44_BYTES and hashlib.sha256(raw).hexdigest() == R44_SHA,
         "R44 source capsule differs")
    images = {r["image_id"]: r["source_image_sha256"] for r in rows}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with zipfile.ZipFile(io.BytesIO(z.read("history/pr131-r43.zip"))) as history:
            for image_id, expected in images.items():
                data = history.read("originals/" + image_id + ".jpg")
                need(hashlib.sha256(data).hexdigest() == expected, "Original photo differs: " + image_id)
                path = Path(destination) / (image_id + ".jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    need(path.read_bytes() == data, "Preserve existing different photo: " + image_id)
                else:
                    path.write_bytes(data)


def replay(root):
    rows, parent_release = audit.load_rows(root)
    need(audit.sha(root/audit.PARENT/"release-status.json") == PARENT_RELEASE_SHA,
         "Immutable R48 release differs")
    need(parent_release["generated_manifest_sha256"] == PARENT_MANIFEST_SHA, "R48 parent lineage differs")
    source = root/SOURCE/"audit-ledger.json"
    ledger = json.loads(source.read_bytes())
    need(ledger["schema_version"] == 1 and ledger["round"] == 52, "R52 ledger version differs")
    need(ledger["parent_audit_sha256"] == audit.AUDIT_SHA
         and ledger["parent_manifest_sha256"] == PARENT_MANIFEST_SHA, "R52 ledger parent differs")
    need(ledger["all_current_rows_inspected"] is True
         and ledger["human_review_performed"] is False, "Review status differs")
    need(ledger["full_originals_inspected"] == sorted({r["image_id"] for r in rows}),
         "Full-photo scope differs")
    decisions = ledger["rows"]
    need([r["stable_id"] for r in decisions] == [r["stable_id"] for r in rows],
         "Stable source identities/order differ")
    need(len(decisions) == len(rows) == 507, "Current row inventory differs")
    after_rows, edits = [], []
    tally = Counter()
    for before, decision in zip(rows, decisions):
        sid = before["stable_id"]
        need(decision["image_id"] == before["image_id"]
             and type(decision["source_row"]) is type(before["source_row"])
             and decision["source_row"] == before["source_row"]
             and decision["output_row"] == before["output_row"]
             and decision["source_image_sha256"] == before["source_image_sha256"], "Identity differs: " + sid)
        need(decision["before_line"] == before["line"], "R48 row differs: " + sid)
        need(decision["updates_px"] == EXPECTED_EDITS.get(sid, {}), "Unfrozen amendment: " + sid)
        need(decision["remaining_holds"] == before["remaining_holds"]
             and decision["hold_resolutions"] == [] and decision["new_holds"] == [],
             "Hold status differs: " + sid)
        need(decision["inspected"] is True and decision["full_original_inspected"] is True
             and decision["human_review_performed"] is False
             and decision["nominal_visual_uncertainty_px"] == 2.0
             and bool(decision["reason"].strip()), "Visual review record incomplete: " + sid)
        after = amend_line(before["line"], decision["updates_px"])
        need(after == decision["after_line"], "Coordinate replay differs: " + sid)
        protect(before["line"], after)
        changed = changed_fields(before["line"], after)
        need(changed == decision["changed_fields"], "Changed-side inventory differs: " + sid)
        box, points = audit.geometry(after)
        expected_sides = [{"field": field, "before_px": before["box_xyxy_px"][audit.EDGES.index(field)],
                           "nominal_after_px": decision["updates_px"][field],
                           "encoded_after_px": box[audit.EDGES.index(field)]} for field in changed]
        need(decision["side_changes"] == expected_sides, "Displayed side coordinates differ: " + sid)
        check_geometry(parse_pose_labels(after, 1280, 720, sid), 1280, 720, sid)
        after_rows.append({**before, "line": after, "box_xyxy_px": box, "points": points})
        if changed:
            edits.append({"stable_id": sid, "image_id": before["image_id"],
                          "source_row": before["source_row"], "output_row": before["output_row"],
                          "before_box_xyxy_px": before["box_xyxy_px"], "after_box_xyxy_px": box,
                          "changed_fields": changed, "updates_px": decision["updates_px"],
                          "side_changes": decision["side_changes"], "reason": decision["reason"],
                          "nominal_visual_uncertainty_px": 2.0,
                          "evidence_png": "amendments/" + sid.replace(":", "-") + ".png"})
        tally["reviewed_rows"] += 1
        tally["amended_rows"] += bool(changed)
        tally["changed_side_boundaries"] += len(changed)
        tally["held_rows"] += bool(before["remaining_holds"])
        tally["held_fields"] += len(before["remaining_holds"])
        tally["locatable_corners"] += len(points)
    tally["retained_rows"] = tally["reviewed_rows"]-tally["amended_rows"]
    tally["protected_corner_slots"] = 2*len(rows)
    tally["unavailable_corners"] = tally["protected_corner_slots"]-tally["locatable_corners"]
    need(tally["amended_rows"] == 10 and tally["changed_side_boundaries"] == 11
         and tally["held_rows"] == 12 and tally["held_fields"] == 31, "Review tallies differ")
    before_screen = audit.make_report(root, rows)
    before_screen["stage"] = "R48_before_R52"
    after_screen = audit.make_report(root, after_rows)
    after_screen["stage"] = "R52_after_visual_amendments"
    after_screen["method"]["inputs"] = "All 507 exact R48 rows with the 10 recorded R52 side-only amendments replayed; no predictions used."
    files = {"geometry-before.json": audit.encoded(before_screen),
             "geometry-after.json": audit.encoded(after_screen)}
    frames = []
    for image_id in sorted({r["image_id"] for r in after_rows}):
        frame_rows = [r for r in after_rows if r["image_id"] == image_id]
        complete = not any(r["remaining_holds"] for r in frame_rows)
        need(complete == (image_id in parent_release["annotation_complete_ids"]), "Whole-photo gate differs")
        payload = ("\n".join(r["line"] for r in frame_rows)+"\n").encode()
        files["proposals/"+image_id+".txt"] = payload
        if complete:
            files["complete-candidates/labels/"+image_id+".txt"] = payload
        frames.append({"image_id": image_id, "source_image_sha256": frame_rows[0]["source_image_sha256"],
                       "annotation_complete": complete, "eligible_for_candidate_export": complete,
                       "human_review_performed": False, "training_split": "unassigned",
                       "rows": [{"stable_id": r["stable_id"], "source_row": r["source_row"],
                                 "output_row": r["output_row"], "after_line": r["line"],
                                 "unresolved_fields": r["remaining_holds"]} for r in frame_rows]})
    manifest = {"schema_version": 1, "round": 52, "kind": "supplementary_side_amendment",
                "created_date": "2026-11-27", "coordinate_space": "Original 1280x720 pixels, x right/y down",
                "parent_manifest_sha256": PARENT_MANIFEST_SHA, "parent_audit_sha256": audit.AUDIT_SHA,
                "audit_sha256": audit.sha(source), "builder_sha256": audit.sha(Path(__file__)),
                "checker_sha256": audit.sha(root/"scripts/audit_front_geometry_r52.py"),
                "review_status": "Agent-assisted visual review of original images; no independent human adjudication",
                "all_current_rows_inspected": True, "human_review_performed": False,
                "counts": dict(tally), "historical_identities": 508,
                "added_or_deleted_rows": 0, "all_corner_tokens_unchanged": True,
                "all_hold_fields_unchanged": True, "all_whole_photo_gates_unchanged": True,
                "annotation_complete_ids": [f["image_id"] for f in frames if f["annotation_complete"]],
                "held_image_ids": [f["image_id"] for f in frames if not f["annotation_complete"]],
                "before_geometry_counts": before_screen["counts"],
                "after_geometry_counts": after_screen["counts"], "amendments": edits, "frames": frames,
                "files": {name: hashlib.sha256(data).hexdigest() for name,data in files.items()},
                "limitations": ["Nominal 1.8-2.6 pixel refinements with 2px visual ambiguity are not calibrated physical measurements.",
                                "Geometry flags are review cues, not confirmed errors; eliminating flags is not the objective.",
                                "No model is trained here and no accuracy gain or review-efficiency gain is claimed.",
                                "Six held photographs remain ineligible as complete training photographs."]}
    files["manifest.json"] = audit.encoded(manifest)
    return manifest, files, rows, after_rows


def render_amendments(manifest, originals, destination):
    from PIL import Image, ImageDraw, ImageFont
    destination = Path(destination)
    output = destination/"amendments"
    output.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = small = ImageFont.load_default()
    for edit in manifest["amendments"]:
        path = Path(originals)/(edit["image_id"]+".jpg")
        frame = next(f for f in manifest["frames"] if f["image_id"] == edit["image_id"])
        need(audit.sha(path) == frame["source_image_sha256"], "Evidence image differs")
        image = Image.open(path).convert("RGB")
        before,after = edit["before_box_xyxy_px"],edit["after_box_xyxy_px"]
        x0,y0,x1,y1=before
        crop=(max(0,math.floor(x0)-10), max(0,math.floor(y0)-8),
              min(1280,math.ceil(x1)+10), min(720,math.ceil(y1)+8))
        raw=image.crop(crop)
        scale=min(5., 420/raw.width, 200/raw.height)
        size=(round(raw.width*scale),round(raw.height*scale))
        raw=raw.resize(size,Image.Resampling.NEAREST)
        panel=Image.new("RGB",(1320,310),"white")
        draw=ImageDraw.Draw(panel)
        draw.text((12,8),edit["stable_id"]+" | original pixels / R48 / R52",font=font,fill="black")
        row=next(r for r in frame["rows"] if r["stable_id"] == edit["stable_id"])
        _,points=audit.geometry(row["after_line"])
        for n,(title,box,color) in enumerate((("Raw",None,None),("R48 before",before,(220,130,0)),("R52 after",after,(0,110,205)))):
            offset=12+n*440
            draw.text((offset,36),title,font=font,fill="black")
            display=raw.copy()
            d=ImageDraw.Draw(display)
            tr=lambda x,y:((x-crop[0])*scale,(y-crop[1])*scale)
            if box:
                d.rectangle((*tr(box[0],box[1]),*tr(box[2],box[3])),outline=color,width=2)
                for p in points:
                    px,py=tr(p["x"],p["y"])
                    d.ellipse((px-3,py-3,px+3,py+3),outline=(0,255,255),width=1)
            panel.paste(display,(offset,62))
        change="; ".join(f"{s['field']} {s['before_px']:.3f} -> {s['nominal_after_px']:.3f} px" for s in edit["side_changes"])
        draw.text((12,266),change,font=small,fill="black")
        draw.text((12,286),"2px nominal visual uncertainty; cyan = unchanged locatable corners; nearest-neighbor enlargement adds no detail.",font=small,fill="black")
        panel.save(destination/edit["evidence_png"])
    (destination/"amendment-index.json").write_bytes(audit.encoded(manifest["amendments"]))


def run(root, output, originals=None, evidence=None, check=False):
    manifest,files,rows,after_rows=replay(root)
    for name,payload in files.items():
        path=output/name
        if check:
            need(path.exists() and path.read_bytes() == payload, "Committed replay differs: "+str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                need(path.read_bytes() == payload, "Preserve existing different output: "+str(path))
            else:
                path.write_bytes(payload)
    if originals:
        images = {r["image_id"]: r["source_image_sha256"] for r in rows}
        for image_id,expected in images.items():
            need(audit.sha(originals/(image_id+".jpg")) == expected, "Original photo hash differs")
    if evidence:
        need(originals is not None,"Evidence rendering requires verified originals")
        render_amendments(manifest,originals,evidence)
    return manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--originals",type=Path)
    p.add_argument("--evidence-dir",type=Path)
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=run(args.root,args.output_dir or args.root/SOURCE,args.originals,args.evidence_dir,args.check)
    print(json.dumps({"counts":result["counts"],"before":result["before_geometry_counts"],
                      "after":result["after_geometry_counts"],
                      "annotation_complete_ids":result["annotation_complete_ids"],
                      "held_image_ids":result["held_image_ids"]},indent=2))
