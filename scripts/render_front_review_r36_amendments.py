#!/usr/bin/env python3
"""Render the R36 verified subset and unresolved landmark evidence from original pixels."""
import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from front_review_r36 import verify_review
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE, BEFORE, AFTER,
    require, sha, register_fonts, text, para, geometry, display,
)

AMBER = colors.HexColor("#9B5E16")
RED = colors.HexColor("#B84040")
TOTAL = 9


def footer(pdf, page):
    pdf.setStrokeColor(RULE); pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, "R35 TO R36  /  ORIGINAL-PIXEL REVIEW  /  26 NOV 2026", MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7); pdf.drawRightString(WIDTH-MARGIN, 18, f"{page} / {TOTAL}")


def panel(pdf, image, bounds, row, mode, x, top, width, max_height):
    crop = image.crop(bounds)
    scale = min(width/crop.width, max_height/crop.height)
    w,h = crop.width*scale,crop.height*scale
    ox,oy = x+(width-w)/2,top-h
    raster = crop.resize((max(crop.width,math.ceil(w*3)),max(crop.height,math.ceil(h*3))), Image.Resampling.NEAREST)
    pdf.drawImage(ImageReader(raster),ox,oy,w,h)
    pdf.setStrokeColor(RULE);pdf.setLineWidth(.5);pdf.rect(ox,oy,w,h,stroke=1,fill=0)
    if mode:
        c=BEFORE if mode=="before_line" else AFTER
        g=geometry(row[mode]);a,b,d,e=g["box"]
        pdf.setStrokeColor(c);pdf.setLineWidth(1)
        pdf.rect(ox+(a-bounds[0])*scale,oy+(bounds[3]-e)*scale,(d-a)*scale,(e-b)*scale,stroke=1,fill=0)
        for label,(qx,qy,v) in zip(("L","R"),g["points"]):
            if not v:continue
            xx,yy=ox+(qx-bounds[0])*scale,oy+(bounds[3]-qy)*scale
            pdf.circle(xx,yy,2.8,stroke=1,fill=0)
            tx=min(ox+w-10,xx+4);ty=min(top-10,yy+5)
            pdf.setFillColor(colors.white);pdf.rect(tx-1,ty-2,10,11,stroke=0,fill=1)
            text(pdf,label,tx,ty,8,c,True)
    return h


def changes(row):
    before,after=geometry(row["before_line"]),geometry(row["after_line"])
    entries=[]
    for i,name in enumerate(("Left / xmin","Top / ymin","Right / xmax","Bottom / ymax")):
        if abs(before["box"][i]-after["box"][i])>1e-5:
            entries.append((name,display(before["box"][i]),display(after["box"][i])))
    if not entries:entries=[("Right / xmax",display(before["box"][2]),"503 (held)")]
    return entries


def cover(pdf):
    text(pdf,"ANNOTATION REVIEW  /  VERIFIED FIELDS AND OPEN QUESTIONS",MARGIN,HEIGHT-40,9,MUTED,True)
    text(pdf,"R36 | Front contours and landmarks",MARGIN,HEIGHT-79,27,INK,True)
    y=para(pdf,"Five requested rows, four original photographs. Four box amendments are supported by the pixels. One entire row and two requested landmark movements remain pending; no point coordinate changes in this round.",MARGIN,HEIGHT-101,WIDTH-2*MARGIN,11,16)
    cards=[(4,"box amendments"),(2,"training rows"),(0,"corner changes"),(3,"pending requests")]
    cw=(WIDTH-2*MARGIN-36)/4
    for i,(n,label) in enumerate(cards):
        x=MARGIN+i*(cw+12);pdf.setFillColor(PALE);pdf.roundRect(x,y-82,cw,64,4,fill=1,stroke=0)
        text(pdf,str(n),x+12,y-48,24,INK,True);text(pdf,label,x+12,y-69,9,MUTED)
    y-=112
    sections=[
        ("Include the owned white rim", "new_000010 row 24 expands from x910 to x913. The pale strip joins the same front and rounded lower contour, before the dark rack separator. Pages 6 and 8 show the observation and the saved labels."),
        ("A box extremum is a different feature from a corner", "new_000034 row 35 restores xmin 511 and physical top 255. The lower-left corner stays at (518.35,297.09): a move to x511 at that height lands on the dark rack transition. Page 9 makes the distinction visible."),
        ("Do not infer an absent marking", "The yellow xmax marker and red outline for new_000006 row 53 were not available. Its R35 row remains exact. The pending questions are carried into the notebook and machine-readable ledger."),
    ]
    for heading,body in sections:
        text(pdf,heading,MARGIN,y,12.5,INK,True)
        y=para(pdf,body,MARGIN,y-17,WIDTH-2*MARGIN,10,14)-23
    y=para(pdf,"<b>Next paired run:</b> two eligible box-only replacements in one training photograph, new_000034 rows 34 and 35. Development changes are separate. The review still contains 659 development fronts and 583 fronts with two locatable corners. No instance is added or removed.",MARGIN,y+3,WIDTH-2*MARGIN,9,13)
    require(y>43,"Cover overlaps footer");footer(pdf,1);pdf.showPage()


def reading_page(pdf):
    text(pdf,"READING GUIDE  /  SAME CROP, SAME SCALE",MARGIN,HEIGHT-40,9,MUTED,True)
    text(pdf,"Trace the front, then place its features",MARGIN,HEIGHT-79,26,INK,True)
    sections=[
        ("Original pixels are the evidence", "Each case shows an untouched crop, the saved R35 box in magenta, and the R36 saved box in green. Enlarged pixels use nearest-neighbor sampling. No image detail is synthesized. L and R identify the saved left and right corner fields."),
        ("Geometry first; colour supports ownership", "Follow a front's continuous plate, lip, rim, stack position and occlusion junctions. Include its white and shaded plastic. Exclude separate rack, gripper and neighboring-only strips. An axis-aligned box can contain internal finger pixels when it encloses a sloping or occluded front."),
        ("Coordinate convention", "Original size is 1280 x 720. Stored x grows rightward and stored y downward. Physical top is ymin; physical bottom is ymax. The tables use these physical edge names. Typical visual ambiguity is approximately 2-2.5 pixels, not calibrated physical distance."),
        ("The point-name correction in new_000034 row 34", "The physical left contour is near x435 and the saved LEFT point at (436,545). The saved right point is at (544.76,530.61). Literal xmin at that right point would collapse the 119-pixel box to 0.24 pixels. R36 follows the visible contour and the stated rack-exclusion goal; both points stay unchanged."),
        ("Three requests remain open", "Row006:53 needs its missing marked crop. Row034:35 needs the intended lower-left turn marked if it differs from the current observed turn. Row010:24 needs the requested point identified: the stored left corner is unavailable behind the jaw, while the visible saved point is the right corner."),
        ("Evaluation and scope", "The four changes are original-pixel annotation decisions, not a model accuracy claim. PR119 is first evaluated against preserved R35; its R36 rescore is reported separately. Earlier review scope is inherited, and this round does not certify the full dataset."),
    ]
    y=HEIGHT-111
    for heading,body in sections:
        text(pdf,heading,MARGIN,y,11.5,INK,True)
        y=para(pdf,body,MARGIN,y-15,WIDTH-2*MARGIN,9.5,13.3)-18
    require(y>43,"Reading page overlaps footer");footer(pdf,2);pdf.showPage()


def case_page(pdf,row,image,page):
    pending=row["disposition"]!="verified"
    status={"verified":"VERIFIED BOX AMENDMENT","partially_verified":"BOX AMENDED / LANDMARK REQUEST PENDING","pending":"MISSING MARKUP / ROW RETAINED"}[row["disposition"]]
    text(pdf,f"{'TRAINING' if row['split']=='train' else 'DEVELOPMENT'}  /  {status}",MARGIN,HEIGHT-33,8,AMBER if pending else MUTED,True)
    text(pdf,f"{row['image_id']} | row {row['row']}",MARGIN,HEIGHT-63,22,INK,True)
    bounds=row["crop_xyxy"];top=HEIGHT-105;pw=(WIDTH-2*MARGIN-24)/3
    heights=[]
    for i,(label,mode,color) in enumerate((("Original pixels",None,MUTED),("R35 | saved annotation","before_line",BEFORE),("R36 | retained" if row["action"]=="retain" else "R36 | verified fields","after_line",AFTER))):
        x=MARGIN+i*(pw+12);text(pdf,label,x,top+11,9,color,True)
        heights.append(panel(pdf,image,bounds,row,mode,x,top,pw,212))
    y=top-max(heights)-25
    text(pdf,"Coordinates | original pixels",MARGIN,y,10,INK,True)
    text(pdf,"Observed basis",448,y,10,INK,True)
    ty=y-22;cols=[MARGIN+4,MARGIN+145,MARGIN+265]
    pdf.setFillColor(PALE);pdf.rect(MARGIN,ty-4,395,17,fill=1,stroke=0)
    for x,label in zip(cols,("Field","R35","R36")):text(pdf,label,x,ty+1,8,MUTED,True)
    ty-=21
    for entry in changes(row):
        for x,value in zip(cols,entry):text(pdf,str(value),x,ty,9,INK)
        pdf.setStrokeColor(RULE);pdf.line(MARGIN,ty-6,MARGIN+395,ty-6);ty-=21
    ty-=10;text(pdf,"Saved corners | both unchanged",MARGIN,ty,9,INK,True);ty-=17
    for name,p in zip(("Left","Right"),geometry(row["after_line"])["points"]):
        value="Unavailable (v0)" if not p[2] else f"({display(p[0])}, {display(p[1])}), v2"
        text(pdf,f"{name}: {value}",MARGIN,ty,8.5,MUTED);ty-=17
    if row["wording_correction"]:
        ty=para(pdf,"<b>Point-name correction:</b> "+escape(row["wording_correction"]),MARGIN,ty-5,395,8.3,11.5,MUTED)
    py=para(pdf,escape(row["reason"]),448,y-17,WIDTH-MARGIN-448,9,12.5)
    if pending:
        py=para(pdf,"<b>Pending:</b> "+escape(row["pending_reason"]),448,py-10,WIDTH-MARGIN-448,8.5,12,AMBER)
    py=para(pdf,f"Visual ambiguity: approximately {row['coordinate_uncertainty_px']} original pixels.",448,py-11,WIDTH-MARGIN-448,8,11.5,MUTED)
    yy=min(ty,py)-19
    text(pdf,f"Crop {tuple(bounds)} (x0,y0,x1,y1). Original PNG SHA-256 verified. No point visibility changes.",MARGIN,yy,7.3,MUTED)
    require(yy>43,f"Case overlaps footer: {row['image_id']}:{row['row']}")
    footer(pdf,page);pdf.showPage()
    return {"page":page,"image_id":row["image_id"],"parent_row":row["row"],"current_row":row["row"],"crop_xyxy":bounds,"decision":row["action"],"disposition":row["disposition"],"source_image_sha256":row["source_image_sha256"]}


def pixel_detail(pdf,image,bounds,x,top,width,height,verticals=(),points=()):
    crop=image.crop(bounds);s=min(width/crop.width,height/crop.height);w,h=crop.width*s,crop.height*s;oy=top-h
    raster=crop.resize((math.ceil(w*3),math.ceil(h*3)),Image.Resampling.NEAREST)
    pdf.drawImage(ImageReader(raster),x,oy,w,h)
    pdf.setStrokeColor(colors.Color(.3,.45,.52,alpha=.35));pdf.setLineWidth(.25)
    for q in range(bounds[0],bounds[2]+1,2):
        xx=x+(q-bounds[0])*s;pdf.line(xx,oy,xx,top);text(pdf,str(q),xx-7,top+6,7,MUTED)
    for q in range(bounds[1],bounds[3]+1,2):
        yy=top-(q-bounds[1])*s;pdf.line(x,yy,x+w,yy);text(pdf,str(q),x-24,yy-2,7,MUTED)
    for q,c in verticals:
        xx=x+(q-bounds[0])*s;pdf.setStrokeColor(c);pdf.setLineWidth(1.7);pdf.line(xx,oy,xx,top)
    for qx,qy,c,label in points:
        xx=x+(qx-bounds[0])*s;yy=top-(qy-bounds[1])*s
        pdf.setStrokeColor(c);pdf.setLineWidth(1.5);pdf.circle(xx,yy,4,stroke=1,fill=0)
        pdf.setFillColor(colors.white);pdf.rect(xx+5,yy+2,12,13,stroke=0,fill=1);text(pdf,label,xx+7,yy+5,8,c,True)
    return h


def ownership_page(pdf,root,rows):
    row=next(r for r in rows if r["image_id"]=="new_000010")
    image=Image.open(root/"data/images/val/new_000010.png").convert("RGB")
    text(pdf,"EXPLANATION  /  NEW_000010 ROW 24",MARGIN,HEIGHT-33,8,MUTED,True)
    text(pdf,"Why this white rim belongs to the front",MARGIN,HEIGHT-67,25,INK,True)
    text(pdf,"Context | R36 box; saved R point unchanged",MARGIN,HEIGHT-98,9,AFTER,True)
    panel(pdf,image,row["crop_xyxy"],row,"after_line",MARGIN,HEIGHT-111,410,242)
    dx=490;dt=HEIGHT-121
    text(pdf,"Original pixels | right rim",dx,HEIGHT-98,9,INK,True)
    pixel_detail(pdf,image,(900,480,924,502),dx,dt,307,282,
        verticals=((910,BEFORE),(913,AFTER)),points=((909,491,INK,"R"),))
    text(pdf,"Magenta x910 | green x913 | R: saved point",dx,dt-300,8,MUTED)
    y=207
    text(pdf,"Contour continuity supports ownership",MARGIN,y,11,INK,True)
    yy=para(pdf,"The pale frontal plate continues into x911-912 and rounds into the same lower contour. The darker separator around x913-914 continues downward into the rack. Another cassette is beyond that separator. The connected outline supports xmax 913; whiteness alone would not.",MARGIN,y-16,410,9.5,13.5)
    yy=para(pdf,"The gray jaw on the left hides the target's left end. Its occlusion boundary is not an observed cassette corner. The left point therefore remains unavailable.",MARGIN,yy-12,410,9.5,13.5)
    ry=para(pdf,"<b>Landmark question:</b> the visible saved point is RIGHT at (909,491). If the requested rightward movement refers to that point, the apparent turn is around x909-913,y489-493. This range is visual ambiguity, not a newly saved point. Please identify the intended feature before changing it.",dx,dt-319,307,9,13,AMBER)
    require(min(yy,ry)>43,"Ownership page overlaps footer");footer(pdf,8);pdf.showPage()
    return {"page":8,"image_id":"new_000010","row":24,"kind":"ownership_explanation","context_crop":row["crop_xyxy"],"detail_crop":[900,480,924,502]}


def landmark_page(pdf,root,rows):
    row=next(r for r in rows if r["image_id"]=="new_000034" and r["row"]==35)
    image=Image.open(root/"data/images/train/new_000034.png").convert("RGB")
    text(pdf,"LANDMARK CHECK  /  NEW_000034 ROW 35",MARGIN,HEIGHT-33,8,MUTED,True)
    text(pdf,"The leftmost edge and the lower-left turn",MARGIN,HEIGHT-67,25,INK,True)
    text(pdf,"Context | restored box; lower-left point retained",MARGIN,HEIGHT-98,9,AFTER,True)
    panel(pdf,image,row["crop_xyxy"],row,"after_line",MARGIN,HEIGHT-111,410,242)
    dx=491;dt=HEIGHT-121
    text(pdf,"Original pixels | lower-left turn",dx,HEIGHT-98,9,INK,True)
    pixel_detail(pdf,image,(503,284,529,308),dx,dt,306,282,
        points=((511,297.09,RED,"A"),(518.3500032,297.09,AFTER,"B")))
    text(pdf,"A: x511 at current y (not applied) | B: saved turn",dx,dt-301,7.8,MUTED)
    y=207
    text(pdf,"One sloping boundary, two different features",MARGIN,y,11,INK,True)
    yy=para(pdf,"An upper portion of this front can set xmin 511 while the lower-left turn lies farther right. The lower silhouette slopes across the crop. At the saved point height y297.09, the projected x511 position is on the dark rack/edge transition.",MARGIN,y-16,410,9.5,13.5)
    yy=para(pdf,"Restore xmin 511 and physical top 255 as requested. Keep bottom 300 and the current observed corner (518.35,297.09). A box edge does not require its corner to share the same x coordinate.",MARGIN,yy-12,410,9.5,13.5)
    ry=para(pdf,"<b>Pending:</b> if a different lower-front feature is intended, mark that point on the crop. The visible pale turn is approximately x516-519,y296-299. Moving a point merely to match a box extremum would change its physical meaning. The red analytical marker A is illustrative and is not a user-supplied outline.",dx,dt-319,306,9,13,AMBER)
    require(min(yy,ry)>43,"Landmark page overlaps footer");footer(pdf,9);pdf.showPage()
    return {"page":9,"image_id":"new_000034","row":35,"kind":"landmark_explanation","context_crop":row["crop_xyxy"],"detail_crop":[503,284,529,308]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inventory",type=Path)
    args=parser.parse_args();root=args.root.resolve();receipt=verify_review(root)
    ledger=json.loads((root/"results/review/front-review-20261126-r36/row-review.json").read_text())
    register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True)
    pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT),pageCompression=1)
    pdf.setTitle("R36 annotation amendments and pending landmarks");pdf.setAuthor("Siuuuers/dl annotation review")
    cover(pdf);reading_page(pdf);pages=[]
    for page,row in enumerate(ledger["rows"],3):
        path=root/"data/images"/row["split"]/(row["image_id"]+".png")
        require(sha(path)==row["source_image_sha256"],"Original image changed")
        pages.append(case_page(pdf,row,Image.open(path).convert("RGB"),page))
    pages.append(ownership_page(pdf,root,ledger["rows"]))
    pages.append(landmark_page(pdf,root,ledger["rows"]))
    pdf.save()
    inventory={"pdf_pages":TOTAL,"requested_rows":5,"amended_rows":4,"fully_held_rows":1,
        "pending_request_rows":3,"changed_points":0,"deleted_rows":0,
        "review_manifest_sha256":receipt["review_manifest_sha256"],"pdf_sha256":sha(args.output),"pages":pages}
    if args.inventory:args.inventory.write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"pages":TOTAL,"bytes":args.output.stat().st_size,"sha256":sha(args.output)}))


if __name__=="__main__":main()
