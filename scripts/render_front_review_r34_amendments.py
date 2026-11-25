#!/usr/bin/env python3
"""Render every R34 user-request decision with original, before and after crops."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.pdfgen import canvas

from front_review_r34 import verify_review
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE, BEFORE, AFTER,
    require, sha, register_fonts, text, para, geometry, display, bounds_for, draw_panel,
)


def footer(pdf, number, total):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, "R33 TO R34  /  ORIGINAL-PIXEL FOLLOW-UP REVIEW  /  25 NOV 2026", MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f"{number} / {total}")


def changes(row):
    before, after = geometry(row["before_line"]), geometry(row["after_line"])
    out = []
    for index, name in enumerate(("Left / xmin", "Top / ymin", "Right / xmax", "Bottom / ymax")):
        if abs(before["box"][index]-after["box"][index]) > 1e-5:
            out.append((row["row"], name, display(before["box"][index]), display(after["box"][index])))
    for index, name in enumerate(("Left corner", "Right corner")):
        a, b = before["points"][index], after["points"][index]
        if max(abs(x-y) for x,y in zip(a,b)) > 1e-5:
            def point(p):
                return "Unavailable (v0)" if p[2] == 0 else f"({display(p[0])}, {display(p[1])}), v2"
            out.append((row["row"], name, point(a), point(b)))
    if not out:
        out.append((row["row"], "Top / ymin retained", "474", "474"))
    return out


def cover(pdf, summary, total):
    text(pdf, "ANNOTATION REVIEW  /  ALL 17 REQUESTED ROW DECISIONS", MARGIN, HEIGHT-40, 9, MUTED, True)
    text(pdf, "R34 | Precise boundaries, visible corners", MARGIN, HEIGHT-79, 27, INK, True)
    y = para(pdf, "Your R33 follow-up, preserved as exact labels and a small reproducible training input. Each case uses the same original crop and scale in all three panels.", MARGIN, HEIGHT-100, WIDTH-2*MARGIN, 11, 16)
    cards = [(16,"amended rows"),(14,"photographs changed"),(7,"training amendments"),(9,"development amendments")]
    cw=(WIDTH-2*MARGIN-36)/4
    for i,(n,label) in enumerate(cards):
        x=MARGIN+i*(cw+12)
        pdf.setFillColor(PALE);pdf.roundRect(x,y-83,cw,64,4,fill=1,stroke=0)
        text(pdf,str(n),x+12,y-49,24,INK,True)
        text(pdf,label,x+12,y-70,8.8,MUTED)
    y-=113
    text(pdf,"Visible pieces through the gripper belong to the same front",MARGIN,y,13,INK,True)
    y=para(pdf,"Restore the previous extents for new_000034 row 43 and new_000040 rows 34-35. The visible cassette pieces inside the opening are part of those same fronts. Their box can span the intervening gripper; their hidden right corners remain unavailable.",MARGIN,y-18,WIDTH-2*MARGIN,10,15)
    text(pdf,"One observed corner restored",MARGIN,y-28,13,INK,True)
    y=para(pdf,"new_000066 row 14 has a locatable lower-left turn at (451,673). Its left annotation changes from unavailable to observed. This round therefore changes boundaries and corners, including one visibility flag; it is not a box-only treatment.",MARGIN,y-46,WIDTH-2*MARGIN,10,15)
    text(pdf,"One requested edge already agrees",MARGIN,y-28,13,INK,True)
    y=para(pdf,"new_000009 p58 maps to TXT row 51. Its top is 473.932 px, while R33 already uses 474 px. Retain the label: 0.068 px is below the useful precision of this image. All requested decisions, including this retention, appear here.",MARGIN,y-46,WIDTH-2*MARGIN,10,15)
    y=para(pdf,"<b>Scope:</b> 17 focused row decisions. The earlier R33 screen is accepted and inherited. No objects are added or deleted. The original 137/20 split, trainer-validation labels and frozen 666/663 scoring references are preserved. R34 training has not run.",MARGIN,y-22,WIDTH-2*MARGIN,9,13)
    require(y>44,"Cover overlaps footer")
    footer(pdf,1,total);pdf.showPage()


def reading_page(pdf,total):
    text(pdf,"REVIEW METHOD  /  WHAT EACH MARK MEANS",MARGIN,HEIGHT-40,9,MUTED,True)
    text(pdf,"Ownership first. Original pixels decide.",MARGIN,HEIGHT-79,27,INK,True)
    items=[
        ("A box encloses observed frontal material", "Include the front's own molded lip and rim, including separate same-instance pieces seen through an opening. Remove a boundary strip only when it contains no owned target material. An axis-aligned rectangle can include internal gripper or background pixels."),
        ("A corner is a physical lower-front turn", "A visible box boundary at an occluder is not automatically a physical corner. Rings are drawn only for observed v2 points. An unavailable point remains the canonical 0 0 0 triplet and is never plotted at the image origin."),
        ("Image y grows downward", "Stored ymin is the top, and stored ymax is the bottom. The user's physical-up instruction means decreasing stored y. Tables use Top/Bottom as well as the stored field name."),
        ("pIDs identify a particular model output", "The cited magenta pIDs come from PR113 candidate, seed47, NMS0.70, in saved prediction order. They are not TXT row numbers and must not be transferred to PR114 outputs. Original pixels decide the final coordinates; accepting one edge does not accept an entire duplicate box."),
        ("Precision and scope", "Typical coordinate uncertainty is 1-2 original pixels, recorded by row. Exact TXT values are retained for reproducibility, not as a claim of subpixel measurement. Unrequested fields and historical snapshots are preserved."),
    ]
    y=HEIGHT-112
    for title,body in items:
        text(pdf,title,MARGIN,y,12,INK,True)
        y=para(pdf,body,MARGIN,y-15,WIDTH-2*MARGIN,10,14)-24
    require(y>40,"Reading guide overlaps footer")
    footer(pdf,2,total);pdf.showPage()


def case(pdf,rows,image,page,total):
    iid=rows[0]["image_id"]; ids=[r["row"] for r in rows]
    split="TRAINING" if rows[0]["split"]=="train" else "DEVELOPMENT"
    retained=all(r["action"]=="retain" for r in rows)
    text(pdf,f"{split}  /  {'RETAINED WITH REASON' if retained else 'ACCEPTED AMENDMENT'}",MARGIN,HEIGHT-33,8,MUTED,True)
    text(pdf,f"{iid} | {'row' if len(ids)==1 else 'rows'} "+", ".join(map(str,ids)),MARGIN,HEIGHT-63,22,INK,True)
    bounds=bounds_for(rows);crop=image.crop(bounds)
    entries=[entry for row in rows for entry in changes(row)]
    top=HEIGHT-105;pw=(WIDTH-2*MARGIN-24)/3;ph=min(233,350-17*len(entries));heights=[]
    for i,(label,mode,color) in enumerate((("Original pixels",None,MUTED),("R33 | saved annotation","before_line",BEFORE),("R34 | reviewed annotation","after_line",AFTER))):
        x=MARGIN+i*(pw+12);text(pdf,label,x,top+11,9,color,True)
        heights.append(draw_panel(pdf,crop,bounds,rows,mode,x,top,pw,ph))
    y=top-max(heights)-24
    text(pdf,"Coordinates | original pixels",MARGIN,y,10,INK,True)
    text(pdf,"Observed basis",448,y,10,INK,True)
    ty=y-21;cols=[MARGIN+4,MARGIN+34,MARGIN+153,MARGIN+274]
    pdf.setFillColor(PALE);pdf.rect(MARGIN,ty-4,398,17,fill=1,stroke=0)
    for x,label in zip(cols,("Row","Field","R33","R34")):text(pdf,label,x,ty+1,8,MUTED,True)
    ty-=20
    for entry in entries:
        for x,value in zip(cols,entry):text(pdf,str(value),x,ty,8.2,INK)
        pdf.setStrokeColor(RULE);pdf.line(MARGIN,ty-6,MARGIN+398,ty-6);ty-=19
    reasons=" ".join(f"<b>Row {r['row']}:</b> "+escape(r['reason']) for r in rows)
    py=para(pdf,reasons,448,y-16,WIDTH-MARGIN-448,9,13)
    py=para(pdf,"Coordinate ambiguity: "+", ".join(str(r["coordinate_uncertainty_px"])+" px" for r in rows)+". Same crop and scale in all panels. Rings show observed corners only.",448,py-13,WIDTH-MARGIN-448,8,11.5,MUTED)
    yy=min(ty,py)-18
    cue=rows[0].get("prediction_cue")
    if cue:
        yy=para(pdf,f"Source cue: PR113 candidate p{cue['prediction_id']} -> {iid} TXT row {ids[0]}. Prediction is an inspection cue; the accepted boundary follows the visible front.",MARGIN,yy,WIDTH-2*MARGIN,8,11,MUTED)-9
    text(pdf,f"Context crop: {bounds} (x0, y0, x1, y1). Source PNG SHA-256 verified.",MARGIN,yy,7.5,MUTED)
    require(yy>43,f"Case overlaps footer: {iid}:{ids}")
    footer(pdf,page,total);pdf.showPage()
    return {"page":page,"image_id":iid,"rows":ids,"crop_xyxy":bounds,"decisions":[r["action"] for r in rows]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inventory",type=Path)
    args=parser.parse_args();root=args.root.resolve();receipt=verify_review(root)
    ledger=json.loads((root/"results/review/front-review-20261125-r34/row-review.json").read_text())
    groups=[]
    for row in ledger["rows"]:
        if groups and row["image_id"]=="new_000040" and groups[-1][0]["image_id"]==row["image_id"]:groups[-1].append(row)
        else:groups.append([row])
    total=2+len(groups);register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True)
    pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT),pageCompression=1)
    pdf.setTitle("R34 annotation amendments");pdf.setAuthor("Siuuuers/dl annotation review")
    cover(pdf,receipt,total);reading_page(pdf,total);pages=[]
    for page,rows in enumerate(groups,3):
        row=rows[0];path=root/"data/images"/row["split"]/(row["image_id"]+".png")
        require(sha(path)==row["source_image_sha256"],"Original image changed")
        pages.append(case(pdf,rows,Image.open(path).convert("RGB"),page,total))
    pdf.save()
    inventory={"pdf_pages":total,"decisions":17,"amended_rows":16,"retained_rows":1,
               "review_manifest_sha256":receipt["review_manifest_sha256"],"pages":pages}
    if args.inventory:args.inventory.write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"pages":total,"bytes":args.output.stat().st_size}))


if __name__=="__main__":main()
