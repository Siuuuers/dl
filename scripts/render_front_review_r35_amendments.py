#!/usr/bin/env python3
"""Render all eleven R35 decisions from verified original-image crops."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.pdfgen import canvas

from front_review_r35 import verify_review
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE, BEFORE, AFTER,
    require, sha, register_fonts, text, para, geometry, display, bounds_for, draw_panel,
)


def footer(pdf, page, total):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, "R34 TO R35  /  ORIGINAL-PIXEL FOLLOW-UP REVIEW  /  25 NOV 2026", MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f"{page} / {total}")


def changes(row):
    if row["action"] == "delete":
        return [(row["row"], "Instance", "Included", "Excluded")]
    before, after = geometry(row["before_line"]), geometry(row["after_line"])
    result = []
    for i, name in enumerate(("Left / xmin", "Top / ymin", "Right / xmax", "Bottom / ymax")):
        if abs(before["box"][i]-after["box"][i]) > 1e-5:
            result.append((row["row"], name, display(before["box"][i]), display(after["box"][i])))
    for i, name in enumerate(("Left corner", "Right corner")):
        if max(abs(a-b) for a,b in zip(before["points"][i],after["points"][i])) > 1e-5:
            def point(p):
                return "Unavailable" if not p[2] else f"({display(p[0])}, {display(p[1])}), v2"
            result.append((row["row"],name,point(before["points"][i]),point(after["points"][i])))
    return result


def cover(pdf, total):
    text(pdf,"ANNOTATION REVIEW  /  ALL 11 REQUESTED ROW DECISIONS",MARGIN,HEIGHT-40,9,MUTED,True)
    text(pdf,"R35 | Shaded seams and owned fronts",MARGIN,HEIGHT-79,27,INK,True)
    y=para(pdf,"A focused follow-up after PR117, across seven original photographs. Each case uses the same crop and scale for original pixels, the accepted R34 label, and the R35 decision.",MARGIN,HEIGHT-100,WIDTH-2*MARGIN,11,16)
    cards=[(9,"amended rows"),(5,"training amendments"),(4,"development amendments"),(2,"development exclusions")]
    cw=(WIDTH-2*MARGIN-36)/4
    for i,(n,label) in enumerate(cards):
        x=MARGIN+i*(cw+12)
        pdf.setFillColor(PALE);pdf.roundRect(x,y-83,cw,64,4,fill=1,stroke=0)
        text(pdf,str(n),x+12,y-49,24,INK,True)
        text(pdf,label,x+12,y-70,8.5,MUTED)
    y-=113
    sections=[
        ("Geometry establishes ownership; colour supports it", "Include the front's own molded rim and shaded portions. For new_000034 row 43, extend the observed left-jaw slot fragment to x640. Pale material in the wider opening near x700 belongs to another column."),
        ("One corner follows its corrected contour", "new_000010 row 24 narrows its right edge to x910 and moves its right corner to (909,491). This is development-only. The five training-row changes alter boxes only."),
        ("Two explicit instance exclusions", "Remove new_000009 rows 56 and 57 as requested. Their previous front interpretations remain in R34. This is a user-directed reference decision, not a confirmed duplicate finding or a model improvement. Retained rows 1-55 keep their identities."),
    ]
    for heading,body in sections:
        text(pdf,heading,MARGIN,y,12.5,INK,True)
        y=para(pdf,body,MARGIN,y-17,WIDTH-2*MARGIN,10,14)-24
    y=para(pdf,"<b>Evaluation:</b> the revised development reference has 659 fronts and 583 two-corner objects. Completed PR117 predictions are assessed against R34 first; their R35 rescore is recorded separately. The next notebook compares R34 and R35 training annotations under the same native recipe.",MARGIN,y+3,WIDTH-2*MARGIN,9,13)
    require(y>44,"Cover overlaps footer")
    footer(pdf,1,total);pdf.showPage()


def reading_page(pdf,total):
    text(pdf,"READING GUIDE  /  SAME CROP, SAME SCALE",MARGIN,HEIGHT-40,9,MUTED,True)
    text(pdf,"Follow the object through the shade",MARGIN,HEIGHT-79,27,INK,True)
    sections=[
        ("What belongs to this front", "Trace the broad frontal plate, its own lip and rim, and the occlusion junctions. Shaded plastic can be target material, and bright metal can be rack. The box encloses the owned visible extrema; internal gripper pixels can lie between separated pieces of one front."),
        ("Top and bottom are the visible shaded transitions", "For new_000034 rows 34-35 and new_000009 row 51, follow each front's own transition and supporting rim. These are the requested new_000034 rows, distinct from the new_000040 pair in R34. A broad upper plane or a neighboring cassette does not extend the box."),
        ("Stored image y grows downward", "Top is ymin and bottom is ymax. Moving upward decreases stored y. Tables name the physical edge and the stored field. The brief positive request for new_000057 row 1 is interpreted as increasing xmax, continuing the prior right-edge request."),
        ("Rings indicate observed physical corners", "Only observed v2 corners are drawn. Hidden or ambiguous corners retain the canonical 0 0 0 triplet. An occluder edge is not a cassette corner. The small new_000010 point follow-up keeps the landmark consistent with the corrected contour."),
        ("Precision, exclusions and scope", "Coordinates are in the original 1280 x 720 image. Visual ambiguity is roughly 1.5-2.5 pixels, with softer seams identified by row; exact saved tokens support replay, not subpixel metrology. Exclusion pages show no R35 box for the removed instance. They do not relabel the surrounding scene. This is a focused 11-row follow-up."),
    ]
    y=HEIGHT-110
    for heading,body in sections:
        text(pdf,heading,MARGIN,y,12,INK,True)
        y=para(pdf,body,MARGIN,y-15,WIDTH-2*MARGIN,10,14)-23
    require(y>43,"Reading guide overlaps footer")
    footer(pdf,2,total);pdf.showPage()


def case(pdf,row,image,page,total):
    iid,index=row["image_id"],row["row"]
    deleted=row["action"]=="delete"
    split="TRAINING" if row["split"]=="train" else "DEVELOPMENT"
    text(pdf,f"{split}  /  {'USER-DIRECTED INSTANCE EXCLUSION' if deleted else 'ACCEPTED AMENDMENT'}",MARGIN,HEIGHT-33,8,MUTED,True)
    text(pdf,f"{iid} | row {index}",MARGIN,HEIGHT-63,22,INK,True)
    bounds=bounds_for([{**row,"after_line":row["after_line"] or row["before_line"]}])
    crop=image.crop(bounds);entries=changes(row)
    top=HEIGHT-105;pw=(WIDTH-2*MARGIN-24)/3;ph=min(233,350-17*len(entries));heights=[]
    panels=(("Original pixels",None,MUTED),("R34 | saved annotation","before_line",BEFORE),
            ("R35 | instance excluded" if deleted else "R35 | reviewed annotation","after_line",AFTER))
    for i,(label,mode,color) in enumerate(panels):
        x=MARGIN+i*(pw+12);text(pdf,label,x,top+11,9,color,True)
        visible_rows=[] if deleted and mode=="after_line" else [row]
        heights.append(draw_panel(pdf,crop,bounds,visible_rows,mode,x,top,pw,ph))
    y=top-max(heights)-24
    text(pdf,"Decision | original pixels",MARGIN,y,10,INK,True)
    text(pdf,"Observed basis and decision",448,y,10,INK,True)
    ty=y-21;columns=[MARGIN+4,MARGIN+34,MARGIN+153,MARGIN+274]
    pdf.setFillColor(PALE);pdf.rect(MARGIN,ty-4,398,17,fill=1,stroke=0)
    for x,label in zip(columns,("Row","Field","R34","R35")):text(pdf,label,x,ty+1,8,MUTED,True)
    ty-=20
    for entry in entries:
        for x,value in zip(columns,entry):text(pdf,str(value),x,ty,8.2,INK)
        pdf.setStrokeColor(RULE);pdf.line(MARGIN,ty-6,MARGIN+398,ty-6);ty-=19
    py=para(pdf,escape(row["reason"]),448,y-16,WIDTH-MARGIN-448,9,13)
    note=("Parent row maps to deleted. Earlier interpretation remains in R34; retained new_000009 rows 1-55 keep their identities." if deleted else
          f"Visual coordinate ambiguity: approximately {row['coordinate_uncertainty_px']} px. Rings mark observed corners only.")
    py=para(pdf,note,448,py-13,WIDTH-MARGIN-448,8,11.5,MUTED)
    yy=min(ty,py)-18
    text(pdf,f"Context crop: {bounds} (x0, y0, x1, y1). Original PNG SHA-256 verified.",MARGIN,yy,7.5,MUTED)
    require(yy>43,f"Case overlaps footer: {iid}:{index}")
    footer(pdf,page,total);pdf.showPage()
    return {"page":page,"image_id":iid,"parent_row":index,"current_row":None if deleted else index,
            "crop_xyxy":bounds,"decision":row["action"],"source_image_sha256":row["source_image_sha256"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inventory",type=Path)
    args=parser.parse_args();root=args.root.resolve();receipt=verify_review(root)
    ledger=json.loads((root/"results/review/front-review-20261125-r35/row-review.json").read_text())
    total=2+len(ledger["rows"]);register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True)
    pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT),pageCompression=1)
    pdf.setTitle("R35 annotation amendments");pdf.setAuthor("Siuuuers/dl annotation review")
    cover(pdf,total);reading_page(pdf,total);pages=[]
    for page,row in enumerate(ledger["rows"],3):
        path=root/"data/images"/row["split"]/(row["image_id"]+".png")
        require(sha(path)==row["source_image_sha256"],"Original image changed")
        pages.append(case(pdf,row,Image.open(path).convert("RGB"),page,total))
    pdf.save()
    inventory={"pdf_pages":total,"decisions":11,"amended_rows":9,"deleted_rows":2,
               "review_manifest_sha256":receipt["review_manifest_sha256"],"pdf_sha256":sha(args.output),"pages":pages}
    if args.inventory:args.inventory.write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"pages":total,"bytes":args.output.stat().st_size,"sha256":sha(args.output)}))


if __name__=="__main__":main()
