#!/usr/bin/env python3
"""Render R37's supplied-markup amendments and the separate inherited landmark hold."""
import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from front_review_r37 import verify_review
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE, BEFORE, AFTER,
    require, sha, register_fonts, text, para, geometry, display,
)
from render_front_review_r36_amendments import panel, pixel_detail

AMBER = colors.HexColor("#9B5E16")
ORANGE = colors.HexColor("#D27929")
TOTAL = 5
LEDGER = "results/review/front-review-20261126-r37"


def footer(pdf, page):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, "R36 TO R37  /  SUPPLIED MARKINGS AND ORIGINAL PIXELS  /  26 NOV 2026",
         MARGIN, 18, 7, MUTED)
    pdf.setFont("Amend", 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f"{page} / {TOTAL}")


def title(pdf, kicker, heading, size=25):
    text(pdf, kicker, MARGIN, HEIGHT-35, 8.5, MUTED, True)
    text(pdf, heading, MARGIN, HEIGHT-72, size, INK, True)


def cover(pdf):
    title(pdf, "ANNOTATION REVIEW  /  TWO CURRENT MARKED REQUESTS RESOLVED",
          "R37 | Marked boundaries and observed turns", 26)
    y = para(pdf, "The two supplied JPEGs now identify the intended front boundary and lower-left turn. Original-image inspection supports two local amendments. The earlier new_000010 landmark question remains a separate inherited hold.",
             MARGIN, HEIGHT-96, WIDTH-2*MARGIN, 11, 15.5)
    cards = [(2,"supplied marked cases"),(1,"right-edge amendment"),
             (1,"observed-corner amendment"),(1,"inherited pending request")]
    cw = (WIDTH-2*MARGIN-36)/4
    for i,(number,label) in enumerate(cards):
        x = MARGIN+i*(cw+12)
        pdf.setFillColor(PALE)
        pdf.roundRect(x,y-81,cw,64,4,fill=1,stroke=0)
        text(pdf,str(number),x+12,y-47,24,INK,True)
        text(pdf,label,x+12,y-67,8.5,MUTED)
    y -= 111
    sections = [
        ("new_000006 row 53 | right edge 503 to 495",
         "The yellow mark selects the end of the lower sloping front near the finger. The lighter strip reaching the old x503 lies above that selected seam. Keep the other three box edges, the observed left point, and the unavailable right point."),
        ("new_000034 row 35 | left point to (517,294)",
         "The orange mark identifies a soft turn above the broader shaded rim. Move the observed left point from (518.35,297.09) to the integer coordinate (517,294), retaining visibility2. The box and right point remain exact."),
        ("One inherited question remains",
         "Neither new attachment identifies the requested point in new_000010 row24. Its complete R36 row remains unchanged. Page5 keeps that scope distinct from the two resolved cases."),
    ]
    for heading, body in sections:
        text(pdf,heading,MARGIN,y,12.2,INK,True)
        y = para(pdf,body.replace("visibility2","visibility 2").replace("row24","row 24").replace("Page5","Page 5"),
                 MARGIN,y-17,WIDTH-2*MARGIN,10,14)-23
    y = para(pdf,"<b>Controlled next pair:</b> two training rows in two photographs; one box and one observed-point correction. All20 development label files are byte-identical to R36. No class, visibility, instance or row identity changes. Typical visual ambiguity is about2 original pixels; screenshot transforms do not provide subpixel physical certainty.".replace("All20","All 20").replace("about2","about 2"),
             MARGIN,y+3,WIDTH-2*MARGIN,9,13)
    require(y>43,"R37 cover overlaps footer")
    footer(pdf,1)
    pdf.showPage()
    return {"page":1,"kind":"scope_and_summary"}


def triptych(pdf,row,image,top,max_height):
    pw = (WIDTH-2*MARGIN-24)/3
    heights = []
    for i,(label,mode,color) in enumerate((("Original pixels",None,MUTED),
                 ("R36 | saved annotation","before_line",BEFORE),
                 ("R37 | accepted amendment","after_line",AFTER))):
        x = MARGIN+i*(pw+12)
        text(pdf,label,x,top+11,9,color,True)
        heights.append(panel(pdf,image,row["crop_xyxy"],row,mode,x,top,pw,max_height))
    return top-max(heights)


def table(pdf,entries,x,top,width):
    cols=[x+4,x+width*.43,x+width*.73]
    pdf.setFillColor(PALE)
    pdf.rect(x,top-15,width,18,fill=1,stroke=0)
    for xx,label in zip(cols,("Field","R36","R37")):
        text(pdf,label,xx,top-9,8,MUTED,True)
    y=top-31
    for entry in entries:
        for xx,value in zip(cols,entry):
            text(pdf,str(value),xx,y,9,INK)
        pdf.setStrokeColor(RULE)
        pdf.line(x,y-7,x+width,y-7)
        y-=23
    return y


def boundary_page(pdf,row,image):
    title(pdf,"TRAINING  /  SUPPLIED YELLOW BOUNDARY TARGET","new_000006 | row 53",23)
    bottom=triptych(pdf,row,image,HEIGHT-109,205)
    y=bottom-25
    text(pdf,"Coordinates | original pixels",MARGIN,y,10,INK,True)
    text(pdf,"Why the right extent changes",448,y,10,INK,True)
    ty=table(pdf,[("Right / xmax","503","495")],MARGIN,y-16,395)
    ty=para(pdf,"<b>Held edges:</b> left464, top563, bottom586.<br/><b>Held points:</b> left(465,580), v2; right unavailable, v0.".replace("left464","left 464").replace("top563","top 563").replace("bottom586","bottom 586").replace("left(","left ("),
            MARGIN,ty-9,395,9,13)
    ty=para(pdf,"The lower pink stroke reaches about y587.6, but its thickness and the soft seam do not justify a second edge change. Preserve bottom586.".replace("bottom586","bottom 586"),
            MARGIN,ty-13,395,8.8,12.5,MUTED)
    py=para(pdf,"The supplied outline selects a front whose upper seam slopes from about(472,564) to the finger near(495,572). Pale material reaching the previous x503 lies above that seam. The yellow region registers to x493.3-495.8; integer xmax495 follows the specified front and raw contour.".replace("about(","about (").replace("near(","near (").replace("xmax495","xmax 495"),
            448,y-17,WIDTH-MARGIN-448,9.4,13.2)
    py=para(pdf,"The enclosing rectangle may still contain internal finger pixels. That does not make the finger part of the target. Visual ambiguity is approximately2 original pixels.".replace("approximately2","approximately 2"),
            448,py-12,WIDTH-MARGIN-448,9,12.8,MUTED)
    yy=min(ty,py)-21
    text(pdf,"Same crop and scale in all panels. Original PNG and supplied JPEG hashes are bound in the review.",MARGIN,yy,7.4,MUTED)
    require(yy>43,"R37 boundary page overlaps footer")
    footer(pdf,2)
    pdf.showPage()
    return {"page":2,"kind":"current_marked_case","image_id":row["image_id"],"row":row["row"],
            "crop_xyxy":row["crop_xyxy"],"source_image_sha256":row["source_image_sha256"],"changed_fields":row["changed_fields"]}


def landmark_page(pdf,row,image):
    title(pdf,"TRAINING  /  SUPPLIED ORANGE LANDMARK TARGET","new_000034 | row 35",23)
    bottom=triptych(pdf,row,image,HEIGHT-109,180)
    y=bottom-24
    text(pdf,"Observed left point | original pixels",MARGIN,y,10,INK,True)
    ty=table(pdf,[("Left point x","518.35","517"),("Left point y","297.09","294")],MARGIN,y-16,395)
    ty=para(pdf,"<b>Visibility:</b>2 throughout.<br/><b>Box:</b>(511,255,637,300), unchanged.<br/><b>Right point:</b>(635,284), v2, unchanged.".replace("</b>2","</b> 2").replace("</b>(","</b> ("),
            MARGIN,ty-8,395,9,13)
    ty=para(pdf,"The selected turn moves modestly left and mainly upward. The own rim and shaded supporting material still justify the unchanged full box. A landmark need not share a box extremum.",
            MARGIN,ty-12,395,9,12.8,MUTED)
    dx=490
    text(pdf,"Original detail | old B and accepted C",dx,y,9.5,INK,True)
    h=pixel_detail(pdf,image,(509,290,523,302),dx,y-25,290,139,
                   points=((518.3500032,297.09,BEFORE,"B"),(517,294,AFTER,"C")))
    py=para(pdf,"The orange marking maps near(517.2,294.0). C=(517,294) is an integer adjudication with about2px visual ambiguity, not a subpixel measurement.".replace("near(","near (").replace("about2px","about 2 px"),
            dx,y-25-h-15,WIDTH-MARGIN-dx,8.7,12)
    yy=min(ty,py)-18
    text(pdf,"B: previous saved point. C: accepted R37 point. User-supplied markings and registration are shown next.",MARGIN,yy,7.3,MUTED)
    require(yy>43,"R37 landmark page overlaps footer")
    footer(pdf,3)
    pdf.showPage()
    return {"page":3,"kind":"current_marked_case","image_id":row["image_id"],"row":row["row"],
            "crop_xyxy":row["crop_xyxy"],"detail_crop":[509,290,523,302],
            "source_image_sha256":row["source_image_sha256"],"changed_fields":row["changed_fields"]}


def marked_image(pdf,path,bounds,x,top,width,height):
    im=Image.open(path).convert("RGB").crop(bounds)
    scale=min(width/im.width,height/im.height)
    w,h=im.width*scale,im.height*scale
    ox=x+(width-w)/2
    pdf.drawImage(ImageReader(im),ox,top-h,w,h)
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.5)
    pdf.rect(ox,top-h,w,h,fill=0,stroke=1)
    return h


def registration_page(pdf,root,audit):
    title(pdf,"EVIDENCE  /  ACTUAL SUPPLIED JPEG DETAILS","How the markings map to original pixels",25)
    y=para(pdf,"The original JPEG files are preserved unchanged. These displayed details are cropped for legibility; their colour strokes are user-supplied. Registration links the marked regions to saved geometry and raw image pixels.",
           MARGIN,HEIGHT-94,WIDTH-2*MARGIN,10,14)
    cols=[MARGIN,WIDTH/2+12]
    pw=(WIDTH-2*MARGIN-24)/2
    cases=[(audit["user_references"][0],(70,350,1220,1150),"new_000006 row53 | yellow boundary"),
           (audit["user_references"][1],(330,480,1030,920),"new_000034 row35 | orange turn")]
    top=y-30
    for x,(desc,bounds,label) in zip(cols,cases):
        path=root/desc["path"]
        require(sha(path)==desc["sha256"],"Supplied JPEG bytes changed")
        text(pdf,label.replace("row53","row 53").replace("row35","row 35"),x,top+12,10,INK,True)
        marked_image(pdf,path,bounds,x,top,pw,222)
    by=top-244
    text(pdf,"Saved box edges anchor the first image",cols[0],by,10.5,INK,True)
    a=para(pdf,"The teal box is (464,563,503,586). It gives about26.69 screenshot pixels per original x pixel and26.70 per y pixel. The yellow region maps to x493.3-495.8; select xmax495.".replace("about26","about 26").replace("and26","and 26").replace("xmax495","xmax 495"),
           cols[0],by-16,pw,9.2,13)
    a=para(pdf,"The broad pink strokes establish which sloping front is intended. Their thickness is not an exact box-edge measurement; the lower stroke does not trigger an additional bottom correction.",
           cols[0],a-10,pw,9,12.5,MUTED)
    text(pdf,"A/B and the grid anchor the second image",cols[1],by,10.5,INK,True)
    b=para(pdf,"A=(511,297.09) and B=(518.35,297.09) were analytical marks in the earlier report. The user added orange and pink. A/B plus eleven horizontal grid lines give scales near64.75 and64.78; the orange region maps to x515.8-518.7,y292.8-295.2.".replace("near64","near 64").replace("and64","and 64"),
           cols[1],by-16,pw,9.2,13)
    b=para(pdf,"Choose(517,294) from the marked region and raw turn. Transform decimals describe registration; the accepted physical interpretation remains approximate, about2px.".replace("Choose(","Choose (").replace("about2px","about 2 px"),
           cols[1],b-10,pw,9,12.5,MUTED)
    require(min(a,b)>43,"R37 registration page overlaps footer")
    footer(pdf,4)
    pdf.showPage()
    return {"page":4,"kind":"actual_user_markup_registration","attachment_details":[
        {"path":d["path"],"sha256":d["sha256"],"crop_xyxy":list(bb)} for d,bb,_ in cases]}


def inherited_page(pdf,row,image):
    title(pdf,"INHERITED REQUEST  /  NO NEW MARKING FOR THIS CASE","new_000010 | row 24 remains unchanged",24)
    y=para(pdf,"This is the earlier R36 landmark question carried forward, not a third current amendment. Neither supplied screenshot identifies the intended point in this photograph.",
           MARGIN,HEIGHT-96,WIDTH-2*MARGIN,10.5,14.5)
    top=y-31
    pw=(WIDTH-2*MARGIN-24)/2
    for i,(label,mode,col) in enumerate((("Original pixels",None,MUTED),("R37 | exact R36 row retained","after_line",AFTER))):
        x=MARGIN+i*(pw+24)
        text(pdf,label,x,top+12,9.5,col,True)
        panel(pdf,image,row["crop_xyxy"],row,mode,x,top,pw,190)
    bottom=top-min(190,pw/(row["crop_xyxy"][2]-row["crop_xyxy"][0])*(row["crop_xyxy"][3]-row["crop_xyxy"][1]))
    y=bottom-27
    text(pdf,"Preserve the saved physical meaning",MARGIN,y,11.5,INK,True)
    y=para(pdf,"The actual left end is behind the jaw. Its point remains unavailable(v0), and the visible right point remains(909,491), v2. The box remains(810,464,913,495). The jaw edge is not an observed cassette corner. The requested point must be identified before either triplet changes.".replace("unavailable(v0)","unavailable (v0)").replace("remains(","remains ("),
           MARGIN,y-17,WIDTH-2*MARGIN,10,14)
    y-=24
    text(pdf,"What this round changes in training",MARGIN,y,11.5,INK,True)
    y=para(pdf,"R37 compares two eligible training-row amendments against R36 under the same recipe and budget. All20 development files, all class and visibility tokens, all4,445 object identities, and the inherited training exclusions remain unchanged. The retained new_000009 file still has55 rows.".replace("All20","All 20").replace("all4,445","all 4,445").replace("has55","has 55"),
           MARGIN,y-17,WIDTH-2*MARGIN,9.5,13.5)
    y=para(pdf,"A small box can be a real clipped or occluded front. This round introduces no size-based deletion rule and no new instance exclusion. The two current marked requests are resolved; this inherited landmark question remains open.",
           MARGIN,y-12,WIDTH-2*MARGIN,9.3,13,AMBER)
    require(y>43,"R37 inherited page overlaps footer")
    footer(pdf,5)
    pdf.showPage()
    return {"page":5,"kind":"inherited_pending_request","image_id":row["image_id"],"row":row["row"],
            "crop_xyxy":row["crop_xyxy"],"source_image_sha256":row["source_image_sha256"],"row_unchanged":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inventory",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    receipt=verify_review(root)
    audit_path=root/LEDGER/"row-review.json"
    audit=json.loads(audit_path.read_text())
    rows={(r["image_id"],r["row"]):r for r in audit["rows"]}
    images={}
    for key,row in rows.items():
        path=root/"data/images"/row["split"]/(row["image_id"]+".png")
        require(sha(path)==row["source_image_sha256"],"Original PNG changed")
        images[key]=Image.open(path).convert("RGB")
    register_fonts()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT),pageCompression=1)
    pdf.setTitle("R37 annotation amendments from supplied marked references")
    pdf.setAuthor("Siuuuers/dl annotation review")
    pages=[cover(pdf),boundary_page(pdf,rows['new_000006',53],images['new_000006',53]),
           landmark_page(pdf,rows['new_000034',35],images['new_000034',35]),
           registration_page(pdf,root,audit),inherited_page(pdf,rows['new_000010',24],images['new_000010',24])]
    pdf.save()
    inventory={"schema_version":1,"pdf_pages":TOTAL,"current_requested_rows":2,"current_resolved_rows":2,
        "amended_rows":2,"box_amendments":1,"changed_observed_points":1,"inherited_pending_rows":1,
        "added_rows":0,"deleted_rows":0,"visibility_changes":0,"development_label_files_changed":0,
        "review_manifest_sha256":receipt["review_manifest_sha256"],"row_audit_sha256":sha(audit_path),
        "renderer_sha256":sha(Path(__file__)),"pdf_sha256":sha(args.output),"user_references":audit["user_references"],
        "pages":pages}
    if args.inventory:
        args.inventory.parent.mkdir(parents=True,exist_ok=True)
        args.inventory.write_text(json.dumps(inventory,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"pages":TOTAL,"bytes":args.output.stat().st_size,
                      "sha256":sha(args.output),"review_manifest_sha256":receipt["review_manifest_sha256"]}))


if __name__=="__main__":
    main()
