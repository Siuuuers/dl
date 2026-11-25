#!/usr/bin/env python3
"""Render the complete PR114 candidate error catalogue from saved fixed inference."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE,
    register_fonts, text, para, sha, require,
)

CYAN = colors.HexColor('#00C9E8')
MAGENTA = colors.HexColor('#E946BA')
SOURCE = 'results/evaluation/pr114-r33-20261125'


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, '25 NOV 2026 / ORIGINAL PNG PIXELS / R33 DEVELOPMENT DIAGNOSTIC / PR114', MARGIN, 18, 7, MUTED)
    pdf.setFont('Amend', 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f'{page} / {pages}')


def heading(pdf, title, subtitle):
    text(pdf, 'PR114 / CASSETTE FRONTS', MARGIN, HEIGHT-32, 9, MUTED, True)
    text(pdf, title, MARGIN, HEIGHT-67, 24, INK, True)
    para(pdf, subtitle, MARGIN, HEIGHT-83, WIDTH-2*MARGIN, 9.2, 13)


def table(pdf, report, ref, top, title):
    text(pdf, title, MARGIN, top, 10, MUTED, True)
    cols=[MARGIN+6, MARGIN+300, MARGIN+370, MARGIN+460, MARGIN+555, MARGIN+665]
    y=top-23
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,19,fill=1,stroke=0)
    for x,label in zip(cols,('Model','Miss','Unmatched','Duplicate','Both <=5 px','Both <=10 px')):
        text(pdf,label,x,y,8.5,MUTED,True)
    for role,label in (('reference','R31-label native control'),('candidate','R33-label native candidate'),('pr35','PR35 selected')):
        m=report['references'][ref]['models'][role];d=m['detection'];q=m['corner_quality'];y-=19
        values=[label,d['fn'],d['fp'],d['duplicate_fp'],q['radii'][0]['both_visible_objects_within_radius'],q['radii'][1]['both_visible_objects_within_radius']]
        for x,val in zip(cols,values):text(pdf,str(val),x,y,9,INK,role=='candidate')
    return y-19


def cover(pdf, report, pages):
    heading(pdf,'R33 training changes: a small output shift','The returned PR114 pair is verified. Three checkpoints were inferred in one CPU runtime with the unchanged fixed settings.')
    y=HEIGHT-130
    y=table(pdf,report,'r33',y,'CURRENT R33 REVIEW / 661 FRONTS / 583 TWO-LOCATABLE-CORNER PAIRS')
    y=table(pdf,report,'frozen666',y-8,'FROZEN 666 / PRIMARY HISTORICAL COMPARISON / 584 ELIGIBLE PAIRS')
    y=table(pdf,report,'historical663',y-8,'HISTORICAL 663 / REPORTED SEPARATELY / 584 ELIGIBLE PAIRS')
    y=para(pdf,'<b>Prioritize the partial-front misses.</b> The R33 candidate retains the same 15 missed identities as its control. All are partial fronts: four have no locatable corner and eleven have one. It matches 583/583 two-corner fronts, but only 63/78 partial fronts. Two fewer unmatched outputs are both duplicate-count reductions.',MARGIN,y-5,WIDTH-2*MARGIN,9,12.8)
    y=para(pdf,'The frozen-666 candidate gains one match and has three fewer unmatched outputs; historical-663 misses stay unchanged. Corner-pair counts do not improve. PR35 remains selected under the frozen protocol. This repeatedly reviewed development set does not establish unseen-session reliability.',MARGIN,y-9,WIDTH-2*MARGIN,8.6,12)
    require(y>49,'Cover text overlaps footer')
    footer(pdf,1,pages);pdf.showPage()


def definitions(pdf, report, pages):
    heading(pdf,'Read each category precisely','The complete catalogue follows: 15 missed fronts, 17 matched visible-corner error rows, and all 90 unmatched candidate outputs.')
    gap=28;w=(WIDTH-2*MARGIN-gap)/2;left=MARGIN;right=MARGIN+w+gap
    def block(x,y,title,body):
        text(pdf,title,x,y,11,INK,True)
        return para(pdf,body,x,y-12,w,9.2,13)-22
    y=HEIGHT-137
    y=block(left,y,'Missed front','A reference row has no assigned output at same-class IoU >=0.50. Nine candidate misses have no output with IoU >=0.10; six have a closer output below 0.50. These are geometric triage groups, not final physical diagnoses.')
    y=block(left,y,'Unmatched output','An output has no assigned reference after confidence-ordered one-to-one matching. It may be a duplicate, misplaced box, excluded surface, uncertain front, or omitted reference object. Do not call all unmatched outputs hallucinations.')
    y=block(left,y,'Duplicate is already included','A duplicate is an unmatched output that overlaps an already matched reference at IoU >=0.50. Here, <b>90 unmatched = 30 duplicate + 60 other unmatched.</b> Do not add duplicate counts a second time.')
    y=block(left,y,'Matching was checked','A maximum-cardinality check of the same IoU-eligible graph finds no extra possible matches in any image for any of the three models across all four preserved references. The confidence-greedy scorer remains unchanged; no missing TP count is attributable to greedy assignment in this replay.')
    z=HEIGHT-137
    z=block(right,z,'Corner error','On a matched object, a locatable reference corner is more than 5 original-image pixels from its returned point, or the point is missing/invalid at confidence 0.50. Five pixels is an illustrative review radius, not a calibrated robot tolerance. Pair denominators include eligible missed objects.')
    z=block(right,z,'Unavailable reference corners','Reference visibility 0 has no coordinate truth. A predicted hidden corner is allowed but cannot be scored against (0,0). A confident point is not proof that the physical corner is visible. Cyan draws only locatable reference points; magenta draws returned model points.')
    z=block(right,z,'Coordinates and IDs','Stored image x grows right; y grows down. Top is ymin; bottom is ymax. L/R follow image-X order. Rows are R33 reference TXT rows. <b>pIDs are one-based original PR114 candidate output indices.</b> Earlier PR113 pIDs belong to a different checkpoint and must not be transferred.')
    z=block(right,z,'Fixed procedure','Both arms: native Pose26 loss, seed 46, independent epoch-173 initialization, 30 epochs. Inference: Ultralytics 8.4.182, PyTorch 2.10 CPU, FP32, size1280, one-to-many head, NMS0.70, box confidence0.13, endpoint confidence0.50. No additional threshold or NMS search was performed.')
    require(min(y,z)>35,'Definition page overflow')
    footer(pdf,2,pages);pdf.showPage()


def count_page(pdf, candidate, pages):
    heading(pdf,'Every development image is accounted for','Candidate against current R33 reference; the three empty scenes remain in the inventory and contribute to false-output totals.')
    cols=[MARGIN+6,MARGIN+150,MARGIN+240,MARGIN+335,MARGIN+425,MARGIN+540,MARGIN+665]
    labels=['Image','Fronts','Outputs','Misses','Corner errors','Unmatched','Duplicate']
    y=HEIGHT-131;rowh=17.2
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,19,fill=1,stroke=0)
    for x,label in zip(cols,labels):text(pdf,label,x,y,8.5,MUTED,True)
    ec=Counter(x['image_id'] for x in candidate['corner_error_rows'])
    for index,item in enumerate(candidate['per_image']):
        y-=rowh
        if index%2==1:
            pdf.setFillColor(PALE);pdf.rect(MARGIN,y-4,WIDTH-2*MARGIN,rowh,fill=1,stroke=0)
        values=[item['image_id'],item['gold_count'],item['prediction_count'],item['fn'],ec[item['image_id']],item['fp'],item['duplicate_fp']]
        for x,val in zip(cols,values):text(pdf,str(val),x,y,8.5,INK)
    y-=23;d=candidate['detection']
    for x,val in zip(cols,['TOTAL',d['gold_count'],d['prediction_count'],d['fn'],len(candidate['corner_error_rows']),d['fp'],d['duplicate_fp']]):text(pdf,str(val),x,y,9,INK,True)
    para(pdf,'One row can have two corner errors but appears once in the corner-error catalogue. All missed rows and all unmatched pIDs appear exactly once in their respective catalogues. Duplication here refers to model outputs, not duplicate image files or annotation rows.',MARGIN,y-16,WIDTH-2*MARGIN,8.6,12)
    footer(pdf,3,pages);pdf.showPage()


def bounds_for(boxes, width=1280, height=720, padding=22):
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes);x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
    return [max(0,math.floor(x0-padding)),max(0,math.floor(y0-padding)),min(width,math.ceil(x1+padding)),min(height,math.ceil(y1+padding))]


def intersects(box,bounds):
    return min(box[2],bounds[2])>max(box[0],bounds[0]) and min(box[3],bounds[3])>max(box[1],bounds[1])


def draw_crop(pdf, image, bounds, x, top, width, height, objects):
    crop=image.crop(bounds);scale=min(width/crop.width,height/crop.height)
    dw,dh=crop.width*scale,crop.height*scale;dx=x+(width-dw)/2;dy=top-dh
    pdf.setFillColor(PALE);pdf.rect(x,top-height,width,height,fill=1,stroke=0)
    pdf.drawImage(ImageReader(crop),dx,dy,dw,dh,mask='auto')
    pdf.saveState();clip=pdf.beginPath();clip.rect(dx,dy,dw,dh);pdf.clipPath(clip,stroke=0,fill=0)
    def point(px,py):return dx+(px-bounds[0])*scale,dy+dh-(py-bounds[1])*scale
    for obj,color,label,predicted in objects:
        box=obj['bbox_xyxy'];ax,ay=point(box[0],box[1]);bx,by=point(box[2],box[3])
        pdf.setStrokeColor(color);pdf.setLineWidth(1.25);pdf.rect(ax,by,bx-ax,ay-by,fill=0,stroke=1)
        if label:
            anchor=(bx-6*len(label)-3) if predicted else ax+3
            tx=max(dx+2,min(anchor,dx+dw-30));ty=max(dy+4,min(ay-8,dy+dh-9))
            text(pdf,label,tx,ty,7,color,True)
        for side,k in enumerate(obj['keypoints']):
            valid=(k[2]>=.5 if predicted else k[2]==2) and all(v is not None and math.isfinite(v) for v in k[:2])
            if not valid:continue
            px,py=point(*k[:2]);pdf.setStrokeColor(color);pdf.setLineWidth(1.4);pdf.circle(px,py,2.5,stroke=1,fill=0)
            text(pdf,'LR'[side],px+4 if predicted else px-10,py+3 if predicted else py-10,7,color,True)
    pdf.restoreState()


def detailed_page(pdf, cases, gold, predictions, images, page, pages, inventory):
    heading(pdf,'Every missed front and corner error','Identical original crop and scale in each pair. Cyan: reviewed R33 reference. Magenta: fixed PR114 candidate output.')
    panelw=(WIDTH-2*MARGIN-18)/2
    for slot,case in enumerate(cases):
        top=HEIGHT-126-slot*221;name=case['image_id'];row=case['row'];g=gold[name][row-1];preds=predictions[name]['predictions']
        miss=case['category']=='miss';pi=(case['nearest_prediction_id'] if miss else case['prediction_id']);p=preds[pi-1] if pi else None
        boxes=[g['bbox_xyxy']]
        if p is not None and (not miss or case['nearest_iou']>=.10):boxes.append(p['bbox_xyxy'])
        bounds=bounds_for(boxes)
        title=f"{name} / row {row} / {'missed front' if miss else 'corner error'}"
        text(pdf,title,MARGIN,top,12.2,INK,True)
        text(pdf,'R33 REFERENCE',MARGIN,top-17,8,CYAN,True)
        text(pdf,'PR114 CANDIDATE'+(f' / p{pi}' if pi else ''),MARGIN+panelw+18,top-17,8,MAGENTA,True)
        draw_crop(pdf,images[name],bounds,MARGIN,top-24,panelw,126,[(g,CYAN,f'row {row}',False)])
        models=([(p,MAGENTA,f'p{pi}',True)] if not miss else [(q,MAGENTA,f'p{j+1}',True) for j,q in enumerate(preds) if intersects(q['bbox_xyxy'],bounds)])
        draw_crop(pdf,images[name],bounds,MARGIN+panelw+18,top-24,panelw,126,models)
        if miss:
            description=f"No assigned output at IoU 0.50. {case['locatable_corners']} locatable reference corner(s). "
            description+=f"Closest box: p{pi}, IoU {case['nearest_iou']:.3f}, confidence {case['nearest_confidence']:.3f}." if pi else 'No output in this image.'
            description+=' Model panel includes all surviving boxes that intersect this crop.'
        else:
            point_text='; '.join(f"{p['side']} {p['error_px']:.2f} px" if p['error_px'] is not None else f"{p['side']} point unavailable/invalid" for p in case['points'])
            description=f"p{pi}: box confidence {p['confidence']:.3f}; matched IoU {case['iou']:.3f}. Corner error: {point_text}."
        y=para(pdf,escape(description),MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4)
        text(pdf,'Original crop (x0,y0,x1,y1): '+str(bounds),MARGIN,y-12,7.2,MUTED)
        require(y-12>39,'Detailed card overflow')
        inventory.append({'page':page,'category':case['category'],'image_id':name,'row':row,'prediction_id':pi,'crop_xyxy':bounds})
    footer(pdf,page,pages);pdf.showPage()


def unmatched_page(pdf,cases,gold,predictions,images,page,pages,inventory):
    heading(pdf,'Every unmatched output','Complete catalogue, six per page. Cyan is the nearest reviewed front when overlapping; magenta is the named output. Category is a scoring outcome.')
    gap=20;w=(WIDTH-2*MARGIN-gap)/2
    for slot,item in enumerate(cases):
        col=slot%2;row=slot//2;x=MARGIN+col*(w+gap);top=HEIGHT-128-row*148
        name=item['image_id'];pid=item['prediction_id'];p=predictions[name]['predictions'][pid-1]
        g=gold[name][item['gold_row']-1] if item['gold_row'] else None
        boxes=[p['bbox_xyxy']]
        if g is not None and item['iou']>=.1:boxes.append(g['bbox_xyxy'])
        bounds=bounds_for(boxes,padding=17)
        text(pdf,f"{name} / p{pid} / {item['kind'].replace('_',' ')}",x,top,9.5,INK,True)
        objects=[]
        if g is not None and intersects(g['bbox_xyxy'],bounds):objects.append((g,CYAN,f"row {item['gold_row']}",False))
        objects.append((p,MAGENTA,f'p{pid}',True))
        draw_crop(pdf,images[name],bounds,x,top-8,w,96,objects)
        text(pdf,f"Conf {item['confidence']:.3f} | nearest row {item['gold_row'] or '-'} | IoU {item['iou']:.3f}",x,top-116,8,INK)
        text(pdf,'Crop '+str(bounds),x,top-129,7.2,MUTED)
        inventory.append({'page':page,'category':item['kind'],'image_id':name,'prediction_id':pid,'nearest_gold_row':item['gold_row'],'crop_xyxy':bounds})
    footer(pdf,page,pages);pdf.showPage()


def appendix_footer(pdf, page):
    pdf.setStrokeColor(RULE);pdf.setLineWidth(.6)
    pdf.line(MARGIN,31,WIDTH-MARGIN,31)
    text(pdf,'R34 REFERENCE APPENDIX / UNCHANGED PR114 PREDICTIONS / NO R34 TRAINING RESULT',MARGIN,18,7,MUTED)
    pdf.setFont('Amend',7);pdf.drawRightString(WIDTH-MARGIN,18,f'APPENDIX {page} / 2')


def r34_summary(pdf,report):
    heading(pdf,'R34 amendments: cached rescore','Appendix 1 of 2. The model, images, inference, prediction order, and scorer are unchanged. Only the reviewed reference is amended.')
    cols=[MARGIN+5,MARGIN+260,MARGIN+350,MARGIN+440,MARGIN+535,MARGIN+650]
    y=HEIGHT-136
    text(pdf,'R33 -> R34 / 661 FRONTS / 583 TWO-LOCATABLE-CORNER PAIRS',MARGIN,y,10,MUTED,True);y-=26
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,20,fill=1,stroke=0)
    for x,s in zip(cols,['Same model','Miss','Unmatched','Duplicate','Pairs <=5 px','Pairs <=10 px']):text(pdf,s,x,y,8.3,MUTED,True)
    for role,label in [('reference','PR114 R31-label control'),('candidate','PR114 R33-label candidate'),('pr35','PR35 selected')]:
        old=report['references']['r33']['models'][role];new=report['references']['r34']['models'][role];y-=25
        values=[label]+[f"{old['detection'][k]} -> {new['detection'][k]}" for k in ['fn','fp','duplicate_fp']]+[f"{old['corner_quality']['radii'][i]['both_visible_objects_within_radius']} -> {new['corner_quality']['radii'][i]['both_visible_objects_within_radius']}" for i in range(2)]
        for x,s in zip(cols,values):text(pdf,s,x,y,8.6,INK,role=='candidate')
    y=para(pdf,'<b>These are reference changes, not model changes.</b> The original 34-page R33 catalogue remains intact. The R34 candidate replay has 16 missed fronts, 18 matched corner-error rows, and 91 unmatched outputs = 30 duplicates + 61 other unmatched.',MARGIN,y-33,WIDTH-2*MARGIN,10,14)
    y=para(pdf,'<b>Newly missed:</b> new_000088 / row 51. Tightening the reference box to exclude background moves its best output, p57, below IoU 0.50. The same p57 is now other-unmatched. No previously missed row is recovered by the reference amendments.',MARGIN,y-14,WIDTH-2*MARGIN,9.5,13.5)
    y=para(pdf,'<b>New corner-error row:</b> new_000059 / row 2. Moving the reviewed right corner upward changes its distance from the same returned point to just over five pixels. This is a threshold crossing under the amended target; the prediction did not move.',MARGIN,y-14,WIDTH-2*MARGIN,9.5,13.5)
    y=para(pdf,'All 16 candidate misses remain partial fronts: four with no locatable corner, twelve with one. All 583 two-corner fronts still match. The maximum-cardinality check again finds zero additional possible matches for any model or image under R34. Frozen-666 and historical-663 results are unchanged because their references remain frozen.',MARGIN,y-14,WIDTH-2*MARGIN,9.2,13)
    y=para(pdf,'The following page isolates the two newly flagged rows using original-image crops. Cyan marks the stated reference version; magenta marks the identical PR114 candidate output. The rest of the R33 error catalogue should be read with the identity delta recorded here.',MARGIN,y-14,WIDTH-2*MARGIN,9.2,13)
    require(y>44,'R34 summary overflow');appendix_footer(pdf,1);pdf.showPage()


def r34_cases(pdf,gold,newgold,predictions,images):
    heading(pdf,'Two newly flagged rows, unchanged outputs','Appendix 2 of 2. Each pair uses identical original pixels and crop scale. Cyan: stated reference. Magenta: same PR114 output.')
    w=(WIDTH-2*MARGIN-18)/2;records=[]
    for slot,(name,row,pid,kind) in enumerate([('new_000088',51,57,'new_miss'),('new_000059',2,17,'new_corner_error')]):
        top=HEIGHT-126-slot*221;old=gold[name][row-1];new=newgold[name][row-1];p=predictions[name]['predictions'][pid-1]
        bounds=bounds_for([old['bbox_xyxy'],new['bbox_xyxy'],p['bbox_xyxy']])
        oldiou=_box_iou(old['bbox_xyxy'],p['bbox_xyxy']);newiou=_box_iou(new['bbox_xyxy'],p['bbox_xyxy'])
        olderr=math.dist(old['keypoints'][1][:2],p['keypoints'][1][:2]);newerr=math.dist(new['keypoints'][1][:2],p['keypoints'][1][:2])
        title=f"{name} / row {row} / p{pid} / "+('new miss + other-unmatched' if kind=='new_miss' else 'new right-corner error')
        text(pdf,title,MARGIN,top,12,INK,True)
        text(pdf,'R33 REFERENCE + SAME CANDIDATE',MARGIN,top-17,8,CYAN,True)
        text(pdf,'R34 REFERENCE + SAME CANDIDATE',MARGIN+w+18,top-17,8,CYAN,True)
        for x,g in [(MARGIN,old),(MARGIN+w+18,new)]:draw_crop(pdf,images[name],bounds,x,top-24,w,126,[(g,CYAN,f'row {row}',False),(p,MAGENTA,f'p{pid}',True)])
        if kind=='new_miss':
            require(oldiou>=.5 and newiou<.5,'Expected R34 IoU threshold crossing absent')
            description=f"Same p57, confidence {p['confidence']:.6f}. IoU {oldiou:.6f} -> {newiou:.6f}; matching requires >=0.50. Background exclusion makes this a missed reference and an other-unmatched output. No prediction changed."
        else:
            require(olderr<=5 and newerr>5,'Expected R34 point threshold crossing absent')
            description=f"Same p17, confidence {p['confidence']:.6f}. Right-corner distance {olderr:.6f} -> {newerr:.6f} px; review limit 5 px. The box remains matched (IoU {newiou:.6f}). Image y grows downward: the amended point moves upward."
        y=para(pdf,escape(description),MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4)
        text(pdf,'Original crop (x0,y0,x1,y1): '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'Appendix crop overflow')
        records.append({'page':36,'category':kind,'image_id':name,'row':row,'prediction_id':pid,'crop_xyxy':bounds,'r33_iou':oldiou,'r34_iou':newiou,'r33_right_error_px':olderr,'r34_right_error_px':newerr})
    appendix_footer(pdf,2);pdf.showPage();return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory',type=Path)
    args=parser.parse_args();root=args.root.resolve()
    comparison=args.comparison or root/SOURCE/'comparison.json';report=json.loads(comparison.read_text())
    source=report['models']['candidate'];cache=root/source['prediction_path'];require(sha(cache)==source['predictions_sha256'],'Prediction cache drift')
    header,*rows=[json.loads(x) for x in cache.read_text().splitlines()];predictions={x['image_id']:x for x in rows}
    manifest_path=root/report['references']['r33']['manifest'];require(sha(manifest_path)==report['references']['r33']['manifest_sha256'],'Reference manifest drift')
    manifest=json.loads(manifest_path.read_text());gold={};images={}
    for item in manifest['files']:
        if item['split']!='val':continue
        name=item['image_id'];path=root/'data/images/val'/f'{name}.png';label=manifest_path.parent/item['label_path']
        require(sha(path)==item['source_image_sha256']==predictions[name]['image_sha256'],'Original image drift')
        require(sha(label)==item['label_sha256'],'Reference label drift')
        images[name]=Image.open(path).convert('RGB');gold[name]=parse_pose_labels(label.read_text(),images[name].width,images[name].height)
    candidate=report['references']['r33']['models']['candidate'];cases=[dict(x,category='miss') for x in candidate['misses']]+[dict(x,category='corner') for x in candidate['corner_error_rows']]
    unmatched=candidate['unmatched_outputs']
    require(len(cases)==32 and len(candidate['misses'])==15 and len(unmatched)==90,'Reviewed inventory differs')
    require(len({(x['image_id'],x['row']) for x in candidate['misses']})==15,'Repeated missed row')
    require(len({(x['image_id'],x['prediction_id']) for x in unmatched})==90,'Repeated unmatched output')
    for name in predictions:
        score=score_image(gold[name],predictions[name]['predictions'],images[name].width,images[name].height,.5,.5)
        require({i+1 for i in score['unmatched_gold_indices']}=={x['row'] for x in candidate['misses'] if x['image_id']==name},'Miss ledger differs')
        require({i+1 for i in score['unmatched_prediction_indices']}=={x['prediction_id'] for x in unmatched if x['image_id']==name},'Unmatched ledger differs')
    register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True);pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT))
    pdf.setTitle('PR114 R33 misses and errors');pdf.setAuthor('Cassette-front review')
    pages=3+math.ceil(len(cases)/2)+math.ceil(len(unmatched)/6);inventory=[]
    cover(pdf,report,pages);definitions(pdf,report,pages);count_page(pdf,candidate,pages);page=4
    for offset in range(0,len(cases),2):detailed_page(pdf,cases[offset:offset+2],gold,predictions,images,page,pages,inventory);page+=1
    for offset in range(0,len(unmatched),6):unmatched_page(pdf,unmatched[offset:offset+6],gold,predictions,images,page,pages,inventory);page+=1
    require(page-1==pages,'Page inventory mismatch')
    r34_path=root/report['references']['r34']['manifest'];require(sha(r34_path)==report['references']['r34']['manifest_sha256'],'R34 manifest drift')
    r34_manifest=json.loads(r34_path.read_text());newgold={}
    for item in r34_manifest['files']:
        if item['split']!='val':continue
        name=item['image_id'];label=r34_path.parent/item['label_path'];require(sha(label)==item['label_sha256'],'R34 label drift')
        require(item['source_image_sha256']==predictions[name]['image_sha256'],'R34 image drift')
        newgold[name]=parse_pose_labels(label.read_text(),images[name].width,images[name].height)
    r34_summary(pdf,report);appendix=r34_cases(pdf,gold,newgold,predictions,images);pdf.save()
    result={'schema_version':1,'pages':pages+2,'r33_main_pages':pages,'pdf_sha256':sha(args.output),'source_comparison_sha256':sha(comparison),'candidate_predictions_sha256':sha(cache),'reference_manifest_sha256':sha(manifest_path),'source_image_sha256':{name:predictions[name]['image_sha256'] for name in sorted(images)},'missed_rows':15,'corner_error_rows':17,'unmatched_outputs':90,'duplicates':30,'other_unmatched':60,'inventory':inventory,'r34_appendix':{'pages':[35,36],'interpretation':'Reference-only rescore; unchanged model predictions, not R34 training.','manifest_sha256':sha(r34_path),'inventory':appendix}}
    target=args.inventory or root/SOURCE/'pdf-inventory.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))


if __name__=='__main__':main()
