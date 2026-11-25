#!/usr/bin/env python3
"""Render the complete PR117 candidate error catalogue from saved fixed inference."""
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
SOURCE = 'results/evaluation/pr117-r34-20261125'


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, '25 NOV 2026 / ORIGINAL PNG PIXELS / R34 DEVELOPMENT DIAGNOSTIC / PR117', MARGIN, 18, 7, MUTED)
    pdf.setFont('Amend', 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f'{page} / {pages}')


def heading(pdf, title, subtitle):
    text(pdf, 'PR117 / CASSETTE FRONTS', MARGIN, HEIGHT-32, 9, MUTED, True)
    text(pdf, title, MARGIN, HEIGHT-67, 24, INK, True)
    para(pdf, subtitle, MARGIN, HEIGHT-83, WIDTH-2*MARGIN, 9.2, 13)


def table(pdf, report, ref, top, title):
    text(pdf, title, MARGIN, top, 10, MUTED, True)
    cols=[MARGIN+6, MARGIN+300, MARGIN+370, MARGIN+460, MARGIN+555, MARGIN+665]
    y=top-23
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,19,fill=1,stroke=0)
    for x,label in zip(cols,('Model','Miss','Unmatched','Duplicate','Both <=5 px','Both <=10 px')):
        text(pdf,label,x,y,8.5,MUTED,True)
    for role,label in (('reference','R33-label native control'),('candidate','R34-label native candidate'),('pr35','PR35 selected')):
        m=report['references'][ref]['models'][role];d=m['detection'];q=m['corner_quality'];y-=19
        values=[label,d['fn'],d['fp'],d['duplicate_fp'],q['radii'][0]['both_visible_objects_within_radius'],q['radii'][1]['both_visible_objects_within_radius']]
        for x,val in zip(cols,values):text(pdf,str(val),x,y,9,INK,role=='candidate')
    return y-19


def cover(pdf, report, pages):
    heading(pdf,'R34 changes: frozen recall is unchanged','The returned PR117 native pair is verified. Fresh inference uses the same CPU runtime and fixed settings as the preserved PR114 and PR35 caches.')
    y=HEIGHT-130
    y=table(pdf,report,'r34',y,'CURRENT R34 REVIEW / 661 FRONTS / 583 TWO-LOCATABLE-CORNER PAIRS')
    y=table(pdf,report,'frozen666',y-8,'FROZEN 666 / PRIMARY HISTORICAL COMPARISON / 584 ELIGIBLE PAIRS')
    y=table(pdf,report,'historical663',y-8,'HISTORICAL 663 / REPORTED SEPARATELY / 584 ELIGIBLE PAIRS')
    y=para(pdf,'<b>At the fixed 0.13 confidence, the candidate adds one R34 miss and recovers none.</b> All 17 missed fronts are partial: four have no locatable corner and thirteen have one. It matches 583/583 two-corner fronts, but only 61/78 partial fronts. Unmatched outputs rise by one; duplicate count stays at 30.',MARGIN,y-5,WIDTH-2*MARGIN,9,12.8)
    y=para(pdf,'Frozen-666 misses and unmatched counts do not improve, and one fewer corner pair falls within five pixels. Historical-663 gains one match. PR35 remains selected. The PR117 control exactly reproduces all 20 PR114 R33-candidate prediction records. Reviewed development results do not establish unseen-session reliability.',MARGIN,y-9,WIDTH-2*MARGIN,8.6,12)
    require(y>49,'Cover text overlaps footer')
    footer(pdf,1,pages);pdf.showPage()


def definitions(pdf, report, pages):
    heading(pdf,'Read each category precisely','The complete catalogue follows: 17 missed fronts, 20 matched visible-corner error rows, and all 92 unmatched candidate outputs.')
    gap=28;w=(WIDTH-2*MARGIN-gap)/2;left=MARGIN;right=MARGIN+w+gap
    def block(x,y,title,body):
        text(pdf,title,x,y,11,INK,True)
        return para(pdf,body,x,y-12,w,9.2,13)-22
    y=HEIGHT-137
    y=block(left,y,'Missed front','A reference row has no assigned output at same-class IoU >=0.50. Nine candidate misses have no output with IoU >=0.10; eight have a closer output below 0.50. These are geometric triage groups, not final physical diagnoses.')
    y=block(left,y,'Unmatched output','An output has no assigned reference after confidence-ordered one-to-one matching. It may be a duplicate, misplaced box, excluded surface, uncertain front, or omitted reference object. Do not call all unmatched outputs hallucinations.')
    y=block(left,y,'Duplicate is already included','A duplicate is an unmatched output that overlaps an already matched reference at IoU >=0.50. Here, <b>92 unmatched = 30 duplicate + 62 other unmatched.</b> Do not add duplicate counts a second time.')
    y=block(left,y,'Matching was checked','A maximum-cardinality check of the same IoU-eligible graph finds no extra possible matches in any image for any of the five checkpoint records across the separately preserved references. The confidence-greedy scorer remains unchanged; no missing TP count is attributable to greedy assignment in this replay.')
    z=HEIGHT-137
    z=block(right,z,'Corner error','On a matched object, a locatable reference corner is more than 5 original-image pixels from its returned point, or the point is missing/invalid at confidence 0.50. Five pixels is an illustrative review radius, not a calibrated robot tolerance. Pair denominators include eligible missed objects.')
    z=block(right,z,'Unavailable reference corners','Reference visibility 0 has no coordinate truth. A predicted hidden corner is allowed but cannot be scored against (0,0). A confident point is not proof that the physical corner is visible. Cyan draws only locatable reference points; magenta draws returned model points.')
    z=block(right,z,'Coordinates and IDs','Stored image x grows right; y grows down. Top is ymin; bottom is ymax. L/R follow image-X order. Rows are R34 reference TXT rows. <b>pIDs are one-based original PR117 candidate output indices.</b> Earlier PR113/PR114 pIDs belong to different caches and must not be transferred.')
    z=block(right,z,'Fixed procedure','Both arms: native Pose26 loss, seed 46, independent epoch-173 initialization, 30 epochs. Inference: Ultralytics 8.4.182, PyTorch 2.10 CPU, FP32, size 1280, one-to-many head, NMS 0.70, box confidence 0.13, endpoint confidence 0.50. One separate confidence 0.10 probe appears in Appendix A. The primary setting is unchanged; no NMS sweep was performed.')
    require(min(y,z)>35,'Definition page overflow')
    footer(pdf,2,pages);pdf.showPage()


def count_page(pdf, candidate, pages):
    heading(pdf,'Every development image is accounted for','Candidate against current R34 reference; the three empty scenes remain in the inventory and contribute to false-output totals.')
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
    heading(pdf,'Every missed front and corner error','Identical original crop and scale in each pair. Cyan: reviewed R34 reference. Magenta: fixed PR117 candidate output.')
    panelw=(WIDTH-2*MARGIN-18)/2
    for slot,case in enumerate(cases):
        top=HEIGHT-126-slot*221;name=case['image_id'];row=case['row'];g=gold[name][row-1];preds=predictions[name]['predictions']
        miss=case['category']=='miss';pi=(case['nearest_prediction_id'] if miss else case['prediction_id']);p=preds[pi-1] if pi else None
        boxes=[g['bbox_xyxy']]
        if p is not None and (not miss or case['nearest_iou']>=.10):boxes.append(p['bbox_xyxy'])
        bounds=bounds_for(boxes)
        title=f"{name} / row {row} / {'missed front' if miss else 'corner error'}"
        text(pdf,title,MARGIN,top,12.2,INK,True)
        text(pdf,'R34 REFERENCE',MARGIN,top-17,8,CYAN,True)
        text(pdf,'PR117 CANDIDATE'+(f' / p{pi}' if pi else ''),MARGIN+panelw+18,top-17,8,MAGENTA,True)
        draw_crop(pdf,images[name],bounds,MARGIN,top-24,panelw,126,[(g,CYAN,f'row {row}',False)])
        models=([(p,MAGENTA,f'p{pi}',True)] if not miss else [(q,MAGENTA,f'p{j+1}',True) for j,q in enumerate(preds) if intersects(q['bbox_xyxy'],bounds)])
        draw_crop(pdf,images[name],bounds,MARGIN+panelw+18,top-24,panelw,126,models)
        if miss:
            description=f"No assigned output at IoU 0.50. {case['locatable_corners']} locatable reference corner(s). "
            description+=f"Closest box: p{pi}, IoU {case['nearest_iou']:.3f}, confidence {case['nearest_confidence']:.3f}." if pi else 'No output overlaps this reference box.'
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


def appendix_footer(pdf, section, index, count):
    pdf.setStrokeColor(RULE);pdf.setLineWidth(.6);pdf.line(MARGIN,31,WIDTH-MARGIN,31)
    label='EXPLORATORY CONFIDENCE PROBE / PRIMARY CONFIDENCE 0.13 UNCHANGED' if section=='A' else 'R35 REFERENCE REPLAY / UNCHANGED MODEL / NO R35 TRAINING RESULT'
    text(pdf,label,MARGIN,18,7,MUTED)
    pdf.setFont('Amend',7);pdf.drawRightString(WIDTH-MARGIN,18,f'APPENDIX {section} / {index} OF {count}')


def probe_summary(pdf, probe):
    heading(pdf,'One bounded confidence probe','Appendix A. Same candidate, images, NMS 0.70, and endpoint settings. Only box confidence changes from 0.13 to 0.10 for this exploratory replay.')
    cols=[MARGIN+5,MARGIN+210,MARGIN+310,MARGIN+430,MARGIN+535,MARGIN+650];y=HEIGHT-137
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,20,fill=1,stroke=0)
    for x,s in zip(cols,['Reference','Miss','Unmatched','Duplicate','Pairs <=5 px','Pairs <=10 px']):text(pdf,s,x,y,8.4,MUTED,True)
    for ref,label in [('frozen666','Frozen 666 / 584 pairs'),('historical663','Historical 663 / 584 pairs'),('r34','R34 review / 583 pairs'),('r35','R35 review / 583 pairs')]:
        if ref not in probe['references']:continue
        a=probe['references'][ref]['primary'];b=probe['references'][ref]['probe'];y-=26
        values=[label]+[f"{a['detection'][k]} -> {b['detection'][k]}" for k in ['fn','fp','duplicate_fp']]+[f"{a['corner_quality']['radii'][i]['both_visible_objects_within_radius']} -> {b['corner_quality']['radii'][i]['both_visible_objects_within_radius']}" for i in range(2)]
        for x,s in zip(cols,values):text(pdf,s,x,y,9,INK,ref=='r34')
    y=para(pdf,'<b>736 original outputs are preserved exactly; 16 low-confidence outputs are added.</b> All three empty scenes still have zero outputs. On R34 and R35, two additional front matches cost 14 additional unmatched outputs: one duplicate and thirteen other unmatched. No eligible two-corner pair improves.',MARGIN,y-35,WIDTH-2*MARGIN,10,14)
    y=para(pdf,'The two recovered diagnostic rows are <b>new_000010:34</b> (probe p39, confidence 0.105308) and <b>new_000090:39</b> (probe p64, confidence 0.103099). These are new matches from added outputs, unlike removing an instance from the reference. Both recovered cases and all 16 added outputs are cropped on the following pages.',MARGIN,y-16,WIDTH-2*MARGIN,9.5,13.5)
    y=para(pdf,'<b>The newly lost paired-control match, new_000010:31, remains missed at 0.10.</b> The control had matched it at confidence 0.131017. This probe therefore does not establish that a small global threshold reduction fixes that particular failure. It does establish the exact recall/extra-output tradeoff at the one tested setting.',MARGIN,y-16,WIDTH-2*MARGIN,9.5,13.5)
    y=para(pdf,'Frozen 666 gains one match for fifteen extra unmatched outputs; historical 663 gains none for sixteen extra unmatched outputs. This repeatedly reviewed set cannot select a deployment threshold by itself. Primary confidence remains0.13. Probe pIDs refer to the separate0.10 cache; the catalogue classifies additions against R34.',MARGIN,y-16,WIDTH-2*MARGIN,9.2,13)
    require(y>45,'Probe summary overflow');appendix_footer(pdf,'A',1,5);pdf.showPage()


def probe_recoveries(pdf, probe, gold, primary, trial, images):
    heading(pdf,'Two extra matches at confidence 0.10','Appendix A. Same original crops. Left: surviving primary outputs. Right: the new low-confidence match. Cyan: R34 reference; magenta: model output.')
    w=(WIDTH-2*MARGIN-18)/2;records=[]
    for slot,case in enumerate(probe['references']['r34']['recovered']):
        name,row=case['image_id'],case['row'];g=gold[name][row-1];preds=trial[name]['predictions'];score=score_image(gold[name],preds,1280,720,.5,.5)
        match=next(m for m in score['matches'] if m['gold_index']==row-1);pid=match['prediction_index']+1;p=preds[pid-1]
        bounds=bounds_for([g['bbox_xyxy'],p['bbox_xyxy']]);top=HEIGHT-126-slot*221
        text(pdf,f'{name} / row {row} / added match from probe p{pid}',MARGIN,top,12,INK,True)
        text(pdf,'PRIMARY CONFIDENCE 0.13',MARGIN,top-17,8,CYAN,True);text(pdf,'EXPLORATORY CONFIDENCE 0.10',MARGIN+w+18,top-17,8,MAGENTA,True)
        old=[(g,CYAN,f'row {row}',False)]+[(q,MAGENTA,f'p{i+1}',True) for i,q in enumerate(primary[name]['predictions']) if intersects(q['bbox_xyxy'],bounds)]
        draw_crop(pdf,images[name],bounds,MARGIN,top-24,w,126,old)
        draw_crop(pdf,images[name],bounds,MARGIN+w+18,top-24,w,126,[(g,CYAN,f'row {row}',False),(p,MAGENTA,f'p{pid}',True)])
        y=para(pdf,f"Added confidence {p['confidence']:.6f}; matched IoU {match['iou']:.6f}. The same row is missed at 0.13 and matched at 0.10 under both R34 and R35. No reference coordinate changed between these two inference settings.",MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4)
        text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'Probe recovery overflow')
        records.append({'image_id':name,'row':row,'probe_prediction_id':pid,'confidence':p['confidence'],'iou':match['iou'],'crop_xyxy':bounds})
    appendix_footer(pdf,'A',2,5);pdf.showPage();return records


def probe_additions(pdf,cases,gold,trial,images,index,physical_page):
    heading(pdf,'Every added low-confidence output','Appendix A. Complete 16-output catalogue at 0.10. Categories use R34; pIDs are from the probe cache. Two additions match; fourteen remain unmatched.')
    gap=20;w=(WIDTH-2*MARGIN-gap)/2;records=[]
    for slot,item in enumerate(cases):
        col=slot%2;row=slot//2;x=MARGIN+col*(w+gap);top=HEIGHT-128-row*148
        name=item['image_id'];pid=item['prediction_id'];p=trial[name]['predictions'][pid-1];g=gold[name][item['gold_row']-1] if item['gold_row'] else None
        boxes=[p['bbox_xyxy']]+([g['bbox_xyxy']] if g is not None and item['iou']>=.1 else []);bounds=bounds_for(boxes,padding=17)
        text(pdf,f"{name} / probe p{pid} / {item['kind'].replace('_',' ')}",x,top,9.3,INK,True)
        objects=([(g,CYAN,f"row {item['gold_row']}",False)] if g is not None and intersects(g['bbox_xyxy'],bounds) else [])+[(p,MAGENTA,f'p{pid}',True)]
        draw_crop(pdf,images[name],bounds,x,top-8,w,96,objects)
        text(pdf,f"Conf {p['confidence']:.6f} | row {item['gold_row'] or '-'} | IoU {item['iou']:.3f}",x,top-116,8,INK)
        text(pdf,'Crop '+str(bounds),x,top-129,7.2,MUTED)
        records.append(dict(item,page=physical_page,crop_xyxy=bounds))
    appendix_footer(pdf,'A',index,5);pdf.showPage();return records


def render_probe(pdf,root,report,gold,primary,images,first_page):
    path=root/SOURCE/'confidence-probe.json';probe=json.loads(path.read_text());cache=root/probe['probe_prediction_path']
    require(sha(cache)==probe['probe_predictions_sha256'],'Probe cache drift')
    require(probe['all_primary_outputs_exactly_preserved'],'Primary output consistency differs')
    header,*rows=[json.loads(x) for x in cache.read_text().splitlines()];trial={r['image_id']:r for r in rows};cases=[]
    for name,r in trial.items():
        score=score_image(gold[name],r['predictions'],r['width'],r['height'],.5,.5)
        matches={m['prediction_index']:m for m in score['matches']}
        for i,p in enumerate(r['predictions']):
            if p['confidence']>=.13:continue
            if i in matches:kind='added_match';row=matches[i]['gold_index']+1;iou=matches[i]['iou']
            else:
                kind='duplicate' if i in score['duplicate_prediction_indices'] else 'other_unmatched'
                gi=max(range(len(gold[name])),key=lambda j:_box_iou(p['bbox_xyxy'],gold[name][j]['bbox_xyxy'])) if gold[name] else None
                row=gi+1 if gi is not None else None;iou=_box_iou(p['bbox_xyxy'],gold[name][gi]['bbox_xyxy']) if gi is not None else 0
            if iou==0:row=None
            cases.append({'image_id':name,'prediction_id':i+1,'kind':kind,'gold_row':row,'iou':iou,'confidence':p['confidence']})
    require(len(cases)==16 and Counter(c['kind'] for c in cases)=={'added_match':2,'duplicate':1,'other_unmatched':13},'Probe addition inventory differs')
    probe_summary(pdf,probe);recovered=probe_recoveries(pdf,probe,gold,primary,trial,images);inventory=[]
    for offset in range(0,len(cases),6):inventory+=probe_additions(pdf,cases[offset:offset+6],gold,trial,images,3+offset//6,first_page+2+offset//6)
    return {'pages':list(range(first_page,first_page+5)),'prediction_sha256':sha(cache),'probe_json_sha256':sha(path),'recovered_crops':recovered,'all_added_outputs':inventory}


def r35_summary(pdf,report,delta):
    heading(pdf,'R35 reference replay: two exclusions','Appendix B. The model and predictions are unchanged. R35 has 659 fronts and 583 eligible corner pairs. These are amended-reference scores, not R35 training results.')
    cols=[MARGIN+5,MARGIN+260,MARGIN+350,MARGIN+440,MARGIN+535,MARGIN+650];y=HEIGHT-137
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,20,fill=1,stroke=0)
    for x,s in zip(cols,['Same model / R34 -> R35','Miss','Unmatched','Duplicate','Pairs <=5 px','Pairs <=10 px']):text(pdf,s,x,y,8.2,MUTED,True)
    for role,label in [('reference','PR117 R33-label control'),('candidate','PR117 R34-label candidate'),('pr35','PR35 selected')]:
        a=report['references']['r34']['models'][role];b=report['references']['r35']['models'][role];y-=25
        values=[label]+[f"{a['detection'][k]} -> {b['detection'][k]}" for k in ['fn','fp','duplicate_fp']]+[f"{a['corner_quality']['radii'][i]['both_visible_objects_within_radius']} -> {b['corner_quality']['radii'][i]['both_visible_objects_within_radius']}" for i in range(2)]
        for x,s in zip(cols,values):text(pdf,s,x,y,8.7,INK,role=='candidate')
    y=para(pdf,'<b>Deleting R34 new_000009 rows 56 and 57 removes two previously missed targets.</b> Candidate true positives remain 644. Its displayed misses fall 17 -> 15 because the reference denominator falls 661 -> 659; no retained object is newly detected. The remaining15 misses are three zero-corner and twelve one-corner fronts. Two-corner eligibility stays 583; visible-corner eligibility falls 1225 -> 1224.',MARGIN,y-30,WIDTH-2*MARGIN,9.6,13.3)
    y=para(pdf,'<b>One retained assignment changes:</b> new_000009:51 switches from p52 to p58 after its box moves upward. p52 becomes other-unmatched; p58 changes from duplicate to matched. Total unmatched stays 92, but its composition becomes 29 duplicates + 63 other unmatched. The 20 corner-error rows and pair counts are unchanged.',MARGIN,y-13,WIDTH-2*MARGIN,9.3,13)
    y=para(pdf,'The development right point on new_000010:24 moves from x911 to x909 at the same y491. The unchanged p26 error rises 1.553809 -> 3.534535px and remains within 5 px. The other two development box changes also retain their assigned outputs. Their exact IoUs are listed below.',MARGIN,y-13,WIDTH-2*MARGIN,9.3,13)
    y-=23
    for item in delta['models']['candidate']['changed_reference_rows']:
        if item['action']=='delete':continue
        a,b=item['r34'],item['r35']
        text(pdf,f"{item['image_id']} : {item['r34_row']}    p{a['matched_prediction_id']} -> p{b['matched_prediction_id']}    assigned IoU {a['matched_iou']:.6f} -> {b['matched_iou']:.6f}",MARGIN,y,8.7,INK);y-=18
    y=para(pdf,'The exclusions follow the user\'s explicit instance decision. Earlier reviews interpreted these regions as cassette fragments; that history is preserved. They are not newly proven duplicates, and the lower miss count is not evidence of model improvement.',MARGIN,y-3,WIDTH-2*MARGIN,8.8,12)
    require(y>44,'R35 summary overflow');appendix_footer(pdf,'B',1,3);pdf.showPage()


def r35_changed_cases(pdf,delta,predictions,images):
    heading(pdf,'Reference edits change an assignment','Appendix B. Identical model outputs and original crops. Cyan: stated reference. Magenta: assigned output for that reference; p52 remains present after its reassignment.')
    w=(WIDTH-2*MARGIN-18)/2;records=[]
    for slot,(name,row) in enumerate([('new_000009',51),('new_000010',24)]):
        item=next(c for c in delta['models']['candidate']['changed_reference_rows'] if c['image_id']==name and c['r34_row']==row)
        old,new=item['r34'],item['r35'];g0,g1=old['reference'],new['reference'];p0=predictions[name]['predictions'][old['matched_prediction_id']-1];p1=predictions[name]['predictions'][new['matched_prediction_id']-1]
        bounds=bounds_for([g0['bbox_xyxy'],g1['bbox_xyxy'],p0['bbox_xyxy'],p1['bbox_xyxy']]);top=HEIGHT-126-slot*221
        text(pdf,f'{name} / row {row}',MARGIN,top,12,INK,True)
        text(pdf,f"R34 / ASSIGNED p{old['matched_prediction_id']}",MARGIN,top-17,8,CYAN,True);text(pdf,f"R35 / ASSIGNED p{new['matched_prediction_id']}",MARGIN+w+18,top-17,8,CYAN,True)
        for x,g,p,pid in [(MARGIN,g0,p0,old['matched_prediction_id']),(MARGIN+w+18,g1,p1,new['matched_prediction_id'])]:draw_crop(pdf,images[name],bounds,x,top-24,w,126,[(g,CYAN,f'row {row}',False),(p,MAGENTA,f'p{pid}',True)])
        description=('p52 IoU 0.601569 -> 0.476323 falls below 0.50. p58 IoU 0.616542 -> 0.545912 stays eligible and becomes the assigned output. The row remains matched; duplicate count drops by one as an existing output changes category.' if row==51 else 'The same p26 stays matched: IoU 0.674006 -> 0.667333. Moving the reference right point 2 px left changes its distance 1.553809 -> 3.534535px; neither version exceeds 5 px. This is not a new prediction or a new corner-error row.')
        y=para(pdf,description,MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4);text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'R35 case overflow')
        records.append({'image_id':name,'row':row,'crop_xyxy':bounds,'r34_prediction_id':old['matched_prediction_id'],'r35_prediction_id':new['matched_prediction_id']})
    appendix_footer(pdf,'B',2,3);pdf.showPage();return records


def r35_exclusions(pdf,delta,images):
    heading(pdf,'Two user-directed reference exclusions','Appendix B. R34 identities 56 and 57 map to null in R35. Rows 1-55 keep their identities. The same pixels remain in the image; no model output has changed.')
    w=(WIDTH-2*MARGIN-18)/2;records=[]
    for slot,row in enumerate([56,57]):
        name='new_000009';item=next(c for c in delta['models']['candidate']['changed_reference_rows'] if c['image_id']==name and c['r34_row']==row)
        require(item['r35'] is None and item['r34']['matched_prediction_id'] is None,'Excluded-instance accounting differs')
        g=item['r34']['reference'];bounds=bounds_for([g['bbox_xyxy']],padding=27);top=HEIGHT-126-slot*221
        text(pdf,f'{name} / R34 row {row} -> R35 excluded',MARGIN,top,12,INK,True)
        text(pdf,'R34 / PREVIOUSLY MISSED REFERENCE',MARGIN,top-17,8,CYAN,True);text(pdf,'R35 / INSTANCE REMOVED FROM REFERENCE',MARGIN+w+18,top-17,8,MUTED,True)
        draw_crop(pdf,images[name],bounds,MARGIN,top-24,w,126,[(g,CYAN,f'old row {row}',False)])
        draw_crop(pdf,images[name],bounds,MARGIN+w+18,top-24,w,126,[])
        description=('This row had one locatable corner and was missed in both primary and probe caches. Excluding it removes one front and one visible-corner target. Its previous cassette-fragment interpretation remains preserved in R34.' if row==56 else 'This row had no locatable corner and was missed in both primary and probe caches. Excluding it removes one front but no two-corner pair. The earlier sequence-supported interpretation remains preserved in R34.')
        y=para(pdf,description,MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4);text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'R35 exclusion overflow')
        records.append({'image_id':name,'r34_row':row,'r35_row':None,'crop_xyxy':bounds,'category':'user_directed_reference_exclusion'})
    appendix_footer(pdf,'B',3,3);pdf.showPage();return records


def render_r35(pdf,root,report,predictions,images,first_page):
    path=root/SOURCE/'r35-reference-delta.json';delta=json.loads(path.read_text());manifest=root/report['references']['r35']['manifest']
    require(sha(manifest)==report['references']['r35']['manifest_sha256']==delta['r35_manifest_sha256'],'R35 source drift')
    r35_summary(pdf,report,delta);changed=r35_changed_cases(pdf,delta,predictions,images);excluded=r35_exclusions(pdf,delta,images)
    return {'pages':list(range(first_page,first_page+3)),'manifest_sha256':sha(manifest),'reference_delta_sha256':sha(path),'changed_crops':changed,'excluded_crops':excluded}


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
    manifest_path=root/report['references']['r34']['manifest'];require(sha(manifest_path)==report['references']['r34']['manifest_sha256'],'Reference manifest drift')
    manifest=json.loads(manifest_path.read_text());gold={};images={}
    for item in manifest['files']:
        if item['split']!='val':continue
        name=item['image_id'];path=root/'data/images/val'/f'{name}.png';label=manifest_path.parent/item['label_path']
        require(sha(path)==item['source_image_sha256']==predictions[name]['image_sha256'],'Original image drift')
        require(sha(label)==item['label_sha256'],'Reference label drift')
        images[name]=Image.open(path).convert('RGB');gold[name]=parse_pose_labels(label.read_text(),images[name].width,images[name].height)
    candidate=report['references']['r34']['models']['candidate'];cases=[dict(x,category='miss') for x in candidate['misses']]+[dict(x,category='corner') for x in candidate['corner_error_rows']]
    unmatched=candidate['unmatched_outputs']
    require(len(cases)==37 and len(candidate['misses'])==17 and len(unmatched)==92,'Reviewed inventory differs')
    require(len({(x['image_id'],x['row']) for x in candidate['misses']})==17,'Repeated missed row')
    require(len({(x['image_id'],x['prediction_id']) for x in unmatched})==92,'Repeated unmatched output')
    for name in predictions:
        score=score_image(gold[name],predictions[name]['predictions'],images[name].width,images[name].height,.5,.5)
        require({i+1 for i in score['unmatched_gold_indices']}=={x['row'] for x in candidate['misses'] if x['image_id']==name},'Miss ledger differs')
        require({i+1 for i in score['unmatched_prediction_indices']}=={x['prediction_id'] for x in unmatched if x['image_id']==name},'Unmatched ledger differs')
    register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True);pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT))
    pdf.setTitle('PR117 R34 misses and errors');pdf.setAuthor('Cassette-front review')
    pages=3+math.ceil(len(cases)/2)+math.ceil(len(unmatched)/6);inventory=[]
    cover(pdf,report,pages);definitions(pdf,report,pages);count_page(pdf,candidate,pages);page=4
    for offset in range(0,len(cases),2):detailed_page(pdf,cases[offset:offset+2],gold,predictions,images,page,pages,inventory);page+=1
    for offset in range(0,len(unmatched),6):unmatched_page(pdf,unmatched[offset:offset+6],gold,predictions,images,page,pages,inventory);page+=1
    require(page-1==pages,'Page inventory mismatch')
    probe_appendix=render_probe(pdf,root,report,gold,predictions,images,pages+1)
    r35_appendix=render_r35(pdf,root,report,predictions,images,pages+6)
    pdf.save()
    result={'schema_version':1,'pages':pages+8,'r34_main_pages':pages,'pdf_sha256':sha(args.output),'source_comparison_sha256':sha(comparison),'candidate_predictions_sha256':sha(cache),'reference_manifest_sha256':sha(manifest_path),'source_image_sha256':{name:predictions[name]['image_sha256'] for name in sorted(images)},'missed_rows':17,'corner_error_rows':20,'unmatched_outputs':92,'duplicates':30,'other_unmatched':62,'inventory':inventory,'confidence_probe_appendix':probe_appendix,'r35_reference_appendix':r35_appendix}
    target=args.inventory or root/SOURCE/'pdf-inventory.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2))


if __name__=='__main__':main()
