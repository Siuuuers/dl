#!/usr/bin/env python3
"""Render the complete PR121 candidate error catalogue from saved fixed inference."""
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
from score_pr121_review import score_probe_invalid_as_unmatched
from render_front_review_r27_amendments import (
    WIDTH, HEIGHT, MARGIN, INK, MUTED, RULE, PALE,
    register_fonts, text, para, sha, require,
)

CYAN = colors.HexColor('#00C9E8')
MAGENTA = colors.HexColor('#E946BA')
SOURCE = 'results/evaluation/pr121-r36-20261126'


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, '26 NOV 2026 / ORIGINAL PNG PIXELS / R36 DEVELOPMENT DIAGNOSTIC / PR121', MARGIN, 18, 7, MUTED)
    pdf.setFont('Amend', 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f'{page} / {pages}')


def heading(pdf, title, subtitle):
    text(pdf, 'PR121 / CASSETTE FRONTS', MARGIN, HEIGHT-32, 9, MUTED, True)
    text(pdf, title, MARGIN, HEIGHT-67, 24, INK, True)
    para(pdf, subtitle, MARGIN, HEIGHT-83, WIDTH-2*MARGIN, 9.2, 13)


def table(pdf, report, ref, top, title):
    text(pdf, title, MARGIN, top, 10, MUTED, True)
    cols=[MARGIN+6, MARGIN+300, MARGIN+370, MARGIN+460, MARGIN+555, MARGIN+665]
    y=top-23
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,19,fill=1,stroke=0)
    for x,label in zip(cols,('Model','Miss','Unmatched','Duplicate','Both <=5 px','Both <=10 px')):
        text(pdf,label,x,y,8.5,MUTED,True)
    for role,label in (('reference','R35-label native control'),('candidate','R36-label native candidate'),('pr35','PR35 selected')):
        m=report['references'][ref]['models'][role];d=m['detection'];q=m['corner_quality'];y-=19
        values=[label,d['fn'],d['fp'],d['duplicate_fp'],q['radii'][0]['both_visible_objects_within_radius'],q['radii'][1]['both_visible_objects_within_radius']]
        for x,val in zip(cols,values):text(pdf,str(val),x,y,9,INK,role=='candidate')
    return y-19


def cover(pdf, report, pages):
    heading(pdf,'R36: no gain in reviewed recall','The returned PR121 native pair is verified against its exact source. Fresh CPU inference uses the preserved fixed settings and matches the prior runtime headers.')
    y=HEIGHT-130
    y=table(pdf,report,'r36',y,'CURRENT R36 REVIEW / 659 FRONTS / 583 TWO-LOCATABLE-CORNER PAIRS')
    y=table(pdf,report,'frozen666',y-8,'FROZEN 666 / PRIMARY HISTORICAL COMPARISON / 584 ELIGIBLE PAIRS')
    y=table(pdf,report,'historical663',y-8,'HISTORICAL 663 / REPORTED SEPARATELY / 584 ELIGIBLE PAIRS')
    y=para(pdf,'<b>The same 15 R36 fronts remain missed.</b> All are partial: three have no locatable corner and twelve have one. The candidate matches 583/583 two-corner fronts, but only 61/76 partial fronts. Unmatched outputs rise by one; duplicate and corner-error row counts stay unchanged.',MARGIN,y-5,WIDTH-2*MARGIN,9,12.8)
    y=para(pdf,'Frozen 666 gains one corner pair within five pixels, keeps the same missed fronts, and adds one unmatched output. The appendices test whether size exclusions or the already-trained one-to-one head solve the remaining problem. PR35 remains selected. The reviewed development set does not establish reliability on new acquisition sessions.',MARGIN,y-9,WIDTH-2*MARGIN,8.6,12)
    require(y>49,'Cover text overlaps footer')
    footer(pdf,1,pages);pdf.showPage()


def definitions(pdf, report, pages):
    heading(pdf,'Read each category precisely','The complete catalogue follows: 15 missed fronts, 19 matched visible-corner error rows, and all 94 unmatched candidate outputs.')
    gap=28;w=(WIDTH-2*MARGIN-gap)/2;left=MARGIN;right=MARGIN+w+gap
    def block(x,y,title,body):
        text(pdf,title,x,y,11,INK,True)
        return para(pdf,body,x,y-12,w,9.2,13)-22
    y=HEIGHT-137
    y=block(left,y,'Missed front','A reference row has no assigned output at same-class IoU >=0.50. Seven candidate misses have no output with IoU >=0.10; eight have a closer output below 0.50. These are geometric triage groups, not final physical diagnoses.')
    y=block(left,y,'Unmatched output','An output has no assigned reference after confidence-ordered one-to-one matching. It may be a duplicate, misplaced box, excluded surface, uncertain front, or omitted reference object. Do not call all unmatched outputs hallucinations.')
    y=block(left,y,'Duplicate is already included','A duplicate is an unmatched output that overlaps an already matched reference at IoU >=0.50. Here, <b>94 unmatched = 31 duplicate + 63 other unmatched.</b> Do not add duplicate counts a second time.')
    y=block(left,y,'Matching was checked','A maximum-cardinality check of the same IoU-eligible graph finds no extra possible matches in any image for any of the five checkpoint records across the separately preserved references. The confidence-greedy scorer remains unchanged; no missing TP count is attributable to greedy assignment in this replay.')
    z=HEIGHT-137
    z=block(right,z,'Corner error','On a matched object, a locatable reference corner is more than 5 original-image pixels from its returned point, or the point is missing/invalid at confidence 0.50. Five pixels is an illustrative review radius, not a calibrated robot tolerance. Pair denominators include eligible missed objects.')
    z=block(right,z,'Unavailable reference corners','Reference visibility 0 has no coordinate truth. A predicted hidden corner is allowed but cannot be scored against (0,0). A confident point is not proof that the physical corner is visible. Cyan draws only locatable reference points; magenta draws returned model points.')
    z=block(right,z,'Coordinates and IDs','Stored image x grows right; y grows down. Top is ymin; bottom is ymax. L/R follow image-X order. Rows are R36 reference TXT rows. <b>pIDs are one-based original PR121 candidate output indices.</b> Earlier PR113/PR114/PR117/PR119 pIDs belong to different caches and must not be transferred.')
    z=block(right,z,'Fixed procedure','Both arms: native Pose26 loss, seed 46, independent epoch-173 initialization, 30 epochs. Inference: Ultralytics 8.4.182, PyTorch 2.10 CPU, FP32, size 1280, one-to-many head, NMS 0.70, box confidence 0.13, endpoint confidence 0.50. No confidence or NMS sweep was performed. A separate appendix reports one same-checkpoint one-to-one-head probe; primary inference remains unchanged.')
    require(min(y,z)>35,'Definition page overflow')
    footer(pdf,2,pages);pdf.showPage()


def count_page(pdf, candidate, pages):
    heading(pdf,'Every development image is accounted for','Candidate against current R36 reference; the three empty scenes remain in the inventory and contribute to false-output totals.')
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
    heading(pdf,'Every missed front and corner error','Identical original crop and scale in each pair. Cyan: reviewed R36 reference. Magenta: fixed PR121 candidate output.')
    panelw=(WIDTH-2*MARGIN-18)/2
    for slot,case in enumerate(cases):
        top=HEIGHT-126-slot*221;name=case['image_id'];row=case['row'];g=gold[name][row-1];preds=predictions[name]['predictions']
        miss=case['category']=='miss';pi=(case['nearest_prediction_id'] if miss else case['prediction_id']);p=preds[pi-1] if pi else None
        boxes=[g['bbox_xyxy']]
        if p is not None and (not miss or case['nearest_iou']>=.10):boxes.append(p['bbox_xyxy'])
        bounds=bounds_for(boxes)
        title=f"{name} / row {row} / {'missed front' if miss else 'corner error'}"
        text(pdf,title,MARGIN,top,12.2,INK,True)
        text(pdf,'R36 REFERENCE',MARGIN,top-17,8,CYAN,True)
        text(pdf,'PR121 CANDIDATE'+(f' / p{pi}' if pi else ''),MARGIN+panelw+18,top-17,8,MAGENTA,True)
        draw_crop(pdf,images[name],bounds,MARGIN,top-24,panelw,126,[(g,CYAN,f'row {row}',False)])
        models=([(p,MAGENTA,f'p{pi}',True)] if not miss else [(q,MAGENTA,f'p{j+1}',True) for j,q in enumerate(preds) if intersects(q['bbox_xyxy'],bounds)])
        draw_crop(pdf,images[name],bounds,MARGIN+panelw+18,top-24,panelw,126,models)
        if miss:
            triage='Very low overlap (best IoU <0.10)' if case['nearest_iou']<.1 else 'Output present below matching gate'
            description=f"{triage}. No assigned output at IoU 0.50. {case['locatable_corners']} locatable reference corner(s). "
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


def appendix_footer(pdf, section, label, number=1, total=1):
    pdf.setStrokeColor(RULE);pdf.setLineWidth(.6);pdf.line(MARGIN,31,WIDTH-MARGIN,31)
    text(pdf,label,MARGIN,18,7,MUTED)
    pdf.setFont('Amend',7);pdf.drawRightString(WIDTH-MARGIN,18,f'APPENDIX {section} / {number} OF {total}')


def read_reference(root, report, ref, predictions, images):
    path=root/report['references'][ref]['manifest']
    require(sha(path)==report['references'][ref]['manifest_sha256'],'Reference manifest drift')
    manifest=json.loads(path.read_text());gold={}
    for item in manifest['files']:
        if item.get('split','val')!='val':continue
        name=item['image_id'];label=path.parent/item['label_path']
        require(sha(label)==item['label_sha256'],'Reference label drift')
        require(item.get('source_image_sha256',item.get('image_sha256'))==predictions[name]['image_sha256'],'Reference image drift')
        gold[name]=parse_pose_labels(label.read_text(),images[name].width,images[name].height)
    return gold


def compact_table(pdf,title,headers,rows,x,top,width,ratios=None,font=8.5):
    text(pdf,title,x,top,10,INK,True);y=top-23
    ratios=ratios or [1/len(headers)]*len(headers);positions=[x+5]
    for ratio in ratios[:-1]:positions.append(positions[-1]+width*ratio)
    pdf.setFillColor(PALE);pdf.rect(x,y-5,width,19,fill=1,stroke=0)
    for px,label in zip(positions,headers):text(pdf,str(label),px,y,font,MUTED,True)
    for index,row in enumerate(rows):
        y-=20
        if index%2==1:pdf.setFillColor(PALE);pdf.rect(x,y-5,width,19,fill=1,stroke=0)
        for px,value in zip(positions,row):text(pdf,str(value),px,y,font,INK)
    return y-20


def size_pages(pdf,root,report,predictions,images,gold,start):
    path=root/SOURCE/'size-visibility-audit.json';audit=json.loads(path.read_text());data=audit['references']['r36']['models']['candidate']
    require(audit['references']['r36']['reference_manifest_sha256']==report['references']['r36']['manifest_sha256'],'Size audit reference drift')
    require(audit['source_comparison_sha256']==sha(root/SOURCE/'comparison.json'),'Size audit comparison drift')
    rows=data['by_reference_dimensions'];w=WIDTH-2*MARGIN
    require([(r['gold'],r['fn'],r['duplicate_outputs']) for r in rows['area_px2']]==[(29,11,15),(495,4,9),(135,0,7),(0,0,0)],'Size narrative counts changed')
    require(not data['area_lt1024_reference_deletion_thought_experiment']['retained_row_assignment_changes'],'Deletion thought changed retained matches')
    heading(pdf,'Small targets and partial fronts','Appendix A. Descriptive association at the unchanged primary settings. Bins use original-image pixels and were fixed before reading PR121 counts. No labels are deleted.')
    y=HEIGHT-133
    headers=['Reference area (px2)','Fronts','Misses','Miss rate','Dup outputs','GT with dup','Other near GT']
    values=[[r['bin'],r['gold'],r['fn'],f"{r['fn_rate']:.1%}" if r['fn_rate'] is not None else '-',r['duplicate_outputs'],r['matched_gt_with_duplicate'],r['nearest_other_unmatched_outputs']] for r in rows['area_px2']]
    y=compact_table(pdf,'R36 / CANDIDATE / AREA',headers,values,MARGIN,y,w,[.22,.10,.10,.12,.14,.14,.18])
    values=[[r['bin'],r['gold'],r['fn'],f"{r['fn_rate']:.1%}",r['duplicate_outputs'],r['matched_gt_with_duplicate'],r['nearest_other_unmatched_outputs']] for r in rows['locatable_corners']]
    y=compact_table(pdf,'R36 / LOCATABLE REFERENCE CORNERS',['Corners','Fronts','Misses','Miss rate','Dup outputs','GT with dup','Other near GT'],values,MARGIN,y-9,w,[.22,.10,.10,.12,.14,.14,.18])
    y=para(pdf,'<b>11 of 15 misses fall below 1,024 px2; four are larger.</b> All misses are partial fronts. Yet 18 of the 29 small references are successfully matched, including all five small two-corner fronts. A small target is not automatically unusable.',MARGIN,y-3,w,9.3,13)
    y=para(pdf,'15 of 31 duplicate outputs associate with small references; 16 associate with larger references. Eleven duplicates associate with two-corner fronts. Small/partial targets concentrate difficulty, but neither size nor partial visibility alone explains every duplicate.',MARGIN,y-9,w,9.3,13)
    y=para(pdf,'Duplicate association uses the highest-IoU already-matched reference at IoU >=0.50. Other-unmatched association uses only the nearest positive-IoU reference; it is not an object-identity verdict. Eight other-unmatched outputs overlap no reference and are outside the size bins. Size, perspective, corner availability, and occlusion are confounded here; these counts do not establish causation.',MARGIN,y-9,w,8.7,12)
    require(y>42,'Size summary overflow');appendix_footer(pdf,'A','SIZE AND VISIBILITY / ASSOCIATION DOES NOT ESTABLISH CAUSATION',1,3);pdf.showPage()

    heading(pdf,'Thin boxes and a deletion thought experiment','Appendix A. Height and short side are separate descriptors. The hypothetical area exclusion changes only references in memory; model outputs and canonical labels stay fixed.')
    gap=24;pw=(w-gap)/2;y=HEIGHT-134
    for x,dim,title in [(MARGIN,'short_side_px','SHORT SIDE = MIN(WIDTH, HEIGHT)'),(MARGIN+pw+gap,'height_px','VERTICAL HEIGHT = YMAX - YMIN')]:
        compact_table(pdf,title,['Pixels','Fronts','Misses','Dup outputs'],[[r['bin'],r['gold'],r['fn'],r['duplicate_outputs']] for r in rows[dim]],x,y,pw,[.27,.24,.24,.25])
    y=HEIGHT-287
    thought=data['area_lt1024_reference_deletion_thought_experiment'];frozen=audit['references']['frozen666']['models']['candidate']['area_lt1024_reference_deletion_thought_experiment']
    values=[]
    for label,case in [('R36',thought),('Frozen666',frozen)]:
        for phase in ['before','after']:
            d=case[phase];values.append([label+' '+phase,d['gold_count'],d['tp'],d['fn'],d['fp'],d['duplicate_fp'],d['other_fp']])
    y=compact_table(pdf,'HYPOTHETICAL REMOVAL OF REFERENCE AREA <1,024 px2',['Reference','Fronts','Matched','Misses','Unmatched','Duplicate','Other'],values,MARGIN,y,w,[.23,.12,.12,.12,.14,.13,.14])
    y=para(pdf,'For R36, this erases 29 target rows: 18 matched fronts and 11 missed fronts. Misses fall 15 -> 4 because the requirements were removed, while matched fronts fall 644 -> 626 and unmatched outputs rise 94 -> 112. Duplicate count falls 31 -> 16, but other-unmatched rises 63 -> 96. Five eligible two-corner targets are also removed.',MARGIN,y-2,w,9.1,12.5)
    y=para(pdf,'No retained-row match changes occur in this thought experiment. This is not a model improvement and does not predict retraining behavior. An operational eligibility rule would require an independent physical reason and a frozen evaluation policy; a size cutoff selected to improve these scores would change the task.',MARGIN,y-8,w,8.8,12)
    require(y>42,'Deletion thought overflow');appendix_footer(pdf,'A','HYPOTHETICAL REFERENCE CHANGE / ALL 738 OUTPUTS RETAINED',2,3);pdf.showPage()

    heading(pdf,'Concrete size and overlap examples','Appendix A. Four deterministic size examples, ordered by area, followed by both high-containment reference pairs. Those pairs are physical review flags, not established duplicate labels. Cyan: reference; magenta: named output or second reference.')
    records=data['reference_rows'];chosen=[]
    rules=[('smallest missed',lambda r:not r['matched'],False),('smallest matched two-corner',lambda r:r['matched'] and r['locatable_corners']==2,False),
        ('smallest duplicate-associated',lambda r:bool(r['duplicate_prediction_ids']),False),('largest two-corner with duplicate',lambda r:r['locatable_corners']==2 and bool(r['duplicate_prediction_ids']),True)]
    for label,test,reverse in rules:
        r=sorted((r for r in records if test(r)),key=lambda r:(r['area_px2'],r['image_id'],r['row']),reverse=reverse)[0];chosen.append((label,r))
    chosen.sort(key=lambda v:(v[1]['area_px2'],v[1]['image_id'],v[1]['row']))
    inventory=[];pw=(w-20)/2
    for slot,(label,r) in enumerate(chosen):
        x=MARGIN+(slot%2)*(pw+20);top=HEIGHT-135-(slot//2)*148;name=r['image_id'];g=gold[name][r['row']-1]
        ids=(r['duplicate_prediction_ids'][:1] if 'duplicate' in label else ([r['prediction_id']] if r['prediction_id'] else []))
        objects=[(g,CYAN,f"row {r['row']}",False)]+[(predictions[name]['predictions'][pid-1],MAGENTA,f'p{pid}',True) for pid in ids]
        bounds=bounds_for([g['bbox_xyxy']]+[o[0]['bbox_xyxy'] for o in objects[1:]],padding=16)
        text(pdf,f"{name} / row {r['row']} / {label}",x,top,8.5,INK,True);draw_crop(pdf,images[name],bounds,x,top-8,pw,89,objects)
        text(pdf,f"Area {r['area_px2']:.1f} px2 | short {r['short_side_px']:.1f} px | corners {r['locatable_corners']}",x,top-109,8,INK)
        text(pdf,f"{'Matched' if r['matched'] else 'Missed'} | duplicate outputs {len(r['duplicate_prediction_ids'])} | crop {bounds}",x,top-122,7.3,MUTED)
        inventory.append({'page':start+2,'category':'size_example','selection':label,'image_id':name,'row':r['row'],'prediction_ids':ids,'crop_xyxy':bounds})
    overlap=audit['references']['r36']['label_overlap'];require(len(overlap['pairs'])==2 and overlap['iou_ge_0_5_pairs']==0,'Reference-overlap example inventory changed')
    for col,pair in enumerate(overlap['pairs']):
        x=MARGIN+col*(pw+20);top=HEIGHT-135-2*148;name=pair['image_id'];ra,rb=pair['rows'];a,b=gold[name][ra-1],gold[name][rb-1]
        bounds=bounds_for([a['bbox_xyxy'],b['bbox_xyxy']],padding=17)
        text(pdf,f"{name} / REFERENCE rows {ra} + {rb}",x,top,9,INK,True)
        draw_crop(pdf,images[name],bounds,x,top-8,pw,89,[(a,CYAN,f'row {ra}',False),(b,MAGENTA,f'row {rb}',False)])
        text(pdf,f"IoU {pair['iou']:.3f} | smaller-box containment {pair['intersection_over_smaller_box']:.3f}",x,top-109,8,INK)
        text(pdf,'Both colors are reference labels; containment alone is not duplication.',x,top-122,7.3,MUTED)
        inventory.append({'page':start+2,'category':'reference_overlap_screen','image_id':name,'rows':[ra,rb],'crop_xyxy':bounds,'iou':pair['iou']})
    appendix_footer(pdf,'A','NO EQUAL REFERENCE BOXES / NO REFERENCE PAIR WITH IoU >=0.50',3,3);pdf.showPage()
    return {'source_sha256':sha(path),'pages':[start,start+1,start+2],'inventory':inventory}


def head_pages(pdf,root,report,predictions,images,gold,start):
    path=root/SOURCE/'head-probe-comparison.json';probe=json.loads(path.read_text());cache=root/probe['probe_prediction_path'];require(sha(cache)==probe['probe_predictions_sha256'],'Probe cache drift')
    require(probe['source_comparison_sha256']==sha(root/SOURCE/'comparison.json'),'Head comparison drift')
    _,*records=[json.loads(line) for line in cache.read_text().splitlines()];one={r['image_id']:r for r in records};current=probe['references']['r36'];delta=current['paired_miss_identity_delta'];w=WIDTH-2*MARGIN
    heading(pdf,'One-to-one is a trade-off, not a duplicate cure','Appendix B. One bounded probe of the same candidate checkpoint: trained one-to-one head without NMS. Confidence, image geometry, FP32 CPU runtime, and matching thresholds stay fixed.')
    y=HEIGHT-133
    for ref,label,denom in [('r36','R36 REVIEW',583),('frozen666','FROZEN 666 / PRIMARY',584),('historical663','HISTORICAL 663',584)]:
        values=[]
        for role,title in [('one_to_many','Primary one-to-many'),('one_to_one','Probe one-to-one')]:
            d=probe['references'][ref][role]['detection'];q=probe['references'][ref][role]['corner_quality']
            values.append([title,d['fn'],d['fp'],d['duplicate_fp'],d['other_fp'],f"{q['radii'][0]['both_visible_objects_within_radius']}/{denom}",f"{q['radii'][1]['both_visible_objects_within_radius']}/{denom}"])
        y=compact_table(pdf,label,['Head','Misses','Unmatched','Duplicate','Other','Pairs <=5px','Pairs <=10px'],values,MARGIN,y,w,[.25,.10,.14,.13,.10,.14,.14])
    def ids(rows):return ', '.join(x['image_id'].replace('new_000','')+':'+str(x['row']) for x in rows)
    y=para(pdf,'<b>R36: five recovered, six newly missed.</b> Recovered rows: '+ids(delta['recovered'])+'. Newly missed rows: '+ids(delta['newly_missed'])+'. IDs use the unchanged R36 reference; the next two pages show every changed miss.',MARGIN,y-2,w,8.8,12)
    y=para(pdf,'All 583 two-corner fronts remain matched, but one fewer pair meets the ten-pixel radius. Duplicates rise 31 -> 37 despite lower total unmatched outputs. Preserve the primary head/configuration; this probe does not establish a superior detector or justify deleting small labels.',MARGIN,y-8,w,8.8,12)
    y=para(pdf,'The strict production scorer rejects one zero-height stored probe box (new_000010 / probe p37, y=720, confidence 0.179103). The diagnostic scores 709 valid boxes and explicitly counts the invalid output as other-unmatched, retaining all 710 outputs and their original pIDs. NMS IoU 0.70 is inactive on this head. No latency claim is made.',MARGIN,y-8,w,8.2,11.5)
    require(y>42,'Head summary overflow');appendix_footer(pdf,'B','SAME CHECKPOINT / SEPARATE PROBE / PRIMARY CONFIGURATION RETAINED',1,3);pdf.showPage()

    changes=[dict(x,change='recovered') for x in delta['recovered']]+[dict(x,change='newly missed') for x in delta['newly_missed']]
    require(len(changes)==11,'Head miss-change inventory differs');changes.append({'image_id':'new_000010','prediction_id':37,'change':'invalid stored output'})
    inventory=[];pw=(w-20)/2
    for page_index in range(2):
        heading(pdf,'Every changed R36 miss in the head probe','Appendix B. Cyan: unchanged R36 reference. Magenta: outputs from the explicitly named head. pIDs belong to their own caches; the two heads do not share an ID space.')
        for slot,case in enumerate(changes[page_index*6:page_index*6+6]):
            x=MARGIN+(slot%2)*(pw+20);top=HEIGHT-132-(slot//2)*148;name=case['image_id'];frame=images[name]
            if case['change']=='invalid stored output':
                p=one[name]['predictions'][36];box=p['bbox_xyxy'];require(box[1]==box[3]==720,'Invalid-output illustration differs')
                bounds=bounds_for([box],padding=22);text(pdf,f'{name} / probe p37 / invalid output',x,top,9,INK,True)
                draw_crop(pdf,frame,bounds,x,top-9,pw,91,[(p,MAGENTA,'p37',True)])
                text(pdf,'Stored box height = 0 px at image bottom; confidence 0.179103.',x,top-113,7.8,INK)
                text(pdf,'Kept as invalid other-unmatched; not silently discarded.',x,top-126,7.5,MUTED)
                inventory.append({'page':start+1+page_index,'category':'invalid_probe_output','image_id':name,'prediction_id':37,'crop_xyxy':bounds});continue
            row=case['row'];g=gold[name][row-1];observations=[]
            for label,data,permit in [('PRIMARY',predictions,False),('PROBE',one,True)]:
                r=data[name];score,_,_=score_probe_invalid_as_unmatched(gold[name],r['predictions'],r['width'],r['height'],permit)
                match=next((m for m in score['matches'] if m['gold_index']==row-1),None)
                pi=match['prediction_index'] if match else max(range(len(r['predictions'])),key=lambda i:_box_iou(g['bbox_xyxy'],r['predictions'][i]['bbox_xyxy']))
                p=r['predictions'][pi];iou=_box_iou(g['bbox_xyxy'],p['bbox_xyxy'])
                if iou==0:pi=None;p=None  # No arbitrary p1 label for a zero-IoU tie.
                observations.append({'head':label,'prediction_id':pi+1 if pi is not None else None,'matched':match is not None,'iou':iou,'prediction':p})
            bounds=bounds_for([g['bbox_xyxy']]+[o['prediction']['bbox_xyxy'] for o in observations if o['iou']>=.1],padding=17)
            text(pdf,f"{name} / row {row} / {case['change']}",x,top,9,INK,True);half=(pw-10)/2
            for col,o in enumerate(observations):
                px=x+col*(half+10);label=f"{o['head']} p{o['prediction_id']}" if o['prediction_id'] is not None else f"{o['head']} no overlap"
                text(pdf,label,px,top-14,7,MAGENTA,True)
                objects=[(g,CYAN,f'row {row}',False)]
                if o['iou']>0:objects.append((o['prediction'],MAGENTA,f"p{o['prediction_id']}",True))
                draw_crop(pdf,frame,bounds,px,top-20,half,80,objects)
            a,b=observations;text(pdf,f"IoU {a['iou']:.3f} -> {b['iou']:.3f} | match {int(a['matched'])} -> {int(b['matched'])}",x,top-113,8,INK)
            text(pdf,'Same original crop '+str(bounds),x,top-126,7.5,MUTED)
            inventory.append({'page':start+1+page_index,'category':'head_miss_change','change':case['change'],'image_id':name,'row':row,'crop_xyxy':bounds,'observations':[{k:v for k,v in o.items() if k!='prediction'} for o in observations]})
        appendix_footer(pdf,'B','ALL FIVE RECOVERIES + SIX NEW MISSES / ORIGINAL PROBE OUTPUT IDs',page_index+2,3);pdf.showPage()
    return {'source_sha256':sha(path),'prediction_sha256':sha(cache),'pages':[start,start+1,start+2],'inventory':inventory}


def handoff_page(pdf,root,report,start):
    audit_path=root/SOURCE/'archive-audit.json';audit=json.loads(audit_path.read_text());w=WIDTH-2*MARGIN
    heading(pdf,'Verified artifacts for the next comparison','Appendix C. Archive integrity, exact-source receipts, original-image hashes, and fixed CPU replay were verified independently. These identities permit reuse without guessing a checkpoint.')
    y=HEIGHT-135
    values=[('PR121 artifact commit',audit['archive_revision']),('Published training source',audit['source']['revision']),('Full returned ZIP SHA256',audit['archive_sha256']),('Reference last.pt SHA256',report['models']['reference']['checkpoint_sha256']),('Candidate last.pt SHA256',report['models']['candidate']['checkpoint_sha256'])]
    for title,value in values:
        text(pdf,title,MARGIN,y,9,INK,True);text(pdf,value,MARGIN,y-15,8.2,MUTED);y-=44
    y=para(pdf,'Both roles: native Pose26, independent original epoch-173 initialization, seed 46, 30 epochs, 1,050 batches, 4,110 presentations, 78 optimizer steps. Only two training-row boxes differ (new_000034 rows 34 and 35); no visibility change. PR121 control predictions reproduce PR119 candidate records exactly on all 20 images.',MARGIN,y+2,w,9.1,13)
    result={'page':start,'archive_audit_sha256':sha(audit_path)}
    if 'r37' in report['references']:
        old=report['references']['r36'];new=report['references']['r37'];a=json.loads((root/old['manifest']).read_text());b=json.loads((root/new['manifest']).read_text())
        old_labels={x['image_id']:x['label_sha256'] for x in a['files'] if x.get('split','val')=='val'};new_labels={x['image_id']:x['label_sha256'] for x in b['files'] if x.get('split','val')=='val'}
        require(old_labels==new_labels and len(old_labels)==20,'R37 development labels changed; equivalence claim invalid')
        for role in report['models']:
            for key in ('detection','corner_quality','by_locatable_corners'):
                require(old['models'][role][key]==new['models'][role][key],'R37 score equivalence differs')
        y=para(pdf,'<b>R37 reference equivalence:</b> all 20 development-label files have the same SHA256 values as R36. The same cached predictions therefore retain every count and point metric. R37 changes training annotations only; this equality is a reference replay, not an R37 learning result.',MARGIN,y-12,w,9.2,13)
        text(pdf,'R37 review manifest SHA256',MARGIN,y-18,8.5,INK,True);text(pdf,new['manifest_sha256'],MARGIN,y-33,8,MUTED)
        result.update({'r36_manifest_sha256':old['manifest_sha256'],'r37_manifest_sha256':new['manifest_sha256'],'all_20_development_label_hashes_equal':True})
    require(y>82,'Handoff overflow');appendix_footer(pdf,'C','NO MODEL PROMOTION / FROZEN666 PRESERVED / PR35 REMAINS SELECTED');pdf.showPage();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--comparison',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inventory',type=Path)
    args=parser.parse_args();root=args.root.resolve();comparison=args.comparison or root/SOURCE/'comparison.json';report=json.loads(comparison.read_text())
    source=report['models']['candidate'];cache=root/source['prediction_path'];require(sha(cache)==source['predictions_sha256'],'Candidate cache drift')
    _,*rows=[json.loads(x) for x in cache.read_text().splitlines()];predictions={r['image_id']:r for r in rows};images={}
    for name,r in predictions.items():
        path=root/'data/images/val'/f'{name}.png';require(sha(path)==r['image_sha256'],'Original image drift');images[name]=Image.open(path).convert('RGB')
    gold=read_reference(root,report,'r36',predictions,images);candidate=report['references']['r36']['models']['candidate']
    cases=[dict(x,category='miss') for x in candidate['misses']]+[dict(x,category='corner') for x in candidate['corner_error_rows']];unmatched=candidate['unmatched_outputs'];d=candidate['detection']
    require((d['fn'],len(candidate['corner_error_rows']),d['fp'],d['duplicate_fp'],d['other_fp'])==(15,19,94,31,63),'Frozen narrative totals changed')
    require(len(candidate['misses'])==d['fn'] and len(unmatched)==d['fp'],'Candidate catalogue count differs')
    require(len({(x['image_id'],x['row']) for x in candidate['misses']})==d['fn'],'Repeated missed row')
    require(len({(x['image_id'],x['prediction_id']) for x in unmatched})==d['fp'],'Repeated unmatched output')
    for name,r in predictions.items():
        score=score_image(gold[name],r['predictions'],r['width'],r['height'],.5,.5)
        require({i+1 for i in score['unmatched_gold_indices']}=={x['row'] for x in candidate['misses'] if x['image_id']==name},'Miss ledger differs')
        require({i+1 for i in score['unmatched_prediction_indices']}=={x['prediction_id'] for x in unmatched if x['image_id']==name},'Unmatched ledger differs')
    register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True);pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT));pdf.setTitle('PR121 R36 misses and errors');pdf.setAuthor('Cassette-front review')
    pages=3+math.ceil(len(cases)/2)+math.ceil(len(unmatched)/6);inventory=[]
    cover(pdf,report,pages);definitions(pdf,report,pages);count_page(pdf,candidate,pages);page=4
    for offset in range(0,len(cases),2):detailed_page(pdf,cases[offset:offset+2],gold,predictions,images,page,pages,inventory);page+=1
    for offset in range(0,len(unmatched),6):unmatched_page(pdf,unmatched[offset:offset+6],gold,predictions,images,page,pages,inventory);page+=1
    require(page-1==pages,'Main page inventory mismatch')
    size=size_pages(pdf,root,report,predictions,images,gold,pages+1)
    head=head_pages(pdf,root,report,predictions,images,gold,pages+4)
    handoff=handoff_page(pdf,root,report,pages+7);pdf.save()
    result={'schema_version':1,'pages':pages+7,'r36_main_pages':pages,'pdf_sha256':sha(args.output),'source_comparison_sha256':sha(comparison),'renderer_sha256':sha(Path(__file__)),
        'candidate_predictions_sha256':sha(cache),'r36_reference_manifest_sha256':report['references']['r36']['manifest_sha256'],
        'source_image_sha256':{name:predictions[name]['image_sha256'] for name in sorted(images)},
        'missed_rows':d['fn'],'corner_error_rows':len(candidate['corner_error_rows']),'unmatched_outputs':d['fp'],'duplicates':d['duplicate_fp'],'other_unmatched':d['other_fp'],
        'inventory':inventory,'size_visibility_appendix':size,'head_comparison_appendix':head,'handoff_appendix':handoff}
    target=args.inventory or root/SOURCE/'pdf-inventory.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','source_image_sha256','size_visibility_appendix','head_comparison_appendix')},indent=2))


if __name__=='__main__':main()
