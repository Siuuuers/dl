#!/usr/bin/env python3
"""Render the complete PR123 candidate error catalogue from saved fixed inference."""
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
SOURCE = 'results/evaluation/pr123-r37-20261126'


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, '26 NOV 2026 / ORIGINAL PNG PIXELS / R37 DEVELOPMENT DIAGNOSTIC / PR123', MARGIN, 18, 7, MUTED)
    pdf.setFont('Amend', 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f'{page} / {pages}')


def heading(pdf, title, subtitle):
    text(pdf, 'PR123 / CASSETTE FRONTS', MARGIN, HEIGHT-32, 9, MUTED, True)
    text(pdf, title, MARGIN, HEIGHT-67, 24, INK, True)
    para(pdf, subtitle, MARGIN, HEIGHT-83, WIDTH-2*MARGIN, 9.2, 13)


def table(pdf, report, ref, top, title):
    text(pdf, title, MARGIN, top, 10, MUTED, True)
    cols=[MARGIN+6, MARGIN+300, MARGIN+370, MARGIN+460, MARGIN+555, MARGIN+665]
    y=top-23
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,19,fill=1,stroke=0)
    for x,label in zip(cols,('Model','Miss','Unmatched','Duplicate','Both <=5 px','Both <=10 px')):
        text(pdf,label,x,y,8.5,MUTED,True)
    for role,label in (('reference','R36-label native control'),('candidate','R37-label native candidate'),('pr35','PR35 selected')):
        m=report['references'][ref]['models'][role];d=m['detection'];q=m['corner_quality'];y-=19
        values=[label,d['fn'],d['fp'],d['duplicate_fp'],q['radii'][0]['both_visible_objects_within_radius'],q['radii'][1]['both_visible_objects_within_radius']]
        for x,val in zip(cols,values):text(pdf,str(val),x,y,9,INK,role=='candidate')
    return y-19


def cover(pdf, report, pages):
    heading(pdf,'R37: no gain in reviewed recall','The returned PR123 native pair is verified against its exact source. Fresh CPU inference uses the preserved fixed settings and matches the prior runtime headers.')
    y=HEIGHT-130
    y=table(pdf,report,'r37',y,'PRESERVED R37 REVIEW / 659 FRONTS / 583 TWO-LOCATABLE-CORNER PAIRS')
    y=table(pdf,report,'frozen666',y-8,'FROZEN 666 / PRIMARY HISTORICAL COMPARISON / 584 ELIGIBLE PAIRS')
    y=table(pdf,report,'historical663',y-8,'HISTORICAL 663 / REPORTED SEPARATELY / 584 ELIGIBLE PAIRS')
    y=para(pdf,'<b>The same 15 R37 fronts remain missed.</b> All are partial: three have no locatable corner and twelve have one. The candidate matches 583/583 two-corner fronts, but only 61/76 partial fronts. Unmatched outputs fall by one; duplicate and corner-error row counts stay unchanged.',MARGIN,y-5,WIDTH-2*MARGIN,9,12.8)
    y=para(pdf,'Frozen 666 recovers one front and removes two unmatched outputs, while losing one corner pair within five pixels. The paired crop appendix shows both fixed-threshold crossings. R38 deletions and corrections are replayed separately so they cannot erase the preserved-R37 failure catalogue. PR35 remains selected. The reviewed development set does not establish reliability on new acquisition sessions.',MARGIN,y-9,WIDTH-2*MARGIN,8.6,12)
    require(y>49,'Cover text overlaps footer')
    footer(pdf,1,pages);pdf.showPage()


def definitions(pdf, report, pages):
    heading(pdf,'Read each category precisely','The complete catalogue follows: 15 missed fronts, 19 matched visible-corner error rows, and all 93 unmatched candidate outputs.')
    gap=28;w=(WIDTH-2*MARGIN-gap)/2;left=MARGIN;right=MARGIN+w+gap
    def block(x,y,title,body):
        text(pdf,title,x,y,11,INK,True)
        return para(pdf,body,x,y-12,w,9.2,13)-22
    y=HEIGHT-137
    y=block(left,y,'Missed front','A reference row has no assigned output at same-class IoU >=0.50. Seven candidate misses have no output with IoU >=0.10; eight have a closer output below 0.50. These are geometric triage groups, not final physical diagnoses.')
    y=block(left,y,'Unmatched output','An output has no assigned reference after confidence-ordered one-to-one matching. It may be a duplicate, misplaced box, excluded surface, uncertain front, or omitted reference object. Do not call all unmatched outputs hallucinations.')
    y=block(left,y,'Duplicate is already included','A duplicate is an unmatched output that overlaps an already matched reference at IoU >=0.50. Here, <b>93 unmatched = 31 duplicate + 62 other unmatched.</b> Do not add duplicate counts a second time.')
    y=block(left,y,'Matching was checked','A maximum-cardinality check of the same IoU-eligible graph finds no extra possible matches in any image for any of the five checkpoint records across the separately preserved references. The confidence-greedy scorer remains unchanged; no missing TP count is attributable to greedy assignment in this replay.')
    z=HEIGHT-137
    z=block(right,z,'Corner error','On a matched object, a locatable reference corner is more than 5 original-image pixels from its returned point, or the point is missing/invalid at confidence 0.50. Five pixels is an illustrative review radius, not a calibrated robot tolerance. Pair denominators include eligible missed objects.')
    z=block(right,z,'Unavailable reference corners','Reference visibility 0 has no coordinate truth. A predicted hidden corner is allowed but cannot be scored against (0,0). A confident point is not proof that the physical corner is visible. Cyan draws only locatable reference points; magenta draws returned model points.')
    z=block(right,z,'Coordinates and IDs','Stored image x grows right; y grows down. Top is ymin; bottom is ymax. L/R follow image-X order. Rows are R37 reference TXT rows. <b>pIDs are one-based original PR123 candidate output indices.</b> Earlier PR113/PR114/PR117/PR119/PR121 pIDs belong to different caches and must not be transferred.')
    z=block(right,z,'Fixed procedure','Both arms: native Pose26 loss, seed 46, independent epoch-173 initialization, 30 epochs. Inference: Ultralytics 8.4.182, PyTorch 2.10 CPU, FP32, size 1280, one-to-many head, NMS 0.70, box confidence 0.13, endpoint confidence 0.50. No new head, confidence, or NMS probe was performed. PDF crops are review panels; the network receives full original frames, then resize/letterbox preprocessing.')
    require(min(y,z)>35,'Definition page overflow')
    footer(pdf,2,pages);pdf.showPage()


def count_page(pdf, candidate, pages):
    heading(pdf,'Every development image is accounted for','Candidate against current R37 reference; the three empty scenes remain in the inventory and contribute to false-output totals.')
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
    heading(pdf,'Every missed front and corner error','Identical original crop and scale in each pair. Cyan: reviewed R37 reference. Magenta: fixed PR123 candidate output.')
    panelw=(WIDTH-2*MARGIN-18)/2
    for slot,case in enumerate(cases):
        top=HEIGHT-126-slot*221;name=case['image_id'];row=case['row'];g=gold[name][row-1];preds=predictions[name]['predictions']
        miss=case['category']=='miss';pi=(case['nearest_prediction_id'] if miss else case['prediction_id']);p=preds[pi-1] if pi else None
        boxes=[g['bbox_xyxy']]
        if p is not None and (not miss or case['nearest_iou']>=.10):boxes.append(p['bbox_xyxy'])
        bounds=bounds_for(boxes)
        title=f"{name} / row {row} / {'missed front' if miss else 'corner error'}"
        text(pdf,title,MARGIN,top,12.2,INK,True)
        text(pdf,'R37 REFERENCE',MARGIN,top-17,8,CYAN,True)
        text(pdf,'PR123 CANDIDATE'+(f' / p{pi}' if pi else ''),MARGIN+panelw+18,top-17,8,MAGENTA,True)
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


def load_cache(root,report,role):
    source=report['models'][role];path=root/source['prediction_path']
    require(sha(path)==source['predictions_sha256'],'Prediction cache drift')
    _,*rows=[json.loads(x) for x in path.read_text().splitlines()]
    return {r['image_id']:r for r in rows}


def compact_table(pdf,title,headers,rows,x,top,width,ratios,font=8.4):
    text(pdf,title,x,top,10,INK,True);y=top-23;positions=[x+5]
    for ratio in ratios[:-1]:positions.append(positions[-1]+width*ratio)
    pdf.setFillColor(PALE);pdf.rect(x,y-5,width,19,fill=1,stroke=0)
    for px,label in zip(positions,headers):text(pdf,str(label),px,y,font,MUTED,True)
    for index,row in enumerate(rows):
        y-=19
        if index%2==1:pdf.setFillColor(PALE);pdf.rect(x,y-5,width,18,fill=1,stroke=0)
        for px,value in zip(positions,row):text(pdf,str(value),px,y,font,INK)
    return y-18


def threshold_page(pdf,root,report,predictions,images,physical_page):
    heading(pdf,'Two fixed thresholds, two directions','Appendix A. The reference and original pixels are identical within each pair. Cyan: frozen666 reference. Magenta: returned output from the named PR123 arm. No cutoff changes.')
    control=load_cache(root,report,'reference');gold=read_reference(root,report,'frozen666',predictions,images)
    w=(WIDTH-2*MARGIN-18)/2;inventory=[]
    for slot,(name,row) in enumerate([('new_000089',43),('new_000010',20)]):
        g=gold[name][row-1];observations=[]
        for role,cache in [('reference',control),('candidate',predictions)]:
            preds=cache[name]['predictions'];score=score_image(gold[name],preds,1280,720,.5,.5)
            matched=next((m for m in score['matches'] if m['gold_index']==row-1),None)
            pi=matched['prediction_index'] if matched else max(range(len(preds)),key=lambda i:_box_iou(g['bbox_xyxy'],preds[i]['bbox_xyxy']))
            p=preds[pi];observations.append({'role':role,'prediction_id':pi+1,'matched':matched is not None,
                'iou':_box_iou(g['bbox_xyxy'],p['bbox_xyxy']),'confidence':p['confidence'],
                'corner_errors_px':[math.dist(a[:2],b[:2]) if a[2]==2 else None for a,b in zip(g['keypoints'],p['keypoints'])],'prediction':p})
        old,new=observations;bounds=bounds_for([g['bbox_xyxy']]+[o['prediction']['bbox_xyxy'] for o in observations]);top=HEIGHT-128-slot*221
        text(pdf,f'Frozen666 / {name} / row {row}',MARGIN,top,12,INK,True)
        text(pdf,f"R36-LABEL CONTROL / p{old['prediction_id']}",MARGIN,top-17,8,MAGENTA,True)
        text(pdf,f"R37-LABEL CANDIDATE / p{new['prediction_id']}",MARGIN+w+18,top-17,8,MAGENTA,True)
        for x,o in [(MARGIN,old),(MARGIN+w+18,new)]:draw_crop(pdf,images[name],bounds,x,top-24,w,126,[(g,CYAN,f'row {row}',False),(o['prediction'],MAGENTA,f"p{o['prediction_id']}",True)])
        if name=='new_000089':
            require(not old['matched'] and new['matched'] and old['iou']<.5<=new['iou'],'Recovered fixed-gate illustration differs')
            description=f"Best IoU {old['iou']:.6f} -> assigned IoU {new['iou']:.6f}; the unchanged match gate is 0.50. This is the sole recovered frozen666/historical663 row. Both arms return an overlapping box; the preserved R37 missed-row set is unchanged."
        else:
            a,b=old['corner_errors_px'][1],new['corner_errors_px'][1]
            require(old['matched'] and new['matched'] and a<=5<b,'Corner fixed-gate illustration differs')
            description=f"Both boxes remain matched. Right-point error {a:.6f} -> {b:.6f} px crosses outside the fixed five-pixel radius, losing one frozen-reference pair. The ten-pixel pair count and preserved R37 corner-error ledger stay unchanged."
        y=para(pdf,description,MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4)
        text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'Threshold card overflow')
        inventory.append({'page':physical_page,'reference':'frozen666','image_id':name,'row':row,'crop_xyxy':bounds,'observations':[{k:v for k,v in o.items() if k!='prediction'} for o in observations]})
    appendix_footer(pdf,'A','FIXED IoU 0.50 / FIXED FIVE-PIXEL REVIEW RADIUS / NO THRESHOLD SEARCH');pdf.showPage();return inventory


def context_page(pdf,root,report,predictions,images,physical_page):
    audit_path=root/'results/review/front-review-20261126-r38/input-context-audit.json'
    require(sha(audit_path)=='5b06b952f7000679cec7fbb2b0ddcb794bf5d6bb5bc9d63ec28a0a22af050e30','Context evidence drift')
    heading(pdf,'Rack context is already in the input','Appendix B. These are complete original validation frames, not row crops. The cropped panels elsewhere in this PDF are a review format; they do not describe a cropped-only training pipeline.')
    gap=20;w=(WIDTH-2*MARGIN-gap)/2;top=HEIGHT-138
    for index,name in enumerate(['new_000009','new_000010']):
        x=MARGIN+index*(w+gap);text(pdf,name+' / full 1280 x 720 PNG',x,top,10,INK,True)
        draw_crop(pdf,images[name],[0,0,1280,720],x,top-12,w,212,[])
    y=top-247
    y=para(pdf,'<b>Verified routing:</b> staging uses the complete source PNGs. Both returned PR123 receipts record four sampled training batches of shape [4, 3, 736, 1280]. Rectangular resize/letterbox preprocessing preserves a landscape full-frame input. Inference also passes the complete decoded original frame to the model.',MARGIN,y,WIDTH-2*MARGIN,9.5,13.3)
    y=para(pdf,'Mosaic, mixup, cutmix, and copy-paste are disabled. Training still applies configured rotation (5 degrees), translation (0.05), scaling (0.15), horizontal flipping (0.5), and color changes. These transformations can resample or clip context; the evidence does not claim every original background pixel survives every batch.',MARGIN,y-11,WIDTH-2*MARGIN,9.2,13)
    y=para(pdf,'The rack is present as visual context, while the supervised output class remains cassette_front with two corner slots. Supplying rack pixels does not prove the network has learned an explicit rack coordinate system. This run does not test whether adding rack labels or a geometry model would help; no new model, head, or threshold experiment is introduced here.',MARGIN,y-11,WIDTH-2*MARGIN,9.2,13)
    require(y>45,'Context page overflow');appendix_footer(pdf,'B','FULL FRAMES IN THE PIPELINE / ROW CROPS ONLY FOR REVIEW');pdf.showPage()
    return {'page':physical_page,'source_audit':audit_path.relative_to(root).as_posix(),'source_audit_sha256':sha(audit_path),'full_frame_images':['new_000009','new_000010']}


def r38_summary_page(pdf,root,report,delta,physical_page):
    before=report['references']['r37'];after=report['references']['r38'];candidate=delta['models']['candidate'];w=WIDTH-2*MARGIN
    require(after['manifest_sha256']==delta['r38_manifest_sha256'],'R38 reference delta drift')
    heading(pdf,'R38 changes the reference, not the model','Appendix C. The exact same saved predictions are rescored after five user-directed target exclusions and three corrections. Main pages retain all preserved-R37 failures. No R38 model has been trained.')
    rows=[]
    for role,label in [('reference','PR123 control'),('candidate','PR123 candidate'),('pr35','PR35 selected')]:
        for name,ref in [('R37',before),('R38',after)]:
            m=ref['models'][role];d=m['detection'];q=m['corner_quality']
            rows.append([label+' / '+name,d['gold_count'],d['fn'],d['fp'],d['duplicate_fp'],f"{q['radii'][0]['both_visible_objects_within_radius']}/{q['both_visible_objects']}"])
    y=compact_table(pdf,'UNCHANGED OUTPUTS / SEPARATE TARGET POPULATIONS',['Checkpoint / target','Fronts','Misses','Unmatched','Duplicate','Pairs <=5 px'],rows,MARGIN,HEIGHT-135,w,[.31,.12,.12,.15,.13,.17])
    deleted=[r for r in candidate['changed_reference_rows'] if r['action']=='delete'];values=[]
    for r in deleted:
        old=r['r37'];pid=old['matched_prediction_id'];values.append([r['image_id']+':'+str(r['r37_row']),'null','matched p'+str(pid) if pid else 'missed','retained output becomes unmatched' if pid else 'removed missed target'])
    y=compact_table(pdf,'FIVE EXCLUDED R37 IDENTITIES / CANDIDATE EFFECT',['Old R37 identity','R38 row','Previous state','Reference-change effect'],values,MARGIN,y-4,w,[.27,.12,.21,.40],8.1)
    y=para(pdf,'<b>The four-count miss reduction is entirely target removal.</b> No retained candidate row gains or loses a match. Candidate unmatched outputs increase 93 -> 94; two duplicate outputs become other-unmatched after box changes (007/p59 and 010/p34). These are changed scoring categories, not suppressed detections.',MARGIN,y-2,w,8.7,11.8)
    y=para(pdf,'Available reference points change 1,224 -> 1,220; the 583 eligible pairs remain. The corrected 010:29 left target changes its error 5.131271 -> 0.500253 px, reducing corner-error rows 19 -> 18 without changing the prediction. PR35 has an additional retained miss at 008:7 after the bottom correction.',MARGIN,y-7,w,8.5,11.6)
    require(y>42,'R38 summary overflow');appendix_footer(pdf,'C','USER-DIRECTED REFERENCE EXCLUSIONS / NO DETECTOR RECOVERY CLAIM',1,3);pdf.showPage()
    return {'page':physical_page,'r37_manifest_sha256':before['manifest_sha256'],'r38_manifest_sha256':after['manifest_sha256'],
        'candidate_miss_attribution':candidate['miss_count_change_attribution'],'deleted_old_row_identities':[{'image_id':r['image_id'],'r37_row':r['r37_row'],'r38_row':None} for r in deleted]}


def r38_crop_pages(pdf,root,report,delta,predictions,images,start):
    candidate=delta['models']['candidate'];changes=[r for r in candidate['changed_reference_rows'] if r['action']=='replace']
    matched_deleted=[r for r in candidate['changed_reference_rows'] if r['action']=='delete' and r['r37']['matched_prediction_id'] is not None]
    require(len(changes)==3 and len(matched_deleted)==1,'R38 crop inventory changed');cases=changes+matched_deleted
    w=(WIDTH-2*MARGIN-18)/2;inventory=[]
    explanations={('new_000007',47):'Top 388 -> 385; both corner targets stay unchanged. The same p45 remains matched. Smaller duplicate p59 falls below the revised box-overlap gate and becomes other-unmatched.',
        ('new_000008',7):'Bottom 33 -> 23 excludes the separator below the owned front. The same p36 remains matched; its left-point error stays 8.085506 px. The correction is not copied from the broader prediction.',
        ('new_000010',29):'Top 679.911797 -> 666; observed left target (320.910003,717.269998) -> (325,714). The same p28 remains matched. Its left-point distance changes because the reference changes.'}
    for page_index in range(2):
        heading(pdf,'Exact effects of the R38 reference changes','Appendix C. Identical image crop and unchanged candidate output in each pair. Cyan: named reference version. Magenta: the same original PR123 candidate pID. All IDs on the left are preserved R37 IDs.')
        for slot,r in enumerate(cases[page_index*2:page_index*2+2]):
            name=r['image_id'];row=r['r37_row'];old,new=r['r37'],r['r38'];pid=old['matched_prediction_id'];p=predictions[name]['predictions'][pid-1]
            g0=old['reference'];g1=new['reference'] if new else None;bounds=bounds_for([g0['bbox_xyxy'],p['bbox_xyxy']]+([g1['bbox_xyxy']] if g1 else []));top=HEIGHT-135-slot*218
            text(pdf,f"{name} / R37 row {row} -> R38 {r['r38_row'] if r['r38_row'] is not None else 'excluded'} / same p{pid}",MARGIN,top,11.6,INK,True)
            text(pdf,'R37 REFERENCE + OUTPUT',MARGIN,top-17,8,CYAN,True)
            text(pdf,'R38 REFERENCE + OUTPUT' if g1 else 'R38: NO TARGET FOR THIS EXCLUDED ROW',MARGIN+w+18,top-17,8,CYAN,True)
            draw_crop(pdf,images[name],bounds,MARGIN,top-24,w,120,[(g0,CYAN,f'row {row}',False),(p,MAGENTA,f'p{pid}',True)])
            objects=([(g1,CYAN,f"row {r['r38_row']}",False)] if g1 else [])+[(p,MAGENTA,f'p{pid}',True)]
            draw_crop(pdf,images[name],bounds,MARGIN+w+18,top-24,w,120,objects)
            if new:
                require(pid==new['matched_prediction_id'],'Replacement crop assignment changed')
                description=explanations[(name,row)]+f" Assigned IoU {old['matched_iou']:.6f} -> {new['matched_iou']:.6f}."
            else:
                description=f"This user-directed exclusion removes a previously matched zero-corner target (old IoU {old['matched_iou']:.6f}). The unchanged p{pid} is now other-unmatched. No box was removed by the model and this is not a confirmed duplicate-annotation finding."
            y=para(pdf,escape(description),MARGIN,top-156,WIDTH-2*MARGIN,8.5,11.5);text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'R38 crop overflow')
            inventory.append({'page':start+page_index,'image_id':name,'r37_row':row,'r38_row':r['r38_row'],'action':r['action'],'prediction_id':pid,'crop_xyxy':bounds,'r37_iou':old['matched_iou'],'r38_iou':new['matched_iou'] if new else None})
        appendix_footer(pdf,'C','SAME CHECKPOINT AND OUTPUTS / REFERENCE-ONLY REPLAY',page_index+2,3);pdf.showPage()
    return inventory


def provenance_page(pdf,root,report,delta,physical_page):
    archive_path=root/SOURCE/'archive-audit.json';source_path=root/SOURCE/'inference-source-audit.json';archive=json.loads(archive_path.read_text());source=json.loads(source_path.read_text());w=WIDTH-2*MARGIN
    heading(pdf,'Verified inputs for a cached replay','Appendix D. Both returned training runs and fresh CPU inference were verified. R38 has no training-label delta; a cached replay can review the corrected targets without another GPU training run.')
    y=HEIGHT-134
    entries=[('PR123 artifact commit',archive['archive_revision']),('Published training source',archive['source']['revision']),('Full returned ZIP SHA256',archive['archive_sha256']),('Reference last.pt SHA256',report['models']['reference']['checkpoint_sha256']),('Candidate last.pt SHA256',report['models']['candidate']['checkpoint_sha256'])]
    for label,value in entries:text(pdf,label,MARGIN,y,9,INK,True);text(pdf,value,MARGIN,y-15,8.2,MUTED);y-=42
    y=para(pdf,'Both arms use native Pose26, seed 46, independent epoch-173 initialization, 30 epochs, 1,050 batches, 4,110 presentations and 78 optimizer steps. The R37 training delta is one box correction and one left-point correction; no visibility change. PR123 control predictions exactly reproduce PR121 candidate records on all 20 images.',MARGIN,y+2,w,9,12.5)
    require(source['dirty_path_count']==157 and len(source['source_file_checks'])==47,'Source reconciliation inventory differs')
    y=para(pdf,'The recorded dirty Git paths are all 157 materialized original PNGs. Every image hash/size matches its committed LFS pointer and original manifest; all 47 receipt/source files match the exact commit. Original receipts and cache headers remain intact. The initializer identity is verified in metadata/receipts; its large byte file was not required or downloaded for this replay.',MARGIN,y-10,w,8.6,12)
    y=para(pdf,'R38 keeps all 137 training files byte-identical and changes only development targets. The replay helper uses Python standard-library code, cached predictions, verified label files and stable row maps; it does not need Torch, model weights or image decoding. No new head/threshold probe or model promotion is included.',MARGIN,y-10,w,8.8,12)
    text(pdf,'R38 reference manifest SHA256',MARGIN,y-18,8.5,INK,True);text(pdf,delta['r38_manifest_sha256'],MARGIN,y-33,8,MUTED)
    require(y-33>42,'Provenance page overflow');appendix_footer(pdf,'D','FROZEN666 PRESERVED / PR35 REMAINS SELECTED / NO R38 LEARNING CLAIM');pdf.showPage()
    return {'page':physical_page,'archive_audit_sha256':sha(archive_path),'source_reconciliation_sha256':sha(source_path),'r38_manifest_sha256':delta['r38_manifest_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--comparison',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inventory',type=Path)
    args=parser.parse_args();root=args.root.resolve();comparison=args.comparison or root/SOURCE/'comparison.json';report=json.loads(comparison.read_text())
    delta_path=root/SOURCE/'r38-reference-delta.json';delta=json.loads(delta_path.read_text());require('r38' in report['references'],'Final report requires separate R38 replay')
    predictions=load_cache(root,report,'candidate');images={}
    for name,r in predictions.items():
        path=root/'data/images/val'/f'{name}.png';require(sha(path)==r['image_sha256'],'Original image drift');images[name]=Image.open(path).convert('RGB')
    gold=read_reference(root,report,'r37',predictions,images);candidate=report['references']['r37']['models']['candidate'];d=candidate['detection']
    cases=[dict(x,category='miss') for x in candidate['misses']]+[dict(x,category='corner') for x in candidate['corner_error_rows']];unmatched=candidate['unmatched_outputs']
    require((d['fn'],len(candidate['corner_error_rows']),d['fp'],d['duplicate_fp'],d['other_fp'])==(15,19,93,31,62),'Preserved-R37 narrative totals changed')
    require(len({(x['image_id'],x['row']) for x in candidate['misses']})==d['fn'],'Repeated missed row')
    require(len({(x['image_id'],x['prediction_id']) for x in unmatched})==d['fp'],'Repeated unmatched output')
    for name,r in predictions.items():
        score=score_image(gold[name],r['predictions'],r['width'],r['height'],.5,.5)
        require({i+1 for i in score['unmatched_gold_indices']}=={x['row'] for x in candidate['misses'] if x['image_id']==name},'Miss ledger differs')
        require({i+1 for i in score['unmatched_prediction_indices']}=={x['prediction_id'] for x in unmatched if x['image_id']==name},'Unmatched ledger differs')
    register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True);pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT));pdf.setTitle('PR123 R37 misses and errors with separate R38 replay');pdf.setAuthor('Cassette-front review')
    pages=3+math.ceil(len(cases)/2)+math.ceil(len(unmatched)/6);inventory=[]
    cover(pdf,report,pages);definitions(pdf,report,pages);count_page(pdf,candidate,pages);page=4
    for offset in range(0,len(cases),2):detailed_page(pdf,cases[offset:offset+2],gold,predictions,images,page,pages,inventory);page+=1
    for offset in range(0,len(unmatched),6):unmatched_page(pdf,unmatched[offset:offset+6],gold,predictions,images,page,pages,inventory);page+=1
    require(page-1==pages,'Main page inventory mismatch')
    thresholds=threshold_page(pdf,root,report,predictions,images,pages+1)
    context=context_page(pdf,root,report,predictions,images,pages+2)
    r38summary=r38_summary_page(pdf,root,report,delta,pages+3)
    r38crops=r38_crop_pages(pdf,root,report,delta,predictions,images,pages+4)
    provenance=provenance_page(pdf,root,report,delta,pages+6);pdf.save()
    result={'schema_version':1,'pages':pages+6,'r37_main_pages':pages,'pdf_sha256':sha(args.output),'source_comparison_sha256':sha(comparison),'renderer_sha256':sha(Path(__file__)),
        'candidate_predictions_sha256':report['models']['candidate']['predictions_sha256'],'r37_reference_manifest_sha256':report['references']['r37']['manifest_sha256'],
        'source_image_sha256':{name:predictions[name]['image_sha256'] for name in sorted(images)},'missed_rows':d['fn'],'corner_error_rows':len(candidate['corner_error_rows']),'unmatched_outputs':d['fp'],'duplicates':d['duplicate_fp'],'other_unmatched':d['other_fp'],
        'inventory':inventory,'paired_threshold_appendix':thresholds,'full_frame_context_appendix':context,'r38_reference_appendix':{'reference_delta_sha256':sha(delta_path),'summary':r38summary,'inventory':r38crops},'provenance_appendix':provenance}
    target=args.inventory or root/SOURCE/'pdf-inventory.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','source_image_sha256','paired_threshold_appendix','r38_reference_appendix')},indent=2))


if __name__=='__main__':main()
