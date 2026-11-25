#!/usr/bin/env python3
"""Render the complete PR119 candidate error catalogue from saved fixed inference."""
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
SOURCE = 'results/evaluation/pr119-r35-20261126'


def footer(pdf, page, pages):
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(.6)
    pdf.line(MARGIN, 31, WIDTH-MARGIN, 31)
    text(pdf, '26 NOV 2026 / ORIGINAL PNG PIXELS / R35 DEVELOPMENT DIAGNOSTIC / PR119', MARGIN, 18, 7, MUTED)
    pdf.setFont('Amend', 7)
    pdf.drawRightString(WIDTH-MARGIN, 18, f'{page} / {pages}')


def heading(pdf, title, subtitle):
    text(pdf, 'PR119 / CASSETTE FRONTS', MARGIN, HEIGHT-32, 9, MUTED, True)
    text(pdf, title, MARGIN, HEIGHT-67, 24, INK, True)
    para(pdf, subtitle, MARGIN, HEIGHT-83, WIDTH-2*MARGIN, 9.2, 13)


def table(pdf, report, ref, top, title):
    text(pdf, title, MARGIN, top, 10, MUTED, True)
    cols=[MARGIN+6, MARGIN+300, MARGIN+370, MARGIN+460, MARGIN+555, MARGIN+665]
    y=top-23
    pdf.setFillColor(PALE);pdf.rect(MARGIN,y-5,WIDTH-2*MARGIN,19,fill=1,stroke=0)
    for x,label in zip(cols,('Model','Miss','Unmatched','Duplicate','Both <=5 px','Both <=10 px')):
        text(pdf,label,x,y,8.5,MUTED,True)
    for role,label in (('reference','R34-label native control'),('candidate','R35-label native candidate'),('pr35','PR35 selected')):
        m=report['references'][ref]['models'][role];d=m['detection'];q=m['corner_quality'];y-=19
        values=[label,d['fn'],d['fp'],d['duplicate_fp'],q['radii'][0]['both_visible_objects_within_radius'],q['radii'][1]['both_visible_objects_within_radius']]
        for x,val in zip(cols,values):text(pdf,str(val),x,y,9,INK,role=='candidate')
    return y-19


def cover(pdf, report, pages):
    heading(pdf,'R35: no gain in reviewed recall','The returned PR119 native pair is verified against its exact source. Fresh CPU inference uses the preserved fixed settings and matches the prior runtime headers.')
    y=HEIGHT-130
    y=table(pdf,report,'r35',y,'CURRENT R35 REVIEW / 659 FRONTS / 583 TWO-LOCATABLE-CORNER PAIRS')
    y=table(pdf,report,'frozen666',y-8,'FROZEN 666 / PRIMARY HISTORICAL COMPARISON / 584 ELIGIBLE PAIRS')
    y=table(pdf,report,'historical663',y-8,'HISTORICAL 663 / REPORTED SEPARATELY / 584 ELIGIBLE PAIRS')
    y=para(pdf,'<b>The same 15 R35 fronts remain missed.</b> All are partial: three have no locatable corner and twelve have one. The candidate matches 583/583 two-corner fronts, but only 61/76 partial fronts. Unmatched outputs rise by one; duplicates rise by two. One fewer matched row exceeds the five-pixel corner review radius.',MARGIN,y-5,WIDTH-2*MARGIN,9,12.8)
    y=para(pdf,'Frozen 666 gains two corner pairs within five pixels but adds one miss and two unmatched outputs. The added miss crosses the fixed IoU 0.50 gate; Appendix A shows the actual crops. PR35 remains selected. The reviewed development set does not establish reliability on new acquisition sessions.',MARGIN,y-9,WIDTH-2*MARGIN,8.6,12)
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
    z=block(right,z,'Coordinates and IDs','Stored image x grows right; y grows down. Top is ymin; bottom is ymax. L/R follow image-X order. Rows are R35 reference TXT rows. <b>pIDs are one-based original PR119 candidate output indices.</b> Earlier PR113/PR114/PR117 pIDs belong to different caches and must not be transferred.')
    z=block(right,z,'Fixed procedure','Both arms: native Pose26 loss, seed 46, independent epoch-173 initialization, 30 epochs. Inference: Ultralytics 8.4.182, PyTorch 2.10 CPU, FP32, size 1280, one-to-many head, NMS 0.70, box confidence 0.13, endpoint confidence 0.50. No new confidence or NMS sweep was performed. Appendix A illustrates existing threshold crossings; it does not change a cutoff.')
    require(min(y,z)>35,'Definition page overflow')
    footer(pdf,2,pages);pdf.showPage()


def count_page(pdf, candidate, pages):
    heading(pdf,'Every development image is accounted for','Candidate against current R35 reference; the three empty scenes remain in the inventory and contribute to false-output totals.')
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
    heading(pdf,'Every missed front and corner error','Identical original crop and scale in each pair. Cyan: reviewed R35 reference. Magenta: fixed PR119 candidate output.')
    panelw=(WIDTH-2*MARGIN-18)/2
    for slot,case in enumerate(cases):
        top=HEIGHT-126-slot*221;name=case['image_id'];row=case['row'];g=gold[name][row-1];preds=predictions[name]['predictions']
        miss=case['category']=='miss';pi=(case['nearest_prediction_id'] if miss else case['prediction_id']);p=preds[pi-1] if pi else None
        boxes=[g['bbox_xyxy']]
        if p is not None and (not miss or case['nearest_iou']>=.10):boxes.append(p['bbox_xyxy'])
        bounds=bounds_for(boxes)
        title=f"{name} / row {row} / {'missed front' if miss else 'corner error'}"
        text(pdf,title,MARGIN,top,12.2,INK,True)
        text(pdf,'R35 REFERENCE',MARGIN,top-17,8,CYAN,True)
        text(pdf,'PR119 CANDIDATE'+(f' / p{pi}' if pi else ''),MARGIN+panelw+18,top-17,8,MAGENTA,True)
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


def appendix_footer(pdf, section, label):
    pdf.setStrokeColor(RULE);pdf.setLineWidth(.6);pdf.line(MARGIN,31,WIDTH-MARGIN,31)
    text(pdf,label,MARGIN,18,7,MUTED)
    pdf.setFont('Amend',7);pdf.drawRightString(WIDTH-MARGIN,18,f'APPENDIX {section} / 1 OF 1')


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


def threshold_page(pdf, root, report, predictions, images, gold, physical_page):
    heading(pdf,'Two fixed-threshold crossings','Appendix A. Same original pixels and reference within each pair. Cyan: stated reference. Magenta: returned output from the named PR119 arm. Cutoffs are unchanged.')
    control_path=root/report['models']['reference']['prediction_path'];require(sha(control_path)==report['models']['reference']['predictions_sha256'],'Control cache drift')
    _,*rows=[json.loads(x) for x in control_path.read_text().splitlines()];control={r['image_id']:r for r in rows}
    frozen=read_reference(root,report,'frozen666',predictions,images);w=(WIDTH-2*MARGIN-18)/2;inventory=[]
    cases=[('frozen666','Frozen 666','new_000089',43,frozen),('r35','R35 review','new_000056',19,gold)]
    for slot,(ref,label,name,row,targets) in enumerate(cases):
        g=targets[name][row-1];observations=[]
        for role,cache in [('reference',control),('candidate',predictions)]:
            preds=cache[name]['predictions'];score=score_image(targets[name],preds,1280,720,.5,.5)
            matched=next((m for m in score['matches'] if m['gold_index']==row-1),None)
            pi=matched['prediction_index'] if matched else max(range(len(preds)),key=lambda i:_box_iou(g['bbox_xyxy'],preds[i]['bbox_xyxy']))
            p=preds[pi];observations.append({'role':role,'prediction_id':pi+1,'matched':matched is not None,'iou':_box_iou(g['bbox_xyxy'],p['bbox_xyxy']),'confidence':p['confidence'],'left_error_px':math.dist(g['keypoints'][0][:2],p['keypoints'][0][:2]),'prediction':p})
        old,new=observations;bounds=bounds_for([g['bbox_xyxy']]+[o['prediction']['bbox_xyxy'] for o in observations]);top=HEIGHT-126-slot*221
        text(pdf,f'{label} / {name} / row {row}',MARGIN,top,12,INK,True)
        text(pdf,f"R34-LABEL CONTROL / p{old['prediction_id']}",MARGIN,top-17,8,MAGENTA,True)
        text(pdf,f"R35-LABEL CANDIDATE / p{new['prediction_id']}",MARGIN+w+18,top-17,8,MAGENTA,True)
        for x,o in [(MARGIN,old),(MARGIN+w+18,new)]:draw_crop(pdf,images[name],bounds,x,top-24,w,126,[(g,CYAN,f'row {row}',False),(o['prediction'],MAGENTA,f"p{o['prediction_id']}",True)])
        if ref=='frozen666':
            require(old['matched'] and not new['matched'] and old['iou']>=.5>new['iou'],'Frozen threshold illustration differs')
            description=f"Assigned IoU {old['iou']:.6f} -> best IoU {new['iou']:.6f}; matching requires >=0.50. The candidate still returns this overlapping box at confidence {new['confidence']:.6f}. This added frozen-reference miss is a gate crossing, not an absent output."
        else:
            require(old['matched'] and new['matched'] and old['left_error_px']>5>=new['left_error_px'],'Corner threshold illustration differs')
            description=f"The front remains matched. Left-corner distance {old['left_error_px']:.6f} -> {new['left_error_px']:.6f} px crosses inside the fixed five-pixel review radius. This is the sole R35 corner-error row removed from the candidate catalogue; the ten-pixel pair count is unchanged."
        y=para(pdf,description,MARGIN,top-160,WIDTH-2*MARGIN,8.5,11.4);text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'Threshold card overflow')
        inventory.append({'page':physical_page,'reference':ref,'image_id':name,'row':row,'crop_xyxy':bounds,'observations':[{k:v for k,v in o.items() if k!='prediction'} for o in observations]})
    appendix_footer(pdf,'A','PAIRED PR119 COMPARISON / FIXED IoU 0.50 AND FIVE-PIXEL REVIEW RADIUS');pdf.showPage();return inventory


def r36_page(pdf,root,report,predictions,images,physical_page):
    path=root/SOURCE/'r36-reference-delta.json';delta=json.loads(path.read_text());manifest=root/report['references']['r36']['manifest']
    require(sha(manifest)==delta['r36_manifest_sha256']==report['references']['r36']['manifest_sha256'],'R36 manifest drift')
    changes=delta['models']['candidate'];require(not changes['retained_row_assignment_changes'] and not changes['output_category_changes'],'R36 assignment/category delta needs review')
    heading(pdf,'R36 reference replay: counts are unchanged','Appendix B. Same PR119 candidate and predictions; no R36 model has been trained. All compared counts and assignments stay unchanged: 659 fronts, 583 eligible pairs; candidate 15 misses, 93 unmatched, 31 duplicates, 19 corner-error rows.')
    w=(WIDTH-2*MARGIN-18)/2;inventory=[]
    require(len(changes['changed_reference_rows'])==2,'R36 development inventory differs')
    for slot,item in enumerate(changes['changed_reference_rows']):
        name=item['image_id'];row=item['r35_row'];old,new=item['r35'],item['r36'];pid=old['matched_prediction_id']
        require(pid==new['matched_prediction_id'] and pid is not None,'R36 retained assignment differs')
        p=predictions[name]['predictions'][pid-1];g0,g1=old['reference'],new['reference'];require(g0['keypoints']==g1['keypoints'],'R36 illustration unexpectedly changes a point')
        bounds=bounds_for([g0['bbox_xyxy'],g1['bbox_xyxy'],p['bbox_xyxy']]);top=HEIGHT-137-slot*215
        text(pdf,f'{name} / row {row} / same p{pid}',MARGIN,top,12,INK,True)
        text(pdf,'R35 REFERENCE + SAME OUTPUT',MARGIN,top-17,8,CYAN,True);text(pdf,'R36 REFERENCE + SAME OUTPUT',MARGIN+w+18,top-17,8,CYAN,True)
        for x,g in [(MARGIN,g0),(MARGIN+w+18,g1)]:draw_crop(pdf,images[name],bounds,x,top-24,w,120,[(g,CYAN,f'row {row}',False),(p,MAGENTA,f'p{pid}',True)])
        amendment='Right box edge 910 -> 913; right point remains (909,491).' if name=='new_000010' else 'Left box edge 777.000120 -> 782; both corner coordinates are unchanged.'
        description=f"{amendment} Assigned IoU {old['matched_iou']:.6f} -> {new['matched_iou']:.6f}. No match, duplicate category, available-point error, or pair count changes. This is a revised reference box, not model learning."
        y=para(pdf,description,MARGIN,top-154,WIDTH-2*MARGIN,8.5,11.4);text(pdf,'Original crop '+str(bounds),MARGIN,y-12,7.2,MUTED);require(y-12>39,'R36 card overflow')
        inventory.append({'page':physical_page,'image_id':name,'r35_row':row,'r36_row':item['r36_row'],'prediction_id':pid,'crop_xyxy':bounds,'r35_iou':old['matched_iou'],'r36_iou':new['matched_iou']})
    appendix_footer(pdf,'B','R36 REFERENCE-ONLY RESCORE / UNCHANGED PREDICTIONS AND POINT TARGETS');pdf.showPage();return {'manifest_sha256':sha(manifest),'reference_delta_sha256':sha(path),'inventory':inventory}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--comparison',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inventory',type=Path)
    args=parser.parse_args();root=args.root.resolve();comparison=args.comparison or root/SOURCE/'comparison.json';report=json.loads(comparison.read_text())
    source=report['models']['candidate'];cache=root/source['prediction_path'];require(sha(cache)==source['predictions_sha256'],'Candidate cache drift')
    header,*rows=[json.loads(x) for x in cache.read_text().splitlines()];predictions={r['image_id']:r for r in rows};images={}
    for name,r in predictions.items():
        path=root/'data/images/val'/f'{name}.png';require(sha(path)==r['image_sha256'],'Original image drift');images[name]=Image.open(path).convert('RGB')
    gold=read_reference(root,report,'r35',predictions,images);candidate=report['references']['r35']['models']['candidate']
    cases=[dict(x,category='miss') for x in candidate['misses']]+[dict(x,category='corner') for x in candidate['corner_error_rows']];unmatched=candidate['unmatched_outputs']
    require(len(candidate['misses'])==15 and len(candidate['corner_error_rows'])==19 and len(unmatched)==93,'Candidate catalogue differs')
    require(len({(x['image_id'],x['row']) for x in candidate['misses']})==15,'Repeated missed row')
    require(len({(x['image_id'],x['prediction_id']) for x in unmatched})==93,'Repeated unmatched output')
    for name,r in predictions.items():
        score=score_image(gold[name],r['predictions'],r['width'],r['height'],.5,.5)
        require({i+1 for i in score['unmatched_gold_indices']}=={x['row'] for x in candidate['misses'] if x['image_id']==name},'Miss ledger differs')
        require({i+1 for i in score['unmatched_prediction_indices']}=={x['prediction_id'] for x in unmatched if x['image_id']==name},'Unmatched ledger differs')
    register_fonts();args.output.parent.mkdir(parents=True,exist_ok=True);pdf=canvas.Canvas(str(args.output),pagesize=(WIDTH,HEIGHT));pdf.setTitle('PR119 R35 misses and errors');pdf.setAuthor('Cassette-front review')
    pages=3+math.ceil(len(cases)/2)+math.ceil(len(unmatched)/6);inventory=[]
    cover(pdf,report,pages);definitions(pdf,report,pages);count_page(pdf,candidate,pages);page=4
    for offset in range(0,len(cases),2):detailed_page(pdf,cases[offset:offset+2],gold,predictions,images,page,pages,inventory);page+=1
    for offset in range(0,len(unmatched),6):unmatched_page(pdf,unmatched[offset:offset+6],gold,predictions,images,page,pages,inventory);page+=1
    require(page-1==pages,'Main page inventory mismatch')
    paired=threshold_page(pdf,root,report,predictions,images,gold,pages+1)
    r36=r36_page(pdf,root,report,predictions,images,pages+2);pdf.save()
    result={'schema_version':1,'pages':pages+2,'r35_main_pages':pages,'pdf_sha256':sha(args.output),'source_comparison_sha256':sha(comparison),'candidate_predictions_sha256':sha(cache),'r35_reference_manifest_sha256':report['references']['r35']['manifest_sha256'],'source_image_sha256':{name:predictions[name]['image_sha256'] for name in sorted(images)},'missed_rows':15,'corner_error_rows':19,'unmatched_outputs':93,'duplicates':31,'other_unmatched':62,'inventory':inventory,'paired_threshold_appendix':paired,'r36_reference_appendix':r36}
    target=args.inventory or root/SOURCE/'pdf-inventory.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inventory','source_image_sha256')},indent=2))


if __name__=='__main__':main()
