#!/usr/bin/env python3
"""Render R50 per-case analysis of the fixed PR137 prediction archive.

The frozen R38 inventory is preserved for continuity. R50 is a separately
versioned reference replay, never a new model result. This script reads images,
labels and predictions; it does not load a model or change any annotation.
"""
import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import zipfile

from export_front_error_review_r48 import collect, originals_for, best_overlap, corner_details, sha
from audit_additional_front_pilot_r47_results import ARCHIVE_SHA, PR_HEAD, REFERENCES
from audit_additional_front_pilot_r45_results import corner_groups
from evaluate_pose import parse_pose_labels
from pose_scoring import score_image, aggregate_scores

R50_MANIFEST = 'data/reviews/fronts-2026-11-27-r50/manifest.json'
R50_SHA = 'f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f'
PR140_HEAD = '55a6f68993a63fcdf422014995fada33352720df'
OUT_DIR = 'results/evaluation/front-errors-r50-20261127'
KIND_NAMES = dict(miss='Missed target', corner='Matched corner failure', duplicate='Formal duplicate',
                  near_gate='Other / near gate', positive_overlap='Other / positive overlap', zero_overlap='Other / zero overlap')


def case_id(case):
    if case['kind'] in ('miss', 'corner'):
        return f"{'M' if case['kind']=='miss' else 'K'}:{case['image']}:R{case['row']:03d}"
    return f"P:{case['image']}:P{case['prediction_number']:03d}"


def all_cases(inventory):
    return inventory['misses'] + inventory['corner_failures'] + inventory['unmatched_predictions']


def load_current(root, predictions, frozen_gold):
    path = root / R50_MANIFEST
    assert sha(path.read_bytes()) == R50_SHA, 'R50 manifest pin differs'
    manifest = json.loads(path.read_text())
    gold, label_hashes = {}, {}
    for record in manifest['files']:
        identity = record['image_id']
        label = path.parent / record['label_path']
        digest = sha(label.read_bytes())
        assert digest == record['label_sha256'], identity
        label_hashes[str(label.relative_to(root))] = digest
        gold[identity] = parse_pose_labels(label.read_text(), 1280, 720)
        assert len(gold[identity]) == len(frozen_gold[identity]), 'Row membership changed'
    assert set(gold) == set(frozen_gold) and len(gold) == 20
    assert sum(map(len, gold.values())) == 654
    visibility = Counter(str(sum(p[2] == 2 for p in t['keypoints'])) for ts in gold.values() for t in ts)
    assert visibility == {'2': 583, '1': 54, '0': 17}, visibility
    reports = {}
    for role in ('reference', 'candidate'):
        per_image, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
        for identity, targets in gold.items():
            ps = predictions[role][identity]['postclip']
            score = score_image(targets, ps, 1280, 720, .5, .5)
            per_image[identity] = score
            for key, values in corner_groups(targets, ps, score).items():
                groups[key].update(values)
        reports[role] = dict(detection=aggregate_scores(list(per_image.values())), per_image=per_image,
                             corners_by_locatable_count={k: dict(v) for k, v in groups.items()})
    return gold, reports, label_hashes


def attach_analysis(inventory, ledger, originals_hashes, gold, current, predictions):
    cases = all_cases(inventory)
    observations = {c['case_id']: c for c in ledger['cases']}
    ids = [case_id(c) for c in cases]
    assert len(ids) == len(set(ids)) == 119
    assert set(ids) == set(observations), 'Visual ledger must cover every fixed-R38 case exactly once'
    current_ids, current_kinds = set(), Counter()
    for identity, scores in current['candidate']['per_image'].items():
        ps = predictions['candidate'][identity]['postclip']
        for row in scores['unmatched_gold_indices']:
            current_ids.add(f'M:{identity}:R{row+1:03d}')
            current_kinds['miss'] += 1
        for match in scores['matches']:
            points = corner_details(gold[identity][match['gold_index']], ps[match['prediction_index']])
            if any(p['eligible'] and (not p['accepted'] or p['error_px'] > 5) for p in points):
                current_ids.add(f"K:{identity}:R{match['gold_index']+1:03d}")
                current_kinds['corner'] += 1
        for index in scores['unmatched_prediction_indices']:
            current_ids.add(f'P:{identity}:P{index+1:03d}')
            _, overlap = best_overlap(ps[index]['bbox_xyxy'], gold[identity])
            kind = 'duplicate' if index in scores['duplicate_prediction_indices'] else 'near_gate' if overlap >= .4 else 'positive_overlap' if overlap > 0 else 'zero_overlap'
            current_kinds[kind] += 1
    assert current_ids == set(ids), 'Current reference introduced/resolved a case: extend continuity atlas explicitly'
    transitions = []
    for case in cases:
        cid, identity = case_id(case), case['image']
        note = observations[cid]
        assert note['original_pixels_inspected'] and note['full_image_context_inspected']
        assert note['original_image_sha256'] == originals_hashes[identity], cid
        assert note['observed_type'] in ledger['taxonomy']
        case['case_id'], case['analysis'] = cid, note
        score = current['candidate']['per_image'][identity]
        ps = predictions['candidate'][identity]['postclip']
        if case['kind'] in ('miss', 'corner'):
            row = case['row'] - 1
            pair = next((m for m in score['matches'] if m['gold_index'] == row), None)
            index, overlap = (pair['prediction_index'], pair['iou']) if pair else best_overlap(gold[identity][row]['bbox_xyxy'], ps)
            pred = ps[index] if index is not None else None
            points = corner_details(gold[identity][row], pred) if pred else []
            status = 'miss' if pair is None else 'corner' if any(p['eligible'] and (not p['accepted'] or p['error_px'] > 5) for p in points) else 'matched_pass'
            data = dict(kind=status, row=case['row'], target=gold[identity][row], matched=pair is not None,
                        iou=overlap, prediction_number=index+1 if index is not None else None, corners=points,
                        reference_changed=gold[identity][row] != case['target'])
        else:
            index = case['prediction_index']
            row, overlap = best_overlap(ps[index]['bbox_xyxy'], gold[identity])
            kind = 'duplicate' if index in score['duplicate_prediction_indices'] else 'near_gate' if overlap >= .4 else 'positive_overlap' if overlap > 0 else 'zero_overlap'
            owner = next((m['prediction_index'] for m in score['matches'] if m['gold_index'] == row), None)
            data = dict(kind=kind, best_row=row+1 if row is not None else None, best_iou=overlap,
                        target=gold[identity][row] if row is not None else None,
                        owner_prediction_number=owner+1 if owner is not None else None,
                        reference_changed=(gold[identity][row] if row is not None else None) != case['target'])
        case['current_r50'] = data
        if data['kind'] != case['kind']:
            transitions.append(dict(case_id=cid, r38=case['kind'], r50=data['kind']))
    inventory['schema'] = 'baomai-r50-fixed-prediction-error-review-v1'
    inventory['source_status'] = 'PR137 latest completed model result; PR140 repairs notebook checkout only'
    inventory['review_base_pr'] = 140
    inventory['review_base_head'] = PR140_HEAD
    inventory['current_reference'] = dict(name='r50-proposed', manifest_path=R50_MANIFEST, manifest_sha256=R50_SHA,
       same_model_predictions=True, same_cases=True, reports=current, counts=dict(current_kinds), category_transitions=transitions)
    inventory['visual_type_counts'] = dict(Counter(c['analysis']['observed_type'] for c in cases))
    inventory['visual_types_count_unit'] = 'review cards; overlapping cases may describe the same physical cassette'
    inventory['visual_taxonomy'] = ledger['taxonomy']
    inventory['visual_inspection'] = dict(original_frames=len(originals_hashes), original_crops=119,
        methods='Full original context plus original/overlay crops, with an independent 12-case zero-overlap review',
        causal_claim='Appearance is observed; learning and earlier suppression mechanisms remain unproven',
        human_review_claim=False, label_writes=False)
    return cases


def render(inventory, evidence, originals, predictions, output):
    from PIL import Image
    import reportlab
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle

    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('R50', str(fonts / 'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('R50Bold', str(fonts / 'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('R50', normal='R50', bold='R50Bold', italic='R50', boldItalic='R50Bold')
    W, H = landscape(A4)
    M, BW = 30, W-60
    ink, muted, teal, line, amber, cyan, green = map(HexColor, ('#173043','#526877','#087f85','#d6e1e6','#ffad32','#00d9ee','#a8ff60'))
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(W,H), invariant=1, pageCompression=1)
    pdf.setTitle('R50 - PR137 error types, evidence and next actions')
    pdf.setAuthor('Siuuuers/dl project review')
    page, page_map, layout_checks = 0, [], []
    esc = lambda s: html.escape(str(s))

    def text(x,y,value,size=9,bold=False,color=None):
        pdf.setFillColor(color or ink); pdf.setFont('R50Bold' if bold else 'R50',size)
        pdf.drawString(x,y,str(value))

    def para(value,x,y,width=BW,size=9,leading=13,color=None,bottom=38):
        style=ParagraphStyle('r50',fontName='R50',fontSize=size,leading=leading,textColor=color or ink)
        p=Paragraph(value,style); _,height=p.wrap(width,1000)
        assert y-height >= bottom-0.01, f'Paragraph overflow page {page}: {y-height:.1f} < {bottom}: {value[:65]}'
        p.drawOn(pdf,x,y-height)
        layout_checks.append(dict(page=page,bottom=round(y-height,3),required_bottom=bottom,kind='paragraph'))
        return y-height

    def start(section,title,subtitle):
        nonlocal page
        page+=1
        pdf.setFillColor(teal);pdf.rect(0,H-7,W,7,stroke=0,fill=1)
        text(M,H-30,section.upper(),8,True,teal)
        text(M,H-57,title,19,True)
        para(esc(subtitle),M,H-73,size=8.8,leading=11.5)

    def end():
        pdf.setStrokeColor(line);pdf.line(M,31,W-M,31)
        text(M,17,'R50 review | 27 Nov 2026 HKT | PR137 fixed cache | R38 review + R50 proposed reference',7,color=muted)
        text(W-M-22,17,f'{page:02d}',9,True,teal);pdf.showPage()

    def table(headers,rows,y,widths,row_height=25,size=8.7,header_height=None):
        for i,row in enumerate([headers]+rows):
            height=header_height if i==0 and header_height is not None else row_height
            pdf.setFillColor(HexColor('#e8f2f3') if i==0 else HexColor('#f4f7f9') if i%2 else white)
            pdf.rect(M,y-height,sum(widths),height,stroke=0,fill=1)
            x=M
            for value,width in zip(row,widths):
                para(('<b>'+esc(value)+'</b>') if i==0 else esc(value),x+7,y-6,width-14,size,size+2,bottom=y-height+3)
                x+=width
            y-=height
        return y-13

    start('Result status','PR137 remains the latest completed model result.','PR140 fixes notebook checkout; it does not contain a new training, pre-NMS or Soft-NMS result.')
    y=para('The supplementary-data candidate reduced unmatched outputs but added two misses. R50 reuses those exact predictions and adds original-pixel analysis plus a separately versioned reference. PR35 remains selected; this review does not promote a new model.',M,482,size=10.5,leading=15)
    rows=[]
    for name,reports in [('Frozen R38',None),('Proposed R50',inventory['current_reference']['reports'])]:
        for role in ('reference','candidate'):
            if reports is None:
                values=(641,13,90,18,'569 / 583') if role=='reference' else (639,15,84,14,'566 / 583')
            else:
                d=reports[role]['detection'];c=reports[role]['corners_by_locatable_count']['2']
                values=(d['tp'],d['fn'],d['fp'],d['duplicate_fp'],f"{c['within5']} / {c['eligible']}")
            rows.append((name,role.title(),*values))
    y=table(['Reference labels','Model arm','Matched','Missed','Unmatched','Dup. subset','Both corners <=5px'],rows,y-14,[140,105,72,72,93,93,BW-575],row_height=29,size=8.5)
    y=para('<b>R50 changes the reference, not the model.</b> It inherits the nine R48 development-row amendments and changes only new_000010/R029 bottom from y=720 to y=714, after the user identified the lower band as another cassette. Its corner tokens remain unchanged. Both arms retain all prior target-to-output assignments.',M,y,size=10,leading=14)
    y=para('The atlas retains <b>all 119 frozen-R38 cases</b>: 15 misses, 20 matched fronts with a visible-corner error above 5px, and 84 unmatched outputs. R50 retains the same 119 case identities. Five unmatched diagnostic bins change, and the formal-duplicate subset falls by one overall. This is reference sensitivity, not improved learning.',M,y-12,size=10,leading=14)
    para('Historical frozen666 primary results also remain fixed: reference/candidate misses 33/34, unmatched 98/91, duplicate subsets 22/17. These reference versions describe the same photographs and are not independent test sets.',M,y-12,size=9,leading=13,color=muted)
    end()

    start('Observed types','A scorer category and a physical explanation are different.','All 119 original crops were inspected in full-image context; type counts are review cards, not unique physical cassettes.')
    short={
      'occlusion_extent':'Box crosses a gripper/hand or differs from the visible fragment.',
      'corner_disagreement':'Matched box, accepted point more than 5px from the recorded corner.',
      'repeated_front_prediction':'Additional retained hypothesis for an already assigned front.',
      'image_boundary_extent':'Clipped target or top-edge strip has incompatible retained extent.',
      'noncassette_response':'Keyboard/peripheral or support-band response established from pixels.',
      'front_extent_or_neighbor_mix':'Cap, seam, lower-band or partial-width ownership differs.',
      'ownership_unresolved':'Four top strips remain unproven as complete cassette fronts.',
      'excluded_reference_front':'Three outputs touch one visible fragment explicitly excluded by the user.',
      'occluded_front_no_localized_box':'Three zero-corner targets lack a suitably localized retained box.',
    }
    rows=[(inventory['visual_taxonomy'][k]['label'],n,short[k]) for k,n in sorted(inventory['visual_type_counts'].items(),key=lambda x:(-x[1],x[0]))]
    y=table(['Observed type','Cards','What was actually seen'],rows,484,[240,55,BW-295],row_height=31,size=8.5)
    y=para('<b>Zero overlap is not a background verdict.</b> Among the 12 zero-R38-overlap outputs, five lie on keyboard/peripheral structures, three cover the same user-excluded cassette-associated sliver, and four top strips remain ownership holds. Two additional positive-overlap outputs mainly cover the lower rack/support band.',M,y,size=9,leading=13)
    para('The independent review traced new_000009/P057, P065 and P066 to the region of old R37 row 52, explicitly deleted in R38. Preserve that exclusion. Their overlaps with the old box are 0.3303, 0.4091 and 0.2206, so none is a recovered detection at IoU 0.50 even under that old rectangle.',M,y-10,size=9,leading=13)
    end()

    start('Misses and corner failures','The immediate target is reliable partial-front geometry.','All 15 candidate misses have fewer than two locatable reference corners; original images show truncation or foreground occlusion.')
    y=table(['Locatable corners','R38 targets','Reference misses','Candidate misses','Added training fronts'],[
      ('Two',583,0,0,164),('One',54,10,12,8),('Zero',17,3,3,0)],484,[170,110,160,160,BW-600],row_height=31,size=9)
    y=para('<b>Observed evidence:</b> three miss cards are image-boundary cases. Nine show retained boxes that cross a gripper or thumb rather than isolating a visible front. Three zero-corner references have no suitably localized retained box. The last category cannot be assigned to low confidence, NMS or failure to propose from this cache.',M,y,size=10,leading=14)
    y=para('The added 172 training fronts contain 164 two-corner examples and only eight one-corner examples. That coverage mismatch motivates a controlled partial-front study; it does not prove why the candidate changed. Existing training labels already contain 465 one-corner and 137 zero-corner fronts in the candidate arm.',M,y-11,size=10,leading=14)
    y=para('<b>Corner cases are a separate outcome.</b> All 20 shown targets are box-matched, but a visible accepted endpoint fails the original 5px gate. The cards give exact displacement and local rim/rack/gripper context. Independently re-mark the physical rim/side intersection before assuming either the model or label is correct.',M,y-11,size=10,leading=14)
    y=para('Missed partial-box heights span 10.0-45.56px, median 23px; matched partial boxes overlap that range (11-60px, median 26.18px). Matched two-corner boxes can be as short as 9.5px. Size may matter, but neither size nor dimness is established as the single cause.',M,y-11,size=9,leading=13)
    para('<b>Coordinate rule:</b> stored image y increases downward. ymin is top; ymax is bottom. A smaller cassette does not justify annexing rack, shadow from a neighbor, gripper or another cassette. Ownership and visibility rules stay consistent across scale.',M,y-11,size=9,leading=13,color=teal)
    end()

    start('Existing filter evidence','A filter is useful only with its recall cost measured.','These are CPU replays of retained PR137 predictions on R38; no new inference or filter benefit is invented.')
    rows=[]
    for point in inventory['diagnostics']['confidence_replay']:
        a,b=point['reference'],point['candidate']; da,db=a['detection'],b['detection']
        rows.append((f"{point['confidence']:.2f}",f"{da['fn']} / {db['fn']}",f"{da['fp']} / {db['fp']}",f"{da['duplicate_fp']} / {db['duplicate_fp']}",f"{a['corners_by_locatable_count']['2']['within5']} / {b['corners_by_locatable_count']['2']['within5']}"))
    y=table(['Box confidence','Miss R / C','Unmatched R / C','Duplicates R / C','Both corners <=5px R / C'],rows,484,[120,135,170,155,BW-580],row_height=26,size=8.7)
    y=para('R/C means reference/candidate; the two-corner success denominator is 583. Raising candidate confidence from 0.13 to 0.25 reduces unmatched 84 to 51, but misses rise 15 to 18. That is a trade-off, not free improvement. The selected deployment confidence remains 0.13.',M,y,size=10,leading=14)
    y=para('The already-used extra postclip suppression removes nine candidate outputs: 732 to 723 total, unmatched 93 to 84, formal duplicates 22 to 14, with 15 misses unchanged. The saved "raw" list is already after ordinary NMS; it cannot reveal candidates suppressed earlier.',M,y-11,size=10,leading=14)
    para('The model does not need to "call" a filter. A simple inference pipeline can apply a deterministic postprocessor after decoding. First log confidence gating, ordinary NMS, clipping and postclip suppression separately. Then test one change with fixed matching, original-image geometry, misses, unmatched outputs and corner success. Soft-NMS, enhancement and higher-resolution results remain unmeasured here.',M,y-11,size=10,leading=14,color=teal)
    end()

    start('Actions with available data','Finish diagnosis before another broad training run.','No step below requires waiting for new captures. Keep development photographs out of training.')
    rows=[
      ('1. Resolve ownership','Use the current guide, original contours, neighboring existing frames and recorded exclusions. Preserve R37:52 deletion and the R50 y=714 seam.','A versioned label ledger; unresolved rows remain holds.'),
      ('2. Locate the failure stage','Re-run the completed model with proposals saved before confidence/NMS, plus the current stages. Reproduce fixed-cache counts first.','Misses separated into absent proposal, low score, suppression, extent and reference issues.'),
      ('3. Correct a new label delta','After diagnosis, correct newly supported eligible training-label errors. Keep native loss, whole-image membership and update budget fixed; hold training if no new delta remains.','One annotation-only treatment, or an explicit no-new-training hold.'),
      ('4. Test the smallest filter','Compare current postclip logic with one justified alternative; use endpoint/extent checks only after validating them on partial fronts.','Joint miss, unmatched and corner tables; no per-frame exceptions or label leakage.'),
      ('5. Test scale or brightness','Change one inference factor at a time, retaining coordinate inversion and identical matching. Review failures by visibility/size/context.','Evidence of a benefit and its runtime/recall cost, or a documented null result.'),
    ]
    y=table(['Priority','Concrete existing-data action','Reviewable output'],rows,484,[155,365,BW-520],row_height=62,header_height=30,size=8.5)
    y=para('Hard-negative mining should use verified noncassette regions from training images. An unmatched development box is not automatically a hard negative: it may be an excluded front, a real but ambiguously owned fragment, or a poorly localized output.',M,y,size=9.5,leading=13)
    y=para('Any future claim of generalization should use a capture/session-separated holdout when available. Until then, document the fixed development-set limitation and avoid choosing a new architecture from these 20 repeatedly reviewed photographs alone.',M,y-10,size=9,leading=12)
    para('<b>Prior trials:</b> R29 partial-image sampling and R31 factor-two partial-box loss already ran. R31 was mixed; its seed47 screen failed. See docs/project-direction-r50.md before proposing another treatment.',M,y-7,size=8,leading=10)
    end()

    start('A manageable thesis direction','Make annotation, detection and corner accuracy measurable.','A proposed scope for discussion with the professor; the lecture is background, not an assigned thesis specification.')
    y=para('<b>Candidate research question:</b> How do consistent visible-front annotations and partial-front supervision affect detection and two-corner localization in crowded tissue-cassette images?',M,484,size=12,leading=17)
    y=para('Begin with a reproducible baseline, a frozen label version and a documented visible-extent/corner contract. Then run one annotation or training-data treatment and one postprocessing treatment separately. The unit of independence is a capture/session, not neighboring frames of the same rack.',M,y-14,size=10.5,leading=15)
    y=table(['Report together','Why it belongs in the study'],[
      ('Detection at the operating point','TP, FN, all unmatched, formal duplicates, precision and recall.'),
      ('Visibility strata','Separate zero-, one- and two-locatable-corner targets.'),
      ('Corner success','Use original-pixel endpoint error and unconditional within5/10px success.'),
      ('Error mechanisms','Keep scorer category, observed type, reference uncertainty and cause hypothesis separate.'),
      ('Cost and reproducibility','Record input size, runtime, seeds, image exposure and actual optimizer updates.'),
    ],y-15,[250,BW-250],row_height=32,size=9)
    y=para('<b>Fixed precision/recall is not AP.</b> At the R38 operating point, candidate precision is639/723=88.38% and recall is639/654=97.71%. These are one threshold pair. AP evaluates a ranked precision-recall curve; COCO-style AP also averages over IoU thresholds. The truncated retained cache is insufficient to claim a complete AP benchmark.',M,y,size=10,leading=14)
    para('Lecture connection: upload/detection.pdf p.5 defines IoU; pp.9-11 cover confidence ranking, precision/recall, the precision-recall curve and AP. One-to-one greedy assignment and the duplicate subset in this report are defined by scripts/pose_scoring.py, not inferred from the lecture illustration.',M,y-11,size=9,leading=13,color=muted)
    end()

    def crop_bounds(boxes):
        x0,y0=min(b[0] for b in boxes)-22,min(b[1] for b in boxes)-20
        x1,y1=max(b[2] for b in boxes)+22,max(b[3] for b in boxes)+20
        w,h=min(1280,max(92,x1-x0)),min(720,max(68,y1-y0));cx,cy=(x0+x1)/2,(y0+y1)/2
        x0,y0=min(max(0,cx-w/2),1280-w),min(max(0,cy-h/2),720-h)
        return math.floor(x0),math.floor(y0),math.ceil(x0+w),math.ceil(y0+h)

    def panel(image,bounds,x,y,w,h,target=None,pred=None,owner=None,current_target=None):
        crop=image.crop(bounds); enlarged=crop.resize((crop.width*3,crop.height*3),Image.Resampling.NEAREST)
        scale=min(w/crop.width,h/crop.height);iw,ih=crop.width*scale,crop.height*scale;ox,oy=x+(w-iw)/2,y+(h-ih)/2
        pdf.drawImage(ImageReader(enlarged),ox,oy,width=iw,height=ih)
        def coord(px,py):return ox+(px-bounds[0])*scale,oy+ih-(py-bounds[1])*scale
        pdf.saveState();clip=pdf.beginPath();clip.rect(ox,oy,iw,ih);pdf.clipPath(clip,stroke=0)
        def draw(obj,color,predicted=False,dashed=False):
            bx,by,ex,ey=obj['bbox_xyxy'];left,bottom=coord(bx,ey);right,top=coord(ex,by)
            pdf.setStrokeColor(color);pdf.setLineWidth(.8);pdf.setDash(2,2) if dashed else pdf.setDash()
            pdf.rect(left,bottom,right-left,top-bottom,stroke=1,fill=0);pdf.setDash()
            if dashed:return
            for px,py,status in obj['keypoints']:
                valid=px is not None and py is not None and math.isfinite(px) and math.isfinite(py)
                if not valid or (predicted and (status<.5 or not 0<=px<1280 or not 0<=py<720)) or (not predicted and status!=2):continue
                xx,yy=coord(px,py);r=1.9
                if predicted:pdf.line(xx-r,yy,xx+r,yy);pdf.line(xx,yy-r,xx,yy+r)
                else:pdf.circle(xx,yy,r,stroke=1,fill=0)
        if owner:draw(owner,amber,True,True)
        if target:draw(target,cyan)
        if current_target and current_target!=target:draw(current_target,green,False,True)
        if pred:draw(pred,amber,True)
        pdf.restoreState()

    def point_note(points):
        return '; '.join(f"{p['side']} {p['error_px']:.2f}px" if p['accepted'] else f"{p['side']} rejected" for p in points if p['eligible'])

    def card(case,x,y,w,h):
        note=case['analysis'];current=case['current_r50'];identity=case['image']
        pdf.setStrokeColor(line);pdf.roundRect(x,y,w,h,5,stroke=1,fill=0)
        left_w=454;right_x=x+477;right_w=w-489
        if case['kind'] in ('miss','corner'):
            title=f"{identity} / R{case['row']:03d} | {case['title']}"
            cand=case['arms']['candidate'];stats=f"R38: {case['visible_corners']} locatable corner(s) | candidate IoU {cand['iou']:.4f}"
            if case['kind']=='corner':stats+=' | '+point_note(cand['corners'])
            boxes=[case['target']['bbox_xyxy']]+[a['prediction']['bbox_xyxy'] for a in case['arms'].values() if a['prediction']]
            boxes.append(current['target']['bbox_xyxy']);bounds=crop_bounds(boxes)
            panels=[(None,None,None,'Original pixels')]
            for role in ('reference','candidate'):
                a=case['arms'][role];label=role.title()+(' matched' if a['matched'] else ' missed')
                if a['prediction']:label+=f" | IoU {a['iou']:.3f}"
                else:label+=' | no overlap'
                panels.append((case['target'],a['prediction'],None,label))
            state=f"R50: {KIND_NAMES.get(current['kind'],current['kind'])}; IoU {current['iou']:.4f}. "
            state+=('Green dotted box shows the changed target.' if current['reference_changed'] else 'Reference geometry unchanged for this row.')
            if case['kind']=='miss':state+=' Nearest overlap is diagnostic, not an assignment.'
        else:
            title=f"{identity} / P{case['prediction_number']:03d} | PR137 candidate | primary #{case['primary_prediction_id']}"
            best=f"R{case['best_row']:03d}" if case['best_row'] else 'none'
            stats=f"R38: {KIND_NAMES[case['kind']]} | best {best}, IoU {case['best_iou']:.4f} | confidence {case['prediction']['confidence']:.3f}"
            boxes=[case['prediction']['bbox_xyxy']]+[t['bbox_xyxy'] for t in (case['target'],current['target']) if t]
            bounds=crop_bounds(boxes)
            owner=predictions['candidate'][identity]['postclip'][case['owner_prediction_number']-1] if case['kind']=='duplicate' else None
            owner50=predictions['candidate'][identity]['postclip'][current['owner_prediction_number']-1] if current['kind']=='duplicate' else None
            panels=[(None,None,None,'Original pixels'),(case['target'],case['prediction'],owner,'R38 / retained candidate'),(current['target'],case['prediction'],owner50,'R50 / same candidate')]
            best50=f"R{current['best_row']:03d}" if current['best_row'] else 'none'
            state=f"R50: {KIND_NAMES[current['kind']]}; best {best50}, IoU {current['best_iou']:.4f}. "
            if owner:state+=f"R38 assigned owner P{case['owner_prediction_number']:03d} is dotted."
            else:state+='Physical type is assessed independently from overlap.'
        text(x+10,y+h-18,title,9.5,True)
        text(x+10,y+h-32,stats,8.1,color=muted)
        gap,pw=9,(left_w-18)/3
        for i,(target,pred,owner,label) in enumerate(panels):
            xx=x+10+i*(pw+gap)
            panel(originals[identity],bounds,xx,y+49,pw,125,target,pred,owner,current['target'] if case['kind'] in ('miss','corner') and target else None)
            text(xx,y+39,label,6.9,True)
        para(esc(state),x+10,y+29,left_w,7.5,9.3,bottom=y+7)
        content=(f"<b>{esc(inventory['visual_taxonomy'][note['observed_type']]['label'])}</b><br/>"
                 f"<b>Observed:</b> {esc(note['observation'])}<br/>"
                 f"<b>Conclusion:</b> {esc(note['conclusion'])}<br/>"
                 f"<b>Next:</b> {esc(note['next_action'])}<br/>"
                 f"<font color='#526877'>Evidence: {esc(note['observation_confidence'].replace('_',' '))}. Causal stage: unproven.</font>")
        para(content,right_x,y+h-45,right_w,8.4,10.65,bottom=y+8)
        page_map.append(dict(case_id=case['case_id'],page=page,source_kind=case['kind'],current_r50_kind=current['kind'],observed_type=note['observed_type']))

    groups=[('Missed targets','All 15 fixed-R38 misses',inventory['misses']),
            ('Corner localization','All 20 matched corner failures',inventory['corner_failures']),
            ('Unmatched outputs','All 84 retained unmatched outputs',inventory['unmatched_predictions'])]
    for section,title,cases in groups:
        for offset in range(0,len(cases),2):
            start(section,title,f"Cases {offset+1}-{min(offset+2,len(cases))} of {len(cases)} | Cyan: reference; amber: prediction; circles: reference corners; crosses: accepted endpoints.")
            for i,c in enumerate(cases[offset:offset+2]):card(c,M,270 if i==0 else 42,BW,217)
            end()

    start('Sources and interpretation','Every card is tied to an original image and cached output.','The JSON records exact coordinates, current-reference status, visual observations, uncertainty and targeted next action.')
    y=para('PR137 archive audit passes before rendering. This includes inventory/source integrity, cached postclip replay and score-report recomputation. Original PNGs match their recorded SHA-256. The R50 reference is hash-pinned and replayed for both arms at the same thresholds. Neither frozen labels nor model checkpoints are changed.',M,484,size=10,leading=14)
    sources=[
      'Completed run: https://github.com/Siuuuers/dl/pull/137',
      'PR137 head: '+PR_HEAD,
      'Notebook fix only: https://github.com/Siuuuers/dl/pull/140',
      'PR140 head: '+PR140_HEAD,
      'Archive SHA-256: '+ARCHIVE_SHA,
      'Frozen R38 manifest SHA-256: '+REFERENCES['r38'][1],
      'Proposed R50 manifest SHA-256: '+R50_SHA,
      'R50 labels: '+R50_MANIFEST,
      'Recorded exclusion: results/review/front-review-20261126-r38/row-review.json; old R37 new_000009 row 52.',
      'Lecture: upload/detection.pdf pp.5, 9-11; scorer: scripts/pose_scoring.py.',
      'Visual ledger: '+OUT_DIR+'/case-observations.json',
      'Independent zero-overlap review: '+OUT_DIR+'/zero-overlap-independent-review.json',
    ]
    for source in sources:y=para(esc(source),M,y-9,size=8,leading=11)
    y=para('<b>Limits:</b> This is an agent visual review of a repeatedly inspected development set, not a new human annotation certification or an independent test. Appearance can support an observed type; it does not establish a network learning mechanism. Four ownership holds remain explicit, and the user-excluded fragment stays excluded.',M,y-13,size=9,leading=13)
    para('Box confidence 0.13; ordinary NMS IoU 0.70; extra postclip suppression uses strictly greater than 0.70; matching IoU>=0.50; accepted endpoint confidence>=0.50; original image 1280x720. P### always means the PR137 candidate postclip index in this atlas, never an earlier PR output number.',M,y-10,size=9,leading=13,color=teal)
    end();pdf.save()
    return page,page_map,layout_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--images-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--results-dir',type=Path)
    args=parser.parse_args();root=args.root.resolve();results=args.results_dir or root/OUT_DIR
    results.mkdir(parents=True,exist_ok=True)
    ledger_path=results/'case-observations.json';ledger=json.loads(ledger_path.read_text())
    inventory,evidence,summary,frozen_gold=collect(root,args.archive)
    originals,hashes=originals_for(inventory,root,args.images_dir,False)
    inventory['original_image_sha256']=hashes
    with zipfile.ZipFile(args.archive) as z:predictions=json.loads(z.read('evaluation/predictions.json'))
    gold,current,label_hashes=load_current(root,predictions,frozen_gold)
    cases=attach_analysis(inventory,ledger,hashes,gold,current,predictions)
    pages,page_map,layout=render(inventory,evidence,originals,predictions,args.output)
    inventory['case_pages']=page_map
    assert len(page_map)==119 and len({c['case_id'] for c in page_map})==119
    assert all(sha((root/path).read_bytes())==digest for path,digest in label_hashes.items()), 'Reference changed during render'
    index=results/'error-index.json';index.write_text(json.dumps(inventory,indent=2,sort_keys=True,allow_nan=False)+'\n')
    analyses=dict(schema='r50-case-analysis-v1',source_pr=137,source_archive_sha256=ARCHIVE_SHA,
        reference_manifest_sha256=REFERENCES['r38'][1],current_reference_manifest_sha256=R50_SHA,
        count=119,type_counts=inventory['visual_type_counts'],count_unit=inventory['visual_types_count_unit'],
        cases=[dict(case_id=c['case_id'],scorer_category=c['kind'],current_r50=c['current_r50'],analysis=c['analysis']) for c in cases])
    (results/'case-analysis.json').write_text(json.dumps(analyses,indent=2,sort_keys=True,allow_nan=False)+'\n')
    verification=dict(schema='r50-error-pdf-verification-v1',pages=pages,pdf_bytes=args.output.stat().st_size,
        pdf_sha256=sha(args.output.read_bytes()),error_index_sha256=sha(index.read_bytes()),
        observations_sha256=sha(ledger_path.read_bytes()),case_count=119,unique_case_count=119,
        r38_counts=inventory['counts'],r50_counts=inventory['current_reference']['counts'],
        visual_type_counts=inventory['visual_type_counts'],original_image_sha256=hashes,
        cpu_audit_status=evidence['status'],archive_sha256=ARCHIVE_SHA,r50_manifest_sha256=R50_SHA,
        reference_labels_unchanged_during_render=True,fresh_training=False,fresh_inference=False,
        paragraph_bounds_pass=True,paragraph_count=len(layout),minimum_body_bottom=min(c['bottom'] for c in layout),
        render_inspection='Pending external PNG page inspection; see visual-verification.json when complete')
    (results/'verification.json').write_text(json.dumps(verification,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (results/'r50-reference-replay.json').write_text(json.dumps(dict(schema='r50-fixed-cache-reference-replay-v1',archive_sha256=ARCHIVE_SHA,manifest_sha256=R50_SHA,reports=current),indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),pages=pages,cases=119,type_counts=inventory['visual_type_counts'],r50_counts=inventory['current_reference']['counts']),indent=2))


if __name__=='__main__':
    main()
