#!/usr/bin/env python3
"""Replay preserved PR121/PR119/PR35 prediction caches against separate references.

This does not run inference, train a model, tune thresholds, or alter official
matching. The maximum-cardinality pass is a diagnostic of the same IoU graph.
"""
import argparse
from pathlib import Path
import hashlib
import json
import math
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'results/evaluation/pr121-r36-20261126'
PREVIOUS = 'results/evaluation/pr119-r35-20261126'
SELECTED = 'results/evaluation/pr114-r33-20261125'
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_pose import parse_pose_labels, INFERENCE
from pose_scoring import score_image, aggregate_scores, _box_iou
from diagnose_pose_quality import quality_for_image, aggregate_quality, _returned

REFERENCES = {
    'frozen666': 'data/references/sequence-refined-reference-2026-11-18/manifest.json',
    'historical663': 'data/references/visible-front-completion-2026-11-14/manifest.json',
    'r35': 'data/reviews/fronts-2026-11-25-r35/manifest.json',
    'r36': 'data/reviews/fronts-2026-11-26-r36/manifest.json',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def max_cardinality(gold, predictions):
    edges = [[i for i,g in enumerate(gold) if g['class_id'] == p['class_id'] and _box_iou(g['bbox_xyxy'], p['bbox_xyxy']) >= .5] for p in predictions]
    assigned = {}
    def augment(p, seen):
        for g in edges[p]:
            if g in seen:
                continue
            seen.add(g)
            if g not in assigned or augment(assigned[g], seen):
                assigned[g] = p
                return True
        return False
    for p in sorted(range(len(predictions)), key=lambda i: (-predictions[i]['confidence'],i)):
        augment(p,set())
    return [{'gold_row':g+1,'prediction_id':p+1} for g,p in sorted(assigned.items())]

def nearest_gold(p, gold):
    if not gold:
        return {'gold_row':None,'iou':0}
    i=max(range(len(gold)),key=lambda i:_box_iou(p['bbox_xyxy'],gold[i]['bbox_xyxy']))
    iou = _box_iou(p['bbox_xyxy'],gold[i]['bbox_xyxy'])
    return {'gold_row':i+1 if iou>0 else None,'iou':iou}

def score_probe_invalid_as_unmatched(gold, pred, width, height, permit_invalid=False):
    """Keep clipped zero-area probe outputs in false-output totals and ID space.

    The production scorer still rejects malformed boxes. For this diagnostic
    only, run it on positive-area records, map all indices back, and account for
    each rejected zero-area record as an invalid other-unmatched output.
    """
    invalid=[]
    for i,p in enumerate(pred):
        x0,y0,x1,y1=p['bbox_xyxy']
        if not all(math.isfinite(v) for v in (x0,y0,x1,y1)):raise ValueError('Non-finite probe box')
        if x1<=x0 or y1<=y0:invalid.append(i)
    if invalid and not permit_invalid:
        raise ValueError('Invalid box in primary inference; audit stops rather than changing its scorer')
    valid=[i for i in range(len(pred)) if i not in invalid];filtered=[pred[i] for i in valid]
    score=score_image(gold,filtered,width,height,.5,.5)
    quality=quality_for_image(gold,filtered,width,height,[5,10])
    for match in score['matches']:match['prediction_index']=valid[match['prediction_index']]
    for key in ('unmatched_prediction_indices','duplicate_prediction_indices','other_fp_prediction_indices'):
        score[key]=[valid[i] for i in score[key]]
    for key in ('unmatched_prediction_indices','other_fp_prediction_indices'):score[key]=sorted(score[key]+invalid)
    for target in (score,quality):
        target['prediction_count']+=len(invalid);target['fp']+=len(invalid)
    score['other_fp']+=len(invalid)
    return score,quality,invalid


def audit(paths, box_confidence=.13, inference_settings=None, permit_invalid=False):
    settings=INFERENCE if inference_settings is None else inference_settings
    document = {'schema_version':1,'inference_settings':settings,'box_confidence':box_confidence,'matching_iou':.5,'endpoint_confidence':.5,'radii_px':[5,10],
        'scope':'PR121 fixed inference and separate reference replays; current-review labels are diagnostic, not an independent holdout. Additional graph matching is diagnostic only and does not change reported scores.',
        'source_revision':'83aeca67264e1f98872abeca40debc212c0775e9','scorer_sha256':sha(Path(__file__)),
        'scripts_sha256':{p:sha(ROOT/'scripts'/p) for p in ('evaluate_pose.py','pose_scoring.py','diagnose_pose_quality.py')},'models':{},'references':{}}
    caches={}
    for role,path in paths.items():
        header,*rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==20 and header['inference']==settings and header['box_confidences']==[box_confidence]
        assert len({r['image_id'] for r in rows})==20
        caches[role]={r['image_id']:r for r in rows}
        document['models'][role]={'prediction_path':str(path.relative_to(ROOT)),'predictions_sha256':sha(path),'checkpoint_sha256':header['model_sha256'],'runtime':header['runtime']}
    assert len({json.dumps(m['runtime'], sort_keys=True) for m in document['models'].values()}) == 1
    document['common_inference_runtime_verified'] = True
    if 'reference' in caches and 'pr119_candidate' in caches:
        identical = [name for name in caches['reference']
                     if caches['reference'][name]['predictions'] == caches['pr119_candidate'][name]['predictions']]
        document['control_reproduction'] = {
            'compared': ['PR121 reference', 'PR119 candidate'],
            'exactly_equal_prediction_image_ids': identical,
            'exactly_equal_images': len(identical),
            'total_images': 20,
            'note': 'Checkpoint file hashes differ. Exact equality and indexwise numerical differences are reported separately; equal aggregates do not imply identical prediction records.'
        }
        differences={'box_px':[], 'point_px':[], 'box_confidence':[], 'point_confidence':[]}
        same_counts=all(len(caches['reference'][name]['predictions'])==len(caches['pr119_candidate'][name]['predictions']) for name in caches['reference'])
        if same_counts:
            for name in caches['reference']:
                for a,b in zip(caches['reference'][name]['predictions'],caches['pr119_candidate'][name]['predictions']):
                    differences['box_px'] += [abs(x-y) for x,y in zip(a['bbox_xyxy'],b['bbox_xyxy'])]
                    differences['box_confidence'].append(abs(a['confidence']-b['confidence']))
                    for x,y in zip(a['keypoints'],b['keypoints']):
                        differences['point_px'] += [abs(x[i]-y[i]) for i in range(2)]
                        differences['point_confidence'].append(abs(x[2]-y[2]))
        document['control_reproduction']['all_image_output_counts_equal']=same_counts
        document['control_reproduction']['indexwise_max_absolute_difference']={k:max(v,default=0) for k,v in differences.items()} if same_counts else None
    for ref,relative in REFERENCES.items():
        mp=ROOT/relative;manifest=json.loads(mp.read_text());files=[f for f in manifest['files'] if f.get('split','val')=='val']
        assert len(files)==20
        ref_report={'manifest':relative,'manifest_sha256':sha(mp),'models':{}}
        for role,cache in caches.items():
            scores=[];quality=[];misses=[];errors=[];unmatched=[];images=[];ambiguities=[];invalid_outputs=[]
            groups={str(n):{'gold':0,'matched':0,'missed':0} for n in range(3)}
            for f in files:
                name=f['image_id'];r=cache[name];label=mp.parent/f['label_path']
                assert sha(label)==f['label_sha256']
                assert r['image_sha256']==f.get('source_image_sha256',f.get('image_sha256'))
                assert r['threshold']==box_confidence
                gold=parse_pose_labels(label.read_text(),r['width'],r['height']);pred=r['predictions']
                s,q,invalid=score_probe_invalid_as_unmatched(gold,pred,r['width'],r['height'],permit_invalid);scores.append(s);quality.append(q)
                invalid_outputs.extend({'image_id':name,'prediction_id':i+1,'bbox_xyxy':pred[i]['bbox_xyxy'],'confidence':pred[i]['confidence'],'reason':'non-positive box area after clipping','accounting':'invalid other-unmatched output; retained in prediction and FP totals'} for i in invalid)
                missed_set=set(s['unmatched_gold_indices'])
                for i,g in enumerate(gold):
                    n=str(sum(k[2]==2 for k in g['keypoints']));groups[n]['gold']+=1;groups[n]['matched']+=i not in missed_set;groups[n]['missed']+=i in missed_set
                for i in s['unmatched_gold_indices']:
                    g=gold[i]
                    p=max(range(len(pred)),key=lambda p:_box_iou(pred[p]['bbox_xyxy'],g['bbox_xyxy'])) if pred else None
                    iou=_box_iou(pred[p]['bbox_xyxy'],g['bbox_xyxy']) if p is not None else 0
                    if iou == 0: p = None  # All zero-IoU ties have no geometrically nearest output.
                    misses.append({'image_id':name,'row':i+1,'gold':g,'locatable_corners':sum(k[2]==2 for k in g['keypoints']),
                        'nearest_prediction_id':p+1 if p is not None else None,'nearest_iou':iou,'nearest_confidence':pred[p]['confidence'] if p is not None else None,
                        'geometric_triage':'assignment_collision' if iou>=.5 else 'overlapping_but_below_matching_iou' if iou>=.1 else 'no_substantial_overlapping_output',
                        'geometric_triage_is_not_physical_adjudication':True})
                for pair in s['matches']:
                    gi,pi=pair['gold_index'],pair['prediction_index'];pts=[]
                    for side,(gk,pk) in enumerate(zip(gold[gi]['keypoints'],pred[pi]['keypoints'])):
                        if gk[2]!=2:continue
                        returned=_returned(pk,r['width'],r['height'])
                        e=math.hypot(gk[0]-pk[0],gk[1]-pk[1]) if returned else None
                        if e is None or e>5:pts.append({'side':['left','right'][side],'error_px':e,'returned':returned,'reference':gk,'prediction':pk})
                    if pts:errors.append({'image_id':name,'row':gi+1,'prediction_id':pi+1,'iou':pair['iou'],'points':pts})
                for pi in s['unmatched_prediction_indices']:
                    p=pred[pi]
                    unmatched.append({'image_id':name,'prediction_id':pi+1,'kind':'duplicate' if pi in s['duplicate_prediction_indices'] else 'other_unmatched','confidence':p['confidence'],**nearest_gold(p,gold)})
                possible=max_cardinality(gold,pred)
                if len(possible)>s['tp']:
                    ambiguities.append({'image_id':name,'confidence_greedy_tp':s['tp'],'maximum_cardinality_tp':len(possible),'extra_possible_matches':len(possible)-s['tp'],'alternative_matches':possible})
                images.append({'image_id':name,'gold_count':len(gold),'prediction_count':len(pred),'tp':s['tp'],'fn':s['fn'],'fp':s['fp'],'duplicate_fp':s['duplicate_fp'],'empty_scene':len(gold)==0,'matches':s['matches']})
            d=aggregate_scores(scores);d.pop('endpoint_errors_px')
            ref_report['models'][role]={'detection':d,'corner_quality':aggregate_quality(quality),'by_locatable_corners':groups,'misses':misses,'corner_error_rows':errors,'unmatched_outputs':unmatched,'invalid_outputs':invalid_outputs,'per_image':images,'matching_diagnostic':{'possible_extra_matches':sum(a['extra_possible_matches'] for a in ambiguities),'images':ambiguities}}
        document['references'][ref]=ref_report
    return document


def compact(report):
    """Keep every count/miss identity and the current candidate review catalogues."""
    for reference, ref_data in report['references'].items():
        old_errors={(x['image_id'],x['row']) for x in ref_data['models']['reference']['corner_error_rows']}
        new_errors={(x['image_id'],x['row']) for x in ref_data['models']['candidate']['corner_error_rows']}
        ref_data['paired_corner_error_identity_delta']={
            'radius_px':5, 'reference_rows':len(old_errors), 'candidate_rows':len(new_errors),
            'newly_flagged':[{'image_id':name,'row':row} for name,row in sorted(new_errors-old_errors)],
            'no_longer_flagged':[{'image_id':name,'row':row} for name,row in sorted(old_errors-new_errors)]}
        for role, model in ref_data['models'].items():
            for item in model['per_image']:
                item.pop('matches', None)
            if not (reference in ('r36', 'r37') and role == 'candidate'):
                for key in ('corner_error_rows', 'unmatched_outputs'):
                    model.pop(key)
        a = ref_data['models']['reference']; b = ref_data['models']['candidate']
        old = {(m['image_id'], m['row']) for m in a['misses']}
        new = {(m['image_id'], m['row']) for m in b['misses']}
        ref_data['paired_miss_identity_delta'] = {
            'recovered': [{'image_id': name, 'row': row} for name, row in sorted(old-new)],
            'newly_missed': [{'image_id': name, 'row': row} for name, row in sorted(new-old)]
        }
    if 'r37' in report['references']:
        report['references']['r37']['interpretation'] = (
            'Existing PR121/PR119/PR35 predictions rescored against R37 amendments; '
            'this is a reference change, not an R37 training result.'
        )
    return report

def reference_change(report):
    """Separate revised targets and deleted identities from model recoveries."""
    if 'r37' not in report['references']:
        return None
    manifests={ref:json.loads((ROOT/report['references'][ref]['manifest']).read_text()) for ref in ('r36','r37')}
    descriptor=manifests['r37']['row_maps'];rowmap_path=ROOT/descriptor['path'];assert sha(rowmap_path)==descriptor['sha256']
    rowmaps=json.loads(rowmap_path.read_text())
    descriptor=manifests['r37']['operations'];ops_path=ROOT/descriptor['path'];assert sha(ops_path)==descriptor['sha256']
    operations=[o for o in json.loads(ops_path.read_text()) if o['split']=='val']
    models=dict(report['models'])
    result={'schema_version':1,'kind':'Reference-only cached replay; no R37 model has been trained.',
            'r36_manifest_sha256':report['references']['r36']['manifest_sha256'],
            'r37_manifest_sha256':report['references']['r37']['manifest_sha256'],
            'row_maps':manifests['r37']['row_maps'],
            'excluded_instances':[{'image_id':o['image_id'],'r36_row':o['base_row'],'r37_row':None,
                                   'reason':'Explicit user-directed reference exclusion; not a confirmed duplicate or detector recovery.'}
                                  for o in operations if o['action']=='delete'],
            'models':{}}
    for role,model in models.items():
        header,*rows=[json.loads(x) for x in (ROOT/model['prediction_path']).read_text().splitlines()]
        cache={r['image_id']:r for r in rows};gold={};scores={}
        for ref,manifest in manifests.items():
            mp=ROOT/report['references'][ref]['manifest'];gold[ref]={};scores[ref]={}
            for f in manifest['files']:
                if f['split']!='val':continue
                name=f['image_id'];r=cache[name];label=mp.parent/f['label_path'];assert sha(label)==f['label_sha256']
                gold[ref][name]=parse_pose_labels(label.read_text(),r['width'],r['height'])
                scores[ref][name]=score_image(gold[ref][name],r['predictions'],r['width'],r['height'],.5,.5)
        details=[];pair_changes=[];category_changes=[];deleted_misses=[]
        for name,r in cache.items():
            mapping={row['base_row']:row['current_row'] for row in rowmaps[name]['rows']}
            inverse={current:base for base,current in mapping.items()}
            old=scores['r36'][name];new=scores['r37'][name]
            a={m['gold_index']+1:m['prediction_index']+1 for m in old['matches']}
            b={inverse[m['gold_index']+1]:m['prediction_index']+1 for m in new['matches']}
            pair_changes.extend({'image_id':name,'r36_row':row,'r37_row':mapping[row],
                                 'old_prediction_id':a.get(row),'new_prediction_id':b.get(row)}
                                for row in mapping if a.get(row)!=b.get(row))
            for row in rowmaps[name]['deleted_base_rows']:
                if row-1 in old['unmatched_gold_indices']:deleted_misses.append({'image_id':name,'r36_row':row,'r37_row':None})
            def category(score,i):
                return 'duplicate' if i in score['duplicate_prediction_indices'] else 'other_unmatched' if i in score['unmatched_prediction_indices'] else 'matched'
            for i,pred in enumerate(r['predictions']):
                before,after=category(old,i),category(new,i)
                if before!=after:
                    category_changes.append({'image_id':name,'prediction_id':i+1,'before':before,'after':after,
                        'r36_nearest':nearest_gold(pred,gold['r36'][name]),'r37_nearest':nearest_gold(pred,gold['r37'][name])})
        for op in operations:
            name,row=op['image_id'],op['base_row'];r=cache[name]
            record={'image_id':name,'r36_row':row,'r37_row':None if op['action']=='delete' else next(x['current_row'] for x in rowmaps[name]['rows'] if x['base_row']==row),'action':op['action']}
            for ref,current in [('r36',row),('r37',record['r37_row'])]:
                if current is None:record[ref]=None;continue
                g=gold[ref][name][current-1]
                match=next((m for m in scores[ref][name]['matches'] if m['gold_index']==current-1),None)
                nearest=max(range(len(r['predictions'])),key=lambda i:_box_iou(g['bbox_xyxy'],r['predictions'][i]['bbox_xyxy'])) if r['predictions'] else None
                p=r['predictions'][match['prediction_index']] if match else None
                record[ref]={'reference':g,'matched_prediction_id':match['prediction_index']+1 if match else None,
                    'matched_iou':match['iou'] if match else None,'nearest_prediction_id':nearest+1 if nearest is not None else None,
                    'nearest_iou':_box_iou(g['bbox_xyxy'],r['predictions'][nearest]['bbox_xyxy']) if nearest is not None else 0,
                    'corner_errors_px':[math.dist(gk[:2],pk[:2]) if gk[2]==2 and _returned(pk,r['width'],r['height']) else None for gk,pk in zip(g['keypoints'],p['keypoints'])] if p else None}
            details.append(record)
        result['models'][role]={'changed_reference_rows':details,'retained_row_assignment_changes':pair_changes,
            'output_category_changes':category_changes,'deleted_previously_missed_rows':deleted_misses,
            'prediction_sha256':sha(ROOT/model['prediction_path'])}
    return result


# Descriptive bins fixed before inspecting PR121 output counts. They are not
# training exclusions, calibrated robot tolerances, or optimized cutoffs.
SIZE_BINS = {
    'short_side_px': [(0,16,'<16'),(16,32,'16-<32'),(32,64,'32-<64'),(64,None,'>=64')],
    'height_px': [(0,16,'<16'),(16,32,'16-<32'),(32,64,'32-<64'),(64,None,'>=64')],
    'area_px2': [(0,1024,'<1024'),(1024,4096,'1024-<4096'),(4096,16384,'4096-<16384'),(16384,None,'>=16384')],
    'locatable_corners': [(0,1,'0'),(1,2,'1'),(2,3,'2')],
}


def dimensions(obj, reference=True):
    box=obj['bbox_xyxy'];w=box[2]-box[0];h=box[3]-box[1]
    result={'width_px':w,'height_px':h,'short_side_px':min(w,h),'area_px2':w*h}
    if reference:result['locatable_corners']=sum(p[2]==2 for p in obj['keypoints'])
    return result


def bin_name(value, dimension):
    return next(name for low,high,name in SIZE_BINS[dimension] if value>=low and (high is None or value<high))


def metric_counts(score):
    return {k:score[k] for k in ('gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp')}


def load_reference_rows(report, ref, cache):
    path=ROOT/report['references'][ref]['manifest'];assert sha(path)==report['references'][ref]['manifest_sha256']
    result={}
    for f in json.loads(path.read_text())['files']:
        if f.get('split','val')!='val':continue
        name=f['image_id'];label=path.parent/f['label_path'];assert sha(label)==f['label_sha256']
        assert f.get('source_image_sha256',f.get('image_sha256'))==cache[name]['image_sha256']
        result[name]=parse_pose_labels(label.read_text(),cache[name]['width'],cache[name]['height'])
    return result


def label_overlap_audit(gold):
    pairs=[]
    for name,rows in gold.items():
        for i,a in enumerate(rows):
            for j in range(i+1,len(rows)):
                b=rows[j]
                if a['class_id']!=b['class_id']:continue
                aa,bb=a['bbox_xyxy'],b['bbox_xyxy'];iou=_box_iou(aa,bb)
                inter=max(0,min(aa[2],bb[2])-max(aa[0],bb[0]))*max(0,min(aa[3],bb[3])-max(aa[1],bb[1]))
                containment=inter/min(dimensions(a)['area_px2'],dimensions(b)['area_px2'])
                if iou<.5 and containment<.9:continue
                pairs.append({'image_id':name,'rows':[i+1,j+1],'iou':iou,'intersection_over_smaller_box':containment,
                    'equal_box_within_1e_5_px':max(abs(x-y) for x,y in zip(aa,bb))<=1e-5,
                    'reference_boxes':[aa,bb],
                    'locatable_corners':[dimensions(a)['locatable_corners'],dimensions(b)['locatable_corners']],
                    'shared_available_point_distances_px':[math.dist(x[:2],y[:2]) if x[2]==y[2]==2 else None for x,y in zip(a['keypoints'],b['keypoints'])]})
    pairs.sort(key=lambda p:(-p['iou'],-p['intersection_over_smaller_box'],p['image_id'],p['rows']))
    return {'screening_rule':'same-class reference-box IoU >=0.50 OR intersection/smaller-area >=0.90; geometry alone does not establish two labels for one physical cassette',
            'iou_ge_0_5_pairs':sum(p['iou']>=.5 for p in pairs),
            'containment_ge_0_9_pairs':sum(p['intersection_over_smaller_box']>=.9 for p in pairs),
            'equal_box_pairs':sum(p['equal_box_within_1e_5_px'] for p in pairs),'pairs':pairs}


def size_visibility_audit(report):
    """Describe fixed outputs and a hypothetical target deletion, without edits."""
    caches={}
    for role,m in report['models'].items():
        path=ROOT/m['prediction_path'];assert sha(path)==m['predictions_sha256']
        _,*rows=[json.loads(line) for line in path.read_text().splitlines()]
        caches[role]={r['image_id']:r for r in rows}
    result={'schema_version':1,'scope':'Descriptive association on repeatedly reviewed development images. Size, corner availability, perspective and occlusion are confounded. No causal claim, annotation deletion, retraining, or threshold selection.',
        'coordinate_space':'original 1280x720 image pixels; width=xmax-xmin, height=ymax-ymin; short side=min(width,height); area=width*height',
        'bins':{k:[{'lower_inclusive':low,'upper_exclusive':high,'name':name} for low,high,name in bins] for k,bins in SIZE_BINS.items()},
        'association_rules':{'matched':'assigned reference from the unchanged confidence-ordered IoU >=0.50 scorer',
            'duplicate':'highest-IoU already-matched same-class reference with IoU >=0.50; a scoring duplicate prediction, not a duplicate annotation',
            'other_unmatched':'highest positive IoU reference, if any; geometric proximity is not proof of object identity',
            'no_overlap':'no positive-IoU reference; excluded from reference-size bins and counted explicitly'},
        'deletion_thought_experiment':{'threshold_original_area_px2':1024,'rule':'Delete only reference rows with original area <1024 px2 in memory, hold all predictions fixed, and rerun the unchanged scorer. This changes the target population and cannot establish model improvement. Exactly 1024 is retained.'},'references':{}}
    for ref in report['references']:
        gold=load_reference_rows(report,ref,caches['candidate']);entry={'reference_manifest_sha256':report['references'][ref]['manifest_sha256'],
            'label_overlap':label_overlap_audit(gold),'models':{}}
        for role,cache in caches.items():
            gt_records=[];outputs=[];before_scores=[];after_scores=[];removed=[];retained_assignment_changes=[];category_changes=[]
            for name,rows in gold.items():
                r=cache[name];pred=r['predictions'];score=score_image(rows,pred,r['width'],r['height'],.5,.5);before_scores.append(score)
                matched={m['gold_index']:m['prediction_index'] for m in score['matches']};assigned={p:g for g,p in matched.items()};dups=set(score['duplicate_prediction_indices'])
                current=[]
                for i,g in enumerate(rows):
                    current.append({'image_id':name,'row':i+1,**dimensions(g),'matched':i in matched,
                        'prediction_id':matched[i]+1 if i in matched else None,'duplicate_prediction_ids':[],'nearest_other_prediction_ids':[]})
                for pi,p in enumerate(pred):
                    kind='matched' if pi in assigned else 'duplicate' if pi in dups else 'other_unmatched'
                    if kind=='matched':gi=assigned[pi]
                    elif kind=='duplicate':
                        eligible=[g for g in matched if rows[g]['class_id']==p['class_id'] and _box_iou(p['bbox_xyxy'],rows[g]['bbox_xyxy'])>=.5]
                        gi=min(eligible,key=lambda i:(-_box_iou(p['bbox_xyxy'],rows[i]['bbox_xyxy']),i));current[gi]['duplicate_prediction_ids'].append(pi+1)
                    else:
                        nearest=nearest_gold(p,rows);gi=nearest['gold_row']-1 if nearest['gold_row'] else None
                        if gi is not None:current[gi]['nearest_other_prediction_ids'].append(pi+1)
                    iou=_box_iou(p['bbox_xyxy'],rows[gi]['bbox_xyxy']) if gi is not None else 0
                    outputs.append({'image_id':name,'prediction_id':pi+1,'kind':kind,'confidence':p['confidence'],
                        'prediction_dimensions':dimensions(p,False),'associated_row':gi+1 if gi is not None else None,
                        'reference_dimensions':dimensions(rows[gi]) if gi is not None else None,'association_iou':iou,
                        'overlap_group':'ge_0.5' if iou>=.5 else '0.1_to_lt_0.5' if iou>=.1 else 'positive_lt_0.1' if iou>0 else 'none'})
                gt_records.extend(current)
                retained=[i for i,g in enumerate(rows) if dimensions(g)['area_px2']>=1024]
                after=score_image([rows[i] for i in retained],pred,r['width'],r['height'],.5,.5);after_scores.append(after)
                newmatched={retained[m['gold_index']]:m['prediction_index'] for m in after['matches']}
                for i,g in enumerate(rows):
                    if i not in retained:removed.append({'image_id':name,'row':i+1,'area_px2':dimensions(g)['area_px2'],
                        'locatable_corners':dimensions(g)['locatable_corners'],'was_matched':i in matched,
                        'prediction_id':matched[i]+1 if i in matched else None})
                    elif matched.get(i)!=newmatched.get(i):retained_assignment_changes.append({'image_id':name,'row':i+1,'before_prediction_id':matched[i]+1 if i in matched else None,'after_prediction_id':newmatched[i]+1 if i in newmatched else None})
                def category(s,p):return 'duplicate' if p in s['duplicate_prediction_indices'] else 'other_unmatched' if p in s['unmatched_prediction_indices'] else 'matched'
                for pi in range(len(pred)):
                    a,b=category(score,pi),category(after,pi)
                    if a!=b:category_changes.append({'image_id':name,'prediction_id':pi+1,'before':a,'after':b})
            tables={};own_prediction_tables={}
            for dim,bins in SIZE_BINS.items():
                table=[]
                for _,_,name in bins:
                    group=[g for g in gt_records if bin_name(g[dim],dim)==name]
                    table.append({'bin':name,'gold':len(group),'tp':sum(g['matched'] for g in group),'fn':sum(not g['matched'] for g in group),
                        'fn_rate':sum(not g['matched'] for g in group)/len(group) if group else None,
                        'duplicate_outputs':sum(len(g['duplicate_prediction_ids']) for g in group),
                        'matched_gt_with_duplicate':sum(bool(g['duplicate_prediction_ids']) for g in group),
                        'nearest_other_unmatched_outputs':sum(len(g['nearest_other_prediction_ids']) for g in group),
                        'locatable_corner_counts':dict(sorted(Counter(str(g['locatable_corners']) for g in group).items()))})
                tables[dim]=table
                if dim!='locatable_corners':own_prediction_tables[dim]=[{'bin':name,**dict(Counter(o['kind'] for o in outputs if bin_name(o['prediction_dimensions'][dim],dim)==name))} for _,_,name in bins]
            cross=[]
            for _,_,area_bin in SIZE_BINS['area_px2']:
                for visible in range(3):
                    group=[g for g in gt_records if bin_name(g['area_px2'],'area_px2')==area_bin and g['locatable_corners']==visible]
                    cross.append({'area_bin':area_bin,'locatable_corners':visible,'gold':len(group),'fn':sum(not g['matched'] for g in group),
                        'duplicate_outputs':sum(len(g['duplicate_prediction_ids']) for g in group),
                        'matched_gt_with_duplicate':sum(bool(g['duplicate_prediction_ids']) for g in group)})
            before=metric_counts(aggregate_scores(before_scores));after=metric_counts(aggregate_scores(after_scores))
            assert before==metric_counts(report['references'][ref]['models'][role]['detection'])
            thought={'before':before,'after':after,'delta':{k:after[k]-before[k] for k in before},
                'removed_rows':len(removed),'removed_previously_matched':sum(x['was_matched'] for x in removed),
                'removed_previously_missed':sum(not x['was_matched'] for x in removed),
                'removed_eligible_two_corner_pairs':sum(x['locatable_corners']==2 for x in removed),
                'retained_row_assignment_changes':retained_assignment_changes,'prediction_category_changes':category_changes,
                'removed_row_inventory':removed}
            summary={'by_reference_dimensions':tables,'by_prediction_dimensions':own_prediction_tables,
                'area_by_corner_availability':cross,'output_overlap_counts':dict(sorted(Counter(o['kind']+' / '+o['overlap_group'] for o in outputs).items())),
                'unmatched_without_reference_overlap':sum(o['kind']!='matched' and o['associated_row'] is None for o in outputs),
                'area_lt1024_reference_deletion_thought_experiment':thought}
            if ref in ('r36','r37') and role=='candidate':summary.update({'reference_rows':gt_records,'prediction_associations':outputs})
            entry['models'][role]=summary
        result['references'][ref]=entry
    return result


def head_probe_comparison(report):
    """Score the single authorized head probe separately from primary inference."""
    path=ROOT/SOURCE/'predictions/candidate-one2one.jsonl'
    if not path.is_file():return None
    header,*rows=[json.loads(line) for line in path.read_text().splitlines()]
    settings={**INFERENCE,'nms':False}
    assert header['inference']==settings and header['box_confidences']==[.13]
    assert header['model_sha256']==report['models']['candidate']['checkpoint_sha256']
    assert header['runtime']['backend_end2end'] is True and header['nms_iou_applies'] is False
    assert header['endpoint_confidence']==.5
    base_runtime=dict(report['models']['candidate']['runtime']);probe_runtime=dict(header['runtime'])
    assert base_runtime.pop('backend_end2end') is False and probe_runtime.pop('backend_end2end') is True
    assert base_runtime==probe_runtime
    audit_path=ROOT/header['head_audit_path'];assert audit_path.is_file()
    full=audit({'candidate_one2one':path},inference_settings=settings,permit_invalid=True)
    baseline_path=ROOT/report['models']['candidate']['prediction_path'];_,*base_rows=[json.loads(line) for line in baseline_path.read_text().splitlines()]
    baseline_cache={r['image_id']:r for r in base_rows};probe_cache={r['image_id']:r for r in rows}
    result={'schema_version':1,'scope':'One bounded same-checkpoint inference-head probe; no retraining, confidence/NMS sweep, baseline change, or automatic promotion. The repeated development reference does not establish performance on new sessions.',
        'primary':'trained one-to-many branch followed by NMS at IoU 0.70',
        'probe':'trained one-to-one branch, NMS-free; retained iou argument 0.70 is ignored on this path',
        'fixed':{'checkpoint_sha256':header['model_sha256'],'box_confidence':.13,'endpoint_confidence':.5,'matching_iou':.5,'radii_px':[5,10]},
        'probe_prediction_path':str(path.relative_to(ROOT)),'probe_predictions_sha256':sha(path),
        'model_audit_path':str(audit_path.relative_to(ROOT)),'model_audit_sha256':sha(audit_path),
        'probe_inference':header['inference'],'probe_runtime':header['runtime'],
        'common_runtime_except_head_verified':True,'timing_claim':'No latency comparison; CPU jobs may overlap.',
        'invalid_output_policy':'Strict production scorer rejects non-positive-area boxes. This probe retains each such output as an invalid other-unmatched FP, scores only positive-area records, and maps all IDs back to the untouched original cache. No output is silently dropped.',
        'references':{}}
    for ref,data in full['references'].items():
        model=data['models']['candidate_one2one'];base=report['references'][ref]['models']['candidate']
        old_misses={(x['image_id'],x['row']) for x in base['misses']};new_misses={(x['image_id'],x['row']) for x in model['misses']}
        gold=load_reference_rows(report,ref,baseline_cache);base_errors=set()
        for name,targets in gold.items():
            r=baseline_cache[name];pred=r['predictions'];score=score_image(targets,pred,r['width'],r['height'],.5,.5)
            for m in score['matches']:
                gi,pi=m['gold_index'],m['prediction_index']
                if any(gk[2]==2 and (not _returned(pk,r['width'],r['height']) or math.dist(gk[:2],pk[:2])>5) for gk,pk in zip(targets[gi]['keypoints'],pred[pi]['keypoints'])):base_errors.add((name,gi+1))
        new_errors={(x['image_id'],x['row']) for x in model['corner_error_rows']}
        result['references'][ref]={'manifest':data['manifest'],'manifest_sha256':data['manifest_sha256'],
            'one_to_many':{'detection':base['detection'],'corner_quality':base['corner_quality'],'by_locatable_corners':base['by_locatable_corners'],'corner_error_rows_count':len(base_errors)},
            'one_to_one':model,
            'paired_miss_identity_delta':{'recovered':[{'image_id':name,'row':row} for name,row in sorted(old_misses-new_misses)],'newly_missed':[{'image_id':name,'row':row} for name,row in sorted(new_misses-old_misses)]},
            'paired_corner_error_identity_delta':{'reference_rows':len(base_errors),'candidate_rows':len(new_errors),
                'newly_flagged':[{'image_id':name,'row':row} for name,row in sorted(new_errors-base_errors)],
                'no_longer_flagged':[{'image_id':name,'row':row} for name,row in sorted(base_errors-new_errors)]}}
    return result


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--next-reference', default='data/reviews/fronts-2026-11-26-r37/manifest.json')
    args = parser.parse_args()
    ROOT = args.root.resolve()
    paths = {role: ROOT / SOURCE / 'predictions' / f'{role}.jsonl'
             for role in ('reference', 'candidate')}
    paths.update({role: ROOT / PREVIOUS / 'predictions' / f'{oldrole}.jsonl'
                  for role, oldrole in (('pr119_reference', 'reference'), ('pr119_candidate', 'candidate'))})
    paths['pr35'] = ROOT / SELECTED / 'predictions' / 'pr35.jsonl'
    if (ROOT / args.next_reference).is_file():
        REFERENCES['r37'] = args.next_reference
    report = compact(audit(paths))
    destination = args.output or ROOT / SOURCE / 'comparison.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    delta = reference_change(report)
    if delta is not None:
        (destination.parent / 'r37-reference-delta.json').write_text(json.dumps(delta, indent=2) + '\n')
    size=size_visibility_audit(report);size['source_comparison_sha256']=sha(destination)
    (destination.parent/'size-visibility-audit.json').write_text(json.dumps(size,indent=2)+'\n')
    probe=head_probe_comparison(report)
    if probe is not None:
        probe['source_comparison_sha256']=sha(destination)
        (destination.parent/'head-probe-comparison.json').write_text(json.dumps(probe,indent=2)+'\n')
    print('Wrote', destination)
    for ref, data in report['references'].items():
        for role, m in data['models'].items():
            d = m['detection']; q = m['corner_quality']
            print(ref, role, {k:d[k] for k in ('gold_count', 'tp', 'fn', 'fp', 'duplicate_fp')},
                  'pairs', [x['both_visible_objects_within_radius'] for x in q['radii']],
                  'denominator', q['both_visible_objects'],
                  'maxcard_delta', m['matching_diagnostic']['possible_extra_matches'])


if __name__ == '__main__':
    main()
