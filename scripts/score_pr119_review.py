#!/usr/bin/env python3
"""Replay preserved PR119/PR117/PR35 prediction caches against separate references.

This does not run inference, train a model, tune thresholds, or alter official
matching. The maximum-cardinality pass is a diagnostic of the same IoU graph.
"""
import argparse
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'results/evaluation/pr119-r35-20261126'
PREVIOUS = 'results/evaluation/pr117-r34-20261125'
SELECTED = 'results/evaluation/pr114-r33-20261125'
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_pose import parse_pose_labels, INFERENCE
from pose_scoring import score_image, aggregate_scores, _box_iou
from diagnose_pose_quality import quality_for_image, aggregate_quality, _returned

REFERENCES = {
    'frozen666': 'data/references/sequence-refined-reference-2026-11-18/manifest.json',
    'historical663': 'data/references/visible-front-completion-2026-11-14/manifest.json',
    'r34': 'data/reviews/fronts-2026-11-25-r34/manifest.json',
    'r35': 'data/reviews/fronts-2026-11-25-r35/manifest.json',
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

def audit(paths, box_confidence=.13):
    document = {'schema_version':1,'inference_settings':INFERENCE,'box_confidence':box_confidence,'matching_iou':.5,'endpoint_confidence':.5,'radii_px':[5,10],
        'scope':'PR119 fixed inference and separate reference replays; current-review labels are diagnostic, not an independent holdout. Additional graph matching is diagnostic only and does not change reported scores.',
        'source_revision':'bfe57e5ceb95ffa1d94ee56880010052ed84c86d','scripts_sha256':{p:sha(ROOT/'scripts'/p) for p in ('evaluate_pose.py','pose_scoring.py','diagnose_pose_quality.py')},'models':{},'references':{}}
    caches={}
    for role,path in paths.items():
        header,*rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==20 and header['inference']==INFERENCE and header['box_confidences']==[box_confidence]
        assert len({r['image_id'] for r in rows})==20
        caches[role]={r['image_id']:r for r in rows}
        document['models'][role]={'prediction_path':str(path.relative_to(ROOT)),'predictions_sha256':sha(path),'checkpoint_sha256':header['model_sha256'],'runtime':header['runtime']}
    assert len({json.dumps(m['runtime'], sort_keys=True) for m in document['models'].values()}) == 1
    document['common_inference_runtime_verified'] = True
    if 'reference' in caches and 'pr117_candidate' in caches:
        identical = [name for name in caches['reference']
                     if caches['reference'][name]['predictions'] == caches['pr117_candidate'][name]['predictions']]
        document['control_reproduction'] = {
            'compared': ['PR119 reference', 'PR117 candidate'],
            'exactly_equal_prediction_image_ids': identical,
            'exactly_equal_images': len(identical),
            'total_images': 20,
            'note': 'Checkpoint file hashes differ. Exact equality and indexwise numerical differences are reported separately; equal aggregates do not imply identical prediction records.'
        }
        differences={'box_px':[], 'point_px':[], 'box_confidence':[], 'point_confidence':[]}
        same_counts=all(len(caches['reference'][name]['predictions'])==len(caches['pr117_candidate'][name]['predictions']) for name in caches['reference'])
        if same_counts:
            for name in caches['reference']:
                for a,b in zip(caches['reference'][name]['predictions'],caches['pr117_candidate'][name]['predictions']):
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
            scores=[];quality=[];misses=[];errors=[];unmatched=[];images=[];ambiguities=[]
            groups={str(n):{'gold':0,'matched':0,'missed':0} for n in range(3)}
            for f in files:
                name=f['image_id'];r=cache[name];label=mp.parent/f['label_path']
                assert sha(label)==f['label_sha256']
                assert r['image_sha256']==f.get('source_image_sha256',f.get('image_sha256'))
                assert r['threshold']==box_confidence
                gold=parse_pose_labels(label.read_text(),r['width'],r['height']);pred=r['predictions']
                s=score_image(gold,pred,r['width'],r['height'],.5,.5);scores.append(s)
                quality.append(quality_for_image(gold,pred,r['width'],r['height'],[5,10]))
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
            ref_report['models'][role]={'detection':d,'corner_quality':aggregate_quality(quality),'by_locatable_corners':groups,'misses':misses,'corner_error_rows':errors,'unmatched_outputs':unmatched,'per_image':images,'matching_diagnostic':{'possible_extra_matches':sum(a['extra_possible_matches'] for a in ambiguities),'images':ambiguities}}
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
            if not (reference in ('r35', 'r36') and role == 'candidate'):
                for key in ('corner_error_rows', 'unmatched_outputs'):
                    model.pop(key)
        a = ref_data['models']['reference']; b = ref_data['models']['candidate']
        old = {(m['image_id'], m['row']) for m in a['misses']}
        new = {(m['image_id'], m['row']) for m in b['misses']}
        ref_data['paired_miss_identity_delta'] = {
            'recovered': [{'image_id': name, 'row': row} for name, row in sorted(old-new)],
            'newly_missed': [{'image_id': name, 'row': row} for name, row in sorted(new-old)]
        }
    if 'r36' in report['references']:
        report['references']['r36']['interpretation'] = (
            'Existing PR119/PR117/PR35 predictions rescored against R36 amendments; '
            'this is a reference change, not an R36 training result.'
        )
    return report

def reference_change(report):
    """Separate revised targets and deleted identities from model recoveries."""
    if 'r36' not in report['references']:
        return None
    manifests={ref:json.loads((ROOT/report['references'][ref]['manifest']).read_text()) for ref in ('r35','r36')}
    descriptor=manifests['r36']['row_maps'];rowmap_path=ROOT/descriptor['path'];assert sha(rowmap_path)==descriptor['sha256']
    rowmaps=json.loads(rowmap_path.read_text())
    descriptor=manifests['r36']['operations'];ops_path=ROOT/descriptor['path'];assert sha(ops_path)==descriptor['sha256']
    operations=[o for o in json.loads(ops_path.read_text()) if o['split']=='val']
    models=dict(report['models'])
    result={'schema_version':1,'kind':'Reference-only cached replay; no R36 model has been trained.',
            'r35_manifest_sha256':report['references']['r35']['manifest_sha256'],
            'r36_manifest_sha256':report['references']['r36']['manifest_sha256'],
            'row_maps':manifests['r36']['row_maps'],
            'excluded_instances':[{'image_id':o['image_id'],'r35_row':o['base_row'],'r36_row':None,
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
            old=scores['r35'][name];new=scores['r36'][name]
            a={m['gold_index']+1:m['prediction_index']+1 for m in old['matches']}
            b={inverse[m['gold_index']+1]:m['prediction_index']+1 for m in new['matches']}
            pair_changes.extend({'image_id':name,'r35_row':row,'r36_row':mapping[row],
                                 'old_prediction_id':a.get(row),'new_prediction_id':b.get(row)}
                                for row in mapping if a.get(row)!=b.get(row))
            for row in rowmaps[name]['deleted_base_rows']:
                if row-1 in old['unmatched_gold_indices']:deleted_misses.append({'image_id':name,'r35_row':row,'r36_row':None})
            def category(score,i):
                return 'duplicate' if i in score['duplicate_prediction_indices'] else 'other_unmatched' if i in score['unmatched_prediction_indices'] else 'matched'
            for i,pred in enumerate(r['predictions']):
                before,after=category(old,i),category(new,i)
                if before!=after:
                    category_changes.append({'image_id':name,'prediction_id':i+1,'before':before,'after':after,
                        'r35_nearest':nearest_gold(pred,gold['r35'][name]),'r36_nearest':nearest_gold(pred,gold['r36'][name])})
        for op in operations:
            name,row=op['image_id'],op['base_row'];r=cache[name]
            record={'image_id':name,'r35_row':row,'r36_row':None if op['action']=='delete' else next(x['current_row'] for x in rowmaps[name]['rows'] if x['base_row']==row),'action':op['action']}
            for ref,current in [('r35',row),('r36',record['r36_row'])]:
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


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--next-reference', default='data/reviews/fronts-2026-11-26-r36/manifest.json')
    args = parser.parse_args()
    ROOT = args.root.resolve()
    paths = {role: ROOT / SOURCE / 'predictions' / f'{role}.jsonl'
             for role in ('reference', 'candidate')}
    paths.update({role: ROOT / PREVIOUS / 'predictions' / f'{oldrole}.jsonl'
                  for role, oldrole in (('pr117_reference', 'reference'), ('pr117_candidate', 'candidate'))})
    paths['pr35'] = ROOT / SELECTED / 'predictions' / 'pr35.jsonl'
    if (ROOT / args.next_reference).is_file():
        REFERENCES['r36'] = args.next_reference
    report = compact(audit(paths))
    destination = args.output or ROOT / SOURCE / 'comparison.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    delta = reference_change(report)
    if delta is not None:
        (destination.parent / 'r36-reference-delta.json').write_text(json.dumps(delta, indent=2) + '\n')
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
