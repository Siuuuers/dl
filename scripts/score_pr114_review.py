#!/usr/bin/env python3
"""Replay preserved PR114/PR35 prediction caches against separate references.

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
SOURCE = 'results/evaluation/pr114-r33-20261125'
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_pose import parse_pose_labels, INFERENCE
from pose_scoring import score_image, aggregate_scores, _box_iou
from diagnose_pose_quality import quality_for_image, aggregate_quality, _returned

REFERENCES = {
    'frozen666': 'data/references/sequence-refined-reference-2026-11-18/manifest.json',
    'historical663': 'data/references/visible-front-completion-2026-11-14/manifest.json',
    'r32': 'data/reviews/fronts-2026-11-25-r32/manifest.json',
    'r33': 'data/reviews/fronts-2026-11-25-r33/manifest.json',
    'r34': 'data/reviews/fronts-2026-11-25-r34/manifest.json',
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
    return {'gold_row':i+1,'iou':_box_iou(p['bbox_xyxy'],gold[i]['bbox_xyxy'])}

def audit(paths):
    document = {'schema_version':1,'inference_settings':INFERENCE,'box_confidence':.13,'matching_iou':.5,'endpoint_confidence':.5,'radii_px':[5,10],
        'scope':'PR114 fixed inference and separate reference replays; current-review labels are diagnostic, not an independent holdout. Additional graph matching is diagnostic only and does not change reported scores.',
        'source_revision':'d4631e07af44efc3c35d84f57f55c10c6ff06a65','scripts_sha256':{p:sha(ROOT/'scripts'/p) for p in ('evaluate_pose.py','pose_scoring.py','diagnose_pose_quality.py')},'models':{},'references':{}}
    caches={}
    for role,path in paths.items():
        header,*rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==20 and header['inference']==INFERENCE and header['box_confidences']==[.13]
        assert len({r['image_id'] for r in rows})==20
        caches[role]={r['image_id']:r for r in rows}
        document['models'][role]={'prediction_path':str(path.relative_to(ROOT)),'predictions_sha256':sha(path),'checkpoint_sha256':header['model_sha256'],'runtime':header['runtime']}
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
                assert r['threshold']==.13
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
    """Keep all counts and the current candidate's complete review catalogues."""
    for reference, ref_data in report['references'].items():
        for role, model in ref_data['models'].items():
            for item in model['per_image']:
                item.pop('matches', None)
            if not (reference in ('r33', 'r34') and role == 'candidate'):
                for key in ('misses', 'corner_error_rows', 'unmatched_outputs'):
                    model.pop(key)
    report['references']['r34']['interpretation'] = (
        'Existing PR114/PR35 predictions rescored against R34 amendments; '
        'this is a reference change, not an R34 training result. '
        'R33 PDF remains tied to the preserved r33 reference.'
    )
    return report


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    paths = {role: ROOT / SOURCE / 'predictions' / f'{role}.jsonl'
             for role in ('reference', 'candidate', 'pr35')}
    report = compact(audit(paths))
    destination = args.output or ROOT / SOURCE / 'comparison.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
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
