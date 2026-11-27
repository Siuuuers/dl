#!/usr/bin/env python3
"""Fixed 1.5x crop feasibility check on saved original and weighted outputs."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

REFS=('frozen666','historical663','r38','r50')

def load(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def usable(pred):
    return all(all(isinstance(x,(int,float)) and math.isfinite(x) for x in point)
        and 0<=point[0]<1280 and 0<=point[1]<720 and point[2]>=.5 for point in pred['keypoints'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--scorer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compress-rows',action='store_true',help='Write full rows as deterministic gzip JSON.')
    args=parser.parse_args()
    assert sha(args.inputs)=='b6059a3810e29ba3afb245f1585704521c2633d8caca6fddd908be015e249125'
    assert sha(args.scorer)=='04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018'
    data=load(args.inputs)
    spec=importlib.util.spec_from_file_location('scorer',args.scorer)
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    arms={'original_control':{im:rows['postclip'] for im,rows in load(args.control)['control'].items()},
          'weighted_candidate':data['predictions']['candidate']}
    results={}; case_rows={}
    for ref in REFS:
        results[ref]={};case_rows[ref]={}
        for arm,images in arms.items():
            c=Counter({key:0 for key in ('all_reference_fronts','all_two_corner_reference_fronts','partial_reference_fronts',
                'complete_reference_missed_box','complete_reference_matched','complete_reference_matched_pair_unavailable',
                'complete_reference_pair_within5','complete_reference_pair_over5','all_matched_crops_both_corners_inside',
                'all_matched_crops_one_corner_outside','all_matched_crops_both_corners_outside','all_matched_crops_clipped_at_image_edge',
                'over5_crops_both_corners_inside','over5_crops_one_corner_outside','over5_crops_both_corners_outside','over5_crops_clipped_at_image_edge',
                'all_matched_reference_corners_outside_image')})
            rows=[]; scores=[]
            for im,targets in data['references'][ref]['gold'].items():
                score=scorer.score_image(targets,images[im],1280,720,.5,.5);scores.append(score)
                if arm=='weighted_candidate':assert score==data['expected_baseline'][ref]['candidate']['per_image'][im]
                matches={m['gold_index']:m for m in score['matches']}
                for gi,g in enumerate(targets):
                    c['all_reference_fronts']+=1
                    if not all(point[2]==2 for point in g['keypoints']):
                        c['partial_reference_fronts']+=1;continue
                    c['all_two_corner_reference_fronts']+=1
                    if gi not in matches:c['complete_reference_missed_box']+=1;continue
                    c['complete_reference_matched']+=1
                    pred=images[im][matches[gi]['prediction_index']]
                    x0,y0,x1,y1=pred['bbox_xyxy'];w=x1-x0;h=y1-y0
                    assert w>0 and h>0
                    # Factor 1.5 was fixed before this feasibility calculation.
                    # Integer slicing bounds expand outwards, then clip to the frame.
                    unbounded=[math.floor(x0-.25*w),math.floor(y0-.25*h),math.ceil(x1+.25*w),math.ceil(y1+.25*h)]
                    crop=[max(0,unbounded[0]),max(0,unbounded[1]),min(1280,unbounded[2]),min(720,unbounded[3])]
                    clipped=crop!=unbounded
                    inside=[crop[0]<=point[0]<crop[2] and crop[1]<=point[1]<crop[3] for point in g['keypoints']]
                    state={2:'both_corners_inside',1:'one_corner_outside',0:'both_corners_outside'}[sum(inside)]
                    c['all_matched_crops_'+state]+=1
                    c['all_matched_crops_clipped_at_image_edge']+=clipped
                    c['all_matched_reference_corners_outside_image']+=sum(not(0<=p[0]<1280 and 0<=p[1]<720) for p in g['keypoints'])
                    row={'image':im,'reference_index':gi,'prediction_index':matches[gi]['prediction_index'],
                        'crop_xyxy':crop,'unclipped_crop_xyxy':unbounded,'clipped_at_image_edge':clipped,
                        'reference_corners_inside':inside,'containment':state}
                    if usable(pred):
                        error=max(math.hypot(p[0]-t[0],p[1]-t[1]) for p,t in zip(pred['keypoints'],g['keypoints']))
                        row['maximum_corner_error_px']=error
                        c['complete_reference_pair_within5' if error<=5 else 'complete_reference_pair_over5']+=1
                        if error>5:
                            c['over5_crops_'+state]+=1
                            c['over5_crops_clipped_at_image_edge']+=clipped
                    else:
                        c['complete_reference_matched_pair_unavailable']+=1
                    rows.append(row)
            assert c['all_two_corner_reference_fronts']+c['partial_reference_fronts']==c['all_reference_fronts']
            assert c['complete_reference_missed_box']+c['complete_reference_matched']==c['all_two_corner_reference_fronts']
            assert sum(c['all_matched_crops_'+v] for v in ('both_corners_inside','one_corner_outside','both_corners_outside'))==c['complete_reference_matched']
            assert sum(c['over5_crops_'+v] for v in ('both_corners_inside','one_corner_outside','both_corners_outside'))==c['complete_reference_pair_over5']
            assert sum(c[v] for v in ('complete_reference_pair_within5','complete_reference_pair_over5','complete_reference_matched_pair_unavailable'))==c['complete_reference_matched']
            aggregate=scorer.aggregate_scores(scores)
            results[ref][arm]={'counts':dict(c),'detection':{k:aggregate[k] for k in ('gold_count','prediction_count','tp','fn','fp','duplicate_fp')},
                'over5_by_image':dict(Counter(r['image'] for r in rows if r.get('maximum_corner_error_px',0)>5))}
            case_rows[ref][arm]=rows
    result={'schema':'fixed-crop-feasibility-v1','factor':1.5,'image_size_px':[1280,720],
        'description':'Containment of annotated lower corners inside a 1.5x expanded predicted-box crop; no refinement model or new predictions.',
        'fixed_parameters':{'box_confidence_strict':.13,'endpoint_confidence_inclusive':.5,'match_iou_inclusive':.5,'pixel_bounds':'floor minima, ceil maxima, clip to [0,1280] x [0,720]; half-open containment'},
        'source_hashes':{k:sha(v) for k,v in {'inputs':args.inputs,'control':args.control,'scorer':args.scorer,'script':Path(__file__)}.items()},
        'results':results}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'crop-containment-summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    payload=(json.dumps(case_rows,indent=2,sort_keys=True)+'\n').encode()
    if args.compress_rows:
        (args.output/'crop-containment-rows.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    else:
        (args.output/'crop-containment-rows.json').write_bytes(payload)
    for ref in ('frozen666','r50'):
        for arm,entry in results[ref].items():print(ref,arm,json.dumps(entry,sort_keys=True))
if __name__=='__main__':main()
