#!/usr/bin/env python3
"""Explore fixed quality-ranking signals on saved cassette pose outputs."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

INPUT_HASH='b6059a3810e29ba3afb245f1585704521c2633d8caca6fddd908be015e249125'
SCORER_HASH='04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018'
REFS=('frozen666','historical663','r38','r50')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,data):path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
def ranks(values):
    ordered=sorted(range(len(values)),key=values.__getitem__)
    result=[0.0]*len(values)
    start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[ordered[end]]==values[ordered[start]]:end+=1
        rank=(start+1+end)/2
        for i in ordered[start:end]:result[i]=rank
        start=end
    return result

def spearman(xs,ys):
    if len(xs)<2:return None
    a,b=ranks(xs),ranks(ys)
    ma,mb=statistics.mean(a),statistics.mean(b)
    cov=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    denom=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return cov/denom if denom else None

def auc(scores,labels):
    pos=[v for v,l in zip(scores,labels) if l]
    neg=[v for v,l in zip(scores,labels) if not l]
    if not pos or not neg:return None
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))

def usable(pred):
    return all(all(isinstance(x,(int,float)) and math.isfinite(x) for x in p)
        and 0<=p[0]<1280 and 0<=p[1]<720 and p[2]>=.5 for p in pred['keypoints'])

def mutual_match(first,second,iou):
    if not first or not second:return {}
    matrix=[[iou(a['bbox_xyxy'],b['bbox_xyxy']) if a['class_id']==b['class_id'] else 0 for b in second] for a in first]
    forward=[max(range(len(second)),key=lambda j:(row[j],-j)) for row in matrix]
    backward=[max(range(len(first)),key=lambda i:(matrix[i][j],-i)) for j in range(len(second))]
    return {i:j for i,j in enumerate(forward) if backward[j]==i and matrix[i][j]>=.5}

def ranking(rows,features):
    errors=[r['worst_error_px'] for r in rows]; labels=[e>5 for e in errors]
    result={}
    for key,risk_direction in features.items():
        values=[r[key] for r in rows]
        result[key]={'spearman_with_worst_error':spearman(values,errors),
            'error_over5_ranking_auc':auc([risk_direction*v for v in values],labels),
            'risk_direction':'larger is higher risk' if risk_direction>0 else 'smaller is higher risk',
            'median_for_success':statistics.median([v for v,l in zip(values,labels) if not l]) if any(not l for l in labels) else None,
            'median_for_failure':statistics.median([v for v,l in zip(values,labels) if l]) if any(labels) else None}
    images=sorted({r['image'] for r in rows})
    for key,risk_direction in features.items():
        correct_pairs=0.0;total_pairs=0;within_images=[]
        for im in images:
            subset=[r for r in rows if r['image']==im]
            scores=[risk_direction*r[key] for r in subset]
            local_labels=[r['worst_error_px']>5 for r in subset]
            value=auc(scores,local_labels)
            if value is not None:
                pairs=sum(local_labels)*(len(local_labels)-sum(local_labels))
                correct_pairs+=value*pairs;total_pairs+=pairs
                within_images.append(value)
        result[key]['within_image_pair_weighted_auc']=correct_pairs/total_pairs if total_pairs else None
        result[key]['within_image_macro_auc']=statistics.mean(within_images) if within_images else None
        result[key]['images_with_both_success_and_failure']=len(within_images)
    return {'rows':len(rows),'success_within5':sum(not x for x in labels),'failure_over5':sum(labels),
        'represented_images':len(images),'failure_by_image':dict(Counter(r['image'] for r in rows if r['worst_error_px']>5)),
        'features':result}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--scorer',type=Path,required=True)
    p.add_argument('--clahe',type=Path,required=True);p.add_argument('--scale',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compress-rows',action='store_true',help='Write full rows as deterministic gzip JSON.')
    a=p.parse_args()
    assert sha(a.inputs)==INPUT_HASH and sha(a.scorer)==SCORER_HASH
    data,contrast,scale=map(read,(a.inputs,a.clahe,a.scale))
    spec=importlib.util.spec_from_file_location('scorer',a.scorer);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    assert auc([.9,.1],[True,False])==1 and auc([.1,.9],[True,False])==0 and auc([.5,.5],[True,False])==.5
    assert spearman([3,1,2],[30,10,20])==1
    baseline=[]
    for ref in REFS:
        for role in ('reference','candidate'):
            scores={im:s.score_image(g,data['predictions'][role][im],1280,720,.5,.5) for im,g in data['references'][ref]['gold'].items()}
            assert scores==data['expected_baseline'][ref][role]['per_image']
            assert s.aggregate_scores(list(scores.values()))==data['expected_baseline'][ref][role]['detection']
            baseline.append(ref+'/'+role)
    arms={
        'original':{im:r['postclip'] for im,r in scale['control'].items()},
        'contrast':{im:r['postclip'] for im,r in contrast['treatment'].items()},
        'enlarged':{im:r['postclip'] for im,r in scale['treatment'].items()},
    }
    # Native output coordinates of the two controls are byte-equal; confidence can drift slightly.
    for im,rows in arms['original'].items():
        other=contrast['control'][im]['postclip'];assert len(rows)==len(other)
        for x,y in zip(rows,other):
            assert x['bbox_xyxy']==y['bbox_xyxy']
            assert [k[:2] for k in x['keypoints']]==[k[:2] for k in y['keypoints']]
    associations={name:{im:mutual_match(arms['original'][im],rows,s._box_iou) for im,rows in values.items()} for name,values in arms.items() if name!='original'}
    summary={};all_rows={}
    for ref in REFS:
        gold=data['references'][ref]['gold']
        evaluated={name:{im:s.score_image(gold[im],rows,1280,720,.5,.5) for im,rows in arm.items()} for name,arm in arms.items()}
        correspondence={name:{im:{pair['prediction_index']:pair['gold_index'] for pair in score['matches']} for im,score in images.items()} for name,images in evaluated.items()}
        full=Counter({k:0 for k in ('all_eligible_two_corner_fronts','missed_box','matched_but_pair_unavailable','both_within5','pair_error_over5','both_within10','pair_error_over10')}); rows=[]
        for im,targets in gold.items():
            matches={p['gold_index']:p for p in evaluated['original'][im]['matches']}
            for gi,g in enumerate(targets):
                if not all(k[2]==2 for k in g['keypoints']):continue
                full['all_eligible_two_corner_fronts']+=1
                if gi not in matches:full['missed_box']+=1;continue
                pair=matches[gi];pi=pair['prediction_index'];pred=arms['original'][im][pi]
                if not usable(pred):full['matched_but_pair_unavailable']+=1;continue
                errs=[math.hypot(k[0]-t[0],k[1]-t[1]) for k,t in zip(pred['keypoints'],g['keypoints'])]
                error=max(errs)
                full['both_within5' if error<=5 else 'pair_error_over5']+=1
                full['both_within10' if error<=10 else 'pair_error_over10']+=1
                row={'case_id':ref+'/'+im+'/R'+str(gi+1).zfill(3),'image':im,'prediction_index':pi,
                    'corner_errors_px':errs,'worst_error_px':error,'box_score':pred['confidence'],
                    'minimum_corner_score':min(k[2] for k in pred['keypoints']),
                    'reference_box_iou':pair['iou'],'comparison_views':{}}
                for name in ('contrast','enlarged'):
                    ai=associations[name][im].get(pi)
                    if ai is None:
                        row['comparison_views'][name]={'association':'none'};continue
                    other=arms[name][im][ai]
                    matchgi=correspondence[name][im].get(ai)
                    status='same_reference_assignment' if matchgi==gi else 'unmatched_assignment' if matchgi is None else 'different_reference_assignment'
                    forward_ious=[s._box_iou(pred['bbox_xyxy'],v['bbox_xyxy']) for v in arms[name][im] if v['class_id']==pred['class_id']]
                    reverse_ious=[s._box_iou(other['bbox_xyxy'],v['bbox_xyxy']) for v in arms['original'][im] if v['class_id']==other['class_id']]
                    info={'association':status,'prediction_index':ai,'usable_pair':usable(other),
                        'predicted_box_iou':s._box_iou(pred['bbox_xyxy'],other['bbox_xyxy']),
                        'eligible_forward_candidates':sum(v>=.5 for v in forward_ious),
                        'eligible_reverse_candidates':sum(v>=.5 for v in reverse_ious),
                        'multiple_eligible_candidates':sum(v>=.5 for v in forward_ious)>1 or sum(v>=.5 for v in reverse_ious)>1,
                        'exact_best_iou_tie':sum(v==max(forward_ious) for v in forward_ious)>1 or sum(v==max(reverse_ious) for v in reverse_ious)>1}

                    if usable(other):info['maximum_corner_displacement_px']=max(math.hypot(k[0]-t[0],k[1]-t[1]) for k,t in zip(pred['keypoints'],other['keypoints']))
                    row['comparison_views'][name]=info
                views=row['comparison_views']
                if all(v.get('usable_pair') for v in views.values()):
                    row['maximum_view_disagreement_px']=max(v['maximum_corner_displacement_px'] for v in views.values())
                rows.append(row)
        assert sum(full[k] for k in ('missed_box','matched_but_pair_unavailable','both_within5','pair_error_over5'))==full['all_eligible_two_corner_fronts']
        common=[r for r in rows if 'maximum_view_disagreement_px' in r]
        features={'box_score':-1,'minimum_corner_score':-1,'reference_box_iou':-1}
        summary[ref]={'all_eligible_partition':dict(full),'matched_returned_pair_ranking':ranking(rows,features),
            'fixed_comparison_subset_ranking':ranking(common,{**features,'maximum_view_disagreement_px':1}),
            'view_association_counts':{name:{'outcome_counts':{status:sum(r['comparison_views'][name]['association']==status for r in rows) for status in ('same_reference_assignment','unmatched_assignment','different_reference_assignment','none')},
                'multiple_eligible_candidates':sum(r['comparison_views'][name].get('multiple_eligible_candidates',False) for r in rows),
                'exact_best_iou_ties':sum(r['comparison_views'][name].get('exact_best_iou_tie',False) for r in rows),
                'associated_but_pair_unavailable':sum(r['comparison_views'][name]['association']!='none' and not r['comparison_views'][name].get('usable_pair') for r in rows)} for name in ('contrast','enlarged')},
            'strong_score_failures_over5':sum(r['worst_error_px']>5 and r['minimum_corner_score']>=.8 for r in rows),
            'strong_score_failures_over10':sum(r['worst_error_px']>10 and r['minimum_corner_score']>=.8 for r in rows),
            'case_examples_largest_error':sorted(rows,key=lambda r:-r['worst_error_px'])[:5]}
        all_rows[ref]=rows
    output={'schema':'r57-corner-quality-diagnostic-v1','source_hashes':{'inputs':sha(a.inputs),'scorer':sha(a.scorer),'clahe_predictions':sha(a.clahe),'scale_predictions':sha(a.scale),'analysis_script':sha(Path(__file__))},
        'baseline_checks':baseline,'fixed_parameters':{'box_cutoff_strict':.13,'matching_iou_inclusive':.5,'endpoint_cutoff_inclusive':.5,'view_association':'Mutual-best same-class predicted-box IoU >=0.50, independent of reference','view_disagreement':'Maximum corresponding-corner distance across contrast and enlarged views'},
        'limitations':['Post-hoc descriptive ranking on 20 reused development images; outputs from the same photo are correlated.','Reference box overlap is an evaluation-only oracle feature.','Ranking cohort contains matched fronts with two reference corners and both returned points; it excludes unmatched/incomplete-reference outputs.','All quality scores are compared on the same fixed common-view subset; its exclusions are reported.','No thresholds fitted or recommended. No confidence calibration or generalization claim.','Reference assignments are evaluation identities, not physical proof of correspondence.'],
        'results':summary}
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'corner-quality-summary.json',output)
    if a.compress_rows:
        payload=(json.dumps(all_rows,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        (a.output/'corner-quality-rows.json.gz').write_bytes(gzip.compress(payload,mtime=0))
    else:
        write(a.output/'corner-quality-rows.json',all_rows)
    print(json.dumps({ref:{'partition':x['all_eligible_partition'],'all_ranking':x['matched_returned_pair_ranking'],'common_ranking':x['fixed_comparison_subset_ranking'],'associations':x['view_association_counts']} for ref,x in summary.items()},indent=2))
if __name__=='__main__':main()
