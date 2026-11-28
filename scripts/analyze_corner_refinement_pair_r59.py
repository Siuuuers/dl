#!/usr/bin/env python3
"""Independently compare the two returned local-corner refinement runs."""
from collections import Counter
import argparse,gzip,hashlib,json,math,statistics,sys
from pathlib import Path
from zipfile import ZipFile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--rgb-return',type=Path,required=True)
parser.add_argument('--hints-return',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
ROOT=args.root.resolve();RGB=args.rgb_return.resolve();HINTS=args.hints_return.resolve()
sys.path.insert(0,str(ROOT/'scripts'))
from audit_reflection_return import load_references
from pose_scoring import score_image,aggregate_scores
import run_inference_ablation_r53 as r53
import corner_refinement_r58 as geometry

def read(p):return json.loads(p.read_text())
def encoded(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def pointvalid(p):return len(p)==3 and all(v is not None and isinstance(v,(int,float)) and math.isfinite(v) for v in p) and 0<=p[0]<1280 and 0<=p[1]<720 and p[2]>=.5

def independent_crop(box):
 x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
 b=[max(0,math.floor(x0-w/4)),max(0,math.floor(y0-h/4)),min(1280,math.ceil(x1+w/4)),min(720,math.ceil(y1+h/4))]
 cw,ch=b[2]-b[0],b[3]-b[1];g=min(256/cw,256/ch)
 rw,rh=max(1,min(256,round(cw*g))),max(1,min(256,round(ch*g)))
 return {'xyxy':b,'resized_wh':[rw,rh],'scale_xy':[rw/cw,rh/ch],'pad_xy':[(256-rw)//2,(256-rh)//2],'canvas_wh':[256,256]}

def independent_trace(t,g):
 status=t['status'];l,top,r,b=g['xyxy'];sx,sy=g['scale_xy'];px,py=g['pad_xy'];rw,rh=g['resized_wh']
 if status not in ('decoded','outside_crop'):
  assert status in ('nonfinite_heatmap','no_valid_content','nonpositive_peak')
  if status=='nonpositive_peak':assert math.isfinite(t['peak_value']) and t['peak_value']<=0
  return None
 x,y=t['peak_xy'];assert 0<=x<64 and 0<=y<64
 inside=lambda a,b:px<=(a+.5)*4<px+rw and py<=(b+.5)*4<py+rh
 assert inside(x,y) and t['peak_value']>0 and math.isfinite(t['peak_value'])
 offsets=[]
 for axis,enabled in [('x',0<x<63 and inside(x-1,y) and inside(x+1,y)),('y',0<y<63 and inside(x,y-1) and inside(x,y+1))]:
  n=t['neighbours'][axis]
  if enabled:
   assert len(n)==2 and all(math.isfinite(v) and v<=t['peak_value'] for v in n)
   d=n[1]-n[0];offsets.append(.25*(1 if d>0 else -1 if d<0 else 0))
  else:assert n is None;offsets.append(0.)
 assert offsets==t['quarter_offset_xy']
 uv=[(x+.5+offsets[0])*4,(y+.5+offsets[1])*4];assert uv==t['canvas_xy']
 p=[(uv[0]-px)/sx+l,(uv[1]-py)/sy+top]
 insidecrop=l<=p[0]<r and top<=p[1]<b
 assert insidecrop==(status=='decoded')
 return p if insidecrop else None

for mode,path,expected in [('rgb',RGB,'16587610aa30a743798905d5cfe371abb959ed8779389fb6a0032403e0b457d3'),('hints',HINTS,'ec67e56d59b9774c676aa5ff8b9906c817ee0682a917a9facb7f277f547bdbdf')]:
 assert sha((path/'inventory.json').read_bytes())==expected,'Returned inventory differs: '+mode
 for name,metadata in read(path/'inventory.json').items():
  payload=(path/name).read_bytes();assert metadata=={'bytes':len(payload),'sha256':sha(payload)},name

summaries={m:read(p/'summary.json') for m,p in [('rgb',RGB),('hints',HINTS)]}
predictions={m:read(p/'predictions.json') for m,p in [('rgb',RGB),('hints',HINTS)]}
refs=load_references(ROOT,summaries['rgb']['contract'])
checks={}
for name in ['staging-r58.json','detector-training.json','training-crops.json','detector-runtime.json']:
 checks[name]=(RGB/name).read_bytes()==(HINTS/name).read_bytes();assert checks[name]
checks['control_predictions']=predictions['rgb']['control']==predictions['hints']['control'];assert checks['control_predictions']
checks['initial_model']=read(RGB/'model-smoke.json')['initial_model']==read(HINTS/'model-smoke.json')['initial_model'];assert checks['initial_model']
logs={m:read(p/'training-log.json') for m,p in [('rgb',RGB),('hints',HINTS)]}
exposure=lambda rows:[{k:r[k] for k in ('epoch','samples','updates','sample_order_sha256','next_learning_rate')} for r in rows]
checks['training_exposure_and_order']=exposure(logs['rgb'])==exposure(logs['hints']);assert checks['training_exposure_and_order']
for mode,path in [('rgb',RGB),('hints',HINTS)]:
 contract=summaries[mode]['contract'];assert contract['source']=='5b1650b8fa3959a0917680305d7b499af783c5aa'
 for script in ['run_corner_refinement_r58.py','corner_refinement_r58.py']:
  assert (path/script).read_bytes()==(ROOT/'scripts'/script).read_bytes()
 expected={arm:r53.reports(refs,p) for arm,p in predictions[mode].items()}
 assert expected==summaries[mode]['reports']
 details=read(path/'crop-transforms.json');seen=set();states=Counter()
 for row in details:
  identity,index=row['image_id'],row['raw_index'];assert (identity,index) not in seen;seen.add((identity,index))
  before=predictions[mode]['control'][identity]['raw'][index]
  after=predictions[mode]['treatment'][identity]['raw'][index]
  assert row['geometry']==independent_crop(before['bbox_xyxy'])
  pts=[independent_trace(t,row['geometry']) for t in row['decoder_trace']]
  states.update(t['status'] for t in row['decoder_trace'])
  assert pts==row['decoded_points']
  assert after['keypoints']==[[None,None,p[2]] if q is None else [q[0],q[1],p[2]] for q,p in zip(pts,before['keypoints'])]
  assert {k:v for k,v in before.items() if k!='keypoints'}=={k:v for k,v in after.items() if k!='keypoints'}
 assert len(seen)==sum(len(r['raw']) for r in predictions[mode]['control'].values())
 checks[mode+'_score_tables_replayed']=16;checks[mode+'_decoder_records_replayed']=len(seen);checks[mode+'_decoder_states']=dict(states)

all_rows={};reports={}
arms={'control':predictions['rgb']['control'],'rgb':predictions['rgb']['treatment'],'hints':predictions['hints']['treatment']}
for reference,gold in refs.items():
 rows=[]
 for image,targets in sorted(gold.items()):
  outputs={arm:values[image]['postclip'] for arm,values in arms.items()}
  scores={arm:score_image(targets,values,1280,720,.5,.5) for arm,values in outputs.items()}
  matches={arm:{m['gold_index']:m['prediction_index'] for m in score['matches']} for arm,score in scores.items()}
  assert matches['control']==matches['rgb']==matches['hints']
  for index,target in enumerate(targets):
   if not all(p[2]==2 for p in target['keypoints']):continue
   pi=matches['control'].get(index);bbox=target['bbox_xyxy'];height=bbox[3]-bbox[1]
   row={'image':image,'row':index+1,'id':image+'/R'+str(index+1).zfill(3),'prediction_index':pi,'reference_bbox_xyxy':bbox,
        'height_group':'<16' if height<16 else '16-<32' if height<32 else '32-<64' if height<64 else '>=64','arms':{}}
   for arm,values in outputs.items():
    p=values[pi] if pi is not None else None
    valid=p is not None and all(pointvalid(k) for k in p['keypoints'])
    signed=[[a[0]-b[0],a[1]-b[1]] for a,b in zip(p['keypoints'],target['keypoints'])] if valid else None
    errors=[math.hypot(*d) for d in signed] if valid else None
    row['arms'][arm]={'errors_px':errors,'signed_errors_xy_px':signed,'maximum_error_px':max(errors) if valid else None,'within5':valid and max(errors)<=5,'within10':valid and max(errors)<=10}
   if pi is not None:
    crop=geometry.crop_geometry(outputs['control'][pi]['bbox_xyxy'])
    heatmaps,available,_=geometry.encode_targets(target['keypoints'],crop)
    ideal=geometry.decode_heatmaps(heatmaps,crop)
    assert all(available) and all(x is not None for x in ideal)
    row['ideal_target_decode_errors_px']=[math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(ideal,target['keypoints'])]
    row['ideal_target_decode_max_error_px']=max(row['ideal_target_decode_errors_px'])
   rows.append(row)
 summary={'eligible':len(rows),'arms':{},'changes':{}}
 for arm in arms:
  e=[r['arms'][arm]['maximum_error_px'] for r in rows if r['arms'][arm]['maximum_error_px'] is not None]
  sorted_errors=sorted(e)
  percentile=lambda q:sorted_errors[min(len(e)-1,max(0,math.ceil(q*len(e))-1))]
  summary['arms'][arm]={'available_matched_pairs':len(e),'within5':sum(r['arms'][arm]['within5'] for r in rows),'within10':sum(r['arms'][arm]['within10'] for r in rows),
   'mean_maximum_error_px':statistics.mean(e),'median_maximum_error_px':statistics.median(e),'p90_maximum_error_px':percentile(.9),'p95_maximum_error_px':percentile(.95),'maximum_error_px':max(e),
   'per_image':{image:{'eligible':sum(r['image']==image for r in rows),'within5':sum(r['image']==image and r['arms'][arm]['within5'] for r in rows),'within10':sum(r['image']==image and r['arms'][arm]['within10'] for r in rows)} for image in gold},
   'per_height':{height:{'eligible':sum(r['height_group']==height for r in rows),'within5':sum(r['height_group']==height and r['arms'][arm]['within5'] for r in rows)} for height in sorted({r['height_group'] for r in rows})},
   'signed_error_by_slot':[{'mean_dx_px':statistics.mean(r['arms'][arm]['signed_errors_xy_px'][slot][0] for r in rows if r['arms'][arm]['signed_errors_xy_px'] is not None),
   'mean_dy_px':statistics.mean(r['arms'][arm]['signed_errors_xy_px'][slot][1] for r in rows if r['arms'][arm]['signed_errors_xy_px'] is not None)} for slot in (0,1)]}
 for arm in ['rgb','hints']:
  summary['changes'][arm]={}
  for threshold in (5,10):
   key='within'+str(threshold);gained=[r for r in rows if not r['arms']['control'][key] and r['arms'][arm][key]];lost=[r for r in rows if r['arms']['control'][key] and not r['arms'][arm][key]]
   summary['changes'][arm][str(threshold)]={'gained':len(gained),'lost':len(lost),'gained_ids':[r['id'] for r in gained],'lost_ids':[r['id'] for r in lost],'losses_by_image':dict(Counter(r['image'] for r in lost))}
 ideal=[r['ideal_target_decode_max_error_px'] for r in rows if 'ideal_target_decode_max_error_px' in r]
 endpoint_ideal=[e for r in rows if 'ideal_target_decode_errors_px' in r for e in r['ideal_target_decode_errors_px']]
 summary['ideal_target_decode']={'matched_pairs':len(ideal),'endpoints':len(endpoint_ideal),'mean_endpoint_error_px':statistics.mean(endpoint_ideal),'maximum_endpoint_error_px':max(endpoint_ideal),'mean_error_px':statistics.mean(ideal),'maximum_error_px':max(ideal),'within5':sum(e<=5 for e in ideal),'interpretation':'Synthetic reference-target heatmaps passed through the fixed decoder; measures representation/decoder capacity, not achieved network accuracy or a prediction method.'}
 reports[reference]=summary;all_rows[reference]=rows

common={}
from predict_pose_folder import postclip_nms
for arm,outputs in arms.items():
 filtered={}
 for image,row in outputs.items():
  raw=[p for p in row['raw'] if p['confidence']>.25]
  kept,audit=postclip_nms(raw)
  expected=[p for p in row['postclip'] if p['confidence']>.25]
  strip=lambda rows:[{k:v for k,v in p.items() if k!='primary_prediction_id'} for p in rows]
  assert strip(kept)==strip(expected),'Suppression/filter commutation differs'
  filtered[image]={'raw':raw,'postclip':kept,'filter':audit}
 scored=r53.reports(refs,filtered)
 common[arm]={ref:{'detection':{k:v['postclip']['detection'][k] for k in ('gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp')},'complete_pairs':v['postclip']['corners_by_locatable_count']['2'],
   'per_image':{im:{'detection':{k:score[k] for k in ('gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp')}} for im,score in v['postclip']['per_image'].items()}} for ref,v in scored.items()}

result={'schema':'local-refinement-paired-analysis-r59-v1','status':'passed','source_commit':'5b1650b8fa3959a0917680305d7b499af783c5aa','cross_arm_checks':checks,
 'baseline':'Expanded-data research detector, not the supplied original baomai model.','common_box_confidence_025':{'threshold':.25,'comparison':'strictly greater','suppression_filter_commutation_verified':True,'arms':common},'source_sha256':sha(Path(__file__).read_bytes()),'references':reports,'runtime':read(RGB/'detector-runtime.json'),
 'training':{'usable_crops':read(RGB/'training-crops.json')['usable_crops'],'counts':read(RGB/'training-crops.json')['counts'],
 'epochs':30,'updates':3600,'sample_exposures':114450,'initial_state_sha256':read(RGB/'model-smoke.json')['initial_model']['state_sha256'],
 'loss_samples':{m:[{k:r[k] for k in ('epoch','mean_loss')} for r in v if r['epoch'] in (1,5,10,20,30)] for m,v in logs.items()}},
 'interpretation':'Both returned refiners reduce five-pixel pair success and increase mean maximum-corner error on the same development photographs. Saved decoder arithmetic is verified; full-heatmap inference is a separate diagnostic.'}
out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
(out/'paired-analysis.json').write_bytes(encoded(result));(out/'paired-case-changes.json.gz').write_bytes(gzip.compress(encoded(all_rows),mtime=0))
(out/'ideal-heatmap-diagnostic.json').write_bytes(encoded({'schema':'ideal-heatmap-decoder-check-r59-v1','status':'completed','source_sha256':sha(Path(__file__).read_bytes()),'canvas_size':[256,256],'heatmap_size':[64,64],'gaussian_sigma_cells':2.0,'crop_expansion':1.5,'decoder':'Valid-content maximum plus signed quarter-cell neighbour offset, then inverse actual rounded crop scale.','reference_use':'Gaussian targets are constructed at annotated reference corners, solely to test representation and decoder capacity. These are synthetic target responses, not model predictions.','references':{ref:v['ideal_target_decode'] for ref,v in reports.items()}}))
print(json.dumps({'checks':checks,'references':{ref:{'eligible':v['eligible'],'counts':{arm:{k:a[k] for k in ['within5','within10','mean_maximum_error_px','median_maximum_error_px']} for arm,a in v['arms'].items()},'ideal_target_decode':v['ideal_target_decode']} for ref,v in reports.items()}},indent=2))
