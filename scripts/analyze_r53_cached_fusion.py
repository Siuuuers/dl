#!/usr/bin/env python3
"""Replay fixed R53 output fusion from pinned PR149/150 archives, without inference.

The default output is compact evidence. Use --include-image-details for complete
per-image count rows and --write-fused-predictions to regenerate prediction arrays.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

EXPECTED_ARCHIVES={
 'clahe':'1e1dde4528729deced36a7943f1a3fc68e4e4993e05c0289e10fcf03f2cfa56d',
 'scale':'b1638b1eaf5e3877a77efb840c342bd1dfe7783ca5951041ed37b692df59abca'}
SOURCE_COMMIT='7b36e23dbbc24769f92a88cb93ae44acfc0406c5'
SOURCE_URL='https://github.com/Siuuuers/dl/tree/'+SOURCE_COMMIT
ARCHIVE_LOCATIONS={
 'clahe':{'pull_request':149,'commit':'343f8d759ccaba671fa38ad4e9674d75a1bb5ba7',
          'path':'results/l4/baomai-clahe-inference-r53-l4-001/baomai-clahe-inference-r53-results.zip'},
 'scale':{'pull_request':150,'commit':'e8a70f67cba1bf99e2d8a7bce9c93b5d4d7996b2',
          'path':'results/a100/baomai-scale-inference-r53-a100-001/baomai-scale-inference-r53-results.zip'}}
CONF=.13
FUSION_IOU=.70
MAX_DET=300
PAIR_CUTOFFS=(.5,.8)


def digest(data):
 return hashlib.sha256(data).hexdigest()


def read_return(path,kind):
 assert digest(path.read_bytes())==EXPECTED_ARCHIVES[kind], 'Returned archive hash differs'
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None
  inventory=json.loads(z.read('inventory.json'))
  for name,meta in inventory.items():
   data=z.read(name)
   assert meta=={'bytes':len(data),'sha256':digest(data)},name
  values={k:json.loads(z.read(k+'.json')) for k in ('summary','predictions','cases','preflight','baseline-equivalence')}
  values['inventory']=inventory
  values['archive_bytes']=path.stat().st_size
 assert values['summary']['status']=='completed' and values['summary']['interpretation_allowed']
 return values


def supported_sets(gold,outputs,scorer):
 metrics={}; detected=set(); pair5=set(); pair10=set()
 for image,targets in sorted(gold.items()):
  ps=outputs[image]; score=scorer.score_image(targets,ps,1280,720,.5,.5)
  metrics[image]=score
  for m in score['matches']:
   g,p=targets[m['gold_index']],ps[m['prediction_index']]
   identity=f'{image}/R{m["gold_index"]+1:03d}'
   detected.add(identity)
   if not all(x[2]==2 for x in g['keypoints']):continue
   if not all(valid_point(x,.5) for x in p['keypoints']):continue
   worst=max(math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(g['keypoints'],p['keypoints']))
   if worst<=5:pair5.add(identity)
   if worst<=10:pair10.add(identity)
 return metrics,{'detected':detected,'pair5':pair5,'pair10':pair10}


def valid_point(p,q):
 return all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) and 0<=p[0]<1280 and 0<=p[1]<720 and p[2]>=q


def pair_screen(gold,outputs,scorer,q):
 totals={k:0 for k in ('front_outputs','offered_pairs','withheld_pairs','eligible_pairs','unmatched','duplicates','reference_incomplete','supported5','supported10','measured_error_over5','measured_error_over10')}
 ids={5:set(),10:set()}; per_image={}
 for image,targets in sorted(gold.items()):
  ps=outputs[image]; accepted=[p for p in ps if all(valid_point(k,q) for k in p['keypoints'])]
  score=scorer.score_image(targets,accepted,1280,720,.5,q)
  c={k:0 for k in totals}
  c.update(front_outputs=len(ps),offered_pairs=len(accepted),withheld_pairs=len(ps)-len(accepted),eligible_pairs=sum(all(k[2]==2 for k in g['keypoints']) for g in targets),unmatched=score['fp'],duplicates=score['duplicate_fp'])
  for m in score['matches']:
   g,p=targets[m['gold_index']],accepted[m['prediction_index']]
   if not all(x[2]==2 for x in g['keypoints']):
    c['reference_incomplete']+=1;continue
   worst=max(math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(g['keypoints'],p['keypoints']))
   for tolerance in (5,10):
    c[f'supported{tolerance}']+=worst<=tolerance
    c[f'measured_error_over{tolerance}']+=worst>tolerance
    if worst<=tolerance:ids[tolerance].add(f'{image}/R{m["gold_index"]+1:03d}')
  for tolerance in (5,10):assert c['offered_pairs']==c[f'supported{tolerance}']+c[f'measured_error_over{tolerance}']+c['unmatched']+c['reference_incomplete']
  for k,v in c.items():totals[k]+=v
  per_image[image]=c
 for t in (5,10):
  totals[f'supported_fraction{t}']=totals[f'supported{t}']/totals['offered_pairs'] if totals['offered_pairs'] else None
  totals[f'eligible_recall{t}']=totals[f'supported{t}']/totals['eligible_pairs']
 return {'endpoint_cutoff_inclusive':q,**totals,'supported_target_ids':{str(t):sorted(ids[t]) for t in (5,10)},'per_image':per_image}


def main():
 if not __debug__:
  raise RuntimeError('Run without -O: the archive and arithmetic assertions must remain enabled.')
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1],help='Project source with unchanged R53 scoring helpers and references (default: this checkout).')
 ap.add_argument('--clahe-archive',type=Path,required=True,help='Exact PR149 return ZIP; hash verified.')
 ap.add_argument('--scale-archive',type=Path,required=True,help='Exact PR150 return ZIP; hash verified.')
 ap.add_argument('--output',type=Path,required=True,help='Directory for compact result JSON and Markdown.')
 ap.add_argument('--include-image-details',action='store_true',help='Include all per-image detection and pair-screen count rows in JSON.')
 ap.add_argument('--write-fused-predictions',action='store_true',help='Also write regeneratable full source-preserving fused-predictions.json.')
 args=ap.parse_args();args.root=args.root.expanduser().resolve();args.output.mkdir(parents=True,exist_ok=True)
 returns={k:read_return(getattr(args,k+'_archive'),k) for k in EXPECTED_ARCHIVES}
 contract=returns['scale']['summary']['contract']
 assert contract['source']==SOURCE_COMMIT==returns['clahe']['summary']['contract']['source']
 assert contract['references']==returns['clahe']['summary']['contract']['references']
 assert contract['checkpoint_sha256']==returns['clahe']['summary']['contract']['checkpoint_sha256']
 checked_source=[]
 for rel in ('scripts/pose_scoring.py','scripts/predict_pose_folder.py','scripts/evaluate_pose.py','scripts/run_inference_ablation_r53.py','scripts/audit_additional_front_pilot_r45_results.py'):
  assert digest((args.root/rel).read_bytes())==contract['source_files_sha256'][rel],rel
  checked_source.append(rel)
 sys.path.insert(0,str(args.root/'scripts'))
 import pose_scoring as scorer
 from predict_pose_folder import postclip_nms
 from evaluate_pose import parse_pose_labels
 from run_inference_ablation_r53 import baseline_equivalence
 # Verify local transitive imports as well as the explicit scoring entry points.
 for module in list(sys.modules.values()):
  filename=getattr(module,'__file__',None)
  if filename is None:continue
  path=Path(filename).resolve()
  if path==Path(__file__).resolve() or not path.is_relative_to(args.root/'scripts'):continue
  rel=path.relative_to(args.root).as_posix()
  assert rel in contract['source_files_sha256'],rel
  assert digest(path.read_bytes())==contract['source_files_sha256'][rel],rel
  if rel not in checked_source:checked_source.append(rel)
 checked_source.sort()
 refs={}
 for name in ('frozen666','historical663','r38','r50'):
  desc=contract['references'][name]
  assert digest((args.root/desc['manifest_path']).read_bytes())==desc['manifest_sha256'],desc['manifest_path']
  refs[name]={}
  for image,meta in desc['labels'].items():
   raw=(args.root/meta['path']).read_bytes()
   assert digest(raw)==meta['sha256'],meta['path']
   refs[name][image]=parse_pose_labels(raw.decode(),1280,720,name+'/'+image)
  assert sum(map(len,refs[name].values()))==desc['objects'],name
 controls=baseline_equivalence(refs,returns['clahe']['predictions']['control'],returns['scale']['predictions']['control'])
 assert controls['passed']
 saved={
  'identity':returns['scale']['predictions']['control'],
  'clahe':returns['clahe']['predictions']['treatment'],
  'scale1600':returns['scale']['predictions']['treatment']}
 original_checks=[]
 for kind,ret in returns.items():
  for arm,outputs in ret['predictions'].items():
   for image,row in outputs.items():
    kept,notes=postclip_nms(row['raw'])
    assert kept==row['postclip'] and notes==row['filter']
    assert all(p['confidence']>CONF for p in row['raw'])
   for ref,gold in refs.items():
    for view in ('raw','postclip'):
     per={i:scorer.score_image(gs,outputs[i][view],1280,720,.5,.5) for i,gs in gold.items()}
     expected=ret['summary']['reports'][arm][ref][view]
     assert per==expected['per_image'],(kind,arm,ref,view)
     assert scorer.aggregate_scores(list(per.values()))==expected['detection']
     original_checks.append(f'{kind}/{arm}/{ref}/{view}')
 outputs={name:{i:r['postclip'] for i,r in rows.items()} for name,rows in saved.items()}
 fusion_audit={}
 for other in ('clahe','scale1600'):
  name='identity_plus_'+other;outputs[name]={};fusion_audit[name]={}
  for image in saved['identity']:
   pool=[]
   for view in ('identity',other):
    for index,p in enumerate(saved[view][image]['raw'],1):
     if p['confidence']>CONF:pool.append({**p,'fusion_source_view':view,'fusion_source_raw_index':index})
   first,audit=postclip_nms(pool)
   assert audit['iou_threshold']==FUSION_IOU
   capped=first[:MAX_DET]
   final,postaudit=postclip_nms(capped)
   assert len(final)==len(capped) and not postaudit['suppressed_predictions']
   for p in final:
    origin=saved[p['fusion_source_view']][image]['raw'][p['fusion_source_raw_index']-1]
    assert all(p[k]==origin[k] for k in ('bbox_xyxy','confidence','class_id','keypoints'))
   outputs[name][image]=final
   fusion_audit[name][image]={'pooled_raw_outputs':len(pool),'after_global_nms':len(first),'after_cap':len(capped),'final_outputs':len(final),'suppressed':len(pool)-len(first),'cap_removed':len(first)-len(capped),'extra_postclip_suppressed':0,'retained_source_counts':{v:sum(p['fusion_source_view']==v for p in final) for v in ('identity',other)}}
 result={'schema':'r53-fixed-cached-fusion-v1','post_hoc':True,'fresh_inference':False,'fresh_training':False,'automatic_promotion':False,'controls_equivalence':controls,'original_score_checks':original_checks,'checked_source_files':checked_source,'provenance':{'archive_sha256':EXPECTED_ARCHIVES,'script_sha256':digest(Path(__file__).read_bytes()),'checkpoint_sha256':contract['checkpoint_sha256'],'source':contract['source']},'protocol':{'identity_control':'PR150 control used consistently; PR149 control discrete outcomes agree','confidence_keep':'strictly greater than 0.13','fusion':'Concatenate per-view raw arrays already native-NMS and clipped to original image; identity first for equal-score ties. Apply classwise score-ordered greedy NMS with IoU>0.70 suppression, cap at300, then original postclip NMS0.70 (verified idempotent). Preserve each retained full bbox and original landmark pair.','native_output_limit':'Caches do not contain pre-native-NMS or sub0.13 proposals, so this is not a reconstruction of joint native decoded-proposal fusion.','oracle_union':'Retrospective union of reference-specific successes in the separate original-view matchings; ceiling only for an oracle choosing among those already scored successful rows, not an attainable predictor or a universal bound on new assignments.','pair_definition':'All offered pairs denominator; all eligible two-v2 reference fronts recall denominator; rematch offered predictions against all reference fronts. Unavailable/unmatched outputs unverified, not all proven physical errors.'},'references':{},'fusion_audit':fusion_audit,'pair_screen_scale':{}}
 for ref,gold in refs.items():
  per_view={};sets={};eligible=sum(all(k[2]==2 for k in g['keypoints']) for gs in gold.values() for g in gs)
  for name,ps in outputs.items():
   per,support=supported_sets(gold,ps,scorer);aggregate=scorer.aggregate_scores(list(per.values()))
   aggregate.pop('endpoint_errors_px')
   sets[name]=support
   per_view[name]={'detection':aggregate,'eligible_pairs':eligible,'pair5':len(support['pair5']),'pair10':len(support['pair10']),'empty_scene_outputs':{i:len(ps[i]) for i,gs in gold.items() if not gs},'per_image_counts':{i:{k:s[k] for k in ('gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp')} for i,s in per.items()}}
   if name!='identity':
    per_view[name]['changes_vs_identity']={m:{'gained':sorted(support[m]-sets['identity'][m]),'lost':sorted(sets['identity'][m]-support[m])} for m in ('detected','pair5','pair10')}
  unions={}
  for names in (('identity','clahe'),('identity','scale1600'),('identity','clahe','scale1600')):
   u={}
   for metric in ('detected','pair5','pair10'):
    ids=set().union(*(sets[n][metric] for n in names))
    u[metric]={'count':len(ids),'additional_to_identity':sorted(ids-sets['identity'][metric])}
   unions['+'.join(names)]=u
  result['references'][ref]={'views':per_view,'oracle_unions':unions}
  result['pair_screen_scale'][ref]={name:[pair_screen(gold,outputs[name],scorer,q) for q in PAIR_CUTOFFS] for name in ('identity','scale1600')}
  for name,rows in result['pair_screen_scale'][ref].items():
   base={tol:set(rows[0]['supported_target_ids'][tol]) for tol in ('5','10')}
   for row in rows:
    ids=row.pop('supported_target_ids')
    row['supported_id_changes_vs_endpoint05']={tol:{'gained':sorted(set(ids[tol])-base[tol]),'lost':sorted(base[tol]-set(ids[tol]))} for tol in ('5','10')}
  for name,ps in outputs.items():
   direct=scorer.aggregate_scores([scorer.score_image(gold[i],ps[i],1280,720,.5,.5) for i in sorted(gold)])
   direct.pop('endpoint_errors_px');assert direct==per_view[name]['detection']
 result['verification']={'native_raw_to_postclip_all_exact':True,'original_score_reports_exact':len(original_checks),'fusion_predictions_preserve_entire_source_records':True,'all_views_direct_rescore_exact':True,'pair_categories_partition_each_image':True}
 result['schema']='r53-fixed-cached-fusion-v2'
 result['provenance']['source_url']=SOURCE_URL
 result['provenance']['source_files_sha256']={rel:contract['source_files_sha256'][rel] for rel in checked_source}
 result['provenance']['archives']={}
 for kind,where in ARCHIVE_LOCATIONS.items():
  result['provenance']['archives'][kind]={**where,'bytes':returns[kind]['archive_bytes'],'sha256':EXPECTED_ARCHIVES[kind],
   'url':f'https://github.com/Siuuuers/dl/blob/{where["commit"]}/{where["path"]}',
   'download_url':f'https://raw.githubusercontent.com/Siuuuers/dl/{where["commit"]}/{where["path"]}',
   'used_members':{name:meta for name,meta in returns[kind]['inventory'].items() if name in ('summary.json','predictions.json','cases.json','preflight.json','baseline-equivalence.json')}}
 result['provenance']['reference_inputs']={name:{**contract['references'][name],
  'manifest_url':f'https://github.com/Siuuuers/dl/blob/{SOURCE_COMMIT}/{contract["references"][name]["manifest_path"]}'} for name in refs}
 result['projection']={'image_details_included':args.include_image_details,'fused_arrays_written':args.write_fused_predictions,
  'meaning':'Compact output retains aggregates, all paired gain/loss identities, oracle additions, empty scenes, and input provenance. Per-image count rows are optional; full fused arrays are reproducible with --write-fused-predictions.'}
 if not args.include_image_details:
  for reference in result['references'].values():
   for view in reference['views'].values():view.pop('per_image_counts')
  for reference in result['pair_screen_scale'].values():
   for rows in reference.values():
    for row in rows:row.pop('per_image')
 (args.output/'cached-fusion-results.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 if args.write_fused_predictions:
  (args.output/'fused-predictions.json').write_text(json.dumps({k:v for k,v in outputs.items() if k.startswith('identity_plus_')},indent=2,sort_keys=True,allow_nan=False)+'\n')
 lines=['# R53 fixed cached output fusion diagnostic','','This post-hoc replay tests complementarity and fixed, score-ordered NMS fusion of the returned views. It performs no inference or training, changes no defaults, and does not establish a deployment result.','', 'The raw records already passed each view\'s native NMS and were clipped/mapped into original image coordinates. They are pooled before the extra postclip pass, with confidence >0.13, fixed IoU >0.70 suppression, maximum 300 and no coordinate averaging. A final original postclip pass is verified idempotent. Removed native proposals cannot be recovered.','']
 lines+=['## Immutable inputs and reproduction','',f'The existing R53 scoring and input protocol is pinned to [source `{SOURCE_COMMIT}`]({SOURCE_URL}). The replay verifies source-helper hashes, all four reference manifests, all 80 reference label files, both archive hashes and their member inventories. The JSON records every input reference path and SHA-256.','']
 for kind,meta in result['provenance']['archives'].items():
  lines.append(f'- [PR{meta["pull_request"]} {kind} archive]({meta["url"]}), commit `{meta["commit"]}`, {meta["bytes"]:,} bytes, SHA-256 `{meta["sha256"]}`; [download]({meta["download_url"]}).')
 lines+=['',f'Checkpoint identity: `{contract["checkpoint_sha256"]}` (the PR137 candidate used by both R53 runs). The saved R53 returns contain the control records required here; PR146, weights, source photographs, a GPU, and model inference are not needed for this CPU replay. This diagnostic does not independently rerun the archived model.','',
  'From this repository checkout with Python 3, after saving the two linked ZIPs outside the checkout:','','```bash',
  'python -B scripts/analyze_r53_cached_fusion.py \\',
  '  --clahe-archive /path/to/baomai-clahe-inference-r53-results.zip \\',
  '  --scale-archive /path/to/baomai-scale-inference-r53-results.zip \\',
  '  --output /path/to/replay-results','```','',
  'The default `--root` is the checkout containing this script. If its old scoring helpers or reference files have changed, pass `--root /path/to/pinned-r53-source` from the immutable source commit above; do not substitute scoring code. This path uses the Python standard library and project helpers. Do not run with `python -O`, which disables validation assertions.','',
  'The default results are compact. Add `--include-image-details` to include all per-image count rows, and `--write-fused-predictions` to regenerate the complete fused arrays with original source-view and source-index provenance. The compact JSON retains all gained/lost identities and empty-scene counts. No fused arrays need to be committed.','']
 for ref,r in result['references'].items():
  lines+=['## '+ref,'','| Output view | Matched | Missed | Unmatched | Duplicate subset | Pair ≤5 px | Pair ≤10 px |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
  for name,v in r['views'].items():
   d=v['detection'];lines.append(f'| {name} | {d["tp"]} | {d["fn"]} | {d["fp"]} | {d["duplicate_fp"]} | {v["pair5"]}/{v["eligible_pairs"]} | {v["pair10"]}/{v["eligible_pairs"]} |')
  lines+=['','### Retrospective success unions','','These use reference knowledge and separate per-view assignments. They are oracle ceilings for choosing among those already successful rows, not attainable output rules or universal bounds on every possible fusion assignment.','']
  for name,u in r['oracle_unions'].items():lines.append(f'- {name}: matched union {u["detected"]["count"]}; pair ≤5 union {u["pair5"]["count"]}; pair ≤10 union {u["pair10"]["count"]}.')
  lines+=['','### Pair screen on standalone scale outputs','','| View | Endpoint cutoff | Offered | Supported ≤5 | Supported fraction | Eligible recall | Unmatched | Incomplete reference pair | Measured >5 px |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
  for name,rows in result['pair_screen_scale'][ref].items():
   for v in rows:lines.append(f'| {name} | {v["endpoint_cutoff_inclusive"]:.2f} | {v["offered_pairs"]} | {v["supported5"]} | {v["supported_fraction5"]:.4%} | {v["eligible_recall5"]:.4%} | {v["unmatched"]} | {v["reference_incomplete"]} | {v["measured_error_over5"]} |')
  lines.append('')
 lines+=['## Verification and limits','','All 32 original raw/postclip model-reference tables reproduce exactly. Identity controls have identical coordinates and discrete outcomes; maximum confidence drift is recorded in JSON. All new views were directly rescored, source pairs were preserved exactly, and pair-support categories partition each image. The four references share the same 20 repeatedly used development photographs. Thresholds were fixed for this diagnostic and were not optimized. Unmatched and unavailable-reference pairs are not all proven physical errors. Full gains/losses and empty-scene counts are in the JSON.','']
 (args.output/'cached-fusion-results.md').write_text('\n'.join(lines))
 print(json.dumps({'output':str(args.output),'original_tables':len(original_checks),'new_output_variants':2,'all_direct_scores_match':True}))


if __name__=='__main__':main()
