#!/usr/bin/env python3
"""Audit immutable R27 source files independently and write a verification receipt.

Usage from the repository root:
    python results/review/front-review-20261123-r27/verify_independent.py

The audit does not import the R27 builder or its validators. It reads source
files without changing them; only the requested receipt path is written.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,math,subprocess
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3],
                    help='Repository root; inferred from this script by default.')
parser.add_argument('--output', type=Path,
                    default=Path('results/verification/front-review-r27-independent.json'),
                    help='Receipt path, relative to --root unless absolute.')
args = parser.parse_args()
R = args.root.resolve()
OUT = args.output.resolve() if args.output.is_absolute() else R / args.output
SCRIPT = Path(__file__).resolve()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
checks=[];failures=[]
def ck(ok,name,detail=None):
 item={'check':name,'passed':bool(ok)}
 if detail is not None:item['detail']=detail
 checks.append(item)
 if not ok:failures.append(item)
def geom(line):
 v=list(map(float,line.split()));return {'box_xyxy':[(v[1]-v[3]/2)*1280,(v[2]-v[4]/2)*720,(v[1]+v[3]/2)*1280,(v[2]+v[4]/2)*720], 'keypoints':[[v[5]*1280,v[6]*720,int(v[7])],[v[8]*1280,v[9]*720,int(v[10])]]}
def close(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(close(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
 return abs(a-b)<1e-7
mp=R/'data/reviews/fronts-2026-11-23-r27/manifest.json';m=read(mp);bp=R/'data/reviews/fronts-2026-11-23-r26/manifest.json';base=read(bp);origin=read(R/'data/manifest.json')
ld=R/m['review_evidence'][0]['path'];ledger=read(ld);ops=read(R/m['operations']['path']);coverage=read(R/m['request_coverage']['path']);maps=read(R/m['row_maps']['path']);summary=read(R/m['amendment_summary']['path'])
ck(sha(mp)=='9e3675c83cb6735242b3b99266e36d5a33248c13e51b8c91bd89a65d2c4b92af','stable_manifest_hash')
ck(sha(ld)=='c5598af2c8f71da10f38c074ad7757f3befd5ca50e350a569cbb42afdb24680d','stable_ledger_hash')
ck(sha(bp)==m['parent']['sha256']=='b999b9491b18ecd2d55684ec14383d1c98160a6bf080a4a751ea83844fca7468','immutable_R26_parent_hash')
ck(sha(R/'data/manifest.json')==m['source_manifest_sha256']=='72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926','original_source_manifest_hash')
# Independently resolve all file descriptors bound by the final manifest.
def descriptors(value):
 if isinstance(value,dict):
  if 'path' in value and 'sha256' in value:yield value
  for v in value.values():yield from descriptors(v)
 elif isinstance(value,list):
  for v in value:yield from descriptors(v)
descs=list(descriptors(m));bad=[d['path'] for d in descs if not (R/d['path']).is_file() or sha(R/d['path'])!=d['sha256']];ck(not bad,'all_manifest_descriptors_resolve_and_hash',{'descriptors':len(descs),'mismatches':bad})
basefiles={f['image_id']:f for f in base['files']};current={f['image_id']:f for f in m['files']};originals={f['id']:f for f in origin['records']}
ck(len(basefiles)==len(current)==len(originals)==157 and set(basefiles)==set(current)==set(originals),'157_image_inventory_identity')
opmap={};byimage=defaultdict(list)
for op in ops:
 key=(op['image_id'],op['base_row']);ck(key not in opmap,'unique_operation_'+str(key));opmap[key]=op;byimage[key[0]].append(op)
ck(len(ops)==21 and all(op['action']=='replace' for op in ops),'exactly_21_replacements_no_add_delete')
ck(dict(Counter(op['split'] for op in ops))=={'val':14,'train':7},'operation_split_14_development_7_train')
expected_keys={('new_000007',r) for r in [24,33,37,41,42]}|{('new_000008',r) for r in [33,34,35]}|{('new_000009',r) for r in [48,49,50,51,52]}|{('new_000013',27),('new_000014',35),('new_000016',26),('new_000016',27),('new_000049',3),('new_000051',20),('new_000056',12),('new_000083',12)}
ck(set(opmap)==expected_keys,'exact_accepted_replacement_identities')
parent_bytes={};saved_bytes={};stats=defaultdict(Counter);origin_stats=defaultdict(Counter);changed_files=[];rows_total=0;unmodified_rows=0;zero_triplets=0
for iid in sorted(current):
 old,new,raw=basefiles[iid],current[iid],originals[iid];split=raw['split'];pold=bp.parent/old['label_path'];pnew=mp.parent/new['label_path'];old_bytes=pold.read_bytes();new_bytes=pnew.read_bytes();parent_bytes[iid]=old_bytes;saved_bytes[iid]=new_bytes
 ck(old['split']==new['split']==split and old['label_path']==new['label_path']==raw['label'],'split_and_label_path_'+iid)
 ck(sha(pold)==old['label_sha256'],'R26_label_hash_'+iid)
 ck(sha(pnew)==new['label_sha256'],'R27_label_hash_'+iid)
 ck(sha(R/'data'/raw['image'])==raw['image_sha256']==old['source_image_sha256']==new['source_image_sha256'],'original_image_hash_'+iid)
 ck(sha(R/'data'/raw['label'])==raw['label_sha256'],'original_label_hash_'+iid)
 before=old_bytes.decode().splitlines();after=new_bytes.decode().splitlines();ck(len(before)==len(after)==old['objects']==new['objects'],'unchanged_row_count_'+iid)
 replay=old_bytes.splitlines(keepends=True)
 for op in byimage[iid]:
  j=op['base_row']-1;ck(before[j]==op['before_line'],'operation_before_'+iid+':'+str(j+1));ending=replay[j][len(replay[j].rstrip(b'\r\n')):];replay[j]=op['after_line'].encode()+ending
 ck(b''.join(replay)==new_bytes,'independent_exact_byte_replay_'+iid)
 ck(bool(byimage[iid])==new['changed_in_r27'],'changed_file_flag_'+iid)
 if old_bytes!=new_bytes:changed_files.append(iid)
 ck(not maps[iid]['deleted_base_rows'] and len(maps[iid]['rows'])==len(before) and all(x['current_row']==x['base_row'] and x['changed']==((iid,x['base_row']) in opmap) for x in maps[iid]['rows']),'identity_row_map_'+iid)
 stats[split]['images']+=1;stats[split]['objects']+=len(after);stats[split]['empty_images']+=not after
 for row,(a,b) in enumerate(zip(before,after),1):
  at,bt=a.split(),b.split();rows_total+=1
  ck(len(at)==len(bt)==11,'eleven_tokens_'+iid+':'+str(row))
  ck(all(at[k]==bt[k] for k in [0,7,10]),'class_visibility_exact_'+iid+':'+str(row))
  numeric=list(map(float,bt));ck(all(math.isfinite(x) for x in numeric) and all(0<=numeric[k]<=1 for k in [1,2,3,4,5,6,8,9]) and numeric[3]>0 and numeric[4]>0 and numeric[0]==0 and numeric[7] in [0,2] and numeric[10] in [0,2],'normalized_geometry_valid_'+iid+':'+str(row))
  if (iid,row) not in opmap:unmodified_rows+=1;ck(a==b,'unmodified_row_exact_'+iid+':'+str(row))
  for k in [5,8]:
   vis=int(bt[k+2]);stats[split]['visible_endpoints' if vis==2 else 'missing_endpoints']+=1
   if int(at[k+2])==0:zero_triplets+=1;ck(at[k:k+3]==bt[k:k+3] and all(float(x)==0 for x in bt[k:k+3]),'unknown_triplet_exact_'+iid+':'+str(row)+':'+str(k))
  if int(bt[7])==int(bt[10])==2:ck(float(bt[5])<=float(bt[8]),'visible_corner_x_order_'+iid+':'+str(row))
 raw_lines=(R/'data'/raw['label']).read_text().splitlines();origin_stats[split]['images']+=1;origin_stats[split]['objects']+=len(raw_lines)
ck(len(changed_files)==10 and sorted(changed_files)==m['changed_image_ids'],'10_changed_147_unchanged_files')
ck({k:dict(v) for k,v in stats.items()}==m['expected_counts']==base['expected_counts'],'review_counts_unchanged_from_R26',dict(stats))
ck(origin_stats['val']['objects']==664 and origin_stats['val']['images']==20 and origin_stats['train']['images']==137,'original_trainer_validation_664_and_split_preserved',dict(origin_stats))
# Cross-check the final ledger against the separately saved reviewer evidence.
independent={}
for descriptor in m['independent_review_evidence']:
 evidence=read(R/descriptor['path'])
 if 'proposals' in evidence:
  entries=[(x['image_id'],x['source_row_1based'],x['before']['line'],x['after']['line']) for x in evidence['proposals']]+[(x['image_id'],x['source_row_1based'],x['line'],x['line']) for x in evidence['retained_rows']]
 else:entries=[(x['image_id'],x['row'],x['before_line'],x['after_line']) for x in evidence.get('reviewed_rows',evidence.get('reviewed_cases',[]))]
 for iid,row,a,b in entries:ck((iid,row) not in independent,'unique_independent_evidence_'+iid+':'+str(row));independent[iid,row]=(a,b)
rows={(x['image_id'],x['row']):x for x in ledger['rows']};ck(len(rows)==len(ledger['rows'])==len(independent)==74 and set(rows)==set(independent),'74_complete_reviewed_row_decisions')
field_tokens={'xmin':{1,3},'xmax':{1,3},'ymin':{2,4},'ymax':{2,4},'left.x':{5},'left.y':{6},'right.x':{8},'right.y':{9}}
groups=Counter()
for key,x in rows.items():
 iid,row=key;a,b=x['before_line'],x['after_line'];at,bt=a.split(),b.split();changed=[j for j in range(11) if at[j]!=bt[j]];allowed=set().union(*(field_tokens[f] for f in x['changed_fields'])) if x['changed_fields'] else set()
 ck((a,b)==independent[key] and a==parent_bytes[iid].decode().splitlines()[row-1] and b==saved_bytes[iid].decode().splitlines()[row-1],'ledger_matches_evidence_and_saved_'+str(key))
 ck(changed==x['changed_token_indices_0based'] and set(changed)<=allowed,'declared_tokens_only_'+str(key))
 bg,ag=geom(a),geom(b);ck(close(bg,x['before_pixels']) and close(ag,x['after_pixels']),'ledger_original_pixel_geometry_'+str(key))
 actual=[name for name,u,v in zip(['xmin','ymin','xmax','ymax'],bg['box_xyxy'],ag['box_xyxy']) if abs(u-v)>1e-6]
 for side,index in [('left',0),('right',1)]:
  for axis,j in [('x',0),('y',1)]:
   if abs(bg['keypoints'][index][j]-ag['keypoints'][index][j])>1e-6:actual.append(side+'.'+axis)
 ck(actual==x['changed_fields'],'declared_semantic_fields_'+str(key))
 actualgroups=[name for name,sl in [('box',slice(1,5)),('left',slice(5,7)),('right',slice(8,10))] if at[sl]!=bt[sl]];ck(actualgroups==x['changed_groups'],'declared_groups_'+str(key));groups.update(actualgroups)
 ck((x['action']=='replace')==(key in opmap) and (a!=b)==(key in opmap),'ledger_action_'+str(key))
 ck(x['source_image_sha256']==current[iid]['source_image_sha256'] and x['source_label_sha256']==basefiles[iid]['label_sha256'],'ledger_source_hashes_'+str(key))
 if key in opmap:ck(opmap[key]['before_line']==a and opmap[key]['after_line']==b and opmap[key]['changed_fields']==actual,'operation_ledger_equal_'+str(key))
ck(dict(groups)=={'box':16,'left':6,'right':3},'16_box_6_left_3_right_field_groups',dict(groups))
expected_named={('new_000007',24),('new_000007',41),('new_000008',33),('new_000009',49)}
ck({(x['image_id'],x['row']) for x in coverage['requested_rows']}=={(x['image_id'],x['row']) for x in ledger['named_rows']}==expected_named and all(rows[key]['scope']=='user_named' and rows[key]['split']=='val' for key in expected_named),'four_named_cases_covered_as_development')
standards=ledger['preserved_standards'];ck(len(standards)==11,'eleven_protected_standards_inventory')
for x in standards:
 iid,row=x['image_id'],x['row'];ck(parent_bytes[iid].decode().splitlines()[row-1]==saved_bytes[iid].decode().splitlines()[row-1]==x['line'],'protected_standard_exact_'+iid+':'+str(row))
# Original and frozen reference files are checked by content hashes and raw counts.
frozen={}
for role,expected in [('primary',666),('diagnostic',663)]:
 d=m['frozen_references'][role];path=R/d['path'];ref=read(path);ck(sha(path)==d['sha256'],'frozen_manifest_hash_'+role);ct=Counter(images=0,objects=0,visible_endpoints=0,missing_endpoints=0,empty_images=0)
 for f in ref['files']:
  p=path.parent/f['label_path'];lines=p.read_text().splitlines();ck(sha(p)==f['label_sha256'],'frozen_label_hash_'+role+':'+f['image_id']);ct['images']+=1;ct['objects']+=len(lines);ct['empty_images']+=not lines
  for line in lines:
   t=line.split()
   for k in [7,10]:ct['visible_endpoints' if int(t[k])==2 else 'missing_endpoints']+=1
 ck(dict(ct)==ref['counts'] and ct['objects']==expected and ct['images']==20,'frozen_reference_counts_'+role,dict(ct));frozen[role]={'manifest_sha256':sha(path),'counts':dict(ct),'label_files_hash_verified':len(ref['files'])}
protected_paths=['data/labels','data/references','data/manifest.json','data/reviews/fronts-2026-11-23-r26','metadata/model_card.json','scripts/pose_scoring.py','scripts/stage_pose_dataset.py','scripts/pose_checkpoint.py','scripts/evaluate_pose.py']
gitdiff=subprocess.check_output(['git','diff','--name-only','HEAD','--',*protected_paths],cwd=R,text=True).splitlines();ck(not gitdiff,'baseline_protected_paths_git_unchanged',gitdiff)
newtraining=subprocess.check_output(['git','status','--short','--','data/candidates','notebooks'],cwd=R,text=True).splitlines();ck(not newtraining,'no_new_candidate_or_notebook_files',newtraining)
ck(m['review_only'] and not m['full_dataset_certified'] and not summary['training_run'] and not summary['candidate_created'] and not summary['notebook_created'],'scope_review_only_not_training')
categories=defaultdict(Counter)
prefixes=('unique_operation_','split_and_label_path_','R26_label_hash_','R27_label_hash_','original_image_hash_','original_label_hash_','unchanged_row_count_','operation_before_','independent_exact_byte_replay_','changed_file_flag_','identity_row_map_','eleven_tokens_','class_visibility_exact_','normalized_geometry_valid_','unmodified_row_exact_','unknown_triplet_exact_','visible_corner_x_order_','unique_independent_evidence_','ledger_matches_evidence_and_saved_','declared_tokens_only_','ledger_original_pixel_geometry_','declared_semantic_fields_','declared_groups_','ledger_action_','ledger_source_hashes_','operation_ledger_equal_','protected_standard_exact_','frozen_label_hash_')
for c in checks:
 category=next((p.rstrip('_') for p in prefixes if c['check'].startswith(p)),c['check']);categories[category]['passed' if c['passed'] else 'failed']+=1
receipt={'schema_version':1,'kind':'independent_read_only_r27_snapshot_audit','status':'pass' if not failures else 'fail','method':'Independent raw-byte replay, source hashes, own numeric geometry parsing and declared-field comparison. Does not import or invoke the R27 builder or its validators. Source files were read only.','manifest':{'path':str(mp.relative_to(R)),'sha256':sha(mp)},'ledger':{'path':str(ld.relative_to(R)),'sha256':sha(ld)},'audit_script_path':str(SCRIPT.relative_to(R)) if SCRIPT.is_relative_to(R) else str(SCRIPT),'audit_script_sha256':sha(SCRIPT),'summary':{'checks_run':len(checks),'failed_checks':len(failures),'image_files_hash_verified':157,'original_label_files_hash_verified':157,'parent_label_files_hash_verified':157,'R27_label_files_hash_verified':157,'exact_replayed_files':157,'replacement_rows':21,'replacement_rows_by_split':{'val':14,'train':7},'changed_files':len(changed_files),'unchanged_files':157-len(changed_files),'total_current_rows':rows_total,'unmodified_rows_byte_identical':unmodified_rows,'zero_endpoint_triplets_preserved':zero_triplets,'reviewed_rows':74,'retained_reviewed_rows':53,'named_cases':4,'protected_standards':11,'field_groups':dict(groups)},'review_counts':{k:dict(v) for k,v in stats.items()},'frozen_references':frozen,'failures':failures,'checks_by_category':{k:dict(v) for k,v in categories.items()},'limitations':['This audit checks provenance, replay and annotation-token invariants. It does not certify every original pixel boundary or all157 photos visually.','Current R27 labels are a review snapshot; no new training/model promotion is implied.']}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'summary':receipt['summary'],'failures':failures,'receipt':str(OUT)},indent=2))
