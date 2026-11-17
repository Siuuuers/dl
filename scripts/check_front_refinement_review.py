#!/usr/bin/env python3
"""Independent standard-library replay of the r2 reviewed-label ledger.

Does not import or call project composers, label parsers, or renderers. Writes only
its requested report; input dataset bytes are never modified.
"""
from pathlib import Path
import argparse, collections, hashlib, json, math, sys

EXPECTED_CANDIDATE='de3cd6e5605ec7bd1250b53b4eb52c3a688e2431a19f4016bbfa08f2e09b6849'
EXPECTED_REVIEW='8408aa6bd84590bcb5ed56de290528ce76c8a74bec32f899199a633a205a14f6'
EXPECTED_PARENT='fb52d6946e598e977289ca025ac386c5044e9f15e9461b6e55f7c00a6b9c9171'
EXPECTED_REFERENCE='0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def js(path): return json.loads(path.read_text())
def raw_sha(raw): return hashlib.sha256(raw).hexdigest()
def identity(entry):
    return ('original',entry['original_row']) if entry['original_row'] is not None else ('addition',entry['addition_id'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--candidate',default='data/candidates/sequence-refined-fronts-2026-11-18-r2/manifest.json')
    p.add_argument('--report',default='results/review/front-refinement-20261118-r2/verification.json')
    args=p.parse_args();root=args.root.resolve();errors=[];check_count=0
    def check(ok,code,**details):
        nonlocal check_count
        check_count+=1
        if not ok: errors.append({'check':code,**details})
    def bound(desc):
        path=root/desc['path'];check(path.is_file(),'descriptor_file_exists',path=str(path));check(sha(path)==desc['sha256'],'descriptor_sha256',path=desc['path']);return path
    def labels(raw,iid):
        result=[]
        for n,line in enumerate(raw.decode('utf-8').splitlines(),1):
            t=line.split();check(len(t)==11,'eleven_fields',image_id=iid,row=n)
            if len(t)!=11:continue
            a=list(map(float,t));check(all(math.isfinite(x) for x in a),'finite_values',image_id=iid,row=n)
            check(a[0]==0,'class_zero',image_id=iid,row=n)
            cx,cy,w,h=a[1:5];x1,y1,x2,y2=cx-w/2,cy-h/2,cx+w/2,cy+h/2
            check(w>0 and h>0 and min(x1,y1)>=-1e-6 and max(x2,y2)<=1+1e-6,'positive_in_image_box',image_id=iid,row=n)
            for i in [5,8]:
                x,y,v=a[i:i+3];check(v in [0,2],'visibility_zero_or_two',image_id=iid,row=n,endpoint=i)
                if v==0:check(x==0 and y==0,'canonical_missing_point',image_id=iid,row=n,endpoint=i)
                else:check(x1-1e-5<=x<=x2+1e-5 and y1-1e-5<=y<=y2+1e-5,'visible_point_in_box',image_id=iid,row=n,endpoint=i)
            if a[7]==2 and a[10]==2:check(a[5]<=a[8]+1e-6,'left_right_point_order',image_id=iid,row=n)
            result.append(a)
        return result
    def counts(items):
        return {'images':len(items),'objects':sum(len(x) for x in items),'visible_endpoints':sum(int(r[7]==2)+int(r[10]==2) for x in items for r in x),'missing_endpoints':sum(int(r[7]==0)+int(r[10]==0) for x in items for r in x),'empty_images':sum(not x for x in items)}

    candidate_path=root/args.candidate;candidate=js(candidate_path)
    check(sha(candidate_path)==EXPECTED_CANDIDATE,'frozen_candidate_hash')
    parent_path=bound(candidate['parent']);parent=js(parent_path)
    check(sha(parent_path)==EXPECTED_PARENT,'frozen_parent_hash')
    review_path=bound(candidate['review']);review=js(review_path)
    check(sha(review_path)==EXPECTED_REVIEW,'frozen_review_hash')
    status_path=bound(candidate['review_status']);status=js(status_path)
    source_path=root/'data/manifest.json';source=js(source_path)
    check(sha(source_path)==candidate['source_manifest_sha256'],'original_manifest_hash')
    reference_path=bound(candidate['development_reference']);reference=js(reference_path)
    check(sha(reference_path)==EXPECTED_REFERENCE,'frozen_development_reference_hash')
    bound(candidate['annotation_policy'])
    for desc in candidate['review_sources']:bound(desc)
    for desc in review['review_sources']:bound(desc)
    check(candidate['review_sources']==review['review_sources'],'review_source_descriptors_match')
    for key in ['parent','parent_status','development']:
        bound(review['composition'][key])
    check(review['composition']['parent']==candidate['parent'],'review_candidate_parent_matches')
    check(review['composition']['development']==candidate['development_reference'],'review_candidate_development_matches')

    originals={r['id']:r for r in source['records']};parent_files={f['image_id']:f for f in parent['files']};out_files={f['image_id']:f for f in candidate['files']}
    training_ids={i for i,r in originals.items() if r['split']=='train'};validation_ids=set(originals)-training_ids
    check(len(originals)==157 and len(training_ids)==137 and len(validation_ids)==20,'original_inventory_157')
    check(set(out_files)==set(parent_files)==training_ids,'training_inventory_exact')
    original_counts={'train':[],'val':[]};original_hashes=[]
    for iid,record in originals.items():
        image=root/'data'/record['image'];label=root/'data'/record['label']
        check(sha(image)==record['image_sha256'],'original_image_hash',image_id=iid)
        check(sha(label)==record['label_sha256'],'original_label_hash',image_id=iid)
        parsed=labels(label.read_bytes(),iid+' original');original_counts[record['split']].append(parsed)
        check(len(parsed)==record['instances'],'original_instance_count',image_id=iid)
        original_hashes.append([iid,record['image_sha256'],record['label_sha256']])
    original_counts={split:counts(items) for split,items in original_counts.items()}

    operations=review['operations'];grouped=collections.defaultdict(dict);operation_ids=set();opcounts=collections.Counter()
    for op in operations:
        key=op['base_row'];iid=op['image_id'];action=op['operation'];opcounts[action]+=1
        check(iid in training_ids and op['split']=='train','operation_train_only',operation_id=op['operation_id'])
        check(op['operation_id'] not in operation_ids,'operation_id_unique',operation_id=op['operation_id']);operation_ids.add(op['operation_id'])
        check(key not in grouped[iid],'one_operation_per_parent_row',image_id=iid,row=key);grouped[iid][key]=op
        check(action in ['replace','delete'],'no_additions_or_unsupported_operations',operation_id=op['operation_id'])
        if action=='replace':
            bt,at=op['before_line'].split(),op['after_line'].split()
            check(len(bt)==len(at)==11,'replacement_fields')
            check(bt[0]==at[0] and bt[5:]==at[5:],'class_and_keypoint_tokens_unchanged',image_id=iid,row=key)
            check(bt[1:5]!=at[1:5],'bbox_actually_changes',image_id=iid,row=key)
        else:check(op.get('after_line') is None,'deletion_has_no_after_line',image_id=iid,row=key)
    check(dict(opcounts)=={'replace':387,'delete':1},'expected_operation_counts',actual=dict(opcounts))
    check(candidate['operation_counts']==dict(opcounts),'manifest_operation_counts')
    deletes=[o for o in operations if o['operation']=='delete']
    check([(o['image_id'],o['base_row']) for o in deletes]==[('new_000049',18)],'only_authorized_deletion')

    output_rows={};parent_rows={};expected_row_maps={};changed=[];file_summary=[];unchanged_rows=0;replayed_files=0;mapping_entries=0;preserved_replacement_points=0
    for iid in sorted(training_ids):
        prior=parent_files[iid];item=out_files[iid];original=originals[iid]
        pp=parent_path.parent/prior['candidate_label'];op=candidate_path.parent/item['candidate_label'];before=pp.read_bytes();actual=op.read_bytes()
        check(raw_sha(before)==prior['candidate_label_sha256'],'parent_label_hash',image_id=iid)
        check(bound(item['base_label']).resolve()==pp.resolve(),'base_label_is_parent',image_id=iid)
        check(item['source_label']==original['label'] and item['source_label_sha256']==original['label_sha256'] and item['source_image_sha256']==original['image_sha256'],'output_source_provenance',image_id=iid)
        check(raw_sha(actual)==item['candidate_label_sha256'],'output_label_hash',image_id=iid)
        physical=before.splitlines(keepends=True);replay=[];map_by_parent={};row_counter=0
        for n,line in enumerate(physical,1):
            action=grouped[iid].get(n)
            if action:
                check(line.rstrip(b'\r\n')==action['before_line'].encode(),'exact_before_line',image_id=iid,row=n)
                if action['operation']=='delete':map_by_parent[n]=None;continue
                replay.append(action['after_line'].encode()+b'\n');preserved_replacement_points+=1
            else:replay.append(line);unchanged_rows+=1
            row_counter+=1;map_by_parent[n]=row_counter
        check(all(1<=n<=len(physical) for n in grouped[iid]),'operation_rows_in_parent',image_id=iid)
        rebuilt=b''.join(replay);check(rebuilt==actual,'independent_byte_reconstruction',image_id=iid);replayed_files+=1
        parsed=labels(actual,iid+' output');baseline=labels(before,iid+' parent');output_rows[iid]=parsed;parent_rows[iid]=baseline
        check(item['candidate_objects']==len(parsed) and item['base_objects']==len(physical) and item['source_objects']==original['instances'],'per_file_counts',image_id=iid)
        if actual!=before:changed.append(iid)
        check(item['changed_from_parent']==(actual!=before),'changed_from_parent_flag',image_id=iid)
        if not grouped[iid]:check(actual==before,'untouched_file_byte_identical',image_id=iid)
        prior_map={identity(e):e for e in prior['row_map']};new_map={identity(e):e for e in item['row_map']}
        check(len(prior_map)==len(prior['row_map']) and len(new_map)==len(item['row_map']),'stable_identity_unique',image_id=iid)
        check(set(prior_map)==set(new_map),'stable_identities_preserved_including_tombstones',image_id=iid)
        check({e['original_row'] for e in item['row_map'] if e['original_row'] is not None}==set(range(1,original['instances']+1)),'all_original_identities_accounted',image_id=iid)
        expected_row_maps[iid]={}
        for ident,old in prior_map.items():
            new=new_map[ident];old_current=old['candidate_row'];expected_current=map_by_parent.get(old_current) if old_current is not None else None
            check(new['candidate_row']==expected_current,'stable_identity_current_row',image_id=iid,identity=list(ident),expected=expected_current,actual=new['candidate_row'])
            check(new['parent_row']==old_current and new['base_row']==old_current,'parent_and_base_row_mapping',image_id=iid,identity=list(ident))
            expected_row_maps[iid][ident]=expected_current;mapping_entries+=1
        check(sorted(e['candidate_row'] for e in item['row_map'] if e['candidate_row'] is not None)==list(range(1,len(parsed)+1)),'complete_bijective_current_row_map',image_id=iid)
        file_summary.append({'image_id':iid,'parent_objects':len(baseline),'output_objects':len(parsed),'replacements':sum(o['operation']=='replace' for o in grouped[iid].values()),'deletions':sum(o['operation']=='delete' for o in grouped[iid].values()),'output_label_sha256':raw_sha(actual),'unchanged_file':actual==before})
    check(len(changed)==28 and candidate['changed_images_from_parent']==changed,'changed_inventory_28',actual=changed)
    check(preserved_replacement_points==387 and unchanged_rows==3410,'all_retained_rows_partitioned',replacements=preserved_replacement_points,unchanged=unchanged_rows)
    check(len(candidate['operations'])==len(operations),'manifest_operation_log_length')
    manifest_ops={o['operation_id']:o for o in candidate['operations']}
    for approved in operations:
        logged=manifest_ops[approved['operation_id']];iid=approved['image_id'];br=approved['base_row']
        check(all(logged.get(k)==v for k,v in approved.items()),'logged_operation_preserves_approval',operation_id=approved['operation_id'])
        identity_for_row=next(identity(e) for e in parent_files[iid]['row_map'] if e['candidate_row']==br)
        check(logged['output_row']==expected_row_maps[iid][identity_for_row],'operation_output_row',operation_id=approved['operation_id'])
        check(logged['source_image_sha256']==originals[iid]['image_sha256'],'operation_image_hash',operation_id=approved['operation_id'])
        check(logged['base_label_sha256']==parent_files[iid]['candidate_label_sha256'],'operation_parent_label_hash',operation_id=approved['operation_id'])
    result_counts={'train':counts(list(output_rows.values())),'val':original_counts['val']};baseline_counts={'train':counts(list(parent_rows.values())),'val':original_counts['val']}
    check(result_counts==candidate['expected_counts']==review['expected_counts'],'computed_counts_match_review_and_manifest',actual=result_counts)
    check(baseline_counts==candidate['baseline_counts']==parent['expected_counts'],'computed_parent_counts')
    check(result_counts['train']=={'images':137,'objects':3797,'visible_endpoints':6860,'missing_endpoints':734,'empty_images':13},'expected_training_counts')

    # Status reconciliation uses approved case decisions and stable source identities,
    # independently from the newly written status's current-row numbers.
    inherited=js(bound(review['composition']['parent_status']));expected_cases={c['case_id']:dict(c) for c in inherited['cases']}
    for d in review['case_decisions']:
        check(d['case_id'] in expected_cases,'decision_has_inherited_case',case_id=d['case_id'])
        c=expected_cases[d['case_id']];c['status']=d['status']
        for k in ['held_original_rows','held_addition_ids']:
            if k in d:c[k]=d[k]
    written_cases={c['case_id']:c for c in status['cases']}
    check(set(written_cases)==set(expected_cases),'all_status_case_identities_preserved')
    pending=[]
    for cid,expected in expected_cases.items():
        actual=written_cases[cid];iid=expected['image_id'];held=[]
        for key in expected.get('held_original_rows',[]):held.append(expected_row_maps[iid].get(('original',key)))
        for key in expected.get('held_addition_ids',[]):held.append(expected_row_maps[iid].get(('addition',key)))
        mapped=sorted({r for r in held if r is not None})
        check(actual['status']==expected['status'],'approved_case_status',case_id=cid)
        check(actual.get('held_original_rows',[])==expected.get('held_original_rows',[]) and actual.get('held_addition_ids',[])==expected.get('held_addition_ids',[]),'approved_held_identities',case_id=cid)
        check(actual['held_current_rows']==mapped,'independent_held_current_rows',case_id=cid,expected=mapped,actual=actual['held_current_rows'])
        if expected['status']!='resolved':
            check(all(r is not None for r in held),'pending_case_has_live_rows',case_id=cid)
            pending.append({'case_id':cid,'image_id':iid,'current_rows':mapped})
    check(len(pending)==10 and len({c['image_id'] for c in pending})==9,'ten_training_groups_nine_photos')
    check(status['remaining_all_case_groups']==candidate['remaining_case_groups']==len(pending),'manifest_status_hold_group_counts')
    check(status['remaining_all_training_images']==candidate['remaining_training_images']==len({c['image_id'] for c in pending}),'manifest_status_hold_image_counts')
    check(written_cases['new_000049-064']['status']=='resolved' and written_cases['new_000049-064']['held_current_rows']==[],'deleted_04918_hold_resolved')

    # Development stays a separately pinned, unchanged 20-photo reference.
    reference_files={f['image_id']:f for f in reference['files']};check(set(reference_files)==validation_ids,'development_inventory_twenty')
    ref_parsed={};ref_rowmap_entries=0
    for iid,f in reference_files.items():
        raw=(reference_path.parent/f['label_path']).read_bytes();check(raw_sha(raw)==f['label_sha256'],'development_label_hash',image_id=iid)
        check(f['source_label_sha256']==originals[iid]['label_sha256'] and f['image_sha256']==originals[iid]['image_sha256'],'development_original_provenance',image_id=iid)
        rr=labels(raw,iid+' development');ref_parsed[iid]=rr;check(len(rr)==f['row_count'],'development_row_count',image_id=iid)
        maps=f['row_map'];check(sorted(e['candidate_row'] for e in maps if e['candidate_row'] is not None)==list(range(1,len(rr)+1)),'development_row_map_complete',image_id=iid)
        originals_mapped={e['original_row']:e['candidate_row'] for e in maps if e['original_row'] is not None}
        check([originals_mapped.get(n) for n in range(1,f['source_row_count']+1)]==f['retained_source_to_revised_row_one_based'],'development_original_row_mapping',image_id=iid)
        check(sorted(e['candidate_row'] for e in maps if e['original_row'] is None and e['candidate_row'] is not None)==f['added_revised_rows_one_based'],'development_added_row_mapping',image_id=iid)
        ref_rowmap_entries+=len(maps)
    check(counts(list(ref_parsed.values()))==reference['counts'],'development_computed_counts')
    check(reference['remaining_rows_are_authoritative'] is True,'development_authoritative_current_holds')
    devholds=reference['remaining_current_rows'];check(len(devholds)==8 and len({x['image_id'] for x in devholds})==7,'eight_development_rows_seven_photos')
    for h in devholds:check(h['image_id'] in reference_files and 1<=h['row']<=len(ref_parsed[h['image_id']]),'development_held_row_exists',image_id=h['image_id'],row=h['row'])
    for ev in candidate['evidence_images']:bound(ev)

    report={
        'schema_version':1,'kind':'independent_r2_byte_reconstruction','result':'pass' if not errors else 'fail','checks_executed':check_count,'errors':errors,
        'method':'Independent standard-library replay from exact parent row bytes and accepted operations. No composer, project geometry parser, or renderer imported/called. Original hashes, row provenance and held-row identities independently recomputed.',
        'inputs':{'candidate':{'path':args.candidate,'sha256':sha(candidate_path)},'parent':candidate['parent'],'accepted_review':candidate['review'],'status':candidate['review_status'],'development_reference':candidate['development_reference'],'original_manifest_sha256':sha(source_path)},
        'original_source_preservation':{'images_verified':157,'labels_verified':157,'image_and_label_inventory_sha256':raw_sha(json.dumps(sorted(original_hashes),separators=(',',':')).encode()),'all_match_original_manifest':not any(e['check'].startswith('original_') for e in errors)},
        'training_reconstruction':{'files_replayed':replayed_files,'unchanged_files':137-len(changed),'changed_images':changed,'operations':dict(opcounts),'unmodified_retained_rows_byte_identical':unchanged_rows,'replacement_class_and_keypoint_tokens_unchanged':preserved_replacement_points,'row_map_entries_verified':mapping_entries,'counts':result_counts['train'],'deleted_identity':{'image_id':'new_000049','parent_row':18,'original_row':18,'current_row':None,'original_19_now_current':expected_row_maps['new_000049'][('original',19)],'original_26_now_current':expected_row_maps['new_000049'][('original',26)],'prior_added_parent_27_now_current':expected_row_maps['new_000049'][('addition','sequence-20261117-sequence-aperture-007')]}},
        'held_training':{'groups':len(pending),'photos':len({x['image_id'] for x in pending}),'distinct_current_rows':len({(x['image_id'],r) for x in pending for r in x['current_rows']}),'cases':pending},
        'frozen_original_validation':result_counts['val'],
        'development_reference':{'counts':counts(list(ref_parsed.values())),'row_maps_verified':20,'row_map_entries_verified':ref_rowmap_entries,'held_rows':devholds,'held_row_count':len(devholds),'held_photo_count':len({x['image_id'] for x in devholds})},
        'changed_file_details':[f for f in file_summary if not f['unchanged_file']],
        'scope_limit':'This QA verifies faithful composition, bytes, geometric validity and provenance. It does not independently certify physical annotation correctness or model accuracy.',
        'verification_script':{'path':str(Path(__file__).resolve().relative_to(root)),'sha256':sha(Path(__file__))},
        'reproduce':'python scripts/check_front_refinement_review.py --root . --report results/review/front-refinement-20261118-r2/verification.json'
    }
    rp=root/args.report;rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'result':report['result'],'checks':check_count,'errors':errors,'report':str(rp),'report_sha256':sha(rp),'training':result_counts['train'],'operations':dict(opcounts)},indent=2))
    return 0 if not errors else 1

if __name__=='__main__':sys.exit(main())
