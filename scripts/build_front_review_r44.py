#!/usr/bin/env python3
"""Replay the complete R44 visual-audit ledger and build an offline frame viewer.

This CPU tool verifies provenance, exact recorded decisions and export gates.
It cannot establish physical truth from checksums, approve labels for a human,
train a model, or estimate performance. The pinned PR131 archive is preserved.
"""
import argparse
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import zipfile

import build_front_review_r43 as r43
import review_front_mining_r42 as legacy
from render_front_review_r44 import render_html

RETURN_PR='https://github.com/Siuuuers/dl/pull/131'
RETURN_COMMIT='7b6c312c492a65a216a678b855f62cb63ee2bdd4'
RETURN_SHA256='ae3d7310fd6b2d3a623bd7b8760dc9d6ffc75e6df1be28bbf0a2555fdb380d6e'
RETURN_BYTES=4293004
PREVIOUS_MANIFEST_SHA256='d40787277c0da7487ee8712da659be847ee9c972918b5cee420488dbafdc499f'
HISTORY='history/pr131-r43.zip'
TOLERANCE_PX=0.0001
require,digest,load_json=legacy.require,legacy.digest,legacy.load_json
json_bytes=r43.json_bytes
SCRIPTS=['build_front_review_r43.py','review_front_mining_r42.py',
         'render_front_review_r43.py','render_front_review_r44.py']
FIELDS={'xmin':[1,3],'xmax':[1,3],'ymin':[2,4],'ymax':[2,4],
        'left_point':[5,6],'right_point':[8,9]}


def geometry(line):
    a=legacy.pose(line,'audit row')
    return dict(xmin=(a[1]-a[3]/2)*1280,ymin=(a[2]-a[4]/2)*720,
                xmax=(a[1]+a[3]/2)*1280,ymax=(a[2]+a[4]/2)*720,
                left_point=[a[5]*1280,a[6]*720],right_point=[a[8]*1280,a[9]*720])


def changed_fields(before,after):
    a,b=geometry(before),geometry(after)
    return [k for k in FIELDS if (max(abs(x-y) for x,y in zip(a[k],b[k]))
            if isinstance(a[k],list) else abs(a[k]-b[k]))>TOLERANCE_PX]


def prior_archive(raw):
    require(len(raw)==RETURN_BYTES and digest(raw)==RETURN_SHA256,'PR131 archive identity')
    payload=legacy.archive_payload(io.BytesIO(raw))
    r43.verify_capsule(payload)
    require(digest(payload['manifest.json'])==PREVIOUS_MANIFEST_SHA256,'R43 manifest identity')
    r42,_,figures=r43.reconstruct_previous(payload)
    return load_json(payload['manifest.json']),r42,payload,figures


def verify_outlines(outlines,manifest):
    require(outlines.get('schema_version')==1 and outlines.get('training_mask') is False,'outline scope')
    items=outlines['items'];seen=set();rows=r43.all_rows(manifest)
    for item in items:
        sid=item['stable_id']
        require(sid in rows and rows[sid]['after_line'] is not None and sid not in seen,'outline identity')
        seen.add(sid)
        require(item['closed'] is True and len(item['vertices_px'])>=3,'closed outline vertices')
        require(all(isinstance(p,list) and len(p)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in p)
                    and 0<=p[0]<1280 and 0<=p[1]<720 for p in item['vertices_px']),'outline coordinate bounds')
        require(isinstance(item['reason'],str) and item['reason'].strip(),'outline reason')
        require(type(item['nominal_uncertainty_px']) in (int,float) and
                math.isfinite(item['nominal_uncertainty_px']) and item['nominal_uncertainty_px']>0,'outline uncertainty')


def apply_audit(old,audit,outlines):
    require(audit['schema_version']==1 and audit['round']==44,'audit schema/round')
    require(audit['previous_manifest_sha256']==PREVIOUS_MANIFEST_SHA256,'audit predecessor')
    require(audit['previous_return']==dict(pr=RETURN_PR,commit=RETURN_COMMIT,sha256=RETURN_SHA256,bytes=RETURN_BYTES),'audit return')
    require(audit['human_review_performed'] is False and audit['all_current_rows_inspected'] is True,'audit review scope')
    old_rows=r43.all_rows(old);current={k:v for k,v in old_rows.items() if v['after_line'] is not None}
    decisions=audit['rows'];ids=[r['stable_id'] for r in decisions]
    require(len(ids)==507 and len(set(ids))==len(ids) and set(ids)==set(current),'all 507 current rows must be covered exactly once')
    require(audit['full_originals_inspected']==[f['image_id'] for f in old['frames']],'full original inspection coverage/order')
    result=deepcopy(old);new_rows=r43.all_rows(result);frame_map={f['image_id']:f for f in result['frames']}
    latest=dict(reviewed_rows=507,amended_rows=0,retained_rows=0,changed_tokens=0,
                corner_coordinate_slots_changed=0,box_boundaries_changed=0,
                added_rows=0,deleted_rows=0,visibility_changes=0,hold_changes=0,
                newly_held_rows=0,newly_held_fields=0,resolved_fields=0)
    for decision in decisions:
        sid=decision['stable_id'];before=current[sid];row=new_rows[sid];fid=sid.split(':')[0]
        require(decision['image_id']==fid and decision['source_row']==before['source_row'],'audit identity '+sid)
        require(decision['source_image_sha256']==frame_map[fid]['source_image_sha256'],'audit raster '+sid)
        require(decision['inspected'] is True and decision['all_current_fields_inspected'] is True,'audit inspection '+sid)
        require(decision['before_line']==before['after_line'],'audit before text '+sid)
        after=decision['after_line'];require(isinstance(after,str),'R44 cannot add/delete '+sid)
        a,b=legacy.pose(before['after_line'],sid),legacy.pose(after,sid)
        delta=r43.changed(before['after_line'],after)
        require(all(before['after_line'].split()[k]==after.split()[k] for k in [0,7,10]),'class/visibility changed '+sid)
        actual_fields=changed_fields(before['after_line'],after)
        require(decision['changed_fields']==actual_fields,'changed field declaration '+sid)
        require(decision['changed_token_indices']==delta,'changed token declaration '+sid)
        allowed={k for field in actual_fields for k in FIELDS[field]}
        require(set(delta)<=allowed,'unrequested token changed '+sid)
        for k in [5,8]:
            if a[k+2]==0:require(a[k:k+3]==b[k:k+3]==[0,0,0],'missing corner changed '+sid)
        require(isinstance(decision['reason'],str) and decision['reason'].strip(),'audit reason '+sid)
        holds=decision['unresolved_fields'];why=decision['unresolved_reasons']
        require(isinstance(holds,list) and len(set(holds))==len(holds) and all(isinstance(v,str) and v for v in holds),'hold fields '+sid)
        require(isinstance(why,list) and all(isinstance(v,str) and v for v in why) and bool(holds)==bool(why),'hold reasons '+sid)
        require(set(before['unresolved_fields'])<=set(holds) and set(before['unresolved_reasons'])<=set(why),'R44 cannot silently clear existing holds '+sid)
        require(decision['resolved_fields']==[],'unexpected hold clearance '+sid)
        new_holds=set(holds)-set(before['unresolved_fields'])
        require(not new_holds or any(v not in before['unresolved_reasons'] for v in why),'new hold without new reason '+sid)
        if before.get('latest_update'):
            row.setdefault('review_history',[]).append({'round':43,**deepcopy(before['latest_update'])})
        row['after_line']=after
        row['action']='add' if row['source_row'] is None else ('retain' if after==row['before_line'] else 'replace')
        row['changed_token_indices']=r43.changed(row['before_line'],after)
        row['reason']=before['reason']+'\nR44: '+decision['reason']
        row['unresolved_fields']=deepcopy(holds);row['unresolved_reasons']=deepcopy(why)
        row['nominal_visual_uncertainty_px']=deepcopy(decision['nominal_visual_uncertainty_px'])
        row['latest_update']=dict(status='amended' if delta else ('held' if holds else 'retained'),
            reason=decision['reason'],requested_fields=['front_ownership','box_bounds','left_point','right_point'],
            changed_fields=actual_fields,changed_token_indices=delta,
            newly_unresolved_fields=sorted(new_holds),all_current_fields_inspected=True)
        latest['amended_rows']+=bool(delta);latest['retained_rows']+=not delta;latest['changed_tokens']+=len(delta)
        latest['corner_coordinate_slots_changed']+=sum(a[k:k+2]!=b[k:k+2] for k in [5,8])
        latest['box_boundaries_changed']+=sum(k in actual_fields for k in ['xmin','ymin','xmax','ymax'])
        latest['hold_changes']+=holds!=before['unresolved_fields'] or why!=before['unresolved_reasons']
        latest['newly_held_rows']+=not before['unresolved_fields'] and bool(holds)
        latest['newly_held_fields']+=len(new_holds)
    result.update(kind='r44-complete-current-front-visual-audit',round=44,
        return_pr=RETURN_PR,return_commit=RETURN_COMMIT,return_archive_sha256=RETURN_SHA256,return_archive_bytes=RETURN_BYTES,
        previous_manifest_sha256=PREVIOUS_MANIFEST_SHA256,audit_sha256=digest(json_bytes(audit)),
        outlines_sha256=digest(json_bytes(outlines)),latest_counts=latest,
        latest_review_scope='Fresh visual inspection of all 12 originals and all 507 current fronts. All 508 stable identities, including the prior deletion, remain in history. Clear corner/rack changes are separate from unresolved ownership holds. This is assistant review, not human approval or training.')
    result.pop('amendments_sha256',None)
    result['previous_round']=dict(round=43,manifest_sha256=PREVIOUS_MANIFEST_SHA256,latest_counts=old['latest_counts'])
    result['policy_notes'].append('R44 revisits every current front. Color is evidence with contour continuity and geometry, not a global material classifier. No previous-ymax or white-band-only top cropping is applied. New ownership uncertainty can revoke a prior whole-image export gate. Illustrative traced outlines are not training segmentation masks.')
    complete=[];held=[]
    for frame in result['frames']:
        frame['unresolved_stable_ids']=[r['stable_id'] for r in frame['rows']+frame['additions'] if r['unresolved_fields']]
        frame['annotation_complete']=not frame['unresolved_stable_ids']
        frame['eligible_for_candidate_export']=frame['annotation_complete']
        frame['disposition']='assistant_review_complete' if frame['annotation_complete'] else 'held'
        frame['latest_reviewed_stable_ids']=[r['stable_id'] for r in frame['rows']+frame['additions'] if r['after_line'] is not None]
        frame['full_image_inspected']=frame['all_baseline_rows_inspected']=True
        frame['all_current_rows_inspected_round']=44
        (complete if frame['annotation_complete'] else held).append(frame['image_id'])
    result['annotation_complete_ids']=complete;result['held_image_ids']=held
    r43.recount(result);verify_outlines(outlines,result)
    require(set(new_rows)==set(old_rows),'stable identities changed')
    require(all(new_rows[k]==v for k,v in old_rows.items() if v['after_line'] is None),'historical deleted identity changed')
    return result


def verification_for(manifest,old_payload):
    result=legacy.verify(manifest,r43.original_source(old_payload,manifest))
    result.update(round=44,latest_counts=manifest['latest_counts'],
        previous_return_pr=RETURN_PR,previous_return_sha256=RETURN_SHA256,
        prior_r43_archive_members_verified=73,all_507_current_rows_in_audit=True,
        all_508_stable_identities_preserved=True,prior_holds_preserved=True,
        visibility_and_missing_corner_triplets_preserved=True,
        historical_prediction_figures=5,inference_performed=False)
    result['scope']='Pinned PR131/R43 archive and full R42 history; exact 507-row audit replay; original raster hashes; immutable identities and visibility; strict geometry; current whole-image gates; canonical viewer. Physical judgments remain assistant visual proposals.'
    return result


def verify_capsule(payload):
    for member,path in [('review.py',Path(__file__))]+[(name,Path(__file__).with_name(name)) for name in SCRIPTS]:
        require(payload[member]==path.read_bytes(),'executing code differs: '+member)
    old,r42,old_payload,figures=prior_archive(payload[HISTORY])
    audit=load_json(payload['audit-ledger.json']);outlines=load_json(payload['outlines.json'])
    require(payload['audit-ledger.json']==json_bytes(audit) and payload['outlines.json']==json_bytes(outlines),'canonical decision serialization')
    expected=apply_audit(old,audit,outlines)
    require(payload['manifest.json']==json_bytes(expected),'manifest differs from exact audit replay')
    proposals={'proposals/'+f['image_id']+'.txt':legacy.candidate_bytes(f) for f in expected['frames']}
    candidates={'complete-candidates/labels/'+f['image_id']+'.txt':legacy.candidate_bytes(f) for f in expected['frames'] if f['eligible_for_candidate_export']}
    for prefix,members in [('proposals/',proposals),('complete-candidates/labels/',candidates)]:
        require({k for k in payload if k.startswith(prefix)}==set(members),'export member set: '+prefix)
        require(all(payload[k]==v for k,v in members.items()),'export bytes: '+prefix)
    require(payload['review.html']==render_html(expected,old,r42,old_payload,figures,outlines),'canonical R44 viewer')
    result=verification_for(expected,old_payload)
    require(payload['verification.json']==json_bytes(result),'stored verification differs from replay')
    return result


def build(return_path,audit_path,outlines_path,output,manifest_output=None,html_output=None):
    raw=return_path.read_bytes();old,r42,prior,figures=prior_archive(raw)
    audit=load_json(audit_path.read_bytes());outlines=load_json(outlines_path.read_bytes())
    manifest=apply_audit(old,audit,outlines)
    payload={HISTORY:raw,'audit-ledger.json':json_bytes(audit),'outlines.json':json_bytes(outlines),
        'manifest.json':json_bytes(manifest),'review.py':Path(__file__).read_bytes()}
    for name in SCRIPTS:payload[name]=Path(__file__).with_name(name).read_bytes()
    for frame in manifest['frames']:
        data=legacy.candidate_bytes(frame);fid=frame['image_id']
        payload['proposals/'+fid+'.txt']=data
        if frame['eligible_for_candidate_export']:payload['complete-candidates/labels/'+fid+'.txt']=data
    payload['complete-candidates/README.txt']=(
        'These are whole-image assistant-reviewed annotation candidates, not a training split or human sign-off.\n'
        'Only frames with no unresolved fields are exported here. Do not train on proposal-only files from held frames.\n'
        'The original training dataset is unchanged. Keep complete image annotations together and decide capture-group splits before a controlled training comparison.\n').encode()
    payload['README.txt']=(
        'Open review.html. All row cards for one photograph are on the same page; Next/Previous changes photographs.\n'
        'All 507 current fronts were inspected; all 508 stable identities remain visible, including the prior deletion.\n'
        'Choose R42 or R43 history beside the latest R44 annotations. Unchanged rows remain present.\n'
        'A manually traced outline is provided for frame_0160 source002 as an explanatory review aid, not a segmentation training mask.\n'
        'history/pr131-r43.zip preserves the exact returned R43 package, including original photographs and complete R42 evidence.\n'
        'The five saved prediction figures remain R42 historical comparisons. No updated scores, inference, model training or promotion are claimed.\n'
        'A new uncertainty can revoke a former complete-image export gate; consult manifest.json.\n'
        'Offline verification: python review.py --check-capsule PATH_TO_THIS_ZIP\n').encode()
    payload['review.html']=render_html(manifest,old,r42,prior,figures,outlines)
    result=verification_for(manifest,prior);payload['verification.json']=json_bytes(result)
    payload['inventory.json']=json_bytes({name:dict(sha256=digest(data),bytes=len(data)) for name,data in sorted(payload.items())})
    verify_capsule(payload)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(2026,11,26,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    verify_capsule(legacy.archive_payload(output))
    for path,data in [(manifest_output,payload['manifest.json']),(html_output,payload['review.html'])]:
        if path is not None:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    return dict(verification=result,archive_sha256=digest(output.read_bytes()),archive_bytes=output.stat().st_size,
                members=len(payload),manifest_sha256=digest(payload['manifest.json']),html_sha256=digest(payload['review.html']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ['return-archive','audit-ledger','outlines','output','manifest-output','html-output','check-capsule']:
        parser.add_argument('--'+option,type=Path)
    args=parser.parse_args()
    if args.check_capsule:result=verify_capsule(legacy.archive_payload(args.check_capsule))
    else:
        require(args.return_archive and args.audit_ledger and args.outlines and args.output,'supply return archive, audit ledger, outlines and output')
        result=build(args.return_archive,args.audit_ledger,args.outlines,args.output,args.manifest_output,args.html_output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
