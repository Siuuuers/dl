#!/usr/bin/env python3
"""Consolidate documented unresolved cases; never change labels or infer closure."""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path,required=True,help='New or empty output directory')
args=parser.parse_args()
ROOT=args.root.resolve()
OUT=args.output.resolve()
VIS=Path('results/review/label-refinement-20261114/visual')
if OUT.exists() and any(OUT.iterdir()):
    parser.error('--output must be new or empty; existing review records are preserved')
OUT.mkdir(parents=True,exist_ok=True)
sources={}

def read(path):
    p=Path(path)
    if not p.is_absolute():p=ROOT/p
    b=p.read_bytes(); key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    sources[key]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
    return json.loads(b)

frozen=read('data/manifest.json')
records={x['id']:x for x in frozen['records']}
labels={i:(ROOT/'data'/x['label']).read_text().splitlines() for i,x in records.items()}
cases=[];resolved=[];validation=[]

def bounds(image,row):
    a=list(map(float,labels[image][row-1].split()));w,h=records[image]['original_size']
    return [(a[1]-a[3]/2)*w,(a[2]-a[4]/2)*h,(a[1]+a[3]/2)*w,(a[2]+a[4]/2)*h]

def add(image,rows,issue,known,needed,source,pointer,kind='identity_or_extent',region=None,
        status='open',details=None,target=None):
    assert image in records
    rows=sorted(set(rows or []))
    assert all(isinstance(r,int) and 1<=r<=len(labels[image]) for r in rows),(image,rows)
    boxes=[bounds(image,r) for r in rows]
    if region is None and boxes:
        w,h=records[image]['original_size']
        region=[max(0,math.floor(min(b[0] for b in boxes)-20)),
                max(0,math.floor(min(b[1] for b in boxes)-20)),
                min(w,math.ceil(max(b[2] for b in boxes)+20)),
                min(h,math.ceil(max(b[3] for b in boxes)+20))]
        region_basis='Suggested review crop: union of original row boxes plus 20px; not an accepted target rectangle.'
    elif region is None:
        w,h=records[image]['original_size'];region=[0,0,w,h]
        region_basis='Whole-frame scope; the source does not record exact pixel crop bounds.'
    else:region_basis='Region explicitly recorded in review, or a described full-width border region; not an accepted target rectangle.'
    entry={'case_id':f'{image}-{len(cases)+len(resolved)+len(validation)+1:03d}',
           'image_id':image,'split':records[image]['split'],'original_rows_1based':rows,
           'region_xyxy_px':region,'region_basis':region_basis,'kind':kind,'status':status,
           'uncertainty':issue,'already_known':known,'resolution_needed':needed,
           'evidence':[{'path':str(source),'selector':pointer}]}
    if details is not None:entry['details']=details
    if target is None:target=cases if status.startswith('open') else resolved
    target.append(entry);return entry

POLICY='One identifiable physical front per row; front-facing visible extent; actual lower-front endpoints independently observed or canonical 0 0 0. Missing slots, overlap, thinness, or failure to find a model proposal are not deletion/restoration evidence.'

# Preserve every explicit unresolved field in the early complete row ledgers.
p='data/candidates/box-refinement-2026-11-07/review.json';d=read(p)
accepted={(x['image_id'],x['original_row'],x['edge']) for x in d['root_adjudication']}
for j,review in enumerate(d['primary_reviews']):
    image=review['image_id'];details=[]
    for ri,row in enumerate(review['rows']):
        n=row['row']; fields=[]
        ident=row.get('identity',row.get('identity_decision',{}))
        if 'unresolved' in ident.get('decision',''):
            fields.append({'field':'identity','reason':ident['reason']})
        for edge,ev in row.get('box_edges',row.get('box_edge_decisions',{})).items():
            if 'unresolved' in ev.get('decision','') and (image,n,edge) not in accepted:
                fields.append({'field':'bbox.'+edge,'reason':ev['reason']})
        ends=row.get('endpoints',row.get('endpoint_decisions',{}))
        items=ends.items() if isinstance(ends,dict) else [(e.get('slot','unknown'),e) for e in ends]
        for slot,ev in items:
            if 'unresolved' in ev.get('decision',''):
                fields.append({'field':'endpoint.'+slot,'reason':ev['reason']})
        if fields:
            details.append({'original_row':n,'source_bbox_xyxy_px':row['source_bbox_xyxy_px'],
                            'unresolved_fields':fields,'source_selector':f'$.primary_reviews[{j}].rows[{ri}]'})
    add(image,[r['original_row'] for r in details],
        'Explicit unresolved legacy fields remain after the 16 accepted single-edge corrections. See per-row details.',
        'All original rows were inspected; supported changes were accepted independently. Retaining other fields did not certify them. Identity rows: '+str(d['unresolved_identity_rows'].get(image,[]))+'.',
        'Resolve physical identity first for the listed identity rows; then adjudicate the named observed bounds/points. Preserve all unrelated tokens and canonical missing slots.',
        p,f'$.primary_reviews[{j}].rows','legacy_fields',details=details)

early='docs/training-pilot-2026-11-06.json';read(early)
resolved.append({'case_id':'early128-inclusion-policy','image_id':'new_000128','split':'train',
    'original_rows_1based':[1,12],'status':'policy_question_resolved_identity_still_open',
    'already_known':'The historical general inclusion-rule question is settled by the visible-front policy: an identifiable positive front may have neither corner locatable. Whether these particular thin strips are separate fronts remains open in the box-pilot ledger.',
    'evidence':[{'path':early,'selector':'$.primary_ordinary_case_decisions[2]'},
                {'path':p,'selector':'$.unresolved_identity_rows.new_000128'}]})

# The broad annotation pass: only explicitly supported later decisions supersede it.
p='data/candidates/annotations-2026-11-10/review.json';d=read(p)
for group,items in d['unresolved_and_retained'].items():
    for idx,x in enumerate(items):
        images=x.get('images',[x.get('image_id',x.get('image'))])
        for image in images:
            rows=x.get('source_rows_1based',[])
            issue=x.get('issue',x.get('reason',''))
            if image=='new_000079' and group=='crosscheck':
                add(image,[15,16,34],issue,'Later visible-extent candidate accepts the three identity-to-box remaps, preserving existing endpoint tokens.',
                    'No remaining hold for those three accepted remaps; row14/context and any lower sliver remain separately recorded.',p,f'$.unresolved_and_retained.{group}[{idx}]',status='superseded_accepted_geometry')
                continue
            if image=='new_000082' and group=='crosscheck':
                add(image,[26],issue,'Later visible-extent candidate refits row26 to the fourth front and appends the separately observed fifth front.',
                    'Those exact issues are superseded; do not infer a sixth front below the accepted region.',p,f'$.unresolved_and_retained.{group}[{idx}]',status='superseded_accepted_geometry_and_addition')
                continue
            if group=='occlusion_review' and idx==2:
                rows={'new_000094':[29,30,31,32],'new_000095':[33,34,35,36,37,38],
                      'new_000096':[39,40,41],'new_000097':[36,37,38,39,40]}[image]
                if image=='new_000094':
                    add(image,rows,issue,'Four later visible-extent replacements directly address these rows.',
                        'No remaining hold for the accepted four replacements; lower identity rows23/24/28 remain open.',p,f'$.unresolved_and_retained.{group}[{idx}]',status='superseded_accepted_geometry')
                    continue
            known=x.get('action',x.get('decision','Source row or unlabeled region retained; no accepted correction.'))
            if image=='new_000061' and rows==[9,10,14,15]:
                known+=' Later one-bound repairs address rows9/14/15; row10 remains explicit. Those selective repairs do not certify every other edge.'
            if image in ('new_000062','new_000063'):
                known+=' A later visible-extent pass explicitly retained the same unresolved opening extents.'
            region=[151,694,301,720] if image=='new_000065' and not rows else None
            status='open_scope_not_localized' if not rows and region is None else 'open'
            add(image,rows,issue,known,
                'Inspect the complete original and then the indicated region/rows; identify distinct fronts and observed limits before any deletion, split, addition or edge change. If the source did not localize rows, first record the affected row IDs.',
                p,f'$.unresolved_and_retained.{group}[{idx}]',region=region,status=status)

# Later visible-extent findings augment, rather than erase, the preceding cases.
p='data/candidates/visible-extent-2026-11-11/review.json';d=read(p)
for idx,x in enumerate(d['unresolved_and_retained']):
    image=x['image_id'];rows=x['source_rows_1based']
    add(image,rows,x['reason'],'Later targeted extent review still retained this specific concern.',
        'Resolve observed continuation through the jaw or border and distinguish a further front from upper/neighbor material. Context rows are not asserted to be wrong merely because they were outside the edit.',
        p,f'$.unresolved_and_retained[{idx}]',kind=x['disposition'])

# Partial-front review: none of these unresolved records is changed by its additions.
p='data/candidates/partial-fronts-2026-11-11/review.json';d=read(p)
for idx,x in enumerate(d['unresolved_and_retained']):
    add(x['image_id'],x.get('source_rows_1based',x.get('source_rows_context',[])),
        x.get('issue',x.get('observation','')),
        x.get('reason_no_edit','The exact region was reviewed and retained without this proposed change.'),
        'Settle physical-instance and face-versus-upper-material identity from original pixels, then bound only observed front material; do not transfer identity or points from neighboring views.',
        p,f'$.unresolved_and_retained[{idx}]',region=x.get('region_original_xyxy'))

# Consistency pass includes nine held complete boxes and separate identity/point concerns.
p='results/review/baomai-training-consistency-20261111/adjudication.json';d=read(p);u=d['unresolved_and_retained']
for idx,x in enumerate(u['held_complete_bbox_replacements']):
    add(x['image_id'],[x['original_row']],x['reason'],
        'Some individual width defects are supported; the full four-bound proposal was not accepted.',
        'Adjudicate the disputed face/roof transition or slanted seam. A supported single-edge repair may be reviewed separately without certifying all four bounds.',
        p,f'$.unresolved_and_retained.held_complete_bbox_replacements[{idx}]','bbox',
        details={'historical_unaccepted_proposal_xyxy':x['initial_proposal_not_adopted']})
identity_notes={'new_000018':'Row27 is a bottom-frame roof/front fragment.',
 'new_000034':'Row15 is a top-border identity/point association; rows48–50 cannot be reliably mapped to three distinct faces in the jaw opening.',
 'new_000049':'Row18 is roughly four pixels tall without a secure separate seam; row21 may be front or lip; row6 behind the gripper remains uncertain.'}
for image,rows in u['identity_source_rows_retained'].items():
    add(image,rows,identity_notes[image],'Source rows retained; overlapping boxes and missing points do not settle identity.',
        'Determine physical front ownership and count from original pixels before any row deletion, split or reassignment.',p,'$.unresolved_and_retained.identity_source_rows_retained.'+image,'identity')
for idx,x in enumerate(u['whole_image_regions']):
    add(x['image_id'],[],x['decision'],'No distinct additional front was accepted in that region.',
        'Resolve separate front versus lip/roof/structure before adding any row; keep unlocatable endpoints missing.',
        p,f'$.unresolved_and_retained.whole_image_regions[{idx}]','possible_omission',x['region_xyxy'])
for idx,x in enumerate(u['point_localization_concerns']):
    add(x['image_id'],x['rows'],x['note'],'Any accepted box edit preserves the old endpoint tokens and is not endpoint certification.',
        'Independently identify each actual lower-front terminal junction; retain current tokens until a supported precise correction is adjudicated.',
        p,f'$.unresolved_and_retained.point_localization_concerns[{idx}]','point_localization')

# Border follow-up preserves the remaining pilot cases.
p='results/review/border-fronts-20261113/adjudication.json';read(p)
for image,rows,issue,region in [
 ('new_000027',[],'Farther-right thin top-border fragments were not adjudicated as separate fronts.',[317,0,1280,45]),
 ('new_000001',[42,43],'The six-image pilot flags visible bounds as unresolved; no exact replacement is recorded in the border follow-up.',None),
 ('new_000013',[34],'The six-image pilot flags visible bounds as unresolved; no exact replacement is recorded in the border follow-up.',None)]:
    add(image,rows,issue,'The accepted new027 top-left additions and old010 top-bound edit do not resolve these different regions.',
        'Inspect each physical front and propose only supported observed bounds. Preserve original row identity and endpoint semantics.',p,'$.held_cases','bbox_or_possible_omission',region)

# Remaining57, reviewer1: named holds plus a documented lower-fragment extent concern.
p='results/review/remaining-training-20261114/reviewer-1.json';d=read(p)
for idx,x in enumerate(d['held_cases']):
    image=x['image_id'];rows=x.get('source_row');rows=rows if isinstance(rows,list) else ([] if rows is None else [rows])
    if image=='new_000033':continue  # exact slot resolved in the fresh review below
    if image=='new_000014':
        issue='Rows21/22 retain inherited physical-front identity/visible-extent uncertainty in the jaw opening. Endpoint occlusion is now resolved separately.'
        known='Fresh review supports keeping both endpoints missing in both rows; it does not newly certify their boxes or physical-instance mapping.'
    else:issue=x['reason'];known='No row addition/deletion or endpoint restoration was supported by this pass.'
    add(image,rows,issue,known,'Resolve the indicated physical identity or observed boundary using original pixels; do not infer hidden corners or targets from spacing.',
        p,f'$.held_cases[{idx}]',region=x.get('region_original_pixels'))
for idx,x in enumerate(d['records']):
    image=x['image_id']
    if image.startswith('old_') and image in ['old_000013','old_000069','old_000079','old_000093','old_000101']:
        add(image,[1],x['label_comparison_findings'],
            'The initial ymin-to-zero idea was retracted; excluded pixels may belong to a preceding projecting lip.',
            'Establish continuous same-front surface and its observed top before changing ymin. Do not copy the old010 decision by analogy.',p,f'$.records[{idx}].label_comparison_findings','bbox_top_ownership')
    if image=='new_000003':
        add(image,list(range(53,58))+list(range(59,63)),
            'Existing bottom-fragment bounds include occluder pixels; the whole-frame pass supplied no exact replacements.',
            'These fragments have source rows and missing corners remain missing; no supported omission or restoration was identified.',
            'Review visible extents row by row; do not automatically shrink disconnected visible portions to one fragment.',p,f'$.records[{idx}].label_comparison_findings','bbox')

# Remaining57 reviewer2 explicit held findings, excluding fully superseded new038.
p='results/review/remaining-training-20261114/reviewer-2.json';d=read(p)
for ii,x in enumerate(d['images']):
    image=x['id']
    for fi,f in enumerate(x['findings']):
        if not f['status'].startswith('held') or image=='new_000038':continue
        add(image,f.get('source_rows',[]),f['reason'],
            'No exact correction was accepted; original tokens remain. This is not permission to delete a thin/occluded positive.',
            'Settle the named identity or physical-corner junction using the complete original and detailed crop before any edit.',
            p,f'$.images[{ii}].findings[{fi}]',f['kind'],f.get('region_xyxy',f.get('crop_xyxy')))
    if image in ['new_000042','old_000054','old_000068']:
        add(image,[],x['disposition'],
            'The specifically described lower-right/top region did not support an added row; its target identity remains unaccepted.',
            'Localize the described region on the complete original and distinguish front from roof/lip or fixture before adding anything.',
            p,f'$.images[{ii}].disposition','possible_omission',
            [0,0,1280,100] if image.startswith('old_') else None)
    if image=='new_000005':
        add(image,[32,33,34,35,42,43,44,45,48,49,50],
            'Broad boxes around occluders were not fully reannotated in the targeted whole-frame pass.',
            'Observed missing sides were retained due clipping or occlusion; no supported omitted front was found.',
            'Adjudicate exact observed extent if migrating this image fully; do not treat the missing slots themselves as errors.',
            p,f'$.images[{ii}].disposition','extent_review_gap',status='open_review_gap')

# Remaining57 reviewer3 distinguishes an actual unresolved identity from a supported missing slot.
p='results/review/remaining-training-20261114/reviewer-3.json';d=read(p)
for ii,x in enumerate(d['images']):
    image=x['id']
    for hi,h in enumerate(x.get('held',[])):
        rows=h.get('rows',[])
        resolved_missing=(image=='new_000004' or (image=='new_000037' and rows))
        status='resolved_keep_missing' if resolved_missing else 'open'
        known='Missing-state retention is supported by actual clipping/occlusion; it does not certify unrelated box fields.' if resolved_missing else 'The targeted pass retained the source rows with this explicit issue.'
        needed='No endpoint restoration is supported; keep canonical missing triplets.' if resolved_missing else 'Adjudicate physical identity/observed extent or precise junction for the stated rows; no deletion or point restoration solely from overlap, small area, or missing slots.'
        if image=='new_000041' and rows==[9]:
            known='Fresh review again keeps both endpoints missing because the physical turns are occluded; separate front identity/extent remains open.'
        if image=='new_000006' and rows==[53,59]:
            needed='User/physical-cassette interpretation is requested for whether the small row59 strip is a separate cassette or the lip/part of row53. Preserve both rows pending the answer and a pixel-backed row map.'
        add(image,rows,h['issue'],known,needed,p,f'$.images[{ii}].held[{hi}]',
            'identity_or_extent',h.get('region'),status)

# The acceptance of precise new038 rows supersedes each historical hold explicitly.
np=VIS/'new_000038-accepted-adjudication.json';nd=read(np)
hp=VIS/'held-endpoint-review-status.json';hd=read(hp)
add('new_000038',[1],
    'Historical single-front A-versus-B interpretation, three-front repair hold, and bottom-clipped exact-box hold.',
    'Root and visual reviewer accepted a row1 box/left-point replacement and appended rows16–18: three thumb-adjacent fronts total plus the separate bottom-clipped positive. Ordinary recorded placement uncertainty remains.',
    'Apply only the exact accepted operation tokens through the verified composition; do not retain the superseded A-to-lowest-point proposal.',
    str(np),'$',status='superseded_accepted_incremental',
    details={'record_status':nd.get('status'),'accepted_operations_source':str(np)})
for image,rows,slots in [('new_000033',[46],['left']),('new_000014',[21,22],['left','right']),('new_000041',[9],['left','right'])]:
    add(image,rows,'Earlier missing-corner restoration question.',
        'Fresh original-image review finds the actual physical junction occluded; keeping the missing state is now a resolved annotation decision.',
        'Keep the named slots exactly 0 0 0. Any separate inherited identity/bbox question remains open, not resolved by this endpoint decision.',
        str(hp),'$',kind='endpoint_state',status='resolved_keep_missing',details={'slots':slots})

add('new_000015',[],
    'Three top-border rectangles initially proposed as omitted cassettes.',
    'The proposals were withdrawn after structural-identity dissent and comparison of complete originals: the apparent strips track rack-top backer/retainer and fastening posts.',
    'Do not add the withdrawn three rows. No new lower-front label follows from their precise historical rectangles.',
    'results/review/remaining-training-20261114/adjudication.json','$.held_or_withdrawn[0]',
    region=[0,0,1280,60],status='withdrawn_not_an_accepted_target')
read('results/review/remaining-training-20261114/adjudication.json')

# Keep validation issues separate from training blockers and original-row numbering explicit.
p='data/references/visible-front-2026-11-10/manifest.json';vd=read(p)
for idx,x in enumerate(vd['queue_dispositions']):
    if x['disposition']!='retained_unresolved':continue
    add(x['image_id'],[x['source_row_one_based']],' '.join(x['unresolved_issues']),x['reason'],
        'Review in the separately versioned development reference; do not put this validation image into training or silently rewrite historical scoring labels.',
        p,f'$.queue_dispositions[{idx}]','validation_identity_or_extent',target=validation)
add('new_000011',[61],
    'Pale central fragment has unresolved extent and relationship to neighboring row59/front-versus-lip structure.',
    'Both actual outer lower corners are unavailable; a model match did not settle identity or make the box tight.',
    'Resolve in a separately versioned evaluation reference; this is outside the training overlay.',
    'docs/focused-case-review-2026-11-11.md','case 011:61','validation_identity_or_extent',target=validation)
for row,issue in [(13,'The narrow pale sliver near the holder does not establish a reliably visible actual lower-left corner; no new restoration is supported.'),
                  (14,'Front/material identity around the holder aperture remains unresolved.'),
                  (30,'Bottom-clipped front keeps both endpoints missing; exact fragment box extent was not adjudicated.')]:
    add('new_000010',[row],issue,
        'The later six-left-point adjudication addresses rows25–28/31/32, not this separate case.',
        'Retain the specific disclosed limitation in the development reference; no training label change.',
        'results/review/visible-corners-20261112/review.json','$.retained_dispositions','validation_point_or_identity',target=validation)
read('results/review/visible-corners-20261112/review.json');read('results/review/corner-adjudication-20261112/review.json')

# Distinguish actual outstanding decisions from general precision/scope caveats.
# Keep the latter visible without turning them into human-decision blockers.
scope_limits=[]
scope_only={
 ('new_000005','extent_review_gap'),
 ('new_000079','unchanged_context'),('new_000082','unchanged_context'),
 ('new_000018','point_localization'),('new_000049','point_localization'),
}
retained=[]
for c in cases:
    key=(c['image_id'],c['kind'])
    general_held=c['image_id'] in ('new_000050','old_000022')
    if key in scope_only or general_held:
        c['status']='documented_scope_limit_not_a_separate_blocker'
        c['resolution_needed']='Preserve this recorded limit. Reopen only when a concrete physical-identity or localization discrepancy is identified; lack of recertification alone does not establish a defect.'
        scope_limits.append(c)
    else:retained.append(c)
cases=retained

# These later records explicitly revisit the same exact row groups. Combine the
# history rather than counting them twice as separate physical problems.
for image,rows in [('new_000062',[8,9]),('new_000063',[8,9]),('new_000094',[23,24,28])]:
    matches=[c for c in cases if c['image_id']==image and c['original_rows_1based']==rows]
    assert len(matches)==2
    earlier,later=matches
    later['evidence']=earlier['evidence']+later['evidence']
    later['earlier_documented_issue']=earlier['uncertainty']
    cases.remove(earlier)
for c in cases:
    if c['image_id']=='new_000061' and c['original_rows_1based']==[9,10,14,15]:
        c['status']='superseded_specific_width_repairs_row10_still_open'
        c['already_known']='Later accepted specific-bound repairs address rows9/14/15; the later explicit row10 case carries the continuing opening-extent uncertainty. Other fields are not newly certified.'
        resolved.append(c)
        cases.remove(c)
        break

extent_images={'new_000064','new_000067','new_000069','new_000071','new_000073','new_000074',
 'new_000075','new_000080','new_000081','new_000084','new_000085','new_000095','new_000096',
 'new_000097','new_000130','new_000016','new_000002'}
for c in cases:
    if c['kind']=='legacy_fields':
        category='legacy_extent_with_separate_identity_rows'
    elif c['kind'] in ('point_localization','uncertain_existing_points') or c['image_id']=='new_000044':
        category='physical_corner_junction'
    elif c['kind'] in ('bbox','bbox_top_ownership','unchanged_uncertain_extent'):
        category='observed_extent_or_surface_boundary'
    elif c['kind']=='bbox_or_possible_omission' and c['original_rows_1based']:
        category='legacy_visible_extent_tightness'
    elif c['image_id'] in extent_images and c['original_rows_1based']:
        category='legacy_visible_extent_tightness'
    elif c['status']=='open_scope_not_localized' and c['image_id'] in extent_images and c['image_id'] not in ('new_000095','new_000096'):
        category='legacy_extent_not_localized_to_rows'
    else:category='physical_instance_identity_or_possible_omission'
    c['decision_category']=category
    c['requires_user_answer']=c['image_id']=='new_000006' and c['original_rows_1based']==[53,59]

# Register text evidence without trying to decode it as JSON.
for path in ('docs/focused-case-review-2026-11-11.md','docs/training-annotations-2026-11-10.md',
             'docs/training-consistency-review-2026-11-11.md'):
    b=(ROOT/path).read_bytes();sources[path]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}

# Row12's original box was explicitly left unresolved while its donor corner was
# accepted. Preserve that known concern even though the decision list is accepted.
add('new_000067',[12],
    'The accepted aggregate-row consolidation transfers a supported left point while deliberately retaining row12\'s existing box; the broad inherited extent is not migrated.',
    'Row25 is a redundant aggregate and its donor point belongs to row12. The instance mapping/point transfer is settled, independently of this retained box extent.',
    'If refining the visible extent, adjudicate the same physical front\'s observed bounds without undoing the accepted donor-corner ownership.',
    'docs/training-annotations-2026-11-10.md','bounded-intervention paragraph: row12 in067',kind='legacy_bbox')
cases[-1]['decision_category']='legacy_visible_extent_tightness';cases[-1]['requires_user_answer']=False

latest128=VIS/'new_000128-fragment-review.json';read(latest128)
for c in cases:
    if c['image_id']=='new_000128' and c['kind']=='legacy_fields':
        c['evidence'].append({'path':str(latest128),'selector':'$'})
        c['latest_identity_recheck']='Fresh original-pixel review still holds rows1/12. Row1 is the roughly 5px top band; row12 is the image-right strip x1269.95–1280,y2.94–68.60, not a bottom fragment. Distinct identity versus row2 roof/lip or row10/rack material remains unresolved. No operation accepted; neighboring images supply context without coordinate transfer.'

# Coverage is membership in targeted passes, never all-row approval.
coverage=read('results/review/baomai-review-coverage-20261112/coverage.json')
selection=read('results/review/remaining-training-20261114/selection.json')
empty=read('results/review/baomai-empty-training-20261114/review.json')
prior=set(selection['prior_review_ids'])
remaining=set()
for group in selection['groups']:
    # The selected manifest stores concrete image records under each group.
    if isinstance(group,dict):
        vals=group.get('images',group.get('image_ids',[]))
        remaining.update(v if isinstance(v,str) else v.get('id',v.get('image_id')) for v in vals)
if not remaining:
    for n,key in [(1,'records'),(2,'images'),(3,'images')]:
        r=read(f'results/review/remaining-training-20261114/reviewer-{n}.json')
        remaining.update(x.get('id',x.get('image_id')) for x in r[key])
train={i for i,r in records.items() if r['split']=='train'}
assert len(prior)==80 and len(remaining)==57 and prior.isdisjoint(remaining)
assert prior|remaining==train and len(train)==137
assert all(c['split']=='train' for c in cases+resolved)
assert all(c['split']=='val' for c in validation)

by_image=[]
for image in sorted(train):
    open_ids=[c['case_id'] for c in cases if c['image_id']==image]
    done=[c['case_id'] for c in resolved if c['image_id']==image]
    limits=[c['case_id'] for c in scope_limits if c['image_id']==image]
    by_image.append({'image_id':image,'source_image_path':'data/'+records[image]['image'],
        'source_image_sha256':records[image]['image_sha256'],'source_label_path':'data/'+records[image]['label'],
        'source_label_sha256':records[image]['label_sha256'],'original_rows':len(labels[image]),
        'targeted_review_membership':True,'fully_certified':False,
        'open_case_ids':open_ids,'resolved_or_superseded_case_ids':done,'scope_limit_case_ids':limits,
        'absence_of_open_case_means':'No explicit remaining issue extracted from this documented corpus; not all-row correctness certification.'})

result={'schema_version':1,'kind':'documented-unresolved-training-review-ledger',
 'date':'2026-11-14','status':'review_needed','source_repository':'Siuuuers/dl',
 'scope':'Consolidation of explicit existing review findings and fresh accepted new038/occlusion decisions. No new pixel adjudication or label edits. Complete for the enumerated documentary sources, not a proof that every possible defect is known.',
 'contract':POLICY,'row_numbering':'One-based original frozen source rows unless explicitly named append rows. Suggested crop regions are review aids, not accepted annotation boxes.',
 'coverage':{'training_images':137,'targeted_review_membership':137,'membership_before_remaining_pass':80,
             'remaining_pass':57,'full_annotation_certification':False},
 'counts':{'open_case_records':len(cases),'training_images_with_open_case_records':len({c['image_id'] for c in cases}),
           'resolved_or_superseded_records':len(resolved),'documented_scope_limit_records':len(scope_limits),'validation_only_records':len(validation),
           'early_box_pilot_rows_with_explicit_unresolved_fields':sum(len(c.get('details',[])) for c in cases if c['kind']=='legacy_fields')},
 'training_open_cases':cases,'resolved_or_superseded_training_cases':resolved,
 'documented_scope_limits_not_separate_blockers':scope_limits,
 'decision_groups':{category:{'case_records':sum(c['decision_category']==category for c in cases),
    'image_ids':sorted({c['image_id'] for c in cases if c['decision_category']==category})}
    for category in sorted({c['decision_category'] for c in cases})},
 'validation_only':{'training_blocker':False,'original_nineteen_dispositions_preserved':19,
     'cases':validation,'note':'Evaluation-reference uncertainty remains relevant to interpreting results; it is not a training-image omission. Later accepted new010 corner decisions are not reopened merely because ordinary manual uncertainty/dissent exists.'},
 'per_training_image_index':by_image,'source_files':sources,
 'separate_out_of_scope':[
  'The additional-frame pilot frame_0001/frame_0160/frame_0383 and its 36 unresolved boundaries are outside the frozen 137 training IDs; they are quarantined supplementary material, not blockers mislabeled as existing training rows.',
  'A model failing a promotion rule does not revoke independently accepted annotation corrections. Composition must still preserve their exact provenance and tokens.',
  'General unreviewed precision/padding is not silently converted into a known defect. No-case images remain uncertified, and resolved missing slots are not counted as erroneous omissions.'
 ]}
(OUT/'ledger.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result['counts'],indent=2))
