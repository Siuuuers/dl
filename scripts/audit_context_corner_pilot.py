#!/usr/bin/env python3
"""Replay the fixed pilot audit with independent arithmetic and targeted mathematical checks.

Requires Python 3.12 and NumPy. Reads source files; writes only the requested new JSON.
The six original PNGs must be restored, not Git LFS pointers. The cohort hash is fixed.
"""

import argparse
import copy, hashlib, importlib.util, json, math, struct
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description="Independently audit the frozen contextual-corner pilot.")
parser.add_argument("--pilot-root", required=True, type=Path,
                    help="Checkout containing scripts/context_corner_pilot.py and the pilot result directory")
parser.add_argument("--repo-root", required=True, type=Path,
                    help="Checkout containing the frozen corrected labels and all six actual original PNGs")
parser.add_argument("--output", required=True, type=Path, help="New audit JSON; existing files are refused")
args = parser.parse_args()
root = args.pilot_root.resolve()
base = root / "results/review/context-corners-20261112"
repo = args.repo_root.resolve()
script = root / "scripts/context_corner_pilot.py"
output = args.output.resolve()
if output.exists():
    parser.error("Output already exists; choose a new audit path")
digest = lambda b: hashlib.sha256(b).hexdigest()
cohort_bytes = (base/'cohort.json').read_bytes()
cohort = json.loads(cohort_bytes)
eval_bytes = (base/'evaluation.json').read_bytes()
ev = json.loads(eval_bytes)
script_bytes = script.read_bytes()
assert digest(cohort_bytes) == '67fe072944cc7dbe60f2be20b0ddb6c87029864cb7c0c81601a201c72c80196c'
assert ev['provenance']['cohort']['sha256'] == digest(cohort_bytes)
assert ev['provenance']['script_sha256'] == digest(script_bytes), 'Evaluation does not bind current script'
manifest_bytes = (repo/cohort['reference_manifest']).read_bytes()
assert digest(manifest_bytes) == cohort['reference_manifest_sha256']
manifest = json.loads(manifest_bytes)
entries = {x['image_id']: x for x in manifest['files']}
loaded = {}
checks = 0
max_coordinate_difference = 0.
def close(a,b,where):
    global checks
    checks += 1
    if isinstance(a,dict):
        assert set(a)==set(b),(where,'keys',set(a)^set(b))
        for k in a: close(a[k],b[k],where+'/'+str(k))
    elif isinstance(a,list):
        assert len(a)==len(b),(where,'length')
        for i,(x,y) in enumerate(zip(a,b)): close(x,y,where+'/'+str(i))
    elif isinstance(a,float):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(where,a,b)
    else:
        assert a==b,(where,a,b)

for identity in ev['provenance']['source_files']:
    key = identity['image_id']; ent=entries[key]
    raw=(repo/identity['image_path']).read_bytes()
    assert digest(raw)==identity['image_sha256']==ent['image_sha256']
    assert len(raw)==identity['image_bytes']
    assert raw[:8]==b'\x89PNG\r\n\x1a\n' and raw[12:16]==b'IHDR'
    width,height=struct.unpack('>II',raw[16:24])
    assert [width,height]==identity['dimensions']==[ent['width'],ent['height']]
    labels=(repo/identity['label_path']).read_bytes()
    assert digest(labels)==identity['label_sha256']==ent['label_sha256']
    assert len(labels)==identity['label_bytes']
    rows={}
    for number,line in enumerate(labels.decode().splitlines(),1):
        fields=list(map(float,line.split()))
        assert len(fields)==11 and fields[0]==0 and fields[7] in [0,2] and fields[10] in [0,2]
        rows[number]={'xy':np.array([[fields[5]*width,fields[6]*height],[fields[8]*width,fields[9]*height]]),
                      'visible':np.array([fields[7]==2,fields[10]==2])}
    loaded[key]={'rows':rows,'width':width,'height':height}

def project(h,points):
    q=np.column_stack((np.asarray(points),np.ones(len(points))))@h.T
    return q[:,:2]/q[:,2,None]
def independent_pair(method,first,last,t):
    if method=='linear':return np.asarray(first)*(1-t)+np.asarray(last)*t
    # Independent, direct 8x8 solution with h33 fixed to 1; no DLT/SVD or pilot imports.
    src=[(0.,0.),(1.,0.),(1.,1.),(0.,1.)]
    dst=[first[0],first[1],last[1],last[0]]
    design=[];target=[]
    for (x,y),(u,v) in zip(src,dst):
        design.append([x,y,1,0,0,0,-u*x,-u*y]);target.append(u)
        design.append([0,0,0,x,y,1,-v*x,-v*y]);target.append(v)
    h=np.append(np.linalg.solve(np.asarray(design),np.asarray(target)),1).reshape(3,3)
    return project(h,[(0.,t),(1.,t)])

stacks={x['id']:x for x in cohort['stacks']}
all_derived={}
donor_audit=[]
for lane in ['interpolation','extrapolation']:
    recorded=ev['lanes'][lane]['cases'];derived=[]
    expected_ids=[f"{s['image_id']}:{r}" for s in cohort['stacks']
                  for r in s['holdout_rows' if lane=='interpolation' else 'extrapolation_holdout_rows']]
    assert [x['case_id'] for x in recorded]==expected_ids
    for case in recorded:
        st=stacks[case['stack_id']];rows=loaded[case['image_id']]['rows']
        ranks={m['row']:m['rank'] for m in st['members']}
        excluded={case['row']} if lane=='interpolation' else set(st['extrapolation_holdout_rows'])
        assert case['excluded_rows']==sorted(excluded)
        eligible=sorted([m for m in st['members'] if m['row'] not in excluded and rows[m['row']]['visible'].all()],
                        key=lambda m:m['rank'])
        if lane=='interpolation':
            ds=[max((m for m in eligible if m['rank']<ranks[case['row']]),key=lambda m:m['rank']),
                min((m for m in eligible if m['rank']>ranks[case['row']]),key=lambda m:m['rank'])]
        else:
            first_rank=min(ranks[r] for r in excluded)
            ds=[m for m in eligible if m['rank']<first_rank][-2:]
        assert [d['row'] for d in case['donors']]==[m['row'] for m in ds]
        assert not(excluded & {m['row'] for m in ds})
        for rec,donor in zip(case['donors'],ds):
            close(rec['xy'],rows[donor['row']]['xy'].tolist(),case['case_id']+'/donor')
            assert rec['rank']==donor['rank']
        t=(ranks[case['row']]-ds[0]['rank'])/(ds[1]['rank']-ds[0]['rank'])
        assert case['rank_ratio']==t and t in ([.5] if lane=='interpolation' else [2.,3.])
        truth=rows[case['row']]['xy']
        close(case['target_xy_for_scoring_only'],truth.tolist(),case['case_id']+'/truth')
        assert case['target_bbox_used'] is False and case['target_endpoints_used_for_prediction'] is False
        d={'stack_id':st['id'],'methods':{}}
        for method in ['linear','homography']:
            pair=independent_pair(method,rows[ds[0]['row']]['xy'],rows[ds[1]['row']]['xy'],t)
            raw=np.asarray(case['methods'][method]['raw_pair_xy'])
            max_coordinate_difference=max(max_coordinate_difference,float(np.max(np.abs(pair-raw))))
            close(pair.tolist(),raw.tolist(),case['case_id']+'/'+method+'/xy')
            error=[math.hypot(*(p-q)) for p,q in zip(pair,truth)]
            close(error,case['methods'][method]['endpoint_errors_px'],case['case_id']+'/'+method+'/errors')
            close(float(max(error)),case['methods'][method]['maximum_endpoint_error_px'],case['case_id']+'/'+method+'/maxerror')
            assert case['methods'][method]['status']=='estimated'
            assert all(0<=p[0]<loaded[case['image_id']]['width'] and 0<=p[1]<loaded[case['image_id']]['height'] for p in pair)
            successes={str(radius):max(error)<=radius for radius in [5,10]}
            close(successes,case['methods'][method]['both_within_radius'],case['case_id']+'/'+method+'/successes')
            d['methods'][method]={'pair':pair,'errors':error}
        d['disagreement']=max(math.hypot(*(a-b)) for a,b in zip(d['methods']['linear']['pair'],d['methods']['homography']['pair']))
        derived.append(d)
        donor_audit.append({'lane':lane,'case_id':case['case_id'],'excluded_rows':sorted(excluded),
                            'donors':[m['row'] for m in ds],'ratio':t})
    all_derived[lane]=derived

def stats(values):
    values=sorted(values)
    def quantile(q):
        i=(len(values)-1)*q;l=math.floor(i);h=math.ceil(i)
        return float(values[l]+(values[h]-values[l])*(i-l))
    return {'count':len(values),'median_px':quantile(.5) if values else None,
            'p90_px':quantile(.9) if values else None,'maximum_px':float(max(values)) if values else None}
def summary(rows):
    ans={'target_pair_denominator':len(rows),'endpoint_denominator':2*len(rows),'methods':{}}
    for method in ['linear','homography']:
        errors=[e for r in rows for e in r['methods'][method]['errors']]
        maxima=[max(r['methods'][method]['errors']) for r in rows]
        ans['methods'][method]={'estimated_pairs':len(rows),'failed_pairs':0,'failure_reasons':{},
            'both_within_radius':{str(t):sum(e<=t for e in maxima) for t in [5,10]},
            'endpoints_within_radius':{str(t):sum(e<=t for e in errors) for t in [5,10]},
            'endpoint_error':stats(errors),'pair_maximum_error':stats(maxima),
            'error_statistic_basis':'All finite raw estimates, including rejected out-of-frame outputs; failures remain in success denominators.'}
    ans['maximum_pair_method_disagreement']=stats([r['disagreement'] for r in rows])
    return ans
for lane,rows in all_derived.items():
    close(summary(rows),ev['lanes'][lane]['summary'],lane+'/summary')
    for stack_id in stacks:
        close(summary([r for r in rows if r['stack_id']==stack_id]),ev['lanes'][lane]['per_stack'][stack_id],lane+'/'+stack_id)

hidden_checks=[]
for case in ev['hidden_cases']:
    assert case['accepted_for_training'] is False and case['accepted_as_observed_reference'] is False
    assert case['confidence'] is None and case['confidence_calibrated'] is False
    assert case['target_bbox_used'] is False and case['target_endpoints_used_for_prediction'] is False
    if case['disposition']=='abstain':
        assert case['status']=='abstained' and case['donors']==[] and case['methods']=={}
        continue
    assert case['image_id']=='new_000010' and case['row'] in [23,24]
    assert [x['row'] for x in case['donors']]==[21,22]
    t=case['row']-21
    assert case['rank_ratio']==t
    rows=loaded[case['image_id']]['rows']
    assert rows[case['row']]['visible'].tolist()==[False,True]
    methods={}
    for method in ['linear','homography']:
        xy=independent_pair(method,rows[21]['xy'],rows[22]['xy'],t)
        got=case['methods'][method]
        close(xy.tolist(),got['raw_pair_xy'],case['case_id']+'/'+method)
        assert len(got['proposals'])==1 and got['proposals'][0]['slot']=='left'
        proposal=got['proposals'][0]
        assert proposal['status']=='estimated_from_context' and proposal['accepted_for_training'] is False
        assert proposal['ground_truth_available'] is False and proposal['geometric_error_px'] is None
        close(xy[0].tolist(),proposal['xy'],'hidden_left_proposal')
        check=got['visible_endpoint_checks']
        assert len(check)==1 and check[0]['slot']=='right'
        assert check[0]['used_in_predictor'] is False and check[0]['certifies_hidden_side_accuracy'] is False
        err=math.hypot(*(xy[1]-rows[case['row']]['xy'][1]))
        close(float(err),check[0]['error_px'],'right_only_check')
        methods[method]={'left_estimate':xy[0].tolist(),'right_check_error_px':err,'hidden_left_error':None}
    hidden_checks.append({'case_id':case['case_id'],'donors':[21,22],'ratio':t,'methods':methods})

# Exercise only two mathematical properties of the implementation, not its general test suite.
spec=importlib.util.spec_from_file_location('context_pilot_under_audit',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
synthetic=[]
for name,h in [
 ('known_affine',np.array([[130.,-8.,400.],[10.,42.,200.],[0.,0.,1.]])),
 ('known_projective',np.array([[180.,3.,350.],[6.,30.,120.],[.015,.12,1.]]))]:
    first=project(h,[(0.,0.),(1.,0.)]);last=project(h,[(0.,1.),(1.,1.)])
    errors=[]
    for t in [.25,.5,.75,2.,3.]:
        known=project(h,[(0.,t),(1.,t)])
        pred=module.estimate_pair('homography',first,last,t,allow_extrapolation=t>1)
        err=float(np.max(np.abs(pred-known)))
        assert err<1e-9,(name,t,err)
        errors.append({'rank_ratio':t,'max_abs_coordinate_error_px':err})
    synthetic.append({'name':name,'checks':errors})
metamorphic_count=0
for lane,info in ev['lanes'].items():
    for rec in info['cases']:
        st=stacks[rec['stack_id']];data=copy.deepcopy(loaded[rec['image_id']])
        ranks={m['row']:m['rank'] for m in st['members']};excluded=set(rec['excluded_rows'])
        for r in excluded: data['rows'][r]['xy'][:]=np.array([[999999.,-888888.],[-777777.,666666.]])
        rerun=module.evaluate_holdout(st,data,rec['row'],ranks,excluded,lane=='extrapolation')
        assert rerun['donors']==rec['donors']
        for method in ['linear','homography']:
            assert rerun['methods'][method]['raw_pair_xy']==rec['methods'][method]['raw_pair_xy']
        metamorphic_count+=1
for original,rec in zip(cohort['hidden_cases'],ev['hidden_cases']):
    if original['disposition']=='abstain':continue
    changed=copy.deepcopy(loaded)
    changed[original['image_id']]['rows'][original['row']]['xy'][1]=[-54321.,12345.]
    rerun=module.evaluate_hidden(original,stacks,changed)
    assert rerun['donors']==rec['donors']
    for method in ['linear','homography']:
        assert rerun['methods'][method]['raw_pair_xy']==rec['methods'][method]['raw_pair_xy']
    metamorphic_count+=1
assert script.read_bytes()==script_bytes,'Script changed during audit'
assert (base/'evaluation.json').read_bytes()==eval_bytes,'Evaluation changed during audit'
result={'status':'verified','method':'Independent direct 8x8 projective solve, raw-label/image identity checks, independent Euclidean errors and linear quantiles; plus synthetic known-H and target-coordinate perturbation checks.',
 'audit_script_sha256':digest(Path(__file__).read_bytes()),
 'script_sha256':digest(script_bytes),'evaluation_sha256':digest(eval_bytes),'cohort_sha256':digest(cohort_bytes),
 'scalar_structure_checks':checks,'original_pngs_and_label_files_verified':len(loaded),
 'holdout_scenarios':sum(len(x['cases']) for x in ev['lanes'].values()),
 'distinct_holdout_objects':len({x['case_id'] for z in ev['lanes'].values() for x in z['cases']}),
 'maximum_independent_coordinate_difference_px':max_coordinate_difference,
 'all_68_method_estimates_recomputed':True,'all_summary_and_per_stack_metrics_recomputed':True,
 'metamorphic_target_perturbations':metamorphic_count,'synthetic_known_homographies':synthetic,
 'donor_checks':donor_audit,'hidden_cases':hidden_checks,'abstentions':3,
 'limits':['Coordinates withheld for the target, but neighbour labels and manual rank are supplied.',
           'No new visual certification of cohort membership or equal pitch/coplanarity.',
           'The two missing left corners have no measured ground-truth error.',
           'A right-endpoint check cannot certify the hidden left endpoint.',
           '34 overlapping scenarios cover 30 unique observed objects; not independent test scenes.']}
output.parent.mkdir(parents=True, exist_ok=True)
with output.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write('\n')
print(json.dumps({k:result[k] for k in ['status','script_sha256','evaluation_sha256','scalar_structure_checks','original_pngs_and_label_files_verified','holdout_scenarios','distinct_holdout_objects','maximum_independent_coordinate_difference_px','metamorphic_target_perturbations']},indent=2))
print('Audit file:',output)

