#!/usr/bin/env python3
"""Compare the supplied detector with the trained detector and fixed corner average."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
from audit_reflection_return import (verify_archive, load_references, encoded, digest, require, valid_point)
from run_corner_average_r58 import average_predictions, report, state
from evaluate_pose import INFERENCE, parse_pose_labels, _prediction_records
from pose_scoring import score_image, aggregate_scores
from predict_pose_folder import postclip_nms
from run_flip_inference_r56 import inverse_horizontal_predictions

ORIGINAL_SHA = 'e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e'
TRAINED_SHA = '222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9'
ORIGINAL_MANIFEST_SHA = '72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926'
ARCHIVE_COMMIT = '0d8f7c6c8a48a4d63ce6641bbed841a663c172e9'
ORIGINAL_ARCHIVE_PATH = 'results/evaluation/baomai-annotations-seed46-20261111/evaluation.zip'
ORIGINAL_ARCHIVE_SHA = '3dfe3415841b47a605e2fd4b9e490b2c12b348b0014c6c2a8c78c06f323ffc1a'
ORIGINAL_ARCHIVE_BYTES = 1364789
REFLECTION_URL = 'https://raw.githubusercontent.com/Siuuuers/dl/dfe5392109433fcee51f41716710001bb123a3d5/results/a100/baomai-flip-inference-r56-a100-001/baomai-flip-inference-r56-results.zip'
REFLECTION_SHA = '88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114'
RESULT_NAME = 'baomai-original-comparison-r59-results.zip'
VIEWS = ('original', 'trained', 'reflected')
PROTOCOL = {'box_confidence_strictly_greater': .25, 'endpoint_confidence_inclusive': .5,
    'matching_iou_inclusive': .5, 'native_nms_iou': .7, 'postclip_suppression_iou_strictly_greater': .7,
    'image_size': 1280, 'precision': 'float32', 'max_det': 300,
    'association_iou_inclusive': .5, 'original_view_coordinate_weight': .5,
    'reflected_view_coordinate_weight': .5, 'fallback': 'unchanged trained original-view record',
    'primary_reference': 'r50', 'sensitivity_reference': 'frozen666',
    'comparison': 'Supplied detector weights versus trained detector plus equal corner averaging, with common processing.',
    'scores': 'Original-view detector scores retained for comparison; not calibrated for averaged coordinates.',
    'development_only': True, 'training': False, 'automatic_promotion': False}


def checked(path, sha, size=None):
    b=Path(path).read_bytes()
    require(digest(b)==sha and (size is None or len(b)==size), 'Input differs: '+str(path))
    return b


def fetch(path, url, sha, size=None):
    path=Path(path)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=180) as response:b=response.read()
        require(digest(b)==sha and (size is None or len(b)==size), 'Downloaded bytes differ: '+path.name)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return checked(path,sha,size)


def original_archive(path, root):
    checked(path,ORIGINAL_ARCHIVE_SHA,ORIGINAL_ARCHIVE_BYTES)
    with ZipFile(path) as z:
        inventory=json.loads(z.read('inventory.json'))['payload_files']
        require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(inventory)|{'inventory.json'} and z.testzip() is None,'Original archive membership/CRC differs')
        for name,m in inventory.items():
            b=z.read(name);require(len(b)==m['size_bytes'] and digest(b)==m['sha256'],'Original inventory differs: '+name)
        lines=[json.loads(line) for line in z.read('evidence/seed-46/m0/predictions.jsonl').splitlines()]
        header=lines[0];old_summary=json.loads(z.read('evidence/seed-46/m0/summary.json'))
        old_per_image=json.loads(z.read('evidence/seed-46/m0/per_image.json'))
    require(header['model_sha256']==ORIGINAL_SHA and header['inference']==INFERENCE,'Original model or inference differs')
    require(old_summary['matching_iou']==.5 and old_summary['endpoint_confidence']==.5,'Original score setting differs')
    checked(root/'data/manifest.json',header['manifest_sha256'])
    records={r['id']:r for r in header['images']};preds={q:{} for q in (.15,.25,.4)};scores={q:[] for q in preds}
    old_rows={(r['image_id'],r['threshold']):r for r in old_per_image}
    for row in lines[1:]:
        identity,q=row['image_id'],row['threshold'];require(identity in records and q in preds,'Unexpected original prediction')
        require(identity not in preds[q] and row['image_sha256']==records[identity]['image_sha256'],'Original record identity differs')
        label=checked(root/'data'/records[identity]['label'],records[identity]['label_sha256'])
        gold=parse_pose_labels(label.decode(),1280,720,identity)
        score=score_image(gold,row['predictions'],1280,720,.5,.5)
        require(all(old_rows[identity,q][k]==v for k,v in score.items()),'Original per-image scoring failed: '+identity)
        scores[q].append(score);preds[q][identity]=row['predictions']
    for q in preds:
        require(set(preds[q])==set(records),'Missing original prediction')
        expected=next(x for x in old_summary['metrics'] if x['threshold']==q)
        require(all(expected[k]==v for k,v in aggregate_scores(scores[q]).items() if k!='endpoint_errors_px'),'Original aggregate scoring failed')
    return {**header,'runtime':old_summary['runtime']},preds[.25]


def load_archives(root,cache):
    cache.mkdir(parents=True,exist_ok=True)
    original_path=cache/'original-epoch173-evaluation.zip'
    fetch(original_path,f'https://raw.githubusercontent.com/Siuuuers/dl/{ARCHIVE_COMMIT}/{ORIGINAL_ARCHIVE_PATH}',ORIGINAL_ARCHIVE_SHA,ORIGINAL_ARCHIVE_BYTES)
    fetch(cache/'reflection-results.zip',REFLECTION_URL,REFLECTION_SHA,2931102)
    header,original=original_archive(original_path,root)
    reflection=verify_archive(cache/'reflection-results.zip')
    refs=load_references(root,reflection['summary']['contract'])
    images={r['id']:r for r in header['images']}
    require({i:r['image_sha256'] for i,r in images.items()}==reflection['summary']['contract']['images'],'Original and current photograph identities differ')
    archived={'original':original,
        'trained':{i:r['raw'] for i,r in reflection['predictions']['control'].items()},
        'reflected':{i:r['raw'] for i,r in reflection['predictions']['treatment'].items()}}
    return {'header':header,'reflection':reflection,'references':refs,'archived':archived,'images':images}


def assemble(native):
    arms={};suppression={}
    for arm in VIEWS:
        arms[arm]={};suppression[arm]={}
        for image,raw in native[arm].items():
            selected=[p for p in raw if p['confidence']>.25]
            arms[arm][image],suppression[arm][image]=postclip_nms(selected)
    arms['proposed']={};associations={}
    for image in arms['trained']:
        arms['proposed'][image],associations[image]=average_predictions(arms['trained'][image],arms['reflected'][image])
    return arms,suppression,associations


def measure(refs,native):
    arms,suppression,associations=assemble(native)
    result={'protocol':PROTOCOL,'references':{},'association_counts':dict(Counter(r['status'] for rows in associations.values() for r in rows))}
    cases={};per_image={}
    for reference,gold in refs.items():
        result['references'][reference]={};cases[reference]={};per_image[reference]={}
        for arm,predictions in arms.items():
            summary,scores,states=report(gold,predictions)
            result['references'][reference][arm]={'detection':{k:v for k,v in summary['detection'].items() if k!='endpoint_errors_px'},'pairs':summary['pairs']}
            cases[reference][arm]=states;per_image[reference][arm]=scores
        baseline=cases[reference]['original'];proposed=cases[reference]['proposed'];changes={}
        for tolerance in (5,10):
            key='within'+str(tolerance);gain=[];loss=[]
            for a,b in zip(baseline,proposed):
                require((a['image'],a['reference_index'])==(b['image'],b['reference_index']),'Case order differs')
                identity=a['image']+'/R'+str(a['reference_index']+1).zfill(3)
                if a['complete_reference'] and not a[key] and b[key]:gain.append(identity)
                if a['complete_reference'] and a[key] and not b[key]:loss.append(identity)
            changes[str(tolerance)]={'gained':len(gain),'lost':len(loss),'gained_ids':gain,'lost_ids':loss}
        result['references'][reference]['original_to_proposed_changes']=changes
    return result,{'native-predictions':native,'predictions':arms,'suppression':suppression,'associations':associations,'cases':cases,'per-image':per_image}


def source_check(root,commit,driver_sha):
    root=root.resolve();require(root==Path(__file__).resolve().parents[1],'Driver source root differs')
    require(re.fullmatch('[0-9a-f]{40}',commit) and re.fullmatch('[0-9a-f]{64}',driver_sha),'Complete immutable source identities required')
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    require(git('rev-parse','HEAD')==commit and not git('status','--porcelain','--untracked-files=all'),'Source checkout changed')
    checked(Path(__file__),driver_sha)
    files={}
    for line in git('ls-tree','-r','HEAD','scripts').splitlines():
        meta,rel=line.split('\t',1);mode,kind,oid=meta.split()
        if not rel.endswith('.py'):continue
        b=(root/rel).read_bytes();require(kind=='blob' and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==oid,'Source file changed: '+rel)
        files[rel]=digest(b)
    return {'commit':commit,'driver_sha256':driver_sha,'files_sha256':files}


def prepare(root,cache,source):
    data=load_archives(root,cache)
    original_manifest=json.loads(checked(root/'data/manifest.json',ORIGINAL_MANIFEST_SHA))
    assets=cache/'original-inputs';items=[('data/'+r['image'],r['image_sha256']) for r in data['images'].values()]
    items.append(('weights/cassette_lower_epoch173.pt',ORIGINAL_SHA))
    def obtain(item):
        rel,sha=item;url=f'https://media.githubusercontent.com/media/Siuuuers/dl/ea4ac955f2dfaf1add24c243599715f802eb41a7/{rel}'
        b=fetch(assets/rel,url,sha,106838542 if rel.endswith('.pt') else None)
        if rel.endswith('.png'):require(b[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',b[16:24])==(1280,720),'Image dimensions differ')
        return rel,{'sha256':sha,'bytes':len(b)}
    with ThreadPoolExecutor(max_workers=5) as pool:asset_receipts=dict(pool.map(obtain,items))
    import audit_additional_front_pilot_r47_results as prior
    archive=prior.obtain_archive(cache)
    with ZipFile(archive) as z:b=z.read('candidate/fit/weights/last.pt')
    require(digest(b)==TRAINED_SHA,'Trained detector differs')
    trained=cache/'research-detector.pt'
    if trained.exists():checked(trained,TRAINED_SHA)
    else:trained.write_bytes(b)
    contract={'schema':'baomai-original-comparison-r59-inputs-v1','source':source,'protocol':PROTOCOL,
        'original_archive':{'commit':ARCHIVE_COMMIT,'path':ORIGINAL_ARCHIVE_PATH,'sha256':ORIGINAL_ARCHIVE_SHA,'bytes':ORIGINAL_ARCHIVE_BYTES},
        'reflection_archive_sha256':REFLECTION_SHA,'research_training_archive_sha256':prior.ARCHIVE_SHA,
        'original_model_sha256':ORIGINAL_SHA,'trained_model_sha256':TRAINED_SHA,
        'reference_contract':data['reflection']['summary']['contract']['references'],'assets':asset_receipts,
        'original_image_manifest_sha256':ORIGINAL_MANIFEST_SHA}
    return {**data,'root':root,'cache':cache,'assets':assets,'models':{'original':assets/'weights/cassette_lower_epoch173.pt','trained':trained},'contract':contract}


def infer(context,device):
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from pose_checkpoint import check_loaded_pose_model
    from run_corner_refinement_r58 import runtime_setup,check_runtime
    runtime_setup(device);runtime=check_runtime(device);output={arm:{} for arm in VIEWS};native_reflected={}
    for model_name,views in (('original',('original',)),('trained',('trained','reflected'))):
        model=YOLO(str(context['models'][model_name]));check_loaded_pose_model(model)
        for _ in range(2):model.predict(np.zeros((720,1280,3),dtype=np.uint8),conf=.25,device=device,augment=False,**INFERENCE)
        require(not model.predictor.model.end2end and not model.predictor.model.fp16,'Unexpected inference backend')
        require(str(model.predictor.device)==runtime['device'],'Unexpected actual device')
        for image,record in sorted(context['images'].items()):
            decoded=cv2.imread(str(context['assets']/'data'/record['image']));require(decoded is not None and decoded.shape==(720,1280,3),'Image decoding differs')
            for arm in views:
                transformed=np.ascontiguousarray(decoded[:,::-1,:]) if arm=='reflected' else decoded
                result=model.predict(transformed,conf=.25,device=device,augment=False,**INFERENCE)[0]
                raw=_prediction_records(result)
                if arm=='reflected':native_reflected[image]=raw;raw=inverse_horizontal_predictions(raw,1280)
                output[arm][image]=raw
            print(model_name,image,'complete',flush=True)
        del model
        if device!='cpu':torch.cuda.empty_cache()
    runtime.update(precision='float32',tf32=False,inference=INFERENCE,box_confidence=.25,model_passes=3,images=20)
    return output,native_reflected,runtime


def continuity(archived,fresh,refs):
    tests={};all_passed=True
    for arm in VIEWS:
        maximum_coordinate=0.;maximum_score=0.;equal_structure=True;discrete=True
        for image in archived[arm]:
            a=[p for p in archived[arm][image] if p['confidence']>.25];b=fresh[arm][image]
            if len(a)!=len(b):equal_structure=False;continue
            for x,y in zip(a,b):
                equal_structure &= x['class_id']==y['class_id']
                maximum_score=max(maximum_score,abs(x['confidence']-y['confidence']))
                for u,v in zip(x['bbox_xyxy'],y['bbox_xyxy']):maximum_coordinate=max(maximum_coordinate,abs(u-v))
                for u,v in zip(x['keypoints'],y['keypoints']):
                    maximum_score=max(maximum_score,abs(u[2]-v[2]))
                    for c,d in zip(u[:2],v[:2]):
                        if c is None or d is None:equal_structure &= c is None and d is None
                        else:maximum_coordinate=max(maximum_coordinate,abs(c-d))
            for gold in refs.values():
                old=score_image(gold[image],postclip_nms(a)[0],1280,720,.5,.5)
                new=score_image(gold[image],postclip_nms(b)[0],1280,720,.5,.5)
                keys=('tp','fn','fp','duplicate_fp','other_fp')
                discrete &= all(old[k]==new[k] for k in keys)
                discrete &= [(m['gold_index'],m['prediction_index']) for m in old['matches']]==[(m['gold_index'],m['prediction_index']) for m in new['matches']]
                for score,predictions in ((old,postclip_nms(a)[0]),(new,postclip_nms(b)[0])):
                    matches={m['gold_index']:m['prediction_index'] for m in score['matches']}
                    states=[state(g,predictions[matches[gi]] if gi in matches else None) for gi,g in enumerate(gold[image])]
                    outcomes=[(s['detected'],s['within5'],s['within10']) for s in states]
                    if score is old:old_outcomes=outcomes
                    else:discrete &= outcomes==old_outcomes
        passed=equal_structure and discrete and maximum_coordinate<=.05 and maximum_score<=1e-5
        tests[arm]={'passed':passed,'same_record_structure':equal_structure,'same_detection_assignments':discrete,'max_coordinate_component_difference_px':maximum_coordinate,'max_score_difference':maximum_score}
        all_passed &= passed
    return {'passed':all_passed,'coordinate_tolerance_px':.05,'score_tolerance':1e-5,'arms':tests,
        'meaning':'Checks historical prediction continuity; fresh comparison is measured separately using both models in the same recorded runtime.'}


def write_evidence(work,summary,records,extras):
    payload={'summary.json':encoded(summary),**{n:encoded(v) for n,v in extras.items()}}
    for name,value in records.items():payload[name+'.json.gz']=gzip.compress(encoded(value),mtime=0)
    payload['inventory.json']=encoded({n:{'bytes':len(b),'sha256':digest(b)} for n,b in payload.items()})
    target=work/RESULT_NAME
    with ZipFile(target.with_suffix('.part'),'w',ZIP_DEFLATED) as z:
        for name,b in payload.items():z.writestr(name,b)
    target.with_suffix('.part').replace(target)
    (work/'summary.json').write_bytes(payload['summary.json'])
    return target


def verify(work,context):
    with ZipFile(work/RESULT_NAME) as z:
        inv=json.loads(z.read('inventory.json'));require(len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(inv)|{'inventory.json'} and z.testzip() is None,'Result membership/CRC differs')
        for n,m in inv.items():b=z.read(n);require(m=={'bytes':len(b),'sha256':digest(b)},'Result inventory differs: '+n)
        summary=json.loads(z.read('summary.json'));require(summary['contract']==context['contract'],'Result input contract differs')
        native=json.loads(gzip.decompress(z.read('native-predictions.json.gz')))
        recomputed,records=measure(context['references'],native)
        require(summary['measurement']==recomputed,'Result score replay differs')
        for name,value in records.items():require(json.loads(gzip.decompress(z.read(name+'.json.gz')))==value,'Result record replay differs: '+name)
        reflected=json.loads(z.read('native-reflected.json'))
        require({i:inverse_horizontal_predictions(p,1280) for i,p in reflected.items()}==native['reflected'],'Inverse reflection differs')
        require(json.loads(z.read('historical-continuity.json'))==continuity(context['archived'],native,context['references']),'Historical continuity replay differs')
        require(summary['status']=='completed' and summary['fresh_inference'] and not summary['training'],'Wrong experiment status')
    print('Verified complete common-runtime comparison and saved score replay.',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','verify','archive'))
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--source-commit');p.add_argument('--script-sha256');p.add_argument('--device',default='0');a=p.parse_args()
    a.root=a.root.resolve();a.cache_dir=a.cache_dir.resolve();a.work=a.work.resolve();require(not a.work.is_relative_to(a.root),'Keep result files outside source')
    a.work.mkdir(parents=True,exist_ok=True)
    if a.action=='archive':
        data=load_archives(a.root,a.cache_dir);summary,records=measure(data['references'],data['archived'])
        summary={'schema':'baomai-original-comparison-r59-archived-v1','status':'completed','fresh_inference':False,'training':False,
            'source_archive_sha256':ORIGINAL_ARCHIVE_SHA,'reflection_archive_sha256':REFLECTION_SHA,'original_model_sha256':ORIGINAL_SHA,'trained_model_sha256':TRAINED_SHA,
            'original_runtime':data['header']['runtime'],'trained_runtime':data['reflection']['runtime'],'measurement':summary}
        (a.work/'archived-summary.json').write_bytes(encoded(summary))
        for name,value in records.items():(a.work/(name+'.json.gz')).write_bytes(gzip.compress(encoded(value),mtime=0))
        print('Verified original archive scoring and compared saved predictions at common confidence 0.25.');return
    source=source_check(a.root,a.source_commit,a.script_sha256);context=prepare(a.root,a.cache_dir,source)
    preflight=a.work/'preflight.json';value=encoded(context['contract'])
    require(not preflight.exists() or preflight.read_bytes()==value,'Different work preserved; choose a fresh directory')
    preflight.write_bytes(value)
    if a.action=='prepare':print('Verified both model files, all 20 photographs, annotations and historical outputs.');return
    if a.action=='run' and not (a.work/RESULT_NAME).exists():
        native,reflected,runtime=infer(context,a.device);measurement,records=measure(context['references'],native)
        historical=continuity(context['archived'],native,context['references'])
        summary={'schema':'baomai-original-comparison-r59-fresh-v1','status':'completed','fresh_inference':True,'training':False,
            'automatic_promotion':False,'development_only':True,'contract':context['contract'],'runtime':runtime,
            'historical_reproduction_passed':historical['passed'],'measurement':measurement}
        write_evidence(a.work,summary,records,{'native-reflected.json':reflected,'historical-continuity.json':historical,'preflight.json':context['contract']})
    verify(a.work,context)

if __name__=='__main__':main()
