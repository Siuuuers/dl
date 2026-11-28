#!/usr/bin/env python3
"""Verify the two returned local-corner experiments and replay their measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile

SOURCE = '5b1650b8fa3959a0917680305d7b499af783c5aa'
DRIVER = '80d52d827e3b0a63318ebb0a611356bd6534da7df1d9891a3f409f57076841b6'
RETURNS = {
    'initial': {'pr':153, 'head':'fc0ec1a92e5d9357808e3c9f3bcf87d87450cdd3',
        'bytes':3118004, 'sha256':'f57e0a53a24a6159e4cb9b91613099146f05e01a483b02e304419a520e69202b'},
    'none': {'pr':154, 'head':'bf718680bc5f2888e51ac084610e678fc761244a',
        'bytes':3117954, 'sha256':'1f66eae2410c9e9d866f924956e1dc94916c86dd115ec2e43a0c27f5c3fa0b60'},
}
MEMBERS = {'baseline-equivalence.json','cases.json','corner_refinement_r58.py','crop-transforms.json',
    'detector-evaluation.json','detector-receipt.json','detector-runtime.json','detector-training.json',
    'input-audit.json','inventory.json','model-smoke.json','pair-accounting.json','predictions.json',
    'preflight.json','run_corner_refinement_r58.py','staging-r58.json','summary.json','training-crops.json',
    'training-log.json','training-receipt.json'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def checked_source(root, relative, expected, overrides=None):
    path = root/relative
    data = path.read_bytes() if path.is_file() else None
    if data is None or sha(data) != expected:
        path = overrides/relative if overrides else path
        need(path.is_file(), 'Missing pinned source: '+relative)
        data = path.read_bytes()
    need(sha(data) == expected, 'Pinned source hash differs: '+relative)
    return data


def load_return(directory, mode):
    identity = RETURNS[mode]
    path = directory/f'baomai-corner-refinement-r58-{mode}-results.zip'
    payload = path.read_bytes()
    need(len(payload)==identity['bytes'] and sha(payload)==identity['sha256'], 'Returned ZIP differs')
    with ZipFile(path) as z:
        need(set(z.namelist())==MEMBERS and len(z.namelist())==len(MEMBERS)
             and z.testzip() is None, 'Returned ZIP membership/CRC differs')
        members = {name:z.read(name) for name in z.namelist()}
    inventory = json.loads(members['inventory.json'])
    need(set(inventory)==MEMBERS-{'inventory.json'}, 'Inventory membership differs')
    for name,item in inventory.items():
        need(item=={'bytes':len(members[name]),'sha256':sha(members[name])}, 'Member hash differs: '+name)
    data = {name[:-5]:json.loads(value) for name,value in members.items() if name.endswith('.json')}
    summary, preflight = data['summary'], data['preflight']
    contract = summary['contract']
    need(summary['status']=='completed' and summary['hint_mode']==mode and summary['baseline_passed']
         and summary['interpretation_allowed'] and not summary['automatic_promotion'], 'Experiment status differs')
    need(contract['source']==SOURCE and contract['script_sha256']==DRIVER, 'Implementation identity differs')
    need(sha(members['run_corner_refinement_r58.py'])==DRIVER, 'Archived driver differs')
    need(contract==preflight['contract'] and sha(encoded(contract))==preflight['contract_sha256'], 'Input contract differs')
    need(data['input-audit']==preflight['input_audit'], 'Input audit differs')
    need(sha(members['staging-r58.json'])==contract['training_staging_sha256'], 'Training staging differs')
    return members, data


def model_parts(directory, data):
    manifest_data = (directory/'return-manifest.json').read_bytes()
    manifest = json.loads(manifest_data)
    mode = data['summary']['hint_mode']
    need(manifest['status']=='completed' and manifest['evidence_file']==f'baomai-corner-refinement-r58-{mode}-results.zip'
         and manifest['evidence_sha256']==RETURNS[mode]['sha256'], 'Return manifest differs')
    digest=hashlib.sha256();total=0
    need(len(manifest['parts'])==9, 'Model part count differs')
    for i,part in enumerate(manifest['parts']):
        name=f'refiner-final.pt.part{i:02d}'
        need(part['file']==name+'.zip' and part['member']==name, 'Model part order differs')
        payload=(directory/part['file']).read_bytes()
        need(len(payload)==part['bytes'] and sha(payload)==part['sha256'], 'Wrapped model part differs')
        with ZipFile(directory/part['file']) as z:
            need(z.namelist()==[name] and z.testzip() is None, 'Model part membership/CRC differs')
            content=z.read(name)
        need(len(content)==part['member_bytes'] and sha(content)==part['member_sha256'], 'Model member differs')
        digest.update(content);total+=len(content)
    receipt=data['training-receipt']
    need(total==manifest['model_bytes']==receipt['model_bytes']
         and digest.hexdigest()==manifest['model_sha256']==receipt['model_sha256'], 'Full model identity differs')
    return {'parts':9,'model_bytes':total,'model_sha256':digest.hexdigest(),'manifest_sha256':sha(manifest_data)}


def audit(root, directory, mode, original_archive, overrides=None):
    members,data=load_return(directory,mode)
    contract=data['summary']['contract']
    for relative,expected in contract['source_files_sha256'].items():
        checked_source(root,relative,expected,overrides)
    for name in ('run_corner_refinement_r58.py','corner_refinement_r58.py'):
        need(members[name]==checked_source(root,'scripts/'+name,
             contract['source_files_sha256']['scripts/'+name],overrides), 'Archived implementation differs')
    sys.path.insert(0,str(root/'scripts'))
    from audit_reflection_return import load_references
    import audit_additional_front_pilot_r47_results as prior
    import run_inference_ablation_r53 as r53
    import run_corner_refinement_r58 as runner
    from evaluate_pose import parse_pose_labels
    import torch
    need(runner.RECIPE==contract['settings']['recipe'] and contract['settings']['hint_mode']==mode,
         'Training recipe differs')
    refs=load_references(root,contract)
    maps=json.loads(checked_source(root,contract['row_maps']['path'],contract['row_maps']['sha256'],overrides))
    prior_receipt=json.loads(encoded(prior.audit(original_archive,root)))
    need(prior_receipt==data['input-audit'], 'Original input archive audit differs')
    with ZipFile(original_archive) as z:
        old_predictions=z.read('evaluation/predictions.json')
        old_summary=z.read('evaluation/summary.json')
        old_staging=json.loads(z.read('staging.json'))
        entries=old_staging['roles']['candidate']
        training_labels={e['id']:z.read('datasets/candidate/'+e['label']) for e in entries}
    need(sha(old_predictions)==contract['saved_predictions_sha256']
         and sha(old_summary)==contract['saved_summary_sha256'], 'Archived control source differs')
    arms=data['predictions']
    need(arms['control']==data['detector-evaluation'], 'Control identity differs')
    runner.frozen_detection_check(arms['control'],arms['treatment'])
    for output in arms.values():runner.replay_postclip(output)
    runner.replay_postclip(data['detector-training'])
    runner.replay_transforms(arms['control'],arms['treatment'],data['crop-transforms'])
    gate=r53.baseline_equivalence(refs,json.loads(old_predictions)['candidate'],arms['control'])
    need(gate['passed'] and gate==data['baseline-equivalence'], 'Baseline-equivalence replay differs')
    scored={arm:r53.reports(refs,output) for arm,output in arms.items()}
    need(scored==data['summary']['reports'], 'Score replay differs')
    need(data['cases']==r53.case_inventory(refs,maps,arms,scored), 'Case replay differs')
    need(data['pair-accounting']==runner.pair_accounting(refs,arms), 'Offered-pair accounting differs')
    stage=data['staging-r58']; stage_rows={row['id']:row for row in stage['records']}
    need(len(stage_rows)==162 and set(stage_rows)=={e['id'] for e in entries}, 'Photograph membership differs')
    train_ids={row['id'] for row in stage_rows.values() if row['split']=='train'}
    val_ids={row['id'] for row in stage_rows.values() if row['split']=='val'}
    need(len(train_ids)==142 and len(val_ids)==20 and not train_ids&val_ids, 'Photograph split differs')
    with tempfile.TemporaryDirectory(prefix='refiner-audit-') as tmp:
        work=Path(tmp);fronts=known=0
        for entry in entries:
            row=stage_rows[entry['id']]
            expected={'id':entry['id'],'split':entry['split'],'image':'dataset/'+entry['image'],
                'label':'dataset/'+entry['label'],'image_sha256':entry['image_sha256'],
                'label_sha256':entry['label_sha256']}
            need(row==expected, 'Staged record differs: '+row['id'])
            label=training_labels[row['id']]
            need(sha(label)==row['label_sha256'], 'Training label bytes differ')
            path=work/row['label'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(label)
            if row['split']=='train':
                targets=parse_pose_labels(label.decode(),1280,720,row['id'])
                fronts+=len(targets);known+=sum(p[2]==2 for t in targets for p in t['keypoints'])
        need(fronts==3999 and known==7259, 'Training endpoint inventory differs')
        crop_replay=runner.training_crops({'stage':stage,'work':work},data['detector-training'],save=False)
    need(crop_replay==data['training-crops'], 'Training crops/supervision replay differs')
    receipt,history=data['training-receipt'],data['training-log']
    for key,name in [('training_log_sha256','training-log.json'),('training_crops_sha256','training-crops.json'),
                     ('preflight_sha256','preflight.json')]:
        need(receipt[key]==sha(members[name]), 'Training receipt differs: '+key)
    need(receipt==data['summary']['training_receipt'], 'Summary training receipt differs')
    n=len(crop_replay['records']);updates=math.ceil(n/runner.BATCH)
    need(receipt['epochs']==30 and receipt['samples']==30*n and receipt['updates']==receipt['expected_updates']==30*updates,
         'Training exposure differs')
    need([row['epoch'] for row in history]==list(range(1,31)), 'Epoch membership differs')
    for row in history:
        need(row['samples']==n and row['updates']==updates, 'Epoch exposure differs')
        order=torch.randperm(n,generator=torch.Generator().manual_seed(runner.SEED+row['epoch'])).tolist()
        need(sha(encoded([crop_replay['records'][i]['identity'] for i in order]))==row['sample_order_sha256'],
             'Epoch sample order differs')
    detector_receipt=data['detector-receipt']
    need(detector_receipt['preflight_sha256']==sha(members['preflight.json'])
         and detector_receipt['runtime']==data['detector-runtime'], 'Detector receipt differs')
    for name,digest in detector_receipt['files'].items():
        need(sha(members[name])==digest, 'Detector cached member differs: '+name)
    model=model_parts(directory,data)
    compact={}
    for reference in refs:
        compact[reference]={}
        for arm in arms:
            item=scored[arm][reference]['postclip']
            compact[reference][arm]={'detection':{k:item['detection'][k] for k in
                ('gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp')},
                'complete_pairs':item['corners_by_locatable_count']['2'],
                'offered_pairs':data['pair-accounting']['references'][reference][arm]}
    return {'schema':'cassette-local-refinement-return-audit-v1','status':'passed','mode':mode,
        'input':RETURNS[mode],'implementation_commit':SOURCE,'driver_sha256':DRIVER,
        'verified_archive_members':len(MEMBERS),'verified_source_data_files':len(contract['source_files_sha256']),
        'replayed_aggregate_tables':16,'replayed_per_image_tables':320,
        'raw_refinement_records':len(data['crop-transforms']),
        'replayed_training_photographs':142,'training_fronts':fronts,'known_training_endpoints':known,
        'usable_training_crops':n,'epochs':30,'samples':receipt['samples'],'updates':receipt['updates'],
        'sample_order_hashes_replayed':30,'sample_order_replay_torch':str(torch.__version__),
        'initial_model_state_sha256':data['model-smoke']['initial_model']['state_sha256'],
        'recorded_runtime':data['detector-runtime'],'model':model,'baseline_equivalence':gate,
        'results':compact,'training_loss_first':history[0]['mean_loss'],'training_loss_last':history[-1]['mean_loss'],
        'all_original_detection_records_preserved':True,'automatic_promotion':False,
        'interpretation':'Development comparison with the frozen research detector. This control is not the original supplied baomai model.',
        'verification_limit':'Saved decoder arithmetic and scoring are replayed; this audit does not independently recompute model heatmaps. Training sample order is reproduced with the stated CPU Torch version, not the original T4 runtime.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--mode',choices=RETURNS,required=True)
    p.add_argument('--original-archive',type=Path,required=True)
    p.add_argument('--pinned-sources',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root.resolve(),a.directory.resolve(),a.mode,a.original_archive.resolve(),a.pinned_sources)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(encoded(result))
    print(json.dumps({'status':result['status'],'mode':a.mode,'output':str(a.output),
          'source_files':result['verified_source_data_files'],'model_sha256':result['model']['model_sha256']}))


if __name__=='__main__':main()
