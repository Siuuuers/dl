#!/usr/bin/env python3
"""Verify actual R23-to-R24 staged bytes, recorded recipe/runtime, and matched-control tensors."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--r23-bundle',type=Path,required=True)
    parser.add_argument('--r24-bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    bundles={'r23':args.r23_bundle.resolve(),'r24':args.r24_bundle.resolve()}
    indexes={name:read(path/'gpu-results-manifest.json') for name,path in bundles.items()}
    a,b=indexes.values()
    assert a['runtime']==b['runtime'] and a['shared_training_settings']==b['shared_training_settings']
    report={'script_sha256':sha(__file__),'recorded_training_runtime_equal':True,'training_settings_equal':True,'runtime':a['runtime'],'shared_training_settings':a['shared_training_settings'],'roles':{}}
    for role in ('reference','candidate'):
        manifests={name:read(path/role/'staging-manifest.json') for name,path in bundles.items()}
        files={name:{f['path']:f for f in m['variants'][role]['files']} for name,m in manifests.items()}
        assert files['r23'].keys()==files['r24'].keys()
        receipts={name:read(path/role/'training-receipt.json') for name,path in bundles.items()}
        for field in ('recipe_id','recipe_sha256','initial_checkpoint_sha256','checkpoint_selection','completed_epochs','optimizer_steps','batches_completed','images_processed'):
            assert receipts['r23'][field]==receipts['r24'][field],field
        changes=[]
        for path,old in files['r23'].items():
            new=files['r24'][path]
            for item in (old,new):
                assert sha(root/item['source_path'])==item['source_sha256']==item['output_sha256']
            if old['output_sha256']==new['output_sha256']:continue
            assert path.startswith('labels/train/')
            oldlines=(root/old['source_path']).read_text().splitlines();newlines=(root/new['source_path']).read_text().splitlines()
            assert len(oldlines)==len(newlines)
            rows=[]
            for row,(x,y) in enumerate(zip(oldlines,newlines),1):
                if x==y:continue
                xx,yy=x.split(),y.split();assert len(xx)==len(yy)==11
                changed=[i for i in range(11) if xx[i]!=yy[i]]
                assert set(changed)<={5,6,8,9}
                rows.append({'row':row,'before':x,'after':y,'changed_fields':changed,'point_count':sum(bool(set(changed)&set(pair)) for pair in ((5,6),(8,9)))})
            changes.append({'path':path,'r23_source':old['source_path'],'r24_source':new['source_path'],'r23_sha256':old['output_sha256'],'r24_sha256':new['output_sha256'],'rows':rows})
        report['roles'][role]={'changes':changes,'recipe_id':receipts['r24']['recipe_id'],'recipe_sha256':receipts['r24']['recipe_sha256'],'initial_checkpoint_sha256':receipts['r24']['initial_checkpoint_sha256'],'files_verified_per_arm':len(files['r23'])}
    assert not report['roles']['reference']['changes']
    changes=report['roles']['candidate']['changes']
    assert [x['path'] for x in changes]==['labels/train/new_000012.txt','labels/train/new_000014.txt','labels/train/new_000016.txt']
    assert sum(len(x['rows']) for x in changes)==18
    assert sum(row['point_count'] for x in changes for row in x['rows'])==22
    report['changed_training_files']=3;report['changed_training_rows']=18;report['changed_training_endpoints']=22
    report['original_images_and_trainer_validation_unchanged']=True
    report['new_000010_development_review_not_used_for_training']=True
    import torch
    states={}
    for name,path in bundles.items():
        checkpoint=path/'reference/fit/weights/last.pt'
        m=read(path/'reference/checkpoint-manifest.json')
        assert sha(checkpoint)==m['checkpoint']['sha256']
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        states[name]=(data.get('ema') or data['model']).state_dict()
    x,y=states.values();assert list(x)==list(y)
    different=[name for name in x if x[name].shape!=y[name].shape or x[name].dtype!=y[name].dtype or not torch.equal(x[name],y[name])]
    assert not different
    report['matched_control_tensors']={'compared':len(x),'identical':len(x),'changed':different}
    report['limits']=['A single matched seed/runtime pair is descriptive evidence, not a population-level causal guarantee.','This crossround comparison isolates the recorded changed training supervision; PR99 candidate versus PR35-control instead tests the accumulated eligible review.','R24 review labels on development-only photo010 are used identically when scoring R23 and R24; they are never training inputs.']
    assert not args.output.exists()
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'changed_files':3,'changed_rows':18,'changed_points':22,'identical_control_tensors':len(x)},indent=2))


if __name__=='__main__':main()
