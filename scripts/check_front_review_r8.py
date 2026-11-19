#!/usr/bin/env python3
"""Stage the actual r8 arms and verify their unchanged comparison inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
from stage_pose_training import stage_datasets
from verify_front_review_r8 import REVIEW, PARENT, CANDIDATE, PROTOCOL_PATH, verify_candidate

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    assert not out.exists() and not out.is_relative_to(root), 'Choose a fresh directory outside the checkout.'
    out.mkdir(parents=True)
    inspection=verify_candidate(root)
    records=json.loads((root/'data/manifest.json').read_text())['records']
    original_before={r['id']:sha(root/'data'/r['label']) for r in records}
    assert all(original_before[r['id']]==r['label_sha256'] for r in records)
    manifest=json.loads((root/CANDIDATE).read_text())
    protocol=json.loads((root/PROTOCOL_PATH).read_text())
    prior=json.loads((root/'experiments/front-review-2026-11-19-r7.json').read_text())
    staged=stage_datasets(root,root/CANDIDATE,out/'staging')
    assert staged['actual_original_pngs_verified']==157
    control_path=root/manifest['control']['manifest']
    control=json.loads(control_path.read_text())
    changes={c['image_id']:c for c in control['changes']}
    held=set(manifest['retained_legacy_image_ids']);val=0;held_count=0
    for r in records:
        original=(root/'data'/r['label']).read_bytes()
        arms=[(out/'staging'/arm/r['label']).read_bytes() for arm in ('reference','candidate')]
        if r['split']=='val':
            assert arms[0]==arms[1]==original;val+=1
        if r['id'] in held:
            change=changes.get(r['id'])
            expected=(control_path.parent/change['candidate_label']).read_bytes() if change else original
            assert arms[0]==arms[1]==expected;held_count+=1
    assert val==20 and held_count==len(held)
    assert len([r for r in records if r['id'].startswith('old_') and r['id'] in held])==31
    assert not held.intersection(manifest['changed_images_vs_control'])
    for role,objects in [('primary',666),('diagnostic',663)]:
        for suffix in ('reference','reference_sha256','counts'):
            assert protocol['evaluation'][role+'_'+suffix]==prior['evaluation'][role+'_'+suffix]
        assert sha(root/protocol['evaluation'][role+'_reference'])==protocol['evaluation'][role+'_reference_sha256']
        assert protocol['evaluation'][role+'_counts']['objects']==objects
    assert sha(root/PARENT['path'])==PARENT['sha256']
    restored=root/'data/reviews/fronts-2026-11-19-r8/labels/train/old_000013.txt'
    assert restored.read_bytes()==(root/'data/labels/train/old_000013.txt').read_bytes()
    assert all(sha(root/'data'/r['label'])==original_before[r['id']] for r in records)
    report={'status':'verified','training_run':False,'review':REVIEW,'parent':PARENT,
            'candidate_manifest_sha256':sha(root/CANDIDATE),'protocol_sha256':sha(root/PROTOCOL_PATH),
            'original_images_verified':157,'all_original_label_bytes_unchanged':True,
            'staged_counts':staged['counts'],'original_validation_photos_byte_identical_in_both_arms':val,
            'held_training_photos_exact_pr35_in_both_arms':held_count,'held_old_series_photos':31,
            'old_000013_review_restored_exact_original_bytes':True,
            'frozen_reference_object_counts':{'primary':666,'diagnostic':663},
            'changed_training_photos_vs_control':len(manifest['changed_images_vs_control']),
            'staging_manifest_sha256':sha(out/'staging/staging-manifest.json'),
            'source_files_sha256':{f'scripts/{name}':sha(root/'scripts'/name) for name in
                ['check_front_review_r8.py','verify_front_review_r8.py','front_review_experiment.py',
                 'stage_pose_training.py','train_pose_comparison.py','verify_incremental_pose_candidate.py']},
            'limits':['Mechanical verification binds the saved decisions; it is not independent physical certification.',
                      'This check stages labels and images. It runs no model training.']}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
