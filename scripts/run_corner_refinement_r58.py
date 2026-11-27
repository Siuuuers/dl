#!/usr/bin/env python3
"""Train one fixed local refiner, preserving every detector box and its scores."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata as md
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import urllib.request
import zipfile

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import corner_refinement_r58 as geometry
import diagnose_fronts_r50 as r50
import run_inference_ablation_r53 as r53
import run_partial_classification_r51 as r51
import audit_additional_front_pilot_r47_results as prior
from evaluate_pose import INFERENCE, _prediction_records, parse_pose_labels
from pose_scoring import score_image
from predict_pose_folder import postclip_nms

need, encoded, put = r50.need, r50.encoded, r50.put
BACKBONE_SHA = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
BACKBONE_BYTES = 102540417
SEED, EPOCHS, BATCH = 46, 30, 32
RECIPE = {'seed': SEED, 'epochs': EPOCHS, 'batch_size': BATCH, 'precision': 'float32',
    'optimizer': 'AdamW', 'learning_rate': .0001, 'weight_decay': .0001,
    'scheduler': 'CosineAnnealingLR', 'minimum_learning_rate': .000001,
    'gradient_clip_norm': 10., 'workers': 0, 'drop_last': False,
    'checkpoint_selection': 'fixed final epoch; no development-based selection',
    'model': 'TorchVision ResNet50 V1.5 plus three 256-channel 4x4 stride2 deconvolutions and two heatmaps',
    'input_channels': 5, 'crop_factor': 1.5, 'canvas': [256, 256], 'heatmap': [64, 64],
    'target_sigma_heatmap_cells': 2., 'hint_sigma_canvas_pixels': 8.,
    'training_hint_noise_std_canvas_pixels': 4., 'rgb_augmentation': 'none',
    'decoder': 'valid-content argmax plus 0.25-cell signed-neighbour offset',
    'coordinate_mapping': 'continuous crop edges; canvas=(image-crop_origin)*actual_xy_scale+padding; heatmap cell centre=(index+0.5)*4',
    'corner_confidence': 'original detector scores retained only for a fixed comparison, not recalibrated',
    'training_proposals': 'frozen detector on training photographs; one-to-one IoU>=0.5 matches',
    'negative_proposal_supervision': 'none; all proposals are still refined and accounted for during evaluation'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen(path, value):
    put(Path(path), encoded(value))


def checked(path, expected):
    r50.checked(Path(path), expected)
    return Path(path)


def obtain_backbone(cache):
    path = cache/'resnet50-11ad3fa6.pth'
    if not path.exists():
        with urllib.request.urlopen(geometry.WEIGHTS_URL, timeout=180) as response:
            payload = response.read()
        need(len(payload) == BACKBONE_BYTES and r50.sha(payload) == BACKBONE_SHA, 'Backbone download differs')
        put(path, payload)
    checked(path, BACKBONE_SHA)
    need(path.stat().st_size == BACKBONE_BYTES, 'Backbone file size differs')
    return path


def stage_training(root, work, cache, archive):
    plan, payload, _ = r51.label_plan(root, archive)
    entries = plan['roles']['candidate']
    assets = cache/'assets'
    r51.r47.obtain_inputs(root, assets)
    r51.obtain_supplementary(root, assets, entries)
    records = []
    for entry in entries:
        image = checked(assets/entry['image_source'], entry['image_sha256'])
        need(r51.image_size(image.read_bytes()) == (1280, 720), 'Image dimensions differ')
        target_image = work/'dataset'/entry['image']
        target_label = work/'dataset'/entry['label']
        put(target_image, image.read_bytes()); put(target_label, payload[entry['id']])
        records.append({'id': entry['id'], 'split': entry['split'],
            'image': str(target_image.relative_to(work)), 'label': str(target_label.relative_to(work)),
            'image_sha256': entry['image_sha256'], 'label_sha256': entry['label_sha256']})
    train = {r['id'] for r in records if r['split'] == 'train'}
    val = {r['id'] for r in records if r['split'] == 'val'}
    need(len(train) == 142 and len(val) == 20 and not train & val, 'Photograph split differs')
    count = sum(len(parse_pose_labels((work/r['label']).read_text(),1280,720,r['id']))
                for r in records if r['split'] == 'train')
    need(count == 3999, 'Training-front inventory differs')
    stage = {'records': records, 'training_photographs': 142, 'evaluation_photographs': 20,
        'training_fronts': count, 'annotation_source_archive_sha256': prior.ARCHIVE_SHA,
        'split': 'Existing photograph membership. No individual-crop random split.',
        'capture_groups': 'Not independently available; this is development evaluation.'}
    frozen(work/'staging-r58.json', stage)
    return stage


def source_and_inputs(args):
    root, work = args.root.resolve(), args.work.resolve()
    need(not work.is_relative_to(root), 'Keep work outside the pinned source checkout')
    need(re.fullmatch('[0-9a-f]{64}', args.script_sha256) is not None, 'Full script hash required')
    checked(root/'scripts/run_corner_refinement_r58.py', args.script_sha256)
    need(Path(__file__).resolve() == root/'scripts/run_corner_refinement_r58.py', 'Run from the pinned checkout')
    context = r50.source_check(root, args.source_commit, r53.R50_DRIVER_SHA, r50.R50_SHA)
    need(context['contract']['source_files_sha256'].get('scripts/run_corner_refinement_r58.py') == args.script_sha256,
         'Refiner driver must be tracked by the published source')
    work.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve() if args.cache_dir else work.parent/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve() if args.archive else prior.obtain_archive(cache)
    stage = stage_training(root, work, cache, archive)
    backbone = obtain_backbone(cache)
    context['contract'].update(schema='baomai-local-refiner-r58-inputs-v1',
        script_sha256=args.script_sha256, settings={'experiment': 'local_corner_refinement',
        'hint_mode': args.hint_mode, 'recipe': RECIPE, 'detector': {**INFERENCE,'conf': .13,'augment': False}},
        training=True, training_staging_sha256=sha(work/'staging-r58.json'),
        pretrained_backbone={'url': geometry.WEIGHTS_URL,'sha256': BACKBONE_SHA,'bytes': BACKBONE_BYTES},
        automatic_promotion=False)
    args.cache_dir, args.archive = cache, archive
    context = r50.prepare(args, context)
    frozen(work/'input-audit.json',context['receipt'])
    context.update(stage=stage, backbone=backbone, hint_mode=args.hint_mode)
    return context


def runtime_setup(device):
    import numpy as np
    import torch
    need(device == 'cpu' or torch.cuda.is_available(), 'Select a GPU runtime')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return torch.device('cpu' if device == 'cpu' else 'cuda:'+device)


def model_smoke(context):
    import numpy as np
    from PIL import Image
    import torch
    runtime_setup('cpu')
    rows = [r for r in context['stage']['records'] if r['split'] == 'train']
    samples = []
    for row in rows:
        gold = parse_pose_labels((context['work']/row['label']).read_text(),1280,720,row['id'])
        if not gold: continue
        g = next((x for x in gold if any(p[2] == 2 for p in x['keypoints'])),None)
        if g is None: continue
        proposal = {'bbox_xyxy': g['bbox_xyxy'], 'keypoints': [[p[0],p[1], .9 if p[2] == 2 else 0.] for p in g['keypoints']]}
        geom = geometry.crop_geometry(proposal['bbox_xyxy'])
        target, available, _ = geometry.encode_targets(g['keypoints'],geom)
        if not available.any() or not geometry.valid_grid(geom).any(): continue
        with Image.open(context['work']/row['image']) as image:
            x = geometry.crop_tensor(image.convert('RGB'),proposal,geom,context['hint_mode'],identity=row['id'])
        samples.append((x,target,available,geometry.valid_grid(geom),row['id']))
        if len(samples)==2: break
    need(len(samples)==2,'Two training crops required for the model mechanics check')
    network = geometry.build_model(context['backbone'])
    initial_identity = geometry.model_identity(network)
    x, target, availability, valid = [torch.from_numpy(np.stack([s[i] for s in samples])).float() for i in range(4)]
    before = network.head[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(network.parameters(),lr=RECIPE['learning_rate'],foreach=False)
    network.train(); prediction = network(x)
    need(list(prediction.shape)==[2,2,64,64],'Refiner output shape differs')
    loss = geometry.masked_heatmap_loss(prediction,target,availability,valid)
    need(torch.isfinite(loss).item(),'Nonfinite model smoke loss')
    loss.backward(); optimizer.step()
    need(not torch.equal(before,network.head[-1].weight.detach()),'Model smoke did not update parameters')
    probe = torch.randn(1,2,64,64,requires_grad=True)
    zero = torch.zeros_like(probe)
    a=torch.tensor([[1.,0.]]); v=torch.ones(1,64,64)
    geometry.masked_heatmap_loss(probe,zero,a,v).backward()
    need(probe.grad[:,1].abs().sum().item()==0,'Unavailable endpoint contributes a gradient')
    receipt={'status':'passed','device':'cpu','training_result':False,
        'description':'Two real training images, annotation-box mechanics fixtures; one disposable optimization step. Model is reinitialized before experiment training.',
        'image_ids':[s[4] for s in samples],'shape':list(prediction.shape),'finite_loss':float(loss.item()),
        'parameters_changed':True,'unavailable_channel_gradient_zero':True,'backbone_sha256':BACKBONE_SHA,
        'initial_model':initial_identity,
        'torch':str(torch.__version__),'torchvision':md.version('torchvision')}
    path=context['work']/'model-smoke.json'
    if path.exists():
        old=read(path)
        need(old['status']=='passed' and old['backbone_sha256']==BACKBONE_SHA and old['torch']==str(torch.__version__)
             and old['torchvision']==md.version('torchvision')
             and old['initial_model']==initial_identity, 'Model smoke runtime or initialization differs')
    else: frozen(path,receipt)
    print('Actual pretrained CPU model check passed: two training images, finite loss, parameter update and zero unavailable-corner gradient.',flush=True)
    del network, optimizer, prediction, loss
    return receipt


def check_runtime(device, previous=None):
    import torch
    from partial_classification_r51 import verify_native_sources
    for name,expected in r53.PINNED_PACKAGES.items():need(md.version(name)==expected,'Package differs: '+name)
    native=verify_native_sources()
    pkg=Path(importlib.util.find_spec('ultralytics').origin).parent
    for relative,expected in r53.INFERENCE_SOURCE_HASHES.items():r50.checked(pkg/relative,expected)
    current={'torch':str(torch.__version__),'torchvision':md.version('torchvision'),
        'python':platform.python_version(),'device':'cpu' if device=='cpu' else 'cuda:'+device,
        'gpu':None if device=='cpu' else torch.cuda.get_device_name(int(device)),
        'cuda':torch.version.cuda,'packages':{n:md.version(n) for n in r53.PINNED_PACKAGES},
        'native_sources':native}
    if previous is not None:
        need(all(previous[key]==value for key,value in current.items()),'Resume requires the same runtime, device and package sources')
    return current


def detector_outputs(context, device):
    import numpy as np
    import cv2
    import torch
    from ultralytics import YOLO
    from pose_checkpoint import check_loaded_pose_model
    runtime_setup(device)
    runtime=check_runtime(device)
    model=YOLO(str(context['weights'])); check_loaded_pose_model(model)
    for _ in range(2): model.predict(np.zeros((720,1280,3),dtype=np.uint8),device=device,conf=.13,augment=False,**INFERENCE)
    need(not model.predictor.model.end2end and not model.predictor.model.fp16,'Unexpected detector backend')
    need(str(model.predictor.device)==runtime['device'],'Detector device differs')
    runtime['checkpoint_sha256']=sha(context['weights'])
    outputs={}
    by_id={r['id']:r for r in context['stage']['records']}
    for identity in sorted(context['records']):
        image=cv2.imread(str(context['work']/by_id[identity]['image']))
        result=model.predict(image,device=device,conf=.13,augment=False,**INFERENCE)[0]
        raw=_prediction_records(result); kept,audit=postclip_nms(raw)
        outputs[identity]={'raw':raw,'postclip':kept,'filter':audit}
    gate=r53.baseline_equivalence(context['refs'],context['prior'],outputs)
    frozen(context['work']/'baseline-equivalence.json',gate)
    frozen(context['work']/'detector-evaluation.json',outputs)
    frozen(context['work']/'detector-runtime.json',runtime)
    if not gate['passed']:
        return outputs,None,runtime
    train={}
    for count,row in enumerate(sorted((r for r in by_id.values() if r['split']=='train'),key=lambda r:r['id']),1):
        image=cv2.imread(str(context['work']/row['image']))
        result=model.predict(image,device=device,conf=.13,augment=False,**INFERENCE)[0]
        raw=_prediction_records(result); kept,audit=postclip_nms(raw)
        train[row['id']]={'raw':raw,'postclip':kept,'filter':audit}
        if count%20==0: print(f'Frozen detector training images: {count}/142',flush=True)
    frozen(context['work']/'detector-training.json',train)
    del model
    if device!='cpu': torch.cuda.empty_cache()
    receipt={'preflight_sha256':sha(context['work']/'preflight.json'),'runtime':runtime,
             'files':{name:sha(context['work']/name) for name in ('detector-evaluation.json','detector-training.json','baseline-equivalence.json','detector-runtime.json')}}
    frozen(context['work']/'detector-receipt.json',receipt)
    return outputs,train,runtime


def restored_detector(context,device):
    path=context['work']/'detector-receipt.json'
    if not path.exists(): return detector_outputs(context,device)
    receipt=read(path)
    need(receipt['preflight_sha256']==sha(context['work']/'preflight.json'),'Detector cache input identity differs')
    for name,expected in receipt['files'].items(): checked(context['work']/name,expected)
    check_runtime(device,receipt['runtime'])
    outputs,train=read(context['work']/'detector-evaluation.json'),read(context['work']/'detector-training.json')
    need(read(context['work']/'baseline-equivalence.json')==r53.baseline_equivalence(context['refs'],context['prior'],outputs),'Cached baseline replay differs')
    need(read(context['work']/'baseline-equivalence.json')['passed'],'Cached detector baseline failed')
    return outputs,train,receipt['runtime']


def training_crops(context, predictions, save=True):
    records=[]; accounting=Counter(); image_counts={}
    for entry in sorted(context['stage']['records'],key=lambda x:x['id']):
        if entry['split']!='train': continue
        image=entry['id']; gold=parse_pose_labels((context['work']/entry['label']).read_text(),1280,720,image)
        output=predictions[image]['postclip']; score=score_image(gold,output,1280,720,.5,.5)
        accounting.update(images=1,reference_fronts=len(gold),predictions=len(output),matched=score['tp'],missed=score['fn'],unmatched=score['fp'])
        kept=0
        for match in score['matches']:
            gi,pi=match['gold_index'],match['prediction_index']; proposal=output[pi]
            geom=geometry.crop_geometry(proposal['bbox_xyxy'])
            _,availability,states=geometry.encode_targets(gold[gi]['keypoints'],geom)
            accounting.update(states)
            if not availability.any() or not geometry.valid_grid(geom).any():
                accounting['matched_crops_without_usable_supervision']+=1;continue
            records.append({'identity':f'{image}:p{pi}','image_id':image,'image_path':entry['image'],
                'reference_index':gi,'prediction_index':pi,'prediction':proposal,
                'geometry':geom,'reference_keypoints':gold[gi]['keypoints'],'availability':availability.tolist()})
            kept+=1
        image_counts[image]=kept
    need(accounting['images']==142 and accounting['reference_fronts']==3999 and len(records)>0,'Training crop inventory differs')
    result={'records':records,'counts':dict(accounting),'usable_crops':len(records),'usable_crops_by_photo':image_counts,
        'input_predictions_sha256':r50.sha(encoded(predictions))}
    if save: frozen(context['work']/'training-crops.json',result)
    return result


def trained_checkpoint(context,crops,device):
    import numpy as np
    from PIL import Image
    import torch
    runtime_setup(device)
    net=geometry.build_model(context['backbone']).to(device if device=='cpu' else 'cuda:'+device)
    need(geometry.model_identity(net)==read(context['work']/'model-smoke.json')['initial_model'],
         'Training architecture or initialization differs from the CPU smoke')
    optimizer=torch.optim.AdamW(net.parameters(),lr=RECIPE['learning_rate'],weight_decay=RECIPE['weight_decay'],foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=RECIPE['minimum_learning_rate'])
    work=context['work']; ckpt=work/'last-training.pt'; metadata=work/'last-training.json'
    contract_sha=sha(work/'preflight.json'); crop_sha=sha(work/'training-crops.json')
    start=0; history=[]
    if ckpt.exists() or metadata.exists():
        need(ckpt.exists() and metadata.exists(),'Incomplete training checkpoint preserved; use a fresh work directory')
        receipt=read(metadata);checked(ckpt,receipt['sha256'])
        need(receipt['preflight_sha256']==contract_sha and receipt['crops_sha256']==crop_sha,'Training resume inputs differ')
        need(receipt['runtime']==read(work/'detector-runtime.json'),'Training resume runtime differs')
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        need(saved['recipe']==RECIPE and saved['hint_mode']==context['hint_mode'],'Resume recipe differs')
        net.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):state[key]=value.to(next(net.parameters()).device)
        start=saved['epoch'];history=saved['history']
        torch.set_rng_state(saved['torch_rng'])
        if device!='cpu':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    data=crops['records']; images={}
    for record in data:
        key=record['image_id']
        if key not in images:
            with Image.open(work/record['image_path']) as im:images[key]=im.convert('RGB')
    updates=sum(h['updates'] for h in history)
    for epoch in range(start+1,EPOCHS+1):
        generator=torch.Generator().manual_seed(SEED+epoch)
        order=torch.randperm(len(data),generator=generator).tolist()
        net.train();loss_total=0.;count=0;steps=0;started=time.perf_counter()
        for begin in range(0,len(order),BATCH):
            batch=[data[i] for i in order[begin:begin+BATCH]]
            inputs=[];targets=[];available=[];valid=[]
            for item in batch:
                inputs.append(geometry.crop_tensor(images[item['image_id']],item['prediction'],item['geometry'],context['hint_mode'],epoch,item['identity'],SEED))
                target,avail,_=geometry.encode_targets(item['reference_keypoints'],item['geometry'])
                targets.append(target);available.append(avail);valid.append(geometry.valid_grid(item['geometry']))
            x,t,a,v=[torch.from_numpy(np.stack(values)).float().to(next(net.parameters()).device) for values in (inputs,targets,available,valid)]
            optimizer.zero_grad(set_to_none=True);out=net(x)
            loss=geometry.masked_heatmap_loss(out,t,a,v)
            need(torch.isfinite(loss).item(),'Nonfinite training loss; current completed epoch preserved')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),RECIPE['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();loss_total+=float(loss.item())*len(batch);count+=len(batch);steps+=1
        scheduler.step();updates+=steps
        history.append({'epoch':epoch,'samples':count,'updates':steps,'mean_loss':loss_total/count,
            'next_learning_rate':scheduler.get_last_lr()[0],'seconds':time.perf_counter()-started,
            'sample_order_sha256':r50.sha(encoded([data[i]['identity'] for i in order]))})
        saved={'model':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'recipe':RECIPE,'hint_mode':context['hint_mode'],
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device!='cpu' else []}
        tmp=ckpt.with_suffix('.part');torch.save(saved,tmp);tmp.replace(ckpt)
        receipt={'sha256':sha(ckpt),'preflight_sha256':contract_sha,'crops_sha256':crop_sha,'completed_epoch':epoch,
                 'runtime':read(work/'detector-runtime.json')}
        temp=metadata.with_suffix('.part');temp.write_bytes(encoded(receipt));temp.replace(metadata)
        print(f'Refiner {context["hint_mode"]}: epoch {epoch}/{EPOCHS}, loss {loss_total/count:.6g}, updates {updates}',flush=True)
    need(len(history)==EPOCHS and [h['epoch'] for h in history]==list(range(1,EPOCHS+1)),'Training history incomplete')
    final=work/'refiner-final.pt'
    if not final.exists():torch.save({'model':{k:v.detach().cpu() for k,v in net.state_dict().items()},'recipe':RECIPE,'hint_mode':context['hint_mode'],'backbone_sha256':BACKBONE_SHA},final)
    final_saved=torch.load(final,map_location='cpu',weights_only=True)
    need(final_saved['recipe']==RECIPE and final_saved['hint_mode']==context['hint_mode'],'Final checkpoint identity differs')
    need(all(torch.equal(value.detach().cpu(),final_saved['model'][key]) for key,value in net.state_dict().items()),'Final checkpoint differs from completed training')
    frozen(work/'training-log.json',history)
    frozen(work/'training-receipt.json',{'epochs':EPOCHS,'samples':len(data)*EPOCHS,'updates':updates,
        'expected_updates':math.ceil(len(data)/BATCH)*EPOCHS,'hint_mode':context['hint_mode'],
        'model_sha256':sha(final),'model_bytes':final.stat().st_size,'training_crops_sha256':crop_sha,
        'preflight_sha256':contract_sha,'selection':'fixed final epoch','training_log_sha256':sha(work/'training-log.json')})
    net.load_state_dict(final_saved['model'],strict=True);net.eval()
    return net


def refine_outputs(context,network,control):
    import numpy as np
    from PIL import Image
    import torch
    result={};details=[];device=next(network.parameters()).device
    by_id={r['id']:r for r in context['stage']['records']}
    with torch.inference_mode():
        for identity,record in sorted(control.items()):
            with Image.open(context['work']/by_id[identity]['image']) as original: image=original.convert('RGB')
            raw=[]
            for index,prediction in enumerate(record['raw']):
                geom=geometry.crop_geometry(prediction['bbox_xyxy'])
                array=geometry.crop_tensor(image,prediction,geom,context['hint_mode'],identity=f'{identity}:p{index}')
                heatmaps=network(torch.from_numpy(array[None]).to(device))[0].cpu().numpy()
                decoded,trace=geometry.decode_heatmaps(heatmaps,geom,with_trace=True)
                raw.append(geometry.replace_points(prediction,decoded))
                details.append({'image_id':identity,'raw_index':index,'geometry':geom,'decoded_points':decoded,
                    'decoder_trace':trace,
                    'heatmap_maxima':[float(h.max()) if np.isfinite(h).all() else None for h in heatmaps]})
            kept,audit=postclip_nms(raw)
            result[identity]={'raw':raw,'postclip':kept,'filter':audit}
    return result,details


def frozen_detection_check(control,treatment):
    need(set(control)==set(treatment),'Image membership changed')
    for image,a in control.items():
        b=treatment[image]
        for view in ('raw','postclip'):
            need(len(a[view])==len(b[view]),'Proposal count changed')
            for x,y in zip(a[view],b[view]):
                need({k:v for k,v in x.items() if k!='keypoints'}=={k:v for k,v in y.items() if k!='keypoints'},'Detector record changed')
                need([p[2] for p in x['keypoints']]==[p[2] for p in y['keypoints']],'Original point score changed')
        need(a['filter']['retained_primary_prediction_ids']==b['filter']['retained_primary_prediction_ids'],'Suppression identity changed')
    return True


def replay_transforms(control,treatment,details):
    expected={(image,index) for image,row in control.items() for index in range(len(row['raw']))}
    observed=set()
    for item in details:
        key=(item['image_id'],item['raw_index'])
        need(key in expected and key not in observed,'Refinement record membership differs')
        observed.add(key); image,index=key
        original=control[image]['raw'][index]
        need(item['geometry']==geometry.crop_geometry(original['bbox_xyxy']),'Crop transform replay differs')
        need(len(item['decoder_trace'])==2,'Two endpoint decoder traces required')
        decoded=[geometry.replay_decoding(trace,item['geometry']) for trace in item['decoder_trace']]
        need(decoded==item['decoded_points'],'Decoder inverse-coordinate replay differs')
        need(treatment[image]['raw'][index]==geometry.replace_points(original,item['decoded_points']),
             'Saved decoded coordinates differ from prediction records')
    need(observed==expected,'Some raw detector records were not refined')


def replay_postclip(outputs):
    for row in outputs.values():
        kept,audit=postclip_nms(row['raw'])
        need(kept==row['postclip'] and audit==row['filter'],'Postclip suppression replay differs')


def blocked_package(context,control):
    """Preserve a failed-control diagnostic and never start experiment training."""
    work=context['work'];gate=read(work/'baseline-equivalence.json')
    need(not gate['passed'],'Blocked bundle requires a failed baseline')
    summary={'schema':'baomai-local-refiner-r58-results-v1','status':'blocked_baseline',
        'contract':context['contract'],'hint_mode':context['hint_mode'],
        'reports':{'control':r53.reports(context['refs'],control)},'baseline_passed':False,
        'training':False,'refinement_executed':False,'interpretation_allowed':False,
        'automatic_promotion':False,'development_only':True}
    frozen(work/'summary.json',summary)
    names=['summary.json','preflight.json','input-audit.json','staging-r58.json',
           'baseline-equivalence.json','detector-evaluation.json','detector-runtime.json','model-smoke.json']
    entries={name:(work/name).read_bytes() for name in names}
    entries['run_corner_refinement_r58.py']=Path(__file__).read_bytes()
    entries['corner_refinement_r58.py']=Path(geometry.__file__).read_bytes()
    entries['inventory.json']=encoded({n:{'bytes':len(data),'sha256':r50.sha(data)} for n,data in entries.items()})
    evidence=work/f'baomai-corner-refinement-r58-{context["hint_mode"]}-results.zip'
    with zipfile.ZipFile(evidence,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):z.writestr(name,data)
    manifest={'status':'blocked_baseline','model_file':None,'model_sha256':None,'model_bytes':0,'parts':[],
              'evidence_file':evidence.name,'evidence_sha256':sha(evidence)}
    frozen(work/'return-manifest.json',manifest)
    bundle=work/f'baomai-corner-refinement-r58-{context["hint_mode"]}-upload.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_STORED) as z:
        z.write(evidence,evidence.name);z.write(work/'return-manifest.json','return-manifest.json')
    return verify(context)


def pair_accounting(refs,arms):
    result={}
    for ref,gold in refs.items():
        result[ref]={}
        for arm,images in arms.items():
            counts=Counter({'offered_pairs':0,'confirmed_within5':0,'known_pair_errors_over5':0,'matched_incomplete_reference':0,'unmatched_offered_pairs':0})
            for image,targets in gold.items():
                output=images[image]['postclip'];score=score_image(targets,output,1280,720,.5,.5)
                matches={m['prediction_index']:m['gold_index'] for m in score['matches']}
                for pi,prediction in enumerate(output):
                    if not all(geometry.point_available(p) for p in prediction['keypoints']):continue
                    counts['offered_pairs']+=1
                    gi=matches.get(pi)
                    if gi is None:counts['unmatched_offered_pairs']+=1;continue
                    target=targets[gi]
                    if not all(p[2]==2 for p in target['keypoints']):counts['matched_incomplete_reference']+=1;continue
                    worst=max(math.hypot(p[0]-g[0],p[1]-g[1]) for p,g in zip(prediction['keypoints'],target['keypoints']))
                    counts['confirmed_within5' if worst<=5 else 'known_pair_errors_over5']+=1
            need(counts['offered_pairs']==sum(counts[k] for k in counts if k!='offered_pairs'),'Pair accounting differs')
            result[ref][arm]=dict(counts)
    return {'matching_convention':'Match all fixed detector boxes first, then account for offered pairs under those same assignments. Accepted subsets are not rematched.',
            'scores_recalibrated':False,'references':result}


def ensure_upload_bundle(work,manifest,hint_mode):
    """Finish an interrupted outer packaging step from verified inner members."""
    paths=[work/manifest['evidence_file'],work/'return-manifest.json',*[work/p['file'] for p in manifest['parts']]]
    expected={p.name:sha(p) for p in paths}
    bundle=work/f'baomai-corner-refinement-r58-{hint_mode}-upload.zip'
    valid=False
    if bundle.exists():
        try:
            with zipfile.ZipFile(bundle) as z:
                valid=(len(z.namelist())==len(expected) and set(z.namelist())==set(expected)
                       and z.testzip() is None and all(r50.sha(z.read(n))==h for n,h in expected.items()))
        except (OSError,zipfile.BadZipFile):pass
    if not valid:
        temp=bundle.with_suffix('.part')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_STORED) as z:
            for path in paths:z.write(path,path.name)
        temp.replace(bundle)
    return bundle


def package(context,arms,details):
    work=context['work'];frozen_detection_check(arms['control'],arms['treatment'])
    scored={arm:r53.reports(context['refs'],output) for arm,output in arms.items()}
    summary={'schema':'baomai-local-refiner-r58-results-v1','status':'completed','contract':context['contract'],
        'hint_mode':context['hint_mode'],'reports':scored,'frozen_detection_records':True,
        'training':True,'fresh_inference':True,'automatic_promotion':False,'development_only':True,
        'refinement_executed':True,'interpretation_allowed':True,
        'training_receipt':read(work/'training-receipt.json'),'baseline_passed':read(work/'baseline-equivalence.json')['passed'],
        'original_corner_scores_reused':True,'scores_recalibrated':False}
    content={'summary.json':summary,'predictions.json':arms,'crop-transforms.json':details,
        'cases.json':r53.case_inventory(context['refs'],context['maps'],arms,scored),
        'pair-accounting.json':pair_accounting(context['refs'],arms)}
    for name,obj in content.items():frozen(work/name,obj)
    names=list(content)+['preflight.json','input-audit.json','staging-r58.json','detector-evaluation.json',
        'detector-training.json','detector-runtime.json','detector-receipt.json','baseline-equivalence.json',
        'training-crops.json','training-log.json','training-receipt.json','model-smoke.json']
    entries={n:(work/n).read_bytes() for n in names}
    entries['run_corner_refinement_r58.py']=Path(__file__).read_bytes()
    entries['corner_refinement_r58.py']=Path(geometry.__file__).read_bytes()
    entries['inventory.json']=encoded({n:{'bytes':len(data),'sha256':r50.sha(data)} for n,data in entries.items()})
    evidence=work/f'baomai-corner-refinement-r58-{context["hint_mode"]}-results.zip'
    with zipfile.ZipFile(evidence.with_suffix('.part'),'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):z.writestr(name,data)
    evidence.with_suffix('.part').replace(evidence)
    model=work/'refiner-final.pt';payload=model.read_bytes();parts=[]
    for index,start in enumerate(range(0,len(payload),16*1024*1024)):
        chunk=payload[start:start+16*1024*1024];member=f'refiner-final.pt.part{index:02d}'
        path=work/(member+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(member,chunk)
        parts.append({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path),
            'member':member,'member_bytes':len(chunk),'member_sha256':r50.sha(chunk)})
    manifest={'status':'completed','model_file':'refiner-final.pt','model_sha256':sha(model),'model_bytes':len(payload),'parts':parts,
        'evidence_file':evidence.name,'evidence_sha256':sha(evidence),
        'instructions':'Read parts in this listed order, verify each wrapper and member, concatenate member bytes, then verify the complete model SHA256.'}
    frozen(work/'return-manifest.json',manifest)
    bundle=work/f'baomai-corner-refinement-r58-{context["hint_mode"]}-upload.zip'
    with zipfile.ZipFile(bundle.with_suffix('.part'),'w',zipfile.ZIP_STORED) as z:
        for path in [evidence,work/'return-manifest.json',*[work/p['file'] for p in parts]]:z.write(path,path.name)
    bundle.with_suffix('.part').replace(bundle)
    return verify(context)


def verify(context):
    work=context['work'];manifest=read(work/'return-manifest.json')
    need(manifest['status'] in ('completed','blocked_baseline'),'Unknown return status')
    checked(work/manifest['evidence_file'],manifest['evidence_sha256'])
    model_hash=hashlib.sha256();model_bytes=0
    for part in manifest['parts']:
        path=checked(work/part['file'],part['sha256'])
        with zipfile.ZipFile(path) as z:
            need(z.namelist()==[part['member']] and z.testzip() is None,'Checkpoint part membership/CRC differs')
            data=z.read(part['member']);need(len(data)==part['member_bytes'] and r50.sha(data)==part['member_sha256'],'Checkpoint part bytes differ')
            model_hash.update(data);model_bytes+=len(data)
    if manifest['status']=='completed':
        need(model_bytes==manifest['model_bytes'] and model_hash.hexdigest()==manifest['model_sha256'],'Reassembled checkpoint differs')
    else:
        need(not manifest['parts'] and manifest['model_sha256'] is None and manifest['model_bytes']==0,
             'Blocked experiment contains a model result')
    with zipfile.ZipFile(work/manifest['evidence_file']) as z:
        inventory=json.loads(z.read('inventory.json'))
        need(set(z.namelist())==set(inventory)|{'inventory.json'} and len(z.namelist())==len(set(z.namelist())) and z.testzip() is None,'Evidence membership/CRC differs')
        for name,item in inventory.items():
            data=z.read(name);need(item=={'bytes':len(data),'sha256':r50.sha(data)},'Evidence member hash differs: '+name)
        need(z.read('run_corner_refinement_r58.py')==Path(__file__).read_bytes() and z.read('corner_refinement_r58.py')==Path(geometry.__file__).read_bytes(),'Saved implementation differs')
        summary=json.loads(z.read('summary.json'))
        need(summary['contract']==context['contract'],'Input contract differs')
        need(summary['status']==manifest['status'],'Result status differs')
        if manifest['status']=='blocked_baseline':
            control=json.loads(z.read('detector-evaluation.json'));replay_postclip(control)
            gate=r53.baseline_equivalence(context['refs'],context['prior'],control)
            need(not gate['passed'] and gate==json.loads(z.read('baseline-equivalence.json')),'Blocked baseline replay differs')
            need(summary['reports']=={'control':r53.reports(context['refs'],control)},'Blocked score replay differs')
            need(not summary['training'] and not summary['refinement_executed'] and not summary['interpretation_allowed'],
                 'Failed baseline must block treatment interpretation')
            ensure_upload_bundle(work,manifest,context['hint_mode'])
            print('Verified failed-control diagnostic; no experiment training or refinement.',flush=True)
            return summary
        arms=json.loads(z.read('predictions.json'))
        need(arms['control']==json.loads(z.read('detector-evaluation.json')),'Control cache identity differs')
        frozen_detection_check(arms['control'],arms['treatment'])
        for output in arms.values():replay_postclip(output)
        replay_transforms(arms['control'],arms['treatment'],json.loads(z.read('crop-transforms.json')))
        gate=r53.baseline_equivalence(context['refs'],context['prior'],arms['control'])
        need(gate['passed'] and gate==json.loads(z.read('baseline-equivalence.json')),'Baseline replay differs')
        scored={arm:r53.reports(context['refs'],output) for arm,output in arms.items()}
        need(scored==summary['reports'],'Score replay differs')
        need(json.loads(z.read('cases.json'))==r53.case_inventory(context['refs'],context['maps'],arms,scored),'Case replay differs')
        need(json.loads(z.read('pair-accounting.json'))==pair_accounting(context['refs'],arms),'Pair-accounting replay differs')
        receipt=json.loads(z.read('training-receipt.json'));history=json.loads(z.read('training-log.json'))
        crops=json.loads(z.read('training-crops.json'))
        need(crops==training_crops(context,json.loads(z.read('detector-training.json')),save=False),'Training supervision replay differs')
        need(receipt['training_log_sha256']==r50.sha(z.read('training-log.json'))
             and receipt['training_crops_sha256']==r50.sha(z.read('training-crops.json'))
             and receipt['preflight_sha256']==r50.sha(z.read('preflight.json')),'Training receipt input identity differs')
        need(receipt['epochs']==EPOCHS and receipt['updates']==receipt['expected_updates'] and receipt['updates']==sum(h['updates'] for h in history),'Training budget differs')
        need([h['epoch'] for h in history]==list(range(1,EPOCHS+1)) and receipt['samples']==len(crops['records'])*EPOCHS,
             'Fixed-epoch sample exposure differs')
        need(all(h['samples']==len(crops['records']) and h['updates']==math.ceil(len(crops['records'])/BATCH) for h in history),
             'Per-epoch training exposure differs')
        need(receipt['expected_updates']==math.ceil(len(crops['records'])/BATCH)*EPOCHS,'Expected optimizer updates differ')
        import torch
        for epoch in history:
            order=torch.randperm(len(crops['records']),generator=torch.Generator().manual_seed(SEED+epoch['epoch'])).tolist()
            expected=r50.sha(encoded([crops['records'][i]['identity'] for i in order]))
            need(expected==epoch['sample_order_sha256'],'Training sample order differs')
        need(receipt['model_sha256']==manifest['model_sha256'] and receipt['model_bytes']==manifest['model_bytes'],'Model identity differs')
    ensure_upload_bundle(work,manifest,context['hint_mode'])
    print('Verified saved score replay and exact model parts.',flush=True)
    return summary


def run(context,device):
    if (context['work']/'return-manifest.json').exists():return verify(context)
    runtime_setup(device);model_smoke(context)
    control,training,_=restored_detector(context,device)
    if training is None:return blocked_package(context,control)
    crops=training_crops(context,training)
    network=trained_checkpoint(context,crops,device)
    refined,details=refine_outputs(context,network,control)
    return package(context,{'control':control,'treatment':refined},details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','smoke','run','verify'))
    parser.add_argument('--source-commit',required=True);parser.add_argument('--script-sha256',required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--archive',type=Path);parser.add_argument('--originals-dir',type=Path)
    parser.add_argument('--hint-mode',choices=('none','initial'),required=True);parser.add_argument('--device',default='0')
    args=parser.parse_args();context=source_and_inputs(args)
    if args.action=='prepare':print('Verified 142 training photographs, 20 evaluation photographs, detector checkpoint and ImageNet backbone. No model run.',flush=True)
    elif args.action=='smoke':print(json.dumps(model_smoke(context),indent=2))
    elif args.action=='run':run(context,args.device)
    else:verify(context)

if __name__=='__main__':main()
