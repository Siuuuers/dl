"""Check coordinate frames, masked supervision, fixed detections and audit recovery."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from PIL import Image
import corner_refinement_r58 as geometry
import run_corner_refinement_r58 as run
import build_corner_refinement_r58_notebooks as notebook
from predict_pose_folder import postclip_nms


def record(box=(20.,30.,80.,50.), points=None, confidence=.9):
    return {'class_id':0,'confidence':confidence,'bbox_xyxy':list(box),
            'keypoints':points if points is not None else [[20.,50.,.8],[80.,50.,.9]]}


def output(raw):
    kept,audit=postclip_nms(raw)
    return {'raw':raw,'postclip':kept,'filter':audit}


class GeometryTest(unittest.TestCase):
    def test_clipped_crop_uses_exact_rounded_scales_and_inverse(self):
        for box in ([0.2,5.7,83.3,19.4],[1250.1,700.2,1279.9,719.9],[240.2,180.7,423.4,291.1]):
            transform=geometry.crop_geometry(box)
            for point in ((box[0],box[1]),(box[2]-.01,box[3]-.01),((box[0]+box[2])/2,(box[1]+box[3])/2)):
                np.testing.assert_allclose(geometry.inverse_point(geometry.forward_point(point,transform),transform),point,atol=1e-12,rtol=0)
            left,top,right,bottom=transform['xyxy'];rw,rh=transform['resized_wh']
            self.assertEqual(transform['scale_xy'],[rw/(right-left),rh/(bottom-top)])
        self.assertEqual(geometry.crop_geometry([0,0,40,20])['xyxy'],[0,0,50,25])

    def test_target_decoder_mapping_has_known_subcell_error_bound(self):
        transform=geometry.crop_geometry([100,100,164,164])
        points=[[115.3,151.7,2],[154.1,150.2,2]]
        targets,available,states=geometry.encode_targets(points,transform)
        self.assertEqual(states,['supervised','supervised'])
        self.assertEqual(available.tolist(),[1.,1.])
        decoded,trace=geometry.decode_heatmaps(targets,transform,with_trace=True)
        self.assertEqual(decoded,[geometry.replay_decoding(item,transform) for item in trace])
        for original,recovered in zip(points,decoded):
            error=np.abs(np.asarray(geometry.forward_point(original,transform))-geometry.forward_point(recovered,transform))
            self.assertTrue(np.all(error<=1.+1e-5),error)
        changed=deepcopy(trace[0]);changed['quarter_offset_xy'][0]+=.25
        with self.assertRaises(ValueError):geometry.replay_decoding(changed,transform)

    def test_padding_cannot_win_and_unknown_or_outside_corners_are_masked(self):
        transform=geometry.crop_geometry([100,100,250,115])
        valid=geometry.valid_grid(transform)
        self.assertFalse(valid.all())
        target,available,states=geometry.encode_targets([[130,110,2],[0,0,0]],transform)
        altered=target.copy();altered[:,~valid]=1e9
        self.assertEqual(geometry.decode_heatmaps(target,transform)[0],geometry.decode_heatmaps(altered,transform)[0])
        self.assertEqual(available.tolist(),[1.,0.]);self.assertFalse(target[1].any())
        outside,available,states=geometry.encode_targets([[1279,719,2],[0,0,0]],transform)
        self.assertEqual(states,['outside_predicted_crop','unavailable'])
        self.assertFalse(available.any());self.assertFalse(outside.any())

    def test_nonpositive_or_nonfinite_heatmap_abstains_without_changing_scores(self):
        transform=geometry.crop_geometry([100,100,200,200])
        heatmaps=np.zeros((2,64,64),dtype='float32');heatmaps[1,10,10]=np.nan
        decoded=geometry.decode_heatmaps(heatmaps,transform)
        self.assertEqual(decoded,[None,None])
        original=record();refined=geometry.replace_points(original,decoded)
        self.assertEqual(refined['keypoints'],[[None,None,.8],[None,None,.9]])
        self.assertEqual(refined['bbox_xyxy'],original['bbox_xyxy'])
        self.assertFalse(geometry.point_available(refined['keypoints'][0]))

    def test_hint_ablation_preserves_rgb_and_deterministic_example_noise(self):
        image=Image.fromarray(np.random.default_rng(8).integers(0,256,(720,1280,3),dtype='uint8'))
        prediction=record();transform=geometry.crop_geometry(prediction['bbox_xyxy'])
        plain=geometry.crop_tensor(image,prediction,transform,'none',epoch=3,identity='photo:p1')
        hinted=geometry.crop_tensor(image,prediction,transform,'initial',epoch=3,identity='photo:p1')
        again=geometry.crop_tensor(image,prediction,transform,'initial',epoch=3,identity='photo:p1')
        other=geometry.crop_tensor(image,prediction,transform,'initial',epoch=4,identity='photo:p1')
        np.testing.assert_array_equal(plain[:3],hinted[:3]);self.assertFalse(plain[3:].any())
        self.assertTrue(hinted[3:].any());np.testing.assert_array_equal(hinted,again)
        self.assertFalse(np.array_equal(hinted[3:],other[3:]))
        px,py=transform['pad_xy'];rw,rh=transform['resized_wh']
        self.assertFalse(hinted[3:,:py].any());self.assertFalse(hinted[3:,py+rh:].any())
        self.assertFalse(hinted[3:,:,:px].any());self.assertFalse(hinted[3:,:,px+rw:].any())

    def test_unavailable_endpoint_and_padding_have_exactly_zero_gradient(self):
        try:import torch
        except ImportError:self.skipTest('Torch is optional for pure geometry checks')
        prediction=torch.randn(1,2,64,64,requires_grad=True)
        target=torch.zeros_like(prediction);availability=torch.tensor([[1.,0.]])
        valid=torch.zeros(1,64,64);valid[:,8:56,4:60]=1
        run.geometry.masked_heatmap_loss(prediction,target,availability,valid).backward()
        self.assertEqual(prediction.grad[:,1].abs().sum().item(),0.)
        self.assertEqual(prediction.grad[:,0,:8].abs().sum().item(),0.)
        self.assertGreater(prediction.grad[:,0,8:56,4:60].abs().sum().item(),0.)


class AuditTest(unittest.TestCase):
    def test_fixed_box_score_and_suppression_identity_reject_mutation(self):
        a={'scene':output([record(),record(confidence=.8)])}
        b=deepcopy(a)
        b['scene']=output([geometry.replace_points(p,[[22,49],[79,51]]) for p in a['scene']['raw']])
        self.assertTrue(run.frozen_detection_check(a,b));run.replay_postclip(b)
        for mutation in ('box','score','point_score'):
            changed=deepcopy(b)
            if mutation=='box':changed['scene']['raw'][0]['bbox_xyxy'][0]+=1
            elif mutation=='score':changed['scene']['raw'][0]['confidence']-=.1
            else:changed['scene']['raw'][0]['keypoints'][0][2]-=.1
            with self.assertRaises(RuntimeError):run.frozen_detection_check(a,changed)

    def test_all_raw_records_require_replayable_transforms(self):
        control={'scene':output([record(),record(confidence=.8)])};details=[];raw=[]
        for index,prediction in enumerate(control['scene']['raw']):
            transform=geometry.crop_geometry(prediction['bbox_xyxy'])
            target,_,_=geometry.encode_targets([[22,49,2],[79,51,2]],transform)
            points,trace=geometry.decode_heatmaps(target,transform,with_trace=True)
            raw.append(geometry.replace_points(prediction,points))
            details.append({'image_id':'scene','raw_index':index,'geometry':transform,
                            'decoded_points':points,'decoder_trace':trace})
        treatment={'scene':output(raw)};run.replay_transforms(control,treatment,details)
        with self.assertRaises(RuntimeError):run.replay_transforms(control,treatment,details[:1])
        wrong=deepcopy(details);wrong[0]['geometry']['pad_xy'][0]+=1
        with self.assertRaises(RuntimeError):run.replay_transforms(control,treatment,wrong)

    def test_offered_pairs_account_for_incomplete_reference_and_unmatched_outputs(self):
        gold=[{'class_id':0,'bbox_xyxy':[20,30,80,50],'keypoints':[[20,50,2],[0,0,0]]}]
        raw=[record(),record(box=(300,300,360,320),points=[[300,320,.8],[360,320,.9]])]
        result=run.pair_accounting({'reference':{'scene':gold}},{'control':{'scene':output(raw)}})
        counts=result['references']['reference']['control']
        self.assertEqual(counts['offered_pairs'],2)
        self.assertEqual(counts['matched_incomplete_reference'],1)
        self.assertEqual(counts['unmatched_offered_pairs'],1)
        self.assertEqual(counts['confirmed_within5'],0)

    def test_failed_baseline_bundle_replays_and_recovers_interrupted_outer_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            gold=[{'class_id':0,'bbox_xyxy':[20,30,80,50],'keypoints':[[20,50,2],[80,50,2]]}]
            refs={'reference':{'scene':gold}};prior={'scene':output([record()])};control={'scene':output([])}
            gate=run.r53.baseline_equivalence(refs,prior,control)
            self.assertFalse(gate['passed'])
            context={'work':work,'contract':{'synthetic_test':True},'hint_mode':'none','refs':refs,'prior':prior}
            payloads={'baseline-equivalence.json':gate,'detector-evaluation.json':control}
            for name in ('preflight.json','input-audit.json','staging-r58.json','detector-runtime.json','model-smoke.json'):
                payloads[name]={'synthetic_test':True}
            for name,value in payloads.items():run.frozen(work/name,value)
            summary=run.blocked_package(context,control)
            self.assertFalse(summary['training']);self.assertFalse(summary['interpretation_allowed'])
            upload=work/'baomai-corner-refinement-r58-none-upload.zip';upload.unlink()
            run.verify(context);self.assertTrue(upload.is_file())
            with zipfile.ZipFile(upload) as z:self.assertEqual(len(z.namelist()),2)
            manifest=json.loads((work/'return-manifest.json').read_text())
            evidence=work/manifest['evidence_file'];evidence.write_bytes(evidence.read_bytes()+b'changed')
            with self.assertRaises(RuntimeError):run.verify(context)

    def test_notebooks_compile_and_isolate_only_hint_mode(self):
        outputs=[notebook.build('a'*40,'b'*64,mode) for mode in ('none','initial')]
        for content in outputs:
            codes=[c for c in content['cells'] if c['cell_type']=='code']
            self.assertEqual(len(codes),4)
            for code in codes:compile(''.join(code['source']),'<notebook-test>','exec')
            self.assertTrue(content['metadata']['training']);self.assertFalse(content['metadata']['automatic_promotion'])
        self.assertNotEqual(outputs[0]['metadata']['hint_mode'],outputs[1]['metadata']['hint_mode'])
        with self.assertRaises(ValueError):notebook.build('branch','b'*64,'none')


class ActualModelTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('R58_BACKBONE_WEIGHTS'),'Set R58_BACKBONE_WEIGHTS for the actual pretrained forward check')
    def test_same_initialized_network_ignores_hints_until_hint_weights_are_trained(self):
        import torch
        path=Path(os.environ['R58_BACKBONE_WEIGHTS'])
        self.assertEqual(run.sha(path),run.BACKBONE_SHA)
        run.runtime_setup('cpu');network=geometry.build_model(path).eval()
        prediction=record();transform=geometry.crop_geometry(prediction['bbox_xyxy'])
        image=Image.new('RGB',(1280,720),(63,80,91))
        arrays=[geometry.crop_tensor(image,prediction,transform,mode) for mode in ('none','initial')]
        with torch.inference_mode():actual=network(torch.from_numpy(np.stack(arrays)))
        self.assertEqual(list(actual.shape),[2,2,64,64]);self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[0],actual[1],rtol=0,atol=0)
        self.assertEqual(network.backbone[0].weight[:,3:].abs().sum().item(),0.)
        self.assertEqual(sum(p.numel() for p in network.parameters()),34002114)


if __name__=='__main__':unittest.main()
