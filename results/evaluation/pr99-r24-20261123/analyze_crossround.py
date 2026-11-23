#!/usr/bin/env python3
"""Reproduce the R23/R24 comparison on identical R24 references and original-pixel case sheets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, evidence, output = args.root.resolve(), args.evidence.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'scripts'))
    from front_review_r24 import experiment
    from evaluate_pose import parse_pose_labels
    from pose_scoring import _box_iou
    from PIL import Image, ImageDraw, ImageFont
    source = {
        'r23': evidence/'r23-same-runtime-comparison.json',
        'r24': evidence/'fixed-comparison.json',
    }
    reports = {name: read(path) for name, path in source.items()}
    prediction_paths = {
        'r23': evidence/'inference/eval-r23-candidate/predictions.jsonl',
        'r24': evidence/'inference/eval-candidate/predictions.jsonl',
    }
    headers, predictions = {}, {}
    for name, path in prediction_paths.items():
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        headers[name] = lines[0]
        predictions[name] = {line['image_id']: line['predictions'] for line in lines[1:]}
        assert reports[name]['models']['candidate']['predictions_sha256'] == digest(path)
    for field in ('runtime', 'inference', 'box_confidences'):
        assert headers['r23'][field] == headers['r24'][field], field
    assert reports['r23']['thresholds'] == reports['r24']['thresholds']
    summary = {
        'kind': 'R23-to-R24 matched-runtime comparison',
        'source_hashes': {name: digest(path) for name,path in source.items()},
        'script_sha256': digest(__file__),
        'runtime_and_settings_equal': True,
        'references': {},
    }
    for label, ref24 in reports['r24']['references'].items():
        ref23 = reports['r23']['references'][label]
        assert ref23['manifest_sha256'] == ref24['manifest_sha256']
        a,b = ref23['models']['candidate'],ref24['models']['candidate']
        per = {
            'reference_id': ref24['reference_id'],
            'reference_sha256': ref24['manifest_sha256'],
            'r23': {}, 'r24': {},
            'gained_matches': sorted(set(b['matched_identities'])-set(a['matched_identities'])),
            'lost_matches': sorted(set(a['matched_identities'])-set(b['matched_identities'])),
            'success_changes': {},
        }
        for name, model in [('r23',a),('r24',b)]:
            per[name] = {k:model['detection'][k] for k in ('tp','fn','fp','duplicate_fp','other_fp','returned_endpoints')}
            per[name]['corner_quality'] = model['corner_quality']
            per[name]['miss_free_nonempty_scenes'] = model['miss_free_nonempty_scenes']
            per[name]['empty_scene_outputs'] = model['empty_scene_outputs']
        for radius in ('5','10'):
            per['success_changes'][radius] = {
                'gained':sorted(set(b['success_identities'][radius])-set(a['success_identities'][radius])),
                'lost':sorted(set(a['success_identities'][radius])-set(b['success_identities'][radius])),
            }
        summary['references'][label] = per
    (_,_,_,_,_,_,reviewed) = experiment(root).read_review(root)
    gold = {iid:parse_pose_labels(reviewed[iid]['bytes'].decode(),1280,720) for iid in predictions['r23']}
    label='current_r24_exploratory'
    matches={name:{iid:{p['gold_index']:p['prediction_index'] for p in trace['matches']}
                   for iid,trace in report['references'][label]['matching_traces']['candidate'].items()}
             for name, report in reports.items()}
    changes=[]
    for iid in gold:
        for row in set(matches['r23'][iid]) & set(matches['r24'][iid]):
            for side,g in enumerate(gold[iid][row]['keypoints']):
                if g[2]!=2: continue
                pts={name:predictions[name][iid][matches[name][iid][row]]['keypoints'][side] for name in predictions}
                if not all(pt[2]>=.5 for pt in pts.values()): continue
                errors={name:math.dist(g[:2],pt[:2]) for name,pt in pts.items()}
                changes.append({'image_id':iid,'row':row+1,'side':('left','right')[side],
                                'gold':g,'predictions':pts,'r23_error':errors['r23'],'r24_error':errors['r24'],
                                'delta':errors['r24']-errors['r23'],'bbox_xyxy':gold[iid][row]['bbox_xyxy']})
    summary['same_locatable_endpoints_compared']=len(changes)
    summary['mean_point_error_change_px']=sum(x['delta'] for x in changes)/len(changes)
    summary['point_error_improved']=sum(x['delta']<0 for x in changes)
    summary['point_error_worsened']=sum(x['delta']>0 for x in changes)
    summary['largest_gains']=sorted(changes,key=lambda x:x['delta'])[:5]
    summary['largest_losses']=sorted(changes,key=lambda x:-x['delta'])[:5]
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'point-error-changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    cases=[('new_000007',14,'Largest measured point improvement'),
           ('new_000059',5,'Largest measured point regression'),
           ('new_000053',11,'Second measured point regression'),
           ('new_000089',49,'Only new match on R24 review'),
           ('new_000087',15,'Primary-reference match loss')]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    details=[]
    for iid,row,title in cases:
        obj=gold[iid][row-1];x1,y1,x2,y2=obj['bbox_xyxy']
        bounds=(max(0,int(x1)-16),max(0,int(y1)-16),min(1280,math.ceil(x2)+16),min(720,math.ceil(y2)+16))
        original=root/'data/images/val'/f'{iid}.png'
        im=Image.open(original).convert('RGB').crop(bounds)
        scale=min(6,470/im.width,300/im.height)
        size=(round(im.width*scale),round(im.height*scale))
        enlarged=im.resize(size,Image.Resampling.NEAREST)
        panel=Image.new('RGB',(1500,470),'#fafafa');draw=ImageDraw.Draw(panel)
        draw.text((12,8),f'{iid}:{row} - {title} (rows in the accepted R24 review)',font=font,fill='black')
        record={'image_id':iid,'row':row,'title':title,'image_sha256':digest(original),'crop':bounds,'gold':obj,'models':{}}
        for col,name in enumerate(('original','r23','r24')):
            ox,oy=12+col*496,78
            panel.paste(enlarged,(ox,oy));draw.text((ox,43),name.upper(),font=font,fill='black')
            if name=='original':continue
            index=matches[name][iid].get(row-1)
            matched=index is not None
            if index is None:
                index=max(range(len(predictions[name][iid])),key=lambda i:_box_iou(obj['bbox_xyxy'],predictions[name][iid][i]['bbox_xyxy']))
            pred=predictions[name][iid][index]
            record['models'][name]={'prediction_index':index,'matched':matched,'prediction':pred,'iou':_box_iou(obj['bbox_xyxy'],pred['bbox_xyxy'])}
            def xy(x,y):return ox+(x-bounds[0])*scale,oy+(y-bounds[1])*scale
            for box,color in [(obj['bbox_xyxy'],'#00ccff'),(pred['bbox_xyxy'],'#e5007d')]:
                draw.rectangle((*xy(box[0],box[1]),*xy(box[2],box[3])),outline=color,width=2)
            for point in obj['keypoints']:
                if point[2]!=2:continue
                x,y=xy(*point[:2]);draw.line((x-5,y,x+5,y),fill='#00ff88',width=2);draw.line((x,y-5,x,y+5),fill='#00ff88',width=2)
            for point in pred['keypoints']:
                x,y=xy(*point[:2]);draw.ellipse((x-3,y-3,x+3,y+3),outline='#ff168c',width=2)
            draw.text((ox,390),f"IoU {record['models'][name]['iou']:.4f}; conf {pred['confidence']:.3f}; {'match' if matched else 'unmatched proposal'}",font=font,fill='black')
        draw.text((12,438),'Cyan box/green cross: accepted R24 reference. Magenta: model box and point.',font=font,fill='black')
        panel.save(output/f'{iid}-row{row:02d}-comparison.png')
        details.append(record)
    (output/'cases.json').write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps({'references':{k:{'r23_matches':v['r23']['tp'],'r24_matches':v['r24']['tp'],'r23_extras':v['r23']['fp'],'r24_extras':v['r24']['fp']} for k,v in summary['references'].items()},'mean_point_error_change_px':summary['mean_point_error_change_px'],'same_points':len(changes)},indent=2))


if __name__=='__main__':main()
