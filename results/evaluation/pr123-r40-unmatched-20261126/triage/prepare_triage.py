import argparse,sys,json,hashlib,math
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
parser=argparse.ArgumentParser(description='Create read-only R39 unmatched-output crop and geometry review material.')
parser.add_argument('--root',type=Path,required=True,help='dl repository root')
parser.add_argument('--image-root',type=Path,required=True,help='Directory containing original validation PNGs')
parser.add_argument('--output',type=Path,required=True,help='Review output directory')
args=parser.parse_args()
ROOT=args.root.resolve();OUT=args.output.resolve();IMG=args.image_root.resolve();OUT.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_pose import parse_pose_labels
from pose_scoring import score_image,_box_iou
mp=ROOT/'data/reviews/fronts-2026-11-26-r38/manifest.json';mf=json.loads(mp.read_text());op=json.loads((ROOT/mf['operations']['path']).read_text());deleted={}
for o in op:
 if o['action']=='delete':deleted.setdefault(o['image_id'],[]).append({'row':o['base_row'],'object':parse_pose_labels(o['before_line'],1280,720)[0]})
cachepath=ROOT/'results/evaluation/pr123-r39-diagnosis-20261126/clipped-nms-predictions.jsonl';head,*rs=[json.loads(l) for l in cachepath.read_text().splitlines()];cache={r['image_id']:r for r in rs};cases=[];per_image={};refs={};font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
for f in mf['files']:
 if f['split']!='val':continue
 name=f['image_id'];r=cache[name];gold=parse_pose_labels((mp.parent/f['label_path']).read_text(),r['width'],r['height']);refs[name]=gold;p=r['predictions'];s=score_image(gold,p,r['width'],r['height'],.5,.5);matches={v['gold_index']:v['prediction_index'] for v in s['matches']};per_image[name]={'other':s['other_fp'],'duplicate':s['duplicate_fp'],'missed':s['fn']}
 for i in s['other_fp_prediction_indices']:
  pred=p[i];box=pred['bbox_xyxy'];j=max(range(len(gold)),key=lambda j:_box_iou(box,gold[j]['bbox_xyxy'])) if gold else None;iou=_box_iou(box,gold[j]['bbox_xyxy']) if j is not None else 0
  ex=sorted([{'r37_row':q['row'],'iou':_box_iou(box,q['object']['bbox_xyxy'])} for q in deleted.get(name,[])],key=lambda q:-q['iou'])
  near=gold[j]['bbox_xyxy'] if iou>.1 else box
  x0=max(0,int(min(box[0],near[0])-30));y0=max(0,int(min(box[1],near[1])-35));x1=min(1280,int(math.ceil(max(box[2],near[2])+30)));y1=min(720,int(math.ceil(max(box[3],near[3])+35)))
  cases.append({'case':len(cases)+1,'image_id':name,'primary_prediction_id':r['primary_prediction_ids'][i],'clip_prediction_id':i+1,'prediction':pred,'best_reference_row':j+1 if iou>0 else None,'best_reference_iou':iou,'best_reference_is_missed':j in s['unmatched_gold_indices'] if iou>0 else None,'matched_primary_prediction_id_for_best_reference':r['primary_prediction_ids'][matches[j]] if j in matches and iou>0 else None,'r38_excluded_overlaps':ex,'crop':[x0,y0,x1,y1],'border':{'top':box[1]==0,'bottom':box[3]==720,'left':box[0]==0,'right':box[2]==1280}})
for name in {c['image_id'] for c in cases}:
 image=Image.open(IMG/(name+'.png')).convert('RGB');d=ImageDraw.Draw(image)
 for j,g in enumerate(refs[name],1):d.rectangle(g['bbox_xyxy'],outline=(0,220,230),width=1)
 for c in [c for c in cases if c['image_id']==name]:
  b=c['prediction']['bbox_xyxy'];d.rectangle(b,outline=(255,0,110),width=2);d.text((max(0,b[0]),max(0,b[1]-15)),f"p{c['primary_prediction_id']}",fill=(255,0,110),font=font,stroke_width=1,stroke_fill=(0,0,0))
 image.save(OUT/f'full_{name}.png')
for start in range(0,len(cases),6):
 subset=cases[start:start+6];sheet=Image.new('RGB',(1660,1050),'white');draw=ImageDraw.Draw(sheet)
 for pos,c in enumerate(subset):
  cx=(pos%2)*830;cy=(pos//2)*350;name=c['image_id'];box=c['prediction']['bbox_xyxy'];crop=c['crop'];original=Image.open(IMG/(name+'.png')).convert('RGB');raw=original.crop(crop);annot=raw.copy();dd=ImageDraw.Draw(annot)
  def translate(b):return [b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]]
  for j,g in enumerate(refs[name],1):
   b=g['bbox_xyxy']
   if b[2]<crop[0] or b[0]>crop[2] or b[3]<crop[1] or b[1]>crop[3]:continue
   dd.rectangle(translate(b),outline=(0,220,230),width=1);dd.text((max(0,b[0]-crop[0]),max(0,b[1]-crop[1])),f'r{j}',fill=(0,200,210),font=small)
  dd.rectangle(translate(box),outline=(255,0,110),width=2)
  for x,y,conf in c['prediction']['keypoints']:
   if x is not None and y is not None and crop[0]<=x<=crop[2] and crop[1]<=y<=crop[3]:dd.ellipse((x-crop[0]-2,y-crop[1]-2,x-crop[0]+2,y-crop[1]+2),fill=(255,0,110))
  scale=min(400/raw.width,240/raw.height);size=(max(1,int(raw.width*scale)),max(1,int(raw.height*scale)))
  sheet.paste(raw.resize(size,Image.Resampling.NEAREST),(cx+8,cy+65));sheet.paste(annot.resize(size,Image.Resampling.NEAREST),(cx+420,cy+65))
  draw.text((cx+8,cy+6),f"{c['case']:02d} {name} / p{c['primary_prediction_id']} conf={c['prediction']['confidence']:.3f}",fill='black',font=font)
  draw.text((cx+8,cy+26),f"best r{c['best_reference_row']} IoU={c['best_reference_iou']:.3f}; ref missed={c['best_reference_is_missed']}; existing p{c['matched_primary_prediction_id_for_best_reference']}",fill='black',font=small)
  draw.text((cx+8,cy+44),f"Original | cyan reference, magenta prediction | crop={crop}",fill='black',font=small)
  c['sheet']=f'sheet_{start//6+1:02d}.png'
 sheet.save(OUT/f'sheet_{start//6+1:02d}.png')
document={'scope':'All 64 other-unmatched outputs after accepted R39 clipped NMS; accepted R38 labels unchanged. Mechanical geometric association is not physical adjudication.','cache_sha256':hashlib.sha256(cachepath.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'per_image':per_image,'cases':cases};(OUT/'mechanical.json').write_text(json.dumps(document,indent=2)+'\n')
names=list(dict.fromkeys(c['image_id'] for c in cases))
context_font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for start in range(0,len(names),4):
 context=Image.new('RGB',(1280,770),'white');d=ImageDraw.Draw(context)
 for i,name in enumerate(names[start:start+4]):
  x=(i%2)*640;y=(i//2)*385
  original=Image.open(IMG/(name+'.png')).convert('RGB').resize((640,360))
  context.paste(original,(x,y+25));d.text((x+5,y+2),name,fill='black',font=context_font)
 context.save(OUT/f'context_{start//4+1:02}.png')
def source_sequence(names,crop,size,step,filename):
 canvas=Image.new('RGB',size,'white');d=ImageDraw.Draw(canvas)
 for i,name in enumerate(names):
  original=Image.open(IMG/(name+'.png')).convert('RGB').crop(crop)
  canvas.paste(original,(0,i*step+25))
  d.text((crop[2]-crop[0]+10,i*step+35),name+' original pixels\nsource crop '+str(crop),fill='black',font=context_font)
 canvas.save(OUT/filename)
source_sequence(['new_000087','new_000088','new_000089','new_000090'],(330,0,1030,140),(1400,700),175,'top_sequence.png')
source_sequence(['new_000053','new_000054','new_000055','new_000056'],(350,0,1050,120),(1200,600),150,'top_sequence_053_056.png')
print('Count',len(cases),'per_image',per_image)
print('Bins',Counter('none' if c['best_reference_iou']==0 else 'low<.1' if c['best_reference_iou']<.1 else '.1-.3' if c['best_reference_iou']<.3 else '.3-.5' for c in cases))
print('Excluded overlap>=.5',[(c['case'],c['image_id'],c['primary_prediction_id'],c['r38_excluded_overlaps'][0]) for c in cases if c['r38_excluded_overlaps'] and c['r38_excluded_overlaps'][0]['iou']>=.5])
for c in cases:print(c['case'],c['image_id'],c['primary_prediction_id'],round(c['prediction']['confidence'],3),c['best_reference_row'],round(c['best_reference_iou'],3),c['best_reference_is_missed'],c['matched_primary_prediction_id_for_best_reference'])
