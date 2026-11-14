#!/usr/bin/env python3
"""Render unresolved new_000128 fragments from pinned original pixels."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,__version__ as PILLOW_VERSION
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo-root',type=Path,required=True)
p.add_argument('--spec',type=Path,required=True)
p.add_argument('--out-dir',type=Path,required=True)
p.add_argument('--font',type=Path,default=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
a=p.parse_args();raw_spec=a.spec.read_bytes();s=json.loads(raw_spec)
h=lambda b:hashlib.sha256(b).hexdigest()
for rel,sha in [(s['source_image_path'],s['source_image_sha256']),(s['source_label_path'],s['source_label_sha256'])]:
 if h((a.repo_root/rel).read_bytes())!=sha:raise ValueError(f'Source hash mismatch: {rel}')
im=Image.open(a.repo_root/s['source_image_path']).convert('RGB');a.out_dir.mkdir(parents=True,exist_ok=True)
font=lambda n:ImageFont.truetype(str(a.font),n)
ink=(25,31,40);orange=(255,190,35);outputs=[]
for case in s['cases']:
 box=case['crop_xyxy'];scale=case['scale'];crop=im.crop(box)
 crop=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST)
 crop.save(a.out_dir/case['raw_crop']);marked=crop.copy();d=ImageDraw.Draw(marked)
 v=list(map(float,case['source_before_text'].split()));cx,cy,w,hh=v[1:5]
 bounds=[(cx-w/2)*1280,(cy-hh/2)*720,(cx+w/2)*1280,(cy+hh/2)*720]
 rr=((bounds[0]-box[0])*scale,(bounds[1]-box[1])*scale,(bounds[2]-box[0])*scale,(bounds[3]-box[1])*scale)
 d.rectangle(rr,outline='black',width=5);d.rectangle(rr,outline=orange,width=2)
 if case['source_row']==1:
  canvas=Image.new('RGB',(crop.width+40,crop.height*2+180),(250,250,250));cd=ImageDraw.Draw(canvas)
  cd.text((20,14),'new_000128 row 1: physical ownership remains unresolved',font=font(26),fill=ink)
  cd.text((20,51),'Original crop above; existing source box below, shown as reference only.',font=font(18),fill=ink)
  canvas.paste(crop,(20,86));canvas.paste(marked,(20,crop.height+110))
  yy=crop.height*2+128
  cd.text((20,yy),'The top-edge band cannot be cleanly assigned to a separate front versus the lower cassette’s roof/lip.',font=font(18),fill=ink)
  cd.text((20,yy+27),'No change accepted. Small fragment size alone does not decide inclusion.',font=font(18),fill=ink)
 else:
  canvas=Image.new('RGB',(crop.width*2+60,crop.height+190),(250,250,250));cd=ImageDraw.Draw(canvas)
  cd.text((20,14),'new_000128 row 12: right-edge fragment remains unresolved',font=font(26),fill=ink)
  cd.text((20,51),'Original pixels',font=font(18),fill=ink);cd.text((40+crop.width,51),'Existing box; not an approved boundary',font=font(18),fill=ink)
  canvas.paste(crop,(20,86));canvas.paste(marked,(40+crop.width,86));yy=crop.height+107
  for line in ['The narrow strip overlaps row10 surface and dark rack/gap.',
               'A distinct owning front is not traceable here; overlap alone does not prove a duplicate.',
               'No change accepted. This is the image right edge, not a bottom fragment.']:
   cd.text((20,yy),line,font=font(18),fill=ink);yy+=27
 path=a.out_dir/case['figure'];canvas.save(path);outputs.append({'filename':path.name,'sha256':h(path.read_bytes()),'crop_xyxy':box,'scale':scale,'output_size_px':list(canvas.size)})
index={'schema_version':1,'status':'review_needed','spec_sha256':h(raw_spec),'renderer_sha256':h(Path(__file__).read_bytes()),'source_image_sha256':s['source_image_sha256'],'source_label_sha256':s['source_label_sha256'],'font_sha256':h(a.font.read_bytes()),'pillow_version':PILLOW_VERSION,'figures':outputs,'accepted_operations':[]}
(a.out_dir/'new_000128-fragment-render-index.json').write_text(json.dumps(index,indent=2)+'\n')
for out in outputs:print(a.out_dir/out['filename'])
