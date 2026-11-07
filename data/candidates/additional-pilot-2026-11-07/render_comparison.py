"""Render original pixels, source labels and explicit annotation proposals without models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


REGIONS = {
    'frame_0001': {'upper': [570,185,980,280], 'middle': [550,255,975,478], 'lower': [605,505,875,674]},
    'frame_0160': {'upper': [350,282,675,343], 'middle': [340,330,675,490], 'lower': [390,538,610,672]},
    'frame_0383': {'upper': [465,243,765,295], 'middle': [460,288,770,434], 'lower': [518,480,720,587]},
}


def rows(path, width, height):
    result = []
    for index,line in enumerate(path.read_text().splitlines(),1):
        _,cx,cy,w,h,lx,ly,lv,rx,ry,rv = map(float,line.split())
        result.append({'row':index,'bbox':[(cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height],
                       'points':[(lx*width,ly*height,lv),(rx*width,ry*height,rv)]})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.candidate/'manifest.json').read_text())
    rendered=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    for record in manifest['images']:
        name=record['id']
        image_path=args.source_root/record['source_image']
        source_path=args.source_root/record['source_label']
        candidate_path=args.candidate/record['candidate_label']
        original=Image.open(image_path).convert('RGB')
        source=rows(source_path,*original.size)
        candidate=rows(candidate_path,*original.size)
        unresolved=set(record['unresolved_source_rows'])
        for region,crop in REGIONS[name].items():
            aspect=(crop[3]-crop[1])/(crop[2]-crop[0])
            fig,axes=plt.subplots(1,3,figsize=(16.5,max(3.8,5.1*aspect+1.5)),sharex=True,sharey=True)
            for axis,title,annotations,color in zip(axes,
                    ['Original pixels','Supplied annotations','Proposed annotations'],
                    [[],source,candidate],['black','#e09000','#00aa78']):
                axis.imshow(original,interpolation='nearest')
                for row in annotations:
                    x1,y1,x2,y2=row['bbox']
                    if x2<crop[0] or x1>crop[2] or y2<crop[1] or y1>crop[3]:
                        continue
                    pending=annotations is candidate and row['row'] in unresolved
                    line_color='#bd158b' if pending else color
                    axis.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=line_color,
                                             linewidth=.9,linestyle='--' if pending else '-'))
                    axis.text(max(crop[0]+1,x1+.5),max(crop[1]+1,y1+.5),str(row['row'])+('?' if pending else ''),
                              fontsize=7,color='black',va='top',clip_on=True,
                              bbox={'facecolor':'white','edgecolor':'none','alpha':.7,'pad':.3})
                    for point,point_color,marker in zip(row['points'],['#d82456','#147ad0'],['o','+']):
                        if point[2]==2:
                            axis.scatter(point[0],point[1],s=16,color=point_color,marker=marker,linewidths=.8)
                axis.set_xlim(crop[0],crop[2]);axis.set_ylim(crop[3],crop[1])
                axis.set_title(title)
                axis.set_xlabel('Original X (px)')
            axes[0].set_ylabel('Original Y (px)')
            fig.suptitle(f'{name} · {region} fronts · {record["disposition"]}',fontsize=14)
            fig.text(.5,.025,'No model predictions. Red: left corner. Blue: right corner. Dashed magenta / ?: unresolved row.',
                     ha='center',fontsize=9)
            fig.subplots_adjust(left=.05,right=.995,top=.83,bottom=.19,wspace=.10)
            path=args.output/f'{name}-{region}.png'
            fig.savefig(path,dpi=140,facecolor='white');plt.close(fig)
            rendered.append({'id':name,'region':region,'crop_xyxy':crop,'file':path.name,
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'source_image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                             'source_label_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
                             'candidate_label_sha256':hashlib.sha256(candidate_path.read_bytes()).hexdigest()})
    (args.output/'render-inputs.json').write_text(json.dumps(rendered,indent=2)+'\n')
    print(json.dumps({'rendered':len(rendered),'model_predictions_used':False,'output':str(args.output)}))


if __name__=='__main__':
    main()
