#!/usr/bin/env python3
"""Render thesis figures from frozen measured metrics and one original image."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch, ConnectionPatch
from matplotlib.ticker import PercentFormatter
from PIL import Image

BLUE='#0072B2'
ORANGE='#D55E00'


def setup():
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':9.5,
        'axes.labelsize':10, 'axes.titlesize':10, 'xtick.labelsize':9,
        'ytick.labelsize':9, 'legend.fontsize':9,
        'text.color':'black', 'axes.labelcolor':'black',
        'xtick.color':'black', 'ytick.color':'black',
        'axes.edgecolor':'black', 'figure.facecolor':'white',
        'axes.facecolor':'white', 'savefig.facecolor':'white',
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.7, 'xtick.major.width':.7,
        'ytick.major.width':.7, 'grid.color':'#D9D9D9',
        'grid.linewidth':.6, 'svg.fonttype':'none'})


def save(fig,path):
    fig.savefig(path,dpi=300,facecolor='white')
    plt.close(fig)
    with Image.open(path) as im:
        assert im.width==1950, (path,im.size)
        assert im.info.get('dpi',(0,0))[0]>299


def target_figure(data,image_path,output):
    source=image_path.read_bytes()
    assert hashlib.sha256(source).hexdigest()==data['source_sha256']
    im=Image.open(image_path).convert('RGB')
    assert list(im.size)==data['original_wh']
    record=data['record'];box=record['bbox_xyxy'];points=record['keypoints']
    assert len(points)==2 and all(k[2]==2 for k in points)
    x0,y0,x1,y1=box
    crop=(max(0,int(x0)-12),max(0,int(y0)-9),min(im.width,int(x1)+13),min(im.height,int(y1)+10))
    fig=plt.figure(figsize=(6.5,3.2))
    full=fig.add_axes([.025,.17,.65,.76])
    zoom=fig.add_axes([.725,.30,.25,.49])
    full.imshow(im,interpolation='nearest');full.set_axis_off()
    full.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=BLUE,linewidth=1.3))
    full.scatter([k[0] for k in points],[k[1] for k in points],s=8,c=ORANGE,zorder=4)
    full.set_title('(a) Original scene',loc='left',pad=6)
    zoom.imshow(im,interpolation='nearest')
    zoom.set_xlim(crop[0],crop[2]);zoom.set_ylim(crop[3],crop[1]);zoom.set_axis_off()
    zoom.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=BLUE,linewidth=1.25))
    zoom.plot([k[0] for k in points],[k[1] for k in points],color=ORANGE,linewidth=.8)
    zoom.scatter([k[0] for k in points],[k[1] for k in points],s=23,c=ORANGE,zorder=4)
    zoom.set_title('(b) R50 row 028',loc='center',pad=9)
    line=ConnectionPatch(xyA=(x1,y0),coordsA=full.transData,xyB=(0,.5),coordsB=zoom.transAxes,color='#555555',linewidth=.7,linestyle=':',zorder=0)
    fig.add_artist(line)
    handles=[Patch(facecolor='none',edgecolor=BLUE,linewidth=1.3,label='Annotated front box'),Line2D([],[],color=ORANGE,marker='o',markersize=4,linewidth=.8,label='Annotated lower-front corners')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,ncol=2,columnspacing=1.8,handlelength=1.7)
    save(fig,output/'figure-1-target.png')


def gains_figure(data,output):
    treatments=data['treatments']
    fig,ax=plt.subplots(figsize=(6.5,3.25))
    fig.subplots_adjust(left=.23,right=.97,top=.79,bottom=.20)
    positions=[1,0]
    for y,row in zip(positions,treatments):
        ax.barh(y,-row['losses'],height=.43,color=ORANGE)
        ax.barh(y,row['gains'],height=.43,color=BLUE)
        ax.text(-row['losses']-.8,y,str(row['losses']),va='center',ha='right',color='black')
        ax.text(row['gains']+.8,y,str(row['gains']),va='center',ha='left',color='black')
    ax.set_yticks(positions,[r['label']+'\nNet '+str(r['net']).replace('-','−') for r in treatments])
    ax.tick_params(axis='y',length=0,pad=9)
    ax.set_xlim(-30,23);ax.set_ylim(-.55,1.55)
    ax.set_xticks([-30,-20,-10,0,10,20],['30','20','10','0','10','20'])
    ax.set_xlabel('Number of reference fronts',labelpad=7)
    ax.axvline(0,color='black',linewidth=.7)
    ax.xaxis.grid(True,zorder=0);ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    fig.legend(handles=[Patch(facecolor=ORANGE,label='Lost ≤5 px pair success'),Patch(facecolor=BLUE,label='Gained ≤5 px pair success')],loc='upper center',bbox_to_anchor=(.58,.99),ncol=2,frameon=False,columnspacing=1.5)
    fig.text(.58,.845,'Primary frozen666 reference · 584 eligible two-corner fronts',ha='center',fontsize=9)
    save(fig,output/'figure-3-r53-changes.png')


def occlusion_figure(data,output):
    conditions=data['conditions']
    fig,ax=plt.subplots(figsize=(6.5,3.55))
    fig.subplots_adjust(left=.12,right=.98,top=.82,bottom=.23)
    xs=list(range(4));width=.34
    for delta,key,color,label in [(-width/2,'within5',BLUE,'Both errors ≤5 px'),(width/2,'within10',ORANGE,'Both errors ≤10 px')]:
        vals=[r[key] for r in conditions]
        bars=ax.bar([x+delta for x in xs],vals,width=width,color=color,label=label)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.55,str(val),ha='center',va='bottom',fontsize=9)
    ax.set_xticks(xs,['Intact','Left corner\nmasked','Right corner\nmasked','Center face\nmasked'])
    ax.set_ylabel('Successful corner pairs / 30',labelpad=9)
    ax.set_ylim(0,33.5);ax.set_yticks([0,5,10,15,20,25,30]);ax.set_xlim(-.65,3.65)
    ax.yaxis.grid(True);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.57,1),ncol=2,frameon=False,columnspacing=1.8)
    fig.text(.57,.855,'Fixed cohort of 30 targets in four source photographs',ha='center',fontsize=9)
    save(fig,output/'figure-2-occlusion.png')


def support_figure(data,output):
    fig,ax=plt.subplots(figsize=(6.5,4.05))
    fig.subplots_adjust(left=.13,right=.91,top=.87,bottom=.19)
    offsets={'frozen666':[(-34,-10),(-34,-2),(-34,16),(-18,8),(6,6)],'r50':[(6,-7),(6,-2),(6,0),(6,5),(6,0)]}
    for ref,color,marker in [('frozen666',BLUE,'o'),('r50',ORANGE,'s')]:
        row=data['references'][ref];pts=row['points']
        x=[100*p['eligible_recall'] for p in pts];y=[100*p['supported_fraction'] for p in pts]
        ax.plot(x,y,marker=marker,markersize=4.2,linewidth=1.2,color=color,label=row['label'])
        for p,px,py,offset in zip(pts,x,y,offsets[ref]):
            ax.annotate(f'{p["q"]:.2f}',(px,py),xytext=offset,textcoords='offset points',fontsize=8.5,color='black',annotation_clip=False)
    ax.set_xlim(0,100);ax.set_ylim(65,100)
    ax.set_xticks([0,20,40,60,80,100]);ax.set_yticks([65,70,75,80,85,90,95,100])
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.set_xlabel('Eligible-pair recall within 5 original pixels',labelpad=8)
    ax.set_ylabel('Reference-supported pair fraction',labelpad=8)
    ax.grid(True);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,.99),frameon=False,ncol=2,columnspacing=2)
    fig.text(.54,.025,'Point labels are endpoint cutoffs q; q = 0.99 offers no pairs and is omitted.',ha='center',fontsize=8.4)
    save(fig,output/'figure-4-pair-tradeoff.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--metrics',type=Path,default=root/'docs/thesis/figures/figure-data-r56.json')
    parser.add_argument('--original-image',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=root/'docs/thesis/figures')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.metrics.read_text())
    assert data['schema']=='thesis-r56-figure-metrics-v1'
    setup()
    target_figure(data['figure1'],args.original_image,args.output)
    occlusion_figure(data['figure2'],args.output)
    gains_figure(data['figure3'],args.output)
    support_figure(data['figure4'],args.output)
    info={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(args.output.glob('figure-*.png'))}
    (args.output/'figure-render-manifest.json').write_text(json.dumps({'schema':'thesis-r56-figure-render-v1','dpi':300,'width_inches':6.5,'metrics_sha256':hashlib.sha256(args.metrics.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'figures':info},indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'figures':list(info)}))


if __name__=='__main__':main()
