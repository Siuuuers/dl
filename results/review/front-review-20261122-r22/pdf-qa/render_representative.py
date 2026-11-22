#!/usr/bin/env python3
"""Render the independent validator's final-PDF visual plan with Poppler.

Requires a successful vector report for the exact final PDF. Rendering prepares
reviewable PNGs; it does not mark visual inspection as completed.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,math,shutil,subprocess

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path,required=True,help='Report produced by check_vectors.py')
    ap.add_argument('--output-dir',type=Path,required=True,help='Raster output directory outside the repository')
    ap.add_argument('--dpi',type=int,default=96)
    args=ap.parse_args();report=json.loads(args.report.read_text());pdf=Path(report['pdf'])
    assert report['status']=='vector_and_structure_checks_passed_visual_review_pending'
    assert sha(pdf)==report['pdf_sha256'];exe=shutil.which('pdftoppm');assert exe
    args.output_dir.mkdir(parents=True,exist_ok=True)
    def render(item):
        prefix=args.output_dir/f'page-{item["page"]:03d}-{item["name"]}'
        cmd=[exe,'-f',str(item['page']),'-l',str(item['page']),'-r',str(args.dpi),'-singlefile','-png',str(pdf),str(prefix)]
        subprocess.run(cmd,check=True,capture_output=True)
        path=prefix.with_suffix('.png');return {**item,'path':str(path),'sha256':sha(path),'dpi':args.dpi,'review_status':'not_yet_visually_inspected'}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages=list(pool.map(render,report['representative_render_plan']))
    # These additional raster crops show the two specifically requested repairs
    # at comfortable scale without changing any PDF object or source pixels.
    manifest_path=Path(report['review_manifest']['path']);manifest=json.loads(manifest_path.read_text());entries={r['image_id']:r for r in manifest['files']}
    receipts={p['image_id']:p for p in report['pages']};crops=[]
    for iid,rr,name in [('new_000094',[23,30],'094-row23-ymin-and-row30-reference'),('new_000096',[37,38],'096-rows37-38-no-right-markers')]:
        page=receipts[iid];source=(manifest_path.parent/entries[iid]['label_path']).read_text().splitlines();boxes=[]
        for row in rr:
            v=list(map(float,source[row-1].split()));w,h=page['original_size'];boxes.append([(v[1]-v[3]/2)*w,(v[2]-v[4]/2)*h,(v[1]+v[3]/2)*w,(v[2]+v[4]/2)*h])
        original=[max(0,math.floor(min(b[0] for b in boxes))-15),max(0,math.floor(min(b[1] for b in boxes))-15),min(w,math.ceil(max(b[2] for b in boxes))+15),min(h,math.ceil(max(b[3] for b in boxes))+15)]
        ix,iy,ir,ib=page['image_bbox_pdf_points'];sx=(ir-ix)/w;sy=(ib-iy)/h;dpi=144;s=dpi/72
        x=math.floor((ix+original[0]*sx)*s);y=math.floor((iy+original[1]*sy)*s)
        right=math.ceil((ix+original[2]*sx)*s);bottom=math.ceil((iy+original[3]*sy)*s)
        prefix=args.output_dir/f'detail-{name}'
        subprocess.run([exe,'-f',str(page['page']),'-l',str(page['page']),'-r',str(dpi),'-x',str(x),'-y',str(y),'-W',str(right-x),'-H',str(bottom-y),'-singlefile','-png',str(pdf),str(prefix)],check=True,capture_output=True)
        path=prefix.with_suffix('.png');crops.append({'image_id':iid,'rows':rr,'page':page['page'],'original_pixel_region':original,'pdf_raster_crop':[x,y,right,bottom],'dpi':dpi,'path':str(path),'sha256':sha(path),'review_status':'not_yet_visually_inspected'})
    receipt={'status':'representative_pages_rendered_visual_inspection_pending','pdf':str(pdf),'pdf_sha256':sha(pdf),'source_vector_report_sha256':sha(args.report),'renderer':'Poppler pdftoppm','pages':pages,'detail_crops':crops}
    path=args.output_dir.parent/'rendering.json';path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'receipt':str(path),'pages_rendered':len(pages),'detail_crops':len(crops)},indent=2))

if __name__=='__main__':main()
