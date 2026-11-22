#!/usr/bin/env python3
"""Independent r22 all-nonempty PDF readback; never imports the PDF builder.

The photo inventory, row/point counts, field colours and marker availability are
computed from actual immutable r21 and saved r22 label bytes. Builder QA is a
cross-check, not the source of those expected values. Run after the final build.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math, re, sys
import fitz
from PIL import Image
from pypdf import PdfReader

sys.dont_write_bytecode = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CYAN = (65/255, 225/255, 237/255)
MAGENTA = (1, 54/255, 209/255)
INK = (19/255, 43/255, 56/255)
COLOR_INTS = {'cyan':0x41e1ed, 'magenta':0xff36d1}
GROUPS = {'box':slice(1,5), 'left':slice(5,8), 'right':slice(8,11)}
TOL = .002

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def close(a, b, tol=2e-6):
    return a is not None and len(a)==len(b) and max(abs(x-y) for x,y in zip(a,b))<tol

def color_name(value):
    return 'magenta' if close(value,MAGENTA) else 'cyan' if close(value,CYAN) else None

def error(a,b):
    assert len(a)==len(b)
    return max(abs(x-y) for x,y in zip(a,b))

def ranges(values):
    groups=[]
    for v in sorted(set(values)):
        if groups and v==groups[-1][-1]+1:groups[-1].append(v)
        else:groups.append([v])
    return ', '.join(str(g[0]) if len(g)==1 else f'{g[0]}-{g[-1]}' for g in groups) or 'none'

def compact(value):
    return re.sub(r'\s+','',value)

def parse(path, size):
    lines=Path(path).read_text().splitlines();out=[];w,h=size
    for row,line in enumerate(lines,1):
        values=list(map(float,line.split()))
        assert len(values)==11 and all(math.isfinite(v) for v in values),(path,row,'invalid row')
        assert values[0]==0 and values[3]>0 and values[4]>0,(path,row,'invalid class/box')
        points=[]
        for start in (5,8):
            x,y,v=values[start:start+3]
            assert v in (0,2),(path,row,'visibility')
            assert v!=0 or (x,y)==(0,0),(path,row,'noncanonical missing point')
            points.append([x*w,y*h,int(v)])
        cx,cy,bw,bh=values[1:5]
        out.append({'line':line,'values':values,'bbox':[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h],'points':points})
    return out

def text_spans(page):
    return [span for block in page.get_text('dict')['blocks'] if block['type']==0
            for line in block['lines'] for span in line['spans']]

def text_within_page(page, label):
    spans=text_spans(page)
    for span in spans:
        r=fitz.Rect(span['bbox'])
        assert r.x0>=-.01 and r.y0>=-.01 and r.x1<=page.rect.width+.01 and r.y1<=page.rect.height+.01,(label,span['text'],'text outside page')
    return spans

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=REPOSITORY_ROOT,help='Repository root; inferred from this script by default')
    ap.add_argument('--pdf',type=Path,required=True,help='Downloaded final PDF to verify')
    ap.add_argument('--review-root',type=Path,default=Path('data/reviews/fronts-2026-11-22-r22'))
    ap.add_argument('--builder-qa',type=Path,default=Path('results/review/front-review-20261122-r22/pdf-qa.json'))
    ap.add_argument('--report',type=Path,required=True,help='Write the vector report outside the repository')
    args=ap.parse_args();root=args.root.resolve();pdf=args.pdf.resolve()
    review=args.review_root if args.review_root.is_absolute() else root/args.review_root
    qa_path=args.builder_qa if args.builder_qa.is_absolute() else root/args.builder_qa
    manifest_path=review/'manifest.json';m=read(manifest_path);q=read(qa_path)
    assert m['review_id']=='fronts-2026-11-22-r22'
    parent_path=root/m['parent']['path'];assert sha(parent_path)==m['parent']['sha256']
    parent=read(parent_path);assert parent['review_id']=='fronts-2026-11-22-r21'
    source_path=root/'data/manifest.json';assert sha(source_path)==m['source_manifest_sha256']
    records={v['id']:v for v in read(source_path)['records']}
    entries={v['image_id']:v for v in m['files']};oldentries={v['image_id']:v for v in parent['files']}
    assert len(records)==157 and set(records)==set(entries)==set(oldentries)
    rows={};oldrows={};source_checks=[]
    for iid,record in sorted(records.items()):
        item=entries[iid];olditem=oldentries[iid]
        assert item['label_path']==record['label'] and item['split']==record['split']
        path=review/item['label_path'];oldpath=parent_path.parent/olditem['label_path']
        assert sha(path)==item['label_sha256'] and sha(oldpath)==olditem['label_sha256'],(iid,'label hash')
        image_path=root/'data'/record['image']
        assert sha(image_path)==record['image_sha256']==item['source_image_sha256'],(iid,'source image hash')
        size=record['original_size'];assert size==[1280,720]
        rows[iid]=parse(path,size);oldrows[iid]=parse(oldpath,size)
        assert len(rows[iid])==item['objects']==len(oldrows[iid]),(iid,'stable row count')
        source_checks.append({'image_id':iid,'current_label_sha256':sha(path),'parent_label_sha256':sha(oldpath),'source_image_sha256':sha(image_path),'rows':len(rows[iid])})
    ids=sorted(i for i in records if rows[i]);empty=sorted(i for i in records if not rows[i]);oldids=[i for i in ids if i.startswith('old_')]
    assert len(ids)==141 and len(empty)==16 and len(oldids)==31
    assert sum(len(rows[i]) for i in ids)==4448
    index_count=math.ceil(len(ids)/36);assert index_count==4
    expected_changed=sorted(i for i in records if [r['line'] for r in rows[i]]!=[r['line'] for r in oldrows[i]])
    assert expected_changed==m['changed_image_ids']
    assert q['pdf_sha256']==sha(pdf) and q['included_image_ids']==ids
    assert q['all_nonempty_display'] and q['changes_only_display'] and q['compact_index']
    assert q['excluded_old_image_ids']==[] and q['excluded_empty_image_ids']==empty
    assert q['index_page_count']==index_count and q['page_count']==145
    assert q['review_manifest']['sha256']==sha(manifest_path)
    assert not q['truncated_footer_blocks'] and not q['unresolved_tag_collisions'] and not q['clipped_tags']
    assert q['current_tag_count']==4448 and q['deletion_tag_count']==0
    qa_pages={p['image_id']:p for p in q['pages']};assert list(qa_pages)==ids
    expected_visible=sum(bool(p[2]) for iid in ids for row in rows[iid] for p in row['points'])
    parent_visible=sum(bool(p[2]) for iid in ids for row in oldrows[iid] for p in row['points'])
    removed=[]
    for iid in ids:
        for n,(row,old) in enumerate(zip(rows[iid],oldrows[iid]),1):
            for k,side in enumerate(('left','right')):
                if old['points'][k][2] and not row['points'][k][2]:removed.append((iid,n,side))
    assert set(removed)=={('new_000096',37,'right'),('new_000096',38,'right')},('requested point deletion set',removed)
    assert parent_visible==8064 and expected_visible==parent_visible-2
    d=fitz.open(pdf);assert len(d)==len(ids)+index_count==145
    reader=PdfReader(pdf)
    refs={(p.indirect_reference.idnum,p.indirect_reference.generation):i for i,p in enumerate(reader.pages)}
    def links(page_number):
        out=[]
        for ref in reader.pages[page_number].get('/Annots',[]):
            link=ref.get_object();assert link['/Subtype']=='/Link'
            assert '/Dest' in link and '/A' not in link
            dest=link['/Dest'][0];out.append(refs[(dest.idnum,dest.generation)])
        return out
    all_links=[x for j in range(index_count) for x in links(j)]
    assert all_links==list(range(index_count,len(d)))
    return_index={target:j for j in range(index_count) for target in links(j)}
    photo_toc=[v for v in d.get_toc() if re.fullmatch(r'(?:new|old)_[0-9]{6}',v[1])]
    assert [(v[1],v[2]) for v in photo_toc]==list(zip(ids,range(index_count+1,len(d)+1)))
    index_toc=[v for v in d.get_toc() if v[1].startswith('Photo index ')]
    assert [(v[1],v[2]) for v in index_toc]==[(f'Photo index {j+1} of 4',j+1) for j in range(4)]
    assert 'all labeled photographs' in d.metadata['title'].lower()
    assert 'including unchanged and old-series photographs' in d.metadata['subject']
    page_size=list(d[0].rect)[2:];assert error(page_size,[1190.5511811023623,841.8897637795277])<.02
    for j in range(index_count):
        page=d[j];assert not page.get_images();texts=page.get_text();text_within_page(page,f'index {j+1}')
        wanted=ids[j*36:(j+1)*36]
        assert re.findall(r'(?:new|old)_[0-9]{6}',texts)==wanted,(j+1,'index photo text')
        assert 'All labeled photographs / Round 22' in texts
        assert 'Only empty scenes are omitted' in texts
        assert '141 photographs' in texts and '4448 current fronts' in texts
        assert len(links(j))==len(wanted)
    receipts=[];counts=Counter();max_box=0.;max_point=0.;changed_row_count=0
    for offset,iid in enumerate(ids,index_count):
        page=d[offset];record=records[iid];current=rows[iid];before=oldrows[iid];qp=qa_pages[iid]
        assert error(list(page.rect)[2:],page_size)<.02
        texts=page.get_text();assert texts.splitlines()[1]==iid
        assert links(offset)==[return_index[offset]]
        info=page.get_image_info(xrefs=True);assert len(info)==1,(iid,'image count')
        inf=info[0];w,h=record['original_size'];assert [inf['width'],inf['height']]==[w,h]
        ix,iy,ir,ib=inf['bbox'];sx=(ir-ix)/w;sy=(ib-iy)/h
        assert abs(sx-sy)<1e-6 and inf['transform'][1]==0 and inf['transform'][2]==0
        def xy(x,y):return [(x-ix)/sx,(y-iy)/sy]
        pix=fitz.Pixmap(d,inf['xref']);assert pix.n==3 and not pix.alpha
        original=Image.open(root/'data'/record['image']).convert('RGB')
        assert original.size==(w,h) and pix.samples==original.tobytes(),(iid,'original RGB pixels')
        boxes=[];points=[];tag_rects=[];unknown_annotations=[]
        for draw in page.get_drawings():
            kinds=[v[0] for v in draw['items']];stroke=draw.get('color');fill=draw.get('fill')
            if draw['type']=='s' and kinds==['re'] and color_name(stroke):
                assert draw['dashes']=='[] 0';rr=draw['rect']
                boxes.append({'box':xy(rr.x0,rr.y0)+xy(rr.x1,rr.y1),'color':color_name(stroke),'seqno':draw['seqno'],'width':draw['width']});points.append({})
            elif draw['type']=='fs' and color_name(fill) and close(stroke,INK) and kinds in [['re'],['c']*4]:
                assert boxes;side='right' if kinds==['re'] else 'left';assert side not in points[-1]
                rr=draw['rect'];coord=xy((rr.x0+rr.x1)/2,(rr.y0+rr.y1)/2)
                assert error(coord,[0,0])>TOL,(iid,'marker at canonical missing coordinate')
                points[-1][side]={'xy':coord,'color':color_name(fill),'diameter_pt':[rr.width,rr.height],'seqno':draw['seqno']}
            elif draw['type']=='fs' and close(fill,INK) and color_name(stroke) and kinds==['re']:
                tag_rects.append(draw['rect'])
            elif draw['type']=='fs' and color_name(fill):
                unknown_annotations.append(draw['seqno'])
        assert not unknown_annotations,(iid,'unexpected annotation shapes',unknown_annotations)
        assert len(boxes)==len(current)==len(points)==len(tag_rects),(iid,'box/marker-group/tag counts')
        changes={};page_box_error=0.;page_point_error=0.;missing={'left':[],'right':[]};page_removed={'left':[],'right':[]}
        for n,(row,old,box,pts) in enumerate(zip(current,before,boxes,points),1):
            changed={g for g,sl in GROUPS.items() if row['values'][sl]!=old['values'][sl]};changes[n]=changed
            counts['changed_rows']+=bool(changed)
            assert box['color']==('magenta' if 'box' in changed else 'cyan'),(iid,n,'box colour')
            assert abs(box['width']-(.95 if 'box' in changed else .55))<1e-5
            e=error(box['box'],row['bbox']);assert e<TOL,(iid,n,'box geometry',e)
            page_box_error=max(page_box_error,e);counts['boxes']+=1;counts['magenta_boxes']+=box['color']=='magenta'
            for k,side in enumerate(('left','right')):
                p=row['points'][k];assert (side in pts)==bool(p[2]),(iid,n,side,'marker availability')
                if old['points'][k][2] and not p[2]:page_removed[side].append(n)
                if not p[2]:missing[side].append(n);counts['missing_points']+=1;continue
                point=pts[side];expected='magenta' if side in changed else 'cyan'
                assert point['color']==expected,(iid,n,side,'point colour')
                e=error(point['xy'],p[:2]);assert e<TOL,(iid,n,side,'point geometry',e)
                page_point_error=max(page_point_error,e)
                assert error(point['diameter_pt'],[4,4] if side in changed else [3.2,3.2])<.001
                counts['visible_points']+=1;counts['magenta_points']+=expected=='magenta';counts['magenta_'+side+'_points']+=expected=='magenta'
        # Extract each tag independently from its own painted rectangle. This also
        # handles adjacent same-colour tags that a PDF text extractor joins.
        labels={}
        for rr in tag_rects:
            assert rr.x0>=ix-TOL and rr.x1<=ir+TOL and rr.y0>=iy-TOL and rr.y1<=ib+TOL,(iid,'clipped tag')
            spans=[span for block in page.get_text('dict',clip=rr)['blocks'] if block['type']==0 for line in block['lines'] for span in line['spans']]
            label=''.join(s['text'] for s in spans).strip()
            assert re.fullmatch(r'[0-9]+\*?',label),(iid,'tag text',label)
            n=int(label.rstrip('*'));assert n not in labels,(iid,'duplicate row tag',n)
            assert label==str(n)+('*' if changes[n] else ''),(iid,n,'tag star')
            assert all(s['color']==COLOR_INTS['magenta' if changes[n] else 'cyan'] for s in spans),(iid,n,'tag colour')
            labels[n]=label
        assert set(labels)==set(range(1,len(current)+1)),(iid,'row identities')
        for j,ra in enumerate(tag_rects):
            for k,rb in enumerate(tag_rects[j+1:],j+1):
                intersection=ra&rb
                assert intersection.is_empty or intersection.width<=TOL or intersection.height<=TOL,(iid,'overlapping row tags',j+1,k+1)
        text_within_page(page,iid)
        availability=compact(page.get_text(clip=fitz.Rect(384,page.rect.height-124,756,page.rect.height-25)))
        for side in ('left','right'):
            assert compact(f'0 0 0 {side}: {ranges(missing[side])}') in availability,(iid,side,'missing-point footer')
        changed_footer=compact(page.get_text(clip=fitz.Rect(29,page.rect.height-124,361,page.rect.height-25)))
        for group in GROUPS:
            assert compact(f'{group.capitalize()}: {ranges(n for n,g in changes.items() if group in g)}') in changed_footer,(iid,group,'changed-field footer')
        assert qp['missing_point_rows']==missing and qp['removed_point_rows']==page_removed,(iid,'point QA metadata')
        assert qp['changed_fields']=={str(n):sorted(g) for n,g in changes.items() if g},(iid,'field QA metadata')
        assert qp['displayed_rows']==list(range(1,len(current)+1)) and qp['page']==offset+1
        assert qp['label_sha256']==entries[iid]['label_sha256'] and qp['original_image_sha256']==record['image_sha256']
        assert not qp['held_fields'] and not qp['deletion_tag_count']
        counts['row_tags']+=len(labels);max_box=max(max_box,page_box_error);max_point=max(max_point,page_point_error)
        receipts.append({'page':offset+1,'image_id':iid,'rows':len(current),'changed_rows':[n for n,g in changes.items() if g],'current_label_sha256':entries[iid]['label_sha256'],'parent_label_sha256':oldentries[iid]['label_sha256'],'original_png_sha256':record['image_sha256'],'image_bbox_pdf_points':[ix,iy,ir,ib],'original_size':[w,h],'box_max_error_original_px':page_box_error,'point_max_error_original_px':page_point_error,'visible_point_count':sum(len(v) for v in points),'missing_point_rows':missing,'removed_point_rows':page_removed,'exact_rgb_image_match':True,'field_colours_match_actual_saved_label_diff':True,'all_row_tags_present_and_correct':True,'row_tag_collisions':0,'clipped_tags':0})
    assert counts['boxes']==counts['row_tags']==4448
    assert counts['visible_points']==expected_visible and counts['missing_points']==2*4448-expected_visible
    early=[i for i in expected_changed if i.startswith('new_') and int(i[4:])<=80]
    assert early, 'The final PDF has no early-phase changed photo to inspect'
    densest=max(ids,key=lambda i:len(rows[i]));assert len(rows[densest])==63
    selections=[(early[0],'early amendment'),('new_000094','row 23 ymin amendment; ymax retained'),('new_000096','rack edits and two removed right points'),('new_000100','named right rack margins'),('new_000015','new_000015 saved-label inventory'),(oldids[0],'first old-series photo'),(oldids[-1],'last old-series photo'),(densest,'densest 63-row photo')]
    by_id={v['image_id']:v for v in receipts}
    visual=[{'page':j+1,'name':f'index-{j+1}','reason':'full compact index coverage'} for j in range(index_count)]
    for iid,reason in selections:
        if not any(v.get('image_id')==iid for v in visual):visual.append({'page':by_id[iid]['page'],'name':iid,'image_id':iid,'reason':reason})
    report={'schema_version':1,'pdf':str(pdf),'pdf_sha256':sha(pdf),'status':'vector_and_structure_checks_passed_visual_review_pending','review_manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)},'parent_manifest':{'path':str(parent_path),'sha256':sha(parent_path)},'builder_qa':{'path':str(qa_path),'sha256':sha(qa_path)},'checks':{'page_count':len(d),'index_pages':index_count,'photo_pages':len(ids),'old_photo_pages':len(oldids),'empty_photos_excluded':len(empty),'all_nonempty_photo_inventory_exact':True,'index_links_and_bookmarks_match':True,'all_header_return_links_match':True,'annotations_compared_with_actual_saved_labels':True,'field_colours_compared_with_actual_parent_current_numeric_differences':True,'maximum_box_geometry_error_original_px':max_box,'maximum_point_geometry_error_original_px':max_point,'geometry_tolerance_original_px':TOL,'all_embedded_RGB_pixels_equal_original_PNGs':True,'all_visible_row_tags_present':True,'no_row_tag_collisions':True,'no_clipped_row_tags':True,'no_text_outside_page':True,'no_marker_at_canonical_missing_coordinates':True,'missing_and_removed_point_metadata_verified':True,'requested_right_point_deletions_verified':True,'all_missing_point_and_changed_field_footer_lists_verified':True,'coordinates_not_moved_by_renderer':True,'tag_placement_separate_from_annotation_geometry':True,'expected_visible_points_from_saved_labels':expected_visible,**dict(counts)},'pages':receipts,'excluded_empty_image_ids':empty,'source_inventory':source_checks,'representative_render_plan':visual,'source_files_modified':False}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(args.report),'status':report['status'],'checks':report['checks']},indent=2))

if __name__=='__main__':
    main()
