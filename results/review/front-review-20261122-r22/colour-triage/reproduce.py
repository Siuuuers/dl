"""Reproduce r22 colour-assisted inspection ranking; never edits labels."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
OUT = None
REVIEW = ROOT / 'data/reviews/fronts-2026-11-22-r21'
SOURCE = json.loads((ROOT / 'data/manifest.json').read_text())
RECORDS = {r['id']: r for r in SOURCE['records']}
MANIFEST = json.loads((REVIEW / 'manifest.json').read_text())
FILES = {f['image_id']: f for f in MANIFEST['files']}
FONT = ImageFont.truetype('DejaVuSans.ttf', 12)
SMALL = ImageFont.truetype('DejaVuSans.ttf', 10)
BOLD = ImageFont.truetype('DejaVuSans-Bold.ttf', 14)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read_photo(iid):
    rec, f = RECORDS[iid], FILES[iid]
    image_path, label_path = ROOT / 'data' / rec['image'], REVIEW / f['label_path']
    image_bytes, label_bytes = image_path.read_bytes(), label_path.read_bytes()
    assert sha(image_bytes) == rec['image_sha256']
    assert sha(label_bytes) == f['label_sha256']
    image = Image.open(image_path).convert('RGB')
    assert list(image.size) == rec['original_size']
    rows = []
    for number, line in enumerate(label_bytes.decode().splitlines(), 1):
        v = list(map(float, line.split())); w, h = image.size
        rows.append({'row': number, 'source_line': line, 'source_row_sha256': sha(line.encode()),
                     'bbox_xyxy': [(v[1] - v[3] / 2) * w, (v[2] - v[4] / 2) * h,
                                   (v[1] + v[3] / 2) * w, (v[2] + v[4] / 2) * h],
                     'keypoints': [[v[5] * w, v[6] * h, int(v[7])],
                                   [v[8] * w, v[9] * h, int(v[10])]]})
    provenance = {'image_id': iid, 'image_path': str(image_path.relative_to(ROOT)),
                  'source_image_sha256': rec['image_sha256'], 'source_label_path': str(label_path.relative_to(ROOT)),
                  'source_label_sha256': f['label_sha256'], 'rows': len(rows), 'split': rec['split']}
    return image, rows, provenance


def measure(rgb, row, side):
    x0, y0, x1, y1 = row['bbox_xyxy']; width, height = x1-x0, y1-y0
    k = min(5, max(2, math.ceil(width * .04)))
    ya, yb = max(0, math.ceil(y0 + .15*height)), min(rgb.shape[0], math.floor(y1 - .15*height)+1)
    if side == 'left':
        edge = math.ceil(x0)
        band = [edge, ya, edge+k, yb]
        inner = [edge+2*k, ya, edge+4*k, yb]
    else:
        edge = math.floor(x1)+1
        band = [edge-k, ya, edge, yb]
        inner = [edge-4*k, ya, edge-2*k, yb]
    result = {'side': side, 'band_width_px': k, 'edge_band_xyxy': band, 'inner_band_xyxy': inner}
    if width < 5*k or yb-ya < 4 or min(band[0], inner[0]) < 0 or max(band[2],inner[2]) > rgb.shape[1]:
        return {**result, 'measurable': False, 'score': 0.0, 'flagged': False, 'reason': 'Too few interior pixels; whole-image review required.'}
    def bands(box):
        crop = rgb[box[1]:box[3], box[0]:box[2]].astype(float)
        lum = crop @ np.array([.2126, .7152, .0722])
        chroma = crop.max(axis=2) - crop.min(axis=2)
        return np.median(lum,axis=1), np.median(chroma,axis=1)
    edge_luma, edge_chroma = bands(band)
    inner_luma, inner_chroma = bands(inner)
    delta = inner_luma-edge_luma
    median = float(np.median(delta))
    dark_fraction = float(np.mean(delta >= 12))
    low_chroma_fraction = float(np.mean(edge_chroma <= inner_chroma+6))
    score = max(0, median) * math.sqrt(dark_fraction) * (.5+.5*low_chroma_fraction)
    return {**result, 'measurable': True, 'edge_luma_median': float(np.median(edge_luma)),
            'inner_luma_median': float(np.median(inner_luma)), 'median_inner_minus_edge_luma': median,
            'edge_chroma_median': float(np.median(edge_chroma)), 'inner_chroma_median': float(np.median(inner_chroma)),
            'darker_row_fraction': dark_fraction, 'relative_low_chroma_row_fraction': low_chroma_fraction,
            'score': score,
            'low_threshold_colour_cue': median >= 12 and dark_fraction >= .6 and low_chroma_fraction >= .6,
            'flagged': score >= 45 and dark_fraction >= .6 and low_chroma_fraction >= .6}


def inspect(iid):
    image, rows, provenance = read_photo(iid); rgb = np.asarray(image)
    scored = [{**row, 'edges': [measure(rgb,row,side) for side in ('left','right')]} for row in rows]
    return image, scored, provenance


def whole_overlay(image, rows, iid):
    out=Image.new('RGB',(image.width,image.height+44),'#17212b');out.paste(image,(0,44))
    d=ImageDraw.Draw(out)
    d.text((8,4),f'{iid} / r21 labels / TRIAGE ONLY: orange edges are colour candidates, not confirmed rack',font=BOLD,fill='white')
    d.text((8,24),'Cyan = saved boxes and visible corner points. All row identities retained. Original photo size; no resampling.',font=SMALL,fill='white')
    for row in rows:
        x0,y0,x1,y1=row['bbox_xyxy'];d.rectangle((x0,y0+44,x1,y1+44),outline='#00e0e8',width=1)
        for edge in row['edges']:
            if edge['flagged']:
                xx=x0 if edge['side']=='left' else x1
                d.line((xx,y0+44,xx,y1+44),fill='#ffb000',width=2)
        for side,(px,py,v) in enumerate(row['keypoints']):
            if v:
                box=(px-1.5,py+42.5,px+1.5,py+45.5)
                (d.ellipse if side==0 else d.rectangle)(box,outline='#00e0e8',width=1)
        label=str(row['row']);rect=d.textbbox((x0+1,y0+44),label,font=SMALL)
        d.rectangle(rect,fill='#17212b');d.text((x0+1,y0+44),label,font=SMALL,fill='white')
    path=OUT/'whole'/f'{iid}.png';path.parent.mkdir(exist_ok=True);out.save(path)
    return str(path.relative_to(OUT))


def select_edges(rows, limit=6):
    ranked=sorted([(row,e) for row in rows for e in row['edges'] if e['flagged']],key=lambda pair:pair[1]['score'],reverse=True)
    selected=[]
    for side in ('left','right'):
        selected.extend([pair for pair in ranked if pair[1]['side']==side][:limit//2])
    seen={(row['row'],e['side']) for row,e in selected}
    selected.extend([pair for pair in ranked if (pair[0]['row'],pair[1]['side']) not in seen][:limit-len(selected)])
    return sorted(selected,key=lambda pair:pair[1]['score'],reverse=True)


def sheet(items, name, title):
    out=Image.new('RGB',(1320,650),'#17212b');d=ImageDraw.Draw(out)
    d.text((8,5),title,font=BOLD,fill='white')
    d.text((8,25),'Raw pixels / saved r21 overlay. Cyan = box, magenta = visible points. Nearest-neighbour integer zoom only. No edits.',font=FONT,fill='white')
    cards=[]
    for j,(iid,image,row,edge) in enumerate(items):
        x0,y0,x1,y1=row['bbox_xyxy'];bx=x0 if edge['side']=='left' else x1
        cx0,cx1=max(0,math.floor(bx)-21),min(image.width,math.floor(bx)+23)
        cy0,cy1=max(0,math.floor(y0)-5),min(image.height,math.ceil(y1)+5)
        clipped=False
        if cy1-cy0>95:
            middle=(cy0+cy1)//2;cy0,cy1=middle-47,middle+48;clipped=True
        crop=image.crop((cx0,cy0,cx1,cy1))
        z=max(1,min(4,190//crop.width,220//crop.height))
        raw=crop.resize((crop.width*z,crop.height*z),Image.Resampling.NEAREST)
        overlay=raw.copy();ov=ImageDraw.Draw(overlay)
        ov.rectangle(((x0-cx0)*z,(y0-cy0)*z,(x1-cx0)*z,(y1-cy0)*z),outline='#00ffff',width=1)
        for px,py,v in row['keypoints']:
            if v and cx0<=px<cx1 and cy0<=py<cy1:
                xx,yy=(px-cx0)*z,(py-cy0)*z;ov.ellipse((xx-2,yy-2,xx+2,yy+2),outline='#ff37d0',width=1)
        xx,yy=(j%3)*440+8,55+(j//3)*295
        d.text((xx,yy),f'{iid}:{row["row"]} {edge["side"]} / score {edge["score"]:.1f}',font=BOLD,fill='white')
        d.text((xx,yy+20),f'x={bx:.2f}; crop [{cx0},{cy0},{cx1},{cy1}]; zoom {z}x'+(' / clipped height' if clipped else ''),font=SMALL,fill='white')
        d.text((xx,yy+36),'RAW',font=SMALL,fill='white');d.text((xx+215,yy+36),'CURRENT OVERLAY',font=SMALL,fill='white')
        out.paste(raw,(xx,yy+52));out.paste(overlay,(xx+215,yy+52))
        cards.append({'image_id':iid,'row':row['row'],'side':edge['side'],'score':edge['score'],
                      'source_row_sha256':row['source_row_sha256'],'source_image_sha256':RECORDS[iid]['image_sha256'],
                      'source_label_sha256':FILES[iid]['label_sha256'],'crop_xyxy':[cx0,cy0,cx1,cy1],
                      'source_crop_rgb_sha256':sha(crop.tobytes()),'integer_nearest_zoom':z,'height_bounded':clipped})
    path=OUT/'sheets'/f'{name}.png';path.parent.mkdir(exist_ok=True);out.save(path)
    return {'path':str(path.relative_to(OUT)),'cards':cards}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh output directory for scores and original-pixel inspection previews')
    args = parser.parse_args()
    OUT = args.output.resolve()
    if OUT.exists():
        raise SystemExit('Choose a fresh output directory; existing inspection evidence is not overwritten.')
    assert sha((REVIEW / 'manifest.json').read_bytes()) == '7e66bfb6f5889b81a7ba6fadc0d66e28979c5ac79dae9995b4f17eeea281fb49'
    OUT.mkdir(parents=True,exist_ok=False)
    early = sorted(i for i in RECORDS if i.startswith('new_') and 1 <= int(i[4:]) <= 80)
    photos = [];sheets=[]
    for iid in early:
        image, rows, provenance = inspect(iid)
        selected=select_edges(rows)
        if selected:
            sheets.append(sheet([(iid,image,row,edge) for row,edge in selected],iid,f'{iid} / bounded colour triage: top six edges, balanced by side'))
        photos.append({**provenance, 'scored_rows': rows,
                       'whole_overlay':whole_overlay(image,rows,iid),'sheet_candidate_edges':[{'row':row['row'],'side':edge['side'],'score':edge['score']} for row,edge in selected],
                       'flagged_edges': sum(e['flagged'] for r in rows for e in r['edges'])})
    calibration=[]
    targets={'new_000082': [4,11],
             'new_000096': [*range(5,10),*range(16,20),*range(25,34),35,36],
             'new_000100': list(range(34,49)), 'new_000094':[31,32]}
    for iid, targets_rows in targets.items():
        _, rows, provenance = inspect(iid)
        for row in rows:
            if row['row'] in targets_rows:
                for edge in row['edges']:
                    if iid=='new_000094' or edge['side']==('left' if iid=='new_000082' else 'right'):
                        calibration.append({**provenance,**row,'edge':edge,'known_reference':'accepted_standard' if iid=='new_000094' else 'user_reported_rack_margin'})
    payload={'schema_version':1,'purpose':'colour-assisted inspection triage only; no annotation decisions or automatic edits',
             'source_review':{'path':'data/reviews/fronts-2026-11-22-r21/manifest.json','sha256':sha((REVIEW/'manifest.json').read_bytes())},
             'method':{'sampling':'Central 70 percent of box height; near-edge inner strip compared with an adjacent deeper interior strip.',
                       'margin_width_px':'min(5, max(2, ceil(box_width * 0.04)))',
                       'flag_rule':'score >= 45 AND fraction of rows with contrast >= 12 is >= 0.6 AND fraction with edge_chroma <= inner_chroma + 6 is >= 0.6',
                       'score':'max(0, median contrast) * sqrt(darker row fraction) * (0.5 + 0.5 * relative low-chroma row fraction)',
                       'global_whiteness_threshold':False,'source_pixels_modified':False,'labels_modified':False},
             'photos':photos,'calibration':calibration,'sheets':sheets,
             'limits':['These scores do not establish material ownership.','Legitimate shaded rims, bevels, text, glare and partial occlusion can trigger or suppress the cue.','The two accepted standard rows are a small calibration example, not a specificity estimate.','Only the top six qualifying edges per photo receive initial contact cards; whole overlays and all edge measurements are retained.','Every photo still requires a whole-image manual scan, including unflagged rows.']}
    (OUT/'triage.json').write_text(json.dumps(payload,indent=2)+'\n')
    summary={'photos':len(photos),'rows':sum(p['rows'] for p in photos),'flagged_edges':sum(p['flagged_edges'] for p in photos),
             'photos_with_flags':sum(bool(p['flagged_edges']) for p in photos),
             'flagged_rows':sum(any(e['flagged'] for e in r['edges']) for p in photos for r in p['scored_rows']),
             'low_threshold_colour_cue_edges':sum(e.get('low_threshold_colour_cue',False) for p in photos for r in p['scored_rows'] for e in r['edges']),
             'contact_sheets':len(sheets),'contact_sheet_edges':sum(len(s['cards']) for s in sheets),
             'reported_right_rack_096_100_flagged':sum(r['edge']['flagged'] for r in calibration if r['image_id'] in ('new_000096','new_000100')),
             'reported_right_rack_096_100_total':sum(r['image_id'] in ('new_000096','new_000100') for r in calibration),
             'reported_left_rack_082_flagged':sum(r['edge']['flagged'] for r in calibration if r['image_id']=='new_000082'),
             'reported_left_rack_082_total':sum(r['image_id']=='new_000082' for r in calibration),
             'accepted_standard_flags':sum(r['edge']['flagged'] for r in calibration if r['known_reference']=='accepted_standard'),
             'accepted_standard_edges':sum(r['known_reference']=='accepted_standard' for r in calibration),
             'by_image':{p['image_id']:p['flagged_edges'] for p in photos}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    calibration_items=[]
    for row in calibration:
        if row['image_id']=='new_000082' or row['known_reference']=='accepted_standard' or not row['edge']['flagged'] or (row['image_id'],row['row']) in [('new_000096',5),('new_000096',28),('new_000100',34),('new_000100',47)]:
            image,_,_=read_photo(row['image_id']);calibration_items.append((row['image_id'],image,row,row['edge']))
    calibration_sheets=[sheet(calibration_items[n:n+6],f'calibration-{n//6+1}','Calibration examples: named rack margins and accepted standards; no annotation decisions') for n in range(0,len(calibration_items),6)]
    (OUT/'calibration-sheets.json').write_text(json.dumps(calibration_sheets,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
