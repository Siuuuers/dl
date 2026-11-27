#!/usr/bin/env python3
"""Render the R52 amendment atlas and PR146 result/thesis report from evidence.

Only original photographs, versioned labels and verified saved predictions are
used. This program does not load a model, train, change labels or promote weights.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import zipfile

from PIL import Image
import reportlab
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

REVIEW = Path('data/reviews/additional-fronts-2026-11-27-r52')
ARCHIVE_SHA = '9538932145e90201d407c339b7da1f73c5c905935853191ab9683b221bd5a09e'
ARCHIVE_BYTES = 43528788
INK, MUTED, TEAL, RULE, CYAN, AMBER, MAGENTA = map(HexColor,
    ('#173344', '#536974', '#087b80', '#d6e2e5', '#00b9c8', '#eaa427', '#d32682'))
W, H = landscape(A4)
M = 32
BW = W - 2*M


def sha(data):
    return hashlib.sha256(data).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n').encode()


def esc(value):
    return html.escape(str(value))


def parse_line(line):
    v = [float(x) for x in line.split()]
    need(len(v) == 11, 'Expected one 11-field pose row')
    return {'bbox_xyxy': [(v[1]-v[3]/2)*1280, (v[2]-v[4]/2)*720,
                         (v[1]+v[3]/2)*1280, (v[2]+v[4]/2)*720],
            'keypoints': [[v[5]*1280, v[6]*720, v[7]], [v[8]*1280, v[9]*720, v[10]]]}


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0.


def region(boxes, margin=18, minimum=(105, 72)):
    x0, y0 = min(b[0] for b in boxes)-margin, min(b[1] for b in boxes)-margin
    x1, y1 = max(b[2] for b in boxes)+margin, max(b[3] for b in boxes)+margin
    width, height = min(1280, max(minimum[0], x1-x0)), min(720, max(minimum[1], y1-y0))
    x = min(max(0, (x0+x1-width)/2), 1280-width)
    y = min(max(0, (y0+y1-height)/2), 720-height)
    return math.floor(x), math.floor(y), math.ceil(x+width), math.ceil(y+height)


def collect(root, archive, originals, supplementary, audit_dir):
    verification = read(audit_dir/'verification.json')
    need(verification['status'] == 'passed', 'Independent PR146 audit must pass')
    data = archive.read_bytes()
    need(len(data) == ARCHIVE_BYTES and sha(data) == ARCHIVE_SHA, 'PR146 archive differs')
    with zipfile.ZipFile(archive) as z:
        need(z.testzip() is None, 'PR146 ZIP CRC failed')
        predictions = json.loads(z.read('evaluation/predictions.json'))
    summary, cases = read(audit_dir/'summary.json'), read(audit_dir/'case-inventory.json')
    need(summary['archive_sha256'] == ARCHIVE_SHA and cases['source_pr'] == 146, 'Wrong result provenance')
    manifest, ledger = read(root/REVIEW/'manifest.json'), read(root/REVIEW/'audit-ledger.json')
    need(manifest['counts']['amended_rows'] == 10 and manifest['counts']['changed_side_boundaries'] == 11,
         'R52 amendment scope differs')
    need(manifest['audit_sha256'] == sha((root/REVIEW/'audit-ledger.json').read_bytes()), 'R52 ledger differs')
    need(manifest['all_corner_tokens_unchanged'] and manifest['all_hold_fields_unchanged'], 'Protected fields changed')
    original_manifest = read(root/'data/manifest.json')
    images, image_hashes = {}, {}
    for entry in original_manifest['records']:
        if entry['split'] != 'val':
            continue
        path = originals/(entry['id']+'.png')
        need(sha(path.read_bytes()) == entry['image_sha256'], 'Development original differs: '+entry['id'])
        images[entry['id']] = Image.open(path).convert('RGB')
        image_hashes[entry['id']] = entry['image_sha256']
    for frame in manifest['frames']:
        identity = frame['image_id']
        path = supplementary/(identity+'.jpg')
        need(sha(path.read_bytes()) == frame['source_image_sha256'], 'Supplementary original differs: '+identity)
        images[identity] = Image.open(path).convert('RGB')
        image_hashes[identity] = frame['source_image_sha256']
    need(len(images) == 32 and all(im.size == (1280,720) for im in images.values()), 'Original dimensions/membership differ')
    reference = read(root/'data/reviews/fronts-2026-11-27-r50/manifest.json')
    gold = {}
    for f in reference['files']:
        path = root/'data/reviews/fronts-2026-11-27-r50'/f['label_path']
        need(sha(path.read_bytes()) == f['label_sha256'], 'R50 reference differs')
        gold[f['image_id']] = [parse_line(line) for line in path.read_text().splitlines() if line.strip()]
    visual_path = root/'results/evaluation/pr146-r51-20261127/visual-observations.json'
    visual = read(visual_path)
    observations = visual.get('cases', visual.get('observations', []))
    lookup = {(x['image'], x.get('prediction_number',x.get('candidate_prediction_number'))): x for x in observations}
    need(len(lookup) == len(cases['candidate_unmatched']) == 112, 'Expected every unmatched output in visual review')
    need(set(lookup) == {(x['image'], x['prediction_number']) for x in cases['candidate_unmatched']}, 'Visual identity set differs')
    return dict(root=root, verification=verification, summary=summary, cases=cases, predictions=predictions,
        manifest=manifest, ledger=ledger, images=images, image_hashes=image_hashes, gold=gold, visual=visual,
        observations=lookup, before=read(root/REVIEW/'geometry-before.json'), after=read(root/REVIEW/'geometry-after.json'))


class Report:
    def __init__(self, output, title, footer):
        fonts = Path(reportlab.__file__).parent/'fonts'
        if 'R52' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('R52', str(fonts/'Vera.ttf')))
            pdfmetrics.registerFont(TTFont('R52B', str(fonts/'VeraBd.ttf')))
            pdfmetrics.registerFontFamily('R52', normal='R52', bold='R52B', italic='R52', boldItalic='R52B')
        output.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(output), pagesize=(W,H), invariant=1, pageCompression=1)
        self.c.setTitle(title); self.c.setAuthor('Siuuuers/dl | evidence review')
        self.footer = footer
        self.page = 0
        self.index, self.bounds = [], []

    def text(self, value, x, y, size=9, bold=False, color=INK):
        font = 'R52B' if bold else 'R52'
        width = pdfmetrics.stringWidth(str(value), font, size)
        need(M-0.1 <= x and x+width <= W-M+0.1 and 12 <= y <= H-12,
             f'Text outside page {self.page}: {value}')
        self.c.setFillColor(color); self.c.setFont(font,size); self.c.drawString(x,y,str(value))
        self.bounds.append({'page':self.page,'kind':'text','left':x,'right':x+width,'bottom':y})

    def para(self, value, x, y, width=BW, size=9, leading=13, bottom=40, color=INK):
        style = ParagraphStyle('p',fontName='R52',fontSize=size,leading=leading,textColor=color)
        p = Paragraph(value,style); _,height=p.wrap(width,1200)
        need(y-height >= bottom-0.1, f'Paragraph below allotted region on page {self.page}: {value[:80]}')
        p.drawOn(self.c,x,y-height)
        self.bounds.append({'page':self.page,'kind':'paragraph','left':x,'right':x+width,'bottom':y-height,'minimum':bottom})
        return y-height

    def start(self, section, title, subtitle=''):
        self.page += 1
        self.c.setFillColor(TEAL); self.c.rect(0,H-7,W,7,fill=1,stroke=0)
        self.text(section.upper(),M,H-28,8.2,True,TEAL)
        self.text(title,M,H-54,20,True)
        if subtitle:
            self.para(esc(subtitle),M,H-68,size=8.8,leading=11.5,bottom=H-99,color=MUTED)

    def end(self, section, **metadata):
        self.c.setStrokeColor(RULE);self.c.line(M,31,W-M,31)
        self.text(self.footer,M,17,7.1,color=MUTED)
        self.text(f'{self.page:02d}',W-M-20,17,9,True,TEAL)
        self.index.append({'page':self.page,'section':section,**metadata})
        self.c.showPage()

    def table(self, headers, rows, y, widths, height=27, size=8.6):
        for i,row in enumerate([headers]+rows):
            self.c.setFillColor(HexColor('#e6f2f3' if i==0 else '#f3f6f8' if i%2 else '#ffffff'))
            self.c.rect(M,y-height,sum(widths),height,stroke=0,fill=1)
            x=M
            for value,width in zip(row,widths):
                value=esc(value)
                self.para('<b>'+value+'</b>' if i==0 else value,x+6,y-5,width-12,size,size+2,bottom=y-height+2)
                x+=width
            y-=height
        return y-13

    def panel(self, im, crop, x, y, width, height, overlays=(), labels=False):
        need(y-height >= 38 and x>=M-0.1 and x+width<=W-M+0.1,'Image outside page')
        raw=im.crop(crop)
        scale=min(width/raw.width,height/raw.height)
        iw,ih=raw.width*scale,raw.height*scale
        ox,oy=x+(width-iw)/2,y-ih
        # Nearest-neighbor enlargement reveals the original pixel sampling;
        # it adds no image detail and never changes the source photograph.
        enlarged=raw.resize((raw.width*3,raw.height*3),Image.Resampling.NEAREST)
        self.c.drawImage(ImageReader(enlarged),ox,oy,iw,ih)
        def transform(px,py):return ox+(px-crop[0])*scale,y-(py-crop[1])*scale
        self.c.saveState()
        clip=self.c.beginPath();clip.rect(ox,oy,iw,ih)
        self.c.clipPath(clip,stroke=0,fill=0)
        for obj,color,dashed,label in overlays:
            self.c.saveState()
            self.c.setStrokeColor(color);self.c.setFillColor(color);self.c.setLineWidth(1.05)
            self.c.setDash(3,2) if dashed else self.c.setDash()
            a,b,c,d=obj['bbox_xyxy'];xx,yy=transform(a,b)
            self.c.rect(xx,yy-(d-b)*scale,(c-a)*scale,(d-b)*scale,stroke=1,fill=0)
            self.c.setDash()
            for px,py,v in obj.get('keypoints',[]):
                if v>=0.5:
                    qx,qy=transform(px,py)
                    if ox-1<=qx<=ox+iw+1 and oy-1<=qy<=y+1:self.c.circle(qx,qy,1.7,stroke=1,fill=0)
            if labels and label:
                font='R52B';fs=6.4;tw=pdfmetrics.stringWidth(label,font,fs)
                tx=min(max(xx+1,ox),ox+iw-tw-2);ty=min(max(yy-6,oy+2),y-6)
                self.c.setFillColor(white);self.c.rect(tx-1,ty-1,tw+2,7.3,fill=1,stroke=0)
                self.c.setFillColor(color);self.c.setFont(font,fs);self.c.drawString(tx,ty,label)
            self.c.restoreState()
        self.c.restoreState()
        return iw,ih

    def finish(self):
        self.c.save()
        return {'pages':self.page,'index':self.index,'layout_checks':self.bounds}


def amendments(data, output):
    p=Report(output,'R52 | Cassette-front amendments','R52 | 27 Nov 2026 HKT | Original pixels | Agent-assisted review; human adjudication pending')
    m=data['manifest'];counts=m['counts']
    p.start('Annotation review','Small boundary refinements, explicit evidence','507 existing rows across 12 supplementary photographs; parent R48 to proposed R52.')
    y=p.para('<b>10 rows / 11 sides amended in frame_0160.</b> The other 497 lines are byte-for-byte retained. All 1,014 corner slots, 31 held fields and whole-photo gates remain unchanged. These are nominal visual refinements, with approximately ±2 px boundary uncertainty; decimal serialization does not imply subpixel physical accuracy.',M,H-112,size=10,leading=14)
    rows=[]
    for a in m['amendments']:
        changes='; '.join(f"{s['field']} {s['before_px']:.3f} -> {s['nominal_after_px']:g}" for s in a['side_changes'])
        rows.append([f"s{a['source_row']:03d}",changes,'inward side refinement'])
    y=p.table(['Source row','Original-image pixels: R48 -> R52','Decision'],rows,y-16,[90,425,BW-515],height=23,size=8.5)
    p.para('The eight requested rows are s015, s016, s019, s020, s024, s025, s029 and s030. The same local contour pattern also supports s018 and s022, plus the right side of s030. Preserve the cassette\'s own gray rim; remove an outer strip only when it belongs to rack/background.',M,y,size=9,leading=13)
    p.end('amendment_summary',stable_ids=[a['stable_id'] for a in m['amendments']])

    p.start('Interpretation','A neighbour check is a review cue','The earlier containment check is retained and expanded to left, top, right and bottom contact diagnostics.')
    before,after=data['before']['counts'],data['after']['counts']
    rows=[['Strict interior: foreign corner >2 px inside',before['strict_interior_rows'],after['strict_interior_rows']],
          ['At least one near-edge contact (±2 px)',before['edge_contact_rows'],after['edge_contact_rows']],
          ['Any diagnostic cue',before['flagged_rows'],after['flagged_rows']],
          ['Own locatable corner outside tolerance',before['own_corner_outside_rows'],after['own_corner_outside_rows']]]
    y=p.table(['Rows flagged / all 507 rows','R48','R52'],rows,H-112,[BW-180,90,90],height=29)
    y=p.para('<b>These are workload counts, not verified error counts.</b> Ordinary stacked-front contacts account for many flags. Perspective and slanted visible surfaces can place another instance\'s projected corner inside a valid axis-aligned envelope. The neighbouring point may itself be wrong. The script never clips a box, moves a point, deletes a row or lifts a hold.',M,y-3,size=10,leading=14)
    y=p.para('<b>Follow surface ownership.</b> Include a visible cassette surface even when the rack or another cassette casts a shadow onto it. Exclude rack, gripper and neighbouring material from the target\'s extrema. A tight rectangle may still enclose background between those extrema. Bright turns and dark undersides are visual clues, not laws of physics.',M,y-17,size=10,leading=14)
    y=p.para('<b>Context remains in the input.</b> Full-frame images already show the surrounding rack. A future crop experiment can add context around an unchanged target. Changing the target box to include rack would instead change the supervision. The professor\'s generic detection lecture does not provide a cassette-specific instruction to do that.',M,y-17,size=10,leading=14)
    p.para('<b>Limits of this pass.</b> All 12 originals and all 507 existing row crops were inspected by an agent. This is not independent human adjudication, timed review research or proof that every unlabelled region is empty. Six complete and six held photograph statuses remain unchanged. The label-only checker cannot find a cassette for which no row exists.',M,y-17,size=9.5,leading=13.5)
    p.end('geometry_check_interpretation')

    lookup={r['stable_id']:r for r in data['ledger']['rows']}
    for offset in range(0,len(m['amendments']),2):
        part=m['amendments'][offset:offset+2]
        p.start('Before / after','Each change against the unmarked original','Raw crop at left. At right: R48 dashed amber, R52 solid magenta, retained corner points on both geometries.')
        for i,a in enumerate(part):
            y=H-113-i*216
            row=lookup[a['stable_id']];old=parse_line(row['before_line']);new=parse_line(row['after_line'])
            p.text(a['stable_id'],M,y,11,True)
            detail='; '.join(f"{s['field']}: {s['before_px']:.3f} -> {s['nominal_after_px']:g} px" for s in a['side_changes'])
            p.text(detail+' | nominal uncertainty ±2 px',M+223,y,8.5,color=MUTED)
            crop=region([old['bbox_xyxy'],new['bbox_xyxy']],margin=15,minimum=(105,62))
            half=(BW-18)/2
            p.panel(data['images'][a['image_id']],crop,M,y-12,half,118)
            p.panel(data['images'][a['image_id']],crop,M+half+18,y-12,half,118,
                    [(old,AMBER,True,''),(new,MAGENTA,False,'')])
            p.para(esc(a['reason']),M,y-139,size=9,leading=12.5,bottom=y-199)
        p.end('amendment_detail',stable_ids=[a['stable_id'] for a in part])

    changed={a['stable_id'] for a in m['amendments']}
    for offset in range(0,len(m['frames']),2):
        part=m['frames'][offset:offset+2]
        p.start('Review coverage','Every supplementary frame, current proposal','Numbers are original source-row identities. Cyan retains R48 geometry; magenta identifies an R52-amended box.')
        for i,f in enumerate(part):
            x=M+i*(BW/2+8);width=BW/2-8
            p.text(f['image_id'],x,H-115,12,True)
            held=sum(bool(r['unresolved_fields']) for r in f['rows'])
            p.text(f"{len(f['rows'])} rows | {'complete proposal' if f['annotation_complete'] else 'held'} | {held} rows with holds",x,H-131,8.5,color=MUTED)
            objects=[parse_line(r['after_line']) for r in f['rows']]
            crop=region([o['bbox_xyxy'] for o in objects],margin=20,minimum=(180,180))
            overlays=[(o,MAGENTA if r['stable_id'] in changed else CYAN,False,str(r['source_row']) if r['source_row'] is not None else 'add1') for o,r in zip(objects,f['rows'])]
            p.panel(data['images'][f['image_id']],crop,x,H-144,width,358,overlays,labels=True)
            p.para('Source photograph and every existing row reviewed. Retention is a review decision; cyan is not a certificate of physical truth.',x,78,width,size=8.2,leading=11,bottom=43,color=MUTED)
        p.end('supplementary_frame_overview',images=[f['image_id'] for f in part])
    return p.finish()


VISUAL_NAMES={
    'formal_duplicate':'Formal duplicate',
    'front_extent_or_fragment':'Front extent / fragment',
    'rack_or_gripper':'Rack / gripper',
    'keyboard_or_other_background':'Keyboard / background',
    'reference_policy_exclusion':'Reference-policy exclusion',
    'ambiguous_partial_fragment':'Uncertain partial fragment',
}


def visual_type(obs):
    return obs.get('visual_subtype',obs.get('subtype',obs.get('category','uncertain')))


def card(p,data,case,x,y,width,height,kind):
    identity,row=case['image'],case['row']
    target=case['target'];candidate=case['candidate']
    pred=candidate['prediction']
    if pred is None:
        values=data['predictions']['candidate'][identity]['postclip']
        pred=max(values,key=lambda v:iou(target['bbox_xyxy'],v['bbox_xyxy'])) if values else None
        overlap=iou(target['bbox_xyxy'],pred['bbox_xyxy']) if pred else 0.
        detail=f"{candidate['locatable_corners']} locatable corner(s); nearest retained IoU {overlap:.3f}"
    else:
        detail=f"out{candidate['prediction_number']:03d} (P{pred['primary_prediction_id']:03d}); confidence {pred['confidence']:.3f}; {candidate['locatable_corners']} corner(s)"
    p.text(f'{identity} / R{row:03d}',x,y,9.5,True)
    p.text(detail,x,y-14,7.8,color=MUTED)
    crop=region([target['bbox_xyxy']],margin=22,minimum=(120,80))
    overlays=[(target,CYAN,False,'')]
    # Only draw the retained box when its centre/extent is relevant to this crop.
    if pred and iou(target['bbox_xyxy'],pred['bbox_xyxy'])>0:
        overlays.append((pred,MAGENTA,True,''))
    half=(width-10)/2
    p.panel(data['images'][identity],crop,x,y-24,half,108)
    p.panel(data['images'][identity],crop,x+half+10,y-24,half,108,overlays)
    if kind=='recovery':
        errors=[f'{v:.2f}' for v in candidate['errors_px'] if v is not None]
        note='Recovered by the fixed one-to-one matcher. '+('Locatable-corner error: '+', '.join(errors)+' px.' if errors else 'No locatable corner is scored for this target.')
    else:
        box=target['bbox_xyxy']
        if box[1]<1:appearance='A narrow front is truncated by the upper image boundary.'
        elif box[3]>719:appearance='A partial front reaches the lower image boundary.'
        else:appearance='A partially visible front remains unmatched at the fixed IoU rule.'
        note=appearance+' Review its owned visible extent and partial-front supervision. Saved final outputs do not reveal whether the current miss arose before confidence filtering or suppression.'
    p.para(esc(note),x,y-144,width,size=8.2,leading=11.2,bottom=y-height+9)


def results(data,output):
    p=Report(output,'R52 | PR146 results, errors and annotation thesis','R52 | 27 Nov 2026 HKT | PR146 saved-output replay | Same 20 development scenes; no model promotion')
    reports=data['summary']['reports'];cases=data['cases']
    p.start('Completed experiment','Four recovered fronts, more unmatched outputs','PR146: classification BCE weight 1 versus 2 on anchors assigned to partially visible fronts. Same 142 training photographs.')
    rows=[]
    for name in ['frozen666','historical663','r38','r50']:
        a,b=reports[name]['postclip']['reference'],reports[name]['postclip']['candidate']
        rows.append([name,f"{a['detection']['fn']} -> {b['detection']['fn']}",f"{a['detection']['fp']} -> {b['detection']['fp']}",f"{a['detection']['duplicate_fp']} -> {b['detection']['duplicate_fp']}",f"{a['corners_by_locatable_count']['2']['within5']} -> {b['corners_by_locatable_count']['2']['within5']} / {b['corners_by_locatable_count']['2']['eligible']}"])
    y=p.table(['Reference version','Missed targets','Unmatched outputs','Formal duplicates','Both corners <=5 px'],rows,H-114,[133,142,150,145,BW-570],height=31)
    y=p.para('<b>Recommendation: retain the selected model and use this as evidence about the tradeoff.</b> Under R50, the treatment recovers four targets with no newly missed targets, while unmatched outputs rise by 28: ten more formal duplicates and eighteen more other unmatched outputs. The three empty scenes still have zero retained outputs.',M,y-2,size=10,leading=14)
    a,b=reports['r50']['postclip']['reference']['detection'],reports['r50']['postclip']['candidate']['detection']
    precision=lambda x:100*x['tp']/x['prediction_count'];recall=lambda x:100*x['tp']/x['gold_count']
    y=p.para(f"At the fixed operating point, R50 precision changes from {precision(a):.2f}% to {precision(b):.2f}% and recall from {recall(a):.2f}% to {recall(b):.2f}%. These are fixed-threshold counts, not AP. R38 and R50 are alternate annotations of the same scenes, not independent confirmation sets.",M,y-15,size=9.5,leading=13.5)
    y=p.para('<b>Evidence checks.</b> The 43,528,788-byte archive and all 518 inventoried payload files verify. Sixteen score tables, 320 per-image reports and 40 postclip filters replay exactly. Both returned training arms completed 30 epochs, 1,080 batches, 80 optimizer updates and 4,260 image presentations. The reference predictions exactly reproduce PR137 on all 20 scenes.',M,y-15,size=9,leading=13)
    p.para('The audit replays saved evidence on CPU; it does not retrain or rerun inference. A returned on-screen historical663 reference miss count differs from the stored JSON: the replayed value is 30. Model selection remains PR35. There is no automatic promotion.',M,y-15,size=9,leading=13,color=MUTED)
    p.end('result_summary')

    p.start('Error analysis','Separate presence, extent and corner accuracy','R50 is the current development-review reference; the frozen666 result remains separately reported.')
    rows=[]
    for n in ['2','1','0']:
        a=reports['r50']['postclip']['reference']['corners_by_locatable_count'][n]
        b=reports['r50']['postclip']['candidate']['corners_by_locatable_count'][n]
        rows.append([n,b['eligible'],f"{a['eligible']-a['matched']} -> {b['eligible']-b['matched']}",f"{a['within5']} -> {b['within5']}" if n!='0' else 'not applicable'])
    y=p.table(['Locatable corners','Reference targets','Misses: control -> treatment','All available corners <=5 px'],rows,H-114,[150,145,230,BW-525],height=31)
    y=p.para('<b>All 11 remaining R50 misses are partial fronts:</b> nine have one locatable corner and two have none. All 583 two-corner targets match. A matched box can still have inaccurate corners: the candidate has 16 two-corner and three one-corner matches failing the 5 px criterion. These 19 are distinct from the 11 misses.',M,y-3,size=10,leading=14)
    y=p.para('<b>The one additional two-corner success is near the threshold.</b> For new_000010 / R020, the worst corner error changes from 5.094 px to 4.984 px, approximately 0.11 px. Both-corner success at 10 px stays 582/583. This is not evidence of a broad improvement in landmark precision.',M,y-16,size=10,leading=14)
    y=p.para('<b>Unmatched is a relationship to the current reference.</b> A formal duplicate overlaps an already matched same-class target at IoU >=0.50. The other 89 unmatched outputs can be malformed real-front predictions, background responses, excluded fragments or uncertain cases. The visual inventory below keeps these interpretations separate from the formal count.',M,y-16,size=10,leading=14)
    p.para('The existing postclip filter removes nine outputs per arm without changing any matched/missed identity. It helps these outputs, but does not explain the 11 remaining misses. PR146 does not return full decoded proposal tensors. Earlier confidence/NMS stage diagnoses belong to the previous checkpoint; they cannot be asserted as this candidate\'s internal failure causes.',M,y-16,size=9.5,leading=13.5,color=MUTED)
    p.end('error_interpretation')

    p.start('Recovered targets','The four recovered development fronts','Cyan: R50 target. Dashed magenta: PR146 candidate. Raw crop at left; identical pixels with overlays at right.')
    width=(BW-20)/2
    for i,c in enumerate(cases['recoveries']):card(p,data,c,M+(i%2)*(width+20),H-115-(i//2)*219,width,211,'recovery')
    p.end('recovered_targets',cases=[f"{c['image']}:R{c['row']:03d}" for c in cases['recoveries']])
    for offset in range(0,len(cases['remaining_misses']),4):
        part=cases['remaining_misses'][offset:offset+4]
        p.start('Remaining misses',f"Partial fronts still missed: {offset+1}-{offset+len(part)}",'Cyan: reference target. Dashed magenta, when shown: nearest retained candidate; it is not a recovered target.')
        for i,c in enumerate(part):card(p,data,c,M+(i%2)*(width+20),H-115-(i//2)*219,width,211,'miss')
        p.end('remaining_misses',cases=[f"{c['image']}:R{c['row']:03d}" for c in part])

    p.start('Corner errors','Matched boxes with a corner error above 5 px','The denominator is all eligible targets; missing detections are reported separately, rather than disappearing from the result.')
    rows=[]
    for c in cases['candidate_corner_failures']:
        errors=', '.join(f'{v:.3f}' for v in c['candidate']['errors_px'] if v is not None)
        rows.append([f"{c['image']} / R{c['row']:03d}",c['candidate']['locatable_corners'],errors,'yes' if c['candidate']['within10'] else 'no'])
    y=p.table(['R50 identity','Available corners','Candidate errors (px)','All <=10 px'],rows,H-112,[285,105,240,BW-630],height=19,size=8.2)
    p.para('Coordinates and matched prediction IDs for every case are in case-inventory.json. Review corner identity and visible contour before adjusting any annotation. Do not tune the 5 px threshold to turn a near-boundary failure into a success.',M,y+3,size=8.6,leading=12,bottom=39)
    p.end('corner_error_inventory',cases=[f"{c['image']}:R{c['row']:03d}" for c in cases['candidate_corner_failures']])

    p.start('Unmatched outputs','Visual types guide different next actions','These are agent-assisted interpretations of original pixels, not newly certified background labels or an automatic relabeling rule.')
    tally=Counter(visual_type(v) for v in data['observations'].values())
    actions={'formal_duplicate':'Inspect instance identity and suppression/ranking; preserve distinct adjacent fronts.',
        'front_extent_or_fragment':'Review target ownership and localization; a confidence filter cannot repair extent.',
        'rack_or_gripper':'Consider verified difficult negatives only after confirming annotation completeness.',
        'keyboard_or_other_background':'Check scene-level context and negative coverage without adding training leakage.',
        'reference_policy_exclusion':'Preserve the recorded exclusion and report its scoring effect.',
        'ambiguous_partial_fragment':'Keep uncertainty; widen the original-image review before changing labels.'}
    rows=[[VISUAL_NAMES.get(k,k),v,actions.get(k,'Review the original pixels and keep uncertainty explicit.')] for k,v in sorted(tally.items())]
    y=p.table(['Visual subtype','Outputs','Smallest useful next action'],rows,H-113,[180,65,BW-245],height=43,size=8.8)
    y=p.para('The pages that follow include all 112 candidate unmatched outputs, with stable candidate prediction positions and original crops. Magenta is the prediction. Cyan, when shown, is the nearest overlapping R50 target. Visual subtype and formal matcher status are independent fields; see the machine-readable ledger for confidence and action details.',M,y-5,size=9.5,leading=13.5)
    p.para('The earlier user-directed exclusion of the visible partial fragment in new_000009 remains in force. Outputs on that fragment must not be declared verified background merely because they are absent from R50. This report makes no new annotation from a prediction and changes no deployment filter.',M,y-16,size=9.5,leading=13.5)
    p.end('unmatched_visual_taxonomy',visual_counts=dict(tally))

    for offset in range(0,len(cases['candidate_unmatched']),8):
        part=cases['candidate_unmatched'][offset:offset+8]
        p.start('Unmatched atlas',f"Candidate outputs {offset+1}-{offset+len(part)} of 112",'Raw left, overlay right. out = postclip position; P = retained primary prediction ID, both in the PR146 candidate arm.')
        for i,c in enumerate(part):
            x=M+(i%2)*(width+20);y=H-110-(i//2)*113
            obs=data['observations'][(c['image'],c['prediction_number'])]
            pred=c['prediction'];subtype=VISUAL_NAMES.get(visual_type(obs),visual_type(obs))
            p.text(f"{c['image']} / out{c['prediction_number']:03d} (P{pred['primary_prediction_id']:03d}) | {pred['confidence']:.3f}",x,y,8.1,True)
            p.text(subtype+(' | duplicate by matcher' if c['category']=='formal_duplicate' else ''),x,y-11,7.1,color=MUTED)
            crop=region([pred['bbox_xyxy']],margin=22,minimum=(110,72))
            overlays=[]
            if c['nearest_row'] is not None:
                target=data['gold'][c['image']][c['nearest_row']-1]
                overlays.append((target,CYAN,False,''))
            overlays.append((pred,MAGENTA,False,''))
            half=(width-10)/2
            p.panel(data['images'][c['image']],crop,x,y-17,half,58)
            p.panel(data['images'][c['image']],crop,x+half+10,y-17,half,58,overlays)
            observation=obs.get('observation',obs.get('visual_observation','Inspect original context; uncertainty retained.'))
            p.para(esc(observation),x,y-80,width,size=7.1,leading=9.3,bottom=y-109)
        p.end('unmatched_atlas',cases=[{'image':c['image'],'postclip_position':c['prediction_number'],'primary_prediction_id':c['prediction']['primary_prediction_id']} for c in part])

    p.start('Thesis proposal','Study annotation quality with a fixed detector','Suggested title: Geometry-guided annotation quality control for partially occluded cassette-front detection and corner localization.')
    y=p.para('<b>Research question.</b> Can explicit ownership rules and simple checks between neighbouring instances help a reviewer find real annotation problems more efficiently, and do accepted corrections improve a detector when other factors are fixed?',M,H-112,size=11,leading=15)
    rows=[['1. Define the target','Visible cassette front; own rim and shaded material; actual lower corners; explicit unavailable states.'],
          ['2. Build an auditable queue','All-four-side geometry checks, stable IDs, signed distances, full original context and human decisions.'],
          ['3. Measure review value','Confirmed issues in the first K reviewed cases; compare with random/sequential review at the same budget.'],
          ['4. Check what the queue misses','Exhaustive review of a subset or a prespecified sample of unflagged regions; label-only checks miss absent rows.'],
          ['5. Test accepted label changes','Same images, model, initialization, seed, recipe and training budget; change only eligible training labels.'],
          ['6. Report limitations','Agent review is preliminary. Record real reviewer time prospectively; do not invent efficiency or agreement.']]
    y=p.table(['Part','Evidence to collect'],rows,y-18,[175,BW-175],height=44,size=9)
    p.para('This is a proposed empirical annotation study, not a professor-approved topic or a claim of a novel detector. It can produce a sound thesis even if downstream gains are mixed, provided the protocol and negative results are reported honestly. The complete proposal and chapter outline are in docs/thesis-plan-r52.md.',M,y-2,size=9.5,leading=13.5)
    p.end('thesis_question_and_review_study')

    p.start('Controlled evaluation','Keep label effects separate from model effects','Two training conditions can be scored against two references. Four score cells do not require four training runs.')
    y=p.table(['Model','Existing frozen reference','Adjudicated reference'],[
        ['Trained with previous labels','A','B'],['Trained with accepted corrected labels','C','D']],H-112,[300,238,BW-538],height=37,size=10)
    y=p.para('<b>A versus C, and B versus D:</b> training-label effects under a fixed measuring reference. <b>A versus B, or C versus D:</b> the effect of changing the measuring reference for unchanged predictions. Never report a better score caused only by changing the answer key as a model improvement.',M,y-5,size=10,leading=14)
    y=p.para('<b>The immediate R52 notebook uses CPU only.</b> It rebuilds the return audit, geometry checks and these reports from pinned evidence. The nominal side refinements are approximately 2 px on one training photograph. That supports a reviewable label proposal, not an expectation that another expensive training run will improve the model.',M,y-17,size=10,leading=14)
    y=p.para('<b>A later label-only pilot:</b> keep the 142-photo membership and native loss fixed. Compare the previous labels with accepted corrections on already eligible images. Adding frame_0001, increasing resolution or changing the loss belongs to a separate experiment. Keep all held photographs under the existing whole-photo rule.',M,y-17,size=10,leading=14)
    y=p.para('<b>Evidence limits.</b> The 20 repeatedly reviewed photographs are a development set. Different annotation versions are not independent tests. Recover capture/near-duplicate groups before designing a future split. Many boxes from a few related scenes are not independent experimental samples. Use additional paired seeds only when a consequential result needs replication.',M,y-17,size=9.5,leading=13.5)
    p.para('Possible chapters: problem and target definition; related work; versioned dataset and review method; review-efficiency experiment; controlled training/evaluation; limitations and conclusions. A narrower alternative is annotation uncertainty and score stability for small partially occluded fronts.',M,y-16,size=9.5,leading=13.5)
    p.end('thesis_controlled_evaluation')

    p.start('Primary reading','Six references with a clear purpose','These papers motivate mechanisms and study design. None establishes that a particular cassette amendment must improve detection.')
    refs=[
      ('Ma, Ushiku and Sagara | CVPR Workshops 2022','The Effect of Improving Annotation Quality on Object Detection Datasets: A Preliminary Study','https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf','Professional reannotation and explicit guidelines produced mixed detector results. Measure annotation consistency and downstream performance separately.'),
      ('Tkachenko, Thyagarajan and Mueller | ICML DMLR Workshop 2023','ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data','https://arxiv.org/abs/2309.00832','Prioritize overlooked, misplaced and wrongly classified boxes for human review. Their prediction-based scoring uses out-of-sample models; fitted training predictions are not an unbiased substitute.'),
      ('Bolya, Foley, Hays and Hoffman | ECCV 2020','TIDE: A General Toolbox for Identifying Object Detection Errors','https://arxiv.org/abs/2008.08115','Separate detection error types and their influence on AP. Our fixed-threshold project counts are complementary; this report does not claim to have run TIDE.'),
      ('Mottaghi et al. | CVPR 2014','The Role of Context for Object Detection and Semantic Segmentation in the Wild','https://cs.stanford.edu/~roozbeh/pascal-context/mottaghi_et_al_cvpr14.pdf','Studies local and global context as recognition evidence. This supports examining surroundings without treating their material as the target extent.'),
      ('Murrugarra-Llerena and Jung | WACV 2025','Noise-Aware Evaluation of Object Detectors','https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html','Quantifies annotation discrepancies and their effect on evaluation, especially small objects and strict IoU acceptance. Use sensitivity analysis, not post-hoc threshold relaxation.'),
      ('Zhang, Benenson and Schiele | CVPR 2017','CityPersons: A Diverse Dataset for Pedestrian Detection','https://arxiv.org/abs/1702.05693','Distinguishes visible tight boxes from full-body amodal boxes. Declare the target convention explicitly; our current cassette task uses the visible front.')]
    y=H-111
    for authors,title,url,note in refs:
        p.text(authors,M,y,8.5,True,TEAL)
        y=p.para('<link href="'+html.escape(url,quote=True)+'" color="#173344"><b>'+esc(title)+'</b></link><br/>'+esc(note),M,y-7,size=8.6,leading=11.6,bottom=39)-17
    p.end('primary_references')
    return p.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--originals',type=Path,required=True)
    parser.add_argument('--supplementary',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();out=args.output_dir.resolve()
    need(root not in out.parents and out!=root,'Keep binary deliverables outside the source repository')
    data=collect(root,args.archive,args.originals,args.supplementary,args.audit_dir)
    out.mkdir(parents=True,exist_ok=True)
    outputs={}
    for filename,renderer in [('r52-cassette-amendments.pdf',amendments),('r52-results-and-thesis.pdf',results)]:
        path=out/filename
        metadata=renderer(data,path)
        metadata.update({'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size})
        outputs[filename]=metadata
    receipt={'schema':'baomai-r52-report-verification-v1','status':'passed','scope':'original-image rendering and layout bounds; no training or inference',
        'archive_sha256':ARCHIVE_SHA,'report_builder_sha256':sha(Path(__file__).read_bytes()),
        'review_manifest_sha256':sha((root/REVIEW/'manifest.json').read_bytes()),
        'visual_observations_sha256':sha((root/'results/evaluation/pr146-r51-20261127/visual-observations.json').read_bytes()),
        'original_image_sha256':data['image_hashes'],'outputs':outputs,
        'visual_qa':'Rendered page inspection is recorded separately; layout bounds alone do not establish visual quality.'}
    (out/'report-verification.json').write_bytes(encoded(receipt))
    print(json.dumps({k:{f:v[f] for f in ['pages','bytes','sha256']} for k,v in outputs.items()},indent=2))


if __name__=='__main__':
    main()
