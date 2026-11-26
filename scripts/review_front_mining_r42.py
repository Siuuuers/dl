#!/usr/bin/env python3
"""Verify and package the R42 annotation review; never infer, train or approve labels.

The viewer draws original pixels with recorded overlays. It cannot establish
physical correctness. Complete-image status comes from the documented review.
"""
import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


COUNT_FIELDS = {
    'baseline_rows', 'candidate_rows', 'retained_source_rows',
    'replaced_source_rows', 'deleted_source_rows', 'added_rows',
    'held_source_rows', 'corner_coordinate_slots_changed',
    'corner_availability_slots_changed',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f'nonfinite JSON: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def archive_payload(path):
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        names = [item.filename for item in items]
        require(len(names) == len(set(names)), 'duplicate ZIP member')
        require(sum(i.file_size for i in items) < 100_000_000, 'unexpected expanded size')
        for item in items:
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.filename,
                    f'unsafe ZIP member: {item.filename}')
            require(not item.is_dir() and ((item.external_attr >> 16) & 0o170000) != 0o120000,
                    f'non-file member: {item.filename}')
        require(archive.testzip() is None, 'ZIP CRC failure')
        data = {name: archive.read(name) for name in names}
    inventory = load_json(data['inventory.json'])
    require(set(data) == set(inventory) | {'inventory.json'}, 'inventory/member mismatch')
    for name, entry in inventory.items():
        require(entry == {'sha256': digest(data[name]), 'bytes': len(data[name])}, f'inventory: {name}')
    return data


def pose(line, name):
    t = line.split()
    require(len(t) == 11, f'{name}: expected11 tokens')
    a = list(map(float, t))
    require(all(math.isfinite(x) for x in a) and a[0] == 0, f'{name}: class/finite')
    x, y, w, h = a[1:5]
    box = [(x-w/2)*1280, (y-h/2)*720, (x+w/2)*1280, (y+h/2)*720]
    eps = .0001
    require(w > 0 and h > 0 and box[0] >= -eps and box[1] >= -eps and
            box[2] <= 1280+eps and box[3] <= 720+eps, f'{name}: box')
    for k in [5, 8]:
        px, py, v = a[k:k+3]
        require(v in [0, 2], f'{name}: visibility')
        if v == 0:
            require(px == py == 0, f'{name}: canonical missing')
        else:
            require(0 <= px < 1 and 0 <= py < 1, f'{name}: point bounds')
            require(box[0]-eps <= px*1280 <= box[2]+eps and box[1]-eps <= py*720 <= box[3]+eps,
                    f'{name}: point containment')
    require(not (a[7] and a[10]) or a[5] <= a[8], f'{name}: point order')
    return a


def candidate_bytes(frame):
    rows = [r['after_line'] for r in frame['rows'] + frame['additions']
            if r['after_line'] is not None]
    return ('\n'.join(rows) + ('\n' if rows else '')).encode()


def figure_contract(payload, manifest):
    """Return the exact relative figure paths and hashes declared by the report."""
    name = 'fixed-prediction-diagnostic.json'
    if name not in payload:
        return {}
    diagnostic = load_json(payload[name])
    require(diagnostic['provenance']['manifest_sha256'] == digest(payload['manifest.json']),
            'diagnostic manifest identity')
    require(set(diagnostic['scored_image_ids']) == set(manifest['annotation_complete_ids']) and
            set(diagnostic['excluded_held_image_ids']) == set(manifest['held_image_ids']),
            'diagnostic image scope')
    require(diagnostic['inference_performed'] is False and
            diagnostic['training_performed'] is False and
            diagnostic['human_review_claimed'] is False, 'diagnostic review scope')
    cases = diagnostic.get('reviewed_unmatched_triage', {}).get('cases', [])
    figures = {}
    for case in cases:
        fid, row, name = case['image_id'], case['reference_row'], case['figure']
        require(fid in manifest['annotation_complete_ids'] and type(row) is int and row > 0,
                'diagnostic figure reference')
        expected = f'figures/{fid}_row_{row:03d}.png'
        require(name == expected and PurePosixPath(name).parts == ('figures', PurePosixPath(name).name),
                f'unsafe or mismatched diagnostic figure: {name}')
        sha = case['figure_sha256']
        require(isinstance(sha, str) and len(sha) == 64 and
                all(c in '0123456789abcdef' for c in sha), f'{name}: figure SHA256')
        require(name not in figures, f'duplicate diagnostic figure: {name}')
        figures[name] = sha
    return figures


def render_html(manifest, payload, figures):
    """Render a self-contained viewer, also valid in a notebook iframe srcdoc."""
    display_data = dict(frames=manifest['frames'], images={})
    for f in manifest['frames']:
        fid = f['image_id']
        jpg = payload['originals/'+fid+'.jpg']
        display_data['images'][fid] = 'data:image/jpeg;base64,'+base64.b64encode(jpg).decode()
    if figures:
        display_data['diagnostic_figures'] = [
            dict(name=name, image='data:image/png;base64,'+base64.b64encode(payload[name]).decode())
            for name in sorted(figures)
        ]
    encoded = json.dumps(display_data, separators=(',', ':')).replace('<', '\\u003c')
    return HTML.replace('__DATA__', encoded).encode()


def verify(manifest, source):
    require(manifest['schema_version'] == 1, 'schema')
    require(all(manifest[key] is False for key in
                ['training_run', 'model_promotion', 'active_dataset_modified',
                 'human_review_performed']), 'review scope')
    require(set(manifest['counts']) == COUNT_FIELDS and
            all(type(n) is int and n >= 0 for n in manifest['counts'].values()),
            'count schema')
    counts = dict.fromkeys(manifest['counts'], 0)
    complete, held, ids = [], [], set()
    for f in manifest['frames']:
        fid = f['image_id']
        require(fid not in ids and f['image_size'] == [1280, 720], 'frame identity/size')
        ids.add(fid)
        prefix = 'review/' + fid + '/'
        for name, key in [('original.jpg','source_image_sha256'), ('source.txt','source_label_sha256'),
                          (fid+'.txt','baseline_label_sha256')]:
            require(digest(source[prefix+name]) == f[key], f'{fid}: {name} hash')
        original = source[prefix+'source.txt'].decode().splitlines()
        baseline = source[prefix+fid+'.txt'].decode().splitlines()
        require(len(original) == len(baseline) == f['baseline_rows'] == len(f['rows']), f'{fid}: row count')
        require(all(type(r['source_row']) is int for r in f['rows']) and
                all(r['source_row'] is None for r in f['additions']), f'{fid}: row identity types')
        require([r['source_row'] for r in f['rows']] == list(range(1,len(baseline)+1)), f'{fid}: source ID order')
        output, row_ids, held_ids = [], set(), []
        for r in f['rows'] + f['additions']:
            sid = r['stable_id']
            require(sid not in row_ids and sid.startswith(fid+':'), f'{fid}: stable identity')
            row_ids.add(sid)
            if r['source_row'] is not None:
                n = r['source_row']
                require(sid == f'{fid}:source:{n:03d}', f'{sid}: source identity mapping')
                require(r['before_line'] == baseline[n-1] and r['source_line'] == original[n-1], f'{sid}: source text')
                expected_action = 'delete' if r['after_line'] is None else ('retain' if r['after_line']==r['before_line'] else 'replace')
                require(r['action'] == expected_action, f'{sid}: operation')
                counts[{'retain':'retained_source_rows','replace':'replaced_source_rows','delete':'deleted_source_rows'}[expected_action]] += 1
                counts['held_source_rows'] += bool(r['unresolved_fields'])
            else:
                require(r['action'] == 'add' and r['before_line'] is None and
                        r.get('source_line') is None and sid.startswith(fid+':add:'),
                        f'{sid}: addition identity')
                counts['added_rows'] += 1
            after = r['after_line']
            actual = list(range(11)) if after is None or r['before_line'] is None else [i for i,(a,b) in enumerate(zip(r['before_line'].split(),after.split())) if a!=b]
            require(r['changed_token_indices'] == actual, f'{sid}: changed token declaration')
            require(isinstance(r['reason'], str) and bool(r['reason'].strip()), f'{sid}: missing reason')
            for key in ['unresolved_fields', 'unresolved_reasons']:
                require(isinstance(r[key], list) and
                        all(isinstance(v, str) and bool(v.strip()) for v in r[key]),
                        f'{sid}: {key} must be a list of nonempty strings')
            require(len(r['unresolved_fields']) == len(set(r['unresolved_fields'])) and
                    bool(r['unresolved_fields']) == bool(r['unresolved_reasons']),
                    f'{sid}: hold fields/reasons')
            if r['unresolved_fields']:
                held_ids.append(sid)
                require(bool(r['unresolved_reasons']), f'{sid}: hold reason')
            if after is None:
                require(r['output_row'] is None, f'{sid}: deleted output row')
            else:
                b = pose(after, sid)
                output.append(after)
                require(type(r['output_row']) is int and r['output_row'] == len(output), f'{sid}: output mapping')
                if r['before_line'] is not None:
                    a = list(map(float,r['before_line'].split()))
                    for k in [5,8]:
                        counts['corner_coordinate_slots_changed'] += a[k+2]==b[k+2]==2 and a[k:k+2]!=b[k:k+2]
                        counts['corner_availability_slots_changed'] += a[k+2]!=b[k+2]
        require(digest(candidate_bytes(f)) == f['candidate_label_sha256'], f'{fid}: replay hash')
        require(len(output) == f['candidate_rows'] and held_ids == f['unresolved_stable_ids'], f'{fid}: review accounting')
        require(f['full_image_inspected'] is True and f['all_baseline_rows_inspected'] is True, f'{fid}: review scope')
        require(f['eligible_for_independent_evaluation'] is False and f['training_split']=='unassigned', f'{fid}: split scope')
        require(type(f['annotation_complete']) is bool and
                type(f['eligible_for_candidate_export']) is bool and
                f['eligible_for_candidate_export'] == f['annotation_complete'], f'{fid}: export gate')
        if f['annotation_complete']:
            require(not held_ids and f['disposition']=='assistant_review_complete', f'{fid}: complete image has holds')
            complete.append(fid)
        else:
            require(bool(held_ids) and f['disposition']=='held', f'{fid}: held image missing hold ledger')
            held.append(fid)
        counts['baseline_rows'] += len(baseline)
        counts['candidate_rows'] += len(output)
    require(counts == manifest['counts'], 'aggregate counts')
    require(complete == manifest['annotation_complete_ids'] and held == manifest['held_image_ids'], 'image disposition lists')
    require(len(ids)==12 and counts['baseline_rows']==507, 'selection coverage')
    return dict(passed=True,scope='Artifact integrity, exact operation replay, stable identities, declared whole-image gates and strict geometry; physical decisions remain assistant visual review.',counts=counts,
                annotation_complete_ids=complete,held_image_ids=held,human_review_performed=False,training=False,promotion=False)


def verify_capsule(payload):
    """Check relationships between packaged files as well as ZIP byte integrity.

    Checksums establish internal consistency, not authorship or human approval.
    Verify the delivered archive hash against the trusted publication separately.
    """
    manifest = load_json(payload['manifest.json'])
    source, expected_candidates = {}, {}
    for f in manifest['frames']:
        fid = f['image_id']; prefix = 'review/' + fid + '/'
        jpg = payload['originals/' + fid + '.jpg']
        source[prefix+'original.jpg'] = jpg
        source[prefix+'source.txt'] = payload['source-labels/'+fid+'.txt']
        source[prefix+fid+'.txt'] = payload['baseline-labels/'+fid+'.txt']
        proposal = payload['proposals/'+fid+'.txt']
        require(proposal == candidate_bytes(f), fid+': proposal artifact differs from replay')
        if f['eligible_for_candidate_export']:
            expected_candidates['complete-candidates/labels/'+fid+'.txt'] = proposal
    result = verify(manifest, source)
    actual = {k for k in payload if k.startswith('complete-candidates/labels/')}
    require(actual == set(expected_candidates), 'candidate export includes wrong paths/images')
    for name, expected in expected_candidates.items():
        require(payload[name] == expected, name+': candidate artifact differs from proposal')
    require(load_json(payload['verification.json']) == result, 'stored verification differs from replay')
    figures = figure_contract(payload, manifest)
    require({k for k in payload if k.startswith('figures/')} == set(figures),
            'diagnostic figure member set')
    for name, sha in figures.items():
        require(payload[name].startswith(b'\x89PNG\r\n\x1a\n') and digest(payload[name]) == sha,
                f'{name}: diagnostic figure identity')
    require(payload['review.html'] == render_html(manifest, payload, figures),
            'review HTML differs from canonical viewer/manifest/images')
    return result


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R42 cassette-front review</title><style>
:root{font-family:system-ui,sans-serif;color:#18353b;background:#f2f6f6}body{max-width:1440px;margin:auto;padding:20px}h1{margin:0;font-size:1.65rem}p{line-height:1.55}.muted{color:#586e72}header,.card{background:white;border:1px solid #d8e5e5;border-radius:12px;padding:18px;margin-bottom:16px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}select,button{padding:10px;border:1px solid #90aaa9;border-radius:7px;background:white;color:#18353b;font:inherit}button{cursor:pointer}button:focus-visible,select:focus-visible{outline:3px solid #007f85}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}canvas{width:100%;height:auto;display:block;background:#111;image-rendering:pixelated}.panel{min-width:0}h2{font-size:1rem;margin:0 0 8px}#full{max-width:1280px;margin:auto}#notes{white-space:pre-wrap}details{margin-top:10px}pre{overflow:auto;font-size:.78rem}#status{font-weight:650}.held{color:#95550b}.complete{color:#076f55}#help{font-size:.9rem}@media(max-width:760px){body{padding:10px}.grid{grid-template-columns:1fr}}
</style><body><header><h1>R42 · Cassette-front review</h1><p>Original photographs, exact label changes, and unresolved fields. Assistant review; no human sign-off or model-training result.</p><div class="controls"><label>Image <select id="frame"></select></label><label>Rows <select id="filter"><option value="changed">Changed or held</option><option value="all">All source rows</option><option value="held">Held only</option></select></label><button id="prev">Previous row</button><select id="row" aria-label="Stable source row"></select><button id="next">Next row</button></div><p id="status"></p><p id="help" class="muted">Every row refers to the immutable upstream identity. Additions keep a separate ID. Green = left corner; blue = right corner; cyan = unchanged field; magenta = changed field. Boxes and points are judged separately. Select “All source rows” to inspect completeness.</p></header>
<section class="card"><div class="grid"><div class="panel"><h2>Original pixels</h2><canvas id="raw"></canvas></div><div class="panel"><h2>Working baseline</h2><canvas id="before"></canvas></div><div class="panel"><h2>Reviewed proposal</h2><canvas id="after"></canvas></div></div><p id="identity"></p><p id="notes"></p><details><summary>Exact row text and declared changes</summary><pre id="tokens"></pre></details></section>
<section class="card"><div class="controls"><h2>Whole original frame</h2><label><input type="checkbox" id="overlay" checked> Show proposal overlay</label><label><input type="checkbox" id="sourceOverlay"> Show upstream source instead</label></div><canvas id="full"></canvas></section>
<section class="card" id="diagnostics" hidden><h2>Unmatched reference cases</h2><p>Saved predictions compared with the reviewed labels. These figures show a fixed-output diagnostic, with no new training or independent test.</p><div id="figureCases"></div></section>
<footer class="muted"><p>Held images are excluded as whole images, including their useful corrections. Coordinates on soft contours are nominal estimates. Annotation-complete images remain candidate annotations with no assigned training split; none is an independent test scene.</p></footer>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
let frame,rows=[],active,img=new Image(),generation=0,imageReady=false;
for(const figure of data.diagnostic_figures||[]){const detail=document.createElement('details'),summary=document.createElement('summary'),picture=document.createElement('img');summary.textContent=figure.name.split('/').pop();picture.src=figure.image;picture.alt='Fixed-prediction diagnostic: '+summary.textContent;picture.style.width='100%';picture.style.height='auto';picture.style.display='block';detail.append(summary,picture);$('figureCases').appendChild(detail);$('diagnostics').hidden=false}
data.frames.forEach(f=>{const o=document.createElement('option');o.value=f.image_id;o.textContent=f.image_id+' · '+(f.annotation_complete?'complete candidate':'held');$('frame').appendChild(o)});
function parse(line){if(!line)return null;let a=line.trim().split(/\s+/).map(Number);return{a,b:[(a[1]-a[3]/2)*1280,(a[2]-a[4]/2)*720,(a[1]+a[3]/2)*1280,(a[2]+a[4]/2)*720]}}
function draw(ctx,line,ox,oy,changed=[],label=''){const p=parse(line);if(!p)return;const a=p.a,b=p.b;ctx.lineWidth=.7;ctx.strokeStyle=changed.some(i=>i>=1&&i<=4)?'#ff33cf':'#16dce3';ctx.strokeRect(b[0]-ox,b[1]-oy,b[2]-b[0],b[3]-b[1]);for(const k of[5,8])if(a[k+2]){ctx.beginPath();ctx.strokeStyle=changed.some(i=>i>=k&&i<=k+2)?'#ff33cf':k===5?'#24ff65':'#008eff';ctx.arc(a[k]*1280-ox,a[k+1]*720-oy,1.4,0,Math.PI*2);ctx.stroke()}if(label){ctx.fillStyle='#fff';ctx.strokeStyle='#172428';ctx.lineWidth=2;ctx.font='10px sans-serif';ctx.strokeText(label,b[0]-ox,b[1]-oy-2);ctx.fillText(label,b[0]-ox,b[1]-oy-2)}}
function whole(){const c=$('full'),ctx=c.getContext('2d');c.width=1280;c.height=720;if(!imageReady)return;ctx.drawImage(img,0,0);if($('overlay').checked)for(const r of frame.rows.concat(frame.additions)){const line=$('sourceOverlay').checked?r.source_line:r.after_line;draw(ctx,line,0,0,$('sourceOverlay').checked?[]:r.changed_token_indices,String(r.source_row??'new'))}}
function clearRow(message){active=null;$('identity').textContent='';$('notes').textContent=message;$('notes').className='muted';$('tokens').textContent='';$('prev').disabled=true;$('next').disabled=true;for(const id of['raw','before','after']){const c=$(id);c.width=400;c.height=60}}
function render(){whole();if(!imageReady){clearRow('Loading original image…');return}active=rows[Number($('row').value)];if(!active){clearRow('No rows in this filter.');return}$('prev').disabled=$('next').disabled=rows.length<2;const boxes=[parse(active.before_line)?.b,parse(active.after_line)?.b].filter(Boolean);const x=Math.max(0,Math.floor(Math.min(...boxes.map(b=>b[0]))-16)),y=Math.max(0,Math.floor(Math.min(...boxes.map(b=>b[1]))-16)),x2=Math.min(1280,Math.ceil(Math.max(...boxes.map(b=>b[2]))+16)),y2=Math.min(720,Math.ceil(Math.max(...boxes.map(b=>b[3]))+16));for(const [id,line]of[['raw',null],['before',active.before_line],['after',active.after_line]]){const c=$(id);c.width=(x2-x)*4;c.height=(y2-y)*4;const ctx=c.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.scale(4,4);ctx.drawImage(img,x,y,x2-x,y2-y,0,0,x2-x,y2-y);draw(ctx,line,x,y,id==='after'?active.changed_token_indices:[])}$('identity').textContent=active.stable_id+' · '+active.action+' · output row '+(active.output_row??'deleted');$('notes').textContent=active.reason+(active.unresolved_fields.length?'\n\nHELD: '+active.unresolved_fields.join(', ')+'\n'+active.unresolved_reasons.join('\n'):'');$('notes').className=active.unresolved_fields.length?'held':'';$('tokens').textContent=JSON.stringify({source:active.source_line??null,baseline:active.before_line,proposal:active.after_line,changed_token_indices:active.changed_token_indices},null,2)}
function rowOptions(){const mode=$('filter').value;rows=frame.rows.concat(frame.additions).filter(r=>mode==='all'||(mode==='held'?r.unresolved_fields.length>0:r.action!=='retain'||r.unresolved_fields.length>0));$('row').replaceChildren();rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(r.source_row?'Source '+r.source_row:'Added front')+(r.unresolved_fields.length?' · held':'');$('row').appendChild(o)});render()}
function changeFrame(){frame=data.frames.find(f=>f.image_id===$('frame').value);$('status').className=frame.annotation_complete?'complete':'held';$('status').textContent=(frame.annotation_complete?'Complete assistant-reviewed candidate':'Whole image held')+' · '+frame.candidate_rows+' rows · '+frame.unresolved_stable_ids.length+' held rows';imageReady=false;rows=[];$('row').replaceChildren();$('row').disabled=$('filter').disabled=true;render();const g=++generation;img=new Image();img.onload=()=>{if(g===generation){imageReady=true;$('row').disabled=$('filter').disabled=false;rowOptions()}};img.onerror=()=>{if(g===generation)clearRow('Unable to load the original image.')};img.src=data.images[frame.image_id]}
$('frame').onchange=changeFrame;$('filter').onchange=rowOptions;$('row').onchange=render;$('overlay').onchange=whole;$('sourceOverlay').onchange=whole;for(const[id,step]of[['prev',-1],['next',1]])$(id).onclick=()=>{if(rows.length){$('row').value=(Number($('row').value)+step+rows.length)%rows.length;render()}};changeFrame();
</script></body></html>'''


def build(manifest_path, return_path, output, extras, figures_dir=None):
    manifest_bytes = manifest_path.read_bytes()
    manifest = load_json(manifest_bytes)
    returned = return_path.read_bytes()
    require(len(returned)==manifest['return_archive_bytes'] and digest(returned)==manifest['return_archive_sha256'], 'R41 return identity')
    source = archive_payload(io.BytesIO(returned))
    verification = verify(manifest, source)
    payload = {'manifest.json':manifest_bytes, 'verification.json':(json.dumps(verification,indent=2)+'\n').encode(),
               'review.py':Path(__file__).read_bytes()}
    for f in manifest['frames']:
        fid=f['image_id'];prefix='review/'+fid+'/'
        jpg=source[prefix+'original.jpg'];payload['originals/'+fid+'.jpg']=jpg
        payload['source-labels/'+fid+'.txt']=source[prefix+'source.txt']
        payload['baseline-labels/'+fid+'.txt']=source[prefix+fid+'.txt']
        candidate=candidate_bytes(f)
        payload['proposals/'+fid+'.txt']=candidate
        if f['eligible_for_candidate_export']:
            payload['complete-candidates/labels/'+fid+'.txt']=candidate
    payload['complete-candidates/README.txt']=('Annotation-complete candidates only. Original JPEGs are in ../originals/. No training split is assigned. Held images are excluded in their entirety. Do not use this mined pool as an independent test set.\n').encode()
    require(set(extras) <= {'audit.json', 'independent-review.json', 'fixed-prediction-diagnostic.json'},
            'unexpected extra artifact name')
    require(not set(extras).intersection(payload), 'extra artifact overwrites a required member')
    for name, path in extras.items(): payload[name]=path.read_bytes()
    figures = figure_contract(payload, manifest)
    require(not figures or figures_dir is not None, 'supply --diagnostic-figures for the declared case PNGs')
    if figures_dir is not None:
        require(figures_dir.is_dir() and bool(figures), 'diagnostic figure directory/report')
        files = {p.name: p for p in figures_dir.glob('*.png')}
        require(set(files) == {PurePosixPath(name).name for name in figures},
                'diagnostic figure directory differs from declared cases')
        for name, sha in figures.items():
            path = files[PurePosixPath(name).name]
            require(path.is_file() and not path.is_symlink(), f'{name}: expected regular PNG')
            data = path.read_bytes()
            require(data.startswith(b'\x89PNG\r\n\x1a\n') and digest(data) == sha,
                    f'{name}: diagnostic figure identity')
            payload[name] = data
    payload['review.html']=render_html(manifest, payload, figures)
    inventory={name:{'sha256':digest(data),'bytes':len(data)} for name,data in sorted(payload.items())}
    payload['inventory.json']=(json.dumps(inventory,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(2026,11,26,0,0,0));info.external_attr=0o100644<<16
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    verify_capsule(archive_payload(output))
    return dict(verification=verification,archive_sha256=digest(output.read_bytes()),archive_bytes=output.stat().st_size,members=len(payload))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--return-archive',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--independent-review',type=Path)
    p.add_argument('--diagnostic',type=Path)
    p.add_argument('--diagnostic-figures',type=Path,
                   help='Directory containing the exact case PNGs and hashes declared by --diagnostic')
    p.add_argument('--check-capsule',type=Path)
    a=p.parse_args()
    if a.check_capsule:
        result=verify_capsule(archive_payload(a.check_capsule))
    else:
        require(a.manifest and a.return_archive and a.output,'supply manifest, return archive and output')
        extras={name:path for name,path in [('audit.json',a.audit),('independent-review.json',a.independent_review),('fixed-prediction-diagnostic.json',a.diagnostic)] if path}
        result=build(a.manifest,a.return_archive,a.output,extras,a.diagnostic_figures)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
