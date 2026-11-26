"""Render one complete frame per page with preserved R42 and latest R43 labels."""
import base64
import json
from pathlib import PurePosixPath
import re


def render_html(manifest, previous_manifest, payload, figures):
    """Return self-contained HTML bytes; all images also work inside iframe srcdoc."""
    previous = {frame['image_id']: frame for frame in previous_manifest['frames']}
    if len(previous) != len(previous_manifest['frames']):
        raise ValueError('Repeated previous frame identity')
    if (len(manifest['frames']) != len(previous) or
            set(previous) != {frame['image_id'] for frame in manifest['frames']}):
        raise ValueError('Current and previous frame collections differ')
    frames = []
    for frame in manifest['frames']:
        fid = frame['image_id']
        old_frame = previous[fid]
        old_rows = {row['stable_id']: row for row in old_frame['rows'] + old_frame['additions']}
        current_rows = frame['rows'] + frame['additions']
        ids = [row['stable_id'] for row in current_rows]
        if len(ids) != len(set(ids)) or not set(old_rows).issubset(ids):
            raise ValueError(f'{fid}: missing or repeated stable row identity')
        merged = []
        for row in current_rows:
            old = old_rows.get(row['stable_id'])
            merged.append({**row, 'r42_line': old['after_line'] if old else None,
                'r42_reason': old['reason'] if old else 'No entry in the R42 review.',
                'r42_action': old['action'] if old else None,
                'r42_output_row': old['output_row'] if old else None,
                'r42_unresolved_fields': old['unresolved_fields'] if old else [],
                'r42_unresolved_reasons': old['unresolved_reasons'] if old else []})
        jpg = payload['originals/' + fid + '.jpg']
        frames.append({key: value for key, value in frame.items() if key not in ('rows', 'additions')} |
                      {'items': merged, 'r42_annotation_complete': old_frame['annotation_complete'],
                       'image': 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode()})
    diagnostic_figures = []
    for name in sorted(figures):
        filename = PurePosixPath(name).name
        match = re.fullmatch(r'(frame_\d+)_row_(\d+)\.png', filename)
        if match is None or match.group(1) not in previous:
            raise ValueError(f'Unrecognized historical diagnostic figure: {name}')
        diagnostic_figures.append({'name': filename, 'image_id': match.group(1),
            'source_row': int(match.group(2)),
            'image': 'data:image/png;base64,' + base64.b64encode(payload[name]).decode()})
    data = {'frames': frames, 'diagnostic_figures': diagnostic_figures}
    encoded = json.dumps(data, ensure_ascii=True, separators=(',', ':')).replace('<', '\\u003c')
    return HTML.replace('__DATA__', encoded).encode()


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R43 cassette-front review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#18353b;background:#f2f6f6;color-scheme:light}
*{box-sizing:border-box}body{max-width:1520px;margin:auto;padding:20px}h1{font-size:1.7rem;margin:0 0 8px}h2{font-size:1.15rem;margin:0 0 10px}h3{font-size:.93rem;margin:0 0 8px}p{line-height:1.5;margin:10px 0}a{color:#076e78}
header,.card,.frame-navigation{background:white;border:1px solid #d8e5e5;border-radius:12px;padding:18px;margin-bottom:16px}.muted{color:#526c71}.held{color:#92520a}.complete{color:#076b53}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.controls label{display:flex;gap:6px;align-items:center;font-size:.92rem}
button,select{padding:10px;border:1px solid #89a5a7;border-radius:7px;background:white;color:#18353b;font:inherit;max-width:100%}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #00868e;outline-offset:2px}summary{cursor:pointer;line-height:1.5}details{margin-top:12px}details p{margin:8px 0}code,pre{font-family:ui-monospace,monospace}code{overflow-wrap:anywhere}pre{font-size:.78rem;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;background:#f4f7f7;padding:12px;border-radius:7px}
.legend{font-size:.9rem}.legend span{display:inline-block;margin-right:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.row-card{scroll-margin-top:16px}.row-heading{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}.row-heading h2{margin:0}.badge{font-size:.78rem;padding:4px 8px;border-radius:5px;background:#edf3f3;color:#405d64}.badge.amended{background:#fbe8f7;color:#8a216d}.badge.held{background:#fff0d9;color:#92520a}.row-id{font-size:.78rem;margin:7px 0 14px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{min-width:0}.panel canvas{width:100%;height:auto;display:block;image-rendering:pixelated;background:#172428}.panel-note{font-size:.77rem;min-height:1.3em;margin:6px 0;color:#526c71}.latest-panel h3{color:#8a216d}.latest-note,.held-note{white-space:pre-wrap}.held-note{border-left:3px solid #d59637;padding-left:10px}.history p{white-space:pre-wrap}.frame-position{font-size:.9rem;font-weight:650}.frame-navigation{display:flex;gap:12px;align-items:center;justify-content:space-between}.status{font-weight:650}#full{display:block;width:100%;max-width:1280px;height:auto;margin:12px auto 0;image-rendering:pixelated;background:#172428}.empty{padding:22px;border:1px dashed #9bb2b4;border-radius:9px;margin-bottom:16px}.diagnostic-image{display:block;width:100%;max-width:960px;height:auto;margin:12px auto}.count{font-size:.9rem}.section-intro{margin:20px 0 12px}.history-label{font-weight:650}
@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){body{padding:9px}header,.card,.frame-navigation{padding:12px;margin-bottom:12px;border-radius:9px}h1{font-size:1.4rem}.grid{gap:8px}.panel h3{font-size:.82rem}.controls{gap:8px}button,select{padding:9px;font-size:.89rem}.row-heading h2{font-size:1.02rem}.row-id{font-size:.72rem}.frame-navigation{gap:7px}.frame-position{font-size:.8rem}.panel-note{font-size:.71rem}.legend{font-size:.8rem}}
@media print{body{max-width:none;background:white;padding:0}.controls,.frame-navigation,#photoOrigin{display:none}.card{break-inside:avoid}.grid{grid-template-columns:repeat(4,minmax(0,1fr))}}
</style></head><body>
<header id="top"><h1>R43 · Cassette-front review</h1>
<p>One photograph per page, with every row together. The earlier labels remain beside the newest amendments.</p>
<div class="controls" aria-label="Frame navigation"><button id="previousFrame">Previous frame</button><label>Frame <select id="frame"></select></label><button id="nextFrame">Next frame</button><span id="positionTop" class="frame-position"></span></div>
<p id="status" class="status" aria-live="polite"></p>
<div class="controls"><label>Show <select id="filter"><option value="all">All rows</option><option value="latest">Reviewed this round</option><option value="changed">Changed or held in either round</option><option value="held">Currently held</option></select></label><label>Jump to <select id="jump"><option value="">Choose a row</option></select></label></div>
<p id="count" class="count muted"></p>
<p class="legend"><span><i class="dot" style="background:#17cbe0"></i>Box</span><span><i class="dot" style="background:#1dde54"></i>Left corner</span><span><i class="dot" style="background:#087bff"></i>Right corner</span><span><i class="dot" style="background:#ed38ca"></i>Changed field</span></p>
<p class="muted legend">R42 highlights changes from the working baseline. R43 highlights only changes since R42. A box and its two physical corners are judged separately; unavailable corners have no dot.</p>
<details id="photoOrigin"><summary>Where did the frame_* photographs come from?</summary><p>The <code>frame_*</code> names are the original filenames of an additional 320-image upstream collection, under <code>grasp-vision / box_in_basket / white_box_pose</code>. This viewer contains 12 selected originals from that collection. The earlier frozen collection contains 157 <code>new_*</code> and <code>old_*</code> photographs. The prefixes identify different collections; matching numbers do not establish a correspondence between photos.</p></details>
</header>
<section class="card" aria-label="Whole original photograph"><h2>Whole original frame</h2><div class="controls"><label><input type="checkbox" id="showOverlay" checked> Show labels</label><label>Version <select id="overlayVersion"><option value="latest">Latest R43</option><option value="r42">Previous R42</option><option value="baseline">Working baseline</option><option value="source">Original upstream</option></select></label></div><canvas id="full" aria-label="Whole original photograph with optional label overlay"></canvas></section>
<h2 class="section-intro">Rows in this frame</h2><main id="rows" aria-label="Annotation rows"></main>
<section class="card" id="diagnostics" hidden><h2>Earlier R42 prediction comparisons</h2><p class="muted">These preserved figures use the R42 reference labels and saved predictions. They are historical comparisons, not updated R43 scores. The latest row cards above show any newer corner or boundary amendment.</p><div id="figureCases"></div></section>
<nav class="frame-navigation" aria-label="Next or previous photograph"><button id="previousFrameBottom">Previous frame</button><span id="positionBottom" class="frame-position"></span><button id="nextFrameBottom">Next frame</button></nav>
<footer class="muted"><p>These are assistant-reviewed proposals. A held whole image stays outside the complete-candidate export, even when some of its rows have useful corrections. Soft contours support nominal coordinates. No human sign-off, new training result, or independent test performance is claimed.</p></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
let frameIndex=0,frame=null,img=null,imageReady=false,generation=0,visibleRows=[];
const frameButtons=[['previousFrame',-1],['previousFrameBottom',-1],['nextFrame',1],['nextFrameBottom',1]];
function element(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function parse(line){if(!line)return null;const a=line.trim().split(/\s+/).map(Number);if(a.length!==11||!a.every(Number.isFinite))return null;return{a,b:[(a[1]-a[3]/2)*1280,(a[2]-a[4]/2)*720,(a[1]+a[3]/2)*1280,(a[2]+a[4]/2)*720]}}
function changed(before,after){if(before===after)return[];if(!before||!after)return Array.from({length:11},(_,i)=>i);const a=before.split(/\s+/),b=after.split(/\s+/);return b.map((v,i)=>v!==a[i]?i:-1).filter(i=>i>=0)}
function rowAnchor(row){return 'row-'+row.stable_id.replace(/[^a-zA-Z0-9_-]/g,'-')}
function rowName(row){return row.source_row===null?'Added front':`Source row ${row.source_row}`}
function draw(ctx,line,ox,oy,changes=[],label=''){
 const p=parse(line);if(!p)return;const {a,b}=p;ctx.lineWidth=.7;ctx.strokeStyle=changes.some(i=>i>=1&&i<=4)?'#ed38ca':'#17cbe0';ctx.strokeRect(b[0]-ox,b[1]-oy,b[2]-b[0],b[3]-b[1]);
 for(const k of[5,8])if(a[k+2]){ctx.beginPath();ctx.strokeStyle=changes.some(i=>i>=k&&i<=k+2)?'#ed38ca':k===5?'#1dde54':'#087bff';ctx.arc(a[k]*1280-ox,a[k+1]*720-oy,1.4,0,Math.PI*2);ctx.stroke()}
 if(label){ctx.font='10px sans-serif';ctx.lineWidth=2;ctx.strokeStyle='#172428';ctx.fillStyle='#fff';ctx.strokeText(label,b[0]-ox,b[1]-oy-2);ctx.fillText(label,b[0]-ox,b[1]-oy-2)}
}
function panelNote(line,version,row){const p=parse(line);if(p)return`Left ${p.a[7]?'visible':'unavailable'} · Right ${p.a[10]?'visible':'unavailable'}`;if(version==='raw')return'Unfiltered original pixels';if(version==='baseline')return'Not present in the working baseline';if(version==='r42')return row.before_line?'Deleted in R42':'Not present in R42';return row.r42_line?'Deleted in R43':row.before_line?'Deletion retained':'No current annotation'}
function cropFor(row){const boxes=[row.before_line,row.r42_line,row.after_line].map(parse).filter(Boolean).map(p=>p.b);if(!boxes.length)return[0,0,1280,720];return[Math.max(0,Math.floor(Math.min(...boxes.map(b=>b[0]))-16)),Math.max(0,Math.floor(Math.min(...boxes.map(b=>b[1]))-16)),Math.min(1280,Math.ceil(Math.max(...boxes.map(b=>b[2]))+16)),Math.min(720,Math.ceil(Math.max(...boxes.map(b=>b[3]))+16))]}
function renderPanel(row,title,version,line,crop,changes){const panel=element('div','panel'+(version==='latest'?' latest-panel':''));panel.appendChild(element('h3','',title));const canvas=element('canvas');canvas.setAttribute('aria-label',`${rowName(row)}: ${title}`);const[x,y,x2,y2]=crop;canvas.width=(x2-x)*4;canvas.height=(y2-y)*4;const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.scale(4,4);ctx.drawImage(img,x,y,x2-x,y2-y,0,0,x2-x,y2-y);draw(ctx,line,x,y,changes);panel.appendChild(canvas);panel.appendChild(element('p','panel-note',panelNote(line,version,row)));return panel}
function renderCard(row){
 const card=element('article','card row-card');card.id=rowAnchor(row);card.dataset.stableId=row.stable_id;
 const heading=element('div','row-heading');heading.appendChild(element('h2','',rowName(row)));
 const update=row.latest_update||null,delta=changed(row.r42_line,row.after_line);
 heading.appendChild(element('span','badge'+(delta.length?' amended':update?.status==='held'?' held':''),delta.length?'Amended in R43':update?.status==='held'?'R43 request held':update?'Reviewed in R43 · retained':'Unchanged since R42'));
 if(row.unresolved_fields.length)heading.appendChild(element('span','badge held','Held fields'));
 if(row.after_line===null)heading.appendChild(element('span','badge','Deleted identity retained in history'));
 card.appendChild(heading);card.appendChild(element('p','row-id muted',`${row.stable_id} · Current output row ${row.output_row===null?'none (deleted)':row.output_row}`));
 const grid=element('div','grid'),crop=cropFor(row);
 for(const [title,version,line,delta]of[['Original pixels','raw',null,[]],['Working baseline','baseline',row.before_line,[]],['Previous R42','r42',row.r42_line,changed(row.before_line,row.r42_line)],['Latest R43','latest',row.after_line,changed(row.r42_line,row.after_line)]])grid.appendChild(renderPanel(row,title,version,line,crop,delta));
 card.appendChild(grid);
 card.appendChild(element('p','latest-note',update?`R43 review: ${update.reason}`:'No new amendment to this row; its R42 proposal is preserved.'));
 if(row.unresolved_fields.length)card.appendChild(element('p','held held-note',`Current hold: ${row.unresolved_fields.join(', ')}\n${row.unresolved_reasons.join('\n')}`));
 const history=element('details','history');history.appendChild(element('summary','','Review history and exact row text'));
 history.appendChild(element('p','history-label','R42 decision'));history.appendChild(element('p','',row.r42_reason));
 if(row.r42_unresolved_fields.length)history.appendChild(element('p','muted',`R42 holds: ${row.r42_unresolved_fields.join(', ')}\n${row.r42_unresolved_reasons.join('\n')}`));
 if(update){history.appendChild(element('p','history-label','R43 request and decision'));history.appendChild(element('p','',`Requested fields: ${(update.requested_fields||[]).join(', ')||'See decision'}\nStatus: ${update.status||'reviewed'}\n${update.reason}`))}
 history.appendChild(element('pre','',JSON.stringify({stable_id:row.stable_id,source:row.source_line??null,working_baseline:row.before_line,previous_r42:row.r42_line,latest_r43:row.after_line,r42_changed_token_indices:changed(row.before_line,row.r42_line),r43_changed_token_indices:changed(row.r42_line,row.after_line),current_unresolved_fields:row.unresolved_fields},null,2)));
 card.appendChild(history);return card
}
function renderWhole(){const c=$('full');c.width=1280;c.height=720;const ctx=c.getContext('2d');if(!imageReady)return;ctx.drawImage(img,0,0);if(!$('showOverlay').checked)return;const version=$('overlayVersion').value;for(const row of frame.items){const line=version==='source'?row.source_line:version==='baseline'?row.before_line:version==='r42'?row.r42_line:row.after_line;const changes=version==='latest'?changed(row.r42_line,row.after_line):version==='r42'?changed(row.before_line,row.r42_line):[];draw(ctx,line,0,0,changes,row.source_row===null?'new':String(row.source_row))}}
function renderDiagnostics(){const panel=$('diagnostics'),container=$('figureCases');container.replaceChildren();const figures=data.diagnostic_figures.filter(f=>f.image_id===frame.image_id);panel.hidden=!figures.length;for(const f of figures){const detail=element('details'),summary=element('summary','',`R42 source row ${f.source_row} · saved prediction comparison`),picture=element('img','diagnostic-image');picture.src=f.image;picture.alt=`Historical R42 comparison for ${f.image_id}, source row ${f.source_row}`;picture.loading='lazy';detail.append(summary,picture);container.appendChild(detail)}}
function renderRows(){
 const mode=$('filter').value;visibleRows=frame.items.filter(row=>mode==='all'||(mode==='latest'?Boolean(row.latest_update)||row.r42_line!==row.after_line:mode==='held'?row.unresolved_fields.length>0:row.action!=='retain'||row.r42_action!=='retain'||row.r42_line!==row.after_line||row.unresolved_fields.length>0||row.r42_unresolved_fields.length>0));
 $('rows').replaceChildren();$('jump').replaceChildren();const start=element('option','','Choose a row');start.value='';$('jump').appendChild(start);
 const amended=frame.items.filter(row=>row.r42_line!==row.after_line).length;
 $('count').textContent=`Showing ${visibleRows.length} of ${frame.items.length} row identities · ${frame.candidate_rows} current fronts · ${amended} amended in R43. Deleted identities and additions remain separate.`;
 if(!visibleRows.length)$('rows').appendChild(element('p','empty muted','No rows match this filter. Choose “All rows” to view the entire frame.'));
 else for(const row of visibleRows){$('rows').appendChild(renderCard(row));const option=element('option','',rowName(row)+(row.unresolved_fields.length?' · held':''));option.value=rowAnchor(row);$('jump').appendChild(option)}
 $('jump').disabled=!visibleRows.length;
}
function navigation(){const position=`Frame ${frameIndex+1} of ${data.frames.length}`;$('positionTop').textContent=position;$('positionBottom').textContent=position;for(const[id,step]of frameButtons)$(id).disabled=step<0?frameIndex===0:frameIndex===data.frames.length-1}
function changeFrame(index,scroll=true){
 frameIndex=Math.min(data.frames.length-1,Math.max(0,index));frame=data.frames[frameIndex];$('frame').value=frame.image_id;navigation();imageReady=false;visibleRows=[];$('rows').replaceChildren(element('p','empty muted','Loading original pixels…'));$('jump').replaceChildren();$('jump').disabled=$('filter').disabled=true;$('count').textContent='';$('figureCases').replaceChildren();$('diagnostics').hidden=true;
 $('status').className='status '+(frame.annotation_complete?'complete':'held');$('status').textContent=`${frame.annotation_complete?'Complete assistant-reviewed candidate':'Whole image held'} · ${frame.image_id} · ${frame.candidate_rows} current fronts · ${frame.unresolved_stable_ids.length} held source rows`;
 renderWhole();const token=++generation;const nextImage=new Image();img=nextImage;
 nextImage.onload=()=>{if(token!==generation)return;imageReady=true;$('filter').disabled=false;renderWhole();renderRows();renderDiagnostics()};
 nextImage.onerror=()=>{if(token!==generation)return;$('rows').replaceChildren(element('p','empty held','The embedded original photograph could not be loaded.'));$('count').textContent='Image loading failed; no annotation panels were drawn.'};
 nextImage.src=frame.image;if(scroll)window.scrollTo({top:0,behavior:'auto'});
}
for(const frame of data.frames){const option=element('option','',`${frame.image_id} · ${frame.annotation_complete?'complete candidate':'held'}`);option.value=frame.image_id;$('frame').appendChild(option)}
$('filter').value='all';$('frame').onchange=()=>changeFrame(data.frames.findIndex(f=>f.image_id===$('frame').value));$('filter').onchange=()=>{if(imageReady)renderRows()};$('showOverlay').onchange=renderWhole;$('overlayVersion').onchange=renderWhole;
$('jump').onchange=()=>{const target=$($('jump').value);if(target)target.scrollIntoView({block:'start',behavior:'smooth'})};for(const[id,step]of frameButtons)$(id).onclick=()=>changeFrame(frameIndex+step);changeFrame(0,false);
</script></body></html>'''
