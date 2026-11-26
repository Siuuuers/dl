"""Render complete frame pages with R41, R42, R43 and latest R44 history."""
import base64
import json
import math
from pathlib import PurePosixPath
import re


def _frame_index(manifest, label):
    frames = manifest['frames']
    indexed = {frame['image_id']: frame for frame in frames}
    if len(indexed) != len(frames):
        raise ValueError(f'Repeated {label} frame identity')
    return indexed


def _row_index(frame, label):
    rows = frame['rows'] + frame.get('additions', [])
    indexed = {row['stable_id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{frame['image_id']}: repeated {label} stable row identity")
    return indexed


def render_html(manifest, previous_manifest, r42_manifest, payload, figures, outlines):
    """Return self-contained HTML bytes with original pixels and complete history.

    ``previous_manifest`` is R43. R42's before_line is the R41 working baseline;
    it is deliberately independent of the newest round's immediate predecessor.
    Outlines are supplied physical review aids, never inferred box polygons.
    """
    current = _frame_index(manifest, 'R44')
    previous = _frame_index(previous_manifest, 'R43')
    r42 = _frame_index(r42_manifest, 'R42')
    if set(current) != set(previous) or set(current) != set(r42):
        raise ValueError('R44, R43 and R42 frame collections differ')
    frames = []
    identities = {}
    for frame in manifest['frames']:
        fid = frame['image_id']
        current_rows = _row_index(frame, 'R44')
        old_rows = _row_index(previous[fid], 'R43')
        older_rows = _row_index(r42[fid], 'R42')
        if not set(old_rows).issubset(current_rows) or not set(older_rows).issubset(current_rows):
            raise ValueError(f'{fid}: missing historical stable row identity')
        size = frame['image_size']
        if (len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size)
                or previous[fid]['image_size'] != size or r42[fid]['image_size'] != size):
            raise ValueError(f'{fid}: inconsistent image dimensions')
        merged = []
        for sid, row in current_rows.items():
            if sid in identities:
                raise ValueError(f'Repeated global stable row identity: {sid}')
            identities[sid] = size
            old, older = old_rows.get(sid), older_rows.get(sid)
            baseline = older.get('before_line') if older else old.get('before_line') if old else None
            source = row.get('source_line', old.get('source_line') if old else None)
            item = {**row, 'source_line': source, 'baseline_line': baseline,
                    'r42_record': older, 'r43_record': old}
            for version, record in [('r42', older), ('r43', old)]:
                item.update({f'{version}_line': record['after_line'] if record else None,
                             f'{version}_reason': record.get('reason', '') if record else f'No entry in {version.upper()}.',
                             f'{version}_action': record.get('action') if record else None,
                             f'{version}_output_row': record.get('output_row') if record else None,
                             f'{version}_unresolved_fields': record.get('unresolved_fields', []) if record else [],
                             f'{version}_unresolved_reasons': record.get('unresolved_reasons', []) if record else [],
                             f'{version}_latest_update': record.get('latest_update') if record else None})
            item.setdefault('unresolved_fields', [])
            item.setdefault('unresolved_reasons', [])
            merged.append(item)
        jpg = payload['originals/' + fid + '.jpg']
        frames.append({key: value for key, value in frame.items() if key not in ('rows', 'additions')} |
                      {'items': merged,
                       'r42_annotation_complete': r42[fid]['annotation_complete'],
                       'r43_annotation_complete': previous[fid]['annotation_complete'],
                       'image': 'data:image/jpeg;base64,' + base64.b64encode(jpg).decode()})
    outline_data = outlines if outlines is not None else {'kind': 'physical-front-review-outline', 'items': []}
    if not isinstance(outline_data, dict) or not isinstance(outline_data.get('items'), list):
        raise ValueError('Outlines must be an object with an items list')
    outline_ids = set()
    for item in outline_data['items']:
        sid = item.get('stable_id')
        if sid not in identities or sid in outline_ids:
            raise ValueError(f'Unknown or repeated outline identity: {sid}')
        outline_ids.add(sid)
        vertices = item.get('vertices_px')
        if not isinstance(vertices, list) or len(vertices) < 3 or item.get('closed') is not True:
            raise ValueError(f'{sid}: review outline needs at least three vertices and closed=true')
        width, height = identities[sid]
        for point in vertices:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in point)
                    or not (0 <= point[0] <= width and 0 <= point[1] <= height)):
                raise ValueError(f'{sid}: invalid original-pixel outline vertex')
        if not isinstance(item.get('reason'), str):
            raise ValueError(f'{sid}: review outline reason is required')
    diagnostic_figures = []
    for name in sorted(figures):
        filename = PurePosixPath(name).name
        match = re.fullmatch(r'(frame_\d+)_row_(\d+)\.png', filename)
        if match is None or match.group(1) not in previous:
            raise ValueError(f'Unrecognized historical diagnostic figure: {name}')
        diagnostic_figures.append({'name': filename, 'image_id': match.group(1),
            'source_row': int(match.group(2)),
            'image': 'data:image/png;base64,' + base64.b64encode(payload[name]).decode()})
    data = {'frames': frames, 'diagnostic_figures': diagnostic_figures, 'outlines': outline_data}
    encoded = json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return HTML.replace('__DATA__', encoded).encode()


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R44 cassette-front review</title>
<style>
:root{font-family:system-ui,sans-serif;color:#18353b;background:#f2f6f6;color-scheme:light}
*{box-sizing:border-box}body{max-width:1520px;margin:auto;padding:20px}h1{font-size:1.7rem;margin:0 0 8px}h2{font-size:1.15rem;margin:0 0 10px}h3{font-size:.93rem;margin:0 0 8px}p{line-height:1.5;margin:10px 0}a{color:#076e78}
header,.card,.frame-navigation{background:white;border:1px solid #d8e5e5;border-radius:12px;padding:18px;margin-bottom:16px}.muted{color:#526c71}.held{color:#92520a}.complete{color:#076b53}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.controls label{display:flex;gap:6px;align-items:center;font-size:.92rem}
button,select{padding:10px;border:1px solid #89a5a7;border-radius:7px;background:white;color:#18353b;font:inherit;max-width:100%}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #00868e;outline-offset:2px}summary{cursor:pointer;line-height:1.5}details{margin-top:12px}details p{margin:8px 0}code,pre{font-family:ui-monospace,monospace}code{overflow-wrap:anywhere}pre{font-size:.78rem;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;background:#f4f7f7;padding:12px;border-radius:7px}
.legend{font-size:.9rem}.legend span{display:inline-block;margin-right:12px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.row-card{scroll-margin-top:16px}.row-heading{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}.row-heading h2{margin:0}.badge{font-size:.78rem;padding:4px 8px;border-radius:5px;background:#edf3f3;color:#405d64}.badge.amended{background:#fbe8f7;color:#8a216d}.badge.held{background:#fff0d9;color:#92520a}.row-id{font-size:.78rem;margin:7px 0 14px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{min-width:0}.panel canvas{width:100%;height:auto;display:block;image-rendering:pixelated;background:#172428}.panel-note{font-size:.77rem;min-height:1.3em;margin:6px 0;color:#526c71}.latest-panel h3{color:#8a216d}.latest-note,.held-note,.outline-note{white-space:pre-wrap}.held-note{border-left:3px solid #d59637;padding-left:10px}.outline-note{border-left:3px solid #b45d00;padding-left:10px;color:#79511e}.history p{white-space:pre-wrap}.frame-position{font-size:.9rem;font-weight:650}.frame-navigation{display:flex;gap:12px;align-items:center;justify-content:space-between}.status{font-weight:650}#full{display:block;width:100%;max-width:1280px;height:auto;margin:12px auto 0;image-rendering:pixelated;background:#172428}.empty{padding:22px;border:1px dashed #9bb2b4;border-radius:9px;margin-bottom:16px}.diagnostic-image{display:block;width:100%;max-width:960px;height:auto;margin:12px auto}.count{font-size:.9rem}.section-intro{margin:20px 0 12px}.history-label{font-weight:650}
@media(max-width:1050px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){body{padding:9px}header,.card,.frame-navigation{padding:12px;margin-bottom:12px;border-radius:9px}h1{font-size:1.4rem}.grid{gap:8px}.panel h3{font-size:.82rem}.controls{gap:8px}button,select{padding:9px;font-size:.89rem}.row-heading h2{font-size:1.02rem}.row-id{font-size:.72rem}.frame-navigation{gap:7px}.frame-position{font-size:.8rem}.panel-note{font-size:.71rem}.legend{font-size:.8rem}}
@media print{body{max-width:none;background:white;padding:0}.controls,.frame-navigation,#photoOrigin{display:none}.card{break-inside:avoid}.grid{grid-template-columns:repeat(4,minmax(0,1fr))}}
</style></head><body>
<header id="top"><h1>R44 · Cassette-front review</h1>
<p>One photograph per page, with every row together. Original pixels, the working labels, review history and the latest decision stay side by side.</p>
<div class="controls" aria-label="Frame navigation"><button id="previousFrame">Previous frame</button><label>Frame <select id="frame"></select></label><button id="nextFrame">Next frame</button><span id="positionTop" class="frame-position"></span></div>
<p id="status" class="status" aria-live="polite"></p>
<div class="controls"><label>Show <select id="filter"><option value="all">All rows</option><option value="latest">Reviewed in R44</option><option value="changed">Changed or held in any review</option><option value="held">Currently held</option></select></label><label>History panel <select id="historyVersion"><option value="r43">Previous R43</option><option value="r42">Earlier R42</option></select></label><label>Jump to <select id="jump"><option value="">Choose a row</option></select></label></div>
<p id="count" class="count muted"></p>
<p class="legend"><span><i class="dot" style="background:#17cbe0"></i>Box</span><span><i class="dot" style="background:#1dde54"></i>Left corner</span><span><i class="dot" style="background:#087bff"></i>Right corner</span><span><i class="dot" style="background:#ed38ca"></i>Changed field</span><span><i class="dot" style="background:#b45d00"></i>Review outline</span></p>
<p id="changeLegend" class="muted legend"></p>
<details id="photoOrigin"><summary>Where did the frame_* photographs come from?</summary><p>The <code>frame_*</code> names are the original filenames of an additional 320-image upstream collection, under <code>grasp-vision / box_in_basket / white_box_pose</code>. This viewer contains selected originals from that collection. The earlier frozen collection contains 157 <code>new_*</code> and <code>old_*</code> photographs. The prefixes identify different collections; matching numbers do not establish a correspondence between photos.</p></details>
</header>
<section class="card" aria-label="Whole original photograph"><h2>Whole original frame</h2><div class="controls"><label><input type="checkbox" id="showOverlay" checked> Show labels</label><label>Version <select id="overlayVersion"><option value="latest">Latest R44</option><option value="r43">Previous R43</option><option value="r42">Earlier R42</option><option value="baseline">R41 working baseline</option><option value="source">Original upstream</option></select></label><label><input type="checkbox" id="showOutlines"> Show review outline, not training mask</label></div><p id="outlineHelp" class="muted legend"></p><canvas id="full" aria-label="Whole original photograph with optional label overlay"></canvas></section>
<h2 class="section-intro">Rows in this frame</h2><main id="rows" aria-label="Annotation rows"></main>
<section class="card" id="diagnostics" hidden><h2>Earlier R42 prediction comparisons</h2><p class="muted">These preserved figures use the R42 reference labels and saved predictions. They are historical comparisons, not updated R43 or R44 scores. The latest row cards above show newer corner, boundary and hold decisions.</p><div id="figureCases"></div></section>
<nav class="frame-navigation" aria-label="Next or previous photograph"><button id="previousFrameBottom">Previous frame</button><span id="positionBottom" class="frame-position"></span><button id="nextFrameBottom">Next frame</button></nav>
<footer class="muted"><p>These are assistant-reviewed proposals. A held whole image stays outside the complete-candidate export, even when some of its rows have useful corrections. A traced outline is a nominal review aid, not a segmentation label or training mask. No human sign-off, new training result, or independent test performance is claimed.</p></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const outlineById=new Map(data.outlines.items.map(item=>[item.stable_id,item]));
let frameIndex=0,frame=null,img=null,imageReady=false,generation=0,visibleRows=[];
const frameButtons=[['previousFrame',-1],['previousFrameBottom',-1],['nextFrame',1],['nextFrameBottom',1]];
function element(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function parse(line){if(!line)return null;const a=line.trim().split(/\s+/).map(Number);if(a.length!==11||!a.every(Number.isFinite))return null;const[w,h]=frame.image_size;return{a,b:[(a[1]-a[3]/2)*w,(a[2]-a[4]/2)*h,(a[1]+a[3]/2)*w,(a[2]+a[4]/2)*h]}}
function changed(before,after){if(before===after)return[];if(!before||!after)return Array.from({length:11},(_,i)=>i);const a=before.trim().split(/\s+/),b=after.trim().split(/\s+/);return b.map((v,i)=>v!==a[i]?i:-1).filter(i=>i>=0)}
function rowAnchor(row){return 'row-'+row.stable_id.replace(/[^a-zA-Z0-9_-]/g,'-')}
function rowName(row){return row.source_row===null?'Added front':`Source row ${row.source_row}`}
function versionLine(row,version){return version==='source'?row.source_line:version==='baseline'?row.baseline_line:version==='r42'?row.r42_line:version==='r43'?row.r43_line:row.after_line}
function versionChanges(row,version){return version==='latest'?changed(row.r43_line,row.after_line):version==='r43'?changed(row.r42_line,row.r43_line):version==='r42'?changed(row.baseline_line,row.r42_line):[]}
function draw(ctx,line,ox,oy,changes=[],label=''){
 const p=parse(line);if(!p)return;const{a,b}=p,[w,h]=frame.image_size;ctx.lineWidth=.7;ctx.strokeStyle=changes.some(i=>i>=1&&i<=4)?'#ed38ca':'#17cbe0';ctx.strokeRect(b[0]-ox,b[1]-oy,b[2]-b[0],b[3]-b[1]);
 for(const k of[5,8])if(a[k+2]){ctx.beginPath();ctx.strokeStyle=changes.some(i=>i>=k&&i<=k+2)?'#ed38ca':k===5?'#1dde54':'#087bff';ctx.arc(a[k]*w-ox,a[k+1]*h-oy,1.4,0,Math.PI*2);ctx.stroke()}
 if(label){ctx.font='10px sans-serif';ctx.lineWidth=2;ctx.strokeStyle='#172428';ctx.fillStyle='#fff';ctx.strokeText(label,b[0]-ox,b[1]-oy-2);ctx.fillText(label,b[0]-ox,b[1]-oy-2)}
}
function drawOutline(ctx,row,ox,oy){
 const outline=outlineById.get(row.stable_id);if(!outline||!$('showOutlines').checked)return;
 ctx.save();ctx.beginPath();outline.vertices_px.forEach(([x,y],i)=>i?ctx.lineTo(x-ox,y-oy):ctx.moveTo(x-ox,y-oy));if(outline.closed)ctx.closePath();ctx.lineWidth=1.15;ctx.strokeStyle='#fff';ctx.stroke();ctx.lineWidth=.65;ctx.strokeStyle='#b45d00';ctx.stroke();ctx.restore();
}
function panelNote(line,version,row){
 const p=parse(line);if(p)return`Left ${p.a[7]?'visible':'unavailable'} · Right ${p.a[10]?'visible':'unavailable'}`;
 if(version==='raw')return'Unfiltered original pixels';if(version==='baseline')return'Not present in the R41 working baseline';
 if(version==='r42'||version==='r43')return row[version+'_record']?`Deleted in ${version.toUpperCase()}; history retained`:`No entry in ${version.toUpperCase()}`;
 return row.after_line===null?'Deleted identity; history retained':'No current annotation';
}
function cropFor(row){
 const[w,h]=frame.image_size,boxes=[row.baseline_line,row.r42_line,row.r43_line,row.after_line].map(parse).filter(Boolean).map(p=>p.b),outline=outlineById.get(row.stable_id);
 if(outline){const xs=outline.vertices_px.map(p=>p[0]),ys=outline.vertices_px.map(p=>p[1]);boxes.push([Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)])}
 if(!boxes.length)return[0,0,w,h];return[Math.max(0,Math.floor(Math.min(...boxes.map(b=>b[0]))-16)),Math.max(0,Math.floor(Math.min(...boxes.map(b=>b[1]))-16)),Math.min(w,Math.ceil(Math.max(...boxes.map(b=>b[2]))+16)),Math.min(h,Math.ceil(Math.max(...boxes.map(b=>b[3]))+16))];
}
function renderPanel(row,title,version,line,crop,changes){
 const panel=element('div','panel'+(version==='latest'?' latest-panel':''));panel.dataset.version=version;panel.appendChild(element('h3','',title));const canvas=element('canvas');canvas.setAttribute('aria-label',`${rowName(row)}: ${title}`);const[x,y,x2,y2]=crop;canvas.width=(x2-x)*4;canvas.height=(y2-y)*4;const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.scale(4,4);ctx.drawImage(img,x,y,x2-x,y2-y,0,0,x2-x,y2-y);if(version==='latest')drawOutline(ctx,row,x,y);draw(ctx,line,x,y,changes);panel.appendChild(canvas);panel.appendChild(element('p','panel-note',panelNote(line,version,row)));return panel;
}
function appendHistory(history,round,record){
 history.appendChild(element('p','history-label',`${round} decision`));if(!record){history.appendChild(element('p','',`No entry in ${round}.`));return}
 history.appendChild(element('p','',record.reason||'No separate decision text.'));
 if(record.unresolved_fields?.length)history.appendChild(element('p','muted',`${round} holds: ${record.unresolved_fields.join(', ')}\n${(record.unresolved_reasons||[]).join('\n')}`));
 if(record.latest_update){const update=record.latest_update;history.appendChild(element('p','',`${round} latest update\nRequested fields: ${(update.requested_fields||[]).join(', ')||'See decision'}\nStatus: ${update.status||'reviewed'}\n${update.reason||''}`))}
}
function renderCard(row){
 const card=element('article','card row-card');card.id=rowAnchor(row);card.dataset.stableId=row.stable_id;const heading=element('div','row-heading');heading.appendChild(element('h2','',rowName(row)));
 const update=row.latest_update||null,delta=changed(row.r43_line,row.after_line),held=row.unresolved_fields.length>0,historicalDeletion=row.after_line===null&&!update;
 const badge=historicalDeletion?'Historical deletion · retained':delta.length?'Amended in R44':update?(held?'Reviewed in R44 · held':'Reviewed in R44 · retained'):'No R44 review record';
 heading.appendChild(element('span','badge'+(delta.length?' amended':held?' held':''),badge));if(held)heading.appendChild(element('span','badge held','Held fields'));if(row.after_line===null)heading.appendChild(element('span','badge','Deleted identity retained in history'));
 card.appendChild(heading);card.appendChild(element('p','row-id muted',`${row.stable_id} · Current output row ${row.output_row===null?'none (deleted)':row.output_row}`));
 const grid=element('div','grid'),crop=cropFor(row),historyVersion=$('historyVersion').value;
 const panels=[['Original pixels','raw',null,[]],['R41 working baseline','baseline',row.baseline_line,[]],[historyVersion==='r43'?'Previous R43':'Earlier R42',historyVersion,versionLine(row,historyVersion),versionChanges(row,historyVersion)],['Latest R44','latest',row.after_line,versionChanges(row,'latest')]];
 for(const[title,version,line,changes]of panels)grid.appendChild(renderPanel(row,title,version,line,crop,changes));card.appendChild(grid);
 card.appendChild(element('p','latest-note',historicalDeletion?`Earlier decision retained in history: ${row.reason||'No separate earlier decision text supplied.'}`:update?`R44 decision: ${update.reason||row.reason||'No separate decision text supplied.'}`:'No R44 review record supplied.'));
 if(update)card.appendChild(element('p','muted legend',`Requested fields: ${(update.requested_fields||[]).join(', ')||'See decision'} · Status: ${update.status||'reviewed'}`));
 if(held)card.appendChild(element('p','held held-note',`Current hold: ${row.unresolved_fields.join(', ')}\n${row.unresolved_reasons.join('\n')}`));
 const outline=outlineById.get(row.stable_id);if(outline){const uncertainty=Array.isArray(outline.nominal_uncertainty_px)?outline.nominal_uncertainty_px.join('–'):outline.nominal_uncertainty_px;card.appendChild(element('p','outline-note',`Review outline, not training mask. ${outline.reason}${uncertainty===undefined?'':` Nominal uncertainty: ${uncertainty} original pixels.`} Use the outline toggle above to show the trace on the latest panel and latest whole-frame overlay.`))}
 const history=element('details','history');history.appendChild(element('summary','','Review history and exact row text'));appendHistory(history,'R42',row.r42_record);appendHistory(history,'R43',row.r43_record);
 history.appendChild(element('p','history-label',historicalDeletion?'R44 history retention':'R44 request and decision'));history.appendChild(element('p','',historicalDeletion?'This identity was deleted in an earlier round. Its recorded deletion and exact row history are preserved; it is not a current front.':update?`Requested fields: ${(update.requested_fields||[]).join(', ')||'See decision'}\nStatus: ${update.status||'reviewed'}\n${update.reason||row.reason||''}`:'No R44 latest_update record supplied.'));
 history.appendChild(element('pre','',JSON.stringify({stable_id:row.stable_id,source:row.source_line??null,working_baseline_r41:row.baseline_line,earlier_r42:row.r42_line,previous_r43:row.r43_line,latest_r44:row.after_line,r42_changed_token_indices:versionChanges(row,'r42'),r43_changed_token_indices:versionChanges(row,'r43'),r44_changed_token_indices:versionChanges(row,'latest'),r42_review_record:row.r42_record,r43_review_record:row.r43_record,r44_latest_update:update,current_unresolved_fields:row.unresolved_fields,current_unresolved_reasons:row.unresolved_reasons,review_outline:outline||null},null,2)));card.appendChild(history);return card;
}
function renderWhole(){
 const c=$('full');if(!frame)return;[c.width,c.height]=frame.image_size;const ctx=c.getContext('2d');if(!imageReady)return;ctx.drawImage(img,0,0);const version=$('overlayVersion').value;
 for(const row of frame.items){if(version==='latest')drawOutline(ctx,row,0,0);if($('showOverlay').checked)draw(ctx,versionLine(row,version),0,0,versionChanges(row,version),row.source_row===null?'new':String(row.source_row))}
}
function renderDiagnostics(){const panel=$('diagnostics'),container=$('figureCases');container.replaceChildren();const figures=data.diagnostic_figures.filter(f=>f.image_id===frame.image_id);panel.hidden=!figures.length;for(const f of figures){const detail=element('details'),summary=element('summary','',`R42 source row ${f.source_row} · saved prediction comparison`),picture=element('img','diagnostic-image');picture.src=f.image;picture.alt=`Historical R42 comparison for ${f.image_id}, source row ${f.source_row}`;picture.loading='lazy';detail.append(summary,picture);container.appendChild(detail)}}
function changedOrHeld(row){return row.action!=='retain'||row.r42_action==='replace'||row.r42_action==='delete'||row.r43_action==='replace'||row.r43_action==='delete'||row.baseline_line!==row.r42_line||row.r42_line!==row.r43_line||row.r43_line!==row.after_line||row.unresolved_fields.length>0||row.r42_unresolved_fields.length>0||row.r43_unresolved_fields.length>0}
function renderRows(){
 const mode=$('filter').value;visibleRows=frame.items.filter(row=>mode==='all'||(mode==='latest'?Boolean(row.latest_update):mode==='held'?row.unresolved_fields.length>0:changedOrHeld(row)));
 $('rows').replaceChildren();$('jump').replaceChildren();const start=element('option','','Choose a row');start.value='';$('jump').appendChild(start);
 const amended=frame.items.filter(row=>row.r43_line!==row.after_line).length,reviewed=frame.items.filter(row=>row.latest_update).length;
 $('count').textContent=`Showing ${visibleRows.length} of ${frame.items.length} row identities · ${frame.candidate_rows} current fronts · ${reviewed} reviewed in R44 · ${amended} amended in R44. Unchanged, deleted and added identities remain in the review.`;
 if(!visibleRows.length)$('rows').appendChild(element('p','empty muted','No rows match this filter. Choose “All rows” to view the entire frame.'));
 else for(const row of visibleRows){$('rows').appendChild(renderCard(row));const option=element('option','',rowName(row)+(row.unresolved_fields.length?' · held':''));option.value=rowAnchor(row);$('jump').appendChild(option)}$('jump').disabled=!visibleRows.length;
}
function navigation(){const position=`Frame ${frameIndex+1} of ${data.frames.length}`;$('positionTop').textContent=position;$('positionBottom').textContent=position;for(const[id,step]of frameButtons)$(id).disabled=step<0?frameIndex===0:frameIndex===data.frames.length-1}
function updateLegend(){const previous=$('historyVersion').value==='r43'?'R43 highlights changes since R42.':'R42 highlights changes from the R41 working baseline.';$('changeLegend').textContent=`${previous} Latest R44 always highlights changes since R43. Boxes and physical corners are judged separately; unavailable corners have no dot.`}
function changeFrame(index,scroll=true){
 frameIndex=Math.min(data.frames.length-1,Math.max(0,index));frame=data.frames[frameIndex];$('frame').value=frame.image_id;navigation();imageReady=false;visibleRows=[];$('rows').replaceChildren(element('p','empty muted','Loading original pixels…'));$('jump').replaceChildren();$('jump').disabled=$('filter').disabled=true;$('count').textContent='';$('figureCases').replaceChildren();$('diagnostics').hidden=true;
 const heldCount=frame.items.filter(row=>row.unresolved_fields.length).length;$('status').className='status '+(frame.annotation_complete?'complete':'held');$('status').textContent=`${frame.annotation_complete?'Complete assistant-reviewed candidate':'Whole image held'} · ${frame.image_id} · ${frame.candidate_rows} current fronts · ${heldCount} held row identities`;
 const outlineCount=frame.items.filter(row=>outlineById.has(row.stable_id)).length;$('showOutlines').disabled=!outlineCount;$('outlineHelp').textContent=outlineCount?`${outlineCount} physical front outline${outlineCount===1?'':'s'} available in this frame. The toggle draws only the supplied trace on Latest R44; earlier labels and original pixels remain untouched.`:'No traced front outline is supplied for this frame. No polygon is inferred from a box.';
 renderWhole();const token=++generation,nextImage=new Image();img=nextImage;
 nextImage.onload=()=>{if(token!==generation)return;if(nextImage.naturalWidth!==frame.image_size[0]||nextImage.naturalHeight!==frame.image_size[1]){showImageError('Embedded photograph dimensions differ from the annotation frame.');return}imageReady=true;$('filter').disabled=false;renderWhole();renderRows();renderDiagnostics()};
 nextImage.onerror=()=>{if(token===generation)showImageError('The embedded original photograph could not be loaded.')};nextImage.src=frame.image;if(scroll)window.scrollTo({top:0,behavior:'auto'});
}
function showImageError(message){imageReady=false;$('rows').replaceChildren(element('p','empty held',message));$('count').textContent='Image loading failed; no annotation panels were drawn.'}
for(const f of data.frames){const option=element('option','',`${f.image_id} · ${f.annotation_complete?'complete candidate':'held'}`);option.value=f.image_id;$('frame').appendChild(option)}
$('filter').value='all';$('historyVersion').value='r43';$('overlayVersion').value='latest';$('showOutlines').checked=false;
$('frame').onchange=()=>changeFrame(data.frames.findIndex(f=>f.image_id===$('frame').value));$('filter').onchange=()=>{if(imageReady)renderRows()};$('historyVersion').onchange=()=>{updateLegend();if(imageReady)renderRows()};$('showOverlay').onchange=renderWhole;$('overlayVersion').onchange=renderWhole;$('showOutlines').onchange=()=>{if(imageReady){renderWhole();renderRows()}};
$('jump').onchange=()=>{const target=$($('jump').value);if(target)target.scrollIntoView({block:'start',behavior:'smooth'})};for(const[id,step]of frameButtons)$(id).onclick=()=>changeFrame(frameIndex+step);updateLegend();if(data.frames.length)changeFrame(0,false);else $('rows').appendChild(element('p','empty muted','No frames supplied.'));
</script></body></html>'''
