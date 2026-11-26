#!/usr/bin/env python3
"""Replay the targeted R43 amendments and package a frame-at-a-time review.

Original pixels and the complete R42 history are retained. This CPU-only tool
checks recorded decisions; it does not infer physical truth, train, or approve
annotations. The external archive hash is the publication's trust anchor.
"""
import argparse
from copy import deepcopy
import io
import json
from pathlib import Path
import zipfile

import review_front_mining_r42 as previous
from render_front_review_r43 import render_html


RETURN_SHA256 = '716f81f9b72fffccfb52120d90b30a50b3d605ebd649736ed1f5ad189b4cb2ef'
RETURN_BYTES = 4181563
PREVIOUS_MANIFEST_SHA256 = '5fad09d5e1922a4b176200459bcd53a698577472feb05a03d1be329aa45896b7'
PREVIOUS_INVENTORY_SHA256 = '8eb1e7e1ae4687e8069730321cd916650141ebd1deb0ce176a6804e279345c26'
PREVIOUS_SCRIPT_SHA256 = 'a81fbe0059de9e9e11df69335c31a0ad3b0a2deee932a0547d14d186a5ab36a8'
RETURN_COMMIT = '3644cd90434d755da9ace3c77cbf193bf0e3ba87'
RETURN_PR = 'https://github.com/Siuuuers/dl/pull/129'
HISTORY = 'history/r42/'
REQUESTS = {
    'frame_0001:source:001': ['left_point'],
    'frame_0001:source:036': ['left_point'],
    'frame_0001:source:043': ['right_point', 'xmax'],
    'frame_0146:source:001': ['left_point'],
    'frame_0160:source:001': ['right_point'],
    'frame_0160:source:024': ['left_point'],
    'frame_0160:source:026': ['right_point'],
    'frame_0160:source:031': ['left_point'],
    'frame_0160:source:037': ['right_point'],
    'frame_0383:source:010': ['right_point'],
}
require, digest, load_json = previous.require, previous.digest, previous.load_json


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True) + '\n').encode()


def changed(before, after):
    if before == after:
        return []
    if before is None or after is None:
        return list(range(11))
    return [i for i, (a, b) in enumerate(zip(before.split(), after.split())) if a != b]


def all_rows(manifest):
    return {r['stable_id']: r for f in manifest['frames'] for r in f['rows'] + f['additions']}


def recount(manifest):
    counts = dict.fromkeys(previous.COUNT_FIELDS, 0)
    for frame in manifest['frames']:
        counts['baseline_rows'] += len(frame['rows'])
        frame['candidate_rows'] = sum(r['after_line'] is not None for r in frame['rows'] + frame['additions'])
        counts['candidate_rows'] += frame['candidate_rows']
        for row in frame['rows'] + frame['additions']:
            counts[{'retain': 'retained_source_rows', 'replace': 'replaced_source_rows',
                    'delete': 'deleted_source_rows', 'add': 'added_rows'}[row['action']]] += 1
            if row['source_row'] is not None:
                counts['held_source_rows'] += bool(row['unresolved_fields'])
            if row['before_line'] is not None and row['after_line'] is not None:
                a, b = (list(map(float, row[key].split())) for key in ('before_line', 'after_line'))
                for k in (5, 8):
                    counts['corner_coordinate_slots_changed'] += a[k+2] == b[k+2] == 2 and a[k:k+2] != b[k:k+2]
                    counts['corner_availability_slots_changed'] += a[k+2] != b[k+2]
        frame['candidate_label_sha256'] = digest(previous.candidate_bytes(frame))
    # A stable order keeps the generated JSON reproducible across Python processes.
    manifest['counts'] = {key: counts[key] for key in sorted(counts)}


def apply_amendments(old, amendments):
    """Preserve every unrequested token, identity, hold and whole-image gate."""
    require(amendments['schema_version'] == 1, 'amendment schema')
    require(amendments['previous_manifest_sha256'] == PREVIOUS_MANIFEST_SHA256,
            'amendment previous manifest identity')
    require(amendments['previous_return'] == dict(pr=RETURN_PR, commit=RETURN_COMMIT,
                sha256=RETURN_SHA256, bytes=RETURN_BYTES), 'amendment return provenance')
    requests = amendments['requests']
    ids = [r['stable_id'] for r in requests]
    require(len(ids) == len(set(ids)) and set(ids) == set(REQUESTS), 'requested row coverage')
    result = deepcopy(old)
    before_rows, after_rows = all_rows(old), all_rows(result)
    frame_map = {f['image_id']: f for f in result['frames']}
    latest = dict(reviewed_rows=len(requests), amended_rows=0, retained_rows=0,
                  changed_tokens=0, corner_coordinate_slots_changed=0,
                  box_boundaries_changed=0, added_rows=0, deleted_rows=0,
                  hold_changes=0, visibility_changes=0)
    for request in requests:
        sid = request['stable_id']
        before, row = before_rows[sid], after_rows[sid]
        fid = sid.split(':')[0]
        require(request['image_id'] == fid and request['source_row'] == before['source_row'], sid+': request identity')
        require(request['source_image_sha256'] == frame_map[fid]['source_image_sha256'], sid+': original image identity')
        require(request['requested_fields'] == REQUESTS[sid], sid+': request fields')
        require(request['before_line'] == before['after_line'], sid+': R42 baseline mismatch')
        for key in ('unresolved_fields', 'unresolved_reasons'):
            require(request[key] == before[key], sid+': an existing hold changed')
        after = request['after_line']
        require(isinstance(after, str), sid+': R43 cannot add or delete a front')
        a, b = previous.pose(before['after_line'], sid), previous.pose(after, sid)
        delta = changed(before['after_line'], after)
        require(delta == request['changed_token_indices'], sid+': latest changed tokens')
        require(request['status'] == ('amended' if delta else 'retained'), sid+': request status')
        require(isinstance(request['reason'], str) and request['reason'].strip(), sid+': request reason')
        allowed = set()
        for field in request['requested_fields']:
            allowed.update({'left_point': [5, 6], 'right_point': [8, 9], 'xmax': [1, 3]}[field])
        require(set(delta) <= allowed, sid+': unrelated token changed')
        if 'xmax' in request['requested_fields']:
            require(abs((a[1]-a[3]/2-b[1]+b[3]/2)*1280) <= .0001, sid+': xmin was not preserved')
            require((b[1]+b[3]/2) < (a[1]+a[3]/2), sid+': requested right bound must contract')
            latest['box_boundaries_changed'] += 1
        latest['amended_rows'] += bool(delta)
        latest['retained_rows'] += not delta
        latest['changed_tokens'] += len(delta)
        latest['corner_coordinate_slots_changed'] += sum(a[k:k+2] != b[k:k+2] for k in (5, 8))
        row['after_line'] = after
        row['action'] = 'retain' if after == row['before_line'] else 'replace'
        row['changed_token_indices'] = changed(row['before_line'], after)
        row['reason'] = before['reason'] + '\nR43: ' + request['reason']
        row['latest_update'] = {key: deepcopy(request[key]) for key in
                                ('status', 'reason', 'requested_fields', 'changed_token_indices')}
    result['kind'] = 'r43-targeted-corner-review-with-complete-frame-history'
    result['round'] = 43
    result['baseline_return'] = {key: old[key] for key in
                                ('return_pr', 'return_commit', 'return_archive_sha256', 'return_archive_bytes')}
    result.update(return_pr=RETURN_PR, return_commit=RETURN_COMMIT,
                  return_archive_sha256=RETURN_SHA256, return_archive_bytes=RETURN_BYTES)
    result['previous_manifest_sha256'] = PREVIOUS_MANIFEST_SHA256
    result['amendments_sha256'] = digest(json_bytes(amendments))
    result['latest_counts'] = latest
    result['latest_review_scope'] = (
        'Ten requested source rows were checked against the original photos and independently cross-reviewed. '
        'Other rows and whole-frame completeness are inherited unchanged from R42. '
        'Full-image inspection flags record that earlier review; R43 is a targeted amendment, not a fresh review of all 507 fronts.')
    result['policy_notes'].append('R43 preserves every R42 identity, hold and whole-image gate. Historical R42 prediction figures retain their original reference labels; they are not R43 scores.')
    for frame in result['frames']:
        frame['latest_reviewed_stable_ids'] = [r['stable_id'] for r in frame['rows'] if 'latest_update' in r]
    recount(result)
    return result


def original_source(payload, manifest):
    source = {}
    for frame in manifest['frames']:
        fid = frame['image_id']; prefix = 'review/' + fid + '/'
        for member, suffix in [('originals/'+fid+'.jpg', 'original.jpg'),
                               ('source-labels/'+fid+'.txt', 'source.txt'),
                               ('baseline-labels/'+fid+'.txt', fid+'.txt')]:
            source[prefix+suffix] = payload[member]
    return source


def reconstruct_previous(payload):
    """Reconstruct all 65 R42 members without duplicating images or old HTML."""
    old_manifest_bytes = payload[HISTORY+'manifest.json']
    require(digest(old_manifest_bytes) == PREVIOUS_MANIFEST_SHA256, 'R42 manifest hash')
    old = load_json(old_manifest_bytes)
    inventory_bytes = payload[HISTORY+'inventory.json']
    require(digest(inventory_bytes) == PREVIOUS_INVENTORY_SHA256, 'R42 inventory hash')
    inventory = load_json(inventory_bytes)
    prior = {'inventory.json': inventory_bytes}
    proposals = {f['image_id']: previous.candidate_bytes(f) for f in old['frames']}
    for name in inventory:
        if name == 'review.html':
            continue
        if name == 'review.py':
            prior[name] = payload['review_front_mining_r42.py']
        elif name.startswith(('proposals/', 'complete-candidates/labels/')):
            prior[name] = proposals[Path(name).stem]
        elif name.startswith(('originals/', 'source-labels/', 'baseline-labels/', 'figures/')):
            prior[name] = payload[name]
        else:
            prior[name] = payload[HISTORY+name]
    figures = previous.figure_contract(prior, old)
    prior['review.html'] = previous.render_html(old, prior, figures)
    for name, entry in inventory.items():
        require(entry == dict(sha256=digest(prior[name]), bytes=len(prior[name])), 'R42 history differs: '+name)
    previous.verify_capsule(prior)
    return old, prior, figures


def verification_for(manifest, payload):
    result = previous.verify(manifest, original_source(payload, manifest))
    result.update(round=43, latest_counts=manifest['latest_counts'],
                  previous_return_pr=RETURN_PR, previous_return_sha256=RETURN_SHA256,
                  r42_history_members_verified=65,
                  all_prior_holds_and_frame_gates_preserved=True,
                  unrequested_row_tokens_preserved=True,
                  historical_prediction_figures=5,
                  inference_performed=False)
    result['scope'] = 'Pinned R42 history, exact targeted R43 replay, unchanged identities/tokens/holds, original images, strict geometry, export gates and canonical frame viewer. Physical decisions remain assistant visual review.'
    return result


def verify_capsule(payload):
    require(digest(payload['review_front_mining_r42.py']) == PREVIOUS_SCRIPT_SHA256,
            'R42 verifier identity')
    for member, path in [('review.py', Path(__file__)),
                         ('render_front_review_r43.py', Path(__file__).with_name('render_front_review_r43.py'))]:
        require(payload[member] == path.read_bytes(), member+': verifier/renderer differs from executing code')
    old, prior, figures = reconstruct_previous(payload)
    amendments = load_json(payload['amendments.json'])
    require(payload['amendments.json'] == json_bytes(amendments), 'amendment serialization')
    expected = apply_amendments(old, amendments)
    manifest = load_json(payload['manifest.json'])
    require(payload['manifest.json'] == json_bytes(expected), 'manifest differs from R42 plus exact amendment replay')
    result = verification_for(manifest, payload)
    proposals = {'proposals/'+f['image_id']+'.txt': previous.candidate_bytes(f) for f in manifest['frames']}
    candidates = {'complete-candidates/labels/'+f['image_id']+'.txt': previous.candidate_bytes(f)
                  for f in manifest['frames'] if f['eligible_for_candidate_export']}
    for prefix, expected_members in [('proposals/', proposals), ('complete-candidates/labels/', candidates)]:
        require({name for name in payload if name.startswith(prefix)} == set(expected_members), prefix+': member set')
        for name, data in expected_members.items():
            require(payload[name] == data, name+': label artifact differs from replay')
    require({name for name in payload if name.startswith('figures/')} == set(figures), 'historical figure member set')
    require(payload['review.html'] == render_html(manifest, old, payload, figures), 'HTML differs from canonical frame viewer')
    require(load_json(payload['verification.json']) == result, 'stored verification differs from replay')
    return result


def build(return_path, amendments_path, output, manifest_output=None, html_output=None):
    raw = return_path.read_bytes()
    require(len(raw) == RETURN_BYTES and digest(raw) == RETURN_SHA256, 'PR129 return archive identity')
    prior = previous.archive_payload(io.BytesIO(raw))
    previous.verify_capsule(prior)
    require(digest(prior['manifest.json']) == PREVIOUS_MANIFEST_SHA256 and
            digest(prior['inventory.json']) == PREVIOUS_INVENTORY_SHA256, 'R42 baseline identity')
    old = load_json(prior['manifest.json'])
    amendments = load_json(amendments_path.read_bytes())
    manifest = apply_amendments(old, amendments)
    payload = {'manifest.json': json_bytes(manifest), 'amendments.json': json_bytes(amendments),
               'review.py': Path(__file__).read_bytes(),
               'review_front_mining_r42.py': Path(previous.__file__).read_bytes(),
               'render_front_review_r43.py': Path(__file__).with_name('render_front_review_r43.py').read_bytes()}
    for name, data in prior.items():
        if name.startswith(('originals/', 'source-labels/', 'baseline-labels/', 'figures/')):
            payload[name] = data
        elif name not in ('review.py', 'review.html') and not name.startswith(('proposals/', 'complete-candidates/labels/')):
            payload[HISTORY+name] = data
    for frame in manifest['frames']:
        fid = frame['image_id']; data = previous.candidate_bytes(frame)
        payload['proposals/'+fid+'.txt'] = data
        if frame['eligible_for_candidate_export']:
            payload['complete-candidates/labels/'+fid+'.txt'] = data
    payload['complete-candidates/README.txt'] = prior['complete-candidates/README.txt']
    payload['README.txt'] = (
        'Open review.html: all rows of one frame appear together; Previous/Next navigates frames.\n'
        'Original pixels, working baseline, previous R42 and latest R43 labels are shown together.\n'
        'All 508 identities remain: 507 current fronts plus the earlier deleted identity.\n'
        'Ten requested rows were reviewed this round. See amendments.json for exact decisions.\n'
        'The history/r42 directory preserves earlier evidence; original images and figures are shared without duplication.\n'
        'The five prediction figures are historical R42 diagnostics, not updated R43 scores.\n'
        'Complete candidates remain frame_0146, frame_0371 and frame_0383; nine whole frames remain held.\n'
        'No training split, human approval, inference or training is created by this package.\n'
        'Offline check: python review.py --check-capsule PATH_TO_THIS_ZIP\n').encode()
    figures = previous.figure_contract(prior, old)
    payload['review.html'] = render_html(manifest, old, payload, figures)
    result = verification_for(manifest, payload)
    payload['verification.json'] = json_bytes(result)
    payload['inventory.json'] = json_bytes({name: dict(sha256=digest(data), bytes=len(data)) for name, data in sorted(payload.items())})
    verify_capsule(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 26, 0, 0, 0)); info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    verify_capsule(previous.archive_payload(output))
    for path, data in [(manifest_output, payload['manifest.json']), (html_output, payload['review.html'])]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return dict(verification=result, archive_sha256=digest(output.read_bytes()),
                archive_bytes=output.stat().st_size, members=len(payload),
                manifest_sha256=digest(payload['manifest.json']), html_sha256=digest(payload['review.html']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-archive', type=Path)
    parser.add_argument('--amendments', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest-output', type=Path)
    parser.add_argument('--html-output', type=Path)
    parser.add_argument('--check-capsule', type=Path)
    args = parser.parse_args()
    if args.check_capsule:
        result = verify_capsule(previous.archive_payload(args.check_capsule))
    else:
        require(args.return_archive and args.amendments and args.output, 'supply return archive, amendments and output')
        result = build(args.return_archive, args.amendments, args.output, args.manifest_output, args.html_output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
