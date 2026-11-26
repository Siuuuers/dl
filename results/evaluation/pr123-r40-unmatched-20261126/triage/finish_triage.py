"""Read-only manual+mechanical audit of the 64 R39 other-unmatched outputs."""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(description='Attach the recorded source-pixel triage to read-only R39 mechanical results.')
parser.add_argument('--root', type=Path, required=True, help='dl repository root')
parser.add_argument('--image-root', type=Path, required=True, help='Directory containing original validation PNGs')
parser.add_argument('--output', type=Path, required=True, help='Directory containing mechanical.json and review images')
args = parser.parse_args()
ROOT = args.root.resolve()
OUT = args.output.resolve()
IMAGE = args.image_root.resolve()
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou

data = json.loads((OUT / 'mechanical.json').read_text())
manifest_path = ROOT / 'data/reviews/fronts-2026-11-26-r38/manifest.json'
cache_path = ROOT / 'results/evaluation/pr123-r39-diagnosis-20261126/clipped-nms-predictions.jsonl'
expected_cache_sha = 'd4b404997b891192839c6551fa4513fea49f2c51bbc7cfb9212185ace45ff0f2'
expected_reference_sha = '0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5'
if hashlib.sha256(cache_path.read_bytes()).hexdigest() != expected_cache_sha or data['cache_sha256'] != expected_cache_sha:
    raise ValueError('Manual review belongs to the fixed R39 clipped cache; rebuild and re-review changed inputs.')
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_reference_sha or data['reference_sha256'] != expected_reference_sha:
    raise ValueError('Manual review belongs to the accepted R38 manifest; do not reuse it on another reference.')
manifest = json.loads(manifest_path.read_text())
file_map = {f['image_id']: f for f in manifest['files']}
_, *records = [json.loads(l) for l in cache_path.read_text().splitlines()]
cache = {r['image_id']: r for r in records}

background = {40, 41, 50, 55, 59, 61, 62, 64}
excluded = {13, 19, 21, 24}
uncertain = {6, 39, 42}
missed = {27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 47, 48, 58}
mixed_occluder = {5, 44}
high_association = {1, 2, 4, 7, 22, 26, 29, 30, 31, 33, 34, 35, 36, 37, 38, 43, 45, 46, 47, 48, 54, 56, 58, 60, 63}

# Observations were made only after viewing every full image in context_01..04
# and all exact-coordinate paired source/overlay sheets_01..11.
notes = {
 1: 'The proposal is a shorter/lower envelope of the visible bottom front already represented by R38 row47. No separate front is suggested inside this proposal.',
 2: 'The proposal covers a shallow top-frame strip of the same broad clipped front as row1; its short height explains the low IoU. A second front is not independently visible in that strip.',
 3: 'The proposal shares the lower visible edge of row17 and extends down over the right jaw/rack. It is associated with that occluded front, but the portion behind the jaw is not directly observable.',
 4: 'The proposal is a lower subset of the right-frame-clipped row8 front, already matched by p33.',
 5: 'Most of the proposal is the right jaw and rack below row17; only its top overlaps the accepted front. A shifted extent variant and an occluder false response cannot be separated confidently from these pixels. No independent new cassette is established.',
 6: 'The proposal is mainly on the left white jaw/dark rack below row11, with only 0.092 IoU to that front. No separate frontal seam is independently visible. Keep this as an occluder/hidden-target ambiguity, not a new label or a confirmed nonexistent object.',
 7: 'The proposal covers the narrow visible left-of-jaw row51 front with a modest shift/extent difference; p58 already matches that accepted instance.',
 8: 'The visible row49 sliver is present at the left of the proposal; the right side extends into the white jaw. This is an extent variant around an already matched front, with hidden extent unobservable.',
 9: 'The visible row50 sliver is present at the left; the right side extends into the white jaw. It is not evidence for an additional independent cassette.',
 10: 'The proposal covers row49 and extends rightward beneath the jaw, with a slightly higher top boundary. It duplicates an accepted target region rather than establishing a separate visible front.',
 11: 'The proposal straddles the row50/row51 boundary and extends into the jaw. Its best association is row50; exact seam ownership is less certain than the numerical nearest-reference assignment.',
 12: 'The proposal starts on the visible row50 region but extends into the jaw and toward row49. No independent additional front is established.',
 13: 'The small bottom left-of-jaw front lies in the explicitly excluded R37 row52 region. The pixels show a partial front; the exclusion is a chosen reference-population decision, not proof of background. This proposal uses a wider extent than the removed row.',
 14: 'The proposal covers the visible row51 sliver and extends rightward into the jaw/slot. The accepted instance already has matched p58.',
 15: 'The proposal is lower than the row51 box and partly approaches the explicitly excluded lower row52 region. Best accepted-row association is row51; exact ownership across the lower seam remains uncertain.',
 16: 'The proposal covers the lower part of row50 and crosses toward row51. It is an extent/seam variant of the existing narrow stack, not evidence of an additional front.',
 17: 'The proposal spans the row49/row50 transition and extends under the jaw; row50 is only the best geometric association, not a definitive physical ownership claim.',
 18: 'The proposal overlaps the visible row50 region but is substantially wider toward the jaw and somewhat vertically shifted.',
 19: 'The proposal covers the same small bottom front excluded as R37 row52; it has no overlap with the accepted R38 targets. Exclusion remains authoritative.',
 20: 'The proposal overlaps lower row51, extends rightward into the jaw, and approaches the excluded lower fragment. No extra independent accepted target is established.',
 21: 'The proposal covers the excluded R37 row52 partial front and a wider area toward the jaw. Its zero accepted-target overlap is partly a policy-population effect, not evidence that the visible front is nonexistent.',
 22: 'The very shallow top-clipped proposal is a subset of the already matched row12 region. Its height is about 4.5 source pixels, making IoU sensitive to a few pixels.',
 23: 'The proposal straddles the lower row50/row51 region and extends rightward beneath the jaw. Existing row51 is its best accepted association.',
 24: 'This bottom-frame proposal overlaps the explicitly excluded R37 row30 box at IoU 0.779704. Accepted R38 row29 is only a weak neighbor. This is a direct exclusion-policy association; no label restoration is proposed.',
 25: 'The small visible triangular row31 region beside the finger is inside the proposal, which extends farther onto the finger. The accepted front already matches p30; its precise hidden-side extent is not visible.',
 26: 'The proposal covers the lower portion of the larger visible bottom-frame row29 front and omits much of its upper extent; p28 already matches that target.',
 27: 'The proposal covers part of the sloped front beside the finger near missed row30 and crosses toward row31. It is a localization/ownership variant, not an absence of model activity. Exact sloping seam ownership is uncertain.',
 28: 'The proposal associates with missed row30 beside the finger but is lower and partly wider; it also approaches row31. The accepted row remains missed at the fixed IoU gate.',
 29: 'The model proposes the visible row55 left-of-jaw front at high confidence, but its box extends farther into the jaw. Best IoU is 0.487830, just below the fixed 0.50 gate. The front was proposed; localization failed the accepted convention.',
 30: 'The high-confidence proposal includes the visible row56 front and extends into the jaw. Accepted row56 is still unmatched because the proposed envelope differs substantially.',
 31: 'A second proposal covers the same missed row55 front and a larger jaw-occluded extent. This is an alternative geometry for one target, not another independent cassette.',
 32: 'The proposal covers the accepted right-of-jaw row59 sliver but extends leftward over the jaw; p65 already matches that instance.',
 33: 'The proposal covers the bottom visible row57 front but its envelope is wider/lower than the small accepted fragment. This is one of three alternatives associated with the same missed target.',
 34: 'The proposal covers row57 and extends farther into the jaw than p61. No extra front is independently visible.',
 35: 'The proposal is another envelope of the same missed row55 front, with a rightward jaw extension. It does not represent a third physical target.',
 36: 'The proposal covers the missed row56 region with a narrower envelope than p57, but still fails the accepted IoU gate.',
 37: 'The proposal includes row57 and a broad jaw extension. It is an alternative geometry in the same occluded-front region.',
 38: 'The proposal includes row56 and extends rightward into the jaw, with some vertical overlap into adjacent rows.',
 39: 'The proposal is only 3.835 source pixels tall at the top frame. It overlaps both the current row1 region and the explicitly excluded R37 row1 sliver (IoU 0.239 with the latter). The pixels do not resolve a separate extra cassette or a single unambiguous ownership assignment. Do not change either accepted decision.',
 40: 'The proposal is below the bottom accepted cassette on the continuous rack/base support strip, next to a binder clip. The full frame and source crop show fixture material rather than a separate cassette front.',
 41: 'The proposal is another segment of the same rack/base support strip in an empty slot, not an observed cassette front. Only a negligible corner overlap exists with row19.',
 42: 'A 9-pixel-high pale strip at the top frame could be a clipped front continuation/neighbor or a fixture/rim. The remaining pixels and existing decisions do not establish an independent target. Preserve this as a possible unlabeled clipped-target/fixture ambiguity; no automatic annotation.',
 43: 'The proposal is a smaller envelope of the visible row14 front beside the finger; the wider accepted target already has p21. The higher proposal confidence does not make its box the better accepted match.',
 44: 'The proposal shares the bottom of row18 but lies mainly across the jaw tips and central opening below it. It may be a shifted/overextended front response or an occluder response. No independent additional cassette is established.',
 45: 'The proposal covers a shallow upper rim of the same row53 front near the image boundary, while p13 already matches its larger accepted extent.',
 46: 'The proposal covers a narrower left subset of the bottom row52 front; p55 already matches the accepted full visible extent.',
 47: 'The proposal includes the missed row51 fragment to the right of the jaw but extends far leftward across the jaw. R33 explicitly changed this target to the observed fragment (left edge 802 instead of 749); that convention explains why broad predictions can fail IoU.',
 48: 'A second proposal covers the same missed row51 fragment, with less leftward jaw extension than p56. Its IoU 0.456572 still fails the fixed accepted-reference threshold.',
 49: 'The proposal includes the visible row42 sliver beside the jaw and extends rightward into the jaw; p53 already matches that instance.',
 50: 'The proposal covers the white horizontal rack fixture above the mesh. Full-frame context and adjacent original frame090 reveal the same continuous rail and its mounting bolts; it is not a cassette front.',
 51: 'The proposal covers the visible row43 sliver and extends rightward across the jaw/slot; p59 already matches the same accepted target.',
 52: 'The proposal spans the row41/row42 seam and extends across the jaw, with row42 the best accepted association. No independent new target is shown.',
 53: 'The proposal is a lower/overextended envelope at row41 that overlaps row42 and the jaw. Row41 already matches p50; exact seam ownership should not be inferred from this box alone.',
 54: 'The proposal covers only the left visible portion of row47 beside the right jaw; p54 already matches the larger accepted front.',
 55: 'The proposal covers another segment of the same white top rack fixture as p52. The adjacent original090 view exposes the rail and bolts, resolving the misleading cassette-like pale strip.',
 56: 'The proposal covers a left subset of row46 beside the right jaw; p47 already matches the accepted front.',
 57: 'The proposal covers the visible right-of-jaw row49 front but extends farther left over the jaw. That target already matches p56.',
 58: 'The high-confidence proposal includes missed row39 beside the jaw but is wider and slightly shifted. Its IoU 0.489707 is just below the fixed gate; this is clear proposal presence with localization mismatch.',
 59: 'The proposal covers a blue/gray illuminated object at the lower-right image boundary outside the rack. The full context identifies this as background furniture/peripheral context, not a cassette front.',
 60: 'The proposal covers the visible row41 region but extends to the right into the jaw and slightly across the row40 boundary. Row41 already matches p43.',
 61: 'The proposal is a second extent of the same lower-right blue/gray background item as p51.',
 62: 'The proposal is another extent of the same lower-right blue/gray background item as p51, not another independent false object.',
 63: 'The proposal is a vertically smaller/shifted subset of the lowest visible row50 front beside the finger. That accepted target already matches p52.',
 64: 'The proposal covers the lower strip of the same lower-right blue/gray background item. Four remaining outputs respond to that single background region.',
}

history = []
for rel in ['results/review/front-review-20261125-r35/operations.json', 'results/review/front-review-20261126-r38/operations.json']:
    for op in json.loads((ROOT / rel).read_text()):
        if op['action'] == 'delete':
            history.append({
                'image_id': op['image_id'], 'source': rel,
                'base_row': op['base_row'],
                'base_reference': 'R34' if 'r35/' in rel else 'R37',
                'bbox_xyxy': parse_pose_labels(op['before_line'], 1280, 720)[0]['bbox_xyxy'],
                'reason': op['reason'],
            })

groups = Counter()
images = {}
missed_groups = defaultdict(list)
outranking = []
for case in data['cases']:
    n = case['case']
    category = ('background' if n in background else 'excluded_target_region' if n in excluded
                else 'uncertain' if n in uncertain else 'missed_target_localization_variant' if n in missed
                else 'matched_target_region')
    groups[category] += 1
    physical = {
        'category': category,
        'association_confidence': ('high' if n in background or n in excluded or n in high_association
                                   else 'low' if n in uncertain or n in mixed_occluder else 'medium'),
        'finding': notes[n],
        'independent_additional_target_established': False,
        'hidden_corners_adjudicated': False,
        'label_change': 'none',
        'review_status': 'manual source-frame context and exact-coordinate crop reviewed',
        'ambiguous_front_vs_occluder': n in mixed_occluder or n in uncertain,
    }
    if n in excluded:
        physical['excluded_identity'] = f"R37/{case['image_id']}:{30 if n == 24 else 52}"
    if n in background:
        physical['background_region'] = ('054_base_support' if n in {40,41} else '089_top_fixture' if n in {50,55} else '090_blue_peripheral')
    case['physical_review'] = physical
    case['exclusion_history_overlaps'] = [dict(h, iou=_box_iou(case['prediction']['bbox_xyxy'], h['bbox_xyxy'])) for h in history if h['image_id'] == case['image_id']]
    if n in missed:
        missed_groups[(case['image_id'], case['best_reference_row'])].append(case)
    paired_id = case['matched_primary_prediction_id_for_best_reference']
    if paired_id is not None:
        record = cache[case['image_id']]
        paired = record['predictions'][record['primary_prediction_ids'].index(paired_id)]
        case['matched_prediction_pair'] = {
            'primary_prediction_id': paired_id,
            'confidence': paired['confidence'],
            'pair_box_iou': _box_iou(case['prediction']['bbox_xyxy'], paired['bbox_xyxy']),
            'other_prediction_has_higher_confidence': case['prediction']['confidence'] > paired['confidence'],
        }
        if category == 'matched_target_region' and case['matched_prediction_pair']['other_prediction_has_higher_confidence']:
            outranking.append({'image_id': case['image_id'], 'other_primary_prediction_id': case['primary_prediction_id'], **case['matched_prediction_pair']})
    name = case['image_id']
    if name not in images:
        source = IMAGE / f'{name}.png'
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        expected = file_map[name]['source_image_sha256']
        images[name] = {'path': str(source), 'sha256': sha, 'reference_source_sha256': expected, 'matches_reference_source': sha == expected}
    assert images[name]['matches_reference_source'], name

data.update({
    'scope': 'Manual and mechanical audit of all 64 other-unmatched predictions after R39 clipped NMS, with accepted R38 labels unchanged. Prediction counts are not independent object counts.',
    'reviewed_on': '2026-11-26',
    'reviewer': 'assistant source-pixel review, not a new human-approved annotation',
    'reference_policy_sources': ['docs/annotation-boundaries-r33.md', 'docs/front-review-r38.md', 'results/review/front-review-20261126-r38/operations.json', 'results/review/front-review-20261125-r35/operations.json'],
    'reference_policy': 'Boxes follow observed frontal material including owned lip/rim; do not include all-occluder boundary strips. Hidden-corner predictions remain allowed and do not turn v=0 annotations into observed truth. Case-specific exclusions remain accepted.',
    'counts': dict(groups),
    'counts_are': 'predictions, not independent physical cassettes',
    'matched_target_region_detail': {'predictions': groups['matched_target_region'], 'distinct_best_reference_targets': len({(c['image_id'],c['best_reference_row']) for c in data['cases'] if c['physical_review']['category']=='matched_target_region'}), 'mixed_front_occluder_cases': [5,44], 'interpretation': 'Geometric/source-region associations; all 35 are not asserted to be verified physical duplicates.'},
    'missed_target_associations': [
        {'image_id': k[0], 'r38_row': k[1], 'primary_prediction_ids': [c['primary_prediction_id'] for c in cs],
         'best_iou': max(c['best_reference_iou'] for c in cs),
         'best_iou_primary_prediction_id': max(cs,key=lambda c:c['best_reference_iou'])['primary_prediction_id']}
        for k,cs in missed_groups.items()
    ],
    'higher_confidence_worse_match_pairs': outranking,
    'mechanical_overlap_counts': {'best_reference_iou_at_least_0_1':sum(c['best_reference_iou']>=.1 for c in data['cases']), 'best_reference_iou_at_least_0_3':sum(c['best_reference_iou']>=.3 for c in data['cases']), 'best_reference_iou_exactly_zero':sum(c['best_reference_iou']==0 for c in data['cases'])},
    'source_images': images,
    'review_materials': {'full_context': [f'context_{i:02}.png' for i in range(1,5)], 'all_case_crops': [f'sheet_{i:02}.png' for i in range(1,12)], 'top_fixture_context': 'top_sequence.png', 'clipped_top_ambiguity_context': 'top_sequence_053_056.png'},
    'limitations': ['This is one assistant review, not a new accepted annotation round.', 'Occluded material and hidden corner truth cannot be established from these single images.', 'A nearest-reference IoU association is not itself a physical identity proof.', 'No model inference or training was performed; no metric gate or label was changed.', 'The reviewed 20-image development batch is not an independent deployment test.', 'No independent additional cassette was established; three cases remain explicitly uncertain, including possible clipped front case42.'],
    'no_training_guidance': 'Retain accepted clipped-NMS baseline. Any next filter must be label-independent at runtime and replayed against the frozen reference for matched identity, misses, and corner behavior. Avoid broad confidence or NMS tightening: six unmatched extent variants outrank already matched predictions. Background-region observations justify a conditional rack-ROI experiment only if a general rack location is independently available; do not encode validation-image masks. Keep review-only geometric associations out of runtime filtering.',
})

assert len(data['cases']) == 64 and sum(groups.values()) == 64
assert set(notes) == set(range(1,65))
assert len(missed_groups) == 6 and len(outranking) == 6
(OUT / 'review.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps({'counts':data['counts'],'missed_target_associations':data['missed_target_associations'],'source_images_verified':len(images),'higher_confidence_worse_match_pairs':len(outranking)},indent=2))
