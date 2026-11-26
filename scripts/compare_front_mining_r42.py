#!/usr/bin/env python3
"""Compare three R42 label versions using fixed R41 postclip predictions.

This CPU-only replay performs no inference or training. It scores only the three
annotation-complete mined-development frames and verifies the returned receipt,
source files, labels, and exact upstream match pairs. Visual case attribution is
kept in the separately saved assistant review; it is not regenerated here.

Example:
    python scripts/compare_front_mining_r42.py \
        --return-root /path/to/extracted-r41-return \
        --output /path/to/comparison.json
"""
from pathlib import Path
import argparse, hashlib, json, sys

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()

def require(condition, message):
    # Input contract checks must remain active even when Python runs with -O.
    if not condition:
        raise ValueError(message)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1],
                    help='Repository containing the receipt-pinned scoring scripts (default: this script repo)')
    ap.add_argument('--manifest', type=Path,
                    help='R42 normalized review manifest (default: repository R42 manifest)')
    ap.add_argument('--return-root', type=Path, required=True,
                    help='Extracted R41 return containing receipt.json, predictions.jsonl, and review/')
    ap.add_argument('--output', type=Path, required=True,
                    help='JSON destination for the quantitative comparison; visual triage stays separate')
    args=ap.parse_args()
    args.repo=args.repo.resolve()
    args.return_root=args.return_root.resolve()
    manifest_path=(args.manifest or args.repo/'data/reviews/additional-fronts-2026-11-26-r42/manifest.json').resolve()
    receipt_path=args.return_root/'receipt.json'
    cache_path=args.return_root/'predictions.jsonl'
    require(args.output.resolve() not in {manifest_path, receipt_path, cache_path},
            'Output must not overwrite an input manifest, receipt, or prediction cache')
    manifest=json.loads(manifest_path.read_text());receipt=json.loads(receipt_path.read_text())
    selected=['frame_0146','frame_0371','frame_0383']
    require(set(selected)==set(manifest['annotation_complete_ids']), 'Unexpected annotation-complete cohort')
    require(not set(selected).intersection(manifest['held_image_ids']), 'A selected frame is held')
    # Use the exact scoring/parser implementation recorded by the R41 receipt.
    for name in ['evaluate_pose.py','pose_scoring.py','pose_checkpoint.py']:
        require(sha(args.repo/'scripts'/name)==receipt['source_scripts_sha256'][name], f'Receipt script hash mismatch: {name}')
    sys.path.insert(0,str(args.repo/'scripts'))
    from evaluate_pose import parse_pose_labels
    from pose_scoring import score_image, aggregate_scores, _box_iou
    cache={}
    for line in cache_path.read_text().splitlines():
        r=json.loads(line)
        if r['image_id'] in selected:
            require(r['image_id'] not in cache, 'Duplicate selected prediction record')
            cache[r['image_id']]=r
    require(set(cache)==set(selected), 'Selected prediction records are incomplete')
    frames={f['image_id']:f for f in manifest['frames'] if f['image_id'] in selected}
    version_names=['upstream_source','working_baseline','reviewed']
    scores_by_version={v:[] for v in version_names};results=[]
    numeric_keys=['gold_count','prediction_count','tp','fn','fp','duplicate_fp','other_fp','visible_endpoints','returned_endpoints','missing_endpoints','endpoint_coverage','conditional_median_error_px','conditional_p95_error_px']
    for fid in selected:
        f=frames[fid];pred_record=cache[fid];width,height=f['image_size']
        require(f['annotation_complete'] and f['eligible_for_candidate_export'], f'Incomplete annotation: {fid}')
        require(not f['unresolved_stable_ids'] and not f['eligible_for_independent_evaluation'], f'Unexpected review/evaluation status: {fid}')
        require(pred_record['source_image_sha256']==f['source_image_sha256'], f'Prediction image hash mismatch: {fid}')
        require(sha(args.return_root/'review'/fid/'original.jpg')==f['source_image_sha256'], f'Original image hash mismatch: {fid}')
        comparison=json.loads((args.return_root/'review'/fid/'comparison.json').read_text())
        require(comparison==pred_record, f'Per-frame/cache records disagree: {fid}')
        predictions=pred_record['predictions']
        primary_ids=[p['primary_prediction_id'] for p in predictions]
        require(primary_ids==pred_record['postclip_nms']['retained_primary_prediction_ids'], f'Postclip prediction IDs disagree: {fid}')
        require(len(predictions)==pred_record['filtered_outputs'], f'Postclip output count mismatch: {fid}')
        require(receipt['inference']['conf']==.13 and receipt['inference']['source_comparison_iou']==.5, 'Unexpected box confidence/matching contract')
        require(receipt['inference']['endpoint_confidence']==.5 and receipt['inference']['postclip_nms_iou']==.7, 'Unexpected point/postclip contract')
        source_rows=sorted(f['rows'],key=lambda r:r['source_row'])
        require([r['source_row'] for r in source_rows]==list(range(1,len(source_rows)+1)), f'Source row identities are not contiguous: {fid}')
        candidate_rows=sorted([r for r in f['rows'] if r['after_line'] is not None]+f['additions'],key=lambda r:r['output_row'])
        lines={
            'upstream_source':[r['source_line'] for r in source_rows],
            'working_baseline':[r['before_line'] for r in source_rows],
            'reviewed':[r['after_line'] for r in candidate_rows],
        }
        ids={v:[r['stable_id'] for r in (candidate_rows if v=='reviewed' else source_rows)] for v in version_names}
        texts={v:'\n'.join(lines[v])+'\n' for v in version_names}
        expected_hashes={'upstream_source':f['source_label_sha256'],'working_baseline':f['baseline_label_sha256'],'reviewed':f['candidate_label_sha256']}
        for v in version_names:
            require(sha_text(texts[v])==expected_hashes[v], f'Label hash mismatch: {fid}/{v}')
        require(texts['upstream_source']==(args.return_root/'review'/fid/'source.txt').read_text(), f'Original source labels differ: {fid}')
        require(texts['working_baseline']==(args.return_root/'review'/fid/(fid+'.txt')).read_text(), f'Working labels differ: {fid}')
        image_result={'image_id':fid,'image_size':[width,height],'source_image_sha256':f['source_image_sha256'],'baseline_origin':f['baseline_origin'],'fixed_primary_outputs':pred_record['primary_outputs'],'fixed_postclip_outputs':len(predictions),'retained_primary_prediction_ids':primary_ids,'prediction_object_sha256':sha_text(json.dumps(predictions,sort_keys=True,separators=(',',':'))),'versions':{}}
        pair_sets={}
        for version in version_names:
            gold=parse_pose_labels(texts[version],width,height,f'{fid}:{version}')
            score=score_image(gold,predictions,width,height,matching_iou=.5,endpoint_conf=.5)
            scores_by_version[version].append(score)
            if version=='upstream_source':
                require(score['fn']==pred_record['source_unmatched_rows'], f'Source unmatched references do not replay: {fid}')
                require(score['fp']==pred_record['source_unmatched_outputs'], f'Source unmatched predictions do not replay: {fid}')
                require(score['duplicate_fp']==pred_record['source_duplicate_subset'], f'Source duplicate subset does not replay: {fid}')
                replay_pairs=[{'source_row':m['gold_index']+1,'primary_prediction_id':primary_ids[m['prediction_index']],'iou':m['iou']} for m in score['matches']]
                require(replay_pairs==pred_record['source_matches'], f'Exact upstream match pairs do not replay: {fid}')
            pairs=[{'reference_row':m['gold_index']+1,'stable_id':ids[version][m['gold_index']],'primary_prediction_id':primary_ids[m['prediction_index']],'iou':m['iou']} for m in score['matches']]
            pair_sets[version]={(m['stable_id'],m['primary_prediction_id']) for m in pairs}
            unmatched_details=[]
            for gidx in score['unmatched_gold_indices']:
                candidates=[(i,_box_iou(p['bbox_xyxy'],gold[gidx]['bbox_xyxy'])) for i,p in enumerate(predictions) if p['class_id']==gold[gidx]['class_id']]
                best=max(candidates,key=lambda p:(p[1],-p[0])) if candidates else None
                unmatched_details.append({'reference_row':gidx+1,'stable_id':ids[version][gidx],'best_primary_prediction_id':primary_ids[best[0]] if best else None,'best_box_iou':best[1] if best else None,'best_prediction_confidence':predictions[best[0]]['confidence'] if best else None})
            image_result['versions'][version]={
                'label_sha256':expected_hashes[version],
                'metrics':{k:score[k] for k in numeric_keys},
                'unscorable_reference_corner_slots':2*len(gold)-score['visible_endpoints'],
                'matches':pairs,
                'unmatched_reference_rows':unmatched_details,
                'unmatched_primary_prediction_ids':[primary_ids[i] for i in score['unmatched_prediction_indices']],
                'duplicate_subset_primary_prediction_ids':[primary_ids[i] for i in score['duplicate_prediction_indices']],
                'endpoint_errors_px':score['endpoint_errors_px'],
            }
        base=image_result['versions']['working_baseline']['metrics'];rev=image_result['versions']['reviewed']['metrics']
        image_result['reviewed_minus_working_baseline']={k:rev[k]-base[k] if rev[k] is not None and base[k] is not None else None for k in numeric_keys}
        image_result['match_assignment_change_from_working_baseline']={
            'retained_pair_count':len(pair_sets['reviewed']&pair_sets['working_baseline']),
            'new_pairs':[{'stable_id':x[0],'primary_prediction_id':x[1]} for x in sorted(pair_sets['reviewed']-pair_sets['working_baseline'])],
            'lost_pairs':[{'stable_id':x[0],'primary_prediction_id':x[1]} for x in sorted(pair_sets['working_baseline']-pair_sets['reviewed'])],
        }
        results.append(image_result)
    totals={v:aggregate_scores(scores_by_version[v]) for v in version_names}
    result={
        'kind':'fixed-r41-prediction-label-version-diagnostic',
        'scored_image_ids':selected,
        'excluded_held_image_ids':manifest['held_image_ids'],
        'scope':'Only the three annotation-complete mined-development frames. No held image was scored in this diagnostic.',
        'interpretation':'Differences arise solely from reference labels and matching assignments. The checkpoint and every saved prediction are fixed. This is not model improvement, independent accuracy, or a generalization estimate.',
        'inference_performed':False,'training_performed':False,'human_review_claimed':False,
        'visual_triage':'Separate assistant review decisions; not regenerated by quantitative replay.',
        'provenance':{
            'manifest_path':str(manifest_path),'manifest_sha256':sha(manifest_path),
            'prediction_cache_path':str(cache_path),'prediction_cache_sha256':sha(cache_path),
            'receipt_sha256':sha(receipt_path),'return_pr':manifest['return_pr'],'return_commit':manifest['return_commit'],
            'checkpoint_sha256':receipt['checkpoint_sha256'],
            'parser_sha256':sha(args.repo/'scripts/evaluate_pose.py'),'scorer_sha256':sha(args.repo/'scripts/pose_scoring.py'),
            'parser_and_scorer_match_r41_receipt':True,'original_source_counts_and_exact_match_pairs_reproduced':True,
            'replay_script_sha256':sha(__file__),
        },
        'contract':{
            'matching_iou':.5,'matching':'Same-class confidence-descending greedy one-to-one; choose highest IoU remaining reference, stable index ties.',
            'predictions':'Already confidence-filtered and postclip-filtered saved R41 outputs; no new inference or additional filtering.',
            'box_confidence':.13,'ordinary_nms_iou':.7,'postclip_nms_iou':.7,
            'duplicates':'Subset of unmatched predictions overlapping an already-matched same-class reference at IoU>=0.5; do not add them again to unmatched totals.',
            'endpoint_confidence':.5,'point_metrics':'Reference visibility2 only. Error measured only for matched boxes with finite in-image predicted point and confidence>=0.5; reference-visible points without such returns count missing, including points on unmatched boxes. Pooled quantiles are conditional on returned endpoints.',
            'unavailable_reference_corners':'Reference visibility0 is unscorable for corner error. Predicted hidden corners are permitted and do not turn into missing/false reference truth. Prediction confidence never changes annotation visibility.',
            'selection_caveat':'Mined disagreements and correlated pool images; not an independent test. Reference edits may change IoU assignment and the conditional error population.',
        },
        'images':results,'aggregate':totals,
        'aggregate_reviewed_minus_working_baseline':{k:totals['reviewed'][k]-totals['working_baseline'][k] if totals['reviewed'][k] is not None and totals['working_baseline'][k] is not None else None for k in numeric_keys},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'per_image':[{ 'image_id':i['image_id'], **{v:i['versions'][v]['metrics'] for v in version_names}} for i in results],'aggregate':{v:{k:x[k] for k in numeric_keys} for v,x in totals.items()}},indent=2))

if __name__=='__main__':main()
