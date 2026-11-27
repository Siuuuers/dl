#!/usr/bin/env python3
"""Independent rank-sum check of reported pairwise AUROC values."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def rank_sum_auc(scores, labels):
    """Mann-Whitney rank identity; independent of pairwise implementation."""
    ordered=sorted(zip(scores, labels))
    positives=sum(labels); negatives=len(labels)-positives
    if not positives or not negatives:return None
    positive_rank_sum=0.0;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        midrank=((i+1)+j)/2
        positive_rank_sum+=midrank*sum(l for _,l in ordered[i:j])
        i=j
    return (positive_rank_sum-positives*(positives+1)/2)/(positives*negatives)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    summary=json.loads((a.directory/'corner-quality-summary.json').read_text())
    rows_path=a.directory/'corner-quality-rows.json.gz'
    if rows_path.exists():
        rows_payload=gzip.decompress(rows_path.read_bytes())
    else:
        rows_path=a.directory/'corner-quality-rows.json'
        rows_payload=rows_path.read_bytes()
    rows=json.loads(rows_payload)
    checks=[]
    for ref,record in summary['results'].items():
        for scope in ('matched_returned_pair_ranking','fixed_comparison_subset_ranking'):
            cohort=[r for r in rows[ref] if scope=='matched_returned_pair_ranking' or 'maximum_view_disagreement_px' in r]
            assert len(cohort)==record[scope]['rows']
            for feature,metrics in record[scope]['features'].items():
                direction=1 if feature=='maximum_view_disagreement_px' else -1
                scores=[direction*r[feature] for r in cohort];labels=[r['worst_error_px']>5 for r in cohort]
                value=rank_sum_auc(scores,labels)
                assert abs(value-metrics['error_over5_ranking_auc'])<1e-12
                checks.append(ref+'/'+scope+'/'+feature)
    report={'passed':True,'method':'Independent Mann-Whitney rank-sum identity checked against pairwise ranking values; ties receive midranks.',
            'absolute_tolerance':1e-12,'check_count':len(checks),'checks':checks,
            'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'summary_sha256':hashlib.sha256((a.directory/'corner-quality-summary.json').read_bytes()).hexdigest(),
            'row_file':rows_path.name,'row_file_sha256':hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            'row_payload_sha256':hashlib.sha256(rows_payload).hexdigest()}
    (a.directory/'ranking-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Passed {len(checks)} independent rank-sum checks.')
if __name__=='__main__':main()
