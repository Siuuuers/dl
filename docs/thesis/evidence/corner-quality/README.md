# Corner quality and crop containment evidence

28 November 2026, Asia/Hong_Kong.

This folder records a CPU analysis of saved predictions from twenty development photographs. The [research note](../../research-corner-quality.md) explains the findings and their practical limits. The calculation generates no model predictions, changes no reference annotations, and selects no output threshold.

## Principal result

A fixed crop, 1.5 times the predicted bounding-box width and height, contains both annotated lower corners in every matched complete-reference target. Missing complete targets remain in the accounting.

| Reference and saved output | Eligible complete targets | Missed complete targets | Matched crops containing both corners | Current pairs over 5 px contained |
|---|---:|---:|---:|---:|
| Fixed reference, original control | 584 | 7 | 577/577 | 66/66 |
| Fixed reference, weighted candidate | 584 | 7 | 577/577 | 66/66 |
| Reviewed reference, original control | 583 | 0 | 583/583 | 17/17 |
| Reviewed reference, weighted candidate | 583 | 0 | 583/583 | 16/16 |

For each saved output, 34 crops are clipped at the image boundary. These include eleven inaccurate pairs under the fixed reference and five under the reviewed reference. No matched complete pair is unavailable under the declared score and bounds rules. A crop containing the annotations establishes feasibility of the input region; it does not establish refinement accuracy.

The fixed reference contains 666 fronts and is not the upstream 664-front validation reference. Internal JSON reference keys remain `frozen666`, `historical663`, `r38`, and `r50` for reproducibility. The original control and weighted candidate come from different experiment records; this is a descriptive comparison of their saved outputs, not a new controlled training comparison.

## Files

| File | Content |
|---|---|
| `corner-quality-summary.json` | All-eligible partition, conditional ranking statistics, source hashes, image conditioning, and view-association ambiguity counts |
| `corner-quality-rows.json.gz` | Complete per-target ranking and comparison-view records; gzip-compressed JSON |
| `crop-containment-summary.json` | Detection counts, all-eligible corner accounting, crop containment, image-edge counts, and source hashes |
| `crop-containment-rows.json.gz` | Complete per-target crop coordinates and containment records; gzip-compressed JSON |
| `ranking-verification.json` | Independent rank-sum verification of 28 pairwise-AUROC values |
| `input-provenance.json` | Immutable archive locations, exact member names, byte counts, SHA-256 values, and checkpoint identities |
| `protocol.md` | Original diagnostic definitions and explicitly labelled exploratory follow-up checks |
| `manifest.json` | Hashes of this evidence and the packaged scripts and research note |

Full row evidence occupies approximately 442 kB compressed. The scripts can also produce uncompressed JSON by omitting `--compress-rows`. The verification script accepts either representation and prefers `.json.gz` if both exist, so use a clean output directory for a replay.

## Cached inputs

Four files are needed. Python 3 and its standard library are sufficient; weights, images, a GPU, and model libraries are unnecessary.

| Required input | Immutable source and member |
|---|---|
| Reference/candidate snapshot | Repository file `docs/thesis/evidence/r55-automatic-output-audit.zip`, member `r55-automatic-output-audit/r54-analysis/inputs.json` |
| Frozen scorer | The same repository ZIP, member `r55-automatic-output-audit/r54-analysis/pose_scoring.py` |
| Original and contrast-view predictions | [Contrast experiment archive](https://raw.githubusercontent.com/Siuuuers/dl/343f8d759ccaba671fa38ad4e9674d75a1bb5ba7/results/l4/baomai-clahe-inference-r53-l4-001/baomai-clahe-inference-r53-results.zip), member `predictions.json` |
| Original and enlarged-view predictions | [Enlarged-input experiment archive](https://raw.githubusercontent.com/Siuuuers/dl/e8a70f67cba1bf99e2d8a7bce9c93b5d4d7996b2/results/a100/baomai-scale-inference-r53-a100-001/baomai-scale-inference-r53-results.zip), member `predictions.json` |

The archives correspond to the completed [contrast experiment](https://github.com/Siuuuers/dl/pull/149) and [enlarged-input experiment](https://github.com/Siuuuers/dl/pull/150). Their unchanged controls share box and corner coordinates. The enlarged-input archive's control is used consistently for the new quality and crop analysis. The weighted candidate and reference annotations come from the existing audit snapshot; its lineage is recorded in `input-provenance.json`.

### Extract and verify the required members

From the repository root, save the two linked archives and substitute their paths and a new cache directory below. This checks each complete archive and each required member against the recorded SHA-256 before writing the four local inputs.

```bash
python - /path/to/baomai-clahe-inference-r53-results.zip \
  /path/to/baomai-scale-inference-r53-results.zip \
  /path/to/cassette-cache <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ledger = json.loads(Path(
    'docs/thesis/evidence/corner-quality/input-provenance.json'
).read_text())
archive_paths = [
    Path('docs/thesis/evidence/r55-automatic-output-audit.zip'),
    Path(sys.argv[1]),
    Path(sys.argv[2]),
]
cache = Path(sys.argv[3])
for archive_path, category, record in zip(
    archive_paths, ('snapshot', 'contrast', 'enlarged'), ledger['archives']
):
    actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if actual != record['archive_sha256']:
        raise ValueError(f'Archive hash mismatch: {archive_path}')
    with ZipFile(archive_path) as archive:
        for member, metadata in record['members'].items():
            payload = archive.read(member)
            if hashlib.sha256(payload).hexdigest() != metadata['sha256']:
                raise ValueError(f'Member hash mismatch: {member}')
            destination = cache / category / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
print('Verified and extracted the four required inputs.')
PY
```

### Reproduce the diagnostics

From the repository root, set the same cache path. Outputs go to a separate replay directory.

```bash
CASSETTE_CACHE=/path/to/cassette-cache

python -B scripts/analyze_corner_quality.py \
  --inputs "$CASSETTE_CACHE/snapshot/r55-automatic-output-audit/r54-analysis/inputs.json" \
  --scorer "$CASSETTE_CACHE/snapshot/r55-automatic-output-audit/r54-analysis/pose_scoring.py" \
  --clahe "$CASSETTE_CACHE/contrast/predictions.json" \
  --scale "$CASSETTE_CACHE/enlarged/predictions.json" \
  --output "$CASSETTE_CACHE/replay" \
  --compress-rows

python -B scripts/check_corner_crop_containment.py \
  --inputs "$CASSETTE_CACHE/snapshot/r55-automatic-output-audit/r54-analysis/inputs.json" \
  --scorer "$CASSETTE_CACHE/snapshot/r55-automatic-output-audit/r54-analysis/pose_scoring.py" \
  --control "$CASSETTE_CACHE/enlarged/predictions.json" \
  --output "$CASSETTE_CACHE/replay" \
  --compress-rows

python -B scripts/verify_corner_quality_ranking.py "$CASSETTE_CACHE/replay"
```

Do not use `python -O`, because the scripts use assertions for baseline and accounting verification. The ranking calculation reproduces all eight saved reference/candidate baseline combinations. The crop calculation retains complete-reference misses, matched unavailable pairs, and all inaccurate returned pairs. The separate verification script checks all 28 pooled ranking results using the independent Mann–Whitney rank-sum formula.

## Interpretation of ranking and correspondence

The quality comparison uses 577 matched returned complete pairs under the fixed reference and 583 under the reviewed reference. Predicted boxes alone select mutual-best cross-view associations at fixed overlap of at least 0.50. All cohort members have usable associated point pairs in both views. Ten contrast associations and thirteen enlarged-view associations have multiple eligible overlap candidates; none has an exact best-overlap tie. These cases remain in the analysis. Ground truth is consulted only afterward to score errors and audit reference-assignment consistency.

All inaccurate pairs have both corner scores at least 0.8. Comparison-view disagreement gives pooled AUROC of 0.745 and 0.839, but these decrease to 0.587 and 0.659 when successful and inaccurate outputs are compared only within the same photograph. This diagnostic does not establish a calibrated correctness probability or an automatic rejection rule. The full research note reports the definitions, comparison denominators, reference sensitivity, and relevant primary literature.
