# Saved-model replay on the revised reference

This is a **provisional development diagnostic** of the retained PR35 model and
the two previously trained PR64 models. No model has been trained on the
18 November labels by this replay. The `candidate` role means the **previous
PR64 candidate**, which trained on the 3,788-object accepted-fronts candidate.

The exact saved predictions are scored separately against the 14 November
completion reference and the 18 November reference. The older completion
reference is distinct from PR66's 12 November primary adjudication reference.
Every model uses the same selected reference within each comparison.

| Reference | Model | Matched | Misses | Unmatched proposals | Duplicates | Both corners within 5 px | Both within 10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 November, 663 objects | PR35 | 648 | 15 | 80 | 32 | 533 | 582 |
| 14 November, 663 objects | PR64 control | 646 | 17 | 83 | 29 | 531 | 583 |
| 14 November, 663 objects | PR64 candidate | 643 | 20 | 103 | 31 | 532 | 583 |
| 18 November, 666 objects | PR35 | 648 | 18 | 80 | 32 | 533 | 582 |
| 18 November, 666 objects | PR64 control | 646 | 20 | 83 | 29 | 531 | 583 |
| 18 November, 666 objects | PR64 candidate | 642 | 24 | 104 | 30 | 532 | 583 |

Both references have 584 objects with two locatable corners, including eligible
missed objects in the corner-success denominator. All three models produce zero
outputs on the three empty photos. The previous candidate remains worse than its
matched control on object misses and unmatched proposals. No promotion follows.

Changed scores between references are annotation effects on fixed predictions,
not model learning. Unmatched proposals are association outcomes; they have not
all been individually adjudicated as background false positives. These repeatedly
used development photos do not establish reliability on an unseen camera session.

## Reproduce without a GPU or network

From the repository root, with the original validation PNGs already materialized:

```bash
python scripts/replay_sequence_reference.py --root . --verify-only
```

The script binds both reference hashes, the original image/label identities, the
prediction ZIP, checkpoint identities and original scoring helper hashes. It
reconciles all saved 14 November model summaries before reporting the revised
reference. The historical comparison reproduces exactly: 3,951 scalar checks,
maximum absolute difference zero.

The preserved ZIP contains the three prediction files, prior comparison and
training audit. The script extracts only its five expected members into a fresh
temporary directory under `tmp/`, checks member paths, and removes the extracted
files afterward. It never loads model weights, performs inference or runs training.

The `--source-archive`, `--source-helper` and `--output-directory` options accept
paths relative to `--root`. To generate a separate copy, select a fresh output
directory and omit `--verify-only`. Existing outputs must match exactly; the
script does not overwrite different historical evidence.

## Evidence

- `pr66-evaluation-receipts.zip`: immutable source artifact from GitHub Actions
  run 43717526943, artifact 14162004655. Its recorded GitHub expiry is
  15 December 2026; the preserved local ZIP removes that retention dependency.
- `measure_accepted_fronts_pr64.py`: unchanged measurement helper extracted from
  PR66 commit `d71303a2a3e2c6abed278fe91ba0f0a910fea63d`.
- `sequence-reference-replay.json`: aggregates, per-image counts, visibility-group
  counts, matching traces, historical reconciliation and provenance.
- `affected-row-status.json`: revised/added row association status, matched
  prediction identities and best box IoUs. New010 row33 is the upper added front;
  row34 is the lower added front.
- `source-inventory.json`: archive member digests, source/output hashes, source
  revisions, artifact retention metadata and reproduction instructions.

Source: [PR66 measurement artifact](https://github.com/Siuuuers/dl/actions/runs/43717526943/artifacts/14162004655).
