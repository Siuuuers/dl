# PR101: completed R25 model comparison

23 November 2026, Asia/Hong_Kong.

**Retain PR35. R25 does not establish a replacement, and it does not improve on R24 overall.** Both returned R25 training arms are complete and valid. The candidate has slightly worse detection counts than R24 on the same references; its corner results are unchanged or mixed. This evaluation performs fresh inference and saved-prediction scoring only. No training was repeated and no annotation was amended.

## Comparison on identical references

All four checkpoints were freshly inferred in one CPU runtime. Settings remain input 1280, FP32, rectangular preprocessing, box confidence 0.13, NMS IoU 0.70, matching IoU 0.50 and endpoint confidence 0.50. The denominator of 584 includes every reference front with two locatable corners, including eligible misses. All four models produce zero outputs on the three empty images.

| Frozen 666-front primary reference | Matched | Missed | Unmatched outputs | Both corners ≤5 px | Both ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected PR35 | 648 | 18 | 80 | 533/584 | 582/584 |
| R25 matched control | 646 | 20 | 83 | 531/584 | 583/584 |
| R24 candidate | 631 | 35 | 105 | 518/584 | 573/584 |
| R25 candidate | 629 | 37 | 108 | 518/584 | 574/584 |

The separate historical 663-front reference gives PR35/control/R24/R25 miss counts **15/17/31/34** and unmatched-output counts **80/83/104/108**. Its five- and ten-pixel pair totals equal those in the table above. Neither reference supports replacing PR35 with R25.

The next table uses the **same accepted R25 review bytes for every model**, saved before PR101 returned. Its 662 fronts are an exploratory reference, separate from both frozen references and the original 664-front validation used inside training.

| Current R25 exploratory reference | Matched | Missed | Unmatched outputs | Both corners ≤5 px | Both ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR35 | 625 | 37 | 103 | 543/584 | 573/584 |
| R25 matched control | 624 | 38 | 105 | 539/584 | 573/584 |
| R24 candidate | 646 | 16 | 90 | 557/584 | 583/584 |
| R25 candidate | 645 | 17 | 92 | 557/584 | 583/584 |

Relative to R24 on this same current reference, R25 loses one match and adds two unmatched outputs. Miss-free nonempty scenes decrease from 10/17 to 9/17. The five- and ten-pixel successful object identities are exactly unchanged, not merely equal aggregate counts. Duplicates decrease from 33 to 32 while other unmatched outputs increase from 57 to 60. R25 matches every one of the 584 two-locatable-corner targets; its 17 misses comprise seven targets with no locatable corner and ten with one.

The current reference favors the recent candidates over PR35, while the frozen primary favors PR35. This preserves the existing reference-sensitive conclusion: selection under the frozen protocol is not proof that PR35 is physically best under every accepted annotation version. Changing the reference can change a score without changing the model. Do not compare the earlier R24-reference PR35 total of 544 five-pixel pairs directly with this R25-reference total of 543 and attribute the difference to model learning.

The [full comparison](fixed-comparison.json) contains every per-image outcome, matching identity and gained/lost corner identity. The [compact summary](summary.json) keeps all three reference versions explicit.

## What changed in training

The [return audit](return-audit.json) verifies the archive, all 12 indexed payloads, source and notebook pins, protocol, completed receipts, staged inputs, original image hashes and initializer bytes. Each arm completed 30 epochs, 1,050 batches, 4,110 image presentations and 78 optimizer updates on an NVIDIA L4, with exit code zero and final-epoch checkpoint selection.

The [cross-round input audit](crossround-input-verification.json) verifies that R24 and R25 share the same recorded training runtime, settings and epoch173 initializer. Their matched-control state dictionaries have **879/879 identical parameter/buffer tensors**. The R25 candidate differs from R24 only in **four left-corner locations in one training file, `new_000012`, rows 1, 2, 3 and 28**. Images, boxes, classes, visibility, object counts and original trainer validation are unchanged. Corrections in development photos 008 and 010 do not enter training.

Thus R25 versus R24 tests this four-point training delta under the recorded seed/runtime. R25 versus its PR35-label control instead tests the accumulated eligible review, which differs in 92 training label files. Neither comparison establishes performance on an independent camera session. The supported annotation corrections remain accepted; a model regression does not by itself invalidate the image evidence for a label.

## Concrete remaining cases

On the current R25 reference, the sole R24-to-R25 lost match is **`new_000089:49`**. R24's retained matching box has confidence 0.144224 and IoU 0.710913. R25 still has a retained overlapping alternative at confidence 0.338856, but its best retained IoU is 0.445040, below the fixed 0.50 matching requirement. The official miss is preserved. These filtered outputs do not establish whether another raw proposal was confidence-filtered, suppressed or never generated.

The frozen reference additionally loses **`089:43`**, whose best retained IoU moves from 0.500023 to 0.486658. Its confidence remains high, 0.545116 to 0.554145. This is an overlap-boundary loss rather than evidence that confidence tuning would repair it. The frozen ten-pixel pair gain is `008:20`; no five-pixel pair identity changes. On the current reference there are no pair-success gains or losses.

All 17 current-reference misses are:

| Photograph | R25 review rows |
| --- | --- |
| `new_000008` | 33, 34, 36 |
| `new_000009` | 55, 56, 57 |
| `new_000010` | 14, 33, 34 |
| `new_000011` | 55, 56, 57, 60 |
| `new_000053` | 1 |
| `new_000086` | 57 |
| `new_000088` | 57 |
| `new_000089` | 49 |

These are scored target identities, not proposed label changes. The prior [R24 partial-front diagnosis](../pr99-r24-20261123/partial-front-failure-queue.md) explains the other 16 cases; this report does not claim to repeat its original-pixel inspection for the new R25 predictions.

The evidence continues to favor a bounded review of analogous eligible training examples for visible box extent and complete partial-front coverage. Development-only corrections improve the reference but do not supply new training information. A further training experiment should follow supported changes to training supervision or another specific mechanism; repeating the unchanged run would not test a new hypothesis. These results do not establish a need for a different architecture, learning rate or general threshold sweep.

## Provenance and replay

- [PR101](https://github.com/Siuuuers/dl/pull/101) head: `606655962be6716b19ec29285192c86c5aca03b6`.
- Archive: 39,783,961 bytes; SHA-256 `8cb47ae13ad14e325b540c19ccd6fccb3198b85406f3238a581bd2db5b9e3762`.
- Training source: `86a7ba078b348f2b1cbacf73b4c271e9216fdb63`.
- Notebook: `ace99d4aa71e5c20d7c5d2445f18fdab58d63496`, SHA-256 `11d0cac7a492f7234e4778490c1ef70d445d5aa309f2bca11af4d02329553877`.
- R25 control weight SHA-256: `561665025b30718916e838e6d829e050d0fae622498a44733c1dc52b49bf05df`.
- R25 candidate weight SHA-256: `6ada96ede2a33604fcb6d5b42322798c8055d3c3298f36de4a3b66a15e827e9c`.

Fresh inference covers four models ×20 original images, plus one excluded warmup per model. Runtime is Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93, with four CPU threads. The exact headers and prediction hashes are in [inference-provenance.json](inference-provenance.json); commands are in [inference-commands.json](inference-commands.json). Raw evaluator summary files score the original 664-front labels; use `fixed-comparison.json` for the three post-training references above.

For the retained model's visual report, use [the fresh PR35 prediction records](inference/eval-pr35/predictions.jsonl). Checkpoint receipts, original logs preserved as `train-log.txt`, CSVs and predictions are included. Weights, archives and full camera images remain external inputs.

Replay scoring without weights, inference or training:

```bash
python3 results/evaluation/pr101-r25-20261123/replay_evidence.py \
  --root /path/to/dl \
  --output /tmp/pr101-replay-verification.json
```

The checkout must contain the pinned scorer/reference bytes and hash-matching original validation PNGs. To rerun the return audit after shared training scripts change, use the immutable training source with the audit adapter supplied here; its byte checks intentionally reject a different training implementation. The adapter patches record the limited changes from PR99's existing workflow.
