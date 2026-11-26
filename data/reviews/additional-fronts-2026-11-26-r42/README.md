# R42 — full-frame review of the PR127 mining return

**Three complete assistant-reviewed annotation candidates; nine whole images held.**
This review covers all 12 selected original images and all 507 starting rows from
[PR127](https://github.com/Siuuuers/dl/pull/127). It does not train a model, assign
the candidates to a training split, alter the active 137/20 dataset, or create an
independent evaluation set.

The [manifest](manifest.json) records every immutable source identity, original
upstream line, working baseline line, proposed line, exact changed token, reason,
output-row mapping and unresolved field. The working baseline includes the older
quarantined proposals for 0001/0160/0383. New R42 counts are relative to that
working baseline, not a second count of the earlier pilot corrections.

| Image | Baseline rows | R42 replaced rows | Held source rows | Disposition |
| --- | ---: | ---: | ---: | --- |
| frame_0001 |43|1|4|Held|
| frame_0160 |43|5|1|Held|
| frame_0383 |43|4|0|Complete candidate|
| frame_0038 |35|9|4|Held|
| frame_0074 |43|5|2|Held|
| frame_0128 |43|42|1|Held; also delete 1/add 1|
| frame_0146 |43|17|0|Complete candidate|
| frame_0178 |43|12|2|Held|
| frame_0225 |42|28|21|Held|
| frame_0280 |43|24|11|Held|
| frame_0301 |43|18|9|Held|
| frame_0371 |43|1|0|Complete candidate|
| **Total** |**507**|**166**|**55**|**3 complete /9 held**|

There are 340 retained source rows, 166 replacements, one deletion and one addition:
507 final rows. Among retained identities, 63 observed corner coordinate slots
change and six corner-availability slots change. These are overlapping field
counts, not additional objects. Five availability changes remove an unsupported
visible-corner claim; one restores a corner actually visible in 0301. The 0128
addition/deletion has its own identity and is counted separately.

The [independent-review record](independent-review.json) describes the second
assistant checks, accepted challenges, final complete-image label hashes and
held rows. Neither reviewer is a human annotator. Soft contours support nominal
coordinates rather than claims of subpixel physical accuracy.

## View and verify

Download the [review capsule](../../../../results/review/front-mining-r42/review.zip),
extract it, and open `review.html` in a browser. It is self-contained and reads no
network service. Original pixels, working labels, proposed labels and exact
reasons are available for every source row and the added front. The full-frame
view can switch between the upstream source and proposed overlays.

The capsule also contains all 12 original JPEGs, immutable upstream TXT files,
working-baseline TXT files, proposed TXT files, the audit and row verifier.
`complete-candidates/labels/` contains only 0146/0371/0383. It is an annotation
handoff, not a complete training dataset. Held images are never empty negatives.

```bash
python scripts/review_front_mining_r42.py --check-capsule /path/to/review.zip
```

To recreate the deterministic capsule from the pinned PR127 archive:

```bash
python scripts/review_front_mining_r42.py \
  --manifest data/reviews/additional-fronts-2026-11-26-r42/manifest.json \
  --return-archive /path/to/baomai-front-mining-r41-results.zip \
  --audit docs/front-mining-r41-audit.json \
  --independent-review data/reviews/additional-fronts-2026-11-26-r42/independent-review.json \
  --diagnostic docs/front-mining-r42-diagnostic.json \
  --diagnostic-figures docs/images/front-mining-r42 \
  --output /path/to/review.zip
```

The [R42 CPU notebook](../../../../notebooks/baomai_front_review_r42.ipynb)
downloads the capsule by immutable Git blob identity, verifies the archive and
manifest hashes, checks all operations, and displays the same viewer. It needs no
GPU or inference dependencies.

## Preserve the task

The policy remains
[baomai-visible-front-v1](../../../../docs/annotation-policy.md): one identifiable
physical cassette front including its own visible rim, a tight observed envelope,
and independently judged actual lower-front corners. Broad roofs, rack and
gripper do not define the target's boundary. Visible target patches may enclose
an occluder inside their axis-aligned box. A hidden or materially ambiguous
corner is `0 0 0`; a real partly visible front is retained.

For 0128, source row 43 is unsupported below the last lower-left front. The missing
top front is added as `frame_0128:add:lower_left_top:001`. Later source identities
are never shifted to disguise the missing/additional-instance pair. The frame
remains held because source 35's upper boundary is unresolved.

See the [R42 findings](../../../../docs/front-mining-review-r42.md) for the fixed
prediction comparison and next experiment decision. The returned 320-frame pool
has already been mined for development. Distinct file hashes, filename spacing
and assistant annotation review do not establish independent captures or
independent test performance.
