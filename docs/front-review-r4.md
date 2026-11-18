# Round 4: labels, focused review and the next comparison

The 31 requested photographs have been reviewed against their original PNGs. The pass contains **521 visual row replacements, three additions and three user-requested deletions**. Two additional inherited PDF-rounding discrepancies receive separately recorded numerical box repairs. The total is **523 replacements, three additions and three deletions across 33 changed photos**.

The one unresolved training item is **new_000014, base row 11**. Its narrow material at the right image boundary cannot be assigned confidently to a cassette rather than neighboring fronts or rack. The supported edits on this photo remain in the review, while the training comparison keeps the entire photograph's exact PR35 labels in both arms.

## Review the changes

The separately delivered `cassette-front-changes-r4-2026-11-18.pdf` is 42 pages. It contains a linked index, every current label on all 33 changed photos, seven enlarged before/current details, and a page explaining the three deletions.

| Mark | Meaning |
| --- | --- |
| Cyan | Inherited row |
| Magenta and an asterisk | Row edited or added in this pass |
| Amber | Unresolved row retained for review |
| Circle / square | Image-left / image-right actual lower-front corner |
| No corner marker | Unavailable point; its numeric triple is `0 0 0` |
| Red dashed box, before panel only | Deleted base row |

Magenta marks the affected row; it does not imply every field in the row changed. All requests and operation keys use immutable **base rows from PR68**. The current-photo pages use **current rows**. The [row maps](../results/review/front-review-20261118-r4/row-maps.json) preserve both identities.

### Count and visibility decisions

- `new_000015`: four visible fronts below base row 56. Base 57 and 58 become current 57 and 58; two additions become current 59 and 60. Former base 59 becomes current 61. Both added fronts have no locatable corners.
- `new_000023`: base row 22 becomes an upper fragment and an added lower fragment, current rows 22 and 23. Former base rows 23–35 become current 24–36. Both fragments retain missing corners. The upper box spans two observed patches separated by gripper occlusion and does not duplicate the neighboring right-stack row.
- `new_000003:15`, `new_000004:15` and `new_000033:4` are deleted by the user's specific adjudications. These are not a general size filter or proof that the visible material is physically absent. Surviving rows renumber normally.
- Requested visible corners are restored in `new_000005:47–49` and `new_000030:21`; unavailable left points in `new_000002:38–39` and right points in `new_000019:58–62` become missing.
- `new_000051:18` and other documented no-change rows were inspected and retained. A request for review does not justify an arbitrary pixel shift.

The [request coverage](../results/review/front-review-20261118-r4/request-coverage.json) accounts for all 37 request groups, including point amendments and every broad/specific box request. `new_00002` is interpreted as `new_000002`. The [operation ledger](../results/review/front-review-20261118-r4/operations.json) records exact before/after lines and changed field groups.

### Two numerical repairs

Inherited PDF reconstruction left the unchanged left point just outside its box in `new_000061:10` and `new_000119:26`. Their left bounds expand by **0.00012096 px** and **0.00011904 px**, respectively. Right, top and bottom bounds and every corner token remain unchanged. These are separately classified serialization repairs, not fresh visual reviews. The existing `1e-4` px geometry tolerance is unchanged. The verifier restricts these repairs to the two named rows and an expansion below `0.001` original pixel.

## Run the prepared notebook

[Open the r4 notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r4-20261118/notebooks/baomai_front_review_r4.ipynb), choose **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. The notebook pins its source commit and protocol hash, verifies inputs, performs a four-batch resource trial, trains missing arms, backs up completed work to Drive, and packages one return ZIP. Return that ZIP for the fixed-reference comparison.

| Item | Reference arm | Candidate arm |
| --- | --- | --- |
| Training photos / empty scenes | 137 / 13 | 137 / 13 |
| Training fronts | 3,771 | 3,789 |
| Locatable / missing corners | 6,841 / 701 | 6,853 / 725 |
| Supervision | Exact PR35 labels | r4 review, with exact PR35 on held new_000014 |
| Initialization | Original epoch173 | Original epoch173, independently |
| Recipe | Seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW, lr 0.0001 | Identical |
| Checkpoint selection | Final epoch | Final epoch |

**94 photos differ between the arms.** This compares the accumulated accepted review, including r3 and r4, against the selected PR35 supervision. It is not a 31-photo-only ablation or a continuation of the PR67 candidate checkpoint. No GPU training is performed by preparing these files.

The [candidate](../data/candidates/front-review-2026-11-18-r4/manifest.json) and [prospective protocol](../experiments/front-review-2026-11-18-r4.json) are separate from the complete review. Both use `full_dataset_certified: false`. Code and staging receipts bind the new verifier and every effective label.

## Keep the four label populations separate

| Population | Photos | Fronts | Role |
| --- | ---: | ---: | --- |
| Original training validation | 20 | 664 | Unchanged validation inside both training runs |
| Frozen sequence reference | 20 | 666 | Primary comparison, same as the previous protocol |
| Frozen historical completion reference | 20 | 663 | Mandatory diagnostic, reported separately |
| Inherited r3/r4 development review | 20 | 662 | Separate exploratory replay only |

The current pass changes no development-reference labels. The 662-front review retains its two inherited amber rows, `new_000007:7,14`; the 666/663 references retain their own eight/thirteen uncertain rows. None is silently substituted for another.

Compare both fresh final checkpoints and retained PR35 with the fixed box threshold `0.13`, endpoint threshold `0.50`, NMS IoU `0.70`, matching IoU `0.50`, and radii `5/10` original pixels. Report misses first, then extra/duplicate detections, empty scenes, corner-availability groups, and both-corner successes with eligible misses retained in denominators. Inspect gained/lost identities. Conflicting reference rankings are inconclusive. Do not search seeds, epochs or thresholds and call the selected result independent validation.

**PR35 remains the selected model.** Corrected labels are a hypothesis about better supervision; only a completed comparison can establish the effect on these scenes. The reused development photos do not establish reliability on an unseen camera session.

## Verification and reproduction

The [verification evidence](../results/verification/front-review-r4-20261118/) records the exact reconstruction and three passing checks: real two-arm staging, rejection of a self-consistent attempt to release the held photo, and rejection of reference substitution/automatic promotion. Actual staging checks all 157 original PNGs, 314 copied images and 314 copied labels. The prior v2 candidate still verifies under its own unchanged protocol.

```bash
python scripts/verify_front_review_r4.py
python scripts/check_front_review_r4.py
python scripts/build_front_review_r4_pdf.py \
  --output /path/to/cassette-front-changes-r4-2026-11-18.pdf \
  --audit-output /path/to/pdf-audit.json
```

The PDF renderer needs Pillow, ReportLab and DejaVu Sans. Its geometry comes from verified labels; it never chooses or adjusts annotations. The recorded PDF hash identifies the delivered file; regenerated PDF metadata may change its bytes while preserving the same reviewed geometry.
