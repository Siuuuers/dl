# Video-continuity label preview: review before merge

**This is a separate DRAFT proposal stacked on PR #63.** It does not change the frozen PR63 review candidate, PR35 checkpoint, or the pinned accepted-fronts notebook.

## PDF review first

The companion combined review PDF has **175 pages**: original atlas pages 1-166 unchanged, then a 9-page video comparison appendix (full-PDF pages 167-175). The slim 9-page appendix is also available in the conversation. Exact SHA-256 digests are recorded in [manifest.json](manifest.json).

Most relevant appendix pages: **4** for the new116 deletion; **7** for the two new128 proposed fronts; **5** for new122-124 owner questions. The PDF labels the changes *PROPOSED* until physical verification.

## Two proposed TXT changes

- **new_000116:** remove old row **24** (a 142x35 px box spanning gripper/hand, both corners missing). All other rows are preserved verbatim and later rows shift their position in the TXT file. Do not apply this deletion to new117.
- **new_000128:** add two new rows **37** and **38** from the apparent continued regions of new127 rows37/38. Proposed observed boxes in original 1280x720 coordinates: **[675,458,744,500]** and **[697,504,724,530]**. Both lower corners remain `0 0 0` until actually observable. Existing row18 is separate and lies above these proposals.

The change preview would adjust the **full 3,791-object review candidate to 3,792 objects**, a net +1. It is **not** the previous limited 3,788-object training treatment, which remains exactly pinned.

## Open review holds

- new030 row12 shoulder ↔ upper diamond patch association suggests top y443 instead of 455.72, but lower triangle owner and full bounds are not certified.
- new041/042/044/049/062/074 and new123/124 have useful cross-frame tier clues, but no automatic count/box change is justified without identifiable front-plane seams.
- The top-clipped bands in new128 (rows1,6,9) remain open; size alone is no deletion rule.
- The apparent pair of new128 surfaces could also include gripper-interior material. Neighboring-frame position and high image correlation support continuity, but only the actual front-plane identification justifies final labels. This draft invites that physical check.

## Baseline and upstream

The branch `an94-doll/embedding_vision:baomai` and the earliest imported version of `Siuuuers/dl` both point to original SHA `351eb3cf960ccc34829e2bc72e1fa4645acc6295` (2026-10-28). No new upstream baomai revision exists to bring forward as of this review.

## Acceptance gate

Review the PDF first. After physical confirmation, validate all labels in a **separate versioned candidate**, verify counts and corner visibility, and only then consider merging the draft or preparing a newly pinned training experiment. This PR itself performs **no training and no model promotion**.
