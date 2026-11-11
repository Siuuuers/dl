# Partial-front training annotations

**Status: proposed cumulative annotation candidate.** This extends [annotations-2026-11-10](../annotations-2026-11-10/README.md) with five independently reviewed omitted fronts, one duplicate consolidation and one gripper-only deletion. All eight inherited label files, their edit records and accepted decisions are preserved. The [manifest](manifest.json) reconstructs all 13 changed training files from the original labels; the [review](review.json) distinguishes inherited evidence from this new review.

## New changes

Rows below refer to the original label file unless explicitly identified as candidate rows. Boxes use original 1280 × 720 pixels in `[left, top, right, bottom]` order.

| Image | Accepted change | Final box or point handling |
| --- | --- | --- |
| `new_000119` | Retain row 34, consolidate redundant row 36 into it, then delete row 36 | `[688, 443, 739, 467]`; transfer row 36's exact visible left-corner tokens to row 34; right corner stays missing |
| `new_000120` | Delete gripper-only source row 56; append an independent omitted front as candidate row 56 | `[576, 526, 606, 553]`; both corners missing |
| `new_000122` | Append candidate row 57 | `[676, 475, 727, 516]`; both corners missing |
| `new_000125` | Append candidate rows 53 and 54 | `[674, 460, 738, 481]` and `[677, 482, 734, 500]`; both corners missing on each |
| `new_000130` | Append candidate row 59 | `[699, 503, 726, 520]`; both corners missing |

The new review covers **12 training images and 472 original rows**: 111, 114, 115, 116, 117, 119, 120, 122, 124, 125, 127 and 130. Original PNGs were inspected before label overlays, and accepted geometry received an independent root review. These integer bounds were frozen from the original pixels before new training or scoring. No new visible corner was invented; all five added fronts have two canonical missing point slots.

No edit was accepted for 111 or 114. Identity questions in 115, 116, 117, 124 and 127 remain recorded and unchanged. Existing loose extents in 130 also remain flagged. This is a bounded correction, not certification of every row or a completed migration to the visible-front policy. The inherited eight files were reviewed previously and are not newly certified here.

| Dataset state | Training objects | Visible endpoints | Missing endpoints |
| --- | ---: | ---: | ---: |
| Original frozen labels | 3,772 | 6,839 | 705 |
| Inherited eight-file candidate | 3,771 | 6,841 | 701 |
| This cumulative 13-file candidate | **3,774** | **6,841** | **707** |

Relative to the inherited candidate, this adds five rows, deletes two rows, amends one retained row and transfers one existing endpoint exactly. Across the cumulative candidate, there are ten additions, eight deletions, four retained-row amendments and three exact endpoint transfers. All 137 training images and 13 empty training images remain. The original 20-image validation split remains byte-identical at 664 objects, 1,227 visible endpoints, 101 missing endpoints and three empty scenes. Separate versioned evaluation references are not part of this training overlay.

## Verify

```bash
python scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/partial-fronts-2026-11-11/manifest.json
```

The existing verifier checks source and candidate label hashes, every original-row mapping, explicit accepted operations, exact donor transfers, visibility and geometry. It does not adjudicate image identity or demonstrate model improvement. The review records source-image hashes, inherited provenance, peer-review file hashes, final decisions and unresolved observations. All original labels, images and the inherited candidate remain unchanged.
