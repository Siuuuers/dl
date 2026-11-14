# Remaining training-label decisions

All **137 training IDs have targeted review membership**. That does not mean every row or field has been approved. The consolidated ledger preserves **88 documented open case records across 64 training images**, with exact original row IDs, review crop hints, evidence selectors, source hashes, known facts and the decision still needed. These are documentary case groups, not 88 proven label errors: some contain many rows, some describe a possible omitted front, and some concern the same image from different angles.

The ledger is complete for its enumerated review sources, not a guarantee that there are no unknown defects. It does not change any label.

## What remains

| Decision group | Case records | Images | Practical interpretation |
|---|---:|---:|---|
| Early complete box-pilot ledgers | 3 | 3 | 81 rows retain 163 explicitly unresolved fields: seven identity decisions and 156 box fields. These are actual recorded unresolved statuses, not ordinary placement ranges. |
| Legacy visible-extent tightness | 13 | 12 | Identifiable fronts have unresolved bounds, often extending onto an occluder or neighboring surface. |
| Observed extent / surface boundary | 18 | 11 | A roof/front transition, slanted seam, or same-front ownership must be settled before a precise replacement. |
| Actual lower-front corner junction | 3 | 3 | Specific physical point associations remain questionable; missing slots are not filled from symmetry or a model. |
| Instance identity / possible omitted front | 44 | 34 | Separate cassette versus lip, roof, rack or neighboring front; some regions have no accepted new row. |
| Old extent flags lacking row localization | 7 | 7 | Earlier reports flagged visible extent but did not identify exact rows; localize those issues on full originals before editing. |

Images overlap between groups; their counts must not be added as distinct images.

The early box pilot is especially important. It retained explicit unresolved fields in **new_000030: 22 rows; new_000012: 38 rows; new_000128: 21 rows**. The seven identity rows are `new_000030:12`, `new_000012:23`, and `new_000128:1,6,9,12,19`. The accepted sixteen edge corrections did not settle these other fields. Fresh review still holds `new_000128:1/12`; row12 is the **right-edge strip**, not a bottom fragment.

Other concentrated groups are:

- `new_000018` and `new_000034`: nine held full-box proposals, separate thin-fragment identities, and a small number of explicit point-association concerns. A clear width problem does not automatically approve disputed top/bottom bounds.
- `new_000061/062/063`: visible strips are real targets, but their continuation through the jaws is uncertain. The later accepted selective width repairs do not justify deleting those identities.
- `new_000066`, `new_000115/116/117`, `new_000124/127`: possible merged or additional fronts and unresolved roof/front/occluder ownership.
- `new_000006:53/59`: the current concrete user question—does the small row59 strip represent a separate cassette or a lip/part of row53? Both source rows remain until that physical interpretation is settled. Rows51/52/60 have a separate inherited identity question.
- `new_000014:21/22` and `new_000041:9`: endpoint occlusion is settled, while independent identity/extent questions remain explicit.
- `old_000013/069/079/093/101:1`: proposed top expansions were retracted because the upper material may belong to a preceding lip. The accepted old010 decision cannot be copied by analogy.

## What has already been settled

The ledger retains **13 resolved or superseded records**. In particular, the fresh accepted `new_000038` adjudication supersedes the earlier A-to-B single-front interpretation, the three-front repair hold, and the bottom-clipped front-box hold. It accepts the exact row1 replacement and three appended rows; ordinary manual placement uncertainty is recorded without reopening those accepted operations.

`new_000033:46` left and both slots in `new_000014:21/22` and `new_000041:9` are **resolved keep-missing decisions**. Actual corner occlusion supports `0 0 0`. That decision says nothing new about an independent box/identity question. The three proposed top additions in `new_000015` remain withdrawn after the structural-identity review.

Seven general scope/precision caveats are preserved in a separate **non-blocking** section. They do not become new human-decision blockers simply because an old point was not recertified. Conversely, absence of an open case for an image does not certify every label in that image.

## Validation is separate

The ledger preserves **23 validation-only records** outside the training blockers: the nineteen historical unresolved reference dispositions, the separate `new_000011:61` caveat, and `new_000010:13/14/30`. Later accepted `new_000010` corner adjudications are not reopened merely because reviewers recorded ordinary uncertainty or dissent. These images remain validation/development data and must not be moved into training to resolve the ledger.

No deletion, addition or visible-corner restoration is justified solely by box overlap, thinness, an absent model proposal, or missing endpoint flags. The next work is to settle the listed physical ownership questions and exact supported fields while preserving all other tokens. This ledger establishes review scope; it does not claim a fully corrected release or a model improvement.

[`ledger.json`](ledger.json) is the complete machine-readable record, with one index entry for every training image. [`build_training_review_ledger.py`](../../../../scripts/build_training_review_ledger.py) reproduces it from the recorded sources using repository-relative paths. Suggested crop bounds derived from source boxes are review aids, not accepted annotation geometry.
