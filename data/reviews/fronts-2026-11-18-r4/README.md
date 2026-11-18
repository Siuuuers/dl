# Round 4 cassette-front review

This version applies the latest user-requested original-image review to the immutable PR68/r3 baseline. It includes **521 visual row replacements, 3 additions, 3 deletions**, plus **2 narrowly bounded serialization repairs** inherited from PDF recovery. All 157 label files are present; only 33 training photos change. The 31 requested photos received fresh visual review; the two numerical-only photos did not.

| Split | Images | Fronts | Locatable corners | Missing corners | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training review | 137 | 3,789 | 6,853 | 725 | 13 |
| Separate development review | 20 | 662 | 1,226 | 98 | 3 |

The [manifest](manifest.json) pins the parent, operation ledger, row maps, request coverage, case ledger, policy and user reference images. `new_000014` base row 11 remains unresolved. Development rows `new_000007:7,14` retain inherited amber status. Other unmodified labels inherit the previous review and its limits; absence of a hold is not full physical certification.

One row remains one cassette front with two lower-front corner slots:

```text
0 cx cy width height left_x left_y left_v right_x right_y right_v
```

Coordinates are normalized; `kpt_shape: [2,3]`, `flip_idx: [1,0]`. A locatable corner uses `v=2`; an unavailable corner uses numeric `0 0 0`. Boxes include the front's own visible rim/lip. A box may span an occluder between separate visible patches belonging to the same front. No entirely hidden front or hidden corner is inferred from spacing.

The [separate training candidate](../../candidates/front-review-2026-11-18-r4/manifest.json) retains exact PR35 labels for the entire held new_000014 photo in both arms. It stages the original 664-front training validation, not this directory's 662-front development review. The full review remains uncertified and is not a model result.

See the [focused review and notebook guide](../../../docs/front-review-r4.md). Reconstruct and verify every label against the immutable row operations with `python scripts/verify_front_review_r4.py` from the repository root.
