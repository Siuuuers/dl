# PR137 return audit and R48 error review evidence

This directory contains a read-only CPU audit of the completed R47 T4 return
in [PR137](https://github.com/Siuuuers/dl/pull/137) at
`dcb43d7ad061fc3a43f1392398bb8794d9d7168d`. It does not contain a new GPU
training run or new inference. Current R38 and historical reference labels
are unchanged. PR35 remains selected.

| File | Purpose |
| --- | --- |
| `return-verification.json` | Exact source, archive inventory, labels, completed exposure/runtime receipts, 40 filter replays and 12 scoring reports |
| `prior-reference-identity.json` | Exact parsed prediction identity: PR133 candidate equals PR137 reference, all 20 photographs and raw/postclip/filter views |
| `error-index.json` | All 119 current-R38 candidate cases, exact output/target identities, coordinates, image hashes, page membership and diagnostics |
| `verification.json` | PDF, index and source hashes; complete case coverage; render/layout QA scope |

The result archive is 42,395,650 bytes with SHA-256
`fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`.
The replay uses 654 current-R38 targets on 20 development photographs:
misses 13 to 15; unmatched outputs 90 to 84; formal duplicates 18 to 14;
two-corner 5px successes 569 to 566 out of 583. Frozen666 and
historical663 reports are kept separate within the receipt.

The index contains 15 misses, 20 matched corner failures and 84 unmatched
outputs (14 formal duplicates, 21 near-gate, 37 lower-positive-overlap and
12 zero-overlap). There are no recovered targets or matches in the diagnostic
IoU interval `[0.50, 0.55)`. The 46-page PDF's first five pages explain the
training result, definitions, per-photograph metrics, visibility/height
strata and post-hoc confidence trade-offs.

New PDF `P###` identities are PR137 candidate postclip positions. The user's
old amendment cues are PR133 candidate positions, which bind exactly to
PR137 reference predictions; these namespaces must remain explicit. `R###`
always names a one-based current-R38 development label row in this report.

See [the result analysis](../../../docs/additional-front-pilot-r47-results.md)
and [the error review and reproduction commands](../../../docs/front-error-review-r48.md).
The reproducible renderer writes `output/pdf/r48-missed-errors-unmatched.pdf`
and verifies all original-image hashes. Training evidence is the returned
run's archived receipts and saved predictions; this CPU review does not
independently regenerate those predictions from checkpoint weights.
