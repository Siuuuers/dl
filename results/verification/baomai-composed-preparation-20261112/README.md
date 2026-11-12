# Composed comparison preparation evidence

**Preparation passed on the original assets. No GPU trial, full training or new inference was executed.** The [handoff](../../../docs/training-composed-fronts-2026-11-12.md) supplies the next actual GPU run. PR35 remains selected.

## Actual data

- [Asset restoration](assets.json): all 137 training PNGs, 20 validation PNGs and original epoch173 weights match their original SHA-256 identities. PNG integrity and 1280×720 dimensions pass. Materialization preserves the committed LFS pointer contents.
- [Staging and read-only re-verification](staging-run.json): two complete datasets were copied from the actual originals and independently verified. The control has 3,771 objects / 6,842 locatable endpoints / 700 missing slots; the treatment has 3,770 / 6,842 / 698. Both have 137 training images and retain the original validation labels.
- [Notebook input preflight](notebook-inline-preflight.json): the exact embedded Python input program ran with the real protocol and original checkpoint. It separately verified the adjudicated development reference with 663 objects, 1,227 locatable endpoints and 584 eligible pairs.

The staging receipt identifies the executed file bytes. Its workspace paths record where this preparation occurred; users should use fresh paths through the delivered notebook. Dataset staging verifies bytes and geometry, not annotation truth or training success.

## Focused code checks

| Evidence | Scope | Passing tests |
| --- | --- | ---: |
| [Contract checks and final review](contract-checks.json) | Real composed labels, row maps, rehashed undeclared edits, policy snapshots, strict verifier preservation | 14 |
| [Integration checks](integration-checks.json) | New arm/label identity, stager, worker, completed checkpoint and generic collector/evaluator integration | 8 |
| [Return package checks](package-checks.json) | Real frozen protocol plus synthetic archive transport, integrity and role/runtime rejection | 5 |
| [Final pinned notebook check](pinned-handoff.json) | Actual notebook code-cell flow with substituted GPU, network and training operations | 16 |
| Historical checkpoint/trainer/annotation checks in the integration record | Existing behavior and recipe inventories | 60 |
| Unchanged endpoint checks in the integration record | Correct historical-policy input snapshot | 14 |

Synthetic weights/receipts were temporary test fixtures and were never deserialized or treated as trained models. The new tests total 43; historical compatibility tests total 74. The notebook pin check is repeated after the final implementation/protocol pins are installed, because that last edit determines what the user actually runs.

The integration record preserves an initial failure when old endpoint fixtures were given the newer live policy. The same unchanged tests pass with their pinned historical policy. No validator was weakened or historical manifest rewritten to make that mismatch disappear.

## Identities

- Composed manifest: `86522ed6549390e8e6fb147aeb8244c4aef0b2f1273f57cdd0fc07b85074b38a`.
- Exact endpoint control manifest: `9c489bf57a526509de73e40cc73b02117398fba4ec97107bf047aa71e4951c41`.
- Prospective protocol: `2e73d4d81585e8ed1759bdb02aea98d8c9db3a856f8887ef9f41069eb799fa3f`.
- Numerical recipe: `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`.
- Adjudicated evaluation reference: `75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e`.

The original source labels, the accepted endpoint candidate, historical reference labels/replays and live annotation policy remain unchanged. The new composed candidate is a separate version. The changed earlier proposal README contains only a follow-up notice to this implemented preparation.
