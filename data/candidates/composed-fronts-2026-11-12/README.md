# Composed visible-front training candidate

This candidate makes one controlled training comparison runnable: hold the
accepted ten-slot endpoint supervision fixed, then add the previously recorded
seventeen PR49 bounding-box corrections and one duplicate deletion. It is a
composition of existing decisions, not a new image review or a measured model
improvement.

## Exact arms

| Training arm | Complete overlay | Images | Objects | Visible endpoints | Missing endpoints | Empty images |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Control | Current `endpoint-visibility-2026-11-12`, ten files | 137 | 3,771 | 6,842 | 700 | 13 |
| Treatment | This `composed-fronts-2026-11-12`, eleven files | 137 | 3,770 | 6,842 | 698 | 13 |

The control manifest SHA-256 is
`9c489bf57a526509de73e40cc73b02117398fba4ec97107bf047aa71e4951c41`.
Its ten changed endpoint slots include the current `new_000023` original row33
right corner. The older nine-slot endpoint result is not substituted.

The treatment contains the exact eight PR35 parent overlay files and complete
composed label files for `new_000023`, `new_000031`, and `new_000045`. It adds the
seventeen recorded box changes to that control, deletes only `new_000023`
original source row22, and retains original source row36 as candidate row35.
Original rows23–36 shift down by one; original row33 becomes candidate row32.
Every retained class and endpoint token is identical to the corresponding
control row. The fourteen different PR47 box edits are outside this candidate.

Both arms retain all twenty original validation labels byte-for-byte: 664
objects, 1,227 visible endpoints, 101 missing endpoints, and three empty images.
The separately versioned adjudicated evaluation reference is not a training
input and must not replace these validation-label bytes.

## Frozen decisions and mechanical verification

`review.json` binds the operations and complete source-to-candidate row maps to
the frozen compatibility specification and recovered correction record. The
manifest exposes eleven complete `changes` records in the existing overlay
shape and a separate `control` descriptor. It is self-contained as an overlay
on the original source labels; an older historical candidate is not merged
implicitly at staging time.

Run the label contract without images or a GPU:

```bash
python scripts/verify_pose_composed_candidate.py \
  --candidate data/candidates/composed-fronts-2026-11-12/manifest.json
python scripts/check_pose_composed_candidate.py
```

The new kind is `training_composed_overlay`; its paired recipe is
`reviewed-composed-seed46-v1`. The verifier reconstructs all bytes independently
from the frozen inputs, verifies the box-only recovery payload as well as the
composed result, checks all 157 effective label geometries and counts, and
rejects extra files, changed row identities, unrecorded operations, or altered
endpoint supervision. Its receipt separately identifies the control and its
counts. Successful label verification does not certify image payload readiness
or launch training.

## Historical policy compatibility

The current policy remains `docs/annotation-policy.md`, SHA-256
`74dd943976a4abc22ecbf73551c69f715666060d5961b2bed81e6a9c73a3afc9`.
The immutable snapshot
`docs/policies/annotation-policy-pr35-2026-11-10.md` preserves historical policy
SHA-256 `3cda361ac4c5c9ba36b037e7e8968e0629eb3bca8d489c187beb8dd66e208a91`.

Verification checks that the current document contains those historical bytes
unchanged and appends only its pinned section8. That section expressly retains
sections1–7 for observed reference/current training labels and approves no
inferred training point. Both hashes and the appended-byte hash are pinned.

The unmodified endpoint validator invokes the unmodified PR35 strict validator
in an isolated temporary snapshot containing all 157 original labels, the exact
parent/control manifests, labels and evidence, and the historical policy at
its original expected path. The live policy and historical manifests are never
rewritten. The five existing strict validation/parsing modules are hash-pinned;
the checkpoint module is copied only as the endpoint recipe-ID support
dependency, and the historical endpoint recipe ID is checked in its receipt.

No inferred coordinate, new image adjudication, model inference, or training is
performed by this contract. Actual image restoration, checkpoint verification,
paired staging, and the fixed training run remain separate readiness steps.
