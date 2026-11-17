# PR67 result audit — 18 November 2026

**Recommendation: retain PR35.** The new candidate improves one narrow metric but does not support broad replacement under the fixed comparison. The strongest next step is resolving the explicit partial-front identity/extent disagreements and replaying these saved predictions; another GPU run or a parameter sweep is unnecessary for that step.

## Verified execution

PR67 head `58871404154f061126e5c931dcc2de91fef45f3f` contains a results-only archive. Its SHA-256 is `943da561f1f3175c04846423c5664059c1f5c845125f6ab76bebcfdf3aee52ce`; all 12 indexed member hashes/sizes and both complete 30-epoch receipts were independently checked. Models are YOLO26s two-point pose models, class `cassette_front`, `kpt_shape=[2,3]`.

Training starts each arm independently from epoch173 with seed 46, 30 epochs, batch 4, input size 1280, FP32 and AdamW at 0.0001. Control has 3,771 fronts/6,841 locatable corners; candidate has 3,791/6,859;84 training photos differ. Nine unresolved photos retain exact PR35 labels in both arms. Training validation remains 20 photos/664 original fronts. The complete reviewed dataset is a different artifact and was not trained wholesale.

Fresh inference ran for retained PR35, PR67 control and PR67 candidate using exact evaluator sources from commit `9304d4feb8aac481638ff08829dfb6e86e437c95`, Ultralytics 8.4.182, PyTorch 2.14.1+cpu, Python 3.13.5, NumPy 2.4.6 and OpenCV 5.0.0. All 20 original validation PNG hashes were verified. Thresholds remain box 0.13, endpoint 0.50, NMS 0.70, matching IoU 0.50, image size 1280. Full prediction records are in `eval-pr35`, `eval-reference`, `eval-candidate`. Existing source, labels, selected model, seeds and thresholds were not edited.

The `eval-*` directories preserve the original inference output, including summaries against the original 664-front validation labels. The fixed 666/663-front comparisons reported below are the subsequent saved-prediction scoring in `comparison.json`. The later 662-front exploratory result has its own directory. Compare models within one named reference; do not mix counts from these files.

## Fixed primary reference: 666 fronts

| Model | Matched | Missed | Unmatched outputs | Duplicate subset | Both corners ≤5px | Both ≤10px | Miss-free nonempty scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| PR35 |648|18|80|32|533/584|582/584|6/17|
| PR67 control |646|20|83|29|531/584|583/584|4/17|
| PR67 candidate |641|25|105|28|532/584|584/584|5/17|

All models produce zero outputs on the three empty photos. None has an error-free nonempty scene under the fixed association rule. Candidate minus control is 5 more misses and 22 more unmatched outputs. Candidate minus PR35 is 7 more misses and 25 more unmatched outputs.

**584/584 is confined to the 584 fronts with two locatable reference corners.** It is not 100% detection accuracy, accuracy on all 666 fronts, unseen-scene performance or correct complete-scene output. Relative to the matched control, the single new 10 px success is `new_000087:15`: the same already-visible proposal crosses IoU 0.50 from 0.495163 to 0.505702 with no box edge moving more than 0.59 original pixels.

| Reference corner availability | Objects | PR35 misses | Control misses | Candidate misses |
|---|---:|---:|---:|---:|
| Zero locatable corners |23|10|10|13|
| One locatable corner |59|7|9|12|
| Two locatable corners |584|1|1|0|

The remaining detection difficulty is concentrated in the partial/zero-corner groups. Missing reference corners have no verified position: a returned coordinate on them cannot be called correct or incorrect from this dataset alone.

## Required historical reference: 663 fronts

PR35/control/candidate misses are 15/17/21; unmatched outputs 80/83/104; duplicate subset 32/29/28. Two-corner success counts remain 533/531/532 at 5 px and 582/583/584 at 10 px. This ranking also fails to support broad replacement. The two development references share many inherited labels; agreement between their rankings is not independent physical verification of every boundary.

## What changed in the images

Compared with control, three reference objects gain matches: 009:53,056:14,087:15. Eight lose matches: 008:33,009:56,010:31,010:32,011:61,056:15,090:41,090:42. Every identity, best overlap, full prediction and matching trace is preserved in `comparison.json`.

- **008:33 is a missed visible image-edge front at the fixed threshold.** The nearby candidate proposal follows the next front below.
- **009:53 improves the box for already-proposed partial material.** It is not first discovery of that object.
- **056:14 and 087:15 cross the matching boundary.** The underlying fronts were already proposed by the control.
- **010:31/32, 056:15 and 090:41/42 still have candidate proposals.** Their extents contract toward exposed white material beside the thumb/gripper, while frozen gold boxes cover larger extents. These are extent/ownership disagreements, not proof that the candidate stopped detecting the material. A box may legitimately span an occluder when separated visible pieces belong to the same front. Therefore broad-over-occluder alone does not establish wrong gold or a correct candidate; inspect same-front ownership before revising labels.
- **009:56 and 011:61 involve the gripper aperture.** Their scored loss must be reported separately from unresolved surface/identity questions and the user's later annotation decisions.

The 22 extra-output increase is concentrated in 010(+9),090(+9),089(+7),087(+3),088(+2), and053/055/056/086 (+1 each), offset by 009(−5),011/059(−2 each),007/054/058(−1 each). The mathematical duplicate subset actually falls 29→28; other unmatched outputs rise 54→77. Most additional output count is at low confidence: extras in [0.13, 0.25) rise 29→45. This describes the saved outputs and is not a new threshold search.

One visually clear regression is the new keyboard proposal in 089, candidate prediction 67: box confidence 0.14576 and point confidences 0.877/0.933. It passes the point cutoff despite being on the wrong object. The control has no proposal there. Not all 105 unmatched outputs have been individually adjudicated as background; partial-front fragment boxes and reference mismatches also contribute.

## Reproduce the fixed comparison from saved predictions

Run the following from this repository's root, using a checkout that contains this evidence directory and `scripts/measure_sequence_fronts_pr67.py`. Python 3.11 or newer, Git LFS, `curl`, `sha256sum` and `cmp` are sufficient. This replay uses the Python standard library; it does not require a GPU, PyTorch or Ultralytics and does not load the checkpoint weights.

The 20 original validation PNGs must be materialized, rather than Git LFS pointer files. The original validation labels, model card and both frozen reference directories must also remain intact. The scorer checks their recorded identities. Its three scoring helpers come from source commit `9304d4feb8aac481638ff08829dfb6e86e437c95`; their hashes and the fixed protocol hash are enforced by the script. Do not replace these inputs with current reviewed labels.

Download the two original completed-result archives at their immutable PR heads. Extract each complete archive: the checkpoint manifest is verified together with its neighboring training receipt, staging manifest and training results. The retained PR35 model is the **candidate** arm of the PR35 archive; the PR67 control is its archive's **reference** arm.

```bash
set -e
pr67_evidence="$PWD/results/evaluation/sequence-fronts-pr67-20261118"
pr67_replay="$(mktemp -d "${TMPDIR:-/tmp}/dl-pr67-replay.XXXXXX")"

git lfs pull --include='data/images/val/*.png' --exclude=''

curl --fail --location --output "$pr67_replay/pr35.zip" \
  'https://raw.githubusercontent.com/Siuuuers/dl/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip'
curl --fail --location --output "$pr67_replay/pr67.zip" \
  'https://raw.githubusercontent.com/Siuuuers/dl/58871404154f061126e5c931dcc2de91fef45f3f/results/gpu/baomai-sequence-fronts-l4-001/baomai-sequence-fronts-seed46-v2-results.zip'

sha256sum --check <<EOF
df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91  $pr67_replay/pr35.zip
943da561f1f3175c04846423c5664059c1f5c845125f6ab76bebcfdf3aee52ce  $pr67_replay/pr67.zip
EOF

python -m zipfile -e "$pr67_replay/pr35.zip" "$pr67_replay/pr35"
python -m zipfile -e "$pr67_replay/pr67.zip" "$pr67_replay/pr67"

python -B scripts/measure_sequence_fronts_pr67.py \
  --root . \
  --pr35 "$pr67_evidence/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$pr67_replay/pr35/candidate/checkpoint-manifest.json" \
  --reference "$pr67_evidence/eval-reference/predictions.jsonl" \
  --reference-manifest "$pr67_replay/pr67/reference/checkpoint-manifest.json" \
  --candidate "$pr67_evidence/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$pr67_replay/pr67/candidate/checkpoint-manifest.json" \
  --output "$pr67_replay/comparison.json"

cmp "$pr67_evidence/comparison.json" "$pr67_replay/comparison.json"
```

A successful `cmp` exits with status zero and prints nothing: the replay reproduces the committed JSON byte for byte. The output path must be new. If the verified archives are already extracted, reuse those two extraction directories in the three `--*-manifest` arguments and skip the downloads and extraction. Predictions remain the saved records in this directory; no new inference or parameter search is performed.

`comparison.json` contains both predefined references, all three model summaries, per-image counts, matching traces and gained/lost row identities. The separate [r3 exploratory replay](r3-exploratory/README.md) explains how to reproduce the later recovered-reference diagnostic without replacing this fixed result.

## Minimal next work

1. Keep PR35 selected while preserving both PR67 outcomes and the frozen comparison.
2. Review the named boundary/ownership cases against original pixels and relevant neighboring frames. Use annotation semantics already agreed; do not infer unknown corners.
3. If the reference changes, create a separate version and replay the saved predictions. Keep this preregistered 666/663 comparison intact. A later 662-front r3 reference is a separate exploratory comparison, not a substitute result.
4. Make the single current model/label/reference combination explicit in the project's entry point. Continue checking whole images for missed fronts, not just the model's proposed boxes.
5. Do not turn a previously trained frame into an 'unseen' holdout simply by moving its split. A future grouped evaluation needs a baseline without exposure to held-out frames; any split of this existing video still cannot establish robustness across new camera sessions.

Sources: [PR67](https://github.com/Siuuuers/dl/pull/67), [fixed protocol](https://github.com/Siuuuers/dl/blob/5718111dc1c78a9d8127c88f6a5eb0b39c5cfc47/experiments/sequence-front-corrections-2026-11-18.json), [training guide](https://github.com/Siuuuers/dl/blob/5718111dc1c78a9d8127c88f6a5eb0b39c5cfc47/docs/sequence-front-training.md), and the new local measured evidence in this directory.
