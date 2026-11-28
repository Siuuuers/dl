# Independent replay of the returned QFL experiment

Source: `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`.
Return: PR157, `f971e880e8aff2859ce4d8ae2923efea826eddb4`.
Evidence ZIP SHA-256:
`cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a`.

The complete archive remains in that return commit at
`results/a100/baomai-quality-focal-r61-a100-001/baomai-quality-focal-r61-results.zip`.
This directory stores compact derived evidence rather than another checkpoint
or evidence-archive copy.

After obtaining and extracting the hash-verified archive, run:

```bash
python docs/thesis/evidence/quality-focal-return/replay.py \
  --repo /path/to/dl \
  --pack /path/to/extracted-evidence \
  --archive /path/to/baomai-quality-focal-r61-results.zip \
  --output /path/to/replayed-evidence
```

The repository must contain the published source commit. The replay verifies
every inventory member, all 95 source Git identities, all 60 reference-label Git
identities, staged label hashes, paired receipts and controller traces. It imports
the archive's exact evaluator, including the explicit PR156 handling of clipped
nonpositive boxes, and reproduces all 18 metric groups, all nine AP50 rankings,
and the complete prediction/association/case details exactly. No inference,
training, model promotion or threshold search occurs.

`verification.json` records what this script verifies. It does not independently
reconstruct checkpoint bytes or rerun the saved gradient guard. The guard's frozen
receipt verifier was also run separately and passed. A separate [checkpoint-byte receipt](checkpoint-verification.json) verifies all four committed wrappers, their member sizes/hashes and concatenation order, and both complete model hashes. It makes no inference claim.

`metrics.csv` contains every reference/model/view result. `paired-cases.json`
retains detection and corner gains/losses. `ranking.json` gives the prespecified
low-floor metric without its large curve arrays. `score-distributions.json`
records descriptive score pools, clipped boxes and empty-scene outputs.
`emitted-pair-support.json` distinguishes endpoint-valid output pairs from pairs
that the complete reference can verify. The source evaluator uses strict
in-image bounds and endpoint confidence at least 0.50.

`training-diagnostics.json` and `training-by-epoch.csv` summarize the actual
recorded assignment and loss values. Native/QFL target means are weighted by
foreground-anchor count. Each loss numerator is divided by that batch's original
normalizer, multiplied by its classification gain and branch schedule, then
summed; the batch-size factor is excluded. These scalar loss contributions are
not parameter-gradient contributions. The native BCE in a candidate trace is
evaluated on candidate predictions, not on the control model's predictions.

All outputs are development evidence. The replay cannot establish physical
annotation correctness or independent-session generalization.
