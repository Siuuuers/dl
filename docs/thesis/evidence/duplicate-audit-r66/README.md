# Fixed duplicate and unmatched-output probes

This completed CPU analysis uses saved predictions from the historical retained
method, both PR157 classification-loss arms and both PR158 training-label arms.
It changes no annotations, model weights, object-confidence threshold or primary
evaluation matcher. The same 20 development photographs are scored against the
reviewed, untouched supplied and earlier fixed references.

The source driver is
[`analyze_duplicate_audit_r66.py`](../../../../scripts/analyze_duplicate_audit_r66.py).
The four-cell notebook is
[`baomai_duplicate_audit_r66.ipynb`](../../../../notebooks/baomai_duplicate_audit_r66.ipynb).
All four actual code cells completed unchanged from published implementation
`9d457b56ee43a3514bd17a0e9070a5a243d69775`. The
[execution receipt](notebook-execution.json),
[executed notebook](executed-notebook.ipynb), and
[package identity](package-manifest.json) record the completed run. Saved analysis
bytes reproduced exactly during the fourth cell. Execution used one local
IPython namespace, with a clean source checkout and hash-verified inputs already
available locally; GitHub acquisition and the hosted Colab download interface
were not exercised.

## Fixed interventions

Each of three probes starts from the unchanged completed two-view output:

1. Require an existing mutual-best reflected-view box association at the
   original association IoU threshold of at least 0.50. A usable corner pair is
   not an extra condition for this box-support gate.
2. Add confidence-ordered output suppression at prediction-pair IoU strictly
   above 0.60.
3. Add the same output suppression strictly above 0.50.

The two suppression probes operate after the already completed native 0.70 NMS,
clipping, 0.70 postclip suppression and two-view corner averaging. They are not
replacements for native NMS. No previously discarded proposal can be recovered.
Equal confidence selects the earlier original output index. The retained
records preserve every coordinate and confidence value. Cross-view associations
and averaged coordinates are not recomputed. Evaluation matching is recomputed
for each retained subset, so a reference can be matched to a different retained
prediction. The three probes are not combined.

## Result that matters

For the PR158 reviewed-label arm, unchanged results are 15 misses, 51 extras and
13 duplicates. Requiring mutual view support gives 19/27/6. Extra suppression at
0.60 gives 16/32/2; at 0.50 it gives 19/23/1. All four retain 572 of 583 complete
pairs within five pixels and 583 within ten. The removed matches have incomplete
reference corners. The 0.60 probe loses `r50/new_000056/R014`, with one known
corner. Thus unchanged complete-pair scores conceal detection losses
in partial fronts. No probe is promoted or presented as an overall improvement.

The maximum-cardinality one-to-one matching count equals the existing greedy
count in every image, method, reference and probe in this audit. This diagnostic
uses reference geometry and does not replace the published scorer or provide
an inference rule. It rules out a cardinality gain from changing matching on
these particular retained output sets.

The input archive hashes, every inventoried member, identical reference-label
bytes and complete saved-output replay are checked. The baseline verification
reproduces 840 archived per-image score objects, including methods/views outside
the five two-view methods subjected to the probes. No model or photograph bytes
are loaded. Archive identity and replay do not establish physical label truth.

## Evidence

| File | Contents |
| --- | --- |
| `summary.json` | All 60 method/reference/probe rows; zero-, one- and two-known-corner strata; method, reference, input and source identities. |
| `matrix.csv` | Flat comparison table including all-eligible five/ten-pixel results and diagnostic matching bounds. |
| `paired-changes.json.gz` | Every changed match assignment or detection/corner outcome, with version-qualified reference IDs and original prediction identities. |
| `removed-outputs.json.gz` | Each removed output, its baseline evaluation status, source coordinates, view link and suppression competitor when present. |
| `baseline-cases.json.gz` | Baseline misses/extras with coordinates and geometric overlap cues, without physical-negative verdicts. |
| `matching-bounds.json.gz` | Per-image greedy counts and maximum-cardinality bounds. |
| `selected-original-indices.json.gz` | Exact surviving indices in each original completed output list. |
| `inventory.json` | Byte size and SHA-256 of every generated analysis member. |
| `results.md` | Readable reviewed-reference table and interpretation limits. |

`P001` identities are one-based positions in a named method's original completed
two-view output list. They are stable across its probes, but are not identities
across models or photographs. `R001` identifies a row in the named reference
version and image; equal row numbers across references need not denote the same
physical front. Historical `primary_prediction_id` values remain inside each
saved prediction record.

In the baseline ledger, `nearest_reference_id` denotes the reference with
greatest positive box IoU, with row order breaking ties; it is null when every
overlap is zero. It is an overlap cue, not a spatial-nearest or physical-identity
claim.

## Reproduction

Stage the three archives under an input directory using these filenames:
`historical-r60.zip`, `quality-focal-pr157.zip`, and
`training-labels-pr158.zip`. The driver constants pin all byte hashes, sizes,
return commits and repository paths. The notebook downloads missing archives
at those commits; pre-existing input files are accepted only after the same
checks.

```bash
python -B scripts/analyze_duplicate_audit_r66.py \
  --inputs /path/to/verified-inputs \
  --output /path/to/replayed-results
```

Add `--stage-inputs` to obtain missing archives. No third-party inference or
training package is needed for the analysis. The notebook also repeats the
calculation, compares all saved analysis bytes and creates a verified result ZIP.

## Limits

Extras include evaluation duplicates and other unmatched outputs. The reference
contains documented historical exclusions of visible partial fronts; there is
no implemented ignore-region mechanism. An extra must not become a negative
training example merely because it is unmatched. The reflected image supplies
another network response to the same photograph, not independent scene evidence.
The probes reuse development data and make no independent-session or robotic
positioning claim.
