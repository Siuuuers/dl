# Publication review

The [numerical and causal report review](report-review.json) checks the final report and condensed JSON against independently scored evidence. All **1,456 scalar comparisons agree exactly**, including 36 model/reference/threshold views across the two experiments, common-returned endpoint diagnostics, reference mappings and 130 queue miss flags. The original-pixel interpretations remain a separate recorded review; this verifier does not adjudicate new labels.

The [handoff review](handoff-review.json) checks completed-study status, candidate/default wording, hardware attribution, bounded next steps and preservation of historical methods. Both records identify the exact reviewed files by SHA-256. The report review has no outstanding numerical or wording corrections.

Capsule packaging and replay are verified separately in [verification.json](../verification.json). That fresh extraction reproduces the complete official summary with only the documented timestamp/invocation exclusions and checked root-path normalizations, and regenerates the queue byte-for-byte.

## Repeat the report comparison

First follow the [evidence README](../README.md) to extract the capsule and prepare its clean pinned `dl-replay` source with original images. The report verifier expects this workspace layout:

| Path below the review workspace | Contents |
| --- | --- |
| `dl` | Clean inference/replay source at `e4bb7019bcffd80b08cab28c4877dd562ee1d796`, with original images |
| `seed46-work` | Extracted evidence capsule |
| `dl-confirmation` | Checkout of this publication, containing the report, JSON and published queue |

These can be symlinks to the existing directories; no additional image copies are needed. Add the extracted independent scorer to the Python module path:

```bash
PYTHONPATH=/absolute/path/to/extracted/independent \
  python /absolute/path/to/review_confirmation_report.py \
  --workspace /absolute/path/to/review-workspace \
  --output /absolute/path/to/new-report-review.json
```

The [verifier](review_confirmation_report.py) reads artifacts only. Its generated review records the current paths and time, so compare its findings and input hashes rather than requiring the whole review JSON to be byte-identical. The official summary replay and deterministic queue generation have their stricter equality rules in the evidence README.
