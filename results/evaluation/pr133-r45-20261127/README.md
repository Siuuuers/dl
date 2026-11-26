# PR133 R45 return audit

`cpu-audit.json` is the deterministic output of:

```bash
python3 scripts/audit_additional_front_pilot_r45_results.py \
  --archive /path/to/baomai-additional-front-pilot-r45-rebuilt-results.zip \
  --output results/evaluation/pr133-r45-20261127/cpu-audit.json
```

Use `--download-dir /tmp/baomai-pr133` instead of `--archive` to fetch the pinned PR133 chunks. The audit requires Python's standard library and the repository's hash-pinned scoring helpers. No ML package, GPU, model deserialization, training or fresh inference is needed.

The verified return has 334 inventory entries plus the inventory itself. All forty image-role postclip outputs and twelve reference/view/role reports reproduce exactly. The result records checkpoint hashes and source/runtime consistency, while explicitly separating that from independent prediction generation.

`pdf-verification.json` lists the eighteen case pages, original-image hashes and rendering checks for the separately delivered 21-page results PDF. The PDF is intentionally not committed. Reproduce it after obtaining the return archive:

```bash
python3 scripts/export_additional_front_pilot_r45_pdf.py \
  --archive /tmp/baomai-pr133/baomai-additional-front-pilot-r45-rebuilt-results.zip \
  --images-dir /tmp/baomai-pr133/originals --download-images \
  --output /tmp/baomai-pr133/r45-pilot-results-and-missed-review.pdf
```

Only this PDF command additionally needs Pillow and ReportLab. The nine downloaded originals are checked against the pinned image manifest before rendering. The PDF covers all thirteen R38 candidate misses, its one recovered target, and four other changed identities under frozen666/historical663.

See [the interpretation and decision](../../../docs/additional-front-pilot-r45-results.md). The selected checkpoint is unchanged; automatic promotion is false.
