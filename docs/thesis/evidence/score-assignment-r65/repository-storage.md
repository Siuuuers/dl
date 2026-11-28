# Repository storage and exact restoration

The completed `results/targets.json.gz` is 31,771,125 bytes. The GitHub connector
rejects request bodies above 16 MiB, while its base64 upload would require
42,361,500 encoded bytes. Repository storage therefore uses four ordered
`targets.json.gz.part00` through `targets.json.gz.part03` files, each at most
8 MiB. These are consecutive pieces of the **exact original gzip bytes**; they
are not a recompression, reduced dataset or new diagnostic run.

From the repository root, restore the logical evidence file with:

```sh
python docs/thesis/evidence/score-assignment-r65/results/restore_targets.py
```

The standard-library script verifies the size and SHA-256 of every ordered part,
verifies the assembled bytes, and creates `targets.json.gz` without replacing an
existing different file. If the correct file already exists, it verifies the
parts and existing file and leaves that file unchanged. The storage map is
[results/targets.parts.json](results/targets.parts.json). The restored file is
31,771,125 bytes with SHA-256
`883d5a21dbcd3041916267bdfcc6b0332868ca080a25349ce8e785e6b0f2c620` and Git blob SHA-1
`b6eb84734254e2de31e3ac840e2c924da5d4b04f`.

The original README, execution receipt, analytic manifest, package manifest
and `inventory.json` remain byte-for-byte unchanged. Verify all original
inventory paths after restoration; the reassembled gzip matches its original
size and hash directly. The parts, storage map, restoration script and this
guide are publication assets added after execution, outside the original
inventory. They are not claimed as original notebook outputs. This storage
change preserves every experimental byte and does not change any result.
