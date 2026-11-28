# Reconstructing the R69 result archive

The complete notebook result is stored as five exact binary parts because the GitHub connector accepts requests below 16 MiB. Each full part is 8 MiB before base64 encoding. These are slices of the original ZIP, not independently compressed archives. The standalone ZIP remains in the actual execution workspace; it is omitted from the repository to avoid storing the same large payload twice.

[repository-storage.json](repository-storage.json) lists the parts in reconstruction order, with their byte counts, SHA256 hashes and Git blob identities. Concatenation was verified byte for byte against the completed notebook package. [Independent storage verification](repository-storage-verification.json) confirms the part identities, reconstruction, archive CRC and all 35 member hashes. The package, execution receipt, computed outputs and original inventory are unchanged.

| Original completed artifact | Bytes | SHA256 |
|---|---:|---|
| `baomai-current-proposals-r69-results.zip` | 39,331,256 | `cb6a80abb90d792cb241beb93c13312f3bdcbab9c5631c831523df8cce96ca48` |
| ZIP member `decoded-proposals.npz` | 29,296,868 | `c26ccba90d28a6dab9c2f6bd1049b19374d0ddbf60594377ab9e81332454e2b6` |

The ZIP contains all 35 original members, including the complete raw proposals and the separate corner-suppression results. The adjacent [inventory](inventory.json) describes the original complete run, including its archived NPZ member; it is not an inventory of the multipart repository layout.

Run this from the repository root to verify every part, reconstruct the exact completed ZIP and extract the raw tensor archive:

```python
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

evidence = Path("docs/thesis/evidence/current-proposals-r69")
manifest = json.loads((evidence / "repository-storage.json").read_text())
destination = Path("r69-extracted")
destination.mkdir(exist_ok=True)
archive = destination / manifest["original_archive"]["filename"]

with archive.open("wb") as output:
    for part in manifest["parts"]:
        data = (evidence / part["path"]).read_bytes()
        assert len(data) == part["bytes"]
        assert sha256(data).hexdigest() == part["sha256"]
        output.write(data)

completed = archive.read_bytes()
assert len(completed) == manifest["original_archive"]["bytes"]
assert sha256(completed).hexdigest() == manifest["original_archive"]["sha256"]

member = manifest["decoded_proposals_member"]
with ZipFile(archive) as package:
    assert package.testzip() is None
    raw = package.read(member["name"])
assert len(raw) == member["bytes"]
assert sha256(raw).hexdigest() == member["sha256"]
(destination / member["name"]).write_bytes(raw)
```

This uses only Python's standard library and performs no model inference. Loading the resulting NPZ for analysis requires NumPy. The [evidence guide](README.md), [execution receipt](notebook-execution.json) and [independent verification](independent-verification.json) describe how the unchanged package was produced and checked.
