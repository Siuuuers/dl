# Physical adjudication — 2026-11-14

[decisions.json](decisions.json) records the user's four physical clarifications
and the corresponding review decisions. It is an additive record: the source
full candidate, review status, labels and full-review gate are preserved.
The JSON pins the source manifest, source review status and original image bytes
with SHA-256 hashes. Coordinates refer to the original 1280 × 720 images.

## Accepted decisions

| Case | Decision | Evidence |
| --- | --- | --- |
| `new_000037-100` | Broad upper assembly is non-target; add no cassette. | The user identifies this exact assembly as wall. |
| `new_000017-075` | Broad upper assembly is non-target; add no cassette. | Independent original-image review identifies continuous fixture/backing geometry across several cassette columns. This is a visual inference, not a user-confirmed wall identification. |

Neither decision changes any label row. Neither certifies all annotations in an
image or establishes model readiness.

## Clarifications with no label changes

The user identifies the material in `new_000030`'s gripper opening as shadowed
cassette material. Its target-face ownership and front count remain unresolved.
The other reviewed aperture questions and thin/clipped bands also remain open.
For `new_000116` row24 and `new_000123` row24, the current boxes include more than
the disputed narrow material, so they must not be described as reviewed tight
observed-front boxes.

No white-shoulder requirement, brightness cutoff, minimum height/area or
in-rack-only rule is adopted. Neighboring size and position constrain hypotheses;
they do not establish invisible seams or corners. All missing-corner states
remain unchanged. The record's `label_operations` is empty.

A controlled incremental training experiment may consume these two closures
and preserve the remaining uncertainty under its own explicit scope. This record
does not bypass or rewrite the existing full-review release gate.
