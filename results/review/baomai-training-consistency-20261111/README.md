# Six-image consistency review evidence

**Completed original-image review, with no inference or training.** The [report](../../../docs/training-consistency-review-2026-11-11.md) accepts 14 box-only corrections in three existing training images. It preserves every existing class and corner token in the new delta, holds nine initial box proposals and retains the separate identity uncertainties. The [candidate](../../../data/candidates/consistency-2026-11-11/README.md) exactly inherits the selected PR35 eight-file overlay. Model benefit is unmeasured; PR35 and its explicit 0.13 workflow remain selected.

## Inspect the record

| File | Purpose |
| --- | --- |
| [scope.json](scope.json) | Six-image cohort, selection rule, source hashes and stopping rule fixed before viewing pixels |
| [adjudication.json](adjudication.json) | Fourteen accepted boxes, disposition of all 135 rows, held proposals, uncertainty and provenance |
| [evidence.zip](evidence.zip) | Complete recorded reviews, reconciliation, native-pixel proof crops, ambiguity crops and reproduction code |
| [capsule-verification.json](capsule-verification.json) | Member/hash checks, extracted audit replay and exact candidate rebuild |
| [Candidate verification](../../../data/candidates/consistency-2026-11-11/verification.json) | Result from the unchanged repository annotation verifier |
| [Independent composition verification](../../../data/candidates/consistency-2026-11-11/composition-verification.json) | Frozen selection, original PNGs/labels, exact inheritance, endpoint preservation and protected-source checks |

The capsule contains **108 inventoried payloads / 109 ZIP members**. It is **4,699,376 bytes**, SHA256:

```text
5964d09180c8434378eb7fe08ade6ac8d9d71eb235e84ca92441496fb0942be3
```

| Frozen identity | SHA256 |
| --- | --- |
| Original data manifest | `72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926` |
| Prospective six-image scope | `de9f1d44301fcfb0e4fb07b14a502256529c3e33abc468411d4468511b6d0f7a` |
| Pre-image inventory | `d71c7fb5ecf1f4cf58c0120b76588d5913d653bc3f18c89ce8b59725d9343294` |
| Final adjudication | `60c5cbf8f215c796a28c13c184e88ad7fc9dffac08d821e2b77194ac97b147fb` |
| Selected parent manifest | `28fc8f5da95605089d61fb062132200ed296c3a10b216f1a61dbda1798ea4ba7` |
| New candidate manifest | `2f59ec8a09466922eb2f20fd66c5f0f8a972a6c5694bbb41c17d51277ff3b9a9` |
| New candidate review | `5e22ccc984de4887f62ae11ec38f2aca614a0ef12078ee180a43ad612f602afc` |

The workflow/source base is commit `61ae0e68fc517a6dfc7d8c19b63d846303a79e03`. Original PNGs were read from the clean, hydrated source at `d43a58b9456a782ace14e20f0bf7ed2da3572ca1` and individually matched to the same frozen manifest. Full original PNGs and model weights retain their existing sources.

## What was verified

The independent check confirms all **157 original label hashes**, all **six original PNG hashes/dimensions**, and the exact prospective selection rule. It confirms the inherited **eight label files, eight change objects and twelve decision objects**, the **fourteen disjoint bbox-only replacements**, **270 preserved corner triples** (223 visible/47 missing), and unchanged protected source/reference/protocol/history bytes in 339 tracked files. All 135 reviewed rows are accounted for as fourteen edited plus 121 retained. Nine held box proposals and eight additional identity-concern rows remain unchanged.

The unchanged verifier passes for both parent and new candidate. Effective training counts remain **137 images / 3,771 objects / 6,841 visible endpoints / 701 missing slots / 13 empty images**. Original validation remains **20 images / 664 objects / 1,227 visible endpoints / 101 missing slots / three empty images**. The separate 663-object corrected development reference is unchanged.

The ZIP's exact members, sizes, CRCs and SHA256 digests were checked. Its extracted independent auditor passed against the repository and six original PNGs. Its extracted builder reproduced **all thirteen generated candidate files** byte-for-byte: eleven label files, manifest and review JSON. Documentation and separately produced verification results are outside that builder's output.

Mechanical verification does not certify physical target identity, corner precision, complete annotation coverage or model performance. Those distinctions and the uncertainty decisions are explicit in the report and review records.

## Reproduce

Extract the capsule into a new directory and follow its `REPRODUCE.md`. The main candidate check is also available directly from the repository root:

```bash
python -B scripts/verify_pose_annotation_candidate.py \
  --candidate data/candidates/consistency-2026-11-11/manifest.json
```

The capsule supplies `verify_capsule.py`, the independent `mechanical-review/audit_candidate.py`, the exact candidate builder, and source-pixel figure/reconciliation renderers. Replay paths are configurable. The full independent check needs hydrated original PNGs and the declared historical Git objects; ZIP integrity by itself has a narrower meaning.

The initial review records retain their original bytes, including historical workspace paths. The capsule includes every individual raw/overlay comparison crop and three grouped native overlays. Additional native full-frame renderings named in the proof index can be regenerated with the included renderer. Green marks the fourteen accepted recommendations; amber marks initial estimates held unchanged.
