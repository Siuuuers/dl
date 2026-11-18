# R7 preparation checks

**Verified input preparation; no r7 model training was run.**

[verification.json](verification.json) records actual staging of both arms, all 157 original camera-image hashes, unchanged original label bytes, original 664-front validation in both arms, exact PR35 labels on 44 held training photographs, and unchanged 666/663 scoring references. [staging-manifest.json](staging-manifest.json) records the exact staged files and counts.

The review restores `old_000013.txt` byte for byte from the original. The complete review has 3,788 training fronts; the composed training candidate has 3,780 because held photos use PR35 supervision. The control has 3,771. Both retain 137 training images and 13 empty scenes; 79 photos differ between arms.

Run `python3 scripts/check_front_review_r7.py --output /path/to/a/new/check-directory` from the checkout. The output must be new and outside the repository. All source PNGs must be hydrated through Git LFS. A different output location or run time changes staging metadata hashes without changing the scientific inputs.

The separate notebook pin receipt records its actual source commit, protocol hash, code-cell checks and input preflight. The PDF's rendered-page review is separate from coordinate and provenance validation. None of these checks certifies every physical interpretation or claims improved model accuracy.
