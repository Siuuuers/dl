#!/usr/bin/env python3
"""Build the explicitly limited, immutable correction candidate and protocol."""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from verify_incremental_pose_candidate import (DEFAULT_CANDIDATE, PROTOCOL_PATH, compose,
                                              protocol_document, verify_incremental_candidate)
from verify_reconciled_pose_labels import encoded


def build(root):
    root = Path(root).resolve()
    candidate = root / DEFAULT_CANDIDATE
    protocol = root / PROTOCOL_PATH
    if candidate.parent.exists() or protocol.exists():
        raise FileExistsError("incremental artifacts already exist; verify them instead of overwriting")
    payloads, manifest, source_receipt = compose(root)
    protocol_payload = encoded(protocol_document(root, manifest, source_receipt))
    for relative, payload in payloads.items():
        path = candidate.parent / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_bytes(protocol_payload)
    return verify_incremental_candidate(root, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = build(args.root)
    print(f"Verified {result['candidate_id']}: {len(result['changed_images_vs_control'])} changed photos; "
          f"{len(result['retained_legacy_image_ids'])} photos retain exact PR35 labels in both arms.")


if __name__ == "__main__":
    main()
