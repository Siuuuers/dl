#!/usr/bin/env python3
"""Compose the separately versioned, bounded sequence-front training experiment.

The complete r2 review is never released by this command. Its held photographs
keep exact PR35 label bytes in both arms; the original validation stays frozen.
"""

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from verify_incremental_pose_candidate import (
    SEQUENCE_CANDIDATE, SEQUENCE_PROTOCOL_PATH, compose_sequence,
    sequence_protocol_document, verify_incremental_candidate,
)
from verify_reconciled_pose_labels import encoded


def build(root):
    root = Path(root).resolve()
    candidate, protocol = root / SEQUENCE_CANDIDATE, root / SEQUENCE_PROTOCOL_PATH
    if candidate.parent.exists() or protocol.exists():
        raise FileExistsError("sequence training artifacts already exist; verify instead of overwriting")
    payloads, manifest, source_receipt = compose_sequence(root)
    protocol_payload = encoded(sequence_protocol_document(root, manifest, source_receipt))
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
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (verify_incremental_candidate(args.root, SEQUENCE_CANDIDATE)
              if args.verify_only else build(args.root))
    print(f"Verified {result['candidate_id']}: {len(result['changed_images_vs_control'])} changed photos; "
          f"{len(result['retained_legacy_image_ids'])} photos retain exact PR35 labels in both arms.")


if __name__ == "__main__":
    main()
