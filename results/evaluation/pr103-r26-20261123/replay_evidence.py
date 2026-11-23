#!/usr/bin/env python3
"""Replay the PR103 comparison from its preserved predictions and receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root / "results/evaluation/pr103-r26-20261123"
    if args.output.exists():
        raise ValueError("Choose a new verification output")
    with tempfile.TemporaryDirectory(prefix="pr103-replay-") as temp:
        replay = Path(temp) / "fixed-comparison.json"
        command = [sys.executable, "-B", str(evidence / "measure_pr103.py"),
                   "--root", str(root), "--output", str(replay)]
        for role in ("candidate", "reference", "pr35", "r25", "r25control"):
            command.extend(["--" + role,
                            str(evidence / "inference" / ("eval-" + role) / "predictions.jsonl"),
                            "--" + role + "-manifest",
                            str(evidence / "checkpoint-receipts" / role / "checkpoint-manifest.json")])
        subprocess.run(command, check=True, capture_output=True, text=True)
        expected = (evidence / "fixed-comparison.json").read_bytes()
        actual = replay.read_bytes()
        if actual != expected:
            raise ValueError("Replay differs from the preserved comparison")
    report = {"schema_version": 1, "exact_report_bytes_reproduced": True,
              "fixed_comparison_sha256": hashlib.sha256(expected).hexdigest(),
              "reference_versions": 3, "models": 5, "prediction_records": 100,
              "inference_repeated": False, "training_repeated": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
