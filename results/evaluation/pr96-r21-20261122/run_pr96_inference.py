#!/usr/bin/env python3
"""Reproduce the three fixed PR96 CPU inferences without training or tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED = {
    "scripts/evaluate_pose.py": "3f80bd09efbe97327b705ffe6ad3e662111f3e2bc71b69b51a9d38d7baa90ce0",
    "scripts/pose_scoring.py": "04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018",
    "scripts/pose_checkpoint.py": "e0f85a76fd5234a98e66783d3408e22175ba334c6209f91aca8680221a2f8d5d",
    "experiments/front-review-2026-11-22-r21.json": "6e79440caa989de5327531ebde2242aa3ba7f25de7497ebac8542ee766400122",
}
MODEL_SHA = {
    "pr35": "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1",
    "reference": "3b6054b0da543752ba7202995bce8834dc5daf02ea7e2463ea409da365e6acc6",
    "candidate": "a14467eb1e15258ab8a5f473e1f1abd600766e11d068dba4cd041714384de1d4",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    for role in MODEL_SHA:
        parser.add_argument("--" + role + "-manifest", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    for path, expected in EXPECTED.items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen input differs: {path}")
    manifests = {role: getattr(args, role + "_manifest").resolve() for role in MODEL_SHA}
    for role, path in manifests.items():
        if json.loads(path.read_text())["checkpoint"]["sha256"] != MODEL_SHA[role]:
            raise ValueError(f"Unexpected checkpoint for {role}")
    output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "YOLO_CONFIG_DIR": str(output / "ultralytics-config"), "YOLO_OFFLINE": "true"}
    commands = []
    for role, manifest in manifests.items():
        command = [str(args.python.absolute()), "-B", str(root / "scripts/evaluate_pose.py"), "--root", str(root), "--output", str(output / ("eval-" + role)), "--checkpoint-manifest", str(manifest), "--device", "cpu", "--threads", "4", "--conf", "0.13", "--endpoint-conf", "0.50", "--matching-iou", "0.50", "--overlays", "0"]
        print("Running", role, flush=True)
        with (output / ("inference-" + role + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        commands.append({"role": role, "command": command, "exit_code": result.returncode})
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{role} failed; inspect its log. No later model was run.")
    print("Completed the three fixed CPU inferences.")


if __name__ == "__main__":
    main()
