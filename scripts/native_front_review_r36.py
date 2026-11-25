#!/usr/bin/env python3
"""Verify and package the native-loss R35-versus-R36 supervision comparison."""
import argparse
import json
from pathlib import Path

from evaluate_pose import _relative_file, file_sha256
from front_review_r36_training import PROTOCOL_PATH, verify_candidate
import native_front_review_r33 as native
from pose_checkpoint import recipe_identity
from predict_pose_folder import BASELINE_SHA256
from restore_gpu_inputs import read_json, require
from train_pose_comparison import source_identity

ROLES = ("reference", "candidate")


def load_protocol(root, path=PROTOCOL_PATH):
    root = Path(root).resolve()
    path = Path(path)
    path = path.resolve() if path.is_absolute() else _relative_file(root, str(path))
    require(path == root / PROTOCOL_PATH, "R36 protocol path differs")
    document = read_json(path)
    require(document["experiment_id"] == "front-review-r36"
            and document["recipe_id"] == "reviewed-incremental-fronts-seed46-v1"
            and all(document[k] == v for k, v in recipe_identity(document["recipe_id"]).items()),
            "R36 numerical recipe differs")
    require(document["initial_checkpoint_sha256"] == BASELINE_SHA256
            and document["execution"]["loss"] == "unmodified_native_Pose26"
            and document["execution"]["partial_box_factor"] == 1
            and document["execution"]["automatic_promotion"] is False,
            "R36 native comparison contract differs")
    verified = verify_candidate(root, root / document["candidate"]["path"])
    require(verified["training_input_delta"]["report"]["summary"]["changed_rows"] > 0,
            "Identical-label comparison is not useful")
    require(set(document["roles"]) == set(ROLES), "R36 role inventory differs")
    for prefix in ("primary", "diagnostic"):
        evaluation = document["evaluation"]
        require(file_sha256(_relative_file(root, evaluation[prefix+"_reference"]))
                == evaluation[prefix+"_reference_sha256"], "Frozen evaluation reference differs")
    return {"document": document, "path": path, "sha256": file_sha256(path),
            "candidate_verification": verified}


def verify_trial(root, run, role="candidate", context=None):
    return native.verify_trial(root, run, role, context or load_protocol(root))


def verify_run(root, run, role, context=None):
    return native.verify_run(root, run, role, context or load_protocol(root))


def package_pair(root, reference, candidate, output, verify_existing=False, protocol=PROTOCOL_PATH):
    return native.package_pair(root, reference, candidate, output, verify_existing, protocol,
                               context=load_protocol(root, protocol))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", default=PROTOCOL_PATH)
    args = parser.parse_args()
    context = load_protocol(args.root, args.protocol)
    print(json.dumps({"status": "verified", "protocol_sha256": context["sha256"],
                      "training_delta": context["candidate_verification"]["training_input_delta"]["report"]["summary"],
                      "training_executed": False}, indent=2))
