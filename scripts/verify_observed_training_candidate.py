#!/usr/bin/env python3
"""Verify reviewed visible-front labels and their separate training-readiness gate.

The immutable review composer deliberately never grants a training release.
This adapter verifies its complete output, the historical PR35 supervision, and
an explicit experiment protocol. A pending image question blocks GPU staging.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

from compose_observed_pose_review import KIND, bound, sha, verify_tree
from verify_pose_candidate import read_reference, require
from verify_pose_composed_candidate import (PARENT as PR35_CONTROL, bound_file,
                                           regular_tree, verify_historical_control)


PROTOCOL_PATH = "experiments/observed-fronts-2026-11-14.json"
RECIPE_ID = "reviewed-observed-fronts-seed46-v1"
REVIEW_STATES = {"resolved", "partial", "held"}
# Exact git blobs from e46c562f3b34994db709c9ae4f8f2f0d2a3c5aed are archived
# locally so a shallow production checkout needs no historical git objects.
# The unchanged reconciliation inventory independently requires these hashes.
HISTORICAL_REPLAY_CODE = {
    "scripts/pose_checkpoint.py": {
        "path": "scripts/historical/observed-review-parent/pose_checkpoint.py",
        "sha256": "f902c34720283480f948cca78bd2d6d24fd07bf6bc5a325945172d56d55f5bd5",
    },
    "scripts/stage_pose_training.py": {
        "path": "scripts/historical/observed-review-parent/stage_pose_training.py",
        "sha256": "a733f5e80335204747139e87a723b7bb095123059dc5a58b86ee824a26e5afe6",
    },
}


def compose_observed_review(root, review_path):
    """Replay the unchanged composer with its exact pinned historical support.

    The live training modules may extend recipes. Only disposable snapshot
    copies receive the archived originals; all frozen validators and their
    source, policy, image, label, weight, and evidence checks still execute.
    """
    root = Path(root).resolve()
    review_path = Path(review_path)
    if not review_path.is_absolute():
        review_path = root / review_path
    require(review_path.is_file() and not review_path.is_symlink()
            and review_path.resolve().is_relative_to(root), "review must be a regular repository file")
    relative_review = review_path.resolve().relative_to(root)
    historical = {relative: bound(root, descriptor).read_bytes()
                  for relative, descriptor in HISTORICAL_REPLAY_CODE.items()}

    def link_or_copy(source, destination):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    with tempfile.TemporaryDirectory(prefix="observed-history-replay-") as temporary:
        snapshot, output = Path(temporary) / "snapshot", Path(temporary) / "payloads"
        for relative in ("data", "docs", "results/review", "metadata", "weights", "scripts"):
            folder = root / relative
            regular_tree(folder)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(folder, target, copy_function=link_or_copy)
        for relative, payload in historical.items():
            target = snapshot / relative
            # Never write through a hard link into the live source checkout.
            target.unlink()
            target.write_bytes(payload)
        program = """
import sys
from pathlib import Path
root, review, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / 'scripts'))
from compose_observed_pose_review import compose
payloads, manifest = compose(root, root / review)
for relative, payload in payloads.items():
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(snapshot),
                                 relative_review.as_posix(), str(output)],
                                text=True, capture_output=True, timeout=120)
        require(result.returncode == 0,
                "isolated frozen observed-review replay failed: " + (result.stderr or result.stdout).strip())
        paths = regular_tree(output)
        payloads = {relative: (output / relative).read_bytes() for relative in sorted(paths)}
    return payloads, json.loads(payloads["manifest.json"])


def review_batches(root, review, coverage):
    """Load only batch documents bound by the accepted annotation evidence."""
    evidence = {}
    for item in review["evidence"]:
        require(item["path"] not in evidence, "duplicate accepted-review evidence path")
        evidence[item["path"]] = item
    paths = {item["review_path"] for item in coverage["images"]}
    require(paths <= set(evidence), "coverage references an unbound batch review")
    return {path: json.loads(bound(root, evidence[path]).read_text()) for path in sorted(paths)}


def reconcile_review_inventory(parent_status, batches, status, coverage):
    """Rebuild case ownership and status from evidence, not the release summary.

    This pure check also binds each coverage claim to its batch image record.
    An omitted new issue or an undisposed prior open case must not turn a
    self-consistent summary into permission to train.
    """
    def named_items(items, key, context):
        require(isinstance(items, list), f"{context} must be a list")
        result = {}
        for item in items:
            require(isinstance(item, dict), f"invalid {context} entry")
            name = item.get(key)
            require(isinstance(name, str) and bool(name.strip()), f"invalid {context} {key}")
            require(name not in result, f"duplicate {context} {key}: {name}")
            result[name] = item
        return result

    def case_state(item, context, *, default=None):
        state = item.get("status", default)
        require(isinstance(state, str) and state in REVIEW_STATES,
                f"unknown physical-review decision: {context}")
        return state

    def partial_rows(value, context):
        require(isinstance(value, list)
                and all(type(row) is int and row > 0 for row in value)
                and len(value) == len(set(value)), f"invalid partial-row inventory: {context}")
        return set(value)

    require(parent_status["candidate"] == coverage["input_candidate"],
            "parent review status belongs to a different input candidate")
    prior = named_items(parent_status["cases"], "case_id", "parent cases")
    expected = {}
    for case_id, item in prior.items():
        image_id = item.get("image_id")
        require(isinstance(image_id, str) and bool(image_id), "parent case lacks image ownership")
        expected[case_id] = (image_id, case_state(item, case_id))
    prior_open = {case_id for case_id, (_, state) in expected.items() if state != "resolved"}
    handled, images = set(), {}
    for path, batch in batches.items():
        require(batch["input_candidate"] == coverage["input_candidate"],
                f"batch review uses a different input candidate: {path}")
        for image_id, image in named_items(batch["images"], "image_id", "batch images").items():
            require(image_id not in images, f"image appears in multiple batch reviews: {image_id}")
            images[image_id] = (path, image)
            dispositions = named_items(image.get("case_dispositions", []), "case_id", "case dispositions")
            for case_id, decision in dispositions.items():
                require(case_id in prior, f"disposition names an unknown parent case: {case_id}")
                require(prior[case_id]["image_id"] == image_id,
                        f"case disposition has different image ownership: {case_id}")
                require(case_id not in handled, f"case has multiple dispositions: {case_id}")
                handled.add(case_id)
                expected[case_id] = (image_id, case_state(decision, case_id))
            issues = image.get("new_issues", [])
            require(isinstance(issues, list), f"new issues must be a list: {image_id}")
            for issue in issues:
                require(isinstance(issue, dict), f"invalid new issue: {image_id}")
                case_id = issue.get("issue_id", issue.get("case_id"))
                require(isinstance(case_id, str) and bool(case_id.strip()), "new issue lacks a case ID")
                require("issue_id" not in issue or "case_id" not in issue
                        or issue["issue_id"] == issue["case_id"], "conflicting new issue IDs")
                require(case_id not in expected, f"new issue reuses an existing case ID: {case_id}")
                expected[case_id] = (image_id, case_state(issue, case_id, default="held"))
    require(prior_open <= handled,
            "prior open cases lack explicit dispositions: " + ", ".join(sorted(prior_open - handled)))
    require(all(image_id in images for image_id, _ in expected.values()),
            "review case belongs to an image absent from the batch reviews")

    actual = named_items(status["cases"], "case_id", "review cases")
    require(set(actual) == set(expected),
            "review case inventory differs from parent cases plus discovered issues")
    for case_id, item in actual.items():
        require((item.get("image_id"), case_state(item, case_id)) == expected[case_id],
                f"declared case status or image ownership differs from evidence: {case_id}")

    reviewed = named_items(coverage["images"], "image_id", "coverage images")
    require(set(reviewed) == set(images), "coverage image inventory differs from batch reviews")
    for image_id, item in reviewed.items():
        path, image = images[image_id]
        require(item["review_path"] == path, f"coverage names the wrong batch review: {image_id}")
        require(item["source_image_sha256"] == image["source_original_image"]["sha256"],
                f"coverage source image differs from reviewed pixels: {image_id}")
        require(partial_rows(item["input_partial_rows"], image_id)
                == partial_rows(image["reviewed_partial_rows"], image_id),
                f"coverage partial rows differ from batch review: {image_id}")
        require(type(image["image_review_complete"]) is bool
                and type(item["image_review_complete"]) is bool
                and item["image_review_complete"] == image["image_review_complete"],
                f"coverage completion differs from batch review: {image_id}")
        declared = item["unresolved_case_ids"]
        require(isinstance(declared, list) and all(isinstance(case_id, str) for case_id in declared)
                and len(declared) == len(set(declared)), f"invalid per-image pending cases: {image_id}")
        pending = {case_id for case_id, (owner, state) in expected.items()
                   if owner == image_id and state != "resolved"}
        require(set(declared) == pending, f"coverage pending cases differ from evidence: {image_id}")
    return [case_id for case_id, item in actual.items() if item["status"] != "resolved"]


def check_input_partial_coverage(root, coverage, source_manifest_sha256):
    """Derive review coverage from bound input labels, not two matching summaries."""
    input_path = bound(root, coverage["input_candidate"])
    proposal = json.loads(input_path.read_text())
    require(proposal["source_manifest_sha256"] == source_manifest_sha256,
            "review input uses a different source manifest")
    source = json.loads(bound(root, {"path": "data/manifest.json",
                                     "sha256": source_manifest_sha256}).read_text())
    train = {record["id"]: record for record in source["records"] if record["split"] == "train"}
    changes = {}
    for change in proposal["changes"]:
        image_id = change["image_id"]
        require(image_id in train and image_id not in changes,
                "review input has an unknown, non-training, or repeated changed image")
        record = train[image_id]
        require(change["candidate_label"] == change["source_label"] == record["label"]
                and change["source_label_sha256"] == record["label_sha256"]
                and change["source_image_sha256"] == record["image_sha256"],
                f"review input label mapping differs: {image_id}")
        changes[image_id] = change
    reviewed = {item["image_id"]: item for item in coverage["images"]}
    require(len(reviewed) == len(coverage["images"]) and set(reviewed) == set(train),
            "input coverage must inventory every training original exactly once")
    total = 0
    for image_id, record in train.items():
        change = changes.get(image_id)
        base = input_path.parent if change else root / "data"
        descriptor = {"path": change["candidate_label"] if change else record["label"],
                      "sha256": change["candidate_label_sha256"] if change else record["label_sha256"]}
        rows = []
        for number, line in enumerate(bound(base, descriptor).read_text().splitlines(), 1):
            tokens = line.split()
            require(len(tokens) == 11 and tokens[7] in ("0", "2") and tokens[10] in ("0", "2"),
                    f"invalid review input pose row: {image_id}:{number}")
            if tokens[7] == "0" or tokens[10] == "0":
                rows.append(number)
        require(reviewed[image_id]["input_partial_rows"] == rows,
                f"coverage partial rows differ from bound input labels: {image_id}")
        total += len(rows)
    expected = {"reviewed_training_images": len(train), "reviewed_input_partial_rows": total}
    for name, count in expected.items():
        require(type(coverage.get(name)) is int and coverage[name] == count,
                f"coverage {name} differs from bound input labels")
    return expected


def check_annotation_readiness(protocol, reviewed, pending, reference, *, require_ready):
    """Require settled training and development labels before GPU staging."""
    remaining = reference["remaining_current_rows"]
    require(isinstance(remaining, list), "development-reference pending rows must be a list")
    require(type(protocol["evaluation"]["remaining_reference_rows"]) is int
            and protocol["evaluation"]["remaining_reference_rows"] == len(remaining),
            "declared development-reference pending count differs from its manifest")
    incomplete = sum(item["image_review_complete"] is not True for item in reviewed.values())
    ready = not incomplete and not pending and not remaining
    require(protocol["training_readiness"] == ("ready" if ready else "blocked_label_review"),
            "declared training readiness differs from physical-review decisions")
    if require_ready:
        row_names = ", ".join(f"{item['image_id']}:row{item['row']}" for item in remaining) or "none"
        require(ready, "Training is blocked by the annotation review: "
                f"{len(pending)} unresolved training case groups; {incomplete} incomplete training images; "
                f"{len(remaining)} unresolved development-reference rows ({row_names}). "
                "Resolve the physical questions and issue a new hash-bound protocol before training.")
    return remaining


def verify_observed_training_candidate(root, candidate, *, require_ready=True):
    root, candidate = Path(root).resolve(), Path(candidate).resolve()
    proposal = json.loads(candidate.read_text())
    require(proposal.get("kind") == KIND, "expected observed review composition")
    review_path = bound(root, proposal["review"])
    payloads, rebuilt = compose_observed_review(root, review_path)
    verify_tree(candidate.parent, payloads)
    require(proposal == rebuilt, "observed review manifest reconstruction differs")
    review = json.loads(review_path.read_text())

    protocol_path = root / PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == 1
            and protocol.get("kind") == "baomai-observed-fronts-training-protocol",
            "unsupported observed-front training protocol")
    descriptor = {"path": candidate.relative_to(root).as_posix(), "sha256": sha(candidate)}
    require(protocol["candidate"] == descriptor, "protocol binds another candidate")
    require(protocol["recipe_id"] == RECIPE_ID, "protocol recipe differs")
    from pose_checkpoint import recipe_identity
    require(protocol["recipe_sha256"] == recipe_identity(RECIPE_ID)["recipe_sha256"],
            "protocol numerical recipe differs")
    require(protocol["control"] == PR35_CONTROL, "control must retain exact PR35 training labels")

    _, records, _, source_bytes, _, _ = read_reference(root, proposal["source_manifest_sha256"])
    # The unchanged historical verifier replays the original annotation overlay
    # as the parent of the ten-slot endpoint overlay in an isolated old-policy
    # snapshot. The live policy and all historical files remain untouched.
    verify_historical_control(root, records, source_bytes)
    control_path = bound_file(root, PR35_CONTROL, "PR35 supervision", "manifest")
    control = json.loads(control_path.read_text())
    control_receipt = {
        "manifest": PR35_CONTROL["manifest"], "manifest_sha256": PR35_CONTROL["sha256"],
        "candidate_id": control["candidate_id"], "kind": control["kind"],
        "changed_train_labels": len(control["changes"]), "counts": control["expected_counts"],
    }
    numerical_recipe = recipe_identity(RECIPE_ID)
    expected_roles = {
        arm: {"arm": arm,
              "supervision": "pr35_annotation_control" if arm == "reference" else "observed_fronts_treatment",
              "candidate_manifest": descriptor["path"], "candidate_manifest_sha256": descriptor["sha256"],
              "candidate_id": proposal["candidate_id"], **numerical_recipe,
              "counts": control["expected_counts"] if arm == "reference" else proposal["expected_counts"]}
        for arm in ("reference", "candidate")
    }
    require(protocol["roles"] == expected_roles, "protocol arm roles or counts differ")

    status_path = bound(root, protocol["review_status"])
    status = json.loads(status_path.read_text())
    require(status["candidate"] == descriptor, "review status belongs to another candidate")
    require(status["accepted_review_path"] == review_path.relative_to(root).as_posix(),
            "review status names different accepted evidence")
    parent_status = json.loads(bound(root, status["parent_status"]).read_text())

    coverage_path = bound(root, protocol["coverage"])
    coverage = json.loads(coverage_path.read_text())
    require(coverage["candidate"] == descriptor, "coverage belongs to another candidate")
    pending = reconcile_review_inventory(parent_status, review_batches(root, review, coverage), status, coverage)
    check_input_partial_coverage(root, coverage, proposal["source_manifest_sha256"])
    require(type(status["remaining_all_case_groups"]) is int
            and status["remaining_all_case_groups"] == len(pending), "review pending count differs")
    require(protocol["unresolved_case_ids"] == pending, "protocol omits or changes pending questions")
    train = {record["id"]: record for record in records if record["split"] == "train"}
    reviewed = {item["image_id"]: item for item in coverage["images"]}
    require(len(reviewed) == len(coverage["images"]) and set(reviewed) == set(train),
            "coverage must inventory each of the 137 training originals exactly once")
    changes = {change["image_id"]: change for change in proposal["changes"]}
    for image_id, item in reviewed.items():
        record = train[image_id]
        digest = changes.get(image_id, {}).get("candidate_label_sha256", record["label_sha256"])
        require(item["source_image_sha256"] == record["image_sha256"]
                and item["candidate_label_sha256"] == digest,
                f"coverage source or label identity differs: {image_id}")
    require(protocol["initial_checkpoint_sha256"] ==
            json.loads((root / "metadata/model_card.json").read_text())["sha256"],
            "training initialization differs from original epoch173")
    evaluation = protocol["evaluation"]
    reference_path = root / evaluation["primary_reference"]
    require(sha(reference_path) == evaluation["primary_reference_sha256"],
            "development-reference identity differs")
    reference = json.loads(reference_path.read_text())
    for item in reference["files"]:
        require(sha(reference_path.parent / item["label_path"]) == item["label_sha256"],
                "development-reference label bytes differ")
    require(evaluation["training_validation_unchanged"] is True
            and evaluation["references_reported_separately"] is True,
            "development reference must remain separate from in-training validation")
    remaining_reference = check_annotation_readiness(protocol, reviewed, pending, reference,
                                                     require_ready=require_ready)

    return {
        "status": "verified", "candidate_id": proposal["candidate_id"],
        "candidate_manifest_sha256": sha(candidate),
        "source_manifest_sha256": proposal["source_manifest_sha256"],
        "declared_source_revision": proposal["source_revision"],
        "counts": proposal["expected_counts"], "changed_train_labels": len(proposal["changes"]),
        "removed_objects": sum(change["source_objects"] - change["candidate_objects"]
                               for change in proposal["changes"]),
        "control": control_receipt,
        "training_protocol": {"path": PROTOCOL_PATH, "sha256": sha(protocol_path)},
        "training_readiness": protocol["training_readiness"],
        "unresolved_case_ids": pending, "coverage_images": len(reviewed),
        "remaining_reference_rows": len(remaining_reference),
        "unresolved_reference_rows": remaining_reference,
        "training_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true",
                        help="Report integrity and outstanding review questions without granting readiness")
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = args.candidate if args.candidate.is_absolute() else root / args.candidate
    print(json.dumps(verify_observed_training_candidate(root, candidate,
                                                       require_ready=not args.inspect), indent=2))


if __name__ == "__main__":
    main()
