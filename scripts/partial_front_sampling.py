#!/usr/bin/env python3
"""Fixed R28-label partial-front exposure experiment and returned-pair checks."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import zipfile

sys.dont_write_bytecode = True

from evaluate_pose import _relative_file, file_sha256
from package_gpu_results import INDEX_NAME, check_pair, collect_arm, copy_member, shared_settings, verify_archive
from pose_checkpoint import _same, recipe_identity
from predict_pose_folder import BASELINE_SHA256, _reference_model
from restore_gpu_inputs import read_json, require


DEFAULT_PROTOCOL = "experiments/partial-front-sampling-2026-11-24-r29.json"
PROTOCOL_SHA256 = "2ab005b9d64bdd388a71a1026df4885152762be1f992cd90607d5256a046468f"
ROLES = ("control", "candidate")
COHORT = ["new_000012", "new_000018", "new_000023", "new_000031", "new_000044",
          "new_000045", "new_000094", "new_000116", "new_000128"]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_protocol(root, path=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    path = Path(path)
    path = path.resolve() if path.is_absolute() else _relative_file(root, str(path))
    require(path == root / DEFAULT_PROTOCOL, "sampling protocol path differs")
    require(file_sha256(path) == PROTOCOL_SHA256, "fixed sampling protocol differs")
    document = read_json(path)
    require(document["schema_version"] == 1
            and document["kind"] == "baomai-partial-front-sampling-protocol"
            and document["experiment_id"] == "partial-front-sampling-r29", "wrong sampling experiment")
    require(document["candidate_manifest"] == "data/candidates/front-review-2026-11-23-r28/manifest.json"
            and document["label_arm_for_both_experimental_roles"] == "candidate",
            "both roles must use the accepted eligible R28 label arm")
    identity = recipe_identity(document["recipe_id"])
    require(all(document[k] == v for k, v in identity.items()), "numerical training recipe differs")
    require(document["initial_checkpoint_sha256"] == BASELINE_SHA256, "epoch173 initializer differs")
    for relative, expected in (("data/manifest.json", document["source_manifest_sha256"]),
                               (document["candidate_manifest"], document["candidate_manifest_sha256"])):
        require(file_sha256(_relative_file(root, relative)) == expected, "source identity differs: " + relative)
    for item in (document["accepted_review"], document["evidence"]["training_inventory"],
                 document["evidence"]["cohort"]):
        require(file_sha256(_relative_file(root, item["path"])) == item["sha256"],
                "sampling evidence differs: " + item["path"])
    evaluation = document["evaluation"]
    for prefix in ("primary", "diagnostic"):
        require(file_sha256(_relative_file(root, evaluation[prefix + "_reference"]))
                == evaluation[prefix + "_reference_sha256"], "frozen evaluation reference differs")
    spec, execution = document["sampling"], document["execution"]
    require(spec["seed"] == 46 and spec["cohort_image_ids"] == COHORT
            and spec["samples_per_epoch"] == 137 and spec["extra_presentations_per_cohort_image"] == 1
            and spec["preserve_empty_images"] is True and spec["preserve_held_and_excluded_images"] is True,
            "fixed sampling dose or protection policy differs")
    require(execution["epochs"] == 30 and execution["resource_trial_batches"] == 4
            and execution["expected_batches"] == 1050 and execution["expected_images"] == 4110
            and execution["expected_optimizer_steps"] == 78 and execution["same_runtime_required"] is True
            and execution["automatic_promotion"] is False, "fixed experiment budget differs")
    return {"document": document, "sha256": PROTOCOL_SHA256, "path": path}


def source_identity(root):
    from train_pose_comparison import source_identity as base_source
    root = Path(root).resolve()
    result = base_source(root)
    for name in ("scripts/partial_front_sampling.py", "scripts/train_pose_sampling.py", DEFAULT_PROTOCOL):
        result["files_sha256"][name] = file_sha256(_relative_file(root, name))
    return result


def experiment_identity(context, role):
    require(role in ROLES, "unknown sampling role")
    document = context["document"]
    return {"id": document["experiment_id"], "role": role, "protocol_sha256": context["sha256"],
            "base_recipe_id": document["recipe_id"], "base_recipe_sha256": document["recipe_sha256"],
            "label_arm": "candidate", "label_manifest": document["candidate_manifest"],
            "label_manifest_sha256": document["candidate_manifest_sha256"],
            "sampling_spec_sha256": digest_json(document["sampling"])}


def sampling_inventory(root, context=None):
    """Resolve the existing effective R28 training labels; never read development coordinates."""
    root = Path(root).resolve()
    context = context or load_protocol(root)
    document = context["document"]
    source = read_json(root / "data/manifest.json")
    manifest_path = root / document["candidate_manifest"]
    candidate = read_json(manifest_path)
    changes = {row["image_id"]: row for row in candidate["changes"]}
    selected = {row["image_id"]: row for row in candidate["image_sources"]}
    images = []
    for record in sorted(source["records"], key=lambda item: item["id"]):
        if record["split"] != "train":
            continue
        image_id, change = record["id"], changes.get(record["id"])
        label = (_relative_file(manifest_path.parent, change["candidate_label"]) if change
                 else _relative_file(root / "data", record["label"]))
        expected = change["candidate_label_sha256"] if change else record["label_sha256"]
        require(file_sha256(label) == expected, "effective R28 label differs: " + image_id)
        groups = Counter()
        for line in label.read_text().splitlines():
            fields = line.split()
            require(len(fields) == 11 and all(fields[i] in ("0", "2", "0.0", "2.0") for i in (7, 10)),
                    "unexpected accepted visibility encoding")
            groups[sum(float(fields[i]) > 0 for i in (7, 10))] += 1
        images.append({"image_id": image_id, "source_image_sha256": record["image_sha256"],
                       "label_path": label.relative_to(root).as_posix(), "label_sha256": expected,
                       "selection": selected[image_id]["selection"], "objects": sum(groups.values()),
                       "zero_corner_fronts": groups[0], "one_corner_fronts": groups[1],
                       "two_corner_fronts": groups[2]})
    ids = [row["image_id"] for row in images]
    empties = [row["image_id"] for row in images if row["objects"] == 0]
    protected = sorted(set(empties + candidate["retained_legacy_image_ids"]
                           + candidate["excluded_training_image_ids"]))
    observed_cohort = [row["image_id"] for row in images if row["selection"] == "accepted_r28_review"
                       and row["zero_corner_fronts"] >= 1
                       and row["zero_corner_fronts"] + row["one_corner_fronts"] >= 8
                       and 3 * (row["zero_corner_fronts"] + row["one_corner_fronts"]) >= row["objects"]]
    require(observed_cohort == COHORT, "training-only cohort rule no longer gives the declared nine images")
    donors = sorted(set(ids) - set(protected) - set(COHORT))
    require(len(ids) == len(set(ids)) == 137 and len(empties) == 13
            and len(protected) == 45 and len(donors) == 83 and not set(COHORT) & set(protected),
            "sampling cohort, donor or protected population differs")
    require(all(row["selection"] == "accepted_r28_review" for row in images if row["image_id"] in donors),
            "omission donor is outside eligible reviewed training photos")
    counts = {"images": len(images), "objects": sum(row["objects"] for row in images),
              "empty_images": len(empties), **{name: sum(row[name] for row in images)
              for name in ("zero_corner_fronts", "one_corner_fronts", "two_corner_fronts")}}
    require(counts["objects"] == 3784, "effective R28 training object count differs")
    cohort_counts = {name: sum(row[name] for row in images if row["image_id"] in COHORT)
                     for name in ("zero_corner_fronts", "one_corner_fronts", "two_corner_fronts")}
    require(cohort_counts == {"zero_corner_fronts": 33, "one_corner_fronts": 85, "two_corner_fronts": 176},
            "fixed cohort visibility counts differ")
    result = {"image_ids": ids, "cohort_image_ids": list(COHORT), "empty_image_ids": empties,
              "protected_image_ids": protected, "omission_pool_image_ids": donors,
              "images": images, "counts": counts}
    result["sha256"] = digest_json(result)
    return result


def omission_order(inventory, seed):
    result = list(inventory["omission_pool_image_ids"])
    random.Random(seed).shuffle(result)
    return result


def substitutions(inventory, seed, epoch):
    require(type(epoch) is int and 1 <= epoch <= 30, "sampling epoch must be between 1 and 30")
    donors, cohort = omission_order(inventory, seed), inventory["cohort_image_ids"]
    start = (epoch - 1) * len(cohort)
    return {donors[(start + index) % len(donors)]: image_id for index, image_id in enumerate(cohort)}


def planned_exposure(root, context=None):
    context = context or load_protocol(root)
    inventory = sampling_inventory(root, context)
    document = context["document"]
    roles = {}
    for role in ROLES:
        exposures = Counter({image_id: 0 for image_id in inventory["image_ids"]})
        for epoch in range(1, document["execution"]["epochs"] + 1):
            mapping = substitutions(inventory, document["sampling"]["seed"], epoch) if role == "candidate" else {}
            exposures.update(mapping.get(image_id, image_id) for image_id in inventory["image_ids"])
        roles[role] = {"images": sum(exposures.values()), "batches": 1050,
                       "per_image": dict(sorted(exposures.items())),
                       "object_presentations_by_locatable_corners": {
                           str(index): sum(exposures[row["image_id"]] * row[name] for row in inventory["images"])
                           for index, name in enumerate(("zero_corner_fronts", "one_corner_fronts", "two_corner_fronts"))}}
    return {"inventory_sha256": inventory["sha256"], "epochs": 30,
            "omission_order": omission_order(inventory, document["sampling"]["seed"]), "roles": roles}


class PartialFrontSampling:
    """Wrap the ordinary sampler and record consumed training identities."""

    def __init__(self, root, context, role):
        require(role in ROLES, "unknown sampling role")
        self.context, self.role = context, role
        self.inventory = sampling_inventory(root, context)
        self.seed = context["document"]["sampling"]["seed"]
        self.epochs = []
        self.active_epoch = 0
        self.exposures = Counter({image_id: 0 for image_id in self.inventory["image_ids"]})
        self.base_sampler_class = None

    def install(self, loader):
        from ultralytics.data.build import InfiniteDataLoader
        import numpy as np
        require(loader.num_workers == 0 and loader.batch_size == 4 and loader.drop_last is False,
                "sampling requires the fixed single-process batch-four loader")
        dataset = loader.dataset
        ids = [Path(path).stem for path in dataset.im_files]
        require(len(ids) == 137 and set(ids) == set(self.inventory["image_ids"]), "training loader population differs")
        require(np.all(dataset.batch_shapes == dataset.batch_shapes[0]),
                "the fixed sampling experiment requires equal rectangular batch shapes")
        index_by_id = {image_id: index for index, image_id in enumerate(ids)}
        base_sampler, owner = loader.sampler, self
        self.base_sampler_class = type(base_sampler).__name__

        class ExposureSampler:
            def __len__(self):
                return 137

            def __iter__(self):
                ordinary = list(base_sampler)
                require(len(ordinary) == 137 and set(ordinary) == set(range(137)),
                        "ordinary sampler must visit each training image once")
                names = [ids[index] for index in ordinary]
                mapped = owner.plan_epoch(names)
                return iter(index_by_id[name] for name in mapped)

        loader.close()
        return InfiniteDataLoader(dataset=dataset, batch_size=loader.batch_size, shuffle=False,
                                  num_workers=0, sampler=ExposureSampler(), pin_memory=loader.pin_memory,
                                  collate_fn=loader.collate_fn, worker_init_fn=loader.worker_init_fn,
                                  generator=loader.generator, drop_last=False)

    def start_epoch(self, epoch):
        require(epoch == len(self.epochs) + 1 and 1 <= epoch <= 30, "sampling epochs are not contiguous")
        if self.epochs:
            self.finish_epoch(epoch - 1)
        self.active_epoch = epoch

    def plan_epoch(self, image_ids):
        require(self.active_epoch == len(self.epochs) + 1, "sampler ran outside its training epoch")
        mapping = substitutions(self.inventory, self.seed, self.active_epoch) if self.role == "candidate" else {}
        self.epochs.append({"epoch": self.active_epoch, "base_image_ids": list(image_ids),
                            "observed_image_ids": [], "batches_completed": 0})
        return [mapping.get(image_id, image_id) for image_id in image_ids]

    def record_batch(self, image_ids, epoch):
        require(self.epochs and self.epochs[-1]["epoch"] == epoch, "consumed batch has no matching sampling epoch")
        record = self.epochs[-1]
        mapping = substitutions(self.inventory, self.seed, epoch) if self.role == "candidate" else {}
        planned = [mapping.get(image_id, image_id) for image_id in record["base_image_ids"]]
        offset = len(record["observed_image_ids"])
        require(list(image_ids) == planned[offset:offset + 4] and len(image_ids) in (1, 4),
                "actual training images differ from the fixed sampling plan")
        record["observed_image_ids"].extend(image_ids)
        record["batches_completed"] += 1
        self.exposures.update(image_ids)

    def finish_epoch(self, epoch):
        require(self.epochs and self.epochs[-1]["epoch"] == epoch
                and len(self.epochs[-1]["observed_image_ids"]) == 137
                and self.epochs[-1]["batches_completed"] == 35, "sampling epoch did not complete its fixed budget")

    def state(self):
        return {"schema_version": 1, "role": self.role, "protocol_sha256": self.context["sha256"],
                "inventory_sha256": self.inventory["sha256"],
                "sampling_spec_sha256": digest_json(self.context["document"]["sampling"]),
                "base_sampler_class": self.base_sampler_class,
                "epochs": self.epochs, "images_seen": sum(self.exposures.values()),
                "batches_seen": sum(row["batches_completed"] for row in self.epochs),
                "per_image": dict(sorted(self.exposures.items()))}


def validate_state(state, root, context, role, *, complete=False):
    inventory = sampling_inventory(root, context)
    require(isinstance(state, dict) and state.get("schema_version") == 1 and state.get("role") == role
            and state.get("protocol_sha256") == context["sha256"]
            and state.get("inventory_sha256") == inventory["sha256"]
            and state.get("sampling_spec_sha256") == digest_json(context["document"]["sampling"]),
            "sampling receipt identity differs")
    require(state.get("base_sampler_class") in ("RandomSampler", "SequentialSampler"),
            "sampling did not observe a supported ordinary loader sampler")
    epochs = state["epochs"]
    require(isinstance(epochs, list) and 1 <= len(epochs) <= 30, "missing sampling exposure")
    exposures = Counter({image_id: 0 for image_id in inventory["image_ids"]})
    batches = 0
    for epoch, record in enumerate(epochs, 1):
        require(record["epoch"] == epoch and len(record["base_image_ids"]) == 137
                and set(record["base_image_ids"]) == set(inventory["image_ids"]), "base image order is not a permutation")
        mapping = substitutions(inventory, context["document"]["sampling"]["seed"], epoch) if role == "candidate" else {}
        planned = [mapping.get(image_id, image_id) for image_id in record["base_image_ids"]]
        observed = record["observed_image_ids"]
        require(0 < len(observed) <= 137 and observed == planned[:len(observed)], "consumed image order differs from plan")
        require(record["batches_completed"] == math.ceil(len(observed) / 4)
                and (len(observed) == 137 or len(observed) % 4 == 0), "sampling batch accounting differs")
        if complete or epoch < len(epochs):
            require(len(observed) == 137, "an incomplete sampling epoch was recorded as complete")
        exposures.update(observed)
        batches += record["batches_completed"]
    require(state["images_seen"] == sum(exposures.values()) and state["batches_seen"] == batches
            and _same(state["per_image"], dict(sorted(exposures.items()))), "sampling totals differ from consumed images")
    if complete:
        expected = planned_exposure(root, context)["roles"][role]
        require(len(epochs) == 30 and state["images_seen"] == expected["images"]
                and state["batches_seen"] == expected["batches"]
                and _same(state["per_image"], expected["per_image"]), "completed exposure differs from all thirty epochs")
    return {"images_seen": state["images_seen"], "batches_seen": batches, "epochs_seen": len(epochs)}


def metadata_preflight(root):
    root = Path(root).resolve()
    weights, card_hash = _reference_model(root)
    return {"root": str(root), "weights": str(weights), "model_sha256": BASELINE_SHA256,
            "model_card_sha256": card_hash, "manifest_sha256": file_sha256(root / "data/manifest.json")}


def verify_trial(root, run, role, context=None):
    from pose_checkpoint import (_check_runtime_v2, _check_staging, _contained_path,
                                 check_training_settings, training_label_identity)
    from stage_pose_training import MANIFEST

    root, run = Path(root).resolve(), Path(run).resolve()
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(receipt["purpose"] == "resource-trial" and receipt["status"] == "trial-complete"
            and receipt["arm"] == "candidate" and receipt["checkpoint_selection"] == "none",
            "saved run is not a completed sampling resource trial")
    require(_same(receipt["experiment"], experiment_identity(context, role)), "trial experiment binding differs")
    validate_state(receipt["sampling"], root, context, role, complete=False)
    expected = context["document"]["execution"]["resource_trial_batches"]
    require(receipt["batches_completed"] == receipt["sampling"]["batches_seen"] == expected
            and receipt["images_processed"] == receipt["sampling"]["images_seen"] == expected * 4
            and receipt["optimizer_steps"] > 0 and receipt["optimizer_parameter_change_observed"] is True,
            "trial did not establish the fixed real-training exposure and optimizer update")
    require(receipt["execution"]["exit_code"] == 0 and receipt["execution"]["stop_reason"] is None
            and receipt["inputs_reverified_after_worker"] is True, "trial stopped or lost verified inputs")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"], "saved trial source differs")
    require(all(receipt[key] == value for key, value in recipe_identity(context["document"]["recipe_id"]).items())
            and receipt["initial_checkpoint_sha256"] == context["document"]["initial_checkpoint_sha256"],
            "saved trial recipe or initialization differs")
    staging_path = _contained_path(run, MANIFEST)
    require(file_sha256(staging_path) == receipt["staging_manifest_sha256"], "saved trial staging hash differs")
    staging = read_json(staging_path)
    preflight = metadata_preflight(root)
    variants = _check_staging(preflight, staging)
    require(receipt["reference_manifest_sha256"] == preflight["manifest_sha256"]
            and receipt["model_card_sha256"] == preflight["model_card_sha256"], "saved trial source metadata differs")
    document = context["document"]
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"]
            and _same(receipt["training_labels"], training_label_identity(staging, "candidate")),
            "saved trial did not use the accepted R28 candidate label arm")
    variant = variants["candidate"]
    require(receipt["data_yaml_sha256"] == variant["data_yaml_sha256"], "saved trial selected a different dataset YAML")
    stage_root = Path(staging["output_root"])
    require(stage_root.is_absolute() and ".." not in stage_root.parts and stage_root.name == "staged-data",
            "saved trial staging location differs from the run structure")
    settings = receipt["training_settings"]
    model = Path(settings["model"])
    model_relative = Path(preflight["weights"]).relative_to(root)
    require(model.is_absolute() and ".." not in model.parts
            and model.parts[-len(model_relative.parts):] == model_relative.parts,
            "saved trial model path differs from original initialization")
    check_training_settings(settings, document["recipe_id"], model=model,
                            data=stage_root / variant["data_yaml"], device=settings["device"], project=stage_root.parent)
    _check_runtime_v2(receipt["runtime"], settings)
    log = _contained_path(run, receipt["log"]["path"])
    require(log.stat().st_size > 0 and file_sha256(log) == receipt["log"]["sha256"], "saved trial log hash differs")
    samples = receipt["batch_samples"]
    require(len(samples) == expected and len(receipt["sampling"]["epochs"]) == 1,
            "saved trial lacks the four observed training batches")
    observed = receipt["sampling"]["epochs"][0]["observed_image_ids"]
    for index, sample in enumerate(samples):
        shape = sample["shape"]
        require(len(shape) == 4 and shape[:2] == [4, 3] and min(shape[2:]) > 0 and max(shape[2:]) == 1280
                and sample["epoch"] == 1 and sample["batch"] == index + 1
                and sample["image_ids"] == observed[index * 4:(index + 1) * 4],
                "saved trial batch shape or consumed identities differ from the fixed recipe")
    return receipt


def verify_run(root, run, role, context=None, require_weights=True):
    root, run = Path(root).resolve(), Path(run).resolve()
    require(require_weights is True, "returned sampling runs require actual checkpoint bytes")
    context = context or load_protocol(root)
    receipt = read_json(run / "training-receipt.json")
    require(_same(receipt.get("experiment"), experiment_identity(context, role)), "completed sampling role differs")
    state = receipt.get("sampling")
    validate_state(state, root, context, role, complete=True)
    require(state["images_seen"] == receipt["images_processed"] and state["batches_seen"] == receipt["batches_completed"],
            "sampling hook did not observe every completed training batch")
    execution = context["document"]["execution"]
    require(receipt["optimizer_steps"] == execution["expected_optimizer_steps"], "optimizer-update budget differs")
    require(receipt["source"]["files_sha256"] == source_identity(root)["files_sha256"],
            "completed sampling implementation differs from this checkout")
    bundle = collect_arm(metadata_preflight(root), run, "candidate")
    document, staging = context["document"], bundle["staging"]
    require(receipt["training_settings"]["device"].isdecimal()
            and receipt["runtime"]["device"].startswith("cuda:") and receipt["runtime"]["threads"] == 4,
            "full sampling comparison requires the fixed CUDA/four-thread runtime")
    require(staging["source"]["candidate_manifest"] == document["candidate_manifest"]
            and staging["source"]["candidate_manifest_sha256"] == document["candidate_manifest_sha256"],
            "completed sampling label composition differs")
    for entry in bundle["files"]:
        entry["path"] = role + "/" + entry["path"].split("/", 1)[1]
    return bundle


def package_pair(root, control, candidate, output, verify_existing=False, protocol=DEFAULT_PROTOCOL):
    root = Path(root).resolve()
    context = load_protocol(root, protocol)
    runs = {"control": Path(control).resolve(), "candidate": Path(candidate).resolve()}
    require(all(path.is_dir() and not Path(raw).is_symlink() for raw, path in
                ((control, runs["control"]), (candidate, runs["candidate"]))), "run directories must be regular")
    require(not runs["control"].is_relative_to(runs["candidate"])
            and not runs["candidate"].is_relative_to(runs["control"]), "run directories overlap")
    bundles = {role: verify_run(root, run, role, context) for role, run in runs.items()}
    check_pair(bundles["control"], bundles["candidate"])
    receipts = {role: bundle["receipt"] for role, bundle in bundles.items()}
    require(receipts["control"]["training_labels"] == receipts["candidate"]["training_labels"],
            "experimental roles used different training labels")
    for first, second in zip(receipts["control"]["sampling"]["epochs"], receipts["candidate"]["sampling"]["epochs"]):
        require(first["base_image_ids"] == second["base_image_ids"], "ordinary paired sampler orders differ")
    rows = [entry for bundle in bundles.values() for entry in bundle["files"]]
    rows.append({"path": "experiment-protocol.json", "sha256": context["sha256"],
                 "size_bytes": context["path"].stat().st_size, "source_path": context["path"]})
    rows.sort(key=lambda row: row["path"])
    receipt = receipts["control"]
    index = {"schema_version": 1, "kind": "baomai-partial-front-sampling-results",
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": context["sha256"],
             "source": receipt["source"], "runtime": receipt["runtime"], "shared_base_settings": shared_settings(receipt),
             "roles": {role: {"checkpoint_manifest": f"{role}/checkpoint-manifest.json",
                              "experiment": bundle["receipt"]["experiment"],
                              "sampling": bundle["receipt"]["sampling"],
                              "checkpoint_identity": bundle["identity"]} for role, bundle in bundles.items()},
             "files": [{key: value for key, value in row.items() if key != "source_path"} for row in rows],
             "scope": "Completed fixed-budget sampling comparison only; no accuracy approval or model promotion."}
    output = Path(output).absolute()
    require(not output.is_symlink() and output.suffix.lower() == ".zip", "output must be a regular ZIP path")
    output = output.resolve()
    require(all(not output.is_relative_to(path) for path in (root, *runs.values())), "ZIP overlaps source/run directories")
    if verify_existing:
        with zipfile.ZipFile(output) as archive:
            previous = json.loads(archive.read(INDEX_NAME))
        index["created_at_utc"] = previous["created_at_utc"]
        require(_same(previous, index), "existing return index differs from verified runs")
    else:
        require(not output.exists(), "return ZIP already exists")
    index_bytes = (json.dumps(index, indent=2, allow_nan=False) + "\n").encode()
    if verify_existing:
        verify_archive(output, index_bytes, rows)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sampling-return-", dir=output.parent) as temporary:
            staged = Path(temporary) / "results.zip"
            with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                archive.writestr(INDEX_NAME, index_bytes)
                for entry in rows:
                    copy_member(archive, entry)
            verify_archive(staged, index_bytes, rows)
            os.link(staged, output)
    return {"archive": str(output), "sha256": file_sha256(output), "size_bytes": output.stat().st_size,
            "files": len(rows) + 1, "status": "verified", "accuracy_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "package"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
        item.add_argument("--protocol", default=DEFAULT_PROTOCOL)
        if name == "package":
            for option in ("control", "candidate", "output"):
                item.add_argument("--" + option, type=Path, required=True)
            item.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        context = load_protocol(args.root, args.protocol)
        result = {"status": "verified", "protocol_sha256": context["sha256"],
                  "inventory": sampling_inventory(args.root, context),
                  "planned_exposure": planned_exposure(args.root, context), "training_executed": False}
    else:
        result = package_pair(args.root, args.control, args.candidate, args.output, args.verify_existing, args.protocol)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
