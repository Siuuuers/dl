#!/usr/bin/env python3
"""Run a bounded resource trial or one traceable baomai fine-tuning arm."""

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time

from evaluate_pose import ULTRALYTICS_VERSION, _relative_file, file_sha256, preflight_dataset
from pose_checkpoint import (COMPOSED_RECIPE_ID, ENDPOINT_RECIPE_ID, OBSERVED_RECIPE_ID,
                             ORIGINAL_RECIPE_ID, RECIPE_IDS, WARMUP_RECIPE_ID,
                             check_recipe_candidate_kind, check_training_settings, get_recipe, recipe_identity,
                             training_label_identity)
from stage_pose_training import MANIFEST, stage_datasets
from verify_pose_candidate import DEFAULT_CANDIDATE, require


SOURCE_FILES = ("scripts/train_pose_comparison.py", "scripts/pose_checkpoint.py",
                "scripts/stage_pose_training.py", "scripts/verify_pose_candidate.py",
                "scripts/verify_pose_box_candidate.py", "scripts/verify_pose_annotation_candidate.py",
                "scripts/verify_pose_endpoint_candidate.py",
                "scripts/verify_pose_composed_candidate.py",
                "scripts/verify_observed_training_candidate.py",
                "scripts/compose_observed_pose_review.py",
                "scripts/evaluate_pose.py", "scripts/pose_scoring.py")


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return {"revision": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
            "files_sha256": {name: file_sha256(root / name) for name in SOURCE_FILES}}


def descriptor(path, run, *, checkpoint=False):
    result = {"path": path.relative_to(run).as_posix(), "sha256": file_sha256(path)}
    if checkpoint:
        result["size_bytes"] = path.stat().st_size
    return result


def verify_input_files(run, staging):
    base = run / "staged-data"
    for arm, variant in staging["variants"].items():
        yaml_path = base / variant["data_yaml"]
        require(not yaml_path.is_symlink() and file_sha256(yaml_path) == variant["data_yaml_sha256"],
                "training dataset YAML changed")
        for item in variant["files"]:
            path = base / arm / item["path"]
            require(not path.is_symlink() and file_sha256(path) == item["output_sha256"],
                    f"training input changed: {arm}/{item['path']}")
    require(file_sha256(base / MANIFEST) == file_sha256(run / MANIFEST), "staging receipt changed")


def resolved_runtime(torch, model, device):
    """Observe the effective device/backend after setup without changing any flags."""
    cuda = device.type == "cuda"
    four_dimensional = [parameter for parameter in model.parameters() if parameter.ndim == 4]
    return {"device": str(device), "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
            "device_capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
            "backend": {
                "model_channels_last": bool(four_dimensional) and all(parameter.is_contiguous(
                    memory_format=torch.channels_last) for parameter in four_dimensional),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_version": torch.backends.cudnn.version(),
                "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark}}


def optimizer_learning_rates(optimizer):
    return [{"index": index, "parameter_group": group["param_group"], "lr": float(group["lr"]),
             "initial_lr": float(group["initial_lr"])} for index, group in enumerate(optimizer.param_groups)]


def worker_training_labels(config, staging):
    """Reject a composed arm/label mismatch before importing a training runtime."""
    labels = training_label_identity(staging, config["arm"])
    if config["recipe_id"] in (COMPOSED_RECIPE_ID, OBSERVED_RECIPE_ID) or labels is not None:
        check_recipe_candidate_kind(config["recipe_id"], staging["source"].get("candidate_kind"))
        require(labels is not None and config.get("training_labels") == labels,
                "worker control/treatment label identity differs from staging")
        require(config["candidate_manifest"] == staging["source"]["candidate_manifest"],
                "worker composed proposal differs from staging")
    return labels


def run_worker(config, input_transform=None):
    recipe = get_recipe(config["recipe_id"])
    declared_recipe = recipe_identity(config["recipe_id"])
    require(all(config.get(key) == value for key, value in declared_recipe.items()),
            "worker configuration recipe identity differs")
    require(config["recipe_id"] != WARMUP_RECIPE_ID or config["arm"] == "reference",
            "warmup experiment requires the reference label arm")
    require(config["recipe_id"] != ENDPOINT_RECIPE_ID or config["arm"] == "candidate",
            "endpoint experiment requires the candidate label arm")
    run = Path(config["output"])
    staging = json.loads((run / MANIFEST).read_text())
    arm = config["arm"]
    labels = worker_training_labels(config, staging)

    import resource
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import SETTINGS

    SETTINGS.update({key: False for key in
                     ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")})
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    arm_root = run / "staged-data" / arm
    progress = {**declared_recipe, "status": "starting", "completed_epochs": [], "optimizer_steps": 0,
                "batches_completed": 0, "images_processed": 0, "batch_samples": [],
                "optimizer_step_samples": [],
                "training_returned": False, "finalization_completed": False,
                "optimizer_parameter_change_observed": False, "validation_executed": False,
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "ultralytics": importlib.metadata.version("ultralytics"),
                            "device": config["device"], "torch_cuda": torch.version.cuda,
                            "threads": config["threads"], "interop_threads": 1,
                            "platform": platform.platform(),
                            "packages": {name: importlib.metadata.version(name) for name in
                                         ("torchvision", "numpy", "Pillow", "opencv-python", "psutil")}}}
    if labels is not None:
        progress["training_labels"] = labels
    started = time.perf_counter()

    def save_progress():
        if input_transform is not None:
            progress["input_corruption"] = input_transform.state()
        progress["worker_wall_seconds"] = time.perf_counter() - started
        progress["worker_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write_json(run / "training-progress.json", progress)

    class TrialFinished(Exception):
        pass

    class RecordedPoseTrainer(PoseTrainer):
        def __init__(self, *args, **kwargs):
            self.epoch_batch_count = 0
            self.epoch_image_count = 0
            self.current_batch = None
            self.completed_training_epoch = None
            super().__init__(*args, **kwargs)
            torch.set_num_threads(config["threads"])
            require(self.world_size in (0, 1) and not self.ddp, "only one explicit training device is supported")
            self.add_callback("on_train_epoch_start", self.record_epoch_start)
            self.add_callback("on_train_batch_start", self.record_batch_start)
            self.add_callback("on_train_batch_end", self.record_batch_end)
            self.add_callback("on_train_epoch_end", self.record_epoch_end)
            self.add_callback("on_model_save", self.record_model_save)

        def _build_train_pipeline(self):
            require(self.batch_size == recipe["batch"], "automatic batch-size changes invalidate the comparison")
            super()._build_train_pipeline()
            routed = []
            files = {item["path"]: item["output_sha256"] for item in staging["variants"][arm]["files"]}
            for split, loader in (("train", self.train_loader), ("val", self.test_loader)):
                expected = staging["variants"][arm]["counts"][split]
                require(len(loader.dataset) == expected["images"], f"loader image count changed: {split}")
                labels = loader.dataset.labels
                require(sum(len(item["cls"]) for item in labels) == expected["objects"],
                        f"loader annotation count changed: {split}")
                visible = sum(int((item["keypoints"][:, :, 2] == 2).sum()) for item in labels)
                require(visible == expected["visible_endpoints"], f"loader visible-point count changed: {split}")
                for image, label in zip(loader.dataset.im_files, img2label_paths(loader.dataset.im_files)):
                    for raw in (image, label):
                        path = Path(raw).resolve()
                        require(path.is_relative_to(arm_root), "loader resolved outside its selected arm")
                        name = path.relative_to(arm_root).as_posix()
                        require(name in files and file_sha256(path) == files[name], "loader input content mismatch")
                routed.append({"split": split, "images": len(loader.dataset),
                               "objects": expected["objects"], "visible_endpoints": visible,
                               "batches": len(loader), "batch_size": loader.batch_size})
            progress["loader_verification"] = routed
            progress["training_settings"] = dict(vars(self.args))
            check_training_settings(progress["training_settings"], config["recipe_id"],
                                    model=config["initial_weights"], data=arm_root / "data.yaml",
                                    device=config["device"], project=run)
            progress["runtime"].update(resolved_runtime(torch, self.model, self.device))
            save_progress()

        def preprocess_batch(self, batch):
            result = super().preprocess_batch(batch)
            if input_transform is not None:
                input_transform(result, epoch=self.epoch + 1)
            self.current_batch = {"shape": list(result["img"].shape),
                                  "image_ids": [Path(name).stem for name in result["im_file"]],
                                  "objects": len(result["cls"])}
            return result

        def record_epoch_start(self, trainer):
            self.epoch_batch_count = self.epoch_image_count = 0
            self.completed_training_epoch = None

        def record_batch_start(self, trainer):
            self.batch_started = time.perf_counter()

        def optimizer_step(self):
            require(bool(torch.isfinite(self.loss).all()), "nonfinite training loss")
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients),
                    "missing or nonfinite training gradients")
            witness = None
            for group in self.optimizer.param_groups:
                if group["lr"] <= 0:
                    continue
                for parameter in group["params"]:
                    if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
                        witness = parameter
                        break
                if witness is not None:
                    break
            require(witness is not None, "no nonzero gradient with positive learning rate")
            before = witness.detach().clone()
            rates = optimizer_learning_rates(self.optimizer)
            super().optimizer_step()
            require(bool(torch.isfinite(witness).all()), "nonfinite parameter after optimizer update")
            changed = bool(torch.any(before != witness.detach()))
            progress["optimizer_parameter_change_observed"] |= changed
            progress["optimizer_steps"] += 1
            if len(progress["optimizer_step_samples"]) < 8:
                progress["optimizer_step_samples"].append({"optimizer_step": progress["optimizer_steps"],
                    "epoch": self.epoch + 1, "batch": self.epoch_batch_count + 1,
                    "global_batch": progress["batches_completed"] + 1,
                    "learning_rates": rates, "witness_parameter_changed": changed})

        def record_batch_end(self, trainer):
            require(self.current_batch is not None and bool(torch.isfinite(self.loss).all()),
                    "batch did not complete with finite loss")
            self.epoch_batch_count += 1
            self.epoch_image_count += self.current_batch["shape"][0]
            progress["batches_completed"] += 1
            progress["images_processed"] += self.current_batch["shape"][0]
            if len(progress["batch_samples"]) < config["trial_batches"]:
                progress["batch_samples"].append({**self.current_batch,
                    "epoch": self.epoch + 1, "batch": self.epoch_batch_count,
                    "wall_seconds": time.perf_counter() - self.batch_started,
                    "loss": float(self.loss.detach()),
                    "loss_components": {key: float(value.detach()) for key, value in self.loss_items.items()},
                    "optimizer_steps_total": progress["optimizer_steps"],
                    "learning_rates": optimizer_learning_rates(self.optimizer)})
            progress["status"] = "training"
            save_progress()
            if config["trial"] and progress["batches_completed"] >= config["trial_batches"]:
                require(progress["optimizer_steps"] > 0 and progress["optimizer_parameter_change_observed"],
                        "resource trial did not establish an actual optimizer update")
                raise TrialFinished()

        def record_epoch_end(self, trainer):
            require(self.epoch_batch_count == len(self.train_loader)
                    and self.epoch_image_count == len(self.train_loader.dataset),
                    "partial training epoch cannot be recorded as complete")
            self.completed_training_epoch = self.epoch + 1

        def record_model_save(self, trainer):
            require(self.completed_training_epoch == len(progress["completed_epochs"]) + 1,
                    "saved epochs are not a contiguous completed sequence")
            require(self.last.is_file(), "last.pt was not saved")
            progress["completed_epochs"].append(self.completed_training_epoch)
            save_progress()

        def validate(self):
            result = super().validate()
            progress["validation_executed"] = True
            return result

        def final_eval(self):
            require(not config["trial"] and progress["completed_epochs"] == list(range(1, 31)),
                    "only the completed 30-epoch comparison can finalize")
            super().final_eval()
            progress["finalization_completed"] = True
            save_progress()

    model = None
    try:
        require(importlib.metadata.version("ultralytics") == ULTRALYTICS_VERSION, "pinned Ultralytics required")
        require(config["device"] == "cpu" or torch.cuda.is_available(), "requested accelerator is unavailable")
        model = YOLO(config["initial_weights"])
        require(model.task == "pose" and model.names == {0: "cassette_front"}
                and list(model.model.model[-1].kpt_shape) == [2, 3], "initial loaded pose contract differs")
        settings = {**recipe, "data": str(arm_root / "data.yaml"), "device": config["device"],
                    "workers": 0, "plots": False, "verbose": False,
                    "project": str(run), "name": "fit", "exist_ok": False}
        model.train(trainer=RecordedPoseTrainer, **settings)
        progress["training_returned"] = True
        progress["status"] = "complete"
    except TrialFinished:
        progress["status"] = "trial-complete"
    except Exception as error:
        progress["status"] = "failed"
        progress["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if model is not None and getattr(model, "trainer", None) is not None:
            progress["training_settings"] = dict(vars(model.trainer.args))
            for name in ("train_loader", "test_loader"):
                loader = getattr(model.trainer, name, None)
                if loader is not None and hasattr(loader, "close"):
                    loader.close()
        save_progress()
    return 0


def monitor_worker(command, run, config):
    import psutil
    env = {**os.environ, "YOLO_OFFLINE": "true", "YOLO_CONFIG_DIR": str(run / "runtime-settings"),
           "MPLCONFIGDIR": str(run / "matplotlib"), "OMP_NUM_THREADS": str(config["threads"]),
           "MKL_NUM_THREADS": str(config["threads"]), "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    peak, reason = 0, None
    with (run / "train.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while process.poll() is None:
            try:
                main = psutil.Process(process.pid)
                rss = sum(item.memory_info().rss for item in [main, *main.children(recursive=True)] if item.is_running())
                peak = max(peak, rss)
                if rss > config["max_rss_mib"] * 2**20:
                    reason = "process RSS limit exceeded"
                elif psutil.virtual_memory().available < 512 * 2**20:
                    reason = "system available memory fell below 512 MiB"
            except psutil.NoSuchProcess:
                pass
            if time.perf_counter() - started > config["max_seconds"]:
                reason = "wall-time limit exceeded"
            if reason:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.2)
        code = process.wait()
    return {"exit_code": code, "stop_reason": reason, "wall_seconds": time.perf_counter() - started,
            "sampled_peak_process_tree_rss_mib": peak / 2**20,
            "rss_sampling_interval_seconds": 0.2,
            "max_seconds": config["max_seconds"], "max_rss_mib": config["max_rss_mib"]}


def run_parent(args):
    from pose_checkpoint import write_checkpoint_manifest
    root, run = args.root.resolve(), args.output.absolute()
    require(root == Path(__file__).resolve().parents[1], "--root must identify the executing repository")
    require(not run.exists() and not run.is_symlink(), "run output must be a fresh directory")
    run = run.resolve()
    require(not run.is_relative_to(root) and not root.is_relative_to(run),
            "training output must be outside the repository")
    require(args.threads > 0 and args.max_seconds > 0 and args.max_rss_mib > 0, "resource limits must be positive")
    require(1 <= args.trial_batches <= 8, "trial batches must be between 1 and 8")
    require(args.device == "cpu" or args.device.isdecimal(), "select cpu or one CUDA device index explicitly")
    declared_recipe = recipe_identity(args.recipe)
    require(args.recipe != WARMUP_RECIPE_ID or args.arm == "reference",
            "warmup experiment requires the reference label arm")
    require(args.recipe != ENDPOINT_RECIPE_ID or args.arm == "candidate",
            "endpoint experiment requires the candidate label arm")
    candidate_name = str(getattr(args, "candidate", DEFAULT_CANDIDATE))
    require(Path(candidate_name).parts[:2] == ("data", "candidates") and ".." not in Path(candidate_name).parts,
            "candidate must be a repository-relative data/candidates manifest")
    candidate = _relative_file(root, candidate_name)
    proposal = json.loads(candidate.read_text(encoding="utf-8"))
    require(isinstance(proposal, dict), "candidate manifest must be an object")
    check_recipe_candidate_kind(args.recipe, proposal.get("kind"))
    require(importlib.metadata.version("ultralytics") == ULTRALYTICS_VERSION, "pinned Ultralytics required")
    preflight = preflight_dataset(root, require_weights=True)
    identity = source_identity(root)
    require(args.trial or not identity["working_tree_status"], "full comparison requires a clean source revision")
    config = {"root": str(root), "output": str(run), "trial": args.trial,
              "trial_batches": args.trial_batches, "arm": args.arm, "device": args.device,
              "threads": args.threads, "max_seconds": args.max_seconds,
              "max_rss_mib": args.max_rss_mib, "initial_weights": preflight["weights"],
              "candidate_manifest": candidate_name, **declared_recipe}
    run.mkdir(parents=True, exist_ok=False)
    stage_datasets(root, candidate, run / "staged-data")
    shutil.copyfile(run / "staged-data" / MANIFEST, run / MANIFEST)
    staging = json.loads((run / MANIFEST).read_text())
    labels = training_label_identity(staging, args.arm)
    if args.recipe in (COMPOSED_RECIPE_ID, OBSERVED_RECIPE_ID):
        require(labels is not None, "controlled training requires the staged supervision identity")
        config["training_labels"] = labels
    write_json(run / "run-config.json", config)
    execution = monitor_worker([sys.executable, str(Path(__file__).resolve()),
                                "--worker", str(run / "run-config.json")], run, config)
    progress_path = run / "training-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"status": "failed"}
    receipt = {**progress, "schema_version": 2, "purpose": "resource-trial" if args.trial else "paired-finetune",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "arm": args.arm,
               "initial_checkpoint_sha256": preflight["model_sha256"],
               "reference_manifest_sha256": preflight["manifest_sha256"],
               "model_card_sha256": preflight["model_card_sha256"],
               "staging_manifest_sha256": file_sha256(run / MANIFEST),
               "data_yaml_sha256": staging["variants"][args.arm]["data_yaml_sha256"],
               "source": identity, "execution": execution, **declared_recipe,
               "checkpoint_selection": "final_epoch" if not args.trial else "none",
               "inputs_reverified_after_worker": False,
               "log": descriptor(run / "train.log", run)}
    if labels is not None:
        receipt["training_labels"] = labels
    valid_execution = execution["exit_code"] == 0 and execution["stop_reason"] is None
    try:
        require(all(progress.get(key) == value for key, value in declared_recipe.items()),
                "worker reported a different or missing recipe identity")
        if labels is not None:
            require(progress.get("training_labels") == labels, "worker reported a different or missing control/treatment label identity")
            require(file_sha256(candidate) == staging["source"]["candidate_manifest_sha256"],
                    "composed candidate manifest changed during execution")
            control = staging["source"]["control"]
            require(file_sha256(_relative_file(root, control["manifest"])) == control["manifest_sha256"],
                    "endpoint control manifest changed during execution")
            if args.recipe == OBSERVED_RECIPE_ID:
                protocol = staging["source"]["training_protocol"]
                require(file_sha256(_relative_file(root, protocol["path"])) == protocol["sha256"],
                        "observed training protocol changed during execution")
        verify_input_files(run, staging)
        require(identity == source_identity(root), "training source code or revision changed during execution")
        require(file_sha256(preflight["weights"]) == preflight["model_sha256"], "initial checkpoint changed")
        require(file_sha256(root / "data/manifest.json") == preflight["manifest_sha256"], "reference manifest changed")
        receipt["inputs_reverified_after_worker"] = True
    except (ValueError, OSError) as error:
        valid_execution = False
        receipt["input_verification_error"] = str(error)
    if not valid_execution:
        receipt["status"] = "failed"
    if valid_execution and not args.trial:
        receipt["checkpoint"] = descriptor(run / "fit/weights/last.pt", run, checkpoint=True)
        receipt["results"] = descriptor(run / "fit/results.csv", run)
    write_json(run / "training-receipt.json", receipt)
    if valid_execution and not args.trial:
        try:
            write_checkpoint_manifest(run, preflight)
        except (ValueError, OSError, KeyError, TypeError) as error:
            receipt["status"] = "failed"
            receipt["checkpoint_manifest_error"] = str(error)
            write_json(run / "training-receipt.json", receipt)
            raise
    success = valid_execution and receipt["status"] == ("trial-complete" if args.trial else "complete")
    print(json.dumps({"output": str(run), "purpose": receipt["purpose"], "status": receipt["status"],
                      "execution": execution, "batches_completed": receipt.get("batches_completed", 0),
                      "optimizer_steps": receipt.get("optimizer_steps", 0),
                      "completed_epochs": receipt.get("completed_epochs", []),
                      "checkpoint_manifest_created": success and not args.trial}, allow_nan=False))
    return 0 if success else 1


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) == 2 and arguments[0] == "--worker":
        return run_worker(json.loads(Path(arguments[1]).read_text()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("reference", "candidate"), required=True,
                        help="for the composed recipe: reference is the endpoint control, candidate is the composed treatment")
    parser.add_argument("--recipe", choices=RECIPE_IDS, default=ORIGINAL_RECIPE_ID,
                        help="fixed recipe; the historical warmup experiment requires --arm reference")
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE,
                        help="repository-relative candidate manifest; defaults to the original deletion-only proposal")
    parser.add_argument("--device", required=True, help="cpu or one CUDA index; same device for both arms")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trial", action="store_true", help="stop after bounded real training batches; no eligible checkpoint")
    mode.add_argument("--train", action="store_true", help="complete the fixed 30-epoch recipe")
    parser.add_argument("--trial-batches", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, required=True, help="subprocess wall-time budget including setup/finalization")
    parser.add_argument("--max-rss-mib", type=int, default=6000, help="sampled process-tree RSS stop threshold")
    args = parser.parse_args(arguments)
    try:
        return run_parent(args)
    except (ValueError, OSError, KeyError, TypeError, importlib.metadata.PackageNotFoundError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
