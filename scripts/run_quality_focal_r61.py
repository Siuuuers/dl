#!/usr/bin/env python3
"""Paired native-versus-quality-focal training with unchanged pose supervision."""
import argparse
import ast
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import run_partial_classification_r51 as base
import run_additional_front_pilot_r47 as r47
import rebuild_additional_front_pilot_r45 as r45
import audit_additional_front_pilot_r47_results as prior
import run_training_label_comparison_r61 as shared
from evaluate_pose import INFERENCE, _prediction_records
from predict_pose_folder import postclip_nms
from detection_ranking_r61 import score_detection_ranking
from pose_checkpoint import get_recipe, recipe_identity, check_training_settings, check_loaded_pose_model

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'baomai-quality-focal-r61'
ROLES = ('reference', 'candidate')
BETAS = {'reference': 0, 'candidate': 2}
RECIPE, INIT, VAL = base.RECIPE, base.INIT, base.VAL
COUNTS = base.COUNTS
need, digest, sha, checked = r45.need, r45.digest, r45.sha, r45.checked
encoded, put, save, read, rev, counts, labels = r45.encoded, r45.put, r45.save, r45.read, r45.rev, r45.counts, r45.labels
verify_loader, dataset_config, image_size = base.verify_loader, base.dataset_config, base.image_size
backup_paths, validate_backup = r47.backup_paths, r47.validate_backup


def local_source_files(root):
    pending = {Path(__file__).name, 'quality_focal_r61.py', 'check_quality_focal_r61.py',
               'run_training_label_comparison_r61.py'}
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = root / 'scripts' / name
        need(path.is_file() and not path.is_symlink(), 'Missing local source: ' + name)
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [v.name for v in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = module.split('.')[0] + '.py'
                if (root / 'scripts' / candidate).is_file() and candidate not in seen:
                    pending.add(candidate)
    return {'scripts/' + name: sha(root / 'scripts' / name) for name in sorted(seen)}


def read_refs(root, entries=None):
    original = read(checked(root / 'data/manifest.json', r47.ORIGINAL_MANIFEST_SHA))
    if entries is not None:
        expected = {row['id']: row['image_sha256'] for row in original['records'] if row['split'] == 'val'}
        actual = {row['id']: row['image_sha256'] for row in entries if row['split'] == 'val'}
        need(actual == expected, 'Evaluation photograph identities differ')
    return shared.read_references(root, original['records'])[0]


def label_plan(root, archive):
    from quality_focal_r61 import SPEC, CRITERION_CONTRACT
    plan, payload, provenance = base.label_plan(root, archive)
    references, reference_sources = shared.read_references(root, read(root / 'data/manifest.json')['records'])
    plan.update(schema='baomai-quality-focal-r61-v1', experiment='quality-focal-r61',
        source_files_sha256=local_source_files(root), references=reference_sources,
        evaluation_protocol=shared.PROTOCOL,
        treatment=dict(focal_beta=BETAS, native_equivalent_reference=True,
            weighted_term='native classification BCE multiplied by absolute probability-target difference to beta',
            loss_spec=SPEC, criterion_contract=CRITERION_CONTRACT,
            same_images_and_labels=True, same_initializer=True, equal_image_exposure=True,
            equal_optimizer_updates=True, no_label_changes=True, automatic_promotion=False,
            interpretation='QFL-style modulation of existing native soft targets; not a full GFL detector.'))
    return json.loads(encoded(plan)), payload, provenance


def source_check(root, revision, script_sha):
    """Fail closed on first setup and every resumed command, before GPU imports."""
    need(re.fullmatch(r"[0-9a-f]{40}", revision or "") is not None, "Source revision is not published")
    need(re.fullmatch(r"[0-9a-f]{64}", script_sha or "") is not None, "Script hash is not published")
    need(rev(root) == revision, "Wrong checkout preserved; choose a fresh R61 base directory")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    need(not status.strip(), "Changed checkout preserved; choose a fresh R61 base directory")
    need(Path(__file__).resolve() == (root / "scripts" / Path(__file__).name).resolve(), "Run from the pinned checkout")
    checked(Path(__file__), script_sha)
    tree = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], cwd=root, text=True)
    tracked = {}
    for line in tree.splitlines():
        meta, relative = line.split("\t", 1)
        mode, kind, object_id = meta.split()
        if relative.startswith("scripts/") and relative.endswith(".py"):
            path = root / relative
            need(kind == "blob" and mode in ("100644", "100755") and path.is_file() and not path.is_symlink(), "Missing source closure: " + relative)
            data = path.read_bytes()
            need(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == object_id, "Source Git object differs: " + relative)
            tracked[relative] = digest(data)
    closure = local_source_files(root)
    need(all(tracked.get(name) == value for name, value in closure.items()), "Untracked source dependency")
    original = read(checked(root / "data/manifest.json", r47.ORIGINAL_MANIFEST_SHA))
    read_refs(root, [dict(id=r["id"], split=r["split"], image_sha256=r["image_sha256"]) for r in original["records"]])
    print("R61 source closure verified:", revision, len(closure), "Python files", flush=True)
    return dict(source=revision, script_sha256=script_sha, source_files_sha256=closure)


def prepare(root, work, archive, inputs, source_contract=None):
    plan, payload, provenance = label_plan(root, archive)
    if source_contract is not None:
        need(plan["source"] == source_contract["source"] and plan["source_files_sha256"] == source_contract["source_files_sha256"], "Preflight source changed")
    need(not work.resolve().is_relative_to(root.resolve()), "Work directory must be outside source checkout")
    original = read(root / "data/manifest.json")
    put(work / "initializer/cassette_lower_epoch173.pt", checked(inputs / original["model_checkpoint"], INIT).read_bytes())
    for role in ROLES:
        folder = work / "datasets" / role
        for entry in plan["roles"][role]:
            data = checked(inputs / entry["image_source"], entry["image_sha256"]).read_bytes()
            need(image_size(data) == (1280, 720), "Original size differs: " + entry["id"])
            put(folder / entry["image"], data)
            put(folder / entry["label"], payload[entry["id"]])
        put(folder / "data.yaml", encoded(dataset_config(folder)))
    for name, data in provenance.items():
        put(work / "inputs" / name, data)
    put(work / "staging.json", encoded(plan))
    verify_stage(root, work, archive)
    print("R61 CPU input preflight passed: both roles", COUNTS, "; identical 1080 batches / 80 updates / 4260 image presentations", flush=True)
    return plan


def verify_stage(root, work, archive):
    expected, _, provenance = label_plan(root, archive)
    plan = read(work / "staging.json")
    need(plan == expected, "Staged R61 source/data protocol differs")
    checked(work / "initializer/cassette_lower_epoch173.pt", INIT)
    for name, data in provenance.items():
        checked(work / "inputs" / name, digest(data))
    need(plan["roles"]["reference"] == plan["roles"]["candidate"], "Paired input identities differ")
    for role, entries in plan["roles"].items():
        folder = work / "datasets" / role
        grouped = {"train": [], "val": []}
        for entry in entries:
            data = checked(folder / entry["image"], entry["image_sha256"]).read_bytes()
            need(image_size(data) == (1280, 720), "Staged size differs")
            grouped[entry["split"]].append(labels(checked(folder / entry["label"], entry["label_sha256"]).read_bytes(), entry["id"]))
        for field, sub, suffixes in (("image", "images", {".png", ".jpg", ".jpeg"}), ("label", "labels", {".txt"})):
            found = {p.relative_to(folder).as_posix() for p in (folder / sub).rglob("*") if p.is_file() and p.suffix.lower() in suffixes}
            need(found == {e[field] for e in entries}, "Staged file membership differs")
        need({split: list(counts(values)) for split, values in grouped.items()} == plan["counts"][role], "Staged counts differ")
        need(read(folder / "data.yaml") == dataset_config(folder), "Dataset paths/settings differ")
    read_refs(root, plan["roles"]["reference"])
    return plan


def cpu_loader_preflight(root, work, archive):
    """Construct the real pinned datasets/loaders and consume one CPU batch each."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.utils.torch_utils import init_seeds
    from quality_focal_r61 import verify_native_sources
    plan = verify_stage(root, work, archive)
    native = verify_native_sources()
    result = dict(status="passed", source=plan["source"], staging_sha256=sha(work / "staging.json"),
                  native_source_hashes=native, device="cpu", training=False, fresh_inference=False, datasets={})
    torch.set_num_threads(2)
    for role in ROLES:
        folder = work / "datasets" / role
        cfg = get_cfg(overrides={**get_recipe(RECIPE), "task": "pose", "mode": "train", "workers": 0})
        for split in ("train", "val"):
            init_seeds(46, deterministic=True)
            dataset = build_yolo_dataset(cfg, str(folder / "images" / split), 4, dataset_config(folder), mode=split, rect=split == "val", stride=32)
            evidence = verify_loader(folder, plan, role, split, dataset)
            loader = build_dataloader(dataset, 4, 0, shuffle=split == "train", rank=-1, pin_memory=False, device="cpu")
            need(len(loader) == (36 if split == "train" else 5), "CPU loader batch count differs")
            batch = next(iter(loader))
            need(batch["img"].device.type == "cpu" and len(batch["im_file"]) == 4 and batch["img"].shape[0] == 4, "CPU batch construction failed")
            evidence.update(batches=len(loader), sample_image_ids=[Path(p).stem for p in batch["im_file"]], sample_shape=list(batch["img"].shape))
            result["datasets"][role + "/" + split] = evidence
            del loader, dataset, batch
    for split in ("train", "val"):
        need(result["datasets"]["reference/" + split] == result["datasets"]["candidate/" + split], "CPU paired loader behavior differs")
    put(work / "cpu-loader-preflight.json", encoded(result))
    return result


def verify_cpu_gates(root, work, plan):
    from quality_focal_r61 import NATIVE_SOURCE_HASHES, SPEC, CRITERION_CONTRACT, digest_json
    from check_quality_focal_r61 import verify_guard
    loader, guard = read(work / 'cpu-loader-preflight.json'), read(work / 'loss-guard.json')
    need(loader['status'] == 'passed' and loader['source'] == plan['source']
         and loader['staging_sha256'] == sha(work / 'staging.json')
         and loader['native_source_hashes'] == NATIVE_SOURCE_HASHES
         and loader['device'] == 'cpu' and not loader['training'] and not loader['fresh_inference'],
         'CPU loader gate differs')
    for role in ROLES:
        for split in ('train', 'val'):
            evidence = loader['datasets'][role + '/' + split]
            need(evidence['counts'] == plan['counts'][role][split] and evidence['exact_image_and_label_bytes']
                 and evidence['batches'] == (36 if split == 'train' else 5), 'CPU loader inventory differs')
    verify_guard(guard)
    need(guard['initializer_sha256'] == INIT and guard['dataset_archive_sha256'] == prior.ARCHIVE_SHA
         and guard['native_sources_sha256'] == NATIVE_SOURCE_HASHES
         and guard['optimizer_updates'] == 0
         and guard['loss_weighting_spec_sha256'] == digest_json(SPEC)
         and guard['criterion_contract'] == CRITERION_CONTRACT, 'CPU differentiation gate identity differs')
    for name, expected in guard['source_sha256'].items():
        checked(root / 'scripts' / name, expected)
    need(set(guard['source_sha256']) == {'quality_focal_r61.py', 'check_quality_focal_r61.py',
                                             'partial_classification_r51.py', 'check_partial_classification_r51.py'},
         'CPU differentiation source inventory differs')
    return {name: sha(work / name) for name in ('cpu-loader-preflight.json', 'loss-guard.json')}


def verify_controller(work, plan, role, receipt, trial=False):
    from quality_focal_r61 import validate_state
    ids = sorted(entry['id'] for entry in plan['roles'][role] if entry['split'] == 'train')
    need(receipt['focal_beta'] == BETAS[role], 'Arm focal exponent differs')
    state = receipt['quality_focal_controller']
    validate_state(state, ids, BETAS[role], role=role, protocol_sha256=sha(work / 'staging.json'), complete=not trial, trial=trial)
    need(state['batches_seen'] == receipt['batches'] and state['images_seen'] == receipt['images'],
         'Loss observer and trainer exposure differ')
    need(receipt['cpu_preflight_sha256'] == verify_cpu_gates(ROOT, work, plan), 'CPU gate bytes changed')
    for split in ('train', 'val'):
        evidence = receipt['actual_loader'][split]
        entries = sorted((entry for entry in plan['roles'][role] if entry['split'] == split), key=lambda row: row['image'])
        need(evidence['counts'] == plan['counts'][role][split] and evidence['exact_image_and_label_bytes']
             and evidence['cached_coordinates_atol'] == 1e-7
             and evidence['entries_sha256'] == digest(encoded(entries)), 'Actual loader evidence differs')


def verify_arm(work, role, plan, current=None):
    receipt = r47.verify_arm(work, role, plan, current)
    verify_controller(work, plan, role, receipt)
    return receipt


def verify_trial(work, plan, current=None):
    receipt = r47.verify_trial(work, plan, current)
    verify_controller(work, plan, 'candidate', receipt, trial=True)
    return receipt


def train(root, work, archive, role, trial=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PoseTrainer
    from quality_focal_r61 import QualityFocalLoss, verify_native_sources
    from ultralytics.utils import SETTINGS
    plan, runtime = verify_stage(root, work, archive), r45.runtime()
    cpu_gates = verify_cpu_gates(root, work, plan)
    need(runtime["native"] == verify_native_sources(), "Runtime loss/trainer source differs")
    need(not trial or role == "candidate", "Resource trial uses the candidate treatment")
    folder, out = work / "datasets" / role, work / ("resource-trial" if trial else role)
    budget = plan["budgets"][role]
    if (out / "receipt.json").exists():
        verify_trial(work, plan, runtime) if trial else verify_arm(work, role, plan, runtime)
        print("Reusing verified completed", out.name, flush=True)
        return
    if not trial:
        verify_trial(work, plan, runtime)
        if role == "candidate":
            verify_arm(work, "reference", plan, runtime)
    need(not out.exists(), "Preserved incomplete run; rename before retry: " + str(out))
    out.mkdir(parents=True)
    initializer = checked(work / "initializer/cassette_lower_epoch173.pt", INIT)
    training_ids = sorted(entry["id"] for entry in plan["roles"][role] if entry["split"] == "train")
    controller = QualityFocalLoss(training_ids, BETAS[role], role=role, protocol_sha256=sha(work / "staging.json"))
    progress = dict(focal_beta=BETAS[role], cpu_preflight_sha256=cpu_gates, role=role, trial=trial, source=plan["source"], runtime=runtime, staging_sha256=sha(work / "staging.json"),
                    status="starting", epochs=[], batches=0, images=0, steps=0, optimizer_batch_indices=[], updated=False)
    def record():
        save(out / "progress.json", progress)
    class TrialDone(Exception):
        pass
    class Trainer(PoseTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            need(self.world_size in (0, 1) and not self.ddp, "Single GPU only")
            for event, callback in (("on_train_epoch_start", self.start), ("on_train_batch_end", self.batch_done),
                                    ("on_train_epoch_end", self.epoch_done), ("on_model_save", self.saved)):
                self.add_callback(event, callback)
        def _build_train_pipeline(self):
            super()._build_train_pipeline()
            need(self.batch_size == budget["batch"] and len(self.train_loader) == budget["batches_per_epoch"], "Actual batch budget differs")
            progress["actual_loader"] = {split: verify_loader(folder, plan, role, split, loader.dataset)
                                         for split, loader in (("train", self.train_loader), ("val", self.test_loader))}
            check_training_settings(dict(vars(self.args)), RECIPE, model=initializer, data=folder / "data.yaml", device="0", project=out)
            progress["settings"] = dict(vars(self.args))
            initial = hashlib.sha256()
            for name, value in sorted(self.model.state_dict().items()):
                value = value.detach().cpu().contiguous()
                initial.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes())
            progress["initial_state_sha256"] = initial.hexdigest()
            record()
        def start(self, trainer):
            self.seen, self.nb = Counter(), 0
            controller.start_epoch(self.model, self.ema.ema, self.epoch + 1)
        def preprocess_batch(self, batch):
            result = super().preprocess_batch(batch)
            self.ids = [Path(path).stem for path in result["im_file"]]
            need(len(self.ids) == result["img"].shape[0], "Batch image count differs")
            controller.prepare_batch(result, self.epoch + 1)
            return result
        def optimizer_step(self):
            need(bool(torch.isfinite(self.loss).all()), "Nonfinite loss")
            gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
            need(gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients), "Nonfinite or missing gradients")
            parameter = next((parameter for group in self.optimizer.param_groups if group["lr"] > 0 for parameter in group["params"] if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))), None)
            need(parameter is not None, "No nonzero gradient at positive learning rate")
            before = parameter.detach().clone()
            super().optimizer_step()
            need(bool(torch.isfinite(parameter).all()), "Nonfinite parameter")
            progress["updated"] |= bool(torch.any(before != parameter.detach()))
            progress["steps"] += 1
            progress["optimizer_batch_indices"].append(progress["batches"])
        def batch_done(self, trainer):
            need(bool(torch.isfinite(self.loss).all()), "Nonfinite batch loss")
            controller.finish_batch(self.ids, self.epoch + 1)
            self.seen.update(self.ids)
            self.nb += 1
            progress["batches"] += 1
            progress["images"] += len(self.ids)
            record()
            if trial and progress["batches"] == 4:
                need(progress["updated"] and progress["steps"] == 4, "No actual trial update")
                raise TrialDone()
        def epoch_done(self, trainer):
            controller.finish_epoch(self.epoch + 1)
            need(self.nb == budget["batches_per_epoch"] and self.seen == Counter(entry["id"] for entry in plan["roles"][role] if entry["split"] == "train"), "Epoch image exposure differs")
        def saved(self, trainer):
            need(self.last.is_file() and self.epoch + 1 == len(progress["epochs"]) + 1, "Noncontiguous saved epoch")
            progress["epochs"].append(self.epoch + 1)
            record()
    SETTINGS.update({key: False for key in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = YOLO(str(initializer))
    check_loaded_pose_model(model)
    try:
        model.train(trainer=Trainer, **get_recipe(RECIPE), data=str(folder / "data.yaml"), device="0", workers=0,
                    plots=False, verbose=False, project=str(out), name="fit", exist_ok=False)
        need(not trial, "Trial did not stop")
        need(progress["epochs"] == list(range(1, budget["epochs"] + 1)) and progress["updated"] and
             all(progress[key] == budget[key] for key in ("batches", "images", "steps", "optimizer_batch_indices")), "Full measured budget differs")
        with (out / "fit/results.csv").open() as stream:
            need([int(row["epoch"]) for row in csv.DictReader(stream)] == list(range(1, budget["epochs"] + 1)), "Training CSV epochs differ")
        verify_stage(root, work, archive)
        progress["artifacts"] = {name: sha(out / name) for name in ("fit/weights/last.pt", "fit/results.csv")}
        progress["status"] = "complete"
    except TrialDone:
        progress["status"] = "trial-complete"
    except Exception as error:
        progress["status"], progress["error"] = "failed", str(error)
        record()
        raise
    progress["quality_focal_controller"] = controller.state()
    record()
    save(out / "receipt.json", progress)
    verify_trial(work, plan, runtime) if trial else verify_arm(work, role, plan, runtime)
    print(role, progress["status"], flush=True)


def restore_backup(work, backup, name, plan):
    if backup is None or (work / name).exists() or not (backup / name / 'backup.json').exists():
        return
    source = backup / name
    marker = validate_backup(source, name, sha(work / 'staging.json'), plan['source'])
    for relative in backup_paths(name):
        target = work / name / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    if marker.get('log_sha256'):
        shutil.copyfile(source / 'run.log', work / (name + '.log'))
    verify_trial(work, plan) if name == 'resource-trial' else verify_arm(work, name, plan)
    print('Restored verified completed', name, flush=True)


def save_backup(work, backup, name, plan):
    if backup is None:
        return
    verify_trial(work, plan) if name == 'resource-trial' else verify_arm(work, name, plan)
    source, destination = work / name, backup / name
    marker = dict(source=plan['source'], staging_sha256=sha(work / 'staging.json'), files={relative: sha(source / relative) for relative in backup_paths(name)})
    log = work / (name + '.log')
    if log.is_file():
        marker['log_sha256'] = sha(log)
    if (destination / 'backup.json').exists():
        need(validate_backup(destination, name, marker['staging_sha256'], plan['source']) == marker, 'Different completed backup preserved')
        return
    need(not destination.exists(), 'Incomplete backup preserved: ' + str(destination))
    destination.mkdir(parents=True)
    for relative in backup_paths(name):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    if log.is_file():
        shutil.copyfile(log, destination / 'run.log')
    save(destination / 'backup.json', marker)
    validate_backup(destination, name, marker['staging_sha256'], plan['source'])
    print('Saved verified completed', name, flush=True)


def run_pair(root, work, archive, backup):
    plan = verify_stage(root, work, archive)
    verify_cpu_gates(root, work, plan)
    script = Path(__file__).resolve()
    for name, action, role in (('resource-trial', 'trial', 'candidate'), ('reference', 'train', 'reference'), ('candidate', 'train', 'candidate')):
        restore_backup(work, backup, name, plan)
        command = [sys.executable, '-B', str(script), action, '--root', str(root), '--work', str(work),
                   '--archive', str(archive), '--role', role, '--source-revision', plan['source'], '--script-sha256', sha(script)]
        if (work / name / 'receipt.json').exists():
            subprocess.run(command, check=True)
        else:
            with (work / (name + '.log')).open('a', encoding='utf-8') as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line); log.flush()
                    need(process.wait() == 0, name + ' failed; inspect the preserved run and log')
        save_backup(work, backup, name, plan)


def verify_pair(work, plan, current=None):
    receipts = {role: verify_arm(work, role, plan, current) for role in ROLES}
    need(receipts['reference']['runtime'] == receipts['candidate']['runtime'], 'Paired runtime differs')
    need(receipts['reference']['initial_state_sha256'] == receipts['candidate']['initial_state_sha256'],
         'Initial parameter and buffer states differ')
    a, b = (receipts[role]['quality_focal_controller']['epochs'] for role in ROLES)
    need([row['observed_image_ids'] for row in a] == [row['observed_image_ids'] for row in b],
         'Paired image presentation order differs')
    return receipts


def detection_ranking(references, native):
    predictions = {}
    for arm, views in native.items():
        predictions[arm] = {identity: postclip_nms([p for p in rows if p['confidence'] > .001])[0]
                            for identity, rows in views['original'].items()}
    return {reference: {arm: score_detection_ranking(gold, values) for arm, values in predictions.items()}
            for reference, gold in references.items()}


def evaluate(root, work, archive):
    if (work / 'evaluation/summary.json').exists() or (work / 'evaluation/details.json.gz').exists():
        verify_evaluation(root, work, archive)
        print('Reusing verified saved evaluation', flush=True)
        return
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    plan, runtime = verify_stage(root, work, archive), r45.runtime()
    receipts = verify_pair(work, plan, runtime)
    entries = [row for row in plan['roles']['reference'] if row['split'] == 'val']
    paths = {'supplied': work / 'initializer/cassette_lower_epoch173.pt',
             **{role: work / role / 'fit/weights/last.pt' for role in ROLES}}
    native = {}
    forward_calls = 0
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for arm, path in paths.items():
        model = YOLO(str(path)); check_loaded_pose_model(model)
        model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), device='0', conf=.001, **INFERENCE)
        need(not model.predictor.model.end2end and not model.predictor.model.fp16, 'Expected ordinary NMS and FP32')
        native[arm] = {'original': {}, 'reflected': {}}
        for row in entries:
            image = cv2.imread(str(work / 'datasets/reference' / row['image']))
            need(image is not None and image.shape == (720, 1280, 3), 'Original image dimensions differ')
            for view, pixels in (('original', image), ('reflected', np.ascontiguousarray(image[:, ::-1]))):
                output = model.predict(source=pixels, device='0', conf=.001, **INFERENCE)[0]
                need(tuple(output.orig_shape) == (720, 1280), 'Prediction dimensions differ')
                native[arm][view][row['id']] = _prediction_records(output)
                forward_calls += 1
        del model
        torch.cuda.empty_cache()
    measure, detail = shared.score_native(read_refs(root), native, baseline='reference')
    ranking = detection_ranking(read_refs(root), native)
    summary = dict(schema='baomai-quality-focal-evaluation-r61-v1', source=plan['source'],
        staging_sha256=sha(work / 'staging.json'), runtime=runtime,
        model_sha256={arm: sha(path) for arm, path in paths.items()},
        treatment=plan['treatment'], budgets=plan['budgets'], measurement=measure, detection_ranking=ranking,
        inference={**INFERENCE, 'conf': .001, 'device': '0', 'fp16': False, 'tf32': False},
        actual_forward_calls=forward_calls, warmup_forward_calls=3, accuracy_results_available=True)
    put(work / 'evaluation/details.json.gz', gzip.compress(encoded(detail), mtime=0))
    put(work / 'evaluation/summary.json', encoded(summary))
    verify_evaluation(root, work, archive)


def verify_evaluation(root, work, archive):
    plan = verify_stage(root, work, archive)
    receipts = verify_pair(work, plan)
    summary = read(work / 'evaluation/summary.json')
    detail = json.loads(gzip.decompress((work / 'evaluation/details.json.gz').read_bytes()))
    expected_models = {'supplied': INIT, **{role: receipts[role]['artifacts']['fit/weights/last.pt'] for role in ROLES}}
    need(summary['source'] == plan['source'] and summary['staging_sha256'] == sha(work / 'staging.json')
         and summary['runtime'] == receipts['reference']['runtime']
         and summary['model_sha256'] == expected_models and summary['treatment'] == plan['treatment']
         and summary['budgets'] == plan['budgets'], 'Evaluation provenance differs')
    need(summary['inference'] == {**INFERENCE, 'conf': .001, 'device': '0', 'fp16': False, 'tf32': False}
         and summary['actual_forward_calls'] == 120, 'Evaluation settings or call count differs')
    measure, replay = shared.score_native(read_refs(root), detail['native-predictions'], baseline='reference')
    need(measure == summary['measurement'] and replay == detail, 'Saved score or coordinate replay differs')
    need(summary['detection_ranking'] == detection_ranking(read_refs(root), detail['native-predictions']),
         'Detection ranking replay differs')
    return plan


def package(root, work, archive):
    plan = verify_evaluation(root, work, archive)
    files = {}
    for relative in ('staging.json', 'cpu-loader-preflight.json', 'loss-guard.json',
                     'evaluation/summary.json', 'evaluation/details.json.gz'):
        files[relative] = (work / relative).read_bytes()
    for name in ('resource-trial', *ROLES):
        for relative in ('receipt.json', 'progress.json', 'fit/args.yaml', 'fit/results.csv'):
            path = work / name / relative
            if path.is_file():
                files[name + '/' + relative] = path.read_bytes()
        log = work / (name + '.log')
        if log.is_file():
            files['logs/' + name + '.txt'] = log.read_bytes()
    for role, entries in plan['roles'].items():
        folder = work / 'datasets' / role
        files['datasets/' + role + '/data.yaml'] = (folder / 'data.yaml').read_bytes()
        for row in entries:
            files['datasets/' + role + '/' + row['label']] = checked(folder / row['label'], row['label_sha256']).read_bytes()
    for name, expected in plan['source_files_sha256'].items():
        files['source/' + name] = checked(root / name, expected).read_bytes()
    for name, expected in plan['provenance_sha256'].items():
        files['inputs/' + name] = checked(work / 'inputs' / name, expected).read_bytes()
    files['inputs/source-manifest.json'] = checked(root / 'data/manifest.json', r47.ORIGINAL_MANIFEST_SHA).read_bytes()
    for name, item in plan['references'].items():
        files['references/' + name + '/manifest.json'] = checked(root / item['manifest_path'], item['manifest_sha256']).read_bytes()
        for identity, row in item['labels'].items():
            files['references/' + name + '/labels/' + identity + '.txt'] = checked(root / row['path'], row['sha256']).read_bytes()
    files['README.txt'] = ('Quality focal comparison: native BCE reference versus beta=2 candidate.\n'
        'Both arms use identical 142-photo training supervision and independent epoch173 initialization.\n'
        'Read evaluation/summary.json and retain all modelparts and return-manifest.json.\n'
        'The supplied-model original view is the report baseline; all annotation references are development views.\n').encode()
    return shared.package_evidence(work, files, {role: work / role / 'fit/weights/last.pt' for role in ROLES}, prefix=PREFIX)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('source-check', 'prepare', 'verify', 'cpu-loader', 'verify-cpu',
        'trial', 'train', 'run-pair', 'evaluate', 'verify-evaluation', 'package', 'verify-return'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--inputs-dir', type=Path)
    parser.add_argument('--download-inputs', action='store_true')
    parser.add_argument('--role', choices=ROLES, default='candidate')
    parser.add_argument('--backup-dir', type=Path)
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    contract = source_check(root, args.source_revision, args.script_sha256)
    if args.command == 'source-check':
        return
    need(root != work and not work.is_relative_to(root) and not root.is_relative_to(work),
         'Work and source folders must be separate')
    archive = args.archive.resolve() if args.archive else prior.obtain_archive(work / 'cache/pr137')
    if args.command == 'prepare':
        plan, _, _ = label_plan(root, archive)
        inputs = (args.inputs_dir or work / 'cache/assets').resolve()
        if args.download_inputs:
            r47.obtain_inputs(root, inputs)
            base.obtain_supplementary(root, inputs, plan['roles']['candidate'])
        prepare(root, work, archive, inputs, contract)
    elif args.command in ('verify', 'verify-cpu'):
        plan = verify_stage(root, work, archive)
        if args.command == 'verify-cpu':
            verify_cpu_gates(root, work, plan)
        print('Input/source gates passed; no training performed', flush=True)
    elif args.command == 'cpu-loader':
        print(json.dumps(cpu_loader_preflight(root, work, archive), indent=2))
    elif args.command in ('trial', 'train'):
        train(root, work, archive, args.role, args.command == 'trial')
    elif args.command == 'run-pair':
        run_pair(root, work, archive, args.backup_dir.resolve() if args.backup_dir else None)
        verify_pair(work, read(work / 'staging.json'))
    elif args.command in ('evaluate', 'verify-evaluation'):
        (evaluate if args.command == 'evaluate' else verify_evaluation)(root, work, archive)
    elif args.command == 'package':
        print(json.dumps(package(root, work, archive), indent=2))
    else:
        print(json.dumps(shared.verify_package(work, prefix=PREFIX), indent=2))


if __name__ == '__main__':
    main()
