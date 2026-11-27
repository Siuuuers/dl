#!/usr/bin/env python3
"""R51: paired partial-target classification BCE, with identical PR137 data.

CPU modes verify source, archived supervision, every staged image/label and cached
scores. GPU modes run a four-batch trial, then two fresh matched 30-epoch arms.
No R50 inference rerun, annotation edits, threshold sweep or model promotion.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import urllib.request
import zipfile

sys.dont_write_bytecode = True
import run_additional_front_pilot_r47 as r47
import rebuild_additional_front_pilot_r45 as r45
import audit_additional_front_pilot_r47_results as prior
from evaluate_pose import INFERENCE, _prediction_records
from pose_checkpoint import get_recipe, recipe_identity, check_training_settings, check_loaded_pose_model
from pose_scoring import score_image, aggregate_scores
from predict_pose_folder import postclip_nms

ROOT = Path(__file__).resolve().parents[1]
NAME = "baomai-partial-classification-r51-results.zip"
ROLES = ("reference", "candidate")
FACTORS = {"reference": 1, "candidate": 2}
RECIPE, INIT, VAL = r47.RECIPE, r47.INIT, r47.VAL
COUNTS = [142, 3999, 7259, 739, 13]
R50_PATH = "data/reviews/fronts-2026-11-27-r50/manifest.json"
R50_SHA = "f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f"
REFS = {**r47.REFS, "r50": (R50_PATH, R50_SHA, 654)}
R44_SHA = "100bfaceb9281bcf8c37c48eaaacf5159df14b0d04dd0ec37d5dcd4fc89c672d"
R44_BYTES = 6599840
R44_URL = "https://raw.githubusercontent.com/Siuuuers/dl/61c04d6a6dcb7e9f01ca99c603148b7f1506766b/results/review/front-mining-r44/review.zip"
need, digest, sha, checked = r45.need, r45.digest, r45.sha, r45.checked
encoded, put, save, read, rev, counts, labels = r45.encoded, r45.put, r45.save, r45.read, r45.rev, r45.counts, r45.labels
expected_budget = r47.expected_budget


def local_source_files(root):
    """Hash the complete recursive local Python closure, including lazy imports."""
    pending = {Path(__file__).name, "partial_classification_r51.py", "check_partial_classification_r51.py"}
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = root / "scripts" / name
        need(path.is_file() and not path.is_symlink(), "Missing source: " + name)
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = module.split(".")[0] + ".py"
                if (root / "scripts" / candidate).is_file() and candidate not in seen:
                    pending.add(candidate)
    return {"scripts/" + name: sha(root / "scripts" / name) for name in sorted(seen)}


def source_check(root, revision, script_sha):
    """Fail closed on first setup and every resumed command, before GPU imports."""
    need(re.fullmatch(r"[0-9a-f]{40}", revision or "") is not None, "Source revision is not published")
    need(re.fullmatch(r"[0-9a-f]{64}", script_sha or "") is not None, "Script hash is not published")
    need(rev(root) == revision, "Wrong checkout preserved; choose a fresh R51 base directory")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    need(not status.strip(), "Changed checkout preserved; choose a fresh R51 base directory")
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
    print("R51 source closure verified:", revision, len(closure), "Python files", flush=True)
    return dict(source=revision, script_sha256=script_sha, source_files_sha256=closure)


def label_plan(root, archive):
    """Use the audited PR137 candidate bytes for BOTH roles, including five photos."""
    from partial_classification_r51 import SPEC, CRITERION_CONTRACT
    checked(archive, prior.ARCHIVE_SHA)
    need(archive.stat().st_size == prior.ARCHIVE_BYTES, "PR137 archive size differs")
    receipt = json.loads(encoded(prior.audit(archive, root)))
    need(receipt["status"] == "passed", "PR137 return failed audit")
    with zipfile.ZipFile(archive) as z:
        staging_data = z.read("staging.json")
        need(digest(staging_data) == prior.STAGING_SHA, "PR137 staging differs")
        old = json.loads(staging_data)
        entries = old["roles"]["candidate"]
        payload = {entry["id"]: z.read("datasets/candidate/" + entry["label"]) for entry in entries}
        provenance = {"pr137-staging.json": staging_data, "pr137-input-audit.json": encoded(receipt),
                      "r47-release-status.json": z.read("release-status.json"), "r47-review-manifest.json": z.read("review/manifest.json")}
    need(old["initializer"] == INIT and old["recipe"] == recipe_identity(RECIPE), "Initializer or recipe differs")
    need(old["counts"]["candidate"] == {"train": COUNTS, "val": list(VAL)}, "PR137 candidate inventory differs")
    need(len(entries) == len({entry["id"] for entry in entries}) == 162, "Staged identities differ")
    need(sorted(entry["id"] for entry in entries if entry["source_kind"] != "original") == prior.EXTRAS, "Supplement membership differs")
    for entry in entries:
        need(digest(payload[entry["id"]]) == entry["label_sha256"], "Archived training label differs")
    budget = expected_budget(142)
    need((budget["batches"], budget["steps"], budget["images"]) == (1080, 80, 4260), "Expected accumulation budget differs")
    plan = dict(schema="baomai-r51-partial-classification-v1", source=rev(root), experiment="partial-classification-r51",
        source_files_sha256=local_source_files(root), recipe=recipe_identity(RECIPE), initializer=INIT,
        input_archive=dict(pr=137, head=prior.PR_HEAD, sha256=prior.ARCHIVE_SHA, bytes=prior.ARCHIVE_BYTES, staging_sha256=prior.STAGING_SHA),
        provenance_sha256={name: digest(data) for name, data in provenance.items()},
        counts={role: {"train": COUNTS, "val": list(VAL)} for role in ROLES},
        count_fields=old["count_fields"], budgets={role: budget for role in ROLES},
        treatment=dict(factors=FACTORS, partial_target_rule="fewer than two post-augmentation keypoints have nonzero visibility",
                       weighted_term="classification BCE for foreground anchors assigned to partial targets only",
                       native_equivalent_reference=True, loss_spec=SPEC, criterion_contract=CRITERION_CONTRACT,
                       same_images_and_labels=True, same_initializer=True, equal_image_exposure=True,
                       equal_optimizer_updates=True, no_label_changes=True, automatic_promotion=False),
        references={name: dict(manifest_path=path, manifest_sha256=value, objects=n) for name, (path, value, n) in REFS.items()},
        roles={role: entries for role in ROLES})
    return json.loads(encoded(plan)), payload, provenance


def obtain_supplementary(root, cache, entries):
    """Read five original JPEGs from the pinned R44 capsule; build no review charts."""
    extras = [entry for entry in entries if entry["source_kind"] != "original"]
    if all((cache / entry["image_source"]).is_file() for entry in extras):
        for entry in extras:
            checked(cache / entry["image_source"], entry["image_sha256"])
        return
    archive = cache / "r44-review.zip"
    local = root / "results/review/front-mining-r44/review.zip"
    if not archive.exists():
        if local.is_file() and sha(local) == R44_SHA:
            data = local.read_bytes()
        else:
            with urllib.request.urlopen(R44_URL, timeout=120) as response:
                data = response.read()
        need(digest(data) == R44_SHA and len(data) == R44_BYTES, "R44 archive differs")
        put(archive, data)
    checked(archive, R44_SHA)
    need(archive.stat().st_size == R44_BYTES, "R44 archive size differs")
    with zipfile.ZipFile(archive) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("history/pr131-r43.zip"))) as nested:
            for entry in extras:
                data = nested.read(entry["image_source"])
                need(digest(data) == entry["image_sha256"], "Supplementary JPEG differs: " + entry["id"])
                put(cache / entry["image_source"], data)


def image_size(data):
    """Original PNG/JPEG dimensions without a package install or image rewrite."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    need(data[:2] == b"\xff\xd8", "Unsupported original image")
    index = 2
    while index < len(data):
        need(data[index] == 255, "Invalid JPEG marker")
        while data[index] == 255:
            index += 1
        marker = data[index]; index += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        length = int.from_bytes(data[index:index + 2], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", data[index + 3:index + 7])
            return width, height
        need(length >= 2 and marker != 0xDA, "JPEG has no supported size marker")
        index += length
    raise RuntimeError("JPEG size is missing")


def dataset_config(folder):
    return dict(path=str(folder), train="images/train", val="images/val", names=["cassette_front"], kpt_shape=[2, 3], flip_idx=[1, 0])


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
    print("R51 CPU input preflight passed: both roles", COUNTS, "; identical 1080 batches / 80 updates / 4260 image presentations", flush=True)
    return plan


def verify_stage(root, work, archive):
    expected, _, provenance = label_plan(root, archive)
    plan = read(work / "staging.json")
    need(plan == expected, "Staged R51 source/data protocol differs")
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


def read_refs(root, entries):
    validation = {entry["id"]: entry for entry in entries if entry["split"] == "val"}
    result = {}
    for name, (relative, expected, count) in REFS.items():
        path = checked(root / relative, expected)
        rows = [row for row in read(path)["files"] if row.get("split", "val") == "val"]
        need(len(rows) == 20 and {row["image_id"] for row in rows} == set(validation), "Score reference image membership differs")
        gold = {}
        for row in rows:
            identity = row["image_id"]
            need(row.get("image_sha256", row.get("source_image_sha256")) == validation[identity]["image_sha256"], "Score reference image hash differs")
            gold[identity] = labels(checked(path.parent / row["label_path"], row["label_sha256"]).read_bytes(), name + "/" + identity)
        need(sum(map(len, gold.values())) == count, "Score reference target count differs")
        result[name] = (expected, gold)
    return result


def verify_loader(folder, plan, role, split, dataset):
    """Bind actual native dataset cache coordinates to every immutable TXT byte."""
    import numpy as np
    from ultralytics.data.utils import img2label_paths
    entries = {entry["image"]: entry for entry in plan["roles"][role] if entry["split"] == split}
    cached, seen = dataset.labels, set()
    objects = sum(len(item["cls"]) for item in cached)
    visible = sum(int((item["keypoints"][:, :, 2] == 2).sum()) for item in cached)
    observed = [len(dataset), objects, visible, 2 * objects - visible, sum(not len(item["cls"]) for item in cached)]
    need(observed == plan["counts"][role][split] and len(cached) == len(dataset.im_files), "Actual loader inventory differs")
    for image, label, loaded in zip(dataset.im_files, img2label_paths(dataset.im_files), cached):
        key = Path(image).resolve().relative_to(folder).as_posix()
        need(key in entries, "Unexpected loader image")
        entry = entries[key]
        need(Path(label).resolve() == (folder / entry["label"]).resolve(), "Loader label path differs")
        checked(image, entry["image_sha256"])
        text = checked(label, entry["label_sha256"]).read_text()
        expected = np.array([[float(token) for token in line.split()] for line in text.splitlines() if line.strip()], dtype=np.float32).reshape(-1, 11)
        actual = np.concatenate((loaded["cls"], loaded["bboxes"], loaded["keypoints"].reshape(-1, 6)), axis=1)
        need(loaded.get("normalized") is True and loaded.get("bbox_format") == "xywh" and actual.shape == expected.shape and
             bool(np.allclose(actual, expected, rtol=0, atol=1e-7)), "Native cached coordinates differ: " + entry["id"])
        seen.add(key)
    need(seen == set(entries), "Actual loader membership differs")
    return dict(counts=observed, exact_image_and_label_bytes=True, cached_coordinates_atol=1e-7,
                entries_sha256=digest(encoded([entries[key] for key in sorted(entries)])))


def cpu_loader_preflight(root, work, archive):
    """Construct the real pinned datasets/loaders and consume one CPU batch each."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.utils.torch_utils import init_seeds
    from partial_classification_r51 import verify_native_sources
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
    from partial_classification_r51 import NATIVE_SOURCE_HASHES, SPEC, CRITERION_CONTRACT, digest_json
    loader = read(work / "cpu-loader-preflight.json")
    guard = read(work / "loss-guard.json")
    need(loader["status"] == "passed" and loader["source"] == plan["source"] and loader["staging_sha256"] == sha(work / "staging.json")
         and loader["native_source_hashes"] == NATIVE_SOURCE_HASHES and loader["device"] == "cpu" and not loader["training"] and not loader["fresh_inference"], "CPU loader preflight differs")
    for role in ROLES:
        for split in ("train", "val"):
            evidence = loader["datasets"][role + "/" + split]
            need(evidence["counts"] == plan["counts"][role][split] and evidence["exact_image_and_label_bytes"] is True and evidence["batches"] == (36 if split == "train" else 5), "CPU loader inventory differs")
    need(guard["status"] == "passed" and guard["initializer_sha256"] == INIT and guard["dataset_archive_sha256"] == prior.ARCHIVE_SHA
         and guard["native_sources_sha256"] == NATIVE_SOURCE_HASHES and guard["optimizer_updates"] == 0
         and guard["loss_weighting_spec_sha256"] == digest_json(SPEC) and guard["criterion_contract"] == CRITERION_CONTRACT
         and guard["native_branch_schedule_bitwise_equal"] is True, "CPU gradient/native-equivalence guard differs")
    invariants = ("factor1_native_loss_and_prediction_gradients_bitwise_equal",
                  "factor2_assignment_and_target_geometry_bitwise_unchanged",
                  "factor2_nonclassification_loss_components_bitwise_equal",
                  "factor2_rle_flow_parameter_gradients_bitwise_equal")
    for name in ("mixed_real_batch", "native_class_weights_1_7", "empty_batch", "all_fully_visible_synthetic_batch", "consumed_visibility_mutation"):
        case = guard[name]
        need(all(case[key] is True for key in invariants), "A CPU loss/gradient invariant failed: " + name)
        branches = case["selective_classification_gradient_check"]
        need(set(branches) == {"one2many", "one2one"}, "Missing selective-gradient branch")
        for branch in branches.values():
            need(branch["box_keypoint_and_sigma_gradients_bitwise_unchanged"] is True
                 and branch["full_and_background_score_gradients_bitwise_unchanged"] is True
                 and branch["partial_score_gradient_factor_exact"] == 2, "Selective classification gradient differs")
            if name in ("mixed_real_batch", "native_class_weights_1_7", "consumed_visibility_mutation"):
                need(branch["partial_foreground_anchors"] > 0, "No partial anchors exercised by CPU gradient guard")
    network = guard["full_network_control_equivalence"]
    need(network["factor1_native_full_network_parameter_gradients_bitwise_equal"] is True
         and network["parameter_tensors_with_gradient"] > 0 and network["optimizer_updates"] == 0,
         "Full-network native-equivalence guard failed")
    for name, value in guard["source_sha256"].items():
        checked(root / "scripts" / name, value)
    need(set(guard["source_sha256"]) == {"partial_classification_r51.py", "check_partial_classification_r51.py"}, "Loss guard source membership differs")
    return {name: sha(work / name) for name in ("cpu-loader-preflight.json", "loss-guard.json")}


def verify_controller(work, plan, role, receipt, trial=False):
    from partial_classification_r51 import validate_state
    ids = sorted(entry["id"] for entry in plan["roles"][role] if entry["split"] == "train")
    need(receipt["classification_factor"] == FACTORS[role], "Arm classification factor differs")
    state = receipt["classification_controller"]
    validate_state(state, ids, FACTORS[role], role=role, protocol_sha256=sha(work / "staging.json"), complete=not trial, trial=trial)
    need(state["batches_seen"] == receipt["batches"] and state["images_seen"] == receipt["images"], "Loss observer and trainer exposure differ")
    need(all(value["partial_classification_terms"] > 0 for value in state["assignment_totals"].values()), "No assigned partial classification terms were observed")
    need(receipt["cpu_preflight_sha256"] == verify_cpu_gates(ROOT, work, plan), "Completed CPU-gate identity differs")
    for split in ("train", "val"):
        evidence = receipt["actual_loader"][split]
        entries = sorted((entry for entry in plan["roles"][role] if entry["split"] == split), key=lambda item: item["image"])
        need(evidence["counts"] == plan["counts"][role][split] and evidence["exact_image_and_label_bytes"] is True and evidence["cached_coordinates_atol"] == 1e-7,
             "Actual loader receipt differs")
        need(evidence["entries_sha256"] == digest(encoded(entries)), "Actual loader identity hashes differ")


def verify_arm(work, role, plan, current=None):
    receipt = r47.verify_arm(work, role, plan, current)
    verify_controller(work, plan, role, receipt)
    return receipt


def verify_trial(work, plan, current=None):
    receipt = r47.verify_trial(work, plan, current)
    verify_controller(work, plan, "candidate", receipt, trial=True)
    return receipt


def train(root, work, archive, role, trial=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PoseTrainer
    from partial_classification_r51 import PartialClassificationLoss, verify_native_sources
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
    controller = PartialClassificationLoss(training_ids, FACTORS[role], role=role, protocol_sha256=sha(work / "staging.json"))
    progress = dict(classification_factor=FACTORS[role], cpu_preflight_sha256=cpu_gates, role=role, trial=trial, source=plan["source"], runtime=runtime, staging_sha256=sha(work / "staging.json"),
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
    progress["classification_controller"] = controller.state()
    record()
    save(out / "receipt.json", progress)
    verify_trial(work, plan, runtime) if trial else verify_arm(work, role, plan, runtime)
    print(role, progress["status"], flush=True)


def evaluate(root, work, archive):
    # Completed scores are replayed instead of recomputing predictions.
    if (work / "evaluation/summary.json").exists() or (work / "evaluation/predictions.json").exists():
        verify_evaluation(root, work, archive)
        print("Reusing verified completed evaluation", flush=True)
        return
    import numpy as np
    import torch
    from ultralytics import YOLO
    plan, runtime = verify_stage(root, work, archive), r45.runtime()
    verify_cpu_gates(root, work, plan)
    receipts = {role: verify_arm(work, role, plan, runtime) for role in ROLES}
    entries, references = plan["roles"]["reference"], read_refs(root, plan["roles"]["reference"])
    detail = {}
    summary = dict(source=plan["source"], runtime=runtime, staging_sha256=sha(work / "staging.json"),
                   experiment=plan["experiment"], treatment=plan["treatment"], budgets=plan["budgets"],
                   model_sha256={role: receipts[role]["artifacts"]["fit/weights/last.pt"] for role in ROLES},
                   inference={**INFERENCE, "conf": .13, "matching_iou": .5, "endpoint_conf": .5}, references={})
    for role in ROLES:
        model = YOLO(str(work / role / "fit/weights/last.pt"))
        check_loaded_pose_model(model)
        model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8), device="0", conf=.13, **INFERENCE)
        need(not model.predictor.model.end2end and not model.predictor.model.fp16, "Expected ordinary NMS and FP32")
        detail[role] = {}
        for entry in entries:
            if entry["split"] != "val":
                continue
            prediction = model.predict(source=str(work / "datasets/reference" / entry["image"]), device="0", conf=.13, **INFERENCE)[0]
            need(tuple(prediction.orig_shape) == (720, 1280), "Prediction dimensions differ")
            raw = _prediction_records(prediction)
            kept, filtering = postclip_nms(raw)
            detail[role][entry["id"]] = dict(raw=raw, postclip=kept, filter=filtering)
        del model
        torch.cuda.empty_cache()
    for name, (manifest_sha, targets) in references.items():
        reports = {"manifest_sha256": manifest_sha}
        for view in ("raw", "postclip"):
            reports[view] = {}
            for role in ROLES:
                scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
                for identity, gold in targets.items():
                    predictions = detail[role][identity][view]
                    scores[identity] = score_image(gold, predictions, 1280, 720, .5, .5)
                    for key, values in r45.corners(gold, predictions, scores[identity]).items():
                        groups[key].update(values)
                reports[view][role] = dict(detection=aggregate_scores(list(scores.values())), corners_by_locatable_count=groups, per_image=scores)
            a, b = (reports[view][role]["per_image"] for role in ROLES)
            reports[view]["identity_delta"] = {identity: dict(recovered_rows=[i + 1 for i in sorted(set(a[identity]["unmatched_gold_indices"]) - set(b[identity]["unmatched_gold_indices"]))], newly_missed_rows=[i + 1 for i in sorted(set(b[identity]["unmatched_gold_indices"]) - set(a[identity]["unmatched_gold_indices"]))]) for identity in a}
        summary["references"][name] = reports
    save(work / "evaluation/predictions.json", detail)
    save(work / "evaluation/summary.json", summary)
    verify_evaluation(root, work, archive)


def verify_evaluation(root, work, archive):
    plan = verify_stage(root, work, archive)
    verify_cpu_gates(root, work, plan)
    receipts = {role: verify_arm(work, role, plan) for role in ROLES}
    need(receipts["reference"]["runtime"] == receipts["candidate"]["runtime"], "Mixed-runtime arms")
    summary, predictions = read(work / "evaluation/summary.json"), read(work / "evaluation/predictions.json")
    need(summary["source"] == plan["source"] and summary["staging_sha256"] == sha(work / "staging.json") and
         summary["runtime"] == receipts["reference"]["runtime"] and summary["treatment"] == plan["treatment"] and summary["budgets"] == plan["budgets"], "Evaluation identity differs")
    need(summary["inference"] == {**INFERENCE, "conf": .13, "matching_iou": .5, "endpoint_conf": .5}, "Evaluation settings differ")
    need(summary["model_sha256"] == {role: receipts[role]["artifacts"]["fit/weights/last.pt"] for role in ROLES}, "Evaluation model identity differs")
    references = read_refs(root, plan["roles"]["reference"])
    need(set(summary["references"]) == set(references) and set(predictions) == set(ROLES), "Evaluation membership differs")
    validation = {entry["id"] for entry in plan["roles"]["reference"] if entry["split"] == "val"}
    for role in ROLES:
        need(set(predictions[role]) == validation, "Prediction photograph membership differs")
        for value in predictions[role].values():
            kept, filtering = postclip_nms(value["raw"])
            need(kept == value["postclip"] and filtering == value["filter"], "Postclip replay differs")
    for name, (manifest_sha, targets) in references.items():
        report = summary["references"][name]
        need(report["manifest_sha256"] == manifest_sha, "Evaluation reference hash differs")
        for view in ("raw", "postclip"):
            for role in ROLES:
                scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
                for identity, gold in targets.items():
                    pred = predictions[role][identity][view]
                    scores[identity] = score_image(gold, pred, 1280, 720, .5, .5)
                    for key, values in r45.corners(gold, pred, scores[identity]).items():
                        groups[key].update(values)
                need(report[view][role] == dict(detection=aggregate_scores(list(scores.values())), corners_by_locatable_count=groups, per_image=scores), "Score replay differs")
            a, b = (report[view][role]["per_image"] for role in ROLES)
            expected = {identity: dict(recovered_rows=[i + 1 for i in sorted(set(a[identity]["unmatched_gold_indices"]) - set(b[identity]["unmatched_gold_indices"]))], newly_missed_rows=[i + 1 for i in sorted(set(b[identity]["unmatched_gold_indices"]) - set(a[identity]["unmatched_gold_indices"]))]) for identity in a}
            need(report[view]["identity_delta"] == expected, "Changed-target identities differ")
    return plan


def return_members(root, work, archive, verify_saved_scores=True):
    plan = verify_evaluation(root, work, archive) if verify_saved_scores else verify_stage(root, work, archive)
    receipts = {role: verify_arm(work, role, plan) for role in ROLES}
    need(receipts['reference']['runtime'] == receipts['candidate']['runtime'], 'Mixed-runtime return')
    verify_trial(work, plan, receipts['reference']['runtime'])
    members = {name: work / name for name in ('staging.json', 'cpu-loader-preflight.json', 'loss-guard.json',
                                              'evaluation/summary.json', 'evaluation/predictions.json')}
    for name, expected in plan['provenance_sha256'].items():
        members['inputs/' + name] = checked(work / 'inputs' / name, expected)
    for relative, expected in plan['source_files_sha256'].items():
        members['source/' + relative] = checked(root / relative, expected)
    for role in ('resource-trial', *ROLES):
        for name in backup_paths(role):
            members[role + '/' + name] = work / role / name
        if (work / (role + '.log')).is_file():
            members[role + '.log'] = work / (role + '.log')
    for role in ROLES:
        for entry in plan['roles'][role]:
            members['datasets/' + role + '/' + entry['label']] = work / 'datasets' / role / entry['label']
        members['datasets/' + role + '/data.yaml'] = work / 'datasets' / role / 'data.yaml'
    members['inputs/source-manifest.json'] = checked(root / 'data/manifest.json', r47.ORIGINAL_MANIFEST_SHA)
    for name, (relative, expected, _) in REFS.items():
        path = checked(root / relative, expected)
        members['references/' + name + '/manifest.json'] = path
        for entry in read(path)['files']:
            if entry.get('split', 'val') == 'val':
                members['references/' + name + '/' + entry['label_path']] = checked(path.parent / entry['label_path'], entry['label_sha256'])
    return members


def verify_return(root, work, archive, output):
    members = return_members(root, work, archive)
    with zipfile.ZipFile(output) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == set(members) | {'inventory.json'}, 'Return archive membership differs')
        need(z.testzip() is None, 'Return ZIP CRC failure')
        inventory = json.loads(z.read('inventory.json'))
        need(set(inventory) == set(members), 'Return inventory differs')
        for name, path in members.items():
            expected = dict(bytes=path.stat().st_size, sha256=sha(path))
            need(inventory[name] == expected and z.getinfo(name).file_size == expected['bytes'] and digest(z.read(name)) == expected['sha256'], 'Return member differs: ' + name)
    return dict(path=str(output), bytes=output.stat().st_size, sha256=sha(output), members=len(names))


def restore_return(root, work, archive, output):
    """Only restore absent cached scores; all completed source/run bytes must match."""
    members = return_members(root, work, archive, verify_saved_scores=False)
    evaluation = {'evaluation/summary.json', 'evaluation/predictions.json'}
    payload = {}
    with zipfile.ZipFile(output) as z:
        names = z.namelist()
        need(len(names) == len(set(names)) and set(names) == set(members) | {'inventory.json'} and z.testzip() is None, 'Saved return membership/CRC differs')
        inventory = json.loads(z.read('inventory.json'))
        need(set(inventory) == set(members), 'Saved return inventory differs')
        for name, path in members.items():
            data = z.read(name)
            need(inventory[name] == dict(bytes=len(data), sha256=digest(data)), 'Saved return bytes differ: ' + name)
            if name in evaluation:
                payload[name] = data
            else:
                need(inventory[name] == dict(bytes=path.stat().st_size, sha256=sha(path)), 'Saved return context differs: ' + name)
    for name, data in payload.items():
        put(work / name, data)
    return verify_return(root, work, archive, output)


def package(root, work, archive, output):
    need(not output.resolve().is_relative_to(work.resolve()), 'Return ZIP must be outside work')
    if output.exists():
        return verify_return(root, work, archive, output)
    members = return_members(root, work, archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    need(not temporary.exists(), 'Incomplete archive preserved: ' + str(temporary))
    r47.write_return_zip(members, temporary)
    verify_return(root, work, archive, temporary)
    temporary.replace(output)
    return verify_return(root, work, archive, output)


backup_paths, validate_backup = r47.backup_paths, r47.validate_backup


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('source-check', 'prepare', 'verify', 'cpu-loader', 'verify-cpu', 'trial', 'train', 'run-pair',
                                            'evaluate', 'verify-evaluation', 'package', 'verify-return', 'restore-return'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--inputs-dir', type=Path)
    parser.add_argument('--download-inputs', action='store_true')
    parser.add_argument('--role', choices=ROLES, default='candidate')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backup-dir', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = source_check(root, args.source_revision, args.script_sha256)
    if args.command == 'source-check':
        return
    need(args.work is not None, '--work is required')
    work = args.work.resolve()
    archive = args.archive.resolve() if args.archive else prior.obtain_archive(work / 'cache/pr137')
    if args.command == 'prepare':
        plan, _, _ = label_plan(root, archive)
        inputs = (args.inputs_dir or work / 'cache/assets').resolve()
        if args.download_inputs:
            r47.obtain_inputs(root, inputs)
            obtain_supplementary(root, inputs, plan['roles']['candidate'])
        prepare(root, work, archive, inputs, contract)
    elif args.command in ('verify', 'verify-cpu'):
        plan = verify_stage(root, work, archive)
        if args.command == 'verify-cpu':
            verify_cpu_gates(root, work, plan)
        print('R51 preflight verified; no training performed', flush=True)
    elif args.command == 'cpu-loader':
        print(json.dumps(cpu_loader_preflight(root, work, archive), indent=2))
    elif args.command in ('trial', 'train'):
        train(root, work, archive, args.role, args.command == 'trial')
    elif args.command == 'run-pair':
        run_pair(root, work, archive, args.backup_dir.resolve() if args.backup_dir else None)
    elif args.command in ('evaluate', 'verify-evaluation'):
        (evaluate if args.command == 'evaluate' else verify_evaluation)(root, work, archive)
    else:
        need(args.output is not None, '--output is required')
        result = {'package': package, 'verify-return': verify_return, 'restore-return': restore_return}[args.command](root, work, archive, args.output.resolve())
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
