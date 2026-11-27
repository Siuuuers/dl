#!/usr/bin/env python3
"""R47 expanded-data, equal-epoch pilot; explicit exposure and whole-photo gates.

CPU: plan / prepare / verify / verify-return. GPU: trial / train / evaluate.
The R47 reference preserves the prior R45 candidate's dataset: 137 R37 originals
plus old frame_0383. The candidate uses every complete R47 supplementary photo. Both start fresh
from epoch173. More data and its additional exposure form one treatment.
"""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

sys.dont_write_bytecode = True
import rebuild_additional_front_pilot_r45 as r45
import build_front_review_r47 as review_builder
from evaluate_pose import parse_pose_labels, INFERENCE, _prediction_records
from pose_checkpoint import get_recipe, recipe_identity, check_training_settings, check_loaded_pose_model
from pose_scoring import score_image, aggregate_scores
from predict_pose_folder import postclip_nms

REVIEW = "data/reviews/additional-fronts-2026-11-27-r47"
RELEASE_SHA256 = "1d4f3ac4cb3efc97c6e05ef392b83654d4eed2039f3f167340a425ff837641a7"
SOURCE_INPUTS = "ea4ac955f2dfaf1add24c243599715f802eb41a7"
ORIGINAL_MANIFEST_SHA = "72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926"
OLD_EXTRA = "frame_0383"
OLD_EXTRA_LABEL = "data/reviews/additional-fronts-2026-11-27-r46/complete-candidates/labels/frame_0383.txt"
OLD_EXTRA_LABEL_SHA = "1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0"
OLD_EXTRA_IMAGE_SHA = "cad520edcb4299210090da752b4bdc7bd8e857651cdd2351dfe58ac9a534cbd6"
R45_SOURCE_SHA = "e6a61f4bd6aa792c0bb74ee6a6e4c07259c72494211f40cb90402354a2ab2507"
ROLES = ("reference", "candidate")
RECIPE, INIT, VAL = r45.RECIPE, r45.INIT, r45.VAL
REFS = {**r45.REFS, "r38": (r45.REFS["r38"][0], "0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5", 654)}
need, digest, sha, checked = r45.need, r45.digest, r45.sha, r45.checked
encoded, put, save, read, rev, counts, labels = r45.encoded, r45.put, r45.save, r45.read, r45.rev, r45.counts, r45.labels


def expected_budget(photographs):
    """Replay the pinned trainer's accumulation schedule without importing NumPy.

The runtime verifies exact native trainer source bytes. Actual batches, image
exposure, optimizer updates and update batch indices must match these values.
No padding, supplementary oversampling, last-batch flush or resume is added.
"""
    recipe = get_recipe(RECIPE)
    batch, epochs = recipe["batch"], recipe["epochs"]
    nb = math.ceil(photographs / batch)
    warmup = round(min(recipe["warmup_epochs"], max(epochs - 1, 0)) * nb)
    accumulate = max(round(recipe["nbs"] / batch), 1)
    last, indices = -1, []
    for ni in range(nb * epochs):
        if ni < warmup:
            accumulate = max(1, round(1 + (recipe["nbs"] / batch - 1) * ni / warmup))
        if ni - last >= accumulate:
            indices.append(ni)
            last = ni
    return dict(epochs=epochs, batch=batch, batches_per_epoch=nb, batches=nb * epochs,
                images=photographs * epochs, steps=len(indices), optimizer_batch_indices=indices,
                warmup_batches=warmup, ordinary_once_per_image_per_epoch=True)


def local_source_files(root):
    """Record the recursive local Python source closure, including lazy imports."""
    pending = {"run_additional_front_pilot_r47.py", "rebuild_additional_front_pilot_r45.py",
               "partial_box_loss_r31.py", *review_builder.SCRIPTS}
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = root / "scripts" / name
        need(path.is_file(), "Missing local source: " + name)
        seen.add(name)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = module.split(".")[0] + ".py"
                if (root / "scripts" / candidate).is_file() and candidate not in seen:
                    pending.add(candidate)
    return {"scripts/" + name: sha(root / "scripts" / name) for name in sorted(seen)}


def release_context(root, review):
    checked(root / "scripts/rebuild_additional_front_pilot_r45.py", R45_SOURCE_SHA)
    need(len(RELEASE_SHA256) == 64, "R47 release hash has not been frozen")
    release = read(checked(root / REVIEW / "release-status.json", RELEASE_SHA256))
    report = review_builder.verify_output(review)
    manifest = read(review / "manifest.json")
    need(release["round"] == manifest["round"] == 47, "Wrong review round")
    checked(review / "manifest.json", release["generated_manifest_sha256"])
    for filename, key in (("audit-ledger.json", "audit_sha256"), ("outlines.json", "outlines_sha256")):
        checked(root / REVIEW / filename, release[key])
        checked(review / filename, release[key])
    for name, expected in release["files"].items():
        checked(root / REVIEW / name, expected)
        checked(review / name, expected)
    complete = sorted(release["annotation_complete_ids"])
    frames = {frame["image_id"]: frame for frame in manifest["frames"]}
    need(complete == sorted(manifest["annotation_complete_ids"]) and
         complete == sorted(frame["image_id"] for frame in manifest["frames"] if frame["eligible_for_candidate_export"]),
         "Complete-photo membership differs")
    need(set(complete).isdisjoint(release["held_image_ids"]) and OLD_EXTRA in complete and "frame_0160" in complete,
         "Required complete photographs missing, or held photo included")
    need(2 <= len(complete) <= 12, "Unexpected supplementary count")
    for identity in complete:
        frame = frames[identity]
        need(frame["annotation_complete"] and frame["eligible_for_candidate_export"] and
             not any(row["unresolved_fields"] for row in frame["rows"] + frame["additions"]), "Held supplementary photo: " + identity)
        need(frame["training_split"] == "unassigned", "Review unexpectedly assigned a split")
    return release, manifest, report, frames, complete


def label_plan(root, review):
    release, manifest, report, frames, complete = release_context(root, review)
    original = read(checked(root / "data/manifest.json", ORIGINAL_MANIFEST_SHA))
    overlay_path = root / r45.OVERLAY
    overlay = read(checked(overlay_path, r45.OVERLAY_HASH))
    need(original["model_epoch"] == 173 and original["model_sha256"] == INIT, "Initializer declaration differs")
    need(recipe_identity(RECIPE)["recipe_sha256"] == r45.RECIPE_HASH, "Recipe differs")
    records = original["records"]
    train_ids = {row["id"] for row in records if row["split"] == "train"}
    val_ids = {row["id"] for row in records if row["split"] == "val"}
    need(len(records) == len(train_ids | val_ids) == 157 and len(train_ids) == 137 and len(val_ids) == 20 and not train_ids & val_ids,
         "Original split differs")
    changes = {row["image_id"]: row for row in overlay["changes"]}
    sources = {row["image_id"]: row for row in overlay["image_sources"]}
    need(set(sources) == train_ids and len(changes) == 92 and set(changes) <= train_ids, "Original overlay differs")
    common, payload = [], {}
    original_hashes = {row["image_sha256"] for row in records}
    for row in records:
        identity, split = row["id"], row["split"]
        label = checked(root / "data" / row["label"], row["label_sha256"])
        if identity in changes:
            change = changes[identity]
            need(change["source_label_sha256"] == row["label_sha256"] and change["source_image_sha256"] == row["image_sha256"], "Overlay lineage differs")
            label = checked(overlay_path.parent / change["candidate_label"], change["candidate_label_sha256"])
        if split == "train":
            need(sha(label) == sources[identity]["effective_label"]["sha256"] and row["image_sha256"] == sources[identity]["source_image_sha256"], "Effective R37 labels differ")
        data = label.read_bytes()
        entry = dict(id=identity, split=split, image=f"images/{split}/{identity}.png", label=f"labels/{split}/{identity}.txt",
                     image_sha256=row["image_sha256"], label_sha256=digest(data), image_source="data/" + row["image"],
                     label_source=label.relative_to(root).as_posix(), source_kind="original")
        common.append(entry)
        for role in ROLES:
            payload[(role, identity)] = data
    need(counts([labels(payload[("reference", row["id"])], row["id"]) for row in common if row["split"] == "train"]) == (137, 3784, 6837, 731, 13),
         "Original R37 inventory differs")
    old = checked(root / OLD_EXTRA_LABEL, OLD_EXTRA_LABEL_SHA).read_bytes()
    need(frames[OLD_EXTRA]["source_image_sha256"] == OLD_EXTRA_IMAGE_SHA, "Baseline supplementary image differs")
    base_extra = dict(id=OLD_EXTRA, split="train", image=f"images/train/{OLD_EXTRA}.jpg", label=f"labels/train/{OLD_EXTRA}.txt",
                      image_sha256=OLD_EXTRA_IMAGE_SHA, label_sha256=OLD_EXTRA_LABEL_SHA,
                      image_source="originals/" + OLD_EXTRA + ".jpg", label_source=OLD_EXTRA_LABEL, source_kind="r45_extra")
    payload[("reference", OLD_EXTRA)] = old
    candidate_extra = []
    for identity in complete:
        frame = frames[identity]
        need(identity not in train_ids | val_ids and frame["source_image_sha256"] not in original_hashes, "Supplement duplicates an original")
        relative = "complete-candidates/labels/" + identity + ".txt"
        data = checked(review / relative, release["files"][relative]).read_bytes()
        payload[("candidate", identity)] = data
        candidate_extra.append(dict(id=identity, split="train", image=f"images/train/{identity}.jpg", label=f"labels/train/{identity}.txt",
                                    image_sha256=frame["source_image_sha256"], label_sha256=digest(data),
                                    image_source="originals/" + identity + ".jpg", label_source=relative, source_kind="r47_extra"))
    need(len({row["image_sha256"] for row in candidate_extra}) == len(candidate_extra), "Duplicate supplementary image bytes")
    roles = {"reference": common + [base_extra], "candidate": common + candidate_extra}
    observed = {role: {split: counts([labels(payload[(role, row["id"])], row["id"]) for row in rows if row["split"] == split])
                       for split in ("train", "val")} for role, rows in roles.items()}
    need(observed["reference"]["train"] == (138, 3827, 6923, 731, 13), "R47 reference / prior R45 candidate dataset differs")
    need(all(observed[role]["val"] == VAL for role in ROLES), "Validation training inventory differs")
    shared_changes = [identity for identity in complete if identity == OLD_EXTRA and payload[("candidate", identity)] != old]
    budgets = {role: expected_budget(observed[role]["train"][0]) for role in ROLES}
    plan = dict(schema="baomai-r47-expanded-data-equal-epoch-v1", source=rev(root), experiment="additional-front-pilot-r47",
                source_files_sha256=local_source_files(root), recipe=recipe_identity(RECIPE), initializer=INIT,
                release_sha256=RELEASE_SHA256, review_manifest_sha256=release["generated_manifest_sha256"],
                counts={role: {split: list(value) for split, value in item.items()} for role, item in observed.items()}, budgets=budgets,
                count_fields=["photos", "fronts", "locatable_corner_slots", "missing_corner_slots", "empty_photos"],
                treatment=dict(reference_extra_ids=[OLD_EXTRA], candidate_extra_ids=complete,
                               added_photo_ids=sorted(set(complete) - {OLD_EXTRA}), shared_extra_label_changes=shared_changes,
                               preserved_baseline_extra_label_sha256=OLD_EXTRA_LABEL_SHA,
                               equal_epochs=True, equal_optimizer_updates=budgets["reference"]["steps"] == budgets["candidate"]["steps"],
                               extra_exposure_is_part_of_treatment=True, isolated_data_from_compute=False,
                               original_137_supervision_identical=True, all_original_20_development_photos_excluded_from_training=True,
                               whole_photo_gate=True, held_photo_ids=release["held_image_ids"], automatic_promotion=False), roles=roles)
    return plan, payload


def obtain_inputs(root, directory):
    """Fetch only the 157 pinned original PNGs and epoch173 from a fixed commit."""
    manifest = read(checked(root / "data/manifest.json", ORIGINAL_MANIFEST_SHA))
    assets = [("data/" + row["image"], row["image_sha256"]) for row in manifest["records"]]
    assets.append((manifest["model_checkpoint"], INIT))
    def obtain(item):
        relative, expected = item
        path = directory / relative
        if path.exists():
            checked(path, expected)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://media.githubusercontent.com/media/Siuuuers/dl/{SOURCE_INPUTS}/{relative}"
        with urllib.request.urlopen(url, timeout=180) as response:
            data = response.read()
        need(digest(data) == expected, "Downloaded input differs: " + relative)
        put(path, data)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(obtain, assets))
    print("Verified original image/checkpoint cache:", len(assets), flush=True)


def stage(root, work, review, inputs):
    from PIL import Image
    plan, payload = label_plan(root, review)
    original = read(root / "data/manifest.json")
    initializer = checked(inputs / original["model_checkpoint"], INIT)
    put(work / "initializer/cassette_lower_epoch173.pt", initializer.read_bytes())
    for role, entries in plan["roles"].items():
        folder = work / "datasets" / role
        for entry in entries:
            source = (inputs if entry["source_kind"] == "original" else review) / entry["image_source"]
            checked(source, entry["image_sha256"])
            with Image.open(source) as image:
                need(image.size == (1280, 720), "Input dimensions differ: " + entry["id"])
            put(folder / entry["image"], source.read_bytes())
            put(folder / entry["label"], payload[(role, entry["id"])])
        put(folder / "data.yaml", encoded(dict(path=str(folder), train="images/train", val="images/val", names=["cassette_front"], kpt_shape=[2, 3], flip_idx=[1, 0])))
    put(work / "staging.json", encoded(plan))
    # Store replayable review provenance separately from the training images.
    put(work / "release-status.json", (root / REVIEW / "release-status.json").read_bytes())
    verify_stage(root, work, review)
    read_refs(root, plan["roles"]["reference"])
    print(json.dumps({"preflight": "passed", "counts": plan["counts"], "budgets": {r: {k: v for k, v in b.items() if k != "optimizer_batch_indices"} for r, b in plan["budgets"].items()}, "treatment": plan["treatment"]}, indent=2), flush=True)


def verify_stage(root, work, review):
    expected, _ = label_plan(root, review)
    actual = read(work / "staging.json")
    need(actual == expected, "Staged protocol differs from the frozen source/review plan")
    checked(work / "release-status.json", RELEASE_SHA256)
    checked(work / "initializer/cassette_lower_epoch173.pt", INIT)
    for role, entries in actual["roles"].items():
        folder = work / "datasets" / role
        need(len(entries) == len({entry["id"] for entry in entries}), "Repeated staged image identity")
        groups = {"train": [], "val": []}
        for entry in entries:
            checked(folder / entry["image"], entry["image_sha256"])
            groups[entry["split"]].append(labels(checked(folder / entry["label"], entry["label_sha256"]).read_bytes(), entry["id"]))
        for kind, sub, suffixes in (("image", "images", {".png", ".jpg", ".jpeg"}), ("label", "labels", {".txt"})):
            found = {path.relative_to(folder).as_posix() for path in (folder / sub).rglob("*") if path.is_file() and path.suffix.lower() in suffixes}
            need(found == {entry[kind] for entry in entries}, "Unexpected staged " + role + " " + kind + " files")
        need(all(list(counts(groups[split])) == actual["counts"][role][split] for split in groups), "Staged label counts differ")
        need(read(folder / "data.yaml") == dict(path=str(folder), train="images/train", val="images/val", names=["cassette_front"], kpt_shape=[2, 3], flip_idx=[1, 0]), "Dataset configuration differs")
    return actual


def read_refs(root, entries):
    # Pin all three manifests, including current R38, before delegating parsing.
    for relative, expected, _ in REFS.values():
        checked(root / relative, expected)
    return r45.read_refs(root, entries)


def verify_arm(work, role, plan, current=None):
    out, budget = work / role, plan["budgets"][role]
    receipt = read(out / "receipt.json")
    need(receipt == read(out / "progress.json"), "Final progress differs from receipt")
    need(receipt["status"] == "complete" and receipt["role"] == role and not receipt["trial"] and
         receipt["source"] == plan["source"] and receipt["staging_sha256"] == sha(work / "staging.json"), "Arm identity differs")
    need(receipt["epochs"] == list(range(1, budget["epochs"] + 1)) and receipt["updated"] and
         all(receipt[key] == budget[key] for key in ("batches", "images", "steps", "optimizer_batch_indices")), "Actual arm exposure differs")
    if current is not None:
        need(receipt["runtime"] == current, "A/B runtime differs")
    need(set(receipt["artifacts"]) == {"fit/weights/last.pt", "fit/results.csv"}, "Arm artifact inventory differs")
    for name, expected in receipt["artifacts"].items():
        checked(out / name, expected)
    settings = receipt["settings"]
    check_training_settings(settings, RECIPE, model=settings["model"], data=settings["data"], device="0", project=settings["project"])
    with (out / "fit/results.csv").open() as stream:
        need([int(row["epoch"]) for row in csv.DictReader(stream)] == list(range(1, budget["epochs"] + 1)), "Receipt CSV epochs differ")
    return receipt


def verify_trial(work, plan, current=None):
    receipt = read(work / "resource-trial/receipt.json")
    need(receipt == read(work / "resource-trial/progress.json") and receipt["status"] == "trial-complete" and receipt["trial"] and
         receipt["role"] == "candidate" and receipt["source"] == plan["source"] and
         receipt["staging_sha256"] == sha(work / "staging.json") and receipt["batches"] == 4 and receipt["images"] == 16 and
         receipt["steps"] == 4 and receipt["optimizer_batch_indices"] == [0, 1, 2, 3] and receipt["updated"], "Resource trial differs")
    if current is not None:
        need(receipt["runtime"] == current, "Resource-trial runtime differs")
    settings = receipt["settings"]
    check_training_settings(settings, RECIPE, model=settings["model"], data=settings["data"], device="0", project=settings["project"])
    return receipt


def train(root, work, review, role, trial=False):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.data.utils import img2label_paths
    from ultralytics.utils import SETTINGS
    plan, runtime = verify_stage(root, work, review), r45.runtime()
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
    progress = dict(role=role, trial=trial, source=plan["source"], runtime=runtime, staging_sha256=sha(work / "staging.json"),
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
            for split, loader in (("train", self.train_loader), ("val", self.test_loader)):
                entries = {entry["image"]: entry for entry in plan["roles"][role] if entry["split"] == split}
                cached, seen = loader.dataset.labels, set()
                objects = sum(len(item["cls"]) for item in cached)
                visible = sum(int((item["keypoints"][:, :, 2] == 2).sum()) for item in cached)
                observed = [len(loader.dataset), objects, visible, 2 * objects - visible, sum(not len(item["cls"]) for item in cached)]
                need(observed == plan["counts"][role][split], "Actual loader inventory differs: " + str(observed))
                need(len(cached) == len(loader.dataset.im_files), "Loader cache membership differs")
                for image, label, loaded in zip(loader.dataset.im_files, img2label_paths(loader.dataset.im_files), cached):
                    key = Path(image).resolve().relative_to(folder).as_posix()
                    entry = entries[key]
                    need(Path(label).resolve() == (folder / entry["label"]).resolve(), "Loader label path differs")
                    checked(image, entry["image_sha256"])
                    expected_label = checked(label, entry["label_sha256"]).read_text()
                    seed = np.array([[float(token) for token in line.split()] for line in expected_label.splitlines() if line.strip()], dtype=np.float32).reshape(-1, 11)
                    loaded_rows = np.concatenate((loaded["cls"], loaded["bboxes"], loaded["keypoints"].reshape(-1, 6)), axis=1)
                    need(loaded.get("normalized") is True and loaded.get("bbox_format") == "xywh" and loaded_rows.shape == seed.shape and
                         bool(np.allclose(loaded_rows, seed, rtol=0, atol=1e-7)), "Loader cached coordinates differ: " + entry["id"])
                    seen.add(key)
                need(seen == set(entries), "Loader image membership differs")
            check_training_settings(dict(vars(self.args)), RECIPE, model=initializer, data=folder / "data.yaml", device="0", project=out)
            progress["settings"] = dict(vars(self.args))
            record()
        def start(self, trainer):
            self.seen, self.nb = Counter(), 0
        def preprocess_batch(self, batch):
            result = super().preprocess_batch(batch)
            self.ids = [Path(path).stem for path in result["im_file"]]
            need(len(self.ids) == result["img"].shape[0], "Batch image count differs")
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
            self.seen.update(self.ids)
            self.nb += 1
            progress["batches"] += 1
            progress["images"] += len(self.ids)
            record()
            if trial and progress["batches"] == 4:
                need(progress["updated"] and progress["steps"] == 4, "No actual trial update")
                raise TrialDone()
        def epoch_done(self, trainer):
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
        verify_stage(root, work, review)
        progress["artifacts"] = {name: sha(out / name) for name in ("fit/weights/last.pt", "fit/results.csv")}
        progress["status"] = "complete"
    except TrialDone:
        progress["status"] = "trial-complete"
    except Exception as error:
        progress["status"], progress["error"] = "failed", str(error)
        record()
        raise
    record()
    save(out / "receipt.json", progress)
    verify_trial(work, plan, runtime) if trial else verify_arm(work, role, plan, runtime)
    print(role, progress["status"], flush=True)


def evaluate(root, work, review):
    import numpy as np
    import torch
    from ultralytics import YOLO
    plan, runtime = verify_stage(root, work, review), r45.runtime()
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
    verify_evaluation(root, work, review)


def verify_evaluation(root, work, review):
    plan = verify_stage(root, work, review)
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


def return_members(root, work, review, verify_saved_scores=True):
    plan = verify_evaluation(root, work, review) if verify_saved_scores else verify_stage(root, work, review)
    receipts = {role: verify_arm(work, role, plan) for role in ROLES}
    need(receipts["reference"]["runtime"] == receipts["candidate"]["runtime"], "Mixed-runtime return")
    verify_trial(work, plan, receipts["reference"]["runtime"])
    members = {name: work / name for name in ("staging.json", "release-status.json", "evaluation/summary.json", "evaluation/predictions.json")}
    for relative, expected in plan["source_files_sha256"].items():
        members["source/" + relative] = checked(root / relative, expected)
    for role in ("resource-trial", *ROLES):
        for relative in ("receipt.json", "progress.json", "fit/results.csv", "fit/weights/last.pt"):
            path = work / role / relative
            if path.is_file() and not (role == "resource-trial" and relative.startswith("fit/")):
                members[role + "/" + relative] = path
        if (work / (role + ".log")).is_file():
            members[role + ".log"] = work / (role + ".log")
    for role in ROLES:
        for entry in plan["roles"][role]:
            members["datasets/" + role + "/" + entry["label"]] = work / "datasets" / role / entry["label"]
        members["datasets/" + role + "/data.yaml"] = work / "datasets" / role / "data.yaml"
    members["inputs/source-manifest.json"] = checked(root / "data/manifest.json", ORIGINAL_MANIFEST_SHA)
    members["inputs/r37-overlay-manifest.json"] = checked(root / r45.OVERLAY, r45.OVERLAY_HASH)
    members["inputs/pr133-frame0383-labels.txt"] = checked(root / OLD_EXTRA_LABEL, OLD_EXTRA_LABEL_SHA)
    for name in ("manifest.json", "audit-ledger.json", "outlines.json", "verification.json", "history/r44-manifest.json", "history/r46-manifest.json", "history/r46-audit-ledger.json", "history/r46-outlines.json"):
        members["review/" + name] = review / name
    for path in (review / "complete-candidates/labels").glob("*.txt"):
        members["review/complete-candidates/labels/" + path.name] = path
    for name, (relative, _, _) in REFS.items():
        path = root / relative
        members["references/" + name + "/manifest.json"] = path
        for entry in read(path)["files"]:
            if entry.get("split", "val") == "val":
                members["references/" + name + "/" + entry["label_path"]] = path.parent / entry["label_path"]
    return members


def verify_return(root, work, review, output):
    members = return_members(root, work, review)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) and set(names) == set(members) | {"inventory.json"}, "Return archive membership differs")
        need(archive.testzip() is None, "Return archive CRC failure")
        inventory = json.loads(archive.read("inventory.json"))
        need(set(inventory) == set(members), "Return inventory differs")
        for name, source in members.items():
            need(inventory[name] == dict(sha256=sha(source), bytes=source.stat().st_size), "Return source differs: " + name)
            need(archive.getinfo(name).file_size == source.stat().st_size and digest(archive.read(name)) == sha(source), "Return bytes differ: " + name)
    return dict(path=str(output), bytes=output.stat().st_size, sha256=sha(output), members=len(names))


def restore_return(root, work, review, output):
    """Reuse a verified completed return without regenerating GPU predictions.

Only the two named evaluation JSON files can be restored. Every other archive
member must match the current source, staged labels and completed run bytes.
The archive is never extracted generally, and incompatible local files remain.
"""
    members = return_members(root, work, review, verify_saved_scores=False)
    evaluation = {"evaluation/summary.json", "evaluation/predictions.json"}
    payload = {}
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) and set(names) == set(members) | {"inventory.json"}, "Backup return membership differs")
        need(archive.testzip() is None, "Backup return CRC failure")
        inventory = json.loads(archive.read("inventory.json"))
        need(set(inventory) == set(members), "Backup return inventory differs")
        for name, source in members.items():
            data = archive.read(name)
            need(inventory[name] == dict(sha256=digest(data), bytes=len(data)), "Backup return bytes differ: " + name)
            if name in evaluation:
                payload[name] = data
            else:
                need(inventory[name] == dict(sha256=sha(source), bytes=source.stat().st_size), "Backup return differs from current completed context: " + name)
    for name, data in payload.items():
        put(work / name, data)
    return verify_return(root, work, review, output)


def write_return_zip(members, output):
    """Stable member metadata makes identical evidence independent of mtimes."""
    inventory = {name: dict(sha256=sha(path), bytes=path.stat().st_size) for name, path in members.items()}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name in sorted([*members, "inventory.json"]):
            info = zipfile.ZipInfo(name, (2026, 11, 27, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            data = encoded(inventory) if name == "inventory.json" else members[name].read_bytes()
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)


def package(root, work, review, output):
    need(not output.resolve().is_relative_to(work.resolve()), "Return ZIP must be outside work")
    if output.exists():
        return verify_return(root, work, review, output)
    members = return_members(root, work, review)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    need(not temporary.exists(), "Existing incomplete archive preserved: " + str(temporary))
    write_return_zip(members, temporary)
    verify_return(root, work, review, temporary)
    temporary.replace(output)
    return verify_return(root, work, review, output)


def backup_paths(name):
    return ["receipt.json", "progress.json"] + ([] if name == "resource-trial" else ["fit/results.csv", "fit/weights/last.pt"])


def validate_backup(folder, name, identity, source):
    marker = read(folder / "backup.json")
    need(marker["source"] == source and marker["staging_sha256"] == identity and set(marker["files"]) == set(backup_paths(name)), "Backup protocol differs")
    for relative, expected in marker["files"].items():
        checked(folder / relative, expected)
    if marker.get("log_sha256"):
        checked(folder / "run.log", marker["log_sha256"])
    return marker


def restore_backup(work, backup, name, plan):
    if backup is None or (work / name).exists() or not (backup / name / "backup.json").exists():
        return
    source = backup / name
    marker = validate_backup(source, name, sha(work / "staging.json"), plan["source"])
    for relative in backup_paths(name):
        destination = work / name / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    if marker.get("log_sha256"):
        shutil.copyfile(source / "run.log", work / (name + ".log"))
    verify_trial(work, plan) if name == "resource-trial" else verify_arm(work, name, plan)
    print("Restored verified completed", name, flush=True)


def save_backup(work, backup, name, plan):
    if backup is None:
        return
    verify_trial(work, plan) if name == "resource-trial" else verify_arm(work, name, plan)
    source, destination = work / name, backup / name
    marker = dict(source=plan["source"], staging_sha256=sha(work / "staging.json"), files={relative: sha(source / relative) for relative in backup_paths(name)})
    log = work / (name + ".log")
    if log.is_file():
        marker["log_sha256"] = sha(log)
    if (destination / "backup.json").exists():
        need(validate_backup(destination, name, marker["staging_sha256"], plan["source"]) == marker, "Different completed backup preserved")
        return
    need(not destination.exists(), "Incomplete backup preserved: " + str(destination))
    destination.mkdir(parents=True)
    for relative in backup_paths(name):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    if log.is_file():
        shutil.copyfile(log, destination / "run.log")
    save(destination / "backup.json", marker)
    validate_backup(destination, name, marker["staging_sha256"], plan["source"])
    print("Saved verified completed", name, flush=True)


def run_pair(root, work, review, backup):
    plan = verify_stage(root, work, review)
    script = Path(__file__).resolve()
    for name, action, role in (("resource-trial", "trial", "candidate"), ("reference", "train", "reference"), ("candidate", "train", "candidate")):
        restore_backup(work, backup, name, plan)
        command = [sys.executable, "-B", str(script), action, "--root", str(root), "--work", str(work), "--review", str(review), "--role", role]
        if (work / name / "receipt.json").exists():
            subprocess.run(command, check=True)
        else:
            with (work / (name + ".log")).open("a", encoding="utf-8") as log:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    need(process.wait() == 0, name + " failed; inspect its preserved log and run directory")
        save_backup(work, backup, name, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "prepare", "verify", "trial", "train", "run-pair", "evaluate", "verify-evaluation", "package", "verify-return", "restore-return"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--download-inputs", action="store_true")
    parser.add_argument("--role", choices=ROLES, default="candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    root, work, review = args.root.resolve(), args.work.resolve(), args.review.resolve()
    if args.command == "plan":
        plan, _ = label_plan(root, review)
        print(json.dumps({key: value for key, value in plan.items() if key != "roles"}, indent=2))
    elif args.command == "prepare":
        # Check the complete-photo gate before any large input download.
        label_plan(root, review)
        inputs = (args.inputs_dir or root).resolve()
        if args.download_inputs:
            obtain_inputs(root, inputs)
        stage(root, work, review, inputs)
    elif args.command == "verify":
        verify_stage(root, work, review)
        print("R47 staged inputs verified; no training performed")
    elif args.command in ("trial", "train"):
        train(root, work, review, args.role, args.command == "trial")
    elif args.command == "run-pair":
        run_pair(root, work, review, args.backup_dir.resolve() if args.backup_dir else None)
    elif args.command == "evaluate":
        evaluate(root, work, review)
    elif args.command == "verify-evaluation":
        verify_evaluation(root, work, review)
        print("R47 cached evaluation replay passed")
    else:
        need(args.output is not None, "Specify --output")
        function = {"package": package, "verify-return": verify_return, "restore-return": restore_return}[args.command]
        result = function(root, work, review, args.output.resolve())
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
