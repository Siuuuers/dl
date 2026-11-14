#!/usr/bin/env python3
"""Focused checks for corruption geometry, RNG isolation, and the pinned pose pipeline."""

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from pose_input_corruption import DEFAULT_SPEC, PoseInputCorruption, validate_state


REAL_IDS = ("new_000001", "new_000013", "new_000027", "new_000133", "old_000004", "old_000010")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_batch(count=8):
    image = torch.linspace(0, 1, 3 * 96 * 160).reshape(3, 96, 160)
    boxes = [[70 / 160, 46 / 96, 90 / 160, 32 / 96], [31.5 / 160, 57.5 / 96, 9 / 160, 15 / 96],
             [110.5 / 160, 57.5 / 96, 9 / 160, 15 / 96]]
    points = [[[30 / 160, 60 / 96, 2], [110 / 160, 60 / 96, 2]],
              [[0.125, 0.25, 0], [31 / 160, 61 / 96, 2]],
              [[109 / 160, 61 / 96, 2], [0.6, 0.2, 0]]]
    return {"img": image.repeat(count, 1, 1, 1), "bboxes": torch.tensor(boxes * count),
            "keypoints": torch.tensor(points * count), "cls": torch.zeros(count * 3, 1),
            "batch_idx": torch.arange(count).repeat_interleave(3).float(),
            "im_file": [f"synthetic_{i:02d}.png" for i in range(count)]}


def unchanged_targets(before, after):
    def same(first, second):
        if isinstance(first, torch.Tensor):
            return isinstance(second, torch.Tensor) and torch.equal(first, second)
        if isinstance(first, np.ndarray):
            return isinstance(second, np.ndarray) and np.array_equal(first, second)
        if isinstance(first, dict):
            return first.keys() == second.keys() and all(same(first[k], second[k]) for k in first)
        if isinstance(first, (list, tuple)):
            return type(first) is type(second) and len(first) == len(second) and all(same(a, b) for a, b in zip(first, second))
        return first == second

    assert before.keys() == after.keys()
    for name in before:
        if name == "img":
            continue
        assert same(before[name], after[name]), name


def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def assert_rng_same(before, after):
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def check_pair(batch, control, candidate, epoch=1):
    clean, changed = copy.deepcopy(batch), copy.deepcopy(batch)
    before = rng_state()
    control(clean, epoch=epoch)
    candidate(changed, epoch=epoch)
    assert_rng_same(before, rng_state())
    unchanged_targets(batch, clean)
    unchanged_targets(batch, changed)
    assert torch.equal(batch["img"], clean["img"])
    a, b = control.state(), candidate.state()
    assert a["decision_sha256"] == b["decision_sha256"] and a["examples"] == b["examples"]
    for key in a["counters"]:
        if not key.endswith("_applied"):
            assert a["counters"][key] == b["counters"][key], key
    return changed


def check_mask_pixels(before, after, state, batch_number):
    masks = [e for e in state["examples"] if e["batch"] == batch_number]
    assert state["counters"]["images_planned"] <= 8, "check requires all bounded mask examples"
    allow = torch.zeros_like(before["img"], dtype=torch.bool)
    for event in masks:
        i, (x0, y0, x1, y1) = event["image_index"], event["mask_rect_xyxy"]
        allow[i, :, y0:y1, x0:x1] = True
        expected = torch.full_like(after["img"][i, :, y0:y1, x0:x1], event["fill_gray"] / 255)
        assert torch.equal(after["img"][i, :, y0:y1, x0:x1], expected)
        target, other = event["target_keypoints_xyv"][event["side"]], event["target_keypoints_xyv"][1 - event["side"]]
        assert min(target[0] - x0, target[1] - y0, x1 - target[0], y1 - target[1]) >= 3
        assert other[0] + 3 < x0 or other[0] - 3 >= x1 or other[1] + 3 < y0 or other[1] - 3 >= y1
    assert torch.equal(before["img"][~allow], after["img"][~allow])


def synthetic_checks():
    original = synthetic_batch()
    control, candidate = [PoseInputCorruption(DEFAULT_SPEC, role) for role in ("control", "candidate")]
    changed = check_pair(original, control, candidate)
    state = candidate.state()
    assert state["counters"]["images_applied"] > 0
    assert state["counters"]["collateral_known_endpoints_applied"] > 0
    check_mask_pixels(original, changed, state, 1)
    repeat, repeated = PoseInputCorruption(DEFAULT_SPEC, "candidate"), copy.deepcopy(original)
    repeat(repeated, epoch=1)
    assert torch.equal(changed["img"], repeated["img"]) and state == repeat.state()

    ineligible = synthetic_batch(2)
    for name in ("bboxes", "keypoints", "cls", "batch_idx"):
        ineligible[name] = ineligible[name][4:5]
    untouched = copy.deepcopy(ineligible)
    transform = PoseInputCorruption(DEFAULT_SPEC, "candidate")
    transform(ineligible)
    assert torch.equal(ineligible["img"], untouched["img"])
    unchanged_targets(untouched, ineligible)
    assert transform.state()["counters"]["empty_images"] == 1
    assert transform.state()["counters"]["ineligible_images"] == 1
    assert transform.state()["counters"]["source_missing_slots_seen"] == 1

    narrow = synthetic_batch()
    narrow["keypoints"][::3, :, 0] = torch.tensor([0.3, 0.325])
    bad = PoseInputCorruption(DEFAULT_SPEC, "candidate")
    before_narrow = narrow["img"].clone()
    bad(narrow)
    assert bad.state()["counters"]["geometry_rejected"] > 0
    assert bad.state()["counters"]["images_applied"] == 0 and torch.equal(before_narrow, narrow["img"])

    malformed = copy.deepcopy(state)
    malformed["counters"]["images_applied"] = 0
    try:
        validate_state(malformed, DEFAULT_SPEC, "candidate")
    except ValueError:
        pass
    else:
        raise AssertionError("role-inconsistent receipt accepted")
    return {"passed": True, "state": state,
            "checks": ["control identity", "deterministic private RNG", "unchanged source geometry and v0",
                       "exact mask pixels and protected opposite neighborhood", "collateral known-point inventory",
                       "empty and ineligible images", "single-attempt geometry rejection", "role receipt binding"]}


def check_loss_and_hook(root):
    from ultralytics.utils.loss import KeypointLoss, PoseLoss26

    criterion = SimpleNamespace(keypoint_loss=KeypointLoss(torch.tensor([0.5, 0.5])),
                                rle_loss=None, bce_pose=torch.nn.BCEWithLogitsLoss())
    prediction = torch.tensor([[[[12., 20., 0.], [60., 20., 0.]]]], requires_grad=True)
    known_and_missing = torch.tensor([[[[10., 20., 2.], [50., 20., 0.]]]])
    loss, _, _ = PoseLoss26.calculate_keypoints_loss(
        criterion, torch.tensor([[True]]), torch.tensor([[0]]), known_and_missing,
        torch.tensor([[1.]]), torch.tensor([[[0., 0., 100., 50.]]]), prediction)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0 and prediction.grad[0, 0, 0, 0] != 0
    assert prediction.grad[0, 0, 1, 0] == 0
    tree = ast.parse((root / "scripts/train_pose_comparison.py").read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_worker")
    assert worker.args.args[-1].arg == "input_transform" and isinstance(worker.args.defaults[-1], ast.Constant)
    assert worker.args.defaults[-1].value is None
    call_sites = []
    for method in ast.walk(worker):
        if isinstance(method, ast.FunctionDef):
            direct = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "input_transform"]
            if direct and method.name != "run_worker":
                call_sites.append((method.name, len(direct)))
    assert call_sites == [("preprocess_batch", 1)]
    return {"passed": True, "pinned_known_coordinate_gradient": float(prediction.grad[0, 0, 0, 0]),
            "missing_coordinate_gradient": float(prediction.grad[0, 0, 1, 0]),
            "training_hook_call_sites": call_sites, "default_hook": None}


def full_model_backward(batch, checkpoint, image_index):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from pose_checkpoint import get_recipe

    before_hash = digest(checkpoint)
    assert before_hash == "420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1"
    started = time.perf_counter()
    source_shape = list(batch["img"].shape)
    selected = batch["batch_idx"] == image_index
    batch = {"img": batch["img"][image_index:image_index + 1],
             "cls": batch["cls"][selected], "bboxes": batch["bboxes"][selected],
             "keypoints": batch["keypoints"][selected],
             "batch_idx": torch.zeros(int(selected.sum()), dtype=batch["batch_idx"].dtype),
             "im_file": [batch["im_file"][image_index]]}
    model = YOLO(str(checkpoint)).model
    model.args = get_cfg(overrides=get_recipe("reviewed-annotations-seed46-v1"))
    model.train().requires_grad_(True)
    assert model.names == {0: "cassette_front"} and list(model.model[-1].kpt_shape) == [2, 3]
    loss, _ = model(batch)
    total = loss.sum()
    assert torch.isfinite(total)
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    nonzero = sum(bool(torch.count_nonzero(g)) for g in gradients)
    assert nonzero > 0 and digest(checkpoint) == before_hash
    return {"passed": True, "device": "cpu", "image_shape": list(batch["img"].shape),
            "source_batch_shape": source_shape, "selected_corrupted_image_index": image_index,
            "production_batch_size_verified": False,
            "loss": float(total.detach()), "gradient_parameter_count": len(gradients),
            "nonzero_gradient_parameter_count": nonzero, "checkpoint_sha256": before_hash,
            "wall_seconds": time.perf_counter() - started, "optimizer_step_executed": False,
            "checkpoint_written": False, "scope": "One actual corrupted image forward/backward integration at full input resolution; batch1 smoke check, not a training result or batch4 resource check."}


def real_pipeline_checks(root, image_root, output, checkpoint=None):
    from PIL import Image, ImageDraw
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import init_seeds
    from pose_checkpoint import get_recipe

    records = {r["id"]: r for r in json.loads((root / "data/manifest.json").read_text())["records"]}
    sources, control, candidate = [], PoseInputCorruption(DEFAULT_SPEC, "control"), PoseInputCorruption(DEFAULT_SPEC, "candidate")
    preview, integration = None, None
    with tempfile.TemporaryDirectory(prefix="pose-corruption-check-") as temporary:
        stage = Path(temporary)
        (stage / "images/train").mkdir(parents=True)
        (stage / "labels/train").mkdir(parents=True)
        for image_id in REAL_IDS:
            item = records[image_id]
            image, label = image_root / (image_id + ".png"), root / "data" / item["label"]
            assert item["split"] == "train" and digest(image) == item["image_sha256"] and digest(label) == item["label_sha256"]
            shutil.copyfile(image, stage / "images/train" / image.name)
            shutil.copyfile(label, stage / "labels/train" / label.name)
            sources.append({"image_id": image_id, "image_sha256": item["image_sha256"], "label_sha256": item["label_sha256"]})
        recipe = get_recipe("reviewed-annotations-seed46-v1")
        init_seeds(46, deterministic=True)
        dataset = YOLODataset(img_path=str(stage / "images/train"), imgsz=1280, augment=True,
                              hyp=get_cfg(overrides=recipe), rect=True, batch_size=4, stride=32, pad=0,
                              cache=False, task="pose", data={"names": {0: "cassette_front"}, "kpt_shape": [2, 3], "flip_idx": [1, 0]})
        for start in range(0, len(dataset), 4):
            batch = YOLODataset.collate_fn([dataset[i] for i in range(start, min(start + 4, len(dataset)))])
            batch = DetectionTrainer.preprocess_batch(SimpleNamespace(device=torch.device("cpu"), args=SimpleNamespace(multi_scale=0.0)), batch)
            changed = check_pair(batch, control, candidate)
            state = candidate.state()
            check_mask_pixels(batch, changed, state, state["counters"]["batches_seen"])
            events = [e for e in state["examples"] if e["batch"] == state["counters"]["batches_seen"]]
            if checkpoint is not None and integration is None and events:
                integration = full_model_backward(copy.deepcopy(changed), checkpoint, events[0]["image_index"])
            if preview is None and events:
                event = events[0]
                x0, y0, x1, y1 = event["target_box_xyxy"]
                width, height = event["size_wh"]
                crop = [max(0, int(x0) - 36), max(0, int(y0) - 36), min(width, int(x1) + 37), min(height, int(y1) + 37)]
                images = []
                for source, title in ((batch, "Before corruption"), (changed, "Candidate input")):
                    array = (source["img"][event["image_index"]].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    image = Image.fromarray(array).crop(crop)
                    scale = min(480 / image.width, 240 / image.height)
                    image = image.resize((round(image.width * scale), round(image.height * scale)))
                    canvas = Image.new("RGB", (480, 270), "white")
                    canvas.paste(image, ((480 - image.width) // 2, 30 + (240 - image.height) // 2))
                    ImageDraw.Draw(canvas).text((8, 8), title + " · " + event["image_id"], fill="black")
                    images.append(canvas)
                panel = Image.new("RGB", (960, 270), "white")
                panel.paste(images[0], (0, 0))
                panel.paste(images[1], (480, 0))
                panel.save(output / "real-corruption-preview.png")
                preview = {"event": event, "display_crop_xyxy": crop,
                           "sha256": digest(output / "real-corruption-preview.png"),
                           "meaning": "Display crop of actual pose pipeline tensors before/after corruption; no inference or model prediction."}
    assert candidate.state()["counters"]["images_seen"] == 6 and candidate.state()["counters"]["images_applied"] > 0
    return {"passed": True, "sources": sources, "control": control.state(), "candidate": candidate.state(), "preview": preview,
            "model_training_executed": False, "standard_pose_pipeline_executed": True,
            "full_model_forward_backward": integration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--real-image-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Optional verified PR35 weights for one actual CPU batch backward check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("ultralytics") == "8.4.182"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "runtime": {"torch": torch.__version__, "ultralytics": "8.4.182"},
              "source_sha256": {name: digest(args.repo_root / "scripts" / name) for name in
                                ("pose_input_corruption.py", "check_pose_input_corruption.py", "train_pose_comparison.py")},
              "synthetic": synthetic_checks(), "loss_and_hook": check_loss_and_hook(args.repo_root)}
    if args.real_image_root:
        report["real_pipeline"] = real_pipeline_checks(args.repo_root, args.real_image_root, args.output, args.checkpoint)
    report["status"] = "passed"
    (args.output / "checks.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "real_images": 6 if args.real_image_root else 0,
                      "model_training_executed": False, "pinned_loss_component_backward_executed": True}))


if __name__ == "__main__":
    main()
