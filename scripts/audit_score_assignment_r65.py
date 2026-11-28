#!/usr/bin/env python3
"""Inspect frozen native/QFL pose assignments without training or annotation edits."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from types import MethodType, SimpleNamespace
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED

sys.dont_write_bytecode = True
import run_quality_focal_r61 as r61

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "baomai-score-assignment-r65"
RETURN_COMMIT = "f971e880e8aff2859ce4d8ae2923efea826eddb4"
RETURN_PATH = "results/a100/baomai-quality-focal-r61-a100-001"
RETURN_MANIFEST_SHA = "8e9b60725a701f79048435772df4fdc07385077dc8d5efd7049833ababba3473"
EVIDENCE_FILE = "baomai-quality-focal-r61-results.zip"
EVIDENCE_SHA = "cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a"
STAGING_SHA = "7914e2e622b76b4947067b8b8abeee8e01efdd4c76c013e8c6850fa93be69c45"
MODELS = {
    "reference": "15d7c97d5c58f16e40462807231c7bb01a5ee9a719225cb756493e1b41714d67",
    "candidate": "8760602df814f47685dde25a2e389a554f623a7280473c90d5e94a0f30fe519f",
}
NATIVE_FILES = {
    "utils/loss.py": "f8db64a3093ef310b23cb83d587dd5c8aa3db20a27da94d95a62a734a6c055d2",
    "utils/tal.py": "7fc3127e4b094247ccbbcf8af9829972295f96fc53027313ef51b076d27274cd",
    "nn/modules/head.py": "b471bb8a39cc1e38672c20feaa8c28bdf43739adb8eafd14226807b34b31f3ee",
    "nn/tasks.py": "1048ab8db7504e7c57a77d05bfbd855e9de0ef477e15a5aae8cdb7e8a958246a",
    "data/augment.py": "802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3",
    "utils/instance.py": "ebc8b6337f4636c1457473c079d007693b3d8766c4891e2b5dd521c22c2e5527",
    "cfg/default.yaml": "b91e4622e293270595e6e58d5bc3f3cdcb4a1d4f8b28c9035ec883bb2835081f",
}
SMOKE_IDS = ("new_000001", "new_000002", "new_000131")
RESULT_FILES = ("summary.json", "targets.json.gz", "images.json", "groups.csv", "results.md")
PROTOCOL = {
    "training": False, "optimizer_updates": 0, "annotation_edits": False,
    "models": "the two complete final checkpoints returned by PR157",
    "photos": 142, "source_fronts": 3999, "source_labels": "unchanged r61 training labels",
    "model_mode": "eval, unfused, FP32; all BatchNorm and dropout remain in eval",
    "inputs": "unaugmented BGR images; native LetterBox(auto=True,stride=32,scaleup=False,new_shape=(1280,1280)); RGB tensor divided by255",
    "expected_input_chw": [3, 736, 1280], "batch_size": 1,
    "prediction_pool": "both raw native output heads, before confidence filtering or NMS",
    "known_corner_groups": [0, 1, 2],
    "native_overlap": "mean(clamped CIoU, known-only OKS), or clamped CIoU for zero known corners",
    "counterfactual": "native TaskAlignedAssigner with box-only overlap; same decoded predictions, candidate-center rule, alpha/beta/topk/topk2; descriptive only",
    "anchor_detail": "all eligible IDs and eligible-value summaries; p/q/components for union of pre-conflict and surviving anchors",
    "q_for_unassigned_anchor": "zero for this target; never borrow the target score of another object",
    "ineligible_owner_edge_case": "retain native zero-overlap conflict ownership; flag it and leave unevaluated pair components null",
    "score_differences": "p minus q on anchors assigned to this target; not calibration against observed correctness",
    "historical_training_replay": False, "recognition_accuracy_measurement": False,
}
need, sha, checked, encoded, put, read = r61.need, r61.sha, r61.checked, r61.encoded, r61.put, r61.read


def source_check(root, commit, driver_sha):
    need(root == ROOT and re.fullmatch("[0-9a-f]{40}", commit or "")
         and re.fullmatch("[0-9a-f]{64}", driver_sha or ""), "Complete source identities required")
    need(commit != "0" * 40 and r61.rev(root) == commit, "Wrong or unpublished source checkout")
    need(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                     cwd=root, text=True).strip(), "Changed source preserved; use a fresh audit folder")
    checked(Path(__file__), driver_sha)
    closure = {**r61.local_source_files(root), "scripts/" + Path(__file__).name: driver_sha}
    tree = subprocess.check_output(["git", "ls-tree", "-r", "HEAD"], cwd=root, text=True)
    blobs = {line.split("\t", 1)[1]: line.split()[2] for line in tree.splitlines()}
    for name, expected in closure.items():
        data = checked(root / name, expected).read_bytes()
        need(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == blobs.get(name),
             "Source Git object differs: " + name)
    return {"commit": commit, "driver_sha256": driver_sha, "files_sha256": closure}


def native_contract():
    import ultralytics
    need(ultralytics.__version__ == "8.4.182", "Ultralytics8.4.182 required")
    package = Path(ultralytics.__file__).parent
    for name, expected in NATIVE_FILES.items():
        checked(package / name, expected)
    fixtures = check_edge_fixtures()
    return {"python": platform.python_version(), "packages": {name: version(name) for name in
             ("torch", "torchvision", "ultralytics", "numpy", "opencv-python", "Pillow")},
            "native_files_sha256": dict(NATIVE_FILES), "observation_edge_fixtures": fixtures}


def download(url, path, expected):
    if path.exists():
        return checked(path, expected)
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    need(hashlib.sha256(data).hexdigest() == expected, "Downloaded bytes differ: " + path.name)
    put(path, data)
    return path


def obtain_models(folder, allow_download):
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"https://raw.githubusercontent.com/Siuuuers/dl/{RETURN_COMMIT}/{RETURN_PATH}/"
    manifest_path = folder / "return-manifest.json"
    if allow_download:
        download(prefix + manifest_path.name, manifest_path, RETURN_MANIFEST_SHA)
    manifest = read(checked(manifest_path, RETURN_MANIFEST_SHA))
    for role, expected in MODELS.items():
        spec = manifest["models"][role]
        need(spec["model_sha256"] == expected, "Checkpoint identity differs")
        path = folder / spec["model_file"]
        if not path.exists():
            need(allow_download, "Missing model; provide verified models-dir or enable downloads")
            parts = []
            for item in spec["parts"]:
                wrapper = download(prefix + item["file"], folder / item["file"], item["sha256"])
                need(wrapper.stat().st_size == item["bytes"], "Wrapper size differs")
                with ZipFile(wrapper) as z:
                    need(z.namelist() == [item["member"]], "Unexpected model wrapper members")
                    data = z.read(item["member"])
                need(len(data) == item["member_bytes"] and hashlib.sha256(data).hexdigest() == item["member_sha256"],
                     "Checkpoint part differs")
                parts.append(data)
            data = b"".join(parts)
            need(len(data) == spec["model_bytes"] and hashlib.sha256(data).hexdigest() == expected,
                 "Reconstructed model differs")
            put(path, data)
        checked(path, expected)
        need(path.stat().st_size == spec["model_bytes"], "Checkpoint size differs")
    return manifest


def obtain_evidence(folder, allow_download=True):
    path = folder / EVIDENCE_FILE
    if allow_download:
        download(f"https://raw.githubusercontent.com/Siuuuers/dl/{RETURN_COMMIT}/{RETURN_PATH}/{EVIDENCE_FILE}", path, EVIDENCE_SHA)
    return checked(path, EVIDENCE_SHA)


def make_plan(root, archive, source):
    checked(archive, EVIDENCE_SHA)
    with ZipFile(archive) as z:
        staged = z.read("staging.json")
        need(hashlib.sha256(staged).hexdigest() == STAGING_SHA, "Returned training inventory differs")
        upstream = json.loads(staged)
        entries = sorted((r for r in upstream["roles"]["reference"] if r["split"] == "train"), key=lambda r: r["id"])
        payload = {row["id"]: z.read("datasets/reference/" + row["label"]) for row in entries}
        for row in entries:
            need(hashlib.sha256(payload[row["id"]]).hexdigest() == row["label_sha256"], "Returned training label differs")
    need(upstream["roles"]["reference"] == upstream["roles"]["candidate"], "Paired source data differs")
    need(len(entries) == 142 and upstream["counts"]["reference"]["train"] == [142, 3999, 7259, 739, 13],
         "Training inventory differs")
    return {"schema": "score-assignment-audit-r65-plan-v1", "source": source, "protocol": PROTOCOL,
            "entries": entries, "counts": upstream["counts"]["reference"]["train"],
            "input_archive": {"sha256": EVIDENCE_SHA, "bytes": archive.stat().st_size, "commit": RETURN_COMMIT,
                              "staging_sha256": STAGING_SHA}, "models_sha256": MODELS,
            "return_commit": RETURN_COMMIT, "return_manifest_sha256": RETURN_MANIFEST_SHA}, payload


def prepare(root, work, archive, assets, models, source, allow_download=False):
    plan, payload = make_plan(root, archive, source)
    if allow_download:
        def obtain(row):
            url = f"https://media.githubusercontent.com/media/Siuuuers/dl/{r61.r47.SOURCE_INPUTS}/{row['image_source']}"
            download(url, assets / row["image_source"], row["image_sha256"])
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(obtain, [row for row in plan["entries"] if row["source_kind"] == "original"]))
        r61.base.obtain_supplementary(root, assets, plan["entries"])
    obtain_models(models, allow_download)
    for row in plan["entries"]:
        image = checked(assets / row["image_source"], row["image_sha256"])
        need(r61.image_size(image.read_bytes()) == (1280, 720), "Original image dimensions differ")
        put(work / "dataset" / row["image"], image.read_bytes())
        put(work / "dataset" / row["label"], payload[row["id"]])
    for role, expected in MODELS.items():
        put(work / "models" / (role + "-last.pt"), checked(models / (role + "-last.pt"), expected).read_bytes())
    put(work / "inputs/pr157-return-manifest.json", checked(models / "return-manifest.json", RETURN_MANIFEST_SHA).read_bytes())
    put(work / "plan.json", encoded(plan))
    verify_stage(root, work, archive, source)
    print("Prepared 142 training photographs and two verified frozen checkpoints.", flush=True)


def verify_stage(root, work, archive, source):
    plan, payload = make_plan(root, archive, source)
    need(read(work / "plan.json") == plan, "Saved input/source plan differs")
    for row in plan["entries"]:
        checked(work / "dataset" / row["image"], row["image_sha256"])
        checked(work / "dataset" / row["label"], row["label_sha256"])
    for role, expected in MODELS.items():
        checked(work / "models" / (role + "-last.pt"), expected)
    checked(work / "inputs/pr157-return-manifest.json", RETURN_MANIFEST_SHA)
    return plan


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode() + value.numpy().tobytes())
    return digest.hexdigest()


def input_batch(work, entry, device):
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.instance import Instances
    image = cv2.imread(str(work / "dataset" / entry["image"]))
    need(image is not None and image.shape == (720, 1280, 3), "Input raster differs")
    rows = np.array([list(map(float, line.split())) for line in (work / "dataset" / entry["label"]).read_text().splitlines()
                     if line.strip()], dtype=np.float32).reshape(-1, 11)
    points = rows[:, 5:].reshape(-1, 2, 3).copy()
    labels = {"img": image, "cls": rows[:, :1].copy(),
              "instances": Instances(rows[:, 1:5].copy(), keypoints=points.copy(), bbox_format="xywh", normalized=True)}
    transformed = LetterBox(new_shape=(1280, 1280), auto=True, stride=32, scaleup=False)(labels)
    image, inst = transformed["img"], transformed["instances"]
    h, w = image.shape[:2]
    need((h, w) == (736, 1280) and len(inst) == len(rows), "Letterbox shape or row inventory differs")
    need(np.array_equal(inst.keypoints[..., 2], points[..., 2]), "Letterbox changed availability")
    inst.convert_bbox("xywh")
    boxes = inst.bboxes.copy() / np.array([w, h, w, h], dtype=np.float32)
    keypoints = inst.keypoints.copy()
    keypoints[..., :2] /= np.array([w, h], dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).unsqueeze(0).to(device).float() / 255
    return {"img": tensor, "batch_idx": torch.zeros(len(rows), device=device),
            "cls": torch.from_numpy(rows[:, :1]).to(device),
            "bboxes": torch.from_numpy(boxes).to(device), "keypoints": torch.from_numpy(keypoints).to(device)}


def assignment_inputs(branch, pred, batch):
    import torch
    from ultralytics.utils.tal import make_anchors
    scores = pred["scores"].permute(0, 2, 1).contiguous()
    distri = pred["boxes"].permute(0, 2, 1).contiguous()
    anchors, strides = make_anchors(pred["feats"], branch.stride, .5)
    imgsz = torch.tensor(pred["feats"][0].shape[2:], device=scores.device, dtype=scores.dtype) * branch.stride[0]
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = branch.preprocess(targets.to(scores.device), 1, scale_tensor=imgsz[[1, 0, 1, 0]])
    labels, boxes = targets.split((1, 4), 2)
    pd_boxes = (branch.bbox_decode(anchors, distri).detach() * strides).type(boxes.dtype)
    points = pred["kpts"].permute(0, 2, 1).contiguous().view(1, -1, *branch.kpt_shape)
    points = branch.kpts_decode(anchors, points)
    args = (scores.detach().sigmoid(), pd_boxes, anchors * strides, labels, boxes, boxes.sum(2, keepdim=True).gt_(0.0))
    extra = {"pd_kpts": points.detach()[..., :2] * strides[:, None], "gt_kpts": branch._batch_keypoints(batch, 1, imgsz)}
    return args, extra, strides


def clone_tensors(value):
    if isinstance(value, dict):
        return {k: clone_tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_tensors(v) for v in value)
    return value.detach().clone()


def tensors_equal(first, second):
    import torch
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(tensors_equal(first[k], second[k]) for k in first)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(tensors_equal(a, b) for a, b in zip(first, second))
    return torch.equal(first, second)


def native_input_guard(branch, pred, batch, args, extra):
    observed = {}
    def capture(module, call_args, kwargs, result):
        observed.update(args=clone_tensors(call_args), extra=clone_tensors(kwargs), result=clone_tensors(result))
    handle = branch.assigner.register_forward_hook(capture, with_kwargs=True)
    try:
        branch.loss(pred, batch)
    finally:
        handle.remove()
    need(tensors_equal(observed["args"], args) and tensors_equal(observed["extra"], extra),
         "Diagnostic assignment inputs differ from native PoseLoss26.loss")
    return observed["result"]


def stats(values):
    values = sorted(map(float, values))
    if not values:
        return {"count": 0, "mean": None, "min": None, "median": None, "max": None}
    mid = (len(values) - 1) / 2
    return {"count": len(values), "mean": math.fsum(values) / len(values), "min": values[0],
            "median": (values[math.floor(mid)] + values[math.ceil(mid)]) / 2, "max": values[-1]}


def observe_assignment(assigner, args, extra, *, box_only, guard=None):
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner
    p, pd_boxes, centers, labels, gt_boxes, mask_gt = args
    eligible = assigner.select_candidates_in_gts(centers, gt_boxes, mask_gt) * mask_gt.bool()
    indices = eligible.nonzero(as_tuple=True)
    captured = {}
    original = assigner.get_pos_mask
    def get_pos_mask(this, *pos_args, **kwargs):
        result = original(*pos_args, **kwargs)
        captured["pre"] = result[0].detach().cpu().clone()
        captured["overlap"] = result[2][indices].detach().cpu().clone()
        return result
    assigner.get_pos_mask = MethodType(get_pos_mask, assigner)
    try:
        output = assigner(*args, **({} if box_only else extra))
    finally:
        assigner.get_pos_mask = original
    if guard is not None:
        need(tensors_equal(output, guard), "Observer changed native returned assignment tensors")
    ciou = TaskAlignedAssigner.pair_overlaps(assigner, indices, gt_boxes, pd_boxes)
    gt_points = extra["gt_kpts"][indices[0], indices[1]]
    vis = gt_points[..., 2] > 0
    area = (gt_boxes[indices[:2]][:, 2:] - gt_boxes[indices[:2]][:, :2]).prod(-1, keepdim=True) * .53
    distance = (extra["pd_kpts"][indices[0], indices[2]].float() - gt_points[..., :2]).pow(2).sum(-1)
    sigmas = extra["sigmas"] if "sigmas" in extra else assigner.sigmas
    oks = ((-distance / ((2 * sigmas).pow(2) * (area + assigner.eps) * 2)).exp() * vis).sum(-1) / vis.sum(-1).clamp(min=1)
    expected = ciou if box_only else torch.where(vis.any(-1), (ciou + oks.to(ciou.dtype)) / 2, ciou)
    if len(indices[0]):
        need(torch.equal(expected.cpu(), captured["overlap"]), "Independent CIoU/OKS decomposition differs")
    _, _, targets, foreground, target_index = output
    data = {"p": p[0, :, 0].cpu(), "q": targets[0, :, 0].cpu(), "fg": foreground[0].cpu(),
            "gt": target_index[0].cpu(), "eligible": eligible[0].cpu(),
            "pre": captured.get("pre", torch.zeros_like(eligible).cpu())[0],
            "indices": tuple(v.cpu() for v in indices), "ciou": ciou.cpu(), "oks": oks.cpu(),
            "quality": expected.cpu(), "boxes": gt_boxes[0].cpu(), "centers": centers.cpu(),
            "points": extra["gt_kpts"][0].cpu(), "guard_passed": guard is not None}
    return data


def target_records(data, strides, identity, role, branch, mode):
    import torch
    rows = []
    strides = strides[:, 0].cpu()
    component = {(int(g), int(a)): index for index, (g, a) in enumerate(zip(data["indices"][1], data["indices"][2]))}
    for g in range(len(data["boxes"])):
        eligible = data["eligible"][g].nonzero().flatten().tolist()
        before = data["pre"][g].nonzero().flatten().tolist()
        assigned = ((data["gt"] == g) & data["fg"]).nonzero().flatten().tolist()
        known = int((data["points"][g, :, 2] > 0).sum())
        box = data["boxes"][g].tolist()
        rows.append({"image_id": identity, "source_row": g + 1, "role": role, "branch": branch, "mode": mode,
                     "known_corners": known, "bbox_xyxy_input_px": box, "width_px": box[2] - box[0], "height_px": box[3] - box[1],
                     "eligible_count": len(eligible), "preconflict_count": len(before), "surviving_count": len(assigned),
                     "eligible_anchor_indices": eligible, "preconflict_anchor_indices": before, "surviving_anchor_indices": assigned,
                     "foreground_zero_q_count": sum(float(data["q"][a]) == 0 for a in assigned),
                     "eligible_p": stats(float(data["p"][a]) for a in eligible),
                     "eligible_ciou": stats(float(data["ciou"][component[g, a]]) for a in eligible),
                     "eligible_oks": stats(float(data["oks"][component[g, a]]) for a in eligible) if known else None,
                     "eligible_quality": stats(float(data["quality"][component[g, a]]) for a in eligible),
                     "anchors": []})
        for a in sorted(set(before) | set(assigned)):
            index = component.get((g, a))
            q = float(data["q"][a]) if a in assigned else 0.0
            if index is None:
                need(a in assigned and a not in before and q == 0.0,
                     "Unexpected nonzero assignment outside the native eligible pool")
            rows[-1]["anchors"].append({"anchor_index": a, "stride": float(strides[a]),
                 "center_xy_input_px": data["centers"][a].tolist(), "p": float(data["p"][a]), "q": q,
                 "pair_evaluated_for_assignment": index is not None, "owner_outside_eligible": index is None,
                 "ciou_clamped": float(data["ciou"][index]) if index is not None else None,
                 "oks_known_only": float(data["oks"][index]) if known and index is not None else None,
                 "assignment_quality": float(data["quality"][index]) if index is not None else 0.0,
                 "preconflict": a in before, "surviving": a in assigned,
                 "p_minus_q": float(data["p"][a]) - q if a in assigned else None})
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["role"], row["branch"], row["mode"], row["known_corners"]].append(row)
    result = []
    for key, targets in sorted(grouped.items()):
        anchors = [a for r in targets for a in r["anchors"] if a["surviving"]]
        result.append({"role": key[0], "branch": key[1], "mode": key[2], "known_corners": key[3],
                       "targets": len(targets), "without_eligible_center": sum(not r["eligible_count"] for r in targets),
                       "without_assignment": sum(not r["surviving_count"] for r in targets),
                       "assigned_but_all_q_zero": sum(bool(r["surviving_count"]) and r["surviving_count"] == r["foreground_zero_q_count"] for r in targets),
                       "surviving_owner_outside_eligible": sum(a["owner_outside_eligible"] for a in anchors),
                       "surviving_anchors": len(anchors),
                       **{name: stats(a[name] for a in anchors if a[name] is not None) for name in
                          ("p", "q", "p_minus_q", "ciou_clamped", "oks_known_only", "assignment_quality")}})
    return result


def check_edge_fixtures():
    """Exercise native zero-overlap ties and ownership without checkpoint loading."""
    import torch
    from ultralytics.utils.tal import PoseTaskAlignedAssigner
    assigner = PoseTaskAlignedAssigner(torch.tensor([.5, .5]), topk=3, topk2=3,
                                      num_classes=1, alpha=.5, beta=6, stride=[8, 16, 32])
    gt = torch.tensor([[[100., 100., 110., 110.], [0., 0., 10., 10.], [0., 0., 10., 10.]]])
    args = (torch.full((1, 3, 1), .5), torch.tensor([[[1000., 1000., 1010., 1010.]] * 3]),
            torch.tensor([[3., 3.], [5., 5.], [7., 7.]]), torch.zeros((1, 3, 1)), gt,
            torch.ones((1, 3, 1), dtype=torch.bool))
    extra = {"pd_kpts": torch.zeros((1, 3, 2, 2)), "gt_kpts": torch.zeros((1, 3, 2, 3))}
    with torch.no_grad():
        native = clone_tensors(assigner(*args, **extra))
        observed = observe_assignment(assigner, args, extra, box_only=False, guard=native)
        rows = target_records(observed, torch.ones((3, 1)), "fixture", "fixture", "fixture", "native")
        need(rows[0]["eligible_count"] == 0 and rows[0]["surviving_count"] == 3
             and rows[0]["foreground_zero_q_count"] == 3
             and all(a["owner_outside_eligible"] and a["ciou_clamped"] is None for a in rows[0]["anchors"]),
             "Zero-overlap conflict ownership was hidden or misreported")
        good_args = (args[0], torch.tensor([[[0., 0., 10., 10.]] * 3]), *args[2:])
        good = observe_assignment(assigner, good_args, extra, box_only=False,
                                  guard=clone_tensors(assigner(*good_args, **extra)))
        owned = target_records(good, torch.ones((3, 1)), "fixture", "fixture", "fixture", "native")
        need(owned[1]["surviving_count"] == 3 and all(a["q"] > 0 for a in owned[1]["anchors"])
             and owned[2]["preconflict_count"] == 3 and owned[2]["surviving_count"] == 0
             and all(a["q"] == 0 for a in owned[2]["anchors"]), "Another target's q leaked into losing candidates")
    return {"native_zero_overlap_conflict_retained": True, "per_target_q_ownership": True,
            "fixture_is_observed_checkpoint_case": False}


def audit(work, device="auto", smoke=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.tal import TaskAlignedAssigner
    runtime = native_contract()
    torch.set_num_threads(2)
    torch.manual_seed(46)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    runtime.update(device=device, cuda=torch.version.cuda, cpu_threads=2,
                   gpu_name=torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                   cudnn=torch.backends.cudnn.version(),
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                   tf32_cudnn=torch.backends.cudnn.allow_tf32,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                   cudnn_deterministic=torch.backends.cudnn.deterministic,
                   cudnn_benchmark=torch.backends.cudnn.benchmark)
    plan = read(work / "plan.json")
    entries = [r for r in plan["entries"] if not smoke or r["id"] in SMOKE_IDS]
    need(len(entries) == (3 if smoke else 142), "Audit image scope differs")
    folder = work / ("smoke" if smoke else "results")
    need(not folder.exists(), "Existing audit preserved; verify it or use a fresh work folder")
    rows, image_rows, invariants = [], [], {}
    with torch.inference_mode():
        for role, expected in MODELS.items():
            model = YOLO(str(checked(work / "models" / (role + "-last.pt"), expected))).model.float().to(device).eval()
            need(type(model.model[-1]).__name__ == "Pose26" and all(not m.training for m in model.modules()), "Expected frozen Pose26 eval modules")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            # Exported inference metadata omits some loss gains. Restore the
            # original recipe plus pinned native defaults solely for loss guards.
            model.args = get_cfg(overrides={**r61.get_recipe(r61.RECIPE), "task": "pose", "mode": "train"})
            before_hash = state_hash(model)
            criterion = model.init_criterion()
            need(type(criterion).__name__ == "E2ELoss", "Expected both native pose loss branches")
            guarded = {name: set() for name in ("one2many", "one2one")}
            checks = []
            for image_index, entry in enumerate(entries):
                batch = input_batch(work, entry, device)
                outputs = model(batch["img"])
                need(isinstance(outputs, tuple) and set(outputs[1]) == {"one2many", "one2one"}, "Missing native raw heads")
                raw = outputs[1]
                known = (batch["keypoints"][..., 2] > 0).sum(-1).tolist()
                categories = set(map(int, known)) if known else {"empty"}
                for branch_name in ("one2many", "one2one"):
                    branch = getattr(criterion, branch_name)
                    need(type(branch).__name__ == "PoseLoss26", "Unexpected native criterion")
                    args, extra, strides = assignment_inputs(branch, raw[branch_name], batch)
                    needs_guard = bool(categories - guarded[branch_name])
                    baseline = native_input_guard(branch, raw[branch_name], batch, args, extra) if needs_guard else None
                    original = observe_assignment(branch.assigner, args, extra, box_only=False, guard=baseline)
                    rows.extend(target_records(original, strides, entry["id"], role, branch_name, "native"))
                    attrs = {k: getattr(branch.assigner, k) for k in ("topk", "topk2", "num_classes", "alpha", "beta", "stride", "eps")}
                    box_assigner = TaskAlignedAssigner(**attrs)
                    box_assigner.sigmas = branch.assigner.sigmas
                    counter = observe_assignment(box_assigner, args, extra, box_only=True)
                    rows.extend(target_records(counter, strides, entry["id"], role, branch_name, "box_only_counterfactual"))
                    changed = int(((original["fg"] != counter["fg"]) | ((original["gt"] != counter["gt"]) & (original["fg"] | counter["fg"]))).sum())
                    image_rows.append({"role": role, "image_id": entry["id"], "branch": branch_name,
                          "source_targets": len(known), "known_corner_counts": {str(k): known.count(k) for k in (0, 1, 2)},
                          "raw_anchors": len(original["p"]), "input_shape": list(batch["img"].shape),
                          "native_foreground": int(original["fg"].sum()), "box_only_foreground": int(counter["fg"].sum()),
                          "changed_foreground_or_owner_anchors": changed,
                          "native_background_p": stats(original["p"][~original["fg"]].tolist()),
                          "observer_and_native_input_guard_passed": needs_guard,
                          "assigner_parameters": {k: v for k, v in attrs.items()}, "oks_sigmas": branch.assigner.sigmas.cpu().tolist()})
                    if needs_guard:
                        guarded[branch_name].update(categories)
                        checks.append({"image_id": entry["id"], "branch": branch_name, "categories": sorted(map(str, categories)),
                                       "native_inputs_exact": True, "native_returned_assignments_exact": True})
                if (image_index + 1) % 10 == 0 or image_index + 1 == len(entries):
                    print(f"{role}: {image_index + 1}/{len(entries)} frozen photographs audited", flush=True)
            need(all(value == {0, 1, 2, "empty"} for value in guarded.values()), "Missing 0/1/2-known or empty-image guards")
            after_hash = state_hash(model)
            need(before_hash == after_hash and all(not m.training for m in model.modules())
                 and all(p.grad is None for p in model.parameters()), "Model state/BatchNorm changed or gradient was produced")
            invariants[role] = {"parameter_and_buffer_sha256_before": before_hash,
                                "parameter_and_buffer_sha256_after": after_hash,
                                "all_modules_eval": True, "all_parameter_gradients_none": True, "guards": checks}
            del model, criterion
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    groups = summarize(rows)
    result = {"schema": "score-assignment-audit-r65-v1", "scope": "three_photo_smoke" if smoke else "all142_training_photos",
              "completed": True, "full142_completed": not smoke, "photographs_per_model": len(entries),
              "forward_calls": len(entries) * 2, "source_fronts_per_model": sum(r["source_targets"] for r in image_rows if r["role"] == "reference" and r["branch"] == "one2many"),
              "protocol": PROTOCOL, "source": plan["source"], "plan_sha256": sha(work / "plan.json"),
              "models_sha256": MODELS, "runtime": runtime, "invariants": invariants, "groups": groups,
              "interpretation": "These are frozen, unaugmented training-image diagnostics, not historical training replay, a model improvement, or independent-session evidence."}
    put(folder / "summary.json", encoded(result))
    put(folder / "targets.json.gz", gzip.compress(encoded(rows), mtime=0))
    put(folder / "images.json", encoded(image_rows))
    flat = [{k: (v["mean"] if isinstance(v, dict) else v) for k, v in group.items()} for group in groups]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    put(folder / "groups.csv", buffer.getvalue().encode())
    put(folder / "results.md", markdown(result).encode())
    put(folder / "manifest.json", encoded({name: {"bytes": (folder / name).stat().st_size, "sha256": sha(folder / name)}
        for name in RESULT_FILES}))
    verify_result(work, smoke)
    return result


def markdown(result):
    lines = ["# Frozen score and assignment audit", "", "Scope: " + result["scope"] + ". " + result["interpretation"], "",
             "Native rules and box-only counterfactuals use identical frozen decoded predictions within each checkpoint.", "",
             "| Model | Branch | Overlap | Known corners | Targets | Unassigned | Mean p | Mean q | Mean p−q |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in result["groups"]:
        number = lambda k: "—" if g[k]["mean"] is None else f"{g[k]['mean']:.5f}"
        lines.append(f"| {g['role']} | {g['branch']} | {g['mode']} | {g['known_corners']} | {g['targets']} | {g['without_assignment']} | {number('p')} | {number('q')} | {number('p_minus_q')} |")
    lines.extend(["", "Means above are over surviving anchors. The detailed file preserves per-target eligibility, selection, ownership and quality components. p−q is computed on the same assigned anchor; it is not calibration against observed correctness.", "",
                  "Changing overlap changes assignment and targets together in this descriptive replay. It does not demonstrate what would happen after retraining. No confidence threshold or training objective is selected from these values.", ""])
    return "\n".join(lines)


def verify_result(work, smoke=False):
    folder = work / ("smoke" if smoke else "results")
    manifest, result = read(folder / "manifest.json"), read(folder / "summary.json")
    need(set(manifest) == set(RESULT_FILES), "Output manifest membership differs")
    need({p.name for p in folder.iterdir()} == set(RESULT_FILES) | {"manifest.json"}, "Unexpected or missing output files")
    for name, item in manifest.items():
        need(checked(folder / name, item["sha256"]).stat().st_size == item["bytes"], "Output size differs")
    rows = json.loads(gzip.decompress((folder / "targets.json.gz").read_bytes()))
    need(summarize(rows) == result["groups"] and result["plan_sha256"] == sha(work / "plan.json"), "Summary replay differs")
    need(result["photographs_per_model"] == (3 if smoke else 142) and result["forward_calls"] == (6 if smoke else 284), "Output scope differs")
    need(result["models_sha256"] == MODELS and result["protocol"] == PROTOCOL, "Model/protocol differs")
    need(result["completed"] and result["full142_completed"] == (not smoke)
         and result["runtime"]["native_files_sha256"] == NATIVE_FILES, "Completion/native identity differs")
    images = read(folder / "images.json")
    plan = read(work / "plan.json")
    identities = {r["id"] for r in plan["entries"] if not smoke or r["id"] in SMOKE_IDS}
    need(len(images) == len(identities) * 4 and
         {(r["role"], r["branch"], r["image_id"]) for r in images} ==
         {(role, branch, identity) for role in MODELS for branch in ("one2many", "one2one") for identity in identities},
         "Per-image/head membership differs")
    need(len(rows) == result["source_fronts_per_model"] * 2 * 2 * 2, "Target record count differs")
    need(len({(r["role"], r["branch"], r["mode"], r["image_id"], r["source_row"]) for r in rows}) == len(rows),
         "Duplicate target record identities")
    for value in result["invariants"].values():
        need(value["parameter_and_buffer_sha256_before"] == value["parameter_and_buffer_sha256_after"]
             and value["all_modules_eval"] and value["all_parameter_gradients_none"], "Frozen-state guard differs")
    return result


def package(work):
    verify_result(work)
    path = work / (PREFIX + "-results.zip")
    members = {"plan.json": work / "plan.json", "inputs/pr157-return-manifest.json": work / "inputs/pr157-return-manifest.json",
               "source/audit_score_assignment_r65.py": Path(__file__)}
    members.update({"results/" + name: work / "results" / name for name in (*RESULT_FILES, "manifest.json")})
    if (work / "smoke/summary.json").exists():
        verify_result(work, True)
        members["smoke/summary.json"] = work / "smoke/summary.json"
    r61.shared.write_zip(path, {name: p.read_bytes() for name, p in members.items()})
    print(json.dumps({"archive": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source-check", "native-check", "prepare", "audit", "verify", "package"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--download-inputs", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root, work = args.root.resolve(), args.work.resolve()
    need(not work.is_relative_to(root), "Audit work must be outside the source checkout")
    source = source_check(root, args.source_revision, args.script_sha256)
    if args.command == "source-check":
        print(json.dumps({"commit": source["commit"], "driver_sha256": source["driver_sha256"],
                          "verified_source_files": len(source["files_sha256"])})); return
    if args.command == "native-check":
        print(json.dumps(native_contract(), indent=2)); return
    need(args.archive is not None, "Provide the immutable PR157 evidence archive")
    if args.command == "prepare":
        need(args.inputs_dir is not None and args.models_dir is not None, "Provide input and model cache folders")
        prepare(root, work, args.archive, args.inputs_dir, args.models_dir, source, args.download_inputs); return
    verify_stage(root, work, args.archive, source)
    if args.command == "audit":
        folder = work / ("smoke" if args.smoke else "results")
        if folder.exists():
            saved = verify_result(work, args.smoke)
            print("Reused verified completed audit; no new forward calls. Saved runtime:",
                  json.dumps(saved["runtime"], sort_keys=True), flush=True)
        else:
            audit(work, args.device, args.smoke)
    elif args.command == "verify":
        verify_result(work, args.smoke)
        print("Saved assignment audit replay verified.", flush=True)
    else:
        package(work)


if __name__ == "__main__":
    main()
