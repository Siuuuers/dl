#!/usr/bin/env python3
"""Run one fixed PR121 candidate probe using its retained native one-to-one head."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from evaluate_pose import (INFERENCE, _prediction_records, _runtime_and_model,
                           file_sha256, preflight_dataset)
from predict_pose_folder import verify_trained_checkpoint
from verify_pose_candidate import require

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT = "f501bb80f7e77f6a35cf72630ca5ff2a3b2169690ee5d9852919d1cbd2661ba4"
OUTPUT = "results/evaluation/pr121-r36-20261126/predictions/candidate-one2one.jsonl"
AUDIT = "results/evaluation/pr121-r36-20261126/head-probe-model-audit.json"


def branch_state(head, prefix):
    payload = hashlib.sha256()
    tensors, parameters = 0, 0
    for name, value in sorted(head.state_dict().items()):
        if name.startswith(prefix):
            tensor = value.detach().float().contiguous().cpu()
            payload.update(name.encode())
            payload.update(str(tuple(tensor.shape)).encode())
            payload.update(tensor.numpy().tobytes())
            tensors += 1
            parameters += tensor.numel()
    require(tensors > 0 and parameters > 0, "Expected trained head tensors are absent")
    return {"tensors": tensors, "elements": parameters, "float32_state_sha256": payload.hexdigest()}


def run(checkpoint_manifest):
    import numpy as np
    import torch
    output, audit_path = ROOT / OUTPUT, ROOT / AUDIT
    require(not output.exists() and not audit_path.exists(), "Preserve the existing single head probe")
    data = preflight_dataset(ROOT, require_weights=True)
    checkpoint = verify_trained_checkpoint(ROOT, checkpoint_manifest)
    require(checkpoint["model_sha256"] == EXPECTED_CHECKPOINT, "This probe is only for the PR121 candidate")
    model, runtime = _runtime_and_model(checkpoint, "cpu", 4)
    head = model.model.model[-1]
    modules = ["one2one_cv2", "one2one_cv3", "one2one_cv4", "one2one_cv4_kpts", "one2one_cv4_sigma"]
    require(type(head).__name__ == "Pose26" and head.nc == 1 and list(head.kpt_shape) == [2, 3]
            and all(getattr(head, name, None) is not None for name in modules),
            "The returned checkpoint lacks its expected native one-to-one pose branch")
    before = {"head_class": type(head).__name__, "serialized_end2end": bool(head.end2end),
              "head_modules": list(head._modules), "one2one_state": branch_state(head, "one2one_")}
    initial, _ = _runtime_and_model(data, "cpu", 4)
    initial_state = branch_state(initial.model.model[-1], "one2one_")
    require(initial_state["float32_state_sha256"] != before["one2one_state"]["float32_state_sha256"],
            "The returned one-to-one branch is unchanged from initialization")
    del initial
    inference = {**INFERENCE, "nms": False}
    first = data["records"][0]
    model.predict(source=np.zeros((first["height"], first["width"], 3), dtype=np.uint8),
                  device="cpu", conf=0.13, **inference)
    backend = model.predictor.model
    torch.set_num_threads(4)
    require(bool(backend.end2end) and not backend.fp16 and str(backend.device) == "cpu",
            "The probe did not activate the native NMS-free FP32 CPU head")
    require(model.predictor.args.nms is False and model.predictor.args.max_det == 300,
            "Head selector or fixed output cap differs")
    runtime.update(device_actual=str(backend.device), backend_end2end=bool(backend.end2end),
                   backend_fp16=bool(backend.fp16), torch_threads=torch.get_num_threads())
    audit = {"schema_version": 1, "status": "running", "kind": "same_checkpoint_one2one_head_probe",
             "checkpoint_sha256": checkpoint["model_sha256"],
             "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest),
             "initializer_sha256": data["model_sha256"], "initializer_one2one_state": initial_state,
             "prefusion_model": before, "retained_one2one_state_changed_from_initializer": True,
             "runtime": runtime, "inference": inference, "box_confidence": 0.13,
             "endpoint_confidence": 0.50, "nms_iou_applies": False,
             "nms_note": "nms=False selects the trained one-to-one branch. The retained iou=0.70 argument does not run ordinary NMS on this output.",
             "image_ids": [r["id"] for r in data["records"]],
             "warmup": "One zero image, excluded from measured predictions",
             "source_helper": {"path": "scripts/probe_pr121_one2one.py", "sha256": file_sha256(__file__)},
             "timing_claim": "No latency comparison; concurrent CPU work may affect timings.",
             "training_executed": False, "automatic_promotion": False}
    header = {"record_type": "run", "schema_version": 1, "purpose": audit["kind"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_sha256": checkpoint["model_sha256"], "checkpoint_identity": checkpoint["checkpoint_identity"],
              "manifest_sha256": data["manifest_sha256"], "model_card_sha256": data["model_card_sha256"],
              "runtime": runtime, "inference": inference, "box_confidences": [0.13],
              "endpoint_confidence": 0.50, "nms_iou_applies": False, "nms_note": audit["nms_note"],
              "image_ids": audit["image_ids"], "head_audit_path": AUDIT,
              "source_helper": audit["source_helper"], "timing_claim": audit["timing_claim"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    count = 0
    with output.open("x") as stream:
        stream.write(json.dumps(header, allow_nan=False) + "\n")
        for record in data["records"]:
            started = time.perf_counter()
            result = model.predict(source=record["image_path"], device="cpu", conf=0.13, **inference)[0]
            duration = (time.perf_counter() - started) * 1000
            require(tuple(result.orig_shape) == (record["height"], record["width"]), "Image geometry differs")
            predictions = _prediction_records(result)
            item = {"record_type": "prediction", "image_id": record["id"], "threshold": 0.13,
                    "image_sha256": record["image_sha256"], "label_sha256": record["label_sha256"],
                    "width": record["width"], "height": record["height"], "predictions": predictions,
                    "stage_ms": {key: float(value) for key, value in result.speed.items()},
                    "predict_call_wall_ms": duration}
            stream.write(json.dumps(item, allow_nan=False) + "\n")
            stream.flush()
            count += len(predictions)
            print(record["id"], len(predictions), flush=True)
    require(file_sha256(checkpoint["weights"]) == EXPECTED_CHECKPOINT, "Checkpoint changed during the probe")
    audit.update(status="complete", predictions={"path": OUTPUT, "sha256": file_sha256(output)},
                 processed_images=len(data["records"]), prediction_count=count,
                 finished_at_utc=datetime.now(timezone.utc).isoformat())
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return {"status": "complete", "predictions": OUTPUT, "audit": AUDIT, "prediction_count": count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint_manifest), indent=2))
