#!/usr/bin/env python3
"""Training-only pixel corruption with unchanged known source-view pose targets."""

import copy
import hashlib
import json
import math
from pathlib import Path
import random


DEFAULT_SPEC = {
    "id": "known-source-corner-corruption-v1",
    "seed": 46013,
    "probability": 0.5,
    "max_masks_per_image": 1,
    "side_fraction_range": [0.15, 0.35],
    "outside_fraction": 0.05,
    "vertical_pad_fraction": 0.15,
    "minimum_pad_px": 3,
    "fill_gray_range": [64, 192],
    "point_radius_px": 3,
}
COUNTERS = (
    "batches_seen", "images_seen", "images_planned", "images_applied",
    "empty_images", "ineligible_images", "probability_skipped", "geometry_rejected",
    "source_missing_slots_seen", "known_endpoints_covered_planned",
    "collateral_known_endpoints_planned", "known_endpoints_covered_applied",
    "collateral_known_endpoints_applied",
)
MAX_EXAMPLES = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def spec_sha256(spec):
    require(canonical(spec) == canonical(DEFAULT_SPEC), "unsupported corruption specification")
    return hashlib.sha256(canonical(spec).encode()).hexdigest()


def validate_state(state, spec, role):
    require(isinstance(state, dict), "missing or invalid corruption state")
    require(role in ("control", "candidate"), "invalid corruption role")
    require(state.get("schema_version") == 1 and state.get("role") == role,
            "corruption state identity differs")
    require(canonical(state.get("spec")) == canonical(spec) and
            state.get("spec_sha256") == spec_sha256(spec), "corruption specification differs")
    counts = state.get("counters", {})
    require(set(counts) == set(COUNTERS) and all(type(v) is int and v >= 0 for v in counts.values()),
            "invalid corruption counters")
    require(sum(counts[k] for k in ("images_planned", "empty_images", "ineligible_images",
                                    "probability_skipped", "geometry_rejected")) == counts["images_seen"],
            "corruption image counts do not partition the input")
    require(counts["batches_seen"] <= counts["images_seen"], "invalid corruption batch count")
    require(counts["known_endpoints_covered_planned"] ==
            counts["images_planned"] + counts["collateral_known_endpoints_planned"],
            "planned endpoint coverage differs")
    for applied, planned in (("images_applied", "images_planned"),
                             ("known_endpoints_covered_applied", "known_endpoints_covered_planned"),
                             ("collateral_known_endpoints_applied", "collateral_known_endpoints_planned")):
        require(counts[applied] == (counts[planned] if role == "candidate" else 0),
                "applied corruption counts differ from role")
    digest, examples = state.get("decision_sha256"), state.get("examples")
    require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "invalid corruption decision digest")
    require(isinstance(examples, list) and len(examples) == min(MAX_EXAMPLES, counts["images_planned"]),
            "corruption examples differ from planned count")
    return state


def inside(point, rectangle):
    return rectangle[0] <= point[0] < rectangle[2] and rectangle[1] <= point[1] < rectangle[3]


def neighborhood_intersects(point, rectangle, radius):
    return (point[0] + radius >= rectangle[0] and point[0] - radius < rectangle[2]
            and point[1] + radius >= rectangle[1] and point[1] - radius < rectangle[3])


class PoseInputCorruption:
    """Plan identical masks in both roles; change only candidate image tensors."""

    def __init__(self, spec, role):
        require(role in ("control", "candidate"), "invalid corruption role")
        self.spec_digest = spec_sha256(spec)
        self.spec, self.role = copy.deepcopy(spec), role
        self.rng = random.Random(spec["seed"])
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.decisions, self.examples = hashlib.sha256(), []

    def state(self):
        return validate_state({"schema_version": 1, "role": self.role,
                               "spec": copy.deepcopy(self.spec), "spec_sha256": self.spec_digest,
                               "counters": dict(self.counts), "decision_sha256": self.decisions.hexdigest(),
                               "examples": copy.deepcopy(self.examples)}, self.spec, self.role)

    def _record(self, decision):
        self.decisions.update((canonical(decision) + "\n").encode())
        if decision["status"] == "planned" and len(self.examples) < MAX_EXAMPLES:
            self.examples.append(copy.deepcopy(decision))

    def __call__(self, batch, *, epoch=0):
        import torch

        images, boxes, points = batch["img"], batch["bboxes"], batch["keypoints"]
        require(isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[1] == 3
                and images.shape[0] > 0 and torch.is_floating_point(images), "expected normalized image batch")
        require(type(epoch) is int and epoch >= 0, "invalid corruption epoch")
        require(boxes.ndim == 2 and boxes.shape[1] == 4 and points.shape == (len(boxes), 2, 3),
                "expected two-point pose targets")
        owners = batch["batch_idx"].detach().cpu().reshape(-1).tolist()
        require(len(owners) == len(boxes) == len(batch["cls"]) and
                all(float(i).is_integer() and 0 <= i < len(images) for i in owners), "invalid pose batch ownership")
        names = batch["im_file"]
        require(len(names) == len(images), "image identities differ from batch")
        box_rows, point_rows = boxes.detach().cpu().tolist(), points.detach().cpu().tolist()
        require(all(p[2] in (0, 2) for pair in point_rows for p in pair), "unexpected source visibility")
        self.counts["source_missing_slots_seen"] += sum(p[2] == 0 for pair in point_rows for p in pair)
        height, width = images.shape[-2:]
        self.counts["batches_seen"] += 1
        radius = self.spec["point_radius_px"]
        for image_index in range(len(images)):
            self.counts["images_seen"] += 1
            decision = {"epoch": epoch, "batch": self.counts["batches_seen"],
                        "image_index": image_index, "image_id": Path(names[image_index]).stem,
                        "size_wh": [width, height]}
            objects = []
            for index, owner in enumerate(owners):
                if owner != image_index:
                    continue
                cx, cy, bw, bh = box_rows[index]
                box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                       (cx + bw / 2) * width, (cy + bh / 2) * height]
                pair = [[p[0] * width, p[1] * height, p[2]] for p in point_rows[index]]
                objects.append((box, pair))
            eligible = [i for i, (box, pair) in enumerate(objects)
                        if all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]
                        and all(p[2] == 2 and math.isfinite(p[0]) and math.isfinite(p[1])
                                and 0 <= p[0] < width and 0 <= p[1] < height for p in pair)
                        and pair[0][0] < pair[1][0]]
            if not objects:
                reason = "empty_images"
            elif not eligible:
                reason = "ineligible_images"
            elif self.rng.random() >= self.spec["probability"]:
                reason = "probability_skipped"
            else:
                reason = None
            if reason:
                self.counts[reason] += 1
                self._record({**decision, "status": reason})
                continue
            target = self.rng.choice(eligible)
            side = self.rng.randrange(2)
            inward = self.rng.uniform(*self.spec["side_fraction_range"])
            gray = self.rng.randint(*self.spec["fill_gray_range"])
            box, pair = objects[target]
            x0, y0, x1, y1 = box
            box_width, box_height = x1 - x0, y1 - y0
            outward = self.spec["outside_fraction"] * box_width
            lo, hi = ((x0 - outward, x0 + inward * box_width) if side == 0
                      else (x1 - inward * box_width, x1 + outward))
            padding = max(self.spec["minimum_pad_px"], self.spec["vertical_pad_fraction"] * box_height)
            rectangle = [max(0, math.floor(lo)), max(0, math.floor(y0 - padding)),
                         min(width, math.ceil(hi)), min(height, math.ceil(y1 + padding))]
            point = pair[side]
            margin = min(point[0] - rectangle[0], point[1] - rectangle[1],
                         rectangle[2] - point[0], rectangle[3] - point[1])
            if (rectangle[0] >= rectangle[2] or rectangle[1] >= rectangle[3] or margin < radius
                    or neighborhood_intersects(pair[1 - side], rectangle, radius)):
                self.counts["geometry_rejected"] += 1
                self._record({**decision, "status": "geometry_rejected", "target_object_index": target,
                              "side": side, "mask_rect_xyxy": rectangle})
                continue
            covered = [{"object_index": i, "endpoint_index": j, "known_before_corruption": True,
                        "would_be_unobserved_if_applied": True}
                       for i, (_, corners) in enumerate(objects) for j, corner in enumerate(corners)
                       if corner[2] == 2 and inside(corner, rectangle)]
            collateral = sum(p["object_index"] != target for p in covered)
            require(len(covered) == 1 + collateral, "intended endpoint coverage differs")
            self.counts["images_planned"] += 1
            self.counts["known_endpoints_covered_planned"] += len(covered)
            self.counts["collateral_known_endpoints_planned"] += collateral
            decision.update(status="planned", target_object_index=target, side=side,
                            target_box_xyxy=box, target_keypoints_xyv=pair, mask_rect_xyxy=rectangle,
                            inward_fraction=inward, fill_gray=gray, covered_known_endpoints=covered)
            self._record(decision)
            if self.role == "candidate":
                images[image_index, :, rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = gray / 255
                self.counts["images_applied"] += 1
                self.counts["known_endpoints_covered_applied"] += len(covered)
                self.counts["collateral_known_endpoints_applied"] += collateral
        return batch
