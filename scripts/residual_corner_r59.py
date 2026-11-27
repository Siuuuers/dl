"""One-step corner correction with an exactly identity initial output."""
from copy import deepcopy
import math

import corner_refinement_r58 as crop

crop_geometry = crop.crop_geometry
point_available = crop.point_available
model_identity = crop.model_identity


def input_tensor(image, prediction, geometry, identity=''):
    return crop.crop_tensor(image, prediction, geometry, 'initial', epoch=0, identity=identity)


def correction_targets(reference, prediction, geometry):
    import numpy as np
    targets = np.zeros((2, 2), dtype='float32')
    available = np.zeros(2, dtype='float32')
    states = []
    for slot, (gold, initial) in enumerate(zip(reference, prediction['keypoints'])):
        if gold[2] != 2:
            states.append('reference_unavailable')
        elif not point_available(initial):
            states.append('initial_point_unavailable')
        else:
            available[slot] = 1
            targets[slot] = [(gold[axis]-initial[axis])*geometry['scale_xy'][axis] for axis in range(2)]
            states.append('supervised')
    return targets, available, states


def apply_corrections(prediction, corrections, geometry):
    """Apply canvas offsets directly in image coordinates; preserve unavailable inputs."""
    if len(corrections) != 2 or any(len(delta) != 2 for delta in corrections):
        raise ValueError('Exactly two two-dimensional corrections are required')
    result = deepcopy(prediction)
    for slot, (initial, delta) in enumerate(zip(prediction['keypoints'], corrections)):
        if not point_available(initial):
            continue
        finite = all(value is not None and math.isfinite(value) for value in delta)
        if finite and all(value == 0 for value in delta):
            continue
        point = [initial[axis]+float(delta[axis])/geometry['scale_xy'][axis] for axis in range(2)] if finite else None
        if point is None or not (0 <= point[0] < 1280 and 0 <= point[1] < 720):
            result['keypoints'][slot] = [None, None, initial[2]]
        else:
            result['keypoints'][slot] = [*point, initial[2]]
    return result


def build_model(pretrained_path=None):
    import torch
    from torch import nn
    from torchvision.models import resnet50

    class ResidualCornerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = resnet50(weights=None)
            if pretrained_path is not None:
                self.body.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True), strict=True)
            rgb = self.body.conv1.weight.detach().clone()
            self.body.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                self.body.conv1.weight.zero_()
                self.body.conv1.weight[:, :3].copy_(rgb)
            self.body.fc = nn.Linear(2048, 4)
            nn.init.zeros_(self.body.fc.weight)
            nn.init.zeros_(self.body.fc.bias)

        def forward(self, image):
            return self.body(image).reshape(-1, 2, 2)

    return ResidualCornerModel()


def masked_correction_loss(prediction, target, availability):
    from torch.nn import functional as F
    denominator = availability.sum()*2
    if denominator.item() <= 0:
        raise ValueError('The batch contains no usable correction target')
    loss = F.smooth_l1_loss(prediction, target, reduction='none', beta=1.)
    return (loss*availability[:, :, None]).sum()/denominator
