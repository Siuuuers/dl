"""Geometry, masked supervision and fixed heatmap model for local corner refinement."""
from copy import deepcopy
import hashlib
import math

SIZE, GRID, STRIDE = 256, 64, 4
CROP_FACTOR, SIGMA, HINT_SIGMA, HINT_JITTER = 1.5, 2.0, 8.0, 4.0
WEIGHTS_URL = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'


def crop_geometry(box, width=1280, height=720):
    x0, y0, x1, y1 = box
    if not all(math.isfinite(x) for x in box) or x1 <= x0 or y1 <= y0:
        raise ValueError('A crop requires a finite positive predicted box')
    w, h = x1-x0, y1-y0
    bounds = [max(0, math.floor(x0-.25*w)), max(0, math.floor(y0-.25*h)),
              min(width, math.ceil(x1+.25*w)), min(height, math.ceil(y1+.25*h))]
    cw, ch = bounds[2]-bounds[0], bounds[3]-bounds[1]
    if cw <= 0 or ch <= 0:
        raise ValueError('Predicted crop has no image intersection')
    gain = min(SIZE/cw, SIZE/ch)
    rw, rh = max(1, min(SIZE, round(cw*gain))), max(1, min(SIZE, round(ch*gain)))
    return {'xyxy': bounds, 'resized_wh': [rw, rh], 'scale_xy': [rw/cw, rh/ch],
            'pad_xy': [(SIZE-rw)//2, (SIZE-rh)//2], 'canvas_wh': [SIZE, SIZE]}


def forward_point(point, geometry):
    return [(point[i]-geometry['xyxy'][i])*geometry['scale_xy'][i]+geometry['pad_xy'][i]
            for i in range(2)]


def inverse_point(point, geometry):
    return [(point[i]-geometry['pad_xy'][i])/geometry['scale_xy'][i]+geometry['xyxy'][i]
            for i in range(2)]


def point_available(point, width=1280, height=720, threshold=.5):
    return (len(point) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in point)
            and 0 <= point[0] < width and 0 <= point[1] < height and point[2] >= threshold)


def valid_grid(geometry):
    import numpy as np
    x = (np.arange(GRID)+.5)*STRIDE
    y = (np.arange(GRID)+.5)*STRIDE
    px, py = geometry['pad_xy']; rw, rh = geometry['resized_wh']
    return ((x[None, :] >= px) & (x[None, :] < px+rw)
            & (y[:, None] >= py) & (y[:, None] < py+rh))


def gaussian_grid(x, y, sigma, size):
    import numpy as np
    rows, cols = np.mgrid[:size, :size]
    return np.exp(-((cols-x)**2+(rows-y)**2)/(2*sigma*sigma)).astype('float32')


def encode_targets(points, geometry):
    import numpy as np
    targets = np.zeros((2, GRID, GRID), dtype='float32')
    present = np.zeros(2, dtype='float32')
    states = []
    l, t, r, b = geometry['xyxy']
    mask = valid_grid(geometry)
    for i, point in enumerate(points):
        if point[2] != 2:
            states.append('unavailable'); continue
        if not (l <= point[0] < r and t <= point[1] < b):
            states.append('outside_predicted_crop'); continue
        u, v = forward_point(point, geometry)
        targets[i] = gaussian_grid(u/STRIDE-.5, v/STRIDE-.5, SIGMA, GRID)*mask
        present[i] = 1
        states.append('supervised')
    return targets, present, states


def crop_tensor(image, prediction, geometry, hint_mode, epoch=0, identity='', seed=46):
    import numpy as np
    from PIL import Image
    if hint_mode not in ('none', 'initial'):
        raise ValueError('Unknown hint mode')
    l, t, r, b = geometry['xyxy']; rw, rh = geometry['resized_wh']; px, py = geometry['pad_xy']
    patch = image.crop((l, t, r, b)).resize((rw, rh), Image.Resampling.BILINEAR)
    canvas = np.full((SIZE, SIZE, 3), 0, dtype='uint8')
    canvas[py:py+rh, px:px+rw] = np.asarray(patch.convert('RGB'))
    rgb = canvas.astype('float32')/255
    rgb = (rgb-np.array([.485, .456, .406], dtype='float32'))/np.array([.229, .224, .225], dtype='float32')
    hints = np.zeros((2, SIZE, SIZE), dtype='float32')
    local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{epoch}:{identity}'.encode()).digest()[:8], 'little')
    jitter = np.random.default_rng(local_seed).normal(0, HINT_JITTER if epoch else 0, (2, 2))
    for j, point in enumerate(prediction['keypoints']):
        if point_available(point):
            u, v = forward_point(point, geometry)
            if hint_mode == 'initial':
                hints[j] = gaussian_grid(u-.5+jitter[j, 0], v-.5+jitter[j, 1], HINT_SIGMA, SIZE)
                hints[j, :py] = 0; hints[j, py+rh:] = 0
                hints[j, :, :px] = 0; hints[j, :, px+rw:] = 0
    return np.concatenate((rgb.transpose(2, 0, 1), hints), axis=0).astype('float32')


def decode_heatmaps(heatmaps, geometry, with_trace=False):
    import numpy as np
    values = np.asarray(heatmaps)
    if values.shape != (2, GRID, GRID):
        raise ValueError('Expected two 64 by 64 heatmaps')
    mask = valid_grid(geometry)
    points, traces = [], []
    for h in values:
        if not np.isfinite(h).all():
            points.append(None); traces.append({'status':'nonfinite_heatmap'}); continue
        if not mask.any():
            points.append(None); traces.append({'status':'no_valid_content'}); continue
        masked = np.where(mask, h, -np.inf)
        y, x = np.unravel_index(int(masked.argmax()), masked.shape)
        if masked[y, x] <= 0:
            points.append(None); traces.append({'status':'nonpositive_peak','peak_value':float(masked[y,x])}); continue
        dx = dy = 0.
        neighbours = {'x':None,'y':None}
        if 0 < x < GRID-1 and mask[y, x-1] and mask[y, x+1]:
            dx = .25*float(np.sign(h[y, x+1]-h[y, x-1]))
            neighbours['x'] = [float(h[y,x-1]),float(h[y,x+1])]
        if 0 < y < GRID-1 and mask[y-1, x] and mask[y+1, x]:
            dy = .25*float(np.sign(h[y+1, x]-h[y-1, x]))
            neighbours['y'] = [float(h[y-1,x]),float(h[y+1,x])]
        canvas = [(x+.5+dx)*STRIDE, (y+.5+dy)*STRIDE]
        point = inverse_point(canvas, geometry)
        l, t, r, b = geometry['xyxy']
        trace={'status':'decoded','peak_xy':[int(x),int(y)],'peak_value':float(masked[y,x]),
               'neighbours':neighbours,'quarter_offset_xy':[dx,dy],'canvas_xy':canvas}
        if not (l <= point[0] < r and t <= point[1] < b):
            points.append(None)
            trace['status']='outside_crop'
        else:
            points.append(point)
        traces.append(trace)
    return (points,traces) if with_trace else points


def replay_decoding(trace, geometry):
    """Replay local decoder arithmetic; full heatmap maxima require model inference."""
    import numpy as np
    status=trace['status'];mask=valid_grid(geometry)
    if status=='nonfinite_heatmap':return None
    if status=='no_valid_content':
        if mask.any():raise ValueError('Empty-content decoder state differs')
        return None
    if status=='nonpositive_peak':
        if not math.isfinite(trace['peak_value']) or trace['peak_value']>0:raise ValueError('Peak rejection differs')
        return None
    if status not in ('decoded','outside_crop'):raise ValueError('Unknown decoder state')
    x,y=trace['peak_xy']
    if not (isinstance(x,int) and isinstance(y,int) and 0<=x<GRID and 0<=y<GRID and mask[y,x]):
        raise ValueError('Peak lies outside valid content')
    if not math.isfinite(trace['peak_value']) or trace['peak_value']<=0:raise ValueError('Decoded peak must be positive')
    offsets=[]
    for axis,enabled in (('x',0<x<GRID-1 and mask[y,x-1] and mask[y,x+1]),
                         ('y',0<y<GRID-1 and mask[y-1,x] and mask[y+1,x])):
        neighbour=trace['neighbours'][axis]
        if enabled:
            if neighbour is None or len(neighbour)!=2 or not all(math.isfinite(v) and v<=trace['peak_value'] for v in neighbour):
                raise ValueError('Peak-neighbour trace differs')
            offsets.append(.25*float(np.sign(neighbour[1]-neighbour[0])))
        else:
            if neighbour is not None:raise ValueError('Padded neighbour must not affect decoding')
            offsets.append(0.)
    if offsets!=trace['quarter_offset_xy']:raise ValueError('Quarter-offset replay differs')
    canvas=[(x+.5+offsets[0])*STRIDE,(y+.5+offsets[1])*STRIDE]
    if canvas!=trace['canvas_xy']:raise ValueError('Heatmap-to-canvas replay differs')
    point=inverse_point(canvas,geometry);l,t,r,b=geometry['xyxy']
    inside=l<=point[0]<r and t<=point[1]<b
    if inside != (status=='decoded'):raise ValueError('Decoded crop availability differs')
    return point if inside else None


def replace_points(prediction, decoded):
    result = deepcopy(prediction)
    result['keypoints'] = [[None, None, old[2]] if p is None else [float(p[0]), float(p[1]), old[2]]
                           for p, old in zip(decoded, prediction['keypoints'])]
    return result


def build_model(pretrained_path=None):
    import torch
    from torch import nn
    from torchvision.models import resnet50
    class CornerRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            body = resnet50(weights=None)
            if pretrained_path is not None:
                body.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True), strict=True)
            rgb_conv = body.conv1
            body.conv1 = nn.Conv2d(5, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                body.conv1.weight.zero_()
                body.conv1.weight[:, :3].copy_(rgb_conv.weight)
            self.backbone = nn.Sequential(body.conv1, body.bn1, body.relu, body.maxpool,
                body.layer1, body.layer2, body.layer3, body.layer4)
            stages = []
            channels = 2048
            for _ in range(3):
                stages += [nn.ConvTranspose2d(channels, 256, 4, 2, 1, bias=False),
                           nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
                channels = 256
            self.head = nn.Sequential(*stages, nn.Conv2d(256, 2, 1))
            for module in self.head.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.normal_(module.weight, std=.001)
                    if module.bias is not None: nn.init.zeros_(module.bias)
        def forward(self, image):
            return self.head(self.backbone(image))
    return CornerRefiner()


def model_identity(network):
    """Bind every tensor name, shape, dtype and exact initial value, without pickle."""
    import json
    digest = hashlib.sha256()
    tensors = []
    for name, value in network.state_dict().items():
        record = {'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)}
        digest.update(json.dumps(record, sort_keys=True).encode()+b'\n')
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        tensors.append(record)
    return {'state_sha256': digest.hexdigest(), 'tensors': tensors,
            'trainable_parameters': sum(p.numel() for p in network.parameters() if p.requires_grad)}


def masked_heatmap_loss(prediction, target, availability, valid):
    weights = availability[:, :, None, None]*valid[:, None]
    denominator = weights.sum()
    if denominator.item() <= 0:
        raise ValueError('Training batch contains no supervised valid heatmap cells')
    return ((prediction-target).square()*weights).sum()/denominator
