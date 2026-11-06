#!/usr/bin/env python3
"""Export YOLO-pose draft .txt labels from cassette_lower_epoch173 (CPU ok)."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'weights/cassette_lower_epoch173.pt'
SRC_IMAGES = ROOT / 'data/images'
OUT = ROOT / 'data_pred'  # labeler-compatible: images/ + labels/
SPLITS = ('val',)  # start with val (20); add 'train' later if wanted
CONF = 0.25
IMGSZ = 1280


def row_from_boxes_kpts(boxes_xywhn, kpts_xyn, kpts_conf):
    """One cassette → class cx cy w h kx1 ky1 v1 kx2 ky2 v2."""
    cx, cy, w, h = [float(x) for x in boxes_xywhn]
    # clamp box
    cx = min(1.0, max(0.0, cx)); cy = min(1.0, max(0.0, cy))
    w = min(1.0, max(1e-6, w)); h = min(1.0, max(1e-6, h))
    parts = [0, cx, cy, w, h]
    for i in range(2):
        x, y = float(kpts_xyn[i][0]), float(kpts_xyn[i][1])
        c = float(kpts_conf[i]) if kpts_conf is not None else 1.0
        if c < 0.01 or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            parts.extend([0.0, 0.0, 0])
        else:
            parts.extend([min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), 2])
    return parts


def fmt(row):
    out = []
    for k, v in enumerate(row):
        if k in (0, 7, 10):
            out.append(str(int(v)))
        else:
            out.append(f'{float(v):.8f}')
    return ' '.join(out)


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} already exists. Move it before exporting again to preserve corrected labels.')
    assert WEIGHTS.exists(), WEIGHTS
    model = YOLO(str(WEIGHTS))
    for split in SPLITS:
        (OUT / 'images' / split).mkdir(parents=True)
        (OUT / 'labels' / split).mkdir(parents=True)
        # symlink images so labeler can open data_pred as data folder
        src = SRC_IMAGES / split
        for img in sorted(src.glob('*.png')):
            link = OUT / 'images' / split / img.name
            if not link.exists():
                link.symlink_to(img.resolve())

        results = model.predict(
            source=str(src),
            imgsz=IMGSZ,
            conf=CONF,
            device='cpu',
            save=False,
            verbose=False,
        )
        n_imgs = 0
        n_obj = 0
        for r in results:
            stem = Path(r.path).stem
            lines = []
            if r.boxes is not None and len(r.boxes):
                xywhn = r.boxes.xywhn.cpu().numpy()
                if r.keypoints is None:
                    continue
                xyn = r.keypoints.xyn.cpu().numpy()
                conf = None
                if r.keypoints.conf is not None:
                    conf = r.keypoints.conf.cpu().numpy()
                for i in range(len(xywhn)):
                    kc = conf[i] if conf is not None else None
                    lines.append(fmt(row_from_boxes_kpts(xywhn[i], xyn[i], kc)))
                    n_obj += 1
            (OUT / 'labels' / split / f'{stem}.txt').write_text(('\n'.join(lines) + ('\n' if lines else '')))
            n_imgs += 1
        print(f'{split}: wrote drafts for {n_imgs} images, {n_obj} cassettes → {OUT}/labels/{split}/')


if __name__ == '__main__':
    main()
