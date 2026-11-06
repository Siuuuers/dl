#!/usr/bin/env python3
"""Check draft export and preservation without model dependencies or inference."""
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch


def tensor(values):
    value = Mock()
    value.cpu.return_value.numpy.return_value = values
    return value


def main():
    yolo = Mock()
    spec = importlib.util.spec_from_file_location('export_pose_drafts', Path(__file__).with_name('export_pose_drafts.py'))
    exporter = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=yolo)}):
        spec.loader.exec_module(exporter)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        exporter.WEIGHTS = root / 'weights.pt'
        exporter.WEIGHTS.touch()
        exporter.SRC_IMAGES = root / 'images'
        exporter.OUT = root / 'data_pred'
        images = exporter.SRC_IMAGES / 'val'
        images.mkdir(parents=True)
        for name in ('cassette', 'empty'):
            (images / f'{name}.png').write_bytes(b'fixture image')

        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xywhn = tensor([[0.5, 0.5, 0.4, 0.2]])
        keypoints = SimpleNamespace(xyn=tensor([[[0.3, 0.6], [0.7, 0.6]]]), conf=tensor([[0.9, 0.8]]))
        yolo.return_value.predict.return_value = [
            SimpleNamespace(path=str(images / 'cassette.png'), boxes=boxes, keypoints=keypoints),
            SimpleNamespace(path=str(images / 'empty.png'), boxes=None, keypoints=None),
        ]
        exporter.main()
        labels = exporter.OUT / 'labels' / 'val'
        label = labels / 'cassette.txt'
        assert label.read_text() == '0 0.50000000 0.50000000 0.40000000 0.20000000 0.30000000 0.60000000 2 0.70000000 0.60000000 2\n'
        assert (labels / 'empty.txt').read_text() == ''
        for name in ('cassette', 'empty'):
            assert (exporter.OUT / 'images' / 'val' / f'{name}.png').read_bytes() == (images / f'{name}.png').read_bytes()

        corrected = b'manually corrected label\n'
        label.write_bytes(corrected)
        before = {path.relative_to(exporter.OUT): path.read_bytes() for path in exporter.OUT.rglob('*') if path.is_file()}
        try:
            exporter.main()
        except FileExistsError as error:
            assert 'preserve corrected labels' in str(error)
        else:
            raise AssertionError('Existing drafts must not be overwritten')
        after = {path.relative_to(exporter.OUT): path.read_bytes() for path in exporter.OUT.rglob('*') if path.is_file()}
        assert after == before
        yolo.assert_called_once_with(str(exporter.WEIGHTS))
        yolo.return_value.predict.assert_called_once()

    print('PASS: new drafts, empty labels, and existing corrections preserved before model load')


if __name__ == '__main__':
    main()
