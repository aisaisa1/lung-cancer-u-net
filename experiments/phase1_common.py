"""Shared IO for the read-only target-data audits; no student-model imports."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
CLASSES = ('Normal', 'Benign', 'Malignant')


def class_name(parts):
    for part in reversed(parts):
        name = part.lower().strip().replace('_', ' ')
        for prefix, canonical in (('normal', 'Normal'), ('benign', 'Benign'),
                                  ('bengin', 'Benign'), ('malignant', 'Malignant')):
            if name == prefix or name == prefix + ' cases':
                return canonical
    return 'Unknown'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        # No EXIF rotation, resize, crop, or source write: retain stored coordinates.
        if image.mode not in ('L', 'RGB', 'RGBA', 'P'):
            raise ValueError(f'Unsupported native mode {image.mode}; no silent 8-bit conversion')
        return np.asarray(image.convert('RGB')).copy()


def prepare_output(path, dataset_root, sentinel):
    path = Path(path).resolve()
    dataset_root = Path(dataset_root).resolve()
    if path == dataset_root or dataset_root in path.parents or path in dataset_root.parents:
        raise ValueError('Output must be separate from the input dataset tree')
    if (path / sentinel).exists():
        raise FileExistsError(f'Existing report: {path / sentinel}; choose a new --output-dir')
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)
