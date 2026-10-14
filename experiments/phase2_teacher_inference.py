"""Frozen, manual-box-only teacher inference. One teacher per process avoids SAM import collisions."""

import argparse
from collections import Counter, defaultdict
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

from phase1_common import PROJECT_ROOT, read_rgb, sha256_file, write_csv, write_json

sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
from validate_phase2_annotations import validate

pilot = importlib.import_module('01_teacher_feasibility')
QC_FIELDS = tuple(pilot.QC_FIELDS)
RESULT_FIELDS = ('sample', 'class', 'teacher', 'region', 'source_path', 'width', 'height',
                 'prompt_type', 'technical_success', 'status', 'error', 'runtime_seconds',
                 'embedding_seconds', 'nonempty', *QC_FIELDS, 'native_box', 'model_box',
                 'model_input_size', 'mask_path', 'overlay_path', 'roi_path', 'reviewer_status')
TEACHERS = ('SAM-Med2D', 'MedSAM')


def medsam_box(box, native_shape):
    h, w = native_shape
    return np.asarray(box, dtype=np.float64) * np.array([1024 / w, 1024 / h, 1024 / w, 1024 / h])


def merge_masks(masks, shape):
    if not masks:
        return None
    if any(mask.shape != shape for mask in masks):
        raise ValueError('Cannot merge predictions from different native coordinates')
    return np.logical_or.reduce(masks)


def save_png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


class FrozenTeacher:
    def __init__(self, name, repo, checkpoint, device):
        import torch
        self.name, self.device = name, torch.device(device)
        repo = repo.resolve()
        if 'segment_anything' in sys.modules:
            raise RuntimeError('Run each teacher in a separate process: both official repos use segment_anything')
        sys.path.insert(0, str(repo))
        sam = importlib.import_module('segment_anything')
        if repo not in Path(sam.__file__).resolve().parents:
            raise RuntimeError('Wrong segment_anything implementation imported')
        self.metadata = dict(repo=str(repo), repo_commit=subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            checkpoint=str(checkpoint.resolve()), checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=sha256_file(checkpoint), device=str(self.device))
        if name == 'SAM-Med2D':
            # The official release includes optimizer state: mmap avoids materializing it all in RAM.
            # Official release serializes its Adam optimizer too. Allow only these known
            # torch/stdlib classes, not unrestricted pickle execution or optimizer use.
            with torch.serialization.safe_globals([torch.optim.Adam, defaultdict, dict]):
                checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
            state = checkpoint_data.get('model', checkpoint_data)
            position = state['image_encoder.pos_embed']
            image_size = int(position.shape[1]) * 16
            if position.shape[1] != position.shape[2] or image_size not in (256, 1024):
                raise ValueError('Unexpected official checkpoint positional embedding geometry')
            args = SimpleNamespace(image_size=image_size, sam_checkpoint=None, encoder_adapter=True)
            self.model = sam.sam_model_registry['vit_b'](args)
            self.model.load_state_dict(state, strict=True)
            del checkpoint_data, state
            self.model.to(self.device).eval().requires_grad_(False)
            self.predictor = sam.SamPredictor(self.model)
            self.metadata.update(image_size=image_size, load_policy='strict model state, adapters enabled, mmap weights_only',
                                 preprocessing='Official SamPredictor ResizeLongestSide, RGB mean/std, bottom/right padding',
                                 output_policy='single mask, native logit interpolation, threshold 0, no filtering/cropping')
        else:
            # Use the official model factory's strict pretrained-state loading.
            self.model = sam.sam_model_registry['vit_b'](checkpoint=str(checkpoint))
            self.model.to(self.device).eval().requires_grad_(False)
            self.metadata.update(image_size=1024, load_policy='official strict state loading',
                                 preprocessing='Official MedSAM_Inference.py cubic 1024x1024 resize, uint8 then min-max [0,1]',
                                 output_policy='single mask, native probability interpolation, threshold 0.5, no filtering/cropping')

    def encode(self, rgb):
        import torch
        synchronize(self.device)
        start = time.perf_counter()
        self.native_shape = rgb.shape[:2]
        with torch.inference_mode():
            if self.name == 'SAM-Med2D':
                self.predictor.set_image(rgb, image_format='RGB')
                self.input_size = self.predictor.input_size
            else:
                from skimage.transform import resize
                # Only transient model-input arrays are transformed, never the source file.
                resized = resize(rgb, (1024, 1024), order=3, preserve_range=True,
                                 anti_aliasing=True).astype(np.uint8)
                normalized = (resized - resized.min()) / max(float(resized.max() - resized.min()), 1e-8)
                tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device).permute(2, 0, 1)[None]
                self.embedding = self.model.image_encoder(tensor)
                self.input_size = (1024, 1024)
        synchronize(self.device)
        return time.perf_counter() - start

    def predict_box(self, box):
        import torch
        with torch.inference_mode():
            if self.name == 'SAM-Med2D':
                native_box = np.asarray(box, dtype=np.float64)
                transformed = self.predictor.transform.apply_boxes(native_box[None], self.native_shape)[0]
                # Pass native coordinates to predict; it transforms once. No double scaling.
                masks, _, _ = self.predictor.predict(box=native_box, multimask_output=False)
                binary = masks[0].astype(bool)
            else:
                from torch.nn import functional as F
                transformed = medsam_box(box, self.native_shape)
                prompt = torch.as_tensor(transformed, dtype=torch.float32, device=self.device)[None, None]
                sparse, dense = self.model.prompt_encoder(points=None, boxes=prompt, masks=None)
                logits, _ = self.model.mask_decoder(image_embeddings=self.embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense, multimask_output=False)
                probabilities = F.interpolate(torch.sigmoid(logits), size=self.native_shape,
                                               mode='bilinear', align_corners=False)
                binary = probabilities[0, 0].cpu().numpy() > 0.5
            if binary.shape != self.native_shape:
                raise ValueError(f'Output shape {binary.shape} differs from native {self.native_shape}')
            return binary, transformed.tolist()


def result_row(sample, teacher, region):
    row = {field: '' for field in RESULT_FIELDS}
    row.update(sample=sample['sample_id'], **{'class': sample['class']}, teacher=teacher, region=region,
               source_path=sample['source_path'], width=sample['image_width'], height=sample['image_height'],
               prompt_type='human_manual_boxes_independent_then_union', technical_success=False,
               status='failed', reviewer_status='PENDING')
    return row


def run(args):
    import torch
    out = args.output_dir.resolve()
    if (PROJECT_ROOT / 'DATASET').resolve() in out.parents:
        raise ValueError('Teacher output must not overlap the original dataset tree')
    if (out / 'teacher_summary.json').exists() or (out / 'predictions.csv').exists():
        raise FileExistsError('Teacher run outputs already exist; choose a new --output-dir')
    out.mkdir(parents=True, exist_ok=True)
    selection, annotations, validation = validate(args.prompts_dir, out / 'validation')
    if not validation['ready_for_inference']:
        raise ValueError('PENDING human prompts block inference')
    if args.smoke:
        # One actual manually annotated image per class, not synthetic boxes.
        selection = [next(s for s in selection if s['class'] == cls) for cls in ('Normal', 'Benign', 'Malignant')]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    teacher, setup_error = None, ''
    try:
        teacher = FrozenTeacher(args.teacher, args.repo_dir, args.checkpoint, args.device)
    except Exception as error:
        setup_error = f'{type(error).__name__}: {error}'
        write_json(out / 'setup_error.json', dict(error=setup_error, traceback=traceback.format_exc()))
        print('Teacher setup failed:', setup_error, flush=True)
    setup_seconds = time.perf_counter() - start
    rows, geometry = [], []
    for i, sample in enumerate(selection, 1):
        rgb = read_rgb(PROJECT_ROOT / sample['source_path'])
        sample_masks = []
        visible = [r for r in annotations if r['sample_id'] == sample['sample_id'] and r['visibility_status'] == 'VISIBLE']
        embedding_seconds, image_error = 0.0, setup_error
        if teacher is not None and visible:
            try:
                embedding_seconds = teacher.encode(rgb)
            except Exception as error:
                image_error = f'{type(error).__name__}: {error}'
        image_rows = []
        for annotation in [r for r in annotations if r['sample_id'] == sample['sample_id']]:
            box_id = annotation['box_id']
            row = result_row(sample, args.teacher, box_id)
            row['embedding_seconds'] = embedding_seconds
            if annotation['visibility_status'] == 'NOT_VISIBLE':
                row.update(status='not_visible', error='Human marked this region NOT_VISIBLE; no prediction fabricated')
                rows.append(row)
                image_rows.append(row)
                continue
            coordinates = [annotation[key] for key in ('x_min', 'y_min', 'x_max', 'y_max')]
            row['native_box'] = json.dumps(coordinates)
            synchronize(args.device)
            t0 = time.perf_counter()
            try:
                if image_error:
                    raise RuntimeError(image_error)
                binary, transformed = teacher.predict_box(coordinates)
                synchronize(args.device)
                row.update(runtime_seconds=time.perf_counter() - t0, technical_success=True, status='ok',
                           nonempty=bool(binary.any()), **pilot.technical_qc(binary),
                           model_box=json.dumps(transformed), model_input_size=json.dumps(teacher.input_size))
                relative = Path('individual_masks') / f"{sample['sample_id']}_{box_id}.png"
                save_png(out / relative, binary.astype(np.uint8) * 255)
                row['mask_path'] = relative.as_posix()
                sample_masks.append(binary)
                geometry.append(dict(sample=sample['sample_id'], region=box_id, native_box=coordinates,
                                     transformed_box=transformed, native_shape=list(rgb.shape[:2]),
                                     resized_input_shape=list(teacher.input_size)))
            except Exception as error:
                row.update(error=f'{type(error).__name__}: {error}', runtime_seconds=time.perf_counter() - t0)
            rows.append(row)
            image_rows.append(row)
        merged = result_row(sample, args.teacher, 'merged')
        merged.update(embedding_seconds=embedding_seconds,
                      runtime_seconds=embedding_seconds + sum(float(r['runtime_seconds'] or 0) for r in image_rows))
        binary = merge_masks(sample_masks, rgb.shape[:2])
        if binary is not None:
            complete = len(sample_masks) == len(visible)
            merged.update(status='ok' if complete else 'partial', technical_success=complete,
                          nonempty=bool(binary.any()), **pilot.technical_qc(binary))
            for field, directory, pixels in (
                    ('mask_path', 'merged_masks', binary.astype(np.uint8) * 255),
                    ('overlay_path', 'overlays', pilot.overlay(rgb, binary)),
                    ('roi_path', 'roi', rgb * binary[..., None])):
                path = Path(directory) / f"{sample['sample_id']}.png"
                save_png(out / path, pixels)
                merged[field] = path.as_posix()
            if not complete:
                merged['error'] = 'Partial union: at least one visible manual region failed'
        else:
            merged['error'] = image_error or 'No successful visible box predictions; no union fabricated'
        rows.append(merged)
        if sha256_file(PROJECT_ROOT / sample['source_path']) != sample['file_sha256']:
            raise RuntimeError('Source changed during read-only inference')
        write_csv(out / 'predictions.csv', rows, RESULT_FIELDS)
        print(f"[{i}/{len(selection)}] {args.teacher} {sample['sample_id']}: {merged['status']}", flush=True)
    if sha256_file(args.prompts_dir / 'manual_lung_boxes.json') != validation['json_sha256'] or \
            sha256_file(args.prompts_dir / 'manual_lung_boxes.csv') != validation['csv_sha256']:
        raise RuntimeError('Human annotations changed during inference')
    merged_rows = [r for r in rows if r['region'] == 'merged']
    environment = {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-image', 'pillow')}
    summary = dict(teacher=args.teacher, smoke=args.smoke, selected_images=len(selection),
                   technical_successes=sum(r['technical_success'] for r in merged_rows),
                   failures=sum(not r['technical_success'] for r in merged_rows),
                   empty_successes=sum(r['technical_success'] and not r['nonempty'] for r in merged_rows),
                   setup_error=setup_error, setup_seconds=setup_seconds,
                   model=teacher.metadata if teacher is not None else None, environment=environment,
                   python=platform.python_version(), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                   annotation_csv_sha256=validation['csv_sha256'], annotation_json_sha256=validation['json_sha256'],
                   source_hashes_unchanged=True, no_accuracy_computed=True, no_student_training=True,
                   review_status='PENDING', timing_policy='GPU synchronized; image encoding once + independent decoder calls; '
                   'merged runtime excludes loading/saving/QC, includes shared encoding',
                   union_policy='pixelwise OR in native coordinates; no mask clipping, filtering, closing, or threshold tuning')
    write_json(out / 'geometry.json', geometry)
    write_json(out / 'teacher_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', choices=TEACHERS, required=True)
    parser.add_argument('--prompts-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_prompts')
    parser.add_argument('--repo-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.repo_dir = args.repo_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/.cache' / args.teacher
    filename = 'sam-med2d_b.pth' if args.teacher == 'SAM-Med2D' else 'medsam_vit_b.pth'
    args.checkpoint = args.checkpoint or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/.cache/phase2_weights' / filename
    args.output_dir = args.output_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_teacher_comparison' / (
        args.teacher + ('_smoke' if args.smoke else ''))
    run(args)


if __name__ == '__main__':
    main()
