"""Phase 1: frozen LungMask teacher feasibility and representation stability only."""

import argparse
from collections import Counter
import csv
import hashlib
import inspect
from itertools import combinations
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from phase1_common import (CLASSES, PROJECT_ROOT, prepare_output, read_rgb,
                           sha256_file, write_csv, write_json)
from teacher_environment_audit import inspect_environment

TEACHERS = ('R231', 'R231CovidWeb')
REPRESENTATIONS = ('native', 'linear', 'gamma_lower', 'gamma_higher', 'clahe_mild', 'contrast_mild')
QC_FIELDS = ('foreground_area_ratio', 'connected_components', 'largest_component_fraction',
             'border_touch_fraction', 'centroid_x', 'centroid_y', 'fragmentation', 'empty_mask')
ROW_FIELDS = ['image_id', 'relative_path', 'class_name', 'width', 'height', 'teacher',
              'representation', 'status', 'error', 'elapsed_seconds', 'internal_intensity_views',
              *QC_FIELDS, 'cross_teacher_dice', 'cross_teacher_iou', 'cross_teacher_both_empty',
              'mask_path', 'label_map_path', 'overlay_path', 'roi_path', 'human_review_status']


def representations(rgb):
    """Intensity changes only: no resizing, rotations, differing crops, or anatomy changes."""
    x = rgb.astype(np.float32) / 255.0
    low, high = float(x.min()), float(x.max())
    linear = (x - low) / (high - low) if high > low else x.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8)).apply(gray)
    # Blend to keep CLAHE mild; all diagnostic views keep the full original canvas.
    mild = 0.75 * x + 0.25 * (clahe.astype(np.float32) / 255.0)[..., None]
    convert = lambda a: np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)
    result = dict(native=rgb.copy(), linear=convert(linear), gamma_lower=convert(x ** 0.9),
                  gamma_higher=convert(x ** 1.1), clahe_mild=convert(mild),
                  contrast_mild=convert((x - 0.5) * 1.05 + 0.5))
    if any(a.shape != rgb.shape for a in result.values()):
        raise ValueError('Representation geometry changed')
    return result


def agreement(a, b):
    if a.shape != b.shape:
        raise ValueError(f'Mask coordinates differ: {a.shape} vs {b.shape}')
    a, b = a.astype(bool), b.astype(bool)
    intersection = int(np.logical_and(a, b).sum())
    total = int(a.sum() + b.sum())
    union = int(np.logical_or(a, b).sum())
    return dict(dice=2 * intersection / total if total else 1.0,
                iou=intersection / union if union else 1.0,
                both_empty=total == 0)


def technical_qc(mask):
    mask = mask.astype(bool)
    labels, number = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    foreground = int(mask.sum())
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    border = np.zeros_like(mask)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    y, x = np.nonzero(mask)
    return dict(foreground_area_ratio=foreground / mask.size, connected_components=int(number),
                largest_component_fraction=largest / foreground if foreground else None,
                border_touch_fraction=int((mask & border).sum()) / foreground if foreground else None,
                centroid_x=float(x.mean()) if foreground else None,
                centroid_y=float(y.mean()) if foreground else None,
                fragmentation=1 - largest / foreground if foreground else None,
                empty_mask=foreground == 0)


def overlay(rgb, mask):
    result = rgb.copy()
    result[mask] = np.rint(0.6 * rgb[mask] + 0.4 * np.array([0, 255, 80])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 80, 0), 1)
    return result


def save_image(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def contact_sheet(path, panels, title):
    tile_width, tile_height = 300, 300
    columns = min(5, len(panels))
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * tile_width, rows * (tile_height + 32) + 36), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='black')
    for i, (label, array) in enumerate(panels):
        x, y = (i % columns) * tile_width, 36 + (i // columns) * (tile_height + 32)
        panel = Image.fromarray(array).convert('RGB')
        # Display thumbnails preserve aspect ratio; actual output masks retain native dimensions.
        panel = ImageOps.contain(panel, (tile_width, tile_height))
        canvas.paste(panel, (x + (tile_width - panel.width) // 2, y + 32))
        draw.text((x + 8, y + 8), label, fill='black')
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def select_sample(audit_dir, per_class, seed):
    with (audit_dir / 'dataset_audit.json').open(encoding='utf-8') as handle:
        audit = json.load(handle)
    if audit['smoke'] or audit['scope'] != 'full':
        raise ValueError('Teacher pilot requires a full dataset audit, not smoke inventory')
    source = audit.get('canonical_teacher_source')
    if source is None:
        raise ValueError('Canonical teacher source unresolved; no alternative dataset fallback')
    with (audit_dir / 'dataset_inventory.csv').open(encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['source'] == source and row['status'] == 'ok']
    selected, exclusions = [], []
    seen = set()
    for cls in CLASSES:
        candidates = sorted((row for row in rows if row['class_name'] == cls),
                            key=lambda row: hashlib.sha256(
                                f"{seed}|{row['relative_path']}".encode()).hexdigest())
        chosen = 0
        for row in candidates:
            digest = row['pixel_sha256']
            if digest in seen:
                exclusions.append(dict(relative_path=row['relative_path'], reason='exact_pixel_duplicate'))
                continue
            seen.add(digest)
            selected.append(row)
            chosen += 1
            if chosen == per_class:
                break
        if chosen < per_class:
            raise ValueError(f'Only {chosen} unique {cls} images; requested {per_class}')
    return audit, selected, exclusions


def load_teacher(name, args):
    import torch
    from lungmask import mask
    if 'noHU' not in inspect.signature(mask.LMInferer.__init__).parameters:
        raise RuntimeError('Installed LungMask lacks noHU; use the isolated 0.2.14 environment')
    torch.hub.set_dir(str(args.cache_dir.resolve()))
    url, _ = mask.MODEL_URLS[name]
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / url.rsplit('/', 1)[-1]
    if not checkpoint.exists() and not args.download_weights:
        raise FileNotFoundError(f'{name} weights absent; pass --download-weights for official teacher weights')
    if not checkpoint.exists():
        # Use upstream URLs with filename hash-prefix verification. No dataset download.
        state = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True,
                                                   progress=True, weights_only=True)
        del state
    digest = sha256_file(checkpoint)
    expected = checkpoint.stem.rsplit('-', 1)[-1]
    if not digest.startswith(expected):
        raise ValueError(f'Cached official weight hash mismatch: {checkpoint.name}')
    inferer = mask.LMInferer(modelname=name, noHU=True, force_cpu=args.cpu, batch_size=args.batch_size,
                            volume_postprocessing=not args.no_postprocessing, tqdm_disable=True)
    inferer.model.eval()
    inferer.model.requires_grad_(False)
    return inferer, dict(url=url, file=str(checkpoint), sha256=digest,
                         file_bytes=checkpoint.stat().st_size, device=str(inferer.device))


def predict(inferer, rgb):
    import torch
    from skimage import color, transform
    # Inspect the exact upstream noHU sanity filter to report its technical failures explicitly.
    gray = transform.resize(color.rgb2gray(rgb), [256, 256])
    scales = np.linspace(0.3, 2, 20)
    retained = sum(int((np.minimum(gray * scale, 1) > 0.6).sum()) > 25000 for scale in scales)
    if retained == 0:
        raise ValueError('LungMask noHU sanity filter rejects all 20 internal intensity views')
    with torch.inference_mode():
        labels = np.asarray(inferer.apply(rgb))
    if labels.shape != (1, *rgb.shape[:2]):
        raise ValueError(f'Unexpected label shape {labels.shape}; native coordinates required')
    labels = labels[0]
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError('Unexpected LungMask output semantics (expected background/right/left)')
    return labels.astype(np.uint8), retained


def distribution(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def run(args, output):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    audit, selected, exclusions = select_sample(args.audit_dir, args.samples_per_class, args.seed)
    root = Path(audit['dataset_root']).resolve()
    # Validate all files against the audit before doing expensive inference.
    for row in selected:
        path = (root / row['relative_path']).resolve()
        if root not in path.parents or sha256_file(path) != row['file_sha256']:
            raise ValueError(f"Source changed since audit or escaped dataset: {row['relative_path']}")
    for name in ('masks', 'overlays', 'roi', 'contact_sheets', 'disagreement_maps',
                 'originals', 'representations', 'label_maps'):
        (output / name).mkdir(parents=True, exist_ok=True)
    cached_rows = {}
    run_config = dict(seed=args.seed, samples_per_class=args.samples_per_class,
                      smoke=args.smoke, no_postprocessing=args.no_postprocessing,
                      batch_size=args.batch_size, teachers=list(TEACHERS),
                      representations=list(REPRESENTATIONS))
    if args.resume:
        with (output / 'selected_images.csv').open(encoding='utf-8', newline='') as handle:
            previous = list(csv.DictReader(handle))
        if previous != selected:
            raise ValueError('Resume manifest differs from the current audited sample')
        if (output / 'run_config.json').exists():
            with (output / 'run_config.json').open(encoding='utf-8') as handle:
                if json.load(handle) != run_config:
                    raise ValueError('Resume inference configuration differs')
        with (output / 'teacher_environment.json').open(encoding='utf-8') as handle:
            previous_environment = json.load(handle)
        current_environment = inspect_environment()
        for package in ('torch', 'lungmask', 'numpy', 'scipy', 'skimage'):
            if previous_environment['packages'][package]['version'] != current_environment['packages'][package]['version']:
                raise ValueError(f'Resume package version differs: {package}')
        with (output / 'teacher_feasibility.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                key = (row['image_id'], row['teacher'], row['representation'])
                if key in cached_rows:
                    raise ValueError(f'Duplicate cached row: {key}')
                cached_rows[key] = row
    write_json(output / 'run_config.json', run_config)
    write_csv(output / 'selected_images.csv', selected, list(selected[0]))
    environment = inspect_environment()
    write_json(output / 'teacher_environment.json', environment)
    teachers, teacher_setup = {}, {}
    for name in TEACHERS:
        try:
            inferer, provenance = load_teacher(name, args)
            teachers[name] = inferer
            teacher_setup[name] = dict(status='ready', **provenance)
        except Exception as error:
            teachers[name] = None
            teacher_setup[name] = dict(status='failed', error=f'{type(error).__name__}: {error}',
                                        traceback=traceback.format_exc())
            print(f'{name}: setup failed: {error}', flush=True)

    rows, pairs, stability, review = [], [], [], []
    for index, selected_row in enumerate(selected, start=1):
        relative = selected_row['relative_path']
        cls = selected_row['class_name']
        image_id = cls.lower() + '_' + hashlib.sha256(relative.encode()).hexdigest()[:12]
        rgb = read_rgb(root / relative)
        views = representations(rgb)
        source_hash = selected_row['file_sha256']
        save_image(output / 'originals' / f'{image_id}.png', rgb)
        for rep, view in views.items():
            save_image(output / 'representations' / f'{image_id}_{rep}.png', view)
        masks, native_rows = {}, {}
        for name in TEACHERS:
            masks[name] = {}
            for rep, view in views.items():
                row = {field: '' for field in ROW_FIELDS}
                row.update(image_id=image_id, relative_path=relative, class_name=cls,
                           width=rgb.shape[1], height=rgb.shape[0], teacher=name, representation=rep,
                           human_review_status='pending', status='failed')
                t0 = time.time()
                try:
                    if teachers[name] is None:
                        raise RuntimeError(teacher_setup[name]['error'])
                    cached = cached_rows.get((image_id, name, rep))
                    if cached and cached['status'] == 'failed' and 'sanity filter rejects' in cached['error']:
                        raise ValueError('LungMask noHU sanity filter rejects all 20 internal intensity views')
                    if cached and cached['status'] == 'ok':
                        for field in ('mask_path', 'label_map_path', 'overlay_path', 'roi_path'):
                            artifact = (output / cached[field]).resolve()
                            if output not in artifact.parents or not artifact.is_file():
                                raise ValueError(f'Invalid resume artifact: {field}')
                        with Image.open(output / cached['label_map_path']) as image:
                            labels = np.asarray(image).copy()
                        with Image.open(output / cached['mask_path']) as image:
                            cached_mask = np.asarray(image) > 0
                        if labels.shape != view.shape[:2] or not np.isin(labels, [0, 1, 2]).all():
                            raise ValueError('Cached labels do not match native geometry/semantics')
                        if not np.array_equal(labels > 0, cached_mask):
                            raise ValueError('Cached binary mask and labels disagree')
                        retained = int(cached['internal_intensity_views'])
                    else:
                        labels, retained = predict(teachers[name], view)
                    binary = labels > 0
                    row.update(status='ok', internal_intensity_views=retained, **technical_qc(binary))
                    stem = f'{image_id}_{rep}.png'
                    artifacts = dict(mask_path=Path('masks') / name / stem,
                                     label_map_path=Path('label_maps') / name / stem,
                                     overlay_path=Path('overlays') / name / stem,
                                     roi_path=Path('roi') / name / stem)
                    save_image(output / artifacts['mask_path'], binary.astype(np.uint8) * 255)
                    save_image(output / artifacts['label_map_path'], labels)
                    save_image(output / artifacts['overlay_path'], overlay(view, binary))
                    # Full-frame masked ROI, not a differently cropped representation.
                    save_image(output / artifacts['roi_path'], rgb * binary[..., None])
                    row.update({field: path.as_posix() for field, path in artifacts.items()})
                    masks[name][rep] = binary
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
                cached = cached_rows.get((image_id, name, rep))
                row['elapsed_seconds'] = (float(cached['elapsed_seconds']) if cached
                                          else round(time.time() - t0, 4))
                rows.append(row)
                if rep == 'native':
                    native_rows[name] = row
            successful = masks[name]
            image_pairs = []
            for a, b in combinations(successful, 2):
                metrics = agreement(successful[a], successful[b])
                pair = dict(image_id=image_id, class_name=cls, teacher=name,
                            representation_a=a, representation_b=b, **metrics)
                pairs.append(pair)
                image_pairs.append(pair)
            complete = len(successful) == len(REPRESENTATIONS)
            stats = dict(image_id=image_id, class_name=cls, teacher=name,
                         successful_representations=len(successful), expected_representations=6,
                         complete=complete, successful_pairs=len(image_pairs), expected_pairs=15,
                         mean_dice=None, minimum_dice=None, mean_iou=None, minimum_iou=None,
                         both_empty_pairs=sum(p['both_empty'] for p in image_pairs),
                         nonempty_mean_dice=None, nonempty_mean_iou=None, disagreement_map_path='')
            if image_pairs:
                stats.update(mean_dice=float(np.mean([p['dice'] for p in image_pairs])),
                             minimum_dice=min(p['dice'] for p in image_pairs),
                             mean_iou=float(np.mean([p['iou'] for p in image_pairs])),
                             minimum_iou=min(p['iou'] for p in image_pairs))
                valid = [p for p in image_pairs if not p['both_empty']]
                if valid:
                    stats.update(nonempty_mean_dice=float(np.mean([p['dice'] for p in valid])),
                                 nonempty_mean_iou=float(np.mean([p['iou'] for p in valid])))
                n = len(successful)
                votes = np.stack(list(successful.values())).sum(axis=0).astype(np.float32)
                disagreement = 2 * votes * (n - votes) / (n * (n - 1))
                map_path = Path('disagreement_maps') / name / f'{image_id}.npy'
                (output / map_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(output / map_path, disagreement)
                heat = cv2.cvtColor(cv2.applyColorMap(np.rint(disagreement * 255).astype(np.uint8),
                                                      cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
                save_image((output / map_path).with_suffix('.png'), heat)
                stats['disagreement_map_path'] = map_path.as_posix()
            stability.append(stats)
            if successful:
                panels = [(rep, overlay(views[rep], successful[rep])) for rep in successful]
                contact_sheet(output / 'contact_sheets' / f'{image_id}_{name}_representations.png',
                              panels, f'{cls} | {name} | representation stability (not accuracy)')

        if all('native' in masks[name] for name in TEACHERS):
            metrics = agreement(masks[TEACHERS[0]]['native'], masks[TEACHERS[1]]['native'])
            for row in native_rows.values():
                row.update(cross_teacher_dice=metrics['dice'], cross_teacher_iou=metrics['iou'],
                           cross_teacher_both_empty=metrics['both_empty'])
        panels = [('Original (native)', rgb)]
        for name in TEACHERS:
            if 'native' in masks[name]:
                binary = masks[name]['native']
                panels.extend([(name + ' Mask', binary.astype(np.uint8) * 255),
                               (name + ' Overlay', overlay(rgb, binary))])
        if len(panels) == 1:
            failed = np.full_like(rgb, 235)
            panels.append(('Both native teachers failed', failed))
        sheet = Path('contact_sheets') / f'{image_id}_teachers.png'
        contact_sheet(output / sheet, panels, f'{cls} | {Path(relative).name} | pending human review')
        for name in TEACHERS:
            review.append(dict(image_id=image_id, relative_path=relative, class_name=cls, teacher=name,
                               native_status=native_rows[name]['status'], contact_sheet=sheet.as_posix(),
                               decision='pending', reviewer='', notes=''))
        unchanged = sha256_file(root / relative) == source_hash
        if not unchanged:
            raise RuntimeError(f'Source image changed during read-only pilot: {relative}')
        # Persist progress after each image; failures never disappear from denominators.
        write_csv(output / 'teacher_feasibility.csv', rows, ROW_FIELDS)
        print(f'[{index}/{len(selected)}] {cls}: {Path(relative).name}; '
              + ', '.join(f'{name}={native_rows[name]["status"]}' for name in TEACHERS), flush=True)

    write_csv(output / 'representation_pairwise.csv', pairs, ['image_id', 'class_name', 'teacher',
              'representation_a', 'representation_b', 'dice', 'iou', 'both_empty'])
    write_csv(output / 'representation_stability.csv', stability, list(stability[0]))
    write_csv(output / 'human_visual_review.csv', review, list(review[0]))
    summaries = {}
    for name in TEACHERS:
        native = [r for r in rows if r['teacher'] == name and r['representation'] == 'native']
        ok = [r for r in native if r['status'] == 'ok']
        complete = [s for s in stability if s['teacher'] == name and s['complete']]
        summaries[name] = dict(setup=teacher_setup[name], native_attempted=len(native),
                               native_successes=len(ok), native_failures=len(native) - len(ok),
                               native_empty_masks=sum(r['empty_mask'] for r in ok),
                               all_representation_failures=sum(r['status'] != 'ok' for r in rows
                                                               if r['teacher'] == name),
                               native_successes_by_class=dict(Counter(r['class_name'] for r in ok)),
                               native_area_ratio=distribution([r['foreground_area_ratio'] for r in ok]),
                               representation_complete_images=len(complete),
                               representation_mean_dice=distribution([s['mean_dice'] for s in complete]),
                               representation_minimum_dice=distribution([s['minimum_dice'] for s in complete]),
                               representation_mean_iou=distribution([s['mean_iou'] for s in complete]),
                               representation_minimum_iou=distribution([s['minimum_iou'] for s in complete]),
                               both_empty_representation_pairs=sum(s['both_empty_pairs'] for s in complete))
    cross = [r for r in rows if r['teacher'] == 'R231' and r['representation'] == 'native'
             and r['cross_teacher_dice'] != '']
    summary = dict(phase=1, smoke=args.smoke, seed=args.seed, selected_images=len(selected),
                   selected_class_counts=dict(Counter(r['class_name'] for r in selected)),
                   canonical_source=audit['canonical_teacher_source'], dataset_root=str(root),
                   selection_method='class-stratified SHA256(seed|relative_path) rank; exact pixel deduplication',
                   sampling_is_not_a_training_split=True, exclusions=exclusions,
                   grouping_safely_inferred=audit['grouping']['safely_inferred'],
                   environment=environment, teachers=summaries,
                   cross_teacher_agreement=dict(n=len(cross), dice=distribution([r['cross_teacher_dice'] for r in cross]),
                                                iou=distribution([r['cross_teacher_iou'] for r in cross]),
                                                both_empty_pairs=sum(r['cross_teacher_both_empty'] for r in cross)),
                   representations=dict(native='stored RGB export, no intensity modification',
                                        linear='global min-max scaling to 0..255, constant images unchanged',
                                        gamma_lower='x**0.9, x=image/255', gamma_higher='x**1.1',
                                        clahe_mild='25% blend of grayscale CLAHE clipLimit=1.2 grid=8x8',
                                        contrast_mild='1.05*(x-0.5)+0.5, clipped to [0,1]'),
                   coordinate_policy='Full stored native canvas for every view and output. Upstream noHU '
                                     'resizes internally to 256x256 and remaps labels with nearest-neighbor.',
                   qc_definitions=dict(connectivity='8-connected, 2D',
                                       border_touch_fraction='foreground on outermost image border / all foreground',
                                       fragmentation='1 - largest_component_foreground / all foreground',
                                       centroid='x,y in native pixel coordinates; null for empty masks'),
                   disagreement_definition='Fraction of unordered successful-view pairs disagreeing per pixel: '
                                           '2*k*(n-k)/(n*(n-1)); maps are not calibrated uncertainty.',
                   empty_mask_convention='Both-empty agreement is 1 by set convention, explicitly flagged. '
                                         'Empty masks are technical failures for anatomical use, not evidence of reliability.',
                   postprocessing='disabled' if args.no_postprocessing else 'upstream legacy volume_postprocessing=True',
                   limitations=['Teacher masks are pseudo-label candidates, never ground truth.',
                                'Dice/IoU here measure agreement/stability, never segmentation accuracy.',
                                'noHU performs an internal 20-intensity sweep, rejects dark views, then selects '
                                'the view with maximal right-lung label-1 area. This can confound representation stability.',
                                'Legacy postprocessing operates across synthetic intensity views, not a true CT volume.',
                                'Export orientation and HU calibration are unavailable; left/right semantics require review.',
                                'QC is descriptive, not an anatomical acceptance rule.',
                                'Only complete six-view images enter aggregate stability; partial/failing cases remain in CSV.',
                                'No calibrated uncertainty estimation or final reliability loss implemented.'],
                   human_visual_review='pending for every candidate', recommendation='FIX: human visual review required',
                   student_training_allowed=False, source_hashes_unchanged=True,
                   resumed=args.resume, resumed_cached_rows=len(cached_rows),
                   elapsed_seconds=round(time.time() - started, 2))
    write_json(output / 'teacher_summary.json', summary)
    print('Outputs:', output, flush=True)
    for name, item in summaries.items():
        print(name, 'native successes:', item['native_successes'], '/', item['native_attempted'],
              'complete stability:', item['representation_complete_images'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/audit')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--samples-per-class', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true', help='One image per class; both teachers, all six representations')
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted pilot after validating cached results')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4, help='Internal noHU sweep batch size')
    parser.add_argument('--no-postprocessing', action='store_true')
    parser.add_argument('--download-weights', action='store_true', help='Download official teacher checkpoints if absent')
    parser.add_argument('--cache-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/.cache/torch')
    args = parser.parse_args()
    if min(args.samples_per_class, args.threads, args.batch_size) < 1:
        parser.error('samples-per-class, threads and batch-size must be positive')
    if args.smoke:
        args.samples_per_class = 1
    output = args.output_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/teacher_pilot'
    if args.smoke and args.output_dir is None:
        output = output / 'smoke'
    with (args.audit_dir / 'dataset_audit.json').open(encoding='utf-8') as handle:
        root = Path(json.load(handle)['dataset_root'])
    cache = args.cache_dir.resolve()
    if cache == root.resolve() or root.resolve() in cache.parents:
        parser.error('Weight cache must not be inside the target dataset')
    output = prepare_output(output, root, 'teacher_summary.json')
    if (output / 'teacher_feasibility.csv').exists() and not args.resume:
        parser.error('Incomplete pilot already exists; use --resume or a new --output-dir')
    run(args, output)


if __name__ == '__main__':
    main()
