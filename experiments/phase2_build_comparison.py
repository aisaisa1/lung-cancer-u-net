"""Compare cached teacher behavior in native coordinates, without GT/accuracy or new inference."""

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from phase1_common import PROJECT_ROOT, read_rgb, sha256_file, write_csv, write_json
sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
from draw_lung_boxes import read_csv
from validate_phase2_annotations import validate
pilot = importlib.import_module('01_teacher_feasibility')

COMPARISON_FIELDS = ('sample', 'class', 'teacher', 'prompt_type', 'technical_success', 'nonempty',
                    'reviewer_status', 'error_type', 'notes', 'source_path', 'width', 'height',
                    'foreground_area_ratio', 'connected_components', 'fragmentation', 'border_touch_fraction',
                    'runtime_seconds', 'technical_error', 'mask_path', 'overlay_path')


def read_mask(path, shape):
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.shape != shape or not np.isin(array, [0, 255]).all():
        raise ValueError(f'Invalid binary mask/native geometry: {path}')
    return array > 0


def manual_panel(rgb, records):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for row in records:
        if row['visibility_status'] == 'VISIBLE':
            x0, y0, x1, y1 = [row[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')]
            color = '#ffbf00' if row['box_id'] == 'box_1' else '#00bfff'
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
            draw.text((x0 + 3, y0 + 3), row['box_id'], fill=color)
    return np.asarray(image)


def sheet(path, panels, title, columns=6):
    # Native-sized panels: no display thumbnail interpolation is saved into these sheets.
    widths = [array.shape[1] for _, array in panels]
    heights = [array.shape[0] for _, array in panels]
    width, height = max(widths), max(heights)
    columns = min(columns, len(panels))
    rows = (len(panels) + columns - 1) // columns
    image = Image.new('RGB', (width * columns, 32 + rows * (height + 28)), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((6, 8), title, fill='black')
    for i, (label, array) in enumerate(panels):
        x, y = (i % columns) * width, 32 + (i // columns) * (height + 28)
        draw.text((x + 5, y + 5), label, fill='black')
        tile = Image.fromarray(array).convert('RGB')
        image.paste(tile, (x + (width - tile.width) // 2, y + 28))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_prompts')
    parser.add_argument('--runs-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_teacher_comparison')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output_dir or (args.runs_dir / 'smoke_comparison' if args.smoke else args.runs_dir)
    output = output.resolve()
    if (PROJECT_ROOT / 'DATASET').resolve() in output.parents:
        raise ValueError('Comparison output must not overlap source dataset')
    selection, annotations, validation = validate(args.prompts_dir, output / 'validation')
    if not validation['ready_for_inference']:
        raise ValueError('PENDING prompts remain; comparison is blocked')
    if args.smoke:
        selection = [next(s for s in selection if s['class'] == cls) for cls in ('Normal', 'Benign', 'Malignant')]
    phase1_root = PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/teacher_pilot'
    legacy = {(r['image_id'], r['teacher']): r for r in read_csv(phase1_root / 'teacher_feasibility.csv')
              if r['representation'] == 'native'}
    alternatives, geometries, summaries = {}, {}, {}
    for teacher in ('SAM-Med2D', 'MedSAM'):
        directory = args.runs_dir / teacher
        alternatives[teacher] = {(r['sample'], r['region']): r for r in read_csv(directory / 'predictions.csv')}
        with (directory / 'teacher_summary.json').open(encoding='utf-8') as handle:
            summaries[teacher] = json.load(handle)
        for key in ('annotation_csv_sha256', 'annotation_json_sha256'):
            if summaries[teacher][key] != validation['csv_sha256' if 'csv' in key else 'json_sha256']:
                raise ValueError(f'{teacher} predictions use a different human annotation snapshot')
        with (directory / 'geometry.json').open(encoding='utf-8') as handle:
            geometries[teacher] = {(r['sample'], r['region']): r for r in json.load(handle)}
    comparison, checked_geometry, checked_union = [], 0, 0
    for sample in selection:
        sid, shape = sample['sample_id'], (sample['image_height'], sample['image_width'])
        rgb = read_rgb(PROJECT_ROOT / sample['source_path'])
        records = [r for r in annotations if r['sample_id'] == sid]
        boxes_panel = manual_panel(rgb, records)
        for directory, array in (('originals', rgb), ('manual_boxes', boxes_panel)):
            pilot.save_image(output / directory / f'{sid}.png', array)
        panels = [('Original (native)', rgb), ('Exact manual boxes', boxes_panel)]
        legacy_panels = []
        for teacher in ('SAM-Med2D', 'MedSAM', 'R231', 'R231CovidWeb'):
            row = {field: '' for field in COMPARISON_FIELDS}
            row.update(sample=sid, **{'class': sample['class']}, teacher=teacher, source_path=sample['source_path'],
                       width=shape[1], height=shape[0], reviewer_status='PENDING',
                       prompt_type='none_legacy_noHU_cached' if teacher.startswith('R231') else 'manual_box_1_box_2_union')
            binary = None
            if teacher.startswith('R231'):
                result = legacy[sid, teacher]
                success = result['status'] == 'ok'
                row.update(technical_success=success, technical_error=result['error'],
                           notes='Cached Phase-1 native output; LungMask was not rerun.',
                           runtime_seconds=result['elapsed_seconds'])
                if success:
                    binary = read_mask(phase1_root / result['mask_path'], shape)
            else:
                result = alternatives[teacher][sid, 'merged']
                success = result['technical_success'] == 'True'
                row.update(technical_success=success, technical_error=result['error'],
                           runtime_seconds=result['runtime_seconds'])
                individual = []
                debug = [('Original (native)', rgb), ('Exact manual boxes', boxes_panel)]
                for record in records:
                    region = record['box_id']
                    box_result = alternatives[teacher][sid, region]
                    if record['visibility_status'] != 'VISIBLE':
                        if box_result['mask_path']:
                            raise ValueError('NOT_VISIBLE region must not have a fabricated prediction')
                        continue
                    if box_result['technical_success'] == 'True':
                        mask = read_mask(args.runs_dir / teacher / box_result['mask_path'], shape)
                        individual.append(mask)
                        native = np.array([record[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')], dtype=float)
                        if json.loads(box_result['native_box']) != native.tolist():
                            raise ValueError('Teacher did not use the exact human box')
                        actual = np.array(json.loads(box_result['model_box']))
                        if teacher == 'MedSAM':
                            expected = native * np.array([1024 / shape[1], 1024 / shape[0]] * 2)
                        else:
                            size = summaries[teacher]['model']['image_size']
                            resized_h = int(shape[0] * size / max(shape) + 0.5)
                            resized_w = int(shape[1] * size / max(shape) + 0.5)
                            expected = native * np.array([resized_w / shape[1], resized_h / shape[0]] * 2)
                        if not np.allclose(actual, expected, rtol=0, atol=1e-7):
                            raise ValueError('Incorrect prompt transform into model coordinates')
                        geometry = geometries[teacher][sid, region]
                        if not np.allclose(geometry['transformed_box'], expected, atol=1e-7):
                            raise ValueError('Geometry provenance disagrees with saved prediction rows')
                        checked_geometry += 1
                        debug.append((region + ' independent mask', mask.astype(np.uint8) * 255))
                if result['mask_path']:
                    binary = read_mask(args.runs_dir / teacher / result['mask_path'], shape)
                    if not individual or not np.array_equal(binary, np.logical_or.reduce(individual)):
                        raise ValueError('Merged mask is not the exact native union of individual masks')
                    checked_union += 1
                sheet(output / 'box_debug_contact_sheets' / teacher / f'{sid}.png', debug,
                      f'{teacher} | {sid} | independent region predictions, no GT', columns=4)
            if binary is not None:
                qc = pilot.technical_qc(binary)
                row.update(nonempty=bool(binary.any()), **{key: qc[key] for key in
                    ('foreground_area_ratio', 'connected_components', 'fragmentation', 'border_touch_fraction')})
                relative_mask = Path('comparison_masks') / teacher / f'{sid}.png'
                relative_overlay = Path('comparison_overlays') / teacher / f'{sid}.png'
                pilot.save_image(output / relative_mask, binary.astype(np.uint8) * 255)
                rendered = pilot.overlay(rgb, binary)
                pilot.save_image(output / relative_overlay, rendered)
                row.update(mask_path=(output / relative_mask).relative_to(PROJECT_ROOT).as_posix(),
                           overlay_path=(output / relative_overlay).relative_to(PROJECT_ROOT).as_posix())
                pair_panels = [(teacher + ' Mask', binary.astype(np.uint8) * 255), (teacher + ' Overlay', rendered)]
            else:
                pair_panels = [(teacher + ' unavailable (technical)', np.full_like(rgb, 230)),
                               (teacher + ' no prediction', np.full_like(rgb, 230))]
            if teacher.startswith('R231'):
                legacy_panels.extend(pair_panels)
            else:
                panels.extend(pair_panels)
            comparison.append(row)
        sheet(output / 'contact_sheets' / f'{sid}_comparison.png', panels + legacy_panels,
              f"{sample['class']} | {Path(sample['source_path']).name} | human mask review PENDING | no accuracy")
        if sha256_file(PROJECT_ROOT / sample['source_path']) != sample['file_sha256']:
            raise RuntimeError('Source modified during comparison generation')
    csv_path = output / 'phase2_teacher_comparison.csv' if args.smoke else output.parent / 'phase2_teacher_comparison.csv'
    if csv_path.exists():
        # Retain explicit human grades/notes when regenerating derived artifacts.
        reviewed = {(r['sample'], r['teacher']): r for r in read_csv(csv_path)}
        for row in comparison:
            old = reviewed.get((row['sample'], row['teacher']))
            if old:
                for key in ('reviewer_status', 'error_type', 'notes'):
                    row[key] = old[key]
    write_csv(csv_path, comparison, COMPARISON_FIELDS)
    summary = dict(selected_images=len(selection), annotation_validation=validation,
                   per_teacher={}, verification=dict(individual_geometry_checks=checked_geometry,
                                                     native_union_checks=checked_union,
                                                     sources_unchanged=True, annotations_unchanged=True),
                   mask_review_status_counts=dict(Counter(r['reviewer_status'] for r in comparison)),
                   accuracy_computed=False, limitations=['QC is technical, not anatomical validation.',
                       'Fragmentation = 1-largest-component fraction; two normal lungs can naturally yield a high value.',
                       'No predicted mask was clipped to its prompt box, sharpened, or morphologically filtered.',
                       'Box IDs are arbitrary; anatomical left/right orientation is unverified.'])
    for teacher in ('SAM-Med2D', 'MedSAM', 'R231', 'R231CovidWeb'):
        rows = [r for r in comparison if r['teacher'] == teacher]
        ok = [r for r in rows if r['technical_success']]
        summary['per_teacher'][teacher] = dict(attempted=len(rows), technical_successes=len(ok),
            technical_failures=len(rows) - len(ok), nonempty_outputs=sum(bool(r['nonempty']) for r in ok),
            mean_area_ratio=float(np.mean([r['foreground_area_ratio'] for r in ok])) if ok else None,
            mean_runtime_seconds=float(np.mean([float(r['runtime_seconds']) for r in ok])) if ok else None,
            recommendation='REJECT' if teacher.startswith('R231') else 'CONDITIONAL',
            reason='Phase-1 default noHU route rejected for ungated supervision; cached comparison only.'
                   if teacher.startswith('R231') else 'Inference runs, but expert anatomical mask review is still PENDING.')
    write_json(output / 'comparison_summary.json', summary)
    print(json.dumps(summary['per_teacher'], indent=2))
    print('Verified geometry:', checked_geometry, '; exact native unions:', checked_union)
    print('Comparison CSV:', csv_path)
    print('Contact sheets:', output / 'contact_sheets')


if __name__ == '__main__':
    main()
