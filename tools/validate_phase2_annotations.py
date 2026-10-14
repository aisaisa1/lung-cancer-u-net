"""Read-only validation of completed human prompts and lossless native previews."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from draw_lung_boxes import (PROJECT_ROOT, BOX_IDS, COORDINATES, FIELDS,
                              csv_text, digest_file, read_csv, validate_record, validate_selection,
                              atomic_text)


def validate(prompts, output, project_root=PROJECT_ROOT):
    prompts, output, project_root = Path(prompts), Path(output), Path(project_root)
    dataset = (project_root / 'DATASET').resolve()
    if output.resolve() == dataset or dataset in output.resolve().parents:
        raise ValueError('Validation reports must not be written into the dataset')
    selection = read_csv(prompts / 'phase2_selection.csv')
    validate_selection(selection, project_root)
    with (prompts / 'manual_lung_boxes.json').open(encoding='utf-8') as handle:
        saved = json.load(handle)
    expected_digest = hashlib.sha256(csv_text(selection, tuple(selection[0])).encode()).hexdigest()
    if saved.get('schema_version') != 1 or saved['selection_sha256'] != expected_digest:
        raise ValueError('Annotation JSON does not match the frozen selection')
    records = saved['records']
    if (prompts / 'manual_lung_boxes.csv').read_text(encoding='utf-8') != csv_text(records, FIELDS):
        raise ValueError('Annotation CSV and JSON differ; no repair/overwrite performed')
    expected = {(s['sample_id'], b) for s in selection for b in BOX_IDS}
    if len(records) != 30 or {(r['sample_id'], r['box_id']) for r in records} != expected:
        raise ValueError('Missing or duplicate annotation slots')
    samples = {s['sample_id']: s for s in selection}
    for row in records:
        validate_record(row, samples[row['sample_id']])
    rows, blur_rows = [], []
    for sample in selection:
        source = np.asarray(Image.open(project_root / sample['source_path']).convert('RGB'))
        excluded = Image.new('1', (sample['image_width'], sample['image_height']), 0)
        drawing = ImageDraw.Draw(excluded)
        drawing.rectangle((0, 0, min(sample['image_width'], 420), 19), fill=1)
        for record in records:
            if record['sample_id'] != sample['sample_id']:
                continue
            if record['visibility_status'] == 'VISIBLE':
                x0, y0, x1, y1 = [record[k] for k in COORDINATES]
                drawing.rectangle((max(0, x0 - 3), max(0, y0 - 3), min(sample['image_width'] - 1, x1 + 3),
                                   min(sample['image_height'] - 1, y1 + 3)), outline=1, width=8)
                # Exclude only the tiny text-label zone, not the region interior.
                drawing.rectangle((x0, y0, min(sample['image_width'] - 1, x0 + 60),
                                   min(sample['image_height'] - 1, y0 + 20)), fill=1)
            rows.append(dict(sample_id=sample['sample_id'], source_path=sample['source_path'],
                             box_id=record['box_id'], visibility_status=record['visibility_status'],
                             coordinates_valid=True, reviewer_id=record['reviewer_id']))
        preview_path = prompts / 'previews' / f"{sample['sample_id']}.png"
        preview = np.asarray(Image.open(preview_path).convert('RGB')) if preview_path.exists() else None
        unmarked = np.asarray(excluded) == 0
        same_size = preview is not None and preview.shape == source.shape
        equal = bool(np.array_equal(preview[unmarked], source[unmarked])) if same_size else False
        count = int(np.any(preview[unmarked] != source[unmarked], axis=1).sum()) if same_size else None
        blur_rows.append(dict(sample_id=sample['sample_id'], native_width=source.shape[1], native_height=source.shape[0],
                              preview_exists=preview is not None, preview_native_size=same_size,
                              unmarked_pixels_checked=int(unmarked.sum()), unmarked_pixels_equal=equal,
                              unmarked_pixels_changed=count))
    missing = [r for r in rows if r['visibility_status'] == 'PENDING']
    invisible = [r for r in rows if r['visibility_status'] == 'NOT_VISIBLE']
    report = dict(valid=True, ready_for_inference=not missing, selected_images=len(selection), records=len(records),
                  class_counts=dict(Counter(s['class'] for s in selection)),
                  native_dimensions=dict(Counter(f"{s['image_width']}x{s['image_height']}" for s in selection)),
                  visibility_counts=dict(Counter(r['visibility_status'] for r in rows)),
                  pending_regions=missing, not_visible_regions=invisible,
                  original_hashes_verified=True, annotations_modified=False,
                  csv_sha256=digest_file(prompts / 'manual_lung_boxes.csv'),
                  json_sha256=digest_file(prompts / 'manual_lung_boxes.json'),
                  preview_checks=blur_rows,
                  preview_diagnosis=('Saved PNGs preserve native resolution and unmarked source pixels exactly. '
                                     'Old GUI fit-mode LANCZOS resizing can interpolate/upscale display pixels; '
                                     'this is display softness, not destructive source blur.'
                                     if all(r['unmarked_pixels_equal'] for r in blur_rows) else
                                     'Preview mismatch detected; inspect per-image records before diagnosing blur.'),
                  geometry='Native half-open pixel edges, arbitrary box IDs; no crop homogenization.')
    atomic_text(output / 'annotation_validation.json', json.dumps(report, indent=2) + '\n')
    atomic_text(output / 'region_validation.csv', csv_text(rows, tuple(rows[0])))
    return selection, records, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_prompts')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_teacher_comparison/validation')
    parser.add_argument('--smoke', action='store_true', help='Full 15-image validation is already inexpensive')
    args = parser.parse_args()
    _, _, report = validate(args.prompts_dir, args.output_dir)
    print(json.dumps(report, indent=2))
    if not report['ready_for_inference']:
        sys.exit('Inference blocked by PENDING regions; no missing prompts are inferred.')


if __name__ == '__main__':
    main()
