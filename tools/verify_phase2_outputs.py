"""Read-only independent verification of saved individual/union masks, QC, and original pixels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from draw_lung_boxes import PROJECT_ROOT, read_csv, digest_file, atomic_text
from validate_phase2_annotations import validate

sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))
from phase2_build_comparison import read_mask, pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_teacher_comparison')
    parser.add_argument('--prompts-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_prompts')
    parser.add_argument('--smoke', action='store_true', help='Verify one image per class')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    selection, _, validation = validate(args.prompts_dir, out / 'validation')
    if args.smoke:
        selection = [next(s for s in selection if s['class'] == c) for c in ('Normal', 'Benign', 'Malignant')]
    union_count, masks_count, roi_count, qc_count = 0, 0, 0, 0
    comparison = read_csv(out.parent / 'phase2_teacher_comparison.csv')
    for sample in selection:
        sid, shape = sample['sample_id'], (sample['image_height'], sample['image_width'])
        rgb = np.asarray(Image.open(PROJECT_ROOT / sample['source_path']).convert('RGB'))
        saved = np.asarray(Image.open(out / 'originals' / f'{sid}.png').convert('RGB'))
        if not np.array_equal(rgb, saved):
            raise ValueError('Saved original differs from source pixel values')
        for teacher in ('SAM-Med2D', 'MedSAM'):
            run = out / teacher
            masks = [read_mask(run / 'individual_masks' / f'{sid}_{box}.png', shape) for box in ('box_1', 'box_2')]
            merged = read_mask(run / 'merged_masks' / f'{sid}.png', shape)
            if not np.array_equal(merged, masks[0] | masks[1]):
                raise ValueError('Saved union differs from independent predictions')
            copied = read_mask(out / 'comparison_masks' / teacher / f'{sid}.png', shape)
            if not np.array_equal(merged, copied):
                raise ValueError('Comparison mask differs from the teacher output')
            roi = np.asarray(Image.open(run / 'roi' / f'{sid}.png').convert('RGB'))
            if not np.array_equal(roi, rgb * merged[..., None]):
                raise ValueError('ROI altered source intensities or coordinate canvas')
            qc = pilot.technical_qc(merged)
            row = next(r for r in comparison if r['sample'] == sid and r['teacher'] == teacher)
            for field in ('foreground_area_ratio', 'connected_components', 'fragmentation', 'border_touch_fraction'):
                if not np.isclose(float(row[field]), qc[field]):
                    raise ValueError(f'Saved QC differs: {field}')
                qc_count += 1
            masks_count += 4
            union_count += 1
            roi_count += 1
        if digest_file(PROJECT_ROOT / sample['source_path']) != sample['file_sha256']:
            raise ValueError('Original source hash changed')
    report = dict(passed=True, smoke=args.smoke, source_images=len(selection), exact_unions=union_count,
                  native_masks_checked=masks_count, full_frame_roi_checks=roi_count, qc_values_checked=qc_count,
                  source_hashes_unchanged=True, annotations_unchanged=True,
                  annotation_csv_sha256=validation['csv_sha256'], annotation_json_sha256=validation['json_sha256'],
                  caveat='This verifies implementation/artifact integrity, not anatomical accuracy.')
    atomic_text(out / ('verification_smoke.json' if args.smoke else 'verification.json'), json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
