"""Freeze 15 additional images and prepare human-only prompts; no gold export or inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from draw_lung_boxes import (PROJECT_ROOT, SELECTION_FIELDS, AnnotationStore, atomic_text,
                              csv_text, digest_file, read_csv, sample_id, validate_selection)

CLASSES = ('Normal', 'Benign', 'Malignant')
DEFAULT_OUTPUT = PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase3_additional_prompts'
OLD_PROMPTS = PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_prompts'


def rank(source_path):
    return hashlib.sha256(f'phase3-additional|42|{source_path}'.encode()).hexdigest()


def image_features(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    central = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return dict(central_laplacian_variance=float(cv2.Laplacian(central, cv2.CV_64F).var()),
                central_bright_fraction=float(np.mean(central >= 180)),
                central_intensity_spread=float(np.percentile(central, 95) - np.percentile(central, 5)))


def choose_additional(candidates, old, known_pixel_hashes):
    """Stratify on representation proxies, not human anatomical grades."""
    additional, used_hashes = [], set(known_pixel_hashes)
    old_paths = {s['source_path'] for s in old}
    for cls in CLASSES:
        pool = sorted([r for r in candidates if r['class'] == cls and r['source_path'] not in old_paths
                       and r['pixel_sha256'] not in used_hashes], key=lambda r: rank(r['source_path']))
        selected = {}

        def take(rows, reason):
            if len(selected) >= 5:
                return False
            for row in rows:
                if row['source_path'] in selected or row['pixel_sha256'] in used_hashes:
                    continue
                selected[row['source_path']] = dict(row, selection_reason=reason)
                used_hashes.add(row['pixel_sha256'])
                return True
            return False

        old_sizes = {(int(s['image_width']), int(s['image_height'])) for s in old if s['class'] == cls}
        new_sizes = sorted({(r['image_width'], r['image_height']) for r in pool} - old_sizes)
        # At most two new geometry strata per class, leaving slots for image-quality/hardness probes.
        for width, height in new_sizes[:2]:
            take([r for r in pool if (r['image_width'], r['image_height']) == (width, height)],
                 'native_resolution_not_in_retained_phase2_samples')
        comparable = [r for r in pool if (r['image_width'], r['image_height']) == (512, 512)] or pool
        take(sorted(comparable, key=lambda r: (r['central_laplacian_variance'], rank(r['source_path']))),
             'low_sharpness_proxy_blur_requires_human_confirmation')
        if cls == 'Malignant':
            take(sorted(comparable, key=lambda r: (-r['central_bright_fraction'], rank(r['source_path']))),
                 'malignant_class_high_central_brightness_not_verified_pathology')
        else:
            take(sorted(comparable, key=lambda r: (-r['central_laplacian_variance'], rank(r['source_path']))),
                 'high_sharpness_readability_proxy_not_anatomical_easy_grade')
        take([r for r in pool if r['phase1_native_technical_failure']],
             'documented_phase1_native_technical_failure_no_replacement_mask')
        take(sorted(comparable, key=lambda r: (-r['central_bright_fraction'], rank(r['source_path']))),
             'high_central_brightness_proxy_not_verified_pathology')
        while len(selected) < 5:
            if not take(pool, 'deterministic_remaining_rank'):
                raise ValueError(f'Not enough distinct additional {cls} images')
        additional.extend(sorted(selected.values(), key=lambda r: rank(r['source_path'])))
    return additional


def retained_disagreement(old, runs_dir):
    results = []
    for sample in old:
        arrays = []
        for teacher in ('SAM-Med2D', 'MedSAM'):
            path = runs_dir / teacher / 'merged_masks' / f"{sample['sample_id']}.png"
            with Image.open(path) as image:
                array = np.asarray(image)
            shape = (sample['image_height'], sample['image_width'])
            if array.shape != shape or not np.isin(array, [0, 255]).all():
                raise ValueError(f'Invalid existing native candidate: {path}')
            arrays.append(array > 0)
        disagreement = np.logical_xor(*arrays)
        union = np.logical_or(*arrays)
        results.append(dict(sample_id=sample['sample_id'], source_path=sample['source_path'],
                            disagreement_fraction_native=float(disagreement.mean()),
                            disagreement_fraction_union=float(disagreement.sum() / union.sum()) if union.any() else None,
                            interpretation='sampling/review cue only, not segmentation accuracy or a human grade'))
    return results


def prepare_additional(output, old_dir=OLD_PROMPTS, audit_dir=None, runs_dir=None):
    output, old_dir = Path(output).resolve(), Path(old_dir).resolve()
    audit_dir = Path(audit_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/audit')
    runs_dir = Path(runs_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_teacher_comparison')
    dataset = (PROJECT_ROOT / 'DATASET').resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError('Only output metadata/prompts may be prepared; do not write into DATASET')
    if output == old_dir or old_dir in output.parents or output in old_dir.parents:
        raise ValueError('Additional prompt session must be separate from original Phase-2 annotations')
    old = read_csv(old_dir / 'phase2_selection.csv')
    validate_selection(old, PROJECT_ROOT)
    protected = {str(old_dir / name): digest_file(old_dir / name) for name in
                 ('phase2_selection.csv', 'manual_lung_boxes.csv', 'manual_lung_boxes.json')}
    report_path = output / 'selection_report.json'
    if report_path.exists():
        with report_path.open(encoding='utf-8') as handle:
            report = json.load(handle)
        if protected != report['original_prompt_file_hashes']:
            raise ValueError('Original prompt snapshot changed; refusing to mix it with the frozen new selection')
        new = read_csv(output / 'phase2_selection.csv')
        store = AnnotationStore(output, new)
        total = read_csv(output / 'gold_subset_selection_30.csv')
        if len(total) != 30 or {r['sample_id'] for r in total} != {r['sample_id'] for r in old + new}:
            raise ValueError('Frozen total-30 manifest no longer matches retained/additional selections')
        for row in total:
            if digest_file(PROJECT_ROOT / row['source_path']) != row['file_sha256']:
                raise ValueError('A selected source changed after preparation')
        return store, report
    if any((output / name).exists() for name in ('phase2_selection.csv', 'manual_lung_boxes.json', 'manual_lung_boxes.csv')):
        raise ValueError('Partial/user-authored additional session exists; do not overwrite it')
    with (audit_dir / 'dataset_audit.json').open(encoding='utf-8') as handle:
        audit = json.load(handle)
    inventory = [r for r in read_csv(audit_dir / 'dataset_inventory.csv')
                 if r['source'] == audit['canonical_teacher_source'] and r['status'] == 'ok']
    lookup = {(Path('DATASET') / r['relative_path']).as_posix(): r for r in inventory}
    old_hashes = {lookup[s['source_path']]['pixel_sha256'] for s in old}
    if len(old_hashes) != 15:
        raise ValueError('Retained Phase-2 images contain exact pixel duplicates')
    native = [r for r in read_csv(PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/teacher_pilot/teacher_feasibility.csv')
              if r['representation'] == 'native']
    failures = {(Path('DATASET') / r['relative_path']).as_posix() for r in native if r['status'] != 'ok'}
    candidates = []
    for row in inventory:
        relative = (Path('DATASET') / row['relative_path']).as_posix()
        path = (PROJECT_ROOT / relative).resolve()
        if (dataset / 'The IQ-OTHNCCD lung cancer dataset').resolve() not in path.parents:
            raise ValueError('Inventory source escaped canonical tree')
        if digest_file(path) != row['file_sha256']:
            raise ValueError(f'Original source changed since full audit: {relative}')
        candidates.append(dict(sample_id=sample_id(row['relative_path'], row['class_name']), source_path=relative,
                               **{'class': row['class_name']}, image_width=int(row['width']), image_height=int(row['height']),
                               file_sha256=row['file_sha256'], pixel_sha256=row['pixel_sha256'],
                               phase1_native_technical_failure=relative in failures, **image_features(path)))
    selected = choose_additional(candidates, old, old_hashes)
    new = [{field: row[field] for field in SELECTION_FIELDS} for row in selected]
    validate_selection(new, PROJECT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / 'phase2_selection.csv', csv_text(new, SELECTION_FIELDS))
    atomic_text(output / 'sampling_features.csv', csv_text(candidates, tuple(candidates[0])))
    atomic_text(output / 'additional_selection_details.csv', csv_text(selected, tuple(selected[0])))
    total = []
    for stage, rows in (('RETAINED_PHASE2', old), ('ADDITIONAL_PENDING_HUMAN_BOXES', new)):
        for sample in rows:
            total.append(dict(sample, selection_stage=stage,
                              candidate_status='AVAILABLE_PHASE2' if stage == 'RETAINED_PHASE2' else 'NOT_GENERATED_PENDING_MANUAL_BOXES'))
    atomic_text(output / 'gold_subset_selection_30.csv', csv_text(total, tuple(total[0])))
    disagreement = retained_disagreement(old, runs_dir)
    atomic_text(output / 'retained_teacher_disagreement.json', json.dumps(disagreement, indent=2) + '\n')
    store = AnnotationStore(output, new)
    report = dict(schema_version=1, seed=42, selection_policy='class strata + new resolutions + low/high sharpness, '
                   'central brightness, documented prior technical failures; SHA256 path rank tie-break; exact pixel dedup',
                   scanned_native_exports=len(candidates), retained_images=15, additional_images=15,
                   combined_images=30, combined_classes=dict(Counter(s['class'] for s in old + new)),
                   additional_native_dimensions=dict(Counter(f"{s['image_width']}x{s['image_height']}" for s in new)),
                   combined_native_dimensions=dict(Counter(f"{s['image_width']}x{s['image_height']}" for s in old + new)),
                   additional_reasons=dict(Counter(s['selection_reason'] for s in new)),
                   all_30_distinct_pixel_hashes=len(old_hashes | {r['pixel_sha256'] for r in selected}) == 30,
                   original_prompt_file_hashes=protected,
                   source_images_unchanged=True, original_phase2_annotations_unchanged=True,
                   additional_prompts_initialized='30 PENDING slots, no inferred coordinates or human grades',
                   additional_teacher_inference_executed=False, gold_subset_exported=False,
                   source_quality_analysis='Laplacian variance/bright fraction of central 50% at native resolution, read-only arrays.',
                   retained_disagreement_coverage='All 15 original teacher pairs retained, including their largest disagreements.',
                   limitations=['Low sharpness is a blur candidate, not a verified blur diagnosis.',
                                'High sharpness is a readability proxy, not verified easy anatomy.',
                                'Brightness of JPEG pixels is not HU or confirmed pathology; abnormal anatomy needs expert review.',
                                'No reliable case/patient grouping; 30 images are not assumed to be 30 patients.',
                                'No additional teacher candidates, replacement Otsu/LungMask masks, or manual masks are created.'])
    for path, digest in protected.items():
        if digest_file(path) != digest:
            raise RuntimeError('Original Phase-2 annotation files changed during preparation')
    atomic_text(report_path, json.dumps(report, indent=2) + '\n')
    return store, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--smoke', action='store_true', help='Validate/reload the fixed session; no annotation or inference')
    args = parser.parse_args()
    store, report = prepare_additional(args.output_dir)
    print('Retained: 15; additional: 15; total class counts:', report['combined_classes'])
    for sample in store.selection:
        print(sample['class'], Path(sample['source_path']).name,
              f"{sample['image_width']}x{sample['image_height']}", sample['selection_reason'])
    print('Additional annotation states:', dict(Counter(r['visibility_status'] for r in store.records)))
    print('Output:', store.output)
    print('No teacher inference or Gold Subset export was performed.')


if __name__ == '__main__':
    main()
