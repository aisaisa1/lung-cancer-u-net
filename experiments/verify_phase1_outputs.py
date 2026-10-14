"""Validate native-coordinate artifacts/metrics and target file hashes; no inference."""

import argparse
from collections import Counter, defaultdict
import csv
import importlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase1_common import PROJECT_ROOT, read_rgb, sha256_file, write_json

pilot = importlib.import_module('01_teacher_feasibility')


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, name):
    if expected is None:
        if actual not in ('', None):
            raise ValueError(f'{name}: expected null, got {actual}')
    elif not np.isclose(float(actual), float(expected), rtol=1e-6, atol=1e-7):
        raise ValueError(f'{name}: {actual} != {expected}')


def verify(audit_dir, output, smoke=False):
    with (audit_dir / 'dataset_audit.json').open(encoding='utf-8') as handle:
        audit = json.load(handle)
    with (output / 'teacher_summary.json').open(encoding='utf-8') as handle:
        summary = json.load(handle)
    root = Path(audit['dataset_root']).resolve()
    inventory = csv_rows(audit_dir / 'dataset_inventory.csv')
    checked_sources = inventory[:3] if smoke else inventory
    for row in checked_sources:
        path = (root / row['relative_path']).resolve()
        if root not in path.parents or sha256_file(path) != row['file_sha256']:
            raise ValueError(f"Dataset image changed: {row['relative_path']}")
    selected = csv_rows(output / 'selected_images.csv')
    rows = csv_rows(output / 'teacher_feasibility.csv')
    keys = {(r['image_id'], r['teacher'], r['representation']) for r in rows}
    if len(rows) != len(selected) * 2 * 6 or len(keys) != len(rows):
        raise ValueError('Missing or duplicate teacher/representation rows')
    chosen_ids = {r['image_id'] for r in rows}
    if smoke:
        chosen_ids = set(sorted(chosen_ids)[:3])
    masks = defaultdict(dict)
    source_by_id = {}
    artifact_count = 0
    for row in rows:
        if row['image_id'] not in chosen_ids:
            continue
        key = (row['image_id'], row['teacher'])
        shape = (int(row['height']), int(row['width']))
        source_by_id[row['image_id']] = row['relative_path']
        if row['status'] != 'ok':
            if not row['error'] or any(row[f] for f in ('mask_path', 'label_map_path', 'overlay_path', 'roi_path')):
                raise ValueError('Failed inference must have an error and no fabricated artifacts')
            continue
        artifacts = {}
        for field in ('mask_path', 'label_map_path', 'overlay_path', 'roi_path'):
            path = (output / row[field]).resolve()
            if output.resolve() not in path.parents:
                raise ValueError('Artifact path escapes pilot output')
            with Image.open(path) as image:
                array = np.asarray(image).copy()
            if array.shape[:2] != shape:
                raise ValueError(f'Native geometry mismatch: {path}')
            artifacts[field] = array
            artifact_count += 1
        binary = artifacts['mask_path'] > 0
        if not np.isin(artifacts['mask_path'], [0, 255]).all():
            raise ValueError('Saved binary mask values must be 0/255')
        if not np.array_equal(binary, artifacts['label_map_path'] > 0):
            raise ValueError('Saved label map and binary mask disagree')
        for name, value in pilot.technical_qc(binary).items():
            if name == 'empty_mask':
                if row[name] != str(value):
                    raise ValueError('Empty mask flag differs')
            else:
                close(row[name], value, name)
        masks[key][row['representation']] = binary

    for image_id, relative in source_by_id.items():
        native = read_rgb(root / relative)
        original = read_rgb(output / 'originals' / f'{image_id}.png')
        if not np.array_equal(native, original):
            raise ValueError('Original display differs from decoded source pixels')
        for rep in pilot.REPRESENTATIONS:
            view = read_rgb(output / 'representations' / f'{image_id}_{rep}.png')
            if view.shape != native.shape:
                raise ValueError('Representation canvas differs from native source')

    pairs = csv_rows(output / 'representation_pairwise.csv')
    actual_pair_count = 0
    for row in pairs:
        if row['image_id'] not in chosen_ids:
            continue
        available = masks[row['image_id'], row['teacher']]
        metrics = pilot.agreement(available[row['representation_a']], available[row['representation_b']])
        for name in ('dice', 'iou'):
            close(row[name], metrics[name], name)
        if row['both_empty'] != str(metrics['both_empty']):
            raise ValueError('Both-empty pair flag differs')
        actual_pair_count += 1
    expected_pair_count = sum(len(v) * (len(v) - 1) // 2 for v in masks.values())
    if actual_pair_count != expected_pair_count:
        raise ValueError('Missing representation agreement pairs')
    stability = csv_rows(output / 'representation_stability.csv')
    map_count = 0
    for row in stability:
        if row['image_id'] not in chosen_ids:
            continue
        available = masks[row['image_id'], row['teacher']]
        metrics = [pilot.agreement(a, b) for a, b in combinations(available.values(), 2)]
        if int(row['successful_representations']) != len(available):
            raise ValueError('Stability coverage differs from actual masks')
        if row['complete'] != str(len(available) == 6):
            raise ValueError('Complete-representation flag differs')
        for name in ('dice', 'iou'):
            close(row['mean_' + name], np.mean([m[name] for m in metrics]) if metrics else None, 'mean_' + name)
            close(row['minimum_' + name], min(m[name] for m in metrics) if metrics else None, 'minimum_' + name)
        if metrics:
            stack = np.stack(list(available.values())).astype(bool)
            # Independently derive the map from explicit pairwise XORs.
            expected = np.mean([np.logical_xor(a, b) for a, b in combinations(stack, 2)], axis=0)
            actual = np.load(output / row['disagreement_map_path'], allow_pickle=False)
            if actual.shape != expected.shape or not np.allclose(actual, expected, atol=1e-7):
                raise ValueError('Disagreement map differs from pairwise disagreement fraction')
            map_count += 1
    cross = [r for r in rows if r['teacher'] == 'R231' and r['representation'] == 'native'
             and r['cross_teacher_dice'] != '']
    for row in cross:
        if row['image_id'] not in chosen_ids:
            continue
        metrics = pilot.agreement(masks[row['image_id'], 'R231']['native'],
                                  masks[row['image_id'], 'R231CovidWeb']['native'])
        close(row['cross_teacher_dice'], metrics['dice'], 'cross_teacher_dice')
        close(row['cross_teacher_iou'], metrics['iou'], 'cross_teacher_iou')
    nonempty_cross = [r for r in cross if r['cross_teacher_both_empty'] == 'False']
    duplicate_stats = {}
    for source in audit['sources']:
        source_rows = [r for r in inventory if r['source'] == source]
        hashes = Counter(r['pixel_sha256'] for r in source_rows)
        duplicate_stats[source] = dict(images=len(source_rows), unique_pixel_hashes=len(hashes),
                                       duplicate_pixel_groups=sum(n > 1 for n in hashes.values()),
                                       redundant_images=sum(n - 1 for n in hashes.values()))
    duplicate_rows = csv_rows(audit_dir / 'duplicate_report.csv')
    cross_class_groups = {r['hash'] for r in duplicate_rows
                          if r['duplicate_type'] == 'identical_decoded_pixels' and r['cross_class'] == 'True'}
    summaries = {}
    for name in pilot.TEACHERS:
        native = [r for r in rows if r['teacher'] == name and r['representation'] == 'native']
        ok = [r for r in native if r['status'] == 'ok']
        close(summary['teachers'][name]['native_successes'], len(ok), 'summary_native_successes')
        complete = [s for s in stability if s['teacher'] == name and s['complete'] == 'True']
        valid_min = [float(s['minimum_dice']) for s in complete]
        error_counts = Counter(r['error'] for r in rows if r['teacher'] == name and r['status'] != 'ok')
        summaries[name] = dict(native_nonempty=len([r for r in ok if r['empty_mask'] == 'False']),
                               native_error_counts=dict(Counter(r['error'] for r in native if r['status'] != 'ok')),
                               all_view_error_counts=dict(error_counts),
                               complete_images_with_zero_minimum_dice=sum(v == 0 for v in valid_min),
                               complete_nonempty_pair_mean_dice=pilot.distribution([
                                   float(s['nonempty_mean_dice']) for s in complete if s['nonempty_mean_dice']]),
                               complete_nonempty_pair_mean_iou=pilot.distribution([
                                   float(s['nonempty_mean_iou']) for s in complete if s['nonempty_mean_iou']]))
    report = dict(smoke=smoke, passed=True, source_image_hashes_checked=len(checked_sources),
                  source_images_unchanged=True, image_ids_checked=len(chosen_ids),
                  artifact_images_checked=artifact_count, pairwise_metrics_checked=actual_pair_count,
                  disagreement_maps_checked=map_count, feasibility_rows=len(rows),
                  per_source_duplicate_summary=duplicate_stats, cross_class_exact_pixel_groups=len(cross_class_groups),
                  teachers=summaries,
                  cross_teacher_nonempty_union_agreement=dict(n=len(nonempty_cross),
                      dice=pilot.distribution([float(r['cross_teacher_dice']) for r in nonempty_cross]),
                      iou=pilot.distribution([float(r['cross_teacher_iou']) for r in nonempty_cross])),
                  caveat='Artifact correctness and mathematical agreement are not anatomical validation.')
    write_json(output / ('verification_smoke.json' if smoke else 'verification.json'), report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/audit')
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/teacher_pilot')
    parser.add_argument('--smoke', action='store_true', help='Validate at most three image IDs and three source hashes')
    args = parser.parse_args()
    verify(args.audit_dir.resolve(), args.output_dir.resolve(), args.smoke)


if __name__ == '__main__':
    main()
