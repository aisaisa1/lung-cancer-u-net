"""Check package integrity and human-mask format/provenance; no accuracy or training."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from draw_lung_boxes import PROJECT_ROOT, read_csv, digest_file, atomic_text, csv_text

STATUS = {'NOT_STARTED', 'ANNOTATED_PENDING_REVIEW', 'NEEDS_EXPERT_REVIEW', 'REVIEWED'}
STARTING_TEACHERS = {'', 'SAM-Med2D', 'MedSAM', 'BOTH_REFERENCE', 'FROM_SCRATCH'}
FREEZE_MANIFEST_NAME = 'manual_freeze_manifest.json'


def load_freeze_manifest(gold):
    manifest_path = gold / 'metadata' / FREEZE_MANIFEST_NAME
    if manifest_path.exists():
        with manifest_path.open(encoding='utf-8') as handle:
            return json.load(handle)
    return {}


def save_freeze_manifest(gold, manifest):
    manifest_path = gold / 'metadata' / FREEZE_MANIFEST_NAME
    atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + '\n')


def freeze_manual_masks(gold, base=PROJECT_ROOT):
    gold, base = Path(gold).resolve(), Path(base).resolve()
    manifest = gold / 'metadata/package_manifest.csv'
    with (gold / 'metadata/export_report.json').open(encoding='utf-8') as handle:
        export = json.load(handle)
    if digest_file(manifest) != export['package_manifest_sha256']:
        raise ValueError('Frozen package manifest changed')
    package = read_csv(manifest)
    status_rows = read_csv(gold / 'metadata/manual_annotation_status.csv')
    statuses = {r['sample_id']: r for r in status_rows}
    freeze_manifest = load_freeze_manifest(gold)
    frozen_count = 0
    for row in package:
        sid = row['sample_id']
        info = statuses[sid]
        manual = (gold / row['manual_mask_path']).resolve()
        if not manual.exists():
            continue
        if info['manual_mask_status'] not in {'ANNOTATED_PENDING_REVIEW', 'NEEDS_EXPERT_REVIEW', 'REVIEWED'}:
            continue
        current_hash = digest_file(manual)
        if sid in freeze_manifest:
            if freeze_manifest[sid] != current_hash:
                raise ValueError(f'Manual mask changed after freeze: {sid}')
        else:
            freeze_manifest[sid] = current_hash
            frozen_count += 1
    save_freeze_manifest(gold, freeze_manifest)
    return dict(frozen=frozen_count, total_in_manifest=len(freeze_manifest))


def validate(gold, base=PROJECT_ROOT):
    gold, base = Path(gold).resolve(), Path(base).resolve()
    manifest = gold / 'metadata/package_manifest.csv'
    with (gold / 'metadata/export_report.json').open(encoding='utf-8') as handle:
        export = json.load(handle)
    if digest_file(manifest) != export['package_manifest_sha256']:
        raise ValueError('Frozen package manifest changed')
    package = read_csv(manifest)
    status_rows = read_csv(gold / 'metadata/manual_annotation_status.csv')
    statuses = {r['sample_id']: r for r in status_rows}
    if len(package) != 30 or len(status_rows) != 30 or len(statuses) != 30:
        raise ValueError('Expected exactly 30 package/status records')
    if Counter(r['class'] for r in package) != {'Normal': 10, 'Benign': 10, 'Malignant': 10}:
        raise ValueError('Class-balanced 30-image package required')
    missing, issues, pending_review, counts = [], [], [], Counter()
    expected_manual_paths = set()
    for row in package:
        sid = row['sample_id']
        info = statuses[sid]
        if any(info[k] != row[k] for k in ('source_path', 'class')):
            raise ValueError('Human metadata no longer matches source manifest')
        if info['manual_mask_status'] not in STATUS or info['candidate_teacher'] not in STARTING_TEACHERS:
            raise ValueError('Unsupported annotation status or starting candidate choice')
        counts[info['manual_mask_status']] += 1
        size = (int(row['native_width']), int(row['native_height']))
        for path, expected in ((gold / row['image_path'], row['image_sha256']),
                               (base / row['source_path'], row['image_sha256'])):
            if digest_file(path) != expected:
                raise ValueError(f'Original/copy hash mismatch: {path}')
            with Image.open(path) as image:
                if image.size != size:
                    raise ValueError('Original native dimensions changed')
        for key in ('SAM_Med2D', 'MedSAM'):
            candidate = gold / row[f'{key}_candidate_path']
            if digest_file(candidate) != row[f'{key}_sha256']:
                raise ValueError('Exported candidate changed; edit only masks_manual, not candidate inputs')
            with Image.open(candidate) as image:
                array = np.asarray(image)
            if array.shape != size[::-1] or not np.isin(array, [0, 255]).all():
                raise ValueError('Candidate mask geometry/encoding mismatch')
            if digest_file(gold / row[f'{key}_overlay_path']) != row[f'{key}_overlay_sha256']:
                raise ValueError('Exported overlay preview changed')
        manual = (gold / row['manual_mask_path']).resolve()
        if (gold / 'masks_manual').resolve() not in manual.parents:
            raise ValueError('Manual mask path escaped its designated folder')
        expected_manual_paths.add(manual)
        if not manual.exists():
            missing.append(sid)
            if info['manual_mask_status'] != 'NOT_STARTED':
                issues.append(dict(sample_id=sid, error='Human metadata claims progress but no manual mask exists'))
            continue
        try:
            with Image.open(manual) as image:
                array = np.asarray(image).copy()
                if image.mode not in ('L', '1') or image.size != size:
                    raise ValueError('Manual mask must be native-size grayscale/1-bit PNG, not RGB or a resized image')
            values = set(np.unique(array).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                raise ValueError('Manual mask has non-binary values/antialiasing; no automatic threshold repair performed')
            if not array.any():
                raise ValueError('Empty foreground needs explicit expert adjudication; visible lung prompts exist')
            if not info['annotator'].strip() or not info['notes'].strip() or not info['candidate_teacher']:
                raise ValueError('Record human annotator, starting candidate choice, and corrections/review notes')
            if info['manual_mask_status'] == 'NOT_STARTED':
                raise ValueError('Manual file exists but annotation status has not been recorded by a human')
            if info['manual_mask_status'] != 'REVIEWED' or not info['reviewer'].strip():
                pending_review.append(sid)
        except (ValueError, OSError) as error:
            issues.append(dict(sample_id=sid, error=str(error)))
    unexpected = [p.relative_to(gold).as_posix() for p in (gold / 'masks_manual').rglob('*')
                  if p.is_file() and p.resolve() not in expected_manual_paths]
    if unexpected:
        issues.append(dict(sample_id='', error='Unexpected/unmapped manual files: ' + ', '.join(unexpected)))

    freeze_manifest = load_freeze_manifest(gold)
    freeze_issues = 0
    for row in package:
        sid = row['sample_id']
        manual = (gold / row['manual_mask_path']).resolve()
        if manual.exists() and sid in freeze_manifest:
            current_hash = digest_file(manual)
            if freeze_manifest[sid] != current_hash:
                issues.append(dict(sample_id=sid, error='Manual mask hash differs from freeze manifest'))
                freeze_issues += 1
    protected_path = base / 'OUTPUT_HASIL/segmentation/gold_subset_candidates_30/protected_phase2_hashes.json'
    protected_path = base / 'OUTPUT_HASIL/segmentation/gold_subset_candidates_30/protected_phase2_hashes.json'
    protected_count = 0
    if protected_path.is_file():
        with protected_path.open(encoding='utf-8') as handle:
            protected = json.load(handle)
        for relative, expected in protected.items():
            if digest_file(base / relative) != expected:
                raise ValueError('Original Phase-2 annotation/output changed after consolidation')
            protected_count += 1
    with (gold / 'metadata/candidate_provenance.json').open(encoding='utf-8') as handle:
        provenance = json.load(handle)
    for session, info in provenance['sessions'].items():
        for suffix, key in (('csv', 'annotation_csv_sha256'), ('json', 'annotation_json_sha256')):
            for prompt_dir in (base / info['prompt_directory'], gold / 'metadata/prompt_sessions' / session):
                if digest_file(prompt_dir / f'manual_lung_boxes.{suffix}') != info[key]:
                    raise ValueError('Original/exported prompt snapshot does not match candidate provenance')
    report = dict(package_valid=True, package_images=30, candidates_per_teacher=30,
                  manual_masks_present=30 - len(missing), missing_manual_samples=missing,
                  manual_issues=issues, expert_review_pending=pending_review, annotation_status_counts=dict(counts),
                  manual_complete=not missing and not issues,
                  reviewed_complete=not missing and not issues and not pending_review,
                  source_images_unchanged=True, protected_phase2_files_verified=protected_count,
                  no_masks_fabricated=True,
                  freeze_manifest_entries=len(freeze_manifest),
                  freeze_issues=freeze_issues,
                  limitation='Technical/provenance validation cannot establish anatomical correctness; human expert review remains required.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold-dir', type=Path, default=PROJECT_ROOT / 'DATASET/Lung_Gold_Subset')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--require-reviewed', action='store_true')
    parser.add_argument('--freeze', action='store_true', help='Freeze current manual masks by recording their hashes')
    parser.add_argument('--report', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/gold_subset_candidates_30/gold_package_validation.json')
    args = parser.parse_args()
    if args.freeze:
        result = freeze_manual_masks(args.gold_dir)
        print(json.dumps(result, indent=2))
        return
    report = validate(args.gold_dir)
    atomic_text(args.report, json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if (args.require_complete and not report['manual_complete']) or (args.require_reviewed and not report['reviewed_complete']):
        sys.exit('Manual mask annotation/review is incomplete; no training or automatic mask repair was performed.')


if __name__ == '__main__':
    main()
