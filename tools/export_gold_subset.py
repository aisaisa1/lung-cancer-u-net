"""Export only existing, verified candidates and native originals for human correction."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from draw_lung_boxes import PROJECT_ROOT, read_csv, csv_text, digest_file, atomic_text

TEACHERS = (('SAM-Med2D', 'SAM_Med2D'), ('MedSAM', 'MedSAM'))
DEFAULT_POOL = PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/gold_subset_candidates_30'
DEFAULT_GOLD = PROJECT_ROOT / 'DATASET/Lung_Gold_Subset'
ANNOTATION_FIELDS = ('sample_id', 'source_path', 'class', 'candidate_teacher', 'manual_mask_status',
                     'annotator', 'reviewer', 'ambiguity_flag', 'notes')


def verified_copy(source, destination, expected_sha256):
    if destination.exists():
        raise FileExistsError(f'Existing destination will not be overwritten: {destination}')
    if digest_file(source) != expected_sha256:
        raise ValueError(f'Input changed since consolidation: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest_file(destination) != expected_sha256 or digest_file(source) != expected_sha256:
        raise RuntimeError('Export copy is not byte-identical or source changed')


def export(gold=DEFAULT_GOLD, pool=DEFAULT_POOL, base=PROJECT_ROOT):
    base, gold, pool = Path(base).resolve(), Path(gold).resolve(), Path(pool).resolve()
    canonical = (base / 'DATASET/The IQ-OTHNCCD lung cancer dataset').resolve()
    if gold == canonical or canonical in gold.parents or gold in canonical.parents:
        raise ValueError('Gold export must not overlap the original dataset tree')
    if gold.exists():
        raise FileExistsError('Gold directory already exists; validate it rather than overwriting annotation work')
    master = pool.parent / 'gold_subset_candidates_30.csv'
    rows = read_csv(master)
    with (pool / 'consolidation_summary.json').open(encoding='utf-8') as handle:
        summary = json.load(handle)
    with (pool / 'candidate_provenance.json').open(encoding='utf-8') as handle:
        provenance = json.load(handle)
    if digest_file(master) != summary['master_manifest_sha256']:
        raise ValueError('Consolidated manifest changed after verification')
    if len(rows) != 30 or Counter(r['class'] for r in rows) != {'Normal': 10, 'Benign': 10, 'Malignant': 10}:
        raise ValueError('A complete, balanced 30-image pool is required')
    if summary['failures'] or summary['complete_candidate_pairs'] != 30:
        raise ValueError('Export blocked: both candidate masks must exist for every case')
    records = {r['sample_id']: r for r in provenance['samples']}
    if len(records) != 30 or len({r['sample_id'] for r in rows}) != 30:
        raise ValueError('Duplicate or missing provenance sample IDs')
    image_names, mask_names = set(), set()
    # Preflight the entire pool before creating a partial annotation package.
    for row in rows:
        d = records[row['sample_id']]
        if d['source_path'] != row['source_path'] or d['annotation_session'] != row['annotation_session']:
            raise ValueError('Source/session provenance mismatch')
        image_name, mask_name = Path(row['source_path']).name, Path(row['source_path']).stem + '.png'
        if image_name.casefold() in image_names or mask_name.casefold() in mask_names:
            raise ValueError('Source filenames collide; cannot preserve them in a flat annotation package')
        image_names.add(image_name.casefold())
        mask_names.add(mask_name.casefold())
        source = (base / row['source_path']).resolve()
        if canonical not in source.parents or digest_file(source) != d['source_sha256']:
            raise ValueError('Invalid/changed original source')
        shape = (int(row['native_height']), int(row['native_width']))
        for teacher, key in TEACHERS:
            candidate = d['teachers'][teacher]
            if row[f'{key}_success'] != 'True' or not candidate['success'] or row[f'{key}_mask_path'] != candidate['mask_path']:
                raise ValueError('Missing or mismatched successful candidate')
            mask_path = (base / candidate['mask_path']).resolve()
            if base not in mask_path.parents or digest_file(mask_path) != candidate['mask_sha256']:
                raise ValueError('Candidate provenance/content mismatch')
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
            if mask.shape != shape or not np.isin(mask, [0, 255]).all():
                raise ValueError('Candidate is not a native binary mask')
    protocol = base / 'docs/GOLD_SUBSET_ANNOTATION_PROTOCOL.md'
    if not protocol.is_file():
        raise FileNotFoundError('Write the annotation protocol before exporting')
    for directory in ('images', 'candidate_masks/SAM-Med2D', 'candidate_masks/MedSAM', 'masks_manual', 'metadata'):
        (gold / directory).mkdir(parents=True, exist_ok=True)
    package, annotations = [], []
    for row in rows:
        d = records[row['sample_id']]
        image_path = Path('images') / Path(row['source_path']).name
        manual_path = Path('masks_manual') / (Path(row['source_path']).stem + '.png')
        verified_copy(base / row['source_path'], gold / image_path, d['source_sha256'])
        entry = dict(sample_id=row['sample_id'], source_path=row['source_path'], **{'class': row['class']},
                     native_width=int(row['native_width']), native_height=int(row['native_height']),
                     annotation_session=row['annotation_session'], image_path=image_path.as_posix(),
                     image_sha256=d['source_sha256'], manual_mask_path=manual_path.as_posix())
        for teacher, key in TEACHERS:
            candidate = d['teachers'][teacher]
            target = Path('candidate_masks') / teacher / manual_path.name
            overlay = Path('metadata/overlay_previews') / teacher / manual_path.name
            verified_copy(base / candidate['mask_path'], gold / target, candidate['mask_sha256'])
            verified_copy(base / candidate['overlay_path'], gold / overlay, candidate['overlay_sha256'])
            entry.update({f'{key}_candidate_path': target.as_posix(), f'{key}_sha256': candidate['mask_sha256'],
                          f'{key}_origin_path': candidate['mask_path'], f'{key}_overlay_path': overlay.as_posix(),
                          f'{key}_overlay_sha256': candidate['overlay_sha256']})
        package.append(entry)
        annotations.append(dict(sample_id=row['sample_id'], source_path=row['source_path'], **{'class': row['class']},
                                candidate_teacher='', manual_mask_status='NOT_STARTED', annotator='', reviewer='', ambiguity_flag='', notes=''))
    atomic_text(gold / 'metadata/package_manifest.csv', csv_text(package, tuple(package[0])))
    atomic_text(gold / 'metadata/manual_annotation_status.csv', csv_text(annotations, ANNOTATION_FIELDS))
    verified_copy(master, gold / 'metadata/gold_subset_candidates_30.csv', digest_file(master))
    verified_copy(pool / 'candidate_provenance.json', gold / 'metadata/candidate_provenance.json', digest_file(pool / 'candidate_provenance.json'))
    verified_copy(pool / 'consolidation_summary.json', gold / 'metadata/consolidation_summary.json', digest_file(pool / 'consolidation_summary.json'))
    verified_copy(protocol, gold / 'metadata/ANNOTATION_PROTOCOL.md', digest_file(protocol))
    # Manual prompt provenance is metadata, never a pixel reference mask.
    for session in provenance['sessions']:
        prompt_root = base / provenance['sessions'][session]['prompt_directory']
        for name in ('phase2_selection.csv', 'manual_lung_boxes.csv', 'manual_lung_boxes.json'):
            verified_copy(prompt_root / name, gold / 'metadata/prompt_sessions' / session / name, digest_file(prompt_root / name))
    report = dict(package_images=30, candidate_masks_per_teacher=30, overlay_previews_per_teacher=30,
                  manual_masks_created=0, human_annotation_status='NOT_STARTED',
                  original_files_byte_identical=True, source_sessions=list(provenance['sessions']),
                  package_manifest_sha256=digest_file(gold / 'metadata/package_manifest.csv'),
                  labels_are_candidates_not_ground_truth=True,
                  note='This is a correction/annotation package. It becomes a reviewed reference subset only after genuine human annotation and expert review.')
    atomic_text(gold / 'metadata/export_report.json', json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold-dir', type=Path, default=DEFAULT_GOLD)
    parser.add_argument('--pool-dir', type=Path, default=DEFAULT_POOL)
    args = parser.parse_args()
    report = export(args.gold_dir, args.pool_dir)
    print(json.dumps(report, indent=2))
    print('Gold correction package:', args.gold_dir)


if __name__ == '__main__':
    main()
