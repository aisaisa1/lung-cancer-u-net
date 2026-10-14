"""Inventory local IQ-OTH/NCCD images without modifying or splitting them."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image

from phase1_common import (CLASSES, IMAGE_EXTENSIONS, PROJECT_ROOT, class_name,
                           prepare_output, sha256_file, write_csv, write_json)


def source_kind(source):
    lower = source.lower()
    if 'mask' in lower:
        return 'unverified_mask_or_annotation'
    if 'crop' in lower:
        return 'derived_crop'
    if 'gamma' in lower or 'clahe' in lower:
        return 'derived_gamma_clahe'
    if 'iq-oth' in lower or 'iq_oth' in lower:
        return 'least_processed_export_candidate'
    return 'unknown_provenance'


def audit(dataset_root, output, smoke=False):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Target dataset missing: {dataset_root}')
    files = sorted(p for p in dataset_root.rglob('*') if p.is_file() and not p.is_symlink())
    candidates = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
    source_counts = Counter((p.relative_to(dataset_root).parts[0],
                             class_name(p.relative_to(dataset_root).parts[:-1])) for p in candidates)
    if smoke:
        limited = defaultdict(list)
        for path in candidates:
            rel = path.relative_to(dataset_root)
            key = (rel.parts[0], class_name(rel.parts[:-1]))
            if len(limited[key]) < 2:
                limited[key].append(path)
        candidates = [p for paths in limited.values() for p in paths]

    inventory = []
    for path in candidates:
        relative = path.relative_to(dataset_root)
        source = relative.parts[0]
        row = dict(relative_path=relative.as_posix(), source=source,
                   source_kind=source_kind(source), class_name=class_name(relative.parts[:-1]),
                   extension=path.suffix.lower(), file_bytes=path.stat().st_size,
                   width='', height='', mode='', dtype='', file_sha256='', pixel_sha256='',
                   candidate_identity='', identity_status='not_inferred', verified_case_id='',
                   exif_keys='', status='ok', error='')
        serial = re.fullmatch(r'(?:Normal|Bengin|Benign|Malignant) case\s*\((\d+)\)',
                              path.stem, flags=re.IGNORECASE)
        explicit = re.search(r'(?:patient|subject|case)[_-]([A-Za-z0-9]+)', path.stem, re.I)
        if serial:
            row['candidate_identity'] = serial.group(1)
            row['identity_status'] = 'ambiguous_class_local_filename_serial_not_patient_id'
        elif explicit:
            row['candidate_identity'] = explicit.group(1)
            row['identity_status'] = 'unverified_filename_candidate_requires_metadata'
        try:
            row['file_sha256'] = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                array = np.asarray(image)
                row.update(width=image.width, height=image.height, mode=image.mode,
                           dtype=str(array.dtype), exif_keys=';'.join(map(str, image.getexif().keys())))
                digest = hashlib.sha256()
                digest.update(f'{image.mode}|{array.shape}|{array.dtype}'.encode())
                digest.update(array.tobytes())
                row['pixel_sha256'] = digest.hexdigest()
        except Exception as error:
            row.update(status='error', error=f'{type(error).__name__}: {error}')
        inventory.append(row)

    duplicates = []
    for kind, field in (('identical_file_bytes', 'file_sha256'),
                        ('identical_decoded_pixels', 'pixel_sha256')):
        grouped = defaultdict(list)
        for row in inventory:
            if row[field]:
                grouped[row[field]].append(row)
        for digest, group in sorted(grouped.items()):
            if len(group) < 2:
                continue
            classes = sorted({r['class_name'] for r in group})
            within = Counter(r['source'] for r in group)
            for row in group:
                duplicates.append(dict(duplicate_type=kind, hash=digest, group_size=len(group),
                                       relative_path=row['relative_path'], source=row['source'],
                                       class_name=row['class_name'], cross_class=len(classes) > 1,
                                       within_source_count=within[row['source']]))

    sources = {}
    for source in sorted({s for s, _ in source_counts}):
        rows = [r for r in inventory if r['source'] == source]
        sources[source] = dict(kind=source_kind(source),
                               total_images=sum(n for (s, _), n in source_counts.items() if s == source),
                               class_counts={c: source_counts[source, c] for c in (*CLASSES, 'Unknown')},
                               inspected_images=len(rows),
                               dimensions=dict(Counter(f"{r['width']}x{r['height']}" for r in rows
                                                       if r['status'] == 'ok')),
                               extensions=dict(Counter(r['extension'] for r in rows)),
                               read_errors=sum(r['status'] != 'ok' for r in rows))
    raw_candidates = [s for s, info in sources.items()
                      if info['kind'] == 'least_processed_export_candidate'
                      and all(info['class_counts'][c] > 0 for c in CLASSES)
                      and info['read_errors'] == 0]
    canonical = raw_candidates[0] if len(raw_candidates) == 1 else None
    nonimages = [dict(relative_path=p.relative_to(dataset_root).as_posix(),
                     extension=p.suffix.lower(), file_bytes=p.stat().st_size)
                 for p in files if p.suffix.lower() not in IMAGE_EXTENSIONS]
    report = dict(schema_version=1, smoke=smoke, scope='partial image/hash inspection' if smoke else 'full',
                  dataset_root=str(dataset_root), total_image_files=sum(source_counts.values()),
                  inspected_image_files=len(inventory), sources=sources, non_image_files=nonimages,
                  canonical_teacher_source=canonical,
                  canonical_teacher_reason=('Only full-class, least-processed IQ-OTH/NCCD export candidate. '
                                           'Use stored native dimensions, not Gamma/CLAHE or crops. '
                                           'Folder naming does not establish original DICOM provenance.'
                                           if canonical else 'BLOCKED: no unique least-processed candidate; '
                                           'resolve provenance explicitly, no silent fallback.'),
                  hu_status='Regular image exports are not calibrated HU; do not reconstruct HU without metadata.',
                  grouping=dict(safely_inferred=False, verified_case_count=0,
                                identity_status_counts=dict(Counter(r['identity_status'] for r in inventory)),
                                evidence='Class folders and numbered case (N) filenames alone do not link slices '
                                         'to patients. Numbers may be image serials. No identity is assigned.',
                                required='Obtain original case-to-slice mapping or DICOM study/series metadata '
                                         'before any patient-disjoint training/evaluation split.'),
                  duplicate_group_counts={kind: len({r['hash'] for r in duplicates if r['duplicate_type'] == kind})
                                          for kind in ('identical_file_bytes', 'identical_decoded_pixels')},
                  limitations=['Exact hashes do not detect near-duplicates or intensity-transformed versions.',
                               'Mask folder names do not verify expert annotation or anatomical correctness.',
                               'Smoke counts enumerate filenames, but dimensions/hashes cover only two per source/class.',
                               'No train/validation/test split was created.'])
    write_csv(output / 'dataset_inventory.csv', inventory, list(inventory[0]) if inventory else [
        'relative_path', 'source', 'class_name', 'status'])
    write_csv(output / 'duplicate_report.csv', duplicates, ['duplicate_type', 'hash', 'group_size',
              'relative_path', 'source', 'class_name', 'cross_class', 'within_source_count'])
    write_json(output / 'dataset_audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=PROJECT_ROOT / 'DATASET')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/audit'
    if args.smoke and args.output_dir is None:
        output = output / 'smoke'
    output = prepare_output(output, root, 'dataset_audit.json')
    report = audit(root, output, args.smoke)
    print(f"Inspected {report['inspected_image_files']}/{report['total_image_files']} image files")
    for source, info in report['sources'].items():
        print(source, info['class_counts'], info['dimensions'])
    print('Canonical teacher source:', report['canonical_teacher_source'])
    print('Safe patient/case grouping: NOT ESTABLISHED')
    print('Outputs:', output)


if __name__ == '__main__':
    main()
