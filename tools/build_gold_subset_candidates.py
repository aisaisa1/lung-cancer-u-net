"""Consolidate the two completed human-prompt sessions; never runs a teacher."""

import argparse
from collections import Counter
import importlib
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from draw_lung_boxes import PROJECT_ROOT, read_csv, digest_file, csv_text, atomic_text
from validate_phase2_annotations import validate

sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))
comparison = importlib.import_module('phase2_build_comparison')

FIELDS = ('sample_id', 'source_path', 'class', 'native_width', 'native_height', 'annotation_session',
          'SAM_Med2D_mask_path', 'MedSAM_mask_path', 'SAM_Med2D_success', 'MedSAM_success', 'notes')
TEACHERS = (('SAM-Med2D', 'SAM_Med2D'), ('MedSAM', 'MedSAM'))
SESSIONS = (('phase2_prompts', 'phase2_teacher_comparison'),
            ('phase3_additional_prompts', 'phase3_teacher_comparison'))
DEFAULT_OUTPUT = PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/gold_subset_candidates_30'


def write_json(path, data):
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def protected_files(base):
    """Record original annotations and outputs as immutable consolidation inputs."""
    roots = (base / 'OUTPUT_HASIL/segmentation/phase2_prompts',
             base / 'OUTPUT_HASIL/segmentation/phase2_teacher_comparison')
    return {p.relative_to(base).as_posix(): digest_file(p)
            for root in roots for p in root.rglob('*') if p.is_file()}


def check_geometry(row, annotation, shape, teacher, model_size):
    native = np.array([annotation[k] for k in ('x_min', 'y_min', 'x_max', 'y_max')], dtype=float)
    if json.loads(row['native_box']) != native.tolist():
        raise ValueError('Prediction native box differs from the exact human annotation')
    h, w = shape
    if teacher == 'MedSAM':
        resized_h, resized_w = 1024, 1024
    else:
        resized_h, resized_w = int(h * model_size / max(h, w) + 0.5), int(w * model_size / max(h, w) + 0.5)
    expected = native * np.array([resized_w / w, resized_h / h] * 2)
    if not np.allclose(json.loads(row['model_box']), expected, rtol=0, atol=1e-7):
        raise ValueError('Incorrect native-to-model box coordinate transform')
    if tuple(json.loads(row['model_input_size'])) != (resized_h, resized_w):
        raise ValueError('Unexpected resized model input dimensions')


def consolidate(output=DEFAULT_OUTPUT, base=PROJECT_ROOT):
    base, output = Path(base).resolve(), Path(output).resolve()
    if (base / 'DATASET').resolve() == output or (base / 'DATASET').resolve() in output.parents:
        raise ValueError('Consolidation reports must be outside DATASET')
    if (output / 'consolidation_summary.json').exists():
        raise FileExistsError('Consolidation already completed; choose a new output folder or validate existing files')
    baseline = protected_files(base)
    output.mkdir(parents=True, exist_ok=True)
    rows, details, sessions, reference = [], [], {}, {}
    seen_ids, seen_sources, seen_pixels = set(), set(), set()
    geometry_count, union_count = 0, 0
    for prompt_name, run_name in SESSIONS:
        prompts = base / 'OUTPUT_HASIL/segmentation' / prompt_name
        runs = base / 'OUTPUT_HASIL/segmentation' / run_name
        selection, annotations, validation = validate(prompts, output / 'validation' / prompt_name, base)
        if not validation['ready_for_inference']:
            raise ValueError(f'{prompt_name}: unfinished human prompts')
        session_data = dict(prompt_directory=prompts.relative_to(base).as_posix(),
                            run_directory=runs.relative_to(base).as_posix(),
                            annotation_csv_sha256=validation['csv_sha256'], annotation_json_sha256=validation['json_sha256'],
                            selected_images=len(selection), teachers={})
        predictions, summaries = {}, {}
        for teacher, key in TEACHERS:
            directory = runs / teacher
            with (directory / 'teacher_summary.json').open(encoding='utf-8') as handle:
                summary = json.load(handle)
            if summary['annotation_csv_sha256'] != validation['csv_sha256'] or \
                    summary['annotation_json_sha256'] != validation['json_sha256']:
                raise ValueError(f'{prompt_name}/{teacher}: annotation snapshot mismatch')
            if summary['smoke'] or summary['selected_images'] != 15:
                raise ValueError('Only complete 15-image session outputs may enter this pool')
            if not summary['model']:
                raise ValueError(f'{teacher}: pretrained model setup failed')
            signature = {k: summary['model'][k] for k in ('repo_commit', 'checkpoint_sha256', 'image_size',
                                                        'preprocessing', 'output_policy')}
            if teacher in reference and reference[teacher] != signature:
                raise ValueError(f'{teacher}: original/additional pipeline or checkpoint differs')
            reference[teacher] = signature
            summaries[teacher] = summary
            data = read_csv(directory / 'predictions.csv')
            if len(data) != 45 or len({(r['sample'], r['region']) for r in data}) != 45:
                raise ValueError('Missing/duplicate independent or merged prediction records')
            predictions[teacher] = {(r['sample'], r['region']): r for r in data}
            session_data['teachers'][teacher] = dict(summary_path=(directory / 'teacher_summary.json').relative_to(base).as_posix(),
                                                     technical_successes=summary['technical_successes'], failures=summary['failures'],
                                                     model_signature=signature)
        for sample in selection:
            sid, shape = sample['sample_id'], (sample['image_height'], sample['image_width'])
            if sid in seen_ids or sample['source_path'] in seen_sources:
                raise ValueError('The original and additional sessions overlap')
            seen_ids.add(sid)
            seen_sources.add(sample['source_path'])
            rgb = np.asarray(Image.open(base / sample['source_path']).convert('RGB'))
            pixel_hash = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
            if pixel_hash in seen_pixels:
                raise ValueError('Exact decoded-pixel duplicate in candidate pool')
            seen_pixels.add(pixel_hash)
            record = dict(sample_id=sid, source_path=sample['source_path'], **{'class': sample['class']},
                          native_width=shape[1], native_height=shape[0], annotation_session=prompt_name,
                          SAM_Med2D_mask_path='', MedSAM_mask_path='', SAM_Med2D_success=False, MedSAM_success=False,
                          notes='Teacher candidates only; human whole-lung correction/review is pending.')
            annotation = [a for a in annotations if a['sample_id'] == sid]
            panels = [('Original (native)', rgb), ('Exact manual boxes', comparison.manual_panel(rgb, annotation))]
            detail = dict(sample_id=sid, source_path=sample['source_path'], source_sha256=sample['file_sha256'],
                          pixel_sha256=pixel_hash, annotation_session=prompt_name,
                          native_width=shape[1], native_height=shape[0], teachers={})
            for teacher, key in TEACHERS:
                directory = runs / teacher
                result = predictions[teacher][sid, 'merged']
                visible = [a for a in annotation if a['visibility_status'] == 'VISIBLE']
                success = result['technical_success'] == 'True'
                if not success or not result['mask_path']:
                    detail['teachers'][teacher] = dict(success=False, error=result['error'])
                    panels.extend([(teacher + ' FAILED', np.full_like(rgb, 230)),
                                   (teacher + ' no candidate fabricated', np.full_like(rgb, 230))])
                    continue
                individual = []
                for a in annotation:
                    individual_row = predictions[teacher][sid, a['box_id']]
                    if a['visibility_status'] != 'VISIBLE':
                        if individual_row['mask_path']:
                            raise ValueError('Unexpected fabricated mask for NOT_VISIBLE region')
                        continue
                    if individual_row['technical_success'] != 'True':
                        raise ValueError('Merged success claimed despite a visible-region prediction failure')
                    mask = comparison.read_mask(directory / individual_row['mask_path'], shape)
                    check_geometry(individual_row, a, shape, teacher, summaries[teacher]['model']['image_size'])
                    individual.append(mask)
                    geometry_count += 1
                binary = comparison.read_mask(directory / result['mask_path'], shape)
                if not visible or len(individual) != len(visible) or not np.array_equal(binary, np.logical_or.reduce(individual)):
                    raise ValueError('Whole-lung candidate is not the exact native independent-mask union')
                union_count += 1
                overlay = np.asarray(Image.open(directory / result['overlay_path']).convert('RGB'))
                if overlay.shape != rgb.shape:
                    raise ValueError('Overlay is not on the native canvas')
                candidate = (directory / result['mask_path']).relative_to(base).as_posix()
                record[f'{key}_mask_path'], record[f'{key}_success'] = candidate, True
                detail['teachers'][teacher] = dict(success=True, mask_path=candidate,
                    mask_sha256=digest_file(base / candidate), nonempty=bool(binary.any()),
                    overlay_path=(directory / result['overlay_path']).relative_to(base).as_posix(),
                    overlay_sha256=digest_file(directory / result['overlay_path']),
                    individual_paths=[(directory / predictions[teacher][sid, a['box_id']]['mask_path']).relative_to(base).as_posix()
                                      for a in visible], runtime_seconds=float(result['runtime_seconds']))
                panels.extend([(teacher + ' candidate mask', binary.astype(np.uint8) * 255),
                               (teacher + ' overlay', overlay)])
            comparison.sheet(output / 'contact_sheets' / f'{sid}_comparison.png', panels,
                             f'{sample["class"]} | {Path(sample["source_path"]).name} | {prompt_name} | candidates, not ground truth')
            rows.append(record)
            details.append(detail)
        sessions[prompt_name] = session_data
    if len(rows) != 30 or Counter(r['class'] for r in rows) != {'Normal': 10, 'Benign': 10, 'Malignant': 10}:
        raise ValueError('Consolidated pool must contain exactly 30 class-balanced images')
    failures = [{'sample_id': r['sample_id'], 'teacher': teacher} for r in rows for teacher, key in TEACHERS
                if not r[f'{key}_success']]
    for path, expected in baseline.items():
        if digest_file(base / path) != expected:
            raise RuntimeError('Original Phase-2 input/output changed during consolidation')
    for d in details:
        if digest_file(base / d['source_path']) != d['source_sha256']:
            raise RuntimeError('Source image changed during consolidation')
    manifest = output.parent / 'gold_subset_candidates_30.csv'
    if manifest.exists():
        raise FileExistsError('Master candidate manifest already exists; do not overwrite user metadata')
    atomic_text(manifest, csv_text(rows, FIELDS))
    summary = dict(selected_images=30, class_counts=dict(Counter(r['class'] for r in rows)),
                   native_dimensions=dict(Counter(f"{r['native_width']}x{r['native_height']}" for r in rows)),
                   teacher_successes={teacher: sum(r[f'{key}_success'] for r in rows) for teacher, key in TEACHERS},
                   failures=failures, complete_candidate_pairs=len(rows) - len({r['sample_id'] for r in failures}),
                   independent_box_geometry_checks=geometry_count, native_union_checks=union_count,
                   original_phase2_files_unchanged=True, source_images_unchanged=True,
                   master_manifest_sha256=digest_file(manifest), mask_review_status='PENDING',
                   no_accuracy_computed=True, no_student_training=True, sessions=sessions)
    write_json(output / 'candidate_provenance.json', dict(samples=details, sessions=sessions))
    write_json(output / 'protected_phase2_hashes.json', baseline)
    write_json(output / 'consolidation_summary.json', summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _, report = consolidate(args.output_dir)
    print(json.dumps(report, indent=2))
    if report['failures']:
        sys.exit('Candidate pool has failures; Gold Subset export remains blocked.')


if __name__ == '__main__':
    main()
