"""Annotation-package tests use synthetic fixtures, never research/manual ground truth."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from draw_lung_boxes import digest_file, csv_text, read_csv
from export_gold_subset import export, ANNOTATION_FIELDS
from validate_gold_subset import validate
from build_gold_subset_candidates import check_geometry


class GoldSubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.pool = self.base / 'OUTPUT_HASIL/segmentation/gold_subset_candidates_30'
        self.pool.mkdir(parents=True)
        self.gold = self.base / 'DATASET/Lung_Gold_Subset'
        rows, samples = [], []
        sessions = {
            'phase2_prompts': {'prompt_directory': 'OUTPUT_HASIL/segmentation/phase2_prompts'},
            'phase3_additional_prompts': {'prompt_directory': 'OUTPUT_HASIL/segmentation/phase3_additional_prompts'}}
        for session in sessions.values():
            directory = self.base / session['prompt_directory']
            directory.mkdir(parents=True)
            for name in ('phase2_selection.csv', 'manual_lung_boxes.csv', 'manual_lung_boxes.json'):
                (directory / name).write_text('synthetic fixture\n', encoding='utf-8')
            session['annotation_csv_sha256'] = digest_file(directory / 'manual_lung_boxes.csv')
            session['annotation_json_sha256'] = digest_file(directory / 'manual_lung_boxes.json')
        for cls in ('Normal', 'Benign', 'Malignant'):
            for i in range(10):
                sid = f'{cls.lower()}_{i}'
                relative = Path('DATASET/The IQ-OTHNCCD lung cancer dataset') / f'{cls} cases' / f'{cls} case ({i}).png'
                source = self.base / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                size = (61, 39) if cls == 'Malignant' and i == 9 else (43, 31)
                Image.new('RGB', size, (20 + i, 50, 80)).save(source)
                session = 'phase2_prompts' if i < 5 else 'phase3_additional_prompts'
                row = dict(sample_id=sid, source_path=relative.as_posix(), **{'class': cls}, native_width=size[0],
                           native_height=size[1], annotation_session=session, SAM_Med2D_mask_path='', MedSAM_mask_path='',
                           SAM_Med2D_success=True, MedSAM_success=True, notes='synthetic candidate fixture')
                detail = dict(sample_id=sid, source_path=relative.as_posix(), annotation_session=session,
                              source_sha256=digest_file(source), teachers={})
                for teacher, key in (('SAM-Med2D', 'SAM_Med2D'), ('MedSAM', 'MedSAM')):
                    mask_relative = Path('OUTPUT_HASIL/fixture') / teacher / f'{sid}.png'
                    overlay_relative = Path('OUTPUT_HASIL/fixture') / teacher / f'{sid}_overlay.png'
                    mask_path = self.base / mask_relative
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    mask = Image.new('L', size, 0)
                    ImageDraw.Draw(mask).rectangle((5, 5, 15, 20), fill=255)
                    mask.save(mask_path)
                    Image.new('RGB', size, (20 + i, 70, 80)).save(self.base / overlay_relative)
                    row[f'{key}_mask_path'] = mask_relative.as_posix()
                    detail['teachers'][teacher] = dict(success=True, mask_path=mask_relative.as_posix(),
                        mask_sha256=digest_file(mask_path), overlay_path=overlay_relative.as_posix(),
                        overlay_sha256=digest_file(self.base / overlay_relative))
                rows.append(row)
                samples.append(detail)
        master = self.pool.parent / 'gold_subset_candidates_30.csv'
        master.write_text(csv_text(rows, tuple(rows[0])), encoding='utf-8')
        summary = dict(master_manifest_sha256=digest_file(master), complete_candidate_pairs=30, failures=[])
        (self.pool / 'consolidation_summary.json').write_text(json.dumps(summary), encoding='utf-8')
        (self.pool / 'candidate_provenance.json').write_text(json.dumps(dict(samples=samples, sessions=sessions)), encoding='utf-8')
        (self.base / 'docs').mkdir()
        (self.base / 'docs/GOLD_SUBSET_ANNOTATION_PROTOCOL.md').write_text('Synthetic test protocol only', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_export_byte_identical_candidates_and_empty_manual_directory(self):
        report = export(self.gold, self.pool, self.base)
        self.assertEqual(report['manual_masks_created'], 0)
        self.assertEqual(len(list((self.gold / 'images').iterdir())), 30)
        self.assertEqual(len(list((self.gold / 'candidate_masks/MedSAM').iterdir())), 30)
        self.assertEqual(list((self.gold / 'masks_manual').iterdir()), [])
        result = validate(self.gold, self.base)
        self.assertTrue(result['package_valid'])
        self.assertFalse(result['manual_complete'])
        self.assertEqual(result['manual_masks_present'], 0)
        self.assertEqual(len(result['missing_manual_samples']), 30)
        with self.assertRaises(FileExistsError):
            export(self.gold, self.pool, self.base)

    def test_format_provenance_validation_requires_genuine_annotation_and_review_records(self):
        export(self.gold, self.pool, self.base)
        status_path = self.gold / 'metadata/manual_annotation_status.csv'
        statuses = read_csv(status_path)
        package = read_csv(self.gold / 'metadata/package_manifest.csv')
        for row, info in zip(package, statuses):
            mask = Image.new('L', (int(row['native_width']), int(row['native_height'])), 0)
            ImageDraw.Draw(mask).rectangle((3, 4, 18, 21), fill=255)
            mask.save(self.gold / row['manual_mask_path'])
            info.update(candidate_teacher='FROM_SCRATCH', manual_mask_status='REVIEWED',
                        annotator='human_fixture', reviewer='expert_fixture', notes='Synthetic correction record, not research data')
        status_path.write_text(csv_text(statuses, ANNOTATION_FIELDS), encoding='utf-8')
        self.assertTrue(validate(self.gold, self.base)['reviewed_complete'])
        statuses[0]['reviewer'] = ''
        status_path.write_text(csv_text(statuses, ANNOTATION_FIELDS), encoding='utf-8')
        result = validate(self.gold, self.base)
        self.assertFalse(result['reviewed_complete'])
        self.assertEqual(len(result['expert_review_pending']), 1)

    def test_reject_wrong_manual_geometry_without_resampling_or_repair(self):
        export(self.gold, self.pool, self.base)
        row = read_csv(self.gold / 'metadata/package_manifest.csv')[0]
        manual = self.gold / row['manual_mask_path']
        Image.new('L', (7, 9), 127).save(manual)
        before = digest_file(manual)
        result = validate(self.gold, self.base)
        self.assertFalse(result['manual_complete'])
        self.assertTrue(result['manual_issues'])
        self.assertEqual(digest_file(manual), before)

    def test_block_export_if_candidate_changed_before_any_package_is_created(self):
        row = read_csv(self.pool.parent / 'gold_subset_candidates_30.csv')[0]
        Image.new('L', (43, 31), 128).save(self.base / row['MedSAM_mask_path'])
        with self.assertRaises(ValueError):
            export(self.gold, self.pool, self.base)
        self.assertFalse(self.gold.exists())

    def test_model_geometry_uses_each_native_size(self):
        annotation = dict(x_min=70, y_min=128, x_max=304, y_max=450)
        native = [70, 128, 304, 450]
        med = [70 * 1024 / 623, 256, 304 * 1024 / 623, 900]
        row = dict(native_box=json.dumps(native), model_box=json.dumps(med), model_input_size='[1024, 1024]')
        check_geometry(row, annotation, (512, 623), 'MedSAM', 1024)
        row['model_box'] = json.dumps([v * 2 for v in native])
        with self.assertRaises(ValueError):
            check_geometry(row, annotation, (512, 623), 'MedSAM', 1024)


if __name__ == '__main__':
    unittest.main()
