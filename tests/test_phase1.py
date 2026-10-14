"""Meaningful checks for agreement math, geometry, empty masks and safe IO."""

import importlib
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
pilot = importlib.import_module('01_teacher_feasibility')
audit = importlib.import_module('00_dataset_audit')
from phase1_common import prepare_output


class Phase1Tests(unittest.TestCase):
    def test_agreement_known_overlap_and_empty_masks(self):
        a = np.array([[1, 1], [0, 0]], dtype=bool)
        b = np.array([[0, 1], [1, 0]], dtype=bool)
        self.assertEqual(pilot.agreement(a, b)['dice'], 0.5)
        self.assertAlmostEqual(pilot.agreement(a, b)['iou'], 1 / 3)
        empty = np.zeros_like(a)
        self.assertTrue(pilot.agreement(empty, empty)['both_empty'])
        self.assertEqual(pilot.agreement(a, empty)['dice'], 0)
        with self.assertRaises(ValueError):
            pilot.agreement(a, np.zeros((3, 2)))

    def test_qc_empty_centroid_and_eight_connectivity(self):
        qc = pilot.technical_qc(np.zeros((4, 4), dtype=bool))
        self.assertTrue(qc['empty_mask'])
        self.assertIsNone(qc['centroid_x'])
        self.assertIsNone(qc['fragmentation'])
        diagonal = np.eye(3, dtype=bool)
        qc = pilot.technical_qc(diagonal)
        self.assertEqual(qc['connected_components'], 1)
        self.assertEqual(qc['centroid_x'], 1)
        self.assertAlmostEqual(qc['border_touch_fraction'], 2 / 3)

    def test_representations_preserve_rectangular_canvas_and_source(self):
        rng = np.random.default_rng(17)
        source = rng.integers(0, 256, size=(31, 57, 3), dtype=np.uint8)
        before = source.copy()
        views = pilot.representations(source)
        self.assertEqual(tuple(views), pilot.REPRESENTATIONS)
        self.assertTrue(np.array_equal(source, before))
        self.assertTrue(np.array_equal(views['native'], before))
        for view in views.values():
            self.assertEqual(view.shape, source.shape)
            self.assertEqual(view.dtype, np.uint8)

    def test_constant_linear_normalization_no_nan(self):
        source = np.full((17, 29, 3), 127, dtype=np.uint8)
        self.assertTrue(np.array_equal(pilot.representations(source)['linear'], source))

    def test_audit_does_not_assign_numbered_cases_as_patient_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / 'DATASET'
            for cls in ('Normal cases', 'Bengin cases', 'Malignant cases'):
                directory = root / 'The IQ-OTHNCCD lung cancer dataset' / cls
                directory.mkdir(parents=True)
                Image.fromarray(np.full((13, 21, 3), 80, dtype=np.uint8)).save(directory / 'Normal case (1).png')
            output = prepare_output(base / 'audit', root, 'dataset_audit.json')
            report = audit.audit(root, output)
            self.assertFalse(report['grouping']['safely_inferred'])
            self.assertEqual(report['grouping']['verified_case_count'], 0)
            self.assertEqual(report['duplicate_group_counts']['identical_decoded_pixels'], 1)
            self.assertEqual(report['canonical_teacher_source'], 'The IQ-OTHNCCD lung cancer dataset')
            with self.assertRaises(ValueError):
                prepare_output(root / 'outputs', root, 'anything.json')
            with self.assertRaises(FileExistsError):
                prepare_output(output, root, 'dataset_audit.json')

    def test_one_teacher_failure_keeps_other_outputs_and_failure_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / 'DATASET'
            for i, cls in enumerate(('Normal cases', 'Bengin cases', 'Malignant cases')):
                directory = root / 'The IQ-OTHNCCD lung cancer dataset' / cls
                directory.mkdir(parents=True)
                Image.fromarray(np.full((19, 27, 3), 40 + i * 30, dtype=np.uint8)).save(directory / 'slice.png')
            audit_dir = prepare_output(base / 'audit', root, 'dataset_audit.json')
            audit.audit(root, audit_dir)
            output = prepare_output(base / 'pilot', root, 'teacher_summary.json')
            args = SimpleNamespace(threads=1, seed=42, audit_dir=audit_dir, samples_per_class=1,
                                   smoke=True, resume=False, no_postprocessing=False, batch_size=1)

            def load(name, _):
                if name == 'R231CovidWeb':
                    raise RuntimeError('intentional unavailable teacher in test')
                return object(), {'device': 'cpu', 'test_fixture': True}

            def prediction(_, rgb):
                labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
                labels[3:10, 4:12] = 1
                return labels, 1

            with patch.object(pilot, 'load_teacher', side_effect=load), \
                    patch.object(pilot, 'predict', side_effect=prediction), \
                    patch.object(pilot, 'inspect_environment', return_value={'test_fixture': True}):
                pilot.run(args, output)
            with (output / 'teacher_summary.json').open(encoding='utf-8') as handle:
                summary = json.load(handle)
            self.assertEqual(summary['teachers']['R231']['native_successes'], 3)
            self.assertEqual(summary['teachers']['R231CovidWeb']['native_failures'], 3)
            self.assertEqual(summary['teachers']['R231CovidWeb']['all_representation_failures'], 18)
            self.assertEqual(summary['cross_teacher_agreement']['n'], 0)
            self.assertFalse(summary['student_training_allowed'])
            with (output / 'teacher_feasibility.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 36)
            failed = [r for r in rows if r['status'] == 'failed']
            self.assertEqual(len(failed), 18)
            self.assertTrue(all(not r['mask_path'] and r['error'] for r in failed))
            self.assertEqual(len(list((output / 'contact_sheets').glob('*_teachers.png'))), 3)


if __name__ == '__main__':
    unittest.main()
