"""Verify prompt geometry/union and completed annotation validation, not anatomical accuracy."""

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'tools'))
inference = importlib.import_module('phase2_teacher_inference')
from draw_lung_boxes import Viewport


class Phase2InferenceTests(unittest.TestCase):
    def test_medsam_rectangular_box_scales_x_and_y_independently(self):
        box = [170, 144, 398, 445]
        transformed = inference.medsam_box(box, (512, 801))
        self.assertAlmostEqual(transformed[0], 170 * 1024 / 801)
        self.assertEqual(transformed[1], 288)
        self.assertAlmostEqual(transformed[2], 398 * 1024 / 801)
        self.assertEqual(transformed[3], 890)
        self.assertEqual(box, [170, 144, 398, 445])

    def test_union_keeps_independent_masks_and_rejects_mixed_geometry(self):
        a, b = np.zeros((512, 801), bool), np.zeros((512, 801), bool)
        a[100:200, 50:180] = True
        b[100:200, 170:260] = True
        before_a, before_b = a.copy(), b.copy()
        merged = inference.merge_masks([a, b], (512, 801))
        self.assertEqual(merged.sum(), 100 * 210)
        self.assertTrue(np.array_equal(a, before_a))
        self.assertTrue(np.array_equal(b, before_b))
        with self.assertRaises(ValueError):
            inference.merge_masks([a, np.zeros((512, 512), bool)], (512, 801))
        self.assertIsNone(inference.merge_masks([], (512, 801)))

    def test_native_view_does_not_resample_or_force_square(self):
        view = Viewport(801, 512, 250, 200, native=True)
        self.assertEqual((view.display_width, view.display_height), (801, 512))
        self.assertEqual(view.rectangle((182, 156), (410, 457)), (170, 144, 398, 445))
        fit = Viewport(512, 512, 2000, 1400)
        self.assertEqual((fit.display_width, fit.display_height), (512, 512))


if __name__ == '__main__':
    unittest.main()
