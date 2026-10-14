"""Annotation geometry and persistence tests; fixtures are not research annotations."""

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools/draw_lung_boxes.py'
spec = importlib.util.spec_from_file_location('draw_lung_boxes', MODULE_PATH)
boxes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boxes)


class BoxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = []
        for cls in ('Normal', 'Benign', 'Malignant'):
            for i in range(5):
                relative = Path('DATASET/The IQ-OTHNCCD lung cancer dataset') / f'{cls} cases' / f'{i}.png'
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                width, height = (801, 512) if cls == 'Normal' and i == 0 else (61, 43)
                Image.new('RGB', (width, height), (30 + i, 60, 90)).save(path)
                self.selection.append(dict(sample_id=f'{cls.lower()}_{i}', source_path=relative.as_posix(),
                                           **{'class': cls}, image_width=width, image_height=height,
                                           file_sha256=boxes.digest_file(path), selection_reason='test_fixture'))
        self.output = self.root / 'OUTPUT_HASIL/prompts'

    def tearDown(self):
        self.temporary.cleanup()

    def store(self, **kwargs):
        return boxes.AnnotationStore(self.output, self.selection, self.root, **kwargs)

    def test_initialization_is_pending_with_blank_boxes_and_native_previews(self):
        store = self.store()
        self.assertEqual(len(store.records), 30)
        self.assertTrue(all(r['visibility_status'] == 'PENDING' for r in store.records))
        self.assertTrue(all(all(r[k] == '' for k in boxes.COORDINATES) for r in store.records))
        with Image.open(self.output / 'previews/normal_0.png') as image:
            self.assertEqual(image.size, (801, 512))

    def test_native_coordinates_redraw_not_visible_delete_and_resume(self):
        store = self.store()
        store.set_box(0, 'box_1', (13, 17, 600, 490), 'human_1', 'first drawing')
        store.set_box(0, 'box_1', (20, 30, 590, 500), 'human_1', 'redraw')
        store.set_not_visible(0, 'box_2', 'human_1', 'region not visible')
        store.session = {'sample_index': 4, 'reviewer_id': 'human_1'}
        store.save()
        store.preview('normal_0')
        resumed = self.store()
        self.assertEqual(resumed.session['sample_index'], 4)
        self.assertEqual(resumed.record(0, 'box_1')['x_max'], 590)
        self.assertEqual(resumed.record(0, 'box_2')['visibility_status'], 'NOT_VISIBLE')
        self.assertEqual(resumed.record(0, 'box_2')['x_min'], '')
        resumed.delete(0, 'box_1')
        resumed.save()
        self.assertEqual(self.store().record(0, 'box_1')['visibility_status'], 'PENDING')
        for sample in self.selection:
            self.assertEqual(boxes.digest_file(self.root / sample['source_path']), sample['file_sha256'])
        with resumed.csv_path.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(tuple(rows[0]), boxes.FIELDS)

    def test_reverse_drag_clamping_and_non_square_native_mapping(self):
        view = boxes.Viewport(801, 512, 1053, 721)
        expected = (13, 29, 800, 510)
        display = view.canvas_box(expected)
        self.assertEqual(view.rectangle(display[:2], display[2:]), expected)
        self.assertEqual(view.rectangle(display[2:], display[:2]), expected)
        self.assertEqual(view.rectangle((-1000, -1000), (10000, 10000)), (0, 0, 801, 512))
        self.assertFalse(view.contains(-100, -100))

    def test_rejects_invalid_boxes_and_requires_human_identity(self):
        store = self.store()
        for box in ((1, 1, 1, 10), (-1, 1, 10, 10), (1, 1, 802, 10), (1.5, 1, 10, 10)):
            with self.assertRaises(ValueError):
                store.set_box(0, 'box_1', box, 'human_1')
        with self.assertRaises(ValueError):
            store.set_box(0, 'box_1', (1, 1, 10, 10), '')
        with self.assertRaises(ValueError):
            store.set_not_visible(0, 'box_2', '')
        self.assertTrue(all(r['visibility_status'] == 'PENDING' for r in store.records))

    def test_guard_dataset_and_changed_manifest_and_csv_edits(self):
        with self.assertRaises(ValueError):
            boxes.AnnotationStore(self.root / 'DATASET/annotations', self.selection, self.root)
        store = self.store()
        changed = [dict(s) for s in self.selection]
        changed[0]['selection_reason'] = 'changed'
        with self.assertRaises(ValueError):
            boxes.AnnotationStore(self.output, changed, self.root)
        store.csv_path.write_text('user edit\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.store()
        repaired = self.store(repair_csv=True)
        self.assertEqual(len(boxes.read_csv(repaired.csv_path)), 30)

    def test_gui_click_drag_save_navigation_and_quit(self):
        import tkinter as tk
        store = self.store()
        root = tk.Tk()
        root.withdraw()
        try:
            app = boxes.BoxGUI(root, store, reviewer='human_test_fixture')
            root.update_idletasks()
            app.viewport = boxes.Viewport(801, 512, 900, 600, native=True)
            display = app.viewport.canvas_box((40, 50, 380, 440))
            app.press(SimpleNamespace(x=display[0], y=display[1]))
            app.motion(SimpleNamespace(x=display[2], y=display[3]))
            app.release(SimpleNamespace(x=display[2], y=display[3]))
            self.assertEqual(tuple(store.record(0, 'box_1')[k] for k in boxes.COORDINATES), (40, 50, 380, 440))
            app.active.set('box_2')
            app.switch_box()
            app.not_visible()
            self.assertTrue(app.save())
            app.navigate(1)
            self.assertEqual(app.index, 1)
            app.navigate(-1)
            self.assertEqual(app.index, 0)
            self.assertEqual(store.record(0, 'box_2')['visibility_status'], 'NOT_VISIBLE')
            app.delete()
            with patch('tkinter.messagebox.askyesnocancel', return_value=False):
                app.navigate(1)
            # Discard restores the prior saved NOT_VISIBLE instead of inferring a box.
            self.assertEqual(store.record(0, 'box_2')['visibility_status'], 'NOT_VISIBLE')
            app.quit()
            root = None
            resumed = self.store()
            self.assertEqual(resumed.session['sample_index'], 1)
            self.assertEqual(resumed.record(0, 'box_1')['reviewer_id'], 'human_test_fixture')
        finally:
            if root is not None:
                root.destroy()


if __name__ == '__main__':
    unittest.main()
