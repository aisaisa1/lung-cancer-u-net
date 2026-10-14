"""Selection coverage and session isolation; no synthetic fixtures enter research outputs."""

import importlib
from pathlib import Path
import random
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
prepare = importlib.import_module('prepare_phase3_additional_prompts')
from draw_lung_boxes import AnnotationStore, digest_file


class Phase3SelectionTests(unittest.TestCase):
    def pool(self):
        old, candidates = [], []
        for cls in prepare.CLASSES:
            for i in range(5):
                old.append(dict(sample_id=f'{cls.lower()}_old_{i}', source_path=f'{cls}/old_{i}.jpg',
                                **{'class': cls}, image_width=512, image_height=512))
            for i in range(10):
                width, height = 512, 512
                if cls == 'Normal' and i == 8:
                    width, height = 506, 331
                if cls == 'Malignant' and i in (8, 9):
                    width, height = ((511, 404), (623, 512))[i - 8]
                candidates.append(dict(sample_id=f'{cls.lower()}_{i}', source_path=f'{cls}/{i}.jpg',
                                       **{'class': cls}, image_width=width, image_height=height,
                                       pixel_sha256=f'{cls}_pixel_{i}', file_sha256=f'{cls}_file_{i}',
                                       central_laplacian_variance=float(i + 1),
                                       central_bright_fraction=float(i) / 10,
                                       phase1_native_technical_failure=i == 3))
        return old, candidates

    def test_selection_deterministic_under_input_shuffle_and_class_balanced(self):
        old, pool = self.pool()
        chosen = prepare.choose_additional(pool, old, {'retained_pixel'})
        random.Random(25).shuffle(pool)
        reordered = prepare.choose_additional(pool, old, {'retained_pixel'})
        self.assertEqual(chosen, reordered)
        self.assertEqual(prepare.Counter(r['class'] for r in chosen), {'Normal': 5, 'Benign': 5, 'Malignant': 5})
        self.assertTrue({r['source_path'] for r in old}.isdisjoint(r['source_path'] for r in chosen))
        self.assertEqual(len({r['pixel_sha256'] for r in chosen}), 15)

    def test_additional_selection_includes_rare_sizes_low_sharpness_and_prior_failures(self):
        old, pool = self.pool()
        chosen = prepare.choose_additional(pool, old, set())
        sizes = {(r['image_width'], r['image_height']) for r in chosen}
        self.assertTrue({(506, 331), (511, 404), (623, 512)} <= sizes)
        for cls in prepare.CLASSES:
            rows = [r for r in chosen if r['class'] == cls]
            self.assertTrue(any('low_sharpness' in r['selection_reason'] for r in rows))
            self.assertTrue(any(r['phase1_native_technical_failure'] for r in rows))
        self.assertTrue(any('malignant_class_high' in r['selection_reason'] for r in chosen))

    def test_duplicates_and_retained_pixels_are_not_selected(self):
        old, pool = self.pool()
        pool[0]['pixel_sha256'] = 'retained_pixel'
        pool[1]['pixel_sha256'] = pool[2]['pixel_sha256']
        chosen = prepare.choose_additional(pool, old, {'retained_pixel'})
        hashes = [r['pixel_sha256'] for r in chosen]
        self.assertNotIn('retained_pixel', hashes)
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_new_session_cannot_overwrite_original_annotation_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for cls in prepare.CLASSES:
                for i in range(5):
                    relative = Path('DATASET/The IQ-OTHNCCD lung cancer dataset') / cls / f'{i}.png'
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (32, 24), (i * 10, 40, 90)).save(path)
                    samples.append(dict(sample_id=f'{cls.lower()}_{i}', source_path=relative.as_posix(),
                                        **{'class': cls}, image_width=32, image_height=24,
                                        file_sha256=digest_file(path), selection_reason='fixture'))
            old = AnnotationStore(root / 'OUTPUT_HASIL/phase2', samples, root)
            old.set_box(0, 'box_1', (1, 2, 20, 22), 'original_human')
            old.save()
            csv_before, json_before = digest_file(old.csv_path), digest_file(old.json_path)
            new = AnnotationStore(root / 'OUTPUT_HASIL/additional', samples, root)
            self.assertTrue(all(r['visibility_status'] == 'PENDING' for r in new.records))
            self.assertTrue(all(r['x_min'] == '' for r in new.records))
            self.assertEqual(digest_file(old.csv_path), csv_before)
            self.assertEqual(digest_file(old.json_path), json_before)


if __name__ == '__main__':
    unittest.main()
