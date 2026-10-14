"""Human-only, native-coordinate lung prompt annotation; never runs a teacher."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/phase2_prompts'
BOX_IDS = ('box_1', 'box_2')
VISIBILITY = ('PENDING', 'VISIBLE', 'NOT_VISIBLE')
FIELDS = ('sample_id', 'source_path', 'class', 'image_width', 'image_height',
          'box_id', 'x_min', 'y_min', 'x_max', 'y_max', 'visibility_status', 'reviewer_id', 'notes')
SELECTION_FIELDS = ('sample_id', 'source_path', 'class', 'image_width', 'image_height',
                    'file_sha256', 'selection_reason')
COORDINATES = ('x_min', 'y_min', 'x_max', 'y_max')


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def csv_text(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_rgb(path):
    with Image.open(path) as image:
        image.load()
        # Retain stored pixel coordinates; do not apply EXIF rotation or windowing.
        return image.convert('RGB')


def sample_id(relative_path, class_name):
    return class_name.lower() + '_' + hashlib.sha256(relative_path.encode()).hexdigest()[:12]


def make_selection(pilot_dir):
    """Select images, never boxes: anchors + failed/empty/nonempty technical strata."""
    originals = read_csv(pilot_dir / 'selected_images.csv')
    native = [r for r in read_csv(pilot_dir / 'teacher_feasibility.csv') if r['representation'] == 'native']
    lookup = {}
    for row in native:
        lookup.setdefault(row['relative_path'], []).append(row)
    anchors = {'Normal': 'Normal case (222).jpg', 'Benign': 'Bengin case (120).jpg',
               'Malignant': 'Malignant case (202).jpg'}
    selection = []
    for cls in ('Normal', 'Benign', 'Malignant'):
        candidates = [r for r in originals if r['class_name'] == cls]
        rank = lambda r: hashlib.sha256(f"42|{r['relative_path']}".encode()).hexdigest()
        candidates.sort(key=rank)
        chosen = {}

        def take(rows, number, reason):
            added = 0
            for row in rows:
                relative = row['relative_path']
                if relative in chosen:
                    continue
                chosen[relative] = (row, reason)
                added += 1
                if added >= number or len(chosen) >= 5:
                    break

        take([r for r in candidates if Path(r['relative_path']).name == anchors[cls]],
             1, 'documented_phase1_review_anchor_not_anatomical_grade')
        take([r for r in candidates if any(v['status'] != 'ok' for v in lookup.get(r['relative_path'], []))],
             2, 'phase1_native_technical_failure')
        take([r for r in candidates if any(v['status'] == 'ok' and v['empty_mask'] == 'True'
                                           for v in lookup.get(r['relative_path'], []))],
             1, 'phase1_native_empty_mask')
        if len(chosen) < 5:
            take(candidates, 5 - len(chosen), 'deterministic_phase1_remaining_sample')
        if len(chosen) != 5:
            raise ValueError(f'Cannot select five {cls} images from the Phase-1 pilot')
        for row, reason in sorted(chosen.values(), key=lambda pair: rank(pair[0])):
            if row['source_kind'] != 'least_processed_export_candidate':
                raise ValueError('Only the audited least-processed exports may be selected')
            selection.append(dict(sample_id=sample_id(row['relative_path'], cls),
                                  source_path=(Path('DATASET') / row['relative_path']).as_posix(),
                                  **{'class': cls}, image_width=int(row['width']), image_height=int(row['height']),
                                  file_sha256=row['file_sha256'], selection_reason=reason))
    return selection


def validate_selection(selection, project_root):
    if len(selection) != 15 or Counter(s['class'] for s in selection) != {
            'Normal': 5, 'Benign': 5, 'Malignant': 5}:
        raise ValueError('Selection must contain exactly 15 images, five per class')
    ids, source_paths = set(), set()
    canonical = (project_root / 'DATASET/The IQ-OTHNCCD lung cancer dataset').resolve()
    for row in selection:
        if set(row) != set(SELECTION_FIELDS):
            raise ValueError('Unexpected selection manifest schema')
        if row['sample_id'] in ids or row['source_path'] in source_paths:
            raise ValueError('Duplicate sample ID or source in selection')
        ids.add(row['sample_id'])
        source_paths.add(row['source_path'])
        if not row['sample_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in row['sample_id']):
            raise ValueError('Invalid sample ID')
        path = (project_root / row['source_path']).resolve()
        if canonical not in path.parents:
            raise ValueError(f'Source is outside the canonical export tree: {path}')
        if digest_file(path) != row['file_sha256']:
            raise ValueError(f'Source hash changed: {path}')
        with Image.open(path) as image:
            if image.size != (int(row['image_width']), int(row['image_height'])):
                raise ValueError(f'Source native dimensions changed: {path}')
        row['image_width'], row['image_height'] = int(row['image_width']), int(row['image_height'])


def blank_record(sample, box_id):
    return dict(sample_id=sample['sample_id'], source_path=sample['source_path'],
                **{'class': sample['class']}, image_width=sample['image_width'],
                image_height=sample['image_height'], box_id=box_id,
                x_min='', y_min='', x_max='', y_max='', visibility_status='PENDING',
                reviewer_id='', notes='')


def validate_record(row, sample):
    if set(row) != set(FIELDS):
        raise ValueError('Unexpected annotation record schema')
    for key in ('sample_id', 'source_path', 'class', 'image_width', 'image_height'):
        if row[key] != sample[key]:
            raise ValueError(f'Annotation does not match the fixed sample: {key}')
    if row['box_id'] not in BOX_IDS or row['visibility_status'] not in VISIBILITY:
        raise ValueError('Unknown box ID or visibility status')
    if not isinstance(row['reviewer_id'], str) or not isinstance(row['notes'], str):
        raise ValueError('Reviewer and notes must be text')
    if row['visibility_status'] == 'VISIBLE':
        coords = [row[k] for k in COORDINATES]
        if any(type(v) is not int for v in coords):
            raise ValueError('Box coordinates must be native integer pixel edges')
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= sample['image_width'] and 0 <= y0 < y1 <= sample['image_height']):
            raise ValueError('Box must have positive area within the native image')
    elif any(row[k] != '' for k in COORDINATES):
        raise ValueError('PENDING/NOT_VISIBLE records must have blank coordinates')
    if row['visibility_status'] != 'PENDING' and not row['reviewer_id'].strip():
        raise ValueError('Every drawn or NOT_VISIBLE region requires a human reviewer ID')


class Viewport:
    """Aspect-preserving canvas geometry using actual rounded display dimensions."""

    def __init__(self, native_width, native_height, canvas_width, canvas_height, native=False):
        self.width, self.height = native_width, native_height
        scale = min(1.0, max(canvas_width - 24, 1) / native_width,
                    max(canvas_height - 24, 1) / native_height)
        if native:
            scale = 1.0
        self.display_width = max(1, round(native_width * scale))
        self.display_height = max(1, round(native_height * scale))
        self.left = 12 if native else (canvas_width - self.display_width) / 2
        self.top = 12 if native else (canvas_height - self.display_height) / 2

    def contains(self, x, y):
        return (self.left <= x <= self.left + self.display_width and
                self.top <= y <= self.top + self.display_height)

    def native_point(self, x, y):
        return (min(self.width, max(0.0, (x - self.left) * self.width / self.display_width)),
                min(self.height, max(0.0, (y - self.top) * self.height / self.display_height)))

    def canvas_box(self, box):
        x0, y0, x1, y1 = box
        return (self.left + x0 * self.display_width / self.width,
                self.top + y0 * self.display_height / self.height,
                self.left + x1 * self.display_width / self.width,
                self.top + y1 * self.display_height / self.height)

    def rectangle(self, start, end):
        a, b = self.native_point(*start), self.native_point(*end)
        # Native edge convention: [x_min, x_max), [y_min, y_max).
        return (max(0, math.floor(min(a[0], b[0]) + 1e-7)),
                max(0, math.floor(min(a[1], b[1]) + 1e-7)),
                min(self.width, math.ceil(max(a[0], b[0]) - 1e-7)),
                min(self.height, math.ceil(max(a[1], b[1]) - 1e-7)))


class AnnotationStore:
    def __init__(self, output, selection, project_root=PROJECT_ROOT, repair_csv=False):
        self.output, self.selection = Path(output).resolve(), selection
        self.project_root = Path(project_root).resolve()
        dataset = (self.project_root / 'DATASET').resolve()
        if self.output == dataset or dataset in self.output.parents or self.output in dataset.parents:
            raise ValueError('Annotation output must be separate from the original dataset tree')
        validate_selection(selection, self.project_root)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'previews').mkdir(exist_ok=True)
        self.csv_path = self.output / 'manual_lung_boxes.csv'
        self.json_path = self.output / 'manual_lung_boxes.json'
        self.fingerprint = hashlib.sha256(csv_text(selection, SELECTION_FIELDS).encode()).hexdigest()
        self.session = {'sample_index': 0, 'reviewer_id': ''}
        self.by_sample = {s['sample_id']: s for s in selection}
        self.records = [blank_record(s, box) for s in selection for box in BOX_IDS]
        self.last_saved = None
        if self.json_path.exists():
            with self.json_path.open(encoding='utf-8') as handle:
                data = json.load(handle)
            if data.get('schema_version') != 1 or data.get('selection_sha256') != self.fingerprint:
                raise ValueError('Saved annotation selection/schema differs; do not mix pilot samples')
            self.records, self.session = data['records'], data['session']
            self.validate()
            text = csv_text(self.records, FIELDS)
            if not self.csv_path.exists() or self.csv_path.read_text(encoding='utf-8') != text:
                if not repair_csv:
                    raise ValueError('CSV/JSON differ. Preserve any manual CSV edits; use --repair-csv '
                                     'only to regenerate the CSV mirror from the saved JSON.')
                atomic_text(self.csv_path, text)
            self.last_saved = [dict(r) for r in self.records]
        elif self.csv_path.exists():
            raise ValueError('CSV exists without authoritative JSON; refusing to overwrite user annotations')
        else:
            self.save()
            for sample in selection:
                self.preview(sample['sample_id'])

    def validate(self):
        expected = {(s['sample_id'], b) for s in self.selection for b in BOX_IDS}
        actual = {(r['sample_id'], r['box_id']) for r in self.records}
        if actual != expected or len(self.records) != len(expected):
            raise ValueError('Exactly two annotation slots per selected sample are required')
        for record in self.records:
            validate_record(record, self.by_sample[record['sample_id']])
        if not (type(self.session.get('sample_index')) is int and
                0 <= self.session['sample_index'] < len(self.selection)):
            raise ValueError('Invalid resume index')

    def record(self, index, box_id):
        sample = self.selection[index]['sample_id']
        return next(r for r in self.records if r['sample_id'] == sample and r['box_id'] == box_id)

    def set_box(self, index, box_id, box, reviewer, notes=''):
        row = dict(self.record(index, box_id))
        row.update(dict(zip(COORDINATES, box)), visibility_status='VISIBLE',
                   reviewer_id=reviewer.strip(), notes=notes)
        validate_record(row, self.selection[index])
        self.record(index, box_id).update(row)

    def set_not_visible(self, index, box_id, reviewer, notes=''):
        if not reviewer.strip():
            raise ValueError('Enter your reviewer ID before marking NOT_VISIBLE')
        row = self.record(index, box_id)
        row.update({k: '' for k in COORDINATES}, visibility_status='NOT_VISIBLE',
                   reviewer_id=reviewer.strip(), notes=notes)

    def delete(self, index, box_id):
        self.record(index, box_id).update(blank_record(self.selection[index], box_id))

    def save(self):
        self.validate()
        for sample in self.selection:
            if digest_file(self.project_root / sample['source_path']) != sample['file_sha256']:
                raise ValueError(f"Source changed: {sample['source_path']}")
        now = datetime.now(timezone.utc).isoformat()
        data = dict(schema_version=1, selection_sha256=self.fingerprint,
                    coordinate_convention='Native stored pixel edges; x_max/y_max exclusive; origin top-left',
                    orientation='Unverified DICOM orientation; box_1/box_2 are slots, not anatomical sides',
                    prompt_type='human_drawn_regions_for_later_whole_lung_union',
                    updated_at_utc=now, session=self.session, records=self.records)
        # JSON is authoritative, CSV is an export; both files are individually atomic.
        atomic_text(self.json_path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        atomic_text(self.csv_path, csv_text(self.records, FIELDS))
        if self.last_saved is not None and self.last_saved != self.records:
            changes = [{'before': before, 'after': after} for before, after in zip(self.last_saved, self.records)
                       if before != after]
            with (self.output / 'annotation_history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'saved_at_utc': now, 'changes': changes}, ensure_ascii=False) + '\n')
        self.last_saved = [dict(r) for r in self.records]

    def preview(self, image_id):
        sample = self.by_sample[image_id]
        image = load_rgb(self.project_root / sample['source_path'])
        draw = ImageDraw.Draw(image)
        colors = {'box_1': '#ffbf00', 'box_2': '#00bfff'}
        statuses = []
        for row in self.records:
            if row['sample_id'] != image_id:
                continue
            statuses.append(f"{row['box_id']}: {row['visibility_status']}")
            if row['visibility_status'] == 'VISIBLE':
                x0, y0, x1, y1 = [row[k] for k in COORDINATES]
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=colors[row['box_id']], width=2)
                draw.text((x0 + 3, y0 + 3), row['box_id'], fill=colors[row['box_id']])
        label = ' | '.join(statuses)
        draw.rectangle((0, 0, min(image.width, 420), 19), fill='black')
        draw.text((3, 3), label, fill='white')
        path = self.output / 'previews' / f'{image_id}.png'
        temp = path.with_suffix('.png.tmp')
        image.save(temp, format='PNG')
        os.replace(temp, path)


def prepare(output, pilot_dir, selection_path=None, repair_csv=False):
    output = Path(output).resolve()
    dataset = (PROJECT_ROOT / 'DATASET').resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError('Output must not overlap the dataset tree')
    manifest = Path(selection_path).resolve() if selection_path else output / 'phase2_selection.csv'
    if manifest.exists():
        selection = read_csv(manifest)
    else:
        if selection_path:
            raise FileNotFoundError(f'Specified selection manifest not found: {manifest}')
        if (output / 'manual_lung_boxes.json').exists() or (output / 'manual_lung_boxes.csv').exists():
            raise ValueError('Annotations exist without their selection manifest; do not regenerate selection')
        selection = make_selection(Path(pilot_dir))
        validate_selection(selection, PROJECT_ROOT)
        atomic_text(manifest, csv_text(selection, SELECTION_FIELDS))
    store = AnnotationStore(output, selection, repair_csv=repair_csv)
    return store


class BoxGUI:
    def __init__(self, root, store, reviewer=''):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
        self.tk, self.ttk, self.ImageTk = tk, ttk, ImageTk
        self.root, self.store = root, store
        self.index = store.session['sample_index']
        self.active = tk.StringVar(value='box_1')
        self.native_view = tk.BooleanVar(value=True)
        self.reviewer = tk.StringVar(value=reviewer or store.session.get('reviewer_id', ''))
        self.heading, self.info, self.status = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.dirty = False
        self.drag_start, self.viewport, self.photo = None, None, None
        root.title(f'Manual lung boxes — {store.output.name} — no anatomical left/right assignment')
        root.geometry('1150x820')
        root.minsize(820, 650)
        top = ttk.Frame(root, padding=8)
        top.pack(fill='x')
        ttk.Label(top, textvariable=self.heading, font=('Segoe UI', 11, 'bold')).pack(anchor='w')
        ttk.Label(top, text=f'Annotation session: {store.output.name}').pack(anchor='w')
        ttk.Label(top, text='Draw each visible lung region in one slot; the two will later be merged. '
                            'Slots are not anatomical sides. No missing region is inferred.').pack(anchor='w')
        bar = ttk.Frame(root, padding=(8, 0))
        bar.pack(fill='x')
        ttk.Label(bar, text='Reviewer ID:').pack(side='left')
        ttk.Entry(bar, textvariable=self.reviewer, width=24).pack(side='left', padx=6)
        for box in BOX_IDS:
            ttk.Radiobutton(bar, text=box, variable=self.active, value=box,
                            command=self.switch_box).pack(side='left', padx=5)
        ttk.Button(bar, text='Redraw selected', command=self.redraw).pack(side='left', padx=4)
        ttk.Button(bar, text='Delete selected', command=self.delete).pack(side='left', padx=4)
        ttk.Button(bar, text='Mark NOT_VISIBLE', command=self.not_visible).pack(side='left', padx=4)
        ttk.Checkbutton(bar, text='Native 1:1', variable=self.native_view, command=self.render).pack(side='left', padx=4)
        ttk.Label(root, textvariable=self.info, padding=(8, 4)).pack(fill='x')
        canvas_frame = ttk.Frame(root)
        canvas_frame.pack(fill='both', expand=True, padx=8)
        self.canvas = tk.Canvas(canvas_frame, background='#202020', highlightthickness=0)
        vertical = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.canvas.yview)
        horizontal = ttk.Scrollbar(canvas_frame, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind('<Configure>', lambda event: self.render())
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        notes_bar = ttk.Frame(root, padding=8)
        notes_bar.pack(fill='x')
        ttk.Label(notes_bar, text='Notes for selected slot:').pack(anchor='w')
        self.notes = tk.Text(notes_bar, height=3, wrap='word')
        self.notes.pack(fill='x')
        self.notes.bind('<<Modified>>', self.notes_modified)
        bottom = ttk.Frame(root, padding=(8, 0, 8, 8))
        bottom.pack(fill='x')
        ttk.Button(bottom, text='Previous', command=lambda: self.navigate(-1)).pack(side='left')
        ttk.Button(bottom, text='Next', command=lambda: self.navigate(1)).pack(side='left', padx=6)
        ttk.Button(bottom, text='Save (Ctrl+S)', command=self.save).pack(side='left', padx=6)
        ttk.Button(bottom, text='Quit', command=self.quit).pack(side='left', padx=6)
        ttk.Label(bottom, textvariable=self.status).pack(side='left', padx=10)
        root.bind('<Control-s>', lambda event: self.save())
        root.protocol('WM_DELETE_WINDOW', self.quit)
        self.previous_active = 'box_1'
        self.load_sample()

    def note_text(self):
        return self.notes.get('1.0', 'end-1c')

    def collect_notes(self):
        row = self.store.record(self.index, self.previous_active)
        text = self.note_text()
        if text != row['notes']:
            if row['visibility_status'] != 'PENDING' and not self.reviewer.get().strip():
                raise ValueError('Enter a reviewer ID to update a reviewed region')
            row['notes'] = text
            if row['visibility_status'] != 'PENDING':
                row['reviewer_id'] = self.reviewer.get().strip()
            self.dirty = True

    def notes_modified(self, event=None):
        if self.notes.edit_modified():
            self.dirty = True
            self.status.set('Unsaved changes')
            self.notes.edit_modified(False)

    def set_notes(self):
        row = self.store.record(self.index, self.active.get())
        self.notes.delete('1.0', 'end')
        self.notes.insert('1.0', row['notes'])
        self.notes.edit_modified(False)

    def load_sample(self):
        sample = self.store.selection[self.index]
        self.image = load_rgb(self.store.project_root / sample['source_path'])
        self.heading.set(f"{self.index + 1}/{len(self.store.selection)} | {sample['class']} | {Path(sample['source_path']).name} | "
                         f"native {sample['image_width']}x{sample['image_height']} | {sample['sample_id']}")
        self.previous_active = self.active.get()
        self.set_notes()
        self.dirty = False
        self.status.set('Loaded saved state; both slots may remain PENDING')
        self.render()

    def render(self):
        if not hasattr(self, 'image'):
            return
        sample = self.store.selection[self.index]
        self.viewport = Viewport(sample['image_width'], sample['image_height'],
                                 max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1),
                                 native=self.native_view.get())
        display_size = (self.viewport.display_width, self.viewport.display_height)
        view = self.image.copy() if display_size == self.image.size else self.image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = self.ImageTk.PhotoImage(view)
        self.canvas.delete('all')
        self.canvas.create_image(self.viewport.left, self.viewport.top, image=self.photo, anchor='nw')
        self.canvas.configure(scrollregion=(0, 0, max(self.canvas.winfo_width(), self.viewport.left + display_size[0] + 12),
                                            max(self.canvas.winfo_height(), self.viewport.top + display_size[1] + 12)))
        summaries = []
        for box_id, color in (('box_1', '#ffbf00'), ('box_2', '#00bfff')):
            row = self.store.record(self.index, box_id)
            label = f"{box_id}: {row['visibility_status']}"
            if row['visibility_status'] == 'VISIBLE':
                coords = tuple(row[k] for k in COORDINATES)
                label += ' ' + str(coords)
                display = self.viewport.canvas_box(coords)
                self.canvas.create_rectangle(*display, outline=color, width=3)
                self.canvas.create_text(display[0] + 4, display[1] + 4, text=box_id, fill=color, anchor='nw')
            summaries.append(label)
        self.info.set(' | '.join(summaries) + f' | Display {display_size[0]}x{display_size[1]}'
                      + (' (1:1, no resampling)' if self.native_view.get() else ' (fit, never upscaled)'))
        self.drag_start = None

    def alert(self, error):
        from tkinter import messagebox
        messagebox.showerror('Annotation not saved', str(error), parent=self.root)

    def switch_box(self):
        try:
            self.collect_notes()
        except ValueError as error:
            self.active.set(self.previous_active)
            self.alert(error)
            return
        self.previous_active = self.active.get()
        self.set_notes()
        self.render()

    def press(self, event):
        if not self.reviewer.get().strip():
            self.alert('Enter your reviewer ID before drawing a human prompt.')
            return
        point = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.viewport and self.viewport.contains(*point):
            self.drag_start = point

    def motion(self, event):
        if self.drag_start is None:
            return
        box = self.viewport.rectangle(self.drag_start, (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)))
        self.canvas.delete('drag')
        self.canvas.create_rectangle(*self.viewport.canvas_box(box), outline='white', width=2, dash=(4, 3), tags='drag')

    def release(self, event):
        if self.drag_start is None:
            return
        start = self.drag_start
        self.drag_start = None
        try:
            end = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            if abs(end[0] - start[0]) < 2 or abs(end[1] - start[1]) < 2:
                self.canvas.delete('drag')
                return
            box = self.viewport.rectangle(start, end)
            self.store.set_box(self.index, self.active.get(), box, self.reviewer.get(), self.note_text())
            self.dirty = True
            self.render()
            self.status.set('Unsaved box; draw the other slot or mark NOT_VISIBLE explicitly')
        except ValueError as error:
            self.alert(error)
            self.render()

    def redraw(self):
        self.status.set('Click-drag on the image to replace the selected slot; old box stays until release')
        self.canvas.focus_set()

    def delete(self):
        self.store.delete(self.index, self.active.get())
        self.set_notes()
        self.dirty = True
        self.render()
        self.status.set('Deleted selected slot; now PENDING. Save to persist.')

    def not_visible(self):
        try:
            self.store.set_not_visible(self.index, self.active.get(), self.reviewer.get(), self.note_text())
            self.dirty = True
            self.render()
            self.status.set('NOT_VISIBLE set by human; coordinates cleared. Save to persist.')
        except ValueError as error:
            self.alert(error)

    def save(self):
        try:
            self.collect_notes()
            self.store.session = dict(sample_index=self.index, reviewer_id=self.reviewer.get().strip())
            self.store.save()
            self.store.preview(self.store.selection[self.index]['sample_id'])
            self.dirty = False
            self.status.set('Saved CSV, JSON, and native-resolution preview')
            return True
        except (ValueError, OSError) as error:
            self.alert(error)
            return False

    def may_leave(self):
        from tkinter import messagebox
        try:
            self.collect_notes()
        except ValueError as error:
            self.alert(error)
            return False
        if not self.dirty:
            return True
        decision = messagebox.askyesnocancel('Unsaved annotations', 'Save this image before leaving?', parent=self.root)
        if decision is None:
            return False
        if decision:
            return self.save()
        # Discard only this sample's unsaved edits, retaining other saved samples.
        sample_id_value = self.store.selection[self.index]['sample_id']
        for row in self.store.records:
            if row['sample_id'] == sample_id_value:
                saved = next(r for r in self.store.last_saved if r['sample_id'] == sample_id_value
                             and r['box_id'] == row['box_id'])
                row.update(saved)
        self.dirty = False
        return True

    def navigate(self, step):
        target = max(0, min(len(self.store.selection) - 1, self.index + step))
        if target == self.index or not self.may_leave():
            return
        self.index = target
        self.store.session = dict(sample_index=self.index, reviewer_id=self.reviewer.get().strip())
        try:
            self.store.save()
        except (ValueError, OSError) as error:
            self.alert(error)
            return
        self.load_sample()

    def quit(self):
        if not self.may_leave():
            return
        self.store.session = dict(sample_index=self.index, reviewer_id=self.reviewer.get().strip())
        try:
            self.store.save()
        except (ValueError, OSError) as error:
            self.alert(error)
            return
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--pilot-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/teacher_pilot')
    parser.add_argument('--selection', type=Path, help='Reuse an existing manifest with the documented schema')
    parser.add_argument('--reviewer-id', default='', help='Optional; can also be entered in the GUI')
    parser.add_argument('--prepare-only', action='store_true', help='Create/reload the manifest and PENDING records, no GUI')
    parser.add_argument('--smoke', action='store_true', help='Validate files and instantiate a hidden GUI, without annotating')
    parser.add_argument('--repair-csv', action='store_true', help='Explicitly regenerate only CSV from authoritative JSON')
    args = parser.parse_args()
    try:
        store = prepare(args.output_dir, args.pilot_dir, args.selection, args.repair_csv)
        print('Fixed selection:', len(store.selection), 'images; classes:', dict(Counter(s['class'] for s in store.selection)))
        print('Annotation slots:', len(store.records), '; status:', dict(Counter(r['visibility_status'] for r in store.records)))
        print('CSV:', store.csv_path)
        if args.prepare_only:
            return
        import tkinter as tk
        root = tk.Tk()
        if args.smoke:
            root.withdraw()
        app = BoxGUI(root, store, args.reviewer_id)
        if args.smoke:
            root.update_idletasks()
            app.canvas.configure(width=900, height=500)
            root.update()
            app.render()
            if len(app.canvas.find_all()) < 1:
                raise RuntimeError('GUI did not render the native image')
            root.destroy()
            print('Smoke passed: source hashes and native dimensions valid, saved state loads, Tk image renders; no boxes inferred.')
            return
        root.mainloop()
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f'Annotation tool: {error}\n')


if __name__ == '__main__':
    main()
