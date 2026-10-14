# Manual lung-box workflow — Phase-2 handoff

This stage prepares human prompts only. It does not run SAM-Med2D, MedSAM,
LungMask, Otsu, a student model, classification, or Grad-CAM.

## Launch (PowerShell, project root)

```powershell
& ".\budi\Scripts\python.exe" tools/draw_lung_boxes.py
```

Enter your reviewer ID in the GUI before drawing or marking a region NOT_VISIBLE.
Alternatively:

```powershell
& ".\budi\Scripts\python.exe" tools/draw_lung_boxes.py --reviewer-id "YOUR_REVIEWER_ID"
```

The existing Python environment already has Tkinter and Pillow. No package
installation, environment downgrade, or checkpoint download is needed for the GUI.

## Fixed sample selection

No Phase-2 manifest existed at this handoff. The tool initializes and freezes
`OUTPUT_HASIL/segmentation/phase2_prompts/phase2_selection.csv` from the existing
45-image Phase-1 pilot. There are **15 native/least-processed images: five Normal,
five Benign, five Malignant**. Selection is reproducible: seed-42 path-hash rank,
documented review anchors, up to two native technical failures per class, an
empty-output case where available, then remaining ranked samples.

The documented `Normal case (222)`, `Bengin case (120)`, and rectangular
`Malignant case (202)` are included. These selection strata are technical/image
sampling reasons, not automatically assigned anatomical review grades.

Once created, the saved manifest is reused without resampling. File SHA256s and
stored native dimensions are checked. Source images come only from
`DATASET/The IQ-OTHNCCD lung cancer dataset/`, never derived Gamma/CLAHE or crops.
The manifest does not establish patient identity or a patient-disjoint split.

## Mouse workflow

1. Select **box_1**. Click-drag a rectangle around one visible lung region.
2. Select **box_2** and draw the other region. The two regions will later be merged
   into the whole-lung field. Do not exclude dense pathological regions merely
   because they are bright; include their extent when drawing the prompt.
3. A slot/region that is not visible must be explicitly marked **NOT_VISIBLE**.
   If uncertain or not finished, leave it **PENDING** and describe the concern in
   Notes. The program never generates the missing region.
4. **Redraw selected**: click-drag again to replace that slot. The previous box
   remains until a valid replacement is released.
5. **Delete selected**: clears that slot and restores PENDING with blank coordinates.
6. **Save** or **Ctrl+S** saves both slots, notes, CSV, JSON, and the native-size preview.
7. **Previous/Next** navigates the fixed sample list. Unsaved edits offer Save,
   Discard, or Cancel. Slots are independent for each sample.
8. **Quit** saves the last image index/reviewer for resume, after handling unsaved
   edits. Relaunch the same command to continue.

There are at most two slots per sample. **box_1/box_2 are arbitrary slot IDs, not
anatomical left/right labels**, because trustworthy DICOM orientation is absent.
Bounding boxes are prompts, not lung pixel ground truth or mask correctness grades.

## Outputs and coordinate convention

```text
OUTPUT_HASIL/segmentation/phase2_prompts/
    phase2_selection.csv
    manual_lung_boxes.csv
    manual_lung_boxes.json
    previews/<sample_id>.png
    annotation_history.jsonl     # created after actual annotation changes
```

Each annotation record includes:

```text
sample_id, source_path, class, image_width, image_height, box_id,
x_min, y_min, x_max, y_max, visibility_status, reviewer_id, notes
```

- Native stored image coordinates, origin top-left; x increases rightward and y
  downward. `x_max`/`y_max` are **exclusive pixel-edge bounds**, allowed up to
  native width/height. A full image would span `[0, 0, width, height]`.
- The GUI now defaults to Native 1:1 with scrollbars: no display resampling.
  Optional Fit mode may downsample for overview but never enlarges the image.
  Saved coordinates always map back to native dimensions, including 801x512 images. Dragging in either
  direction works; endpoints are clamped to the image canvas.
- Visibility states: **VISIBLE**, **NOT_VISIBLE**, **PENDING**. NOT_VISIBLE and
  PENDING have blank coordinate fields, never zero-area dummy boxes.
- Initially all 30 slots are PENDING, with no inferred coordinates or reviewer.
- Reviewer identity is required for VISIBLE and NOT_VISIBLE records. Notes are
  per slot. Anatomical PASS/FAIL judgments belong to a later separate mask review.
- JSON is the authoritative saved state, including resume index and selection
  fingerprint. CSV is its portable export. Individual writes use atomic replace.
  If CSV was manually edited or a write was interrupted and files differ, the tool
  refuses to overwrite those edits. `--repair-csv` explicitly regenerates CSV from
  JSON; preserve/reconcile any edits first.
- Previews are new PNG copies with display annotations. The original JPEGs are
  read-only and verified by SHA256; neither EXIF orientation nor anatomy is changed.

## Noninteractive checks

```powershell
& ".\budi\Scripts\python.exe" tools/draw_lung_boxes.py --prepare-only
& ".\budi\Scripts\python.exe" tools/draw_lung_boxes.py --smoke
& ".\budi\Scripts\python.exe" -m unittest discover -s tests -p "test_draw_lung_boxes.py" -v
```

Smoke validates the real manifest/source files and instantiates a hidden Tk GUI
without drawing boxes. Tests use separate synthetic image fixtures, not real
research annotations. No teacher may run until the human prompts are finished
and the user explicitly continues the pilot.
