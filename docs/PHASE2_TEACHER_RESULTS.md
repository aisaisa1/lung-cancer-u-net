# Phase-2 manual-prompt teacher comparison

## 1. Annotation validation

All **15** fixed samples have two valid saved **VISIBLE** region boxes: **30/30**.
CSV and authoritative JSON match; reviewer ID is `aisya`. Every rectangle is a
positive-area integer native-coordinate box within its own source bounds.
PENDING regions: **0**. NOT_VISIBLE regions: **0**. Missing slots: **0**.
Native source sizes are **14 × 512x512** and **1 × 801x512**. These differences are
expected; neither the boxes nor images were forced to a common native size.

Validation reports:
`OUTPUT_HASIL/segmentation/phase2_teacher_comparison/validation/annotation_validation.json`
and `region_validation.csv`.

Original image hashes, manual CSV SHA256
`3cf91326a767a1259fd4b30eb491edd7cf10d0cac7aa2c0db5798b883ec40599`, and manual JSON
SHA256 `a206225937a062967ce9cd7bd3650955cab0d562a746d24d5499f9960a6a6c85`
were unchanged after inference. No pending prompt was inferred or filled.

## 2. Preview blur diagnosis and display fix

All 15 saved annotation PNG previews have their exact source-native resolution.
After excluding only text/bounding-box drawing pixels, **zero** unmarked pixels
differ from the decoded source JPEGs. There is no destructive source/preview blur.

The original GUI used LANCZOS fit-to-window display resizing; this can soften or
enlarge display pixels. `tools/draw_lung_boxes.py` now defaults to **Native 1:1**,
with horizontal/vertical scrollbars and no display resampling. Optional Fit mode
can downsample for overview but never upscales. Scrolled mouse coordinates are
mapped through canvas coordinates. Saved annotations were not rewritten.

Comparison contact sheets also save **native-sized image panels**, without
thumbnail resizing. Opening a wide sheet in a viewer's Fit mode can still hide
detail; inspect it at 100% or use the individual native masks/overlays.

## 3. Geometry and independent prompt inference

Each teacher encoded the full stored image once. Each human region then had an
independent decoder call: no second-box mask input, prior prediction, automatic
prompt, crop-generated prompt, mask clipping, or morphology was applied.
Predictions were returned to the source HxW canvas before pixelwise union.

- SAM-Med2D: official `SamPredictor`, 256 longest-side resize, aspect-preserving
  rounded dimensions plus bottom/right padding. The native box is passed once to
  the predictor, which applies its own exact x/y scaling. A 512x801 image becomes
  164x256 before padding; this differs correctly from a 512x512 source.
- MedSAM: official in-memory cubic 1024x1024 resize/normalization. Native x box
  coordinates scale by `1024/W`, y by `1024/H`; sigmoid probabilities are
  interpolated back to original HxW, then thresholded at 0.5.
- For `Malignant case (202)`, both individual predictions and merged masks retain
  native **801x512** dimensions. Full-frame ROI is source RGB multiplied by the
  merged mask, not a separately resized/cropped image.
- Both slots are arbitrary image-region IDs, not anatomical left/right labels.

Verification checked **60 independent native-to-model box transforms** and
**30 exact native mask unions**. No accuracy metric was calculated.

Independent artifact verification also passed **120 native binary-mask reads**,
**30 full-frame ROI pixel checks**, and **120 saved QC value comparisons**.
The source-original PNG copies were pixel-identical to decoded sources. Six GUI
tests, three geometry/union tests, and the hidden Native 1:1 GUI smoke passed.

## 4. Results on the same 15 samples

| Teacher | Technical successes | Technical failures | Nonempty merged outputs | Mean area ratio | Mean inference seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM-Med2D | 15/15 | 0 | 15/15 | 0.12253 | 0.1568 |
| MedSAM | 15/15 | 0 | 15/15 | 0.12806 | 3.5597 |
| R231, cached Phase-1 native | 7/15 | 8/15 | 3/15 | 0.04798 among successful calls | 1.8570, earlier CPU run |
| R231CovidWeb, cached Phase-1 native | 7/15 | 8/15 | 6/15 | 0.04960 among successful calls | 1.8534, earlier CPU run |

Both alternative teachers also produced all **30/30 individual region masks**.
This is technical/nonempty status, not human anatomical acceptance. All **60**
sample/teacher review statuses remain **PENDING**; no automated PASS/FAIL or error
type was assigned. The required comparison CSV records area, 8-connected component
count, fragmentation, border contact, runtime and technical errors separately.

Fragmentation here is `1 - largest_component_foreground / total_foreground`.
A bilateral lung mask naturally has two components; a value near 0.5 can occur for
two correctly separated lungs and is **not** an anatomical rejection rule.

## 5. Representative observations, requiring human review

- `Normal case (26)`: both alternative outputs cover bilateral regions that
  default LungMask could not process. SAM-Med2D has more internal holes and small
  islands; MedSAM's field appears more filled. No formal anatomical grade assigned.
- `Bengin case (120)`: both alternatives appear to follow the two lung fields;
  cached LungMask outputs instead mark tiny external patches.
- `Malignant case (202)`, 801x512: both alternatives avoid the large exterior
  background foreground seen in cached LungMask outputs. This is also the key
  rectangular-coordinate transformation check.
- `Malignant case (95)`: inspect dense/pathological regions and the smaller
  box_2 region carefully. SAM-Med2D leaves visible interior holes around bright
  structures; MedSAM includes more of them. Boundary completeness and inclusion
  of abnormal tissue still need expert assessment.

## 6. Artifacts

Main table:
`OUTPUT_HASIL/segmentation/phase2_teacher_comparison.csv`.

Base directory: `OUTPUT_HASIL/segmentation/phase2_teacher_comparison/`:

```text
contact_sheets/<sample_id>_comparison.png
originals/<sample_id>.png
manual_boxes/<sample_id>.png
comparison_masks/<teacher>/<sample_id>.png
comparison_overlays/<teacher>/<sample_id>.png
box_debug_contact_sheets/<teacher>/<sample_id>.png
SAM-Med2D/individual_masks/<sample_id>_box_1.png
SAM-Med2D/individual_masks/<sample_id>_box_2.png
SAM-Med2D/merged_masks/<sample_id>.png
MedSAM/individual_masks/<sample_id>_box_1.png
MedSAM/individual_masks/<sample_id>_box_2.png
MedSAM/merged_masks/<sample_id>.png
```

Each alternative directory additionally includes `overlays/`, `roi/`,
`predictions.csv`, `geometry.json`, and `teacher_summary.json`. ROI retains the
full original image canvas. Contact sheet first row is Original | Manual Boxes |
SAM-Med2D Mask | SAM-Med2D Overlay | MedSAM Mask | MedSAM Overlay; cached R231 and
R231CovidWeb panels follow on the second row. Unavailable legacy outputs are
explicitly labeled, not fabricated as empty successful masks.

## 7. Recommendation and stop

- **SAM-Med2D: CONDITIONAL.** Successful prompt-based feasibility, but holes/dense
  region exclusions and boundaries need human review.
- **MedSAM: CONDITIONAL.** Encouraging bilateral fields in inspected cases, but
  no trusted target segmentation ground truth or completed human mask review.
- **R231 and R231CovidWeb default noHU: REJECT for ungated supervision**, retaining
  the Phase-1 conclusion. They were not rerun in this phase.

This phase stops after comparison generation. No U-Net training, pseudo-label
generation for the whole dataset, classification, Grad-CAM or robustness run.

## Execution record

Primary commands (PowerShell, project root):

```powershell
python tools/validate_phase2_annotations.py
py -3.10 -m venv .venv_phase2_teachers
& '.\.venv_phase2_teachers\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv_phase2_teachers\Scripts\python.exe' -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
& '.\.venv_phase2_teachers\Scripts\python.exe' -m pip install --only-binary=:all: -r requirements_phase2_teachers.txt
& '.\.venv_phase2_teachers\Scripts\python.exe' -m pip check
& '.\.venv_phase2_teachers\Scripts\python.exe' tools/download_phase2_weights.py --teacher SAM-Med2D
& '.\.venv_phase2_teachers\Scripts\python.exe' tools/download_phase2_weights.py --teacher MedSAM
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher SAM-Med2D --smoke
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher SAM-Med2D --smoke --output-dir OUTPUT_HASIL/segmentation/phase2_teacher_comparison/SAM-Med2D_smoke_v2
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher MedSAM --smoke
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher SAM-Med2D
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher MedSAM
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_build_comparison.py
& '.\.venv_phase2_teachers\Scripts\python.exe' tools/verify_phase2_outputs.py
python -m unittest discover -s tests -p "test_draw_lung_boxes.py" -v
& '.\.venv_phase2_teachers\Scripts\python.exe' -m unittest discover -s tests -p "test_phase2_inference.py" -v
python tools/draw_lung_boxes.py --smoke
```

Official repositories were shallow/sparse cloned into the ignored cache and their
`segment_anything` directories selected with `git sparse-checkout set`.
Initial SAM-Med2D smoke exposed the serialized Adam class; the loader was fixed
using scoped torch/stdlib allowlisting, then smoke/full inference succeeded.
The failed smoke remains separately recorded and is not part of the 15-image
results. A GUI test using fractional synthetic event coordinates was updated to
exercise the new exact Native 1:1 display; real saved annotations were not changed.
