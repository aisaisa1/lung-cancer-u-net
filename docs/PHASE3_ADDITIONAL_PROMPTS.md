# Phase-3 additional 15-image manual-prompt handoff

Per the updated instruction, this handoff **only** selects additional images and
prepares human box annotation. It does not export the Gold Subset, generate any
new candidate mask, run a teacher, or train a model.

## Launch the additional-image GUI

From the project root in PowerShell:

```powershell
& ".\budi\Scripts\python.exe" "tools\draw_lung_boxes.py" --output-dir "OUTPUT_HASIL\segmentation\phase3_additional_prompts"
```

The window identifies its session as **phase3_additional_prompts**. The original
Phase-2 completed annotations are kept in their separate `phase2_prompts` folder.
Enter your reviewer ID, then draw `box_1` and `box_2` independently. These are
arbitrary slot IDs, never anatomical left/right labels. Use NOT_VISIBLE explicitly
for a region that cannot be seen; leave uncertain/unfinished slots PENDING with
notes rather than guessing. Native 1:1 view and scrollbars preserve stored geometry.

Save with Ctrl+S or Save, navigate Previous/Next, and Quit/relaunch to resume. The
additional session starts with **30 PENDING slots** (two slots per image), no
coordinates, no reviewer, and no inferred annotations.

## Exact additional selection

| Class | Original filename | Native width x height | Sampling cue |
| --- | --- | --- | --- |
| Normal | `Normal case (155).jpg` | 512x512 | Prior native technical failure |
| Normal | `Normal case (198).jpg` | 512x512 | High-frequency/sharpness proxy; visual noise/texture present |
| Normal | `Normal case (344).jpg` | 506x331 | Additional native-resolution stratum |
| Normal | `Normal case (280).jpg` | 512x512 | High central brightness proxy |
| Normal | `Normal case (112).jpg` | 512x512 | Low-sharpness/possible-blur proxy |
| Benign | `Bengin case (11).jpg` | 512x512 | Remaining deterministic path-hash rank |
| Benign | `Bengin case (79).jpg` | 512x512 | Prior native technical failure |
| Benign | `Bengin case (76).jpg` | 512x512 | High central brightness proxy |
| Benign | `Bengin case (52).jpg` | 512x512 | Low-sharpness/possible-blur proxy |
| Benign | `Bengin case (102).jpg` | 512x512 | High-frequency/sharpness proxy; visual noise/texture present |
| Malignant | `Malignant case (234).jpg` | 623x512 | Additional native-resolution stratum |
| Malignant | `Malignant case (548).jpg` | 512x512 | Prior native technical failure |
| Malignant | `Malignant case (180).jpg` | 511x404 | Additional native-resolution stratum; difficult ambiguous anatomy |
| Malignant | `Malignant case (181).jpg` | 512x512 | Low-sharpness/possible-blur proxy |
| Malignant | `Malignant case (458).jpg` | 512x512 | Malignant-class high brightness; dense-region review needed |

The full original filename/path and SHA256 are retained in the manifest. Source
folder spelling `Bengin cases` is preserved. All original 15 Phase-2 samples remain
in the total-30 selection, including the prior difficult/dense example
`Malignant case (95)` and the 801x512 `Malignant case (202)`.

Combined counts: **10 Normal, 10 Benign, 10 Malignant**, with **30 distinct exact
pixel hashes**. Combined resolutions: 26 at 512x512, one each at 801x512, 506x331,
623x512, and 511x404. No source is resized, enhanced, cropped, sharpened, or replaced.

## Reproducible selection and its limits

`tools/prepare_phase3_additional_prompts.py` reads the prior full inventory,
verifies all 1,097 canonical export hashes, and measures native central-window
Laplacian variance/intensity statistics. It retains the old 15, excludes their
pixel duplicates, and selects five extra per class using new native-resolution
strata, low/high-frequency proxies, central brightness and prior technical failure
strata. Ties use SHA256 of `phase3-additional|42|source_path`. The saved manifests
are then frozen and reused without reshuffling or resetting annotations.

These cues are **not human anatomical review fields**. Low Laplacian variance is
not proof of blur; high variance can indicate noise, not an easy or sharp case.
JPEG brightness is not HU and does not establish pathology. Dense or ambiguous
regions need human/expert inspection. Prior LungMask failures are a sampling cue
only, never replacement candidate masks or prompts.

All 15 original teacher pairs are retained, so their disagreements remain in the
30-image selection. `retained_teacher_disagreement.json` records pixel disagreement
as a sampling/review cue, not accuracy. Additional teacher disagreement is unknown
until manual boxes and the later explicitly requested inference are complete.
No patient IDs were inferred: 30 images are not assumed to be 30 patients.

## Prepared outputs

```text
OUTPUT_HASIL/segmentation/phase3_additional_prompts/
    phase2_selection.csv                # schema reused by the existing GUI; NEW 15 only
    manual_lung_boxes.csv               # NEW human annotations only
    manual_lung_boxes.json
    previews/<sample_id>.png
    gold_subset_selection_30.csv        # selection metadata only, not a Gold Subset export
    additional_selection_details.csv
    sampling_features.csv
    retained_teacher_disagreement.json
    selection_report.json
```

The previous `OUTPUT_HASIL/segmentation/phase2_prompts/manual_lung_boxes.*` files
are unchanged. No directory/package at `DATASET/Lung_Gold_Subset/` is exported in
this handoff. The new 15 candidate pairs are marked **NOT_GENERATED_PENDING_MANUAL_BOXES**.

## Checks and stop condition

```powershell
python tools/prepare_phase3_additional_prompts.py --smoke
python tools/draw_lung_boxes.py --output-dir OUTPUT_HASIL/segmentation/phase3_additional_prompts --smoke
python -m unittest discover -s tests -p "test_phase3_selection.py" -v
```

Preparation/checks never draw boxes. The same GUI supports save/resume, deletion,
redraw, notes and NOT_VISIBLE. Keep all pathological lung tissue within the intended
whole-lung field rather than excluding it merely because it is dense. Mark uncertain
extent in notes for expert review.

**Next human action:** complete these additional 15 images. Only after you explicitly
confirm the saved prompts are complete should they be validated and passed through
the same SAM-Med2D/MedSAM inference pipeline. Only after all 30 candidate pairs are
available will the full Gold Subset correction package be exported. No U-Net training
or BTRC-CT implementation is authorized by this handoff.
