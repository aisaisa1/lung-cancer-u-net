# Phase-1 results: teacher feasibility and representation stability

## Decision

**NO-GO for using the tested default LungMask 0.2.14 noHU outputs as ungated
student supervision. FIX/manual review is needed before any further method
decision.** The software environment works; anatomical teacher reliability has
not passed the feasibility gate. This conclusion concerns this adapter and these
non-HU target exports, not all possible use of LungMask or the existing student
architecture.

Phase 1 is complete. No student model, classifier, Grad-CAM, reliability loss,
U-Net++, U-Net3+, or full benchmark was trained/run. All expert visual-review
decisions remain pending.

## Dataset audit

Full inventory covers 6,222 JPEG files across six local versions, with zero image
read errors. The canonical teacher source is:

`DATASET/The IQ-OTHNCCD lung cancer dataset/`

It is the least-processed locally available full-class export candidate. It is
**not raw DICOM or calibrated HU**. Image export/annotation provenance needs
confirmation. No source was resized, modified, moved, or overwritten.

| Source | Normal | Benign | Malignant | Total | Stored dimensions |
| --- | ---: | ---: | ---: | ---: | --- |
| IQ-OTHNCCD export | 416 | 120 | 561 | 1,097 | 512x512: 1,036; 801x512: 28; 623x512: 31; 511x404: 1; 506x331: 1 |
| `Citra_Gamma_CLAHE` | 416 | 120 | 561 | 1,097 | all 512x512 |
| `Crop_Otsu_Gamma_CLAHE` | 416 | 120 | 561 | 1,097 | all 512x512 |
| `Crop_UNet_Gamma_CLAHE` | 416 | 0 | 561 | 977 | all 224x224 |
| `Crop_UNet_Basic_Gamma_CLAHE` | 416 | 0 | 561 | 977 | all 224x224 |
| `Crop_UNet_Light_Gamma_CLAHE` | 416 | 0 | 561 | 977 | all 224x224 |

The Benign folder is spelled `Bengin cases`; reporting normalizes this label
without renaming the original folder. The local dataset description claims 1,190
images/110 cases, which differs from the observed 1,097 exports. It does not
provide a slice-to-case mapping. Treat that discrepancy as provenance to resolve.

Exact decoded-pixel hashes within the export source identify **43 duplicate
groups** (43 redundant files, 1,054 unique pixel hashes). Across all versions,
1,061 exact duplicate groups appear for both file-byte and decoded-pixel hashes;
many are cross-version copies, not distinct patients. One cross-class exact-pixel
group exists across the audited versions; consult `duplicate_report.csv` to locate
it rather than assuming it represents a raw target label conflict. Gamma/CLAHE
has four redundant files in one group; each U-Net crop source has three redundant
files in one group. Exact hashes do not detect all near duplicates or transformed
versions.

No usable manual mask directory or DICOM data was found. Two `*1` extensionless
files are old machine-path text references, not image datasets. There is no safe
patient/case grouping: `Normal/Bengin/Malignant case (N)` is a class-local serial
pattern with no verified case metadata. The pilot is a diagnostic sample, not a
training/test split and not 45 independent patients.

## Pilot protocol

- Frozen R231 and R231CovidWeb; official pretrained weight hashes recorded.
- Isolated Python 3.10.11 / torch 2.5.1+cpu / LungMask 0.2.14 environment.
- `LMInferer(noHU=True, volume_postprocessing=True, batch_size=4)`.
- 45 images: 15 per class; seed 42; deterministic SHA256-ranked relative paths,
  with exact pixel duplicate exclusion. Every selected source hash verified.
- No alternative dataset, external target download, or adaptive sample replacement.
- The same anatomical image is retained at its native stored dimensions for all
  six intensity representations: native, min-max linear, gamma exponent 0.9,
  gamma exponent 1.1, mild CLAHE 25% blend, mild contrast factor 1.05.
- Upstream performs internal 256x256 resizing and nearest-neighbor remapping.
  A selected 801x512 export is retained as 801x512 in saved outputs.
- Saved ROI is a full-frame native image multiplied by the mask, not a differently
  cropped view. Labels >0 form the binary union of predicted left/right lung.
- Failures remain in denominators. No mask file is fabricated for failed inference.
- Both-empty mask Dice/IoU equals 1 by set convention, but is explicitly flagged
  and excluded in the additional nonempty-union analyses below.

## R231 and R231CovidWeb results

| Result | R231 | R231CovidWeb |
| --- | ---: | ---: |
| Native images attempted | 45 | 45 |
| Native technical successes | 27 (60%) | 27 (60%) |
| Native technical failures | 18 (40%) | 18 (40%) |
| Successful native predictions by class, Normal/Benign/Malignant | 8 / 9 / 10 | 8 / 9 / 10 |
| Empty masks among technical successes | 7 / 27 | 3 / 27 |
| Nonempty native outputs, not anatomical acceptance | 20 / 45 | 24 / 45 |
| Failed representation calls | 117 / 270 | 117 / 270 |
| Complete six-representation images | 13 / 45 | 13 / 45 |

All recorded per-image inference exceptions were the same upstream sanity-filter
condition: every internal intensity view was rejected. Environment/model loading
succeeded for both teachers. An inference returning a shaped label array is only
a technical success; masks may still cover nothing, background, or partial lung.

Cross-teacher **agreement** on the 27 images with both native outputs:

| Denominator | Mean Dice | Mean IoU |
| --- | ---: | ---: |
| All 27 technically successful pairs, including 3 both-empty pairs | 0.481725 | 0.398620 |
| 24 pairs with nonempty union | 0.416941 | 0.323448 |

These are not segmentation accuracy estimates. Failures are not assigned a
synthetic zero Dice; their count is reported separately.

## Cross-representation stability

Aggregate stability uses only the **13 complete six-view images per teacher**,
not all 45 images. Each complete image has 15 unordered view pairs. Partial-view
statistics and coverage are preserved in `representation_stability.csv`.

| Complete-view result | R231 | R231CovidWeb |
| --- | ---: | ---: |
| Mean of per-image mean pairwise Dice | 0.565227 | 0.585680 |
| Mean of per-image mean pairwise IoU | 0.536999 | 0.537600 |
| Mean of per-image minimum pairwise Dice | 0.074294 | 0.072788 |
| Worst pairwise Dice across complete images | 0 | 0 |
| Complete images with minimum pairwise Dice = 0 | 12 / 13 | 11 / 13 |
| Both-empty representation pairs in complete images | 11 | 1 |
| Mean per-image Dice excluding both-empty pairs | 0.509527 | 0.581041 |
| Mean per-image IoU excluding both-empty pairs | 0.481130 | 0.532820 |

Disagreement maps are the empirical fraction of successful unordered view pairs
that differ at each pixel, `2*k*(n-k)/(n*(n-1))`. NPY values retain their original
0–1 scale; PNG heat maps use that fixed scale. Maps with partial coverage must be
interpreted alongside successful-representation counts. They are not calibrated
uncertainty or final reliability weights.

## Visual observations requiring expert review

1. `Bengin case (120)` (`benign_bdd00f97a62b`): native outputs highlight tiny
   exterior patches instead of enclosing the visible lung regions; teacher Dice
   is zero. Technical nonempty status is not anatomical success.
2. `Malignant case (202)` (`malignant_bc3ffd00292a`, native 801x512): R231 mean
   representation Dice is **0.984531** (CovidWeb 0.963069), yet the contact sheets
   show a substantial predicted foreground outside the body and incomplete lung
   coverage. This is a concrete **stable-but-incorrect** counterexample.
3. `Normal case (222)` (`normal_4df864c5ac1e`): R231 native/most view masks are
   empty, while the linear representation labels background patches. The
   both-empty convention can inflate a mean stability score.

These are preliminary engineering observations from displayed contact sheets,
not expert annotations, measured anatomical error, or completed human acceptance.
Every entry of `human_visual_review.csv` remains `pending`.

## Verification

- Five initial mathematical/IO/geometry tests passed; a further single-teacher
  failure integration test checks graceful output adaptation and denominators.
- Real smoke run: both models loaded, all 3 classes/all 6 representations attempted.
- Additional smoke in the unchanged main environment: both missing teachers were
  recorded as failed, with originals/contact sheets and null agreement summaries.
- Independent artifact verification passed: all **6,222** source image SHA256s
  unchanged; **540** feasibility rows; **1,224** mask/label/overlay/ROI images
  checked; **654** pairwise metrics recomputed; **54** disagreement maps checked
  against explicit pairwise XOR calculations.
- The full pilot hit the shell's 600-second limit after image 42. Resume validated
  the fixed sample manifest, package versions and cached artifacts, reused 504
  completed representation rows, and completed the remaining images. Resume
  wall time in JSON is the continuation time, not the full pilot duration.

## Exact review outputs

Base: `OUTPUT_HASIL/segmentation/teacher_pilot/`.

Start with:

```text
contact_sheets/benign_bdd00f97a62b_teachers.png
contact_sheets/malignant_bc3ffd00292a_teachers.png
contact_sheets/malignant_bc3ffd00292a_R231_representations.png
contact_sheets/malignant_bc3ffd00292a_R231CovidWeb_representations.png
contact_sheets/normal_4df864c5ac1e_teachers.png
contact_sheets/normal_4df864c5ac1e_R231_representations.png
contact_sheets/normal_fc7dd0331035_teachers.png
disagreement_maps/R231/malignant_bc3ffd00292a.png
disagreement_maps/R231CovidWeb/malignant_bc3ffd00292a.png
```

Then review all teacher contact sheets against originals at full native resolution,
using `selected_images.csv` to map image IDs to filenames. Review masks/overlays
for both lungs, exterior background, missing pathological/dense regions, empty
masks, annotations and export orientation. Record decisions and reviewer/notes
in the local `human_visual_review.csv`; no training acceptance follows automatically.

Quantitative/provenance outputs:

- `teacher_feasibility.csv`, `teacher_summary.json`
- `representation_pairwise.csv`, `representation_stability.csv`
- `human_visual_review.csv`, `verification.json`, `run_config.json`
- `originals/`, `representations/`, `masks/`, `label_maps/`, `overlays/`, `roi/`
- Audit inventory, duplicate report and JSON under `OUTPUT_HASIL/segmentation/audit/`

## Remaining blockers

Reliable teacher behavior on non-HU exports has not been established. The legacy
adapter rejects many slices and can be consistently wrong; its internal view
selection/postprocessing also confounds representation stability. Expert review,
trusted correction/reference masks and case-to-slice provenance are needed before
supervision/splitting decisions. Any different adapter or further experiments
require explicit continuation instruction. Phase 1 stops here.
