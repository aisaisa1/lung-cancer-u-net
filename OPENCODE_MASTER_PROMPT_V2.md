# OpenCode Master Prompt V2
## IQ-OTH/NCCD Lung Segmentation Research Project

You are working on an existing PyTorch research repository.

Before making any research-method changes, read these two local documents:

1. `GITHUB_REFERENCE_STACK_LUNG_Q1.md`
2. `ROADMAP_Q1_LUNG_SEGMENTATION.md` if present

The GitHub reference stack takes precedence if there is a methodological conflict with the older roadmap.

# CRITICAL RULE

Do NOT invent an implementation from scratch before inspecting the closest public reference repositories.

The implementation must be an original adaptation, not a copy.

For every major component you implement, document:
- reference repository
- exact file/function inspected
- what concept was borrowed
- what was changed
- why it differs scientifically

Create this documentation in:
`docs/REFERENCE_IMPLEMENTATION_MAP.md`

# REFERENCE REPOSITORIES

Study these in this order:

## A. LungMask
https://github.com/JoHof/lungmask

Inspect:
- README
- `lungmask/mask.py`
- `lungmask/resunet.py`

Purpose:
pretrained CT lung teacher behavior, R231/R231CovidWeb inference, model loading, output semantics, post-processing.

## B. FPL+
https://github.com/HiLab-git/FPL-plus

Inspect:
- README
- `data/get_pixel_weight.py`
- `merge_pixelw.py`
- pseudo-label generator configs
- final segmentor training path

Purpose:
closest methodological precedent for filtered/weighted pseudo-labels.

IMPORTANT:
The project MUST NOT claim that simple cross-representation agreement or pixel weighting is novel because FPL+ already implements a closely related idea.

## C. UPL-SFDA
https://github.com/HiLab-git/UPL-SFDA

Inspect:
- `run_2d_upl.py`
- `unet.py`
- `loss.py`
- `utils.py`

Purpose:
source-free adaptation, transformed-view predictions, multi-decoder consistency, pseudo-label generation, entropy regularization.

## D. UC-Seg
https://github.com/taozh2017/UCSeg

Inspect:
- 2D trainer
- uncertainty/pseudo-label modules
- prediction utilities

Purpose:
modern uncertainty-aware pseudo-label generation and cross-subnet consistency.

## E. Medical Image Segmentation Benchmarks
https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks

Inspect:
- `src/`
- `main.py`
- `infer.py`
- `split.py`

Purpose:
multi-model registry, unified training/inference, fair U-Net comparisons.

# CURRENT NOVELTY POSITION

Working concept:

**BTRC-CT: Black-box Teacher Reliability Calibration for Pseudo-Label Distillation under Non-HU CT Representation Loss**

Do NOT describe this as proven novel.

The research hypothesis is:

A pretrained lung CT teacher may become unreliable when HU information is irreversibly lost and the CT slice exists only as a regular JPG/PNG-like image representation.

Instead of assuming the teacher pseudo-mask is correct, estimate its reliability using signals that do not require changing the teacher:

1. same-anatomy intensity-representation stability;
2. optional agreement between R231 and R231CovidWeb;
3. conservative anatomical plausibility;
4. calibration against a small manually corrected target-domain subset.

The most important novelty test is NOT:
"Does disagreement produce weights?"

FPL+ already contains a related idea.

The important test is:
"Can a reliability score calibrated for non-HU representation loss predict REAL teacher segmentation error and improve distillation?"

# REQUIRED VALIDATION OF THE RELIABILITY SCORE

Before student training, the project must support a small real-GT target subset.

On held-out gold masks calculate:

- teacher Dice/IoU
- reliability score
- correlation(reliability, teacher GT error)
- AUROC/AUPRC for detecting poor pseudo-labels
- reliability calibration plot
- selective-risk curve

Example selective-risk question:
If the bottom 10%, 20%, 30% least reliable pseudo-labels are rejected, does mean teacher error among retained samples fall?

If reliability score cannot predict real teacher error:
STOP.
Do not proceed with BTRC-CT as the method.

# IMMEDIATE TASK ONLY

Do NOT implement the final novelty yet.

Phase 1 consists of:

1. repository audit;
2. dataset audit;
3. teacher environment audit;
4. teacher feasibility pilot;
5. reference implementation map.

After Phase 1, STOP and report.

# 1. REPOSITORY AUDIT

Locate actual active files for:
- segmentation models
- data loading
- pseudo-mask generation
- segmentation training
- preprocessing
- classification
- Grad-CAM

Search all occurrences of:
- `generate_pseudo_mask`
- `THRESH_OTSU`
- `cv2.threshold`
- `Citra_Gamma_CLAHE`
- `Crop_UNet_Gamma_CLAHE`
- `build_pairs`
- `split_pairs`
- `best_unet.pt`

Create:
`docs/CURRENT_PIPELINE_AUDIT.md`

Do not delete or rewrite old code yet.

# 2. REFERENCE IMPLEMENTATION MAP

Create:
`docs/REFERENCE_IMPLEMENTATION_MAP.md`

Use a table:

Component | Our current code | Reference repo | Reference file/function | Reuse concept | Important difference

Required components:
- teacher inference
- pseudo-label generation
- pseudo-label filtering/weighting
- transform consistency
- uncertainty handling
- U-Net model registry
- unified train loop
- evaluation
- data split

# 3. DATASET AUDIT

Create:
`experiments/00_dataset_audit.py`

Required outputs:
`OUTPUT_HASIL/segmentation/audit/dataset_inventory.csv`
`OUTPUT_HASIL/segmentation/audit/duplicate_report.csv`
`OUTPUT_HASIL/segmentation/audit/dataset_audit.json`

Audit:
- classes
- counts
- file types
- dimensions
- duplicates
- derived/preprocessed folders
- least-processed available images
- possible patient/case identifiers

Do not claim patient-level independence unless supported by filenames/folders/metadata.

# 4. TEACHER FEASIBILITY

Create:
`experiments/01_teacher_feasibility.py`

Teachers:
- LungMask R231
- LungMask R231CovidWeb

Use least-processed/native-resolution target images.

Do NOT generate teacher masks from Gamma+CLAHE images unless the script is explicitly testing that representation.

The script should support multiple intensity representations for DIAGNOSTIC analysis only:
- native target image
- linear normalized
- mild gamma lower
- mild gamma higher
- mild CLAHE

These are not yet pseudo-label training targets.

For each image and representation:
- run teacher
- map output into common image coordinates
- create binary lung mask
- save overlay
- save ROI
- record mask area
- connected components
- border contact
- centroid
- fragmentation

If both teachers run:
- compute R231 vs R231CovidWeb Dice/IoU

Across representations:
- compute pairwise mask Dice
- mean Dice
- minimum Dice
- pixel disagreement map

IMPORTANT:
Call these:
- stability
- agreement
- QC

Do NOT call them accuracy.

Outputs:
`OUTPUT_HASIL/segmentation/teacher_pilot/teacher_feasibility.csv`
`OUTPUT_HASIL/segmentation/teacher_pilot/teacher_summary.json`
`OUTPUT_HASIL/segmentation/teacher_pilot/contact_sheets/`
`OUTPUT_HASIL/segmentation/teacher_pilot/disagreement_maps/`

# 5. TEACHER PACKAGE SAFETY

Do not break the current Python/PyTorch environment.

Before installation:
- log Python
- PyTorch
- CUDA
- torchvision
- numpy
- scipy
- SimpleITK

If the LungMask version required for non-HU behavior conflicts with the active environment:
- create a separate environment specification
- do not downgrade working packages automatically

# 6. STOP CONDITION

After Phase 1:
STOP.

Report:

1. Current repository structure
2. Current pseudo-mask logic
3. Dataset format
4. Patient/case split feasibility
5. LungMask environment status
6. Teacher pilot success/failure
7. Representation stability statistics
8. Teacher agreement statistics
9. Files created
10. Exact blockers
11. GO / FIX / NO-GO
12. Whether a manually corrected gold subset is now required

Do NOT:
- train all U-Nets
- train final student
- run classification
- run Grad-CAM
- claim BTRC-CT novelty
in Phase 1.

# FUTURE PHASES AFTER USER APPROVAL

## Phase 2: Gold target subset
Build export/import workflow for manually corrected masks.

## Phase 3: Reliability validation
Implement candidate reliability score and verify it predicts true teacher error on held-out gold masks.

Only if successful:
proceed.

## Phase 4: Frozen pseudo-label dataset
Freeze:
- pseudo-mask
- teacher provenance
- probability if available
- reliability map
- image reliability score

## Phase 5: Fair architecture benchmark
Use a single model factory inspired by:
https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks

Initial architecture set:
- Vanilla U-Net
- U-Net Light
- Attention U-Net
- ResUNet
- ResNet18 U-Net
- U-Net++
- U-Net3+

Optional:
- TransUNet or SegFormer as non-classical comparator

All models must use the exact same:
- split
- pseudo-targets
- augmentations
- input size
- optimizer policy
- training budget
- threshold-selection rule
- test evaluator

## Phase 6: BTRC-CT distillation
Compare:
- Otsu
- naive R231
- naive R231CovidWeb
- naive teacher ensemble
- representation-stability weight
- anatomy-QC weight
- calibrated BTRC-CT

## Phase 7: Real-GT segmentation evaluation
Never mix pseudo-label agreement and manual-GT performance.

## Phase 8: Downstream diagnosis
Compare full image vs ROI sources.

## Phase 9: Robustness/XAI
Only after segmentation/classification are stable.

# ENGINEERING REQUIREMENTS

- Windows compatible
- `num_workers=0` available
- deterministic seed handling
- CLI `--smoke`
- no destructive dataset operations
- never silently choose "first file" if named file is missing
- reusable model factory
- config-driven experiments
- save per-image CSV
- save experiment JSON/YAML
- save selected visualizations
- do not commit unless explicitly asked

# FINAL BEHAVIOR

When uncertain about a research-method implementation:
1. inspect the closest reference repo;
2. document the precedent;
3. avoid inventing unnecessary complexity;
4. do not silently turn an existing method into a "novel" contribution.
