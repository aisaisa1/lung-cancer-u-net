# Q1 Research Roadmap
## Reliability-Gated Cross-Representation Pseudo-Label Distillation for Lung Segmentation in Non-HU CT

## 0. Locked Research Position

### Target dataset
IQ-OTH/NCCD Lung Cancer CT dataset.

Important facts for this project:
- The target data used locally are regular image exports such as JPG/PNG rather than reliable raw HU-valued CT volumes.
- The target dataset is primarily a diagnostic/classification dataset and does not provide a trusted pixel-level lung mask set for the local segmentation pipeline.
- The current project therefore cannot treat automatically generated masks as true ground truth.

### Segmentation target
The segmentation task is **whole-lung field segmentation**, not tumor segmentation.

Output convention:
- 0 = background / non-lung
- 1 = lung field

If a teacher returns left and right lung as separate labels, they may be merged into one binary lung mask for the main experiment.

### Downstream goal
The lung mask is used to isolate lung ROI before downstream lung cancer classification.

Conceptual pipeline:

RAW IQ-OTH CT image
-> lung segmentation
-> lung ROI
-> optional image enhancement for classifier
-> normal / benign / malignant or explicitly justified binary classifier
-> optional XAI and robustness analysis

---

# 1. Why the Current Pipeline Is Not Yet Valid

## P1. No real target-domain segmentation ground truth
IQ-OTH/NCCD does not give the current code a reliable pixel-level lung target.

Consequence:
Dice or IoU against an automatically generated pseudo-mask only measures agreement with the pseudo-label source. It is not absolute anatomical accuracy.

Required fix:
- create a small target-domain gold subset by manual correction/annotation and expert review if possible;
- and/or use an external CT lung dataset with real masks such as LCTSC for absolute segmentation evaluation.

Never call pseudo-labels "ground truth".

## P2. Current pseudo-mask generation uses Otsu
The current `generate_pseudo_mask()` approximately does:

image
-> grayscale
-> Gaussian blur
-> Otsu threshold
-> morphology
-> fill holes
-> pseudo-mask

This is only a classical image-processing heuristic.

Consequence:
A student U-Net trained against this target learns to imitate Otsu errors.

Required fix:
Otsu remains only:
- a classical baseline;
- never the primary supervision for the proposed method.

## P3. Current pseudo-mask is generated after resize and image preprocessing
Teacher/pseudo-mask generation must not use an aggressively modified 224 x 224 Gamma/CLAHE image as the canonical source if a less-processed target image exists.

Required canonical order:

RAW/least-processed target image at native resolution
-> teacher inference and reliability estimation
-> pseudo-mask
-> QC
-> freeze/cache target
-> resize image and mask together for student training

## P4. Pretrained lung teachers may fail on non-HU JPG/PNG
LungMask R231/R231CovidWeb was developed for CT and warns that results on non-HU regular images, cropped images, annotated images, and large intensity shifts can be unreliable.

This failure mode is not merely an inconvenience. It motivates the proposed research problem.

## P5. Random slice-level splitting risks patient leakage
IQ-OTH/NCCD contains many slices per case/patient.

A split such as:
Patient A slice 12 -> train
Patient A slice 13 -> test

is invalid for a strong paper.

Required fix:
Recover patient/case groups if possible and use patient-level/group-level split.

If exact patient IDs cannot be reconstructed, document this explicitly as a dataset limitation and do not describe a random slice split as independent patient testing.

## P6. Validation is not test
Current code builds `test_pairs` but the existing training loop is primarily reporting validation performance.

Required:
- train set for optimization;
- validation set for checkpoint/threshold/hyperparameter selection;
- held-out test set for final reporting only.

## P7. Current metrics are insufficient
Final segmentation evaluation should include at least:
- Dice
- IoU/Jaccard
- sensitivity/recall
- precision
- HD95
- ASD/ASSD if practical
- number of trainable parameters
- inference latency
- GPU memory or approximate resource use

Calculate segmentation metrics per image and report mean, SD, median and confidence interval where practical.

## P8. Current Grad-CAM experiment is a classifier experiment
Grad-CAM on `ResNet18_CBAM` does not measure U-Net segmentation quality.

Keep three experiment families separate:
1. segmentation,
2. classification,
3. explanation/robustness.

---

# 2. Locked Novelty

## Working name
**RCR-PD: Reliability-Gated Cross-Representation Pseudo-Label Distillation**

This is a working research name, not a final paper title.

## Scientific problem
A target CT repository may be available only as regular non-HU image exports and without lung masks.

A pretrained lung segmentation teacher trained on proper HU CT can be confidently wrong after:
- loss of HU values;
- intensity remapping;
- export to JPG/PNG;
- contrast enhancement;
- scanner/window differences.

Naively distilling every teacher mask propagates teacher errors into the student.

## Core hypothesis
A trustworthy pseudo-label should be **stable across anatomy-preserving representations of the same CT slice**.

Instead of using only teacher confidence:

x -> teacher -> pseudo-mask

we use:

same anatomical slice x
-> representation r1
-> representation r2
-> representation r3
...
-> teacher predictions
-> map predictions to one coordinate system
-> measure agreement
-> anatomical sanity checks
-> reliability map/score
-> reliable pseudo-supervision only

## Cross-representation probes
Begin conservatively.

Candidate probes:
- raw/non-HU image
- simple linear intensity normalization
- mild gamma transform
- mild CLAHE
- mild contrast transform

These transforms are **reliability probes**, not automatically accepted training preprocessing.

Do not use transformations that alter anatomy.

Do not assume a pseudo-HU conversion is scientifically valid without a defensible mapping.

## Reliability components

### A. Representation consistency
For K predictions of the same anatomy:

P_1(x), P_2(x), ..., P_K(x)

estimate pixel-wise and/or mask-level agreement.

Possible measures:
- mean probability
- variance
- pairwise Dice
- pairwise IoU
- entropy of ensemble probability

High agreement -> more reliable.
High disagreement -> less reliable.

### B. Optional inter-teacher agreement
If both R231 and R231CovidWeb are operational:

Teacher 1 predictions
vs
Teacher 2 predictions

can provide another reliability signal.

This is auxiliary, not the main novelty.

### C. Anatomical plausibility
Use conservative lung-shape sanity checks, for example:
- foreground area is not nearly zero;
- foreground does not fill nearly the entire image;
- expected number of dominant connected components is plausible;
- foreground centroid and laterality are plausible;
- excessive contact with image border is flagged;
- extreme fragmentation is flagged.

Do not use aggressive anatomical priors that remove large tumors or abnormal lung morphology.

### D. Boundary uncertainty
Teacher disagreement near boundaries should produce lower pixel weights rather than automatically forcing a hard pseudo-label.

## Reliability-gated supervision

For each pixel i:

w_i = reliability(i)

where:
- w_i close to 1 -> strong supervision;
- intermediate -> soft/downweighted supervision;
- w_i below threshold -> ignore.

Candidate loss:

L_total =
lambda_seg * L_weighted_BCE_Dice
+ lambda_cons * L_representation_consistency
+ optional lambda_boundary * L_boundary

The first implementation should stay simple.

## What is NOT claimed as novelty
Do not claim any of these alone:
- U-Net
- Attention U-Net
- ResNet encoder
- CBAM
- pseudo-labeling
- uncertainty
- multi-teacher learning
- consistency regularization
- CLAHE
- Gamma correction

These are established components.

The novelty claim is the **specific reliability formulation for pseudo-supervision under lost-HU / non-HU CT representation shift**, plus its validation across U-Net students and downstream diagnostic utility.

---

# 3. Mandatory Research Questions

## RQ1
How reliable are off-the-shelf CT lung segmentation teachers when applied to non-HU IQ-OTH/NCCD image exports?

## RQ2
Can cross-representation consistency predict which teacher pseudo-labels are trustworthy?

## RQ3
Does reliability-gated pseudo-label distillation improve lung segmentation compared with:
- Otsu pseudo-supervision;
- naive single-teacher pseudo-supervision;
- naive multi-teacher/ensemble supervision;
- ungated consistency training?

## RQ4
Is the benefit architecture-independent across multiple U-Net variants?

## RQ5
Do improved lung masks improve downstream diagnosis, rather than only improving pseudo-label agreement?

## RQ6
Does the proposed approach remain more stable under mild image corruption/intensity shift?

---

# 4. Experimental Gates

Do not jump directly to full training.

## GATE 0 - Repository and dataset audit
Output:
- repository tree;
- Python environment report;
- dataset folders;
- image count per class;
- native image dimensions;
- extension distribution;
- patient/case grouping feasibility;
- duplicate detection;
- exact files currently used by segmentation and classification.

Status values:
PASS / FIX / BLOCKED

## GATE 1 - Teacher feasibility pilot
Take a stratified pilot set, for example 30-60 images including:
- Normal
- Benign
- Malignant
- different apparent axial positions
- diverse contrast/intensity

Teachers:
- LungMask R231
- LungMask R231CovidWeb

Test on least-processed/native-resolution images.

Save for every sample:
- original
- teacher binary mask
- overlay
- ROI
- mask area ratio
- connected components
- border-touch metric
- pairwise teacher agreement when available

Create contact sheets.

Decision:
- GO: majority anatomically plausible
- FIX: systematic but correctable failure
- NO-GO: teacher cannot reliably segment this representation

Do not train student U-Nets before this gate.

## GATE 2 - Target-domain gold anchor
Create a manually corrected subset if feasible.

Minimum research objective:
- approximately 50-100 representative slices
- balanced across diagnostic groups as far as reasonable
- corrections performed using a reproducible annotation protocol
- expert/radiology review if available

Keep:
- raw teacher mask
- corrected mask
- annotator/reviewer metadata
- reason for correction

If this is unavailable, clearly downgrade claims from anatomical accuracy to teacher agreement and rely more heavily on external GT validation.

## GATE 3 - Freeze pseudo-label dataset
Once teacher strategy is accepted:
- generate pseudo-labels once;
- save mask;
- save probability map when available;
- save reliability map;
- save QC metrics;
- save provenance/config/hash.

Every U-Net comparison must use the exact same split and supervision targets.

## GATE 4 - Baseline U-Net benchmark
Recommended main models:
1. Vanilla U-Net
2. U-Net Light
3. Attention U-Net
4. ResUNet
5. ResNet18 U-Net
6. U-Net++
7. U-Net3+ if feasible

Optional non-U-Net control:
- DeepLabV3+ or SegFormer

Do not train every architecture until a common training/evaluation harness passes a smoke test.

## GATE 5 - Proposed RCR-PD method
Run the proposed reliability-gated method on selected student architectures.

At minimum:
- Vanilla U-Net
- strongest conventional U-Net baseline
- ResNet18 Attention U-Net used in the current project

Then test whether the benefit transfers across architectures.

## GATE 6 - Ablation
Required ablations:
A0: Otsu pseudo-label
A1: single teacher, no gating
A2: teacher ensemble, no gating
A3: confidence-only gating if probability available
A4: representation-consistency gating only
A5: anatomy sanity gate only
A6: representation + anatomy
A7: full proposed method

Optional:
- boundary weighting
- consistency loss

## GATE 7 - External or gold-standard segmentation evaluation
Primary preference:
- target-domain manually corrected subset

Additional external source:
- CT lung dataset with true masks such as LCTSC

Report absolute segmentation metrics only against real/manual labels.

Report pseudo-label agreement separately.

## GATE 8 - Downstream diagnosis
Compare identical classifier under:
1. full image
2. Otsu ROI
3. teacher ROI
4. best baseline U-Net ROI
5. RCR-PD ROI

Important:
The classifier split must be patient-level/group-safe.

If using binary Normal vs Malignant:
- explicitly state that Benign is excluded and why.

Prefer additionally reporting a 3-class experiment if scientifically appropriate.

Metrics:
- accuracy
- balanced accuracy
- macro F1
- per-class sensitivity/recall
- specificity
- AUROC one-vs-rest for multiclass
- calibration if relevant

## GATE 9 - Robustness and XAI
Only after segmentation and classification are stable.

Corruptions:
- Gaussian noise
- blur
- mild brightness/contrast
- compression if clinically/repository relevant

Measure:
- segmentation Dice degradation
- mask stability
- classification degradation
- confidence/calibration change
- Grad-CAM stability for classifier only

Do not conflate Grad-CAM with segmentation.

---

# 5. Fair U-Net Benchmark Protocol

All compared architectures must share:
- exact same train/val/test split;
- same input resolution per experiment;
- same supervision masks;
- same augmentations unless architecture requires otherwise;
- same optimizer family;
- same training budget;
- same early stopping rule;
- same threshold-selection rule;
- same random seeds;
- at least 3 seeds for final important comparisons if compute permits.

Initial debug loss:
- BCE + Dice

After baseline works, ablate:
- BCE + Dice
- Lovasz + BCE
- boundary-enhanced objective

Do not change model, loss, augmentation, resolution and pseudo-label source simultaneously.

---

# 6. Data Leakage Rules

This project must actively search for patient/case grouping.

Never silently perform slice-level random train/test split if multiple slices belong to the same case.

Write a dataset audit table containing:
- filename
- class
- inferred case ID
- image size
- hash
- duplicate group
- split

Any uncertain case mapping must be logged.

---

# 7. Required Output Files

Recommended output structure:

OUTPUT_HASIL/
  segmentation/
    audit/
    teacher_pilot/
    pseudo_labels/
    reliability/
    benchmarks/
    ablations/
    external_eval/
    figures/
    tables/
  classification/
  robustness/
  logs/

CSV/JSON outputs should include:
- config
- model
- seed
- split
- checkpoint
- per-image metrics
- aggregate metrics
- runtime
- parameter count
- provenance

---

# 8. Reproducibility

Set and log seeds for:
- Python random
- NumPy
- PyTorch CPU
- CUDA

Enable deterministic behavior where practical, but log when nondeterministic kernels are required.

Every experiment must save:
- config JSON/YAML
- Git commit if repository uses Git
- environment/package versions
- start/end time
- model checkpoint
- metrics CSV
- predictions for selected samples

---

# 9. Immediate Next Task

DO NOT start the seven-model benchmark yet.

Immediate task:

**Build and run a teacher feasibility audit on native/least-processed IQ-OTH/NCCD images.**

Required script name:

`experiments/01_teacher_feasibility.py`

It must:
1. locate dataset safely;
2. never overwrite source images;
3. sample stratified images;
4. run R231 and R231CovidWeb if available;
5. support failure of one teacher without crashing the whole audit;
6. produce masks, overlays and ROIs;
7. compute simple QC;
8. generate `teacher_feasibility.csv`;
9. generate visual contact sheets;
10. print GO/FIX/NO-GO summary without pretending to know anatomical truth.

No student U-Net training until this audit is reviewed.

---

# 10. Literature Boundary For Novelty

The following ideas already exist in medical segmentation literature and must not be sold individually as new:
- uncertainty-guided pseudo-label selection;
- source-free pseudo-label adaptation;
- mean-teacher consistency;
- multi-teacher distillation;
- topology/anatomical constraints;
- probabilistic U-Net uncertainty;
- U-Net + GRU uncertainty on IQ-OTH/NCCD.

Therefore the paper must emphasize:
1. non-HU CT representation shift as the supervision reliability problem;
2. cross-representation teacher stability as an explicit pseudo-label reliability signal;
3. reliability-gated distillation rather than naive teacher imitation;
4. fair cross-architecture evaluation;
5. absolute validation against manual/external GT;
6. downstream diagnostic utility.

---

# 11. Candidate Paper Framing

Working title:

**Reliability-Gated Cross-Representation Pseudo-Label Distillation for Lung Segmentation in Unlabeled Non-HU CT Images**

Alternative application-oriented title:

**Reliable Lung ROI Learning from Unlabeled Non-HU CT: Cross-Representation Pseudo-Label Distillation and Downstream Cancer Diagnosis**

Core contribution statement, provisional:

1. We characterize the failure of HU-trained lung segmenters when transferred to non-HU CT image exports.
2. We propose a cross-representation reliability mechanism that estimates pseudo-label trustworthiness from segmentation invariance under anatomy-preserving intensity representations, optionally augmented with teacher agreement and conservative anatomical checks.
3. We distill only reliable pixel/sample supervision into multiple U-Net students using reliability-weighted segmentation loss.
4. We evaluate whether the resulting lung masks improve both segmentation quality and downstream diagnostic performance under patient-safe evaluation.
5. We evaluate robustness under controlled image perturbations.

These claims are provisional until experiments support them.
