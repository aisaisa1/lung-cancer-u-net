# GitHub Reference Stack for IQ-OTH/NCCD Lung Segmentation Project

This file defines the implementation references that OpenCode should study before modifying the user's repository.

The goal is NOT to copy these repositories. The goal is to extract validated implementation patterns and adapt them to the current repository.

---

## Reference 1 — LungMask
Repository:
https://github.com/JoHof/lungmask

Role in our project:
- pretrained lung CT segmentation teacher
- teacher inference
- output semantics for right/left lung
- teacher model loading
- post-processing reference
- R231 and R231CovidWeb comparison

Important files:
- `lungmask/mask.py`
- `lungmask/resunet.py`
- README

Important behavior to inspect:
- model loading from pretrained weights
- `LMInferer`
- R231 versus R231CovidWeb
- output labels: background / right lung / left lung
- limitations for regular-image / non-HU inputs
- volume and connected-component post-processing

Do NOT:
- assume its prediction is target-domain ground truth
- silently treat non-HU JPEG inference as equivalent to HU CT inference

---

## Reference 2 — FPL+
Repository:
https://github.com/HiLab-git/FPL-plus

Paper:
Filtered Pseudo Label-based Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation, IEEE TMI 2024.

Role in our project:
This is the CLOSEST methodological reference found so far.

Why it is important:
- separate pseudo-label generator and final segmentor
- generates pseudo-labels before final student training
- derives uncertainty/reliability weights
- has pixel-level weighting
- compares predictions generated from related domain representations
- trains a final segmentor using filtered/weighted pseudo-label supervision

Important locations:
- `data/get_pixel_weight.py`
- configuration files under `config_dual`
- pseudo-label generator pipeline
- final segmentor pipeline
- `merge_pixelw.py`
- PyMIC training code

Critical overlap with our previous novelty idea:
`get_pixel_weight.py` compares two related pseudo-label predictions and assigns lower pixel weights to disagreement regions.

Therefore:
"cross-representation agreement -> pixel weight"
CANNOT be claimed by itself as our novelty.

What to reuse conceptually:
- separation of pseudo-label generation and student training
- frozen pseudo-label workflow
- pixel-weight map concept
- ablation logic
- provenance of reliability weights

What must be different:
- no CycleGAN/cross-modality translation required
- no access to labeled source target adaptation pipeline is assumed
- our target problem is non-HU representation loss in exported CT
- our teacher can be treated as a pretrained/black-box segmentation teacher
- reliability is estimated from same-anatomy intensity representations, teacher behavior, and conservative anatomical plausibility
- validation must directly test whether reliability score predicts actual pseudo-label error on a manually corrected or external-GT subset

---

## Reference 3 — UPL-SFDA
Repository:
https://github.com/HiLab-git/UPL-SFDA

Paper:
Uncertainty-aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation, IEEE TMI 2023.

Role:
Reference for source-free target adaptation and perturbation-based pseudo-label reliability.

Important files:
- `run_2d_upl.py`
- `unet.py`
- `loss.py`
- `utils.py`

Important implementation pattern:
- shared encoder with several auxiliary decoders
- multiple transformed/rotated views
- predictions mapped back to the same orientation
- mean prediction map
- thresholded pseudo-label
- largest-component filtering
- target training with pseudo-label Dice supervision
- entropy minimization

What to learn:
- transformation consistency implementation
- mapping predictions back into one coordinate system
- auxiliary-head diversity
- separating source model loading from target adaptation
- target-domain-only adaptation loop

Do NOT copy:
- its uncertainty formulation as our novelty
- its exact rotation strategy as a substitute for our non-HU intensity-shift problem

---

## Reference 4 — UC-Seg
Repository:
https://github.com/taozh2017/UCSeg

Paper:
Uncertainty-Aware Cross-Training for Semi-Supervised Medical Image Segmentation, IEEE TIP 2025.

Role:
Reference for modern uncertainty-aware pseudo-label generation.

Important locations:
- `UC-Seg_2D_github/UC_trainer.py`
- `UC-Seg_2D_github/module/`
- `UC-Seg_2D_github/utils/`
- 2D prediction scripts

Method concepts:
- two distinct subnets
- cross-subnet consistency
- Dirichlet/subjective-logic uncertainty
- adaptive high-confidence pseudo-label generation

Why this matters:
It demonstrates that generic:
- uncertainty-aware pseudo-labeling
- dual-subnet consensus
- consistency preservation
are already established.

Therefore these are baselines/components, not our novelty.

---

## Reference 5 — UPC-Pytorch
Repository:
https://github.com/AIforMS/UPC-Pytorch

Role:
Older, simpler reference for:
- uncertainty-aware pseudo-labeling
- consistency loss
- semi-supervised medical segmentation training

Important locations:
- `code/`
- `train_LA_UPC.py`
- dataloaders and evaluation scripts

Use:
Useful as a simpler engineering example if UC-Seg is too entangled for first implementation.

---

## Reference 6 — Medical-Image-Segmentation-Benchmarks
Repository:
https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks

Role:
PRIMARY reference for fair multi-architecture benchmarking.

Supported/reference models include:
- U-Net
- Attention U-Net
- U-Net++
- U-Net3+
- TransUNet
- MedT
- UNeXt
- SwinUnet
- CMU-Net
- CMUNeXt
- Mobile U-ViT

Important files/folders:
- `src/`
- `main.py`
- `infer.py`
- `split.py`

What OpenCode should borrow:
- model registry/factory idea
- same training entry point for different models
- unified CLI
- shared dataset interface
- parameter/FLOPs/FPS comparison
- model-independent evaluation

What OpenCode must improve for our project:
- patient/case-safe split instead of naive random slice split
- held-out test set
- per-image metrics
- HD95/ASSD
- pseudo-label versus real-GT metric separation
- fixed target masks for fair comparison

---

# Revised Novelty Position

## What has been ruled out as novelty
Do NOT claim any of these alone:
- U-Net variant
- Attention Gate
- ResNet encoder
- uncertainty-guided pseudo-labeling
- multi-teacher learning
- pseudo-label filtering
- cross-representation agreement
- pixel-level reliability weights
- consistency training
- anatomy post-processing

All have close precedents.

## Safer working novelty

Working concept:

**Black-box Teacher Reliability Calibration for Pseudo-Label Distillation under Non-HU CT Representation Loss**

Optional short name:
**BTRC-CT**

Core distinction:
We do not assume that a pretrained HU-CT teacher is trustworthy after irreversible intensity information loss caused by export to regular images.

We explicitly evaluate whether an external teacher is reliable for each target sample/pixel when:
- HU information is unavailable
- only non-HU image representation remains
- no target segmentation labels exist for most images
- no source training data are required
- no teacher architecture modification is required

The teacher may be treated as a black-box predictor.

### Proposed reliability evidence

1. Same-anatomy intensity representation stability
   - original non-HU export
   - mild linear normalization
   - mild gamma variations
   - mild CLAHE
   - mild contrast changes

2. Teacher-model agreement where available
   - R231
   - R231CovidWeb

3. Conservative anatomical plausibility
   - area ratio
   - component structure
   - fragmentation
   - border contact
   - centroid/laterality plausibility

4. Calibration against a small manually corrected target subset
   - THIS is critical
   - learn or tune reliability mapping ONLY on validation portion of gold subset
   - evaluate reliability-error correlation on held-out gold subset

### Key scientific test
The novel reliability score must predict real segmentation error.

Required analysis:
- correlation between reliability score and Dice-to-real-GT
- AUROC/AUPRC for identifying low-quality pseudo-labels
- calibration plot
- selective-risk curve:
  remove/downweight least reliable samples and observe GT segmentation error

This separates the proposal from methods that merely compute disagreement weights.

### Distillation
Use reliability to:
- accept
- downweight
- ignore

pseudo-label pixels/samples when training multiple student U-Net architectures.

### Main claim, only if experimentally supported
A reliability score specifically calibrated for non-HU representation loss can identify unreliable off-the-shelf CT lung pseudo-labels and improve pseudo-label distillation across multiple segmentation architectures.

---

# Recommended Development Order

1. Study LungMask implementation.
2. Study FPL+ pixel/image weighting pipeline.
3. Study UPL-SFDA transformation-consistency implementation.
4. Study UC-Seg uncertainty pseudo-label generator.
5. Study Medical-Image-Segmentation-Benchmarks architecture registry/training harness.
6. Audit current repository.
7. Build teacher feasibility pilot.
8. Build gold-subset support.
9. Validate whether proposed reliability signals actually predict teacher error.
10. ONLY IF step 9 works, implement reliability-gated distillation.
11. Benchmark multiple U-Net architectures.
12. Run ablations.
13. Evaluate downstream classifier.
14. Run robustness/XAI as a later experiment.

---

# Rule for OpenCode

Before implementing a component:
- identify which reference repository is closest;
- cite repository + exact file/function in the implementation notes;
- explain what is reused conceptually;
- explain what is intentionally different;
- never paste large portions of external source;
- respect repository licenses;
- write original implementation adapted to this project.
