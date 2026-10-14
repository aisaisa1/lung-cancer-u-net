# Reference implementation map — Phase 1

Public implementations were inspected before the Phase-1 scripts were written.
The added code is original project-specific audit/IO/diagnostic code. Teacher
inference calls the installed LungMask implementation instead of copying its
network or non-HU algorithm. Related-method concepts are credited, not novel claims.

| Component | Current project implementation | Reference repository | Reference file/function | Concept reused | What is different in our implementation |
| --- | --- | --- | --- | --- | --- |
| Teacher inference | `experiments/01_teacher_feasibility.py:load_teacher,predict` | [LungMask](https://github.com/JoHof/lungmask) | `README.md`; `lungmask/mask.py:MODEL_URLS,get_model,LMInferer`; `lungmask/resunet.py:UNet`; legacy tag `v0.2.14` | Frozen pretrained R231/R231CovidWeb, native-coordinate label mapping, binary union of lung labels | Uses supported legacy `noHU=True` explicitly on single JPEG exports, checks the actual installed API and official weight hash prefixes, records rejected intensity sweeps; never treats 0–255 as calibrated HU |
| Pseudo-label generation | Legacy `LungSegDataset.generate_pseudo_mask`; new pilot saves candidate teacher masks and provenance | [FPL+](https://github.com/HiLab-git/FPL-plus) | `README.md` pseudo-label-generator/final-segmentor pipeline; `config_dual/data_vs/vs_t1s_g.cfg`, `vs_t1s_g_fake.cfg`, `vs_t1s_S.cfg` | Separate generation/inference artifacts from eventual student optimization; retain frozen manifests | External pretrained lung teachers, target exports only, no source generator/CycleGAN/DSBN training and no final student segmentor in Phase 1 |
| Pseudo-label filtering | Technical QC CSV and `human_visual_review.csv`; no mask accepted automatically | FPL+ | `data/get_pixel_weight.py`; `merge_pixelw.py`; README image-level weight workflow | Reliability can involve transformed/native prediction disagreements; filtering is a distinct step | No pixel weights or filtering loss implemented. QC is descriptive and visual-review decisions remain pending. FPL+ already gives related agreement-derived pixel weighting (disagreement downweighting); simple agreement weighting is not claimed novel |
| Representation consistency | `representations`, `agreement`, `representation_pairwise.csv`, `representation_stability.csv` | [UPL-SFDA](https://github.com/HiLab-git/UPL-SFDA) | `run_2d_upl.py`; `unet.py:UNet.forward,save_nii,trian_target`; `utils.py:rotate_single_with_label`; `loss.py:DiceLoss,diceLoss_weight` | Compare predictions in a shared anatomical coordinate system; track prediction consistency | Six mild intensity-only views with identical stored canvas, same frozen teacher; UPL rotates inputs, inverse-rotates multiple decoder outputs, constructs thresholded pseudo-labels and optimizes target models. No rotations, auxiliary decoders, target training, entropy loss or imported UPL loss here |
| Uncertainty handling | Disagreement maps plus explicit missing/empty predictions; no calibrated uncertainty model | [UC-Seg](https://github.com/taozh2017/UCSeg) | `README.md`; `UC-Seg_2D_github/module/generate_pseudo_map_attention_2D.py:Generate_pseudo_Transformer`; `utils/UMIS.py:infer,pseudo_infer,dce_evidence_u_loss` | Confidence/uncertainty and pseudo-label acceptance must be distinguished from hard prediction values | UC-Seg uses subnet evidence/Dirichlet-style uncertainty and learned cross-attention pseudo generation. Here maps are empirical hard-mask disagreement only, never evidential/calibrated uncertainty; no UC-Seg training or learned pseudo head |
| U-Net architecture registry | Existing `models_to_train`, `models_info`, `create_models`; `modul_unet.py` contains the three existing architectures | [Medical Image Segmentation Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks) | `README.md` network collection/model CLI; `src/network/` directory | Future unified architecture selection/comparison interface | Reference only in Phase 1. Primary ResNet18 Attention U-Net is retained. No U-Net++, U-Net3+, registry replacement or new benchmark model created |
| Training harness | Existing `01_train_unet.py:train_all_models,train_single_model` | FPL+; Medical Image Segmentation Benchmarks | FPL+ README and generator/final configs; benchmark README `main.py --model ...` usage | Keep stages and reproducibility configurations explicit | Existing harness untouched; teacher pilot has no optimizer/backward call and imports no student model. Training remains outside Phase 1 |
| Evaluation | Legacy segmentation evaluator; new technical QC and teacher/representation agreement | LungMask; UPL-SFDA; Medical Image Segmentation Benchmarks | LungMask README output semantics, `utils.py:postprocessing`; UPL `run_2d_upl.py` held-out validation; benchmark README evaluation interface | Named output semantics, deterministic diagnostics, separate reference-based versus reference-free metrics | No trusted target GT means no accuracy, anatomical Dice or Grad-CAM segmentation score. Failure denominators, empty-mask flags, partial-view coverage and pending review are explicit |
| Data splitting | Legacy `split_pairs` / classifier `train_test_split`; new audit reports unverified identity and pilot samples only | UPL-SFDA; Medical Image Segmentation Benchmarks | `run_2d_upl.py:get_data_loader`; `utils.py:niiDataset`; benchmark `split.py` | Use explicit manifests and distinguish train/validation/test inputs | No split created. Benchmark's random image-ID split is deliberately not adopted for patient-linked CT. Patient/case mapping and exact duplicate handling must precede any future patient-disjoint split |

## Inspected reference files and reproducibility anchors

- **LungMask**: public `master` README, `mask.py`, `resunet.py`; legacy
  [`v0.2.14/mask.py`](https://github.com/JoHof/lungmask/blob/v0.2.14/lungmask/mask.py),
  `utils.py`, `setup.py`. Installed version 0.2.14 is recorded in each run.
  Official R231 weights SHA256:
  `d5d2fc3df432115933491f115b283e94d1f5c841c0e82f6e7bd2d65d3ccde69f`;
  CovidWeb:
  `0de78a7e829a114f5c4c6d67810daaaf2e66b82492de79e820d5b1ea6145a271`.
- **FPL+**: tree inspected at `29575002e774bf25058fc4ecffefa5c34d178ddf`;
  README, pixel-weight scripts, generator and final-segmentor configs, `run.sh`.
  README commands omit the `data_vs/` directory found in the current tree;
  `run.sh` references an older config. Some provided flags disable pixel weights;
  one cannot assume every supplied config activates the full published method.
- **UPL-SFDA**: tree at `1f62b92052b4c18cf4add0fea39f23f845ff257e`;
  all four requested Python files inspected. `trian_target` is the upstream spelling.
- **UC-Seg**: tree at `cd5b9a88d9c597543b9a42ea7158d10ed7817612`;
  README, 2D pseudo-map module, and `UMIS.py` evidence/uncertainty routines inspected.
- **Medical Image Segmentation Benchmarks**: README, `split.py`, `src/` and
  `src/network/` listings inspected. It benchmarks labeled BUSI, not IQ-OTH/NCCD
  exports. Its random image-level splitting is not evidence of appropriate CT
  patient-disjoint splitting. A guessed `src/archs.py` URL was 404; the network
  directory was resolved through the repository API instead.

## Non-HU limitations inherited from LungMask

The modern README explicitly restricts JPEG/PNG `noHU` support to 0.2.5–0.2.14.
In v0.2.14 the native export is converted to grayscale, resized internally to
256x256, scaled through 20 factors from 0.3 to 2, clipped, and filtered by a
brightness-area heuristic. A view with maximum label-1/right-lung area is selected
after inference/postprocessing and mapped back with nearest-neighbor interpolation.
This is not HU reconstruction. Its synthetic intensity stack is not a physical
CT volume, and its adaptive selection confounds pure external representation
stability. These properties are measured/reported, not silently patched away.

High agreement between two teachers or six representations can occur when all
predictions are wrong. Human review and future trusted reference annotations are
needed before any student supervision or reliability-method claim.
