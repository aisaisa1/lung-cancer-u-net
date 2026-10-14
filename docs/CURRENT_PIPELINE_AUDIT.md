# Current pipeline audit — Phase 1

Audit date: 2026-10-08. Baseline Git commit: `5cd6971` on `main`.
Repository: <https://github.com/aisaisa1/lung-cancer-u-net>.

## 1. Current pipeline and active source files

The primary segmentation model is **ResNet18 pretrained encoder + Attention-Gated
U-Net decoder**, not a plain U-Net. Phase 1 adds independent audit/inference scripts;
it does not invoke the existing end-to-end executor.

| Experiment/component | Active file | Function/class or entry point |
| --- | --- | --- |
| End-to-end legacy executor | `JALANKAN_SEMUA_PIPELINE.py` | `main`, `run_script`; runs training, crops, evaluation, classification, Grad-CAM in sequence |
| Segmentation architectures | `SCRIPTS/01_SEGMENTASI_UNET/modul_unet.py` | `ResNet18_UNet`, `UNet_Basic`, `UNet_Light`; `UNet = ResNet18_UNet` is an alias |
| Segmentation attention | same module | `AttentionGate`, `AttentionDecoderBlock` |
| Segmentation data | same module | `LungSegDataset`, `build_pairs`; pairs are image paths, not image/GT pairs |
| Otsu baseline pseudo-masks | same module | `LungSegDataset.generate_pseudo_mask`, `refine_lung_mask` |
| Segmentation training | `SCRIPTS/01_SEGMENTASI_UNET/01_train_unet.py` | `split_pairs`, `train_single_model`, `train_all_models` |
| Three-model ROI cropping | `SCRIPTS/01_SEGMENTASI_UNET/02_crop_dataset_unet.py` | `crop_image_with_mask`, `crop_dataset_for_model`, `main` |
| Otsu-agreement visualizations | `SCRIPTS/01_SEGMENTASI_UNET/02_visualisasi_3_unet.py` | `load_image_and_pseudo_mask`, `predict_mask`, `visualize_sample`, `main` |
| Legacy purported GT evaluation | `SCRIPTS/01_SEGMENTASI_UNET/03_evaluasi_unet_dice_iou.py` | `evaluate_model_variant`, `main` |
| Classification model/data | `SCRIPTS/02_KLASIFIKASI_MODEL/modul_klasifikasi.py` | `ResNet18_CBAM`, `CBAM`, `ClassificationDataset`, `load_dataset_files` |
| Classifier training/evaluation | `SCRIPTS/02_KLASIFIKASI_MODEL/01_train_klasifikasi.py`, `02_evaluasi_klasifikasi.py` | `train_classifier`, `evaluate_classifier` |
| Layer Grad-CAM | `SCRIPTS/03_UJI_LAYER_GRADCAM/01_uji_layer_gradcam.py` | `QuantusGradCAM`; hooks classifier CBAM1–4 |
| Noise robustness | `SCRIPTS/04_UJI_GRADCAM_NOISE/01_uji_gradcam_noise.py` | `GradCAM_Binary`, `add_gaussian_noise`; classifier CBAM4, CAM correlations |

Legacy flow:

```text
DATASET/Citra_Gamma_CLAHE
  -> 224x224 RGB
  -> on-the-fly Otsu pseudo-mask
  -> three segmentation models, validation against Otsu masks
  -> checkpoint selection using validation Otsu agreement
  -> per-model dataset crops
  -> separate binary classifier on primary-model crops
  -> classifier evaluation and Grad-CAM experiments
```

The new flow is separate:

```text
read-only dataset inventory -> native/least-processed export selection
-> frozen external teacher pilot + same-canvas intensity representations
-> diagnostics and pending human visual review
-> STOP (no student training or downstream experiments)
```

## 2. Old/duplicate and auxiliary code

- `OUTPUT_HASIL/visualisasi.py` is an ignored local duplicate of
  `SCRIPTS/01_SEGMENTASI_UNET/02_visualisasi_3_unet.py`, including its direct
  `from modul_unet import ...` import. Running it from the output folder normally
  needs an appropriate Python module path. It is not called by the executor.
- `auto/lungmask/` is a nested third-party Git checkout, ignored by explicit prior
  user decision. It is a reference, not an active project segmentation model.
  Its modern `LMInferer` lacks the old `noHU` argument. The pilot imports the
  isolated installed `lungmask==0.2.14`, not this checkout.
- `DATASET/Citra_Gamma_CLAHE1` and `DATASET/Mask_Manual_GT1` are small text files
  containing old machine-specific macOS paths. They are not image/mask directories
  or usable annotation datasets. They remain local and ignored.
- Existing untracked roadmap/prompt/reference Markdown and launcher files are
  user materials, not executed pipeline code.
- Gamma/CLAHE and Otsu preprocessing producer scripts are not present in the active
  source tree. Existing derived folders alone do not establish transformation
  parameters, execution history, or source provenance.
- No architecture definition was replaced, deleted, simplified, or copied into the
  teacher experiment. Otsu remains in its existing code as a legacy baseline;
  the legacy training entry point still uses it and must not be run for the new
  scientific supervision experiment without a later explicitly authorized change.

## 3. Actual ResNet18 Attention U-Net architecture

Source: `modul_unet.py:25–130`. `models.resnet18(weights='DEFAULT')` selects
ImageNet pretrained encoder weights when `pretrained=True` (the training default).

For 224x224, three-channel input:

| Encoder feature | Operations | Channels | Spatial size |
| --- | --- | ---: | --- |
| `x0` | `conv1` 7x7/stride2 -> `bn1` -> ReLU | 64 | 112x112 |
| pooled | ResNet maxpool | 64 | 56x56 |
| `x1` | `layer1` | 64 | 56x56 |
| `x2` | `layer2` | 128 | 28x28 |
| `x3` | `layer3` | 256 | 14x14 |
| `x4` | `layer4` | 512 | 7x7 |

Decoder:

| Block | Input / skip / output channels | Skip |
| --- | --- | --- |
| `dec4` | 512 / 256 / 256 | `x3` |
| `dec3` | 256 / 128 / 128 | `x2` |
| `dec2` | 128 / 64 / 64 | `x1` |
| `dec1` | 64 / 64 / 64 | `x0` |

Every decoder block uses transposed-convolution upsampling, **an attention gate on
the skip**, concatenation, and two 3x3 Conv–BN–ReLU layers. Gates project decoder
and skip features with 1x1 Conv–BN, align spatial sizes if needed, add and apply
ReLU, then 1x1 Conv–BN–sigmoid; the skip is multiplied by the scalar spatial map.
The output head is transposed convolution 64->32, BN/ReLU, and 1x1 convolution
32->1. The model returns binary logits; sigmoid is applied externally.

`UNet_Basic`: four downsamplings, 64->128->256->512->1024 channels, vanilla
concatenated skips and double convolutions. `UNet_Light`: 16->32->64->128 with
three downsamplings. These are separate architectures, not teacher models.

## 4. Actual pseudo-mask and preprocessing logic

`LungSegDataset.__getitem__` reads BGR, converts to RGB, and resizes the image to
224x224 **before** calling `generate_pseudo_mask`. That method:

1. Converts RGB to grayscale.
2. Applies Gaussian blur with 5x5 kernel.
3. Uses `cv2.threshold(..., THRESH_BINARY + THRESH_OTSU)`.
4. Inverts if the thresholded foreground has higher mean grayscale intensity,
   attempting to select dark areas.
5. Applies 5x5 elliptical morphological opening and closing.
6. Calls `refine_lung_mask`: fills binary holes, then 3x3 elliptical closing.
7. Returns a float32 binary pseudo-mask.

It explicitly does **not** load `Mask_Manual_GT`. Flips are applied jointly to image
and pseudo-mask. RGB intensities are scaled by 255 and ImageNet mean/std normalized.
There is no annotation-quality verification, background exclusion, or lung-specific
component/position constraint. Uniform images can produce empty intensity subsets
for the means; hole filling can fill regions enclosed by exterior background.
These are methodological failure modes, not anatomical labeling guarantees.

## 5. Current training split and validation/test usage

- `CLASSES` is only `Normal cases`, `Malignant cases`; Benign is excluded from the
  legacy segmentation/classification experiments despite being locally available.
- `split_pairs` randomly shuffles paths within each class, seed 42, and assigns
  approximately 70/15/15 percent **at image/slice level** using integer floor.
  For the current 416/561 image counts, code implies 685 train, 146 validation,
  and 146 test images. No split was executed for Phase 1.
- `test_pairs` is calculated but never used or persisted by segmentation training.
- Loss: 0.50 Lovasz hinge + 0.30 BCEWithLogits + 0.20 boundary contour MSE.
  AdamW LR 2e-4, weight decay 1e-4, cosine scheduling, batch size 16, 15 epochs.
- Validation thresholds sigmoid at 0.48 and pools pixels within a batch, then
  averages batch metrics equally. This is not per-patient or independent
  anatomical segmentation performance. Checkpoints maximize Otsu agreement IoU.
- Visualization uses threshold 0.5, first five sorted images per available binary
  class, and agreement against the same loader-generated Otsu masks.
- `03_evaluasi_unet_dice_iou.py` iterates all available images with same-name masks,
  **not the held-out test list**. It expects `Mask_Manual_GT`, but if absent falls
  back to a parent `Crop_Manual_Gamma_CLAHE` folder. A crop image is not necessarily
  a pixel annotation; `gt_mask > 20` cannot establish anatomical ground truth.
- Legacy cropping/evaluation may continue with random initialized weights if
  checkpoints are missing, and have parent-directory dataset fallbacks.
- Classification uses a separate stratified random image-level 70/15/15 split;
  its test split is saved. If missing, evaluation takes the first 100 images.
- Grad-CAM is classifier interpretability/robustness, not lung-mask evaluation.

## 6. Existing checkpoints and output evidence

Present locally under `OUTPUT_HASIL/` (ignored, neither loaded nor overwritten by
Phase 1):

- `best_unet_resnet18_attention_u_net.pt`
- `best_unet.pt` (primary legacy alias)
- `best_unet_u_net_basic.pt`
- `best_unet_u_net_light.pt`
- `best_classifier.pt`

Other existing artifacts include classifier split/history, Grad-CAM figures/CSV,
`training_histories_3_unet.json`, and `tabel_perbandingan_3_unet.csv`.
The segmentation comparison CSV currently has **blank Dice/IoU/time values**,
consistent with missing usable mask pairs. Existing training histories include
near-100% validation Otsu-agreement values; these do not demonstrate accurate lung
anatomy and need target/foreground sanity investigation. No old results were
regenerated or reinterpreted as newly measured results.

## 7. Required symbol search results

Exact code search covered the active root/SCRIPTS Python files and the ignored
local visualization duplicate. Virtual-environment dependency copies were not
treated as project implementations. Locations below refer to original source
line numbers; newly added audit documentation naturally repeats these names.

| Search term | Occurrences in existing project code |
| --- | --- |
| `generate_pseudo_mask` | `modul_unet.py:391,468` |
| `THRESH_OTSU` | `modul_unet.py:408` |
| `cv2.threshold` | `modul_unet.py:404` |
| `build_pairs` | `modul_unet.py:509`; `01_train_unet.py:13,120`; `02_visualisasi_3_unet.py:12,447`; local duplicate `OUTPUT_HASIL/visualisasi.py:12,447` |
| `split_pairs` | `01_train_unet.py:26,125` |
| `Citra_Gamma_CLAHE` | `modul_unet.py:510`; `01_train_unet.py:121`; `02_crop_dataset_unet.py:51`; `03_evaluasi_unet_dice_iou.py:82`; `02_visualisasi_3_unet.py:448`; local duplicate `visualisasi.py:448` |
| `Crop_UNet_Gamma_CLAHE` | executor `:36`; crop script `:106`; classifier training `:32`, evaluation `:41`, module `:114`; layer Grad-CAM `:98,100`; noise Grad-CAM `:95,97` |
| `best_unet.pt` | training `:110`; crop script `:41`; evaluation script `:104` |
| `ResNet18_UNet` | `modul_unet.py:75,81,318`; training `:13,136`; crop `:10,106`; evaluation `:13,93`; visualization `:8,122`; local duplicate `visualisasi.py:8,122` |

## 8. Potential methodological problems and Phase-1 conclusion

1. Untrusted Otsu masks used as both training targets and validation targets:
   circular supervision/selection, not independent anatomical evaluation.
2. No verified lung pixel-level annotations in the target export tree.
3. Class-local serial filenames do not establish patient identities. Related
   slices or exact duplicates may leak across legacy image-level splits.
4. Derived Gamma/CLAHE/crop versions lose intensity/resolution information and
   should not replace the least-processed teacher input.
5. Silent fallback to other folders/random weights can conceal invalid evaluation.
6. Legacy test subset is unused; full-dataset evaluation can include training data.
7. Existing documentation describing manual GT does not match the active loader
   or the local mask-folder availability.
8. Benign exclusion and preprocessing/threshold differences affect comparability.
9. HU-trained teachers can fail on windowed JPEG exports; teacher agreement and
   stability can be high on consistently incorrect background masks.
10. Simple agreement weighting is related to FPL+ and must not be claimed as novel.

Phase-1 findings and exact commands are in `docs/PHASE1_RESULTS.md` and
`docs/PHASE1_COMMANDS.md`. Human review is pending; no student, classifier,
Grad-CAM, new architecture, reliability loss, or full benchmark was run.
