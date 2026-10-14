# Phase-3 additional inference and 30-case correction package

## Validation and inference

The additional session `phase3_additional_prompts` validated **15/15 images** and
**30/30 VISIBLE boxes**, reviewer `aisya`, all positive-area integer rectangles
inside each native image. PENDING and NOT_VISIBLE regions: **0** each. Native
sizes: 12 at 512x512 and one each at 506x331, 623x512 and 511x404.
CSV/JSON match and source SHA256s remained unchanged. Original Phase-2 annotations
were not reused for these new images and were not overwritten.

The exact existing `experiments/phase2_teacher_inference.py` ran in the existing
isolated GPU environment, with explicit additional prompt/output directories.
Teacher architectures, checkpoints, preprocessing and thresholds matched the
original session. Each visible box had an independent decoder call; masks were
returned to native coordinates before pixelwise union. No clipping, Otsu,
LungMask substitution, morphology or teacher retraining was applied.

| Teacher | Additional native successes | Technical failures | Nonempty unions |
| --- | ---: | ---: | ---: |
| SAM-Med2D | 15/15 | 0 | 15/15 |
| MedSAM | 15/15 | 0 | 15/15 |

All **30 individual masks per teacher** are retained in
`OUTPUT_HASIL/segmentation/phase3_teacher_comparison/<teacher>/individual_masks/`.
Merged outputs, overlays, native full-frame ROIs, box geometry, prediction CSV and
summary JSON are in the same teacher-specific directories.

## Consolidated pool

Master manifest:
`OUTPUT_HASIL/segmentation/gold_subset_candidates_30.csv`.

It contains the requested columns: sample_id, source_path, class, native_width,
native_height, annotation_session, SAM_Med2D_mask_path, MedSAM_mask_path,
SAM_Med2D_success, MedSAM_success and notes. Candidate paths refer to their original
source run; annotation_session distinguishes `phase2_prompts` and
`phase3_additional_prompts`. All 30 distinct decoded images are retained: 10 per
class. Both teachers have **30/30 successful complete candidate masks**.

Combined sizes: 26 at 512x512, one each at 801x512, 506x331, 623x512, 511x404.
**120 independent box transforms** and **60 exact native unions** were verified
across both sessions. Original Phase-2 file hashes were snapshotted and checked
after consolidation/export; no source/old annotation or old output was changed.

Contact sheets:
`OUTPUT_HASIL/segmentation/gold_subset_candidates_30/contact_sheets/<sample_id>_comparison.png`.
Their six native-sized panels are Original | Manual Boxes | SAM-Med2D candidate |
SAM-Med2D Overlay | MedSAM candidate | MedSAM Overlay. No accuracy was computed,
and technical success is not an anatomical PASS. Every candidate still needs
genuine human correction/review before becoming a usable reference mask.

Useful additional-case review sheets:

- `normal_f351ab88323a_comparison.png` — Normal case (344), 506x331.
- `malignant_495bc5732a7f_comparison.png` — Malignant case (234), 623x512.
- `malignant_37c10a275c18_comparison.png` — Malignant case (180), 511x404, ambiguous extent.
- `malignant_d9c0bee29639_comparison.png` — Malignant case (458), dense abnormal regions.
- `normal_a1d74535ecfa_comparison.png` / `benign_2af77330f0e0_comparison.png` — noise/texture cases.

## Gold correction package

Prepared `DATASET/Lung_Gold_Subset/` with 30 byte-identical source JPEG copies,
30 candidate PNGs per teacher, 30 overlay previews per teacher, immutable package
manifest, source/session/checkpoint provenance, manual metadata template and
protocol. Source filenames are preserved; e.g. image `Normal case (155).jpg`
pairs with candidate/manual name `Normal case (155).png`.

`masks_manual/` is **empty**. No manual mask or anatomical review grade was
fabricated. The metadata starts at `NOT_STARTED`, with annotator, reviewer,
candidate choice and correction notes blank.

Post-export validation passed: 30 native originals, 30 candidates per teacher,
0 manual masks, and 392 protected original Phase-2 files verified unchanged.
`manual_complete=false` and `reviewed_complete=false` are the expected state of
an annotation preparation package, not an inference/export failure.
The report is `OUTPUT_HASIL/segmentation/gold_subset_candidates_30/gold_package_validation.json`.

The exact correction procedure and binary/native mask contract are documented in
`docs/GOLD_SUBSET_ANNOTATION_PROTOCOL.md` and copied into the package metadata.
Recommended next step is human correction using an existing editor such as GIMP,
followed by expert review and format/provenance validation. Do not overwrite the
immutable image/candidate copies; export corrected labels to `masks_manual/`.

## Commands executed (PowerShell, project root)

```powershell
python tools/validate_phase2_annotations.py --prompts-dir OUTPUT_HASIL/segmentation/phase3_additional_prompts --output-dir OUTPUT_HASIL/segmentation/phase3_teacher_comparison/validation
python tools/prepare_phase3_additional_prompts.py --smoke
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher SAM-Med2D --prompts-dir OUTPUT_HASIL/segmentation/phase3_additional_prompts --output-dir OUTPUT_HASIL/segmentation/phase3_teacher_comparison/SAM-Med2D
& '.\.venv_phase2_teachers\Scripts\python.exe' experiments/phase2_teacher_inference.py --teacher MedSAM --prompts-dir OUTPUT_HASIL/segmentation/phase3_additional_prompts --output-dir OUTPUT_HASIL/segmentation/phase3_teacher_comparison/MedSAM
python tools/build_gold_subset_candidates.py
python tools/export_gold_subset.py
python -m unittest discover -s tests -p "test_gold_subset.py" -v
python tools/validate_gold_subset.py
```

Five annotation-package tests passed: byte-identical export, empty manual folder,
source/session integrity, refusal to overwrite annotation work, wrong geometry
rejection, human record/review requirements and native-size box scaling.
Tests use separate synthetic fixtures, never research manual masks.

## Stop

This handoff stops after validating additional prompts, running the two teachers
on the additional 15 images, consolidating 30 cases and exporting the correction
package. No U-Net training or full-dataset pseudo-label generation was performed.
The existing primary ResNet18 Attention U-Net is unchanged.
