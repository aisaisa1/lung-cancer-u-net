# Phase-2 teacher candidates and tested environment

Both official implementations were inspected. Project adapters call upstream
model classes; no student architecture was changed or vendor network copied into
the active project source.

| Teacher | Input type | Prompt required | Pretrained weights | Expected regular-2D compatibility | Limitations |
| --- | --- | --- | --- | --- | --- |
| SAM-Med2D | RGB uint8 2D image; official predictor | Human region box; each slot independent | Official README Google Drive `1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl`, `sam-med2d_b.pth` | Designed/fine-tuned on medical 2D images; no calibrated HU required by this API | Not lung-specific; boundaries/dense structures may be excluded; 256px encoder resolution; no GT/accuracy claim |
| MedSAM | Three-channel 2D image; official inference script's transient preprocessing | Human region box; each slot independent | Official README Drive folder; `medsam_vit_b.pth` ID `1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_` | Medical SAM fine-tuning; official script supports regular image files | Domain/windowing shifts remain; box placement affects segmentation; 1024-square resize stretches a rectangular export internally |

## Inspected implementation paths

SAM-Med2D: README, `test.py`, `segment_anything/build_sam.py`,
`predictor.py`, `utils/transforms.py`, `modeling/sam.py`. Its test harness can
construct prompts from labels; that harness was **not** used for this target.
Our adapter uses its official `SamPredictor` with exact human boxes, one mask per
independent call. Source commit: `bfd2b93b1158100c8abd81f61766a2de92c1c175`.

MedSAM: README, `MedSAM_Inference.py`, `segment_anything/build_sam.py`, `setup.py`.
Source commit: `d71e8a1a99ad751840a22a7fa3ecfb4166fb1488`.
The inference adapter follows the official cubic resize, uint8 conversion,
min-max normalization, box coordinate scaling, sigmoid, native probability
interpolation, and 0.5 threshold. These transformations affect only in-memory
model input arrays, never original files or saved manual prompt coordinates.

## Environment and hardware

Created isolated `.venv_phase2_teachers/`, Python 3.10.11, with
`requirements_phase2_teachers.txt`. Existing `budi/` and `.venv_teacher/` remain
unchanged. Packages: torch 2.5.1+cu121, torchvision 0.20.1+cu121, NumPy 1.26.4,
SciPy 1.14.1, scikit-image 0.24.0, OpenCV 4.10.0.84, Pillow 11.0.0.

The actual hardware is RTX 3060 12 GiB, driver 591.86, CUDA 12.1 working in the new
environment. Host RAM is approximately 31.7 GiB, with approximately 8 GiB free at
preflight. `pip check` passed. Models ran separately/sequentially; their shared
upstream package name `segment_anything` is isolated by one model per process.

| Resource measured | SAM-Med2D | MedSAM |
| --- | ---: | ---: |
| Official checkpoint bytes | 2,561,829,701 | 375,049,145 |
| Peak torch GPU allocation | 1,175,094,272 | 2,906,350,080 |
| Setup/loading seconds, full pilot | 9.244 | 3.099 |
| Mean encode + both decoder calls/image, seconds | 0.157 | 3.560 |

Runtime figures are synchronized GPU measurements excluding disk IO and QC;
they are feasibility measurements, not a formal hardware benchmark. First-image
warmup is included. Cached Phase-1 CPU LungMask times are not directly comparable.

The SAM-Med2D official checkpoint serializes an Adam optimizer as well as model
state. Loading uses mmap and narrowly scoped known torch/stdlib class support;
only the model state is copied strictly, the optimizer is never executed.
MedSAM uses its official strict checkpoint loader. No broad editable install of
upstream training/Jupyter/MONAI extras was needed.

Checkpoint content SHA256s (provenance, not an upstream-published trusted digest):

- SAM-Med2D: `921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18`
- MedSAM: `34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74`

Official repository checkouts and checkpoints remain inside ignored
`OUTPUT_HASIL/segmentation/.cache/`. Only the required weights were downloaded;
the upstream Drive folder was listed to locate the exact MedSAM checkpoint, with
`skip_download=True`; no demo or replacement target datasets were downloaded.

## Recommendations

Both candidates are **CONDITIONAL** for supervision until human mask review and
trusted annotation/correction are complete. Preliminary contact-sheet inspection
is encouraging, especially relative to default LungMask noHU, but no anatomical
PASS grades, accuracy, or winner were assigned from QC alone.
