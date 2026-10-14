# Teacher environment audit

## Main working environment (inspected, not changed)

| Component | Observed |
| --- | --- |
| Environment | existing `budi/` |
| Python | 3.12.10 |
| torch / torchvision | 2.13.0 / 0.28.0 |
| CUDA | unavailable; torch compiled without CUDA |
| numpy / scipy | 2.5.2 / 1.18.1 |
| SimpleITK / installed lungmask | absent |
| scikit-image / OpenCV | 0.26.0 / 5.0.0.93 |

The local `auto/lungmask/` checkout is not an installed distribution and its
current API does not expose `noHU`. Do not pass JPEG intensity values to the
modern HU-volume inference API or silently manufacture calibrated HU values.

Upstream only documents JPEG/PNG noHU support from 0.2.5 through 0.2.14. A separate
environment isolates the historical API and older numerical/image libraries.
This does not mean main Python 3.12 is proven inherently incompatible with every
historical package; the isolation avoids changing the working project runtime.

## Isolated teacher environment (created and tested)

Portable specifications: `environment_teacher.yml`, `requirements_teacher.txt`.
Actual local environment: `.venv_teacher/` (already excluded by `.gitignore`).

| Component | Tested version |
| --- | --- |
| Python | 3.10.11 |
| torch / torchvision | 2.5.1+cpu / 0.20.1+cpu |
| numpy / scipy | 1.26.4 / 1.14.1 |
| SimpleITK | 2.4.0 |
| lungmask | 0.2.14 |
| scikit-image / OpenCV | 0.24.0 / 4.10.0.84 |
| pillow | 11.0.0 |
| fill-voids | 2.1.2 (Windows wheel) |
| CUDA | unavailable; inference tested on CPU |

`LMInferer(..., noHU=False, ...)` signature is present and both candidate model
names are registered. The pilot explicitly passes `noHU=True`, freezes parameters,
uses eval/inference mode, batch size 4, and four torch CPU threads.
`pip check` reported no broken requirements. This is a technical compatibility
result, not proof that teacher masks are anatomically trustworthy.

## Installation commands (Windows PowerShell)

```powershell
py -3.10 -m venv .venv_teacher
& ".\.venv_teacher\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv_teacher\Scripts\python.exe" -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
& ".\.venv_teacher\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements_teacher.txt
& ".\.venv_teacher\Scripts\python.exe" -m pip check
```

Conda is an alternative specification, not a second environment created in this
session: `conda env create -f environment_teacher.yml` from the project root.
The requirements pin primary packages; `pip` resolves transitive dependencies.
Use the per-run environment JSON and weight hashes for the measured pilot's provenance.

## Compatibility/install problems actually encountered

1. The newly created venv initially had pip 23.0.1. CPU-index-only installation
   rejected normalized `typing_extensions` metadata and then failed to obtain
   `flit_core`. Fixed by upgrading pip **inside the isolated venv** and allowing
   PyPI for dependencies with the CPU wheel index as an additional index.
2. An initial `fill-voids==2.0.1` pin selected a source archive; its build dependency
   stage could not import numpy. Changed the teacher specification to the tested
   `fill-voids==2.1.2` Windows wheel and installed binary distributions only.
3. The legacy noHU implementation can reject all 20 views and would otherwise
   index an empty array. The pilot detects the same upstream filter condition
   explicitly and records it as a failed inference, without changing thresholds.

Official teacher checkpoints (124,340,403 bytes each) were downloaded to ignored
`OUTPUT_HASIL/segmentation/.cache/torch/checkpoints/`, verified against upstream
filename hash prefixes, and never added to Git. No target dataset was downloaded.

Local machine-specific inspection outputs:

- `OUTPUT_HASIL/segmentation/audit/environment/teacher_environment.json`
- `OUTPUT_HASIL/segmentation/audit/environment_teacher/teacher_environment.json`
- `OUTPUT_HASIL/segmentation/teacher_pilot/teacher_environment.json`

These remain ignored generated outputs, rather than portable project configuration.
