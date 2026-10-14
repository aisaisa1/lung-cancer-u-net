# Phase-1 command record

Commands were run from the project root in Windows PowerShell. Parent directories
were checked with `Test-Path -LiteralPath` before commands creating environments
or output files. Source/data reads and edits used the harness's dedicated tools.
Git was inspected but no commit/push was made during Phase 1.

## Repository/runtime inspection

```powershell
git status --short --branch
git diff --stat
python --version
py -0p
& '.\budi\Scripts\python.exe' --version
& '.\budi\Scripts\python.exe' -m pip list
```

Dedicated content searches located all requested existing project symbols. Public
README/source/tag/config files were read with the web-fetch tool; exact references
are recorded in `REFERENCE_IMPLEMENTATION_MAP.md`. The optional guessed benchmark
`src/archs.py` URL returned 404 and its directory was resolved via GitHub API.

## Dataset and main environment audits

```powershell
python experiments/00_dataset_audit.py --smoke
python experiments/00_dataset_audit.py
python experiments/teacher_environment_audit.py --smoke
```

The main environment audit only imports/inspects packages; missing SimpleITK and
LungMask were recorded, not installed there.

## Isolated environment creation and install attempts

```powershell
py -3.10 -m venv .venv_teacher
# Failed initial attempt (pip 23.0.1/dependency metadata/build issue):
& '.\.venv_teacher\Scripts\python.exe' -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
# Fix:
& '.\.venv_teacher\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv_teacher\Scripts\python.exe' -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# Failed with the initial fill-voids==2.0.1 source archive:
& '.\.venv_teacher\Scripts\python.exe' -m pip install -r requirements_teacher.txt
& '.\.venv_teacher\Scripts\python.exe' -m pip index versions fill-voids
# After correcting the pin to fill-voids==2.1.2:
& '.\.venv_teacher\Scripts\python.exe' -m pip install --only-binary=:all: -r requirements_teacher.txt
& '.\.venv_teacher\Scripts\python.exe' -m pip check
& '.\.venv_teacher\Scripts\python.exe' experiments/teacher_environment_audit.py --output-dir OUTPUT_HASIL/segmentation/audit/environment_teacher --smoke
```

## Real smoke and 45-image pilot

```powershell
& '.\.venv_teacher\Scripts\python.exe' experiments/01_teacher_feasibility.py --smoke --cpu --download-weights
& '.\.venv_teacher\Scripts\python.exe' experiments/01_teacher_feasibility.py --cpu --samples-per-class 15
# Previous command stopped at shell timeout after 42 complete images:
Get-Process -Name python -ErrorAction SilentlyContinue | Format-Table Id, CPU, StartTime, Path -AutoSize
# After adding validated continuation:
& '.\.venv_teacher\Scripts\python.exe' experiments/01_teacher_feasibility.py --cpu --samples-per-class 15 --resume
```

Only official teacher weights were downloaded, into the ignored output cache.
The process query confirmed the timed-out teacher process was no longer running;
unrelated user Python processes were left alone.

## Verification and failure-path smoke

```powershell
& '.\.venv_teacher\Scripts\python.exe' experiments/verify_phase1_outputs.py
& '.\.venv_teacher\Scripts\python.exe' -m unittest discover -s tests -v
python experiments/01_teacher_feasibility.py --smoke --cpu --output-dir OUTPUT_HASIL/segmentation/teacher_pilot/smoke_missing_teacher
& '.\.venv_teacher\Scripts\python.exe' experiments/verify_phase1_outputs.py --smoke --output-dir OUTPUT_HASIL/segmentation/teacher_pilot/smoke_missing_teacher
# Final suite after adding the single-teacher-failure integration check: six passed.
& '.\.venv_teacher\Scripts\python.exe' -m unittest discover -s tests -v
git diff --check
git diff --exit-code -- SCRIPTS JALANKAN_SEMUA_PIPELINE.py README.md requirements.txt install_requirements.py .gitignore
git status --short --branch
git check-ignore -v -- .venv_teacher/pyvenv.cfg OUTPUT_HASIL/segmentation/teacher_pilot/teacher_summary.json OUTPUT_HASIL/segmentation/.cache/torch/checkpoints/unet_r231-d5d2fc3d.pth DATASET/'The IQ-OTHNCCD lung cancer dataset'/'Normal cases'/'Normal case (26).jpg'
```

The missing-teacher smoke deliberately uses the unchanged main interpreter,
verifying that both unavailable teachers are recorded without fabricated masks or
agreement metrics. Unit-test predictions are synthetic mathematical fixtures,
never included in measured teacher results.

## Reproduction/output preservation

All new experimental scripts accept `--smoke`. The audit samples two files per
source/class for expensive dimensions/hashing, while reporting full filename
counts. Teacher smoke uses one image per class and both teachers/all six views.
Environment audit is always import-only. Verification smoke validates at most
three image IDs/source hashes. Completed output reports are not silently replaced;
choose a new `--output-dir` for another run. `--resume` is for interrupted runs.

The existing primary model, training entry points, source dataset and other user
files remain intact. Generated reports include local paths and stay under the
ignored `OUTPUT_HASIL/`; portable code/docs/specifications contain relative paths.
