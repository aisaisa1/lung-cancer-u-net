# Target-domain whole-lung correction protocol

## Package status and target

`DATASET/Lung_Gold_Subset/` is an **annotation preparation package**, not existing
ground truth. Its two teacher candidates per image are proposals. `masks_manual/`
starts empty and must contain only genuinely human-corrected/drawn masks later.

Target: **whole lung field**, merging both visible lung fields into one binary
foreground. Include dense pathological tissue/tumor within the lung field rather
than removing it just because it is bright. Exclude surrounding body/background
and large mediastinal structures outside the lung field. Preserve abnormal anatomy,
including asymmetric extent. When anatomy/extent is ambiguous, record
`NEEDS_EXPERT_REVIEW` instead of pretending certainty.

No left/right orientation is inferred from these JPEGs. Prompt box_1/box_2 IDs
are arbitrary regions. They are not pixel annotations or anatomical truth.

## Package layout and traceability

```text
DATASET/Lung_Gold_Subset/
    images/                              # 30 byte-identical native JPEG copies
    candidate_masks/SAM-Med2D/            # 30 immutable teacher candidate PNGs
    candidate_masks/MedSAM/               # 30 immutable teacher candidate PNGs
    masks_manual/                        # empty until human correction
    metadata/
        package_manifest.csv
        manual_annotation_status.csv
        gold_subset_candidates_30.csv
        candidate_provenance.json
        consolidation_summary.json
        export_report.json
        ANNOTATION_PROTOCOL.md
        manual_freeze_manifest.json      # hash manifest for frozen manual masks
        prompt_sessions/phase2_prompts/
        prompt_sessions/phase3_additional_prompts/
        overlay_previews/SAM-Med2D/
        overlay_previews/MedSAM/
```

Original filenames are preserved: `Normal case (155).jpg` pairs with candidate and
manual mask filename `Normal case (155).png`. The package manifest maps every file
to sample_id, original source_path/class, native dimensions, annotation session,
original candidate path and content SHA256. There are 10 images per class; these
30 slices are not assumed to be 30 independent patients. No split is created.

Native width x height: 26 at 512x512, and one each at 801x512, 506x331, 623x512,
511x404. Do not resize, reorient, enhance or sharpen original/copy images. The
editable label is the mask, never source pixels. Do not overwrite candidate inputs.

## Freeze/integrity mechanism

A **freeze manifest** (`metadata/manual_freeze_manifest.json`) records the SHA256
hash of each manual mask at the moment it is frozen. This creates an immutable
audit trail:

- After the annotator saves a manual mask to `masks_manual/<source_stem>.png` and
  updates the status to `ANNOTATED_PENDING_REVIEW` (or `NEEDS_EXPERT_REVIEW`),
  they run the freeze command to record the mask's hash.
- The freeze manifest is append-only; once a sample_id has a hash entry, that
  hash must never change. Validation will fail if a frozen mask is modified.
- **Candidate masks (SAM-Med2D/MedSAM) must remain hidden from the annotator and
  reviewer until the manual mask reaches final `REVIEWED` state and is frozen.**
  This ensures the manual reference mask is created and reviewed independently
  from the original image only.

Freeze command:

```powershell
& ".\budi\Scripts\python.exe" "tools\validate_gold_subset.py" --freeze
```

## Recommended correction tool: GIMP (offline, native raster workflow)

Use a standard existing editor rather than a new custom painting interface.
GIMP supports aligned native-size image/mask layers, undo/history and explicit
binary PNG export. CVAT/Label Studio can also be used if an annotation team has an
existing installation; final native PNGs must satisfy the same contract below.
No editor was installed or images uploaded automatically in this phase.

For each sample:

1. Open its `images/<original filename>.jpg` in GIMP. Inspect at 100% or greater;
   Fit zoom is only a display setting. Keep the canvas/native dimensions unchanged.
2. **Do not open candidate masks.** Draw the whole-lung mask from the original
   image only. Use hard, non-antialiased painting/erasing or binary selections.
   Foreground must be white (255), background black (0).
3. Save an editable `.xcf` project **outside `masks_manual/`**, for example in an
   annotator workspace. Record its location if desired in notes. Do not edit the
   package's immutable original/candidate copies.
4. For the final mask export, hide the source/other layers; keep only the corrected
   mask at full opacity, black background. Export an 8-bit grayscale PNG, values
   exactly 0/255, to `masks_manual/<source stem>.png`. Verify that the JPG anatomy
   or colored overlay was not accidentally included. Keep width/height unchanged.
5. In `metadata/manual_annotation_status.csv`, set `candidate_teacher` to
   `FROM_SCRATCH` (teacher candidates were not used for annotation); enter your
   `annotator` identity, set `ambiguity_flag` to `true` if anatomy/extent is
   uncertain, notes describing any decisions, and status `ANNOTATED_PENDING_REVIEW`.
6. **Freeze the mask:** run the freeze command to record the manual mask's SHA256
   in `metadata/manual_freeze_manifest.json`. This locks the mask content.
7. A human expert/reviewer inspects the frozen mask against the **original image
   only** (candidate masks remain hidden). Record their identity in `reviewer`
   and status `REVIEWED` only after that actual review. If ambiguity is found,
   use `NEEDS_EXPERT_REVIEW` and notes; do not automatically mark PASS.
8. **After final `REVIEWED` state and freeze**, SAM-Med2D and MedSAM candidates
   may be opened for EVALUATION/COMPARISON ONLY. They must NEVER be used to
   modify/refine a reference mask that will be used to evaluate those teachers.
   If comparison reveals an unresolved anatomical ambiguity, mark
   `NEEDS_EXPERT_REVIEW` instead of editing the mask based on teacher output.

Each status row includes sample_id, source_path, class, candidate_teacher,
manual_mask_status, annotator, reviewer, ambiguity_flag and notes. The initial
`NOT_STARTED` state means no manual mask exists; identities, candidate choice
(always `FROM_SCRATCH`), ambiguity_flag and notes start blank. Teacher masks
remain candidates even if technically nonempty and plausible.

## Validation and next handoff

Check the package at any time:

```powershell
& ".\budi\Scripts\python.exe" "tools\validate_gold_subset.py"
```

After all human masks and expert review are completed:

```powershell
& ".\budi\Scripts\python.exe" "tools\validate_gold_subset.py" --require-complete --require-reviewed
```

Validation checks 30 mapped cases, 10 per class, byte-identical original copies,
immutable candidate/overlay hashes, native mask dimensions, binary values, filename
mapping, nonempty foreground, annotator/reviewer identities, ambiguity_flag, correction
notes, and freeze manifest integrity. It reports missing, malformed, unmapped or
review-pending masks; it does not resize, threshold-repair or create masks. This
validates format/provenance, not anatomical accuracy. Ambiguous or intentionally
empty regions require explicit adjudication before being considered a usable
reviewed reference.

Return the validation report for the next explicit research instruction. No U-Net
training, full-dataset pseudo-label generation, classification, Grad-CAM or BTRC-CT
implementation is initiated by preparing, correcting or validating this package.
