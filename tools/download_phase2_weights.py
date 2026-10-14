"""Download only the two official pretrained checkpoints needed for this pilot."""

import argparse
import json
from pathlib import Path

import gdown

from draw_lung_boxes import PROJECT_ROOT, atomic_text, digest_file

OFFICIAL = {
    'SAM-Med2D': ('1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl', 'sam-med2d_b.pth',
                  'https://github.com/OpenGVLab/SAM-Med2D#-results'),
    'MedSAM': ('1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_', 'medsam_vit_b.pth',
              'https://github.com/bowang-lab/MedSAM#Get-Started'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', choices=OFFICIAL, required=True)
    parser.add_argument('--cache-dir', type=Path, default=PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/.cache/phase2_weights')
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    dataset = (PROJECT_ROOT / 'DATASET').resolve()
    if cache == dataset or dataset in cache.parents:
        parser.error('Checkpoint cache must not overlap the dataset')
    cache.mkdir(parents=True, exist_ok=True)
    file_id, filename, reference = OFFICIAL[args.teacher]
    path = cache / filename
    if not path.exists():
        partial = path.with_suffix('.pth.part')
        result = gdown.download(id=file_id, output=str(partial), quiet=False, resume=True)
        if result is None or partial.stat().st_size < 100_000_000:
            raise RuntimeError('Official checkpoint download failed or returned an unexpected small file')
        partial.replace(path)
    provenance = dict(teacher=args.teacher, source_reference=reference, google_drive_file_id=file_id,
                      filename=filename, file_bytes=path.stat().st_size, sha256=digest_file(path),
                      hash_note='Computed content SHA256 for provenance; upstream does not publish a trusted full digest here.')
    atomic_text(cache / f'{args.teacher}_provenance.json', json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
