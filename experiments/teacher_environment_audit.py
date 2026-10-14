"""Inspect actual import/API compatibility without installing or changing packages."""

import argparse
import importlib
from importlib import metadata
import inspect
from pathlib import Path
import platform
import sys

from phase1_common import PROJECT_ROOT, prepare_output, write_json


def inspect_environment():
    report = dict(python=platform.python_version(), executable=sys.executable,
                  platform=platform.platform(), packages={}, cuda={}, lungmask_api={})
    for module_name, distribution in (('torch', 'torch'), ('torchvision', 'torchvision'),
                                      ('numpy', 'numpy'), ('scipy', 'scipy'),
                                      ('SimpleITK', 'SimpleITK'), ('lungmask', 'lungmask'),
                                      ('skimage', 'scikit-image'), ('cv2', 'opencv-python')):
        item = {}
        try:
            item['version'] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            item['version'] = None
        try:
            module = importlib.import_module(module_name)
            item.update(import_ok=True, module_path=str(module.__file__))
            if module_name == 'torch':
                report['cuda'] = dict(available=module.cuda.is_available(),
                                      compiled_cuda=module.version.cuda,
                                      device_count=module.cuda.device_count(),
                                      devices=[module.cuda.get_device_name(i)
                                               for i in range(module.cuda.device_count())])
            if module_name == 'lungmask':
                mask = importlib.import_module('lungmask.mask')
                signature = inspect.signature(mask.LMInferer.__init__)
                report['lungmask_api'] = dict(signature=str(signature),
                                              noHU_supported='noHU' in signature.parameters,
                                              available_models=list(mask.MODEL_URLS))
        except Exception as error:
            item.update(import_ok=False, error=f'{type(error).__name__}: {error}')
        report['packages'][module_name] = item
    ready = all(report['packages'][p]['import_ok'] for p in
                ('torch', 'torchvision', 'numpy', 'scipy', 'SimpleITK', 'lungmask', 'skimage', 'cv2'))
    report['non_hu_teacher_ready'] = ready and report['lungmask_api'].get('noHU_supported', False)
    report['compatibility_notes'] = [
        'LungMask README limits noHU support to versions 0.2.5 through 0.2.14.',
        'Modern LMInferer.apply expects HU CT volumes; passing JPEG intensities as HU is invalid.',
        'Legacy noHU uses a 20-intensity sweep and heuristic view selection, not HU recovery.',
        'Use environment_teacher.yml / requirements_teacher.txt in a separate environment.',
        'No package installation, model loading, or main-environment changes performed by this audit.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Audit imports only (also the normal behavior)')
    args = parser.parse_args()
    output = args.output_dir or PROJECT_ROOT / 'OUTPUT_HASIL/segmentation/audit/environment'
    output = prepare_output(output, PROJECT_ROOT / 'DATASET', 'teacher_environment.json')
    report = inspect_environment()
    write_json(output / 'teacher_environment.json', report)
    print('Python:', report['python'])
    for name, item in report['packages'].items():
        print(name, item)
    print('CUDA:', report['cuda'])
    print('Non-HU API:', report['lungmask_api'])
    print('Ready:', report['non_hu_teacher_ready'])
    print('Output:', output / 'teacher_environment.json')


if __name__ == '__main__':
    main()
