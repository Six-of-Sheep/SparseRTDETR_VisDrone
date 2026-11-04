"""Direct pixel and spectrum comparison of the two frozen r20 window renderings.

Base canvas: frozen Development548.item.  Control window: frozen zoom_refine.control_pixels.
Method window: frozen zoom_refine.window_pixels on the original image.  No detector module is
imported and CUDA stays hidden; this only renders the existing plan on the CPU.

usage: python -B -I render_check.py --config CONFIG --run-dir DIR --commit SHA
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LUMA = (0.299, 0.587, 0.114)
HIGH_BAND = (0.25, 0.5)  # cycles per canvas pixel, radial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('CUDA must be explicitly hidden')
    run_dir = Path(args.run_dir)
    if (run_dir / 'render-check.json').exists():
        raise SystemExit('render check output already exists')
    started, cpu = time.perf_counter(), time.process_time()
    cfg = json.loads(Path(args.config).read_bytes())
    r20 = cfg['r20']
    manifest = json.loads((run_dir / 'INPUT_MANIFEST.json').read_bytes())
    contract = json.loads(Path(cfg['parent_contract']['path']).read_bytes())
    sys.path.insert(0, str(Path(contract['source_root']) / 'src'))
    import torch
    torch.set_num_threads(1)
    from PIL import Image
    from sparse_rtdetr.r0.evaluation_data import Development548
    from sparse_rtdetr.localization.data import read_bound
    path = Path(r20['release_source']) / r20['code']['zoom_refine']
    if sha(path) != r20['zoom_refine_sha256']:
        raise SystemExit('frozen zoom_refine differs')
    spec = importlib.util.spec_from_file_location('frozen_zoom_refine', path)
    zr = importlib.util.module_from_spec(spec)
    sys.modules['frozen_zoom_refine'] = zr
    spec.loader.exec_module(zr)
    data = Development548(manifest['data_spec'])
    plans = {r['image_id']: r['windows'] for r in json.loads(Path(r20['receipts']['A1']).read_bytes())['plan_audit']}
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    weights = torch.tensor(LUMA).view(3, 1, 1)
    rows_fy = torch.fft.fftfreq(768).abs()
    rows_fx = torch.fft.rfftfreq(1344)
    radial = torch.sqrt(rows_fy[:, None] ** 2 + rows_fx[None, :] ** 2)
    band = torch.logical_and(radial > HIGH_BAND[0], radial <= HIGH_BAND[1])

    def laplacian_energy(x):
        luma = (x * 255. * weights).sum(0, keepdim=True).unsqueeze(0)
        return float(torch.nn.functional.conv2d(luma, kernel).pow(2).mean())

    def high_band_energy(x):
        luma = (x * 255. * weights).sum(0)
        spectrum = torch.fft.rfft2(luma).abs().pow(2)
        return float(spectrum[band].sum() / spectrum.numel())

    records = []
    for position, row in enumerate(data.records):
        pixels, _ = data.item(row, position)
        w = plans[row['stable_image_id']][0]
        x0, y0, x1, y1 = w['crop_box']
        window = zr.Window(1, x0, y0, x1, y1, row['width'], row['height'])
        control = zr.control_pixels(pixels, window)
        raw = read_bound(Path(data.spec['roots']['images']) / Path(row['relative_path']).name,
                         row['image_sha256'], row['image_size_bytes'])
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert('RGB')
            image.load()
        method = zr.window_pixels(image, window)
        image.close()
        difference = (method - control) * 255.
        records.append({'width': row['width'], 'height': row['height'],
                        'control_equals_method': bool(torch.equal(control, method)),
                        'method_equals_base': bool(torch.equal(method, pixels)),
                        'control_equals_base': bool(torch.equal(control, pixels)),
                        'mae': float(difference.abs().mean()), 'rmse': float(difference.pow(2).mean().sqrt()),
                        'laplacian_control': laplacian_energy(control), 'laplacian_method': laplacian_energy(method),
                        'high_band_control': high_band_energy(control), 'high_band_method': high_band_energy(method)})
        if (position + 1) % 100 == 0:
            print(json.dumps({'images': position + 1, 'seconds': time.perf_counter() - started}), flush=True)
    groups = {}
    for r in records:
        groups.setdefault(f"{r['width']}x{r['height']}", []).append(r)
    summary = {}
    for name, items in sorted(groups.items()):
        arr = lambda k: np.array([x[k] for x in items], dtype=np.float64)
        lap = arr('laplacian_control') / np.maximum(arr('laplacian_method'), 1e-12)
        high = arr('high_band_control') / np.maximum(arr('high_band_method'), 1e-12)
        summary[name] = {'images': len(items), 'control_equals_method_images': int(sum(x['control_equals_method'] for x in items)),
                         'method_equals_base_images': int(sum(x['method_equals_base'] for x in items)),
                         'control_equals_base_images': int(sum(x['control_equals_base'] for x in items)),
                         'mae_mean_on_255_scale': float(arr('mae').mean()), 'rmse_mean_on_255_scale': float(arr('rmse').mean()),
                         'laplacian_energy_ratio_control_over_method_median': float(np.median(lap)),
                         'high_band_energy_ratio_control_over_method_median': float(np.median(high))}
    if torch.cuda.is_initialized():
        raise SystemExit('CUDA was initialized')
    if any('rtdetr' in name and 'decoder' in name for name in sys.modules):
        raise SystemExit('detector module imported')
    result = {'commit': args.commit, 'images': len(records), 'high_band_cycles_per_canvas_pixel': list(HIGH_BAND),
              'groups': summary, 'wall_seconds': time.perf_counter() - started,
              'cpu_seconds': time.process_time() - cpu, 'cuda_initialized': False, 'detector_constructed': False}
    with (run_dir / 'render-check.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({'done': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
