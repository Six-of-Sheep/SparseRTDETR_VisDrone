"""Model-free coordinate cases that need the frozen torchvision transforms.

The cached-analysis process forbids every nn.Module construction (frozen softnms_job.cpu_only), and the
torchvision v2 transforms are nn.Modules, so these cases run in their own process.  No detector is
built, CUDA stays hidden, no experiment image or annotation is read: every image is synthetic.

Cases (all against the frozen release named by the parent contract, file SHAs checked first):
  * evaluation resize (RawAugmentation, training=False): the label centre equals the intensity centroid;
  * training flip + resize (training=True, epoch 119: sanitize, p=.5 flip, resize): same check, flipped and not;
  * flip twice restores image and box bit for bit, and the flipped box bounds the flipped pixels;
  * C tiles: integer half-open crop, resize, decode times tile size, add origin (slice_core.map_crop_xyxy).

usage: python -B -I coord_check.py --contract PATH --contract-sha SHA --output FILE --commit SHA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

USED = ('src/sparse_rtdetr/localization/data.py', 'src/sparse_rtdetr/data_protocol/parser.py', 'slice_core.py', 'infer_seed.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def centroid(plane):
    w = plane.astype(np.float64)
    mass = w.sum()
    ys, xs = np.indices(w.shape)
    return float((w * (xs + .5)).sum() / mass), float((w * (ys + .5)).sum() / mass), float(mass)


def rect_image(width, height, left, top, w, h):
    from PIL import Image
    a = np.zeros((height, width, 3), dtype=np.uint8)
    a[top:top + h, left:left + w] = 255
    return Image.fromarray(a, 'RGB')


def aug_case(aug, parse_raw, width, height, left, top, w, h, *, training, epoch, position):
    image = rect_image(width, height, left, top, w, h)
    raw = parse_raw(f'{left},{top},{w},{h},1,4,0,0\n'.encode(), 'synthetic', expected_rows=1)
    pixels, target, trace = aug(image, raw, epoch=epoch, seed=0, position=position, image_id='synthetic', training=training)
    ch, cw = pixels.shape[1], pixels.shape[2]
    cx, cy, mass = centroid(pixels[0].numpy())
    b = target['boxes'][0].double().numpy()
    sx, sy = cw / width, ch / height
    return {'size': f'{width}x{height}', 'rect_xywh': [left, top, w, h], 'training': training, 'epoch': epoch,
            'flipped': any(s['operation'] == 'horizontal_flip' for s in trace['geometry_steps']),
            'steps': [s['operation'] for s in trace['geometry_steps']],
            'error_cx_original_px': float((cx - b[0] * cw) / sx), 'error_cy_original_px': float((cy - b[1] * ch) / sy),
            'mass_over_box_area_canvas': float(mass / (b[2] * cw * b[3] * ch))}


def worst(items, key):
    return max(abs(x[key]) for x in items) if items else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--contract-sha', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--commit', required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('CUDA must be explicitly hidden')
    if sha(args.contract) != args.contract_sha:
        raise SystemExit('parent contract SHA differs')
    contract = json.loads(Path(args.contract).read_bytes())
    source = Path(contract['source_root'])
    files = {x['path']: x['sha256'] for x in contract['source_files']}
    checked = {}
    for rel in USED:
        if rel not in files or sha(source / rel) != files[rel]:
            raise SystemExit('frozen source differs or is not bound: ' + rel)
        checked[rel] = files[rel]
    sys.path[:0] = [str(source / 'src'), str(source)]
    import PIL
    import torch
    import torchvision
    from torchvision import tv_tensors
    from torchvision.transforms.v2 import functional as TF
    from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
    import slice_core
    aug = RawAugmentation((768, 1344))
    cases = {}
    evals, trains = [], []
    for width, height in ((960, 540), (1360, 765), (1920, 1080)):
        for left, top, w, h in ((int(width * .37), int(height * .41), 6, 9), (int(width * .61), int(height * .23), 40, 30)):
            evals.append(aug_case(aug, parse_raw, width, height, left, top, w, h, training=False, epoch=120, position=0))
            for position in range(8):
                trains.append(aug_case(aug, parse_raw, width, height, left, top, w, h, training=True, epoch=119, position=position))
    cases['resize_eval'] = {'cases': evals, 'max_abs_error_cx_px': worst(evals, 'error_cx_original_px'),
                            'max_abs_error_cy_px': worst(evals, 'error_cy_original_px')}
    fl = [x for x in trains if x['flipped']]
    nf = [x for x in trains if not x['flipped']]
    cases['train_flip_resize_epoch119'] = {
        'flipped_cases': len(fl), 'unflipped_cases': len(nf),
        'flipped_max_abs_error_cx_px': worst(fl, 'error_cx_original_px'), 'flipped_max_abs_error_cy_px': worst(fl, 'error_cy_original_px'),
        'unflipped_max_abs_error_cx_px': worst(nf, 'error_cx_original_px'), 'unflipped_max_abs_error_cy_px': worst(nf, 'error_cy_original_px'),
        'flipped_mean_error_cx_px': float(np.mean([x['error_cx_original_px'] for x in fl])) if fl else None,
        'unflipped_mean_error_cx_px': float(np.mean([x['error_cx_original_px'] for x in nf])) if nf else None,
        'cases': trains,
        'note': 'epoch >= 118 removes photometric/zoom/crop, leaving sanitize, p=.5 flip and resize; strong-augmentation geometry is not covered'}
    img = rect_image(37, 11, 5, 2, 7, 3)
    bb = tv_tensors.BoundingBoxes(torch.tensor([[5., 2., 12., 5.]]), format='XYXY', canvas_size=(11, 37))
    f1i, f1b = TF.horizontal_flip(img), TF.horizontal_flip(bb)
    f2i, f2b = TF.horizontal_flip(f1i), TF.horizontal_flip(f1b)
    cols = np.nonzero(np.asarray(f1i)[:, :, 0].max(0))[0]
    flip = {'image_bitwise_restored': bool(np.array_equal(np.asarray(f2i), np.asarray(img))),
            'box_bitwise_restored': bool(torch.equal(f2b.as_subclass(torch.Tensor), bb.as_subclass(torch.Tensor))),
            'flipped_box': f1b.as_subclass(torch.Tensor).tolist()[0], 'flipped_pixel_columns_half_open': [int(cols.min()), int(cols.max()) + 1]}
    flip['pass'] = bool(flip['image_bitwise_restored'] and flip['box_bitwise_restored']
                        and flip['flipped_box'][0] == flip['flipped_pixel_columns_half_open'][0]
                        and flip['flipped_box'][2] == flip['flipped_pixel_columns_half_open'][1])
    cases['flip_twice'] = flip
    tiles = []
    for width, height in ((960, 540), (1360, 765), (1920, 1080)):
        for tile in slice_core.tile_windows(height, width):
            left, top = tile.x0 + 40, tile.y0 + 25
            crop = rect_image(width, height, left, top, 9, 7).crop(tile.crop_box)
            pixels = np.asarray(aug.resize(crop))[:, :, 0]
            cx, cy, _ = centroid(pixels)
            nx, ny = cx / pixels.shape[1], cy / pixels.shape[0]
            px, py = nx * tile.width, ny * tile.height
            box = slice_core.map_crop_xyxy([[px - .5, py - .5, px + .5, py + .5]], tile)[0]
            tiles.append({'size': f'{width}x{height}', 'view': tile.view_id, 'crop_box': list(tile.crop_box),
                          'error_cx_px': float((box[0] + box[2]) / 2 - (left + 4.5)), 'error_cy_px': float((box[1] + box[3]) / 2 - (top + 3.5))})
    cases['c_tile_mapping'] = {'cases': tiles, 'max_abs_error_cx_px': worst(tiles, 'error_cx_px'), 'max_abs_error_cy_px': worst(tiles, 'error_cy_px'),
                               'note': 'infer_seed.pixels = data.transform.resize on the PIL crop; boxes = normalized x tile size + integer origin'}
    result = {'commit': args.commit, 'frozen_source': str(source), 'frozen_files_sha256': checked,
              'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'PIL': PIL.__version__},
              'cases': cases, 'detector_constructed': False, 'cuda_initialized': torch.cuda.is_initialized(),
              'units': 'original image pixels; centroid uses pixel centres j+0.5 (half-open pixel-corner boxes)'}
    with open(args.output, 'x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1, allow_nan=False)
        f.write('\n')
    print(json.dumps({'resize_eval_max': [cases['resize_eval']['max_abs_error_cx_px'], cases['resize_eval']['max_abs_error_cy_px']],
                      'flip': [cases['train_flip_resize_epoch119']['flipped_cases'], cases['train_flip_resize_epoch119']['flipped_max_abs_error_cx_px']],
                      'tile_max': [cases['c_tile_mapping']['max_abs_error_cx_px'], cases['c_tile_mapping']['max_abs_error_cy_px']],
                      'flip_twice': flip['pass']}), flush=True)


if __name__ == '__main__':
    main()
