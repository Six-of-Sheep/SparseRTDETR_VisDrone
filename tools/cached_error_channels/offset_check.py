"""Supplement: signed box residuals of the existing final lists against their nearest same-class GT.

Descriptive only.  Uses list-*.npz of a finished run; no scoring, no list is changed.  The pair for a
GT is the evaluated same-class box with the highest IoU when that IoU is at least 0.5, so this is
not the certified one-to-one match and a box may serve two GT.

usage: python -B -I offset_check.py --run-dir DIR --output FILE --commit SHA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
ARMS = ('A', 'C')
KEYS = ('cx', 'cy', 'w', 'h')


def residuals(z):
    from pycocotools import mask as mask_utils
    gt_box, gt_img, gt_cat, crowd = z['gt_bbox'], z['gt_img'], z['gt_cat'], z['gt_crowd']
    eval_box, eval_img, eval_cat = z['eval_bbox'], z['eval_img'], z['eval_cat']
    out = np.full((len(gt_box), 4), np.nan)
    order = np.argsort(eval_img, kind='stable')
    cuts = np.searchsorted(eval_img[order], np.arange(1, int(z['n_img']) + 2))
    for i in range(int(z['n_img'])):
        dets = order[cuts[i]:cuts[i + 1]]
        gts = np.nonzero(np.logical_and(gt_img == i + 1, np.logical_not(crowd)))[0]
        if len(dets) == 0 or len(gts) == 0:
            continue
        iou = mask_utils.iou(eval_box[dets], gt_box[gts], np.zeros(len(gts), dtype=np.uint8))
        iou = iou * (eval_cat[dets][:, None] == gt_cat[gts][None, :])
        best = iou.argmax(0)
        for column, g in enumerate(gts):
            if iou[best[column], column] < .5:
                continue
            d = eval_box[dets[best[column]]]
            t = gt_box[g]
            out[g] = (d[0] + d[2] / 2 - t[0] - t[2] / 2, d[1] + d[3] / 2 - t[1] - t[3] / 2, d[2] - t[2], d[3] - t[3])
    return out


def block(values):
    values = values[~np.isnan(values[:, 0])]
    n = len(values)
    if n == 0:
        return {'n': 0}
    result = {'n': n}
    for k, key in enumerate(KEYS):
        result[key] = {'mean': round(float(values[:, k].mean()), 4), 'median': round(float(np.median(values[:, k])), 4),
                       'sem_unclustered': round(float(values[:, k].std(ddof=1) / np.sqrt(n)), 4)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / 'INPUT_MANIFEST.json').read_bytes())
    images = manifest['images']
    result = {'commit': args.commit, 'run_manifest_sha256_note': 'list arrays of the named run', 'units': 'original image pixels; detection minus GT',
              'pair_rule': 'nearest evaluated same-class box with IoU at least 0.5, not the certified one-to-one match', 'lists': {}}
    for s in SEEDS:
        for a in ARMS:
            tag = f'{a}-seed{s}'
            z = np.load(run_dir / f'list-{tag}.npz')
            r = residuals(z)
            nc = np.logical_not(z['gt_crowd'])
            area = z['gt_area']
            bands = {'small': np.logical_and(nc, area <= 1024), 'medium': np.logical_and.reduce((nc, area > 1024, area <= 9216)),
                     'large': np.logical_and(nc, area > 9216)}
            size = np.array([f"{images[int(i) - 1]['width']}x{images[int(i) - 1]['height']}" for i in z['gt_img']])
            seq = np.array([images[int(i) - 1]['sequence'] for i in z['gt_img']])
            small = bands['small']
            entry = {'by_area_band': {k: block(r[v]) for k, v in bands.items()},
                     'small_by_image_size': {name: block(r[np.logical_and(small, size == name)]) for name in sorted(set(size.tolist()))}}
            valid = np.logical_and(small, ~np.isnan(r[:, 0]))
            means = []
            for name in sorted(set(seq[valid].tolist())):
                sel = np.logical_and(valid, seq == name)
                if sel.sum() >= 20:
                    means.append(r[sel, :2].mean(0))
            means = np.array(means)
            entry['small_sequences_with_at_least_20_pairs'] = {
                'sequences': len(means), 'cx_positive': int((means[:, 0] > 0).sum()), 'cy_positive': int((means[:, 1] > 0).sum()),
                'cx_mean_of_sequence_means': round(float(means[:, 0].mean()), 4), 'cy_mean_of_sequence_means': round(float(means[:, 1].mean()), 4)}
            result['lists'][tag] = entry
    with open(args.output, 'x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({'output': args.output}))


if __name__ == '__main__':
    main()
