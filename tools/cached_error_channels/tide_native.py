"""Run the original tidecv v1.0.1 code on the certified filtered lists as a cross-check.

Only the computing modules are executed: plotting, cv2 and matplotlib are import-time stubs and
datasets.py is never imported.  Native TIDE keeps its own protocol: one max_dets per image over
all classes, ignore regions with crowd IoU strictly above the threshold, its own AP integration,
classes with data but no GT scored 0, negative dAP clipped to 0 by fix_main_errors.  To stay on
the certified list the input is already filtered and capped at 100 per image per class, and
max_dets is opened so that TIDE truncates nothing further.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

OPEN_MAX_DETS = 10 ** 9
ERROR_NAMES = ('ClassError', 'BoxError', 'OtherError', 'DuplicateError', 'BackgroundError', 'MissedError')
SHORT = {'ClassError': 'Cls', 'BoxError': 'Loc', 'OtherError': 'Both', 'DuplicateError': 'Dupe',
         'BackgroundError': 'Bkg', 'MissedError': 'Miss'}


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)  # inspect walks sys.modules and asks for __file__
        return 0


def load_tide(root):
    """Import tidecv.quantify and tidecv.data from an explicit, hash-checked source directory."""
    root = Path(root)
    for name in ('cv2', 'matplotlib', 'matplotlib.pyplot'):
        sys.modules.setdefault(name, _Stub(name))
    sys.modules['matplotlib'].pyplot = sys.modules['matplotlib.pyplot']
    pkg = types.ModuleType('tidecv')
    pkg.__path__ = [str(root / 'tidecv')]
    sys.modules['tidecv'] = pkg
    plotting = types.ModuleType('tidecv.plotting')
    plotting.Plotter = object
    plotting.print_table = lambda *a, **k: None
    sys.modules['tidecv.plotting'] = plotting
    pkg.plotting = plotting
    return importlib.import_module('tidecv.quantify'), importlib.import_module('tidecv.data')


def build_data(tide_data, n_img, gt_img, gt_cat, gt_bbox, gt_crowd, det_img, det_cat, det_score, det_bbox):
    gt = tide_data.Data('gt', max_dets=OPEN_MAX_DETS)
    pred = tide_data.Data('pred', max_dets=OPEN_MAX_DETS)
    for i in range(1, n_img + 1):
        gt.add_image(i, str(i))
    for img, cat, box, crowd in zip(gt_img.tolist(), gt_cat.tolist(), gt_bbox.tolist(), gt_crowd.tolist()):
        if crowd:
            gt.add_ignore_region(img, cat, box=box)
        else:
            gt.add_ground_truth(img, cat, box=box)
    for img, cat, score, box in zip(det_img.tolist(), det_cat.tolist(), det_score.tolist(), det_bbox.tolist()):
        pred.add_detection(img, cat, score, box=box)
    return gt, pred


def native_run(quantify, gt, pred, tf, bg=0.1):
    start = time.perf_counter()
    run = quantify.TIDERun(gt, pred, tf, bg, quantify.TIDE.BOX, gt.max_dets, True)
    errors = {type(e).__name__: 0 for e in run.errors}
    ignored_pred = 0
    for e in run.errors:
        errors[type(e).__name__] += 1
        if hasattr(e, 'pred') and e.pred['used'] is None:
            ignored_pred += 1
    counts = {SHORT[n]: errors.get(n, 0) for n in ERROR_NAMES}
    signed, clipped = {}, {}
    for name in ERROR_NAMES:
        cls = getattr(quantify, name)
        fixed = run.fix_errors(lambda err, c=cls: isinstance(err, c)).get_mAP()
        signed[SHORT[name]] = fixed - run.ap
        clipped[SHORT[name]] = max(fixed - run.ap, 0)
    small = quantify.Qualifier('Small', lambda x: x['bbox'][2] * x['bbox'][3] <= 32 ** 2)
    small_qualified = {}
    for name in ERROR_NAMES:
        cls = getattr(quantify, name)
        small_qualified[SHORT[name]] = run.fix_errors(small._make_error_func(cls)).get_mAP() - run.ap
    special = run.fix_special_errors()
    return {'iou': tf, 'background': bg, 'AP': run.ap, 'counts': counts,
            'errors_on_ignored_predictions': ignored_pred, 'delta_pp_signed': signed,
            'noop_fix_errors_delta_pp': run.fix_errors(lambda err: False).get_mAP() - run.ap,
            'delta_pp_clipped_native_rule': clipped,
            'delta_pp_all_AP_fixing_only_small_qualified_errors': small_qualified,
            'FP_rank_delta_pp': special[quantify.FalsePositiveError],
            'FN_recall_delta_pp': special[quantify.FalseNegativeError],
            'seconds': time.perf_counter() - start}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tide-root', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--tags', nargs='+', required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    quantify, data = load_tide(args.tide_root)
    for tag in args.tags:
        out = run_dir / f'native-{tag}.json'
        if out.exists():
            continue
        z = np.load(run_dir / f'list-{tag}.npz')
        started = time.perf_counter()
        gt, pred = build_data(data, int(z['n_img']), z['gt_img'], z['gt_cat'], z['gt_bbox'], z['gt_crowd'],
                              z['eval_img'], z['eval_cat'], z['eval_score'], z['eval_bbox'])
        result = {'tag': tag, 'input_npz_sha256': sha256(run_dir / f'list-{tag}.npz'), 'runs': {}}
        for tf in (0.5, 0.75):
            result['runs'][str(tf)] = native_run(quantify, gt, pred, tf)
        result['seconds'] = time.perf_counter() - started
        with open(out, 'x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(json.dumps({'tag': tag, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
