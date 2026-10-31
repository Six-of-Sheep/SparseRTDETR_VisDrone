"""Layout diff probe: why does checkpoint._restore report a model/EMA layout mismatch?

Fresh installs vs post-state variants, using the frozen release layout function.
Throwaway diagnostic for the r10 prototype round.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import detail_path  # noqa: E402
import frozen_r18  # noqa: E402


def layout_diff(tag, a, b, old, limit=20):
    la, lb = old._model_layout(a), old._model_layout(b)
    if la == lb:
        print("LAYOUT-EQUAL", tag, flush=True)
        return
    print("LAYOUT-DIFF", tag, flush=True)
    keys = sorted(set(la) | set(lb))
    shown = 0
    for k in keys:
        if la.get(k) != lb.get(k):
            shown += 1
            if shown <= limit:
                print("  DIFF", k, "| A=", str(la.get(k))[:90], "| B=", str(lb.get(k))[:90], flush=True)
    print("  total-diff-keys", shown, "of", len(keys), flush=True)


def main():
    src = sys.argv[1]
    frozen_r18.require_release_source(src)
    from sparse_rtdetr.baseline import training_v2b_checkpoint as old
    from sparse_rtdetr.protocol_r18.model import CommonEMA, Config, build
    from sparse_rtdetr.protocol_r18.engine import Engine
    from sparse_rtdetr.baseline.training_v2b import _optimizer

    torch.set_num_threads(2)
    conf = Config(seed=0, input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")

    def make(mode):
        model, criterion, info = build(conf, pretrained_path=None, synthetic=True)
        detail_path.install_detail_path(model, mode)
        return model, criterion, info

    model_a, criterion_a, _ = make("detail")
    model_b, criterion_b, _ = make("detail")
    la = old._model_layout(model_a)
    print("layout type", type(la).__name__, "entries", len(la), flush=True)
    sample = list(la.items())[:3]
    for key, value in sample:
        print("  sample", key, "->", str(value)[:120], flush=True)
    layout_diff("fresh-vs-fresh", model_a, model_b, old)

    model_a.backbone._detail_capture["s4"] = torch.zeros(1, 4, 4)
    layout_diff("s4-set-vs-none", model_a, model_b, old)
    model_a.backbone._detail_capture["s4"] = None

    model_a.eval()
    layout_diff("eval-vs-train", model_a, model_b, old)
    model_a.train()

    ema_a = CommonEMA(model_a).module
    ema_b = CommonEMA(model_b).module
    layout_diff("ema-vs-ema", ema_a, ema_b, old)

    def fixture(win):
        return [
            {"boxes": torch.tensor([[.4, .4, .15, .2]]), "labels": torch.tensor([1]),
             "gt_ids": ["probe/w%d/raw0" % win],
             "ignore_union": torch.empty(0, 4, dtype=torch.float64),
             "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
             "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
            {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64), "gt_ids": [],
             "ignore_union": torch.empty(0, 4, dtype=torch.float64),
             "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
             "ignore_class_labels": torch.empty(0, dtype=torch.int64)},
        ]

    engine = Engine(model_a, criterion_a, _optimizer(model_a, conf), CommonEMA(model_a), conf)
    images = torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(21))
    engine.train_window(images, fixture(0), capture=True)
    model_b.train()
    layout_diff("trained-vs-fresh", model_a, model_b, old)
    engine2 = Engine(model_b, criterion_b, _optimizer(model_b, conf), CommonEMA(model_b), conf)
    layout_diff("trained-vs-engine2-fresh", model_a, engine2.model, old)


if __name__ == "__main__":
    main()
