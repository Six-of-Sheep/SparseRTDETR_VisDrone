"""Real-engine synthetic smoke for the tiny-band geometry term (CPU only; CUDA hidden).

For each mode (native / A1 control / A2 method) two real ``FineTuneEngine.train_window``
updates run on fixed synthetic windows (128x128, batch 1 x 2 micro-batches of 2 images,
FP32), one chain checkpoint is saved after the first update, restored into a freshly built
engine and replayed against the recorded second update: the full state and RNG must be
bitwise exact and the term state (draws and generator stream) must continue identically.
The loss-key family must stay native (12/21), the observations must be recorded for every
non-DN family and the draw ledger must match the shifted-call counts.

usage: <frozen python> -I tools/tiny_smoke.py --source-root <release>/source --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def make_targets():
    box = {"ignore_union": torch.empty(0, 4, dtype=torch.float64),
           "ignore_class_boxes": torch.empty(0, 4, dtype=torch.float64),
           "ignore_class_labels": torch.empty(0, dtype=torch.int64)}
    first = {"boxes": torch.tensor([[0.40, 0.40, 0.02, 0.02], [0.60, 0.60, 0.30, 0.30]]),
             "labels": torch.tensor([1, 2]), "gt_ids": ["synthetic/tiny", "synthetic/big"], **box}
    second = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64),
              "gt_ids": [], **box}
    return [first, second]


def windows(seed):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.rand(2, 3, 128, 128, generator=generator), make_targets()) for _ in range(2)]


def observations(result):
    rows = [row for diag in result["normalizers"] for row in diag if "geo_pairs" in row]
    return {"rows": len(rows),
            "pairs": sum(int(row["geo_pairs"]) for row in rows),
            "tiny_pairs": sum(int(row["geo_tiny_pairs"]) for row in rows),
            "native_sum": sum(float(row["geo_tiny_native_sum"]) for row in rows),
            "applied_sum": sum(float(row["geo_tiny_applied_sum"]) for row in rows),
            "shifted_calls": sum(1 for row in rows if row.get("geo_shift") is not None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA must stay hidden for the smoke")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import tiny_geometry_arm as arm
    from sparse_rtdetr.protocol_r18.engine import tensor_digest

    config = Config(input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    report = {}
    for mode in ("A0", "A1", "A2"):
        started = time.monotonic()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        def construct():
            model, criterion, _ = build(config, synthetic=True)
            geo = None if mode == "A0" else mode
            criterion.tiny_geometry_declaration = arm.declaration(geo)
            term = None
            if geo is not None:
                term = arm.install(model, criterion, geo)
            opt, ema, alt = ft.make_optimizer_ema(model, config)
            engine = ft.FineTuneEngine(model, criterion, opt, ema, config, arm=mode,
                                       plan_sha256="a" * 64, parent_sha256="b" * 64, alt_ema=alt)
            engine.geo_mode = geo
            return engine, term

        engine, term = construct()
        inputs = windows(713)
        r1 = engine.train_window(inputs[0][0], inputs[0][1], capture=True)
        checks = {"mode": mode, "update1_keys": sorted(len(row) for row in r1["microbatch_losses"]),
                  "finite1": bool(r1["finite"])}
        cp = ft.save(engine, out / ("SMOKE-%s-C1.pth" % mode))
        expected = ft.capture(engine)
        r2 = engine.train_window(inputs[1][0], inputs[1][1], capture=True)
        obs = observations(r2)
        obs1 = observations(r1)
        draws_after = None if term is None else int(term.state_dict()["draws"])
        if mode == "A1":
            checks["draw_ledger"] = draws_after == obs["shifted_calls"] + obs1["shifted_calls"]
        if mode == "A2":
            checks["no_draw"] = draws_after == 0 and obs["shifted_calls"] == 0
        if mode == "A0":
            checks["applied_is_native"] = obs["applied_sum"] == obs["native_sum"]
        checks["band_active"] = (obs["tiny_pairs"] + obs1["tiny_pairs"]) > 0
        clone, _ = construct()
        restored = ft.restore(clone, cp)
        replay = clone.train_window(inputs[1][0], inputs[1][1], capture=True)
        exact_result = ({k: v for k, v in r2.items() if k != "elapsed_seconds"}
                        == {k: v for k, v in replay.items() if k != "elapsed_seconds"})
        exact_state = native_state.exact_equal(expected, ft.capture(clone))
        checks.update({"save_restore_replay_result_exact": exact_result,
                       "full_state_exact": exact_state,
                       "restore_full_state_RNG_exact": bool(restored.get("full_state_RNG_exact"))})
        assert all(checks.values()), ("SMOKE checks failed", mode, checks)
        body = {"schema": "r18-tiny-geometry-smoke-v1", "checks": checks,
                "observation_update1": obs1, "observation_update2": obs,
                "draws": draws_after, "checkpoint": {k: cp[k] for k in ("path", "bytes", "sha256")},
                "elapsed_seconds": time.monotonic() - started}
        target = out / ("SMOKE-%s.json" % mode)
        encoded = json.dumps(body, indent=1, ensure_ascii=False, sort_keys=True)
        target.write_bytes(encoded.encode("utf-8"))
        report[mode] = {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "checks": checks}
        print(json.dumps({"status": "SMOKE-OK", "mode": mode, "tiny_pairs": obs["tiny_pairs"] + obs1["tiny_pairs"],
                          "sha256": report[mode]["sha256"]}))
    assert not torch.cuda.is_initialized()
    print(json.dumps({"status": "SMOKE-ALL-OK", "modes": sorted(report)}))


if __name__ == "__main__":
    main()
