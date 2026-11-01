"""Real-engine synthetic smoke for the tiny-band extra-positive term (CPU only; CUDA hidden).

For each mode (native / A1 mass control / A2 method) two real ``FineTuneEngine.train_window``
updates run on fixed synthetic windows (128x128, batch 1 x 2 micro-batches of 2 images, FP32),
one chain checkpoint is saved after the first update, restored into a freshly built engine and
replayed against the recorded second update: the full state and RNG must be bitwise exact and the
term state (the deterministic extra-pair counter) must continue identically.  The loss-key family
must stay native (12/21: the extra positive pairs add no loss key), the observations must be
recorded for every non-DN family of every arm, the native arm must add no extra pair at all, the
A1 arm must duplicate exactly the native matched band pairs, and the A2 arm must account for every
tiny-band ground-truth box (admitted + no admissible sibling == band ground truth).

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
    first = {"boxes": torch.tensor([[0.40, 0.40, 0.02, 0.02], [0.30, 0.60, 0.02, 0.02],
                                    [0.60, 0.60, 0.30, 0.30]]),
             "labels": torch.tensor([1, 2, 3]),
             "gt_ids": ["synthetic/tiny-a", "synthetic/tiny-b", "synthetic/big"], **box}
    second = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.int64),
              "gt_ids": [], **box}
    return [first, second]


def windows(seed):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.rand(2, 3, 128, 128, generator=generator), make_targets()) for _ in range(2)]


def observations(result):
    rows = [row for diag in result["normalizers"] for row in diag if "pos_pairs" in row]
    return {"rows": len(rows),
            "pairs": sum(int(row["pos_pairs"]) for row in rows),
            "band_gt": sum(int(row["pos_band_gt"]) for row in rows),
            "band_pairs": sum(int(row["pos_band_pairs"]) for row in rows),
            "extra_pairs": sum(int(row["pos_extra_pairs"]) for row in rows),
            "band_gt_no_sibling": sum(int(row["pos_band_gt_no_sibling"]) for row in rows),
            "extra_iou_sum": sum(float(row["pos_extra_iou_sum"]) for row in rows),
            "modes": sorted({row.get("pos_extra_mode") for row in rows}, key=str)}


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
    from sparse_rtdetr.protocol_r18 import tiny_positives_arm as arm

    config = Config(input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    report = {}
    for mode in ("A0", "A1", "A2"):
        started = time.monotonic()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        def construct():
            model, criterion, _ = build(config, synthetic=True)
            pos = None if mode == "A0" else mode
            criterion.tiny_positives_declaration = arm.declaration(pos)
            term = None
            if pos is not None:
                term = arm.install(model, criterion, pos)
            opt, ema, alt = ft.make_optimizer_ema(model, config)
            engine = ft.FineTuneEngine(model, criterion, opt, ema, config, arm=mode,
                                       plan_sha256="a" * 64, parent_sha256="b" * 64, alt_ema=alt)
            engine.pos_mode = pos
            return engine, term

        engine, term = construct()
        inputs = windows(713)
        r1 = engine.train_window(inputs[0][0], inputs[0][1], capture=True)
        obs1 = observations(r1)
        checks = {"mode": mode, "update1_keys": sorted(len(row) for row in r1["microbatch_losses"]),
                  "finite1": bool(r1["finite"])}
        cp = ft.save(engine, out / ("SMOKE-%s-C1.pth" % mode))
        r2 = engine.train_window(inputs[1][0], inputs[1][1], capture=True)
        expected = ft.capture(engine)
        obs = observations(r2)
        extras = None if term is None else int(term.state_dict()["extras"])
        checks["loss_keys_native"] = sorted(set(checks["update1_keys"])) == [12, 21]
        checks["observations_recorded"] = obs["rows"] > 0 and obs1["rows"] > 0
        if mode == "A0":
            checks["native_adds_nothing"] = (obs["extra_pairs"] + obs1["extra_pairs"]) == 0 \
                and extras is None and obs["modes"] == [None]
        if mode == "A1":
            checks["a1_duplicates_every_band_pair"] = (
                (obs1["extra_pairs"] + obs["extra_pairs"]) == (obs1["band_pairs"] + obs["band_pairs"])
                and (obs1["band_pairs"] + obs["band_pairs"]) > 0)
            checks["a1_no_admission_ledger"] = (obs1["band_gt_no_sibling"] + obs["band_gt_no_sibling"]) == 0
        if mode == "A2":
            checks["a2_accounts_for_every_band_gt"] = (
                (obs1["extra_pairs"] + obs["extra_pairs"] + obs1["band_gt_no_sibling"] + obs["band_gt_no_sibling"])
                == (obs1["band_gt"] + obs["band_gt"]) and (obs1["band_gt"] + obs["band_gt"]) > 0)
            checks["a2_overlap_bar"] = obs1["extra_iou_sum"] >= 0.5 * obs1["extra_pairs"] \
                and obs["extra_iou_sum"] >= 0.5 * obs["extra_pairs"]
            checks["a2_mode_recorded"] = obs["modes"] == ["A2"]
        if extras is not None:
            checks["counter_matches_ledger"] = extras == obs1["extra_pairs"] + obs["extra_pairs"]
        checks["band_active"] = (obs1["band_pairs"] + obs["band_pairs"]) > 0
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
        body = {"schema": "r18-tiny-positives-smoke-v1", "checks": checks, "extras": extras,
                "observation_update1": obs1, "observation_update2": obs,
                "checkpoint": {k: cp[k] for k in ("path", "bytes", "sha256")},
                "elapsed_seconds": time.monotonic() - started}
        target = out / ("SMOKE-%s.json" % mode)
        encoded = json.dumps(body, indent=1, ensure_ascii=False, sort_keys=True)
        target.write_bytes(encoded.encode("utf-8"))
        report[mode] = {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "checks": checks}
        print(json.dumps({"status": "SMOKE-OK", "mode": mode, "band_pairs": obs1["band_pairs"] + obs["band_pairs"],
                          "extras": extras, "sha256": report[mode]["sha256"]}))
    assert not torch.cuda.is_initialized()
    print(json.dumps({"status": "SMOKE-ALL-OK", "modes": sorted(report)}))


if __name__ == "__main__":
    main()
