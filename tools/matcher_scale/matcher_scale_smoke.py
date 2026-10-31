"""CPU synthetic training smoke of the size-relative matching cost through the real engine.

Per mode ('ctrl' or 'rel'): synthetic Config -> install the matcher cost ->
compare against the native matcher on the same frozen window inputs -> real
Engine with a fresh optimizer and EMA -> two train windows -> checkpoint
save/restore/replay exactness -> cross-mode and native restore refusal -> closed
RNG equality against a native engine -> matcher compute cost -> JSON evidence.
CUDA must be hidden; the frozen release and its outputs are untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

matcher_scale = None  # bound by _load() from the staged release package


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _load(root):
    """Import the staged package module; the release tree holds the single copy."""
    global matcher_scale
    sys.path.insert(0, str(Path(root) / "src"))
    sys.path.insert(0, str(root))
    import sparse_rtdetr
    require(str(Path(sparse_rtdetr.__file__).resolve()).startswith(str(Path(root) / "src")),
            "sparse_rtdetr resolved outside the release")
    from sparse_rtdetr.protocol_r18 import matcher_scale as module
    matcher_scale = module
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    _load(root)
    require(args.mode in matcher_scale.MODES, "unknown matching mode: " + args.mode)
    from sparse_rtdetr.protocol_r18.model import ARM, CommonEMA, Config, build
    from sparse_rtdetr.protocol_r18.engine import Engine
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.baseline.training_v2b import _optimizer

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), "CUDA initialized before the CPU smoke")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    conf = Config(seed=0, input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    mode = args.mode

    def reset():
        random.seed(0); np.random.seed(0); torch.manual_seed(0)

    def construct(install):
        model, criterion, info = build(conf, pretrained_path=None, synthetic=True)
        if install is not None:
            matcher_scale.install_matcher_scale(criterion, install)
            matcher_scale.assert_installed(criterion, install)
        return model, criterion, info

    def engine_for(install):
        model, criterion, info = construct(install)
        engine = Engine(model, criterion, _optimizer(model, conf), CommonEMA(model), conf)
        engine.install_mode = install
        return engine, info

    def window(epoch):
        dataset = job.FixtureTrain("D0", epoch, 0, 1)
        return job.collate_window([dataset[i] for i in range(len(dataset))])

    def pairs(indices):
        return [(int(i), int(j)) for i, j in zip(indices[0][0].tolist(), indices[0][1].tolist())]

    reset()
    windows = [window(1), window(2)]

    # --- same inputs, both matchers -------------------------------------------------
    native_model, native_criterion, _ = construct(None)
    model, criterion, info = construct(mode)
    native_keys = [n for n, _ in native_model.named_parameters()]
    keys = [n for n, _ in model.named_parameters()]
    require(native_keys == keys, "installation changed the parameter names")
    require(sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in native_model.parameters()),
            "installation changed the parameter count")
    model.train(); native_model.train()
    with torch.no_grad():
        out_native = native_model(windows[0][0], targets=windows[0][1])
        out_installed = model(windows[0][0], targets=windows[0][1])
    require(all(torch.equal(out_native[k], out_installed[k]) for k in ("pred_logits", "pred_boxes")),
            "installation changed the model forward outputs")
    native_pairs = pairs(native_criterion.matcher(out_native, windows[0][1])["indices"])
    installed_pairs = pairs(criterion.matcher(out_installed, windows[0][1])["indices"])
    cost_info = dict(matcher_scale.LAST_COST_INFO)

    # --- real engine: two windows, save, restore, replay ----------------------------
    engine, _ = engine_for(mode)
    require(set(engine.ema.module.state_dict()) == set(native_model.state_dict()),
            "the EMA copy carries keys the native model does not")
    binding = {"arm": ARM, "scope": "cpu_R18_synthetic",
               "run_id": "r18-matcher-scale-smoke-20261031-001",
               "config": conf.binding(), "model": info}
    r0 = engine.train_window(*windows[0], capture=True)
    require(r0["update"] == 1 and engine.completed_updates == engine.ema.updates == 1 and engine.phase == 0,
            "update clock differs after window 1")
    cp0 = checkpoint.save(engine, binding, out / "CPU-C1.pth")
    expected = checkpoint.capture(engine, binding)
    require(expected["behavior"]["criterion"]["matching"] == matcher_scale.declaration(mode),
            "saved behaviour identity does not carry the declared matching mode")
    r1 = engine.train_window(*windows[1], capture=True)
    require(r1["update"] == 2 and engine.completed_updates == engine.ema.updates == 2,
            "update clock differs after window 2")
    direct = checkpoint.save(engine, binding, out / "CPU-C2-direct.pth")
    rng_after_installed = checkpoint.capture_rng(torch.device("cpu"))

    clone, _ = engine_for(mode)
    restored = checkpoint.restore(clone, binding, cp0["path"], cp0["sha256"])
    r1_replay = clone.train_window(*windows[1], capture=True)
    same_result = ({k: v for k, v in r1.items() if k != "elapsed_seconds"}
                   == {k: v for k, v in r1_replay.items() if k != "elapsed_seconds"})
    state_exact = checkpoint.exact_equal(expected, checkpoint.capture(clone, binding))
    require(same_result, "window-2 replay result differs")
    require(state_exact, "window-2 replay state differs")
    require(bool(restored.get("full_state_exact", False)), "C1 restore is not exact")

    # --- refusals: the declaration travels with the state ---------------------------
    other = "ctrl" if mode == "rel" else "rel"
    clash, _ = engine_for(other)
    clash_refusal = None
    try:
        checkpoint.restore(clash, binding, cp0["path"], cp0["sha256"])
    except Exception as error:  # noqa: BLE001
        clash_refusal = repr(error)
    require(clash_refusal is not None, "a %s checkpoint was accepted by a %s engine" % (mode, other))
    require(clash.failed is True, "a refused restore did not mark the engine failed")
    native_clash, _ = engine_for(None)
    native_refusal = None
    try:
        checkpoint.restore(native_clash, binding, cp0["path"], cp0["sha256"])
    except Exception as error:  # noqa: BLE001
        native_refusal = repr(error)
    require(native_refusal is not None, "an installed checkpoint was accepted by the native engine")

    # --- closed RNG equality against a native engine over the same windows ----------
    reset()
    rng_engine, _ = engine_for(None)
    for win in windows:
        rng_engine.train_window(*win, capture=True)
    rng_after_native = checkpoint.capture_rng(torch.device("cpu"))
    rng_equal = bool(rng_after_installed["python"] == rng_after_native["python"]
                     and torch.equal(rng_after_installed["torch_cpu"], rng_after_native["torch_cpu"])
                     and rng_after_installed["numpy"]["position"] == rng_after_native["numpy"]["position"])
    require(rng_equal, "the installed matcher changed the closed training RNG state")

    # --- matcher compute cost on the real shapes -----------------------------------
    timing = {}
    with torch.no_grad():
        for label, matcher_like in (("native", native_criterion.matcher), (mode, criterion.matcher)):
            matcher_like(out_installed, windows[0][1])
            tick = time.perf_counter()
            for _ in range(20):
                matcher_like(out_installed, windows[0][1])
            timing[label] = round((time.perf_counter() - tick) / 20 * 1000.0, 4)

    report = {
        "status": "PASS", "mode": mode,
        "declared_matching": matcher_scale.declaration(mode),
        "assignment": {"native": native_pairs, "installed": installed_pairs,
                       "changed": bool(native_pairs != installed_pairs),
                       "targets": cost_info.get("targets"), "queries": cost_info.get("queries"),
                       "scale_norm_mean": cost_info.get("scale_norm_mean")},
        "parameter_names_and_count_unchanged": True,
        "ema_state_keys_unchanged": True,
        "training": {"windows": len(windows), "updates": engine.completed_updates,
                     "replay_result_exact": same_result, "replay_state_exact": state_exact,
                     "restore_full_state_exact": bool(restored.get("full_state_exact")),
                     "checkpoints": [cp0, direct]},
        "cross_mode_restore_refused": clash_refusal,
        "native_restore_of_installed_state_refused": native_refusal,
        "closed_rng_equal_to_native_engine": rng_equal,
        "matcher_ms_per_call": timing,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    (out / ("SMOKE-%s.json" % mode)).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("SMOKE_COMPLETE", mode, report["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
