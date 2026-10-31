"""Targeted dual-EMA checks (CPU only, no CUDA, no production data).

Run before the release freeze.  Everything is measured on the fixture
configuration (128x128, physical batch 1, FP32, synthetic native model):

 1. make_optimizer_ema builds two EMA records whose initial modules are bitwise
    identical (same parent/synthetic source state), with counters 0 and 36,480,
    and consumes no CPU RNG draw while building the second record.
 2. A dual-EMA engine and a single-EMA reference engine (the same class with
    alt_ema=None) produce the identical online trajectory: identical update LR,
    losses, pre-clip gradient digest, detector-output digest and DN-noise digest
    on every update, identical closed CPU/Python/NumPy RNG state afterwards.
    The second EMA therefore consumes no training RNG draw.
 3. EMA-R under the dual engine is bitwise equal to a reference record stepped
    over the same post-update model states with the original math (original
    EMA-R behavior unchanged), and EMA-C is bitwise equal to a reference record
    with the same math whose counter starts at 36,480 (only the clock differs).
 4. The complete state payload carries both EMA modules and both counters, and
    restore reproduces the full dual-EMA state exactly (bitwise).

Writes <out>/DUAL_EMA_CHECK.json.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rng_equal(left, right):
    for key in ('python', 'torch_cpu'):
        if key == 'torch_cpu':
            if not torch.equal(left[key], right[key]):
                return False
        elif left[key] != right[key]:
            return False
    numpy_left = (left['numpy']['algorithm'], left['numpy']['position'], left['numpy']['has_gauss'], left['numpy']['cached_gaussian'])
    numpy_right = (right['numpy']['algorithm'], right['numpy']['position'], right['numpy']['has_gauss'], right['numpy']['cached_gaussian'])
    return numpy_left == numpy_right and torch.equal(left['numpy']['keys'], right['numpy']['keys'])


def main():
    started = time.monotonic()
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / 'src'))
    sys.path.insert(0, str(source))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), 'dual-EMA check initialized CUDA')

    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.protocol_r18.model import Config, CommonEMA, build
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state

    fixture = Config(input_hw=(128, 128), physical_batch_size=1, amp_dtype='float32')
    checks = {}

    def reset():
        random.seed(0); np.random.seed(0); torch.random.default_generator.manual_seed(0)

    def window(epoch):
        dataset = job.FixtureTrain('D0', epoch, 0, 1)
        return job.collate_window([dataset[i] for i in range(len(dataset))])

    # ---- run A: dual-EMA engine -------------------------------------------------
    model, criterion, _ = build(fixture, synthetic=True)
    initial = tensor_digest(model.state_dict().items())
    cpu_before = torch.random.get_rng_state()
    opt, ema, alt = ft.make_optimizer_ema(model, fixture)
    checks['second_ema_construction_rng_neutral'] = bool(torch.equal(cpu_before, torch.random.get_rng_state()))
    require(checks['second_ema_construction_rng_neutral'], 'second EMA construction consumed CPU RNG')
    checks['ema_r_start_counter'] = int(ema.updates)
    checks['ema_c_start_counter'] = int(alt.updates)
    require(ema.updates == 0 and alt.updates == ft.ALT_OFFSET, 'dual-EMA start counters differ')
    checks['initial_modules_bitwise_identical'] = bool(
        tensor_digest(ema.module.state_dict().items()) == tensor_digest(alt.module.state_dict().items()) == initial)
    require(checks['initial_modules_bitwise_identical'], 'dual-EMA initial modules differ')
    checks['recipes_identical'] = bool(ema.decay == alt.decay == .9999 and ema.warmups == alt.warmups == 2000)
    require(checks['recipes_identical'], 'dual-EMA decay/warmups differ')

    dual = ft.FineTuneEngine(model, criterion, opt, ema, fixture, arm='D0',
                             plan_sha256='0' * 64, parent_sha256='a' * 64, alt_ema=alt)
    ref_r = CommonEMA(model)
    ref_c = CommonEMA(model); ref_c.updates = ft.ALT_OFFSET
    reset()
    dual_records = []
    for epoch in (1, 2):
        x, t, tr = window(epoch)
        dual_records.append(dual.train_window(x, t, capture=True))
        ref_r.update(model)
        ref_c.update(model)
    dual_rng = native_state.capture_rng(torch.device('cpu'))
    dual_r_digest = tensor_digest(dual.ema.module.state_dict().items())
    dual_c_digest = tensor_digest(dual.alt_ema.module.state_dict().items())
    checks['ema_r_reference_bitwise'] = bool(dual_r_digest == tensor_digest(ref_r.module.state_dict().items()))
    checks['ema_c_reference_bitwise'] = bool(dual_c_digest == tensor_digest(ref_c.module.state_dict().items()))
    checks['ema_r_follows_original_math'] = bool(ref_r.updates == dual.ema.updates == 2)
    checks['ema_c_follows_continued_counter'] = bool(ref_c.updates == dual.alt_ema.updates == 2 + ft.ALT_OFFSET)
    checks['records_differ_after_updates'] = bool(dual_r_digest != dual_c_digest)
    checks['engine_rng_check_count'] = int(dual.alt_rng_neutral_updates)
    require(checks['ema_r_reference_bitwise'] and checks['ema_c_reference_bitwise'], 'EMA record math differs from the reference records')
    require(checks['ema_r_follows_original_math'] and checks['ema_c_follows_continued_counter'], 'EMA record clocks differ from the reference records')
    require(checks['engine_rng_check_count'] == 2, 'engine per-update RNG check count differs')

    # ---- run B: identical model, single-EMA reference engine --------------------
    model_b, criterion_b, _ = build(fixture, synthetic=True)
    checks['synthetic_construction_deterministic'] = bool(tensor_digest(model_b.state_dict().items()) == initial)
    require(checks['synthetic_construction_deterministic'], 'synthetic model construction is not deterministic')
    opt_b, ema_b, _ = ft.make_optimizer_ema(model_b, fixture)
    single = ft.FineTuneEngine(model_b, criterion_b, opt_b, ema_b, fixture, arm='D0',
                               plan_sha256='0' * 64, parent_sha256='a' * 64, alt_ema=None)
    reset()
    single_records = []
    for epoch in (1, 2):
        x, t, tr = window(epoch)
        single_records.append(single.train_window(x, t, capture=True))
    single_rng = native_state.capture_rng(torch.device('cpu'))
    keys = ('lr', 'loss', 'gradient_norm_before_clip', 'gradient_sha256_before_clip',
            'output_sha256', 'dn_noise_sha256', 'microbatch_losses', 'normalizers')
    checks['online_trajectory_identical'] = bool(all(
        all(dual_records[i][k] == single_records[i][k] for k in keys) for i in (0, 1)))
    checks['closed_rng_state_identical'] = bool(rng_equal(dual_rng, single_rng))
    checks['ema_r_identical_across_engines'] = bool(
        dual_r_digest == tensor_digest(single.ema.module.state_dict().items()))
    require(checks['online_trajectory_identical'], 'the second EMA changed the online training trajectory')
    require(checks['closed_rng_state_identical'], 'the second EMA changed the closed RNG state')
    require(checks['ema_r_identical_across_engines'], 'EMA-R differs between the dual and single engines')

    # ---- save/restore of the complete dual-EMA state ---------------------------
    dual.execution_binding = {'scope': 'dual_ema_check', 'arm': 'D0'}
    record = ft.save(dual, out / 'dual_state.pth')
    payload = torch.load(out / 'dual_state.pth', map_location='cpu', weights_only=True)
    checks['payload_keys'] = sorted(payload)
    checks['payload_ema_r_updates'] = int(payload['ema']['updates'])
    checks['payload_ema_c_updates'] = int(payload['ema_alt']['updates'])
    checks['payload_has_both_emas'] = bool('ema' in payload and 'ema_alt' in payload
                                           and payload['ema_alt']['updates'] == payload['ema']['updates'] + ft.ALT_OFFSET)
    require(checks['payload_has_both_emas'], 'saved state does not carry both EMA records with their counters')
    model_c, criterion_c, _ = build(fixture, synthetic=True)
    opt_c, ema_c, alt_c = ft.make_optimizer_ema(model_c, fixture)
    clone = ft.FineTuneEngine(model_c, criterion_c, opt_c, ema_c, fixture, arm='D0',
                              plan_sha256='0' * 64, parent_sha256='a' * 64, alt_ema=alt_c)
    clone.execution_binding = dual.execution_binding
    restored = ft.restore(clone, record)
    checks['restore_full_state_exact'] = bool(restored['full_state_RNG_exact'])
    checks['restore_ema_r_updates'] = int(restored['ema_r_updates'])
    checks['restore_ema_c_updates'] = int(restored['ema_c_updates'])
    checks['restore_modules_bitwise'] = bool(
        tensor_digest(clone.ema.module.state_dict().items()) == dual_r_digest
        and tensor_digest(clone.alt_ema.module.state_dict().items()) == dual_c_digest)
    require(checks['restore_full_state_exact'] and checks['restore_modules_bitwise'], 'dual-EMA restore is not exact')

    checks['cuda_initialized'] = bool(torch.cuda.is_initialized())
    require(not checks['cuda_initialized'], 'dual-EMA check initialized CUDA')
    result = {'status': 'PASS', 'checks': checks, 'elapsed_seconds': time.monotonic() - started,
              'scope': 'CPU synthetic native128; targeted dual-EMA checks before the release freeze',
              'note': ('check 2 proves the second EMA consumes no training RNG draw and cannot change the online '
                       'trajectory; checks 3 prove the two records follow the original math with counters 0 and 36480; '
                       'check 4 proves both records and counters are stored and restored exactly')}
    (out / 'DUAL_EMA_CHECK.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'checks': {k: v for k, v in checks.items() if not k == 'payload_keys'}}))


if __name__ == '__main__':
    main()
