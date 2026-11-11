"""R27-002 L0 checks, CUDA stages (TASK_R27_L0_CONTROL_002 section 4.2-4.4).

zero(c, out, counts):   zero-update identity on the two microbatches of the formal first window.  Reference = native
                        L network from the same parent consuming the explicit I_L = to_low(I_H); production = L0
                        consuming I_H.  Sequential, one graph at a time; 4 full forwards, no optimizer step.
accept(c, out, counts): formal 8x2 / BF16 L0: initial state vs the R27 formal-P e060 state, u1, u2, save u1, restore,
                        replay u2 (3 updates, 6 full forwards), shared audit fields vs R27 formal-P / formal-C.
Both use their own engines, which are discarded; an acceptance state can never continue as formal.
"""
import gc
from pathlib import Path

import l0_common as K
import l0_checks_state as H


def _first_windows(c, n):
    from formal_data import windows
    from sparse_rtdetr.research import scale_baseline as S
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    return list(windows(data, K.PARENT_UPDATE, limit=n))


def _free(torch):
    gc.collect()
    torch.cuda.empty_cache()


def zero(c, out, counts):
    import r27_run as R27  # micro_steps (pure) and native_l (reference constructor) only
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r27_l0_control as L
    from sparse_rtdetr.research import r27_local_memory as R
    torch = K.torch_mod()
    dev = "cuda:0"
    (x, t, inp), = _first_windows(c, 1)
    K.require(inp["update"] == K.PARENT_UPDATE + 1 and tuple(x.shape) == (16, 3, *R.H_HW), "zero-step window")
    xl = R.to_low(x)
    res = {"stage": "cuda_zero_step", "workload": c["workload"]["cuda_zero_step"], "input": inp, "precision": "bfloat16",
           "microbatches": [8, 8], "I_H": list(x.shape), "I_L": list(xl.shape),
           "reference": "r27_run.native_l (native L network, same parent, S15 checkpoint segments) on explicit I_L = to_low(I_H)",
           "informational_I_L_cpu_equals_to_low_on_device": [bool(torch.equal(R.to_low(x[i:i + 8].to(dev)).cpu(), xl[i:i + 8])) for i in (0, 8)]}
    # ---- reference
    K.PHASE["phase"] = "zero:reference"
    ref_e, _ = R27.native_l(c, dev, fixture=False)
    ref_e.model.register_forward_hook(lambda *_: counts.update(reference_forwards=1))
    ref_e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))
    old_names = [n for n, _ in ref_e.model.named_parameters()]
    rng0 = ck.capture_rng(ref_e.device)
    p_ref0, bn_ref0 = tensor_digest(ref_e.model.named_parameters()), H.bn_digest(ref_e.model)
    nl = R27.micro_steps(ref_e.model, ref_e.criterion, xl, t, dev, "bfloat16", old_names=old_names)
    nl_rng = ck.capture_rng(ref_e.device)
    p_ref1, bn_ref1 = tensor_digest(ref_e.model.named_parameters()), H.bn_digest(ref_e.model)
    del ref_e
    _free(torch)
    # ---- L0
    K.PHASE["phase"] = "zero:L0"
    e, checks = L.create(c["parent"]["path"], dev)
    K.register_counts(e, counts)
    names = [n for n, _ in e.model.named_parameters()]
    start_rng_equal = ck.exact_equal(ck.capture_rng(e.device), rng0)
    ck.restore_rng(rng0, e.device)
    p0, bn0 = tensor_digest(e.model.named_parameters()), H.bn_digest(e.model)
    mine = R27.micro_steps(e.model, e.criterion, x, t, dev, "bfloat16", old_names=old_names)
    mine_rng = ck.capture_rng(e.device)
    p1, bn1 = tensor_digest(e.model.named_parameters()), H.bn_digest(e.model)
    del e
    _free(torch)
    eq = {"parameter_names_equal": names == old_names,
          "start_parameters_equal": p0 == p_ref0 == checks["old_parameters_sha256"],
          "start_BN_buffers_equal": bn0 == bn_ref0,
          "start_RNG_equal_after_construction": start_rng_equal,
          "losses_equal": mine["losses"] == nl["losses"] and mine["loss"] == nl["loss"],
          "outputs_equal": mine["outputs"] == nl["outputs"],
          "DN_equal": mine["dn"] == nl["dn"],
          "native_preclip_gradients_equal": mine["old_gradient_sha256"] == nl["old_gradient_sha256"],
          "RNG_after_equal": ck.exact_equal(mine_rng, nl_rng),
          "BN_buffers_after_equal": bn1 == bn_ref1,
          "L0_non_native_gradients_empty": mine["new_gradient_norms"] == {},
          "parameters_unchanged_L0": p1 == p0, "parameters_unchanged_reference": p_ref1 == p_ref0,
          "BN_buffers_changed_by_forward": bn1 != bn0,
          "RNG_consumed_by_forward": not ck.exact_equal(mine_rng, rng0)}
    res.update(comparisons=eq, loss={"L0": mine["loss"], "reference": nl["loss"]},
               microbatch_losses={"L0": mine["losses"], "reference": nl["losses"]},
               loss_terms_per_microbatch=[len(v) for v in mine["losses"]],
               outputs_sha256={"L0": mine["outputs"], "reference": nl["outputs"]},
               DN_sha256={"L0": mine["dn"], "reference": nl["dn"]},
               native_preclip_gradient_sha256={"L0": mine["old_gradient_sha256"], "reference": nl["old_gradient_sha256"]},
               BN_buffers_after={"L0": bn1, "reference": bn_ref1}, native_parameter_tensors=len(old_names),
               L0_non_native_gradients=sorted(mine["new_gradient_norms"]),
               forwards={"reference": counts["reference_forwards"], "L0": counts["forwards"],
                         "total": counts["reference_forwards"] + counts["forwards"]},
               optimizer_steps=counts["actual_updates"], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               formal_state_usable=False)
    K.require(all(eq.values()), ("CUDA zero-step identity failed", {k: v for k, v in eq.items() if not v}))
    K.require(counts["reference_forwards"] == 2 and counts["forwards"] == counts["forward_attempts"] == 2
              and counts["actual_updates"] == 0, f"zero-step counts {dict(counts)}")
    return res


def accept(c, out, counts):
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r27_l0_control as L
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    wins = _first_windows(c, 2)
    res = {"stage": "cuda_accept", "workload": c["workload"]["cuda_accept"]}
    K.PHASE["phase"] = "accept:construct"
    e, checks = L.create(c["parent"]["path"], "cuda:0")
    K.PHASE["phase"] = "accept:initial state vs R27 formal-P e060"
    res["initial_vs_R27_e060"] = H.initial_vs_r27(c, e)
    K.register_counts(e, counts)
    bn0 = _bn_counts(e.model)
    rows, results = [], []
    for i, (x, t, inp) in enumerate(wins, 1):
        K.PHASE["phase"] = f"accept:u{i}"
        r = e.train_window(x, t, capture=True)
        K.require(r["update"] == inp["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates,
                  "acceptance update clock/input")
        results.append(r)
        rows.append({"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": r["loss"], "lr": r["lr"],
                     "gradient_norm_before_clip": r["gradient_norm_before_clip"], "ema_updates": e.ema.updates})
        if i == 1:
            saved = K.save(c, e, out / "u1.pth", "acceptance")
    expected = S.capture(e)
    bn = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    K.require(bn == [4], f"BN delta {bn}")
    del e
    gc.collect()
    torch.cuda.empty_cache()
    K.PHASE["phase"] = "accept:restore+replay u2"
    clone, _ = L.create(c["parent"]["path"], "cuda:0")
    K.register_counts(clone, counts)
    K.load(c, clone, out / "u1.pth", "acceptance")
    x, t, inp = wins[1]
    replay = clone.train_window(x, t, capture=True)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(results[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and ck.exact_equal(expected["rng"], got["rng"]), f"replay differs {diff[:5]}")
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"acceptance counts {dict(counts)}")
    # shared audit fields (declared set) vs R27 formal-P and formal-C; loss / gradient norm are informational only
    K.PHASE["phase"] = "accept:shared fields"
    shared, info = {}, []
    for arm in ("P", "C"):
        theirs = H.r27_rows(c, arm, 2)
        shared[arm] = [K.compare_shared(row, ref) for row, ref in zip(rows, theirs)]
        for row, ref in zip(rows, theirs):
            info.append({"update": row["input"]["update"], "R27_arm": arm,
                         "loss": {"L0": row["loss"], "R27": ref["loss"], "equal": row["loss"] == ref["loss"]},
                         "gradient_norm_before_clip": {"L0": row["gradient_norm_before_clip"], "R27": ref["gradient_norm_before_clip"],
                                                       "equal": row["gradient_norm_before_clip"] == ref["gradient_norm_before_clip"]}})
    K.require(shared["P"] == shared["C"], "shared fields of R27 formal-P and formal-C disagree")
    res.update(create_checks={k: v for k, v in checks.items() if k != "new_parameter_names"},
               first_updates=[{**row, "output_sha256": r["output_sha256"], "dn_noise_sha256": r["dn_noise_sha256"],
                               "gradient_sha256_before_clip": r["gradient_sha256_before_clip"],
                               "elapsed_seconds": r["elapsed_seconds"]} for row, r in zip(rows, results)],
               saved_u1=saved, replay_exact=True, full_state_exact=True, rng_exact=True,
               compared_result_fields=sorted(_result_body(replay)), replay_elapsed_seconds=replay["elapsed_seconds"],
               BN_delta_two_updates=bn,
               shared_fields={"statement": "共享字段相同 (declared shared audit fields equal; not a claim about all training tensors)",
                              "fields": list(K.SHARED_FIELDS), "rows": [1, 2], "R27_formal_P": True, "R27_formal_C": True,
                              "values": shared["P"], "R27_files": c["R27_updates"]},
               informational_not_gated=info,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               formal_state_usable=False)
    return res
