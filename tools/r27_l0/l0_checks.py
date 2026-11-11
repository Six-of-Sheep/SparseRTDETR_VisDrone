"""R27-002 L0 checks, CPU stage (TASK_R27_L0_CONTROL_002 section 4.1); zero / accept live in l0_checks_cuda.py.

cpu(c, out, counts): behaviour declaration, native key set, parent restore, H/L caches, input identity, the small
fixture (physical batch 1, accumulation 2, FP32, H engine canvas / L model canvas; u1, u2, restore, replay u2:
3 fixture updates and 6 full forwards) and the wrong-arm / wrong-scope / endpoint refusals.  The synthetic unit
tests run in a subprocess; their forwards are not part of this stage's account.  Fixture states are scope
"cpu-fixture" and cannot enter formal training.
"""
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

import l0_common as K
import l0_checks_state as H
from l0_checks_cuda import accept, zero  # noqa: F401  (dispatched by l0_run)

TESTS = ("tests/test_r27_l0_control.py", "tests/test_r27_l0_common.py")


def _caches(model):
    d, e = model.decoder, model.encoder
    out = {"decoder_eval_spatial_size": list(d.eval_spatial_size), "encoder_eval_spatial_size": list(e.eval_spatial_size),
           "anchors": list(d.anchors.shape), "valid_mask": list(d.valid_mask.shape), "feat_strides": list(e.feat_strides)}
    for i in e.use_encoder_idx:
        out[f"pos_embed{i}"] = list(getattr(e, f"pos_embed{i}").shape)
    return out


def cpu(c, out, counts):
    torch = K.torch_mod()
    from formal_data import windows
    from r25_run import setup
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.geometry import validate_geometry
    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.research import r27_l0_control as L
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.research import scale_baseline as S
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "CPU checks hide CUDA")
    out = Path(out)
    setup("cpu")  # deterministic algorithms and seeds, as for the CUDA actions (l0_run calls it only for those)
    threads = int(os.environ.get("L0_CPU_THREADS", "16"))
    K.require(threads >= 1, "L0_CPU_THREADS")
    torch.set_num_threads(threads)
    K.require(torch.get_num_threads() == threads, f"intra-op threads {torch.get_num_threads()} != {threads}")
    t0 = time.perf_counter()
    res = {"stage": "cpu_fixture", "workload": c["workload"]["cpu_fixture"],
           "threads": {"L0_CPU_THREADS": os.environ.get("L0_CPU_THREADS"), "default": 16, "torch_num_threads": torch.get_num_threads(),
                       "torch_num_interop_threads": torch.get_num_interop_threads(), "cpu_count": os.cpu_count(),
                       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                       "note": "the gate is the in-process restore/replay identity; fixture losses are not required to "
                               "be bit-identical across processes"}}
    # 1. synthetic unit tests (subprocess; not counted here)
    K.PHASE["phase"] = "cpu:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-4000:])
    # 2. behaviour declaration
    K.PHASE["phase"] = "cpu:declaration"
    old = K.read_json(K.pinned({k: c["R27_contract"][k] for k in ("path", "bytes", "sha256")}))
    I = L.IDENTITY
    decl = {"identity_sha256": L.IDENTITY_SHA256, "contract_identity_equal": c["identity_sha256"] == L.IDENTITY_SHA256 and c["identity"] == I,
            "arm": I["arm"], "no_local_branch": all(I[k] is None for k in ("local_backbone", "local_projection", "local_read",
                                                                          "selector", "route_keys", "local_recompute")),
            "new_learnable_parameters": I["new_learnable_parameters"], "H": I["H"], "L": I["L"],
            "H_to_L": I["H_to_L"], "H_to_L_equals_R27_contract": I["H_to_L"] == old["identity"]["H_to_L"],
            "R27_identity_differs": old["identity_sha256"] != L.IDENTITY_SHA256, "is_not": sorted(I["is_not"])}
    K.require(decl["contract_identity_equal"] and decl["arm"] == K.ARM == L.ARM and decl["no_local_branch"]
              and decl["new_learnable_parameters"] == 0 and decl["H_to_L_equals_R27_contract"] and decl["R27_identity_differs"]
              and (tuple(I["H"]), tuple(I["L"])) == (R.H_HW, R.L_HW) == ((1152, 2016), (768, 1344)), ("L0 declaration", decl))
    res["declaration"] = decl
    # 3. construction from the parent: native key set, parent restore, H/L caches
    K.PHASE["phase"] = "cpu:construct"
    parent = K.pinned(c["parent"])
    e, checks = L.create(parent, "cpu", fixture=True)
    ps = torch.load(parent, map_location="cpu", weights_only=True)
    sd, esd = e.model.state_dict(), e.ema.module.state_dict()
    keys = {"online_equal_parent": list(sd) == list(ps["model"]), "EMA_equal_parent": list(esd) == list(ps["ema"]["module"]),
            "state_keys": len(sd), "R27_keys": sum(H.r27_only(k) for k in sd)}
    K.require(keys["online_equal_parent"] and keys["EMA_equal_parent"] and keys["R27_keys"] == 0, ("native key set", keys))
    validate_geometry(e.model)
    validate_geometry(e.ema.module)
    cache = {"engine_input_hw": list(e.config.input_hw), "online": _caches(e.model), "EMA": _caches(e.ema.module),
             "validate_geometry": "passed (online and EMA)", "global_model_config_input_hw": checks["global_model_config"]["input_hw"]}
    n_l = sum((R.L_HW[0] // s) * (R.L_HW[1] // s) for s in cache["online"]["feat_strides"])
    K.require(tuple(cache["engine_input_hw"]) == R.H_HW and cache["online"] == cache["EMA"]
              and tuple(cache["online"]["decoder_eval_spatial_size"]) == tuple(cache["online"]["encoder_eval_spatial_size"]) == R.L_HW
              and cache["online"]["anchors"][1] == cache["online"]["valid_mask"][1] == n_l, ("H/L caches", cache))
    res.update(native_keys=keys, caches=cache, old_parameters_sha256=checks["old_parameters_sha256"],
               create_checks={k: v for k, v in checks.items() if k != "new_parameter_names"})
    # 4. input identity: formal first window vs the bound R27 formal-P first row
    K.PHASE["phase"] = "cpu:input"
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    (x, t, inp), = list(windows(data, K.PARENT_UPDATE, limit=1))
    theirs = H.r27_rows(c, "P", 1)[0]["input"]
    K.require(set(inp) == set(theirs) == set(K.INPUT_FIELDS), "input record keys")
    for k in K.INPUT_FIELDS:
        K.require(type(inp[k]) is type(theirs[k]) and inp[k] == theirs[k], f"input identity differs from R27 formal-P row 1: {k}")
    res["input_identity"] = {"equal_fields": list(K.INPUT_FIELDS), "record": inp, "R27_file": c["R27_updates"]["P"],
                             "window_shape": list(x.shape)}
    # 5. fixture u1, u2 -> save u1 -> fresh engine, restore, replay u2
    x, t = x[:2], t[:2]
    res["fixture"] = fx = {"images": [tt["image_id"] for tt in t], "physical_batch": 1, "accumulation": 2, "precision": "float32",
                           "engine_canvas": list(R.H_HW), "model_canvas": list(R.L_HW), "same_two_images_for_u1_u2": True}
    K.register_counts(e, counts)
    bn0 = _bn_counts(e.model)
    rows = []
    for i in (1, 2):
        K.PHASE["phase"] = f"cpu:fixture u{i}"
        r = e.train_window(x, t, capture=True)
        K.require(r["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates, "fixture clock")
        rows.append(r)
        if i == 1:
            side = K.save(c, e, out / "cpu-fixture-u1.pth", "cpu-fixture")
    expected = S.capture(e)
    fx["BN_delta_two_updates"] = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    K.require(fx["BN_delta_two_updates"] == [4], f"fixture BN delta {fx['BN_delta_two_updates']}")
    del e
    gc.collect()
    K.PHASE["phase"] = "cpu:fixture restore+replay"
    clone, _ = L.create(parent, "cpu", fixture=True)
    K.register_counts(clone, counts)
    K.load(c, clone, out / "cpu-fixture-u1.pth", "cpu-fixture")
    replay = clone.train_window(x, t, capture=True)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and ck.exact_equal(expected["rng"], got["rng"]), f"fixture replay differs {diff[:5]}")
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"fixture counts {dict(counts)}")
    fx.update(saved_u1=side, full_state_exact=True, rng_exact=True, replay_result_exact=True,
              compared_result_fields=sorted(_result_body(replay)), loss=[r["loss"] for r in rows], replay_loss=replay["loss"],
              output_sha256_u2=rows[1]["output_sha256"], elapsed_seconds=[r["elapsed_seconds"] for r in rows + [replay]],
              cross_process_trace={"note": "informational: lets two processes be compared step by step (not gated)",
                                   "output_sha256": [r["output_sha256"] for r in rows],
                                   "dn_noise_sha256": [r["dn_noise_sha256"] for r in rows],
                                   "gradient_sha256_before_clip": [r["gradient_sha256_before_clip"] for r in rows],
                                   "gradient_norm_before_clip": [r["gradient_norm_before_clip"] for r in rows]})
    # 6. refusals
    K.PHASE["phase"] = "cpu:refusals"
    st = out / "cpu-fixture-u1.pth"

    def r27_wrap():
        m, _, _ = build(Config(seed=0), synthetic=True)
        R.install(m, "P")
        L.wrap(m)

    fake = dict(ps["ema"]["module"])
    fake[R.LOCAL + ".x"] = torch.zeros(1)
    cases = [("fixture state read in formal scope", lambda: K.read_side(c, st, "formal")),
             ("fixture state loaded as formal", lambda: K.load(c, clone, st, "formal")),
             ("R27 formal-P e120 state loaded as formal", lambda: K.load(c, clone, c["P_endpoint"]["path"], "formal")),
             ("R27 accept-N u1 state loaded as acceptance", lambda: K.load(c, clone, K.R27_ROUND / "accept-N/u1.pth", "acceptance")),
             ("build_eval_model with an R27-P shaped key set", lambda: L.build_eval_model(fake, "cpu")),
             ("wrap of a model with R27 installed", r27_wrap)]
    res["refused"] = [H.refused(label, fn) for label, fn in cases[:4]]
    K.require(ck.exact_equal(S.capture(clone), got), "a refused load changed the engine")
    res["refused"] += [H.refused(label, fn) for label, fn in cases[4:]]  # these build fresh models (initialisers draw RNG)
    try:  # before formal training the round has no endpoint; a rerun after it finds one and records that instead
        res["refused"].append(H.refused("endpoint_state without an endpoint", lambda: K.endpoint_state(c)))
    except AssertionError:
        res["refused"].append({"case": "endpoint_state without an endpoint", "skipped": "this round already has an L0 endpoint"})
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"cpu counts {dict(counts)}")
    res.update(fixture_updates=3, full_forwards=6, formal_state_usable=False, cpu_seconds_wall=time.perf_counter() - t0)
    return res
