"""R28-F mapping control (TASK_R28_FOVEA_MAPPING_CONTROL_004): an exploratory direct-precedent control added after the
W and K results were seen.  One new arm F; nothing of the closed W round or of the K follow-up round is rerun or
written, and the closed W decision (closed_fixed_config, K_trigger=false) is neither consulted nor changed.

Arm F = arm W's frozen demand network, parent, canvas, detector, data and schedule with the FOVEA-style separable
attraction map (src/sparse_rtdetr/research/r28_fovea.py) in place of the bounded allocation.

  prepare        contract of the independent round: authorization basis, task and review files, the pinned official
                 FOVEA files, read-only references into the closed W round and the K round, the release difference.
  fovea_check    CPU, affected interfaces only: unit tests; the official rule executed from the pinned official file
                 against the production nodes; identities, authorisation and refusals; geometry on the real first
                 training window (same demand as W, map, GT round trip, complete tent support against a dense
                 reference, arm W's renderer reproduced inside its support); CPU fixture u1 / u2 / restore / replay;
                 the summary arithmetic.
  wire           CUDA wiring check of the code that only runs after training: one evaluation batch and a shortened
                 L0 / W / F timing with stage readings (F on the parent EMA weights; nothing is scored or trained).
  summarize      summary-F, RECORD_ONLY: W - F, F - L0, F - K, costs, limits.  No pass / fail gate.
Accept, formal training, evaluation and timing are r28_checks.accept, r28_loop.formal, r28_eval.evaluate and
r28_cost.cost with R28_ARM=F.
"""
import contextlib
import gc
import importlib.util
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import r28_common as K
import r28_followup as FU

REGISTRATION = "R28-F exploratory direct-precedent mapping control after the W and K results"
ARM = "F"
K_ROUND = K.LD_ROOT / "r28-fixed-prior-control-20261114-001"
TASK = ("TASK_R28_FOVEA_MAPPING_CONTROL_004.md", "0cd799d9f149c0d5108f5f1c5d8a0b881f976aaf0d28cb9b452c5750c26a7d96")
REVIEW = ("REVIEW_R28_K_AND_NEXT_004.md", "4185a80a0563a014acc937eea22fe5c6956b4d37345a8593673bd01fe7065fc1")
OFFICIAL_DIR = "official"
OFFICIAL_FILES = {"grid_generator": ("grid_generator.py", "fovea/models/grid_generator.py"),
                  "plain_kde_grid": ("plain_kde_grid.py", "configs/models/plain_kde_grid.py")}
PIN_K = {  # TASK_R28_FOVEA_MAPPING_CONTROL_004 section 2 and the K round's own receipts
    "commit": "1b4f2a24f7d5a6fbbc953d6ad7e742e3fd2f63ca",
    "manifest": "021a71e68803ca355ff4e4c70680d7600ffb54bf37b02ad88a5637a990429214",
    "contract_file": "80eccd68dd18c6da898f8c8f34c912dfcce3f4b3a29f304fe6220606002214ab",
    "contract": "a16593e791c73bcf8509f1bb6e2123d84f3916b7c8bb5d40547c6dcd313d3988",
    "result": "96c6e14e47fd855c6dd5dbf0b97cdd61f132eae7122f7ac7dba9d32ac03fa9be",
    "endpoint": "5abb8d022d3201c435e13f9001cdf1efe410b88e611b2a7f9b47baa3791e4cbd",
}
INHERITED_K = ("followup-check", "accept-K", "formal-K", "eval-K", "cost-K", "medium", "summary-K")
OWN_STAGES = ("fovea-check", "wire-F", "accept-F", "formal-F", "eval-F", "cost-F", "summary-F")
ALLOWED_CHANGED = ("tools/r28/r28_common.py", "tools/r28/r28_eval.py", "tools/r28/r28_loop.py", "tools/r28/r28_run.py",
                   "tools/r28/r28_cost.py", "tools/r28/r28_checks.py", "tools/r28/r28_followup.py")
ALLOWED_ADDED = ("src/sparse_rtdetr/research/r28_fovea.py", "tests/test_r28_fovea.py", "tests/test_r28_fovea_control.py",
                 "tools/r28/r28_fovea_control.py", "tools/r28/r28_f_control_chain.sh",
                 "tests/test_r28_followup.py", "tools/r28/r28_followup.py", "tools/r28/r28_k_followup_chain.sh")
SCIENCE_ADDED = ("src/sparse_rtdetr/research/r28_fovea.py",)
TESTS = ("tests/test_r28_warp.py", "tests/test_r28_resample.py", "tests/test_r28_summary.py", "tests/test_r28_followup.py",
         "tests/test_r28_fovea.py", "tests/test_r28_fovea_control.py")
ARMS4 = ("L0", "W", "K", "F")
PAIRS = (("W", "F"), ("F", "L0"), ("F", "K"), ("W", "L0"), ("K", "L0"), ("W", "K"))
PRIMARY = (("600", "APS"), ("600", "AP75_small"))
WIRE = {"eval_batches": 1, "cost_warmup": 1, "cost_batches_per_pass": 1, "cost_passes": 1}
TOL = {"fp64_same_formula": 1e-10, "official_fp32_nodes": 1e-5, "official_fp32_dense": 1e-5, "round_trip": 1e-9,
       "render_fp32_vs_fp64": 1e-5, "endpoint": 1e-12}
WORKLOAD = {
    "fovea-check": {"device": "cpu", "updates": 3, "forwards": 6, "COCOeval_calls": 0,
                    "note": "CPU fixture (2 real images, FP32, u1 / u2 / replay); engineering check, listed apart from the formal chain"},
    "wire": {"updates": 0, "forwards": WIRE["eval_batches"] + 3 * (WIRE["cost_warmup"] + WIRE["cost_passes"] * WIRE["cost_batches_per_pass"]),
             "scorer_calls": 0, "note": "wiring of the evaluation batch and of the staged timing; engineering check, listed apart"},
    "accept": K.WORKLOAD["accept"],
    "formal": K.WORKLOAD["formal"],
    "evaluation": K.WORKLOAD["evaluation"],
    "cost-F": K.WORKLOAD["cost-F"],
    "planned_complete_detector_forwards": 6 + 2 * (K.ENDPOINT - K.PARENT_UPDATE) + (K.EVAL_WARMUP + K.EVAL_BATCHES)
    + len(K.COST_ENDPOINTS["cost-F"]) * K.COST_PER_ENDPOINT,
    "planned_updates": 3 + (K.ENDPOINT - K.PARENT_UPDATE),
    "not_run": ["demand network training", "G0", "W training", "K training", "W / K / L0 evaluation or rescoring",
                "a development scoring short chain", "any seed, parameter or schedule variation"],
}
STATEMENTS = {
    "registration": "本对照形成于看过 W 与 K 结果之后，是探索性的直接先例控制；不是最初 R28 预注册链的一部分，也不是未看结果的预注册检验。",
    "W_decision": "旧 W 的 closed_fixed_config（APM600 未过）、K_trigger=false 与旧成本 C 原样保留；F 的任何读数都不追认或改写它们。",
    "scope": "F 是 FOVEA 式映射的受控适配：与 W 共用同一冻结需求网络、父态、画布、检测器、数据与日程，只换映射规则；"
             "不是 FOVEA 整篇方法的复现，不冒用其 KDE／视频历史显著性方案，也不说明相对 LZU、ZoomDet 的优劣。",
    "adaptations": "相对官方代码的适配：显著性来自 R28 冻结需求网络的边际；attraction_fwhm 取发布配置的 4（构造器默认 13 未用，"
                   "两者均未在 VisDrone 上调过）；图像按像素中心、用与 W 相同的 tent 核渲染（非官方 bilinear grid_sample）；"
                   "框用同一分段线性映射及线性延拓（非官方 unwarp_bboxes）；映射算术为 FP64。",
    "record_only": "W−F 等比较登记为 RECORD_ONLY：给出全部绝对值、差值、成本与局限，不设新的通过门，也不套用旧七格作一票否决。",
    "contribution": "方法贡献仍为 not_established：单 seed（seed0）、结果后追加、重复使用的 548 图 development 集；"
                    "直接对照、稳定性与实质差异证据由总控评估。",
    "equivalence": "F 与 W 读数接近不构成等效性证明；W 读数更高也只是单 seed 线索，不自动授权后续实验。",
    "cost": "L0／W／F 同进程测速的新比值单列，不覆盖 W 轮登记的 C，也不与 K 轮的 L0／W／K 比值混为同一次测量。",
    "large": "large 读数由少数稀疏类主导（见逐类表），不作整体优劣依据；APL6 只是补充读数。",
}
LABEL = "mapping_control_completed"
CONTRIBUTION = "not_established"


# ================================================================ release difference
def source_delta(old_files, new_files, *, label):
    """File-by-file difference between a closed round's release and this one.  The science tree may only gain the arm F
    module: no file of src/, vendor/ or configs/ is changed or removed."""
    changed = sorted(k for k in old_files if k in new_files and old_files[k] != new_files[k])
    added = sorted(k for k in new_files if k not in old_files)
    removed = sorted(k for k in old_files if k not in new_files)
    K.require(not removed and set(changed) <= set(ALLOWED_CHANGED) and set(added) <= set(ALLOWED_ADDED),
              f"release differs from {label} outside the arm F adapter: changed {changed}, added {added}, removed {removed}")
    science = ("src/", "vendor/", "configs/")
    K.require(not any(p.startswith(science) for p in changed) and all(p in SCIENCE_ADDED for p in added if p.startswith(science)),
              f"release touches the science tree of {label} beyond the added arm F module")
    return {"against": label, "changed": changed, "added": added, "removed": removed, "unchanged_files": len(old_files) - len(changed),
            "existing_science_files_identical": True, "science_files_added": [p for p in added if p.startswith(science)],
            "note": "every existing file of src/, vendor/ and configs/ (arm W / K operator and wrapper, detector, criterion, "
                    "data) is byte-identical; arm F is one added module"}


# ================================================================ references into the K round (read-only)
def k_references(k_round=None):
    k_round = Path(k_round or K_ROUND).resolve()
    contract = K.ref(k_round / "CONTRACT.json")
    K.require(contract["sha256"] == PIN_K["contract_file"], "K round CONTRACT.json file SHA")
    ck = K.read_json(k_round / "CONTRACT.json")
    body = {k: v for k, v in ck.items() if k != "contract_sha256"}
    K.require(ck["contract_sha256"] == K.sha_bytes(K.canonical(body)) == PIN_K["contract"], "K round contract digest")
    K.require(ck["source_commit"] == PIN_K["commit"] and ck["source_manifest"]["sha256"] == PIN_K["manifest"]
              and ck["authorization_basis"] == K.FOLLOWUP_BASIS and Path(ck["outputs"]).resolve() == k_round
              and ck["parent"]["sha256"] == K.PARENT_SHA, "K round contract identities")
    K.pinned(ck["source_manifest"])
    inherited = {}
    for name in INHERITED_K:
        r = K.ref(k_round / name / "CLOSED.json")
        closed = K.read_json(r["path"])
        K.require(closed.get("status") == "PASS" and closed.get("contract_sha256") == ck["contract_sha256"],
                  f"K round stage {name} is not PASS under the K contract")
        inherited[name] = r
    summary, ev = K.read_json(inherited["summary-K"]["path"]), K.read_json(inherited["eval-K"]["path"])
    K.require(summary["result"]["sha256"] == PIN_K["result"] and ev["endpoint"]["sha256"] == PIN_K["endpoint"]
              and ev["arm"] == "K" and Path(ev["endpoint"]["path"]).parent == k_round / "formal-K", "K result / endpoint SHA")
    for r in (summary["result"], ev["endpoint"], *ev["predictions"].values(), *ev["metrics"].values()):
        K.pinned(r)
    return ck, inherited, {"K_round": str(k_round), "K_contract": contract, "K_contract_sha256": ck["contract_sha256"],
                           "K_commit": ck["source_commit"], "K_manifest": ck["source_manifest"], "K_result": summary["result"],
                           "K_endpoint": ev["endpoint"], "K_weights_sha256": ev["weights_sha256"],
                           "K_predictions": ev["predictions"], "K_metrics": ev["metrics"]}


def official_refs():
    from sparse_rtdetr.research import r28_fovea as FV
    out = {}
    for key, (name, official_path) in OFFICIAL_FILES.items():
        r = K.ref(K.ROUND / OFFICIAL_DIR / name)
        K.require(r["sha256"] == FV.OFFICIAL["files"][official_path], f"official FOVEA file SHA: {name}")
        out[key] = r
    return {**out, "repository": FV.OFFICIAL["repository"], "commit": FV.OFFICIAL["commit"], "rule": FV.OFFICIAL["rule"]}


# ================================================================ contract of the independent round
def prepare(_args=None):
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    round_ = K.ROUND.resolve()
    K.require(K.ROOT.resolve().parent.parent == round_ and K.ROOT.name == "source", "prepare-fovea must run from a frozen release")
    K.require("R28_ROUND" in os.environ and all(round_ != p and p not in round_.parents for p in (FU.PRIOR_ROUND.resolve(), K_ROUND.resolve())),
              "the mapping-control round must be an independent directory named by R28_ROUND")
    # a dev round (inside the round directory) only shakes out the checks: its chain must stop before formal training
    K.require(not K.DEV or round_.parent.parent == K.LD_ROOT.resolve(), "a mapping-control dev round lives inside the round directory")
    manifest = K.read_json(K.ROOT.parent / "MANIFEST.json")
    K.require(manifest["files"] == K.B.source_manifest(K.ROOT), "release source differs from MANIFEST")
    K.require(not manifest.get("dirty", False) or K.DEV, "a release assembled from a dirty work tree is only accepted in a dev round")
    task, review = K.ref(K.ROUND / TASK[0]), K.ref(K.ROUND / REVIEW[0])
    K.require(task["sha256"] == TASK[1] and review["sha256"] == REVIEW[1], "task / review file SHA")
    c1, refs = FU.prior_references()
    ck, k_inherited, k_refs = k_references()
    K.require(set(refs["inherited"]).isdisjoint(k_inherited) and set(OWN_STAGES).isdisjoint(set(refs["inherited"]) | set(k_inherited)),
              "stage names of the three rounds overlap")
    deltas = [source_delta(K.read_json(c1["source_manifest"]["path"])["files"], manifest["files"], label="the closed W round release"),
              source_delta(K.read_json(ck["source_manifest"]["path"])["files"], manifest["files"], label="the K round release")]
    K.require(c1["identity_sha256"] == R.IDENTITY_SHA256 and c1["identity"] == R.IDENTITY and c1["definition"] == W.DEFINITION,
              "arm W / K behaviour identity differs from the closed round")
    c = {k: v for k, v in c1.items() if k != "contract_sha256"}
    c.update(version="R28-F-control-004-contract-v1", authorization_basis=K.FOVEA_BASIS,
             source_root=str(K.ROOT.resolve()), source_manifest=K.ref(K.ROOT.parent / "MANIFEST.json"),
             source_commit=manifest["commit"], source_dirty=bool(manifest.get("dirty", False)),
             identity={**c1["identity"], **FV.IDENTITY}, identity_sha256={**c1["identity_sha256"], **FV.IDENTITY_SHA256},
             fovea_definition=FV.DEFINITION, workload=WORKLOAD,
             cost={**c1["cost"], "cost-F": {"endpoints": list(K.COST_ENDPOINTS["cost-F"]),
                                            "rotations": [list(r) for r in K.COST_ROTATIONS["cost-F"]]}},
             labels={**c1["labels"], "F_completed": LABEL, "F_contribution": CONTRIBUTION},
             resource_plan={"state_bytes": c1["parent"]["bytes"], "formal_states": len(K.SAVE_EPOCHS),
                            "disk_free_bytes_at_prepare": shutil.disk_usage(K.ROUND).free,
                            "gpu_time_reference": "task estimate 11-15 h (W measured 11.31 h, K 11.45 h); the complete tent "
                                                  "support of F may change it; not a cap"},
             outputs=str(K.ROUND), formal_started=False,
             followup={"registration": REGISTRATION, "authorization_basis": K.FOVEA_BASIS, "arm": ARM, "task": task, "review": review,
                       "official": official_refs(), **refs, **k_refs,
                       "inherited": {**refs["inherited"], **k_inherited},
                       "inherited_contracts": {name: ck["contract_sha256"] for name in k_inherited},
                       "source_delta": deltas,
                       "trained_here": ["F detector, 18,240 updates from the e060 parent"],
                       "reused_frozen": ["W demand network (demand/demand-net.pth of the closed W round): the demand source of arm F",
                                         "W e120 endpoint (timing and identity only)", "W / K / L0 predictions, metrics and "
                                         "per-class readings (quoted, not rescored)"],
                       "analysis": {"record_only": True, "primary": [f"W-F top{b} {m}" for b, m in PRIMARY],
                                    "pairs": [f"{x}-{y}" for x, y in PAIRS]},
                       "statements": STATEMENTS})
    c["contract_sha256"] = K.sha_bytes(K.canonical(c))
    K.write_new(K.ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256_F": FV.IDENTITY_SHA256["F"],
                      "authorization_basis": c["authorization_basis"],
                      "source_delta": [{k: d[k] for k in ("against", "changed", "added")} for d in deltas]}))
    return c


# ================================================================ the official rule, executed from the pinned file
@contextlib.contextmanager
def official_module(path):
    """The pinned official grid_generator.py executed unchanged.  Stand-ins exist only for what the file imports but
    the rule does not use (mmcv registry, mmdet builder, the visualisation helper), for its hard-coded .cuda() and
    for the numpy alias np.float that newer numpy removed.  The formula is the official bytes."""
    import numpy as np
    torch = K.torch_mod()

    class Registry:
        def __init__(self, name):
            self.name = name

        def register_module(self, *_a, **_k):
            return lambda cls: cls

    stubs = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.__path__ = []
        stubs[name] = m

    mod("r28_official_fovea"), mod("r28_official_fovea.models"), mod("r28_official_fovea.utils")
    mod("r28_official_fovea.utils.vis", vis_batched_imgs=lambda *_a, **_k: None)
    mod("mmcv"), mod("mmcv.utils", Registry=Registry)
    mod("mmdet"), mod("mmdet.models"), mod("mmdet.models.builder", build=lambda *_a, **_k: None)
    name = "r28_official_fovea.models.grid_generator"
    saved = {n: sys.modules.get(n) for n in (*stubs, name)}
    had_float, cuda = hasattr(np, "float"), torch.Tensor.cuda
    try:
        sys.modules.update(stubs)
        if not had_float:
            np.float = float
        torch.Tensor.cuda = lambda self, *_a, **_k: self
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        yield m
    finally:
        torch.Tensor.cuda = cuda
        if not had_float:
            del np.float
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v


def reference_nodes(s, fwhm, pad):
    """The rule as plain FP64 loops (written apart from r28_fovea): reflect padding, Gaussian window, weighted mean."""
    n = len(s)
    out = []
    for i in range(n):
        num = den = 0.0
        for j in range(i - pad, i + pad + 1):
            src = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)
            k = math.exp(-4.0 * math.log(2.0) * (i - j) ** 2 / fwhm ** 2)
            num += k * s[src] * (j / (n - 1))
            den += k * s[src]
        out.append(num / den)
    return out


def synthetic_demands():
    """Constant, centre-concentrated, edge-concentrated and bimodal 12x21 demand maps."""
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    out = {}
    d = torch.zeros(W.GRID, dtype=torch.float64)
    out["constant"] = d + 0.7
    for name, cells in (("centre", ((5, 7, 9, 12, 6.0),)),            # (row0, row1, col0, col1, value)
                        ("edge", ((0, 2, 0, 2, 8.0), (10, 12, 19, 21, 3.0))),
                        ("bimodal", ((2, 3, 3, 4, 9.0), (9, 10, 17, 18, 5.0)))):
        d = torch.zeros(W.GRID, dtype=torch.float64)
        for r0, r1, c0, c1, v in cells:
            d[r0:r1, c0:c1] = v
        out[name] = W.blur3(d)
    return out


def official_check(fu, demands):
    """demands: name -> (B, 12, 21) FP64 demand without eps.  Production nodes against (a) the official file in its
    own FP32, (b) the FP64 loop reference, (c) the official dense grid (align_corners=True) of the L canvas."""
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research.r27_local_memory import L_HW
    torch = K.torch_mod()
    cfg = {}
    exec(K.pinned(fu["official"]["plain_kde_grid"]).read_text(), {}, cfg)
    cfg = cfg["grid_generator"]
    K.require(cfg["type"] == "PlainKDEGrid" and cfg["separable"] is True and cfg["anti_crop"] is True
              and cfg["attraction_fwhm"] == FV.FWHM == 4, f"official released configuration: {cfg}")
    with official_module(K.pinned(fu["official"]["grid_generator"])) as m:
        mixin = m.RecasensSaliencyToGridMixin
        sig = inspect.signature(mixin.__init__).parameters
        defaults = {"grid_shape": sig["grid_shape"].default, "attraction_fwhm": sig["attraction_fwhm"].default,
                    "anti_crop": sig["anti_crop"].default, "separable": sig["separable"].default}
        kw = {"separable": cfg["separable"], "attraction_fwhm": cfg["attraction_fwhm"], "anti_crop": cfg["anti_crop"]}
        nodes_gen, dense_gen = mixin(output_shape=defaults["grid_shape"], **kw), mixin(output_shape=tuple(L_HW), **kw)
    K.require(tuple(nodes_gen.grid_shape) == FV.GRID_SHAPE == tuple(defaults["grid_shape"]) and nodes_gen.padding_size == FV.PAD
              and nodes_gen.padding_mode == "reflect" and tuple(nodes_gen.filter.shape) == (1, 1, 2 * FV.PAD + 1)
              and nodes_gen.filter.dtype == torch.float32, "official generator constants")
    kernel = FV.kernel_matrix(FV.GRID_SHAPE[0])[0, :FV.PAD + 1].flip(0)          # k(0..PAD)
    kernel_error = float((nodes_gen.filter[0, 0, FV.PAD:].double() - kernel).abs().max())
    K.require(kernel_error <= 1e-7, f"attraction kernel differs from the official filter: {kernel_error}")
    rows, worst = {}, dict.fromkeys(("official_fp32_nodes", "official_fp32_dense", "fp64_same_formula"), 0.0)
    for name, demand in demands.items():
        d = demand.to(torch.float64) + W.EPS
        b = d.shape[0]
        sx, sy = FV.node_saliency(d.sum(1), FV.GRID_SHAPE[1]), FV.node_saliency(d.sum(2), FV.GRID_SHAPE[0])
        (raw_x, gx), (raw_y, gy) = FV.attraction_nodes(sx), FV.attraction_nodes(sy)
        args = (torch.zeros(b, 1, 1, 1), None, sx.float()[:, None], sy.float()[:, None], "cpu")
        grid, dense = nodes_gen.separable_saliency_to_grid(*args), dense_gen.separable_saliency_to_grid(*args)
        K.require(tuple(grid.shape) == (b, *FV.GRID_SHAPE, 2) and tuple(dense.shape) == (b, *L_HW, 2) and grid.dtype == torch.float32,
                  "official grid shape")
        off = {"x": (grid[:, 0, :, 0].double() + 1) / 2, "y": (grid[:, :, 0, 1].double() + 1) / 2}
        den = {"x": (dense[:, 0, :, 0].double() + 1) / 2, "y": (dense[:, :, 0, 1].double() + 1) / 2}
        row = {"images": b}
        for axis, g, raw, s, n_out in (("x", gx, raw_x, sx, L_HW[1]), ("y", gy, raw_y, sy, L_HW[0])):
            ref = torch.tensor([reference_nodes(s[i].tolist(), FV.FWHM, FV.PAD) for i in range(b)], dtype=torch.float64)
            u = (torch.arange(n_out, dtype=torch.float64) / (n_out - 1)).expand(b, -1)       # the official align_corners positions
            e_nodes, e_dense, e_ref = (off[axis] - g).abs(), (den[axis] - FV.unwarp(g, u)).abs(), (ref - raw).abs()
            row[axis] = {"official_fp32_nodes_max": float(e_nodes.max()), "official_fp32_dense_max": float(e_dense.max()),
                         "fp64_same_formula_max": float(e_ref.max()),
                         "endpoint_raw": [float(raw[:, 0].abs().max()), float((raw[:, -1] - 1).abs().max())],
                         "clamped_nodes": int(((raw < 0) | (raw > 1)).sum()),
                         "clamped_interior_nodes": int(((raw[:, 1:-1] < 0) | (raw[:, 1:-1] > 1)).sum()),
                         "clamp_max": float((raw - raw.clamp(0, 1)).abs().max()),
                         "node_slope": [float((g[:, 1:] - g[:, :-1]).min()) * (g.shape[1] - 1), float((g[:, 1:] - g[:, :-1]).max()) * (g.shape[1] - 1)],
                         **({"official_fp32_minus_production_per_node": (off[axis] - g)[0].tolist(),
                             "production_nodes": g[0].tolist()} if b == 1 else {})}
            for k, e in (("official_fp32_nodes", e_nodes), ("official_fp32_dense", e_dense), ("fp64_same_formula", e_ref)):
                worst[k] = max(worst[k], float(e.max()))
        rows[name] = row
    K.require(all(worst[k] <= TOL[k] for k in worst), f"official / reference comparison outside tolerance: {worst}")
    return {"official": fu["official"], "released_configuration": cfg, "constructor_defaults": {k: (list(v) if isinstance(v, tuple) else v)
                                                                                                for k, v in defaults.items()},
            "used": {"grid_shape": list(FV.GRID_SHAPE), "attraction_fwhm": FV.FWHM, "anti_crop": True, "padding_size": FV.PAD},
            "execution": "the pinned official file run unchanged on the CPU; stand-ins only for its unused imports, its "
                         ".cuda() calls and the removed numpy alias np.float",
            "kernel_max_abs_difference_to_official_fp32_filter": kernel_error, "cases": rows, "worst": worst, "tolerance": TOL,
            "comparisons": {"official_fp32_nodes": "official nodes (output_shape = grid_shape) minus production FP64 nodes",
                            "official_fp32_dense": "official grid interpolated to 768x1344 with align_corners=True minus the "
                                                   "production piecewise linear G read at u = i/(L-1); the production image "
                                                   "uses pixel centres instead (declared adaptation)",
                            "fp64_same_formula": "FP64 loop reference minus production nodes before the clamp"}}


# ================================================================ geometry on real training input
def reference_axis(g, n_out, n_src):
    """Dense (n_out, n_src) FP64 tent weights of one axis from the definition (numpy, apart from r28_fovea)."""
    import numpy as np
    n = len(g) - 1
    nodes = np.arange(n + 1) / n
    i = np.arange(n_out)
    left, right, mid = (np.interp(u, nodes, g) for u in (i / n_out, (i + 1) / n_out, (i + 0.5) / n_out))
    p, d = n_src * mid - 0.5, n_src * (right - left)
    h = np.maximum(1.0, d / 1.5)
    reach = int(math.ceil(float(h.max()))) + 2
    k = np.arange(-reach, n_src + reach)
    w = np.maximum(0.0, 1.0 - np.abs(k[None, :] - p[:, None]) / h[:, None])
    support = int((w > 0).sum(1).max())
    w = w / w.sum(1, keepdims=True)
    a = np.zeros((n_out, n_src))
    np.add.at(a, (np.repeat(i, len(k)), np.tile(np.clip(k, 0, n_src - 1), n_out)), w.ravel())
    return a, support, float(d.min()), float(d.max())


def _np_unwarp(g, v):
    import numpy as np
    n = len(g) - 1
    s0, s1 = (g[1] - g[0]) * n, (g[-1] - g[-2]) * n
    return np.where(v < 0, g[0] + v * s0, np.where(v > 1, g[-1] + (v - 1) * s1, np.interp(v, np.arange(n + 1) / n, g)))


def _np_warp(g, u):
    import numpy as np
    n = len(g) - 1
    s0, s1 = (g[1] - g[0]) * n, (g[-1] - g[-2]) * n
    return np.where(u < g[0], (u - g[0]) / s0, np.where(u > g[-1], 1 + (u - g[-1]) / s1, np.interp(u, g, np.arange(n + 1) / n)))


def real_geometry(c, x, t):
    """Checks on one real training window (CPU): the demand of F equals the demand of W bit for bit and is produced
    from the image alone; the map; GT forward / inverse; the complete tent support; arm W's renderer inside its range."""
    import numpy as np
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW
    torch = K.torch_mod()
    F64 = torch.float64
    src_f, src_w = K.demand_source(c, "F"), K.demand_source(c, "W")
    K.require(src_f == src_w and src_f["file"]["sha256"] == FU.PIN["W_demand"], "arm F does not read arm W's demand network file")
    ctx_f, ctx_w = R.Context(src_f), R.Context(src_w)
    with torch.no_grad():
        d_f, d_w = ctx_f.demand(x), ctx_w.demand(x)
    K.require(torch.equal(d_f, d_w) and ctx_f.binding() == ctx_w.binding() and not ctx_f.net.training
              and tuple(d_f.shape) == (len(x), *W.GRID) and bool((d_f >= 0).all()), "F and W demand maps differ on the same input")
    plan, wplan = FV.plan_from_demand(d_f), W.plan_from_demand(d_w)
    K.require(not bool(plan.identity.any()) and plan.identity.tolist() == wplan.identity.tolist(), "identity flags on real input")
    res = {"images": len(x), "demand": {"equal_to_W_bit_for_bit": True, "sha256": tensor_digest([("demand", d_f)]),
                                        "binding": ctx_f.binding(), "min": float(d_f.min()), "max": float(d_f.max()),
                                        "inputs_read": "the image tensor only (Context.demand(x)); targets are not an argument of the "
                                                       "demand or of plan_from_demand"}}
    slopes = {a: ((g[:, 1:] - g[:, :-1]) * (g.shape[1] - 1)) for a, g in (("x", plan.gx), ("y", plan.gy))}
    res["map"] = {"stats": plan.stats,
                  "node_slope_per_image_min": [float(torch.minimum(slopes["x"][i].min(), slopes["y"][i].min())) for i in range(len(x))],
                  "node_slope_per_image_max": [float(torch.maximum(slopes["x"][i].max(), slopes["y"][i].max())) for i in range(len(x))],
                  "magnification_range_F": [1.0 / max(plan.stats[a]["node_slope_max"] for a in "xy"),
                                            1.0 / min(plan.stats[a]["node_slope_min"] for a in "xy")],
                  "magnification_range_W": [float(torch.minimum(wplan.mx.min(), wplan.my.min())),
                                            float(torch.maximum(wplan.mx.max(), wplan.my.max()))],
                  "strictly_increasing": True}
    K.require(all(plan.stats[a]["endpoint_error_max"] <= TOL["endpoint"] and plan.stats[a]["clamp_max"] <= TOL["endpoint"] for a in "xy"),
              f"anti-crop end points: {plan.stats}")
    # GT: forward map against numpy, round trips inside and outside the image
    pad, n = R._padded_xyxy(t)
    tw = FV.warp_targets(plan, t)
    worst = {"dn_forward_vs_numpy": 0.0, "gt_round_trip": 0.0}
    for i, (a, w) in enumerate(zip(t, tw)):
        if not n[i]:
            continue
        e = W.cxcywh_to_xyxy(a["boxes"].to(F64)).numpy()
        gx, gy = plan.gx[i].numpy(), plan.gy[i].numpy()
        ref = np.stack([_np_warp(gx, e[:, 0]), _np_warp(gy, e[:, 1]), _np_warp(gx, e[:, 2]), _np_warp(gy, e[:, 3])], -1)
        mine = W.cxcywh_to_xyxy(FV.map_boxes(plan.select(slice(i, i + 1)), a["boxes"][None].to(F64), inverse=False))[0]
        worst["dn_forward_vs_numpy"] = max(worst["dn_forward_vs_numpy"], float(np.abs(mine.numpy() - ref).max()))
        back = W.cxcywh_to_xyxy(FV.map_boxes(plan.select(slice(i, i + 1)), W.xyxy_to_cxcywh(mine)[None], inverse=True))[0]
        worst["gt_round_trip"] = max(worst["gt_round_trip"], float(np.abs(back.numpy() - e).max()))
        K.require(w["boxes"].dtype == a["boxes"].dtype and bool((w["boxes"][:, 2:] > 0).all())
                  and float((w["boxes"].to(F64) - W.xyxy_to_cxcywh(mine)).abs().max()) <= 1e-6, "denoising targets are not the inverse of G")
    v = torch.linspace(-0.25, 1.25, 3001, dtype=F64).expand(len(x), -1)
    rt = max(float((FV.warp(g, FV.unwarp(g, v)) - v).abs().max()) for g in (plan.gx, plan.gy))
    rt2 = max(float((FV.unwarp(g, FV.warp(g, v)) - v).abs().max()) for g in (plan.gx, plan.gy))
    ref_err = max(float(np.abs(FV.unwarp(g, v)[i].numpy() - _np_unwarp(g[i].numpy(), v[i].numpy())).max())
                  for g in (plan.gx, plan.gy) for i in range(len(x)))
    worst.update(round_trip_warped_source_warped=rt, round_trip_source_warped_source=rt2, unwarp_vs_numpy=ref_err)
    K.require(all(e <= TOL["round_trip"] for e in worst.values()), f"map round trips on real input: {worst}")
    ox, oy = FV.magnification(plan, pad)
    wx, wy = W.magnification(wplan, pad)
    area = torch.cat([tt["raw_area"] for tt in t]).to(F64)
    g_f = torch.cat([torch.sqrt(ox * oy)[i, :k] for i, k in enumerate(n)])
    g_w = torch.cat([torch.sqrt(wx * wy)[i, :k] for i, k in enumerate(n)])

    def geo(g, sel):
        return float(torch.exp(torch.log(g[sel]).mean())) if bool(sel.any()) else None

    small, medium, large = area <= K.SMALL_AREA, (area >= K.SMALL_AREA) & (area <= K.LARGE_AREA), area >= K.LARGE_AREA
    res["gt"] = {"boxes": int(sum(n)), "errors": worst, "tolerance": TOL["round_trip"],
                 "outside_points": "3001 points on [-0.25, 1.25] per image and axis, both directions",
                 "magnification_first_window": {
                     "note": "description of the first training window only (16 images); not a gate and not used to choose anything",
                     "small": {"n": int(small.sum()), "F": geo(g_f, small), "W": geo(g_w, small),
                               "F_share_below_1": float((g_f[small] < 1).double().mean()) if bool(small.any()) else None,
                               "W_share_below_1": float((g_w[small] < 1).double().mean()) if bool(small.any()) else None},
                     "medium": {"n": int(medium.sum()), "F": geo(g_f, medium), "W": geo(g_w, medium)},
                     "large": {"n": int(large.sum()), "F": geo(g_f, large), "W": geo(g_w, large)}}}
    # renderer: complete support against the dense reference; arm W's renderer reproduced inside its support
    t0 = time.perf_counter()
    xw, info = FV.resample(x, plan)
    seconds = time.perf_counter() - t0
    K.require(tuple(xw.shape) == (len(x), 3, *L_HW) and xw.dtype == torch.float32 and bool(torch.isfinite(xw).all())
              and float(xw.min()) >= float(x.min()) - 1e-5 and float(xw.max()) <= float(x.max()) + 1e-5, "rendered image")
    dense = []
    for i in range(min(2, len(x))):
        ax, sx, dx0, dx1 = reference_axis(plan.gx[i].numpy(), L_HW[1], H_HW[1])
        ay, sy, dy0, dy1 = reference_axis(plan.gy[i].numpy(), L_HW[0], H_HW[0])
        ref = np.einsum("ih,chw,jw->cij", ay, x[i].double().numpy(), ax, optimize=True)
        dense.append({"image": i, "max_abs_difference": float(np.abs(xw[i].double().numpy() - ref).max()),
                      "reference_support_x": sx, "reference_support_y": sy, "footprint_px": [min(dx0, dy0), max(dx1, dy1)]})
    K.require(all(r["max_abs_difference"] <= TOL["render_fp32_vs_fp64"] and info["taps_x"] >= r["reference_support_x"]
                  and info["taps_y"] >= r["reference_support_y"] for r in dense), f"complete tent support: {dense} {info}")

    def w_positions(k, n_out):
        j = torch.arange(n_out + 1, dtype=F64)
        return W.inv(k, (j / n_out).expand(k.shape[0], -1)), W.inv(k, ((j[:-1] + 0.5) / n_out).expand(k.shape[0], -1))

    two = wplan.select(slice(0, 2))
    ix, wxx, sx, _ = FV.tent_taps(*w_positions(two.kx, L_HW[1]), H_HW[1])
    iy, wyy, sy, _ = FV.tent_taps(*w_positions(two.ky, L_HW[0]), H_HW[0])
    same = torch.equal(FV.apply_taps(x[:2], ix, wxx, iy, wyy), W.resample(x[:2], two))
    K.require(same and max(sx, sy) <= W.TAPS, "the complete-support code does not reproduce arm W's renderer on arm W's plan")
    res["render"] = {"readings": info, "cpu_seconds_16_images": seconds, "dense_reference": dense, "tolerance": TOL["render_fp32_vs_fp64"],
                     "needs_more_than_4_taps_on_this_window": max(info["taps_x"], info["taps_y"]) > W.TAPS,
                     "more_than_4_taps_case": "tests/test_r28_fovea.py::test_full_support_tent_matches_the_dense_reference[stripes]",
                     "W_plan_through_complete_support_equals_W_resample_bit_exact": True, "W_plan_support": [sx, sy]}
    return res


# ================================================================ refusals of foreign states
def state_refusals(c, e, state, scope):
    """Cross-arm / cross-demand / cross-mapping / cross-scope uses of states that must be refused.  `e` is an arm F
    engine and `state` a state it wrote in `scope`; nothing may change the engine."""
    import l0_checks_state as H
    from sparse_rtdetr.research import r28_resample as R
    fu = K.followup(c)
    other = {**c, "identity_sha256": {**c["identity_sha256"], "F": "0" * 64}}
    real_ctx, synthetic = e.r28_ctx, R.Context({"kind": "synthetic", "seed": 0})

    def other_demand():
        e.r28_ctx = synthetic
        try:
            K.load(c, e, state, scope)
        finally:
            e.r28_ctx = real_ctx

    cases = [("arm F state read as arm W", lambda: K.read_side(c, state, scope, "W")),
             ("arm F state read as arm K", lambda: K.read_side(c, state, scope, "K")),
             ("arm F state read under another mapping identity", lambda: K.read_side(other, state, scope, "F")),
             ("arm F state loaded into an engine with another demand source", other_demand),
             ("W formal e120 state loaded into the arm F engine", lambda: K.load(c, e, fu["W_endpoint"]["path"], "formal")),
             ("K formal e120 state loaded into the arm F engine", lambda: K.load(c, e, fu["K_endpoint"]["path"], "formal")),
             ("L0 formal e120 state loaded into the arm F engine", lambda: K.load(c, e, c["L0_endpoint"]["path"], "formal")),
             ("formal training of arm W in this round", lambda: __import__("r28_loop").guard(c, K.ROUND / "formal-W", None, "W")),
             ("formal training of arm K in this round", lambda: __import__("r28_loop").guard(c, K.ROUND / "formal-K", None, "K"))]
    if scope != "formal":
        cases += [(f"{scope} state read in formal scope", lambda: K.read_side(c, state, "formal", "F")),
                  (f"{scope} state loaded as a formal state", lambda: K.load(c, e, state, "formal"))]
    out = [H.refused(label, fn) for label, fn in cases]
    K.require(e.r28_ctx is real_ctx, "the engine's demand source was not restored")
    return out


# ================================================================ affected-interface check (CPU)
def fovea_check(c, out, counts):
    import l0_checks_state as H
    import r28_checks as CK
    import r28_cost  # noqa: F401  (import check of the timing module)
    import r28_eval as E
    import r28_loop as LP
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "fovea-check").resolve(), "this action writes <round>/fovea-check")
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "the mapping check hides CUDA")
    fu = K.followup(c)
    K.require(fu is not None and K.followup_arm(c) == "F" and K.ARM == "F", "fovea-check runs with R28_ARM=F under the mapping-control contract")
    torch.set_num_threads(int(os.environ.get("R28_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    res = {"stage": "fovea-check", "registration": fu["registration"], "authorization_basis": c["authorization_basis"],
           "workload": c["workload"]["fovea-check"]}

    K.PHASE["phase"] = "fovea-check:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])

    K.PHASE["phase"] = "fovea-check:identities"
    c1, refs = FU.prior_references(fu["prior_round"])
    ck_, k_inherited, k_refs = k_references(fu["K_round"])
    K.require(all(fu[k] == v for k, v in refs.items() if k != "inherited") and all(fu[k] == v for k, v in k_refs.items())
              and fu["inherited"] == {**refs["inherited"], **k_inherited} and fu["official"] == official_refs(),
              "references changed since prepare")
    K.require(K.prior_contract(c)["contract_sha256"] == c1["contract_sha256"] != c["contract_sha256"] != ck_["contract_sha256"],
              "contract digests of the three rounds")
    K.require(K.pinned(fu["task"]) == K.ROUND / TASK[0] and fu["task"]["sha256"] == TASK[1] and fu["review"]["sha256"] == REVIEW[1],
              "task / review files")
    manifest = K.read_json(K.pinned(c["source_manifest"]))
    deltas = [source_delta(K.read_json(K.pinned(fu["prior_manifest"]))["files"], manifest["files"], label="the closed W round release"),
              source_delta(K.read_json(K.pinned(fu["K_manifest"]))["files"], manifest["files"], label="the K round release")]
    K.require(deltas == fu["source_delta"], "source delta changed since prepare")
    same = [k for k in c1 if k not in ("contract_sha256", "version", "source_root", "source_manifest", "source_commit", "source_dirty",
                                       "workload", "labels", "resource_plan", "outputs", "formal_started", "identity", "identity_sha256", "cost")]
    K.require(all(c[k] == c1[k] for k in same) and all(c["identity_sha256"][a] == c1["identity_sha256"][a] == R.IDENTITY_SHA256[a] for a in "WK")
              and c["identity_sha256"]["F"] == FV.IDENTITY_SHA256["F"] and c["identity"]["F"] == FV.IDENTITY["F"]
              and c["fovea_definition"] == FV.DEFINITION and c["definition"] == W.DEFINITION and c["parent"]["sha256"] == FU.PIN["parent"],
              "inherited contract fields / identities")
    res["identities"] = {"W_round_pins": dict(FU.PIN), "K_round_pins": dict(PIN_K), "official": fu["official"],
                         "inherited_stages": {"W_round": list(refs["inherited"]), "K_round": list(k_inherited)},
                         "contract_fields_equal_to_the_W_round": len(same), "source_delta": deltas,
                         "identity_sha256": c["identity_sha256"], "parent_sha256": c["parent"]["sha256"]}

    K.PHASE["phase"] = "fovea-check:authorisation and isolation"
    prior, k_round = Path(fu["prior_round"]), Path(fu["K_round"])
    for name in fu["inherited"]:
        K.require(not (K.ROUND / name).exists(), f"inherited stage name present in this round: {name}")
    w_summary = K.stage_closed(c, "summary-W", "W")
    K.require(w_summary["K_trigger"] is False and w_summary["label"] == "closed_fixed_config"
              and fu["prior_W_decision"] == FU.W_DECISION, "inherited W summary")
    K.require(K.stage_closed(c, "eval-K", "K")["endpoint"] == fu["K_endpoint"] and K.stage_closed(c, "demand")["weights"] == fu["W_demand"],
              "inherited K evaluation / W demand network")
    cases = [
        FU._refused("formal training of arm W in the mapping-control round", lambda: LP.guard(c, K.ROUND / "formal-W", None, "W")),
        FU._refused("formal training of arm K in the mapping-control round", lambda: LP.guard(c, K.ROUND / "formal-K", None, "K")),
        FU._refused("formal F before fovea-check, wire-F and accept-F are closed", lambda: LP.guard(c, K.ROUND / "formal-F", None, "F")),
        FU._refused("a formal W directory lookup in this round", lambda: K.formal_dir(c, "W")),
        FU._refused("a formal K directory lookup in this round", lambda: K.formal_dir(c, "K")),
        FU._refused("an evaluation output inside the closed W round", lambda: E.round_stage(prior / "eval-F", "eval-F")),
        FU._refused("an evaluation output inside the K round", lambda: E.round_stage(k_round / "eval-F", "eval-F")),
        FU._refused("a follow-up block without the registered authorization basis", lambda: K.followup({**c, "authorization_basis": "none"})),
        FU._refused("the mapping-control block under the K follow-up basis",
                    lambda: K.followup({**c, "authorization_basis": K.FOLLOWUP_BASIS, "followup": {**fu, "authorization_basis": K.FOLLOWUP_BASIS}})),
        FU._refused("arm W built by the arm F module", lambda: FV.check_source("W", {"kind": "network"}, False)),
        FU._refused("arm F with a fixed (arm K) demand source", lambda: FV.check_source("F", {"kind": "fixed"}, False)),
    ]
    K.require(not any(v.startswith("R28_DEV_") for v in os.environ), "dev flags are not used by the mapping-control round")
    res["authorisation"] = {"round": str(K.ROUND), "independent_of": [str(prior), str(k_round)], "refusals": cases,
                            "W_summary_inherited": {k: w_summary[k] for k in ("label", "label_key", "K_trigger", "M_failed_cells")},
                            "dev_flags_absent": True}

    K.PHASE["phase"] = "fovea-check:real first window"
    (x, t, inp), = CK._first_windows(c, 1)
    theirs = H.r27_rows(c, "P", 1)[0]["input"]
    K.require(all(type(inp[k]) is type(theirs[k]) and inp[k] == theirs[k] for k in K.INPUT_FIELDS) and set(inp) == set(K.INPUT_FIELDS),
              "input identity differs from R27 formal-P row 1")
    with torch.no_grad():
        real_d = R.Context(K.demand_source(c, "F")).demand(x)

    K.PHASE["phase"] = "fovea-check:official rule"
    res["official_rule"] = official_check(fu, {**{k: v[None] for k, v in synthetic_demands().items()}, "real_first_window": real_d})

    K.PHASE["phase"] = "fovea-check:geometry on the real window"
    res["real_window"] = {"input": inp, "window_shape": list(x.shape), "input_identity_equal_to_R27_formal_P_row_1": True,
                          **real_geometry(c, x, t)}

    K.PHASE["phase"] = "fovea-check:fixture construct"
    parent = K.pinned(c["parent"])
    source = K.demand_source(c, "F")
    e, checks = FV.create(parent, "cpu", "F", source, fixture=True, check=True)
    ps = torch.load(parent, map_location="cpu", weights_only=True)
    sd, esd = e.model.state_dict(), e.ema.module.state_dict()
    keys = {"online_equal_parent": list(sd) == list(ps["model"]), "EMA_equal_parent": list(esd) == list(ps["ema"]["module"]),
            "state_keys": len(sd), "demand_or_r28_keys": sum("r28" in k or "demand" in k or "fovea" in k for k in sd)}
    K.require(keys["online_equal_parent"] and keys["EMA_equal_parent"] and keys["demand_or_r28_keys"] == 0, ("native key set", keys))
    del ps
    res.update(native_keys=keys, create_checks={k: v for k, v in checks.items() if k != "new_parameter_names"})
    x2, t2 = x[:2], t[:2]
    before = [tt["boxes"].clone() for tt in t2]
    res["fixture"] = fx = {"images": [tt["image_id"] for tt in t2], "physical_batch": 1, "accumulation": 2, "precision": "float32",
                           "demand": e.r28_ctx.binding(), "non_identity": True}
    K.register_counts(e, counts)
    obs = LP.install_observer(e)
    bn0 = _bn_counts(e.model)
    rows, readings = [], []
    for i in (1, 2):
        K.PHASE["phase"] = f"fovea-check:fixture u{i}"
        r = e.train_window(x2, t2, capture=True)
        K.require(r["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates, "fixture clock")
        rows.append(r)
        readings.append(obs.update_row(t2))
        if i == 1:
            side = K.save(c, e, out / "cpu-fixture-u1.pth", "cpu-fixture")
    K.require(all(torch.equal(a, tt["boxes"]) for a, tt in zip(before, t2)), "the forward modified the caller's targets")
    K.require(all(r["identity_images"] == 0 and "fovea" in r and r["fovea"]["taps_max"] >= 2 for r in readings), f"fixture readings: {readings}")
    json.dumps(readings, allow_nan=False)          # the ledger writes these rows as strict JSON
    expected = S.capture(e)
    erow = obs.epoch_row(61)
    fx["BN_delta_two_updates"] = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    K.require(fx["BN_delta_two_updates"] == [4], f"fixture BN delta {fx['BN_delta_two_updates']}")
    del e
    gc.collect()
    K.PHASE["phase"] = "fovea-check:fixture restore+replay"
    clone, _ = FV.create(parent, "cpu", "F", source, fixture=True, check=True)
    K.register_counts(clone, counts)
    obs2 = LP.install_observer(clone)
    K.load(c, clone, out / "cpu-fixture-u1.pth", "cpu-fixture")
    replay = clone.train_window(x2, t2, capture=True)
    reading2 = obs2.update_row(t2)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and ck.exact_equal(expected["rng"], got["rng"]) and reading2 == readings[1],
              f"fixture replay differs {diff[:5]}")
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"fixture counts {dict(counts)}")
    fx.update(saved_u1=side, full_state_exact=True, rng_exact=True, replay_result_exact=True, observer_replay_exact=True,
              loss=[r["loss"] for r in rows], replay_loss=replay["loss"], observer_updates=readings, observer_epoch=erow,
              elapsed_seconds=[r["elapsed_seconds"] for r in rows + [replay]])
    K.PHASE["phase"] = "fovea-check:state refusals"
    st = out / "cpu-fixture-u1.pth"
    res["refused"] = state_refusals(c, clone, st, "cpu-fixture") + [
        H.refused("check engine writing a formal state", lambda: K.save(c, clone, out / "refused.pth", "formal")),
        H.refused("formal engine with a synthetic demand", lambda: FV.check_source("F", {"kind": "synthetic", "seed": 0}, False)),
        H.refused("endpoint_state without an endpoint", lambda: K.endpoint_state(c, "F"))]
    K.require(ck.exact_equal(S.capture(clone), got) and not (out / "refused.pth").exists(), "a refused action changed the engine or wrote a file")

    K.PHASE["phase"] = "fovea-check:ledger readings"
    trial_dir = out / "trial-ledger"                       # the fixture's observer rows in the ledger layout (not a formal ledger)
    trial_dir.mkdir()
    with (trial_dir / "UPDATES.jsonl").open("x") as f:
        for r, rd in zip(rows, readings):
            f.write(json.dumps({"update_seconds": r["elapsed_seconds"], "r28": rd}, separators=(",", ":"), allow_nan=False) + "\n")
    folded = formal_observations(trial_dir)
    K.require(folded["updates"] == 2 and folded["taps_max"] >= 2 and folded["clamped_interior_nodes"] == 0
              and folded["node_slope_min"] == min(rd["fovea"]["node_slope_min"] for rd in readings), f"folded fixture readings: {folded}")
    res["ledger_readings_interface"] = {"rows": 2, "folded": folded, "note": "CPU fixture rows; interface check of the summary reader"}

    K.PHASE["phase"] = "fovea-check:summary arithmetic"
    trial = _trial_summary(c, None)
    res["summary_interface"] = {"trial": "summary_f with the stored W evaluation in the F slot (every W - F difference 0)",
                                "label": trial["label"], "contribution": trial["contribution"], "pairs": list(trial["differences_pp"]),
                                "W_decision_kept": trial["W_decision_unchanged"]}
    res.update(fixture_updates=3, full_forwards=6, formal_state_usable=False, cpu_seconds_wall=time.perf_counter() - t0,
               scope="affected interfaces only; the general CPU / identity / evaluation-path evidence of the shared code is the "
                     "closed W round's cpu, zero and evalpath stages and the K round's chain (inherited, pinned)")
    return res


# ================================================================ summary arithmetic (pure)
def size_table(tables, key):
    arms = list(tables)
    K.require(all([r["category_id"] for r in tables[a]] == list(range(1, 11)) and [r["GT"] for r in tables[a]] == [r["GT"] for r in tables[arms[0]]]
                  for a in arms), f"per-class {key} tables")
    return [{"category_id": k, "GT": tables[arms[0]][k - 1]["GT"], **{a: tables[a][k - 1][key] for a in arms}} for k in range(1, 11)]


def summary_f(c, w_result, values, apl, apm, cost_f, k_cost):
    """values / apl / apm: arm -> stored readings of L0, W, K, F.  Pure arithmetic; no gate."""
    import r28_eval as E
    K.require(set(values) == set(apl) == set(apm) == set(ARMS4), "summary arms")
    diffs = {f"{x}-{y}": E.difference_pp(values[x], values[y]) for x, y in PAIRS}
    primary = {f"W-F top{b} {m}": diffs["W-F"][b][m] for b, m in PRIMARY}
    ratios = cost_f["ratios"]
    return {"arm": ARM, "label": LABEL, "contribution": CONTRIBUTION, "record_only": True,
            "values": {**w_result["values"], "K": values["K"], "F": values["F"]},
            "value_sources": {"F": "this round, eval-F (new)", "W": "closed W round eval-W (quoted)", "K": "K round eval-K (quoted)",
                              "L0": "R27-002 eval-L0 via the contract reference (quoted)"},
            "primary_W_minus_F_pp": primary, "differences_pp": diffs,
            "APL6_pp": {f"{x}-{y}": E.apl6(apl[x], apl[y]) for x, y in (("W", "F"), ("F", "L0"), ("F", "K"))},
            "per_class_APL_top600": size_table(apl, "APL"), "per_class_APM_top600": size_table(apm, "APM"),
            "cost_F": {"model_gpu": {k: ratios["model_gpu"][k] for k in ("F/L0", "W/L0", "W/F", "F/W")},
                       "end_to_end": {k: ratios["end_to_end"][k] for k in ("F/L0", "W/L0", "W/F", "F/W")},
                       "medians": cost_f["medians"], "stages": cost_f.get("stages"), "source": "cost-F (L0, W, F in one process)"},
            "cost_K_round_quoted": {**k_cost, "note": "another process (L0, W, K); quoted, not the same measurement"},
            "W_decision_unchanged": w_result["decision"]}


def _inputs(c):
    fu = K.followup(c)
    w_result = K.read_json(K.pinned(fu["prior_W_result"]))
    k_result = K.read_json(K.pinned(fu["K_result"]))
    ev_w, ev_k, chk = K.stage_closed(c, "eval-W", "W"), K.stage_closed(c, "eval-K", "K"), K.stage_closed(c, "followup-check", "K")
    K.require(w_result["decision"] == fu["prior_W_decision"] == FU.W_DECISION and k_result["values"]["K"] == ev_k["values"]
              and k_result["values"]["W"] == ev_w["values"] and k_result["contract_sha256"] == fu["K_contract_sha256"],
              "stored W / K results")
    ref = c["reference"]
    values = {"L0": ref["values"]["L0"], "W": ev_w["values"], "K": ev_k["values"]}
    apl = {"L0": ref["per_class_APL_top600"]["L0"], "W": ev_w["per_class_APL_top600"], "K": ev_k["per_class_APL_top600"]}
    apm = {"L0": chk["per_class"]["L0"]["per_class_APM_top600"], "W": chk["per_class"]["W"]["per_class_APM_top600"],
           "K": ev_k["per_class_APM_top600"]}
    K.require(chk["per_class"]["W"]["per_class_APL_top600"] == apl["W"] and chk["per_class"]["L0"]["per_class_APL_top600"] == apl["L0"],
              "stored per-class large rows of W / L0")
    return w_result, k_result, ev_w, values, apl, apm


def _trial_summary(c, cost):
    """summary_f with the stored W evaluation standing in for F (wiring only; every W - F difference is exactly 0)."""
    w_result, k_result, ev_w, values, apl, apm = _inputs(c)
    if cost is None:
        pairs = ("F/L0", "W/L0", "W/F", "F/W")
        cost = {"ratios": {"model_gpu": dict.fromkeys(pairs, 1.0), "end_to_end": dict.fromkeys(pairs, 1.0)}, "medians": {}}
    trial = summary_f(c, w_result, {**values, "F": ev_w["values"]}, {**apl, "F": apl["W"]}, {**apm, "F": apm["W"]}, cost, k_result["cost_K"])
    K.require(all(v == 0.0 for v in trial["primary_W_minus_F_pp"].values()) and trial["APL6_pp"]["W-F"]["difference_pp"] == 0.0
              and trial["W_decision_unchanged"] == FU.W_DECISION and trial["label"] == LABEL and trial["contribution"] == CONTRIBUTION
              and trial["differences_pp"]["W-K"] == k_result["differences_pp"]["W-K"], "summary arithmetic")
    json.dumps(trial, allow_nan=False)
    return trial


def tensor_tables(path):
    """Per-class readings of the kept COCOeval tensors (numpy only, no scorer): AP, AP50, AP75 and the size classes."""
    import numpy as np
    z = np.load(path, allow_pickle=False)
    p, iou, labels = z["precision"], z["iouThrs"], [str(v) for v in z["areaRngLbl"]]
    K.require(p.ndim == 5 and p.shape[0] == len(iou) and p.shape[3] == len(labels) and labels[0] == "all", "kept COCOeval tensor layout")
    t50, t75 = int(np.argmin(np.abs(iou - 0.5))), int(np.argmin(np.abs(iou - 0.75)))

    def ap(k, area, t=None):
        v = p[:, :, k, labels.index(area), -1] if t is None else p[t, :, k, labels.index(area), -1]
        v = v[v > -1]
        return float(v.mean()) if v.size else None

    rows = []
    for k, cat in enumerate(z["catIds"].tolist()):
        rows.append({"category_id": int(cat), "GT": {lab: int(z["gt_count"][k][a]) for a, lab in enumerate(labels)},
                     "AP": ap(k, "all"), "AP50": ap(k, "all", t50), "AP75": ap(k, "all", t75), "APS": ap(k, "small"),
                     "AP75_small": ap(k, "small", t75), "APM": ap(k, "medium"), "APL": ap(k, "large")})

    def macro(key):
        v = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(v)) if v else None

    return {"classes": rows, "macro": {k: macro(k) for k in ("AP", "AP50", "AP75", "APS", "AP75_small", "APM", "APL")},
            "stats": z["stats"].tolist(), "maxDets": int(z["maxDets"][-1])}


# ================================================================ CUDA wiring of what only runs after training
def wire(c, out, counts):
    import r28_cost as CO
    import r28_eval as E
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_fovea as FV
    torch = K.torch_mod()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "wire-F").resolve() and K.ARM == "F" and K.followup_arm(c) == "F",
              "this action writes <round>/wire-F with R28_ARM=F")
    K.stage_closed(c, "fovea-check", "F")
    K.PHASE["phase"] = "wire:model (parent EMA weights behind the arm F forward)"
    state = torch.load(K.pinned(c["parent"]), map_location="cpu", weights_only=True)["ema"]["module"]
    digest = tensor_digest(state.items())
    model = FV.build_eval_model(state, "cuda:0", "F", K.demand_source(c, "F"))
    K.require(FV.is_wrapped(model) and not model.training, "wire model")
    K.PHASE["phase"] = "wire:timing"
    loaders = {"L0": lambda: CO.load_reference(c, "L0", torch), "W": lambda: CO.load_arm(c, "W", torch),
               "F": lambda: (model, {"weights": "e060 parent EMA (not trained under F; wiring only)", "demand": model.r28_ctx.binding()})}
    order = K.COST_ENDPOINTS["cost-F"]
    timing = CO._run(c, counts, loaders, order, (order,) * WIRE["cost_passes"], warmup=WIRE["cost_warmup"],
                     batches_per_pass=WIRE["cost_batches_per_pass"], stages=True)
    table = timing["stages"]["table"]
    K.require(set(table) == set(order) and all(abs(sum(table[a]["stage_ms_median_pass"].values()) - table[a]["model_gpu_ms_median_pass"])
                                                <= 1e-6 * max(1.0, table[a]["model_gpu_ms_median_pass"]) for a in order)
              and all(table[a]["stage_ms_median_pass"][s] > 0 for a in ("W", "F") for s in ("demand", "map", "render", "detector", "unwarp"))
              and table["L0"]["stage_ms_median_pass"]["render"] > 0 and table["L0"]["stage_ms_median_pass"]["detector"] > 0
              and "probe" not in vars(model.r28_ctx), ("stage readings", table))
    K.PHASE["phase"] = "wire:evaluation batch"
    sub = Counter(forwards=0)
    data, preds, _ = E.infer(c, model, sub, limit=(0, WIRE["eval_batches"]))
    counts.update(sub)
    K.require(all(len(preds[b]) == WIRE["eval_batches"] * K.EVAL_BATCH * b for b in K.BUDGETS)
              and all(math.isfinite(v) for p in preds[600] for v in (p["score"], *p["bbox_xyxy"]))
              and tensor_digest(model.state_dict().items()) == digest, "wire evaluation batch")
    K.PHASE["phase"] = "wire:summary structure"
    trial = _trial_summary(c, timing)
    K.require(counts["forwards"] == c["workload"]["wire"]["forwards"] and counts["actual_updates"] == 0
              and not counts.get("score_budget_calls"), f"wire counts {dict(counts)}")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"stage": "wire-F", "workload": c["workload"]["wire"], "weights": "e060 parent EMA behind the arm F forward", "weights_sha256": digest,
            "purpose": "wiring of code that otherwise runs only after the formal training: the evaluation batch path of arm F and the "
                       "staged L0 / W / F timing; these readings are not results (F is not trained here, the protocol is shortened)",
            "timing": {k: timing[k] for k in ("ratios", "medians", "stages", "protocol", "forwards_per_endpoint", "order_executed")},
            "evaluation_batch": {"images": WIRE["eval_batches"] * K.EVAL_BATCH, "records_top600": len(preds[600]), "scored": False},
            "summary_structure": {"cost_keys": sorted(trial["cost_F"]), "stage_table_arms": sorted(trial["cost_F"]["stages"]["table"])},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "scorer_calls": 0, "formal_state_usable": False}


# ================================================================ summary-F
def formal_observations(d):
    """Map and tap readings of the formal run in directory d, folded from its ledger (correctness and cost record;
    nothing is chosen from them)."""
    d = Path(d)
    acc = {"updates": 0, "node_slope_min": math.inf, "node_slope_max": 0.0, "endpoint_error_max": 0.0, "clamped_nodes": 0, "clamp_max": 0.0,
           "clamped_interior_nodes": 0,
           "taps_max": 0, "footprint_max_px": 0.0, "identity_images": 0, "taps_histogram": Counter(), "seconds": 0.0}
    g = {"g_small": [], "g_medium": [], "g_large": [], "small_g_lt_1": []}
    with (d / "UPDATES.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            r, fv = row["r28"], row["r28"]["fovea"]
            acc["updates"] += 1
            acc["seconds"] += row["update_seconds"]
            acc["node_slope_min"], acc["node_slope_max"] = min(acc["node_slope_min"], fv["node_slope_min"]), max(acc["node_slope_max"], fv["node_slope_max"])
            acc["endpoint_error_max"], acc["clamp_max"] = max(acc["endpoint_error_max"], fv["endpoint_error_max"]), max(acc["clamp_max"], fv["clamp_max"])
            acc["clamped_nodes"] += fv["clamped_nodes"]
            acc["clamped_interior_nodes"] += fv["clamped_interior_nodes"]
            acc["identity_images"] += r["identity_images"]
            if fv["taps_max"] is not None:
                acc["taps_max"] = max(acc["taps_max"], fv["taps_max"])
                acc["taps_histogram"][str(fv["taps_max"])] += 1
                acc["footprint_max_px"] = max(acc["footprint_max_px"], fv["footprint_max_px"])
            for k in g:
                if r[k] is not None:
                    g[k].append(r[k])
    acc["taps_histogram"] = dict(sorted(acc["taps_histogram"].items(), key=lambda kv: int(kv[0])))
    acc["magnification_min"], acc["magnification_max"] = 1.0 / acc["node_slope_max"], 1.0 / acc["node_slope_min"]
    acc["seconds_per_update_mean"] = acc["seconds"] / max(1, acc["updates"])
    acc["per_update_means"] = {k: (sum(v) / len(v) if v else None) for k, v in g.items()}
    acc["note"] = ("node slopes are dG/du of the warped -> source map (1 = uniform); magnification = 1 / node slope; g_* are means over "
                   "updates of the per-update geometric mean magnification of the batch GT by raw-area class")
    return acc


def summarize(c, out, counts):
    import r28_eval as E
    fu = K.followup(c)
    K.require(fu is not None and K.followup_arm(c) == "F" and K.ARM == "F", "the mapping-control round only summarises arm F")
    K.PHASE["phase"] = "summarize/read"
    out = E.round_stage(out, "summary-F")
    w_result, k_result, ev_w, values, apl, apm = _inputs(c)
    ev_f, cost_f = K.stage_closed(c, "eval-F", "F"), K.stage_closed(c, "cost-F", "F")
    K.require(cost_f["weights_sha256"]["F"] == ev_f["weights_sha256"] and cost_f["weights_sha256"]["W"] == ev_w["weights_sha256"]
              == fu["W_weights_sha256"] and cost_f["weights_sha256"]["L0"] == c["L0_weights_sha256"],
              "the cost-F stage timed other models than the evaluated ones")
    K.require(ev_f["demand"]["file"] == fu["W_demand"] and ev_f["demand"] == ev_w["demand"], "arm F was not evaluated with arm W's demand network")
    for b in K.BUDGETS:
        K.pinned(ev_f["metrics"][str(b)]), K.pinned(ev_f["predictions"][str(b)])
    K.PHASE["phase"] = "summarize/compute"
    res = summary_f(c, w_result, {**values, "F": ev_f["values"]}, {**apl, "F": ev_f["per_class_APL_top600"]},
                    {**apm, "F": ev_f["per_class_APM_top600"]}, cost_f, k_result["cost_K"])
    tables = tensor_tables(K.pinned(ev_f["cocoeval_top600_tensors"]))
    f600 = ev_f["values"]["600"]
    consistency = {k: (None if tables["macro"][k] is None else tables["macro"][k] - f600[k]) for k in tables["macro"]}
    formal = K.formal_dir(c, "F").name
    res.update(status="PASS", registration=fu["registration"], authorization_basis=c["authorization_basis"], task=fu["task"],
               review=fu["review"], contract_sha256=c["contract_sha256"], source_commit=c["source_commit"],
               official=fu["official"], fovea_definition=c["fovea_definition"], identity_sha256_F=c["identity_sha256"]["F"],
               W_closed_round={"contract_sha256": fu["prior_contract_sha256"], "result": fu["prior_W_result"], "decision": w_result["decision"],
                               "M_W_minus_L0": w_result["M_W_minus_L0"], "APL6": w_result["APL6"], "C": w_result["C"],
                               "note": "quoted unchanged from the closed round; not recomputed, not overwritten"},
               K_round={"contract_sha256": fu["K_contract_sha256"], "result": fu["K_result"], "A_W_minus_K": k_result["A_W_minus_K"],
                        "note": "quoted unchanged from the K round"},
               F_per_class_top600={**tables, "source": ev_f["cocoeval_top600_tensors"],
                                   "macro_minus_scored_value": consistency,
                                   "note": "read from the tensors kept by the single per-class COCOeval of eval-F; no rescoring"},
               F_demand=ev_f["demand"], F_endpoint=ev_f["endpoint"], F_weights_sha256=ev_f["weights_sha256"],
               F_training_observations=formal_observations(K.formal_dir(c, "F")),
               stages=E._stage_account(c, ["fovea-check", "wire-F", "accept-F", formal, "eval-F", "cost-F"]),
               other_directories=[d for d in E._other_directories() if d["directory"] != OFFICIAL_DIR], statements=STATEMENTS)
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    p = res["primary_W_minus_F_pp"]
    return {"result": K.ref(out / "RESULT.json"), "registration": fu["registration"], "label": res["label"],
            "contribution": res["contribution"], "record_only": True, "W_minus_F_APS600_pp": p["W-F top600 APS"],
            "W_minus_F_AP75_small600_pp": p["W-F top600 AP75_small"], "W_decision_unchanged": w_result["decision"],
            "model_gpu_F_over_L0": res["cost_F"]["model_gpu"]["F/L0"], "model_gpu_W_over_F": res["cost_F"]["model_gpu"]["W/F"]}
