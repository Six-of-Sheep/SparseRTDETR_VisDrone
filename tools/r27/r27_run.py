"""TASK_R27_001 runner: fixed-budget local high-scale memory (N native H / C uniform tiles / P encoder-score tiles).

actions
  prepare      no torch model.  Round CONTRACT (task, parent, S15/A0 references, monitor64, data, identity, gates,
               workload, release manifest).
  cpu          CUDA hidden.  pytest synthetic checks; real e060 CPU fixture: zero-step C/P == native model(I_L)
               (eval/train outputs, losses, old-parameter gradients), C==P, BN separation, gradient start, P route
               invariance to DN/GT, C route stateless, optimizer/EMA coverage, save/restore, refusals, MACs.
  accept       CUDA, one arm (N/C/P).  u1,u2 from e060, save u1, fresh engine restore + replay u2 (exact).  C/P add a
               CUDA zero-step reference against native L (2+2 forwards, no update).  N: u1/u2 vs S15 formal-002.
  diag         CUDA, C or P.  Observation u0, 128 effective updates (S15 windows u18241-u18368), observation u128.
  cost         CUDA.  N (e060) / C,P (u128 online), FP32 batch4 monitor64; 2 warmup/arm; 3 no-hook passes rotated
               NCP/CPN/PNC; then one instrumented pass/arm (stage events, outputs checked equal).
  gate         no model.  Section 8 engineering + cost conditions -> GATE.json (promoted_formal / not_promoted_*).
  formal       CUDA, C or P.  e061-e120 from the e060 parent (18240 updates); refuses without a promoted GATE.json
               and refuses diagnostic/acceptance states; --resume only from a formal state of the same arm.
  evaluate     CUDA+CPU, C or P.  Fixed e120 EMA, view A 548 images, top600/top300, 2 score_budget calls + per-class APL.
  final-cost   CUDA.  Section 7 protocol on N=S15 e120 EMA, C/P e120 EMA in one new process.
  summarize    no model.  E/I/G tables, costs and workload ledger.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"), str(ROOT)]

WS = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
LD_ROOT = WS / "experiments/localization-distribution"
ROUND = Path(os.environ.get("R27_ROUND", str(LD_ROOT / "r27-budgeted-local-memory-20261109-001")))  # dev rounds only via env
PARENT = LD_ROOT / ("rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/"
                    "checkpoint-epoch-060.pth")
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
S15_ROOT = LD_ROOT / "s15-scale-baseline-20261105-001"
S15_CONTRACT = S15_ROOT / "formal-002-CONTRACT.json"
S15_UPDATES = S15_ROOT / "formal-002/train-S15/UPDATES.jsonl"
S15_COMPARISON = S15_ROOT / "formal-002/evaluation/COMPARISON.json"
S15_METRICS = {b: S15_ROOT / f"formal-002/evaluation/S15-EMA-A-top{b}-metrics.json" for b in (600, 300)}
S15_ENDPOINT = S15_ROOT / "formal-002/train-S15/checkpoint-epoch-120.pth"
S15_ENDPOINT_SHA = "656f7be2357f20dce50a718f1093a639c1a32a7faad69d03184e36e43f32871f"
R26_ROUND = LD_ROOT / "r26-constrained-detail-encoder-20261109-001"
R26_CONTRACT = R26_ROUND / "CONTRACT.json"
R26_OBS_N = {k: R26_ROUND / f"diag-N/OBS-{k}.json" for k in ("u0", "u128")}
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
TASK = ROUND / "TASK_R27_001.md"
TASK_SHA = "fefba62135eb090d4d8449c31f7840960432582e8fd126f914ed6a36b606921e"
SCHEMA = "r27-state-v1"
ARMS = ("N", "C", "P")
LEARN = ("C", "P")
DIAG_UPDATES = 128
MONITOR_AT = (1, 2, 16, 64, 128)
OBS_IMAGES, OBS_BATCH = 64, 8
COST_BATCH, COST_WARMUP, COST_PASSES = 4, 2, 3
ROTATIONS = (("N", "C", "P"), ("C", "P", "N"), ("P", "N", "C"))
FORMAL_SAVE_EPOCHS = (60, 75, 90, 105, 120)
RECOVERY_EVERY = 5
EVAL_BATCH, EVAL_WARMUP, EVAL_BATCHES = 4, 4, 137
ENDPOINT = 36480
BUDGETS = (600, 300)
COST_GATE = {"model_gpu_ratio_max": 0.80, "end_to_end_ratio_max": 0.95}
GATE_E = (("600", "AP", -0.30), ("600", "APS", -0.30), ("600", "AP75_small", -0.30), ("300", "AP", -0.30),
          ("300", "APS", -0.30), ("600", "APM", -0.50), ("600", "APL", -0.50))
GATE_I = (("600", "APS", 0.30), ("600", "AP75_small", 0.20), ("600", "AP", -0.10), ("300", "AP", -0.20),
          ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))
GATE_G = (("600", "APS", 0.50), ("600", "AP75_small", 0.30), ("600", "AP", -0.10), ("300", "AP", -0.20),
          ("300", "APS", -0.20), ("600", "APM", -0.50), ("600", "APL", -0.50))
COST_PER_ARM = COST_WARMUP + COST_PASSES * (OBS_IMAGES // COST_BATCH) + OBS_IMAGES // COST_BATCH
WORKLOAD = {
    "acceptance": {"updates_per_arm": 3, "training_forwards_per_arm": 6, "arms": list(ARMS),
                   "cuda_zero_step_reference_forwards_per_learnable_arm": 4},
    "diagnostic": {"updates_per_arm": DIAG_UPDATES, "arms": list(LEARN), "training_forwards_per_arm": 2 * DIAG_UPDATES,
                   "observation_forwards_per_arm": 2 * OBS_IMAGES // OBS_BATCH, "monitor_updates": list(MONITOR_AT)},
    "cost": {"warmup_per_arm": COST_WARMUP, "timed_no_hook_per_arm": COST_PASSES * OBS_IMAGES // COST_BATCH,
             "instrumented_per_arm": OBS_IMAGES // COST_BATCH, "forwards_per_arm": COST_PER_ARM, "batch": COST_BATCH,
             "rotations": [list(r) for r in ROTATIONS]},
    "formal": {"arms": list(LEARN), "updates_per_arm": 18240, "training_forwards_per_arm": 36480,
               "save_epochs": list(FORMAL_SAVE_EPOCHS), "recovery_every_epochs": RECOVERY_EVERY},
    "evaluation": {"endpoint": "e120 EMA", "view": "A", "images": 548, "batch": EVAL_BATCH, "warmup": EVAL_WARMUP,
                   "forwards_per_arm": EVAL_WARMUP + EVAL_BATCHES, "budgets": list(BUDGETS), "score_budget_calls": 4,
                   "per_class_APL_COCOeval_calls": 2, "online": "saved, not inferred or scored"},
    "final_cost": {"forwards_per_arm": COST_PER_ARM, "endpoints": "N=S15 seed0 e120 EMA, C/P e120 EMA"},
    "normal_gpu_updates": {"acceptance": 9, "diagnostic": 2 * DIAG_UPDATES, "formal_conditional": 36480,
                           "total": 9 + 2 * DIAG_UPDATES + 36480},
    "normal_full_forwards": {"acceptance_training": 18, "acceptance_reference": 8, "diagnostic_training": 4 * DIAG_UPDATES,
                             "diagnostic_observation": 4 * OBS_IMAGES // OBS_BATCH, "cost": 3 * COST_PER_ARM,
                             "formal_training_conditional": 72960, "evaluation_conditional": 2 * (EVAL_WARMUP + EVAL_BATCHES),
                             "final_cost_conditional": 3 * COST_PER_ARM},
}
PHASE = {"phase": "initialize"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": file_sha(p)}


def pinned(r):
    require(ref(r["path"]) == r, f"pinned file differs: {r['path']}")
    return Path(r["path"])


def write_new(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x") as f:
        f.write(json.dumps(data, indent=1, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


def read_json(path):
    return json.loads(Path(path).read_text())


def torch_mod():
    import torch
    return torch


# ================================================================ contract
def source_manifest(root):
    files = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            files[str(p.relative_to(root))] = file_sha(p)
    return files


def prepare(_args):
    from sparse_rtdetr.research import r27_local_memory as R
    require(ROOT.resolve().parent.parent == ROUND.resolve() and ROOT.name == "source", "prepare must run from a frozen release")
    manifest = read_json(ROOT.parent / "MANIFEST.json")
    require(manifest["files"] == source_manifest(ROOT), "release source differs from MANIFEST")
    task = ref(TASK)
    require(task["sha256"] == TASK_SHA, "task file SHA")
    s15 = read_json(S15_CONTRACT)
    r26 = read_json(R26_CONTRACT)
    comparison = read_json(S15_COMPARISON)
    a0 = s15["reference"]
    c = {"version": "R27-001-contract-v1", "task": task,
         "parent": ref(PARENT), "parent_update": 18240,
         "S15_contract": ref(S15_CONTRACT), "S15_formal_identity": s15["formal_identity"],
         "S15_updates": ref(S15_UPDATES), "S15_comparison": ref(S15_COMPARISON),
         "S15_metrics": {str(b): ref(p) for b, p in S15_METRICS.items()},
         "S15_endpoint": ref(S15_ENDPOINT), "S15_EMA_values": comparison["scores"]["S15"]["EMA"],
         "A0_EMA_values": comparison["scores"]["A0"]["EMA"],
         "A0_metrics": {b: a0["A0_metrics"]["EMA"][b] for b in ("600", "300")},
         "A0_endpoint": a0["A0_endpoint"],
         "per_class_APL_top600": comparison["per_class_APL_top600"]["EMA"],
         "data": s15["data"], "data_receipt": s15["data_receipt"], "development_ids": s15["development_ids"],
         "evaluation_policy": s15["evaluation_policy"],
         "environment": s15["environment"], "python": PYTHON, "gpu_uuid": s15["gpu_uuid"],
         "R26_contract": ref(R26_CONTRACT), "monitor_rule": r26["monitor_rule"] + " (R26 list reused unchanged)",
         "monitor_ids": r26["monitor_ids"], "monitor_sha256": r26["monitor_sha256"],
         "R26_N_observation": {k: ref(p) for k, p in R26_OBS_N.items()},
         "identity": R.IDENTITY, "identity_sha256": R.IDENTITY_SHA256,
         "source_root": str(ROOT.resolve()), "source_manifest": ref(ROOT.parent / "MANIFEST.json"),
         "source_commit": manifest["commit"], "workload": WORKLOAD,
         "gates": {"cost": COST_GATE, "E_P_minus_S15": GATE_E, "I_P_minus_C": GATE_I, "G_P_minus_A0": GATE_G,
                   "E_cost": COST_GATE, "cost_statistic": "ratio of medians over three per-pass totals (16 batches/pass)"},
         "recipe": {"input_hw": [1152, 2016], "global_hw": [768, 1344], "train_batch": 8, "accumulation": 2,
                    "precision": "bfloat16", "clip": 0.1, "optimizer": "native AdamW groups, global LR clock",
                    "activation_checkpoint": True,
                    "checkpoint_segments": ["backbone.res_layers", "encoder.fpn_blocks", "encoder.pan_blocks",
                                            "r27_local.res_layers"],
                    "allocator_conf": "expandable_segments:True"},
         "outputs": str(ROUND), "formal_started": False}
    require(c["parent"]["sha256"] == PARENT_SHA and c["S15_endpoint"]["sha256"] == S15_ENDPOINT_SHA, "parent/S15 endpoint SHA")
    require(c["monitor_sha256"] == sha_bytes(canonical(c["monitor_ids"])) and len(c["monitor_ids"]) == OBS_IMAGES, "monitor ids")
    c["contract_sha256"] = sha_bytes(canonical(c))
    write_new(ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "identity_sha256": R.IDENTITY_SHA256}))


def load_contract(path=ROUND / "CONTRACT.json"):
    c = read_json(path)
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    require(c["contract_sha256"] == sha_bytes(canonical(body)), "contract digest")
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    require(Path(c["outputs"]).resolve() == ROUND.resolve(), "contract bound to another round")
    pinned(c["source_manifest"])
    pinned(c["parent"])
    from sparse_rtdetr.research import r27_local_memory as R
    require(c["identity_sha256"] == R.IDENTITY_SHA256, "R27 behaviour identity differs from the contract")
    return c


# ================================================================ engines, states
def make(c, device, arm, *, fixture=False):
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.model import Config
    PHASE["phase"] = f"{arm}:construct"
    if arm == "N":
        cfg = Config(seed=0, input_hw=S.S15_HW, physical_batch_size=1, amp_dtype="float32") if fixture else None
        e, info = S.create(c["parent"]["path"], device, cfg, seed=0, fixture=fixture, activation_checkpoint=True)
        info = {"S15_bridge": info}
    else:
        e, info = R.create(c["parent"]["path"], device, arm, fixture=fixture, activation_checkpoint=True)
    e.r27_arm = arm
    return e, info


def binding(c, e, scope, audit=None, role=None):
    from formal_data import cursor
    return {"schema": SCHEMA, "scope": scope, "arm": e.r27_arm, "contract_sha256": c["contract_sha256"],
            "identity_sha256": c["identity_sha256"], "parent_sha256": c["parent"]["sha256"],
            "global_update": e.completed_updates, "cursor": cursor(e.completed_updates), "config": e.config.binding(),
            "audit": audit, "role": role, "activation_checkpoint": e.s15_activation_checkpoint,
            "torch": str(torch_mod().__version__)}


def save(c, e, path, scope, audit=None, role=None):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import assert_healthy
    torch = torch_mod()
    path = Path(path)
    require(e.ema.updates == e.completed_updates, "EMA clock at save")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    b = binding(c, e, scope, audit, role)
    state["r27_binding_sha256"] = sha_bytes(canonical(b))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()
    side = {"status": "COMPLETE", **b, "file": ref(path)}
    write_new(path.with_suffix(".pth.json"), side)
    return side


def read_state(c, path, scope, arm):
    torch = torch_mod()
    path = Path(path).resolve()
    side = read_json(path.with_suffix(".pth.json"))
    require(side["scope"] == scope and side["arm"] == arm and side["contract_sha256"] == c["contract_sha256"]
            and side["schema"] == SCHEMA and side["parent_sha256"] == c["parent"]["sha256"]
            and side["identity_sha256"] == c["identity_sha256"], "R27 state scope/arm/contract")
    require(side["file"] == ref(path), "R27 state file identity")
    s = torch.load(path, map_location="cpu", weights_only=True)
    expect = {k: v for k, v in side.items() if k not in ("status", "file")}
    require(s.pop("r27_binding_sha256") == sha_bytes(canonical(expect)), "R27 binding digest")
    require(s["engine"]["completed_updates"] == side["global_update"] == s["ema"]["updates"], "R27 state clock")
    return side, s


def load(c, e, path, scope):
    """Exact restore into a fresh engine of the same arm; other scope/arm/contract/clock refused."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    side, s = read_state(c, path, scope, e.r27_arm)
    require(side["config"] == e.config.binding() and side["activation_checkpoint"] == e.s15_activation_checkpoint,
            "R27 state config / memory mode")
    require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "R27 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    modes = dict(e.model.named_modules())
    require(set(modes) == set(s["model_modes"]), "R27 mode keys")
    for n, m in s["model_modes"].items():
        modes[n].training = m
    e.load_state_dict(s["engine"])
    ck.restore_rng(s["rng"], e.device)
    require(ck.exact_equal(S.capture(e), s), "R27 full state did not restore exactly")
    return side


def register_counts(e, counts):
    e.model.register_forward_pre_hook(lambda *_: counts.update(forward_attempts=1))
    e.model.register_forward_hook(lambda *_: counts.update(forwards=1))
    e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))


def key_map(data):
    return {r["stable_image_id"]: r["relative_path"] for r in data.records}


def feed_keys(e, keys, ids, scope, epoch):
    if e.r27_arm == "N":
        return
    ctx = e.model.r27_ctx
    require(not ctx.route_keys, "R27 stale route keys")
    ctx.route_keys.extend((i, scope, keys[i], epoch, 0) for i in ids)


def drained(e):
    if e.r27_arm != "N":
        require(not e.model.r27_ctx.route_keys, "R27 route keys not consumed exactly")


# ================================================================ gradient monitor
def groups_of(name):
    from sparse_rtdetr.research import r27_local_memory as R
    if name.startswith(R.LOCAL + ".proj."):
        return ["new", "local_proj"]
    if name.startswith(R.LOCAL + "."):
        return ["new", "local_backbone"]
    if f".{R.READ}." in name:
        layer = name.split("decoder.decoder.layers.")[1].split(".")[0]
        part = name.split(f".{R.READ}.")[1].split(".")[0]
        return ["new", f"read{layer}", f"read{layer}.{part}", f"read.{part}"]
    for prefix, g in (("backbone.", "backbone"), ("encoder.", "encoder"), ("decoder.", "decoder")):
        if name.startswith(prefix):
            return ["old", g]
    return ["old", "other"]


class StepMonitor:
    """Gradient (post-clip, pre-step) norms by group and parameter-change norms of new parameters at registered
    updates; no RNG, no extra forward.  Gradients are task gradients (AdamW weight decay is applied in step)."""

    def __init__(self, e):
        self.e, self.active, self.before = e, False, {}
        e.optimizer.register_step_pre_hook(self.pre)
        e.optimizer.register_step_post_hook(self.post)

    def pre(self, *_):
        if not self.active:
            return
        sq, nz = Counter(), Counter()
        self.before = {}
        for n, p in self.e.model.named_parameters():
            gs = groups_of(n)
            if p.grad is not None:
                g = float(p.grad.detach().float().square().sum())
                for k in gs:
                    sq[k] += g
                    nz[k] += int(g > 0)
            if gs[0] == "new":
                self.before[n] = p.detach().float().clone()
        self.grad = {k: math.sqrt(v) for k, v in sq.items()}
        self.nonzero = dict(nz)

    def post(self, *_):
        if not self.active:
            return
        upd = Counter()
        for n, p in self.e.model.named_parameters():
            if n in self.before:
                d = float((p.detach().float() - self.before[n]).square().sum())
                for k in groups_of(n):
                    upd[k] += d
        self.row = {"grad_norm_post_clip": self.grad, "grad_nonzero_tensors": self.nonzero,
                    "update_norm_new": {k: math.sqrt(v) for k, v in upd.items()}}
        self.before = {}


# ================================================================ audit records
def audit_rows(e, ids):
    """Per-image routing/coverage record of the last update (two microbatches), detached; then cleared."""
    if e.r27_arm == "N":
        return None
    ctx = e.model.r27_ctx
    recs, ctx.records = ctx.records, []
    require(len(recs) == 2, f"R27 audit records per update {len(recs)}")
    out = []
    for rec in recs:
        tiles = rec["tiles"].cpu().tolist()
        mass = rec["mass"].cpu()
        sel_mass = [float(mass[i, tiles[i]].sum() / max(float(mass[i].sum()), 1e-30)) for i in range(len(tiles))]
        cov = rec["top300_center_selected"].cpu().tolist()
        layers = [{k: (v.cpu().tolist() if hasattr(v, "cpu") and v.dim() else (float(v) if hasattr(v, "cpu") else v))
                   for k, v in L.items()} for L in rec["layers"]]
        for i in range(len(tiles)):
            row = {"tiles": tiles[i], "mass": [round(float(m), 6) for m in mass[i]], "selected_mass_fraction": round(sel_mass[i], 6),
                   "top300_centre_selected": round(cov[i], 6),
                   "layers": [{k: (round(v[i], 6) if isinstance(v, list) else v) for k, v in L.items()
                               if k in ("covered", "read", "valid_point_fraction", "dn_covered", "dn_read")} for L in layers]}
            out.append(row)
    require(len(out) == len(ids), "audit/image count")
    for row, i in zip(out, ids):
        row["image_id"] = i
    return out, recs


def epoch_aggregate(rows):
    hist = [0] * 9
    n = len(rows)
    agg = {"images": n, "tile_histogram": hist}
    for r in rows:
        for t in r["tiles"]:
            hist[t] += 1
    agg["mean_selected_mass_fraction"] = sum(r["selected_mass_fraction"] for r in rows) / n
    agg["mean_top300_centre_selected"] = sum(r["top300_centre_selected"] for r in rows) / n
    for k in ("covered", "read", "valid_point_fraction", "dn_covered", "dn_read"):
        agg[f"layer_mean_{k}"] = [sum(r["layers"][j].get(k, 0.) for r in rows) / n for j in range(3)]
    return agg


# ================================================================ monitor64 data and observation
def monitor_data(c):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    from sparse_rtdetr.localization.data import RawAugmentation, parse_raw
    import sparse_rtdetr.localization.data as LD
    import sqc_run as B
    torch = torch_mod()
    data = S.train_dataset(c["data"]["train_core"], hw=S.S15_HW, seed=0)
    rs = {"H": RawAugmentation(Q.H_HW).resize}
    _, annotations = B.roots_of(data)
    rows = [data.by_id[i] for i in c["monitor_ids"]]
    pixels, targets = [], []
    for r in rows:
        px = B.load_pixels(data, r, rs)["H"]
        require(tuple(px.shape) == (3, 1152, 2016), "monitor pixel shape")
        pixels.append((px * 255).round().to(torch.uint8))
        require(torch.equal(pixels[-1].float() / 255., px), "uint8 pixel cache not exact")
        raw = LD.read_bound(annotations / Path(r["annotation_relative_path"]).name, r["annotation_sha256"],
                            r["annotation_size_bytes"])
        parsed = parse_raw(raw, r["stable_image_id"], expected_rows=r["raw_annotation_rows"])
        t = Q.h_training_target(RawAugmentation(Q.H_HW), parsed, (r["height"], r["width"]))
        t["image_id"] = r["stable_image_id"]
        targets.append(t)
    return rows, pixels, targets


def observe(e, monitor, label):
    """Eval/no_grad FP32 final-output readings on monitor64 plus routing/coverage/opportunity descriptions."""
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.baseline.training_v2b_engine import _move
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.research import shared_query_cls as Q
    import torchvision.ops as ops
    torch = torch_mod()
    rows, pixels, targets = monitor
    rng = ck.capture_rng(e.device)
    modes = {n: m.training for n, m in e.model.named_modules()}
    digest = Q.digest_tensors(e.model.state_dict().items())
    ctx = e.model.r27_ctx
    ctx.audit, ctx.monitor, ctx.records = True, [], []
    e.model.eval()
    batches, pairs, images = [], [], []
    try:
        with torch.no_grad(), torch.autocast(e.device.type, enabled=False), forward_guards(e.model):
            for k in range(0, OBS_IMAGES, OBS_BATCH):
                x = torch.stack(pixels[k:k + OBS_BATCH]).to(e.device).float() / 255.
                tg = _move(targets[k:k + OBS_BATCH], e.device)
                ctx.route_keys.extend((r["stable_image_id"], "eval", r["relative_path"], 0, 0) for r in rows[k:k + OBS_BATCH])
                out = e.model(x)
                require(not ctx.route_keys and set(out) == {"pred_logits", "pred_boxes"}, "observation keys/output")
                rec = ctx.records.pop()
                losses = e.criterion({"pred_logits": out["pred_logits"].float(), "pred_boxes": out["pred_boxes"].float()},
                                     [{k2: v for k2, v in t.items() if k2 != "image_id"} for t in tg])
                diag = e.criterion.last_diagnostics
                require(len(diag) == 1 and diag[0]["head"] == "final", "final-only criterion call")
                vals = {k2: float(v) for k2, v in losses.items()}
                batches.append({"images": [t["image_id"] for t in tg], **vals, "final_eval_loss": sum(vals.values()),
                                "GT": sum(len(t["labels"]) for t in tg)})
                tiles = rec["tiles"]
                pb_all = out["pred_boxes"].float()
                for i, t in enumerate(tg):
                    gb = t["boxes"].float()
                    buckets = [Q.size_bucket(float(a)) for a in t["raw_area"]]
                    core = R.core_of(gb[:, 0], gb[:, 1]) if len(gb) else torch.zeros(0, dtype=torch.long, device=e.device)
                    insel = (core[:, None] == tiles[i][None, :]).any(-1).tolist() if len(gb) else []
                    dup = []
                    if len(gb):
                        iou = ops.box_iou(ops.box_convert(pb_all[i], "cxcywh", "xyxy"), ops.box_convert(gb, "cxcywh", "xyxy"))
                        dup = (iou >= 0.5).sum(0).tolist()
                    per = {}
                    for bkt, s_, d_ in zip(buckets, insel, dup):
                        name = ["small", "medium", "large"][bkt]
                        z = per.setdefault(name, {"GT": 0, "GT_in_selected_cores": 0, "GT_with_pred_iou50": 0, "pred_iou50_sum": 0, "pred_iou50_max": 0})
                        z["GT"] += 1; z["GT_in_selected_cores"] += int(s_)
                        z["GT_with_pred_iou50"] += int(d_ > 0); z["pred_iou50_sum"] += int(d_)
                        z["pred_iou50_max"] = max(z["pred_iou50_max"], int(d_))
                    m = rec["mass"][i]
                    images.append({"image_id": t["image_id"], "tiles": tiles[i].tolist(),
                                   "mass": [float(v) for v in m], "selected_mass_fraction": float(m[tiles[i]].sum() / m.sum().clamp_min(1e-30)),
                                   "mass_hhi": float(((m / m.sum().clamp_min(1e-30)) ** 2).sum()),
                                   "top300_centre_selected": float(rec["top300_center_selected"][i]),
                                   "layers": [{"covered": float(L["covered"][i]), "read": float(L["read"][i]),
                                               "valid_point_fraction": float(L["valid_point_fraction"][i])} for L in rec["layers"]],
                                   "by_size": per})
                    q = diag[0]["matched_queries"][i]
                    gid = diag[0]["matched_GT_ids"][i]
                    if not q:
                        continue
                    pos = {g: j for j, g in enumerate(t["gt_ids"])}
                    gi = torch.tensor([pos[g] for g in gid], device=e.device)
                    pb = out["pred_boxes"][i, q].float()
                    g2 = t["boxes"][gi].float()
                    pxy, gxy = ops.box_convert(pb, "cxcywh", "xyxy"), ops.box_convert(g2, "cxcywh", "xyxy")
                    iou = ops.box_iou(pxy, gxy).diag()
                    for j, g in enumerate(gid):
                        qc = int(R.core_of(pb[j:j + 1, 0], pb[j:j + 1, 1])[0])
                        pairs.append({"image": t["image_id"], "gt": g, "query": q[j], "bucket": Q.size_bucket(float(t["raw_area"][pos[g]])),
                                      "iou": float(iou[j]), "pred_core_selected": qc in tiles[i].tolist()})
                layer_res = [{"residual_norm": float(L["residual_norm"]), "residual_norm_top300": float(L["residual_norm_top300"])}
                             for L in rec["layers"]]
                batches[-1]["layers"] = layer_res
                del out, losses
    finally:
        ctx.audit, ctx.monitor, ctx.records = False, None, []
        ctx.route_keys.clear()
        for n, m in e.model.named_modules():
            m.training = modes[n]
    require(ck.exact_equal(ck.capture_rng(e.device), rng), "observation consumed RNG")
    require(Q.digest_tensors(e.model.state_dict().items()) == digest, "observation changed model state (BN?)")
    n = len(batches)
    summary = {"label": label, "batches": n, "final_eval_loss_mean": sum(b["final_eval_loss"] for b in batches) / n}
    for key in ("loss_vfl", "loss_bbox", "loss_giou"):
        summary[key + "_mean"] = sum(b[key] for b in batches) / n
    by = {}
    for p in pairs:
        s = by.setdefault(["small", "medium", "large"][p["bucket"]], Counter())
        s["pairs"] += 1; s["iou"] += p["iou"]; s["sel"] += int(p["pred_core_selected"])
        s["iou_sel"] += p["iou"] if p["pred_core_selected"] else 0.
    summary["by_size_matched_iou"] = {k: {"pairs": v["pairs"], "iou_mean": v["iou"] / v["pairs"],
                                          "pairs_pred_centre_in_selected_core": v["sel"],
                                          "iou_mean_selected": v["iou_sel"] / v["sel"] if v["sel"] else None,
                                          "iou_mean_unselected": (v["iou"] - v["iou_sel"]) / (v["pairs"] - v["sel"]) if v["pairs"] > v["sel"] else None}
                                      for k, v in sorted(by.items())}
    m = len(images)
    summary["routing"] = {"mean_selected_mass_fraction": sum(i["selected_mass_fraction"] for i in images) / m,
                          "mean_mass_hhi": sum(i["mass_hhi"] for i in images) / m,
                          "mean_top300_centre_selected": sum(i["top300_centre_selected"] for i in images) / m,
                          "layer_mean_covered": [sum(i["layers"][j]["covered"] for i in images) / m for j in range(3)],
                          "layer_mean_read": [sum(i["layers"][j]["read"] for i in images) / m for j in range(3)],
                          "layer_mean_valid_point_fraction": [sum(i["layers"][j]["valid_point_fraction"] for i in images) / m for j in range(3)],
                          "tile_histogram": [sum(t == j for i in images for t in i["tiles"]) for j in range(9)],
                          "layer_residual_norm_mean": [sum(b["layers"][j]["residual_norm"] for b in batches) / n for j in range(3)]}
    opp = {}
    for i in images:
        for k, v in i["by_size"].items():
            z = opp.setdefault(k, Counter())
            for kk, vv in v.items():
                if kk == "pred_iou50_max":
                    z[kk] = max(z[kk], vv)
                else:
                    z[kk] += vv
    summary["opportunity_by_size"] = {k: dict(v, GT_in_selected_fraction=v["GT_in_selected_cores"] / v["GT"] if v["GT"] else None,
                                              pred_iou50_per_covered_GT=v["pred_iou50_sum"] / v["GT_with_pred_iou50"] if v["GT_with_pred_iou50"] else None)
                                      for k, v in sorted(opp.items())}
    return {"summary": summary, "batches": batches, "pairs": pairs, "images": images}


# ================================================================ training loop (diag and formal)
def run_loop(c, e, out, stop, scope, counts, *, save_points, monitor_points=(), ledger_rows=None, prior_rows=None):
    from formal_data import windows, WINDOWS_PER_EPOCH
    from sparse_rtdetr.research import scale_baseline as S
    out = Path(out)
    history = out / "UPDATES.jsonl"
    audit_dir = out / "AUDIT"
    audit_dir.mkdir(exist_ok=True)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    keys = key_map(data)
    mon = StepMonitor(e)
    start = e.completed_updates
    rec0 = S.RECOMPUTE["calls"]
    lb0 = e.model.r27_ctx.counters["local_backbone_calls"] if e.r27_arm != "N" else 0
    image_ids, monitored, saves = [], {}, []
    prior_rows = dict(prior_rows or {})
    replayed = 0
    seen = {}
    if e.r27_arm != "N":
        e.model.r27_ctx.audit = True
    with history.open("a", buffering=1) as ledger:
        while e.completed_updates < stop:
            begin = e.completed_updates
            epoch_rows = []
            epoch = begin // WINDOWS_PER_EPOCH + 1
            afile = (audit_dir / f"epoch-{epoch:03d}.jsonl").open("a", buffering=1)
            try:
                for x, t, inp in windows(data, begin, limit=min(stop - begin, WINDOWS_PER_EPOCH - begin % WINDOWS_PER_EPOCH)):
                    ids = [tt["image_id"] for tt in t]
                    s_ = seen.setdefault(inp["epoch"], set())
                    require(len(set(ids)) == len(ids) and not (s_ & set(ids)), "R27 occurrence: image repeated within an epoch")
                    s_.update(ids)
                    rel = e.completed_updates + 1 - start
                    watch = rel in monitor_points
                    if e.r27_arm != "N":
                        e.model.r27_ctx.monitor = [] if watch else None
                    mon.active = watch
                    PHASE["phase"] = f"{e.r27_arm}:update {e.completed_updates + 1}"
                    feed_keys(e, keys, ids, "train", inp["epoch"])
                    res = e.train_window(x, t)
                    drained(e)
                    mon.active = False
                    require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates,
                            "R27 update clock/input")
                    image_ids.extend(ids)
                    row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": res["loss"],
                           "lr": res["lr"], "gradient_norm_before_clip": res["gradient_norm_before_clip"],
                           "microbatch_losses": res["microbatch_losses"], "update_seconds": res["elapsed_seconds"],
                           "ema_updates": e.ema.updates}
                    if e.r27_arm != "N":
                        arows, recs = audit_rows(e, ids)
                        afile.write(json.dumps({"update": res["update"], "images": arows}, separators=(",", ":")) + "\n")
                        epoch_rows.extend(arows)
                        e.model.r27_ctx.monitor = None
                    if watch:
                        factor = min(1., e.config.clip_max_norm / (res["gradient_norm_before_clip"] + 1e-6))
                        m = dict(mon.row)
                        m["grad_norm_pre_clip"] = {k: v / factor for k, v in m["grad_norm_post_clip"].items()}
                        if e.r27_arm != "N":
                            m["layer_residual"] = [[{k: float(v) for k, v in L.items() if k.startswith("residual") or k == "output_proj_input_norm"}
                                                    for L in r["layers"]] for r in recs]
                        monitored[rel] = m
                        row["monitor"] = m
                    if res["update"] in prior_rows:
                        require(prior_rows.pop(res["update"]) == {k: v for k, v in row.items() if k not in ("update_seconds", "monitor")},
                                "R27 recovery replay differs from the recorded trajectory")
                        replayed += 1
                    if ledger_rows is not None:
                        ledger_rows.append(row)
                    ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    if e.completed_updates in save_points:
                        ledger.flush()
                        os.fsync(ledger.fileno())
                        afile.flush()
                        _available(out)
                        ep = e.completed_updates // WINDOWS_PER_EPOCH
                        if scope == "formal":
                            name = f"checkpoint-epoch-{ep:03d}.pth"
                            role = "formal_save" if ep in FORMAL_SAVE_EPOCHS else "recovery"
                        else:
                            name, role = f"state-update-{e.completed_updates}.pth", scope
                        saves.append(save(c, e, out / name, scope, ref(history), role))
                    if e.completed_updates % 16 == 0:
                        print(json.dumps({"arm": e.r27_arm, "update": e.completed_updates, "loss": res["loss"],
                                          "seconds": res["elapsed_seconds"]}), flush=True)
            finally:
                afile.close()
            if epoch_rows and (e.completed_updates % WINDOWS_PER_EPOCH == 0):
                with (out / "AUDIT_EPOCHS.jsonl").open("a") as f:
                    f.write(json.dumps({"epoch": epoch, "end_update": e.completed_updates, **epoch_aggregate(epoch_rows)}) + "\n")
            require(e.completed_updates != begin, "R27 loader made no progress")
    if e.r27_arm != "N":
        e.model.r27_ctx.audit = False
    done = e.completed_updates - start
    rec = S.RECOMPUTE["calls"] - rec0
    per_update = 16 if e.r27_arm == "N" else 20
    require(e.completed_updates == stop and counts["actual_updates"] >= done and not prior_rows, "R27 loop count")
    require(rec == per_update * done, f"R27 block recomputation count {rec}")
    lb = (e.model.r27_ctx.counters["local_backbone_calls"] - lb0) if e.r27_arm != "N" else 0
    require(e.r27_arm == "N" or lb == 2 * done, f"R27 local shallow forward count {lb}")
    return {"updates": done, "block_recomputations": rec, "local_shallow_forwards": lb,
            "local_patches": 2 * done * 8 * 4 if e.r27_arm != "N" else 0, "monitored": monitored, "saves": saves,
            "training_image_ids": image_ids, "prior_rows_replayed_exact": replayed}


def _available(path, need=2 * 1024 ** 3):
    free = shutil.disk_usage(path).free
    require(free > need, f"insufficient disk for the next save: {free} bytes")


# ================================================================ reference micro-steps (zero-step identity)
def micro_steps(model, criterion, images, targets, device, amp, *, old_names):
    """The engine's two equal microbatch forwards/backwards without clip/step; returns losses, output and DN
    digests and the digest of the old-parameter gradients (in ``old_names`` order).  Gradients are cleared."""
    import contextlib
    from sparse_rtdetr.protocol_r18.engine import output_digest, tensor_digest
    from sparse_rtdetr.protocol_r18.model import dn_capture, fp32_outputs
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.baseline.training_v2b_engine import _batchnorm_backward_layout, _move
    torch = torch_mod()
    model.train(); criterion.train()
    for p in model.parameters():
        p.grad = None
    bs = len(targets) // 2
    losses, outs, dns = [], [], []
    with _batchnorm_backward_layout(model, torch.device(device)):
        for micro in range(2):
            sub = _move(targets[micro * bs:(micro + 1) * bs], torch.device(device))
            batch = images[micro * bs:(micro + 1) * bs].to(device)
            ctxm = torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if amp == "bfloat16" else contextlib.nullcontext()
            with ctxm, dn_capture(model) as noise, forward_guards(model):
                out = model(batch, targets=sub)
            with torch.autocast(torch.device(device).type, enabled=False):
                ls = criterion(fp32_outputs(out), sub)
                obj = sum(ls.values()) * .5
            obj.backward()
            losses.append({k: float(v.detach()) for k, v in ls.items()})
            outs.append(output_digest(out))
            dns.append(output_digest(noise))
            del out, ls, obj
    named = dict(model.named_parameters())
    grads = tensor_digest((n, named[n].grad) for n in old_names)
    new_grads = {n: float(p.grad.detach().float().norm()) if p.grad is not None else None
                 for n, p in named.items() if n not in set(old_names)}
    for p in model.parameters():
        p.grad = None
    return {"losses": losses, "loss": sum(sum(v.values()) for v in losses) / 2., "outputs": outs, "dn": dns,
            "old_gradient_sha256": grads, "new_gradient_norms": new_grads}


def native_l(c, device, *, fixture):
    """Native RT-DETRv2-R18 at L from the same parent (the R25 A0 constructor), with the S15 checkpoint segments."""
    from sparse_rtdetr.research.query_local_runtime import create as l_create
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.model import Config
    cfg = Config(seed=0, input_hw=(768, 1344), physical_batch_size=1, amp_dtype="float32") if fixture else Config(seed=0)
    e, info = l_create(c["parent"]["path"], "A0", device, cfg, synthetic=fixture)  # fixture recipe differs from the parent binding
    S.install_activation_checkpoint(e.model)
    e.s15_activation_checkpoint = True
    return e, info


# ================================================================ CPU checks
def cpu(c, out):
    import subprocess
    import copy
    torch = torch_mod()
    from r25_run import setup
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.baseline.training_v2b_engine import _move
    from sparse_rtdetr.protocol_r18.engine import output_digest
    from formal_data import windows
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU checks hide CUDA")
    setup("cpu")
    torch.set_num_threads(int(os.environ.get("R27_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    res = {}
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests/test_r27_local_memory.py")],
                       capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-3:]}
    require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-4000:])
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    keys = key_map(data)
    (x, t, inp), = list(windows(data, 18240, limit=1))
    x, t = x[:2], t[:2]
    ids = [tt["image_id"] for tt in t]
    eng, info = {}, {}
    for a in ("C", "P"):
        eng[a], info[a] = make(c, "cpu", a, fixture=True)
    res["install"] = {a: {k: v for k, v in info[a].items() if k not in ("new_parameter_names",)} for a in info}
    require(info["C"]["new_parameters_sha256"] == info["P"]["new_parameters_sha256"]
            and info["C"]["old_parameters_sha256"] == info["P"]["old_parameters_sha256"], "C/P initial parameters differ")
    ref_e, _ = native_l(c, "cpu", fixture=True)
    old_names = [n for n, _ in ref_e.model.named_parameters()]
    require(old_names == [n for n, _ in eng["P"].model.named_parameters() if not R.is_new(n)], "old parameter order")
    # ---- eval identity: C/P(I_H) == native(I_L), C == P
    zero = {}
    xl = R.to_low(x)
    for m in (ref_e.model, eng["C"].model, eng["P"].model):
        m.eval()
    rng0 = ck.capture_rng("cpu")
    with torch.no_grad():
        oN = ref_e.model(xl)
        oP = eng["P"].model(x)
        feed_keys(eng["C"], keys, ids, "eval", 0)
        oC = eng["C"].model(x)
    require(ck.exact_equal(ck.capture_rng("cpu"), rng0), "eval forward consumed RNG")
    zero["eval_P_equals_native_L"] = all(torch.equal(oN[k], oP[k]) for k in oN)
    zero["eval_C_equals_native_L"] = all(torch.equal(oN[k], oC[k]) for k in oN)
    require(zero["eval_P_equals_native_L"] and zero["eval_C_equals_native_L"], "zero-step eval output differs from native(I_L)")
    zero["I_L_range"] = [float(xl.min()), float(xl.max())]
    # ---- P route uses the native encoder selection (eval: recompute the selection independently)
    with torch.no_grad():
        mP = eng["P"].model
        ctx = mP.r27_ctx
        ctx.audit = True
        mP(x)
        rec = ctx.records.pop()
        ctx.audit = False
        feats = mP.encoder(mP.backbone(R.to_low(x)))
        d = mP.decoder
        memory, shapes = d._get_encoder_input(feats)
        ctx.x = x  # score-head record path active for this independent call only
        _, unact, _, _ = d._get_decoder_input(memory, shapes)
        logits = ctx.enc_logits
        ctx.clear()
        tiles2, mass2, _ = R.route_p(logits, unact.sigmoid())
    zero["P_route_equals_independent_native_selection"] = torch.equal(rec["tiles"], tiles2) and torch.equal(rec["mass"], mass2)
    require(zero["P_route_equals_independent_native_selection"], "P routing differs from the independent native selection")
    zero["P_tiles_first_images"] = rec["tiles"].tolist()
    # ---- train-mode zero step: losses, outputs, DN, old-parameter gradients vs native(I_L); C == P
    bn_global_before = {n: v.clone() for n, v in eng["P"].model.state_dict().items() if "running" in n and not n.startswith(R.LOCAL)}
    steps = {}
    for name, e, xx in (("N_L", ref_e, xl), ("P", eng["P"], x), ("C", eng["C"], x)):
        ck.restore_rng(rng0, "cpu")
        if name == "C":
            feed_keys(e, keys, ids, "train", 61)
        steps[name] = micro_steps(e.model, e.criterion, xx, t, "cpu", "float32", old_names=old_names)
        steps[name]["rng_after"] = ck.capture_rng("cpu")
        if name != "N_L":
            drained(e)
    for a in ("P", "C"):
        zero[f"train_{a}_losses_equal_native_L"] = steps[a]["losses"] == steps["N_L"]["losses"]
        zero[f"train_{a}_outputs_equal_native_L"] = steps[a]["outputs"] == steps["N_L"]["outputs"]
        zero[f"train_{a}_DN_equal_native_L"] = steps[a]["dn"] == steps["N_L"]["dn"]
        zero[f"train_{a}_old_gradients_equal_native_L"] = steps[a]["old_gradient_sha256"] == steps["N_L"]["old_gradient_sha256"]
        zero[f"train_{a}_RNG_after_equal_native_L"] = ck.exact_equal(steps[a]["rng_after"], steps["N_L"]["rng_after"])
        nz = {n: v for n, v in steps[a]["new_gradient_norms"].items() if v}
        zero[f"train_{a}_nonzero_new_gradients"] = sorted(nz)
        require(all(zero[f"train_{a}_{k}"] for k in ("losses_equal_native_L", "outputs_equal_native_L", "DN_equal_native_L",
                                                      "old_gradients_equal_native_L", "RNG_after_equal_native_L")),
                f"zero-step train identity failed for {a}")
        require(sorted(nz) and all("output_proj" in n for n in nz), f"{a}: first-step task gradient outside output_proj {sorted(nz)}")
    zero["train_loss"] = steps["P"]["loss"]
    # BN separation: global BN running stats after the P micro steps equal native L's; local BN moved
    nsd = ref_e.model.state_dict()
    psd = eng["P"].model.state_dict()
    zero["global_BN_equal_native_after_train_forward"] = all(torch.equal(psd[n], nsd[n]) for n in bn_global_before)
    zero["global_BN_changed_by_forward"] = any(not torch.equal(psd[n], bn_global_before[n]) for n in bn_global_before)
    loc = [n for n in psd if n.startswith(R.LOCAL) and "running_mean" in n]
    zero["local_BN_changed"] = sum(not torch.equal(psd[n], psd[n.replace(R.LOCAL + ".conv1.", "backbone.conv1.").replace(R.LOCAL + ".res_layers.", "backbone.res_layers.").replace(R.LOCAL + ".proj.", "encoder.input_proj.0.")]) for n in loc)
    zero["local_BN_tracked"] = sorted(set(int(v) for n, v in psd.items() if n.startswith(R.LOCAL) and n.endswith("num_batches_tracked")))
    zero["global_BN_tracked_delta"] = sorted(set(int(psd[n]) - int(nsd[n]) for n in psd if n.endswith("num_batches_tracked") and not R.is_new(n)))
    require(zero["global_BN_equal_native_after_train_forward"] and zero["global_BN_changed_by_forward"]
            and zero["global_BN_tracked_delta"] == [0], "global BN separation")
    require(zero["local_BN_changed"] == len(loc), "local BN must update from local patches")
    # ---- P route invariance to DN/GT: same images, altered GT (labels/boxes) -> same route on the same encoder output
    with torch.no_grad():
        mP.train()
        ctx.audit = True
        t_alt = copy.deepcopy(t)
        for tt in t_alt:
            tt["labels"] = (tt["labels"] + 3) % 10
            tt["boxes"] = tt["boxes"].clone()
            tt["boxes"][:, :2] = (tt["boxes"][:, :2] * 0.5 + 0.2)
        routes = []
        for tg in (t, t_alt):
            sd = {n: v.clone() for n, v in mP.state_dict().items()}
            ck.restore_rng(rng0, "cpu")
            mP(x[:1], targets=_move(tg[:1], torch.device("cpu")))
            r = ctx.records.pop()
            routes.append((r["tiles"].clone(), r["mass"].clone()))
            mP.load_state_dict(sd)
        ctx.audit = False
    zero["P_route_invariant_to_GT_and_DN"] = torch.equal(routes[0][0], routes[1][0]) and torch.equal(routes[0][1], routes[1][1])
    require(zero["P_route_invariant_to_GT_and_DN"], "P route changed with GT/DN")
    res["zero_step"] = zero
    # ---- one real fixture update on C and P from the same RNG: outputs, DN, loss, RNG equal; then state/refusals
    del ref_e
    gc.collect()
    eng = {a: make(c, "cpu", a, fixture=True)[0] for a in ("C", "P")}
    rows = {}
    for a in ("C", "P"):
        e = eng[a]
        ck.restore_rng(rng0, "cpu")
        feed_keys(e, keys, ids, "train", 61)
        r = e.train_window(x, t, capture=True)
        drained(e)
        rows[a] = {"loss": r["loss"], "outputs": r["output_sha256"], "dn": r["dn_noise_sha256"], "rng": ck.capture_rng("cpu")}
    one = {"C_P_outputs_equal": rows["C"]["outputs"] == rows["P"]["outputs"], "C_P_DN_equal": rows["C"]["dn"] == rows["P"]["dn"],
           "C_P_loss": [rows["C"]["loss"], rows["P"]["loss"]],
           "C_P_RNG_after_equal": ck.exact_equal(rows["C"]["rng"], rows["P"]["rng"])}
    require(one["C_P_outputs_equal"] and one["C_P_DN_equal"] and rows["C"]["loss"] == rows["P"]["loss"] and one["C_P_RNG_after_equal"],
            "C/P first fixture update differs")
    e = eng["P"]
    sd = e.model.state_dict()
    one["output_proj_nonzero_after_update"] = [bool(sd[f"decoder.decoder.layers.{i}.{R.READ}.output_proj.weight"].abs().sum() > 0) for i in range(3)]
    one["adam_step_new"] = sorted(set(int(e.optimizer.state[p]["step"]) for n, p in e.model.named_parameters() if R.is_new(n) and p in e.optimizer.state))
    one["BN_tracked_delta_one_update"] = sorted(set(int(v) for n, v in sd.items() if n.endswith("num_batches_tracked") and n.startswith("backbone.")))
    one["recomputations_per_update"] = None
    res["first_update"] = one
    names = [n for g in e.optimizer.param_groups for n in g["param_names"]]
    res["optimizer"] = {"groups": {g["name"]: len(g["params"]) for g in e.optimizer.param_groups},
                        "covers_all_trainable": sorted(names) == sorted(n for n, p in e.model.named_parameters() if p.requires_grad),
                        "new_group_mapping": info["P"]["new_parameter_group_mapping"]}
    require(res["optimizer"]["covers_all_trainable"], "optimizer coverage")
    emasd = e.ema.module.state_dict()
    res["EMA"] = {"has_new": all(n in emasd for n in sd if R.is_new(n)), "updates": e.ema.updates,
                  "new_requires_grad": any(p.requires_grad for n, p in e.ema.module.named_parameters() if R.is_new(n)),
                  "arm": R.arm_of(e.ema.module)}
    require(res["EMA"]["has_new"] and not res["EMA"]["new_requires_grad"] and res["EMA"]["arm"] == "P", "EMA coverage")
    dc = copy.deepcopy(e.ema.module)
    require(R.arm_of(dc) == "P" and dc.r27_ctx is not e.ema.module.r27_ctx and dc.r27_ctx.local is dc.r27_local
            and all(l.r27_ctx is dc.r27_ctx for l in dc.decoder.decoder.layers), "deepcopy context identity")
    res["deepcopy_context_isolated"] = True
    # C route stateless: same key after save/restore gives the same tiles; RNG untouched by routing
    rng1 = ck.capture_rng("cpu")
    tiles_a = [R.route_c_one("train", keys[i], 61, 0) for i in ids]
    require(ck.exact_equal(ck.capture_rng("cpu"), rng1), "C routing consumed RNG")
    st = Path(out) / "cpu-state-P.pth"
    side = save(c, e, st, "cpu-fixture")
    fresh, _ = make(c, "cpu", "P", fixture=True)
    load(c, fresh, st, "cpu-fixture")
    stc = Path(out) / "cpu-state-C.pth"
    save(c, eng["C"], stc, "cpu-fixture")
    freshc, _ = make(c, "cpu", "C", fixture=True)
    load(c, freshc, stc, "cpu-fixture")
    tiles_b = [R.route_c_one("train", keys[i], 61, 0) for i in ids]
    res["C_route_restore_consistent"] = tiles_a == tiles_b
    res["C_route_tiles_first_window_images"] = tiles_a
    refused = []
    for label, arm, path, scope in (("P state into C engine", "C", st, "cpu-fixture"), ("C state into P engine", "P", stc, "cpu-fixture"),
                                    ("fixture state in diagnostic scope", "P", st, "diagnostic"),
                                    ("fixture state in formal scope", "P", st, "formal"),
                                    ("P state into N engine", "N", st, "cpu-fixture")):
        other, _ = make(c, "cpu", arm, fixture=True)
        try:
            load(c, other, path, scope)
        except (ValueError, RuntimeError, KeyError) as exc:
            refused.append({"case": label, "reason": repr(exc)[:200]})
        else:
            raise AssertionError("accepted: " + label)
        del other
        gc.collect()
    try:
        R.create(LD_ROOT / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/checkpoint-epoch-075.pth"
                 if (LD_ROOT / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/checkpoint-epoch-075.pth").exists()
                 else str(S15_UPDATES), "cpu", "P", fixture=True)
    except Exception as exc:
        refused.append({"case": "non-e060 parent", "reason": repr(exc)[:200]})
    else:
        raise AssertionError("accepted a non-e060 parent")
    for label, fn in (("formal entry resuming a fixture state", lambda: formal_guard(c, "P", st)),
                      ("formal entry without promoted gate", lambda: formal_guard(c, "P", None, gate=Path(out) / "missing-GATE.json"))):
        try:
            fn()
        except ValueError as exc:
            refused.append({"case": label, "reason": repr(exc)[:200]})
        else:
            raise AssertionError("formal entry accepted: " + label)
    res["refused"] = refused
    res["state"] = side
    # MACs (counted, descriptive; not a cost result)
    from torch.utils.flop_counter import FlopCounterMode
    macs = {}
    ne, _ = make(c, "cpu", "N", fixture=True)
    for a, m in (("N", ne.model), ("C", freshc.model), ("P", fresh.model)):
        m.eval()
        if a == "C":
            feed_keys(freshc, keys, ids[:1], "eval", 0)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            m(x[:1])
        macs[a] = {"model_MACs_counted": int(fc.get_total_flops() // 2)}
        if a != "N":
            cnt = fc.get_flop_counts()
            k = [k for k in cnt if k.endswith(R.LOCAL)]
            macs[a]["local_backbone_and_proj_MACs"] = int(sum(cnt[k[0]].values()) // 2) if k else None
    res["MACs"] = macs
    res["parameters"] = {a: sum(p.numel() for p in m.parameters()) for a, m in (("N", ne.model), ("C", freshc.model), ("P", fresh.model))}
    res["formal_plan"] = formal_plan(c, "P")
    res["cpu_seconds_wall"] = time.perf_counter() - t0
    res["cpu_fixture_updates"] = 2
    return res


# ================================================================ CUDA acceptance
def accept(c, out, arm, counts):
    from formal_data import windows
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from s15_formal_train import _differences, _result_body, _bn_counts
    torch = torch_mod()
    out = Path(out)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    keys = key_map(data)
    wins = list(windows(data, 18240, limit=2))
    result = {"arm": arm}
    if arm != "N":
        # CUDA zero-step reference: R27 arm on I_H vs native L on I_L, same window/RNG, no update
        ref_e, _ = native_l(c, "cuda:0", fixture=False)
        old_names = [n for n, _ in ref_e.model.named_parameters()]
        x, t, _ = wins[0]
        rng0 = ck.capture_rng(ref_e.device)
        nl = micro_steps(ref_e.model, ref_e.criterion, R.to_low(x), t, "cuda:0", "bfloat16", old_names=old_names)
        nl_rng = ck.capture_rng(ref_e.device)
        del ref_e
        gc.collect(); torch.cuda.empty_cache()
        e0, _ = make(c, "cuda:0", arm)
        ck.restore_rng(rng0, e0.device)
        feed_keys(e0, keys, [tt["image_id"] for tt in t], "train", 61)
        mine = micro_steps(e0.model, e0.criterion, x, t, "cuda:0", "bfloat16", old_names=old_names)
        drained(e0)
        z = {"losses_equal": mine["losses"] == nl["losses"], "outputs_equal": mine["outputs"] == nl["outputs"],
             "DN_equal": mine["dn"] == nl["dn"], "old_gradients_equal": mine["old_gradient_sha256"] == nl["old_gradient_sha256"],
             "RNG_after_equal": ck.exact_equal(ck.capture_rng(e0.device), nl_rng), "loss": [mine["loss"], nl["loss"]],
             "nonzero_new_gradients": sorted(n for n, v in mine["new_gradient_norms"].items() if v),
             "outputs_sha256": mine["outputs"]}
        counts.update(reference_forwards=4)
        require(z["losses_equal"] and z["outputs_equal"] and z["DN_equal"] and z["old_gradients_equal"] and z["RNG_after_equal"],
                f"CUDA zero-step identity failed {z}")
        require(z["nonzero_new_gradients"] and all("output_proj" in n for n in z["nonzero_new_gradients"]), "CUDA first-step new gradients")
        result["cuda_zero_step_reference"] = z
        del e0
        gc.collect(); torch.cuda.empty_cache()
    e, info = make(c, "cuda:0", arm)
    register_counts(e, counts)
    bn0 = _bn_counts(e.model)
    rows = []
    for i, (x, t, inp) in enumerate(wins, 1):
        feed_keys(e, keys, [tt["image_id"] for tt in t], "train", inp["epoch"])
        r = e.train_window(x, t, capture=True)
        drained(e)
        if arm != "N":
            e.model.r27_ctx.records = []
        rows.append({"input": inp, "update": r})
        if i == 1:
            saved = save(c, e, out / "u1.pth", "acceptance")
    expected = S.capture(e)
    bn = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    require(bn == [4], f"BN delta {bn}")
    del e
    gc.collect(); torch.cuda.empty_cache()
    clone, _ = make(c, "cuda:0", arm)
    register_counts(clone, counts)
    load(c, clone, out / "u1.pth", "acceptance")
    x, t, inp = wins[1]
    feed_keys(clone, keys, [tt["image_id"] for tt in t], "train", inp["epoch"])
    replay = clone.train_window(x, t, capture=True)
    drained(clone)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]["update"]), _result_body(replay))
    require(not diff and ck.exact_equal(expected, got), f"replay differs {diff[:5]}")
    require(counts["actual_updates"] == 3 and counts["forwards"] == 6, f"acceptance counts {dict(counts)}")
    first = [{"input": r["input"], "loss": r["update"]["loss"], "lr": r["update"]["lr"],
              "grad_norm": r["update"]["gradient_norm_before_clip"], "output_sha256": r["update"]["output_sha256"]} for r in rows]
    if arm == "N":
        hist = []
        with S15_UPDATES.open() as f:
            for _ in range(2):
                hist.append(json.loads(f.readline()))
        result["N_vs_S15_formal_002"] = [{"input_equal": h["input"] == r["input"], "loss_equal": h["loss"] == r["loss"],
                                          "lr_equal": h["lr"] == r["lr"], "grad_norm_equal": h["gradient_norm_before_clip"] == r["grad_norm"]}
                                         for h, r in zip(hist, first)]
        require(all(all(v.values()) for v in result["N_vs_S15_formal_002"]), "N acceptance differs from S15 formal-002 history")
    result.update({"install": {k: v for k, v in info.items() if k != "new_parameter_names"} if arm != "N" else info,
                   "first_updates": first, "saved_u1": saved, "replay_exact": True, "BN_delta_two_updates": bn,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "formal_state_usable": False})
    return result


# ================================================================ diagnostic 128
def diag(c, out, arm, counts):
    torch = torch_mod()
    require(arm in LEARN, "diagnostic arms are C and P (N reuses S15 history)")
    out = Path(out)
    monitor = monitor_data(c)
    e, info = make(c, "cuda:0", arm)
    register_counts(e, counts)
    PHASE["phase"] = f"{arm}:observe u0"
    obs0 = observe(e, monitor, "u0")
    write_new(out / "OBS-u0.json", obs0)
    s0 = save(c, e, out / "state-update-18240.pth", "diagnostic")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rows = []
    loop = run_loop(c, e, out, 18240 + DIAG_UPDATES, "diagnostic", counts, save_points={18240 + DIAG_UPDATES},
                    monitor_points=MONITOR_AT, ledger_rows=rows)
    train_wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    PHASE["phase"] = f"{arm}:observe u128"
    obs1 = observe(e, monitor, "u128")
    write_new(out / "OBS-u128.json", obs1)
    require(counts["actual_updates"] == DIAG_UPDATES and counts["forwards"] == 2 * DIAG_UPDATES + 2 * OBS_IMAGES // OBS_BATCH,
            f"diag counts {dict(counts)}")
    ids = loop.pop("training_image_ids")
    overlap = sorted(set(ids) & set(c["monitor_ids"]))
    write_new(out / "MONITORED.json", loop["monitored"])
    hist = []
    with S15_UPDATES.open() as f:
        for _ in range(DIAG_UPDATES):
            hist.append(json.loads(f.readline()))
    s15_loss = [h["loss"] for h in hist]
    return {"arm": arm, "install": {k: v for k, v in info.items() if k != "new_parameter_names"},
            "initial_state": s0, "end_state": loop["saves"][-1],
            "updates": loop["updates"], "block_recomputations": loop["block_recomputations"],
            "local_shallow_forwards": loop["local_shallow_forwards"], "local_patches": loop["local_patches"],
            "train_wall_seconds": train_wall, "mean_update_seconds": sum(r["update_seconds"] for r in rows) / len(rows),
            "median_update_seconds": statistics.median(r["update_seconds"] for r in rows),
            "peak_allocated_bytes_training": peak,
            "loss": {"u1": rows[0]["loss"], "u128": rows[-1]["loss"], "mean_1_16": sum(r["loss"] for r in rows[:16]) / 16,
                     "mean_113_128": sum(r["loss"] for r in rows[-16:]) / 16,
                     "mean_minus_S15_same_window": sum(r["loss"] - s for r, s in zip(rows, s15_loss)) / len(rows),
                     "all_finite": all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm_before_clip"]) for r in rows)},
            "inputs_equal_S15_windows": all(r["input"] == h["input"] for r, h in zip(rows, hist)),
            "lr_equal_S15": all(r["lr"] == h["lr"] for r, h in zip(rows, hist)),
            "monitor64_training_overlap": {"images_in_128_updates": len(ids), "distinct": len(set(ids)),
                                           "monitor_ids_seen": len(overlap)},
            "monitored_summary": {str(k): {"grad_norm_pre_clip": {g: v for g, v in m["grad_norm_pre_clip"].items()
                                                                  if g.startswith(("new", "local", "read"))},
                                           "update_norm_new": m["update_norm_new"], "layer_residual": m.get("layer_residual")}
                                  for k, m in loop["monitored"].items()},
            "observation": {"u0": obs0["summary"], "u128": obs1["summary"]}}


# ================================================================ cost (sections 7 and 10)
def timing(c, out, counts, models, keys_of, label):
    """Section 7 protocol on resident FP32 eval models {arm: model}; returns per-pass records and medians."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    from sparse_rtdetr.localization.data import RawAugmentation
    from sparse_rtdetr.r0.evaluation import rank_candidates
    import sqc_run as B
    torch = torch_mod()
    dev = torch.device("cuda:0")
    data = S.train_dataset(c["data"]["train_core"], hw=S.S15_HW, seed=0)
    rs = {"H": RawAugmentation(Q.H_HW).resize}
    rows = [data.by_id[i] for i in c["monitor_ids"]]
    batches = [list(range(k, k + COST_BATCH)) for k in range(0, OBS_IMAGES, COST_BATCH)]

    def run(arm, idx, inst=None):
        m = models[arm]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = torch.stack([B.load_pixels(data, rows[i], rs)["H"] for i in idx])
        t1 = time.perf_counter()
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
        h0.record()
        xg = x.to(dev)
        h1.record()
        if arm != "N":
            m.r27_ctx.route_keys.extend((rows[i]["stable_image_id"], "eval", rows[i]["relative_path"], 0, 0) for i in idx)
        g0.record()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            o = m(xg)
        g1.record()
        counts.update(forwards=1)  # plain counter outside the model (no hook on the timed path)
        logits, boxes = o["pred_logits"].float().cpu(), o["pred_boxes"].float().cpu()
        t2 = time.perf_counter()
        ranked = [rank_candidates(logits[j], boxes[j], rows[i], 600) for j, i in enumerate(idx)]
        t3 = time.perf_counter()
        require(len(ranked) == len(idx) and (arm == "N" or not m.r27_ctx.route_keys), "timing batch")
        rec = {"load_decode_resize_cpu_s": t1 - t0, "H2D_ms": h0.elapsed_time(h1), "model_gpu_ms": g0.elapsed_time(g1),
               "D2H_after_model_s": t2 - t1, "rank_top600_cpu_s": t3 - t2, "end_to_end_s": t3 - t0}
        if inst is not None:
            rec["stage_ms"] = inst()
        return rec, (logits, boxes)

    result = {"label": label, "warmup": {}, "passes": [], "instrumented": {}}
    ref_out = {}
    for arm in ARMS:
        result["warmup"][arm] = [run(arm, batches[i])[0] for i in range(COST_WARMUP)]
    for p, order in enumerate(ROTATIONS):
        for arm in order:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            recs = []
            for bi, b in enumerate(batches):
                r, o = run(arm, b)
                recs.append(r)
                if p == 0:
                    ref_out[(arm, bi)] = o
            result["passes"].append({"pass": p, "arm": arm, "batches": recs,
                                     "model_gpu_ms_total": sum(r["model_gpu_ms"] for r in recs),
                                     "end_to_end_s_total": sum(r["end_to_end_s"] for r in recs),
                                     "peak_allocated": torch.cuda.max_memory_allocated()})
    # instrumented pass (stage events), outputs checked equal to the no-hook pass
    for arm in ARMS:
        m = models[arm]
        ev, hs = {}, []
        named = {"backbone": m.backbone, "encoder": m.encoder, "decoder": m.decoder}
        if arm != "N":
            named.update({"local_conv1": m.r27_local.conv1, "local_res0": m.r27_local.res_layers[0],
                          "local_res1": m.r27_local.res_layers[1], "local_proj": m.r27_local.proj})
        for k, mod in named.items():
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            ev[k] = (a, b)
            hs.append(mod.register_forward_pre_hook(lambda *_x, a=a: a.record()))
            hs.append(mod.register_forward_hook(lambda *_x, b=b: b.record()))

        def inst(m=m, arm=arm, ev=ev):
            torch.cuda.synchronize()
            st = {k: a.elapsed_time(b) for k, (a, b) in ev.items()}
            if arm != "N":
                e2 = {k: v[-1] for k, v in m.r27_ctx.events.items()}
                st["H_to_L"] = e2["h2l_start"].elapsed_time(e2["h2l_end"])
                st["route"] = e2["prepare_start"].elapsed_time(e2["route_end"])
                st["patch_extract"] = e2["route_end"].elapsed_time(e2["extract_end"])
                st["local_backbone_and_proj"] = e2["extract_end"].elapsed_time(e2["local_backbone_end"])
                st["local_reads"] = sum(e2[f"read{i}_start"].elapsed_time(e2[f"read{i}_end"]) for i in range(3))
                st["decoder_native_rest"] = st["decoder"] - st["route"] - st["patch_extract"] - st["local_backbone_and_proj"] - st["local_reads"]
                m.r27_ctx.events = {}
            return st

        if arm != "N":
            m.r27_ctx.events = {}
        recs = []
        for bi, b in enumerate(batches):
            r, o = run(arm, b, inst)
            recs.append(r)
            require(torch.equal(o[0], ref_out[(arm, bi)][0]) and torch.equal(o[1], ref_out[(arm, bi)][1]),
                    f"instrumented output differs: {arm}")
        for h in hs:
            h.remove()
        if arm != "N":
            m.r27_ctx.events = None
        stages = {k: statistics.median(r["stage_ms"][k] for r in recs) for k in recs[0]["stage_ms"]}
        result["instrumented"][arm] = {"batches": recs, "stage_ms_median_per_batch": stages,
                                       "model_gpu_ms_total": sum(r["model_gpu_ms"] for r in recs),
                                       "outputs_equal_no_hook": True}
    med = {}
    for arm in ARMS:
        ps = [p for p in result["passes"] if p["arm"] == arm]
        med[arm] = {"model_gpu_ms_per_pass": [p["model_gpu_ms_total"] for p in ps],
                    "end_to_end_s_per_pass": [p["end_to_end_s_total"] for p in ps],
                    "model_gpu_ms_median_pass": statistics.median(p["model_gpu_ms_total"] for p in ps),
                    "end_to_end_s_median_pass": statistics.median(p["end_to_end_s_total"] for p in ps),
                    "load_decode_resize_s_median_pass": statistics.median(sum(b["load_decode_resize_cpu_s"] for b in p["batches"]) for p in ps),
                    "H2D_ms_median_pass": statistics.median(sum(b["H2D_ms"] for b in p["batches"]) for p in ps),
                    "rank_s_median_pass": statistics.median(sum(b["rank_top600_cpu_s"] for b in p["batches"]) for p in ps),
                    "per_image_model_gpu_ms": statistics.median(p["model_gpu_ms_total"] for p in ps) / OBS_IMAGES,
                    "peak_allocated": max(p["peak_allocated"] for p in ps),
                    "parameters": sum(p.numel() for p in models[arm].parameters())}
    ratios = {a: {"model_gpu_ratio_vs_N": med[a]["model_gpu_ms_median_pass"] / med["N"]["model_gpu_ms_median_pass"],
                  "end_to_end_ratio_vs_N": med[a]["end_to_end_s_median_pass"] / med["N"]["end_to_end_s_median_pass"]}
              for a in LEARN}
    result["medians"] = med
    result["ratios"] = ratios
    result["cost_condition"] = {a: {"model_gpu": ratios[a]["model_gpu_ratio_vs_N"] <= COST_GATE["model_gpu_ratio_max"],
                                    "end_to_end": ratios[a]["end_to_end_ratio_vs_N"] <= COST_GATE["end_to_end_ratio_max"]}
                                for a in LEARN}
    require(counts["forwards"] == 3 * COST_PER_ARM, f"cost forwards {dict(counts)}")
    return result


def cost(c, out, counts):
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.research import scale_baseline as S
    torch = torch_mod()
    models = {}
    e, _ = S.create(c["parent"]["path"], "cpu", seed=0)
    models["N"] = e.model.eval().requires_grad_(False).to("cuda:0")
    del e
    sources = {"N": {"weights": "S15 bridge of the e060 online state (timing does not depend on values)"}}
    for arm in LEARN:
        p = ROUND / f"diag-{arm}" / "state-update-18368.pth"
        side, s = read_state(c, p, "diagnostic", arm)
        models[arm] = R.build_eval_model(arm, s["model"], "cuda:0")
        sources[arm] = {"state": side["file"], "weights": "u128 online"}
        del s
    gc.collect(); torch.cuda.empty_cache()
    res = timing(c, out, counts, models, None, "pre-formal cost (section 7)")
    res["sources"] = sources
    return res


def final_cost(c, out, counts):
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.model import build
    from sparse_rtdetr.protocol_r18.geometry import validate_geometry
    torch = torch_mod()
    pinned(c["S15_endpoint"])
    s = torch.load(c["S15_endpoint"]["path"], map_location="cpu", weights_only=True)
    require(s["engine"]["completed_updates"] == s["ema"]["updates"] == ENDPOINT and s["config"] == S.production_config(0).binding(),
            "S15 e120 endpoint identity")
    m, _, _ = build(S.production_config(0), synthetic=True)
    m.load_state_dict(s["ema"]["module"], strict=True)
    validate_geometry(m)
    models = {"N": m.eval().requires_grad_(False).to("cuda:0")}
    sources = {"N": {"endpoint": c["S15_endpoint"], "weights": "S15 seed0 e120 EMA"}}
    del s
    for arm in LEARN:
        side, st = read_state(c, ROUND / f"formal-{arm}" / "checkpoint-epoch-120.pth", "formal", arm)
        models[arm] = R.build_eval_model(arm, st["ema"]["module"], "cuda:0")
        sources[arm] = {"endpoint": side["file"], "weights": "e120 EMA"}
        del st
    gc.collect(); torch.cuda.empty_cache()
    res = timing(c, out, counts, models, None, "final deployment timing (section 10 E)")
    res["sources"] = sources
    return res


# ================================================================ gate (section 8)
def gate(c, out):
    closed = {}
    for name in ["cpu", "accept-N", "accept-C", "accept-P", "diag-C", "diag-P", "cost"]:
        p = ROUND / name / "CLOSED.json"
        closed[name] = read_json(p) if p.exists() else None
    checks = {k: bool(v and v["status"] == "PASS" and v["contract_sha256"] == c["contract_sha256"]) for k, v in closed.items()}
    learn = {}
    for arm in LEARN:
        d = closed[f"diag-{arm}"]
        if not d:
            learn[arm] = {"passed": False}
            continue
        mon = d["monitored_summary"]
        late = [k for k in mon if int(k) >= 2]
        parts = ["local_backbone", "local_proj"] + [f"read{i}.{p}" for i in range(3) for p in ("value_proj", "sampling_offsets", "attention_weights", "output_proj")]
        grads = {p: max(mon[k]["grad_norm_pre_clip"].get(p, 0.) for k in late) for p in parts}
        reads = d["observation"]["u128"]["routing"]
        learn[arm] = {"updates": d["updates"], "finite": d["loss"]["all_finite"],
                      "task_gradient_max_after_u2": grads, "all_local_parts_have_task_gradient": all(v > 0 for v in grads.values()),
                      "u128_layer_mean_read": reads["layer_mean_read"], "u128_layer_residual_norm_mean": reads["layer_residual_norm_mean"],
                      "actual_read": all(v > 0 for v in reads["layer_mean_read"]) and all(v > 0 for v in reads["layer_residual_norm_mean"]),
                      "inputs_equal_S15_windows": d["inputs_equal_S15_windows"]}
        learn[arm]["passed"] = (d["updates"] == DIAG_UPDATES and learn[arm]["finite"] and learn[arm]["all_local_parts_have_task_gradient"]
                                and learn[arm]["actual_read"] and d["inputs_equal_S15_windows"])
    engineering = all(checks.values()) and all(v["passed"] for v in learn.values())
    cost_ = closed["cost"]
    cost_ok = bool(cost_) and all(all(v.values()) for v in cost_["cost_condition"].values())
    decision = "promoted_formal" if engineering and cost_ok else ("not_promoted_checks" if not engineering else "not_promoted_cost")
    g = {"status": "PASS", "contract_sha256": c["contract_sha256"], "stages_closed_pass": checks, "learning": learn,
         "engineering_passed": engineering,
         "cost": {"ratios": cost_["ratios"] if cost_ else None, "thresholds": COST_GATE,
                  "conditions": cost_["cost_condition"] if cost_ else None,
                  "medians": {a: {k: v for k, v in cost_["medians"][a].items() if k in ("model_gpu_ms_per_pass", "end_to_end_s_per_pass",
                                                                                       "model_gpu_ms_median_pass", "end_to_end_s_median_pass")}
                              for a in ARMS} if cost_ else None},
         "cost_passed": cost_ok, "decision": decision,
         "rule": "TASK_R27_001 section 8: section-6 checks + 128 finite updates + local task gradients + actual reads; "
                 "C and P each model-GPU ratio <=0.80 and end-to-end ratio <=0.95 vs N (ratio of three-pass medians). "
                 "No AP was read before this decision."}
    write_new(Path(out) / "GATE.json", g)
    return g


# ================================================================ formal (section 9)
def formal_plan(c, arm):
    from formal_data import WINDOWS_PER_EPOCH
    return {"arm": arm, "parent": c["parent"], "start_update": 18240, "stop_update": ENDPOINT, "updates": 18240,
            "save_points": sorted({ep * WINDOWS_PER_EPOCH for ep in range(60, 121, RECOVERY_EVERY)}),
            "formal_save_epochs": list(FORMAL_SAVE_EPOCHS), "scope": "formal",
            "evaluation": {"endpoint": "e120 EMA", "view": "A", "images": 548, "budgets": [600, 300], "score_calls": 2}}


def formal_guard(c, arm, resume, *, gate=ROUND / "gate" / "GATE.json"):
    require(arm in LEARN, "formal arms are C and P")
    if resume is not None:
        side = read_json(Path(resume).with_suffix(".pth.json"))
        require(side.get("scope") == "formal" and side.get("arm") == arm and side.get("contract_sha256") == c["contract_sha256"],
                "formal resume needs a formal state of this arm")
    g = Path(gate)
    require(g.exists(), "formal entry requires the section-8 GATE.json")
    gj = read_json(g)
    require(gj["decision"] == "promoted_formal" and gj["contract_sha256"] == c["contract_sha256"], "formal not promoted by the gate")
    return formal_plan(c, arm)


def formal(c, out, arm, counts, resume):
    from formal_data import WINDOWS_PER_EPOCH
    plan = formal_guard(c, arm, resume)
    out = Path(out)
    e, info = make(c, "cuda:0", arm)
    register_counts(e, counts)
    prior_rows = {}
    if resume is not None:
        side = load(c, e, resume, "formal")
        audit = side["audit"]
        remaining = audit["bytes"]
        with Path(audit["path"]).open("rb") as src, (out / "UPDATES.jsonl").open("xb") as dst:
            while remaining:
                block = src.read(min(remaining, 1 << 20))
                require(bool(block), "prior ledger truncated")
                dst.write(block)
                remaining -= len(block)
        with Path(audit["path"]).open("rb") as src:
            src.seek(audit["bytes"])
            for line in src:
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                prior_rows[row["input"]["update"]] = {k: v for k, v in row.items() if k not in ("update_seconds", "monitor")}
        prior_audit = Path(audit["path"]).parent / "AUDIT"
        (out / "AUDIT").mkdir()
        done_epoch = e.completed_updates // WINDOWS_PER_EPOCH
        for f in sorted(prior_audit.glob("epoch-*.jsonl")):
            if int(f.stem.split("-")[1]) <= done_epoch:
                shutil.copy2(f, out / "AUDIT" / f.name)
        pe = Path(audit["path"]).parent / "AUDIT_EPOCHS.jsonl"
        if pe.exists():
            with pe.open() as src, (out / "AUDIT_EPOCHS.jsonl").open("x") as dst:
                for line in src:
                    if json.loads(line)["epoch"] <= done_epoch:
                        dst.write(line)
        write_new(out / "RESTORE.json", {"resumed_from": side, "prior_rows_after_state": len(prior_rows)})
    else:
        save(c, e, out / "checkpoint-epoch-060.pth", "formal", None, "formal_save")
    loop = run_loop(c, e, out, ENDPOINT, "formal", counts, save_points=set(plan["save_points"]) - {18240},
                    monitor_points=(), prior_rows=prior_rows)
    loop.pop("training_image_ids")
    return {"arm": arm, "plan": plan, "install": {k: v for k, v in info.items() if k != "new_parameter_names"},
            "resumed": resume is not None, **loop}


# ================================================================ evaluation (section 9)
def evaluate(c, out, arm, counts):
    from infer_seed import bridge_values, load_original, pixels, _group, _prefix
    from sparse_rtdetr.research import r27_local_memory as R
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.r0.evaluation import rank_candidates, score_budget
    from s15_formal_eval import per_class_large
    torch = torch_mod()
    out = Path(out)
    path = ROUND / f"formal-{arm}" / "checkpoint-epoch-120.pth"
    side, st = read_state(c, path, "formal", arm)
    require(side["global_update"] == ENDPOINT and st["engine"]["completed_updates"] == st["ema"]["updates"] == ENDPOINT, "e120 clocks")
    state = st["ema"]["module"]
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA not FP32")
    del st
    device = torch.device("cuda:0")
    model = R.build_eval_model(arm, state, device)
    weights = tensor_digest(state.items())
    model.register_forward_pre_hook(lambda *_: counts.update(forwards=1))
    data = S.development(c["data"]["development"], S.S15_HW)
    require([r["stable_image_id"] for r in data.records] == c["development_ids"], "development members/order")
    rng = ck.capture_rng(device)
    ctx = model.r27_ctx
    preds = {b: [] for b in BUDGETS}
    timings = []
    started = time.perf_counter()

    def batch(rows):
        t0 = time.perf_counter()
        imgs = []
        for row in rows:
            with load_original(data, row) as image:
                imgs.append(pixels(data, image))
        x = torch.stack(imgs).to(device)
        ctx.route_keys.extend((r["stable_image_id"], "eval", r["relative_path"], 0, 0) for r in rows)
        with torch.autocast("cuda", enabled=False), forward_guards(model):
            o = model(x)
        require(not ctx.route_keys and o["pred_boxes"].shape == (len(rows), 300, 4) and o["pred_logits"].dtype == torch.float32, "eval output")
        boxes, logits = o["pred_boxes"].cpu(), o["pred_logits"].cpu()
        six = [rank_candidates(lg, bx, row, 600) for lg, bx, row in zip(logits, boxes, rows)]
        timings.append(time.perf_counter() - t0)
        return {600: [p for g in six for p in g], 300: [p for g in six for p in g[:300]]}

    with torch.inference_mode():
        for i in range(EVAL_WARMUP):
            batch(data.records[i * EVAL_BATCH:(i + 1) * EVAL_BATCH])
        for off in range(0, len(data.records), EVAL_BATCH):
            r = batch(data.records[off:off + EVAL_BATCH])
            for b in BUDGETS:
                preds[b].extend(r[b])
    require(counts["forwards"] == EVAL_WARMUP + EVAL_BATCHES, "evaluation forward count")
    require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng), "evaluation changed model/RNG")
    infer_seconds = time.perf_counter() - started
    del model
    gc.collect(); torch.cuda.empty_cache()
    ids = c["development_ids"]
    refs = {}
    for b in BUDGETS:
        p = out / f"top{b}-predictions.json"
        write_new(p, preds[b])
        refs[str(b)] = ref(p)
    grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in BUDGETS}
    _prefix(grouped[600], grouped[300], ids, whole=True)
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    values, metrics_refs, counts_gt, scoring = {}, {}, {}, []
    per_class = None
    for b in BUDGETS:
        t = time.perf_counter()
        metrics = score_budget(rows, parsed, preds[b], c["source_root"])
        counts.update(score_budget_calls=1)
        metrics["R27_binding"] = {"contract_sha256": c["contract_sha256"], "arm": arm, "kind": "EMA", "view": "A", "epoch": 120,
                                  "budget": b, "checkpoint": side["file"], "weights_sha256": weights, "predictions": refs[str(b)]}
        values[str(b)] = bridge_values(metrics)
        counts_gt[str(b)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
        p = out / f"R27-{arm}-EMA-A-top{b}-metrics.json"
        write_new(p, metrics)
        metrics_refs[str(b)] = ref(p)
        scoring.append({"budget": b, "seconds": time.perf_counter() - t})
        if b == 600:
            per_class = per_class_large(rows, parsed, preds[b], metrics["COCO_bridge"])
            counts.update(per_class_COCOeval_calls=1)
    s15_counts = {str(b): {k: read_json(c["S15_metrics"][str(b)]["path"])["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
                  for b in BUDGETS}
    require(counts_gt == s15_counts, f"scored GT set differs from S15: {counts_gt} vs {s15_counts}")
    return {"arm": arm, "endpoint": side["file"], "weights_sha256": weights, "values": values, "metrics": metrics_refs,
            "predictions": refs, "GT_counts": counts_gt, "per_class_APL_top600": per_class, "scoring": scoring,
            "inference_seconds": infer_seconds, "batch_seconds_median": statistics.median(timings)}


# ================================================================ summary (section 10)
def cells(minuend, subtrahend, rule):
    out = {}
    for b, m, minimum in rule:
        d = 100. * (minuend[b][m] - subtrahend[b][m])
        out[f"top{b}_{m}"] = {"difference_pp": d, "minimum_pp": minimum, "passed": d >= minimum,
                              "minuend": minuend[b][m], "subtrahend": subtrahend[b][m]}
    return out


def summarize(c, out):
    ev = {a: read_json(ROUND / f"eval-{a}" / "CLOSED.json") for a in LEARN}
    fc = read_json(ROUND / "final-cost" / "CLOSED.json")
    g = read_json(ROUND / "gate" / "GATE.json")
    vals = {a: ev[a]["values"] for a in LEARN}
    s15, a0 = c["S15_EMA_values"], c["A0_EMA_values"]
    E = cells(vals["P"], s15, GATE_E)
    I = cells(vals["P"], vals["C"], GATE_I)
    G = cells(vals["P"], a0, GATE_G)
    fr = fc["ratios"]["P"]
    e_cost = {"model_gpu_ratio": fr["model_gpu_ratio_vs_N"], "end_to_end_ratio": fr["end_to_end_ratio_vs_N"],
              "model_gpu_passed": fr["model_gpu_ratio_vs_N"] <= COST_GATE["model_gpu_ratio_max"],
              "end_to_end_passed": fr["end_to_end_ratio_vs_N"] <= COST_GATE["end_to_end_ratio_max"]}
    e_pass = all(v["passed"] for v in E.values()) and e_cost["model_gpu_passed"] and e_cost["end_to_end_passed"]
    i_pass = all(v["passed"] for v in I.values())
    g_pass = all(v["passed"] for v in G.values())
    if e_pass and i_pass:
        label = "single_seed_efficiency_gain + routing increment preliminarily established (single seed)"
    elif e_pass:
        label = "single_seed_efficiency_gain (structure efficiency only; routing explanation not established)"
    else:
        label = "E_failed: fixed R27 configuration closed"
    from itertools import chain
    metrics = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
    full = {name: {b: {m: 100. * (x[b][m] - y[b][m]) for m in metrics} for b in ("600", "300")}
            for name, x, y in (("P-S15", vals["P"], s15), ("C-S15", vals["C"], s15), ("P-C", vals["P"], vals["C"]),
                               ("P-A0", vals["P"], a0), ("C-A0", vals["C"], a0))}
    pc = c["per_class_APL_top600"]
    per_class = {"P": ev["P"]["per_class_APL_top600"], "C": ev["C"]["per_class_APL_top600"], "S15": pc["S15"], "A0": pc["A0"]}
    res = {"status": "PASS", "contract_sha256": c["contract_sha256"], "gate": g["decision"],
           "values": {"P": vals["P"], "C": vals["C"], "S15": s15, "A0": a0},
           "E_P_minus_S15": {"cells": E, "cost": e_cost, "passed": e_pass},
           "I_P_minus_C": {"cells": I, "passed": i_pass},
           "G_P_minus_A0": {"cells": G, "passed": g_pass,
                            "APL_cost_remains": not G["top600_APL"]["passed"]},
           "label": label, "differences_pp": full, "per_class_APL_top600": per_class,
           "final_cost": {"ratios": fc["ratios"], "medians": fc["medians"]},
           "pre_formal_cost": g["cost"]}
    write_new(Path(out) / "RESULT.json", res)
    return res


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "cpu", "accept", "diag", "cost", "gate", "formal", "evaluate", "final-cost", "summarize"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    if args.action == "prepare":
        return prepare(args)
    c = load_contract()
    args.out.mkdir(parents=True, exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        from s15_formal_train import bind_seed
        bind_seed(0)
        if args.action in ("accept", "diag", "cost", "formal", "evaluate", "final-cost"):
            require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", "allocator conf")
            setup("cuda:0")
            torch_mod().cuda.reset_peak_memory_stats()
        if args.action == "cpu":
            result = cpu(c, args.out)
        elif args.action == "accept":
            result = accept(c, args.out, args.arm, counts)
        elif args.action == "diag":
            result = diag(c, args.out, args.arm, counts)
        elif args.action == "cost":
            result = cost(c, args.out, counts)
        elif args.action == "gate":
            result = gate(c, args.out)
        elif args.action == "formal":
            result = formal(c, args.out, args.arm, counts, args.resume)
        elif args.action == "evaluate":
            result = evaluate(c, args.out, args.arm, counts)
        elif args.action == "final-cost":
            result = final_cost(c, args.out, counts)
        else:
            result = summarize(c, args.out)
        write_new(args.out / "CLOSED.json", {"status": "PASS", "action": args.action, "arm": args.arm,
                                             "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                             "elapsed_seconds": time.monotonic() - start,
                                             "process_cpu_seconds": time.process_time() - cpu0, **result})
    except BaseException as exc:
        write_new(args.out / "STOP.json", {"status": "STOP", "phase": PHASE["phase"], "action": args.action, "arm": args.arm,
                                           "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                           "counts": dict(counts), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
