"""TASK_R26_001 runner: constrained detail encoder block (N/U/P) on the seed0 S15 H recipe.

actions
  prepare   no torch model.  Writes the round CONTRACT (parent, data, monitor64 ids, workload, source release).
  cpu       CUDA hidden.  Synthetic arithmetic (pytest), real-model native/zero-step identities, optimizer/EMA/
            deepcopy coverage, state save/exact restore and refusals, analytic+counted MACs, formal-entry plan check.
  accept    CUDA, one arm.  u1,u2 from e060, save u1, fresh engine restore + replay u2 (exact).  3 updates.
  diag      CUDA, one arm.  Observation u0, 128 effective updates (u18241-u18368), observation u128.
  cost      CUDA.  Three u128 online endpoints, FP32 batch4 on monitor64: 2 warmup/arm, 3 rotated passes, staged events,
            one uninstrumented batch/arm checked equal.
  formal    CUDA, U or P.  e061-e120 formal entry (18240 updates/arm).  Refuses to start without --authorized-formal;
            --plan-only validates the plan on CPU and exits.  Never continues a diagnostic/acceptance state.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import argparse
import gc
import hashlib
import io
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"), str(ROOT)]

WS = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace")
LD_ROOT = WS / "experiments/localization-distribution"
ROUND = LD_ROOT / "r26-constrained-detail-encoder-20261109-001"
PARENT = LD_ROOT / ("rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/"
                    "checkpoint-epoch-060.pth")
PARENT_SHA = "a2b6412e3c85de7592829f24aff09e355fdd11cb53798bc498bd04b882e1bf8a"
S15_CONTRACT = LD_ROOT / "s15-scale-baseline-20261105-001/formal-002-CONTRACT.json"
S15_CONTRACT_SHA = None  # pinned at prepare time (file sha recorded in the contract)
S15_UPDATES = LD_ROOT / "s15-scale-baseline-20261105-001/formal-002/train-S15/UPDATES.jsonl"
SQC_SPLIT = LD_ROOT / "shared-query-classification-20261109-001/sqc-001-SPLIT.json"
PYTHON = "/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python"
TASK = (Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace") / "experiments/localization-distribution"
        / "r26-constrained-detail-encoder-20261109-001/TASK_R26_001.md")
TASK_SHA = "ea591105d35c8c25f554c94d2fb250b53b3cbbb2699b33ec5809b309024e1311"
SCHEMA = "r26-state-v1"
ARMS = ("N", "U", "P")
DIAG_UPDATES = 128
MONITOR_AT = (1, 2, 16, 64, 128)
OBS_IMAGES, OBS_BATCH = 64, 8
COST_BATCH, COST_WARMUP, COST_PASSES = 4, 2, 3
ROTATIONS = (("N", "U", "P"), ("U", "P", "N"), ("P", "N", "U"))
FORMAL_SAVE_EPOCHS = (60, 75, 90, 105, 120)
WORKLOAD = {"diagnostic_updates_per_arm": DIAG_UPDATES, "acceptance_updates_per_arm": 3,
            "training_forwards_per_update": 2, "observation_batches_per_arm": 2 * OBS_IMAGES // OBS_BATCH,
            "cost_warmup_batches_per_arm": COST_WARMUP, "cost_timed_batches_per_arm": COST_PASSES * OBS_IMAGES // COST_BATCH,
            "cost_uninstrumented_batches_per_arm": 1, "monitor_updates": list(MONITOR_AT),
            "normal_total": {"optimizer_updates": 393, "training_forwards": 786, "observation_forwards": 48,
                             "timed_forwards": 144, "warmup_forwards": 6, "uninstrumented_forwards": 3,
                             "full_forwards": 987}}
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
        f.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


# ================================================================ contract
def monitor_ids(validation):
    return sorted(validation, key=lambda i: (hashlib.sha256(("r26-monitor-v1:" + i).encode()).hexdigest(), i))[:OBS_IMAGES]


def source_manifest(root):
    files = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            files[str(p.relative_to(root))] = file_sha(p)
    return files


def prepare(args):
    import subprocess
    require(ROOT.resolve().parent.parent == ROUND.resolve() and ROOT.name == "source", "prepare must run from a frozen release")
    manifest = json.loads((ROOT.parent / "MANIFEST.json").read_text())
    require(manifest["files"] == source_manifest(ROOT), "release source differs from MANIFEST")
    s15 = json.loads(S15_CONTRACT.read_text())
    split = json.loads(SQC_SPLIT.read_text())
    require(len(split["validation"]) == 512, "SQC validation size")
    ids = monitor_ids(split["validation"])
    task = {"path": str(TASK), **_tsize()}
    pinned(task)
    c = {"version": "R26-001-contract-v1", "task": task,
         "parent": ref(PARENT), "parent_update": 18240,
         "S15_contract": ref(S15_CONTRACT), "S15_formal_identity": s15["formal_identity"],
         "S15_updates": ref(S15_UPDATES), "data": {"train_core": s15["data"]["train_core"]},
         "environment": s15["environment"], "python": PYTHON, "gpu_uuid": s15["gpu_uuid"],
         "SQC_split": ref(SQC_SPLIT), "monitor_rule": "SHA256('r26-monitor-v1:'+image_id) hex ascending, first 64 of the SQC 512 validation",
         "monitor_ids": ids, "monitor_sha256": sha_bytes(canonical(ids)),
         "source_root": str(ROOT.resolve()), "source_manifest": ref(ROOT.parent / "MANIFEST.json"),
         "source_commit": manifest["commit"], "workload": WORKLOAD,
         "recipe": {"input_hw": [1152, 2016], "train_batch": 8, "accumulation": 2, "precision": "bfloat16", "clip": 0.1,
                    "optimizer": "native AdamW groups, global LR clock", "activation_checkpoint": True,
                    "allocator_conf": "expandable_segments:True"},
         "outputs": str(ROUND), "formal_started": False}
    require(c["parent"]["sha256"] == PARENT_SHA, "parent SHA")
    c["contract_sha256"] = sha_bytes(canonical(c))
    write_new(ROUND / "CONTRACT.json", c)
    print(json.dumps({"contract_sha256": c["contract_sha256"], "monitor_sha256": c["monitor_sha256"]}))


def _tsize():
    return {"bytes": TASK.stat().st_size, "sha256": TASK_SHA}


def load_contract(path=ROUND / "CONTRACT.json"):
    c = json.loads(Path(path).read_text())
    body = {k: v for k, v in c.items() if k != "contract_sha256"}
    require(c["contract_sha256"] == sha_bytes(canonical(body)), "contract digest")
    require(Path(c["source_root"]).resolve() == ROOT.resolve(), "not running from the bound release")
    pinned(c["source_manifest"])
    pinned(c["parent"])
    return c


# ================================================================ engine/state
def torch_mod():
    import torch
    return torch


def make(c, device, mode, *, fixture=False):
    from sparse_rtdetr.research import r26_detail as R
    from sparse_rtdetr.protocol_r18.model import Config
    from sparse_rtdetr.research import scale_baseline as S
    PHASE["phase"] = f"{mode}:construct"
    cfg = Config(seed=0, input_hw=S.S15_HW, physical_batch_size=1, amp_dtype="float32") if fixture else None
    e, info = R.create(c["parent"]["path"], device, mode, config=cfg, fixture=fixture, activation_checkpoint=True)
    e.r26_mode = mode
    return e, info


def binding(c, e, scope, audit=None):
    from formal_data import cursor
    from sparse_rtdetr.research import r26_detail as R
    t = getattr(R._block(e.model), R.DETAIL_NAME, None)
    return {"schema": SCHEMA, "scope": scope, "mode": e.r26_mode, "contract_sha256": c["contract_sha256"],
            "parent_sha256": c["parent"]["sha256"], "global_update": e.completed_updates,
            "cursor": cursor(e.completed_updates), "config": e.config.binding(), "audit": audit,
            "activation_checkpoint": e.s15_activation_checkpoint, "T_init_sha256": getattr(e, "r26_T_init", None),
            "torch": str(torch_mod().__version__)}


def save(c, e, path, scope, audit=None):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import assert_healthy
    torch = torch_mod()
    path = Path(path)
    require(e.ema.updates == e.completed_updates, "EMA clock at save")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    b = binding(c, e, scope, audit)
    state["r26_binding_sha256"] = sha_bytes(canonical(b))
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as f:
        torch.save(state, f); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path); tmp.unlink()
    side = {"status": "COMPLETE", **b, "file": ref(path)}
    write_new(path.with_suffix(".pth.json"), side)
    return side


def load(c, e, path, scope):
    """Exact restore into a fresh engine of the same mode; any other scope/mode/contract/clock is refused."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    torch = torch_mod()
    path = Path(path).resolve()
    side = json.loads(path.with_suffix(".pth.json").read_text())
    require(side["scope"] == scope and side["mode"] == e.r26_mode and side["contract_sha256"] == c["contract_sha256"]
            and side["schema"] == SCHEMA and side["parent_sha256"] == c["parent"]["sha256"], "R26 state scope/mode/contract")
    require(side["file"] == ref(path), "R26 state file identity")
    require(side["T_init_sha256"] == getattr(e, "r26_T_init", None) and side["config"] == e.config.binding(),
            "R26 state T init / config")
    s = torch.load(path, map_location="cpu", weights_only=True)
    expect = {k: v for k, v in side.items() if k not in ("status", "file")}
    require(s.pop("r26_binding_sha256") == sha_bytes(canonical(expect)), "R26 binding digest")
    require(s["engine"]["completed_updates"] == side["global_update"] == s["ema"]["updates"], "R26 state clock")
    require(s["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "R26 optimizer names")
    e.model.load_state_dict(s["model"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.ema.load_state_dict(s["ema"])
    modes = dict(e.model.named_modules())
    require(set(modes) == set(s["model_modes"]), "R26 mode keys")
    for n, m in s["model_modes"].items():
        modes[n].training = m
    e.load_state_dict(s["engine"])
    ck.restore_rng(s["rng"], e.device)
    require(ck.exact_equal(S.capture(e), s), "R26 full state did not restore exactly")
    return side


# ================================================================ observers
def register_counts(e, counts):
    e.model.register_forward_pre_hook(lambda *_: counts.update(forward_attempts=1))
    e.model.register_forward_hook(lambda *_: counts.update(forwards=1))
    e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))


def groups_of(name):
    from sparse_rtdetr.research import r26_detail as R
    if f"{R.BLOCK_PATH}.{R.DETAIL_NAME}." in name:
        return ["T", "T." + name.split(f"{R.DETAIL_NAME}.")[1].split(".")[0]]
    if name.startswith(R.BLOCK_PATH + ".bottlenecks."):
        return ["block_replaced", "block_bottlenecks"]
    if name.startswith(R.BLOCK_PATH + "."):
        return ["block_replaced", "block_" + name[len(R.BLOCK_PATH) + 1:].split(".")[0]]
    for prefix, g in (("backbone.", "backbone"), ("encoder.", "encoder_other"), ("decoder.", "decoder")):
        if name.startswith(prefix):
            return [g]
    return ["other"]


class StepMonitor:
    """Gradient (post-clip, pre-step) and update norms by group at registered updates; no RNG, no extra forward."""

    def __init__(self, e):
        self.e, self.active, self.rows, self.before = e, False, {}, {}
        e.optimizer.register_step_pre_hook(self.pre)
        e.optimizer.register_step_post_hook(self.post)

    def pre(self, *_):
        if not self.active:
            return
        torch = torch_mod()
        sq = Counter()
        self.before = {}
        for n, p in self.e.model.named_parameters():
            if p.grad is not None:
                g = float(p.grad.detach().float().square().sum())
                for k in groups_of(n):
                    sq[k] += g
            if groups_of(n)[0] in ("T", "block_replaced"):
                self.before[n] = p.detach().float().clone()
        self.grad = {k: math.sqrt(v) for k, v in sq.items()}

    def post(self, *_):
        if not self.active:
            return
        upd, ref_ = Counter(), Counter()
        for n, p in self.e.model.named_parameters():
            if n in self.before:
                d = float((p.detach().float() - self.before[n]).square().sum())
                for k in groups_of(n):
                    upd[k] += d
                    ref_[k] += float(self.before[n].square().sum())
        self.row = {"grad_norm_post_clip": self.grad,
                    "update_norm": {k: math.sqrt(v) for k, v in upd.items()},
                    "param_norm_before": {k: math.sqrt(v) for k, v in ref_.items()}}
        self.before = {}


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
    """Eval/no_grad FP32 final-output supervision readings on monitor64; modes, BN, EMA, RNG untouched."""
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.baseline.training_v2b_engine import _move
    from sparse_rtdetr.research import r26_detail as R
    from sparse_rtdetr.research import shared_query_cls as Q
    import torchvision.ops as ops
    torch = torch_mod()
    _, pixels, targets = monitor
    rng = ck.capture_rng(e.device)
    modes = {n: m.training for n, m in e.model.named_modules()}
    digest = Q.digest_tensors(e.model.state_dict().items())
    stats = []
    R.set_monitor(e.model, stats)
    e.model.eval()
    batches, pairs = [], []
    try:
        with torch.no_grad(), torch.autocast(e.device.type, enabled=False), forward_guards(e.model):
            for k in range(0, OBS_IMAGES, OBS_BATCH):
                x = torch.stack(pixels[k:k + OBS_BATCH]).to(e.device).float() / 255.
                tg = _move(targets[k:k + OBS_BATCH], e.device)
                out = e.model(x)
                require(set(out) == {"pred_logits", "pred_boxes"}, f"eval output keys {sorted(out)}")
                losses = e.criterion({"pred_logits": out["pred_logits"].float(), "pred_boxes": out["pred_boxes"].float()},
                                     [{k2: v for k2, v in t.items() if k2 != "image_id"} for t in tg])
                diag = e.criterion.last_diagnostics
                require(len(diag) == 1 and diag[0]["head"] == "final", "final-only criterion call")
                vals = {k2: float(v) for k2, v in losses.items()}
                batches.append({"images": [t["image_id"] for t in tg], **vals,
                                "final_eval_loss": sum(vals.values()), "GT": sum(len(t["labels"]) for t in tg),
                                "denominator": diag[0]["denominator"]})
                for i, t in enumerate(tg):
                    q = diag[0]["matched_queries"][i]
                    gid = diag[0]["matched_GT_ids"][i]
                    if not q:
                        continue
                    pos = {g: j for j, g in enumerate(t["gt_ids"])}
                    gi = torch.tensor([pos[g] for g in gid], device=e.device)
                    pb = out["pred_boxes"][i, q].float()
                    gb = t["boxes"][gi].float()
                    pxy, gxy = ops.box_convert(pb, "cxcywh", "xyxy"), ops.box_convert(gb, "cxcywh", "xyxy")
                    iou = ops.box_iou(pxy, gxy).diag()
                    giou = ops.generalized_box_iou(pxy, gxy).diag()
                    l1 = (pb - gb).abs().sum(1)
                    for j, g in enumerate(gid):
                        pairs.append({"image": t["image_id"], "gt": g, "query": q[j],
                                      "bucket": Q.size_bucket(float(t["raw_area"][pos[g]])),
                                      "l1": float(l1[j]), "giou": float(giou[j]), "iou": float(iou[j])})
                del out, losses
    finally:
        R.set_monitor(e.model, None)
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
        s["pairs"] += 1; s["l1"] += p["l1"]; s["giou"] += p["giou"]; s["iou"] += p["iou"]
    summary["by_size"] = {k: {"pairs": v["pairs"], "l1_mean": v["l1"] / v["pairs"], "giou_mean": v["giou"] / v["pairs"],
                              "iou_mean": v["iou"] / v["pairs"]} for k, v in sorted(by.items())}
    summary["block_stats"] = stats
    return {"summary": summary, "batches": batches, "pairs": pairs}


# ================================================================ training loop (diag and formal share it)
def run_loop(c, e, out, stop, scope, counts, *, save_points, monitor_points=(), ledger_rows=None):
    from formal_data import windows, WINDOWS_PER_EPOCH
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import r26_detail as R
    out = Path(out)
    history = out / "UPDATES.jsonl"
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    mon = StepMonitor(e)
    start = e.completed_updates
    rec0 = S.RECOMPUTE["calls"]
    image_ids = []
    monitored = {}
    saves = []
    with history.open("a", buffering=1) as ledger:
        while e.completed_updates < stop:
            begin = e.completed_updates
            for x, t, inp in windows(data, begin, limit=min(stop - begin, WINDOWS_PER_EPOCH - begin % WINDOWS_PER_EPOCH)):
                rel = e.completed_updates + 1 - start
                watch = rel in monitor_points
                stats = [] if watch else None
                R.set_monitor(e.model, stats)
                mon.active = watch
                PHASE["phase"] = f"{e.r26_mode}:update {e.completed_updates + 1}"
                res = e.train_window(x, t)
                R.set_monitor(e.model, None)
                mon.active = False
                require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates,
                        "R26 update clock/input")
                image_ids.extend(tt["image_id"] for tt in t)
                row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": res["loss"],
                       "lr": res["lr"], "gradient_norm_before_clip": res["gradient_norm_before_clip"],
                       "microbatch_losses": res["microbatch_losses"], "update_seconds": res["elapsed_seconds"],
                       "ema_updates": e.ema.updates}
                if watch:
                    factor = min(1., e.config.clip_max_norm / (res["gradient_norm_before_clip"] + 1e-6))
                    m = dict(mon.row)
                    m["grad_norm_pre_clip"] = {k: v / factor for k, v in m["grad_norm_post_clip"].items()}
                    # forward calls: micro0 fwd, micro0 recompute, micro1 fwd, micro1 recompute
                    m["block_forward_stats"] = stats
                    if e.r26_mode != "N":
                        require(len(stats) == 4, f"block stats count {len(stats)}")
                    monitored[rel] = m
                    row["monitor"] = m
                if ledger_rows is not None:
                    ledger_rows.append(row)
                ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                if e.completed_updates in save_points:
                    ledger.flush(); os.fsync(ledger.fileno())
                    name = f"state-update-{e.completed_updates}.pth" if scope != "formal" else \
                        f"checkpoint-epoch-{e.completed_updates // WINDOWS_PER_EPOCH:03d}.pth"
                    saves.append(save(c, e, out / name, scope, ref(history)))
                if e.completed_updates % 16 == 0:
                    print(json.dumps({"mode": e.r26_mode, "update": e.completed_updates, "loss": res["loss"],
                                      "seconds": res["elapsed_seconds"]}), flush=True)
    done = e.completed_updates - start
    rec = S.RECOMPUTE["calls"] - rec0
    require(e.completed_updates == stop and counts["actual_updates"] >= done, "R26 loop count")
    require(rec == 16 * done, f"R26 block recomputation count {rec}")
    return {"updates": done, "block_recomputations": rec, "monitored": monitored, "saves": saves,
            "training_image_ids": image_ids}


# ================================================================ CPU checks
def cpu(c, out):
    import subprocess
    torch = torch_mod()
    from r25_run import setup
    from sparse_rtdetr.research import r26_detail as R
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.baseline.training_v2b_engine import _move
    from formal_data import windows
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU checks hide CUDA")
    setup("cpu")
    torch.set_num_threads(int(os.environ.get("R26_CPU_THREADS", "16")))
    t0 = time.perf_counter()
    res = {}
    # 1. synthetic arithmetic (pytest, separate process; no data/checkpoints)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests/test_r26_detail.py")],
                       capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-3:]}
    require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-4000:])
    # 2. real model (fixture FP32 bs1x2 on the S15 canvas, real e060 weights, real first window images)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    (x, t, inp), = list(windows(data, 18240, limit=1))
    x, t = x[:2], t[:2]
    engines = {}
    info = {}
    for m in ARMS:
        engines[m], info[m] = make(c, "cpu", m, fixture=True)
        engines[m].r26_T_init = info[m]["R26"]["T_init_sha256"]
    res["install"] = info
    require(info["U"]["R26"]["T_init_sha256"] == info["P"]["R26"]["T_init_sha256"]
            and info["U"]["R26"]["new_parameters"] == 12576 == R.detail_parameters(128), "U/P T init / count")
    # native mode N must be bitwise the native module
    from sparse_rtdetr.research.scale_baseline import create as s15_create
    native, _ = s15_create(c["parent"]["path"], "cpu", engines["N"].config, seed=0, fixture=True, activation_checkpoint=True)

    def fwd(e, train=True):
        rng = ck.capture_rng(e.device)
        modes = {n: mm.training for n, mm in e.model.named_modules()}
        e.model.train(train)
        with torch.no_grad():
            o = e.model(x[:1], targets=_move(t[:1], e.device)) if train else e.model(x[:1])
        after = ck.capture_rng(e.device)
        ck.restore_rng(rng, e.device)
        for n, mm in e.model.named_modules():
            mm.training = modes[n]
        return o, after

    bn0 = {n: b.clone() for n, b in engines["N"].model.state_dict().items() if "running" in n or "num_batches" in n}
    oN, rN = fwd(engines["N"], train=False)
    oS, rS = fwd(native, train=False)
    require(all(torch.equal(oN[k], oS[k]) for k in ("pred_logits", "pred_boxes")), "N mode differs from native (eval)")
    oU, rU = fwd(engines["U"], train=False)
    oP, rP = fwd(engines["P"], train=False)
    zero = {"N_equals_native_eval": True,
            "U_equals_P_eval": all(torch.equal(oU[k], oP[k]) for k in ("pred_logits", "pred_boxes")),
            "U_minus_N_eval_max_abs_logits": float((oU["pred_logits"] - oN["pred_logits"]).abs().max()),
            "U_minus_N_eval_max_abs_boxes": float((oU["pred_boxes"] - oN["pred_boxes"]).abs().max())}
    require(zero["U_equals_P_eval"], "U/P zero-step eval output differs")
    del native
    gc.collect()
    # zero-step training losses and RNG: one real update each on U and P, compare loss/output digests/RNG after
    rows = {}
    rng_parent = ck.capture_rng(engines["P"].device)   # every engine restored the same parent RNG at construction
    for m in ("U", "P"):
        e = engines[m]
        ck.restore_rng(rng_parent, e.device)            # engines share the process RNG: each arm starts from the parent RNG
        r = e.train_window(x, t, capture=True)
        rows[m] = {"loss": r["loss"], "output_sha256": r["output_sha256"], "dn": r["dn_noise_sha256"],
                   "rng_after": ck.capture_rng(e.device), "grad_sha": r["gradient_sha256_before_clip"],
                   "T_grads_nonzero": None}
    zero["U_P_first_update_outputs_equal"] = rows["U"]["output_sha256"] == rows["P"]["output_sha256"]
    zero["U_P_first_update_DN_equal"] = rows["U"]["dn"] == rows["P"]["dn"]
    zero["U_P_first_update_loss"] = [rows["U"]["loss"], rows["P"]["loss"]]
    zero["U_P_RNG_after_first_update_equal"] = ck.exact_equal(rows["U"]["rng_after"], rows["P"]["rng_after"])
    zero["U_P_gradients_equal"] = rows["U"]["grad_sha"] == rows["P"]["grad_sha"]
    require(zero["U_P_first_update_outputs_equal"] and zero["U_P_first_update_DN_equal"]
            and rows["U"]["loss"] == rows["P"]["loss"] and zero["U_P_RNG_after_first_update_equal"],
            "U/P zero-step forward/loss/RNG differ")
    T = {m: R._block(engines[m].model).r26_detail for m in ("U", "P")}
    zero["T_after_one_update_changed"] = {m: [bool((T[m][i].weight != 0).any()) for i in (0, 2, 4)] for m in ("U", "P")}
    zero["U_P_T_last_layer_differ_after_one_update"] = not torch.equal(T["U"][4].weight, T["P"][4].weight)
    res["zero_step"] = zero
    # BN updates once per forward under checkpointing (2 forwards per update)
    bn1 = {n: int(b) for n, b in engines["U"].model.state_dict().items() if n.endswith("num_batches_tracked")}
    res["BN_num_batches_tracked_delta_after_one_update"] = sorted(set(v - int(bn0[n]) for n, v in bn1.items()))
    require(res["BN_num_batches_tracked_delta_after_one_update"] == [2], "BN tracked once per forward")
    # optimizer / EMA / deepcopy / mode identity
    import copy
    e = engines["P"]
    names = [n for g in e.optimizer.param_groups for n in g["param_names"]]
    res["optimizer"] = {"groups": {g["name"]: len(g["params"]) for g in e.optimizer.param_groups},
                        "covers_all_trainable": sorted(names) == sorted(n for n, p in e.model.named_parameters() if p.requires_grad),
                        "T_adam_step_after_one_update": [int(e.optimizer.state[T["P"][i].weight]["step"]) for i in (0, 2, 4)]}
    require(res["optimizer"]["covers_all_trainable"], "optimizer coverage")
    emaT = R._block(e.ema.module).r26_detail
    res["EMA"] = {"has_T": True, "updates": e.ema.updates, "T_requires_grad": any(p.requires_grad for p in emaT.parameters()),
                  "mode": R.mode_of(e.ema.module), "T_moved_toward_online": not torch.equal(emaT[4].weight, torch.zeros_like(emaT[4].weight))}
    dc = copy.deepcopy(e.ema.module)
    res["deepcopy_mode"] = R.mode_of(dc)
    require(R.mode_of(dc) == "P" and R.mode_of(e.ema.module) == "P", "deepcopy/EMA mode identity")
    # state save / exact restore / refusals
    st = Path(out) / "cpu-state-P.pth"
    side = save(c, e, st, "cpu-fixture")
    fresh, _ = make(c, "cpu", "P", fixture=True)
    fresh.r26_T_init = info["P"]["R26"]["T_init_sha256"]
    load(c, fresh, st, "cpu-fixture")
    refused = []
    for label, m, scope in (("P state into U engine", "U", "cpu-fixture"), ("fixture state in diagnostic scope", "P", "diagnostic"),
                            ("fixture state in formal scope", "P", "formal"), ("P state into N engine", "N", "cpu-fixture")):
        other, inf = make(c, "cpu", m, fixture=True)
        other.r26_T_init = inf["R26"]["T_init_sha256"]
        try:
            load(c, other, st, scope)
        except (ValueError, RuntimeError, KeyError) as exc:
            refused.append({"case": label, "reason": repr(exc)[:200]})
        else:
            raise AssertionError("accepted: " + label)
        del other
        gc.collect()
    try:
        R.create(LD_ROOT / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/checkpoint-epoch-075.pth"
                 if (LD_ROOT / "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/epoch-recovery-20261014-001/formal/checkpoint-epoch-075.pth").exists()
                 else str(S15_UPDATES), "cpu", "P", config=fresh.config, fixture=True)
    except (ValueError, RuntimeError, KeyError, Exception) as exc:
        refused.append({"case": "non-e060 parent", "reason": repr(exc)[:200]})
    else:
        raise AssertionError("accepted a non-e060 parent")
    try:
        formal_guard(c, "P", Path(out) / "cpu-state-P.pth", plan_only=True)
    except ValueError as exc:
        refused.append({"case": "formal entry resuming a non-formal state", "reason": repr(exc)[:200]})
    else:
        raise AssertionError("formal entry accepted a non-formal state")
    res["refused"] = refused
    res["state"] = side
    # 3. MACs: analytic block and counted whole-model (one FP32 CPU forward per arm at 1152x2016, batch1)
    from torch.utils.flop_counter import FlopCounterMode
    macs = {}
    for m in ARMS:
        e = engines[m]
        e.model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            e.model(x[:1])
        total = fc.get_total_flops() // 2
        counts_ = fc.get_flop_counts()
        key = [k for k in counts_ if k.endswith("encoder.fpn_blocks.1")]
        require(len(key) == 1, f"flop counter block key {key}")
        blk = sum(counts_[key[0]].values()) // 2      # this entry already includes its submodules
        macs[m] = {"model_MACs_counted": int(total), "block_MACs_counted": blk,
                   "block_MACs_analytic": R.macs_block(144, 252, m)}
    res["MACs"] = macs
    res["parameters"] = {m: sum(p.numel() for p in engines[m].model.parameters()) for m in ARMS}
    res["formal_plan"] = formal_guard(c, "P", None, plan_only=True)
    res["cpu_seconds_wall"] = time.perf_counter() - t0
    res["cpu_fixture_updates"] = 2
    return res


# ================================================================ CUDA acceptance
def accept(c, out, arm, counts):
    from formal_data import windows
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from s15_formal_train import _differences, _result_body, _bn_counts
    torch = torch_mod()
    out = Path(out)
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    e, info = make(c, "cuda:0", arm)
    e.r26_T_init = info["R26"]["T_init_sha256"]
    register_counts(e, counts)
    bn0 = _bn_counts(e.model)
    rows = []
    for i, (x, t, inp) in enumerate(windows(data, 18240, limit=2), 1):
        r = e.train_window(x, t, capture=True)
        rows.append({"input": inp, "update": r})
        if i == 1:
            saved = save(c, e, out / "u1.pth", "acceptance")
    expected = S.capture(e)
    bn = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    require(bn == [4], f"BN delta {bn}")
    del e
    gc.collect(); torch.cuda.empty_cache()
    clone, _ = make(c, "cuda:0", arm)
    clone.r26_T_init = info["R26"]["T_init_sha256"]
    register_counts(clone, counts)
    load(c, clone, out / "u1.pth", "acceptance")
    ((x, t, inp),) = list(windows(data, 18241, limit=1))
    replay = clone.train_window(x, t, capture=True)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]["update"]), _result_body(replay))
    require(inp == rows[1]["input"] and not diff and ck.exact_equal(expected, got), f"replay differs {diff[:5]}")
    require(counts["actual_updates"] == 3 and counts["forwards"] == 6, "acceptance counts")
    return {"arm": arm, "install": info, "first_updates": [{"input": r["input"], "loss": r["update"]["loss"],
                                                             "grad_norm": r["update"]["gradient_norm_before_clip"],
                                                             "output_sha256": r["update"]["output_sha256"]} for r in rows],
            "saved_u1": saved, "replay_exact": True, "BN_delta_two_updates": bn,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "formal_state_usable": False}


# ================================================================ diagnostic 128
def diag(c, out, arm, counts):
    torch = torch_mod()
    out = Path(out)
    monitor = monitor_data(c)
    e, info = make(c, "cuda:0", arm)
    e.r26_T_init = info["R26"]["T_init_sha256"]
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
    return {"arm": arm, "install": info, "initial_state": s0, "end_state": loop["saves"][-1],
            "updates": loop["updates"], "block_recomputations": loop["block_recomputations"],
            "train_wall_seconds": train_wall, "mean_update_seconds": sum(r["update_seconds"] for r in rows) / len(rows),
            "peak_allocated_bytes_training": peak,
            "monitor64_training_overlap": {"images_in_128_updates": len(ids), "distinct": len(set(ids)),
                                           "monitor_ids_seen": len(overlap), "ids": overlap},
            "observation": {"u0": obs0["summary"], "u128": obs1["summary"]}}


# ================================================================ cost
def cost(c, out, counts):
    from sparse_rtdetr.research import r26_detail as R
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    from sparse_rtdetr.localization.data import RawAugmentation
    import sqc_run as B
    torch = torch_mod()
    dev = torch.device("cuda:0")
    models = {}
    for arm in ARMS:
        e, info = make(c, "cuda:0", arm)
        e.r26_T_init = info["R26"]["T_init_sha256"]
        load(c, e, ROUND / f"diag-{arm}" / "state-update-18368.pth", "diagnostic")
        m = e.model
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[arm] = m
        del e
    gc.collect(); torch.cuda.empty_cache()
    for m in models.values():
        m.register_forward_pre_hook(lambda *_: counts.update(forwards=1))
    data = S.train_dataset(c["data"]["train_core"], hw=S.S15_HW, seed=0)
    rs = {"H": RawAugmentation(Q.H_HW).resize}
    rows = [data.by_id[i] for i in c["monitor_ids"]]

    def stages(m, arm):
        enc = m.encoder
        blk = R._block(m)
        require(len(enc.encoder) == 1 and len(enc.input_proj) == 3, "encoder layout for timing")
        named = {"backbone": (m.backbone, m.backbone), "input_proj": (enc.input_proj[0], enc.input_proj[2]),
                 "AIFI": (enc.encoder[0], enc.encoder[0]), "encoder": (enc, enc),
                 "fpn_block0": (enc.fpn_blocks[0], enc.fpn_blocks[0]), "target_block": (blk, blk),
                 "blk_conv1": (blk.conv1, blk.conv1), "blk_conv2": (blk.conv2, blk.conv2),
                 "blk_B": (blk.bottlenecks, blk.bottlenecks), "blk_conv3": (blk.conv3, blk.conv3),
                 "decoder": (m.decoder, m.decoder), "model": (m, m)}
        if arm != "N":
            named["blk_T"] = (blk.r26_detail, blk.r26_detail)
        return named

    def instrument(m, arm):
        ev, hs = {}, []
        for k, (first, last) in stages(m, arm).items():
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            ev[k] = (a, b)
            hs.append(first.register_forward_pre_hook(lambda *_x, a=a: a.record()))
            hs.append(last.register_forward_hook(lambda *_x, b=b: b.record()))
        return ev, hs

    def batch(arm, idx, events):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        px = [B.load_pixels(data, rows[i], rs)["H"] for i in idx]
        x = torch.stack(px)
        t1 = time.perf_counter()
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        h0.record()
        xg = x.to(dev, non_blocking=False)
        h1.record()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            o = models[arm](xg)
        o0 = torch.cuda.Event(enable_timing=True); o0.record()
        res = (o["pred_logits"].float().sigmoid().cpu(), o["pred_boxes"].float().cpu())
        t2 = time.perf_counter()
        rec = {"load_resize_cpu_s": t1 - t0, "end_to_end_s": t2 - t0, "H2D_ms": h0.elapsed_time(h1)}
        if events is not None:
            rec["stage_ms"] = {k: a.elapsed_time(b) for k, (a, b) in events.items()}
            rec["output_D2H_ms"] = events["model"][1].elapsed_time(o0) if "model" in events else None
        return rec, o

    result = {"warmup": {}, "passes": [], "uninstrumented_check": {}}
    batches = [list(range(k, k + COST_BATCH)) for k in range(0, OBS_IMAGES, COST_BATCH)]
    inst = {arm: instrument(models[arm], arm) for arm in ARMS}
    first_out = {}
    for arm in ARMS:
        result["warmup"][arm] = []
        for i in range(COST_WARMUP):
            rec, o = batch(arm, batches[i], inst[arm][0])
            result["warmup"][arm].append(rec)
            if i == 0:
                first_out[arm] = {k: v.detach().cpu().clone() for k, v in o.items()}
    for p, order in enumerate(ROTATIONS):
        for arm in order:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            recs = [batch(arm, b, inst[arm][0])[0] for b in batches]
            result["passes"].append({"pass": p, "arm": arm, "batches": recs, "allocated_before": base,
                                     "peak_allocated": torch.cuda.max_memory_allocated()})
    # one uninstrumented batch per arm vs the instrumented warmup output on the same images
    for arm in ARMS:
        for h in inst[arm][1]:
            h.remove()
        _, ou = batch(arm, batches[0], None)
        result["uninstrumented_check"][arm] = all(torch.equal(first_out[arm][k], ou[k].cpu()) for k in first_out[arm])
        require(result["uninstrumented_check"][arm], f"instrumented/uninstrumented outputs differ: {arm}")
    require(counts["forwards"] == 3 * (COST_WARMUP + COST_PASSES * len(batches) + 1), f"cost forwards {dict(counts)}")
    result["parameters"] = {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}
    return result


# ================================================================ formal entry (not launched in R26-001)
def formal_guard(c, arm, resume, *, plan_only):
    """Formal U/P e061-e120 from the certified e060 parent only.  Diagnostic/acceptance/fixture states refused."""
    from formal_data import WINDOWS_PER_EPOCH
    require(arm in ("U", "P"), "formal arms are U and P (N reuses S15 formal-002)")
    if resume is not None:
        side = json.loads(Path(resume).with_suffix(".pth.json").read_text())
        require(side.get("scope") == "formal" and side.get("mode") == arm, "formal resume needs a formal state of this arm")
    plan = {"arm": arm, "parent": c["parent"], "start_update": 18240, "stop_update": 36480, "updates": 18240,
            "save_points": [ep * WINDOWS_PER_EPOCH for ep in FORMAL_SAVE_EPOCHS], "scope": "formal",
            "N_reference": {"contract": c["S15_contract"], "formal_identity": c["S15_formal_identity"]},
            "evaluation": {"endpoint": "e120 EMA", "view": "A", "images": 548, "budgets": [600, 300],
                           "score_calls": 4, "evaluator": "tools/scale_baseline/s15_formal_eval.py (S15 view-A protocol)"}}
    return plan


def formal(c, out, arm, counts, resume):
    torch = torch_mod()
    from formal_data import WINDOWS_PER_EPOCH
    formal_guard(c, arm, resume, plan_only=False)
    e, info = make(c, "cuda:0", arm)
    e.r26_T_init = info["R26"]["T_init_sha256"]
    register_counts(e, counts)
    if resume is not None:
        load(c, e, resume, "formal")
    else:
        save(c, e, Path(out) / "checkpoint-epoch-060.pth", "formal")
    loop = run_loop(c, e, out, 36480, "formal", counts,
                    save_points={ep * WINDOWS_PER_EPOCH for ep in FORMAL_SAVE_EPOCHS}, monitor_points=())
    loop.pop("training_image_ids")
    return {"arm": arm, "install": info, **loop}


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("prepare", "cpu", "accept", "diag", "cost", "formal"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--authorized-formal", default=None, help="exact task path that authorises the formal run")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    if args.action == "prepare":
        return prepare(args)
    c = load_contract()
    if args.action == "formal" and args.plan_only:
        print(json.dumps(formal_guard(c, args.arm, args.resume, plan_only=True), indent=2))
        return
    if args.action == "formal":
        require(args.authorized_formal is not None and Path(args.authorized_formal).is_file(),
                "formal training requires --authorized-formal <task file issued by the controller>; R26-001 does not launch it")
    args.out.mkdir(parents=True, exist_ok=False)
    start, cpu0 = time.monotonic(), time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    try:
        from r25_run import setup
        from s15_formal_train import bind_seed
        bind_seed(0)
        if args.action != "cpu":
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
        else:
            result = formal(c, args.out, args.arm, counts, args.resume)
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
