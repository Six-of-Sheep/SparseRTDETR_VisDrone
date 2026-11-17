"""R28 demand side: the one fixed demand-network training, the fixed train_core prior (arm K) and the G0 reading.

demand(c, out, counts)  30 epochs on train_core through the H augmentation pipeline (CONTRACT demand_training); the
                        last state is saved, nothing is selected, development is not read.
prior(c, out, counts)   label-only: mean over train_core of the per-image deployment demand map (empty images are zero).
g0(c, out, counts)      geometry only: magnification of the certified development GT under the predicted demand, the
                        GT demand and the fixed prior; one gate on the small-GT geometric mean of the predicted demand.
None of these runs the detector or a scorer.
"""
import io
import json
import math
import os
import time
from pathlib import Path

import r28_common as K

F64 = None  # set on first use (torch is imported lazily, as in the L0 runner)


def _torch():
    global F64
    torch = K.torch_mod()
    F64 = torch.float64
    return torch


class DemandItems:
    """Loader-side reduction: (image at 384x672 FP32, demand target FP32, strong flag).  Runs in the worker."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        from sparse_rtdetr.research import r28_warp as W
        pixels, target, trace = self.data[key]
        x = W.demand_input(pixels[None])[0]
        d = W.demand_from_boxes(W.cxcywh_to_xyxy(target["boxes"]))
        return x, d.float(), int(trace["augmentation_strong"]), len(target["labels"])


def _loader(data, pipeline_epoch, workers, limit=None):
    from torch.utils.data import DataLoader
    from sparse_rtdetr.localization.data import WindowSampler, seed_worker
    torch = _torch()
    sampler = WindowSampler(len(data), seed=data.seed, epoch=pipeline_epoch, completed_windows=0, window_limit=limit)
    return DataLoader(DemandItems(data), batch_sampler=sampler, num_workers=workers, worker_init_fn=seed_worker,
                      multiprocessing_context="spawn", prefetch_factor=4, persistent_workers=False,
                      generator=torch.Generator().manual_seed(data.seed), timeout=600)


def learning_rate(step, steps, lr, lr_end):
    """Step 1..steps: cosine from lr (first step) to lr_end (last step)."""
    return lr_end + 0.5 * (lr - lr_end) * (1.0 + math.cos(math.pi * (step - 1) / max(steps - 1, 1)))


def demand(c, out, counts):
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    torch = _torch()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "demand").resolve(), "this action writes <round>/demand")
    cfg = dict(c["demand_training"])
    epochs, limit = cfg["epochs"], None
    if os.environ.get("R28_DEV_DEMAND_EPOCHS"):
        K.require(K.DEV, "R28_DEV_DEMAND_EPOCHS is refused outside a dev round")
        epochs, limit = int(os.environ["R28_DEV_DEMAND_EPOCHS"]), int(os.environ.get("R28_DEV_DEMAND_WINDOWS", "8"))
    pipeline = cfg["pipeline_epochs"][:epochs] if limit else cfg["pipeline_epochs"]
    K.require(len(pipeline) == epochs and (limit or epochs == 30), "demand epochs")
    per_epoch = limit or cfg["windows_per_epoch"]
    steps = epochs * per_epoch
    K.PHASE["phase"] = "demand:construct"
    dev = torch.device(K.device())
    torch.manual_seed(cfg["seed"])
    net = W.DemandNet().to(dev).train()
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    log, step, started = [], 0, time.monotonic()
    with (out / "TRAIN.jsonl").open("x", buffering=1) as ledger:
        for ep, pe in enumerate(pipeline, 1):
            K.PHASE["phase"] = f"demand:epoch {ep}"
            t0, total, seen, strong, gts, mass = time.monotonic(), 0.0, 0, 0, 0, 0.0
            for x, d, s, n in _loader(data, pe, cfg["workers"], limit):
                step += 1
                lr = learning_rate(step, steps, cfg["lr"], cfg["lr_end"])
                for g in opt.param_groups:
                    g["lr"] = lr
                x, d = x.to(dev), d.to(dev)
                K.require(tuple(x.shape) == (cfg["batch"], 3, *W.DEMAND_HW) and tuple(d.shape) == (cfg["batch"], *W.GRID)
                          and x.dtype == d.dtype == torch.float32, "demand batch")
                loss = W.demand_loss(net(x), d)
                K.require(bool(torch.isfinite(loss)), f"demand loss is not finite at step {step}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                counts.update(demand_steps=1)
                total += float(loss.detach())
                seen += 1
                strong += int(s.sum())
                gts += int(n.sum())
                mass += float(d.sum())
            K.require(seen == per_epoch and strong == (seen * cfg["batch"] if pe < 118 else 0), f"demand epoch {ep} windows / strong flag")
            row = {"epoch": ep, "pipeline_epoch": pe, "strong": pe < 118, "steps": seen, "step": step, "loss_mean": total / seen,
                   "lr_last": lr, "gt": gts, "target_mass_mean_per_image": mass / (seen * cfg["batch"]), "seconds": time.monotonic() - t0}
            ledger.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            log.append(row)
    K.require(step == steps == counts["demand_steps"], "demand step count")
    K.PHASE["phase"] = "demand:save"
    net.eval().requires_grad_(False)
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    digest = tensor_digest(state.items())
    K.require(all(bool(torch.isfinite(v).all()) for v in state.values() if v.is_floating_point()), "demand state is not finite")
    path = out / "demand-net.pth"
    with path.open("xb") as f:
        torch.save({"schema": "r28-demand-net-v1", "definition": W.DEFINITION, "model": state, "state_sha256": digest,
                    "training": {**cfg, "epochs_run": epochs, "steps_run": steps, "contract_sha256": c["contract_sha256"]}}, f)
        f.flush()
        os.fsync(f.fileno())
    return {"stage": "demand", "weights": K.ref(path), "state_sha256": digest, "parameters": params, "epochs": epochs,
            "steps": steps, "pipeline_epochs": pipeline, "log": log, "train_wall_seconds": time.monotonic() - started,
            "development_read": False, "endpoint": "last step", "dev_shortened": bool(limit), "device": str(dev),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if dev.type == "cuda" else None}


# ================================================================ deployment geometry from labels
def deployment_xyxy(parsed, width, height):
    """Normalised xyxy (FP64) of the GT a deployment rendering keeps: clipped to the image, positive area; resize only.
    Label arithmetic in FP64, used for the informational readings of g0; the production FP32 boxes of the pipeline can
    put a box centre on the other side of a grid-cell boundary, so the prior of arm K uses the pipeline itself."""
    import numpy as np
    torch = _torch()
    b = np.array([r["box"] for r in parsed["gt"]], dtype=np.float64).reshape(-1, 4)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
    keep = np.all(b[:, 2:] > b[:, :2], axis=1)
    return torch.from_numpy(b[keep] / np.array([width, height, width, height], dtype=np.float64)), keep


def _annotation(data, row):
    import sparse_rtdetr.localization.data as LD
    raw = LD.read_bound(Path(data.annotations_path) / Path(row["annotation_relative_path"]).name,
                        row["annotation_sha256"], row["annotation_size_bytes"])
    return LD.parse_raw(raw, row["stable_image_id"], expected_rows=row["raw_annotation_rows"])


class _PriorItems:
    """Deployment rendering of one train_core image through the production transform (training=False: resize only).
    Box geometry depends on the image size only, so a blank image of the bound size stands in for the pixels; the
    first images are also rendered from their real pixels in `prior` and must give the same boxes bit for bit."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.records)

    def boxes(self, i, image=None):
        from PIL import Image
        row = self.data.records[i]
        parsed = _annotation(self.data, row)
        im = image if image is not None else Image.new("RGB", (row["width"], row["height"]))
        _, target, trace = self.data.transform(im, parsed, epoch=120, seed=0, position=0, image_id=row["stable_image_id"], training=False)
        K.require(not trace["geometry_steps"] and not trace["augmentation_strong"], "deployment rendering must be resize only")
        return target["boxes"]

    def __getitem__(self, i):
        from sparse_rtdetr.research import r28_warp as W
        b = self.boxes(i)
        return W.demand_from_boxes(W.cxcywh_to_xyxy(b)), len(b)


def prior(c, out, counts):
    from PIL import Image
    from torch.utils.data import DataLoader
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    torch = _torch()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "prior").resolve(), "this action writes <round>/prior")
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    K.require(len(data.records) == 4869, "train_core members")
    items = _PriorItems(data)
    K.PHASE["phase"] = "prior:real-pixel rendering of the first images"
    checked = 0
    for i in range(32):
        image_bytes, _ = data.bound_bytes(data.records[i])
        with Image.open(io.BytesIO(image_bytes)) as im:
            real = items.boxes(i, im.convert("RGB"))
        K.require(torch.equal(real, items.boxes(i)), f"blank-image rendering differs from the real image rendering: {i}")
        checked += 1
    K.PHASE["phase"] = "prior:train_core"
    total, empty, boxes, seen = torch.zeros(W.GRID, dtype=F64), 0, 0, 0
    loader = DataLoader(items, batch_size=64, shuffle=False, num_workers=12, multiprocessing_context="spawn", timeout=600)
    for d, n in loader:
        K.require(d.dtype == F64 and tuple(d.shape[1:]) == W.GRID, "prior demand batch")
        total += d.sum(0)
        empty += int((n == 0).sum())
        boxes += int(n.sum())
        seen += len(n)
    K.require(seen == len(data.records), "prior image count")
    mean = total / seen
    K.require(bool(torch.isfinite(mean).all()) and bool((mean >= 0).all()) and float(mean.max()) > 0, "prior demand map")
    digest = tensor_digest([("demand", mean)])
    path = out / "prior.json"
    K.write_new(path, {"schema": "r28-fixed-demand-v1", "definition": W.DEFINITION, "demand": mean.tolist(),
                       "state_sha256": digest, "images": seen, "empty_images": empty, "boxes": boxes,
                       "rule": "arithmetic mean over train_core of blur3(demand) of the deployment rendering (production "
                               "transform, training=False: resize only); images without valid GT contribute a zero map; "
                               "no eps in the stored map",
                       "contract_sha256": c["contract_sha256"]})
    K.require(tensor_digest([("demand", torch.tensor(K.read_json(path)["demand"], dtype=F64))]) == digest, "prior JSON round trip")
    plan = W.plan_from_demand(mean[None])
    return {"stage": "prior", "file": K.ref(path), "state_sha256": digest, "images": seen, "empty_images": empty,
            "boxes": boxes, "demand_sum": float(mean.sum()), "demand_max": float(mean.max()),
            "slopes": {"x": [float(plan.mx.min()), float(plan.mx.max())], "y": [float(plan.my.min()), float(plan.my.max())]},
            "pipeline_rendering_check": {"images": checked, "blank_image_boxes_equal_real_image_boxes": True},
            "development_read": False}


# ================================================================ G0
def _classes(area):
    return {"small": area <= K.SMALL_AREA, "medium": (area >= K.SMALL_AREA) & (area <= K.LARGE_AREA), "large": area >= K.LARGE_AREA}


def _reading(g, area):
    import numpy as np
    out = {}
    for name, sel in _classes(area).items():
        v = g[sel]
        out[name] = {"n": int(sel.sum())}
        if len(v):
            out[name].update(g_geomean=float(np.exp(np.log(v).mean())), g_q10_50_90=[float(q) for q in np.quantile(v, [0.1, 0.5, 0.9])],
                             frac_g_ge_1p2=float((v >= 1.2).mean()), frac_g_lt_1=float((v < 1.0).mean()),
                             frac_g_lt_0p8=float((v < 0.8).mean()))
    return out


def _magnify(W, plan, boxes):
    """g of the (n, 4) normalised xyxy boxes of one image under a one-image plan (CPU, FP64)."""
    torch = _torch()
    if not len(boxes):
        return torch.zeros(0, dtype=F64)
    ox, oy = W.magnification(plan, boxes[None])
    return torch.sqrt(ox * oy)[0]


class _TrainDeploy:
    """Deployment rendering of train_core for the informational train reading: image at 384x672, label boxes."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.records)

    def __getitem__(self, i):
        from PIL import Image
        from torchvision.transforms.v2 import functional as TF
        from sparse_rtdetr.research import r28_warp as W
        row = self.data.records[i]
        image_bytes, _ = self.data.bound_bytes(row)
        with Image.open(io.BytesIO(image_bytes)) as im:
            x = TF.pil_to_tensor(self.data.transform.resize(im.convert("RGB"))).float() / 255.
        return W.demand_input(x[None])[0], i


def g0(c, out, counts):
    import numpy as np
    from infer_seed import load_original, pixels
    from torch.utils.data import DataLoader
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    torch = _torch()
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / "g0").resolve(), "this action writes <round>/g0")
    K.PHASE["phase"] = "g0:sources"
    dev = torch.device(K.device())
    net_ctx = R.Context(K.demand_source(c, "W")).to(dev)
    prior_ctx = R.Context(K.demand_source(c, "K"))
    prior_plan = W.plan_from_demand(prior_ctx.fixed[None])
    data = S.development(c["data"]["development"], S.S15_HW)
    rows = data.records
    K.require([r["stable_image_id"] for r in rows] == c["development_ids"] and len(rows) == 548, "development members/order")
    parsed = [data.raw(row) for row in rows]
    dataset, _, gt_counts = bridge_inputs(rows, parsed, [])
    K.require({k: gt_counts[k] for k in ("GT_before", "GT_after")} == c["reference"]["GT_counts"]["600"],
              f"certified GT set differs from the L0 evaluation: {gt_counts}")
    by_image = {}
    for a in dataset["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    K.PHASE["phase"] = "g0:development"
    per_gt, per_image = [], []
    with torch.inference_mode():
        for off in range(0, len(rows), K.EVAL_BATCH):
            batch = rows[off:off + K.EVAL_BATCH]
            imgs = []
            for row in batch:
                with load_original(data, row) as image:
                    imgs.append(pixels(data, image))
            x = torch.stack(imgs).to(dev)
            K.require(tuple(x.shape[-2:]) == tuple(S.S15_HW) and x.dtype == torch.float32, "G0 input is not the FP32 H canvas")
            with torch.autocast(dev.type, enabled=False):
                d_pred = net_ctx.demand(x)
            counts.update(demand_forwards_development=1)
            plans = W.plan_from_demand(d_pred)
            for j, row in enumerate(batch):
                index = off + j + 1
                anns = [a for a in by_image.get(index, []) if not a["iscrowd"]]
                wh = np.array([row["width"], row["height"]] * 2, dtype=np.float64)
                b = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns],
                             dtype=np.float64).reshape(-1, 4)
                b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, row["width"])
                b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, row["height"])
                K.require(bool(np.all(b[:, 2:] > b[:, :2])), "certified GT with a non-positive clipped side")
                boxes = torch.from_numpy(b / wh)
                d_gt = W.demand_from_boxes(deployment_xyxy(parsed[off + j], row["width"], row["height"])[0])
                g = {"predicted": _magnify(W, plans.select(slice(j, j + 1)), boxes),
                     "gt_demand": _magnify(W, W.plan_from_demand(d_gt[None]), boxes),
                     "prior": _magnify(W, prior_plan, boxes)}
                for n, a in enumerate(anns):
                    per_gt.append({"image": row["stable_image_id"], "raw_annotation_id": a["raw_annotation_id"],
                                   "category_id": a["category_id"], "area": a["area"],
                                   "bbox_xyxy": [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]],
                                   **{k: float(v[n]) for k, v in g.items()}})
                per_image.append({"image": row["stable_image_id"], "gt": len(anns), "demand_sum_predicted": float(d_pred[j].sum()),
                                  "demand_sum_gt": float(d_gt.sum()),
                                  "slopes_predicted": [float(plans.mx[j].min()), float(plans.mx[j].max()),
                                                       float(plans.my[j].min()), float(plans.my[j].max())]})
    area = np.array([r["area"] for r in per_gt])
    crowd = gt_counts["GT_after"] - len(per_gt)
    readings = {k: _reading(np.array([r[k] for r in per_gt]), area) for k in ("predicted", "gt_demand", "prior")}
    g_small = readings["predicted"]["small"]["g_geomean"]
    passed = bool(g_small >= K.G0_MINIMUM)
    K.write_new(out / "per_gt.json", per_gt)
    K.write_new(out / "per_image.json", per_image)
    # informational: the same reading of the predicted demand on train_core (deployment rendering, pipeline-valid GT)
    K.PHASE["phase"] = "g0:train_core"
    train = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    tg, ta = [], []
    loader = DataLoader(_TrainDeploy(train), batch_size=16, shuffle=False, num_workers=12, multiprocessing_context="spawn", timeout=600)
    with torch.inference_mode():
        for x, idx in loader:
            with torch.autocast(dev.type, enabled=False):
                d = net_ctx.net(x.to(dev)).to(F64).cpu()
            counts.update(demand_forwards_train=1)
            plans = W.plan_from_demand(d)
            for j, i in enumerate(idx.tolist()):
                row = train.records[i]
                p = _annotation(train, row)
                boxes, keep = deployment_xyxy(p, row["width"], row["height"])
                tg.append(_magnify(W, plans.select(slice(j, j + 1)), boxes).numpy())
                ta.append(np.array([r["raw_area"] for r, k in zip(p["gt"], keep) if k], dtype=np.float64))
    train_reading = _reading(np.concatenate(tg), np.concatenate(ta))
    return {"stage": "g0", "G0_passed": passed, "G0_small_geomean_predicted": g_small, "G0_minimum": K.G0_MINIMUM,
            "label": None if passed else c["labels"]["G0_failed"],
            "set": {"definition": "certified evaluation GT (r0.evaluation.bridge_inputs: raw-ignore raster filter >= 0.5 removed), "
                                  "non-crowd, COCO area ranges on the raw area with inclusive end points",
                    "GT_before": gt_counts["GT_before"], "GT_after": gt_counts["GT_after"], "crowd_excluded": crowd,
                    "measured": len(per_gt), "small": readings["predicted"]["small"]["n"],
                    "small_equals_26441": readings["predicted"]["small"]["n"] == 26441,
                    "note": "26,441 is a cross-check clue from the earlier analysis, not a constant of this stage; the "
                            "label-script count 26,575 belongs to the older annotation-only convention"},
            "development": readings, "train_core_predicted": train_reading,
            "train_minus_development_small_geomean": train_reading["small"]["g_geomean"] - g_small,
            "train_core_set": "pipeline-valid GT of the deployment rendering, raw-area classes (not the certified evaluation filter)",
            "demand": {"network": net_ctx.binding(), "prior": prior_ctx.binding()},
            "files": {"per_gt": K.ref(out / "per_gt.json"), "per_image": K.ref(out / "per_image.json")},
            "development_use": "development annotations measure one frozen predictor and decide whether the formal W run starts; "
                               "they do not train or select the demand network",
            "detector_forwards": 0, "scorer_calls": 0}
