"""CPU acceptance of the real P3 tap on the frozen seed0 EMA weights (one image, no CUDA).

Scope: tap mechanics on the *real* detector (bitwise-equal outputs with and without the
hook, P3 shape ``[1, 256, 96, 168]`` on every view, exactly one encoder call per forward),
the real strip pooling and the complete GT-free decision enumeration for one frozen image,
and a few fixed comparator updates with the detector held frozen.  CPU numerics differ
from the GPU run, so agreement with the frozen raw views is *reported*, never gated; the
audited engine restore and the bitwise raw-view comparison belong to the GPU engineering
check (``pdra_gpu_check.py``).

Run (frozen interpreter, CUDA hidden):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 python -I tools/pdra/pdra_cpu_tap_acceptance.py \
        --parent <softnms execution-contract.json> --out <.../r18-pdra-wiring-20261030-001/cpu-tap-001>
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _path in (str(HERE), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pdra_core as P
import pdra_gpu_check as G
from pdra_stage1 import _affines

TASK = "r18-pdra-wiring-20261030-001"
IMAGE_ROW = 12          # first image of frozen raw batch 3 (all four images eval-split)
BATCH = 3
UPDATES = 5
require, publish, read, exact = G.require, G.publish, G.read, G.exact


class _Table:
    def __init__(self, table):
        self.table = table

    def r(self, pair):
        return self.table[(pair.identity_i, pair.identity_j)][2]

    def a(self, pair):
        return self.table[(pair.identity_i, pair.identity_j)][3]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", default="UNRECORDED")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(TASK in out.parts, "output must sit inside the task directory " + TASK)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    publish(out / "START.json", {"task": TASK, "mode": "cpu-tap-acceptance", "pid": os.getpid(),
                                 "started_unix": time.time(), "source_commit": args.source_commit,
                                 "image_row": IMAGE_ROW, "threads": args.threads})
    try:
        import torch
        torch.set_num_threads(args.threads)
        require(not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                "CPU acceptance must run with CUDA hidden")
        import pdra_features as FEAT
        import pdra_learn as L
        import pdra_tap as TAP
        inputs = G.resolve_inputs(args.parent, G.FROZEN_BATCHES)
        t0 = time.monotonic()
        loaded = TAP.load_cpu_ema_model_for_acceptance(
            training_source_root=inputs["training_root"], train_contract=inputs["training"],
            checkpoint_path=inputs["checkpoint"]["path"])
        model = loaded["model"]
        from sparse_rtdetr.r0.evaluation_data import Development548
        data = Development548(inputs["data_spec"])
        row = data.records[IMAGE_ROW]
        image = row["stable_image_id"]
        j = IMAGE_ROW - 4 * BATCH
        frozen = torch.load(str(exact(inputs["raw_refs"][BATCH])), map_location="cpu", weights_only=True)
        require(frozen["image_ids"][j] == image, "frozen raw batch image differs")
        load_seconds = time.monotonic() - t0
        digest_before = L.tensor_state_digest(model)
        # 1. forward without and with the tap (CPU)
        t0 = time.monotonic()
        a = TAP.five_view_forward(data, [row], model, tap=None, device="cpu")
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, [row], model, tap=tap, device="cpu")
        forward_seconds = time.monotonic() - t0
        views = []
        for va, vb, vf in zip(a["records"], b["records"], frozen["views"]):
            with torch.no_grad():
                require(torch.equal(va["pred_logits"], vb["pred_logits"]) and torch.equal(va["pred_boxes"], vb["pred_boxes"]),
                        f"tap changed CPU detector output on view {va['view_id']}")
                views.append({"view_id": va["view_id"], "tap_equal": True,
                              "frozen_gpu_raw_max_abs_diff": {
                                  "logits": float((va["pred_logits"][0] - vf["pred_logits"][j]).abs().max()),
                                  "boxes": float((va["pred_boxes"][0] - vf["pred_boxes"][j]).abs().max())}})
        require(tap.calls == 5 and all(s == (1, 256, 96, 168) for s in tap.shapes), f"tap shapes differ: {tap.shapes}")
        require([c.as_dict() for c in a["candidates"][0]] == [c.as_dict() for c in b["candidates"][0]],
                "decoded pools differ between passes")
        require(L.tensor_state_digest(model) == digest_before, "forward changed detector weights")
        # 2. real pooling + complete GT-free decisions for this image
        t0 = time.monotonic()
        pool = b["candidates"][0]
        affine_of = _affines(row, b["geometry"][0])
        bank = FEAT.FeatureBank({v: b["p3_maps"][v][0] for v in range(5)})
        records = P.cluster_pair_records(pool)
        pairs = [P.build_pair_input(r["box_i"].box_xyxy, affine_of(r["box_i"]), r["box_j"].box_xyxy,
                                    affine_of(r["box_j"]), identity_i=P._identity(r["box_i"]),
                                    identity_j=P._identity(r["box_j"])) for r in records]
        valid_pairs = [p for p in pairs if p.valid]
        p0 = time.perf_counter()
        with torch.no_grad():
            d_i = FEAT.pool_sides(bank, [p.left for p in valid_pairs])
            d_j = FEAT.pool_sides(bank, [p.right for p in valid_pairs])
        pool_seconds = time.perf_counter() - p0
        require(torch.isfinite(d_i.float()).all() and torch.isfinite(d_j.float()).all(), "pooled features not finite")
        arms = L.build_arms()
        pdra_arm = arms["arms"]["pdra_difference_strips"]
        adapter = L.PairBankAdapter.build(pdra_arm, bank, pool, affine_of, "pdra")
        decisions = P.pair_decisions(pool, affine_of, _Table(adapter.table))
        merged = L.merge(pool, affine_of, adapter)
        require(len(decisions) == len(records), "decision count differs from the enumeration")
        stage0_rows = G.stage0_rows_for({image})[image]
        c_adapter = L.PairBankAdapter.build(arms["arms"]["c_whole_box_pairwise"], bank, pool, affine_of, "c")
        b_adapter = L.QualityBankAdapter.build(arms["arms"]["b_single_box_quality"], bank, pool, affine_of)
        decide_seconds = time.monotonic() - t0
        same_keys = sum(1 for d in decisions if (d["identity_i"], d["identity_j"]) in stage0_rows)
        # 3. a few fixed comparator updates with the detector frozen (labels from stage-0 rows)
        by_id = {P._identity(c): c for c in pool}
        keep, pref_t, abst_t = [], [], []
        for (ii, jj), r0 in stage0_rows.items():
            if r0[8] not in ("pref", "abstain") or not r0[13] or ii not in by_id or jj not in by_id:
                continue
            pair = P.build_pair_input(by_id[ii].box_xyxy, affine_of(by_id[ii]), by_id[jj].box_xyxy,
                                      affine_of(by_id[jj]), identity_i=ii, identity_j=jj)
            if not pair.valid:
                continue
            keep.append(pair)
            pref_t.append(int(r0[9]) if r0[8] == "pref" else -1)
            abst_t.append(1.0 if r0[8] == "abstain" else 0.0)
        with torch.no_grad():
            m_i = FEAT.pool_sides(bank, [p.left for p in keep])
            m_j = FEAT.pool_sides(bank, [p.right for p in keep])
        tensors = L.PairTensors(m_i, m_j, FEAT.geometry_tensor(keep, "cpu"),
                                torch.tensor(pref_t, dtype=torch.int8), torch.tensor(abst_t))
        comparator_before = L.tensor_state_digest(pdra_arm)
        spec = {**L.frozen_spec(), "updates": UPDATES, "batch_pairs": min(256, len(tensors))}
        train = L.train_pair_arm(pdra_arm, tensors, spec=spec, device="cpu")
        grads = {name: (p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0.0)
                 for name, p in pdra_arm.named_parameters()}
        require(all(grads.values()), "comparator parameters without finite non-zero gradient")
        require(all(p.grad is None for p in model.parameters()), "detector received gradients")
        require(L.tensor_state_digest(model) == digest_before, "comparator training changed detector weights")
        require(L.tensor_state_digest(pdra_arm) != comparator_before, "comparator did not change")
        result = {"status": "PASS", "task": TASK, "mode": "cpu-tap-acceptance", "image_id": image,
                  "image_row": IMAGE_ROW, "frozen_batch": BATCH, "source_commit": args.source_commit,
                  "model": {"ema_updates": loaded["ema_updates"], "scope": loaded["scope"],
                            "load_seconds": load_seconds, "cpu_only": True, "audited_restore": False},
                  "forward": {"calls": a["forward_calls"] + b["forward_calls"], "seconds": forward_seconds,
                              "views": views, "tap_calls": tap.calls, "p3_shape": list(tap.shapes[0]),
                              "pools_identical_between_passes": True},
                  "feature_binding": {"level": P.FEATURE_LEVEL, "stride": list(P.FEATURE_STRIDE),
                                      "p3_extent": [tap.shapes[0][2], tap.shapes[0][3]],
                                      "core_extent": list(P.ViewAffine((0.0, 0.0, 10.0, 10.0)).feature_extent)},
                  "pooling": {"pool": len(pool), "clusters": len(P.nms_clusters(pool)), "pairs": len(records),
                              "readable": len(valid_pairs),
                              "strips": sum(len(p.left.feature_strips) + len(p.right.feature_strips) for p in valid_pairs),
                              "seconds": pool_seconds,
                              "us_per_readable_pair": 1e6 * pool_seconds / max(len(valid_pairs), 1)},
                  "decisions": {"count": len(decisions), "reasons": dict(Counter(d["decision_reason"] for d in decisions)),
                                "stage0_pairs_for_image": len(stage0_rows),
                                "pairs_with_identical_identity_in_stage0": same_keys,
                                "gt_read_in_decision_phase": False,
                                "merge": {k: merged[k] for k in ("cluster_count", "kept_count", "substitutions",
                                                                 "all_abstain_clusters", "rank_shifted_members",
                                                                 "budget_membership_delta_vs_C")},
                                "controls": {"c_pairs": len(c_adapter.table), "b_pairs": len(b_adapter.table)},
                                "seconds": decide_seconds},
                  "learning": {"updates": train["updates"], "batch": train["batch"], "labelled_pairs": len(tensors),
                               "first_losses": train["first_losses"], "parameters_with_gradient": sum(grads.values()),
                               "parameters_total": len(grads), "detector_grads": "none",
                               "detector_digest_unchanged": True, "comparator_changed": True},
                  "vm_hwm_bytes": G.rss_high_water(), "elapsed_seconds": time.monotonic() - started}
        ref = publish(out / "RESULT.json", result)
        publish(out / "CLOSED.json", {"result": ref, "returncode": 0, "automatic_retry": False})
        print(json.dumps({"status": "PASS", "result": ref}), flush=True)
        return 0
    except Exception:
        publish(out / "STOP.json", {"traceback": traceback.format_exc(),
                                    "elapsed_seconds": time.monotonic() - started, "automatic_retry": False})
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
