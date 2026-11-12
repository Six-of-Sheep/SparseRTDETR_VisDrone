#!/usr/bin/env python3
"""R27 post-L0 design check -- script 1 of 2.

Exact shape / block / maskable-cost accounting for the candidate
"set-conditional detection-risk allocation of native residual computation".

Scope fixed by the task book: PResNet18 (variant 'd') C3-C5 residual branches,
H = 1152 x 2016 input, no crop, no extra local backbone.

Pure standard library. No torch, no CUDA, no model, no data. Every structural
constant below is transcribed from the frozen vendored source; make_identity.py
checks those source files against the R27 formal release manifest.

Source of truth:
  vendor/rtdetrv2_pytorch/src/nn/backbone/presnet.py
  vendor/rtdetrv2_pytorch/src/nn/backbone/common.py
  vendor/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml
  vendor/rtdetrv2_pytorch/configs/rtdetrv2/include/rtdetrv2_r50vd.yml

Revision note (2026-11-12, second pass): the first version of this file did
not carry `ch_in = ch_out` into the second block of each stage
(presnet.py:148-149), so it counted block1.branch2a as 64->128 / 128->256 /
256->512 instead of 128->128 / 256->256 / 512->512 and understated the C3-C5
residual MACs (48.158 G instead of 56.184 G). It also reported the INPUT band
read by a stride-2 conv as if it were halo compute. Both are corrected here;
the halo is now counted exactly, per explicit mask, as the extra branch2a
outputs that must be computed (SECTION C).

Run:  python check_backbone_shapes.py [--json]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# 1. Architecture transcribed from presnet.py (PResNet, variant 'd', depth 18)
# --------------------------------------------------------------------------
# ResNet_cfg[18] = [2, 2, 2, 2];  block = BasicBlock (depth < 50), expansion = 1
# ch_out_list = [64, 128, 256, 512]; _out_strides = [4, 8, 16, 32]
# conv1 (variant c/d) = 3 convs, conv1_1 has stride 2.
# max_pool2d(k=3, s=2, p=1) after conv1.
# Blocks(): stride = 2 if i == 0 and stage_num != 2 else 1
#           shortcut = False if i == 0 else True
#           if i == 0: ch_in = ch_out * expansion          (presnet.py:148-149)
#   -> stage 2 (res_layers[0]) runs at stride 1, stages 3..5 at stride 2.
# return_idx = [1, 2, 3]  (include/rtdetrv2_r50vd.yml; HybridEncoder
#   in_channels [128,256,512] = res_layers[1..3] outputs)
# BasicBlock.forward: out = branch2b(branch2a(x)); out = act(out + short)
#   branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=relu)
#   branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
#   short    = x                                   if shortcut
#            = AvgPool2d(2,2,0,ceil) -> 1x1 conv+BN  if variant d and stride 2
#            = 1x1 conv+BN (stride)                 otherwise

H_IN, W_IN = 1152, 2016

CONV1 = [  # (name, cin, cout, k, stride)
    ("conv1_1", 3, 32, 3, 2),
    ("conv1_2", 32, 32, 3, 1),
    ("conv1_3", 32, 64, 3, 1),
]

STAGES = [  # (res_layers index, ch_in, ch_out, count, stage_num)
    (0, 64, 64, 2, 2),
    (1, 64, 128, 2, 3),   # C3
    (2, 128, 256, 2, 4),  # C4
    (3, 256, 512, 2, 5),  # C5
]

C_LABEL = {1: "C3", 2: "C4", 3: "C5"}
STRIDE_OF = {1: 8, 2: 16, 3: 32}
GRID = 4                 # 4 x 4 macro regions


def conv_macs(cin, cout, k, h, w):
    return cin * cout * k * k * h * w


def conv_out(n, k, stride, pad):
    return (n + 2 * pad - k) // stride + 1


def build():
    """Return (per-block records, conv1 MACs, feature-map geometry)."""
    h, w = H_IN, W_IN
    conv1_macs = 0
    geom = {}
    for _name, cin, cout, k, s in CONV1:
        h, w = conv_out(h, k, s, (k - 1) // 2), conv_out(w, k, s, (k - 1) // 2)
        conv1_macs += conv_macs(cin, cout, k, h, w)
    geom["conv1_out"] = (h, w, 64)          # 576 x 1008, stride 2
    h, w = conv_out(h, 3, 2, 1), conv_out(w, 3, 2, 1)   # max-pool 3x3 s2 p1
    geom["stem_out"] = (h, w, 64)           # 288 x 504, stride 4

    records = []
    for idx, ch_in, ch_out, count, stage_num in STAGES:
        cin = ch_in
        for i in range(count):
            stride = 2 if (i == 0 and stage_num != 2) else 1
            shortcut = (i != 0)
            in_h, in_w = h, w
            if stride == 2:
                oh, ow = conv_out(in_h, 3, 2, 1), conv_out(in_w, 3, 2, 1)
            else:
                oh, ow = in_h, in_w
            a_macs = conv_macs(cin, ch_out, 3, oh, ow)      # branch2a
            b_macs = conv_macs(ch_out, ch_out, 3, oh, ow)   # branch2b
            if shortcut:
                s_kind, s_macs, s_bns = "identity", 0, 0
            else:
                s_kind = ("avgpool2d(2,2,0,ceil)+1x1conv" if stride == 2
                          else "1x1conv")
                s_macs, s_bns = conv_macs(cin, ch_out, 1, oh, ow), 1
            records.append(dict(
                stage=idx, block=i, label=C_LABEL.get(idx, f"S{stage_num}"),
                cin=cin, cout=ch_out, stride=stride, shortcut=s_kind,
                in_hw=(in_h, in_w), out_hw=(oh, ow),
                a_macs=a_macs, b_macs=b_macs, s_macs=s_macs,
                res_macs=a_macs + b_macs, bns=2 + s_bns,
                feat_stride=STRIDE_OF.get(idx),
            ))
            h, w = oh, ow
            if i == 0:
                cin = ch_out                # presnet.py:148-149
        geom[f"stage{idx}_out"] = (h, w, ch_out)
    return records, conv1_macs, geom


# --------------------------------------------------------------------------
# 2. Masks: defined once on the C3 grid, mapped upward by 2x2 OR-pooling
# --------------------------------------------------------------------------


def mask_c3(regions, h3, w3):
    """0/1 grid at C3 for a set of (row, col) macro regions of the 4x4 grid."""
    rh, rw = h3 // GRID, w3 // GRID
    assert rh * GRID == h3 and rw * GRID == w3, "4x4 grid must tile C3 exactly"
    m = [[0] * w3 for _ in range(h3)]
    for (r, c) in regions:
        for y in range(r * rh, (r + 1) * rh):
            row = m[y]
            for x in range(c * rw, (c + 1) * rw):
                row[x] = 1
    return m


def or_pool2(m):
    """Mask on the next (stride x2) level: executed if ANY child is executed."""
    h, w = len(m), len(m[0])
    assert h % 2 == 0 and w % 2 == 0
    return [[1 if (m[2 * y][2 * x] or m[2 * y][2 * x + 1]
                   or m[2 * y + 1][2 * x] or m[2 * y + 1][2 * x + 1]) else 0
             for x in range(w // 2)] for y in range(h // 2)]


def dilate1(m):
    """3x3 dilation clipped to the grid: positions whose branch2a output is
    read by branch2b (3x3, stride 1, pad 1) at an executed position."""
    h, w = len(m), len(m[0])
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        for x in range(w):
            x0, x1 = max(0, x - 1), min(w, x + 2)
            hit = 0
            for yy in range(y0, y1):
                row = m[yy]
                if row[x0] or row[x1 - 1] or row[x]:
                    hit = 1
                    break
            out[y][x] = hit
    return out


def area(m):
    return sum(sum(r) for r in m)


def executed_macs(regions, design, geom):
    """Exact residual-branch MACs that must still be computed for a mask.

    Semantics: x_{l+1} = act(short(x_l) + M_l * F_l(x_l)); shortcut and the
    block input x_l are dense. To produce branch2b at the executed positions
    M, branch2a must be computed on dilate1(M) (its 3x3 support). branch2a
    itself only READS x_l, which already exists everywhere, so nothing else
    is recomputed. Elementwise BN/ReLU and gather/scatter are not MACs.
    """
    h3, w3, _ = geom["stage1_out"]
    level = {1: mask_c3(regions, h3, w3)}
    level[2] = or_pool2(level[1])
    level[3] = or_pool2(level[2])
    per_level = {}
    for idx in (1, 2, 3):
        m = level[idx]
        per_level[idx] = dict(mask=area(m), halo=area(dilate1(m)),
                              full=len(m) * len(m[0]))
    macs = 0
    for r in design:
        pl = per_level[r["stage"]]
        macs += r["cin"] * r["cout"] * 9 * pl["halo"]      # branch2a
        macs += r["cout"] * r["cout"] * 9 * pl["mask"]     # branch2b
    return macs, per_level


ALL = [(r, c) for r in range(GRID) for c in range(GRID)]
SCATTER4 = [(0, 0), (0, 2), (2, 1), (2, 3)]
MASKS = [
    ("16/16 all (native)", ALL),
    ("12/16 rows 0-2", [(r, c) for r in (0, 1, 2) for c in range(GRID)]),
    ("12/16 all but 4 scattered", [rc for rc in ALL if rc not in SCATTER4]),
    ("8/16 rows 0-1", [(r, c) for r in (0, 1) for c in range(GRID)]),
    ("8/16 rows 1-2", [(r, c) for r in (1, 2) for c in range(GRID)]),
    ("8/16 checkerboard", [(r, c) for (r, c) in ALL if (r + c) % 2 == 0]),
    ("4/16 corner 2x2", [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ("4/16 scattered", SCATTER4),
    ("1/16 interior (1,1)", [(1, 1)]),
    ("0/16 empty", []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(newline="\n")
    except Exception:
        pass

    records, conv1_macs, geom = build()

    design = [r for r in records if r["stage"] in (1, 2, 3)]
    stem_macs = conv1_macs + sum(r["res_macs"] + r["s_macs"]
                                for r in records if r["stage"] == 0)
    backbone_macs = conv1_macs + sum(r["res_macs"] + r["s_macs"] for r in records)
    res_macs = sum(r["res_macs"] for r in design)
    short_macs = sum(r["s_macs"] for r in design)
    conv_params = (sum(cin * cout * k * k for _n, cin, cout, k, _s in CONV1)
                   + sum(r["cin"] * r["cout"] * 9 + r["cout"] * r["cout"] * 9
                         + (r["cin"] * r["cout"] if r["s_macs"] else 0)
                         for r in records))

    # ---- measured anchors from this project ------------------------------
    # R26 return-001/cost/CLOSED.json, arm N, median over 3 passes x 16
    # batches (FP32, batch4, CUDA-event segments), re-derived from the receipt
    # on 2026-11-12: model 103.218, backbone 47.983, encoder 29.548,
    # decoder 25.657, input_proj 1.448 ms/batch4.
    MEAS = dict(model_ms=103.218, backbone_ms=47.983, encoder_ms=29.548,
                decoder_ms=25.657, input_proj_ms=1.448)
    # R27 L0 cost4 (summary/RESULT.json, one process, four endpoints):
    S15_MODEL_MS_B4 = 1634.606 / 16     # 102.163 ms/batch4 model GPU
    S15_E2E_MS_B4 = 4614.227 / 16       # 288.389 ms/batch4 end to end
    A0_MODEL_MS_B4 = 755.803 / 16       # 47.238 ms/batch4 model GPU (L canvas)
    P_OVER_S15_MODEL = 0.7666815        # R27-P / S15, model GPU
    P_OVER_S15_E2E = 0.9178551          # R27-P / S15, end to end
    MODEL_MACS_COUNTED_G = 169.948286976  # R26 cpu/CLOSED.json, arm N, counted
    # R26 measured block medians (ms/batch4), same receipt:
    R26 = dict(N=dict(block=10.796, B=6.840, model=103.218),
               U=dict(block=7.844, B=1.384, model=98.733),
               P=dict(block=8.455, B=1.384, model=100.835))

    out = []
    p = out.append

    p("=" * 78)
    p("SECTION A -- exact design object: PResNet18(d) C3-C5 residual branches")
    p("=" * 78)
    p(f"input H x W = {H_IN} x {W_IN}  (stride-4 grid after conv1 + max-pool: "
      f"{geom['stem_out'][0]} x {geom['stem_out'][1]})")
    p("")
    hdr = (f"{'stage':>5} {'blk':>3} {'cin':>4} {'cout':>4} {'str':>3} "
           f"{'in HW':>11} {'out HW':>10} {'2a MACs':>8} {'2b MACs':>8} "
           f"{'shortcut':>30} BN")
    p(hdr)
    p("-" * len(hdr))
    for r in design:
        p(f"{r['label']:>5} {r['block']:>3} {r['cin']:>4} {r['cout']:>4} "
          f"{r['stride']:>3} {str(r['in_hw']):>11} {str(r['out_hw']):>10} "
          f"{r['a_macs']/1e9:>7.3f}G {r['b_macs']/1e9:>7.3f}G "
          f"{r['shortcut']:>30} {r['bns']}")
    p("module names: backbone.res_layers[1|2|3].blocks[0|1].{branch2a,branch2b,"
      "short}")
    p("")
    p(f"C3-C5 residual-branch MACs (branch2a+branch2b) : {res_macs/1e9:.3f} G")
    p(f"C3-C5 shortcut-path MACs (always dense)        : {short_macs/1e9:.3f} G")
    p(f"backbone MACs total (conv1 + res_layers 0..3)  : {backbone_macs/1e9:.3f} G")
    p(f"  stem + C2 (always dense, computed before any decision) : "
      f"{stem_macs/1e9:.3f} G ({100*stem_macs/backbone_macs:.1f}% of backbone)")
    p(f"  C3-C5 residual branches (the only maskable part)       : "
      f"{res_macs/1e9:.3f} G ({100*res_macs/backbone_macs:.1f}% of backbone)")
    p(f"backbone conv weights implied by this table : {conv_params:,} "
      f"(self-check: ResNet-18 trunk is ~11.2 M)")
    p(f"R26 counted whole-model MACs (arm N)         : "
      f"{MODEL_MACS_COUNTED_G:.2f} G -> backbone = "
      f"{100*backbone_macs/1e9/MODEL_MACS_COUNTED_G:.1f}% of MACs, "
      f"{100*MEAS['backbone_ms']/MEAS['model_ms']:.1f}% of measured time")

    p("")
    p("=" * 78)
    p("SECTION B -- 4x4 macro-region mapping across C3 / C4 / C5")
    p("=" * 78)
    p(f"{'level':>6} {'stride':>7} {'grid HxW':>10} {'rows/4':>8} {'cols/4':>8} "
      f"{'tiles?':>7}")
    for idx in (1, 2, 3):
        h, w, _ch = geom[f"stage{idx}_out"]
        p(f"{C_LABEL[idx]:>6} {STRIDE_OF[idx]:>7} {f'{h}x{w}':>10} "
          f"{h/GRID:>8.2f} {w/GRID:>8.2f} "
          f"{'yes' if (h % GRID == 0 and w % GRID == 0) else 'NO':>7}")
    p("")
    p("the grid tiles C3 exactly (36 x 63 feature px = 288 x 504 input px per")
    p("region) but not C4 (cols 31.5) or C5 (cols 15.75). Rule used below, and")
    p("the only one examined: the mask is DEFINED on C3 and mapped upward by")
    p("2x2 OR-pooling (a C4/C5 position is executed if any child is).")
    h3, w3, _ = geom["stage1_out"]
    p("resulting single-region footprint (rows x cols, by region column):")
    for idx in (1, 2, 3):
        cells = []
        for c in range(GRID):
            m = mask_c3([(1, c)], h3, w3)
            for _ in range(idx - 1):
                m = or_pool2(m)
            rows = sum(1 for row in m if any(row))
            cols = max(sum(row) for row in m)
            cells.append(f"{rows}x{cols}")
        p(f"   {C_LABEL[idx]}: " + "  ".join(cells))
    p("-> at C4 and C5 neighbouring regions share one boundary column, so the")
    p("   per-region cost is not additive across regions even before any halo.")

    p("")
    p("=" * 78)
    p("SECTION C -- exact residual MACs still executed, per explicit mask")
    p("=" * 78)
    p("counted: branch2b on the mask, branch2a on the mask dilated by 1 px")
    p("(clipped at the border). x_l and the shortcut are dense, so reading the")
    p("branch2a input costs a gather, not a convolution. 'ideal' = regions/16.")
    p("")
    p(f"{'mask':<28} {'ideal':>6} {'executed':>9} {'halo tax':>9} "
      f"{'C3 2b/2a':>13} {'C4 2b/2a':>13} {'C5 2b/2a':>13}")
    mask_rows = []
    for name, regions in MASKS:
        macs, pl = executed_macs(regions, design, geom)
        frac = macs / res_macs
        ideal = len(regions) / (GRID * GRID)
        cells = [f"{100*pl[i]['mask']/pl[i]['full']:>5.1f}/"
                 f"{100*pl[i]['halo']/pl[i]['full']:>5.1f}%" for i in (1, 2, 3)]
        p(f"{name:<28} {100*ideal:>5.1f}% {100*frac:>8.1f}% "
          f"{100*(frac-ideal):>+8.1f}pp " + " ".join(f"{c:>13}" for c in cells))
        mask_rows.append(dict(mask=name, regions=len(regions), ideal=ideal,
                              executed_fraction=frac, executed_macs=macs))
    full_macs = executed_macs(ALL, design, geom)[0]
    empty_macs = executed_macs([], design, geom)[0]
    ok = (full_macs == res_macs and empty_macs == 0)
    p("")
    p(f"self-check: all-16 mask executes exactly the native residual MACs "
      f"({full_macs == res_macs}); empty mask executes none "
      f"({empty_macs == 0}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("mask accounting self-check failed")
    by = {r["mask"]: r["executed_fraction"] for r in mask_rows}
    p("reading: the halo plus the shared C4/C5 boundary columns cost "
      f"{100*(by['8/16 rows 0-1']-0.5):+.1f} pp")
    p(f"for a compact half mask and {100*(by['8/16 checkerboard']-0.5):+.1f} pp "
      "for a checkerboard half mask, so the same")
    p("region COUNT buys different compute. Budgets must be stated in executed")
    p("MACs (or measured time), not in regions, as the task book requires.")

    p("")
    p("=" * 78)
    p("SECTION D -- saving bound: MEASURED anchors x counted MAC split")
    p("=" * 78)
    p("measured anchors (R26 arm N, FP32 batch4, RTX 4090 D):")
    for k, v in MEAS.items():
        p(f"   {k:<16} {v:>9.3f} ms   "
          f"{100*v/MEAS['model_ms']:>5.1f}% of model GPU")
    p("measured L0 cost4 (S15 endpoint at H, same rig, one process):")
    p(f"   model GPU      {S15_MODEL_MS_B4:>9.3f} ms/batch4")
    p(f"   end-to-end     {S15_E2E_MS_B4:>9.3f} ms/batch4 "
      f"(model is {100*S15_MODEL_MS_B4/S15_E2E_MS_B4:.1f}% of it)")
    p(f"   A0 at L        {A0_MODEL_MS_B4:>9.3f} ms/batch4 model GPU")
    p("")
    share_backbone = res_macs / backbone_macs
    res_ms = MEAS["backbone_ms"] * share_backbone
    p(f"C3-C5 residual share of backbone MACs : {100*share_backbone:.1f}%")
    p("ESTIMATE (assumes backbone time is proportional to MACs; the residual")
    p("convs themselves were never timed):")
    p(f"   residual-branch time   = {res_ms:.2f} ms/batch4 "
      f"= {100*res_ms/MEAS['model_ms']:.1f}% of model GPU "
      f"= {100*res_ms/S15_E2E_MS_B4:.1f}% of end-to-end")
    p("that is the CEILING: every C3-C5 residual branch removed (the network")
    p("is then stem + C2 + shortcuts only), zero halo, zero routing, zero")
    p("gather/scatter. It is a bound on the method family, not a design point.")
    p("")
    p("ceiling per mask (same assumption, halo counted, overhead still zero):")
    p(f"{'mask':<28} {'skipped':>8} {'ms/batch4':>10} {'model GPU':>10} "
      f"{'end-to-end':>11}")
    sav_rows = []
    for row in mask_rows:
        skipped = 1.0 - row["executed_fraction"]
        ms = skipped * res_ms
        p(f"{row['mask']:<28} {100*skipped:>7.1f}% {ms:>10.2f} "
          f"{100*ms/MEAS['model_ms']:>9.1f}% {100*ms/S15_E2E_MS_B4:>10.1f}%")
        sav_rows.append(dict(mask=row["mask"], skipped_fraction=skipped,
                             ceiling_ms_batch4=ms,
                             ceiling_frac_model=ms / MEAS["model_ms"],
                             ceiling_frac_e2e=ms / S15_E2E_MS_B4))

    p("")
    p("=" * 78)
    p("SECTION E -- what survives routing and gather/scatter")
    p("=" * 78)
    surv = {}
    p("MEASURED on this codebase (R26: one encoder block restructured to 1/4")
    p(f"area; that block was {100*R26['N']['block']/R26['N']['model']:.1f}% "
      "of model GPU time):")
    for arm in ("U", "P"):
        gross = R26["N"]["B"] - R26[arm]["B"]
        net = R26["N"]["block"] - R26[arm]["block"]
        surv[arm] = net / gross
        dm = 100 * (R26[arm]["model"] / R26["N"]["model"] - 1)
        p(f"   arm {arm}: conv time saved {gross:.2f} ms, block time saved "
          f"{net:.2f} ms -> {100*surv[arm]:.0f}% survived; whole model {dm:+.1f}%")
    lo, hi = min(surv.values()), max(surv.values())
    p("the rest went to added elementwise / resampling ops inside the block.")
    p("")
    p("EXTRAPOLATION, not a measurement: apply that survival range to the")
    p("per-mask ceilings above. R26 restructured one regular dense block; this")
    p("design gathers and scatters irregular region sets in six blocks, so the")
    p("range is more likely optimistic than pessimistic.")
    p(f"{'mask':<28} {'net ms/batch4':>15} {'model GPU':>13} {'end-to-end':>13}")
    ext_rows = []
    for row in sav_rows:
        if row["skipped_fraction"] <= 0:
            continue
        a, b = lo * row["ceiling_ms_batch4"], hi * row["ceiling_ms_batch4"]
        p(f"{row['mask']:<28} {a:>6.1f} - {b:>5.1f}   "
          f"{100*a/MEAS['model_ms']:>4.1f} - {100*b/MEAS['model_ms']:>4.1f}%  "
          f"{100*a/S15_E2E_MS_B4:>4.1f} - {100*b/S15_E2E_MS_B4:>4.1f}%")
        ext_rows.append(dict(mask=row["mask"], net_ms_lo=a, net_ms_hi=b))
    pix_ratio = (768 * 1344) / (H_IN * W_IN)
    opt_d = MEAS["backbone_ms"] * (stem_macs / backbone_macs) * pix_ratio
    p("")
    p("routing cost, to be subtracted from any row above:")
    p("   option A (selector reads the C2 features already computed): ~0 ms")
    p(f"   option D (extra stem+C2 pass on the L canvas; ESTIMATE = backbone "
      f"{MEAS['backbone_ms']:.1f} ms")
    p(f"            x stem+C2 MAC share {100*stem_macs/backbone_macs:.1f}% x "
      f"pixel ratio {pix_ratio:.3f}): {opt_d:.2f} ms/batch4")
    p(f"   option B (full L pass, MEASURED A0): {A0_MODEL_MS_B4:.2f} ms/batch4 "
      f"> ceiling {res_ms:.2f} ms -> never nets positive")
    p("")
    p("scale of the target, MEASURED (L0 cost4):")
    p(f"   H -> L model GPU gap            : {S15_MODEL_MS_B4 - A0_MODEL_MS_B4:.1f} "
      f"ms/batch4 ({100*(1-A0_MODEL_MS_B4/S15_MODEL_MS_B4):.1f}% of the H model)")
    p(f"   R27-P vs S15                    : {100*(P_OVER_S15_MODEL-1):+.1f}% "
      f"model GPU, {100*(P_OVER_S15_E2E-1):+.1f}% end-to-end")
    p(f"   this design, absolute ceiling   : {-100*res_ms/MEAS['model_ms']:+.1f}% "
      f"model GPU, {-100*res_ms/S15_E2E_MS_B4:+.1f}% end-to-end")
    half = [r for r in sav_rows if r["mask"] == "8/16 rows 0-1"][0]
    p(f"   this design, compact half mask  : "
      f"{-100*half['ceiling_frac_model']:+.1f}% model GPU before overhead, "
      f"{-100*lo*half['ceiling_frac_model']:+.1f} to "
      f"{-100*hi*half['ceiling_frac_model']:+.1f}% after the R26 survival range")

    p("")
    p("=" * 78)
    p("SECTION F -- BN inventory inside the design object")
    p("=" * 78)
    for r in design:
        p(f"{r['label']} block{r['block']}: {r['bns']} BN "
          f"(branch2a.norm, branch2b.norm"
          f"{', short.conv.norm' if r['s_macs'] else ''})")
    p(f"total BN modules inside C3-C5             : {sum(r['bns'] for r in design)}")
    p(f"  of which on the MASKED residual path    : "
      f"{sum(2 for r in design)}  (branch2a.norm + branch2b.norm)")
    p("")
    p("project contract asserts these are LIVE nn.BatchNorm2d, not")
    p("FrozenBatchNorm2d:")
    p("   src/sparse_rtdetr/baseline/training_v2a_contract.py:1138-1139")
    p('       "freeze_at": -1, "freeze_norm": False')
    p("   src/sparse_rtdetr/baseline/training_v2b.py:248")
    p("       PResNet(depth=18, pretrained=False, freeze_at=-1, freeze_norm=False, ...)")
    p("   L0 report section 2: 69 BN modules actually updated per training forward")
    p("   (23 backbone + 43 encoder + 3 decoder input_proj)")

    print("\n".join(out))
    if args.json:
        summary = dict(
            residual_macs_G=res_macs / 1e9,
            shortcut_macs_G=short_macs / 1e9,
            backbone_macs_G=backbone_macs / 1e9,
            stem_and_C2_macs_G=stem_macs / 1e9,
            backbone_conv_weights=conv_params,
            residual_share_of_backbone=share_backbone,
            model_macs_counted_G=MODEL_MACS_COUNTED_G,
            residual_share_of_model_macs=res_macs / 1e9 / MODEL_MACS_COUNTED_G,
            best_case_ms_batch4=res_ms,
            best_case_frac_model=res_ms / MEAS["model_ms"],
            best_case_frac_e2e=res_ms / S15_E2E_MS_B4,
            measured=dict(MEAS, S15_model_ms_batch4=S15_MODEL_MS_B4,
                          S15_e2e_ms_batch4=S15_E2E_MS_B4,
                          A0_model_ms_batch4=A0_MODEL_MS_B4,
                          P_over_S15_model=P_OVER_S15_MODEL,
                          P_over_S15_e2e=P_OVER_S15_E2E),
            r26_survival=dict(U=surv["U"], P=surv["P"]),
            routing_option_D_ms_estimate=opt_d,
            masks=mask_rows,
            ceilings=sav_rows,
            extrapolated_net=ext_rows,
            blocks=[{k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in r.items()} for r in design],
        )
        with open(os.path.join(HERE, "check_backbone_shapes.result.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("\n[written] check_backbone_shapes.result.json")


if __name__ == "__main__":
    sys.exit(main())
