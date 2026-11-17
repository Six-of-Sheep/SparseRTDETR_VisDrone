"""TASK_MIX_SCALE_001 read-only supervision-by-canvas description (no torch, no CUDA, no model files).

Compares, update by update, the MIX formal ledger with the two historical seed0 ledgers that trained the same
window range at a single canvas (R25 formal-001 A0 = L, S15 formal-002 = H). The A0 ledger records GT identity
digests but no GT count, so counts are compared only where the other canvas is H (S15 ledger); identities are
compared for both canvases. Also checks that sample order / augmentation seeds agree across the three ledgers.

usage: python3 supervision_by_canvas.py MIX_LEDGER A0_LEDGER S15_LEDGER OUT_JSON
"""
import hashlib
import json
import sys

mix_p, a0_p, s15_p, out_p = sys.argv[1:5]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p, need_gt):
    rows, first_keys = {}, None
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            inp = r.get("input") or {}
            if first_keys is None:
                first_keys = {"top": sorted(r), "input": sorted(inp)}
            u = inp.get("update", r.get("update"))
            g = r.get("gt_count", inp.get("gt_count"))
            if u is None or (need_gt and g is None):
                raise SystemExit("missing update/gt_count in %s: %s" % (p, first_keys))
            if u in rows:
                raise SystemExit("duplicate update %s in %s" % (u, p))
            rows[u] = {"canvas": r.get("canvas"), "gt": g, "epoch": inp.get("epoch"),
                       "image_ids": inp.get("image_ids_sha256"), "gt_ids": inp.get("gt_ids_sha256"),
                       "seeds": inp.get("augmentation_seeds_sha256")}
    return rows, first_keys


mix, mix_keys = load(mix_p, True)
a0, a0_keys = load(a0_p, False)
s15, s15_keys = load(s15_p, True)
if not (set(mix) <= set(a0) and set(mix) <= set(s15)):
    raise SystemExit("MIX updates are not covered by both historical ledgers")


def blank():
    return {"updates": 0, "gt_used": 0, "mix_row_equals_same_canvas_ledger": 0,
            "windows_gt_ids_differ_from_other_canvas": 0,
            "gt_other_canvas": 0, "windows_gt_count_differs_from_other_canvas": 0}


groups = {"L": blank(), "H": blank()}
phases = {"e061_117": {"L": blank(), "H": blank()}, "e118_120": {"L": blank(), "H": blank()}}
order_equal = seeds_equal = 0
totals = {"MIX_actual": 0, "S15_all_H_same_windows": 0}
for u in sorted(mix):
    m, x, y = mix[u], a0[u], s15[u]
    c = m["canvas"]
    if c not in ("L", "H"):
        raise SystemExit("unknown canvas at update %s: %r" % (u, c))
    order_equal += m["image_ids"] == x["image_ids"] == y["image_ids"]
    seeds_equal += m["seeds"] == x["seeds"] == y["seeds"]
    if c == "L":
        same = m["gt_ids"] == x["gt_ids"]
        other = y
    else:
        same = m["gt_ids"] == y["gt_ids"] and m["gt"] == y["gt"]
        other = x
    for g in (groups[c], phases["e061_117" if m["epoch"] <= 117 else "e118_120"][c]):
        g["updates"] += 1
        g["gt_used"] += m["gt"]
        g["mix_row_equals_same_canvas_ledger"] += same
        g["windows_gt_ids_differ_from_other_canvas"] += m["gt_ids"] != other["gt_ids"]
        if other["gt"] is None or g["gt_other_canvas"] is None:
            g["gt_other_canvas"] = None
            g["windows_gt_count_differs_from_other_canvas"] = None
        else:
            g["gt_other_canvas"] += other["gt"]
            g["windows_gt_count_differs_from_other_canvas"] += m["gt"] != other["gt"]
    totals["MIX_actual"] += m["gt"]
    totals["S15_all_H_same_windows"] += y["gt"]

for g in list(groups.values()) + [v for p in phases.values() for v in p.values()]:
    if g["gt_other_canvas"]:
        g["used_minus_other"] = g["gt_used"] - g["gt_other_canvas"]
        g["used_over_other"] = g["gt_used"] / g["gt_other_canvas"]
totals["MIX_minus_S15"] = totals["MIX_actual"] - totals["S15_all_H_same_windows"]
totals["A0_all_L_same_windows"] = None

out = {"rows": len(mix), "first_update": min(mix), "last_update": max(mix),
       "image_order_equal_three_ledgers": order_equal, "augmentation_seeds_equal_three_ledgers": seeds_equal,
       "by_canvas_used": groups, "by_phase": phases, "totals": totals,
       "inputs": {k: {"path": p, "sha256": sha(p)} for k, p in (("MIX", mix_p), ("A0_L", a0_p), ("S15_H", s15_p))},
       "ledger_keys": {"MIX": mix_keys, "A0_L": a0_keys, "S15_H": s15_keys},
       "note": ("gt counts are positive GT after native Resize/Sanitize per 16-image update; the A0 ledger has no "
                "gt_count, so L-canvas counts exist only for MIX L rows; descriptive only")}
with open(out_p, "x") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
