"""Synthetic checks for the r13 selection-ranking prototype (no data, no GPU, no chain).

Verifies only that the formulas implement the declared mechanism:
 1. gradient direction: the matched token's max-class logit is pushed DOWN in the
    loss (i.e. raised in selection), the partner's is pushed UP (suppressed), every
    other entry is exactly zero, and the values equal the declared +-1/N;
 2. an already-satisfied pair contributes exactly zero loss and zero gradient;
 3. an instance with an empty eligible set contributes exactly zero (no gradient);
 4. cross-GT conflict: a token eligible for two instances partners at most one of
    them, chosen by the declared dedup rule, and is dropped from the other;
 5. A1/A2 parity: identical eligibility, dedup, pair counts and margins; the only
    difference is which eligible token is chosen; A1 is reproducible for a seed;
 6. margin monotonicity in the instance scale, with the declared clamp;
 7. the added gradient touches at most 2N tokens and no other entry.

usage: CUDA_VISIBLE_DEVICES= <python> -I test_selection_rank.py [--out out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import selection_rank as sr  # noqa: E402

WEIGHTS = {"cost_bbox": 5.0, "cost_giou": 2.0}
EVIDENCE = {}


def scene(seed, n_tokens, gt_boxes, selected_high):
    """Deterministic synthetic scene: logits with a designed score ordering."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, n_tokens, 3, generator=generator) * 0.1
    boxes = torch.rand(1, n_tokens, 4, generator=generator) * 0.3 + 0.2
    for token, value in selected_high.items():
        logits[0, token, :] = value
    return logits, boxes, torch.tensor(gt_boxes)


def test_gradient_direction():
    # tokens 0..3 with the score ordering 0 > 1 > 2 > 3 ; k=2 selects {0,1}
    logits, boxes, gt = scene(11, 4, [[0.5, 0.5, 0.02, 0.02], [0.8, 0.8, 0.02, 0.02]], {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0})
    # token 0 is the matched token of GT0; token 2 is the best localizer for GT0 among the unselected
    boxes[0, 0] = torch.tensor([0.50, 0.50, 0.020, 0.020])
    boxes[0, 2] = torch.tensor([0.51, 0.51, 0.021, 0.021])
    boxes[0, 3] = torch.tensor([0.90, 0.90, 0.020, 0.020])
    matched = [[(0, 0)]]
    value, evidence = sr.loss(logits, boxes, [gt], matched, k=2, mode="A2", weights=WEIGHTS)
    reference, expected, pairs = sr.check_gradient_direction(logits, boxes, [gt], matched, k=2, mode="A2", weights=WEIGHTS)
    logits2 = logits.clone().requires_grad_(True)
    value2, _ = sr.loss(logits2, boxes, [gt], matched, k=2, mode="A2", weights=WEIGHTS)
    value2.backward()
    got = logits2.grad
    assert pairs and len(pairs) == 1, "expected exactly one pair, got %d" % len(pairs)
    assert evidence["pairs"] == 1 and evidence["active_pairs"] == 1, evidence
    n = 1.0
    partner = evidence["per_image"][0]["records"][0]["partner"]
    assert partner == 2, "partner should be the best eligible localizer (token 2), got %s" % partner
    assert float(got[0, 0].min()) < 0 and abs(float(got[0, 0].min()) + 1.0 / n) < 1e-6, got[0, 0]
    assert float(got[0, partner].max()) > 0 and abs(float(got[0, partner].max()) - 1.0 / n) < 1e-6, got[0, partner]
    assert float(got[0, 1].abs().sum()) == 0.0 and float(got[0, 3].abs().sum()) == 0.0, "untouched tokens have gradients"
    assert torch.allclose(got, expected, atol=1e-6) and abs(float(value) - reference) < 1e-6
    EVIDENCE["gradient_direction"] = {"pairs": evidence["pairs"], "partner": partner,
                                      "matched_grad_min": float(got[0, 0].min()),
                                      "partner_grad_max": float(got[0, partner].max()),
                                      "touched_tokens": evidence["tokens_with_added_gradient"]}


def test_satisfied_pair_is_exactly_zero():
    logits, boxes, gt = scene(12, 4, [[0.5, 0.5, 0.02, 0.02], [0.8, 0.8, 0.02, 0.02]], {0: 5.0, 1: 4.0, 2: -5.0, 3: -6.0})
    boxes[0, 0] = torch.tensor([0.50, 0.50, 0.020, 0.020])
    boxes[0, 2] = torch.tensor([0.505, 0.505, 0.020, 0.020])
    matched = [[(0, 0)]]
    logits2 = logits.clone().requires_grad_(True)
    value, evidence = sr.loss(logits2, boxes, [gt], matched, k=2, mode="A2", weights=WEIGHTS)
    value.backward()
    assert evidence["pairs"] == 1 and evidence["active_pairs"] == 0, evidence
    assert float(value) == 0.0, float(value)
    assert float(logits2.grad.abs().sum()) == 0.0, "a satisfied pair produced a gradient"
    EVIDENCE["satisfied_pair_zero"] = {"pairs": evidence["pairs"], "active": evidence["active_pairs"], "loss": float(value)}


def test_empty_partner_is_exactly_zero():
    logits, boxes, gt = scene(13, 3, [[0.5, 0.5, 0.02, 0.02]], {0: 5.0, 1: 4.0, 2: 3.0})
    boxes[0, 1] = torch.tensor([0.9, 0.9, 0.5, 0.5])      # far outside the size band
    boxes[0, 2] = torch.tensor([0.9, 0.9, 0.5, 0.5])
    matched = [[(0, 0)]]
    logits2 = logits.clone().requires_grad_(True)
    value, evidence = sr.loss(logits2, boxes, [gt], matched, k=1, mode="A2", weights=WEIGHTS)
    value.backward()
    assert evidence["pairs"] == 0 and evidence["per_image"][0]["records"][0]["eligible_count"] == 0, evidence
    assert float(value) == 0.0 and float(logits2.grad.abs().sum()) == 0.0, "empty partner produced a gradient"
    EVIDENCE["empty_partner_zero"] = {"pairs": 0, "loss": float(value)}


def test_cross_gt_conflict_uses_the_declared_dedup():
    # token 2 is eligible for both GT0 (size error small) and GT1 (size error larger)
    logits, boxes, gt = scene(14, 5, [[0.5, 0.5, 0.02, 0.02], [0.8, 0.8, 0.04, 0.04]],
                              {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0})
    boxes[0, 0] = torch.tensor([0.50, 0.50, 0.020, 0.020])
    boxes[0, 1] = torch.tensor([0.80, 0.80, 0.040, 0.040])
    boxes[0, 2] = torch.tensor([0.51, 0.51, 0.020, 0.020])   # nearest to GT0 by size
    boxes[0, 3] = torch.tensor([0.79, 0.79, 0.040, 0.040])
    matched = [[(0, 0), (1, 1)]]
    info = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (1, 1)], k=2, mode="A2", weights=WEIGHTS)
    owners = {g: list(pool) for g, pool in info["eligible"].items()}
    records = {r["gt"]: r for r in info["records"]}
    assert records[0]["partner"] == 2, records
    assert records[1]["partner"] == 3, records
    assert 2 not in owners.get(1, []), "the contested token stayed eligible for the second instance"
    EVIDENCE["cross_gt_conflict"] = {"gt0_partner": records[0]["partner"], "gt1_partner": records[1]["partner"],
                                     "candidate_counts": info["candidate_counts"], "eligible_counts": info["eligible_counts"]}


def test_a1_a2_share_eligibility_counts_and_margins():
    logits, boxes, gt = scene(15, 8, [[0.5, 0.5, 0.02, 0.02], [0.8, 0.8, 0.06, 0.06], [0.3, 0.7, 0.03, 0.03]],
                              {0: 5.0, 1: 4.5, 2: 4.0, 3: 3.5, 4: 3.0, 5: 2.5, 6: 2.0, 7: 1.5})
    for token, box in ((0, [0.50, 0.50, 0.020, 0.020]), (7, [0.80, 0.80, 0.060, 0.060])):
        boxes[0, token] = torch.tensor(box)
    boxes[0, 3] = torch.tensor([0.51, 0.51, 0.020, 0.020])
    boxes[0, 5] = torch.tensor([0.81, 0.81, 0.061, 0.061])
    matched = [[(0, 0), (7, 1)]]
    generator = torch.Generator().manual_seed(1234)
    a2 = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (7, 1)], k=2, mode="A2", weights=WEIGHTS)
    a1 = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (7, 1)], k=2, mode="A1", weights=WEIGHTS,
                    generator=torch.Generator().manual_seed(1234))
    a1_again = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (7, 1)], k=2, mode="A1", weights=WEIGHTS,
                          generator=torch.Generator().manual_seed(1234))
    assert a1["eligible_counts"] == a2["eligible_counts"], (a1["eligible_counts"], a2["eligible_counts"])
    assert a1["candidate_counts"] == a2["candidate_counts"], (a1["candidate_counts"], a2["candidate_counts"])
    assert [r["margin"] for r in a1["records"]] == [r["margin"] for r in a2["records"]]
    assert sum(1 for r in a1["records"] if r["partner"] is not None) == sum(1 for r in a2["records"] if r["partner"] is not None)
    assert [r["partner"] for r in a1["records"]] == [r["partner"] for r in a1_again["records"]], "A1 is not reproducible"
    for record in a1["records"]:
        if record["partner"] is not None:
            assert record["eligible_count"] > 0
    EVIDENCE["a1_a2_parity"] = {"eligible_counts": a1["eligible_counts"], "pairs": len(a1["records"]),
                                "a1_partners": [r["partner"] for r in a1["records"]],
                                "a2_partners": [r["partner"] for r in a2["records"]],
                                "margins": [r["margin"] for r in a2["records"]]}


def test_margin_is_monotone_in_scale_with_clamp():
    values = {}
    for side in (0.01, 0.02, 0.05, 0.10, 0.50):
        logits, boxes, gt = scene(16, 3, [[0.5, 0.5, side, side]], {0: 5.0, 1: 4.0, 2: 3.0})
        boxes[0, 0] = torch.tensor([0.5, 0.5, side, side])
        boxes[0, 1] = torch.tensor([0.51, 0.51, side, side])
        info = sr.pairing(logits[0], boxes[0], gt, [(0, 0)], k=1, mode="A2", weights=WEIGHTS)
        values[side] = info["records"][0]["margin"]
    ordered = [values[s] for s in (0.01, 0.02, 0.05, 0.10, 0.50)]
    assert ordered == sorted(ordered), ordered
    assert ordered[0] == sr.DECLARATION["margin_m0"] * sr.DECLARATION["margin_clamp"][0], ordered
    assert ordered[-1] == sr.DECLARATION["margin_m0"] * sr.DECLARATION["margin_clamp"][1], ordered
    EVIDENCE["margin_monotone"] = {str(k): v for k, v in values.items()}


def test_added_gradient_touches_at_most_two_tokens_per_pair():
    logits, boxes, gt = scene(17, 6, [[0.5, 0.5, 0.02, 0.02], [0.8, 0.8, 0.04, 0.04]],
                              {0: 5.0, 1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.5})
    boxes[0, 0] = torch.tensor([0.50, 0.50, 0.020, 0.020])
    boxes[0, 1] = torch.tensor([0.80, 0.80, 0.040, 0.040])
    boxes[0, 2] = torch.tensor([0.51, 0.51, 0.021, 0.021])
    boxes[0, 3] = torch.tensor([0.81, 0.81, 0.041, 0.041])
    matched = [[(0, 0), (1, 1)]]
    logits2 = logits.clone().requires_grad_(True)
    value, evidence = sr.loss(logits2, boxes, [gt], matched, k=2, mode="A2", weights=WEIGHTS)
    value.backward()
    nonzero = int((logits2.grad.abs().sum(-1) > 0).sum())
    assert evidence["pairs"] == 2, evidence
    assert nonzero <= 2 * evidence["pairs"], (nonzero, evidence["pairs"])
    assert nonzero >= 1
    EVIDENCE["gradient_scope"] = {"pairs": evidence["pairs"], "tokens_with_gradient": nonzero,
                                  "declared_max": 2 * evidence["pairs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    failures = []
    for name in sorted(n for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            globals()[name]()
            print("PASS", name, flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append((name, repr(error), traceback.format_exc()))
            print("FAIL", name, repr(error), flush=True)
            print(traceback.format_exc(), flush=True)
    payload = {"status": "PASS" if not failures else "FAIL",
               "scope": "synthetic logits/boxes only; verifies that the formulas implement the declared mechanism",
               "declaration": sr.DECLARATION, "evidence": EVIDENCE, "failures": failures,
               "elapsed_seconds": round(time.monotonic() - started, 3),
               "cuda_initialized": bool(torch.cuda.is_initialized())}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": failures, "evidence": EVIDENCE}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
