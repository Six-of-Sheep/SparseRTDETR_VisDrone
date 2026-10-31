"""Synthetic checks for the r14 corrected (quality-ordered) selection-ranking prototype.

Verifies only that the corrected formulas implement the declared mechanism:
 1. direction follows QUALITY, not selection status: in a scene where the selected
    (matched) token is the WORSE localizer and an unselected token is measurably
    better, the loss raises the unselected better token and lowers the selected
    worse one, with values +-1/N on their max-class logits and zero elsewhere;
 2. no padding: if the owner is already the better localizer, or no token covers the
    instance above the floor, or the only candidates fail the spatial band, the
    instance contributes zero pairs, exactly zero loss and no gradient;
 3. neighbour protection: a competitor that overlaps ANOTHER instance above tau is
    excluded; the same scene with that neighbour moved away admits it;
 4. exclusivity/dedup: a token eligible for two instances partners at most one, by
    the declared smallest-relative-size-error rule;
 5. a control scene with TWO legal competitors where A1 (uniform draw) and A2
    (highest quality) demonstrably choose differently, with identical eligibility,
    counts and margins;
 6. an already-satisfied pair contributes exactly zero loss and zero gradient;
 7. margin monotonicity in the instance scale with the declared clamp;
 8. the added gradient touches at most 2N tokens and no other entry;
 9. the quality measure is the predicted-box/GT IoU.

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

EVIDENCE = {}


def scene(seed, n_tokens, gt_boxes, scores):
    """Deterministic synthetic scene: designed max-class logits, random boxes to override."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, n_tokens, 3, generator=generator) * 0.1
    boxes = torch.rand(1, n_tokens, 4, generator=generator) * 0.3 + 0.2
    for token, value in scores.items():
        logits[0, token, :] = value
    return logits, boxes, torch.tensor(gt_boxes)


def set_box(boxes, token, values):
    boxes[0, token] = torch.tensor([float(v) for v in values])


def gradient_of(logits, boxes, gt, matched, *, k, mode="A2", generator=None):
    logits = logits.clone().requires_grad_(True)
    value, evidence = sr.loss(logits, boxes, [gt], matched, k=k, mode=mode, generator=generator)
    value.backward()
    return value, evidence, logits.grad


def test_direction_follows_quality_not_selection():
    """The demanded example: selected token worse, unselected token better."""
    gt = torch.tensor([[0.50, 0.50, 0.02, 0.02]])
    logits, boxes, gt = scene(11, 3, gt.tolist(), {0: 5.0, 1: 1.0, 2: 4.0})   # k=2 selects {0,2}
    set_box(boxes, 0, [0.510, 0.510, 0.020, 0.020])   # owner: worse localizer
    set_box(boxes, 1, [0.502, 0.502, 0.020, 0.020])   # unselected competitor: better localizer
    set_box(boxes, 2, [0.900, 0.900, 0.020, 0.020])   # far, no quality
    matched = [[(0, 0)]]
    value, evidence, grad = gradient_of(logits, boxes, gt, matched, k=2)
    record = evidence["per_image"][0]["records"][0]
    assert evidence["pairs"] == 1 and evidence["active_pairs"] == 1, evidence
    assert record["owner"] == 0 and record["better"] == 1 and record["worse"] == 0, record
    assert record["quality_better"] > record["quality_worse"] + sr.DECLARATION["constants"]["q_margin"], record
    assert float(grad[0, 1].min()) < 0 and abs(float(grad[0, 1].min()) + 1.0) < 1e-6, grad[0, 1]
    assert float(grad[0, 0].max()) > 0 and abs(float(grad[0, 0].max()) - 1.0) < 1e-6, grad[0, 0]
    assert float(grad[0, 2].abs().sum()) == 0.0, "an untouched token received a gradient"
    assert abs(float(value) - (record["margin"] - record["score_gap"])) < 1e-6, (float(value), record)
    EVIDENCE["direction_follows_quality"] = {
        "owner": record["owner"], "better": record["better"],
        "quality_owner": round(record["quality_worse"], 6), "quality_better": round(record["quality_better"], 6),
        "better_grad": float(grad[0, 1].min()), "owner_grad": float(grad[0, 0].max()),
        "loss": round(float(value), 6)}


def test_no_pair_when_the_owner_is_already_better():
    logits, boxes, gt = scene(12, 3, [[0.50, 0.50, 0.02, 0.02]], {0: 5.0, 1: 4.0, 2: 1.0})
    set_box(boxes, 0, [0.500, 0.500, 0.020, 0.020])   # owner: nearly perfect
    set_box(boxes, 1, [0.520, 0.520, 0.020, 0.020])   # worse, must not be used
    set_box(boxes, 2, [0.900, 0.900, 0.020, 0.020])
    value, evidence, grad = gradient_of(logits, boxes, gt, [[(0, 0)]], k=1)
    assert evidence["pairs"] == 0 and evidence["per_image"][0]["records"][0]["eligible_count"] == 0, evidence
    assert float(value) == 0.0 and float(grad.abs().sum()) == 0.0, "an already-correct owner produced a term"
    EVIDENCE["no_pair_owner_better"] = {"pairs": 0, "loss": float(value)}


def test_no_pair_without_coverage_or_inside_the_size_band():
    logits, boxes, gt = scene(13, 4, [[0.50, 0.50, 0.02, 0.02]], {0: 5.0, 1: 4.0, 2: 3.0, 3: 1.0})
    set_box(boxes, 0, [0.510, 0.510, 0.020, 0.020])   # owner worse (q ~ 0.14)
    set_box(boxes, 1, [0.560, 0.560, 0.020, 0.020])   # below the coverage floor
    set_box(boxes, 2, [0.500, 0.500, 0.500, 0.500])   # far outside the size band
    set_box(boxes, 3, [0.900, 0.900, 0.020, 0.020])
    value, evidence, grad = gradient_of(logits, boxes, gt, [[(0, 0)]], k=1)
    assert evidence["pairs"] == 0 and evidence["per_image"][0]["records"][0]["eligible_count"] == 0, evidence
    assert float(value) == 0.0 and float(grad.abs().sum()) == 0.0, "an uncovered candidate produced a term"
    EVIDENCE["no_pair_without_coverage"] = {"pairs": 0, "loss": float(value)}


def test_neighbour_protection_excludes_and_admits():
    """Same competitor, two variants: only the neighbour's position differs."""
    matched = [[(0, 0)]]
    near = torch.tensor([[0.500, 0.500, 0.02, 0.02], [0.512, 0.512, 0.03, 0.03]])
    far = torch.tensor([[0.500, 0.500, 0.02, 0.02], [0.900, 0.900, 0.03, 0.03]])
    results = {}
    for label, gt in (("neighbour_overlapping", near), ("neighbour_moved_away", far)):
        logits, boxes, _ = scene(14, 3, gt.tolist(), {0: 5.0, 1: 4.0, 2: 1.0})
        set_box(boxes, 0, [0.510, 0.510, 0.020, 0.020])
        set_box(boxes, 1, [0.500, 0.500, 0.021, 0.021])
        set_box(boxes, 2, [0.900, 0.900, 0.020, 0.020])
        value, evidence, grad = gradient_of(logits, boxes, gt, matched, k=1)
        info = evidence["per_image"][0]
        results[label] = {"pairs": evidence["pairs"], "eligible": info["eligible_counts"].get(0, 0),
                          "loss": round(float(value), 6), "grad": float(grad.abs().sum())}
    assert results["neighbour_overlapping"]["pairs"] == 0 and results["neighbour_overlapping"]["eligible"] == 0, results
    assert results["neighbour_moved_away"]["pairs"] == 1 and results["neighbour_moved_away"]["eligible"] == 1, results
    assert results["neighbour_overlapping"]["loss"] == 0.0 and results["neighbour_overlapping"]["grad"] == 0.0
    EVIDENCE["neighbour_protection"] = results


def two_partner_scene(seed):
    """GT0 (owner worse, competitors blocked) and GT1 with TWO legal competitors."""
    gt = torch.tensor([[0.50, 0.50, 0.02, 0.02], [0.80, 0.80, 0.04, 0.04]])
    logits, boxes, gt = scene(seed, 4, gt.tolist(), {0: 5.0, 1: 4.5, 2: 4.0, 3: 1.0})
    set_box(boxes, 0, [0.510, 0.510, 0.020, 0.020])   # owner of GT0, worse
    set_box(boxes, 1, [0.810, 0.810, 0.040, 0.040])   # owner of GT1, worse (q ~ 0.39)
    set_box(boxes, 2, [0.800, 0.800, 0.032, 0.032])   # competitor for GT1 (q ~ 0.64)
    set_box(boxes, 3, [0.802, 0.802, 0.040, 0.040])   # competitor for GT1, best (q ~ 0.82)
    return logits, boxes, gt


def test_exclusivity_when_a_token_is_eligible_for_two_instances():
    logits, boxes, gt = two_partner_scene(15)
    info = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (1, 1)], k=2, mode="A2")
    records = {r["gt"]: r for r in info["records"]}
    assert info["eligible_counts"][0] == 0, info["eligible_counts"]     # its only candidate went to GT1
    assert info["eligible_counts"][1] == 2, info["eligible_counts"]     # two legal competitors
    assert records[1]["partner"] == 3 and records[1]["better"] == 3, records[1]
    EVIDENCE["exclusivity_two_instances"] = {"eligible_counts": info["eligible_counts"],
                                             "gt1_partner": records[1]["partner"],
                                             "owner_gt1": records[1]["owner"]}


def test_a1_and_a2_can_choose_differently_with_two_legal_partners():
    logits, boxes, gt = two_partner_scene(16)
    a2 = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (1, 1)], k=2, mode="A2")
    choices = set()
    for seed in range(8):
        a1 = sr.pairing(logits[0], boxes[0], gt, [(0, 0), (1, 1)], k=2, mode="A1",
                        generator=torch.Generator().manual_seed(seed))
        assert a1["eligible_counts"] == a2["eligible_counts"], (a1["eligible_counts"], a2["eligible_counts"])
        assert [r["margin"] for r in a1["records"]] == [r["margin"] for r in a2["records"]]
        assert [r["partner"] for r in a1["records"] if r["partner"] is not None] == [r["partner"] for r in a1["records"] if r["partner"] is not None]
        for record in a1["records"]:
            if record["partner"] is not None:
                assert record["partner"] in a1["eligible"][record["gt"]], record
                choices.add(record["partner"])
    a2_partner = [r["partner"] for r in a2["records"] if r["partner"] is not None]
    assert len(choices) >= 2, ("A1 never varied over the seeds", choices)
    assert any(c != a2_partner[-1] for c in choices), ("A1 never differed from A2", choices, a2_partner)
    EVIDENCE["two_legal_partners"] = {"a2_partner": a2_partner[-1], "a1_choices_over_8_seeds": sorted(choices),
                                      "eligible_counts": a2["eligible_counts"]}


def test_satisfied_pair_is_exactly_zero():
    logits, boxes, gt = scene(17, 3, [[0.50, 0.50, 0.02, 0.02]], {0: 1.0, 1: 5.0, 2: 0.5})
    set_box(boxes, 0, [0.510, 0.510, 0.020, 0.020])   # owner, worse, low score
    set_box(boxes, 1, [0.500, 0.500, 0.020, 0.020])   # better competitor, high score (already ranked above)
    set_box(boxes, 2, [0.900, 0.900, 0.020, 0.020])
    value, evidence, grad = gradient_of(logits, boxes, gt, [[(0, 0)]], k=1)
    assert evidence["pairs"] == 1 and evidence["active_pairs"] == 0, evidence
    assert float(value) == 0.0 and float(grad.abs().sum()) == 0.0, "a satisfied pair produced a gradient"
    EVIDENCE["satisfied_pair_zero"] = {"pairs": evidence["pairs"], "active": evidence["active_pairs"],
                                       "loss": float(value)}


def test_margin_is_monotone_in_scale_with_clamp():
    values = {}
    for side in (0.01, 0.02, 0.05, 0.10, 0.50):
        logits, boxes, gt = scene(18, 3, [[0.5, 0.5, side, side]], {0: 5.0, 1: 4.0, 2: 1.0})
        set_box(boxes, 0, [0.5 + 0.15 * side, 0.5, side, side])          # owner: shifted
        set_box(boxes, 1, [0.5, 0.5, side * 1.05, side * 1.05])          # competitor: nearly exact
        set_box(boxes, 2, [0.95, 0.95, side, side])
        info = sr.pairing(logits[0], boxes[0], gt, [(0, 0)], k=1, mode="A2")
        values[side] = info["records"][0]["margin"]
    ordered = [values[s] for s in (0.01, 0.02, 0.05, 0.10, 0.50)]
    clamp = sr.DECLARATION["constants"]["margin_clamp"]
    assert ordered == sorted(ordered), ordered
    assert ordered[0] == sr.DECLARATION["constants"]["m0"] * clamp[0], ordered
    assert ordered[-1] == sr.DECLARATION["constants"]["m0"] * clamp[1], ordered
    EVIDENCE["margin_monotone"] = {str(k): v for k, v in values.items()}


def test_added_gradient_touches_at_most_two_tokens_per_pair():
    logits, boxes, gt = two_partner_scene(19)
    value, evidence, grad = gradient_of(logits, boxes, gt, [[(0, 0), (1, 1)]], k=2)
    nonzero = int((grad.abs().sum(-1) > 0).sum())
    assert evidence["pairs"] == 1, evidence          # GT0 has no eligible competitor in this scene
    assert nonzero <= 2 * evidence["pairs"], (nonzero, evidence["pairs"])
    assert nonzero == 2, (nonzero, evidence["pairs"])
    EVIDENCE["gradient_scope"] = {"pairs": evidence["pairs"], "tokens_with_gradient": nonzero,
                                  "declared_max": 2 * evidence["pairs"]}


def test_quality_measure_is_the_box_iou():
    gt = torch.tensor([[0.50, 0.50, 0.02, 0.02]])
    logits, boxes, gt = scene(20, 3, gt.tolist(), {0: 5.0, 1: 4.0, 2: 1.0})
    set_box(boxes, 0, [0.510, 0.510, 0.020, 0.020])
    set_box(boxes, 1, [0.500, 0.500, 0.020, 0.020])
    set_box(boxes, 2, [0.900, 0.900, 0.020, 0.020])
    info = sr.pairing(logits[0], boxes[0], gt, [(0, 0)], k=1, mode="A2")
    record = info["records"][0]
    boxes_xyxy = sr._xyxy(boxes[0])
    gt_xyxy = sr._xyxy(gt)
    iou = sr.iou_matrix(boxes_xyxy[0:1], gt_xyxy[0:1])[0, 0]
    assert abs(record["quality_worse"] - float(iou)) < 1e-9, (record, float(iou))
    assert abs(record["quality_worse"] - 1.0 / 7.0) < 1e-6, record["quality_worse"]   # offset 0.01 of a 0.02 box
    EVIDENCE["quality_is_iou"] = {"owner_quality": round(record["quality_worse"], 6),
                                  "expected": round(1.0 / 7.0, 6)}


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
               "scope": ("synthetic logits/boxes only; verifies that the corrected formulas implement the "
                         "declared quality-ordered mechanism (direction, no-padding zeros, neighbour protection, "
                         "exclusivity, A1/A2 divergence with two legal partners, margin, gradient scope)"),
               "declaration": sr.DECLARATION, "evidence": EVIDENCE, "failures": failures,
               "elapsed_seconds": round(time.monotonic() - started, 3),
               "cuda_initialized": bool(torch.cuda.is_initialized())}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": [f[0] for f in failures], "evidence": EVIDENCE},
                     ensure_ascii=False))
    if failures:
        raise SystemExit(1)
    print("ALL-OK")


if __name__ == "__main__":
    main()
