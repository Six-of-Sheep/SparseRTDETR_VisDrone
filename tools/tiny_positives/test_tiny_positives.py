"""Unit checks for the tiny-band extra-positive term (r19 round; CPU, no CUDA).

Every check uses the frozen release's own criterion (the native ``loss_boxes`` of the vendored
parent class) as the reference arithmetic, so the augmented-set identity is verified against the
real loss and not against a reimplementation.

usage: <frozen python> -I tools/test_tiny_positives.py [--source-root <release>/source]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = Path(sys.argv[sys.argv.index("--source-root") + 1]).resolve() if "--source-root" in sys.argv \
    else Path(os.environ.get("R18_RELEASE_ROOT", HERE.parents[1]))
for _p in (ROOT, ROOT / "src", ROOT / "vendor/rtdetrv2_pytorch/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sparse_rtdetr.protocol_r18.model import Config, build   # noqa: E402
from sparse_rtdetr.protocol_r18 import tiny_positives as tp   # noqa: E402
from sparse_rtdetr.protocol_r18 import tiny_positives_arm as arm   # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    if not condition:
        print("FAIL %s %s" % (name, detail))


def criterion_pair():
    model, criterion, _ = build(Config(input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32"),
                                synthetic=True)
    native = type(criterion).__mro__[1]
    module = __import__(native.__module__, fromlist=["generalized_box_iou"])
    return model, criterion, native.loss_boxes.__get__(criterion), module


def boxes(rows):
    return torch.tensor(rows, dtype=torch.float32)


def case(q_boxes, gt_boxes, indices):
    outputs = {"pred_boxes": q_boxes.unsqueeze(0)}
    targets = [{"boxes": gt_boxes}]
    return outputs, targets, indices


def main():
    if torch.cuda.is_initialized():
        raise SystemExit("CUDA must stay hidden for the unit checks")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, criterion, base, module = criterion_pair()

    # 1. frozen identity
    check("constants.objective", tp.OBJECTIVE == "tiny_band_positive_multiplicity_v1")
    check("constants.threshold", abs(tp.BAND_AREA_THRESHOLD - 1024.0 / (768.0 * 1344.0)) < 1e-18)
    check("constants.tau", tp.IOU_TAU == 0.5 and tp.EXTRAS_PER_GT == 1 and tp.REQUIRE_BEST_GT is True)
    check("constants.digest", len(tp.constants_sha256()) == 64
          and tp.constants_sha256() == tp.constants_sha256())

    # 2. band boundary (inclusive) on the normalized canvas area
    w = (tp.BAND_AREA_THRESHOLD) ** 0.5
    edge = boxes([[0.5, 0.5, w, w]])
    over = boxes([[0.5, 0.5, w * 1.001, w * 1.0005]])
    check("band.boundary_inclusive", bool(tp.band_mask(edge)[0]) and not bool(tp.band_mask(over)[0]))
    check("band.count", tp.band_gt([{"boxes": torch.cat([edge, over])}]) == 1)

    # hand-built geometry: GT0 tiny and in band, GT1 large and out of band
    gt = boxes([[0.5000, 0.5000, 0.0200, 0.0200], [0.2000, 0.2000, 0.4000, 0.4000]])
    near = boxes([[0.5000, 0.5000, 0.0200, 0.0200]])      # matched to GT0 (IoU 1.0)
    adm = boxes([[0.50649, 0.5000, 0.0200, 0.0200]])      # unmatched, IoU ~0.51 with GT0
    low = boxes([[0.50685, 0.5000, 0.0200, 0.0200]])      # unmatched, IoU ~0.49 with GT0
    other = boxes([[0.2000, 0.2000, 0.4000, 0.4000]])     # unmatched, overlaps GT1 only
    far = boxes([[0.9000, 0.9000, 0.0100, 0.0100]])       # unmatched, far from everything
    big = boxes([[0.2000, 0.2000, 0.4000, 0.4000]])       # matched to GT1
    q = torch.cat([near, adm, low, other, far, big])
    indices = [(torch.tensor([0, 5]), torch.tensor([0, 1]))]
    outputs, targets, idx = case(q, gt, indices)

    # 3. A2 admission: exactly the admissible unmatched query with the highest overlap
    src, tgt, ledger = tp.extra_pairs("A2", q, gt, indices[0][0], indices[0][1], module)
    check("A2.admits_best_sibling", src.tolist() == [1] and tgt.tolist() == [0], (src.tolist(), tgt.tolist()))
    check("A2.ledger", ledger["pos_extra_pairs"] == 1 and ledger["pos_band_gt_no_sibling"] == 0
          and ledger["pos_band_gt"] == 1 and ledger["pos_extra_iou_sum"] > 0.5, ledger)

    # 4. overlap bar: a query just below the bar is not admitted, and the admissible one wins
    q_bar = torch.cat([near, low, far, big])
    bar_idx = (torch.tensor([0, 3]), torch.tensor([0, 1]))
    src_bar, tgt_bar, ledger_bar = tp.extra_pairs("A2", q_bar, gt, bar_idx[0], bar_idx[1], module)
    check("A2.bar_rejects", src_bar.numel() == 0 and ledger_bar["pos_band_gt_no_sibling"] == 1, ledger_bar)

    # 5. own-best-partner rule: a query whose best ground truth is the out-of-band one is never used
    q_best = torch.cat([near, other, far, big])
    src_best, tgt_best, ledger_best = tp.extra_pairs("A2", q_best, gt, bar_idx[0], bar_idx[1], module)
    check("A2.own_best_partner", src_best.numel() == 0 and ledger_best["pos_band_gt_no_sibling"] == 1, ledger_best)

    # 6. A1 and A2 share one admission gate: identical appended set on the same outputs
    src_a1, tgt_a1, ledger_a1 = tp.extra_pairs("A1", q, gt, indices[0][0], indices[0][1], module)
    check("A1.gate_shared_appends_native_pair", src_a1.tolist() == [0] and tgt_a1.tolist() == [0]
          and ledger_a1["pos_extra_pairs"] == 1 and ledger_a1["pos_extra_candidates"] == 1
          and ledger_a1["pos_band_gt_no_sibling"] == 0 and ledger_a1["pos_extra_no_native"] == 0,
          ledger_a1)
    check("A1.A2_set_identical",
          ledger_a1["pos_extra_pairs"] == ledger["pos_extra_pairs"]
          and ledger_a1["pos_extra_candidates"] == ledger["pos_extra_candidates"]
          and ledger_a1["pos_band_gt_no_sibling"] == ledger["pos_band_gt_no_sibling"]
          and ledger_a1["pos_extra_set_digest"] == ledger["pos_extra_set_digest"], (ledger_a1, ledger))
    check("A1.appended_quality_is_the_native_pair",
          abs(ledger_a1["pos_extra_iou_sum"] - 1.0) < 1e-6 and ledger_a1["pos_extra_l1_sum"] == 0.0,
          ledger_a1)
    # no opportunity for the gate -> neither arm appends anything
    src_a1_bar, tgt_a1_bar, ledger_a1_bar = tp.extra_pairs("A1", q_bar, gt, bar_idx[0], bar_idx[1], module)
    check("A1.no_candidate_neither_appends", src_a1_bar.numel() == 0 and tgt_a1_bar.numel() == 0
          and ledger_a1_bar["pos_extra_pairs"] == 0 and ledger_a1_bar["pos_extra_candidates"] == 0
          and ledger_a1_bar["pos_band_gt_no_sibling"] == 1, ledger_a1_bar)

    # 6b. two band ground truths, one with a legitimate extra box and one without
    gt2 = boxes([[0.5000, 0.5000, 0.0200, 0.0200], [0.8000, 0.2000, 0.0200, 0.0200],
                 [0.2000, 0.2000, 0.4000, 0.4000]])
    q2 = torch.cat([boxes([[0.5000, 0.5000, 0.0200, 0.0200]]),      # matched to GT0
                    boxes([[0.50649, 0.5000, 0.0200, 0.0200]]),     # admissible sibling of GT0
                    boxes([[0.8000, 0.2000, 0.0200, 0.0200]]),      # matched to GT1
                    boxes([[0.80685, 0.2000, 0.0200, 0.0200]]),     # IoU ~0.49 with GT1: below the bar
                    boxes([[0.2000, 0.2000, 0.4000, 0.4000]])])     # matched to GT2
    idx2 = (torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2]))
    src_a2_2, tgt_a2_2, ledger_a2_2 = tp.extra_pairs("A2", q2, gt2, idx2[0], idx2[1], module)
    src_a1_2, tgt_a1_2, ledger_a1_2 = tp.extra_pairs("A1", q2, gt2, idx2[0], idx2[1], module)
    check("gate.split_scene", src_a2_2.tolist() == [1] and tgt_a2_2.tolist() == [0]
          and src_a1_2.tolist() == [0] and tgt_a1_2.tolist() == [0]
          and ledger_a1_2["pos_extra_pairs"] == ledger_a2_2["pos_extra_pairs"] == 1
          and ledger_a1_2["pos_band_gt_no_sibling"] == ledger_a2_2["pos_band_gt_no_sibling"] == 1
          and ledger_a1_2["pos_extra_candidates"] == ledger_a2_2["pos_extra_candidates"] == 1
          and ledger_a1_2["pos_extra_set_digest"] == ledger_a2_2["pos_extra_set_digest"],
          (ledger_a1_2, ledger_a2_2))
    # the structural corner: a qualifying ground truth without a native pair (crafted indices)
    corner = tp.extra_pairs("A1", q2, gt2, torch.tensor([2, 4]), torch.tensor([1, 2]), module)
    check("A1.no_native_pair_recorded", corner[0].numel() == 0 and corner[1].numel() == 0
          and corner[2]["pos_extra_no_native"] == 1 and corner[2]["pos_extra_pairs"] == 0
          and corner[2]["pos_extra_pairs"] + corner[2]["pos_band_gt_no_sibling"]
              + corner[2]["pos_extra_no_native"] == corner[2]["pos_band_gt"], corner[2])

    # 7. degenerate inputs
    none_src, none_tgt, ledger_none = tp.extra_pairs("A2", q, torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64),
                                                    torch.zeros(0, dtype=torch.int64), module)
    all_matched = [(torch.arange(6), torch.tensor([0, 0, 0, 1, 1, 1]))]
    dup_src, dup_tgt, ledger_dup = tp.extra_pairs("A2", q, gt, all_matched[0][0], all_matched[0][1], module)
    check("degenerate.no_gt", none_src.numel() == 0 and none_tgt.numel() == 0 and ledger_none["pos_band_gt"] == 0)
    check("degenerate.all_matched", dup_src.numel() == 0 and ledger_dup["pos_band_gt_no_sibling"] == 1, ledger_dup)

    # 8. determinism and no random draw
    before = torch.random.get_rng_state()
    again = tp.extra_pairs("A2", q, gt, indices[0][0], indices[0][1], module)
    check("determinism", torch.equal(src, again[0]) and torch.equal(tgt, again[1])
          and again[2] == ledger)
    check("rng.untouched", torch.equal(before, torch.random.get_rng_state()))

    # 9. the augmented-set identity against the frozen native loss
    num_boxes = 2.0
    native_loss = base(outputs, targets, indices, num_boxes, None)
    aug_a2 = [(torch.tensor([0, 5, 1]), torch.tensor([0, 1, 0]))]
    aug_a1 = [(torch.tensor([0, 5, 0]), torch.tensor([0, 1, 0]))]
    expected_a2 = base(outputs, targets, aug_a2, num_boxes, None)
    expected_a1 = base(outputs, targets, aug_a1, num_boxes, None)
    term_a2 = tp.TinyPositivesTerm("A2")
    term_a1 = tp.TinyPositivesTerm("A1")
    sink_a2 = {}
    got_a2 = term_a2.loss_boxes(base, module, outputs, targets, indices, num_boxes, None, sink=sink_a2)
    got_a1 = term_a1.loss_boxes(base, module, outputs, targets, indices, num_boxes, None, sink={})
    check("identity.A2_is_native_on_augmented_set",
          all(torch.equal(got_a2[k], expected_a2[k]) for k in expected_a2), (got_a2, expected_a2))
    check("identity.A1_is_native_on_duplicated_set",
          all(torch.equal(got_a1[k], expected_a1[k]) for k in expected_a1), (got_a1, expected_a1))
    check("identity.keys_preserved", sorted(got_a2) == sorted(native_loss) == ["loss_bbox", "loss_giou"])
    check("identity.mass_increases",
          all(float(got_a2[k]) > float(native_loss[k]) for k in native_loss), (got_a2, native_loss))
    check("identity.counter", term_a2.state_dict()["extras"] == 1 and term_a1.state_dict()["extras"] == 1)
    check("observation.new_fields",
          all(key in sink_a2["final"] for key in ("pos_extra_candidates", "pos_extra_no_native",
                                                  "pos_extra_set_digest"))
          and sink_a2["final"]["pos_extra_set_digest"] == ledger["pos_extra_set_digest"],
          sink_a2.get("final"))

    # 10. queries that are not admitted have no influence at all
    moved = q.clone()
    moved[2] = moved[2] + torch.tensor([0.05, 0.05, 0.02, 0.02])
    moved[3] = moved[3] + torch.tensor([-0.03, 0.01, 0.0, 0.0])
    moved[4] = moved[4] + torch.tensor([0.02, -0.02, 0.0, 0.0])
    outputs_moved, _, _ = case(moved, gt, indices)
    got_moved = tp.TinyPositivesTerm("A2").loss_boxes(base, module, outputs_moved, targets, indices,
                                                      num_boxes, None, sink={})
    check("no_wrong_supervision.unmatched_inert",
          all(torch.equal(got_a2[k], got_moved[k]) for k in got_a2), (got_a2, got_moved))

    # 11. gradient direction: the extra query is pulled towards its ground truth, others are inert
    qg = torch.cat([near, adm, low, other, far, big]).clone().requires_grad_(True)
    outputs_g, _, _ = case(qg, gt, indices)
    losses = tp.TinyPositivesTerm("A2").loss_boxes(base, module, outputs_g, targets, indices, num_boxes,
                                                   None, sink={})
    total = sum(losses.values())
    total.backward()
    grad = qg.grad
    step = 1e-4
    pull = gt[0, :2] - q[1, :2]
    direction = pull / float(pull.norm())
    losses_probe = []
    for delta in (step, -step):
        probe = torch.cat([near, adm, low, other, far, big]).clone()
        probe[1, :2] = probe[1, :2] + delta * direction
        outputs_p, _, _ = case(probe, gt, indices)
        losses_probe.append(float(sum(tp.TinyPositivesTerm("A2").loss_boxes(
            base, module, outputs_p, targets, indices, num_boxes, None, sink={}).values())))
    # "pulled towards its ground truth" means the descent direction (-gradient) points along
    # ``direction``, so the gradient projected on ``direction`` is negative, and a finite step
    # along ``direction`` lowers the loss
    align = float((grad[1, :2] * direction).sum())
    check("gradient.extra_query_pulled_towards_gt",
          align < 0.0 and losses_probe[0] < losses_probe[1] and float(grad[1].abs().sum()) > 0.0,
          (align, losses_probe, grad[1].tolist()))
    check("gradient.descent_direction_and_finite_difference_agree",
          (align < 0.0) == (losses_probe[0] < losses_probe[1]), (align, losses_probe))
    check("gradient.offset_is_central_axis_only", abs(float(grad[1, 1])) < abs(float(grad[1, 0])),
          grad[1].tolist())
    check("gradient.inert_queries_zero", float(grad[2].abs().sum()) == 0.0 and float(grad[3].abs().sum()) == 0.0
          and float(grad[4].abs().sum()) == 0.0, grad.tolist())

    # 12. term state, declaration and installation guards
    term = tp.TinyPositivesTerm("A2")
    term.extras = 7
    clone = tp.TinyPositivesTerm("A1")
    try:
        clone.load_state_dict(term.state_dict())
        refused = False
    except ValueError:
        refused = True
    check("state.cross_mode_refused", refused)
    reloaded = tp.TinyPositivesTerm("A2")
    reloaded.load_state_dict(term.state_dict())
    check("state.round_trip", reloaded.state_dict() == term.state_dict())
    import copy
    deep = copy.deepcopy(term)
    check("state.deepcopy", deep.state_dict() == term.state_dict() and deep is not term)
    check("declaration.native", arm.declaration(None) == arm.NATIVE_DECLARATION
          and arm.declaration(None)["installed"] is False)
    check("declaration.arms_differ", arm.declaration("A1") != arm.declaration("A2")
          and arm.declaration("A2")["constants_sha256"] == tp.constants_sha256())
    criterion.tiny_positives = None
    installed = arm.install(model, criterion, "A2")
    check("install.armed", arm.assert_installed(model, criterion, "A2")
          and installed.state_dict()["extras"] == 0)
    try:
        arm.install(model, criterion, "A1")
        double = False
    except ValueError:
        double = True
    check("install.double_refused", double)
    try:
        arm.install(model, criterion, None)
        none_refused = False
    except ValueError:
        none_refused = True
    check("install.native_refused", none_refused)
    check("install.model_untouched", not hasattr(model, "tiny_positives"))
    engine = type("E", (), {})()
    engine.criterion = criterion
    engine.pos_mode = "A2"
    check("install.engine_reports_installed", arm.installed(engine) is True)
    check("state.engine_round_trip", arm.state(engine)["term"] == installed.state_dict())
    try:
        arm.load(engine, None)
        none_load = False
    except ValueError:
        none_load = True
    check("load.native_into_armed_refused", none_load)

    # 13. the frozen plan must declare exactly these constants
    plan_path = ROOT / "configs/research/r18_tiny_positives_v1.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        declared = plan.get("tiny_positives") or {}
        check("plan.constants", declared.get("objective") == tp.OBJECTIVE
              and float(declared.get("band_area_px")) == float(tp.BAND_AREA_PX)
              and float(declared.get("iou_tau")) == float(tp.IOU_TAU)
              and int(declared.get("extras_per_gt")) == int(tp.EXTRAS_PER_GT)
              and declared.get("constants_sha256") == tp.constants_sha256(), declared)
        check("plan.arms", plan.get("arms") == ["A0", "A1", "A2"]
              and plan.get("pos_install") == {"A0": None, "A1": "A1", "A2": "A2"})
        check("plan.revision", plan.get("revision") == 2
              and plan.get("version") == "r18-tiny-positives-v2"
              and "duplicated-regression control" in str(plan.get("frozen_science", {}).get("control", "")),
              plan.get("revision"))
    else:
        check("plan.present", False, str(plan_path))

    passed = sum(1 for _, ok in RESULTS if ok)
    print("TESTS %d/%d PASS" % (passed, len(RESULTS)))
    for name, ok in RESULTS:
        if not ok:
            print("  failed:", name)
    if passed != len(RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
