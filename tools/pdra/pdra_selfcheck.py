"""PDRA CPU self-check: constructed inputs only, no detector, no GPU, no training.

Runs the targeted checks of the 2026-10-30 authorizations (rev 2 six items plus the
rev 3 corrections):

  1 coordinate mapping across views        9 control fairness (arm specifications)
  2 input swap (antisymmetry)             10 label ambiguity routing
  3 empty difference regions              11 consistency with the frozen merge core
  4 abstention                            12 one pairing domain (train == inference)
  5 cyclic preference                     13 ordering/budget after substitution
  6 near-neighbour distinct targets       14 cluster builder == frozen arithmetic
  7 original-C fallback (byte-identical)  15 frozen sequence split
  8 region coverage margin (geometry)     16 inference never reads GT labels
                                         17 complete inference domain (all cluster pairs)
                                         18 own-view identity propagation (rev 4)

plus a measured CPU cost count for the budget section.  Everything is deterministic;
the only randomness is a fixed-seed generator used to fabricate *inputs*.

Usage:  python3 -I tools/pdra/pdra_selfcheck.py [--out results.json]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import random
import sys
import time
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

import slice_core
import pdra_core as P


IMAGE_HW = (1500, 2000)          # constructed image: height, width
STRIDE = (8, 8)


@dataclass
class Cand:
    """Minimal candidate carrying exactly the frozen per-candidate fields."""

    box_xyxy: tuple
    score: float
    view_id: int
    query_id: int
    category_id: int


def _tiles() -> dict[int, tuple]:
    return {t.view_id: t.crop_box for t in slice_core.tile_windows(*IMAGE_HW)}


TILES = _tiles()


def affine_of(candidate: Any) -> P.ViewAffine:
    return P.candidate_affine(candidate, IMAGE_HW, TILES, canvas_hw=P.CANVAS_HW,
                              stride=STRIDE)


def _ok(name: str, detail: dict) -> dict:
    return {"check": name, "status": "PASS", **detail}


def _fail(name: str, detail: dict) -> dict:
    return {"check": name, "status": "FAIL", **detail}


# --- check 1: coordinate mapping ---------------------------------------------
def check_coordinate_mapping() -> dict:
    errors = []
    # view 0 crop-local is the identity
    v0 = P.candidate_affine(Cand((0, 0, 1, 1), 0.5, 0, 0, 1), IMAGE_HW, TILES)
    local = v0.to_crop_local((100.0, 200.0, 300.0, 400.0))
    if local != (100.0, 200.0, 300.0, 400.0):
        errors.append(f"view0 crop-local not identity: {local}")
    # known feature value for view 0: x*1344/2000/8, y*768/1500/8
    feat = v0.to_feature((100.0, 200.0, 300.0, 400.0))
    expected = (100 * 1344 / 2000 / 8, 200 * 768 / 1500 / 8,
                300 * 1344 / 2000 / 8, 400 * 768 / 1500 / 8)
    if max(abs(a - b) for a, b in zip(feat, expected)) > 1e-12:
        errors.append(f"view0 feature affine differs: {feat} vs {expected}")
    # round trip original -> feature -> original for every view
    boxes = [(50.0, 60.0, 250.0, 300.0), (900.0, 100.0, 1100.0, 260.0),
             (1500.0, 900.0, 1900.0, 1400.0), (1000.0, 700.0, 1050.0, 780.0)]
    for view_id in (0, 1, 2, 3, 4):
        cand = Cand((0, 0, 1, 1), 0.5, view_id, 0, 1)
        aff = affine_of(cand)
        for box in boxes:
            back = aff.from_feature(aff.to_feature(box))
            if max(abs(a - b) for a, b in zip(back, box)) > 1e-9:
                errors.append(f"view{view_id} round trip failed: {back} vs {box}")
    # visibility domain
    v2 = affine_of(Cand((0, 0, 1, 1), 0.5, 2, 0, 1))
    crop = P.box64(v2.crop_box)
    inside = (crop[0] + 10, crop[1] + 10, crop[0] + 40, crop[1] + 40)
    if abs(v2.visible_fraction(inside) - 1.0) > 1e-12:
        errors.append("fully visible box did not report 1.0")
    half = (crop[0] - 10.0, crop[1] + 10, crop[0] + 10.0, crop[1] + 30)
    if abs(v2.visible_fraction(half) - 0.5) > 1e-12:
        errors.append(f"half visible box reported {v2.visible_fraction(half)}")
    outside = (crop[2] + 5.0, crop[1] + 10, crop[2] + 25.0, crop[1] + 30)
    if v2.visible_fraction(outside) != 0.0:
        errors.append("out-of-crop box reported non-zero visibility")
    # agreement with the frozen geometry helper: the frozen core maps crop-local
    # pixels to original-image pixels by adding the integer crop origin; the PDRA
    # affine must invert exactly that step.
    for tile in slice_core.tile_windows(*IMAGE_HW):
        crop_local = (10.0, 20.0, 110.0, 220.0)
        aff = P.ViewAffine(tile.crop_box)
        original = slice_core.map_crop_xyxy([crop_local], tile)[0]
        mine = aff.to_crop_local(original)
        if max(abs(a - b) for a, b in zip(mine, crop_local)) > 1e-12:
            errors.append(f"view{tile.view_id} origin mapping differs from frozen core")
        # and the whole chain crop-local -> normalised must match the frozen decode
        cw, ch = tile.width, tile.height
        norm = aff.to_crop_normalised(original)
        expected_norm = ((crop_local[0] + crop_local[2]) / 2 / cw,
                         (crop_local[1] + crop_local[3]) / 2 / ch,
                         (crop_local[2] - crop_local[0]) / cw,
                         (crop_local[3] - crop_local[1]) / ch)
        if max(abs(a - b) for a, b in zip(norm, expected_norm)) > 1e-12:
            errors.append(f"view{tile.view_id} normalised decode mismatch")
    return _fail("coordinate_mapping", {"errors": errors}) if errors \
        else _ok("coordinate_mapping", {"views": 5, "round_trip_boxes": len(boxes),
                                        "frozen_core_agreement": True})


# --- check 2: input swap ------------------------------------------------------
def _random_pairs(count: int):
    rng = random.Random(7)
    out = []
    for _ in range(count):
        view_i = rng.randrange(0, 5)
        box_i = (rng.uniform(0, 1900), rng.uniform(0, 1400),
                 rng.uniform(0, 1900), rng.uniform(0, 1400))
        box_i = (min(box_i[0], box_i[2]), min(box_i[1], box_i[3]),
                 max(box_i[0], box_i[2]) + 1.0, max(box_i[1], box_i[3]) + 1.0)
        jx = rng.uniform(-0.4, 0.4)
        jy = rng.uniform(-0.4, 0.4)
        w = box_i[2] - box_i[0]
        h = box_i[3] - box_i[1]
        box_j = (box_i[0] + jx * w, box_i[1] + jy * h,
                 box_i[2] + jx * w, box_i[3] + jy * h)
        view_j = rng.randrange(0, 5)
        out.append((box_i, view_i, box_j, view_j))
    return out


def check_input_swap() -> dict:
    comparator = P.PairComparator(P.GeometricFeatureProvider(), seed=3)
    errors = []
    checked = 0
    for box_i, view_i, box_j, view_j in _random_pairs(60):
        aff_i = affine_of(Cand(box_i, 0.5, view_i, 0, 1))
        aff_j = affine_of(Cand(box_j, 0.5, view_j, 0, 1))
        pair_ij = P.build_pair_input(box_i, aff_i, box_j, aff_j)
        pair_ji = P.build_pair_input(box_j, aff_j, box_i, aff_i)
        if abs((comparator.r(pair_ij) + comparator.r(pair_ji)) - 1.0) > 1e-12:
            errors.append(f"r not antisymmetric for {box_i} {box_j}")
        if abs(comparator.a(pair_ij) - comparator.a(pair_ji)) > 1e-12:
            errors.append(f"a not symmetric for {box_i} {box_j}")
        v_ij = P.decide_pair(pair_ij, comparator.r(pair_ij), comparator.a(pair_ij))
        v_ji = P.decide_pair(pair_ji, comparator.r(pair_ji), comparator.a(pair_ji))
        if v_ij.winner != -v_ji.winner:
            errors.append(f"decision not mirrored for {box_i} {box_j}")
        checked += 1
    return _fail("input_swap", {"errors": errors[:5]}) if errors else \
        _ok("input_swap", {"pairs": checked, "r_antisymmetry": "exact_to_1e-12",
                           "a_symmetry": "exact_to_1e-12"})


# --- check 3: empty difference regions ---------------------------------------
def check_empty_difference_regions() -> dict:
    errors = []
    aff = affine_of(Cand((0, 0, 1, 1), 0.5, 0, 0, 1))
    cases = {
        "identical_boxes": ((100.0, 100.0, 200.0, 200.0), (100.0, 100.0, 200.0, 200.0)),
        "right_side_contained": ((100.0, 100.0, 300.0, 300.0), (120.0, 120.0, 280.0, 280.0)),
        "left_side_contained": ((120.0, 120.0, 280.0, 280.0), (100.0, 100.0, 300.0, 300.0)),
    }
    details = {}
    for name, (bi, bj) in cases.items():
        pair = P.build_pair_input(bi, aff, bj, aff)
        verdict = P.decide_pair(pair, 0.99, 0.0)
        left_empty = pair.left.region_area <= 0 or not pair.left.valid
        right_empty = pair.right.region_area <= 0 or not pair.right.valid
        if (left_empty or right_empty) is False:
            errors.append(f"{name}: expected an empty exclusive side")
        if verdict.winner != 0 or verdict.reason != "infeasible_read":
            errors.append(f"{name}: expected abstention on infeasible read")
        details[name] = {"left_valid": pair.left.valid, "right_valid": pair.right.valid,
                         "left_reason": pair.left.reason, "right_reason": pair.right.reason,
                         "verdict": verdict.reason}
    # a partially overlapping pair must stay feasible
    pair = P.build_pair_input((100.0, 100.0, 200.0, 200.0), aff, (150.0, 100.0, 250.0, 200.0), aff)
    details["partial_overlap"] = {"valid": pair.valid}
    if not pair.valid:
        errors.append("partial overlap wrongly reported infeasible")
    return _fail("empty_difference_regions", {"errors": errors}) if errors \
        else _ok("empty_difference_regions", details)


# --- check 4: abstention ------------------------------------------------------
def check_abstention() -> dict:
    errors = []
    aff = affine_of(Cand((0, 0, 1, 1), 0.5, 0, 0, 1))
    bi = (100.0, 100.0, 200.0, 200.0)
    bj = (150.0, 100.0, 250.0, 200.0)
    pair = P.build_pair_input(bi, aff, bj, aff)
    # abstention head above threshold withholds the vote whatever r says
    v = P.decide_pair(pair, 0.99, P.DELTA_ABSTAIN)
    if v.winner != 0 or v.reason != "abstain_head":
        errors.append("abstain head did not withhold the vote")
    # dead zone
    v = P.decide_pair(pair, 0.5, 0.0)
    if v.winner != 0 or v.reason != "dead_zone":
        errors.append("dead zone did not abstain")
    # a cluster in which every pair abstains keeps the original survivor
    members = [Cand(bi, 0.9, 0, 1, 1), Cand(bj, 0.8, 0, 2, 1)]
    abstaining = P.ScriptedComparator(preference=0.5, abstention=0.0)
    chosen = P.select_representative(members, affine_of, abstaining)
    if chosen["representative"] is not members[0]:
        errors.append("all-abstain cluster did not keep the original survivor")
    if chosen["stats"]["decided"] != 0:
        errors.append("all-abstain cluster reported a decided pair")
    # and it does move when a pair is decided
    decider = P.ScriptedComparator(preference=0.0, abstention=0.0)  # always prefers j
    chosen2 = P.select_representative(members, affine_of, decider)
    if chosen2["representative"] is not members[1]:
        errors.append("decisive comparator did not promote the preferred box")
    return _fail("abstention", {"errors": errors}) if errors else _ok(
        "abstention", {"abstain_head_withholds_vote": True, "dead_zone": True,
                       "all_abstain_keeps_survivor": True,
                       "decided_pair_promotes": True})


# --- check 5: cyclic preference ----------------------------------------------
def check_cyclic_preference() -> dict:
    errors = []
    a = Cand((100.0, 100.0, 200.0, 200.0), 0.90, 0, 1, 1)
    b = Cand((160.0, 100.0, 260.0, 200.0), 0.80, 0, 2, 1)
    c = Cand((220.0, 100.0, 320.0, 200.0), 0.70, 0, 3, 1)

    def pref(pair: P.PairInput) -> float:
        key = (pair.box_i, pair.box_j)
        if key in {(a.box_xyxy, b.box_xyxy), (b.box_xyxy, c.box_xyxy)}:
            return 1.0                      # first member wins
        if key == (a.box_xyxy, c.box_xyxy):
            return 0.0                      # c beats a, closing the cycle
        return 0.5

    comparator = P.ScriptedComparator(preference=pref, abstention=0.0)
    chosen = P.select_representative([a, b, c], affine_of, comparator)
    votes = sorted(chosen["votes"].values())
    if votes != [1, 1, 1]:
        errors.append(f"3-cycle did not yield equal Copeland counts: {votes}")
    if chosen["stats"]["decided"] != 3:
        errors.append("3-cycle did not decide all three pairs")
    # deterministic tie-break: highest original score wins the tie
    if chosen["representative"] is not a:
        errors.append("3-cycle tie-break did not fall back to the highest score")
    # transitivity is not required: a decisive chain resolves without a cycle
    chain_keys = {(a.box_xyxy, b.box_xyxy), (a.box_xyxy, c.box_xyxy), (b.box_xyxy, c.box_xyxy)}
    chain = P.ScriptedComparator(
        preference=lambda p: 1.0 if (p.box_i, p.box_j) in chain_keys else 0.5,
        abstention=0.0)
    chosen2 = P.select_representative([a, b, c], affine_of, chain)
    if chosen2["representative"] is not a:
        errors.append("decisive chain did not select the 2-win member")
    return _fail("cyclic_preference", {"errors": errors}) if errors else _ok(
        "cyclic_preference", {"votes": votes, "decided": 3,
                              "deterministic_tie_break": True, "no_crash": True})


# --- check 6: near-neighbour distinct targets --------------------------------
def check_near_neighbor_distinct_targets() -> dict:
    errors = []
    gts = [P.GTItem(1, 1, (100.0, 100.0, 200.0, 200.0)),
           P.GTItem(2, 1, (205.0, 100.0, 305.0, 200.0))]
    box_i = (100.0, 100.0, 200.0, 200.0)     # matches GT1 only
    box_j = (205.0, 100.0, 305.0, 200.0)     # matches GT2 only
    rec = P.label_pair(box_i, box_j, 1, gts)
    if rec["destination"] != "abstain":
        errors.append(f"distinct adjacent targets routed to {rec['destination']}")
    # an abstaining comparator must not promote the suppressed neighbour
    aff = affine_of(Cand((0, 0, 1, 1), 0.5, 0, 0, 1))
    members = [Cand(box_i, 0.9, 0, 1, 1), Cand(box_j, 0.8, 0, 2, 1)]
    abstaining = P.ScriptedComparator(preference=0.9, abstention=0.9)
    chosen = P.select_representative(members, affine_of, abstaining)
    if chosen["representative"] is not members[0]:
        errors.append("abstention failed to leave the survivor in place")
    contrast = P.ScriptedComparator(preference=0.0, abstention=0.0)
    chosen2 = P.select_representative(members, affine_of, contrast)
    if chosen2["representative"] is not members[1]:
        errors.append("contrast comparator did not promote the neighbour")
    return _fail("near_neighbor_distinct_targets", {"errors": errors}) if errors else _ok(
        "near_neighbor_distinct_targets",
        {"label_destination": rec["destination"], "abstention_holds": True,
         "contrast_promotes": True})


def _slice_candidates(records) -> list:
    out = []
    for rec in records:
        out.append(slice_core.Candidate(rec["box"], rec["score"], rec["view"],
                                        rec["query"], rec["category"], rec["rank"]))
    return out


def _constructed_pool() -> list:
    """A small multi-class pool with several genuine suppression clusters."""
    records = [
        # class 1: three overlapping boxes around one target, then a separate one
        {"box": (100.0, 100.0, 200.0, 200.0), "score": 0.95, "view": 0, "query": 1, "category": 1, "rank": 1},
        {"box": (110.0, 105.0, 205.0, 205.0), "score": 0.90, "view": 1, "query": 2, "category": 1, "rank": 2},
        {"box": (105.0, 102.0, 150.0, 160.0), "score": 0.85, "view": 3, "query": 3, "category": 1, "rank": 3},
        {"box": (900.0, 900.0, 980.0, 980.0), "score": 0.70, "view": 0, "query": 4, "category": 1, "rank": 4},
        # class 3: a pair and a singleton
        {"box": (400.0, 300.0, 460.0, 380.0), "score": 0.88, "view": 0, "query": 5, "category": 3, "rank": 1},
        {"box": (410.0, 305.0, 470.0, 385.0), "score": 0.60, "view": 2, "query": 6, "category": 3, "rank": 2},
        {"box": (1500.0, 100.0, 1600.0, 200.0), "score": 0.55, "view": 4, "query": 7, "category": 3, "rank": 3},
    ]
    return _slice_candidates(records)


# --- check 7: original-C fallback --------------------------------------------
def check_original_c_fallback() -> dict:
    pool = _constructed_pool()
    frozen = slice_core.merge_candidates(pool)
    frozen_top = slice_core.serialize_candidates(frozen["top600"])
    abstaining = P.ScriptedComparator(preference=0.5, abstention=0.0)
    pdra = P.pdra_merge(pool, affine_of, abstaining)
    pdra_top = slice_core.serialize_candidates(pdra["top600"])
    identical = pdra_top == frozen_top
    detail = {"clusters": pdra["cluster_count"], "kept": pdra["kept_count"],
              "substitutions": pdra["substitutions"],
              "all_abstain_clusters": pdra["all_abstain_clusters"],
              "rank_shifted_members": pdra["rank_shifted_members"],
              "byte_identical_to_frozen_C": identical}
    if not identical:
        diff = [i for i, (x, y) in enumerate(zip(frozen_top, pdra_top)) if x != y]
        return _fail("original_c_fallback", {**detail, "first_differing_index": diff[:5]})
    if pdra["kept_count"] != len(frozen_top):
        return _fail("original_c_fallback", {**detail, "reason": "kept count differs"})
    if pdra["rank_shifted_members"] != 0:
        return _fail("original_c_fallback", {**detail, "reason": "re-ranking moved a survivor"})
    return _ok("original_c_fallback", detail)


# --- check 8: region coverage margin (geometry only) --------------------------
def check_region_coverage_margin() -> dict:
    aff = affine_of(Cand((0, 0, 1, 1), 0.5, 0, 0, 1))
    pairs = [P.build_pair_input(bi, aff, bj, aff) for bi, bj in (
        ((100.0, 100.0, 200.0, 200.0), (150.0, 100.0, 250.0, 200.0)),
        ((300.0, 300.0, 400.0, 400.0), (320.0, 330.0, 420.0, 430.0)),
        ((600.0, 100.0, 700.0, 200.0), (650.0, 100.0, 750.0, 200.0)),
    )]
    checks = P.region_coverage_margin(pairs)
    errors = []
    if not checks["c_covers_pdra_domain"]:
        errors.append("control (c) region does not cover the PDRA difference domain")
    if checks["min_margin"] is None or checks["min_margin"] < -1e-9:
        errors.append("control (c) margin went negative")
    # exact expected margin: area(i) + area(j) - (area(i\j) + area(j\i)) = 2 * area(i & j)
    for pair in pairs:
        inter = P.intersect(pair.box_i, pair.box_j)
        margin = 0.0 if inter is None else 2 * ((inter[2] - inter[0]) * (inter[3] - inter[1]))
        c_area = P.CONTROL_WHOLE_BOX_PAIRWISE.read_area(pair.box_i, pair.box_j)
        if abs((c_area - pair.read_area) - margin) > 1e-9:
            errors.append("control (c) area margin is not 2x the intersection area")
    detail = {"pairs": checks["pairs_checked"], "min_margin": checks["min_margin"],
              "property": "geometric coverage only; not a fairness statement"}
    return _fail("region_coverage_margin", {**detail, "errors": errors}) if errors \
        else _ok("region_coverage_margin", detail)


# --- check 9: control fairness on arm specifications --------------------------
def check_control_fairness() -> dict:
    errors = []
    pool_sha = "constructed-pool-sha256"
    arms = [P.ArmSpec("a_original_c", "score", pool_sha),
            P.ArmSpec("b_single_box_quality", "whole_box_quality", pool_sha),
            P.ArmSpec("c_whole_box_pairwise", "whole_box_pairwise", pool_sha),
            P.ArmSpec("pdra", "difference_strips", pool_sha)]
    fair = P.assert_control_fairness(arms)
    if not fair["fair"]:
        errors.append(f"identical arms reported unfair: {fair['violations']}")
    if sorted(fair["reads"].values()) != sorted(a.reads for a in arms):
        errors.append("read rules not reported per arm")
    # a richer PDRA input (finer stride) must be detected
    richer = arms[:3] + [P.ArmSpec("pdra", "difference_strips", pool_sha, feature_stride=(4, 4))]
    if P.assert_control_fairness(richer)["fair"]:
        errors.append("finer PDRA feature stride not detected as unfair")
    # a different split or evaluation membership must be detected
    other_split = arms[:3] + [P.ArmSpec("pdra", "difference_strips", pool_sha,
                                        split_sha256="other")]
    if P.assert_control_fairness(other_split)["fair"]:
        errors.append("different module split not detected as unfair")
    other_eval = arms[:3] + [P.ArmSpec("pdra", "difference_strips", pool_sha,
                                       eval_members_sha256="other")]
    if P.assert_control_fairness(other_eval)["fair"]:
        errors.append("different evaluation members not detected as unfair")
    # a larger parameter or update budget must be detected
    bigger = arms[:3] + [P.ArmSpec("pdra", "difference_strips", pool_sha, parameter_cap=2_000_000)]
    if P.assert_control_fairness(bigger)["fair"]:
        errors.append("larger PDRA parameter cap not detected as unfair")
    # two arms with the same read rule are not two controls
    same_read = arms[:3] + [P.ArmSpec("pdra", "whole_box_pairwise", pool_sha)]
    if P.assert_control_fairness(same_read)["fair"]:
        errors.append("duplicate read rule not detected")
    detail = {"arms": fair["arms"], "equal_fields": fair["equal_fields"],
              "free_field": fair["free_field"], "negative_cases_detected": 5,
              "separate_from_region_coverage": True}
    return _fail("control_fairness", {**detail, "errors": errors}) if errors \
        else _ok("control_fairness", detail)


# --- check 10: label ambiguity routing ---------------------------------------
def check_label_ambiguity() -> dict:
    errors = []
    # near-coincident, shifted GTs so that one large box can match two of them
    gts = [P.GTItem(1, 1, (100.0, 100.0, 200.0, 200.0)),
           P.GTItem(2, 1, (120.0, 120.0, 220.0, 220.0)),
           P.GTItem(3, 1, (140.0, 140.0, 240.0, 240.0))]
    # clean cases use a well-separated GT pair
    clean = [P.GTItem(11, 1, (500.0, 500.0, 600.0, 600.0)),
             P.GTItem(12, 1, (620.0, 500.0, 720.0, 600.0))]
    records = []
    records.append(P.label_pair((500.0, 500.0, 600.0, 600.0),        # pref, gap 0.178
                                (505.0, 505.0, 605.0, 605.0), 1, clean))
    records.append(P.label_pair((500.0, 500.0, 600.0, 600.0),        # abstain, disjoint GTs
                                (620.0, 500.0, 720.0, 600.0), 1, clean))
    conflict = P.label_pair((108.0, 110.0, 208.0, 210.0),           # shares GT2, extras GT1/GT3
                            (125.0, 125.0, 225.0, 225.0), 1, gts)
    records.append(conflict)
    records.append(P.label_pair((900.0, 900.0, 950.0, 950.0),        # excluded_empty
                                (500.0, 500.0, 600.0, 600.0), 1, clean))
    records.append(P.label_pair((500.0, 500.0, 600.0, 600.0),        # excluded_gap
                                (500.0, 500.0, 600.0, 601.0), 1, clean))
    if conflict["destination"] != "excluded_ambiguous":
        errors.append(f"structural conflict routed to {conflict['destination']}")
    if conflict["would_be"] != ["pref", "abstain"]:
        errors.append(f"ambiguous pair did not record would-be destinations: {conflict['would_be']}")
    if not conflict["ambiguous"] or not conflict["multigt"]:
        errors.append("ambiguous pair not flagged ambiguous/multigt")
    expected = ["pref", "abstain", "excluded_ambiguous", "excluded_empty", "excluded_gap"]
    got = [r["destination"] for r in records]
    if got != expected:
        errors.append(f"destinations differ: {got} vs {expected}")
    # no pair may sit in both the preference and the abstention supervision
    pref = {i for i, r in enumerate(records) if r["destination"] == "pref"}
    abst = {i for i, r in enumerate(records) if r["destination"] == "abstain"}
    if pref & abst:
        errors.append("a pair entered both preference and abstention supervision")
    if any(r["destination"] is None for r in records):
        errors.append("a pair left without a destination")
    stats = P.aggregate_labels(records)
    if stats["ambiguous_pairs"] != 1:
        errors.append(f"ambiguity ledger counted {stats['ambiguous_pairs']}")
    # the precomputed-match route must equal the box route bit for bit
    mi = P.match_gts_iou((500.0, 500.0, 600.0, 600.0), 1, clean)
    mj = P.match_gts_iou((505.0, 505.0, 605.0, 605.0), 1, clean)
    if P.route_pair(mi, mj) != records[0]:
        errors.append("route_pair on precomputed matches differs from label_pair")
    # vectorised IoU must equal the scalar IoU bit for bit
    boxes = np.asarray([(500.0, 500.0, 600.0, 600.0), (505.0, 505.0, 605.0, 605.0),
                        (620.0, 500.0, 720.0, 600.0)])
    gt_boxes = np.asarray([g.box_xyxy for g in clean])
    matrix = P.iou_matrix(boxes, gt_boxes)
    for bi, box in enumerate(boxes):
        for gi, g in enumerate(clean):
            if float(matrix[bi, gi]) != P.box_iou(tuple(box), g.box_xyxy):
                errors.append("iou_matrix differs from box_iou")
    detail = {"destinations": got, "conflict_would_be": conflict["would_be"],
              "ledger": stats["destination_counts"],
              "ambiguous_pairs": stats["ambiguous_pairs"],
              "ambiguous_would_be": stats["ambiguous_would_be"],
              "vectorised_iou_bit_identical": True}
    return _fail("label_ambiguity", {**detail, "errors": errors}) if errors \
        else _ok("label_ambiguity", detail)


# --- check 11: consistency with the frozen merge core ------------------------
def check_frozen_core_consistency() -> dict:
    pool = _constructed_pool()
    clusters = P.nms_clusters(pool)
    mine = [cl["survivor"].box_xyxy for cl in clusters]
    theirs = [c.box_xyxy for c in slice_core.classwise_nms(pool)]
    if mine != theirs:
        return _fail("frozen_core_consistency",
                     {"reason": "cluster survivors differ from the frozen kept list"})
    # every suppressed candidate is within IoU>0.5 of its survivor and not kept elsewhere
    for cl in clusters:
        for sup in cl["suppressed"]:
            if P.box_iou(cl["survivor"].box_xyxy, sup.box_xyxy) <= slice_core.NMS_IOU:
                return _fail("frozen_core_consistency", {"reason": "suppressed box not suppressed"})
    return _ok("frozen_core_consistency", {"clusters": len(clusters),
                                           "survivors_match_frozen_nms": True})


# --- check 12: one pairing domain for training and inference ------------------
def _domain_pool() -> list:
    # S suppresses A and B (IoU 0.526 each) but IoU(A, B) = 0.357 < 0.5;
    # D overlaps the suppressed A at IoU 0.526 yet forms its own cluster.
    records = [
        {"box": (100.0, 100.0, 200.0, 200.0), "score": 0.95, "view": 0, "query": 1, "category": 1, "rank": 1},
        {"box": (100.0, 100.0, 200.0, 290.0), "score": 0.90, "view": 1, "query": 2, "category": 1, "rank": 2},
        {"box": (100.0, 10.0, 200.0, 200.0), "score": 0.85, "view": 2, "query": 3, "category": 1, "rank": 3},
        {"box": (100.0, 190.0, 200.0, 290.0), "score": 0.50, "view": 3, "query": 4, "category": 1, "rank": 4},
    ]
    return _slice_candidates(records)


def check_pair_domain_unified() -> dict:
    errors = []
    pool = _domain_pool()
    by_id = {c.identity: c for c in pool}
    S, A, B, D = by_id[(0, 1, 1)], by_id[(1, 2, 1)], by_id[(2, 3, 1)], by_id[(3, 4, 1)]
    clusters = P.nms_clusters(pool)
    if [cl["survivor"].identity for cl in clusters] != [S.identity, D.identity]:
        errors.append("constructed pool did not yield the expected two clusters")
    # training enumeration
    training = {frozenset((r["box_i"].identity, r["box_j"].identity))
                for r in P.cluster_pair_records(pool)}
    # inference enumeration: the pairs the tournament actually visits
    visited = []

    class Recorder:
        def r(self, pair):
            visited.append((pair.box_i, pair.box_j))
            return 0.5

        def a(self, pair):
            return 0.0

    box_to_id = {c.box_xyxy: c.identity for c in pool}
    for cl in clusters:
        P.select_representative(P.cluster_members(cl), affine_of, Recorder())
    inference = {frozenset((box_to_id[bi], box_to_id[bj])) for bi, bj in visited}
    if training != inference:
        errors.append(f"training pairs {sorted(map(sorted, training))} != "
                      f"inference pairs {sorted(map(sorted, inference))}")
    expected = {frozenset((S.identity, A.identity)), frozenset((S.identity, B.identity)),
                frozenset((A.identity, B.identity))}
    if training != expected:
        errors.append(f"pair domain differs from the cluster-internal set: {sorted(map(sorted, training))}")
    iou_ab = P.box_iou(A.box_xyxy, B.box_xyxy)
    iou_ad = P.box_iou(A.box_xyxy, D.box_xyxy)
    if not (iou_ab < 0.5 and frozenset((A.identity, B.identity)) in training):
        errors.append("a below-0.5 cluster-internal pair is missing from the domain")
    if not (iou_ad >= 0.5 and frozenset((A.identity, D.identity)) not in training):
        errors.append("a cross-cluster >=0.5 pair leaked into the domain")
    # the rev-2 pool filter would have produced a different set
    rev2 = {frozenset((x.identity, y.identity)) for i, x in enumerate(pool)
            for y in pool[i + 1:] if x.category_id == y.category_id
            and P.box_iou(x.box_xyxy, y.box_xyxy) >= 0.5}
    if rev2 == training:
        errors.append("rev-2 pool filter coincides with the cluster domain on this case; "
                      "the counter-example does not discriminate")
    detail = {"pair_domain": P.PAIR_DOMAIN, "training_pairs": len(training),
              "inference_pairs": len(inference),
              "below_half_pair_in_domain": round(iou_ab, 4),
              "cross_cluster_pair_excluded": round(iou_ad, 4),
              "rev2_filter_pairs": len(rev2)}
    return _fail("pair_domain_unified", {**detail, "errors": errors}) if errors \
        else _ok("pair_domain_unified", detail)


# --- check 13: ordering and budget truncation after substitution --------------
def check_post_replacement_ranking() -> dict:
    errors = []
    records = [
        {"box": (100.0, 100.0, 200.0, 200.0), "score": 0.95, "view": 0, "query": 1, "category": 1, "rank": 1},
        {"box": (105.0, 100.0, 205.0, 200.0), "score": 0.40, "view": 1, "query": 2, "category": 1, "rank": 2},
        {"box": (500.0, 500.0, 600.0, 600.0), "score": 0.60, "view": 0, "query": 3, "category": 1, "rank": 3},
        {"box": (800.0, 800.0, 900.0, 900.0), "score": 0.50, "view": 0, "query": 4, "category": 1, "rank": 4},
    ]
    pool = _slice_candidates(records)
    by_id = {c.identity: c for c in pool}
    S1, M1, S2, S3 = by_id[(0, 1, 1)], by_id[(1, 2, 1)], by_id[(0, 3, 1)], by_id[(0, 4, 1)]
    frozen_kept = [c.identity for c in slice_core.classwise_nms(pool)]
    if frozen_kept != [S1.identity, S2.identity, S3.identity]:
        errors.append(f"unexpected frozen kept order {frozen_kept}")
    prefer_j = P.ScriptedComparator(preference=0.0, abstention=0.0)
    result = P.pdra_merge(pool, affine_of, prefer_j, budgets=(2, 1))
    kept = [c.identity for c in result["top600"]]
    if result["substitutions"] != 1:
        errors.append(f"expected one substitution, got {result['substitutions']}")
    if kept != [S2.identity, S3.identity]:
        errors.append(f"budget-2 output not re-ranked by the frozen key: {kept}")
    if [c.identity for c in result["top300"]] != [S2.identity]:
        errors.append("budget-1 output did not pick the highest re-ranked representative")
    full = P.pdra_merge(pool, affine_of, prefer_j, budgets=(10, 5))
    order = [c.identity for c in full["top600"]]
    if order != [S2.identity, S3.identity, M1.identity]:
        errors.append(f"full output not in canonical order after substitution: {order}")
    keys = [P.sort_key_of(c) for c in full["top600"]]
    if keys != sorted(keys):
        errors.append("representative list is not sorted by the frozen sort key")
    delta = result["budget_membership_delta_vs_C"]
    if delta.get("2") != {"entered": 1, "left": 1}:
        errors.append(f"budget membership delta mis-reported: {delta}")
    if result["rank_shifted_members"] != 3:
        errors.append(f"rank shift mis-counted: {result['rank_shifted_members']}")
    if result["representative_ranking"] != P.POST_REPLACEMENT_RANKING:
        errors.append("ranking rule not recorded")
    # all-abstain stays identical to C under the same budgets
    abstaining = P.ScriptedComparator(preference=0.5, abstention=0.0)
    same = P.pdra_merge(pool, affine_of, abstaining, budgets=(2, 1))
    if [c.identity for c in same["top600"]] != frozen_kept[:2] or same["rank_shifted_members"]:
        errors.append("all-abstain run no longer equals C after re-ranking")
    detail = {"ranking": P.POST_REPLACEMENT_RANKING, "substituted_representative_score": M1.score,
              "budget2_output": kept, "budget_delta_vs_C": delta,
              "rank_shifted_members": result["rank_shifted_members"]}
    return _fail("post_replacement_ranking", {**detail, "errors": errors}) if errors \
        else _ok("post_replacement_ranking", detail)


# --- check 14: cluster builder equals the frozen arithmetic --------------------
def _random_pool(seed: int, count: int) -> list:
    rng = random.Random(seed)
    out = []
    for index in range(count):
        x = rng.uniform(0, 1800)
        y = rng.uniform(0, 1300)
        w = rng.uniform(5, 120)
        h = rng.uniform(5, 120)
        out.append(slice_core.Candidate((x, y, x + w, y + h), rng.random(),
                                        rng.randrange(0, 5), index % 300, 1 + index % 3,
                                        1 + index))
    return out


def check_cluster_builder_consistency() -> dict:
    errors = []
    pools = [_constructed_pool(), _domain_pool()] + [_random_pool(s, 240) for s in (5, 6, 7)]
    checked = 0
    for pool in pools:
        frozen = [c.identity for c in slice_core.classwise_nms(pool)]
        numpy_clusters, stats = P.nms_clusters(pool, with_stats=True)
        scalar_clusters = P.nms_clusters_scalar(pool)
        mine = [cl["survivor"].identity for cl in numpy_clusters]
        theirs = [cl["survivor"].identity for cl in scalar_clusters]
        if mine != frozen:
            errors.append("numpy mirror survivors differ from the frozen core")
        if theirs != frozen:
            errors.append("scalar builder survivors differ from the frozen core")
        for a, b in zip(numpy_clusters, scalar_clusters):
            if [c.identity for c in a["suppressed"]] != [c.identity for c in b["suppressed"]]:
                errors.append("numpy and scalar builders disagree on a cluster's members")
        if sum(1 + len(cl["suppressed"]) for cl in numpy_clusters) != len(pool):
            errors.append("cluster partition incomplete")
        if stats["cluster_count"] != len(frozen):
            errors.append("cluster count differs from kept count")
        checked += 1
    return _fail("cluster_builder_consistency", {"errors": errors[:5]}) if errors \
        else _ok("cluster_builder_consistency",
                 {"pools": checked, "builder": P.CLUSTER_BUILDER,
                  "survivors_equal_frozen": True, "members_equal_scalar": True})


# --- check 15: frozen sequence split ------------------------------------------
def check_sequence_split_frozen() -> dict:
    errors = []
    rng = random.Random(3)
    sequences = {f"seq{index:02d}": [f"img{index:02d}_{k}" for k in range(rng.randint(10, 30))]
                 for index in range(24)}
    first = P.freeze_sequence_split(sequences)
    second = P.freeze_sequence_split(dict(reversed(list(sequences.items()))))
    if first != second:
        errors.append("split is not deterministic under input order")
    keys = sorted(sequences)
    if first["eval_sequences"] != [k for i, k in enumerate(keys) if i % 3 == 2]:
        errors.append("eval sequences differ from the frozen rule")
    if set(first["train_sequences"]) & set(first["eval_sequences"]):
        errors.append("a sequence sits in both splits")
    if sorted(first["train_sequences"] + first["eval_sequences"]) != keys:
        errors.append("split does not cover every sequence")
    if first["train_images"] + first["eval_images"] != sum(map(len, sequences.values())):
        errors.append("image totals do not add up")
    if first["main_verdict_split"] != "eval":
        errors.append("main verdict split not the module-untrained one")
    detail = {"rule": first["rule"], "sequences": first["sequence_count"],
              "train_sequences": len(first["train_sequences"]),
              "eval_sequences": len(first["eval_sequences"]),
              "deterministic": True, "sequence_level_isolation": True}
    return _fail("sequence_split_frozen", {**detail, "errors": errors}) if errors \
        else _ok("sequence_split_frozen", detail)


# --- check 16: inference never reads GT labels --------------------------------
def check_inference_label_independence() -> dict:
    """The decision path must not consume GT routing records, and label routing
    (pref/abstain/excluded_*) must never narrow the inference domain."""
    import inspect
    errors = []
    params = list(inspect.signature(P.pair_decisions).parameters)
    if params != ["candidates", "affine_of", "comparator"]:
        errors.append(f"pair_decisions grew a non-decision argument: {params}")
    pool = _domain_pool()
    comparator = P.ScriptedComparator(
        preference=lambda pair: 0.9 if pair.identity_i == (0, 1, 1) else 0.5,
        abstention=0.0)
    first = P.pair_decisions(pool, affine_of, comparator)
    second = P.pair_decisions(pool, affine_of, comparator)
    if first != second:
        errors.append("pair_decisions is not deterministic")
    # Build two label routings with and without GT.  They differ (destinations move
    # between pref/abstain and excluded_empty), yet the decision pass is a separate
    # call with no GT argument and identical output.
    gts = [P.GTItem(1, 1, (100.0, 100.0, 200.0, 200.0)),
           P.GTItem(2, 1, (100.0, 190.0, 200.0, 290.0))]
    records = P.cluster_pair_records(pool)
    routed = {P.label_pair(r["box_i"].box_xyxy, r["box_j"].box_xyxy, 1, gts)["destination"]
              for r in records}
    routed_empty = {P.label_pair(r["box_i"].box_xyxy, r["box_j"].box_xyxy, 1, [])["destination"]
                    for r in records}
    if routed == routed_empty:
        errors.append("constructed GT sets do not discriminate the routing check")
    if first != P.pair_decisions(pool, affine_of, comparator):
        errors.append("decisions changed between invocations with different label sets")
    detail = {"pair_decisions_arguments": params, "pairs_decided": len(first),
              "destinations_with_gt": sorted(routed),
              "destinations_without_gt": sorted(routed_empty),
              "decisions_identical_across_label_sets": True}
    return _fail("inference_label_independence", {**detail, "errors": errors}) if errors \
        else _ok("inference_label_independence", detail)


# --- check 17: complete inference domain --------------------------------------
def _feasible_cluster_pool() -> list:
    """One three-member view-0 cluster whose reads are all feasible, plus a singleton.

    S suppresses A and B (IoU 0.667 each) while IoU(A, B) = 0.47 < 0.5; every
    exclusive region is non-empty and fully visible, so no pair abstains for
    geometric reasons.
    """
    records = [
        {"box": (100.0, 100.0, 200.0, 200.0), "score": 0.95, "view": 0, "query": 1, "category": 1, "rank": 1},
        {"box": (120.0, 100.0, 220.0, 200.0), "score": 0.90, "view": 0, "query": 2, "category": 1, "rank": 2},
        {"box": (100.0, 120.0, 200.0, 220.0), "score": 0.85, "view": 0, "query": 3, "category": 1, "rank": 3},
        {"box": (500.0, 500.0, 600.0, 600.0), "score": 0.50, "view": 0, "query": 4, "category": 1, "rank": 4},
    ]
    return _slice_candidates(records)


def check_inference_domain_all_pairs() -> dict:
    """Every cluster-internal pair is decided, including pairs whose labels would be
    excluded from training; exclusions cannot silently shrink the inference domain."""
    errors = []
    pool = _feasible_cluster_pool()
    records, comparator = P.cluster_pair_records(pool), P.ScriptedComparator(
        preference=lambda pair: 0.0, abstention=0.0)      # always prefers j
    decisions = P.pair_decisions(pool, affine_of, comparator)
    if len(decisions) != len(records) or len(records) != 3:
        errors.append(f"pair_decisions enumerated {len(decisions)} of {len(records)} pairs")
    if any(d["winner"] not in (-1, 0, 1) for d in decisions):
        errors.append("a decision left the winner domain")
    if any(not d["valid"] for d in decisions):
        errors.append("constructed feasible pool produced an infeasible read")
    # with empty GT every pair would route to excluded_empty, yet the tournament
    # still substitutes the representative: label exclusion does not reach inference
    empty_routes = {P.label_pair(r["box_i"].box_xyxy, r["box_j"].box_xyxy, 1, [])["destination"]
                    for r in records}
    if empty_routes != {"excluded_empty"}:
        errors.append("constructed pool did not route to excluded_empty with empty GT")
    merged = P.pdra_merge(pool, affine_of, comparator)
    if merged["substitutions"] < 1:
        errors.append("excluded-by-label pairs did not participate in the tournament")
    if merged["cluster_count"] != 2:
        errors.append("constructed pool did not yield the expected two clusters")
    detail = {"cluster_pairs": len(records), "decisions": len(decisions),
              "label_destinations_if_scored": sorted(empty_routes),
              "substitutions_despite_excluded_labels": merged["substitutions"]}
    return _fail("inference_domain_all_pairs", {**detail, "errors": errors}) if errors \
        else _ok("inference_domain_all_pairs", detail)


# --- check 18: own-view identity propagation (rev 4) ---------------------------
def check_source_view_propagation() -> dict:
    errors = []
    box_i = (100.0, 100.0, 200.0, 200.0)
    box_j = (150.0, 100.0, 250.0, 200.0)
    vi, vj = affine_of(Cand(box_i, 0.5, 2, 0, 1)), affine_of(Cand(box_j, 0.5, 3, 0, 1))
    pair = P.build_pair_input(box_i, vi, box_j, vj,
                              identity_i=(2, 5, 1), identity_j=(3, 7, 1))
    if pair.left.source_view != 2 or pair.right.source_view != 3:
        errors.append("reads did not carry their own source views")
    if pair.identity_i != (2, 5, 1) or pair.identity_j != (3, 7, 1):
        errors.append("pair identities were not stored")
    swapped = P.swap_pair(pair)
    if swapped.left.source_view != 3 or swapped.right.source_view != 2:
        errors.append("swap did not exchange source views")
    if swapped.identity_i != (3, 7, 1) or swapped.identity_j != (2, 5, 1):
        errors.append("swap did not exchange identities")
    unspecified = P.ViewAffine((0.0, 0.0, 100.0, 100.0))
    read = P.read_exclusive(P.subtract_strips(box_i, box_j), unspecified)
    if read.source_view is not None:
        errors.append("unspecified affine leaked a source view")
    for view_id in range(5):
        affine = affine_of(Cand((0, 0, 1, 1), 0.5, view_id, 0, 1))
        if affine.view_id != view_id:
            errors.append(f"candidate_affine lost view {view_id}")
    detail = {"left_source_view": pair.left.source_view,
              "right_source_view": pair.right.source_view,
              "swap_exchanges": True, "unspecified_is_none": True}
    return _fail("source_view_propagation", {**detail, "errors": errors}) if errors \
        else _ok("source_view_propagation", detail)


# --- measured CPU cost --------------------------------------------------------
def check_cpu_cost() -> dict:
    """Measure the decision layer's CPU cost on a synthetic pool of realistic size."""
    rng = random.Random(11)
    pool = []
    for cls in range(1, 11):
        for target in range(60):
            x = rng.uniform(0, 1850)
            y = rng.uniform(0, 1360)
            w = rng.uniform(20, 60)
            h = rng.uniform(20, 60)
            for copy in range(5):                      # five near-duplicate candidates
                jx = rng.uniform(-0.3, 0.3) * w
                jy = rng.uniform(-0.3, 0.3) * h
                s = rng.uniform(0.9, 1.1)
                box = (x + jx, y + jy, x + jx + w * s, y + jy + h * s)
                pool.append(Cand(box, rng.random(), copy, target, cls))
    comparator = P.PairComparator(P.GeometricFeatureProvider())
    start = time.perf_counter()
    clusters = P.nms_clusters(pool)
    nms_seconds = time.perf_counter() - start
    pairs = sum(len(cl["suppressed"]) * (len(cl["suppressed"]) + 1) // 2 for cl in clusters)
    start = time.perf_counter()
    result = P.pdra_merge(pool, affine_of, comparator)
    merge_seconds = time.perf_counter() - start
    return _ok("cpu_cost_measured", {
        "candidates": len(pool), "clusters": len(clusters), "intra_cluster_pairs": pairs,
        "nms_seconds": round(nms_seconds, 4), "merge_seconds": round(merge_seconds, 4),
        "seconds_per_pair_us": round(1e6 * merge_seconds / pairs, 3) if pairs else None,
        "kept": result["kept_count"], "substitutions": result["substitutions"],
        "note": "measured on CPU with the geometric provider; stage 1 replaces the "
                "provider with RoIAlign reads and re-measures",
    })


CHECKS = (
    check_coordinate_mapping,
    check_input_swap,
    check_empty_difference_regions,
    check_abstention,
    check_cyclic_preference,
    check_near_neighbor_distinct_targets,
    check_original_c_fallback,
    check_region_coverage_margin,
    check_control_fairness,
    check_label_ambiguity,
    check_frozen_core_consistency,
    check_pair_domain_unified,
    check_post_replacement_ranking,
    check_cluster_builder_consistency,
    check_sequence_split_frozen,
    check_inference_label_independence,
    check_inference_domain_all_pairs,
    check_source_view_propagation,
    check_cpu_cost,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    results = [check() for check in CHECKS]
    failed = [r["check"] for r in results if r["status"] != "PASS"]
    report = {
        "core_version": P.CORE_VERSION,
        "constants": {
            "canvas_hw": P.CANVAS_HW, "feature_stride": list(STRIDE),
            "nms_iou": P.NMS_IOU, "delta_pref": P.DELTA_PREF,
            "delta_abstain": P.DELTA_ABSTAIN, "theta_region": P.THETA_REGION,
            "tau_gap": P.TAU_GAP, "tau_clear": P.TAU_CLEAR, "match_iou": P.MATCH_IOU,
            "pair_domain": P.PAIR_DOMAIN,
            "post_replacement_ranking": P.POST_REPLACEMENT_RANKING,
            "split_rule": P.SPLIT_RULE, "n_min_pairs": P.N_MIN_PAIRS,
            "n_min_images": P.N_MIN_IMAGES,
        },
        "scope": "CONSTRUCTED_INPUTS_CPU_ONLY_NO_DETECTOR_NO_GPU",
        "checks": results,
        "failed": failed,
        "status": "PASS" if not failed else "FAIL",
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
