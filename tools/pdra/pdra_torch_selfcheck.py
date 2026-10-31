"""CPU torch self-checks for the rev-6 PDRA wiring (synthetic inputs only).

Covers: the frozen RoIAlign convention against a float64 reference; the five-slot
descriptor encoding; exact antisymmetry of the torch pair arm; fixed-count training
determinism; byte-equal equivalence of the adapter tournament/merge with the numpy rules
core; sampling-manifest verification (positive and corrupted); arm capacities; tap
mechanics on a synthetic encoder; descriptor dimensions and the fp16 cache budget; the
pref-mask semantics of the pair-arm loss (the rev-6 device-placement regression).

No detector, no CUDA, no real data, no randomness beyond fixed seeds.  Run:
    python -I tools/pdra/pdra_torch_selfcheck.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_HERE, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
from torchvision.ops import roi_align
import torch.nn.functional as F

import pdra_core as P
import pdra_features as FEAT
import pdra_learn as L
import pdra_selfcheck as S


def _ok(name: str, detail: dict) -> dict:
    return {"check": name, "status": "PASS", **detail}


def _fail(name: str, detail: dict) -> dict:
    return {"check": name, "status": "FAIL", **detail}


def _tiny_bank(channels: int = 2, *, fill: float = 1.0) -> FEAT.FeatureBank:
    maps = {view: torch.full((channels, 96, 168), fill) for view in range(5)}
    return FEAT.FeatureBank(maps)


# --- 1: frozen RoIAlign convention -------------------------------------------
def check_roi_align_convention() -> dict:
    errors = []
    rng = np.random.default_rng(7)
    fmap = rng.standard_normal((4, 13, 17)).astype(np.float32)
    tensor = torch.from_numpy(fmap)
    boxes = [(2.3, 1.7, 6.9, 5.2), (0.0, 0.0, 17.0, 13.0), (10.5, 8.25, 16.0, 12.5)]
    worst = 0.0
    for box in boxes:
        ours = roi_align(tensor.unsqueeze(0), torch.tensor([[0.0, *box]]),
                         output_size=(2, 2), spatial_scale=FEAT.SPATIAL_SCALE,
                         sampling_ratio=FEAT.SAMPLING_RATIO, aligned=FEAT.ALIGNED)[0]
        ref = FEAT.reference_roi_align(fmap, box)
        worst = max(worst, float(np.max(np.abs(ours.numpy().astype(np.float64) - ref))))
    if worst > 2e-5:
        errors.append(f"torchvision vs float64 reference differs by {worst:.2e}")
    # aligned=True centres box [c, c+1] on feature cell c: with sampling_ratio=2 the four
    # samples of every output cell sit at +-0.125 / +-0.375 from the centre, so an impulse
    # at the cell centre yields ((0.625 + 0.875) / 2) ** 2 = 0.5625 in each output cell
    impulse = np.zeros_like(fmap)
    impulse[:, 6, 4] = 1.0
    pooled = roi_align(torch.from_numpy(impulse).unsqueeze(0),
                       torch.tensor([[0.0, 4.0, 6.0, 5.0, 7.0]]),
                       output_size=(2, 2), spatial_scale=1.0, sampling_ratio=2,
                       aligned=True)[0]
    if float((pooled - 0.5625).abs().max()) > 1e-5:
        errors.append(f"aligned cell-centre semantics broken: {pooled.flatten()[:4]}")
    # equivalent spelling: canvas units with spatial_scale=1/stride must agree ...
    box = boxes[0]
    canvas = roi_align(tensor.unsqueeze(0),
                       torch.tensor([[0.0, *(v * 8 for v in box)]]),
                       output_size=(2, 2), spatial_scale=1.0 / 8.0, sampling_ratio=2,
                       aligned=True)
    feature_units = roi_align(tensor.unsqueeze(0), torch.tensor([[0.0, *box]]),
                              output_size=(2, 2), spatial_scale=1.0, sampling_ratio=2,
                              aligned=True)
    if float((canvas - feature_units).abs().max()) > 1e-5:
        errors.append("feature-unit and canvas spelling of the same box disagree")
    # ... and the double-scaling mistake must be detectable
    wrong = roi_align(tensor.unsqueeze(0), torch.tensor([[0.0, *box]]),
                      output_size=(2, 2), spatial_scale=1.0 / 8.0, sampling_ratio=2,
                      aligned=True)
    if float((wrong - feature_units).abs().max()) < 1e-3:
        errors.append("double-scaling variant is not distinguishable; guard is void")
    detail = {"reference_max_abs_diff": worst, "boxes": len(boxes),
              "impulse_centre_value": 0.5625, "equivalent_spelling_agrees": True,
              "double_scaling_detected": True}
    return _fail("roi_align_convention", {**detail, "errors": errors}) if errors \
        else _ok("roi_align_convention", detail)


# --- 2: five-slot descriptor encoding ----------------------------------------
def check_slot_encoding() -> dict:
    errors = []
    bank = _tiny_bank()
    affine = P.ViewAffine((0.0, 0.0, 1920.0, 1080.0), stride=(8, 8), view_id=0)
    box_i, box_j = (100.0, 100.0, 200.0, 200.0), (150.0, 100.0, 250.0, 200.0)
    read_left = P.read_exclusive(P.subtract_strips(box_i, box_j), affine)   # single "left"
    read_right = P.read_exclusive(P.subtract_strips(box_j, box_i), affine)  # single "right"
    if [name for name, _ in read_left.strips] != ["left"]:
        errors.append(f"unexpected left decomposition: {read_left.strips}")
    desc = FEAT.pool_sides(bank, [read_left, read_right])
    dim, per_slot = FEAT.side_dim(2), 4 * 2
    slots, mask_start = dim - 5, dim - 5
    expect = torch.zeros(2, dim)
    expect[0, FEAT.SLOT_INDEX["left"] * per_slot:(FEAT.SLOT_INDEX["left"] + 1) * per_slot] = 1.0
    expect[1, FEAT.SLOT_INDEX["right"] * per_slot:(FEAT.SLOT_INDEX["right"] + 1) * per_slot] = 1.0
    expect[0, mask_start + FEAT.SLOT_INDEX["left"]] = 1.0
    expect[1, mask_start + FEAT.SLOT_INDEX["right"]] = 1.0
    if float((desc.float() - expect).abs().max()) > 1e-6:
        errors.append("slot layout differs from the frozen encoding")
    # invalid read -> all-zero descriptor with zero mask
    outside = P.read_exclusive(P.subtract_strips((3000.0, 100.0, 3100.0, 200.0),
                                                 (3050.0, 100.0, 3150.0, 200.0)), affine)
    if outside.valid:
        errors.append("constructed out-of-view read reported valid")
    zero = FEAT.pool_sides(bank, [outside])
    if float(zero.abs().sum()) != 0.0:
        errors.append("invalid read produced non-zero descriptor")
    # whole-box control descriptor: full slot + mask
    whole = FEAT.whole_box_read(box_i, affine)
    wide = FEAT.pool_whole(bank, [whole])
    if float(wide[0, :FEAT.whole_dim(2) - 1].max()) != 1.0 or float(wide[0, -1]) != 1.0:
        errors.append("whole-box descriptor is not the full slot plus mask")
    detail = {"slots": list(FEAT.SLOTS), "side_dim_c2": dim, "mask_bits": 5,
              "invalid_is_zero": True, "whole_dim_c2": FEAT.whole_dim(2)}
    return _fail("slot_encoding", {**detail, "errors": errors}) if errors \
        else _ok("slot_encoding", detail)


# --- 3: exact antisymmetry of the torch pair arm ------------------------------
def check_torch_antisymmetry() -> dict:
    errors = []
    torch.manual_seed(0)
    arm = L.PairArm(FEAT.side_dim(2), hidden=16)
    generator = torch.Generator().manual_seed(3)
    n = 64
    d_i = torch.randn(n, FEAT.side_dim(2), generator=generator).half()
    d_j = torch.randn(n, FEAT.side_dim(2), generator=generator).half()
    dx = torch.randn(n, generator=generator)
    dy = torch.randn(n, generator=generator)
    lw = torch.randn(n, generator=generator)
    lh = torch.randn(n, generator=generator)
    iou = torch.rand(n, generator=generator)
    geom = torch.stack([dx, dy, lw, lh, iou], dim=-1)
    flipped = torch.stack([-dx, -dy, -lw, -lh, iou], dim=-1)
    with torch.no_grad():
        r_ij, a_ij = arm.forward_batch(d_i, d_j, geom)
        r_ji, a_ji = arm.forward_batch(d_j, d_i, flipped)
    worst_r = float((r_ij + r_ji - 1.0).abs().max())
    worst_a = float((a_ij - a_ji).abs().max())
    if worst_r > 1e-5:
        errors.append(f"r antisymmetry broken: {worst_r:.2e}")
    if worst_a > 1e-5:
        errors.append(f"a symmetry broken: {worst_a:.2e}")
    detail = {"pairs": n, "max_abs_r_plus_flip_minus_1": worst_r,
              "max_abs_a_symmetry": worst_a}
    return _fail("torch_antisymmetry", {**detail, "errors": errors}) if errors \
        else _ok("torch_antisymmetry", detail)


# --- 4: fixed-count deterministic training ------------------------------------
def _synthetic_pair_tensors(n: int = 4096, dim: int = 45) -> L.PairTensors:
    generator = torch.Generator().manual_seed(11)
    sides_i = torch.randn(n, dim, generator=generator).half()
    sides_j = torch.randn(n, dim, generator=generator).half()
    geom = torch.rand(n, 5, generator=generator)
    pref_target = torch.randint(0, 2, (n,), generator=generator).to(torch.int8)
    abstain_target = torch.zeros(n)
    abstain_rows = torch.randperm(n, generator=generator)[:160]
    pref_target[abstain_rows] = -1
    abstain_target[abstain_rows] = 1.0
    return L.PairTensors(sides_i=sides_i, sides_j=sides_j, geom=geom,
                         pref_target=pref_target, abstain_target=abstain_target)


def check_learning_fixed_updates() -> dict:
    errors = []
    tensors = _synthetic_pair_tensors()
    spec = {**L.frozen_spec(), "updates": 40, "batch_pairs": 64}
    torch.manual_seed(0)
    arm_a = L.PairArm(45, hidden=16)
    report_a = L.train_pair_arm(arm_a, tensors, spec=spec, device="cpu")
    digest_a = L.tensor_state_digest(arm_a)
    torch.manual_seed(0)
    arm_b = L.PairArm(45, hidden=16)
    digest_b0 = L.tensor_state_digest(arm_b)
    report_b = L.train_pair_arm(arm_b, tensors, spec=spec, device="cpu")
    if report_a["updates"] != 40 or not report_a["steps_exactly_frozen_count"]:
        errors.append(f"pair arm ran {report_a['updates']} updates instead of the frozen 40")
    if not all(np.isfinite(report_a["first_losses"])):
        errors.append("non-finite losses in the first steps")
    if report_a["first_losses"] != report_b["first_losses"]:
        errors.append("training is not deterministic under the frozen generator")
    if digest_a == digest_b0:
        errors.append("training did not change the comparator parameters")
    grads = [name for name, p in arm_a.named_parameters() if p.requires_grad]
    if not grads:
        errors.append("the pair arm has no trainable parameters")
    # quality arm: same frozen count and determinism
    generator = torch.Generator().manual_seed(5)
    descs = torch.randn(2048, FEAT.whole_dim(2), generator=generator).half()
    targets = torch.rand(2048, generator=generator)
    q_spec = {**L.frozen_spec(), "updates": 30, "batch_pairs": 32}
    torch.manual_seed(0)
    quality_a = L.QualityArm(FEAT.whole_dim(2), hidden=16)
    qa = L.train_quality_arm(quality_a, descs, targets, spec=q_spec, device="cpu")
    torch.manual_seed(0)
    quality_b = L.QualityArm(FEAT.whole_dim(2), hidden=16)
    qb = L.train_quality_arm(quality_b, descs, targets, spec=q_spec, device="cpu")
    if qa["updates"] != 30 or qa["first_losses"] != qb["first_losses"]:
        errors.append("quality arm count or determinism differs")
    detail = {"pair_updates": report_a["updates"], "pair_first_losses": report_a["first_losses"][:3],
              "quality_updates": qa["updates"],
              "pair_arm_parameters": sum(p.numel() for p in arm_a.parameters()),
              "deterministic": True}
    return _fail("learning_fixed_updates", {**detail, "errors": errors}) if errors \
        else _ok("learning_fixed_updates", detail)


# --- 5b: rev-6 device-placement regression (pref-mask semantics) ---------------
def check_pair_arm_pref_mask_semantics() -> dict:
    """CPU regression for the rev-5 CUDA device bug.

    rev 5 moved the preference keep-mask to the compute device and then used it to index
    the CPU ``pref_target``, which fails on CUDA ("indices should be either on cpu or on
    the same device as the indexed tensor"); the first real-CUDA update of the r18-pdra
    GPU check stopped there.  On CPU the mask semantics are the testable part: the
    preference loss must be exactly BCE over the rows with ``pref_target >= 0`` in
    original order, and a batch without labelled rows must fall back to the frozen 0.0
    preference term.  The device placement itself is exercised by the real-CUDA updates
    of the GPU engineering check (learn phase).
    """
    errors = []
    batch, n = 64, 512
    generator = torch.Generator().manual_seed(23)
    tensors = L.PairTensors(
        sides_i=torch.randn(n, 45, generator=generator).half(),
        sides_j=torch.randn(n, 45, generator=generator).half(),
        geom=torch.rand(n, 5, generator=generator),
        pref_target=torch.randint(-1, 2, (n,), generator=generator).to(torch.int8),
        abstain_target=torch.rand(n, generator=generator))
    spec = {**L.frozen_spec(), "updates": 3, "batch_pairs": batch}
    torch.manual_seed(0)
    arm = L.PairArm(45, hidden=16)
    digest_before = L.tensor_state_digest(arm)
    report = L.train_pair_arm(arm, tensors, spec=spec, device="cpu")
    # independent reference of the first optimizer step, rebuilt from the frozen spec
    torch.manual_seed(0)
    reference_arm = L.PairArm(45, hidden=16)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(spec["seed"]))
    idx = perm[:batch]
    with torch.no_grad():
        r, a = reference_arm.forward_batch(tensors.sides_i[idx], tensors.sides_j[idx], tensors.geom[idx])
        keep = tensors.pref_target[idx] >= 0
        if bool(keep.any()):
            reference = float(F.binary_cross_entropy(r[keep], tensors.pref_target[idx][keep].float())
                              + F.binary_cross_entropy(a, tensors.abstain_target[idx]))
        else:
            reference = float(F.binary_cross_entropy(a, tensors.abstain_target[idx]))
    if not (bool(keep.any()) and not bool(keep.all())):
        errors.append("the regression batch is not mixed (some but not all rows labelled)")
    if report["updates"] != 3 or not report["steps_exactly_frozen_count"]:
        errors.append(f"pair arm ran {report['updates']} updates instead of the frozen 3")
    if abs(report["first_losses"][0] - reference) > 2e-6:
        errors.append(f"first-step loss {report['first_losses'][0]} differs from the reference {reference}")
    if L.tensor_state_digest(arm) == digest_before:
        errors.append("training did not change the comparator parameters")
    # a batch without preference labels must fall back to the frozen 0.0 preference term
    generator_u = torch.Generator().manual_seed(29)
    unlabelled = L.PairTensors(
        sides_i=torch.randn(batch, 45, generator=generator_u).half(),
        sides_j=torch.randn(batch, 45, generator=generator_u).half(),
        geom=torch.rand(batch, 5, generator=generator_u),
        pref_target=torch.full((batch,), -1, dtype=torch.int8),
        abstain_target=torch.rand(batch, generator=generator_u))
    spec_u = {**L.frozen_spec(), "updates": 1, "batch_pairs": batch}
    torch.manual_seed(0)
    arm_u = L.PairArm(45, hidden=16)
    report_u = L.train_pair_arm(arm_u, unlabelled, spec=spec_u, device="cpu")
    torch.manual_seed(0)
    reference_u = L.PairArm(45, hidden=16)
    with torch.no_grad():
        r_u, a_u = reference_u.forward_batch(unlabelled.sides_i, unlabelled.sides_j, unlabelled.geom)
        reference_abstain = float(F.binary_cross_entropy(a_u, unlabelled.abstain_target))
    if abs(report_u["first_losses"][0] - reference_abstain) > 1e-6:
        errors.append("all-unlabelled batch did not fall back to the abstain-only loss")
    detail = {"mixed_batch_labelled_rows": int(keep.sum()), "mixed_batch_size": batch,
              "first_loss_reference": reference, "first_loss_recorded": report["first_losses"][0],
              "unlabelled_fallback_equals_abstain_term": True, "updates": report["updates"]}
    return _fail("pair_arm_pref_mask_semantics", {**detail, "errors": errors}) if errors else _ok("pair_arm_pref_mask_semantics", detail)


# --- 5: adapter tournament/merge equals the numpy rules core -------------------
class _VerdictStub:
    def __init__(self, r: float, a: float) -> None:
        self.r, self.a = r, a

    def verdict(self, pair: P.PairInput) -> P.PairVerdict:
        return P.decide_pair(pair, self.r, self.a)


def check_adapter_equivalence() -> dict:
    errors = []
    pool = S._constructed_pool()
    clusters = P.nms_clusters(pool)
    for r_value, a_value in ((0.9, 0.0), (0.1, 0.0), (0.5, 0.0), (0.9, 0.9), (0.48, 0.0)):
        core_comparator = P.ScriptedComparator(preference=r_value, abstention=a_value)
        stub = _VerdictStub(r_value, a_value)
        for cluster in clusters:
            members = P.cluster_members(cluster)
            core = P.select_representative(members, S.affine_of, core_comparator)
            mine = L.tournament(members, S.affine_of, stub)
            if core["representative"].identity != mine["representative"].identity:
                errors.append(f"representative differs at r={r_value} a={a_value}")
            if core["stats"] != mine["stats"]:
                errors.append(f"stats differ at r={r_value} a={a_value}")
        core_merge = P.pdra_merge(pool, S.affine_of, core_comparator, budgets=(3, 2))
        mine_merge = L.merge(pool, S.affine_of, stub, budgets=(3, 2))
        for key in ("substitutions", "all_abstain_clusters", "rank_shifted_members",
                    "budget_membership_delta_vs_C"):
            if core_merge[key] != mine_merge[key]:
                errors.append(f"merge {key} differs at r={r_value} a={a_value}")
        if [c.identity for c in core_merge["top600"]] != [c.identity for c in mine_merge["top600"]]:
            errors.append(f"top600 differs at r={r_value} a={a_value}")
    # a real FeatureBank adapter runs end to end on a tiny bank (C=2)
    bank = _tiny_bank()
    torch.manual_seed(0)
    arm = L.PairArm(FEAT.side_dim(2), hidden=16)
    adapter = L.PairBankAdapter.build(arm, bank, pool, S.affine_of, "pdra")
    outcome = L.merge(pool, S.affine_of, adapter, budgets=(3, 2))
    reasons = {adapter.verdict(pair).reason for pair in
               (P.build_pair_input(r["box_i"].box_xyxy, S.affine_of(r["box_i"]),
                                   r["box_j"].box_xyxy, S.affine_of(r["box_j"]),
                                   identity_i=P._identity(r["box_i"]),
                                   identity_j=P._identity(r["box_j"]))
                for r in P.cluster_pair_records(pool))}
    if not reasons <= {"pref_i", "pref_j", "dead_zone", "abstain_head", "infeasible_read"}:
        errors.append(f"unexpected adapter verdict reasons: {reasons}")
    if outcome["cluster_count"] != len(clusters):
        errors.append("adapter merge cluster count differs")
    detail = {"comparators_checked": 5, "clusters": len(clusters),
              "adapter_reasons": sorted(reasons),
              "adapter_substitutions": outcome["substitutions"]}
    return _fail("adapter_equivalence", {**detail, "errors": errors}) if errors \
        else _ok("adapter_equivalence", detail)


# --- 6: sampling-manifest verification ----------------------------------------
def _row(image: str, split: str, destination: str, valid: bool, index: int) -> str:
    return json.dumps([image, "seq", split, 1, [0, index], [1, index + 1], 0, 0.5,
                       destination, None, 0.1 if destination == "pref" else None,
                       1, 10.0, int(valid), "ok"], allow_nan=False)


def check_sampling_verification() -> dict:
    errors = []
    image = "a" * 64
    with tempfile.TemporaryDirectory() as tmp:
        ids_path = os.path.join(tmp, "ids.txt")
        jsonl_path = os.path.join(tmp, "pairs.jsonl")
        rows = [_row(image, "train", "pref", True, i) for i in range(3)]
        rows.append(_row(image, "train", "abstain", True, 3))
        with open(jsonl_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(rows) + "\n")
        with open(ids_path, "w", encoding="utf-8", newline="\n") as stream:
            for i in range(3):
                stream.write(f"pref\t{image}|0,{i},1|1,{i + 1},1\n")
            stream.write(f"abstain\t{image}|0,3,1|1,4,1\n")
        manifest = {"n_pair_pref": 3, "n_pair_abstain_selected": 1, "n_pair_total": 4,
                    "sampling_spec_sha256": "test"}
        report = L.verify_sampling(ids_path, jsonl_path, manifest)
        if report["status"] != "PASS":
            errors.append(f"positive verification failed: {report}")
        # corrupted: one pair in the eval split must be detected
        with open(jsonl_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join([_row(image, "train", "pref", True, 0),
                                    _row(image, "eval", "pref", True, 1),
                                    _row(image, "train", "pref", True, 2),
                                    _row(image, "train", "abstain", True, 3)]) + "\n")
        bad = L.verify_sampling(ids_path, jsonl_path, manifest)
        if bad["status"] != "FAIL" or bad["rows_outside_train_split"] != 1:
            errors.append("corrupted split was not detected")
        # corrupted: missing pair id
        with open(ids_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"pref\t{image}|0,0,1|1,1,1\n")
        try:
            L.verify_sampling(ids_path, jsonl_path, manifest)
            errors.append("wrong count was not rejected")
        except ValueError:
            pass
    detail = {"positive": "PASS", "corrupt_split_detected": True, "wrong_count_rejected": True}
    return _fail("sampling_verification", {**detail, "errors": errors}) if errors \
        else _ok("sampling_verification", detail)


# --- 7: capacity and freeze payload -------------------------------------------
EXPECTED_PARAMS = {"pdra_difference_strips": 689_349, "c_whole_box_pairwise": 164_549,
                   "b_single_box_quality": 147_969}


def check_capacity_report() -> dict:
    errors = []
    payload = L.freeze_payload()
    for key, expected in EXPECTED_PARAMS.items():
        got = payload["arms"][key]["parameters"]
        if got != expected:
            errors.append(f"{key} parameter count {got} differs from expected {expected}")
        if got > L.LEARNING_SPEC["parameter_cap"]:
            errors.append(f"{key} exceeds the parameter cap")
    if payload["learning_spec_sha256"] != L.learning_spec_sha():
        errors.append("spec SHA is not stable")
    if payload["learning_spec"]["updates"] != 20_000 or payload["learning_spec"]["batch_pairs"] != 256:
        errors.append("frozen update/batch count changed")
    hidden = {meta["hidden"] for meta in payload["arms"].values()}
    if hidden != {128}:
        errors.append("arm hidden widths differ")
    detail = {"parameters": {k: v["parameters"] for k, v in payload["arms"].items()},
              "cap": L.LEARNING_SPEC["parameter_cap"],
              "learning_spec_sha256": payload["learning_spec_sha256"],
              "epochs_equivalent": payload["epochs_equivalent"]}
    return _fail("capacity_report", {**detail, "errors": errors}) if errors \
        else _ok("capacity_report", detail)


# --- 8: tap mechanics on a synthetic encoder ----------------------------------
class _Emb(torch.nn.Module):
    def forward(self, x):
        return [x * 2.0, x + 1.0]


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Emb()

    def forward(self, x):
        return self.encoder(x)


def check_tap_mechanics() -> dict:
    import pdra_tap as TAP
    errors = []
    model = _Tiny()
    x = torch.randn(2, 3, 6, 7)
    baseline = model(x)
    tap = TAP.EncoderTap(model)
    with tap:
        observed = model(x)
    if not torch.equal(baseline[0], observed[0]) or not torch.equal(baseline[1], observed[1]):
        errors.append("tap changed the module output")
    if tap.calls != 1 or tap.shapes != [(2, 3, 6, 7)]:
        errors.append(f"tap accounting differs: {tap.calls} {tap.shapes}")
    model(x)
    if tap.calls != 1:
        errors.append("hook was not removed after the context")
    maps = tap.clone_maps()
    if maps[0].requires_grad or maps[0].device != x.device:
        errors.append("cloned map is not a plain tensor on the original device")
    detail = {"calls": 1, "outputs_bitwise_equal": True, "hook_removed": True}
    return _fail("tap_mechanics", {**detail, "errors": errors}) if errors \
        else _ok("tap_mechanics", detail)


# --- 9: descriptor dimensions and fp16 cache budget ---------------------------
def check_dimension_budget() -> dict:
    errors = []
    if FEAT.side_dim(256) != 5125 or FEAT.whole_dim(256) != 1025:
        errors.append("descriptor dimensions differ from the frozen layout")
    pair_bytes = FEAT.cache_budget_bytes(338_735)
    if not 6.9e9 <= pair_bytes <= 7.0e9:
        errors.append(f"stage-1 PDRA cache estimate {pair_bytes} outside [6.9, 7.0] GB")
    if pair_bytes > 8e9:
        errors.append("cache estimate exceeds the 8 GB server-only cap")
    whole_bytes = FEAT.cache_budget_bytes(0, boxes=222_755 * 2)
    detail = {"side_dim_256": FEAT.side_dim(256), "whole_dim_256": FEAT.whole_dim(256),
              "stage1_pdra_cache_gb": round(pair_bytes / 1e9, 3),
              "stage1_whole_cache_gb": round(whole_bytes / 1e9, 3)}
    return _fail("dimension_budget", {**detail, "errors": errors}) if errors \
        else _ok("dimension_budget", detail)


# --- 10: GPU-check runtime constraints and three-arm allocation ---------------
def check_runtime_accounting_and_allocation() -> dict:
    import pdra_gpu_check as G
    errors = []
    if G.CHECK_UPDATES != 50 or sum(G.ARM_ALLOCATION.values()) != G.CHECK_UPDATES:
        errors.append("arm allocation does not sum to the frozen 50 updates")
    if set(G.ARM_ALLOCATION) != set(L.LEARNING_SPEC["arm_keys"]):
        errors.append("allocation keys differ from the frozen arm keys")
    if any(value <= 0 for value in G.ARM_ALLOCATION.values()):
        errors.append("an arm carries no acceptance update")
    if G.CAPS["comparator_updates"] != G.CHECK_UPDATES or G.CAPS["forward_calls"] != 20:
        errors.append("caps differ from the frozen check counts")
    if "gpu_peak_bytes" in G.CAPS:
        errors.append("the artificial memory gate is gone in rev 6 and must not return to CAPS")
    try:
        G.enforce_runtime(started=time.monotonic() - 10.0, label="selfcheck",
                          caps={"wall_seconds": 1, "output_bytes": 1})
        errors.append("a wall-clock breach did not raise")
    except G.RuntimeConstraintError:
        pass
    try:
        G.enforce_runtime(started=time.monotonic(), label="selfcheck",
                          caps={"wall_seconds": 3600, "output_bytes": 1 << 60})
    except G.RuntimeConstraintError:
        errors.append("a within-caps observation raised")
    accounting = G.Accounting(task="selfcheck", mode="cpu")
    accounting.phase("learn")
    accounting.complete("restore")
    accounting.forwards(5)
    accounting.update("pdra_difference_strips")
    snapshot = accounting.snapshot()
    if (snapshot["executed_forward_calls"] != 5
            or snapshot["executed_updates"]["pdra_difference_strips"] != 1
            or snapshot["phases_completed"] != ["restore"]):
        errors.append("accounting snapshot differs")
    snapshot["executed_forward_calls"] = 999
    if accounting.snapshot()["executed_forward_calls"] != 5:
        errors.append("snapshot is not detached from the live accounting")
    detail = {"allocation": dict(G.ARM_ALLOCATION), "check_updates": G.CHECK_UPDATES,
              "wall_breach_raised": True, "memory_gate_absent": True, "accounting_detached": True}
    return _fail("runtime_accounting_and_allocation", {**detail, "errors": errors}) if errors         else _ok("runtime_accounting_and_allocation", detail)


CHECKS = (
    check_runtime_accounting_and_allocation,
    check_roi_align_convention,
    check_slot_encoding,
    check_torch_antisymmetry,
    check_learning_fixed_updates,
    check_pair_arm_pref_mask_semantics,
    check_adapter_equivalence,
    check_sampling_verification,
    check_capacity_report,
    check_tap_mechanics,
    check_dimension_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise SystemExit("torch self-check must not initialize CUDA")
    started = time.monotonic()
    results = []
    for check in CHECKS:
        result = check()
        result["seconds"] = round(time.monotonic() - started, 3)
        results.append(result)
    failed = [r["check"] for r in results if r["status"] != "PASS"]
    report = {"core_version": P.CORE_VERSION,
              "learning_spec_sha256": L.learning_spec_sha(),
              "scope": "CONSTRUCTED_INPUTS_CPU_ONLY_NO_DETECTOR_NO_GPU",
              "cuda_initialized": torch.cuda.is_initialized(),
              "checks": results, "failed": failed,
              "status": "PASS" if not failed else "FAIL"}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
