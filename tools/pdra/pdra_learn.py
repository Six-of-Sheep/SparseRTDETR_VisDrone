"""Real learning modules and controls for PDRA (rev 4).

Three trainable arms share one architecture family and one capacity cap:

* ``pdra_difference_strips``  pair arm reading the two exclusive-region reads (five slots);
* ``c_whole_box_pairwise``    pair arm reading the two whole boxes (superset region);
* ``b_single_box_quality``    independent per-box quality head (IoU-Net style);

arm (a) is the frozen original C and needs no module.  All pairwise heads keep the exact
antisymmetry of the rules core by construction: ``r = sigmoid(z)`` with

    z = u . (E(d_i) - E(d_j)) + v . g_odd        (no bias; swap negates z exactly),

and the abstention head reads the symmetric descriptor ``[E(d_i)+E(d_j), |E(d_i)-E(d_j)|,
IoU]`` so ``a(i,j) == a(j,i)``.  Arm (b) maps its quality difference onto the frozen vote
threshold: ``r = clip(0.5 + (q_i-q_j) * delta_pref / (2*tau_q), 0, 1)`` so it votes iff
``|q_i - q_j| >= tau_q`` and abstains otherwise (``a = 0``: no abstain head).

Training is fully fixed (no search, no early stop): ``LEARNING_SPEC['updates']`` batched
updates of ``LEARNING_SPEC['batch_pairs']`` pairs, AdamW with frozen hyperparameters and a
frozen deterministic shuffle.  Loss normalisation is the arithmetic mean over the batch,
per term; the abstention term averages over *all* pairs of the batch with ``z = 1`` for
cross-GT pairs and ``z = 0`` for same-GT pairs.  Pairs whose labels are excluded from
supervision never enter the batches (the frozen sampling manifest contains only pref and
abstain pairs), and the inference path never reads labels at all.

The decision layer is reused through adapters: :class:`PairBankAdapter` and
:class:`QualityBankAdapter` precompute comparator outputs for every cluster-internal pair
of one image (batched, no GT), and :func:`tournament` / :func:`merge` replay the frozen
Copeland tournament, tie rules and post-substitution re-ranking on top of those verdicts.
The torch self-check verifies byte-equal equivalence with the numpy rules core for a
comparator presented through both paths.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdra_core as P
import pdra_features as FEAT

LEARNING_SPEC = {
    "spec_version": "pdra-learning-v1-20261030",
    "updates": 20_000,
    "batch_pairs": 256,
    "optimizer": {"type": "AdamW", "lr": 3e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                  "weight_decay": 1e-2},
    "grad_clip": None,
    "loss_pref": ("binary_cross_entropy(r, y) averaged over the preference-labelled pairs "
                  "of the batch (term 0.0 when the batch holds none)"),
    "loss_abstain": ("binary_cross_entropy(a, z) averaged over ALL pairs of the batch; "
                     "z = 1 cross-GT (abstain pairs), z = 0 same-GT (preference pairs)"),
    "loss_total": "L_pref + lambda_abstain * L_abstain",
    "lambda_abstain": 1.0,
    "shuffle": ("torch.randperm with torch.Generator().manual_seed(20261030); the generator "
                "advances across epochs, so the order is deterministic and reshuffled"),
    "seed": 20261030,
    "hidden": 128,
    "parameter_cap": 1_000_000,
    "tau_q": 0.05,
    "b_decision": ("vote iff |q_i - q_j| >= tau_q via r = clip(0.5 + (q_i-q_j) * delta_pref / "
                   "(2*tau_q), 0, 1); no abstain head"),
    "b_loss": "mean squared error against the member's matched-GT IoU",
    "c_invalid_read_maps_to": "infeasible (no vote), same outcome as infeasible_read",
    "arm_keys": ["pdra_difference_strips", "c_whole_box_pairwise", "b_single_box_quality"],
}


def learning_spec_sha(spec: Mapping | None = None) -> str:
    import json
    return hashlib.sha256(json.dumps(spec or LEARNING_SPEC, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def frozen_spec() -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in LEARNING_SPEC.items()}


# --- modules ------------------------------------------------------------------
class PairArm(nn.Module):
    """Shared-encoder pairwise arm over one side-descriptor kind (strips or whole boxes)."""

    def __init__(self, side_dim: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.side_dim = int(side_dim)
        self.hidden = int(hidden)
        self.encoder = nn.Sequential(
            nn.Linear(self.side_dim, hidden), nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=False))
        self.pref_u = nn.Linear(hidden + 4, 1, bias=False)   # bias-free: exact antisymmetry
        self.abstain = nn.Sequential(
            nn.Linear(2 * hidden + 1, 64), nn.ReLU(inplace=False), nn.Linear(64, 1))

    def pref_logit(self, d_i: torch.Tensor, d_j: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        e_i, e_j = self.encoder(d_i.float()), self.encoder(d_j.float())
        delta = torch.cat([e_i - e_j, geom.float()[..., :4]], dim=-1)
        return self.pref_u(delta).squeeze(-1)

    def abstain_logit(self, d_i: torch.Tensor, d_j: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
        e_i, e_j = self.encoder(d_i.float()), self.encoder(d_j.float())
        h = torch.cat([e_i + e_j, (e_i - e_j).abs(), iou.float().unsqueeze(-1)], dim=-1)
        return self.abstain(h).squeeze(-1)

    def forward_batch(self, d_i: torch.Tensor, d_j: torch.Tensor,
                      geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.sigmoid(self.pref_logit(d_i, d_j, geom))
        a = torch.sigmoid(self.abstain_logit(d_i, d_j, geom[..., 4]))
        return r, a


class QualityArm(nn.Module):
    """Independent per-box quality head over one whole-box descriptor (control b)."""

    def __init__(self, desc_dim: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.desc_dim = int(desc_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.desc_dim, hidden), nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=False), nn.Linear(hidden, 1))

    def value(self, desc: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.encoder(desc.float()).squeeze(-1))


def build_arms(*, channels: int = FEAT.P3_CHANNELS, hidden: int | None = None,
               seed: int | None = None) -> dict:
    """Fresh arms for the three module controls; capacities recorded against the cap."""
    spec = LEARNING_SPEC
    torch.manual_seed(spec["seed"] if seed is None else seed)
    hidden = spec["hidden"] if hidden is None else hidden
    arms = {
        "pdra_difference_strips": PairArm(FEAT.side_dim(channels), hidden=hidden),
        "c_whole_box_pairwise": PairArm(FEAT.whole_dim(channels), hidden=hidden),
        "b_single_box_quality": QualityArm(FEAT.whole_dim(channels), hidden=hidden),
    }
    report = {}
    for key, arm in arms.items():
        count = sum(p.numel() for p in arm.parameters())
        if count > spec["parameter_cap"]:
            raise ValueError(f"{key} exceeds the frozen parameter cap: {count}")
        report[key] = {"parameters": count, "hidden": hidden,
                       "cap": spec["parameter_cap"],
                       "side_or_desc_dim": int(getattr(arm, "side_dim", 0) or arm.desc_dim)}
    return {"arms": arms, "report": report, "spec_sha256": learning_spec_sha()}


# --- training (fixed update count; no search, no early stop) -------------------
@dataclass
class PairTensors:
    """fp16 side descriptors plus fp32 geometry and targets for one arm."""
    sides_i: torch.Tensor            # [N, D] fp16
    sides_j: torch.Tensor            # [N, D] fp16
    geom: torch.Tensor               # [N, 5] fp32
    pref_target: torch.Tensor        # [N] int8, -1 = pair carries no preference label
    abstain_target: torch.Tensor     # [N] fp32, 1 cross-GT / 0 same-GT
    pair_ids: list = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.sides_i.shape[0])


def train_pair_arm(arm: PairArm, tensors: PairTensors, *, spec: Mapping | None = None,
                   device: str = "cuda:0", step_log: Callable | None = None) -> dict:
    spec = spec or LEARNING_SPEC
    arm = arm.to(device).train()
    optimizer = torch.optim.AdamW(arm.parameters(), lr=spec["optimizer"]["lr"],
                                  betas=tuple(spec["optimizer"]["betas"]),
                                  eps=spec["optimizer"]["eps"],
                                  weight_decay=spec["optimizer"]["weight_decay"])
    generator = torch.Generator().manual_seed(spec["seed"])
    batch = int(spec["batch_pairs"])
    n = len(tensors)
    if n < batch:
        raise ValueError("training set smaller than one frozen batch")
    started = time.monotonic()
    updates, first_losses, running = 0, [], 0.0
    order_position, perm = 0, torch.randperm(n, generator=generator)
    while updates < spec["updates"]:
        if order_position + batch > n:
            perm = torch.randperm(n, generator=generator)
            order_position = 0
        idx = perm[order_position:order_position + batch]
        order_position += batch
        d_i = tensors.sides_i[idx].to(device)
        d_j = tensors.sides_j[idx].to(device)
        geom = tensors.geom[idx].to(device)
        r, a = arm.forward_batch(d_i, d_j, geom)
        pref_mask = tensors.pref_target[idx].to(device) >= 0
        if bool(pref_mask.any()):
            y = tensors.pref_target[idx][pref_mask].float().to(device)
            loss_pref = F.binary_cross_entropy(r[pref_mask], y)
        else:
            loss_pref = r.sum() * 0.0
        z = tensors.abstain_target[idx].float().to(device)
        loss_abstain = F.binary_cross_entropy(a, z)
        loss = loss_pref + float(spec["lambda_abstain"]) * loss_abstain
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at update {updates}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if spec.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(arm.parameters(), spec["grad_clip"])
        optimizer.step()
        updates += 1
        running += float(loss.detach())
        if updates <= 8:
            first_losses.append(round(float(loss.detach()), 6))
        if step_log is not None and (updates == 1 or updates % 2000 == 0):
            step_log({"arm": "pair", "update": updates, "loss": float(loss.detach())})
    arm.eval()
    return {"updates": int(updates), "batch": batch,
            "first_losses": first_losses,
            "mean_loss": running / max(updates, 1),
            "seconds": time.monotonic() - started,
            "steps_exactly_frozen_count": updates == int(spec["updates"])}


def train_quality_arm(arm: QualityArm, descs: torch.Tensor, targets: torch.Tensor, *,
                      spec: Mapping | None = None, device: str = "cuda:0",
                      step_log: Callable | None = None) -> dict:
    spec = spec or LEARNING_SPEC
    arm = arm.to(device).train()
    optimizer = torch.optim.AdamW(arm.parameters(), lr=spec["optimizer"]["lr"],
                                  betas=tuple(spec["optimizer"]["betas"]),
                                  eps=spec["optimizer"]["eps"],
                                  weight_decay=spec["optimizer"]["weight_decay"])
    generator = torch.Generator().manual_seed(spec["seed"])
    batch, n = int(spec["batch_pairs"]), int(descs.shape[0])
    if n < batch:
        raise ValueError("quality training set smaller than one frozen batch")
    started = time.monotonic()
    updates, order_position, perm = 0, 0, torch.randperm(n, generator=generator)
    first_losses, running = [], 0.0
    while updates < spec["updates"]:
        if order_position + batch > n:
            perm = torch.randperm(n, generator=generator)
            order_position = 0
        idx = perm[order_position:order_position + batch]
        order_position += batch
        q = arm.value(descs[idx].to(device))
        loss = F.mse_loss(q, targets[idx].float().to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite quality loss at update {updates}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        updates += 1
        running += float(loss.detach())
        if updates <= 8:
            first_losses.append(round(float(loss.detach()), 6))
    arm.eval()
    return {"updates": int(updates), "batch": batch, "first_losses": first_losses,
            "mean_loss": running / max(updates, 1),
            "seconds": time.monotonic() - started,
            "steps_exactly_frozen_count": updates == int(spec["updates"])}


# --- verdict adapters over the frozen decision layer ---------------------------
class PairBankAdapter:
    """Precomputed comparator outputs for every cluster-internal pair of one image.

    ``kind='pdra'`` feeds the exclusive-region side descriptors and keeps the frozen
    strip-visibility rule.  ``kind='c'`` feeds whole-box descriptors and uses the whole
    box's own visibility rule; a read that fails it maps to an infeasible verdict,
    i.e. the pair casts no vote, exactly like an infeasible strip read for PDRA.
    No ground truth is read anywhere in this path.
    """

    def __init__(self, kind: str) -> None:
        if kind not in ("pdra", "c"):
            raise ValueError("PairBankAdapter kind must be 'pdra' or 'c'")
        self.kind = kind
        self.table: dict[tuple, tuple[bool, str, float, float]] = {}

    @classmethod
    def build(cls, arm: PairArm, bank: FEAT.FeatureBank, candidates: Sequence,
              affine_of: Callable, kind: str, *, chunk: int = 4096) -> "PairBankAdapter":
        adapter = cls(kind)
        records = P.cluster_pair_records(candidates)
        arm = arm.eval()
        with torch.no_grad():
            for start in range(0, len(records), chunk):
                part = records[start:start + chunk]
                pairs = []
                for rec in part:
                    b_i, b_j = rec["box_i"], rec["box_j"]
                    pairs.append(P.build_pair_input(
                        b_i.box_xyxy, affine_of(b_i), b_j.box_xyxy, affine_of(b_j),
                        identity_i=P._identity(b_i), identity_j=P._identity(b_j)))
                if kind == "pdra":
                    d_i = FEAT.pool_sides(bank, [p.left for p in pairs])
                    d_j = FEAT.pool_sides(bank, [p.right for p in pairs])
                    validity = [(p.valid, p.reason) for p in pairs]
                else:
                    reads_i, reads_j = [], []
                    validity = []
                    for p, rec in zip(pairs, part):
                        b_i, b_j = rec["box_i"], rec["box_j"]
                        w_i = FEAT.whole_box_read(p.box_i, affine_of(b_i))
                        w_j = FEAT.whole_box_read(p.box_j, affine_of(b_j))
                        reads_i.append(w_i)
                        reads_j.append(w_j)
                        ok = w_i.valid and w_j.valid
                        reason = ("ok" if ok else
                                  ("left_" + w_i.reason if not w_i.valid else
                                   "right_" + w_j.reason))
                        validity.append((ok, reason))
                    d_i = FEAT.pool_whole(bank, reads_i)
                    d_j = FEAT.pool_whole(bank, reads_j)
                geom = FEAT.geometry_tensor(pairs, bank.device)
                r, a = arm.forward_batch(d_i, d_j, geom)
                for pair, (ok, reason), rv, av in zip(pairs, validity, r.tolist(), a.tolist()):
                    adapter.table[(pair.identity_i, pair.identity_j)] = (ok, reason,
                                                                         float(rv), float(av))
        return adapter

    def verdict(self, pair: P.PairInput) -> P.PairVerdict:
        try:
            ok, reason, r, a = self.table[(pair.identity_i, pair.identity_j)]
        except KeyError as exc:
            raise KeyError(f"pair {pair.identity_i}->{pair.identity_j} was not precomputed") from exc
        if not ok:
            return P.PairVerdict(0, r, a, "infeasible_read")
        if self.kind == "pdra":
            return P.decide_pair(pair, r, a)
        if r >= 0.5 + P.DELTA_PREF:
            return P.PairVerdict(1, r, a, "pref_i")
        if r <= 0.5 - P.DELTA_PREF:
            return P.PairVerdict(-1, r, a, "pref_j")
        return P.PairVerdict(0, r, a, "dead_zone")


class QualityBankAdapter:
    """Control (b): per-box quality over whole-box descriptors, mapped to the frozen rule."""

    kind = "b"

    def __init__(self) -> None:
        self.quality: dict[tuple, float | None] = {}
        self.table: dict[tuple, tuple[bool, str, float, float]] = {}

    @classmethod
    def build(cls, arm: QualityArm, bank: FEAT.FeatureBank, candidates: Sequence,
              affine_of: Callable, *, chunk: int = 4096) -> "QualityBankAdapter":
        adapter = cls()
        arm = arm.eval()
        # one whole-box read per unique member; None = read invalid (cannot be preferred)
        unique = {}
        for candidate in candidates:
            unique.setdefault(P._identity(candidate), candidate)
        ids = sorted(unique)
        with torch.no_grad():
            for start in range(0, len(ids), chunk):
                part = ids[start:start + chunk]
                reads, valid_ids = [], []
                for ident in part:
                    candidate = unique[ident]
                    read = FEAT.whole_box_read(candidate.box_xyxy, affine_of(candidate))
                    if read.valid:
                        reads.append(read)
                        valid_ids.append(ident)
                if reads:
                    desc = FEAT.pool_whole(bank, reads)
                    q = arm.value(desc).tolist()
                    for ident, value in zip(valid_ids, q):
                        adapter.quality[ident] = float(value)
                for ident in part:
                    adapter.quality.setdefault(ident, None)
        scale = P.DELTA_PREF / (2.0 * LEARNING_SPEC["tau_q"])
        for rec in P.cluster_pair_records(candidates):
            i_id, j_id = P._identity(rec["box_i"]), P._identity(rec["box_j"])
            q_i, q_j = adapter.quality[i_id], adapter.quality[j_id]
            if q_i is None or q_j is None:
                adapter.table[(i_id, j_id)] = (False, "infeasible_whole_box_read", 0.5, 0.0)
                continue
            r = min(1.0, max(0.0, 0.5 + (q_i - q_j) * scale))
            adapter.table[(i_id, j_id)] = (True, "ok", float(r), 0.0)
        return adapter

    def verdict(self, pair: P.PairInput) -> P.PairVerdict:
        ok, reason, r, a = self.table[(pair.identity_i, pair.identity_j)]
        if not ok:
            return P.PairVerdict(0, r, a, "infeasible_read")
        return P.decide_pair(pair, r, a)


def tournament(members: Sequence, affine_of: Callable, adapter) -> dict:
    """Copeland tournament on adapter verdicts; tie rules identical to the rules core."""
    votes = {id(m): 0 for m in members}
    stats = {"decided": 0, "abstained": 0, "infeasible": 0, "pairs": 0}
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            m_i, m_j = members[a], members[b]
            stats["pairs"] += 1
            pair = P.build_pair_input(m_i.box_xyxy, affine_of(m_i), m_j.box_xyxy, affine_of(m_j),
                                      identity_i=P._identity(m_i), identity_j=P._identity(m_j))
            verdict = adapter.verdict(pair)
            if verdict.winner == 0:
                stats["infeasible" if verdict.reason == "infeasible_read" else "abstained"] += 1
                continue
            stats["decided"] += 1
            votes[id(m_i if verdict.winner > 0 else m_j)] += 1
    ranked = sorted(members, key=lambda m: (-votes[id(m)],) + P.sort_key_of(m))
    return {"representative": ranked[0], "votes": {id(m): votes[id(m)] for m in members},
            "stats": stats}


def merge(candidates: Sequence, affine_of: Callable, adapter, *,
          budgets: tuple = P.FINAL_BUDGETS) -> dict:
    """Representative replacement + frozen re-ranking/truncation on adapter verdicts."""
    clusters = P.nms_clusters(candidates)
    c_kept = [cluster["survivor"] for cluster in clusters]
    kept, substitutions, all_abstain_clusters = [], 0, 0
    for cluster in clusters:
        members = P.cluster_members(cluster)
        if len(members) == 1:
            kept.append(cluster["survivor"])
            continue
        chosen = tournament(members, affine_of, adapter)
        representative = chosen["representative"]
        if chosen["stats"]["decided"] == 0:
            all_abstain_clusters += 1
        if representative is not cluster["survivor"]:
            substitutions += 1
        kept.append(representative)
    ranked = P.canonical_order(kept)
    c_ranked = P.canonical_order(c_kept)
    rank_shifted = sum(1 for a, b in zip(ranked, c_ranked) if a is not b)
    budget_delta = {}
    for budget in budgets:
        before, after = set(map(id, c_ranked[:budget])), set(map(id, ranked[:budget]))
        budget_delta[str(budget)] = {"entered": len(after - before), "left": len(before - after)}
    return {"core_version": P.CORE_VERSION, "adapter_kind": getattr(adapter, "kind", "?"),
            "nms_iou": P.NMS_IOU, "cluster_count": len(clusters), "kept_count": len(kept),
            "substitutions": substitutions, "all_abstain_clusters": all_abstain_clusters,
            "representative_ranking": P.POST_REPLACEMENT_RANKING,
            "rank_shifted_members": rank_shifted,
            "budget_membership_delta_vs_C": budget_delta,
            "top600": ranked[:budgets[0]], "top300": ranked[:budgets[1]]}


# --- frozen-detector protection and state digests ------------------------------
def tensor_state_digest(module: torch.nn.Module) -> str:
    """SHA-256 over the module's full state dict (names, dtypes, shapes, bytes)."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().to("cpu", copy=True).contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_no_detector_grads(model: torch.nn.Module) -> None:
    offenders = [name for name, p in model.named_parameters() if p.grad is not None]
    if offenders:
        raise ValueError(f"detector carries gradients: {offenders[:5]}")


# --- frozen sampling verification ----------------------------------------------
def verify_sampling(ids_path, jsonl_path, manifest: Mapping, *, progress=None) -> dict:
    """Re-verify the frozen training manifest against stage-0 rows, read-only.

    Checks: file line count and unique ids equal ``manifest['n_pair_total']``; every id is
    a stage-0 ``train`` pair with a valid read whose destination matches its line tag; the
    selected sets equal *all* stage-0 train-readable pairs of that destination (both
    directions), which is the frozen sampling rule given the available counts.
    """
    import json
    from pathlib import Path
    selected = {"pref": set(), "abstain": set()}
    for line in Path(ids_path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        tag, identifier = line.split("\t", 1)
        selected[tag].add(identifier)
    total = len(selected["pref"]) + len(selected["abstain"])
    if len(selected["pref"]) != manifest["n_pair_pref"]:
        raise ValueError("pref count differs from the frozen manifest")
    if len(selected["abstain"]) != manifest["n_pair_abstain_selected"]:
        raise ValueError("abstain count differs from the frozen manifest")
    if total != manifest["n_pair_total"]:
        raise ValueError("total count differs from the frozen manifest")
    all_ids = selected["pref"] | selected["abstain"]
    if len(all_ids) != total:
        raise ValueError("duplicate pair id in the sampling file")
    images = {identifier.split("|", 1)[0] for identifier in all_ids}
    seen = {"pref": set(), "abstain": set()}
    other_split = wrong_destination = invalid = missing = 0
    scanned = 0
    with open(jsonl_path, "r", encoding="utf-8") as stream:
        for line in stream:
            scanned += 1
            if line[2:66] not in images:
                continue
            row = json.loads(line)
            image_id = row[0]
            if image_id not in images:
                continue
            identifier = (f"{image_id}|{row[4][0]},{row[4][1]},{row[3]}"
                          f"|{row[5][0]},{row[5][1]},{row[3]}")
            if identifier not in all_ids:
                missing += 1
                continue
            if row[2] != "train":
                other_split += 1
            if not row[13]:
                invalid += 1
            tag = "abstain" if row[8] == "abstain" else "pref"
            if identifier in selected["pref"] and tag != "pref":
                wrong_destination += 1
            if identifier in selected["abstain"] and tag != "abstain":
                wrong_destination += 1
            seen[tag].add(identifier)
    report = {"sampling_spec_sha256": manifest["sampling_spec_sha256"],
              "pairs_total": total, "pref": len(selected["pref"]),
              "abstain": len(selected["abstain"]),
              "all_ids_found_in_train_rows": len(seen["pref"] | seen["abstain"]) == total,
              "rows_scanned": scanned,
              "rows_outside_train_split": other_split,
              "rows_with_invalid_read": invalid,
              "rows_with_wrong_destination_tag": wrong_destination,
              "selected_equals_all_train_readable": (seen["pref"] == selected["pref"]
                                                     and seen["abstain"] == selected["abstain"]),
              "status": "PASS"}
    if not (report["all_ids_found_in_train_rows"] and other_split == 0 and invalid == 0
            and wrong_destination == 0 and report["selected_equals_all_train_readable"]):
        report["status"] = "FAIL"
    return report


# --- freeze payload ------------------------------------------------------------
def freeze_payload(*, channels: int = FEAT.P3_CHANNELS) -> dict:
    spec = frozen_spec()
    arms = build_arms(channels=channels)
    return {"learning_spec": spec, "learning_spec_sha256": learning_spec_sha(),
            "feature_convention": FEAT.CONVENTION,
            "dims": {"side": FEAT.side_dim(channels), "whole": FEAT.whole_dim(channels),
                     "geometry": FEAT.pair_geometry_dim(), "channels": channels},
            "arms": arms["report"],
            "epochs_equivalent": round(spec["updates"] * spec["batch_pairs"] / 222_755, 3),
            "status": "frozen_pending_gpu_engineering_check"}
