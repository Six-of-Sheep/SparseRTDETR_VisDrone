"""PDRA limited GPU engineering check (one command, fixed scope; rev 6).

Scope (frozen by this file; the brief ``PDRA_GPU_CHECK_R2.md`` repeats it)
--------------------------------------------------------------------------
* Images: the two frozen raw-view batches ``3`` and ``6`` of the seed0 full-C run (eight
  development images, all in the module-eval split; chosen as the two lowest-index raw
  batches whose four images are all eval-split).
* Detector forwards: ``2 batches x 5 views x 2 passes = 20`` batch-of-4 forwards on the
  audited e120 EMA engine -- pass A without the tap, pass B with the read-only P3 tap.
  No warm-up batch, no other forward.  Detector updates: 0.
* Comparator acceptance updates: the frozen total of ``50`` AdamW updates is a fixed
  three-arm allocation ``ARM_ALLOCATION = {pdra: 30, c: 10, b: 10}``.  Every arm trains
  under the frozen learning spec verbatim (optimizer, loss, batch 256, seed); only
  ``updates`` is set to the arm's allocation, exactly as rev 4 set it for its single arm.
  (rev 4 trained the PDRA arm alone for 50; rev 5 fixes the allocation inside the same
  total.)
* Ground truth: pooling and the complete decision enumeration are GT-free
  (``pair_decisions`` has no GT parameter).  Annotations are opened for one purpose
  only -- the (b)-arm member targets (matched-GT IoU, built by the identical stage-1
  ``_gt_items`` helper chain) -- and only after each image's decision phase.  The pair
  labels for the PDRA/(c) arms come from the frozen stage-0 rows, as in rev 4.
* Runtime constraints are enforced *while* the check runs, not only at the end:
  wall <= 1800 s and output <= 256 MiB; checked at input resolution, after the engine
  restore, after every forward pass, after every decided image, after every comparator
  update and after every publish.  rev 6 removed the artificial memory threshold and
  the allocator quota (2026-10-31 authorization): CUDA allocated/reserved peaks are
  recorded for the whole run (restore phase included) but not gated, and a genuine OOM
  surfaces as an exception -> STOP.json (no batch or precision fallback, no retry).
  Host VmHWM is reported per phase, not gated.
* Exception accounting: any failure writes ``STOP.json`` with the exception type and
  message, the full traceback and the executed-count accounting (phase reached, forward
  calls, per-arm updates, observed peaks).  No automatic retry.
* Gates (each FAILs the check, no retry):
  1. tap-vs-no-tap: ``pred_logits``/``pred_boxes`` bitwise equal per view, decoded pools
     identical; tap fires once per forward; P3 shape ``[4, 256, 96, 168]`` on every view;
  2. real pooling: every readable pair of every image pooled through the frozen RoIAlign
     rule on the real P3 maps (counts, strips, microseconds per pair reported);
  3. comparator gradients and frozen detector state: each of the three allocated
     trainings gives every arm parameter a finite non-zero gradient and changes the arm;
     the detector (model and EMA) is asserted gradient-free after every arm, and its
     complete captured state (model, EMA, optimizer, RNG) is exactly equal before and
     after the whole check.  rev 4's "backward on detector tensors" counter was a no-op
     (the patched symbol is not on ``Tensor.backward``'s path and its increment added
     zero); it is removed rather than fixed -- the freeze is evidenced by measured
     parameter gradients, captured-state equality and the EMA digest;
  4. complete pairing decisions: every cluster-internal pair of the eight images is
     decided through the GT-free inference enumeration; the pair count equals the stage-0
     ledger for these images; readability agrees pair by pair with the stage-0 rows when
     the raw views reproduce exactly (otherwise reported, not gated);
  5. resource peaks: wall per phase within the enforced cap; CUDA peak allocated/reserved
     (restore included) and host VmHWM recorded for the whole run (memory is not gated in
     rev 6; a real OOM fails the run as an exception).
* Acceptance boundary: a PASS is an engineering wiring acceptance on eight frozen
  images, nothing more.  ``ACCEPTANCE_SCOPE`` (written into PLAN/RESULT/CLOSED) states
  that it authorizes neither formal learning (stage-1 20,000 updates), nor scoring, nor
  the 548-image pass, and settles no capacity-control question; the only permitted
  follow-up is the commit that flips ``pdra_core.FEATURE_BINDING_VERIFIED`` together
  with this RESULT.json SHA.
* Not in scope: the 548-image pass, stage-1 training, any AP/AR scoring, stage-2
  inference, any change to detector weights or outputs.

``dry-run`` resolves and SHA-checks every input, computes the expected counts from the
stage-0 rows and writes ``PLAN.json`` (allocation, constraints, acceptance scope, zero
executed counts) -- CPU only, no CUDA, no model.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _path in (str(HERE), str(REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pdra_core as P

TASK = "r18-pdra-gpu-check-20261031-002"
CHECK_REVISION = "pdra-gpu-check-rev6-20261031"
PARENT_SHA = "b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28"
STAGE0_DIR = Path("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/"
                  "localization-distribution/r18-pdra-stage0-20261030-001/attempt-001")
STAGE0_SHA = {
    "HPARAMS_FROZEN.json": "be512f47d413842450e1b2528b903ffcad57a65ec5095b0208e92435158485b0",
    "SPLIT_SEQUENCES.json": "2efba5f38aedbc41dbeb0af6d9fdf176f056a3a7b68c1196eda7288cd26e7a76",
    "SAMPLING_MANIFEST.json": "7a931d9289b7c840e10d9a6af48bd3ddf864972735652c4276c5f0c04c366778",
    "STAGE0_SUMMARY.json": "c9c27cf98e39179cc64abac741af5b4701c1c4b027fae3148ec69c5c0251d9ab",
    "seed0-pairs.jsonl": "54c604e089af237e47f3cc68d1d9e407cdd043f9762c8b300330d02e456d1921",
    "seed0-IMAGE_COUNTS.json": "69d555abdcb7fdba7f44c59e08677ae6a7fc43fbbb02fc825197091395a6faf7",
}
FROZEN_BATCHES = (3, 6)
CHECK_UPDATES = 50
# Fixed three-arm acceptance allocation inside the frozen total (rev 5).  The PDRA arm is
# the mechanism under test and keeps the largest share; the two controls receive enough
# updates to prove their own training path (gradients on every parameter, parameters
# change, detector untouched).  Sum must equal CHECK_UPDATES.
ARM_ALLOCATION = {"pdra_difference_strips": 30, "c_whole_box_pairwise": 10, "b_single_box_quality": 10}
# rev 6: the wall clock and output bytes are the enforced caps; CUDA allocated/reserved
# peaks are recorded for the whole run but no longer gated (the rev-5 12 GiB threshold
# and any allocator quota are removed per the 2026-10-31 authorization).  A real OOM is
# an exception -> STOP.json; no batch or precision fallback, no retry.
CAPS = {"wall_seconds": 1800, "output_bytes": 256 * 1024 ** 2,
        "forward_calls": 20, "comparator_updates": CHECK_UPDATES}
EXPECTED_P3 = (256, 96, 168)
ACCEPTANCE_SCOPE = {
    "kind": "engineering_wiring_check_only",
    "authorizes_formal_learning": False,
    "authorizes_stage1_training": False,
    "authorizes_stage2_inference": False,
    "authorizes_ap_ar_scoring": False,
    "authorizes_548_image_pass": False,
    "settles_capacity_control": False,
    "permitted_followups": [
        "a new commit flipping pdra_core.FEATURE_BINDING_VERIFIED to True and registering this RESULT.json SHA",
        "reports and shared handoff records"],
    "statement": ("A PASS verifies the wiring on eight frozen images only. It is not evidence of "
                  "effectiveness, it does not settle the formal-learning capacity-control question, "
                  "and it must not be used to start formal learning (stage 1), scoring or the "
                  "548-image pass; each of those needs a separate user authorization."),
}
GT_USE = {
    "pooling_and_decisions_gt_free": True,
    "annotations_opened_for": ("the (b)-arm member matched-GT IoU targets of its 10 acceptance updates, "
                               "through the identical stage-1 _gt_items chain, after each image's "
                               "decision phase"),
    "annotations_opened_by_any_other_phase": False,
    "pair_labels_source": "frozen stage-0 rows (seed0-pairs.jsonl), as in rev 4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def exact(ref: dict) -> Path:
    p = Path(ref["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), "bad reference: " + str(p))
    require(p.stat().st_size == ref.get("bytes", ref.get("size_bytes")), "size differs: " + str(p))
    require(sha(p) == ref["sha256"], "SHA differs: " + str(p))
    return p


def publish(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "immutable output already exists: " + str(path))
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def rss_high_water() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise ValueError("VmHWM unavailable")


def output_bytes(out: Path) -> int:
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


# --- runtime constraints and accounting (rev 6) ---------------------------------
class RuntimeConstraintError(RuntimeError):
    """A runtime resource constraint was exceeded; ``observation`` holds the values."""

    def __init__(self, message: str, observation: dict) -> None:
        super().__init__(message)
        self.observation = observation


def enforce_runtime(*, started: float, label: str, caps: Mapping = CAPS, out: Path | None = None,
                    torch_mod=None, accounting: "Accounting | None" = None,
                    peak_floor: int | None = None) -> dict:
    """Record resource observations; enforce the wall-clock and output constraints.

    Called at every phase boundary, after every forward pass, after every decided image,
    after every comparator update and after every publish, so a breach fails the run where
    it happens and the failure accounting shows the observed values.  CUDA allocated and
    reserved peaks are recorded for the whole run -- ``peak_floor`` carries the earlier,
    pre-reset restore-phase reserved peak -- but are not gated (rev 6 removed the
    artificial memory threshold and allocator quota; a genuine OOM surfaces as an
    exception).  Without ``torch_mod`` (CPU entries, tests) no CUDA values are consulted.
    """
    elapsed = time.monotonic() - started
    observed = {"label": label, "elapsed_seconds": round(elapsed, 3),
                "cuda_peak_reserved_bytes": None, "cuda_peak_allocated_bytes": None,
                "output_bytes": None}
    if torch_mod is not None and torch_mod.cuda.is_initialized():
        reserved = int(torch_mod.cuda.max_memory_reserved())
        if peak_floor is not None:
            reserved = max(reserved, int(peak_floor))
        observed["cuda_peak_reserved_bytes"] = reserved
        observed["cuda_peak_allocated_bytes"] = int(torch_mod.cuda.max_memory_allocated())
    if out is not None:
        observed["output_bytes"] = output_bytes(out)
    if accounting is not None:
        accounting.observe(observed)
    breach = None
    if elapsed > float(caps["wall_seconds"]):
        breach = "wall_seconds"
    elif observed["output_bytes"] is not None and observed["output_bytes"] > int(caps["output_bytes"]):
        breach = "output_bytes"
    if breach is not None:
        raise RuntimeConstraintError(f"runtime constraint {breach} exceeded at {label}: {observed}", observed)
    return observed


class Accounting:
    """Executed-count accounting of one run; snapshotted into STOP.json and RESULT.json."""

    def __init__(self, task: str, mode: str) -> None:
        self.data = {"task": task, "mode": mode, "phase": "init", "phases_completed": [],
                     "executed_forward_calls": 0,
                     "executed_updates": {key: 0 for key in ARM_ALLOCATION},
                     "budget_observations": 0, "last_budget_observation": None,
                     "cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None}

    def phase(self, name: str) -> None:
        self.data["phase"] = name

    def complete(self, name: str) -> None:
        self.data["phases_completed"].append(name)

    def forwards(self, count: int) -> None:
        self.data["executed_forward_calls"] += int(count)

    def update(self, arm_key: str) -> None:
        self.data["executed_updates"][arm_key] += 1

    def observe(self, observation: Mapping) -> None:
        self.data["budget_observations"] += 1
        self.data["last_budget_observation"] = dict(observation)
        for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
            value = observation.get(key)
            if value is not None and (self.data[key] is None or value > self.data[key]):
                self.data[key] = int(value)

    def snapshot(self) -> dict:
        return json.loads(json.dumps(self.data))


class _Table:
    """Comparator view over a precomputed adapter table (``r``/``a`` per identity pair)."""

    def __init__(self, table):
        self.table = table

    def r(self, pair):
        return self.table[(pair.identity_i, pair.identity_j)][2]

    def a(self, pair):
        return self.table[(pair.identity_i, pair.identity_j)][3]


class _View:
    __slots__ = ("view_id",)

    def __init__(self, view_id):
        self.view_id = view_id


# --- inputs -------------------------------------------------------------------
def resolve_inputs(parent_path: Path, batches) -> dict:
    require(sha(parent_path) == PARENT_SHA, "parent contract SHA differs")
    parent = read(parent_path)
    seed0 = read(exact(parent["seed0_contract"]))
    require(seed0["task_id"] == "r18-sahi-fixed-20261016-001", "seed0 contract task differs")
    endpoint = read(exact(seed0["endpoint"]))
    require(endpoint["status"] == "R18_COMMON_E120_EMA_LOCKED" and endpoint["epoch"] == 120
            and endpoint["checkpoint"] == seed0["checkpoint"], "endpoint/checkpoint lineage differs")
    training = read(exact(seed0["parent_contract"]))
    require(training["task_id"] == "rtdetrv2-r18-common-protocol-seed0-e120-20261013-001",
            "training contract task differs")
    training_root = Path(seed0["parent_source_root"])
    require(training_root.is_dir() and training_root == Path(training["source_root"]),
            "training source root differs")
    binding_path = Path(training["job_root"]) / "formal" / "binding.json"
    require(binding_path.is_file(), "training binding missing")
    full = read(Path(seed0["job_root"]) / "full-C" / "RESULT.json")
    require(full["status"] == "PASS" and full["phase"] == "full-C" and len(full["raw_view_refs"]) == 137,
            "seed0 full-C result differs")
    raw_refs = {k: full["raw_view_refs"][k] for k in batches}
    for ref in raw_refs.values():
        exact(ref)
    stage0 = {}
    for name, expected in STAGE0_SHA.items():
        path = STAGE0_DIR / name
        require(path.is_file() and sha(path) == expected, f"stage-0 artifact differs: {name}")
        stage0[name] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    hparams = read(STAGE0_DIR / "HPARAMS_FROZEN.json")
    split = read(STAGE0_DIR / "SPLIT_SEQUENCES.json")
    sampling = read(STAGE0_DIR / "SAMPLING_MANIFEST.json")
    counts = read(STAGE0_DIR / "seed0-IMAGE_COUNTS.json")
    image_rows = {}
    for k in batches:
        for row in counts[4 * k:4 * k + 4]:
            require(row["split"] == "eval", "frozen batch holds a non-eval image")
            image_rows[row["image_id"]] = row
    require(len(image_rows) == 4 * len(batches), "image rows incomplete")
    arms = [P.ArmSpec(key, reads, hparams["candidate_pool_sha256"],
                      split_sha256=hparams["split_sha256"],
                      eval_members_sha256=hparams["eval_members_sha256"],
                      sampling_spec_sha256=hparams["sampling_spec_sha256"])
            for key, reads in (("a_original_c", "score"), ("b_single_box_quality", "whole_box_quality"),
                               ("c_whole_box_pairwise", "whole_box_pairwise"),
                               ("pdra", "difference_strips"))]
    fairness = P.assert_control_fairness(arms)
    require(fairness["fair"], "arm specifications are not fair")
    return {"parent": parent, "seed0": seed0, "endpoint": endpoint, "training": training,
            "training_root": training_root, "binding_path": binding_path,
            "checkpoint": seed0["checkpoint"], "gpu_uuid": seed0["gpu_uuid"],
            "raw_refs": raw_refs, "stage0": stage0, "hparams": hparams, "split": split,
            "sampling": sampling, "image_rows": image_rows, "fairness": fairness,
            "data_spec": training["data"]["development"]}


def stage0_rows_for(images: set) -> dict:
    """``{image: {(id_i, id_j): row}}`` from the stage-0 jsonl, for the chosen images."""
    rows = {image: {} for image in images}
    with (STAGE0_DIR / "seed0-pairs.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line[2:66] not in images:
                continue
            row = json.loads(line)
            if row[0] not in images:
                continue
            key = ((row[4][0], row[4][1], row[3]), (row[5][0], row[5][1], row[3]))
            rows[row[0]][key] = row
    return rows


def expected_counts(rows_by_image: dict) -> dict:
    total, valid = Counter(), Counter()
    reasons = Counter()
    for image, pairs in rows_by_image.items():
        for row in pairs.values():
            total[row[8]] += 1
            valid[row[8]] += int(row[13])
            if not row[13]:
                reasons[row[14]] += 1
    return {"pairs": sum(total.values()), "valid": sum(valid.values()),
            "by_destination": dict(total), "valid_by_destination": dict(valid),
            "infeasible_reasons": dict(reasons)}


def plan(inputs: dict, batches) -> dict:
    images = set(inputs["image_rows"])
    rows_by_image = stage0_rows_for(images)
    expected = expected_counts(rows_by_image)
    labelled = sum(expected["valid_by_destination"].get(d, 0) for d in ("pref", "abstain"))
    members = set()
    for image, pairs in rows_by_image.items():
        for (id_i, id_j), row in pairs.items():
            if row[8] in ("pref", "abstain") and row[13]:
                members.add((image, id_i))
                members.add((image, id_j))
    clusters = sum(r["clusters"] for r in inputs["image_rows"].values())
    pools = sum(r["pool"] for r in inputs["image_rows"].values())
    return {"task": TASK, "check_revision": CHECK_REVISION, "batches": list(batches),
            "images": sorted(images), "image_rows": inputs["image_rows"],
            "forward_calls": {"pass_A_no_tap": 5 * len(batches), "pass_B_tap": 5 * len(batches),
                              "total": 10 * len(batches), "warmup": 0, "batch_size": 4},
            "comparator_updates": {"total": CHECK_UPDATES, "allocation": dict(ARM_ALLOCATION),
                                   "batch_pairs": 256,
                                   "spec": ("frozen learning spec verbatim (AdamW lr 3e-4, betas 0.9/0.999, "
                                            "eps 1e-8, wd 1e-2, seed 20261030); only 'updates' is set per arm")},
            "detector_updates": 0,
            "detector_gradients_expected": 0,
            "detector_freeze_evidence": [
                "every detector parameter keeps grad None, asserted after each arm",
                "captured full state (model, EMA, optimizer, RNG) exactly equal before and after",
                "EMA state digest unchanged",
                "no backward counter (the rev-4 counter was a no-op and is removed)"],
            "runtime_constraints": {
                "enforced": {"wall_seconds": CAPS["wall_seconds"],
                             "output_bytes": CAPS["output_bytes"]},
                "checked_at": ("input resolution; restore completion; every forward pass; every decided "
                               "image; every comparator update; every publish; final resources"),
                "recorded_not_gated": ["CUDA peak allocated/reserved for the whole run (restore phase "
                                       "included)", "host VmHWM per phase"],
                "memory_gate_removed": ("rev 6 removed the artificial CUDA-reserved threshold and the "
                                        "allocator quota (2026-10-31 authorization); a real OOM fails the "
                                        "run as an exception, with no batch or precision fallback"),
                "on_breach": "RuntimeConstraintError -> STOP.json with accounting; no retry"},
            "exception_accounting": ("STOP.json carries exception type/message, traceback and the executed "
                                     "counts (phase, forward calls, per-arm updates, observed peaks)"),
            "gt_use": GT_USE,
            "acceptance_scope": ACCEPTANCE_SCOPE,
            "expected": {"clusters": clusters, "pool": pools, "labelled_pairs": labelled,
                         "labelled_members_upper_bound": len(members), **expected},
            "caps": CAPS, "feature_level": P.FEATURE_LEVEL, "feature_stride": list(P.FEATURE_STRIDE),
            "expected_p3_shape": [4, *EXPECTED_P3], "gpu_uuid": inputs["gpu_uuid"],
            "checkpoint": inputs["checkpoint"], "fairness": inputs["fairness"],
            "stage0_inputs": inputs["stage0"], "raw_refs": inputs["raw_refs"],
            "not_in_scope": ["548-image pass", "stage-1 training (formal learning, 20,000 updates)",
                             "stage-2 inference", "AP/AR scoring",
                             "detector weight or output changes", "capacity-control decisions"]}


# --- shared comparator acceptance (GPU check and CPU acceptance) ----------------
def parameter_gradient_report(module) -> dict:
    """Per-parameter gradient presence/finiteness/non-zeroness after a training call."""
    import torch
    per_parameter = {}
    for name, parameter in module.named_parameters():
        grad = parameter.grad
        finite = bool(grad is not None and torch.isfinite(grad).all())
        nonzero = bool(grad is not None and float(grad.abs().sum()) > 0.0)
        per_parameter[name] = {"present": grad is not None, "finite": finite, "nonzero": nonzero}
    ok = all(r["present"] and r["finite"] and r["nonzero"] for r in per_parameter.values())
    return {"parameters_total": len(per_parameter),
            "parameters_with_gradient": sum(1 for r in per_parameter.values() if r["present"]),
            "all_with_finite_nonzero_gradient": ok, "per_parameter": per_parameter}


def comparator_acceptance_updates(arms: Mapping, pair_material: Mapping, quality_material: Mapping, *,
                                  allocation: Mapping | None = None, device: str = "cuda:0",
                                  spec_base: Mapping | None = None, on_update: Callable | None = None,
                                  on_arm_complete: Callable | None = None,
                                  allow_small_batch: bool = False) -> dict:
    """Run the fixed three-arm acceptance allocation under the frozen learning spec.

    ``pair_material`` holds the labelled readable pairs (``s_i/s_j`` strip descriptors for
    PDRA, ``w_i/w_j`` whole-box descriptors for (c), shared ``geom``/``pref_target``/
    ``abstain_target``); ``quality_material`` holds the (b) member descriptors and their
    matched-GT IoU targets.  Each arm trains with ``updates`` set to its allocation and every
    other spec field frozen.  ``on_update(arm_key, record)`` fires after every optimizer
    step (runtime accounting/enforcement); ``on_arm_complete(arm_key)`` fires after each arm
    (detector-freeze assertions live in the caller).  ``allow_small_batch`` is for the
    single-image CPU acceptance only; the GPU check keeps the frozen batch of 256.
    """
    import pdra_learn as L
    allocation = dict(ARM_ALLOCATION if allocation is None else allocation)
    require(sum(allocation.values()) == CHECK_UPDATES,
            f"arm allocation {allocation} does not sum to the frozen {CHECK_UPDATES} updates")
    require(set(allocation) == set(arms["arms"]), "allocation keys differ from the arms")
    require(set(allocation) == set(L.LEARNING_SPEC["arm_keys"]), "allocation keys differ from the frozen arm keys")
    spec_base = L.frozen_spec() if spec_base is None else dict(spec_base)
    frozen_batch = int(spec_base["batch_pairs"])
    reports = {}

    def callback(key):
        return None if on_update is None else (lambda record, _key=key: on_update(_key, record))

    def finish(key, arm, train, n, digest_before):
        require(train["updates"] == int(allocation[key]), f"{key} update count differs from its allocation")
        gradient = parameter_gradient_report(arm)
        offenders = [name for name, row in gradient["per_parameter"].items()
                     if not (row["present"] and row["finite"] and row["nonzero"])]
        require(not offenders, f"{key} parameters without finite non-zero gradient: {offenders}")
        digest_after = L.tensor_state_digest(arm)
        require(digest_after != digest_before, f"{key} parameters did not change")
        if on_arm_complete is not None:
            on_arm_complete(key)
        reports[key] = {"updates": train["updates"], "batch": train["batch"], "material_rows": n,
                        "first_losses": train["first_losses"], "mean_loss": train["mean_loss"],
                        "seconds": train["seconds"], "gradient": gradient,
                        "digest_before": digest_before, "digest_after": digest_after,
                        "parameters_changed": True,
                        "steps_exactly_allocated": train["updates"] == int(allocation[key])}

    pair_inputs = {"pdra_difference_strips": (pair_material["s_i"], pair_material["s_j"]),
                   "c_whole_box_pairwise": (pair_material["w_i"], pair_material["w_j"])}
    for key in ("pdra_difference_strips", "c_whole_box_pairwise"):
        sides_i, sides_j = pair_inputs[key]
        n = int(sides_i.shape[0])
        spec = {**spec_base, "updates": int(allocation[key])}
        if allow_small_batch and n < frozen_batch:
            spec["batch_pairs"] = n
        require(n >= int(spec["batch_pairs"]), f"labelled material for {key} ({n}) is smaller than one batch")
        tensors = L.PairTensors(sides_i, sides_j, pair_material["geom"],
                                pair_material["pref_target"], pair_material["abstain_target"])
        arm = arms["arms"][key].to(device)
        digest_before = L.tensor_state_digest(arm)
        train = L.train_pair_arm(arm, tensors, spec=spec, device=device, progress=callback(key))
        finish(key, arm, train, n, digest_before)

    key = "b_single_box_quality"
    descs, targets = quality_material["desc"], quality_material["target"]
    n = int(descs.shape[0])
    spec = {**spec_base, "updates": int(allocation[key])}
    if allow_small_batch and n < frozen_batch:
        spec["batch_pairs"] = n
    require(n >= int(spec["batch_pairs"]) and n > 0, f"member material for {key} ({n}) is smaller than one batch")
    arm = arms["arms"][key].to(device)
    digest_before = L.tensor_state_digest(arm)
    train = L.train_quality_arm(arm, descs, targets, spec=spec, device=device, progress=callback(key))
    finish(key, arm, train, n, digest_before)

    total = sum(r["updates"] for r in reports.values())
    require(total == CHECK_UPDATES, f"executed updates {total} differ from the frozen {CHECK_UPDATES}")
    return {"allocation": allocation, "total_updates": total, "batch_pairs": frozen_batch,
            "small_batch_fallback_used": bool(allow_small_batch and any(r["batch"] < frozen_batch
                                                                        for r in reports.values())),
            "spec_sha256": L.learning_spec_sha(spec_base), "arms": reports}


def b_quality_material(data, row, by_id: Mapping, affine_of: Callable, bank, member_ids) -> dict:
    """(b)-arm member descriptors and matched-GT IoU targets for one image.

    This is the only place the check opens annotations; callers invoke it after the image's
    GT-free pooling and decision phase.  Semantics are the stage-1 ones: one whole-box read
    per unique member, invalid reads skipped, target = best matched-GT IoU of the member's
    category or 0.0 when nothing matches.
    """
    import torch
    import pdra_features as FEAT
    from pdra_stage1 import _gt_items
    from sparse_rtdetr.r0 import evaluation as ev
    gts = _gt_items(data, row, ev)
    kept = []
    for ident in sorted(member_ids):
        candidate = by_id[ident]
        read = FEAT.whole_box_read(candidate.box_xyxy, affine_of(candidate))
        if read.valid:
            kept.append((ident, read))
    if kept:
        with torch.no_grad():
            desc = FEAT.pool_whole(bank, [read for _, read in kept]).cpu()
    else:
        desc = torch.zeros((0, FEAT.whole_dim(bank.channels)), dtype=torch.float16)
    targets = []
    for ident, _ in kept:
        candidate = by_id[ident]
        hits = P.match_gts_iou(candidate.box_xyxy, candidate.category_id,
                               gts.get(candidate.category_id, []))
        targets.append(float(hits[0][0]) if hits else 0.0)
    return {"desc": desc, "target": torch.tensor(targets, dtype=torch.float32),
            "members": len(member_ids), "whole_box_readable": len(kept),
            "zero_targets": sum(1 for t in targets if t == 0.0),
            "gt_chain": "pdra_stage1._gt_items (stage-1 chain), opened after the image's decision phase"}


# --- the GPU check -------------------------------------------------------------
def run(inputs: dict, out: Path, batches, source_commit: str, *, budget_started: float | None = None,
        accounting: Accounting | None = None) -> dict:
    import torch
    import pdra_features as FEAT
    import pdra_learn as L
    import pdra_tap as TAP
    started = time.monotonic() if budget_started is None else budget_started
    accounting = Accounting(TASK, "run") if accounting is None else accounting
    phases = {}
    images = set(inputs["image_rows"])
    plan_record = plan(inputs, batches)
    publish(out / "PLAN.json", plan_record)
    enforce_runtime(started=started, label="plan", out=out, accounting=accounting)
    rows_by_image = stage0_rows_for(images)

    # 1. audited restore of the frozen seed0 e120 EMA engine (training-release imports);
    #    the restore-phase CUDA peak is recorded before the phase stats reset (rev 6:
    #    recorded, not gated)
    accounting.phase("restore")
    enforce_runtime(started=started, label="pre_restore", out=out, accounting=accounting)
    t0 = time.monotonic()
    restored = TAP.restore_frozen_seed0_engine(
        training_source_root=inputs["training_root"], train_contract=inputs["training"],
        binding_path=inputs["binding_path"], checkpoint_path=inputs["checkpoint"]["path"],
        checkpoint_sha256=inputs["checkpoint"]["sha256"], gpu_uuid=inputs["gpu_uuid"])
    engine, binding, model = restored["engine"], restored["binding"], restored["model"]
    from sparse_rtdetr.protocol_r18 import checkpoint
    from sparse_rtdetr.r0.evaluation_data import Development548
    data = Development548(inputs["data_spec"])
    arms = L.build_arms()                       # consumes the global CPU RNG before capture
    before = checkpoint.capture(engine, binding)
    detector_digest_before = L.tensor_state_digest(engine.ema.module)
    restore_peak_allocated = int(torch.cuda.max_memory_allocated())
    restore_peak_reserved = int(torch.cuda.max_memory_reserved())
    worst_peak = {"allocated": restore_peak_allocated, "reserved": restore_peak_reserved}

    def note_peaks() -> dict:
        allocated, reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
        worst_peak["allocated"] = max(worst_peak["allocated"], allocated)
        worst_peak["reserved"] = max(worst_peak["reserved"], reserved)
        return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}

    def guard(label: str) -> dict:
        return enforce_runtime(started=started, label=label, out=out, torch_mod=torch,
                               accounting=accounting, peak_floor=worst_peak["reserved"])

    phases["restore"] = {"seconds": time.monotonic() - t0, "restored": restored["restored"],
                         "runtime": restored["runtime"], "host_hwm_bytes": rss_high_water(),
                         "peak_allocated_bytes": restore_peak_allocated,
                         "peak_reserved_bytes": restore_peak_reserved}
    guard("restore")                            # records the restore-phase peak floor
    torch.cuda.reset_peak_memory_stats()
    accounting.complete("restore")

    # 2. pass A (no tap) and pass B (tap) on the frozen batches
    accounting.phase("forward")
    t0 = time.monotonic()
    pass_b, tap_report = {}, []
    calls = 0
    for k in batches:
        rows = data.records[4 * k:4 * k + 4]
        frozen = torch.load(str(exact(inputs["raw_refs"][k])), map_location="cpu", weights_only=True)
        require(frozen["image_ids"] == [r["stable_image_id"] for r in rows], "raw batch image order differs")
        a = TAP.five_view_forward(data, rows, model, tap=None)
        calls += a["forward_calls"]
        accounting.forwards(a["forward_calls"])
        guard(f"forward/batch-{k}/pass-A")
        with TAP.EncoderTap(model) as tap:
            b = TAP.five_view_forward(data, rows, model, tap=tap)
        calls += b["forward_calls"]
        accounting.forwards(b["forward_calls"])
        guard(f"forward/batch-{k}/pass-B")
        views = []
        for va, vb, vf in zip(a["records"], b["records"], frozen["views"]):
            with torch.no_grad():
                logits_equal = bool(torch.equal(va["pred_logits"], vb["pred_logits"]))
                boxes_equal = bool(torch.equal(va["pred_boxes"], vb["pred_boxes"]))
                frozen_logits_equal = bool(torch.equal(va["pred_logits"], vf["pred_logits"]))
                frozen_boxes_equal = bool(torch.equal(va["pred_boxes"], vf["pred_boxes"]))
                frozen_logits_diff = float((va["pred_logits"] - vf["pred_logits"]).abs().max())
                frozen_boxes_diff = float((va["pred_boxes"] - vf["pred_boxes"]).abs().max())
            require(logits_equal and boxes_equal, f"tap changed detector output on batch {k} view {va['view_id']}")
            views.append({"view_id": va["view_id"], "tap_equal": True,
                          "frozen_raw_equal": frozen_logits_equal and frozen_boxes_equal,
                          "frozen_max_abs_diff": {"logits": frozen_logits_diff, "boxes": frozen_boxes_diff}})
        require(tap.calls == 5 and all(s == (4, *EXPECTED_P3) for s in tap.shapes),
                f"tap shape/count differs: {tap.calls} {tap.shapes[:1]}")
        pools_equal = all([c.as_dict() for c in pa] == [c.as_dict() for c in pb]
                          for pa, pb in zip(a["candidates"], b["candidates"]))
        require(pools_equal, "decoded candidate pools differ between passes")
        require(a["geometry"] == b["geometry"] == frozen["geometry"], "crop geometry differs")
        pass_b[k] = b
        tap_report.append({"batch": k, "image_ids": frozen["image_ids"], "views": views,
                           "tap_calls": tap.calls, "p3_shape": list(tap.shapes[0]),
                           "pools_identical_between_passes": True,
                           "timing_A": a["timing"], "timing_B": b["timing"]})
        del a
    require(calls == CAPS["forward_calls"], f"forward count {calls} differs from the frozen {CAPS['forward_calls']}")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)),
            "forward passes changed the complete detector state")
    L.assert_no_detector_grads(engine.model)
    L.assert_no_detector_grads(engine.ema.module)
    raw_exact = all(v["frozen_raw_equal"] for rep in tap_report for v in rep["views"])
    phases["forward"] = {"seconds": time.monotonic() - t0, "forward_calls": calls,
                         "frozen_raw_views_bitwise_equal": raw_exact,
                         "host_hwm_bytes": rss_high_water(), **note_peaks()}
    publish(out / "TAP_RESULT.json", {"batches": tap_report, "forward_calls": calls,
                                      "frozen_raw_views_bitwise_equal": raw_exact,
                                      "feature_level": P.FEATURE_LEVEL,
                                      "feature_stride": list(P.FEATURE_STRIDE),
                                      "p3_extent_matches_core": [EXPECTED_P3[1], EXPECTED_P3[2]] ==
                                      list(P.ViewAffine((0.0, 0.0, 10.0, 10.0)).feature_extent)})
    guard("forward/published")
    accounting.complete("forward")

    # 3. real pooling + complete pairing decisions per image (no GT in this phase), then
    #    the training material: pair labels from the stage-0 rows, (b) member targets
    #    from annotations opened only after the image's decisions
    accounting.phase("pool_decide")
    t0 = time.monotonic()
    pdra_arm = arms["arms"]["pdra_difference_strips"].to("cuda:0")
    c_arm = arms["arms"]["c_whole_box_pairwise"].to("cuda:0")
    b_arm = arms["arms"]["b_single_box_quality"].to("cuda:0")
    per_image = []
    decision_totals, b_totals = Counter(), Counter()
    mismatch_total = 0
    parts = {name: [] for name in ("s_i", "s_j", "w_i", "w_j", "geom", "pref", "abst", "desc", "target")}
    for k in batches:
        passb = pass_b[k]
        rows = data.records[4 * k:4 * k + 4]
        for j, row in enumerate(rows):
            image = row["stable_image_id"]
            pool = passb["candidates"][j]
            tiles = {t["view_id"]: tuple(float(v) for v in t["crop_box"]) for t in passb["geometry"][j]}
            affines = {v: P.candidate_affine(_View(v), (row["height"], row["width"]), tiles) for v in range(5)}
            affine_of = lambda c, _a=affines: _a[c.view_id]
            bank = FEAT.FeatureBank({v: passb["p3_maps"][v][j] for v in range(5)})
            records = P.cluster_pair_records(pool)
            pairs = [P.build_pair_input(r["box_i"].box_xyxy, affine_of(r["box_i"]),
                                        r["box_j"].box_xyxy, affine_of(r["box_j"]),
                                        identity_i=P._identity(r["box_i"]),
                                        identity_j=P._identity(r["box_j"])) for r in records]
            valid_pairs = [p for p in pairs if p.valid]
            strips = sum(len(p.left.feature_strips) + len(p.right.feature_strips) for p in valid_pairs)
            torch.cuda.synchronize()
            p0 = time.perf_counter()
            with torch.no_grad():
                d_i = FEAT.pool_sides(bank, [p.left for p in valid_pairs])
                d_j = FEAT.pool_sides(bank, [p.right for p in valid_pairs])
            torch.cuda.synchronize()
            pool_seconds = time.perf_counter() - p0
            require(torch.isfinite(d_i.float()).all() and torch.isfinite(d_j.float()).all(), "pooled features not finite")
            adapter = L.PairBankAdapter.build(pdra_arm, bank, pool, affine_of, "pdra")
            decisions = P.pair_decisions(pool, affine_of, _Table(adapter.table))
            merged = L.merge(pool, affine_of, adapter)
            require(len(decisions) == len(records) == len(rows_by_image[image]),
                    f"pair count differs from stage 0 on {image[:8]}")
            stage0_valid = sum(int(r[13]) for r in rows_by_image[image].values())
            mismatches = 0
            for d in decisions:
                r0 = rows_by_image[image][(d["identity_i"], d["identity_j"])]
                if bool(r0[13]) != d["valid"] or (r0[13] and r0[14] != d["read_reason"]):
                    mismatches += 1
            if raw_exact:
                require(mismatches == 0, f"readability differs from stage 0 on {image[:8]}")
            mismatch_total += mismatches
            reasons = Counter(d["decision_reason"] for d in decisions)
            decision_totals.update(reasons)
            first = k == batches[0] and j == 0
            c_adapter = L.PairBankAdapter.build(c_arm, bank, pool, affine_of, "c") if first else None
            b_adapter = L.QualityBankAdapter.build(b_arm, bank, pool, affine_of) if first else None
            # training material (a): labelled readable pairs, labels from the stage-0 rows
            by_id = {P._identity(c): c for c in pool}
            keep, pref_t, abst_t = [], [], []
            for (ii, jj), r0 in rows_by_image[image].items():
                if r0[8] not in ("pref", "abstain") or not r0[13]:
                    continue
                pair = P.build_pair_input(by_id[ii].box_xyxy, affine_of(by_id[ii]), by_id[jj].box_xyxy,
                                          affine_of(by_id[jj]), identity_i=ii, identity_j=jj)
                if not pair.valid:
                    continue
                keep.append(pair)
                pref_t.append(int(r0[9]) if r0[8] == "pref" else -1)
                abst_t.append(1.0 if r0[8] == "abstain" else 0.0)
            with torch.no_grad():
                parts["s_i"].append(FEAT.pool_sides(bank, [p.left for p in keep]).cpu())
                parts["s_j"].append(FEAT.pool_sides(bank, [p.right for p in keep]).cpu())
                parts["w_i"].append(FEAT.pool_whole(bank, [FEAT.whole_box_read(p.box_i, affine_of(by_id[p.identity_i]))
                                                           for p in keep]).cpu())
                parts["w_j"].append(FEAT.pool_whole(bank, [FEAT.whole_box_read(p.box_j, affine_of(by_id[p.identity_j]))
                                                           for p in keep]).cpu())
            parts["geom"].append(FEAT.geometry_tensor(keep, "cpu"))
            parts["pref"].append(torch.tensor(pref_t, dtype=torch.int8))
            parts["abst"].append(torch.tensor(abst_t))
            # training material (b): member targets -- the only annotation access, after the decisions
            member_ids = {ident for pair in keep for ident in (pair.identity_i, pair.identity_j)}
            quality = b_quality_material(data, row, by_id, affine_of, bank, member_ids)
            parts["desc"].append(quality["desc"])
            parts["target"].append(quality["target"])
            b_totals.update({key: quality[key] for key in ("members", "whole_box_readable", "zero_targets")})
            per_image.append({"image_id": image, "pool": len(pool), "clusters": len(P.nms_clusters(pool)),
                              "pairs": len(records), "readable": len(valid_pairs),
                              "stage0_readable": stage0_valid, "readability_mismatches": mismatches,
                              "strips_pooled": strips, "pool_seconds": pool_seconds,
                              "pool_us_per_readable_pair": 1e6 * pool_seconds / max(len(valid_pairs), 1),
                              "decision_reasons": dict(reasons),
                              "merge": {k2: merged[k2] for k2 in ("cluster_count", "kept_count", "substitutions",
                                                                    "all_abstain_clusters", "rank_shifted_members",
                                                                    "budget_membership_delta_vs_C")},
                              "controls_instantiated": {"c_pairs": len(c_adapter.table) if c_adapter else None,
                                                        "b_pairs": len(b_adapter.table) if b_adapter else None},
                              "labelled_pairs": len(keep),
                              "b_material": {key: quality[key] for key in ("members", "whole_box_readable", "zero_targets")}})
            del bank, d_i, d_j
            guard(f"pool_decide/{image[:8]}")
    phases["pool_decide"] = {"seconds": time.monotonic() - t0,
                             "pairs": sum(r["pairs"] for r in per_image),
                             "readable": sum(r["readable"] for r in per_image),
                             "strips": sum(r["strips_pooled"] for r in per_image),
                             "labelled_pairs": sum(r["labelled_pairs"] for r in per_image),
                             "readability_mismatches": mismatch_total,
                             "decision_reasons": dict(decision_totals),
                             "b_material": dict(b_totals),
                             "host_hwm_bytes": rss_high_water(), **note_peaks()}
    require(phases["pool_decide"]["pairs"] == plan_record["expected"]["pairs"], "total pair count differs from the plan")
    publish(out / "POOL_DECISION_RESULT.json", {"images": per_image, **phases["pool_decide"],
                                                "gt_read_in_decision_phase": False,
                                                "gt_use": GT_USE})
    guard("pool_decide/published")
    accounting.complete("pool_decide")

    # 4. the fixed three-arm acceptance allocation with the detector frozen
    accounting.phase("learn")
    t0 = time.monotonic()
    pair_material = {"s_i": torch.cat(parts["s_i"]), "s_j": torch.cat(parts["s_j"]),
                     "w_i": torch.cat(parts["w_i"]), "w_j": torch.cat(parts["w_j"]),
                     "geom": torch.cat(parts["geom"]), "pref_target": torch.cat(parts["pref"]),
                     "abstain_target": torch.cat(parts["abst"])}
    quality_material = {"desc": torch.cat(parts["desc"]), "target": torch.cat(parts["target"])}
    detector_grad_checks = []

    def learn_update(arm_key: str, record: Mapping) -> None:
        accounting.update(arm_key)
        guard(f"learn/{arm_key}/update-{int(record['update'])}")

    def learn_arm_done(arm_key: str) -> None:
        measured = sum(1 for module in (engine.model, engine.ema.module)
                       for _, parameter in module.named_parameters() if parameter.grad is not None)
        detector_grad_checks.append({"after_arm": arm_key, "detector_parameters_with_grad": measured})
        L.assert_no_detector_grads(engine.model)
        L.assert_no_detector_grads(engine.ema.module)

    learn = comparator_acceptance_updates(arms, pair_material, quality_material, device="cuda:0",
                                          on_update=learn_update, on_arm_complete=learn_arm_done)
    require(all(entry["detector_parameters_with_grad"] == 0 for entry in detector_grad_checks),
            "detector carried gradients after an arm")
    require(checkpoint.exact_equal(before, checkpoint.capture(engine, binding)),
            "comparator training touched the detector state")
    require(L.tensor_state_digest(engine.ema.module) == detector_digest_before, "detector EMA digest changed")
    learn["detector_frozen_evidence"] = {
        "detector_grad_checks": detector_grad_checks,
        "detector_full_state_exact_before_after": True,
        "detector_ema_digest_unchanged": True,
        "note": ("no backward counter: the rev-4 counter was a no-op and is removed; the freeze is "
                 "evidenced by measured parameter gradients, captured-state equality and the EMA digest")}
    learn["labelled_pairs"] = int(pair_material["s_i"].shape[0])
    learn["b_material"] = dict(b_totals)
    learn["b_target_source"] = GT_USE["annotations_opened_for"]
    phases["learn"] = {"seconds": time.monotonic() - t0, "total_updates": learn["total_updates"],
                       "per_arm_updates": {key: rep["updates"] for key, rep in learn["arms"].items()},
                       "mean_loss": {key: rep["mean_loss"] for key, rep in learn["arms"].items()},
                       "host_hwm_bytes": rss_high_water(), **note_peaks()}
    publish(out / "LEARN_RESULT.json", {"check_revision": CHECK_REVISION, **learn,
                                        "peak_allocated_bytes": phases["learn"]["peak_allocated_bytes"]})
    guard("learn/published")
    accounting.complete("learn")

    # 5. resources and closure
    accounting.phase("resources")
    wall = time.monotonic() - started
    note_peaks()
    resources = {"wall_seconds": wall, "vm_hwm_bytes": rss_high_water(), "host_hwm_reported_not_gated": True,
                 "restore_peak_allocated_bytes": restore_peak_allocated,
                 "restore_peak_reserved_bytes": restore_peak_reserved,
                 "peak_allocated_bytes": worst_peak["allocated"], "peak_reserved_bytes": worst_peak["reserved"],
                 "memory_peaks_recorded_not_gated": True,
                 "memory_gate_removed": ("rev 6: no artificial CUDA-reserved threshold or allocator quota; "
                                         "a real OOM fails as an exception"),
                 "phase_seconds": {name: round(record["seconds"], 3) for name, record in phases.items()},
                 "output_bytes": output_bytes(out), "caps": CAPS,
                 "constraints_enforced_during_run": ["wall_seconds", "output_bytes"],
                 "budget_observations": accounting.snapshot()["budget_observations"],
                 "within_caps": (wall <= CAPS["wall_seconds"]
                                 and output_bytes(out) <= CAPS["output_bytes"])}
    require(resources["within_caps"], f"resource caps exceeded: {resources}")
    publish(out / "RESOURCE.json", resources)
    guard("resources/published")
    accounting.complete("resources")
    return {"status": "PASS", "task": TASK, "check_revision": CHECK_REVISION, "source_commit": source_commit,
            "batches": list(batches), "images": sorted(images),
            "forward_calls": calls,
            "comparator_updates": {"total": learn["total_updates"], "allocation": learn["allocation"],
                                   "per_arm": {key: rep["updates"] for key, rep in learn["arms"].items()},
                                   "batch_pairs": learn["batch_pairs"]},
            "detector_updates": 0,
            "detector_frozen_evidence": learn["detector_frozen_evidence"],
            "gates": {"tap_outputs_unchanged": True, "tap_shape": [4, *EXPECTED_P3],
                      "frozen_raw_views_bitwise_equal": raw_exact,
                      "real_pooling_pairs": phases["pool_decide"]["readable"],
                      "comparator_updates_total": learn["total_updates"],
                      "comparator_gradients_all_parameters_every_arm": True,
                      "detector_gradients_none_after_every_arm": True,
                      "detector_state_exact_before_after": True,
                      "complete_decisions": phases["pool_decide"]["pairs"],
                      "readability_mismatches_vs_stage0": mismatch_total,
                      "resources_within_caps": True},
            "feature_binding": {"level": P.FEATURE_LEVEL, "stride": list(P.FEATURE_STRIDE),
                                "verified_shape": [4, *EXPECTED_P3], "verified": True},
            "gt_use": GT_USE, "acceptance_scope": ACCEPTANCE_SCOPE,
            "phases": phases, "resources": resources, "accounting": accounting.snapshot(),
            "elapsed_seconds": wall}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "dry-run"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", default="UNRECORDED")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    require(TASK in out.parts, "output must sit inside the task directory " + TASK)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    accounting = Accounting(TASK, args.mode)
    publish(out / "START.json", {"task": TASK, "check_revision": CHECK_REVISION, "mode": args.mode,
                                 "pid": os.getpid(), "started_unix": time.time(),
                                 "source_commit": args.source_commit, "batches": list(FROZEN_BATCHES),
                                 "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 "caps": CAPS, "arm_allocation": ARM_ALLOCATION})
    try:
        accounting.phase("inputs")
        enforce_runtime(started=started, label="inputs", out=out, accounting=accounting)
        inputs = resolve_inputs(args.parent, FROZEN_BATCHES)
        publish(out / "ADMISSION.json", {"task": TASK, "check_revision": CHECK_REVISION,
                                         "parent_contract_sha256": PARENT_SHA,
                                         "seed0_task": inputs["seed0"]["task_id"],
                                         "checkpoint": inputs["checkpoint"],
                                         "training_source_root": str(inputs["training_root"]),
                                         "binding": str(inputs["binding_path"]), "gpu_uuid": inputs["gpu_uuid"],
                                         "data_spec": inputs["data_spec"], "gt_use": GT_USE,
                                         "stage0_inputs": inputs["stage0"], "fairness": inputs["fairness"],
                                         "core_version": P.CORE_VERSION, "mode": args.mode,
                                         "arm_allocation": ARM_ALLOCATION, "caps": CAPS,
                                         "acceptance_scope": ACCEPTANCE_SCOPE,
                                         "own_source": {n: sha(HERE / n) for n in
                                                        ("pdra_gpu_check.py", "pdra_core.py", "pdra_features.py",
                                                         "pdra_learn.py", "pdra_tap.py", "pdra_stage1.py")}})
        enforce_runtime(started=started, label="admission", out=out, accounting=accounting)
        accounting.complete("inputs")
        if args.mode == "dry-run":
            import torch
            require(not torch.cuda.is_initialized(), "dry-run initialized CUDA")
            accounting.phase("plan")
            result = {"status": "PLAN_ONLY", **plan(inputs, FROZEN_BATCHES),
                      "cuda_initialized": False, "model_constructed": False,
                      "executed_forward_calls": 0,
                      "executed_updates": {key: 0 for key in ARM_ALLOCATION}}
            publish(out / "PLAN.json", result)
            enforce_runtime(started=started, label="plan", out=out, accounting=accounting)
            accounting.complete("plan")
            result["accounting"] = accounting.snapshot()
        else:
            result = run(inputs, out, FROZEN_BATCHES, args.source_commit,
                         budget_started=started, accounting=accounting)
        result["elapsed_seconds"] = time.monotonic() - started
        ref = publish(out / "RESULT.json", result)
        publish(out / "CLOSED.json", {"result": ref, "returncode": 0, "automatic_retry": False,
                                      "acceptance_scope": ACCEPTANCE_SCOPE})
        print(json.dumps({"status": result["status"], "result": ref}), flush=True)
        return 0
    except Exception as exc:
        stop = {"status": "FAIL", "task": TASK, "check_revision": CHECK_REVISION, "mode": args.mode,
                "source_commit": args.source_commit, "out": str(out),
                "exception": {"type": type(exc).__name__, "message": str(exc),
                              "observation": getattr(exc, "observation", None)},
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "accounting": accounting.snapshot(),
                "automatic_retry": False,
                "note": ("FAIL stops the check; nothing is retried. A corrected revision and its "
                         "execution need a separate authorization.")}
        try:
            publish(out / "STOP.json", stop)
        except Exception:
            traceback.print_exc()
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
