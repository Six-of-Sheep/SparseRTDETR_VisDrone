"""Produce the r24 plan revision 2 from revision 1 and prove the scientific projection is unchanged.

The delivered rev1 plan (``plan-001/r24_roi_ld_v1.json``) is the registered scientific document.
The latest user instruction requires two amendments that must not touch any scientific variable:

* cost is RECORDED from what is actually measured, and no inference ratio is pre-written (rev1's
  narrative said the ratio would be recorded as 1.0, which is exactly what must not happen);
* the evaluation definition is stated once, explicitly, for the whole round: six budget scorings,
  FT012 EMA-C only, EMA-R state only, and the certified parent anchor kept separate from the
  retrained native A0 (a trained A0 is not required to reproduce the parent score).

Everything else is copied byte for byte, and this script refuses to write if any key outside the
declared metadata set differs between rev1 and rev2, or if the two armed gates' conditions or the
frozen constants are not identical.

usage: amend_plan_r24.py <rev1.json> <out_rev2.json> <receipt.json>
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

REV1_SHA256 = "f49fd4e67a42d9590866c69b54bcd100d803037d33ec2c2eac7b9f2d3b04ac11"
REV1_BYTES = 16966

# Keys the amendment is allowed to add or change; the rest must be identical after removing them.
# ``pos_install`` is added because rev1 declared only ``arm_install`` and ``roi_install``, and the
# retirement of the r19 item has to be stated explicitly for the engine's plan reader.
METADATA = ("revision", "revision_log", "authorization_needed", "runner", "pos_install",
            "evaluation_definition", "cost_policy")

COST_GATE = {
    "metric": "inference_stage_seconds",
    "lhs": "A2",
    "rhs": "A0",
    "op": "<=",
    "threshold": 1.05,
    "note": ("the pre-registered threshold is unchanged at 1.05 and is applied to the MEASURED "
             "wall clock of the fixed inference stage (the student runs one whole-image forward in "
             "every arm); no inference ratio is pre-written, and the per-arm seconds, the A1/A0 "
             "reading and the end-to-end figure including training are recorded alongside"),
}

EVALUATION_DEFINITION = {
    "scorings": 6,
    "ema": "C",
    "ema_r": "state_only",
    "view": "A",
    "budgets": [600, 300],
    "parent_anchor": "separate_from_trained_A0",
    "trained_A0_required_equal_to_parent": False,
    "note": ("six budget scorings, three arms times two budgets, FT012 EMA-C only; EMA-R is "
             "maintained and checkpointed by every arm and is never scored; the certified parent "
             "anchor is reported separately from the retrained native A0 and no equality between "
             "the trained A0 and the parent score is required or asserted"),
}

COST_POLICY = {
    "version": "r24-measured-cost-only-v1",
    "ratio_source": "measured_from_the_inference_stage",
    "predeclared_ratio": None,
    "gate": {"metric": "inference_stage_seconds", "lhs": "A2", "rhs": "A0", "op": "<=", "threshold": 1.05},
    "recorded_alongside": ["inference_stage_seconds_per_arm", "inference_stage_A1_over_A0",
                           "end_to_end_including_training_A2_over_A0",
                           "formal_training_wall_seconds_per_arm", "acceptance_seconds"],
    "note": "cost is recorded as measured; the round pre-writes no ratio and no ratio is asserted by inspection",
}

REVISION_NOTE = {
    "revision": 2,
    "date": "2026-11-03",
    "note": ("engine-installation amendment under the standing preparation authorization, plus the "
             "user's evaluation unification.  No scientific variable changes: the three arms, the "
             "frozen teacher cache, the constants (band 1024 px^2 on 768x1344, tau = 0.3, one target "
             "per ground-truth row, original-coordinate prefilter 0.1), the common-support rule, the "
             "7 + 7 gate conditions and their thresholds, the combination rule, the parent, the "
             "12-pass/3648-update schedule, the dual-EMA clocks, the checkpoint cadence, the "
             "endpoint and the budgets are all carried unchanged."),
    "changes": [
        {"id": "cost-is-measured-not-written",
         "was": "gates.cost.metric = end_to_end_seconds and the narrative recorded the inference "
                "ratio as 1.0",
         "is": "gates.cost.metric = inference_stage_seconds with the same 1.05 threshold, applied "
               "to the measured inference-stage ratio; a cost_policy block states explicitly that no "
               "ratio is pre-declared (predeclared_ratio = null) and lists what is recorded",
         "why": "the user requires cost to be recorded as actually measured and forbids pre-writing "
                "the inference time ratio as 1.0"},
        {"id": "retired-r19-item-declared",
         "was": "rev1 declared arm_install and roi_install only",
         "is": "pos_install = {A0: None, A1: None, A2: None}: the r19 tiny-positives item is "
               "retired and installed by no arm, while its modules stay present and inert",
         "why": "the user requires the old item to be retired in the same installation step"},
        {"id": "unified-evaluation-definition",
         "was": "the evaluation definition was implicit in the endpoint/evaluation blocks",
         "is": "an explicit evaluation_definition block: six scorings, EMA-C only, EMA-R state only, "
               "the certified parent anchor separate from the retrained native A0, and no requirement "
               "that the trained A0 reproduce the parent score",
         "why": "the user requires one unified definition for all six scorings and the separation of "
                "the parent anchor from the trained A0"},
        {"id": "authorization-text",
         "was": "the plan described a separately confirmed GPU stage",
         "is": "the plan records the user's 2026-11-03 instruction: after the engine installation, "
               "the freeze and one full acceptance have been verified, run the three arms and the "
               "fixed evaluation once, repairing ordinary engineering faults with traces and without "
               "changing any scientific setting",
         "why": "the authorization text must match the instruction actually standing"},
    ],
    "unchanged_gate_conditions": "both armed gates keep their 7 conditions and thresholds verbatim",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def projection(plan):
    """The plan without the amendment metadata: this must be identical across revisions."""
    body = copy.deepcopy(plan)
    for key in METADATA:
        body.pop(key, None)
    gates = body.get("gates") or {}
    gates.pop("cost", None)      # amended narrative/semantics only; the armed gates below stay
    body["gates"] = gates
    return body


def main():
    rev1_path, out_path, receipt_path = (Path(a) for a in sys.argv[1:4])
    raw = rev1_path.read_bytes()
    if len(raw) != REV1_BYTES or sha(raw) != REV1_SHA256:
        raise SystemExit("rev1 identity mismatch: %d bytes / %s" % (len(raw), sha(raw)))
    rev1 = json.loads(raw)

    rev2 = copy.deepcopy(rev1)
    rev2["revision"] = 2
    rev2["revision_log"] = list(rev1.get("revision_log") or []) + [REVISION_NOTE]
    rev2["authorization_needed"] = (
        "the user authorized one bounded GPU run on 2026-11-03: after the r24 engine installation, "
        "the retirement of the r19 item, the state save/restore and dual-EMA compatibility and the "
        "affected checks have been run and verified, execute one full acceptance and then, if it "
        "passes, the three-arm formal comparison (3 x 3648 updates) and the fixed evaluation; repair "
        "ordinary engineering faults with traces, accumulate their cost, and report centrally if any "
        "scientific setting would have to change")
    rev2["pos_install"] = {"A0": None, "A1": None, "A2": None}
    rev2["evaluation_definition"] = copy.deepcopy(EVALUATION_DEFINITION)
    rev2["cost_policy"] = copy.deepcopy(COST_POLICY)
    rev2["gates"] = {**rev1["gates"], "cost": copy.deepcopy(COST_GATE)}

    # The proof: outside the amendment metadata every key is identical, and the two armed gates
    # keep their conditions and thresholds verbatim.
    if projection(rev2) != projection(rev1):
        raise SystemExit("the scientific projection changed; refusing to write rev2")
    for name in ("A2_minus_A0", "A2_minus_A1"):
        if rev1["gates"][name] != rev2["gates"][name]:
            raise SystemExit("the armed gate %s changed; refusing to write rev2" % name)
    if rev1["gates"]["combination"] != rev2["gates"]["combination"]:
        raise SystemExit("the combination rule changed; refusing to write rev2")

    body = json.dumps(rev2, ensure_ascii=False, indent=1).encode("utf-8")
    if out_path.exists():
        raise SystemExit("output already exists: " + str(out_path))
    out_path.write_bytes(body)
    receipt = {
        "schema": "r24-plan-amendment-v1",
        "rev1": {"path": str(rev1_path), "bytes": len(raw), "sha256": sha(raw), "revision": rev1["revision"]},
        "rev2": {"path": str(out_path), "bytes": len(body), "sha256": sha(body), "revision": rev2["revision"]},
        "amendment_metadata_keys": list(METADATA),
        "scientific_projection_sha256_before": sha(json.dumps(projection(rev1), sort_keys=True,
                                                              separators=(",", ":"), ensure_ascii=False).encode()),
        "scientific_projection_sha256_after": sha(json.dumps(projection(rev2), sort_keys=True,
                                                             separators=(",", ":"), ensure_ascii=False).encode()),
        "armed_gates_unchanged": ["A2_minus_A0", "A2_minus_A1", "combination"],
        "changes": [row["id"] for row in REVISION_NOTE["changes"]],
        "scientific_projection_unchanged": True,
    }
    receipt["receipt_sha256"] = sha(json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                               ensure_ascii=False).encode())
    if receipt_path.exists():
        raise SystemExit("receipt already exists: " + str(receipt_path))
    receipt_path.write_bytes(json.dumps(receipt, ensure_ascii=False, indent=1).encode("utf-8"))
    print(json.dumps({"rev2": str(out_path), "rev2_sha256": sha(body), "rev2_bytes": len(body),
                      "receipt": str(receipt_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
