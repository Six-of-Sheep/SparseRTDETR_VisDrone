"""Offline H2 scalar receipts -> paper tables/figures; never run a model/evaluator.

Input schema is pinned to PR21 e50bedb. Missing evidence is UNKNOWN; conflicting
evidence is rejected. Original scientific decisions are copied, never replaced
by thresholds applied to rounded table values. See H2_RESULT_TOOL.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys

COMMIT = "e50bedbe8e41bdfd3c47d09f7076d9644e438e26"
CONTRACT = "8450bbaff7d15da0db0343da49aa0d59d9455ff186458a664c2707db0397e8df"
TASK = "r18-h2-formal-20261024-001"
SCOPE = "H2_FORMAL_UD_HISTORICAL_N_V1"
N_STATUS = "AUTHENTICATED_OUTPUT_NOT_ORIGINAL_STAGE_CLOSURE"
N_SHA = {"600": "86be4fcc1478442bd4bb6f3b6e5b0c421e29f854e6977c54ba906814e79dfc7e",
         "300": "5a829d1ed3c68022053605ab5f143d3dd988eb277960918b898ffcaacd7e3129"}
STAGES = ("train-U", "train-D", "infer-U", "infer-D", "score", "cost", "compare")
ARMS = ("N", "U", "D")
BUDGETS = ("600", "300")
PAIRS = ("D-N", "D-U", "U-N")
COCO = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100",
        "AR_small", "AR_medium", "AR_large", "AP75_small")
PRIMARY = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
LIMITS = {600: {"AP75_small": .3, "APS": .2, "AP": 0, "APM": -.2, "APL": -.2, "AR_large": -.2, "primary_AP": -.2},
          300: {"AP75_small": 0, "AP": -.2, "APS": -.2, "primary_AP": -.2}}
ORDERS = [["N", "U", "D"], ["N", "D", "U"], ["U", "N", "D"],
          ["U", "D", "N"], ["D", "N", "U"], ["D", "U", "N"]]
UNKNOWN = "UNKNOWN"
MAX_FILE = 2 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs_unique(items):
    value = {}
    for key, item in items:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def decode(raw):
    def bad_constant(value):
        raise ValueError("non-finite JSON constant: " + value)
    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), "non-finite JSON float: " + text)
        return value
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_unique, parse_constant=bad_constant, parse_float=finite_float)


def number(value, label, minimum=None, maximum=None):
    require(type(value) in (float, int) and math.isfinite(value), label + ": non-finite/non-numeric")
    require(minimum is None or value >= minimum, label + ": below range")
    require(maximum is None or value <= maximum, label + ": above range")
    return value


def same_number(a, b, label):
    number(a, label); number(b, label)
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-10), label + ": stored arithmetic mismatch")


def flag(value):
    require(value is None or type(value) is bool, "scientific flag must be boolean/null")
    return UNKNOWN if value is None else ("PASS" if value else "FAIL")


def reference(ref):
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str), "invalid reference path")
    require(type(ref.get("bytes")) is int and 0 <= ref["bytes"] < MAX_FILE, "invalid reference size")
    require(isinstance(ref.get("sha256"), str) and re.fullmatch("[a-f0-9]{64}", ref["sha256"]), "invalid reference SHA")
    return ref


class Receipts:
    """Resolve only explicit export names. Never open paths embedded in receipts."""

    def __init__(self, index_path, expected_sha):
        self.index_path = Path(index_path).resolve()
        self.folder = self.index_path.parent
        self.rows = {}; self.paths = {}; self.values = {}; self.missing = []
        require(not Path(index_path).is_symlink(), "symlink index refused")
        require(self.index_path.stat().st_size < MAX_FILE, "index too large")
        raw = self.index_path.read_bytes()
        require(sha(raw) == expected_sha, "index SHA mismatch")
        self.index_sha = sha(raw); self.index_bytes = len(raw)
        self.index = decode(raw)
        require(self.index.get("status") == "PASS", "STOP/non-PASS export refused")
        require(self.index.get("original_N_STOP_preserved") is True, "historical N STOP not preserved")
        total = 0
        for row in self.index["files"]:
            name = row["name"]
            require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+\.json", name), "unsafe export name")
            require(name not in self.rows, "duplicate export name")
            require("STOP" not in name, "STOP receipt present: " + name)
            source, returned = reference(row["source"]), reference(row["returned"])
            require((source["sha256"], source["bytes"]) == (returned["sha256"], returned["bytes"]), "source/returned mismatch")
            for ref in (source, returned):
                require(ref["path"] not in self.paths or self.paths[ref["path"]] == name, "ambiguous source path")
                self.paths[ref["path"]] = name
            total += source["bytes"]
            self.rows[name] = row
            path = self.folder / name
            require(not path.is_symlink() and path.resolve().parent == self.folder, "export path escape")
            if not path.exists():
                self.miss(name + ": absent local file")
                continue
            require(path.is_file() and path.stat().st_size == source["bytes"], name + ": bytes mismatch")
            raw = path.read_bytes()
            require(sha(raw) == source["sha256"], name + ": SHA mismatch")
            self.values[name] = decode(raw)
        require(total <= 20 * 1024**2, "export exceeds production 20MiB bound")
        for path in self.folder.glob("*STOP*.json"):
            raise ValueError("STOP file present: " + path.name)

    def miss(self, reason):
        if reason not in self.missing:
            self.missing.append(reason)

    def get(self, name):
        if name not in self.values:
            self.miss(name + ": missing evidence")
        return self.values.get(name)

    def linked(self, ref, name):
        if ref is None:
            self.miss(name + ": missing reference")
            return None
        reference(ref)
        row = self.rows.get(name)
        if row is None:
            self.miss(name + ": missing index entry")
            return None
        require(self.paths.get(ref["path"]) == name, name + ": reference path/role mismatch")
        require(ref["sha256"] == row["source"]["sha256"] and ref["bytes"] == row["source"]["bytes"], name + ": reference SHA/size mismatch")
        return self.get(name)


def metric(score, evaluator, key, missing, label):
    if score is None:
        return None
    section = score.get("COCO_bridge", {}).get("metrics", {}) if evaluator == "COCO_bridge" else score.get("official", {})
    value = section.get(key)
    if value is None:
        missing(label + ": missing metric")
        return None
    number(value, label)
    if evaluator == "COCO_bridge" and value == -1:
        missing(label + ": undefined COCO area/class (-1)")
        return None
    number(value, label, 0, 1 if evaluator == "COCO_bridge" else 100)
    return value * 100 if evaluator == "COCO_bridge" else value


def load(index_path, index_sha, closed_path=None, closed_sha=None, synthetic=False):
    s = Receipts(index_path, index_sha)
    c = s.get("contract.json")
    require(c is not None, "contract is required; no identity-free import")
    require(c.get("task_id") == TASK and c.get("scope") == SCOPE and c.get("seed") == 0, "task/scope/seed mismatch")
    require(c.get("commit") == COMMIT, "running code commit mismatch")
    require(PurePosixPath(c["job_root"]).name == "attempt-formal-001", "attempt mismatch")
    require(c.get("profile") == ("CPU_FIXTURE_ONLY" if synthetic else "LAB_GPU_ONCE"), "synthetic/production mode mismatch")
    if synthetic:
        require(c.get("fixture", {}).get("paper_tool_synthetic") is True, "unmarked synthetic fixture")
    else:
        require("fixture" not in c and c.get("contract_sha256") == CONTRACT, "frozen production contract mismatch")
    cid = c.get("contract_sha256")
    require(cid == digest({k: v for k, v in c.items() if k != "contract_sha256"}), "canonical contract SHA mismatch")
    source = s.linked(c.get("source"), "source-manifest.json")
    if source is not None:
        require(source.get("commit") == COMMIT, "source manifest commit mismatch")
    require(c.get("historical_N", {}).get("status") == N_STATUS, "N historical status mismatch")
    root, comparison, cost, score = (s.get(n) for n in ("RESULT.json", "compare-RESULT.json", "cost-RESULT.json", "score-RESULT.json"))
    # All new stage receipts carry a flat contract identity (not old N identity).
    for name, value in s.values.items():
        if name in ("RESULT.json", "EXECUTION_EXIT.json", "CUMULATIVE_COST.json") or any(name == st + "-" + tail for st in STAGES for tail in ("RESULT.json", "CLOSED.json", "supervisor-exit.json", "worker-exit.json", "binding.json")):
            require(value.get("contract_sha256") == cid, name + ": contract mismatch")
    for st in STAGES:
        cl = s.get(st + "-CLOSED.json")
        result = s.get(st + "-RESULT.json")
        if result is not None:
            require(result.get("status") == "PASS" and result.get("stage") == st, st + ": result status/stage mismatch")
            if not synthetic:
                expected_calls = 72960 if st.startswith("train-") else (137 if st.startswith("infer-") else (384 if st == "cost" else 0))
                require(result.get("forward_calls") == expected_calls, st + ": forward count mismatch")
        if cl is None:
            continue
        require(cl.get("status") == "PASS" and cl.get("arm") == st, st + ": non-PASS stage closure")
        s.linked(cl.get("result"), st + "-RESULT.json")
        for key, name in (("supervisor_exit", "supervisor-exit.json"), ("worker_exit", "worker-exit.json")):
            ex = s.linked(cl.get(key), st + "-" + name)
            if ex is not None:
                require(ex.get("status") == "PASS" and ex.get("error") is None, st + ": non-PASS exit")
                if key == "supervisor_exit":
                    require(ex.get("returncode") == 0 and ex.get("remaining_live") == [] and ex.get("cleanup_errors") == [], st + ": unclean exit")
    ex = s.get("EXECUTION_EXIT.json")
    if ex is not None:
        require(ex.get("status") == "PASS" and ex.get("returncode") == 0 and ex.get("error") is None, "root execution STOP")
    if root is not None:
        require(root.get("status") == "PASS" and root.get("scope") == SCOPE and root.get("stages") == list(STAGES), "root status/scope/stages mismatch")
        require(root.get("fixture_only") is synthetic and root.get("historical_N_status") == N_STATUS, "root mode/N provenance mismatch")
        s.linked(root.get("comparison"), "compare-RESULT.json")
    require((closed_path is None) == (closed_sha is None), "closed path and SHA must be provided together")
    if closed_path is None:
        s.miss("root CLOSED: not supplied separately")
    else:
        p = Path(closed_path)
        require(not p.is_symlink() and p.stat().st_size < MAX_FILE, "invalid root CLOSED file")
        raw = p.read_bytes(); require(sha(raw) == closed_sha, "root CLOSED SHA mismatch")
        cl = decode(raw)
        require(cl.get("status") == "PASS" and cl.get("contract_sha256") == cid, "root CLOSED identity/status mismatch")
        s.linked(cl.get("result"), "RESULT.json")
        ref = reference(cl["export_index"])
        require(ref["sha256"] == s.index_sha and ref["bytes"] == s.index_bytes, "root CLOSED export index mismatch")
        # source/returned paths are remote provenance, never local open targets.
        if root is not None:
            expected = c["job_root"].rstrip("/") + "/slim-export/SHA_INDEX.json"
            require(ref["path"] == expected, "root CLOSED index path mismatch")
    for arm in ("U", "D"):
        tr = s.get(f"train-{arm}-RESULT.json")
        if tr is None:
            continue
        ep = s.linked(tr.get("endpoint"), f"train-{arm}-endpoint.json")
        if not synthetic:
            require(tr.get("actual_updates") == 36480 and tr.get("forward_calls") == tr.get("BF16_forward_calls") == 72960, "training endpoint count mismatch")
        if ep is not None:
            require(ep.get("EMA") is True and ep.get("used_acceptance_state") is False and ep.get("used_old_endpoint") is False, "endpoint provenance mismatch")
            if not synthetic:
                require(ep.get("epoch") == 120 and ep.get("updates") == 36480, "not fixed e120 endpoint")
            b = s.linked(ep.get("binding"), f"train-{arm}-binding.json")
            if b is not None:
                require(b.get("arm") == arm and b.get("source_sha256") == c["source"]["sha256"] and b.get("data_sha256") == digest(c["data"]) and b.get("pretrained_sha256") == c["pretrained"]["sha256"], "endpoint binding mismatch")
    tu, td = s.get("train-U-RESULT.json"), s.get("train-D-RESULT.json")
    if tu is not None and td is not None:
        require(isinstance(tu.get("student_inputs_sha256"), str) and tu["student_inputs_sha256"] == td.get("student_inputs_sha256"), "U/D training input mismatch")
    ledger = s.get("CUMULATIVE_COST.json")
    if ledger is not None:
        require(ledger.get("active") is None and ledger.get("fixture_only") is synthetic, "active/mode ledger mismatch")
        if not synthetic:
            require(ledger.get("forward_calls") == ledger.get("forwards_completed") == 146596 and ledger.get("actual_updates") == ledger.get("updates_completed") == 72960, "cumulative counts mismatch")
        if root is not None:
            require(ledger.get("forward_calls") == root.get("forward_calls"), "root/ledger forward mismatch")
            same_number(ledger["GPU_seconds"], root["GPU_seconds"], "root/ledger GPU seconds")
    identity_complete = not s.missing
    metrics = {}
    if score is not None:
        require(score.get("N_recomputed") is False and score.get("new_groups") == 4 and score.get("new_evaluator_calls") == 8, "score coverage or N reuse mismatch")
        require(set(score["results"]) == set(ARMS), "score arms mismatch")
        for arm in ARMS:
            require(set(score["results"][arm]) == set(BUDGETS), "score budgets mismatch")
            for budget in BUDGETS:
                name = f"{arm}-{budget}.json"; ref = score["results"][arm][budget]
                value = s.linked(ref, name); metrics[arm, budget] = value
                if arm == "N":
                    original = c["historical_N"]["scores"][budget]
                    require((ref["sha256"], ref["bytes"]) == (original["sha256"], original["bytes"]), "historical N score changed")
                    if not synthetic:
                        require(ref["sha256"] == N_SHA[budget], "N fixed score SHA mismatch")
                if value is not None:
                    require(value.get("official_APS") is None, "unexpected official APS")
                    co = value.get("COCO_bridge", {})
                    require(co.get("units") == "0..1 AP, -1 means no GT in an area/class aggregation" and co.get("protocol") == "r0_endpoint_development_raw_ignore_v1" and co.get("maxDets") == [1, 10, 100], "COCO evaluator units/protocol mismatch")
                    count = value.get("official", {}).get("evaluated_image_count")
                    require(count == (8 if synthetic else 548), "evaluated image count mismatch")
    result = build(s, c, root, comparison, cost, metrics, ledger, identity_complete, synthetic)
    result["provenance"]["root_closed_sha256"] = closed_sha
    return result


def build(s, contract, root, comparison, cost, metrics, ledger, identity_complete, synthetic):
    absolute, deltas, gates, cost_rows, trials, centers = [], [], [], [], [], []
    lookup = {}
    for budget in BUDGETS:
        for evaluator, names in (("COCO_bridge", COCO), ("primary", PRIMARY)):
            for arm in ARMS:
                value = metrics.get((arm, budget))
                for key in names:
                    val = metric(value, evaluator, key, s.miss, f"{arm}/{budget}/{evaluator}/{key}")
                    lookup[arm, budget, evaluator, key] = val
                    absolute.append(dict(arm=arm, budget=int(budget), evaluator=evaluator, metric=key, value_percent=val))
    if comparison is not None:
        require(comparison.get("fixture_only") is synthetic, "comparison fixture mismatch")
        if cost is not None:
            require(comparison.get("cost") == cost, "comparison embedded cost differs")
        for (arm, budget), value in metrics.items():
            if value is not None:
                require(comparison.get("absolute", {}).get(arm, {}).get(budget) == value, "comparison embedded score differs")
    for budget in BUDGETS:
        for evaluator, names in (("COCO_bridge", COCO), ("primary", PRIMARY)):
            for pair in PAIRS:
                a, b = pair.split("-")
                for key in names:
                    x, y = lookup[a, budget, evaluator, key], lookup[b, budget, evaluator, key]
                    stored = None
                    if comparison is not None:
                        stored = comparison.get("deltas_pp", {}).get(pair, {}).get(budget, {}).get("COCO_pp" if evaluator == "COCO_bridge" else "primary_pp", {}).get(key)
                    if stored is not None:
                        number(stored, "stored delta")
                        if x is not None and y is not None:
                            same_number(stored, x - y, "stored delta " + pair + "/" + key)
                    if stored is None:
                        s.miss(f"stored delta {pair}/{budget}/{evaluator}/{key}: absent")
                    # Never turn undefined -1 or absent metrics into a numeric gain.
                    val = stored if x is not None and y is not None else None
                    deltas.append(dict(pair=pair, budget=int(budget), evaluator=evaluator, metric=key, delta_pp=val,
                                       recorded_delta_pp=stored, status="RECORDED" if val is not None else UNKNOWN))
    expected = {(pair, b, k): minimum for pair in ("D-N", "D-U") for b, values in LIMITS.items() for k, minimum in values.items()}
    recorded = {}
    if comparison is not None:
        for g in comparison.get("quality_gates", []):
            key = (g["pair"], g["budget"], g["metric"])
            require(key in expected and key not in recorded, "unexpected/duplicate quality gate")
            require(g["minimum"] == expected[key] and type(g["defined"]) is bool and type(g["passed"]) is bool, "changed gate threshold/type")
            number(g["delta_pp"], "gate delta")
            stored = comparison["deltas_pp"][key[0]][str(key[1])]["primary_pp" if key[2] == "primary_AP" else "COCO_pp"]["AP" if key[2] == "primary_AP" else key[2]]
            # The producer reuses this exact float; tolerance could cross a gate.
            require(g["delta_pp"] == stored, "gate vs stored delta: exact duplicate mismatch")
            require(g["defined"] or g["passed"] is False, "undefined gate passed")
            a, b = key[0].split("-")
            if key[2] == "primary_AP":
                require(g["defined"] is True, "primary_AP definition mismatch")
            else:
                values = [(metrics.get((arm, str(key[1]))) or {}).get("COCO_bridge", {}).get("metrics", {}).get(key[2]) for arm in (a, b)]
                if all(v is not None for v in values):
                    require(g["defined"] == all(v >= 0 for v in values), "gate definition disagrees with scores")
            # Verification of the unrounded receipt; not a new scientific decision.
            require(g["passed"] == (g["defined"] and g["delta_pp"] >= g["minimum"]), "contradictory original gate")
            recorded[key] = g
    for key, minimum in expected.items():
        g = recorded.get(key)
        if g is None:
            s.miss("quality gate missing: " + str(key))
        pair, budget, key_name = key
        a, b = pair.split("-"); ev = "primary" if key_name == "primary_AP" else "COCO_bridge"; mk = "AP" if key_name == "primary_AP" else key_name
        known = all(lookup[x, str(budget), ev, mk] is not None for x in (a, b))
        gates.append(dict(pair=pair, budget=budget, metric=key_name, minimum_pp=minimum,
                          delta_pp=g["delta_pp"] if g and known else None, defined=g["defined"] if g else None,
                          recorded_status=flag(g["passed"]) if g else UNKNOWN,
                          status=flag(g["passed"]) if g and known and g["defined"] else UNKNOWN))
    if comparison is not None and len(recorded) == 22 and comparison.get("quality_passed") is not None:
        require(comparison["quality_passed"] == all(g["passed"] for g in recorded.values()), "overall quality differs from original gates")
    for arm in ARMS:
        summary = (cost or {}).get("summary", {}).get(arm, {})
        rr = (cost or {}).get("measurements", {}).get(arm)
        if rr is None:
            s.miss("cost measurements missing: " + arm)
        else:
            require(len(rr) == 6, "must have six measured passes per arm")
            for i, row in enumerate(rr):
                require(row.get("images") == (4 if synthetic else 64) and row.get("batch") == 4 and row.get("dtype") == "FP32", "cost input boundary mismatch")
                for key in ("seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
                    number(row[key], "cost " + key, 0)
                if not synthetic:
                    require(row["seconds"] > 0 and row["peak_allocated_bytes"] > 0, "production cost must be positive")
                trials.append(dict(arm=arm, repeat=i + 1, seconds=row["seconds"], peak_allocated_bytes=row["peak_allocated_bytes"], peak_reserved_bytes=row["peak_reserved_bytes"]))
            for key, expected in (("mean_seconds", statistics.mean(r["seconds"] for r in rr)),
                                  ("sample_SD_seconds", statistics.stdev(r["seconds"] for r in rr)),
                                  ("peak_allocated_bytes", max(r["peak_allocated_bytes"] for r in rr)),
                                  ("peak_reserved_bytes", max(r["peak_reserved_bytes"] for r in rr))):
                if summary.get(key) is None:
                    s.miss("cost summary missing: " + arm + "/" + key)
                else:
                    same_number(summary[key], expected, "cost summary " + key)
        cost_rows.append(dict(arm=arm, mean_seconds=summary.get("mean_seconds"), sample_SD_seconds=summary.get("sample_SD_seconds"),
                              peak_allocated_GiB=summary["peak_allocated_bytes"] / 1024**3 if summary.get("peak_allocated_bytes") is not None else None,
                              peak_reserved_GiB=summary["peak_reserved_bytes"] / 1024**3 if summary.get("peak_reserved_bytes") is not None else None))
    ratios = []
    if cost is not None:
        require(cost.get("orders") == ORDERS and cost.get("scope") == ("CPU_FIXTURE_NOT_COST_EVIDENCE" if synthetic else "FIXED64_GPU_E2E"), "cost order/scope mismatch")
    for pair in ("D/N", "D/U", "U/N"):
        values = (comparison or {}).get("ratios", {}).get(pair, {})
        a, b = pair.split("/")
        for key, source in (("E2E", "mean_seconds"), ("peak_allocated", "peak_allocated_bytes")):
            val = values.get(key); av = (cost or {}).get("summary", {}).get(a, {}).get(source); bv = (cost or {}).get("summary", {}).get(b, {}).get(source)
            if val is None:
                s.miss("cost ratio missing: " + pair + "/" + key)
            else:
                number(val, "cost ratio", 0)
                if av is not None and bv is not None and bv > 0:
                    same_number(val, av / bv, "cost ratio " + pair)
            ratios.append(dict(pair=pair, quantity=key, ratio=val))
    original_cost = (comparison or {}).get("cost_passed")
    dn = (comparison or {}).get("ratios", {}).get("D/N", {})
    if original_cost is not None and all(dn.get(k) is not None for k in ("E2E", "peak_allocated")):
        require(original_cost == (dn["E2E"] <= 1.05 and dn["peak_allocated"] <= 1.05), "original cost decision contradicts stored ratios")
    for arm in ARMS:
        value = None
        if arm != "N":
            inf = s.get(f"infer-{arm}-RESULT.json")
            if inf is not None:
                require(inf.get("images") == (8 if synthetic else 548) and inf.get("EMA") is True, "inference endpoint mismatch")
                value = s.linked(inf.get("centers"), f"infer-{arm}-CENTER_SUMMARY.json")
        keys = ("matched_small", "coarse_mean", "coarse_median", "final_mean", "final_median", "improved", "unchanged", "worsened")
        row = dict(arm=arm, **{k: (value or {}).get(k) for k in keys})
        if value is not None:
            require(value.get("images") == (8 if synthetic else 548) and value.get("N") == "UNKNOWN" and value.get("scope") == "endpoint_local_support_not_training_cause", "center scope mismatch")
            for key in ("matched_small", "improved", "unchanged", "worsened"):
                if row[key] is None:
                    s.miss("center missing: " + arm + "/" + key)
                else:
                    require(type(row[key]) is int and row[key] >= 0, "center count invalid")
            if all(row[k] is not None for k in ("matched_small", "improved", "unchanged", "worsened")):
                require(sum(row[k] for k in ("improved", "unchanged", "worsened")) == row["matched_small"], "center count denominator mismatch")
            for key in ("coarse_mean", "coarse_median", "final_mean", "final_median"):
                if row[key] is not None:
                    number(row[key], "center error", 0)
                    require(row["matched_small"] != 0, "nonempty statistic for empty center sample")
                elif row["matched_small"] != 0:
                    s.miss("center statistic missing: " + arm + "/" + key)
        row["mean_change"] = row["final_mean"] - row["coarse_mean"] if row["final_mean"] is not None and row["coarse_mean"] is not None else None
        row["status"] = "NOT_RECORDED_HISTORICAL_N" if arm == "N" else ("EMPTY_SAMPLE" if row["matched_small"] == 0 else ("RECORDED" if value else UNKNOWN))
        centers.append(row)
    reported = {}
    for key in ("quality_passed", "cost_passed"):
        val = (comparison or {}).get(key); reported[key] = flag(val)
        if root is not None and comparison is not None:
            require(root.get(key) == val, "root/comparison scientific decision mismatch")
        # Constructed fixtures may exercise PASS/FAIL display. The report-level
        # scientific statuses still always say NOT_SCIENTIFIC in this mode.
    complete = not s.missing
    return dict(schema="h2-paper-results-v1", synthetic=synthetic, experiment=TASK, attempt="attempt-formal-001", seed=0,
                runtime_commit=COMMIT, contract_sha256=contract["contract_sha256"],
                engineering="SYNTHETIC_ONLY" if synthetic else ("RECEIPTS_COMPLETE" if identity_complete else UNKNOWN),
                evidence_complete=complete, reported_quality=reported["quality_passed"], reported_cost=reported["cost_passed"],
                quality="NOT_SCIENTIFIC" if synthetic else (reported["quality_passed"] if complete else UNKNOWN),
                cost="NOT_SCIENTIFIC" if synthetic else (reported["cost_passed"] if complete else UNKNOWN),
                absolute=absolute, deltas=deltas, gates=gates, cost_summary=cost_rows, cost_trials=trials, ratios=ratios, centers=centers,
                cumulative_cost=ledger, missing=s.missing,
                provenance=dict(index_sha256=s.index_sha, files=[dict(name=n, sha256=r["source"]["sha256"], bytes=r["source"]["bytes"], present=n in s.values) for n, r in s.rows.items()]),
                limitations=["N is historical; its original STOP and missing training CLOSED remain.",
                             "D-U is the controlled comparison; D-N is historical.",
                             "Center summaries use each arm's own final Hungarian pairs, not common matched GT.",
                             "Normalized center error is pixel L2 divided by sqrt(raw_area); negative final-coarse is improvement.",
                             "Six timing repetitions give sample SD, not cross-seed uncertainty.",
                             "Receipt bytes verified only; no checkpoint, event stream, prediction, GT or server process revalidation.",
                             "Scientific gates are copied; no AP evaluation, training, new inference or novelty verdict."])


def formatted(value):
    if value is None:
        return UNKNOWN
    if type(value) is float:
        return f"{value:.6f}"
    return str(value)


def table(headers, rows):
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(lines + ["|" + "|".join(formatted(v) for v in row) + "|" for row in rows])


def write_csv(path, rows, data_kind):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["data_kind", *rows[0]])
        writer.writeheader()
        writer.writerows(dict(data_kind=data_kind, **{k: UNKNOWN if v is None else v for k, v in row.items()}) for row in rows)


def render(data, output, make_plots=True):
    out = Path(output)
    require(not out.exists(), "output must be a new directory; existing artifacts are never overwritten")
    require(out.parent.is_dir(), "output parent must exist")
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
    out.mkdir()
    marker = "SYNTHETIC TEST DATA — NOT EXPERIMENT RESULTS" if data["synthetic"] else ("H2 seed0 development548" if data["evidence_complete"] else "INCOMPLETE EVIDENCE — UNKNOWN")
    headings = ["# " + marker, "", "工程回执：" + data["engineering"] + "；可报告质量：" + data["quality"] + "；可报告成本：" + data["cost"],
                "", "原比较回执质量/成本：" + data["reported_quality"] + "/" + data["reported_cost"] + "（缺证时不升级为完整裁决）。",
                "", "固定e120 EMA，seed0。COCO原分数0..1转换为百分数；原主评价器已经是百分数；差值为pp。",
                "显示四舍五入仅用于阅读；门槛状态照录原回执。UNKNOWN不以零填充；COCO -1为未定义。", ""]
    for budget in (600, 300):
        for ev, names in (("COCO_bridge", COCO), ("primary", PRIMARY)):
            headings += [f"## top{budget} / {ev} 绝对值（%）", ""]
            headings += [table(["arm", *names], [[arm] + [next(r["value_percent"] for r in data["absolute"] if r["arm"] == arm and r["budget"] == budget and r["evaluator"] == ev and r["metric"] == k) for k in names] for arm in ARMS]), ""]
            headings += [f"### top{budget} / {ev} 成对差值（pp）", "", table(["pair", *names], [[pair] + [next(r["delta_pp"] for r in data["deltas"] if r["pair"] == pair and r["budget"] == budget and r["evaluator"] == ev and r["metric"] == k) for k in names] for pair in PAIRS]), ""]
    for key, title in (("gates", "全部22项原质量条件"), ("cost_summary", "固定64图成本（秒/GiB，合成模式为构造数据）"),
                       ("cost_trials", "18次原测速记录"), ("ratios", "原比较回执中的成本比"), ("centers", "各臂自己的匹配小GT中心摘要")):
        rows = data[key]
        headings += ["## " + title, "", table(list(rows[0]), [list(r.values()) for r in rows]) if rows else UNKNOWN, ""]
        if rows:
            write_csv(out / (key + ".csv"), rows, marker)
    for key in ("absolute", "deltas"):
        write_csv(out / (key + ".csv"), data[key], marker)
    headings += ["## 缺证与解释边界", ""] + ["- " + x for x in data["missing"] + data["limitations"]]
    (out / "TABLES.md").write_text("\n".join(headings) + "\n", encoding="utf-8")
    (out / "NORMALIZED.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tex = ["% " + marker, "% COCO bridge, absolute percent; no uncertainty intervals", r"\begin{tabular}{llrrrrrr}",
           r"\multicolumn{8}{c}{\textbf{" + marker.replace("—", "--") + r"}} \\",
           r"Budget & Arm & AP & APS & AP75-small & APM & APL & AR-large \\", r"\hline"]
    for budget in (600, 300):
        for arm in ARMS:
            vals = [next(r["value_percent"] for r in data["absolute"] if r["arm"] == arm and r["budget"] == budget and r["evaluator"] == "COCO_bridge" and r["metric"] == k) for k in ("AP", "APS", "AP75_small", "APM", "APL", "AR_large")]
            tex.append(" & ".join([str(budget), arm, *(formatted(v) for v in vals)]) + r" \\")
    tex.append(r"\end{tabular}")
    (out / "MAIN_TABLE.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    if make_plots:
        plots(out, data, marker)
    files = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in sorted(out.iterdir()) if p.is_file()]
    (out / "OUTPUT_INDEX.json").write_text(json.dumps(dict(synthetic=data["synthetic"], source_index_sha256=data["provenance"]["index_sha256"], tool_sha256=sha(Path(__file__).read_bytes()), figures=make_plots, files=files), indent=2) + "\n", encoding="utf-8")


def plots(out, data, marker):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    def finish(fig, name):
        # Reserve separate title/footer bands; never overlap tick labels.
        fig.get_layout_engine().set(rect=(0, .07, 1, .86))
        title = fig.suptitle(marker, fontsize=11, y=.985)
        title.set_in_layout(False)
        fig.text(.5, .012, "N historical / D-U controlled / negative values retained / no cross-seed uncertainty", ha="center", fontsize=8)
        if data["synthetic"]:
            fig.text(.5, .5, "SYNTHETIC ONLY", ha="center", fontsize=32, rotation=25, alpha=.18)
        for ext in ("svg", "png"):
            fig.savefig(out / (name + "." + ext), dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, key in zip(axes.flat, ("AP", "APS", "AP75_small", "APM", "APL", "AR_large")):
        for j, budget in enumerate((600, 300)):
            for i, pair in enumerate(PAIRS):
                r = next(r for r in data["deltas"] if r["pair"] == pair and r["budget"] == budget and r["evaluator"] == "COCO_bridge" and r["metric"] == key)
                pos = i + (j - .5) * .34
                if r["delta_pp"] is None:
                    ax.text(pos, 0, "UNKNOWN", rotation=90, ha="center", va="bottom", fontsize=6)
                else:
                    ax.bar(pos, r["delta_pp"], .32, color=("#3176a1", "#d58a42")[j], label=f"top{budget}" if i == 0 else None)
        ax.axhline(0, color="#555", lw=.8); ax.set_xticks(range(3), PAIRS); ax.set_title(key); ax.set_ylabel("Delta (pp)")
    axes.flat[0].legend(); finish(fig, "paired_deltas")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, key, label in zip(axes, ("mean_seconds", "peak_allocated_GiB"), ("Fixed64 seconds (mean +/- timing sample SD)", "Peak allocated GiB (maximum of 6)")):
        for i, row in enumerate(data["cost_summary"]):
            if row[key] is None:
                ax.text(i, 0, "UNKNOWN", ha="center")
            else:
                ax.bar(i, row[key], color="#3176a1", yerr=row["sample_SD_seconds"] if key == "mean_seconds" else None, capsize=4)
        ax.set_xticks(range(3), ARMS); ax.set_ylabel(label)
    finish(fig, "cost")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, suffix in zip(axes, ("mean", "median")):
        for i, arm in enumerate(("U", "D")):
            row = next(r for r in data["centers"] if r["arm"] == arm)
            for j, phase in enumerate(("coarse", "final")):
                val = row[phase + "_" + suffix]; pos = i + (j - .5) * .32
                if val is None:
                    ax.text(pos, 0, "UNKNOWN", rotation=90, va="bottom", ha="center")
                else:
                    ax.bar(pos, val, .30, color=("#8a959e", "#3176a1")[j], label=phase if i == 0 else None)
        ax.set_xticks(range(2), [f"{r['arm']} (n={formatted(r['matched_small'])})" for r in data["centers"] if r["arm"] != "N"])
        ax.set_ylabel("Normalized center error " + suffix); ax.legend()
    finish(fig, "centers")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--index-sha256", required=True)
    p.add_argument("--closed", type=Path)
    p.add_argument("--closed-sha256")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true", help="only explicit constructed test fixtures; watermarked")
    p.add_argument("--no-figures", action="store_true", help="stdlib-only tables; default also renders matplotlib SVG/PNG")
    args = p.parse_args(argv)
    try:
        require(not args.output.exists(), "output must be a new directory")
        require(not args.output.resolve().is_relative_to(args.index.resolve().parent), "output cannot be inside receipt folder")
        data = load(args.index, args.index_sha256, args.closed, args.closed_sha256, args.synthetic)
        render(data, args.output, not args.no_figures)
        print(json.dumps(dict(status="RENDERED", synthetic=data["synthetic"], evidence_complete=data["evidence_complete"], quality=data["quality"], cost=data["cost"], missing_count=len(data["missing"]), output=str(args.output)), ensure_ascii=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        print("H2_PAPER_INPUT_REJECTED: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
