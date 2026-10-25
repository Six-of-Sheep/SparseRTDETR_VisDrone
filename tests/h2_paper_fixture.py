"""Constructed scalar receipts only; no model, image, checkpoint or AP evaluation.

The shape follows e50bedb. Counts/times/metrics are deliberately fabricated for
the offline reader. Some fixtures exercise hypothetical scientific booleans,
unlike the actual CPU experiment chain, which writes null scientific decisions.
"""
import copy
import importlib.util
import json
from pathlib import Path
import statistics
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import h2_paper_results as report


def fixture(folder, edit=None):
    folder = Path(folder); folder.mkdir()
    docs = {}; refs = {}; remote = "/CONSTRUCTED/attempt-formal-001"

    def add(name, value, original=None):
        raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        (folder / name).write_bytes(raw)
        refs[name] = dict(path=original or remote + "/" + name, bytes=len(raw), sha256=report.sha(raw))
        docs[name] = copy.deepcopy(value)
        return refs[name]

    all_scores = {}
    for arm, inc in (("N", 0.), ("U", .001), ("D", .004)):
        all_scores[arm] = {}
        for b in report.BUDGETS:
            co = {k: .30 + inc + (0 if b == "600" else -.01) for k in report.COCO}
            # Keep a negative protected difference for complete reporting.
            if arm == "D":
                co["APL"] = .293 if b == "600" else .283
            val = dict(COCO_bridge=dict(protocol="r0_endpoint_development_raw_ignore_v1",
                units="0..1 AP, -1 means no GT in an area/class aggregation", metrics=co, maxDets=[1, 10, 100]),
                official={**{k: 30. + inc * 100 for k in report.PRIMARY}, "evaluated_image_count": 8},
                official_APS=None, spatial_filter_counts={})
            all_scores[arm][b] = val
    costs = dict(status="PASS", stage="cost", forward_calls=24,
        measurements={a: [dict(seconds=1. + i * .01 + j * .01, images=4, batch=4, dtype="FP32", filter_counts={},
                               peak_allocated_bytes=(i + 2) * 1024, peak_reserved_bytes=(i + 3) * 2048) for j in range(6)] for i, a in enumerate(report.ARMS)},
        orders=report.ORDERS, scope="CPU_FIXTURE_NOT_COST_EVIDENCE", boundary="CONSTRUCTED_ONLY")
    center = {a: dict(images=8, matched_small=n, N="UNKNOWN", scope="endpoint_local_support_not_training_cause",
                coarse_mean=.4, coarse_median=.2, final_mean=.3, final_median=.25,
                improved=n - 3, unchanged=1, worsened=2,
                raw_records=dict(path=remote + f"/infer-{a}/CENTER_RECORDS.json", bytes=1000, sha256="a" * 64)) for a, n in (("U", 10), ("D", 12))}
    options = {"scores": all_scores, "cost": costs, "centers": center}
    if edit:
        edit(options)
    for a, bs in all_scores.items():
        for b, val in bs.items():
            add(f"{a}-{b}.json", val, remote + f"/score/{a}-{b}.json")
    src = add("source-manifest.json", dict(commit=report.COMMIT, source_files=[]), "/CONSTRUCTED/manifest.json")
    c = dict(task_id=report.TASK, scope=report.SCOPE, seed=0, commit=report.COMMIT,
        job_root=remote, profile="CPU_FIXTURE_ONLY", fixture={"paper_tool_synthetic": True}, source=src,
        data={"development": {"manifest": "CONSTRUCTED"}}, pretrained={"sha256": "b" * 64},
        historical_N=dict(status=report.N_STATUS, scores={b: refs[f"N-{b}.json"] for b in report.BUDGETS}))
    c["contract_sha256"] = report.digest(c); cid = c["contract_sha256"]
    add("contract.json", c, "/CONSTRUCTED/contract.json")

    def stage(st, result):
        result = dict(result, contract_sha256=cid)
        add(st + "-RESULT.json", result, remote + "/" + st + "/RESULT.json")
        sup = add(st + "-supervisor-exit.json", dict(status="PASS", arm=st, contract_sha256=cid,
            returncode=0, error=None, remaining_live=[], cleanup_errors=[]), remote + "/" + st + "/supervisor-exit.json")
        worker = add(st + "-worker-exit.json", dict(status="PASS", contract_sha256=cid, error=None), remote + "/" + st + "/worker-exit.json")
        add(st + "-CLOSED.json", dict(status="PASS", arm=st, contract_sha256=cid, result=refs[st + "-RESULT.json"],
            supervisor_exit=sup, worker_exit=worker), remote + "/" + st + "/CLOSED.json")

    for arm in ("U", "D"):
        bind = add(f"train-{arm}-binding.json", dict(arm=arm, contract_sha256=cid, source_sha256=src["sha256"],
            data_sha256=report.digest(c["data"]), pretrained_sha256=c["pretrained"]["sha256"]), remote + f"/train-{arm}/binding.json")
        ep = add(f"train-{arm}-endpoint.json", dict(checkpoint={"path": "CONSTRUCTED_NOT_READ"}, binding=bind,
            epoch=2, updates=2, EMA=True, initial_model_sha256="c" * 64, used_acceptance_state=False, used_old_endpoint=False), remote + f"/train-{arm}/endpoint.json")
        stage("train-" + arm, dict(status="PASS", stage="train-" + arm, endpoint=ep, actual_updates=2,
            forward_calls=4, BF16_forward_calls=4, student_inputs_sha256="d" * 64))
        cen = add(f"infer-{arm}-CENTER_SUMMARY.json", center[arm], remote + f"/infer-{arm}/CENTER_SUMMARY.json")
        stage("infer-" + arm, dict(status="PASS", stage="infer-" + arm, images=8, forward_calls=2, EMA=True, centers=cen))
    stage("score", dict(status="PASS", stage="score", new_groups=4, new_evaluator_calls=8, N_recomputed=False,
        forward_calls=0, results={a: {b: refs[f"{a}-{b}.json"] for b in report.BUDGETS} for a in report.ARMS}))
    costs["summary"] = {a: dict(mean_seconds=statistics.mean(r["seconds"] for r in rr), sample_SD_seconds=statistics.stdev(r["seconds"] for r in rr),
        peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in rr), peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in rr)) for a, rr in costs["measurements"].items()}
    stage("cost", costs)
    deltas = {}; gates = []
    for pair in report.PAIRS:
        a, b = pair.split("-"); deltas[pair] = {}
        for budget in report.BUDGETS:
            x, y = all_scores[a][budget], all_scores[b][budget]
            co = {k: 100 * (v - y["COCO_bridge"]["metrics"][k]) for k, v in x["COCO_bridge"]["metrics"].items() if k in y["COCO_bridge"]["metrics"]}
            pr = {k: x["official"][k] - y["official"][k] for k in report.PRIMARY if k in x["official"] and k in y["official"]}
            deltas[pair][budget] = dict(COCO_pp=co, primary_pp=pr)
            if pair == "U-N":
                continue
            for k, minimum in report.LIMITS[int(budget)].items():
                vals = pr if k == "primary_AP" else co; mk = "AP" if k == "primary_AP" else k
                if mk not in vals:
                    continue
                defined = k == "primary_AP" or (x["COCO_bridge"]["metrics"][k] >= 0 and y["COCO_bridge"]["metrics"][k] >= 0)
                gates.append(dict(pair=pair, budget=int(budget), metric=k, delta_pp=vals[mk], minimum=minimum,
                    defined=defined, passed=defined and vals[mk] >= minimum))
    summary = costs["summary"]
    ratios = {a + "/" + b: dict(E2E=summary[a]["mean_seconds"] / summary[b]["mean_seconds"],
        peak_allocated=summary[a]["peak_allocated_bytes"] / summary[b]["peak_allocated_bytes"] if summary[b]["peak_allocated_bytes"] else None) for a, b in (("D", "N"), ("D", "U"), ("U", "N"))}
    # Fabricated booleans exercise preservation. They are never scientific evidence.
    comp = dict(status="PASS", stage="compare", forward_calls=0, fixture_only=True, absolute=all_scores,
        deltas_pp=deltas, quality_gates=gates, quality_passed=all(g["passed"] for g in gates), cost_passed=False,
        cost=docs["cost-RESULT.json"], ratios=ratios, scientific_interpretation="SYNTHETIC_TEST_ONLY")
    stage("compare", comp)
    add("RESULT.json", dict(status="PASS", scope=report.SCOPE, stages=list(report.STAGES), contract_sha256=cid,
        forward_calls=54, GPU_seconds=0, quality_passed=comp["quality_passed"], cost_passed=comp["cost_passed"],
        comparison=refs["compare-RESULT.json"], historical_N_status=report.N_STATUS, fixture_only=True))
    add("CUMULATIVE_COST.json", dict(task_id=report.TASK, contract_sha256=cid, active=None, fixture_only=True,
        forward_calls=54, forwards_completed=54, actual_updates=4, updates_completed=4, GPU_seconds=0, stages=[]))
    add("EXECUTION_EXIT.json", dict(status="PASS", returncode=0, contract_sha256=cid, error=None))
    idx = dict(status="PASS", original_N_STOP_preserved=True, full_state_predictions_events_centers="SERVER_ONLY",
        files=[dict(name=n, source=r, returned=dict(r, path=remote + "/slim-export/" + n)) for n, r in refs.items()])
    idx_raw = (json.dumps(idx, indent=2) + "\n").encode(); (folder / "SHA_INDEX.json").write_bytes(idx_raw)
    closed = folder.parent / (folder.name + "-ROOT_CLOSED.json")
    closed.write_text(json.dumps(dict(status="PASS", contract_sha256=cid, result=refs["RESULT.json"],
        export_index=dict(path=remote + "/slim-export/SHA_INDEX.json", sha256=report.sha(idx_raw), bytes=len(idx_raw))), indent=2) + "\n", encoding="utf-8")
    return folder / "SHA_INDEX.json", closed


def rewrite(folder, name, edit, propagate=True):
    """Deliberately rebind synthetic graph for adversarial semantic checks."""
    folder = Path(folder)
    values = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json") if p.name != "SHA_INDEX.json"}
    edit(values[name])
    index = json.loads((folder / "SHA_INDEX.json").read_text())
    refs = {r["name"]: r for r in index["files"]}

    def replace(obj, old, new):
        if isinstance(obj, dict):
            if obj.get("sha256") == old["sha256"] and obj.get("bytes") == old["bytes"]:
                obj.update(sha256=new["sha256"], bytes=new["bytes"])
            for v in obj.values():
                replace(v, old, new)
        elif isinstance(obj, list):
            for v in obj:
                replace(v, old, new)
    # Iterate through acyclic reference dependencies until all bytes settle.
    for _ in range(10):
        changed = False
        for n, v in values.items():
            raw = (json.dumps(v, indent=2, ensure_ascii=False) + "\n").encode()
            old = dict(refs[n]["source"]); new = dict(old, bytes=len(raw), sha256=report.sha(raw))
            if old != new:
                changed = True
                (folder / n).write_bytes(raw)
                for r in (refs[n]["source"], refs[n]["returned"]):
                    r.update(bytes=new["bytes"], sha256=new["sha256"])
                if propagate:
                    for vv in values.values():
                        replace(vv, old, new)
        if not changed:
            break
    (folder / "SHA_INDEX.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    closed = folder.parent / (folder.name + "-ROOT_CLOSED.json")
    cl = json.loads(closed.read_text()); rr = refs["RESULT.json"]["source"]
    cl["result"] = rr
    cl["export_index"].update(bytes=(folder / "SHA_INDEX.json").stat().st_size, sha256=report.sha((folder / "SHA_INDEX.json").read_bytes()))
    closed.write_text(json.dumps(cl, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); idx, cl = fixture(args.output)
    print(json.dumps(dict(SYNTHETIC_ONLY=True, index=str(idx), index_sha256=report.sha(idx.read_bytes()), closed=str(cl), closed_sha256=report.sha(cl.read_bytes())), indent=2))
