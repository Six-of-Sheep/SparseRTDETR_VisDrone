"""Fixed three-seed paired readout; no prediction/model access or resampling."""
from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path
import statistics

SEEDS = (0, 1, 2)
BUDGETS = ("600", "300")
ARMS = ("A", "B", "C")
PAIRS = (("B", "A"), ("C", "A"), ("C", "B"))
METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
LIMITS = {"600": {"AP": .30, "APS": .50, "AP75_small": -.20,
                  "APM": -.50, "APL": -.50},
          "300": {"AP": -.20, "APS": -.20}}


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric score required")
    return float(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_stats(values):
    """Exactly three seed-level paired values; no image/query pseudo-replicates."""
    values = [finite(v) for v in values]
    _require(len(values) == 3, "fixed three-seed statistics require all0/1/2")
    return {"n": 3, "values_by_seed_0_1_2": values,
            "mean_pp": statistics.mean(values),
            "sample_sd_pp": statistics.stdev(values), "sample_sd_ddof": 1,
            "worst_pp": min(values)}


def summarize(records):
    """Validate endpoint summaries and derive the frozen strict decision.

    Seed0's certified old-schema comparison is reused as-is by authenticated
    file reference. Its numbers are read, not reevaluated. New seeds must carry
    explicit identity and all engineering evaluation must already be closed.
    """
    _require(set(records) == {"0", "1", "2"}, "all fixed seeds0/1/2 required")
    per_seed = {}
    for seed in SEEDS:
        record = records[str(seed)]
        _require(record["images"] == 548, "not the fixed548 development")
        _require(record["COCO_bridge_original_score_units"] == "0..1"
                 and record["difference_units"] == "percentage points", "comparison units differ")
        if seed:
            _require(type(record.get("seed")) is int and record["seed"] == seed,
                     "new comparison seed identity differs")
            _require(isinstance(record.get("scope"), str)
                     and isinstance(record.get("checkpoint_sha256"), str),
                     "new comparison lacks own e120 identity")
        values = {arm: {budget: {metric: finite(record["scores"][arm][budget][metric])
                               for metric in METRICS} for budget in BUDGETS} for arm in ARMS}
        _require(all(0 <= x <= 1 for arm in ARMS for budget in BUDGETS
                     for x in values[arm][budget].values()), "endpoint score outside original units")
        deltas = {budget: {first + "-" + second: {metric:
            (values[first][budget][metric] - values[second][budget][metric]) * 100.
            for metric in METRICS} for first, second in PAIRS} for budget in BUDGETS}
        for budget in BUDGETS:
            for first, second in PAIRS:
                pair = first + "-" + second
                for metric in METRICS:
                    _require(deltas[budget][pair][metric]
                             == finite(record["differences_pp"][budget][pair][metric]),
                             "stored paired delta disagrees with original scores")
        gates = {}
        for budget, limits in LIMITS.items():
            for metric, threshold in limits.items():
                key = f"top{budget}_{metric}"
                delta = deltas[budget]["C-A"][metric]
                gate = {"difference_pp": delta, "minimum_pp": threshold,
                        "passed": delta >= threshold}
                _require(record["gates_C_minus_A"][key] == gate,
                         "stored gate differs from fixed unrounded rule: " + key)
                gates[key] = gate
        costs = record["cost"]
        times = {arm: finite(costs["seconds"][arm]) for arm in ARMS}
        _require(all(t > 0 for t in times.values()), "cost measurement is not positive")
        ratio = times["C"] / times["A"]
        _require(finite(costs["C_over_A"]) == ratio
                 and costs["maximum_C_over_A"] == 6.
                 and costs["passed"] is (ratio <= 6.), "cost decision disagrees with fixed6x")
        scientific_pass = all(g["passed"] for g in gates.values())
        _require(record["benefit_and_protection_passed"] is scientific_pass,
                 "stored scientific decision differs")
        _require(record["screen_passed"] is (scientific_pass and ratio <= 6.),
                 "stored joint decision differs")
        per_seed[str(seed)] = {"seed": seed, "scores": values, "differences_pp": deltas,
            "gates_C_minus_A": gates, "benefit_and_protection_passed": scientific_pass,
            "matched64_e2e_seconds": times,
            "C_over_A": ratio, "B_over_A": times["B"] / times["A"],
            "C_over_B": times["C"] / times["B"], "cost_passed": ratio <= 6.,
            "screen_passed": scientific_pass and ratio <= 6.,
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "scope": record.get("scope", "certified original seed0 e120 EMA"),
            "official_separate": record["official_separate"]}
    stats = {budget: {first + "-" + second: {metric: paired_stats([
        per_seed[str(seed)]["differences_pp"][budget][first + "-" + second][metric]
        for seed in SEEDS]) for metric in METRICS} for first, second in PAIRS}
        for budget in BUDGETS}
    cost_stats = {}
    for key in ("C_over_A", "B_over_A", "C_over_B"):
        ratios = [per_seed[str(seed)][key] for seed in SEEDS]
        cost_stats[key] = {"n": 3, "values_by_seed_0_1_2": ratios,
            "mean": statistics.mean(ratios), "sample_sd": statistics.stdev(ratios),
            "sample_sd_ddof": 1, "worst_max": max(ratios)}
    science_pass = all(per_seed[str(s)]["benefit_and_protection_passed"] for s in SEEDS)
    cost_pass = all(per_seed[str(s)]["cost_passed"] for s in SEEDS)
    return {"schema": "r18-sahi-fixed-three-seed-decision-v1", "seeds": list(SEEDS),
        "per_seed": per_seed, "paired_delta_statistics_pp": stats,
        "cost_ratio_statistics": cost_stats,
        "strict_benefit_and_protection_passed": science_pass,
        "strict_cost_passed": cost_pass, "strict_repeatability_passed": science_pass and cost_pass,
        "decision": "PASS" if science_pass and cost_pass else "FAIL",
        "decision_rule": "all three seeds individually pass all seven C-A gates and matched C/A<=6; means never override failure",
        "C_minus_B_rule": "descriptive extra slice gains and APL tradeoff; no new invented threshold",
        "statistical_unit": "paired inference-arm increment within each independently trained seed",
        "bootstrap_calls": 0, "seed0_metrics_recomputed": False,
        "new_inference_or_training": False, "holdout_accessed": False,
        "limitations": ["three fixed seeds", "development repeatedly used",
                        "not an independent confirmation set", "SAHI-style practical adaptation",
                        "paired C-B changes context, scale and candidate pool together"],
        "completion": "both fixed new seeds completed; no automatic extra seed, tuning or training"}


def _write_text(path, text):
    from sparse_rtdetr.localization.runio import fsync_directory
    path = Path(path)
    with path.open("x", encoding="utf8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def _report(result):
    lines = ["# 固定SAHI式方案：三seed成对复验", "",
        "结论：**" + result["decision"] + "**。按冻结规则逐seed裁决，平均收益不抵消单seed失败。",
        "共同协议适配RT-DETRv2-R18，seed0原结果复用，seed1/2分别独立训练固定e120 EMA；同seed的A/B/C共用权重。", "",
        "| seed | 七项收益/保护 | C/A耗时 | 成本≤6 | 总判定 |",
        "|---|---|---:|---|---|"]
    for seed in SEEDS:
        record = result["per_seed"][str(seed)]
        lines.append(f"| {seed} | {'PASS' if record['benefit_and_protection_passed'] else 'FAIL'} | "
                     f"{record['C_over_A']:.4f} | {'PASS' if record['cost_passed'] else 'FAIL'} | "
                     f"{'PASS' if record['screen_passed'] else 'FAIL'} |")
    lines += ["", "端点原始分数按百分数展示；差值和门槛采用未舍入pp。", "",
              "| seed | 预算 | 方案 | AP | APS | AP75-small | APM | APL |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        for budget in BUDGETS:
            for arm in ARMS:
                values = result["per_seed"][str(seed)]["scores"][arm][budget]
                lines.append(f"| {seed} | {budget} | {arm} | " +
                    " | ".join(f"{100 * values[m]:.4f}" for m in ("AP", "APS", "AP75_small", "APM", "APL")) + " |")
    lines += ["", "| seed | C−A检查 | 实际pp | 最低pp | 判定 |",
              "|---|---|---:|---:|---|"]
    for seed in SEEDS:
        for name, gate in result["per_seed"][str(seed)]["gates_C_minus_A"].items():
            lines.append(f"| {seed} | {name} | {gate['difference_pp']:+.6f} | "
                         f"{gate['minimum_pp']:+.2f} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines += ["", "以下均为COCO bridge成对差值pp；标准差为3个seed的样本标准差（ddof=1），最差值为最小增量。", "",
              "| 预算 | 差值 | 指标 | seed0 | seed1 | seed2 | 均值 | 样本SD | 最差 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for budget in BUDGETS:
        for pair in ("B-A", "C-A", "C-B"):
            for metric in METRICS:
                stats = result["paired_delta_statistics_pp"][budget][pair][metric]
                vals = stats["values_by_seed_0_1_2"]
                lines.append(f"| {budget} | {pair} | {metric} | " +
                    " | ".join(f"{v:+.4f}" for v in vals) +
                    f" | {stats['mean_pp']:+.4f} | {stats['sample_sd_pp']:.4f} | {stats['worst_pp']:+.4f} |")
    lines += ["", "C−B的AP/APS收益与APL变化单列，不把C−A通过解释成切片单一因素的因果证据。",
        "官方风格完整结果在AGGREGATE.json中另列，原单位保留，未与COCO bridge混算。", "",
        "| seed | A端到端秒 | B端到端秒 | C端到端秒 | C/A | B/A | C/B |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        record = result["per_seed"][str(seed)]
        lines.append(f"| {seed} | " + " | ".join(f"{record['matched64_e2e_seconds'][a]:.6f}" for a in ARMS) +
            f" | {record['C_over_A']:.4f} | {record['B_over_A']:.4f} | {record['C_over_B']:.4f} |")
    lines += ["", "成本为同seed、同硬件、同64图及batch4口径的实测完整端到端时间；整图和四切片真实执行。",
        "seed0成本复用历史实测，不冒充与新seed同时测量；OS页缓存未刷新，不作为冷启动性能。", "",
        "范围限制：仅3个固定seed，development反复使用；不新增bootstrap或holdout。短窗验收不替代终点指标。",
        "本轮正常完成即停，不自动新增seed、调整切片或启动训练。",
        "大型checkpoint、预测和完整审计仅实验室保存，D盘仅必要白名单及SHA索引。",
        "AutoDL建议模式：无需开机；本任务仅使用实验室，AutoDL实际状态本轮未核验。", ""]
    return "\n".join(lines)


def aggregate(c, out, *, comparison_refs=None):
    """CPU closeout; authenticate three small comparisons then write summaries.

    Supply comparison_refs={'0':ref,'1':ref,'2':ref}; otherwise use
    c['seed0_comparison'] and c['seed_comparisons']['1'/'2'].
    """
    from sparse_rtdetr.localization.runio import read, reference, require, publish
    contracts = {}
    if comparison_refs is None:
        comparison_refs = {"0": c["seed0_comparison"]}
        if "seed_comparisons" in c:
            comparison_refs.update(c["seed_comparisons"])
        else:
            for ref in c["contracts"]:
                require(reference(ref["path"]) == ref, "seed execution contract bytes changed")
                contract = read(ref["path"])
                seed = str(contract["seed"])
                require(seed in ("1", "2") and seed not in contracts, "fixed seed contract duplicated")
                contracts[seed] = contract
                root = Path(contract["job_root"]) / "evaluation"
                closed = read(root / "CLOSED.json")
                require(closed["status"] == "PASS" and closed["seed"] == int(seed)
                        and closed["result"] == reference(root / "RESULT.json"),
                        "seed evaluation closure changed")
                result = read(root / "RESULT.json")
                require(result["status"] == "PASS" and result["seed"] == int(seed)
                        and result["source_sha256"] == contract["source_sha256"]
                        and result["contract_sha256"] == contract["contract_sha256"],
                        "seed evaluation contract/source changed")
                comparison_refs[seed] = result["comparison"]
    require(set(comparison_refs) == {"0", "1", "2"}, "exact fixed seed reference set required")
    records = {}
    for seed, ref in comparison_refs.items():
        require(reference(ref["path"]) == ref, "comparison bytes/SHA differ for seed" + seed)
        records[seed] = read(ref["path"])
    result = summarize(records)
    identity = {"task_id": c["task_id"],
                "source_sha256": c.get("source_sha256", records["1"]["source_sha256"])}
    if "execution_sha256" in c:
        identity["execution_sha256"] = c["execution_sha256"]
    if "contract_sha256" in c:
        identity["contract_sha256"] = c["contract_sha256"]
    result.update(**identity, comparison_refs=comparison_refs)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    publish(out / "AGGREGATE.json", result)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["budget", "pair", "metric", "seed0_pp", "seed1_pp", "seed2_pp",
                     "mean_pp", "sample_SD_pp_ddof1", "worst_min_pp"])
    for budget in BUDGETS:
        for first, second in PAIRS:
            pair = first + "-" + second
            for metric in METRICS:
                row = result["paired_delta_statistics_pp"][budget][pair][metric]
                writer.writerow([budget, pair, metric, *row["values_by_seed_0_1_2"],
                                 row["mean_pp"], row["sample_sd_pp"], row["worst_pp"]])
    _write_text(out / "PAIRED_SUMMARY.csv", buffer.getvalue())
    _write_text(out / "REPORT.md", _report(result))
    receipt = {"status": "PASS", "scientific_decision": result["decision"],
        "strict_repeatability_passed": result["strict_repeatability_passed"],
        **identity, "comparison_refs": comparison_refs,
        "aggregate": reference(out / "AGGREGATE.json"), "table": reference(out / "PAIRED_SUMMARY.csv"),
        "report": reference(out / "REPORT.md"), "CPU_only": True, "bootstrap_calls": 0,
        "seed0_metrics_recomputed": False}
    publish(out / "RESULT.json", receipt)
    return receipt


run = aggregate
