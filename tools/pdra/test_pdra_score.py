"""pytest mirror of the PDRA scoring-step logic (numpy-only, no torch, no evaluator).

Covers the frozen criteria constants, the verdict cell logic (pass / fail /
insufficient-evidence), the once-only pp conversion and the wrap byte counting.  The
evaluator call itself is thin glue over the frozen ``score_budget`` and is exercised by
the real scoring step, not here.

Run from the repository root:  python -m pytest tools/pdra/test_pdra_score.py -q
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _path in (_HERE, _REPO):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pdra_score as S


def _passing_metrics():
    """Comfortably passing 0..1 values for every arm/budget/metric."""
    metrics = {arm: {600: {}, 300: {}} for arm in S.ARMS}
    base = {"AP": 0.30, "AP50": 0.50, "AP75": 0.32, "APS": 0.20, "APM": 0.30,
            "APL": 0.40, "AP75_small": 0.10, "AR1": 0.30, "AR10": 0.40, "AR100": 0.45,
            "AR_small": 0.35, "AR_medium": 0.40, "AR_large": 0.50}
    for arm in S.ARMS:
        for budget in (600, 300):
            metrics[arm][budget] = dict(base)
    # PDRA above every control on the gated metrics, controls equal to each other
    for control in S.CONTROLS:
        metrics[control][600]["AP"] = 0.2900
        metrics[control][600]["APS"] = 0.1900
        metrics[control][600]["AP75_small"] = 0.0980   # pdra margin +0.20pp >= 0.0
        metrics[control][600]["APL"] = 0.3950          # pdra margin +0.50pp >= -0.20
        metrics[control][300]["AP"] = 0.3000
        metrics[control][300]["APS"] = 0.2000
    metrics[S.PDRA_ARM][600]["AP"] = 0.2950
    metrics[S.PDRA_ARM][600]["APS"] = 0.1950
    metrics[S.PDRA_ARM][600]["AP75_small"] = 0.1000
    metrics[S.PDRA_ARM][600]["APL"] = 0.4000
    metrics[S.PDRA_ARM][300]["AP"] = 0.2990            # -0.10pp >= -0.20
    metrics[S.PDRA_ARM][300]["APS"] = 0.1990
    return metrics


def test_criteria_frozen_values():
    assert S.CRITERIA["spec_version"] == "pdra-criteria-v1-20261031"
    assert S.CRITERIA["primary_split"] == "eval" and S.CRITERIA["diagnostic_split"] == "train"
    assert S.CRITERIA["primary_images"] == 210 and S.CRITERIA["diagnostic_images"] == 338
    assert S.CRITERIA["budgets"] == [600, 300] and S.CRITERIA["primary_budget"] == 600
    assert S.CRITERIA["gates_primary_600_pp"] == {"AP": 0.20, "APS": 0.30,
                                                  "AP75_small": 0.0, "APL": -0.20}
    assert S.CRITERIA["gates_secondary_300_pp"] == {"AP": -0.20, "APS": -0.20}
    assert set(S.CRITERIA["applies_to"]) == {"a_original_c", "b_single_box_quality",
                                             "c_whole_box_pairwise"}
    first = S.criteria_sha256()
    second = S.criteria_sha256(json.loads(json.dumps(S.CRITERIA)))
    assert first == second and len(first) == 64


def test_verdict_pass_requires_every_cell():
    record = S.verdict(_passing_metrics())
    assert record["status"] == "PASS"
    assert record["failed_cells"] == [] and record["insufficient_cells"] == []
    expected_cells = (3 * 4) + (3 * 2)
    assert len(record["cells"]) == expected_cells
    assert all(cell["status"] == "pass" for cell in record["cells"])


def test_verdict_fail_names_the_cells():
    metrics = _passing_metrics()
    metrics["c_whole_box_pairwise"][600]["APS"] = 0.1940   # pdra margin only +0.10pp < 0.30
    metrics["b_single_box_quality"][600]["APL"] = 0.4300   # pdra margin -3.00pp < -0.20
    record = S.verdict(metrics)
    assert record["status"] == "FAIL"
    assert set(record["failed_cells"]) == {"c_whole_box_pairwise@600:APS",
                                           "b_single_box_quality@600:APL"}
    assert record["insufficient_cells"] == []


def test_verdict_evidence_insufficient_dominates():
    metrics = _passing_metrics()
    metrics[S.PDRA_ARM][600]["APL"] = -1.0                 # frozen evaluator "no GT" code
    record = S.verdict(metrics)
    assert record["status"] == "EVIDENCE_INSUFFICIENT"
    assert record["insufficient_cells"] == ["a_original_c@600:APL",
                                            "b_single_box_quality@600:APL",
                                            "c_whole_box_pairwise@600:APL"]
    assert record["failed_cells"] == []


def test_delta_pp_conversion_exactly_once():
    metrics = _passing_metrics()
    metrics["a_original_c"][600]["AP"] = 0.2950            # margin exactly 0.0pp < 0.20
    record = S.verdict(metrics)
    cell = next(c for c in record["cells"]
                if c["control"] == "a_original_c" and c["budget"] == 600 and c["metric"] == "AP")
    assert abs(cell["delta_pp"] - 0.0) < 1e-9
    assert cell["status"] == "fail"
    assert "a_original_c@600:AP" in record["failed_cells"]


def test_wrap_totals_and_whitelist(tmp_path):
    s1, s2, sco = tmp_path / "s1", tmp_path / "s2", tmp_path / "score"
    for directory in (s1, s2, sco):
        directory.mkdir()

    def write(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    runtime = {"wall_seconds": 100.0, "peak_allocated_bytes": 3, "peak_reserved_bytes": 4,
               "caps": {"wall_seconds": 3600, "output_bytes": 11 * 1024 ** 3}}
    write(s1 / "RESULT.json", {"runtime": runtime})
    write(s1 / "PLAN.json", {"status": "PLAN_ONLY"})
    write(s2 / "RESULT.json", {"runtime": runtime})
    write(sco / "SCORE.json", {"status": "PASS", "elapsed_seconds": 7.0})
    payload = S.wrap(s1, s2, sco)
    assert payload["status"] == "PASS"
    assert payload["total_wall_seconds"] == 207.0
    assert payload["peak_reserved_bytes_max"] == 4
    assert payload["verdict_status"] == "PASS"
    assert payload["server_bytes_added"] == S.dir_bytes(s1) + S.dir_bytes(s2) + S.dir_bytes(sco)
    assert payload["returned_whitelist"]["bytes"] == sum(
        item["bytes"] for item in payload["returned_whitelist"]["files"])
    assert payload["predictions_returned"] is False
