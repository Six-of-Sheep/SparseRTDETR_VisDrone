"""Finite, independent numerical validation; no model or dataset is loaded."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import cvxpy as cp
import clarabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.localization.operators import (
    EPSILON, NumericalIntegrityError, checked_float32, edge_order, entropy,
    moments, physical_support, positive_prior, probabilities64, project,
    weighting_function,
)
from sparse_rtdetr.localization.moment_prior import maximum_entropy


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, atol=1e-8):
    require(np.allclose(actual, expected, rtol=0, atol=atol),
            f"analytic mismatch: {np.asarray(actual).tolist()} != {np.asarray(expected).tolist()}")


def reference(pi, x):
    # Independent primal exponential-cone problem; no main solver helpers.
    t = (x - (x[0] + x[-1]) / 2.) / ((x[-1] - x[0]) / 2.)
    a = np.array([np.ones(len(x)), t, t ** 2])
    r = cp.Variable(len(x))
    objective = cp.Minimize(cp.sum(cp.rel_entr(r, np.ones(len(x)))))
    problem = cp.Problem(objective, [a @ r == a @ pi, r >= .001 / len(x)])
    problem.solve(solver="CLARABEL", max_iter=300, tol_gap_abs=1e-10,
                  tol_gap_rel=1e-10, tol_feas=1e-10)
    require(problem.status == "optimal", "independent convex solver did not certify optimal: " + problem.status)
    result = np.asarray(r.value)
    require(np.max(np.abs(a @ result - a @ pi)) <= 1e-8, "reference primal residual")
    require(np.min(result) >= .001 / len(x) - 1e-8, "reference lower bound")
    return result, {"status": problem.status, "objective": float(problem.value),
                    "iterations": problem.solver_stats.num_iters,
                    "solve_seconds": problem.solver_stats.solve_time}


def run(report):
    fixtures = report["fixtures"]
    x = np.array([-1., 0., 1.]); y = np.array([-1., -.5, .5, 1.])
    p = np.array([.25, .5, .25])
    one = project(p, x, y)
    close(one.probability, [.25] * 4)
    close(one.diagnostics["source_variance"], .5)
    close(one.diagnostics["projected_variance"], .625)
    back = project(one.probability, y, x)
    close(back.probability, [.375, .25, .375])
    close(back.diagnostics["projected_variance"], .75)
    close(.5 * np.abs(back.probability - p).sum(), .25)
    fixtures["one_way_and_roundtrip"] = {"one_way": one.diagnostics, "roundtrip": back.diagnostics,
        "roundtrip_probability": back.probability.tolist(), "roundtrip_tv": .25}
    tail = project(p, 2 * x, x)
    close(tail.diagnostics["tail_mass"], .5)
    close(tail.diagnostics["mean_shift"], 0.)
    close(tail.diagnostics["source_variance"], 2.)
    close(tail.diagnostics["projected_variance"], .5)
    fixtures["tail_cancellation"] = tail.diagnostics
    shifted = project(np.array([0., 1.]), np.array([.1, .3]), np.array([.1, .19, .2]))
    close(shifted.probability, [0., 0., 1.])
    close(shifted.diagnostics["mean_shift"], -.1)
    fixtures["translated_endpoint_roundoff"] = shifted.diagnostics
    for name, pp, shift in [("left", [1., 0., 0.], 1.), ("right", [0., 0., 1.], -1.)]:
        event = project(np.array(pp), 2 * x, x)
        close(event.diagnostics["mean_shift"], shift)
        fixtures[name + "_tail"] = event.diagnostics

    t = np.linspace(-1., 1., 33); u = np.ones(33) / 33; f = .001 / 33
    c = np.zeros(33); c[[0, 8, 16, 24, 32]] = [1, -4, 6, -4, 1]
    raw = u + .005 * c
    same = project(raw, t, t)
    close(same.probability, raw)
    pi, epsdiag = positive_prior(same, t)
    close(pi, u + .004995 * c)
    close(moments(pi, t), [1., 0., 17. / 48.])
    case_list = [("same_moment_shape", pi, u)]
    active_truth = np.full(33, f)
    active_truth[10:23] = (1. - 20. * f) * np.exp(-64. * t[10:23] ** 2) / np.exp(-64. * t[10:23] ** 2).sum()
    active_pi = active_truth.copy(); active_pi[14:19] += .01 * np.array([1, -4, 6, -4, 1])
    case_list.append(("active_lower_bound", active_pi, active_truth))
    for index in (0, 16, 32):
        pp = np.full(33, f); pp[index] += 1. - .001
        case_list.append((f"point_{index}", pp, pp))
    for name, indices in [("adjacent", [16, 17]), ("endpoints", [0, 32])]:
        pp = np.full(33, f); pp[indices] += (1. - .001) / 2.
        case_list.append((name, pp, pp))
    for name, pp, truth in case_list:
        print("checking MM fixture", name, flush=True)
        result = maximum_entropy(pp, t)
        oracle, oracle_meta = reference(pp, t)
        close(result.probability, truth, 2e-6)
        close(result.probability, oracle, 3e-6)
        require(abs(entropy(result.probability) - entropy(oracle)) <= 1e-7, "reference objective differs")
        require(entropy(result.probability) + 1e-7 >= entropy(pp), "entropy below feasible input")
        fixtures[name] = {"input": pp.tolist(), "solution": result.probability.tolist(),
            "certificate": result.certificate, "reference": oracle_meta,
            "reference_probability_max_error": float(np.max(np.abs(result.probability - oracle))),
            "analytic_probability_max_error": float(np.max(np.abs(result.probability - truth))),
            "input_entropy": entropy(pp), "output_entropy": entropy(result.probability),
            "tv": float(.5 * np.abs(pp - result.probability).sum())}

    rho = np.array(fixtures["same_moment_shape"]["solution"])
    close(fixtures["same_moment_shape"]["tv"], .03996)
    def update(prior, residual):
        z = np.log(prior) + residual
        q = np.exp(z - z.max()); return q / q.sum()
    response = {}
    for name, residual in [("zero", np.zeros(33)), ("constant_nonzero", np.ones(33) * .7),
                            ("last_atom_log2", np.r_[np.zeros(32), np.log(2.)])]:
        pa, pb = update(pi, residual), update(rho, residual)
        ma, mb = pa @ t, pb @ t
        if name in ("zero", "constant_nonzero"):
            close(ma, mb)
        else:
            close(ma, pi[-1] / (1 + pi[-1])); close(mb, u[-1] / (1 + u[-1]))
            require(abs(ma - mb) > 1e-4, "expected shape-dependent response missing")
        response[name] = {"pi_mean": float(ma), "mm_mean": float(mb), "difference": float(ma-mb)}
    fixtures["shared_residual_response"] = response
    # The actual torch FP32 log/softmax path consumes the checked FP32 priors.
    import torch
    from sparse_rtdetr.localization.distribution import residual_from_prior
    pi32 = checked_float32(pi, t, lower_bound=f)
    rho32 = checked_float32(rho, t, lower_bound=f)
    for name, residual in [("zero", np.zeros(33)), ("constant_nonzero", np.ones(33) * .7),
                            ("last_atom_log2", np.r_[np.zeros(32), np.log(2.)])]:
        _, pa = residual_from_prior(torch.from_numpy(pi32), torch.tensor(residual, dtype=torch.float32))
        _, pb = residual_from_prior(torch.from_numpy(rho32), torch.tensor(residual, dtype=torch.float32))
        aa, bb = pa.double().numpy(), pb.double().numpy()
        expected_a, expected_b = update(pi, residual), update(rho, residual)
        close(moments(aa, t), moments(expected_a, t), 3e-6)
        close(moments(bb, t), moments(expected_b, t), 3e-6)
        response[name]["fp32_pi_moments"] = moments(aa, t).tolist()
        response[name]["fp32_mm_moments"] = moments(bb, t).tolist()
    fixtures["epsilon"] = epsdiag

    # Four edges, real nonuniform FDR support, and outside-canvas valid boxes.
    w = weighting_function(); close(w[[0, 16, 32]], [-4, 0, 4])
    ref = np.array([1.2, -.1, .03, .06])
    support_unsorted = physical_support(ref, ascending=False)
    support = physical_support(ref)
    pv = np.tile(np.arange(1, 34, dtype=np.float64) / 561, (4, 1))
    close(np.sum(support_unsorted * pv, axis=1), np.sum(support * edge_order(pv), axis=1))
    close(edge_order(edge_order(pv)), pv)
    close(np.mean(support, axis=1), ref[[0, 1, 0, 1]] + np.array([-.5, -.5, .5, .5]) * ref[[2, 3, 2, 3]])
    actual = []
    moved = physical_support(np.array([1.21, -.09, .035, .07]))
    for edge in range(4):
        result = project(edge_order(pv)[edge].astype(np.float32), support[edge], moved[edge])
        positive, diag = positive_prior(result, moved[edge])
        close(moments(positive, moved[edge]),
              (1-EPSILON)*moments(result.probability/result.probability.sum(), moved[edge]) +
              EPSILON*moments(u, moved[edge]))
        mm = maximum_entropy(positive, moved[edge])
        checked_float32(positive, moved[edge], lower_bound=f)
        actual.append({"projection": result.diagnostics, "epsilon": diag, "mm": mm.certificate})
    fixtures["nonuniform_physical_support"] = actual
    # Precheck boundary cases: reject, do not silently normalize invalid inputs.
    rejected = []
    for name, action in [
        ("negative", lambda: probabilities64(np.array([-.1, 1.1]))),
        ("nonfinite", lambda: probabilities64(np.array([np.nan, 1.]))),
        ("mass_over_tolerance", lambda: probabilities64(np.array([.5, .500004], dtype=np.float32))),
        ("duplicate_support", lambda: project(np.array([.5, .5]), np.array([0., 0.]), x)),
        ("tiny_width", lambda: physical_support(np.array([0., 0., 1e-7, .1]))),
        ("negative_height", lambda: physical_support(np.array([0., 0., .1, -.1]))),
    ]:
        try: action()
        except NumericalIntegrityError: rejected.append(name)
        else: raise AssertionError("did not reject " + name)
    close(probabilities64(np.array([.5, .500002], dtype=np.float32)).sum(), 1.)
    fixtures["invalid_rejected"] = rejected


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "localization-operator-validation-v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "numpy": np.__version__, "cvxpy": cp.__version__, "clarabel": clarabel.__version__,
              "fixtures": {}, "model_calls": 0, "cloud_contact": False, "status": "RUNNING"}
    try:
        run(report); report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc), traceback=traceback.format_exc())
        if isinstance(exc, NumericalIntegrityError): report["failure_evidence"] = exc.evidence
    report["elapsed_seconds"] = time.monotonic() - started
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    paths = [Path(__file__)] + list((Path(__file__).resolve().parents[1] / "src/sparse_rtdetr/localization").glob("*.py"))
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (args.output / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"], "fixtures": len(report["fixtures"]), "error": report.get("error")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
