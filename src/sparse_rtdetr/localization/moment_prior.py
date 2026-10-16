"""Moment-matched maximum entropy with an explicit lower-bound KKT certificate.

This is the frozen dual Newton algorithm, not a Gaussian approximation.
Failures include the full input and final solver state for preservation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .operators import EPSILON, NumericalIntegrityError, checked_float32, moments, standardized


@dataclass(frozen=True)
class MomentPrior:
    probability: np.ndarray
    probability32: np.ndarray
    certificate: dict


def maximum_entropy(pi: np.ndarray, support: np.ndarray) -> MomentPrior:
    x = np.asarray(support, dtype=np.float64)
    t, _, _ = standardized(x)
    p = np.asarray(pi, dtype=np.float64)
    f = EPSILON / len(t)
    if (p.shape != t.shape or not np.isfinite(p).all() or abs(p.sum() - 1.) > 1e-8
            or np.min(p) < f - 2e-12):
        raise NumericalIntegrityError("MM needs the validated post-epsilon FP64 prior", pi=p.tolist())
    a = np.stack((np.ones_like(t), t, t * t))
    b = a @ p
    lam = np.zeros(3, dtype=np.float64)
    history = []

    def state(l):
        v = a.T @ l
        exponent = -1. - v
        if np.max(exponent) >= 700 or not np.isfinite(v).all():
            return None
        unbounded = np.exp(exponent)
        rho = np.maximum(f, unbounded)
        free = unbounded > f
        g = a @ rho - b
        nu = np.where(free, 0., np.log(f) + 1. + v)
        primal = float(np.sum(rho * np.log(rho)))
        dual = float(primal + v @ rho - l @ b)
        residuals = {
            "primal_inf": float(np.max(np.abs(g))),
            "lower_bound_violation": float(max(0., f - np.min(rho))),
            "dual_negative_inf": float(max(0., -np.min(nu))),
            "stationarity_inf": float(np.max(np.abs(np.log(rho) + 1. + v - nu))),
            "complementarity_inf": float(np.max(np.abs(nu * (rho - f)))),
            "objective_gap_abs": float(abs(primal - dual)),
            "primal_relative_inf": float(np.max(np.abs(g)) / max(1., np.max(np.abs(b)))),
            "stationarity_relative_inf": float(np.max(np.abs(np.log(rho) + 1. + v - nu)) /
                                                max(1., np.max(np.abs(v)), np.max(np.abs(nu)))),
            "objective_gap_relative": float(abs(primal - dual) / max(1., abs(primal), abs(dual))),
        }
        passed = (residuals["primal_inf"] <= 1e-8 and residuals["lower_bound_violation"] <= 2e-12
                  and residuals["dual_negative_inf"] <= 1e-8 and residuals["stationarity_inf"] <= 1e-6
                  and residuals["complementarity_inf"] <= 1e-7 and residuals["objective_gap_abs"] <= 1e-7)
        return rho, free, g, nu, primal, dual, residuals, passed

    def fail(reason, s, iteration):
        details = {"pi": p.tolist(), "support": x.tolist(), "standardized_support": t.tolist(),
                   "target_moments": b.tolist(), "lambda": lam.tolist(), "iterations": iteration,
                   "history": history, "lower_bound": f}
        if s is not None:
            details.update(probability=s[0].tolist(), active=(~s[1]).tolist(), nu=s[3].tolist(), residuals=s[6])
        raise NumericalIntegrityError(reason, **details)

    for iteration in range(65):
        s = state(lam)
        if s is None:
            fail("MM nonfinite state", s, iteration)
        rho, free, g, nu, primal, dual, residuals, passed = s
        history.append({"iteration": iteration, "dual": dual, "free_bins": int(free.sum()), **residuals})
        if passed:
            try:
                out = checked_float32(rho, x, target=b, lower_bound=f)
            except NumericalIntegrityError as exc:
                history[-1]["conversion_failure"] = exc.evidence
                fail("MM final FP32 conversion failed: " + str(exc), s, iteration)
            certificate = {"iterations": iteration, "lambda": lam.tolist(), "nu": nu.tolist(),
                           "active": (~free).tolist(), "target_moments": b.tolist(),
                           "actual_moments": moments(rho, t).tolist(), "primal_objective": primal,
                           "dual_objective": dual, "lower_bound": f, "residuals": residuals,
                           "history": history, "passed": True}
            return MomentPrior(rho, out, certificate)
        if iteration == 64:
            fail("MM exhausted 64 Newton iterations", s, iteration)
        h = (a[:, free] * rho[free]) @ a[:, free].T
        eigen, vec = np.linalg.eigh(h)
        cutoff = max(1e-14, 1e-12 * float(eigen[-1]))
        keep = eigen > cutoff
        d = vec[:, keep] @ ((vec[:, keep].T @ g) / eigen[keep])
        dot = float(g @ d)
        if not np.isfinite(dot) or dot <= 0:
            norm = float(np.linalg.norm(g))
            if norm == 0:
                fail("MM zero gradient without a valid certificate", s, iteration)
            d = g / max(1., norm)
            dot = float(g @ d)
        accepted = False
        for halving in range(21):
            alpha = 2. ** (-halving)
            trial = state(lam + alpha * d)
            if trial is not None and trial[5] >= dual + 1e-4 * alpha * dot:
                lam = lam + alpha * d
                accepted = True
                history[-1].update(step=alpha, hessian_rank=int(keep.sum()))
                break
        if not accepted:
            fail("MM line search exhausted 20 halvings", s, iteration)
    raise AssertionError("unreachable")
