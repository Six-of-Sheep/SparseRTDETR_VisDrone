"""FP64 physical-support projection; independent of model and GPU runtimes.

The D-FINE weighting function follows the pinned upstream formula. Projection
is barycentric with endpoint tails, not an information-preserving transform.
No failed invariant is repaired by clipping an output distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

K = 33
EPSILON = 1e-3
MASS_TOL32 = 3e-6
MOMENT_TOL64 = 1e-8


class NumericalIntegrityError(RuntimeError):
    def __init__(self, message: str, **evidence: Any):
        super().__init__(message)
        self.evidence = evidence


def weighting_function(reg_max: int = 32, up: float = .5,
                       reg_scale: float = 4.) -> np.ndarray:
    if reg_max != 32 or up <= 0 or reg_scale <= 0:
        raise ValueError("This phase fixes reg_max=32 and positive support scales")
    upper = abs(up) * abs(reg_scale)
    step = (upper + 1.) ** (2. / (reg_max - 2))
    inner = np.array([step ** i - 1. for i in range(1, reg_max // 2)], dtype=np.float64)
    return np.r_[-2. * upper, -inner[::-1], 0., inner, 2. * upper]


def validate_support(support: np.ndarray) -> np.ndarray:
    x = np.asarray(support, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise NumericalIntegrityError("nonfinite or invalid support", support=x.tolist())
    gap = 64. * np.finfo(np.float64).eps * max(1., float(np.max(np.abs(x))))
    if not np.all(np.diff(x) > gap):
        raise NumericalIntegrityError("unresolvable/non-increasing support", support=x.tolist(), min_gap=gap)
    return x


def physical_support(reference: np.ndarray, *, up: float = .5,
                     reg_scale: float = 4., ascending: bool = True) -> np.ndarray:
    """cxcywh -> (..., left/top/right/bottom, 33), with no canvas clipping."""
    r = np.asarray(reference, dtype=np.float64)
    if r.shape[-1:] != (4,) or not np.isfinite(r).all() or np.any(r[..., 2:] <= 0):
        raise NumericalIntegrityError("invalid positive-area reference", reference=r.tolist())
    w = weighting_function(up=up, reg_scale=reg_scale)
    centers = r[..., [0, 1, 0, 1]]
    sizes = r[..., [2, 3, 2, 3]]
    sign = np.array([-1., -1., 1., 1.])
    support = centers[..., None] + sign[:, None] * sizes[..., None] * (.5 + w / reg_scale)
    if ascending:
        support = support.copy()
        support[..., :2, :] = support[..., :2, ::-1]
    rows = support.reshape(-1, K)
    for row in rows:
        validate_support(row if ascending else np.sort(row))
    return support


def edge_order(values: np.ndarray) -> np.ndarray:
    """Self-inverse permutation between W order and ascending physical order."""
    p = np.asarray(values).copy()
    if p.shape[-2:] != (4, K):
        raise ValueError("expected (..., 4, 33)")
    p[..., :2, :] = p[..., :2, ::-1]
    return p


def probabilities64(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities)
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all() or np.any(p < 0):
        raise NumericalIntegrityError("invalid probabilities", probabilities=p.tolist())
    p = p.astype(np.float64)
    mass = float(p.sum())
    if abs(mass - 1.) > MASS_TOL32:
        raise NumericalIntegrityError("input probability mass exceeds tolerance", mass=mass, probabilities=p.tolist())
    return p / mass


def standardized(support: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = validate_support(support)
    center = float((x[0] + x[-1]) / 2.)
    half = float((x[-1] - x[0]) / 2.)
    return (x - center) / half, center, half


def moments(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.array([p.sum(), p @ x, p @ (x * x)], dtype=np.float64)


def entropy(p: np.ndarray) -> float:
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def checked_float32(p: np.ndarray, support: np.ndarray, *, target: np.ndarray | None = None,
                    lower_bound: float | None = None) -> np.ndarray:
    """Validate the actual converted values, without post-conversion repair."""
    p = np.asarray(p, dtype=np.float64)
    t, _, _ = standardized(support)
    b = moments(p, t) if target is None else np.asarray(target, dtype=np.float64)
    out = p.astype(np.float32)
    actual = out.astype(np.float64)
    errors = np.abs(moments(actual, t) - b)
    if not np.isfinite(actual).all() or np.any(actual < 0) or np.max(errors) > MASS_TOL32:
        raise NumericalIntegrityError("FP32 prior conversion failed", moment_errors=errors.tolist())
    if lower_bound is not None:
        allowance = 2. * float(np.spacing(np.float32(lower_bound)))
        if np.min(actual) < lower_bound - allowance:
            raise NumericalIntegrityError("FP32 lower bound failed", minimum=float(np.min(actual)), allowance=allowance)
    return out


@dataclass(frozen=True)
class Projection:
    probability: np.ndarray
    diagnostics: dict[str, float]


def project(probabilities: np.ndarray, source: np.ndarray, destination: np.ndarray) -> Projection:
    x = validate_support(source)
    y = validate_support(destination)
    p = probabilities64(probabilities)
    if p.shape != x.shape:
        raise ValueError("probability/source shape mismatch")
    t, center, half = standardized(y)
    sx = (x - center) / half
    # The transformed endpoints can differ from +/-1 by a rounding bit.
    # Clamp to the actual target atoms, avoiding negative barycentric mass.
    clipped = np.clip(sx, t[0], t[-1])
    lo = np.searchsorted(t, clipped, side="right") - 1
    lo = np.clip(lo, 0, len(y) - 2)
    right = (clipped - t[lo]) / (t[lo + 1] - t[lo])
    q = np.zeros(len(y), dtype=np.float64)
    np.add.at(q, lo, p * (1. - right))
    np.add.at(q, lo + 1, p * right)
    before = moments(p, sx)
    after = moments(q, t)
    low, high = x < y[0], x > y[-1]
    tail = float(p[low | high].sum())
    predicted_drift = float(p @ (clipped - sx))
    mass_error = abs(after[0] - before[0])
    mean_error = abs(after[1] - before[1] - predicted_drift)
    if (mass_error > MOMENT_TOL64 or mean_error > MOMENT_TOL64 or
            np.any(q < 0) or not np.isfinite(q).all()):
        raise NumericalIntegrityError("projection invariant failed", mass_error=mass_error, mean_error=mean_error)
    diagnostics = {
        "tail_mass": tail, "left_tail_mass": float(p[low].sum()), "right_tail_mass": float(p[high].sum()),
        "source_mean": float(p @ x), "projected_mean": float(q @ y),
        "mean_shift": float((after[1] - before[1]) * half),
        "standardized_mean_shift": float(after[1] - before[1]),
        "predicted_tail_mean_shift": predicted_drift * half,
        "mass_error": float(mass_error), "tail_adjusted_mean_error": float(mean_error),
        "source_variance": float((before[2] - before[1] ** 2) * half ** 2),
        "projected_variance": float((after[2] - after[1] ** 2) * half ** 2),
        "source_entropy": entropy(p), "projected_entropy": entropy(q),
    }
    return Projection(q, diagnostics)


def positive_prior(projection: Projection, support: np.ndarray,
                   epsilon: float = EPSILON) -> tuple[np.ndarray, dict[str, float]]:
    if epsilon != EPSILON:
        raise ValueError("epsilon is frozen at 0.001")
    q = np.asarray(projection.probability, dtype=np.float64)
    if abs(q.sum() - 1.) > MOMENT_TOL64 or np.any(q < 0):
        raise NumericalIntegrityError("invalid projection before epsilon")
    q = q / q.sum()
    pi = (1. - epsilon) * q + epsilon / len(q)
    x = validate_support(support)
    t, _, _ = standardized(x)
    delta = moments(pi, t) - moments(q, t)
    checked_float32(pi, x, lower_bound=epsilon / len(q))
    return pi, {"epsilon_mean_shift": float((pi - q) @ x),
                "epsilon_standardized_mean_shift": float(delta[1]),
                "epsilon_standardized_second_moment_shift": float(delta[2])}
