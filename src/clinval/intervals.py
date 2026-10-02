"""Confidence intervals: Wilson for proportions, clustered bootstrap for the rest.

Why clustering matters: patient-window rows from the same patient are
correlated. Resampling *rows* treats 400 windows from 100 patients as 400
independent observations and produces intervals that are too narrow. clinval
resamples **patients**, which is why :class:`~clinval.schemas.config.Bootstrap`
defaults ``cluster_by`` to ``patient_id``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with a confidence interval.

    ``low``/``high`` are ``nan`` when the interval is undefined -- an empty
    sample, or a bootstrap in which the statistic was never computable. That is
    reported rather than coerced to the point estimate, so a reader can tell an
    interval of zero width apart from an absent one.
    """

    point: float
    low: float = float("nan")
    high: float = float("nan")
    method: str = ""
    alpha: float = 0.05
    n: int = 0
    n_clusters: int = 0

    @property
    def width(self) -> float:
        """Interval width, or ``nan`` if undefined."""
        return self.high - self.low

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON export."""
        return {
            "point": _clean(self.point),
            "low": _clean(self.low),
            "high": _clean(self.high),
            "method": self.method,
            "alpha": self.alpha,
            "n": self.n,
            "n_clusters": self.n_clusters,
        }

    def format(self, digits: int = 3) -> str:
        """Render as ``0.812 (0.740-0.869)``, or ``0.812 (CI undefined)``."""
        if math.isnan(self.point):
            return "n/a"
        if math.isnan(self.low) or math.isnan(self.high):
            return f"{self.point:.{digits}f} (CI undefined)"
        return f"{self.point:.{digits}f} ({self.low:.{digits}f}-{self.high:.{digits}f})"


def _clean(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval because it stays inside ``[0, 1]`` and
    keeps sensible coverage at small ``n`` and at proportions near 0 or 1 --
    both routine in phenotype validation, where a rare label may have five
    positives.

    Args:
        successes: Number of successes.
        n: Number of trials.
        alpha: Two-sided significance level.

    Returns:
        An :class:`Interval`. For ``n == 0`` every field is ``nan``.
    """
    if n < 0 or successes < 0:
        raise ValueError(f"successes and n must be non-negative, got {successes} / {n}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")
    if n == 0:
        return Interval(float("nan"), method="wilson", alpha=alpha, n=0)

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(
        point=p,
        low=max(0.0, center - half),
        high=min(1.0, center + half),
        method="wilson",
        alpha=alpha,
        n=n,
    )


def clustered_bootstrap_ci(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster_by: str = "patient_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    point: float | None = None,
) -> Interval:
    """Percentile bootstrap CI that resamples whole clusters.

    Clusters are drawn with replacement to the original cluster count; every
    row of a drawn cluster is included. This is the standard cluster bootstrap
    for correlated observations.

    Determinism: given the same ``seed``, the same frame, and the same
    statistic, the result is reproducible. The generator is seeded once and
    draws are taken in order.

    Args:
        data: Rows to resample.
        statistic: Maps a frame to a scalar. May return ``nan`` for a resample
            where the statistic is undefined (an all-one-class draw, say);
            those are dropped before taking percentiles and counted.
        cluster_by: Column naming the cluster. When the column is missing or
            entirely null, each row becomes its own cluster and the method is
            recorded as ``bootstrap_unclustered``.
        n_boot: Number of resamples.
        alpha: Two-sided significance level.
        seed: Seed for the random generator.
        point: Point estimate. Computed from ``data`` when omitted.

    Returns:
        An :class:`Interval`. With a single cluster the point estimate is
        returned and the bounds are ``nan``, because one cluster carries no
        between-cluster information to resample.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    point_est = float(statistic(data)) if point is None else float(point)
    n_rows = len(data)
    if n_rows == 0:
        return Interval(float("nan"), method="clustered_bootstrap", alpha=alpha, n=0)

    unclustered = cluster_by not in data.columns or data[cluster_by].isna().all()
    method = "bootstrap_unclustered" if unclustered else "clustered_bootstrap"
    if unclustered:
        keys = np.arange(n_rows)
        groups: dict[Any, np.ndarray] = {i: np.array([i]) for i in keys}
    else:
        codes = data[cluster_by].astype("object").fillna("__missing__")
        groups = {k: np.asarray(idx) for k, idx in codes.groupby(codes).indices.items()}
        keys = np.array(list(groups.keys()), dtype=object)

    n_clusters = len(keys)
    if n_clusters < 2:
        return Interval(
            point=point_est,
            method=method + "_single_cluster",
            alpha=alpha,
            n=n_rows,
            n_clusters=n_clusters,
        )

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    positions = np.arange(n_clusters)
    for b in range(n_boot):
        drawn = rng.choice(positions, size=n_clusters, replace=True)
        row_idx = np.concatenate([groups[keys[d]] for d in drawn])
        try:
            values[b] = statistic(data.iloc[row_idx])
        except (ValueError, ZeroDivisionError, KeyError):
            values[b] = np.nan

    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return Interval(
            point=point_est,
            method=method + "_degenerate",
            alpha=alpha,
            n=n_rows,
            n_clusters=n_clusters,
        )
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(
        point=point_est,
        low=float(low),
        high=float(high),
        method=method,
        alpha=alpha,
        n=n_rows,
        n_clusters=n_clusters,
    )


def paired_bootstrap_difference(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster_by: str = "patient_id",
    unit_col: str = "unit_key",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[Interval, float]:
    """Compare two systems on the same units with a paired cluster bootstrap.

    Pairing is what makes the comparison powerful: both systems are scored on
    the identical set of units in every resample, so unit difficulty cancels.
    Units present for only one system are dropped, and the count of dropped
    units is not silently hidden -- it shows up as a smaller ``n``.

    Args:
        data_a: Scored rows for system A.
        data_b: Scored rows for system B.
        statistic: Maps a frame to a scalar.
        cluster_by: Cluster column for resampling.
        unit_col: Column identifying the shared unit.
        n_boot: Number of resamples.
        alpha: Two-sided significance level.
        seed: Random seed.

    Returns:
        ``(interval_for_a_minus_b, two_sided_p)``. The p-value is the bootstrap
        proportion of resamples whose difference crosses zero, doubled and
        clipped to 1.0; it is approximate by construction and reported to three
        decimals in the report.
    """
    shared = set(data_a[unit_col]) & set(data_b[unit_col])
    a = data_a[data_a[unit_col].isin(shared)].sort_values(unit_col).reset_index(drop=True)
    b = data_b[data_b[unit_col].isin(shared)].sort_values(unit_col).reset_index(drop=True)
    if len(shared) == 0:
        return Interval(float("nan"), method="paired_bootstrap", alpha=alpha), float("nan")

    point = float(statistic(a)) - float(statistic(b))

    cluster_src = a if cluster_by in a.columns else None
    if cluster_src is None or cluster_src[cluster_by].isna().all():
        keys = np.arange(len(a))
        groups = {i: np.array([i]) for i in keys}
    else:
        codes = cluster_src[cluster_by].astype("object").fillna("__missing__")
        groups = {k: np.asarray(idx) for k, idx in codes.groupby(codes).indices.items()}
        keys = np.array(list(groups.keys()), dtype=object)

    n_clusters = len(keys)
    if n_clusters < 2:
        return (
            Interval(
                point,
                method="paired_bootstrap_single_cluster",
                alpha=alpha,
                n=len(a),
                n_clusters=n_clusters,
            ),
            float("nan"),
        )

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    positions = np.arange(n_clusters)
    for i in range(n_boot):
        drawn = rng.choice(positions, size=n_clusters, replace=True)
        row_idx = np.concatenate([groups[keys[d]] for d in drawn])
        try:
            diffs[i] = statistic(a.iloc[row_idx]) - statistic(b.iloc[row_idx])
        except (ValueError, ZeroDivisionError, KeyError):
            diffs[i] = np.nan

    finite = diffs[np.isfinite(diffs)]
    if finite.size < 2:
        return (
            Interval(
                point,
                method="paired_bootstrap_degenerate",
                alpha=alpha,
                n=len(a),
                n_clusters=n_clusters,
            ),
            float("nan"),
        )
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two_sided = 2.0 * min((finite <= 0).mean(), (finite >= 0).mean())
    return (
        Interval(
            point=point,
            low=float(low),
            high=float(high),
            method="paired_bootstrap",
            alpha=alpha,
            n=len(a),
            n_clusters=n_clusters,
        ),
        float(min(1.0, p_two_sided)),
    )


def bootstrap_many(
    data: pd.DataFrame,
    statistics: dict[str, Callable[[pd.DataFrame], float]],
    *,
    cluster_by: str = "patient_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Interval]:
    """Bootstrap several statistics over one set of resamples.

    Sharing resamples across metrics is both faster than looping and more
    coherent: sensitivity and specificity are then computed on the same draws,
    so their intervals describe one resampling distribution.
    """
    names = list(statistics)
    point = {name: _safe(statistics[name], data) for name in names}
    n_rows = len(data)
    if n_rows == 0:
        return {n: Interval(float("nan"), method="clustered_bootstrap", alpha=alpha) for n in names}

    unclustered = cluster_by not in data.columns or data[cluster_by].isna().all()
    method = "bootstrap_unclustered" if unclustered else "clustered_bootstrap"
    if unclustered:
        keys: Sequence[Any] = list(range(n_rows))
        groups = {i: np.array([i]) for i in keys}
    else:
        codes = data[cluster_by].astype("object").fillna("__missing__")
        groups = {k: np.asarray(idx) for k, idx in codes.groupby(codes).indices.items()}
        keys = list(groups.keys())

    n_clusters = len(keys)
    if n_clusters < 2:
        return {
            n: Interval(
                point[n],
                method=method + "_single_cluster",
                alpha=alpha,
                n=n_rows,
                n_clusters=n_clusters,
            )
            for n in names
        }

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {n: [] for n in names}
    positions = np.arange(n_clusters)
    for _ in range(n_boot):
        drawn = rng.choice(positions, size=n_clusters, replace=True)
        row_idx = np.concatenate([groups[keys[d]] for d in drawn])
        sample = data.iloc[row_idx]
        for name in names:
            draws[name].append(_safe(statistics[name], sample))

    out: dict[str, Interval] = {}
    for name in names:
        arr = np.asarray(draws[name], dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            out[name] = Interval(
                point[name],
                method=method + "_degenerate",
                alpha=alpha,
                n=n_rows,
                n_clusters=n_clusters,
            )
            continue
        low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = Interval(
            point=point[name],
            low=float(low),
            high=float(high),
            method=method,
            alpha=alpha,
            n=n_rows,
            n_clusters=n_clusters,
        )
    return out


def _safe(fn: Callable[[pd.DataFrame], float], data: pd.DataFrame) -> float:
    try:
        return float(fn(data))
    except (ValueError, ZeroDivisionError, KeyError):
        return float("nan")
