"""Confidence intervals: Wilson against statsmodels, bootstrap reproducibility."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

from clinval.agreement import Confusion, accuracy, cohen_kappa
from clinval.intervals import (
    Interval,
    bootstrap_many,
    clustered_bootstrap_ci,
    paired_bootstrap_difference,
    wilson_ci,
)


def frame(pred: list[bool], gold: list[bool], patients: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pred": pd.array(pred, dtype="boolean"),
            "gold": pd.array(gold, dtype="boolean"),
            "patient_id": patients,
            "unit_key": [f"U{i}" for i in range(len(pred))],
        }
    )


def acc(df: pd.DataFrame) -> float:
    return accuracy(Confusion.from_frame(df))


class TestWilson:
    @pytest.mark.parametrize(
        ("k", "n"),
        [(0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (0, 1), (1, 1), (3, 7), (50, 100), (1, 500)],
    )
    def test_matches_statsmodels(self, k, n):
        """SPEC §8.1: Wilson bounds must match the statsmodels reference."""
        ours = wilson_ci(k, n)
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert ours.low == pytest.approx(lo, abs=1e-10)
        assert ours.high == pytest.approx(hi, abs=1e-10)

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20])
    def test_matches_statsmodels_across_alpha(self, alpha):
        ours = wilson_ci(7, 23, alpha=alpha)
        lo, hi = proportion_confint(7, 23, alpha=alpha, method="wilson")
        assert ours.low == pytest.approx(lo, abs=1e-10)
        assert ours.high == pytest.approx(hi, abs=1e-10)

    def test_stays_in_unit_interval_at_the_boundary(self):
        """Where the Wald interval would leave [0, 1], Wilson does not."""
        for k, n in [(0, 5), (5, 5), (0, 1), (1, 1)]:
            iv = wilson_ci(k, n)
            assert 0.0 <= iv.low <= iv.high <= 1.0

    def test_contains_point_estimate(self):
        rng = np.random.default_rng(1)
        for _ in range(100):
            n = int(rng.integers(1, 400))
            k = int(rng.integers(0, n + 1))
            iv = wilson_ci(k, n)
            assert iv.low - 1e-12 <= iv.point <= iv.high + 1e-12

    def test_n_zero_is_nan_not_an_exception(self):
        iv = wilson_ci(0, 0)
        assert math.isnan(iv.point)
        assert math.isnan(iv.low)
        assert iv.n == 0

    def test_successes_exceeding_n_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            wilson_ci(5, 3)

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            wilson_ci(-1, 3)

    def test_narrows_as_n_grows(self):
        widths = [wilson_ci(n // 2, n).width for n in (10, 50, 200, 1000)]
        assert widths == sorted(widths, reverse=True)


class TestClusteredBootstrap:
    def test_reproducible_with_fixed_seed(self):
        """SPEC §8.1: the same seed must give the same interval."""
        df = frame(
            [True] * 20 + [False] * 20,
            [True] * 15 + [False] * 25,
            [f"P{i // 4}" for i in range(40)],
        )
        a = clustered_bootstrap_ci(df, acc, n_boot=200, seed=7)
        b = clustered_bootstrap_ci(df, acc, n_boot=200, seed=7)
        assert (a.low, a.high, a.point) == (b.low, b.high, b.point)

    def test_different_seeds_differ(self):
        df = frame(
            [True] * 20 + [False] * 20,
            [True] * 15 + [False] * 25,
            [f"P{i // 4}" for i in range(40)],
        )
        a = clustered_bootstrap_ci(df, acc, n_boot=200, seed=1)
        b = clustered_bootstrap_ci(df, acc, n_boot=200, seed=2)
        assert (a.low, a.high) != (b.low, b.high)

    def test_contains_point_estimate(self):
        df = frame(
            [True] * 30 + [False] * 30,
            [True] * 28 + [False] * 32,
            [f"P{i // 3}" for i in range(60)],
        )
        iv = clustered_bootstrap_ci(df, acc, n_boot=400, seed=0)
        assert iv.low <= iv.point <= iv.high

    def test_single_cluster_reports_nan_bounds(self):
        """One cluster carries no between-cluster information to resample."""
        df = frame([True, False, True], [True, True, False], ["P1", "P1", "P1"])
        iv = clustered_bootstrap_ci(df, acc, n_boot=100, seed=0)
        assert iv.n_clusters == 1
        assert math.isnan(iv.low)
        assert not math.isnan(iv.point)
        assert "single_cluster" in iv.method

    def test_clustering_widens_the_interval(self):
        """The whole reason cluster_by exists: ignoring it understates the CI."""
        n = 120
        # Strong within-patient correlation: each patient is entirely right or wrong.
        pred, gold, pats = [], [], []
        rng = np.random.default_rng(5)
        for p in range(20):
            correct = bool(rng.random() < 0.7)
            for _ in range(6):
                g = bool(rng.random() < 0.5)
                gold.append(g)
                pred.append(g if correct else (not g))
                pats.append(f"P{p}")
        assert len(pred) == n
        df = frame(pred, gold, pats)
        clustered = clustered_bootstrap_ci(df, acc, cluster_by="patient_id", n_boot=600, seed=0)
        naive = clustered_bootstrap_ci(df, acc, cluster_by="__absent__", n_boot=600, seed=0)
        assert clustered.width > naive.width

    def test_missing_cluster_column_falls_back_and_says_so(self):
        df = frame([True, False], [True, True], ["P1", "P2"]).drop(columns=["patient_id"])
        iv = clustered_bootstrap_ci(df, acc, n_boot=50, seed=0)
        assert iv.method.startswith("bootstrap_unclustered")

    def test_empty_frame(self):
        df = frame([], [], [])
        iv = clustered_bootstrap_ci(df, acc, n_boot=50, seed=0)
        assert math.isnan(iv.point)
        assert iv.n == 0

    def test_statistic_raising_is_tolerated(self):
        df = frame([True] * 10, [True] * 10, [f"P{i}" for i in range(10)])

        def boom(_: pd.DataFrame) -> float:
            raise ValueError("undefined on this resample")

        iv = clustered_bootstrap_ci(df, boom, n_boot=20, seed=0, point=0.5)
        assert iv.point == 0.5
        assert "degenerate" in iv.method

    def test_n_boot_zero_rejected(self):
        df = frame([True], [True], ["P1"])
        with pytest.raises(ValueError, match="n_boot must be"):
            clustered_bootstrap_ci(df, acc, n_boot=0)


class TestBootstrapMany:
    def test_shares_resamples_across_metrics(self):
        df = frame(
            [True] * 25 + [False] * 25,
            [True] * 20 + [False] * 30,
            [f"P{i // 5}" for i in range(50)],
        )
        out = bootstrap_many(
            df,
            {"accuracy": acc, "kappa": lambda d: cohen_kappa(Confusion.from_frame(d))},
            n_boot=200,
            seed=0,
        )
        assert set(out) == {"accuracy", "kappa"}
        for iv in out.values():
            assert iv.low <= iv.point <= iv.high
            assert iv.n_clusters == 10

    def test_reproducible(self):
        df = frame([True] * 20, [True] * 18 + [False] * 2, [f"P{i // 4}" for i in range(20)])
        stats = {"accuracy": acc}
        a = bootstrap_many(df, stats, n_boot=100, seed=3)["accuracy"]
        b = bootstrap_many(df, stats, n_boot=100, seed=3)["accuracy"]
        assert (a.low, a.high) == (b.low, b.high)

    def test_single_cluster(self):
        df = frame([True, False], [True, True], ["P1", "P1"])
        out = bootstrap_many(df, {"accuracy": acc}, n_boot=50, seed=0)
        assert "single_cluster" in out["accuracy"].method

    def test_empty(self):
        out = bootstrap_many(frame([], [], []), {"accuracy": acc}, n_boot=50, seed=0)
        assert math.isnan(out["accuracy"].point)


class TestPairedBootstrap:
    def test_identical_systems_give_zero_difference(self):
        df = frame(
            [True] * 20 + [False] * 20,
            [True] * 18 + [False] * 22,
            [f"P{i // 4}" for i in range(40)],
        )
        iv, p = paired_bootstrap_difference(df, df.copy(), acc, n_boot=200, seed=0)
        assert iv.point == pytest.approx(0.0)
        assert iv.low <= 0.0 <= iv.high
        assert p == pytest.approx(1.0)

    def test_detects_a_real_difference(self):
        gold = [True] * 40 + [False] * 40
        good = list(gold)
        bad = [not g for g in gold[:30]] + list(gold[30:])
        pats = [f"P{i // 4}" for i in range(80)]
        a = frame(good, gold, pats)
        b = frame(bad, gold, pats)
        iv, p = paired_bootstrap_difference(a, b, acc, n_boot=400, seed=0)
        assert iv.point > 0.3
        assert iv.low > 0.0
        assert p < 0.05

    def test_only_shared_units_compared(self):
        gold = [True, True, False, False]
        a = frame([True] * 4, gold, ["P1", "P2", "P3", "P4"])
        b = frame([True] * 4, gold, ["P1", "P2", "P3", "P4"])
        b["unit_key"] = ["U0", "U1", "U9", "U8"]
        iv, _ = paired_bootstrap_difference(a, b, acc, n_boot=50, seed=0)
        assert iv.n == 2

    def test_no_shared_units(self):
        a = frame([True], [True], ["P1"])
        b = frame([True], [True], ["P2"])
        b["unit_key"] = ["ZZ"]
        iv, p = paired_bootstrap_difference(a, b, acc, n_boot=50, seed=0)
        assert math.isnan(iv.point)
        assert math.isnan(p)

    def test_reproducible(self):
        df_a = frame([True] * 20, [True] * 15 + [False] * 5, [f"P{i // 4}" for i in range(20)])
        df_b = frame([False] * 20, [True] * 15 + [False] * 5, [f"P{i // 4}" for i in range(20)])
        r1 = paired_bootstrap_difference(df_a, df_b, acc, n_boot=100, seed=9)
        r2 = paired_bootstrap_difference(df_a, df_b, acc, n_boot=100, seed=9)
        assert r1[0].low == r2[0].low
        assert r1[1] == r2[1]


class TestIntervalFormatting:
    def test_format_with_bounds(self):
        assert Interval(0.8123, 0.74, 0.869).format() == "0.812 (0.740-0.869)"

    def test_format_without_bounds(self):
        assert "CI undefined" in Interval(0.5).format()

    def test_format_nan_point(self):
        assert Interval(float("nan")).format() == "n/a"

    def test_as_dict_nulls_nan(self):
        d = Interval(float("nan")).as_dict()
        assert d["point"] is None
        assert d["low"] is None
