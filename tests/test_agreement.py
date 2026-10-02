"""Agreement metrics, checked against scikit-learn where a reference exists."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from clinval.agreement import (
    Confusion,
    accuracy,
    cohen_kappa,
    compute_agreement,
    compute_metrics,
    f1,
    inter_annotator_agreement,
    npv,
    pabak,
    ppv,
    prevalence,
    sensitivity,
    specificity,
)
from clinval.schemas.config import Bootstrap


def frame(pred: list[bool], gold: list[bool], patients: list[str] | None = None) -> pd.DataFrame:
    """Build a scorable unit-level frame."""
    n = len(pred)
    return pd.DataFrame(
        {
            "pred": pd.array(pred, dtype="boolean"),
            "gold": pd.array(gold, dtype="boolean"),
            "pred_present": [True] * n,
            "gold_present": [True] * n,
            "label": ["l"] * n,
            "patient_id": patients or [f"P{i}" for i in range(n)],
            "unit_key": [f"U{i}" for i in range(n)],
        }
    )


class TestConfusion:
    def test_counts(self):
        c = Confusion.from_frame(frame([True, True, False, False], [True, False, True, False]))
        assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
        assert c.n == 4

    def test_nulls_excluded(self):
        df = frame([True, True], [True, True])
        df.loc[1, "pred"] = pd.NA
        c = Confusion.from_frame(df)
        assert c.n == 1
        assert c.tp == 1

    def test_empty(self):
        assert Confusion.from_frame(frame([], [])).n == 0

    def test_derived_counts(self):
        c = Confusion(tp=3, fp=2, fn=4, tn=5)
        assert c.n_positive_gold == 7
        assert c.n_positive_pred == 5


class TestPointMetrics:
    def test_known_table(self):
        c = Confusion(tp=8, fp=2, fn=1, tn=9)
        assert sensitivity(c) == pytest.approx(8 / 9)
        assert specificity(c) == pytest.approx(9 / 11)
        assert ppv(c) == pytest.approx(8 / 10)
        assert npv(c) == pytest.approx(9 / 10)
        assert accuracy(c) == pytest.approx(17 / 20)
        assert prevalence(c) == pytest.approx(9 / 20)

    @pytest.mark.parametrize(
        ("metric", "table"),
        [
            (sensitivity, Confusion(tp=0, fp=3, fn=0, tn=5)),  # no gold positives
            (specificity, Confusion(tp=4, fp=0, fn=2, tn=0)),  # no gold negatives
            (ppv, Confusion(tp=0, fp=0, fn=3, tn=5)),  # predicted no positives
            (npv, Confusion(tp=3, fp=5, fn=0, tn=0)),  # predicted no negatives
        ],
    )
    def test_undefined_is_nan_not_zero(self, metric, table):
        """An undefined metric must not read as 'the system got everything wrong'."""
        assert math.isnan(metric(table))

    def test_accuracy_of_empty_is_nan(self):
        assert math.isnan(accuracy(Confusion()))

    def test_f1_matches_sklearn(self):
        rng = np.random.default_rng(3)
        for _ in range(25):
            p = rng.random(60) < 0.5
            g = rng.random(60) < 0.4
            c = Confusion.from_frame(frame(list(p), list(g)))
            expected = f1_score(g, p, zero_division=np.nan)
            ours = f1(c)
            if math.isnan(ours) or (isinstance(expected, float) and math.isnan(expected)):
                assert math.isnan(ours) and math.isnan(float(expected))
            else:
                assert ours == pytest.approx(float(expected))


class TestCohenKappa:
    def test_matches_sklearn_on_random_tables(self):
        """SPEC §8.1: kappa must agree with the reference implementation."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = int(rng.integers(5, 200))
            p = rng.random(n) < rng.uniform(0.05, 0.95)
            g = rng.random(n) < rng.uniform(0.05, 0.95)
            if len({*p.tolist(), *g.tolist()}) < 2:
                continue
            c = Confusion.from_frame(frame(list(p), list(g)))
            expected = cohen_kappa_score(p, g, labels=[False, True])
            ours = cohen_kappa(c)
            if math.isnan(expected):
                continue
            assert ours == pytest.approx(expected, abs=1e-9), (c.as_dict(), expected, ours)

    def test_perfect_agreement_single_class_is_one(self):
        """Both raters said 'no' to everything. They agreed on everything."""
        assert cohen_kappa(Confusion(tp=0, fp=0, fn=0, tn=40)) == 1.0
        assert cohen_kappa(Confusion(tp=40, fp=0, fn=0, tn=0)) == 1.0

    def test_complete_disagreement_single_class_is_zero(self):
        assert cohen_kappa(Confusion(tp=0, fp=20, fn=0, tn=0)) == 0.0
        assert cohen_kappa(Confusion(tp=0, fp=0, fn=20, tn=0)) == 0.0

    def test_empty_is_nan(self):
        assert math.isnan(cohen_kappa(Confusion()))

    def test_rare_label_collapses_kappa_but_not_accuracy(self):
        """The reason clinval reports kappa, PABAK, and prevalence together."""
        c = Confusion(tp=1, fp=2, fn=2, tn=95)
        assert accuracy(c) > 0.95
        assert cohen_kappa(c) < 0.35
        assert pabak(c) > 0.9


class TestPabak:
    def test_formula(self):
        c = Confusion(tp=8, fp=2, fn=1, tn=9)
        assert pabak(c) == pytest.approx(2 * accuracy(c) - 1)

    def test_independent_of_prevalence(self):
        """Two tables with the same agreement rate and different base rates."""
        a = Confusion(tp=45, fp=5, fn=5, tn=45)
        b = Confusion(tp=5, fp=5, fn=5, tn=85)
        assert accuracy(a) == pytest.approx(0.90)
        assert accuracy(b) == pytest.approx(0.90)
        assert pabak(a) == pytest.approx(pabak(b))
        assert cohen_kappa(a) != pytest.approx(cohen_kappa(b))

    def test_empty_is_nan(self):
        assert math.isnan(pabak(Confusion()))


class TestComputeMetrics:
    def test_all_known(self):
        out = compute_metrics(Confusion(tp=5, fp=1, fn=2, tn=8))
        assert "cohen_kappa" in out
        assert "pabak" in out

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValueError, match="unknown metric"):
            compute_metrics(Confusion(tp=1, fp=1, fn=1, tn=1), ["auroc"])


class TestComputeAgreement:
    def test_interval_method_per_metric(self):
        """Wilson for proportions, bootstrap for kappa -- and the row says which."""
        df = frame([True] * 30 + [False] * 30, [True] * 25 + [False] * 35)
        rep = compute_agreement(
            df, metrics=["sensitivity", "cohen_kappa"], bootstrap=Bootstrap(n=60, seed=0)
        )
        assert rep.metrics["sensitivity"].method == "wilson"
        assert rep.metrics["cohen_kappa"].method.startswith("clustered_bootstrap")

    def test_unscored_counted_not_dropped_silently(self):
        df = frame([True, True, False], [True, False, True])
        df.loc[2, "pred"] = pd.NA
        rep = compute_agreement(df, metrics=["accuracy"], bootstrap=Bootstrap(n=20, seed=0))
        assert rep.n_units == 2
        assert rep.unscored == 1
        assert rep.coverage == pytest.approx(2 / 3)

    def test_clusters_counted(self):
        df = frame(
            [True] * 8, [True] * 8, patients=["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"]
        )
        rep = compute_agreement(df, metrics=["accuracy"], bootstrap=Bootstrap(n=20, seed=0))
        assert rep.n_units == 8
        assert rep.n_clusters == 4

    def test_per_label_breakdown(self):
        df = frame([True, False, True, False], [True, True, False, False])
        df["label"] = ["a", "a", "b", "b"]
        rep = compute_agreement(df, metrics=["accuracy"], bootstrap=Bootstrap(n=20, seed=0))
        assert set(rep.per_label) == {"a", "b"}
        assert rep.per_label_confusion["a"].n == 2

    def test_markdown_renders_all_sections(self):
        df = frame([True, False, True, False], [True, True, False, False])
        rep = compute_agreement(
            df, metrics=["accuracy", "cohen_kappa"], bootstrap=Bootstrap(n=20, seed=0)
        )
        md = rep.to_markdown()
        assert "gold +" in md
        assert "Per label" in md
        assert "accuracy" in md

    def test_as_dict_is_json_safe(self):
        import json

        df = frame([True, True], [True, False])
        rep = compute_agreement(df, metrics=["specificity"], bootstrap=Bootstrap(n=10, seed=0))
        json.dumps(rep.as_dict())  # must not raise on nan

    def test_empty_frame(self):
        rep = compute_agreement(
            frame([], []), metrics=["accuracy"], bootstrap=Bootstrap(n=10, seed=0)
        )
        assert rep.n_units == 0
        assert math.isnan(rep.metrics["accuracy"].point)


class TestInterAnnotatorAgreement:
    def test_two_raters(self):
        df = pd.DataFrame(
            {
                "annotator": ["a"] * 4 + ["b"] * 4,
                "note_id": ["N1", "N2", "N3", "N4"] * 2,
                "label": ["l"] * 8,
                "value": [True, True, False, False, True, False, False, False],
            }
        )
        out = inter_annotator_agreement(df)
        assert out["n_annotators"] == 2
        entry = out["pairs"]["a|b"]
        assert entry["n_units"] == 4
        assert entry["observed_agreement"] == pytest.approx(0.75)

    def test_no_overlap_reported_not_crashed(self):
        df = pd.DataFrame(
            {
                "annotator": ["a", "b"],
                "note_id": ["N1", "N2"],
                "label": ["l", "l"],
                "value": [True, True],
            }
        )
        out = inter_annotator_agreement(df)
        assert out["pairs"]["a|b"]["n_units"] == 0

    def test_three_raters_gives_three_pairs(self):
        df = pd.DataFrame(
            {
                "annotator": ["a", "b", "c"],
                "note_id": ["N1"] * 3,
                "label": ["l"] * 3,
                "value": [True, True, False],
            }
        )
        out = inter_annotator_agreement(df)
        assert set(out["pairs"]) == {"a|b", "a|c", "b|c"}
