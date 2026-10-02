"""Property-based tests. SPEC §8.2.

Four invariants the harness must never violate, checked over generated inputs
rather than hand-picked examples.
"""

from __future__ import annotations

import math

import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from clinval.aggregate import AlignedFrame, aggregate_to_windows, align_predictions
from clinval.agreement import Confusion, accuracy, cohen_kappa, f1
from clinval.intervals import clustered_bootstrap_ci, wilson_ci
from clinval.schemas.annotation import Annotation
from clinval.schemas.config import EvalConfig, Unit
from clinval.schemas.prediction import Prediction, Span

FAST = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

note_ids = st.text(alphabet="ABCDEFGH0123456789", min_size=1, max_size=4).map(lambda s: f"N{s}")
labels = st.sampled_from(["diarrhea", "constipation", "vomiting"])


@st.composite
def pred_gold_pairs(draw) -> tuple[list[Prediction], list[Annotation]]:
    """A set of (note, label) pairs, each present on either side or both."""
    n = draw(st.integers(min_value=1, max_value=12))
    pairs = draw(st.lists(st.tuples(note_ids, labels), min_size=n, max_size=n, unique=True))
    preds: list[Prediction] = []
    golds: list[Annotation] = []
    for note_id, label in pairs:
        side = draw(st.sampled_from(["both", "pred_only", "gold_only"]))
        if side in ("both", "pred_only"):
            preds.append(
                Prediction(
                    system="sysA",
                    note_id=note_id,
                    label=label,
                    value=draw(st.booleans()),
                    patient_id="P1",
                )
            )
        if side in ("both", "gold_only"):
            golds.append(
                Annotation(
                    annotator="adjudicated",
                    note_id=note_id,
                    label=label,
                    value=draw(st.booleans()),
                    patient_id="P1",
                    adjudicated=True,
                )
            )
    return preds, golds


class TestAlignmentNeverDropsRecords:
    """SPEC §8.2: alignment accounts for every input record."""

    @FAST
    @given(pred_gold_pairs())
    def test_row_count_is_the_union_of_pairs(self, data):
        preds, golds = data
        aligned = align_predictions(preds, golds, EvalConfig(unit=Unit.NOTE))
        expected = len(
            {(p.note_id, p.label) for p in preds} | {(g.note_id, g.label) for g in golds}
        )
        assert len(aligned) == expected

    @FAST
    @given(pred_gold_pairs())
    def test_every_prediction_appears(self, data):
        preds, golds = data
        aligned = align_predictions(preds, golds, EvalConfig(unit=Unit.NOTE))
        present = set(
            zip(
                aligned.df.loc[aligned.df["pred_present"], "note_id"],
                aligned.df.loc[aligned.df["pred_present"], "label"],
                strict=True,
            )
        )
        assert {(p.note_id, p.label) for p in preds} == present

    @FAST
    @given(pred_gold_pairs())
    def test_every_annotation_appears(self, data):
        preds, golds = data
        aligned = align_predictions(preds, golds, EvalConfig(unit=Unit.NOTE))
        present = set(
            zip(
                aligned.df.loc[aligned.df["gold_present"], "note_id"],
                aligned.df.loc[aligned.df["gold_present"], "label"],
                strict=True,
            )
        )
        assert {(g.note_id, g.label) for g in golds} == present

    @FAST
    @given(pred_gold_pairs())
    def test_scorable_plus_unscored_equals_total(self, data):
        preds, golds = data
        aligned = align_predictions(preds, golds, EvalConfig(unit=Unit.NOTE))
        pred_only = int((aligned.df["pred_present"] & ~aligned.df["gold_present"]).sum())
        gold_only = int((~aligned.df["pred_present"] & aligned.df["gold_present"]).sum())
        assert len(aligned.scorable) + pred_only + gold_only == len(aligned)


class TestAggregationIsIdempotent:
    """SPEC §8.2: aggregating an already-aggregated frame is a no-op."""

    @FAST
    @given(pred_gold_pairs())
    def test_note_unit_idempotent(self, data):
        preds, golds = data
        aligned = align_predictions(preds, golds, EvalConfig(unit=Unit.NOTE))
        once = aggregate_to_windows(aligned, unit=Unit.NOTE)
        twice = aggregate_to_windows(
            AlignedFrame(df=once.df, unit=Unit.NOTE, annotators=aligned.annotators),
            unit=Unit.NOTE,
        )
        assert len(once) == len(twice)
        assert once.df["unit_key"].tolist() == twice.df["unit_key"].tolist()
        assert once.df["pred"].tolist() == twice.df["pred"].tolist()
        assert once.df["gold"].tolist() == twice.df["gold"].tolist()

    @FAST
    @given(pred_gold_pairs())
    def test_patient_unit_idempotent(self, data):
        preds, golds = data
        cfg = EvalConfig(unit=Unit.PATIENT)
        aligned = align_predictions(preds, golds, cfg)
        once = aggregate_to_windows(aligned, unit=Unit.PATIENT)
        twice = aggregate_to_windows(
            AlignedFrame(df=once.df, unit=Unit.PATIENT, annotators=aligned.annotators),
            unit=Unit.PATIENT,
        )
        assert once.df["pred"].tolist() == twice.df["pred"].tolist()


confusion_counts = st.integers(min_value=0, max_value=200)


class TestIntervalsContainThePointEstimate:
    """SPEC §8.2: a CI must bracket its own point estimate."""

    @FAST
    @given(k=confusion_counts, extra=confusion_counts)
    def test_wilson(self, k, extra):
        n = k + extra
        assume(n > 0)
        iv = wilson_ci(k, n)
        assert iv.low - 1e-12 <= iv.point <= iv.high + 1e-12
        assert 0.0 <= iv.low <= iv.high <= 1.0

    @FAST
    @given(
        values=st.lists(
            st.tuples(st.booleans(), st.booleans(), st.integers(min_value=0, max_value=6)),
            min_size=4,
            max_size=40,
        )
    )
    def test_clustered_bootstrap(self, values):
        df = pd.DataFrame(
            {
                "pred": pd.array([v[0] for v in values], dtype="boolean"),
                "gold": pd.array([v[1] for v in values], dtype="boolean"),
                "patient_id": [f"P{v[2]}" for v in values],
            }
        )
        assume(df["patient_id"].nunique() >= 2)
        iv = clustered_bootstrap_ci(
            df, lambda d: accuracy(Confusion.from_frame(d)), n_boot=120, seed=0
        )
        if math.isnan(iv.low):
            return  # degenerate resamples are reported, not fabricated
        # Monte Carlo tolerance: a percentile interval can sit a hair off the
        # point estimate when the statistic is near-constant.
        assert iv.low - 0.02 <= iv.point <= iv.high + 0.02


class TestMetricInvariants:
    @FAST
    @given(tp=confusion_counts, fp=confusion_counts, fn=confusion_counts, tn=confusion_counts)
    def test_kappa_is_bounded(self, tp, fp, fn, tn):
        c = Confusion(tp=tp, fp=fp, fn=fn, tn=tn)
        assume(c.n > 0)
        k = cohen_kappa(c)
        assert -1.0 - 1e-9 <= k <= 1.0 + 1e-9

    @FAST
    @given(tp=confusion_counts, fp=confusion_counts, fn=confusion_counts, tn=confusion_counts)
    def test_rates_are_in_unit_interval_or_nan(self, tp, fp, fn, tn):
        c = Confusion(tp=tp, fp=fp, fn=fn, tn=tn)
        for value in (accuracy(c), f1(c)):
            assert math.isnan(value) or 0.0 <= value <= 1.0

    @FAST
    @given(n=st.integers(min_value=1, max_value=300))
    def test_perfect_agreement_gives_kappa_one(self, n):
        assert cohen_kappa(Confusion(tp=n, fp=0, fn=0, tn=n)) == 1.0


class TestFreezeHashSensitivity:
    """SPEC §8.2: the hash changes if and only if config content changes."""

    @FAST
    @given(overlap=st.floats(min_value=0.0, max_value=1.0))
    def test_changing_a_value_changes_the_hash(self, overlap):
        base = EvalConfig()
        other = base.with_overrides(min_overlap=overlap)
        if abs(overlap - base.matching.span.min_overlap) < 1e-12:
            assert other.content_hash() == base.content_hash()
        else:
            assert other.content_hash() != base.content_hash()

    @FAST
    @given(seed=st.integers(min_value=0, max_value=10_000))
    def test_equal_content_hashes_equally(self, seed):
        a = EvalConfig.from_dict({"bootstrap": {"seed": seed}})
        b = EvalConfig.from_dict({"bootstrap": {"seed": seed}})
        assert a.content_hash() == b.content_hash()

    @FAST
    @given(
        unit=st.sampled_from(["note", "patient", "patient_window"]),
        n=st.integers(min_value=1, max_value=500),
    )
    def test_distinct_configs_hash_distinctly(self, unit, n):
        a = EvalConfig.from_dict({"unit": unit, "bootstrap": {"n": n}})
        b = EvalConfig.from_dict({"unit": unit, "bootstrap": {"n": n + 1}})
        assert a.content_hash() != b.content_hash()


class TestSpanInvariants:
    @FAST
    @given(
        a_start=st.integers(min_value=0, max_value=200),
        a_len=st.integers(min_value=0, max_value=60),
        b_start=st.integers(min_value=0, max_value=200),
        b_len=st.integers(min_value=0, max_value=60),
    )
    def test_jaccard_is_symmetric_and_bounded(self, a_start, a_len, b_start, b_len):
        from clinval.match.spans import jaccard_overlap

        a = Span(start=a_start, end=a_start + a_len)
        b = Span(start=b_start, end=b_start + b_len)
        j1, j2 = jaccard_overlap(a, b), jaccard_overlap(b, a)
        assert j1 == j2
        assert 0.0 <= j1 <= 1.0

    @FAST
    @given(
        start=st.integers(min_value=0, max_value=200), length=st.integers(min_value=1, max_value=60)
    )
    def test_identical_spans_have_jaccard_one(self, start, length):
        from clinval.match.spans import jaccard_overlap

        s = Span(start=start, end=start + length)
        assert jaccard_overlap(s, s) == 1.0
