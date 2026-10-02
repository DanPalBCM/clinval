# Agreement statistics and confidence intervals

Every metric is computed from a 2×2 table of boolean prediction against boolean
gold, at the evaluation unit.

|  | gold + | gold − |
| --- | --- | --- |
| **pred +** | TP | FP |
| **pred −** | FN | TN |

Rows where either side is null are excluded and counted as `unscored`.

## Three families, and why all three are reported

**Accuracy-style** metrics answer *how often is the system right*, conditioned
on truth or on the prediction:

$$
\text{sensitivity} = \frac{TP}{TP+FN} \qquad
\text{specificity} = \frac{TN}{TN+FP}
$$

$$
\text{PPV} = \frac{TP}{TP+FP} \qquad
\text{NPV} = \frac{TN}{TN+FN} \qquad
\text{accuracy} = \frac{TP+TN}{N}
$$

$$
F_1 = \frac{2TP}{2TP+FP+FN} \qquad
\text{prevalence} = \frac{TP+FN}{N}
$$

**Chance-corrected.** Cohen's κ asks how much better than guessing at the
observed base rate:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

with observed agreement \(p_o = (TP+TN)/N\) and expected agreement

$$
p_e = \frac{TP+FP}{N}\cdot\frac{TP+FN}{N} + \frac{FN+TN}{N}\cdot\frac{FP+TN}{N}
$$

**Prevalence-adjusted.** PABAK asks the same question with the base rate
neutralised, so it depends only on the observed agreement rate:

$$
\text{PABAK} = 2p_o - 1
$$

### Why this matters

A rare phenotype can show 97% accuracy and a κ near zero. Consider
`TP=1, FP=2, FN=2, TN=95`:

| metric | value |
| --- | --- |
| accuracy | 0.960 |
| PABAK | 0.920 |
| **Cohen's κ** | **0.306** |
| prevalence | 0.030 |

Reporting only accuracy is the standard way clinical extraction results get
oversold. clinval computes all three and prints prevalence beside them, so the
divergence is visible rather than something a reader has to suspect.

### Edge cases, handled explicitly

| Situation | κ | Rationale |
| --- | --- | --- |
| every unit TN, or every unit TP | `1.0` | the raters agreed on everything; the formula would divide by zero |
| complete disagreement, one class | `0.0` | no agreement beyond none |
| no units | `nan` | undefined, and reported as such |

An **undefined metric is `nan`, never `0.0`.** A specificity with no gold
negatives is genuinely undefined; printing `0.000` would read as "the system
got every negative wrong", which is a different and false claim. The report
renders these as `n/a`.

κ is checked against `sklearn.metrics.cohen_kappa_score` over 200 randomized
tables in `tests/test_agreement.py`.

## Confidence intervals

The interval method is **chosen per metric and named in every row**, because a
reader comparing a κ interval to a sensitivity interval should know they came
from different machinery.

### Wilson, for proportions of a fixed denominator

Applies to sensitivity, specificity, PPV, NPV, accuracy, and prevalence — each
is \(k\) successes out of a well-defined \(n\).

$$
\frac{1}{1+\frac{z^2}{n}}\left[
\hat{p}+\frac{z^2}{2n}
\pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^2}{4n^2}}
\right]
$$

Preferred over Wald because it stays inside \([0,1]\) and keeps sensible
coverage at small \(n\) and at proportions near 0 or 1 — both routine in
phenotype validation, where a rare label may have five positives. Checked
against `statsmodels.stats.proportion.proportion_confint` to 1e-10.

### Clustered bootstrap, for everything else

κ, PABAK, and F₁ are not single proportions, so they get a percentile bootstrap
that **resamples patients, not rows**.

This is the point of `bootstrap.cluster_by`. Patient-window rows from the same
patient are correlated. Resampling rows treats 400 windows from 100 patients as
400 independent observations and produces intervals that are too narrow. A test
demonstrates the effect directly: on data with strong within-patient
correlation, the clustered interval is measurably wider than the naive one.

```yaml
bootstrap:
  n: 2000
  cluster_by: patient_id
  seed: 0
  alpha: 0.05
```

Given the same seed, frame, and statistic, the result is reproducible.

| Situation | Behaviour |
| --- | --- |
| one cluster | point estimate returned, bounds `nan`, method suffixed `_single_cluster` |
| statistic undefined on most resamples | bounds `nan`, method suffixed `_degenerate` |
| `cluster_by` column absent | falls back to per-row, method `bootstrap_unclustered` |

An absent interval is reported as absent. Collapsing it onto the point estimate
would make a zero-width interval indistinguishable from an unknown one.

Metrics requested together share one set of resamples, so sensitivity and
specificity describe the same resampling distribution.

## Comparing two systems

`paired_bootstrap_difference` resamples clusters and scores **both systems on
the identical set of units in every resample**, so unit difficulty cancels.
Units present for only one system are dropped and the smaller `n` is reported.

```python
report = cv.run_eval(preds, golds, config, index=index)
for row in report.comparisons:
    print(row["system_a"], row["system_b"], row["metric"],
          row["difference"], row["ci_low"], row["ci_high"], row["p_value"])
```

The p-value is the bootstrap proportion of resamples whose difference crosses
zero, doubled and clipped to 1.0. **It is an approximation by construction**,
not an exact test, and the report says so.

## The human ceiling

When you pass multi-annotator records, `inter_annotator_agreement` reports
pairwise κ, PABAK, and observed agreement between annotators on the pairs they
both labelled.

Read your system against that number, not against 1.0. A system at κ 0.70
looks very different when two clinicians managed 0.72 with each other than when
they managed 0.95. On the demo data the annotators agree at κ 0.813 and
`careful_reader` scores 0.912 — above the ceiling, which is itself worth
noticing and interrogating.
