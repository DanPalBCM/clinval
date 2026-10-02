# The freeze protocol

Threshold tuning is the easiest way to overfit a validation study without
noticing. You try a few overlap thresholds, one looks better, you report it —
and the number you report is an in-sample fit dressed up as a validation
result.

clinval makes the protocol mechanical rather than a matter of discipline.

## Three guards

```bash
clinval tune     --config eval.yaml  --pred dev_pred.jsonl  --gold dev_gold.jsonl  --out tuned.yaml
clinval freeze   --config tuned.yaml --out frozen.json
clinval evaluate --frozen frozen.json --pred test_pred.jsonl --gold test_gold.jsonl --out report/
```

**1. `tune` refuses test data.** A config whose `split` is `test` raises
`SplitGuardError`. There is an `allow_test` escape hatch in the Python API that
exists so the test suite can exercise the guard; the CLI never sets it.

**2. `freeze` records a hash.** SHA-256 over the canonical JSON form of the
config body, plus a timestamp, the clinval version, and the git commit when
available. The config's `split` is forced to `test`, since a config is frozen
in order to be run out of sample.

**3. `evaluate` verifies the hash.** If the config body changed after freezing,
it refuses:

```
frozen config failed verification: the config body was modified after freezing.
  recorded hash: 05100d073c1cf06904676ca5bf6858b998af191c0149341913644b76f4ff2ec4
  actual hash:   eaed509277aa24797c41cc77439cf358010734dde67454709e36693353557ca7
Re-run `clinval freeze` if the change was intentional.
```

Editing the recorded hash instead is caught the same way, because the check
recomputes the hash from the config body.

## Canonical hashing

Two configs that differ only in key order, or in `6M` versus
`Duration(n=6, unit="M")`, hash **identically**. Otherwise the check would fire
on a no-op reformatting and people would learn to ignore it. A property test
asserts the hash changes if and only if content changes.

The hash covers the config **only** — not the timestamp or the commit — so
re-freezing an identical config is idempotent in hash terms.

## The unfrozen path

`clinval run` does a single-shot run without the protocol. It refuses a config
marked `test` outright, and refuses a `dev` config unless you pass
`--i-understand-this-is-dev`. The flag is deliberately unwieldy: an unfrozen
number is not a validation result, and the command line should say so.

Any report produced without a frozen config carries a prominent banner:

> **These numbers are not an out-of-sample result.** No frozen config was
> supplied, so any threshold in this config may have been chosen on the same
> data it is evaluated on.

## Declaring the grid

Only parameters listed in `freeze.grid` may be tuned, and the grid is declared
*before* tuning rather than chosen after seeing results.

```yaml
freeze:
  objective: cohen_kappa
  grid:
    min_overlap: [0.2, 0.35, 0.5, 0.65, 0.8]
```

Tunable: `min_overlap`, `fuzzy_threshold`, `span_mode`, `require_span`,
`consensus`. Anything else is a config error at load time, not a `KeyError`
halfway through a bootstrap.

With several systems and no `--system`, the objective is the **mean** of the
per-system scores. That tunes the harness to work across systems rather than to
flatter one of them, and the choice is recorded in the tuning note.

## Two hazards the tuning record surfaces

### A flat grid

`TuningResult.is_flat` is true when every candidate scored within 0.005 of the
winner. Then the parameter did not matter on dev, and the chosen value is
effectively arbitrary:

> The objective varies by less than 0.005 across the whole grid, so the chosen
> value is effectively arbitrary on this dev set. Report it as a default, not
> as a tuned finding.

This is how you discover that `min_overlap` does nothing unless `require_span`
is set — see [decisions](decisions.md) D-006.

### A winner that answers fewer questions

Tuning `consensus` exposes a trap. Under `majority` with two systems a 1–1
split is **undecided**, undecided units are excluded from the confusion table,
and the rule therefore posts a near-perfect score by declining to answer the
hard cases. On the demo data:

| consensus | κ | decided |
| --- | --- | --- |
| majority | 1.000 | 70.5% |
| unanimous | 0.872 | 100% |
| any | 0.596 | 100% |

clinval does not invent a score for undecided units — routing a case to a human
is a legitimate outcome, not an error. Instead every candidate records
`decided_fraction`, it is printed as a column, and the warning is explicit:

> The winning setting `{'consensus': 'majority'}` decided only 70.5% of units,
> while `{'consensus': 'unanimous'}` decided 100.0% at cohen_kappa=0.8718. Part
> of the winner's advantage is that undecided units are excluded from its score
> rather than counted against it. Choose on review burden as well as on the
> metric.

See [decisions](decisions.md) D-007.

## The whole tuning record is kept

`frozen.json` carries every candidate and its score, not just the winner, so a
reviewer can see whether the objective was flat and judge the choice for
themselves. It shows up as a "Tuning record" section in the report.

## What the protocol does not guarantee

**It cannot tell whether your dev and test files really hold different
patients.** That is your study design, not clinval's. `check_split_disjoint`
makes it checkable rather than assumed:

```bash
clinval evaluate --frozen frozen.json ... --dev-gold dev_gold.jsonl
```

With `--dev-gold`, overlap is reported in the report and the command exits 2.
In the Python API:

```python
report.split_check = cv.check_split_disjoint(dev_gold, test_gold)
```

A shared patient means the tuned threshold was chosen on a patient it was then
tested on, and the out-of-sample claim is void — while the frozen hash still
verifies perfectly. The hash attests to the config, not to the design.

Split **by patient**, never by note. `SyntheticDataset.split_by_patient` does
this correctly and explains why in its docstring.
