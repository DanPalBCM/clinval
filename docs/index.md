# clinval

**The missing evaluation layer for clinical LLM extraction.**

LLM extraction is easy to demo and hard to trust. Given note-level predictions
from any extractor, clinval answers the questions a reviewer will actually ask:
which patient-level phenotypes are supported, how well does the system agree
with clinician annotators, and *where do the residual errors come from*.

## Where it sits

| | Scores | Knows about |
| --- | --- | --- |
| LLM eval frameworks | prompts and responses | nothing about patients, time windows, or inter-rater agreement |
| Extraction libraries | nothing — they stop at extraction | spans and schemas |
| **clinval** | patient-by-time-window phenotypes | agreement with CIs, error mechanisms, the dev/test protocol |

clinval **never calls an LLM**. It consumes an extractor's output, which is what
makes it extractor-agnostic and its results reproducible.

## Four commitments, each enforced in code

**Extractor-agnostic.** Anything emitting clinval's [JSONL format](formats.md)
can be evaluated.

**[A frozen dev protocol](freeze.md).** Thresholds are tuned on dev, locked
with a SHA-256 hash, and verified before a test-set run. `evaluate` refuses a
config that was edited after freezing.

**[The patient is the unit](concepts.md).** Metrics are computed on
patient-windows with bootstrap intervals clustered by patient, because two
windows from the same patient are not independent observations.

**[Errors are first-class](errors.md).** Every disagreement gets a mechanism
tag with the evidence that produced it, because the point is to learn, not just
to score.

## Start here

- [Getting started](getting-started.md) — install and run the demo
- [Tune, freeze, evaluate](recipes/walkthrough.md) — the protocol end to end
- [Agreement statistics](statistics.md) — the formulas, and which interval
  method applies where
- [Error mechanism taxonomy](errors.md) — the seven tags and the rules that
  assign them

## What it will not do

clinval measures agreement. It does not certify clinical validity, cannot
verify that your dev and test splits really hold different patients, and cannot
tell you whether your gold standard means what you think. The
[limitations section](getting-started.md#limitations) is deliberately blunt.
