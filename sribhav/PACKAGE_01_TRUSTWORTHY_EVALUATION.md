# Level 1 — Build a Trustworthy Evaluation System

> **Level of progression:** after this level you can honestly measure the per-example
> answer instability that VisPruner's aggregate accuracy may be hiding. This is the
> measurement foundation every later level reports against.
>
> **Core milestone:** Level 1 of 4 — part of the required one-month core milestone.

Source phase: *Phase 1* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

## Question answered

**Is the apparent accuracy stability of VisPruner actually hiding substantial per-example
answer instability?**

## Work

First repair the evaluation methodology.

The current heterogeneous evaluation set has known problems that must be resolved before
making claims, including class imbalance and insufficient samples in some categories.

Build a reproducible evaluation framework supporting several token budgets, for example:

* 576
* 288
* 144/128
* 64

For every example record:

* dataset
* task/category
* ground-truth answer
* answer at each token budget
* correctness at each budget
* answer changed?
* correctness transition
* token count
* configuration

Calculate:

* aggregate accuracy
* accuracy delta
* answer-change/churn rate
* correct → wrong rate
* wrong → correct rate
* wrong → wrong rate
* confidence intervals
* McNemar tests where appropriate
* per-category breakdowns

## Tools

### PyTorch

Already part of the model/evaluation path.

**Why it belongs:** running and instrumenting LLaVA/VisPruner inference.

### Hugging Face Datasets

Where appropriate for standardized dataset loading and preprocessing.

**Why it belongs:** reproducible dataset ingestion rather than hand-written ad-hoc loaders.

### Pandas / NumPy

For per-example result analysis.

**Why it belongs:** the core problem requires joining and comparing outputs from multiple
pruning configurations.

### SciPy / statsmodels

For significance testing and confidence intervals.

**Why it belongs:** we need to distinguish real accuracy effects from benchmark noise.

### MLflow or Weights & Biases

Introduce experiment tracking here if the number of configurations warrants it.

Track:

* dataset version
* model version
* pruning budget
* accuracy
* churn
* artifacts
* configuration
* run IDs

**Why it belongs:** comparisons across many datasets/configurations need reproducibility.

## Deliverable

A benchmark report answering:

> How different does VisPruner look when evaluated using per-example stability rather than
> only aggregate accuracy?

---

**Next level:** [Level 2 — Identify What Makes Pruning Unsafe](LEVEL_02_UNSAFE_PRUNING_SIGNALS.md)