# Sribhav Plan — Levels of Progression (Index)

This index is the master document for Sribhav's accuracy-side roadmap, split out of
`SRIBHAV_PLAN_REVISED_FULL.md`. Each phase of the original plan is now its own markdown
file, ordered as a ladder of increasing progression. Every file is a level: you complete
it, you can make a claim you could not make before, and the next level builds on it.

- **Core milestone:** Levels 1–4 constitute the required milestone, complete within one
  month. At the end of Level 4 the project is already complete and résumé-worthy.
- **Optional ladder:** Levels 5–10 are optional extensions. Everything from Level 5 onward
  is optional.

---

## Problem Statement

VisPruner is typically evaluated using aggregate task accuracy. However, a small change in
aggregate accuracy can hide much larger per-example instability.

In our current ScienceQA results, aggressive pruning changed roughly one in nine image-based
answers even though the overall accuracy difference remained small. Some correct answers
became wrong, while some wrong answers became correct. This suggests that **net accuracy
alone may not adequately describe the reliability of visual-token pruning at the request
level.**

The central problem is therefore:

> **Can we characterize the per-example instability introduced by visual-token pruning,
> understand what predicts that instability, and use inexpensive runtime signals to
> selectively allocate more visual tokens only to requests that are likely to be harmed by
> aggressive pruning?**

The project has three connected questions.

## Question A — Measurement

**How much instability does VisPruner introduce that aggregate accuracy hides?**

Instead of measuring only:

`accuracy@576 − accuracy@64`

measure individual answer transitions:

* correct → correct
* correct → wrong
* wrong → correct
* wrong → wrong
* unchanged answer
* changed answer

This establishes whether there is actually a deployment-relevant reliability problem.

## Question B — Explanation / Prediction

**Why are some requests sensitive to pruning while others are unaffected, and can that
sensitivity be predicted cheaply?**

Candidate signals include:

* retained attention mass
* attention entropy
* output logit margin
* selected-token spatial distribution
* token-selection stability
* hidden-state / representation drift

The objective is not simply to correlate random metrics with accuracy.

The engineering question is:

> Can information already available during inference tell us whether aggressive pruning is
> risky?

## Question C — Engineering

**Can we exploit that signal to create adaptive token-budget routing?**

Rather than giving every request 576 visual tokens or every request 64 tokens:

1. Start with a low token budget.
2. Estimate pruning risk.
3. Keep the cheap result when risk is low.
4. Retry with 128/288/576 tokens when risk is high.

The final system should attempt to preserve near-baseline accuracy while substantially
reducing average visual-token processing.

---

## The Level Ladder

| Level | File | Source phase | Core? | Claim you can make after completing it |
|---|---|---|---|---|
| 1 | [`LEVEL_01_TRUSTWORTHY_EVALUATION.md`](LEVEL_01_TRUSTWORTHY_EVALUATION.md) | Phase 1 — Trustworthy Evaluation System | ✅ Core | "I can measure per-example answer instability honestly." |
| 2 | [`LEVEL_02_UNSAFE_PRUNING_SIGNALS.md`](LEVEL_02_UNSAFE_PRUNING_SIGNALS.md) | Phase 2 — Identify Unsafe Pruning | ✅ Core | "I can predict which requests pruning will damage." |
| 3 | [`LEVEL_03_ADAPTIVE_TOKEN_ROUTING.md`](LEVEL_03_ADAPTIVE_TOKEN_ROUTING.md) | Phase 3 — Adaptive Token-Budget Routing | ✅ Core | "I can act on that signal to route token budgets." |
| 4 | [`LEVEL_04_VALIDATION_PACKAGING.md`](LEVEL_04_VALIDATION_PACKAGING.md) | Phase 4 — Final Validation & Packaging | ✅ Core | "The mechanism survives validation and is reproducible." **Core milestone DONE.** |
| 5 | [`LEVEL_05_REPRESENTATION_DRIFT.md`](LEVEL_05_REPRESENTATION_DRIFT.md) | Phase 5 — Representation Drift | 🔸 Optional | "I know how pruning changes internal representations." |
| 6 | [`LEVEL_06_TOKEN_SELECTION_STABILITY.md`](LEVEL_06_TOKEN_SELECTION_STABILITY.md) | Phase 6 — Token-Selection Stability | 🔸 Optional | "I know whether the selected visual evidence is stable." |
| 7 | [`LEVEL_07_DEEPER_RISK_MODELING.md`](LEVEL_07_DEEPER_RISK_MODELING.md) | Phase 7 — Deeper Risk Modeling | 🔸 Optional | "A more complex model improves the Pareto frontier — or not." |
| 8 | [`LEVEL_08_PRODUCTION_OBSERVABILITY.md`](LEVEL_08_PRODUCTION_OBSERVABILITY.md) | Phase 8 — Production Observability | 🔸 Optional | "I can observe the deployed router like an ML system." |
| 9 | [`LEVEL_09_LOCAL_AI_GRADER.md`](LEVEL_09_LOCAL_AI_GRADER.md) | Phase 9 — Local AI Grader | 🔸 Optional | "I reject or justify a local grader with evidence." |
| 10 | [`LEVEL_10_GENERALIZATION.md`](LEVEL_10_GENERALIZATION.md) | Phase 10 — Generalization | 🔸 Optional | "I know whether this generalizes beyond the training workload." |

---

## Final Project Narrative

The accuracy side of the project should eventually be explainable as:

> **VisPruner appeared nearly lossless under aggregate accuracy, but we found that the mean
> concealed substantial per-example answer instability. I built a reproducible evaluation
> system to quantify that instability, instrumented the model to identify inexpensive
> runtime signals associated with risky pruning decisions, trained and validated a
> lightweight risk predictor, and used it to implement adaptive visual-token routing that
> selectively increased compute for sensitive requests. We then measured the resulting
> accuracy–compute tradeoff across multimodal workloads.**

## Tool Story

The tool story naturally follows from the problem:

**PyTorch**
→ model inference and instrumentation.

**Hugging Face/Datasets**
→ reproducible multimodal benchmark ingestion.

**Pandas / NumPy / SciPy**
→ rigorous per-example evaluation and statistics.

**MLflow/W&B**
→ reproducible management of many model/pruning/routing experiments.

**scikit-learn**
→ interpretable runtime risk prediction.

**XGBoost/LightGBM**
→ only if nonlinear prediction materially improves routing.

**OpenTelemetry / Prometheus / Grafana**
→ optional production monitoring once adaptive routing exists.

**Ollama/vLLM/Transformers**
→ optional local grader only if cheaper signals prove inadequate.

The tools are therefore consequences of the engineering questions, rather than the purpose
of the project.

---

*Source: [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md) — this index
and the LEVEL_XX files are the same plan, split by level of progression.*