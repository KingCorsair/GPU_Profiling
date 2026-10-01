# Problem Statement

VisPruner is typically evaluated using aggregate task accuracy. However, a small change in aggregate accuracy can hide much larger per-example instability.

In our current ScienceQA results, aggressive pruning changed roughly one in nine image-based answers even though the overall accuracy difference remained small. Some correct answers became wrong, while some wrong answers became correct. This suggests that **net accuracy alone may not adequately describe the reliability of visual-token pruning at the request level.**

The central problem is therefore:

> **Can we characterize the per-example instability introduced by visual-token pruning, understand what predicts that instability, and use inexpensive runtime signals to selectively allocate more visual tokens only to requests that are likely to be harmed by aggressive pruning?**

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

**Why are some requests sensitive to pruning while others are unaffected, and can that sensitivity be predicted cheaply?**

Candidate signals include:

* retained attention mass
* attention entropy
* output logit margin
* selected-token spatial distribution
* token-selection stability
* hidden-state / representation drift

The objective is not simply to correlate random metrics with accuracy.

The engineering question is:

> Can information already available during inference tell us whether aggressive pruning is risky?

## Question C — Engineering

**Can we exploit that signal to create adaptive token-budget routing?**

Rather than giving every request 576 visual tokens or every request 64 tokens:

1. Start with a low token budget.
2. Estimate pruning risk.
3. Keep the cheap result when risk is low.
4. Retry with 128/288/576 tokens when risk is high.

The final system should attempt to preserve near-baseline accuracy while substantially reducing average visual-token processing.

---

# Core Milestone — Complete Within One Month

The first four phases constitute the required milestone.

At the end of Phase 4, the project should already be complete and résumé-worthy.

---

# Phase 1 — Build a Trustworthy Evaluation System

## Question answered

**Is the apparent accuracy stability of VisPruner actually hiding substantial per-example answer instability?**

## Work

First repair the evaluation methodology.

The current heterogeneous evaluation set has known problems that must be resolved before making claims, including class imbalance and insufficient samples in some categories.

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

**Why it belongs:** the core problem requires joining and comparing outputs from multiple pruning configurations.

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

> How different does VisPruner look when evaluated using per-example stability rather than only aggregate accuracy?

---

# Phase 2 — Identify What Makes Pruning Unsafe

## Question answered

**Can we explain or predict which requests will become unstable under aggressive pruning?**

Instrument the inference path.

For every example collect inexpensive internal signals such as:

### Retained attention mass

$$
R=
\frac{\sum_{i \in selected} a_i}
{\sum_i a_i}
$$

Interpretation:

If the selected tokens contain almost all the model's visual attention, aggressive pruning may be safe.

If the attention distribution is diffuse and 64 selected tokens capture relatively little attention mass, the request may be more sensitive.

### Attention entropy

Measure how concentrated or diffuse the visual attention distribution is.

### Output confidence

For example:

* top-1 versus top-2 logit margin
* entropy of answer-token probabilities

### Spatial properties

Measure whether selected visual tokens are:

* tightly clustered
* distributed throughout the image
* concentrated in one region

### Token-selection stability

Compare which tokens survive at different budgets.

## Create the prediction target

For each example:

```text
target = 1 if aggressive pruning produces an unsafe change
target = 0 otherwise
```

Initially define several possible targets and compare them:

### Instability target

Did the answer change relative to 576?

### Regression target

Did a correct 576-token answer become incorrect?

The second target is more directly useful but requires ground truth during training/evaluation.

## Modeling

Start simple.

### Logistic regression

Use as an interpretable baseline.

### Decision tree / Random Forest

Useful for nonlinear thresholds and feature importance.

### XGBoost / LightGBM

Only introduce if simpler models leave meaningful predictive performance on the table.

Do not use XGBoost merely to add XGBoost to the project.

The question is:

> Does a nonlinear model materially improve our ability to predict unsafe pruning?

## Tools

* **PyTorch hooks / model instrumentation**
* **scikit-learn**
* **XGBoost or LightGBM**, conditionally
* **MLflow/W&B** for experiment comparison
* **SHAP**, optionally, if the stronger model needs interpretability

## Metrics

Measure:

* ROC-AUC
* precision
* recall
* PR-AUC
* false-negative rate
* calibration
* performance by task type

False negatives matter particularly strongly:

> A false negative means the router considered pruning safe when it actually damaged the answer.

## Deliverable

A result answering:

> Can VisPruner cheaply predict its own risky pruning decisions?

This phase may also produce a valuable negative result.

If retained attention mass, entropy, confidence and other internal signals have almost no predictive ability, report that.

Do not force the hypothesis to succeed.

---

# Phase 3 — Build Adaptive Token-Budget Routing

## Question answered

**Can prediction of pruning risk be converted into a useful inference policy?**

Build an adaptive inference controller.

Conceptually:

```text
Image + question
       ↓
Run aggressive VisPruner
       ↓
Compute cheap risk signal
       ↓
       ├── LOW RISK → return result
       │
       └── HIGH RISK
              ↓
       increase visual-token budget
              ↓
        rerun / continue
              ↓
         return result
```

Potential policies:

### Policy A

Always 576 tokens.

This is the quality baseline.

### Policy B

Always 64 tokens.

This is the aggressive efficiency baseline.

### Policy C

64 → 128 fallback.

### Policy D

64 → 288 fallback.

### Policy E

64 → 128 → 288/576 multi-stage routing.

Do not assume that the most complicated router wins.

Compare policies empirically.

## Metrics

Measure:

* final accuracy
* correct → wrong rate
* fallback percentage
* average visual tokens/request
* percentage reduction in tokens
* percentage of lost accuracy recovered
* average latency
* p50 latency
* p95 latency
* additional router overhead

Coordinate latency measurements with the GPU-profiling side of the project so evaluation overhead is reported separately.

## Tools

### PyTorch

Inference/router implementation.

### scikit-learn / XGBoost

Risk prediction.

### MLflow/W&B

Threshold and policy experiments.

### YAML/Hydra or simple configuration files

If the number of experiment combinations warrants configuration management.

Do not introduce Hydra unless configuration complexity actually becomes a problem.

## Deliverable

A table such as:

| Policy    | Accuracy | Avg tokens | Fallback | Latency |
| --------- | -------: | ---------: | -------: | ------: |
| 576 fixed |        X |        576 |        — |       X |
| 64 fixed  |        X |         64 |       0% |       X |
| adaptive  |        X |          X |       X% |       X |

The primary success condition is something resembling:

> Near-baseline quality at substantially below the baseline token budget.

The existing ScienceQA simulation suggests adaptive routing may be promising, but the proper experiment must establish whether that result generalizes.

---

# Phase 4 — Final Validation and ML-System Packaging

## Question answered

**Does the mechanism survive a proper evaluation and can another engineer reproduce it?**

Run the final benchmark on the repaired evaluation suite.

Produce:

### Evaluation artifacts

* accuracy-vs-token-budget curves
* churn-vs-token-budget curves
* category-specific failure analysis
* predictor ROC/PR curves
* calibration plots
* adaptive-routing tradeoff plots
* latency/token/accuracy frontier

### Error analysis

Study representative:

* correct → wrong examples
* wrong → correct examples
* router false negatives
* router false positives
* cases immune to pruning
* cases highly sensitive to pruning

### Experiment reproducibility

Every important result should have:

* config
* run ID
* dataset version
* model version
* raw results
* analysis script

## Tools

* **MLflow/W&B**
* **Pandas**
* **Matplotlib**
* **PyTorch**
* **Git/GitHub**
* optionally **Docker** if reproducible deployment requires it

Docker should only be used if environment reproducibility is genuinely an issue.

## Core Month-1 Completion Criterion

At the end of this phase, Sribhav should be able to answer four questions with evidence:

1. **How much answer instability does VisPruner introduce?**
2. **Where does that instability occur?**
3. **Can cheap inference-time signals predict it?**
4. **Can adaptive routing protect accuracy without surrendering most of the pruning benefit?**

At this point, the core milestone is DONE.

Everything below is optional.

---

# Optional Phase 5 — Representation Drift

## Research question

**Does aggressive pruning substantially alter internal multimodal representations, and does that change predict answer instability?**

Extract hidden states under:

* 576 tokens
* 288 tokens
* 128 tokens
* 64 tokens

Compare representations using:

* cosine similarity
* L2 distance
* layer-wise similarity
* CKA if justified

Then correlate representation drift with answer churn/regression.

## Tools

* PyTorch
* NumPy
* scikit-learn
* optionally CKA tooling
* MLflow/W&B

## Why this is résumé-relevant

It demonstrates genuine **model interpretability / representation analysis**, rather than simply calling embeddings because they are fashionable.

---

# Optional Phase 6 — Token-Selection Stability

## Research question

**How stable is VisPruner's selected visual evidence?**

Compare token sets across:

* budgets
* precision modes
* repeated runs
* model variants

Metrics could include:

* Jaccard similarity
* rank correlation
* spatial overlap
* attention-mass overlap

Then ask:

> Does unstable token selection correspond to unstable answers?

## Tools

* PyTorch
* NumPy/Pandas
* visualization tooling

This connects the internal pruning algorithm directly to observed reliability.

---

# Optional Phase 7 — Deeper Risk Modeling

If simple models show signal but are insufficient, test:

* XGBoost
* LightGBM
* small MLP
* calibrated ensemble
* cost-sensitive classification

The objective becomes:

> Improve the quality/compute Pareto frontier without adding meaningful runtime cost.

Measure the **incremental value** of each more complex model.

If logistic regression performs equally well, use logistic regression.

That itself is a good engineering finding.

---

# Optional Phase 8 — Production Observability

Once adaptive routing actually exists, instrument it like a deployed ML system.

For every request record:

```text
model
pruning budget
attention statistics
risk score
routing decision
fallback?
final token count
latency
```

During offline evaluation also attach correctness.

Build views answering:

* What percentage of traffic triggers fallback?
* Which workloads are most risky?
* Has risk-score distribution shifted?
* How much compute is fallback consuming?
* Is the router's behavior changing across datasets?

## Tools

Possible tools:

* **OpenTelemetry** for structured runtime instrumentation
* **Prometheus** for metrics
* **Grafana** for operational dashboards

or a simpler ML-focused setup with MLflow/W&B if that solves the actual requirement.

Do not add an observability stack until there is a deployed mechanism worth observing.

---

# Optional Phase 9 — Local AI Grader

Only revisit the grader if the cheap signals from the core milestone are inadequate.

The problem then becomes:

> Cheap internal signals cannot reliably distinguish safe and unsafe pruning. Can an additional lightweight evaluator improve routing enough to justify its compute overhead?

At that point compare:

### Internal-signal router

versus

### Local AI-grader router

versus

### Strong external grader as an experimental upper bound

The local grader must justify its cost.

If it needs a second expensive multimodal inference and eliminates the savings from pruning, reject the idea.

Potential tools would only be selected after determining the required grader architecture:

* Ollama
* vLLM
* Hugging Face Transformers
* quantization such as bitsandbytes/AWQ if appropriate

Again:

**The project should not start with “we want to use Ollama.”**

It should reach Ollama only if the experiment gives us a reason to run a local model.

---

# Optional Phase 10 — Generalization

After the mechanism works, test:

* additional VQA datasets
* different LLaVA variants
* different VisPruner configurations
* potentially another visual-token reduction technique

The question is no longer:

> Does VisPruner work on more datasets?

The stronger question is:

> **Does the instability predictor and adaptive-routing mechanism generalize beyond the model/workload on which it was developed?**

That is a legitimate generalization experiment.

---

# Final Project Narrative

The accuracy side of the project should eventually be explainable as:

> **VisPruner appeared nearly lossless under aggregate accuracy, but we found that the mean concealed substantial per-example answer instability. I built a reproducible evaluation system to quantify that instability, instrumented the model to identify inexpensive runtime signals associated with risky pruning decisions, trained and validated a lightweight risk predictor, and used it to implement adaptive visual-token routing that selectively increased compute for sensitive requests. We then measured the resulting accuracy–compute tradeoff across multimodal workloads.**

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

The tools are therefore consequences of the engineering questions, rather than the purpose of the project.
