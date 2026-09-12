# Level 2 — Identify What Makes Pruning Unsafe

> **Level of progression:** after this level you can explain or predict which requests will
> become unstable under aggressive pruning, using signals that are cheap to collect. This
> turns the measurement of Level 1 into a prediction.
>
> **Core milestone:** Level 2 of 4 — part of the required one-month core milestone.

Source phase: *Phase 2* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

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

If the selected tokens contain almost all the model's visual attention, aggressive pruning
may be safe.

If the attention distribution is diffuse and 64 selected tokens capture relatively little
attention mass, the request may be more sensitive.

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

The second target is more directly useful but requires ground truth during
training/evaluation.

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

> A false negative means the router considered pruning safe when it actually damaged the
> answer.

## Deliverable

A result answering:

> Can VisPruner cheaply predict its own risky pruning decisions?

This phase may also produce a valuable negative result.

If retained attention mass, entropy, confidence and other internal signals have almost no
predictive ability, report that.

Do not force the hypothesis to succeed.

---

**Next level:** [Level 3 — Build Adaptive Token-Budget Routing](LEVEL_03_ADAPTIVE_TOKEN_ROUTING.md)