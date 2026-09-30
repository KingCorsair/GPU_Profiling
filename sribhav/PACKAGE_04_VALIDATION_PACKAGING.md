# Level 4 — Final Validation and ML-System Packaging

> **Level of progression:** after this level the core milestone is DONE — the mechanism has
> survived a proper evaluation and another engineer can reproduce it.
>
> **Core milestone:** Level 4 of 4 — completes the required one-month core milestone.

Source phase: *Phase 4* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

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
4. **Can adaptive routing protect accuracy without surrendering most of the pruning
   benefit?**

**At this point, the core milestone is DONE. Everything below is optional.**

---

**Next level:** [Level 5 — Representation Drift](LEVEL_05_REPRESENTATION_DRIFT.md)