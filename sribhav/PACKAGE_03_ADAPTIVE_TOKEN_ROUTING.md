# Level 3 — Build Adaptive Token-Budget Routing

> **Level of progression:** after this level you have converted risk prediction into a usable
> inference policy — a controller that spends tokens only where pruning is risky. This turns
> the Level 2 prediction into an engineering action.
>
> **Core milestone:** Level 3 of 4 — part of the required one-month core milestone.

Source phase: *Phase 3* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

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

Coordinate latency measurements with the GPU-profiling side of the project so evaluation
overhead is reported separately.

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

The existing ScienceQA simulation suggests adaptive routing may be promising, but the proper
experiment must establish whether that result generalizes.

---

**Next level:** [Level 4 — Final Validation and ML-System Packaging](LEVEL_04_VALIDATION_PACKAGING.md)