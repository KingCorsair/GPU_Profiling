# Level 7 — Deeper Risk Modeling

> **Level of progression:** after this level you have either sharpened the quality/compute
> Pareto frontier or proven — with measurements — that the simple Level 2 model was already
> the right choice.
>
> **Optional milestone:** not required for the core milestone. Everything from Level 5
> onward is optional.

Source phase: *Phase 7* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

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

**Next level:** [Level 8 — Production Observability](LEVEL_08_PRODUCTION_OBSERVABILITY.md)