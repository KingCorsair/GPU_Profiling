# Level 6 — Token-Selection Stability

> **Level of progression:** after this level you know whether VisPruner's chosen visual
> evidence is itself stable, and whether unstable selection is what produces unstable
> answers. This connects the pruning algorithm itself to the observed reliability.
>
> **Optional milestone:** not required for the core milestone. Everything from Level 5
> onward is optional.

Source phase: *Phase 6* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

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

**Next level:** [Level 7 — Deeper Risk Modeling](LEVEL_07_DEEPER_RISK_MODELING.md)