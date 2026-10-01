# Level 5 — Representation Drift

> **Level of progression:** after this level you understand *why* aggressive pruning changes
> answers, at the level of the model's internal representations. This deepens the Level 2
> prediction story with mechanistic evidence.
>
> **Optional milestone:** not required for the core milestone. Everything from Level 5
> onward is optional.

Source phase: *Phase 5* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

## Research question

**Does aggressive pruning substantially alter internal multimodal representations, and does
that change predict answer instability?**

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

It demonstrates genuine **model interpretability / representation analysis**, rather than
simply calling embeddings because they are fashionable.

---

**Next level:** [Level 6 — Token-Selection Stability](LEVEL_06_TOKEN_SELECTION_STABILITY.md)