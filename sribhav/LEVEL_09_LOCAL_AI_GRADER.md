# Level 9 — Local AI Grader

> **Level of progression:** after this level you know — with evidence — whether a lightweight
> local grader earns its compute as a safety net, or whether the cheap internal signals from
> Levels 2–3 were already enough. The grader is a fallback, not a starting point.
>
> **Optional milestone:** not required for the core milestone. Everything from Level 5
> onward is optional.

Source phase: *Phase 9* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

Only revisit the grader if the cheap signals from the core milestone are inadequate.

The problem then becomes:

> Cheap internal signals cannot reliably distinguish safe and unsafe pruning. Can an
> additional lightweight evaluator improve routing enough to justify its compute overhead?

At that point compare:

### Internal-signal router

versus

### Local AI-grader router

versus

### Strong external grader as an experimental upper bound

The local grader must justify its cost.

If it needs a second expensive multimodal inference and eliminates the savings from pruning,
reject the idea.

Potential tools would only be selected after determining the required grader architecture:

* Ollama
* vLLM
* Hugging Face Transformers
* quantization such as bitsandbytes/AWQ if appropriate

Again:

**The project should not start with "we want to use Ollama."**

It should reach Ollama only if the experiment gives us a reason to run a local model.

---

**Next level:** [Level 10 — Generalization](LEVEL_10_GENERALIZATION.md)