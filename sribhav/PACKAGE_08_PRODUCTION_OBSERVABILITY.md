# Level 8 — Production Observability

> **Level of progression:** after this level the adaptive router is instrumented like a real
> deployed ML system, so you can see in traffic what Level 3 only measured offline.
>
> **Optional milestone:** not required for the core milestone. Everything from Level 5
> onward is optional.

Source phase: *Phase 8* of [`../SRIBHAV_PLAN_REVISED_FULL.md`](../SRIBHAV_PLAN_REVISED_FULL.md)

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

**Next level:** [Level 9 — Local AI Grader](LEVEL_09_LOCAL_AI_GRADER.md)