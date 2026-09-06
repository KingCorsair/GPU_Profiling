Short answer: **we don't need one for the benchmark — and that's the right conclusion to be forced to state clearly.** An on-prem AI grader is only justified *if and when* the runtime quality gate is a real engineering goal, and even then it competes against cheaper non-LLM signals and may lose. SRIBHAV_WORK.md's Phase 6 asks exactly this question, so let me answer it rigorously rather than defensively.

---

## 1. The benchmark does not need an on-prem AI grader

For scoring a 3-budget sweep, we have two things a runtime grader never does:

- **Ground-truth labels.** For ScienceQA and the hetero set's binary/short answers, the labels are the best judge that exists — better than any model, cloud or local.
- **Deterministic scoring for free-form answers.** Exact match → BERTScore → NLI. These already run locally, are cheap, deterministic, and version-pinned. They're *metrics*, not AI judges — they need no justification because they're just evaluation.

Where those metrics fall short (free-text semantic equivalence), the honest move for an **offline** benchmark is to use the *strongest available judge once* — i.e., Claude — as a **reference** to validate that our local scorer agrees with it (rule 13: scorer validation). A 3B local LLM is a strictly *worse* judge of semantic nuance than Claude, and we'd only send the outputs once, so cost is trivial. **On-prem "AI grading" loses to Claude on quality, on cost, and on effort for the offline use case.** If the gate never gets built, the correct architecture is: deterministic local metrics + one-off Claude reference validation, and nothing else.

---

## 2. The runtime quality gate is the *only* thing that justifies on-prem — and it does so decisively

A gate runs **continuously on the request path**, and that changes every constraint:

| Gate requirement | Cloud API (Claude) | On-prem |
|---|---|---|
| Runs on every request, forever | Recurs on cost per request | One-time fixed cost | 
| Zero/stable latency budget | +network round-trip, variable | Deterministic, local | 
| Doesn't fail when network/API degrades | A gate that depends on a third party isn't a reliability feature | Works offline | 
| No data egress | Test images/questions leave the pod | Stays on the pod | 
| Deterministic, pinned thresholds | API models change versions under you | Reproducible, pinned | 

So the justification is **structural, not aesthetic**: "local is cooler" is not an argument, but "the quality gate is a thing we want to exist, and its requirements are local-only" is one. That's the exact Phase 6 distinction, and it's why the SRIRBHAV_WORK.md doc insists the local grader must *not* exist merely because local is cool.

---

## 3. Even then: don't assume the "AI" part survives

The runtime gate needs a **quality signal**, not necessarily an LLM. And Phase 10's limitation is the crux: an answer-only judge (local or cloud) **cannot know factual correctness**. So the LLM-judge is only one candidate for a signal that is actually honest about what it knows:

| Signal | What it's really measuring | Cost |
|---|---|---|
| Two-budget disagreement (run at n144 and n576, compare) | *Pruning-induced instability* — a change in the answer where there shouldn't be one | One cheap cheap extra forward pass, no model | 
| Token-level model confidence | Weak self-signal, notoriously miscalibrated | Free, extracted from existing run |
| Small local LLM plausibility judge | "Does this answer *look like* a competent answer" — not correctness | GPU/CPU + latency |
| Multimodal local judge (image + Q + A) | Closest to real correctness, still not certain | Expensive, likely kills the efficiency benefit |

The plan should **measure the cheap signals first** and only reach for a small local LLM if the cheap ones demonstrably add no signal. Otherwise we've built exactly the thing Phase 10 warns against: an AI judge that *appears* to grade correctness but doesn't.

---

## 4. What this changes in the plan

The AI-grader phase becomes **conditional and properly gated**:

1. **First decide the gate is real scope** (SRIBHAV_WORK calls it "a possible engineering extension"). If not — done, no on-prem grader, offline benchmark uses deterministic scorers + one-off Claude reference validation.
2. **Benchmark first (steps 1–8)** exactly as planned — the gate idea is meaningless without a trustworthy accuracy axis.
3. **Prototype the *cheapest* runtime signals first**: two-budget disagreement, then confidence. Measure whether they detect the failure classes from the error taxonomy (step 12).
4. **Only if those fail** do we compare small local LLMs (Ollama/llama3.2, already prototyped in `sribhav/ollama_as_a_judge.py`), and only against the criteria from Phase 5 — with the mandate to pick between "primary + fallback" or **"no LLM needed"** as a legitimate outcome.
5. Whatever is adopted gets pinned in the **Dockerfile and image rebuild** (Ollama isn't in `requirements.txt` today), and its latency is measured in its own Phase-11 bucket, never fused into VisPruner inference time.

So the honest summary: **on-prem AI grader = justified only as the engine of a runtime quality gate; the benchmark doesn't need it; and even as gate engine, it must compete with — and prove itself against — cheap deterministic signals.** If it loses, that's a finding, not a failure — and it's precisely the kind of negative result this project is designed to produce honestly.