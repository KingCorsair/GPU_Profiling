# The plan in full — ScienceQA hand-check + pod smoke-test run-book

**Context recap.** An earlier 20-sample ScienceQA hand-check was already flagged in `sribhav/work.md` as the open first step. Evidence I've gathered from the repo:

- **Ground truth** lives in two places: `ScienceQA/data/scienceqa/problems.json` (21,208 problems; `question`, `choices`, `answer` index, `image`; the 4,241 eval rows all resolve) and `eval/scienceqa/llava_test_CQM-A.json` (the exact prompts the model saw, with the official `gpt` letter answer).
- **Model outputs** already exist for every budget at r=0.5: `eval/scienceqa/answers/llava_test_CQM-A/llava-v1.5-7b/n_{576,288,144,128,64}/r_0.5.jsonl` (4,241 rows each, fields `question_id / prompt / text / answer_id / model_id / metadata`).
- **Existing scorer** `vis_pruner_copy/llava/eval/eval_science_qa.py` parses three ways: raw letter ("B"), `"X. …"`, or `"The answer is (X)."`; everything else is `FAILED`. That's the exact-match logic the hand-check will pit against your judgment — not a new scorer.
- **Current numbers** (already scored): n576 = **70.24%**, n288 = 69.51, n144 = 69.68, n128 = 69.98, n64 = 69.84 — near-flat, itself a finding.
- **Where exact match actually bites** (n576 preds): 3,960/4,241 are clean single letters; ~248 are `"B. …"`; the divergent strata are **11 bare `"A."`**, **~26 explanatory sentences** ("To compare the motion of the three ships, we need to determine…"), and **2 embedded-letter sentences**. Those ~39 rows are where scorer-vs-human disagreement lives, and they're the preview of what free-text scoring will face on the heterogeneous set.
- **Local constraint:** no ScienceQA image files exist on this Mac (only `vispruner_eval_dataset/images` for the heterogeneous set). So the hand-check here is text-led; image-dependent rows (maps, OCR) get flagged for a visual pass on the pod.

---

## Deliverable A — ScienceQA 20-sample hand-check (`sribhav/`, no GPU needed)

### A.1 Design decisions (the reasoning)

1. **Sample 20 with stratified oversampling, not uniform random.** This tool's goal is *discovery of where exact match gets it wrong*, not an unbiased accuracy estimate. So: **all 11 bare-`"A."` rows** (tiny stratum, exhaust it) + **7 of the ~26 long-form rows** (seeded random) + **2 embedded-letter rows** + **5 randomly drawn clean single-letter rows** (calibration: confirms the scorer agrees with you on the easy 99%). The sampled JSONL's header records the stratum composition so we never mislabel the sample as a 4,241-representative slice.
2. **Judge = you, not a model.** The tool only *surfaces*; it never decides. Per project rules I will not write scoring logic; I'll build the comparison scaffolding, and the correct/incorrect judgment column is yours to fill.
3. **Show both the file `text` and the parsed letter**, so a disagreement can be attributed to *parse failure* vs *genuinely wrong answer*.
4. **Reuse the exact existing parser** from `eval_science_qa.py`, reproduced (not importing the repo module) so the hand-check measures the same logic that produced the published numbers.
5. **No new dependencies.** Cohen's kappa implemented inline (sklearn isn't installed in your local Python; it's pinned only on the pod image). Everything runs on stdlib + the already-present `json`.

### A.2 The three small scripts (all in `sribhav/`)

**`sqa_handcheck_build.py`** — build the 20-row set.
- Inputs: `eval/scienceqa/llava_test_CQM-A.json` (prompts + official letter), `ScienceQA/data/scienceqa/problems.json` (ground-truth index, choices, image filename), `eval/scienceqa/answers/llava_test_CQM-A/llava-v1.5-7b/n_576/r_0.5.jsonl` (model preds).
- Reproduces the parser; tags each row `form ∈ {clean, dotted, bare_A_dot, long_form, embedded}` and `image_dependent` from the question text ({map, figure, diagram, chart, read, identify on image…} heuristics — heuristic only, editable).
- Applies the stratified draw above with `--seed` (default fixed so it's reproducible; `--count`, `--answers-file`, and `--seed` flags for re-runs on other budgets).
- Writes `sribhav/outputs/sqa_handcheck_n576.jsonl`, one object per row with fields:
  `question_id, prompt, choices, gt_index, gt_letter, model_text, parsed_letter, exact_match (bool), form, image_dependent, image (path or null), human_verdict (blank), human_notes (blank)`.
- Prints a compact summary: stratum counts drawn, how many exact-match rows are `FAILED`.

**`sqa_handcheck_render.py`** — render `sqa_handcheck_n576.html` (self-contained, no server):
- One card per row: question+choices, model's raw text quoted, ground-truth letter, a chip for the exact-match verdict, and two controls — a **Correct/Incorrect** toggle and a **notes** text box.
- A toolbar at top: progress (n/20), "lock & export" button that writes your verdicts back into the JSONL (`human_verdict`, `human_notes`).
- Image shown **when a local file exists**; otherwise a placeholder chip reads `[image not available locally — check on pod if image_dependent]`.

**`sqa_handcheck_compare.py`** — analyze the filled set.
- Loads the JSONL; computes: agreement table (counts of both-correct, both-incorrect, and both disagreement directions), % agreement, **Cohen's kappa inline**, disagreeing rows only, and a final "what exact match got wrong" summary split by `form` (e.g. "3 of 11 bare-A rows were actually the right letter; 0 of 5 clean rows disagreed").
- Exits 0 always (it's a report), prints to stdout, optionally `--json` for a machine-readable copy.

**Usage flow:** `python3 sribhav/sqa_handcheck_build.py` → open the HTML → judge + export → `python3 sribhav/sqa_handcheck_compare.py`.

### A.3 What evidence A gives us

- Whether exact-match scoring is trustworthy on the clean 4000-row bulk (expect near-100% agreement on the 5 calibration rows) and **where precisely it fails** (the parse-failure strata) — with an exact list of the rows.
- Whether the long-form/`"A."` preds contain recoverable letters (→ a safe extractor improvement), or genuinely scramble (→ real quality loss that will look far worse in free-text).
- A validated floor for the strategy that then transfers to the heterogeneous set's free-text scoring (`question_type`-keyed max-over-references), per H2/H3.

---

## Deliverable B — Pod smoke-test run-book (`sribhav/docs/pod_smoke_runbook.md`)

A copy-paste doc for the RunPod pod (no new code; the runner `model_vqa_heterogeneous.py` already exists, untested).

1. **Preflight:** `nvidia-smi`; `cd /workspace/GPU_Profiling && git pull`; `uv pip sync --system requirements.txt`.
2. **Checkout sanity:** checkpoint present at `vis_pruner_copy/checkpoints/llava-v1.5-7b`; dev.json present.
3. **Smoke #1 (unpruned baseline, 5 rows):**
   ```bash
   cd /workspace/GPU_Profiling/vis_pruner_copy
   python llava/eval/model_vqa_heterogeneous.py \
     --question-file ../4_vispruner_eval_dataset/dev.json
   ```
   *(actual paths verified in the script defaults: `vispruner_eval_dataset` sits next to `llava/`, so run from `vis_pruner_copy/` and the default paths resolve; template will contain the exact verified commands.)*
4. **Success criteria checklist** (each is a pass/fail line in the run-book):
   - exit 0 and no CUDA OOM;
   - `/tmp/smoke_het.jsonl` has 5 lines, each containing all of `question_id, category, source_dataset, question_type, answers, image, text` with non-empty `text`;
   - timing sidecar JSON written with `question_count/model_load_s/generation_s`, and model-load time separated (rules 1–4 upheld);
   - `--limit` respected, shuffle=False, deterministic order.
5. **Smoke #2 (pruned path):** same command with `--visual_token_num 144 --important_ratio 0.5 --answers-file /tmp/smoke_het_n144.jsonl --limit 5`. Verifies the real-removal knob actually changes the tensor path.
6. **Cleanup and hand-back:** remove `/tmp` files; note the command + commit hash at the top of the run-book (rule 7).

**Why smoke before any full sweep:** the scaffolded runner was never run on a GPU; a bug here costs 5 minutes, not a 40-minute rerun (rule 19). It also double-checks that **576 is genuinely unpruned** (masks all-True) before the budgets own any claim.

---

## Sequencing, effort, evidence

| # | Step | Effort | Evidence it yields | Blocks |
|---|---|---|---|---|
| 1 | Build A scripts + generate the 20-row HTML | small | Disagreement list, exact-match error pattern | — |
| 2 | You fill the 20 judgments + I discuss disagreements with you | small (your time) | Scorer trust decision | 4, 5 |
| 3 | Write Pod run-book; (on pod) run Smokes 1+2 | small | Runner verified, 576=unpruned confirmed | 5, 6 |
| 4 | Design the `question_type`-keyed scorer (your call; I review) | medium | Correct free-text metric | 6 |
| 5 | 3-budget dev sweep (n576/288/144, r=0.5), 90 questions | medium (pod) | Per-category accuracy-vs-tokens curve | 7 |
| 6 | Random-answer baseline on the heterogeneous outputs (rule 17) | small | Scorer discrimination sanity | 7 |
| 7 | Second-scorer agreement on a sample (rule 13) | small | Scorer reliability | 8 |
| 8 | Per-category accuracy report (rule 14) | small | R6 result material | — |

**Boundaries I'm holding to:** nothing touches `test.json` (rule 11 — all A/B work is dev-only); the hand-check and smoke are in `sribhav/` + `sribhav/docs/` only, no code in Amay's or Rithvik's slices; the scoring *judgment* and *conclusion* stay yours — the tooling only surfaces the comparison; the final reportable numbers still flow through Rithvik's harness, and this work feeds him R6.

## The load-bearing sequencing logic

> You can't validate a *quality signal* without a trustworthy benchmark, and you can't design a *quality gate* without a validated signal.

That decides the order of everything downstream:

```
[A: ScienceQA hand-check] ──► [Benchmark steps 4–8, dev only] ──► data foundation
                                                                       │
[B: pod smoke run-book] ──► [Step 5: 3-budget dev sweep]             ▼
                                                                  [Step 9: AI grader candidates]
                                                                          │
                                                              [Step 11: validate the grader] ──► [Step 14: runtime quality gate]
                                                                          │
                                                    [Step 12: error analysis] [Step 13: frontier] ◄── team's timing (Amay/Rithvik)
                                                                          │
                                                                  [Step 16/17: success criteria + deliverables + writeup]
```

The hand-check and smoke run-book stay the first execution steps. Everything from the AI grader onward is **gated on evidence** from the benchmark: a grader no one trusts is a quality gate that lies.

---

## Step 9 — Local AI grader candidates (Phase 5)

**When:** after the accuracy sweep (step 5) and scorer agree/random-baseline sanity (steps 6–7) show the benchmark is trustworthy. A grader trained or validated against a broken benchmark is garbage-in.

**What it investigates:** a small, self-contained grader that runs locally — no network during normal operation, reproducible, low overhead, runs on our machine/GPU.

**Candidate comparison** (your call to make, per criteria in SRIBHAV_WORK Phase 5 — I'll scaffold the comparison harness; I will not write the grader's scoring logic):

| Criterion | What it means here |
|---|---|
| Accuracy | Agreement with human/VQA labels on dev |
| Memory / GPU or CPU | Does it need a GPU at all; VRAM footprint on our A40/4090 |
| Latency | Per-answer grading cost, separately timed |
| Context requirements | Answer-only vs question+answer vs image+question+answer |
| Reliability / determinism | Same input → same score; is temperature/seed controllable |
| Ease of deployment | Docker pinning, disk, startup |
| **Blind spot check** | Negation, OCR errors, count-off-by-one, hallucinated detail (your own Phase 7 list) |

Candidates to put through it: the two existing local signals (BERTScore + NLI) as a "statistical grader," vs small local LLMs (the **Ollama/**llama stack already prototyped in `sribhav/ollama_as_a_judge.py`), vs a hybrid. Recommend a **primary + one fallback** candidate, with measurements, not vibes.

**Mandatory conceptual check first (Phase 10's IMPORTANT LIMITATION, front-loaded):** an answer-only grader *cannot know whether an answer is factually correct*. So decide — before choosing candidates — what signal the grader is actually producing:
- (a) semantic agreement with labeled references *(impossible at runtime — no labels exist at runtime)*,
- (b) answer plausibility / self-consistency *(possible, but only a proxy)*,
- (c) agreement with a *second*, differently-pruned run of the same question *(detects pruning-induced instability — promising, costs one extra cheap run)*,
- (d) model confidence / token-level signal,
- (e) a multimodal grader with the actual image *(most faithful, but likely too expensive at runtime — measure before rejecting)*.

This decision shapes the entire gate design and will also be an interviewer's first question ("what does the grader actually know?"). The offline grader can score against labels; the *runtime* grader cannot, and the plan must not conflate the two.

**Why not just use Claude (Phase 6), built in as the rule:** Claude is a perfectly good grader for **offline research evaluation** (score the sweep once, done). The local grader is only justified by the **continuous engineering QA / runtime-gating** use case — no external API, no network latency, no recurring cost, reproducible, deployable, can run on every request. So the local grader exists *because of the quality gate*, not because "local is cooler." That framing goes in the writeup.

**Deployment reality check:** `ollama`/`chromadb` are *not* in `requirements.txt` — any adopted grader must be pinned in the **Dockerfile + image rebuild**, never `pip install` on a pod (CLAUDE.md infra rule).

---

## Step 10 — Validate the local grader (Phase 7)

**References** (in order of trust): ground-truth dataset labels (ScienceQA + hetero answers) → the human labels you produce in Step 2 / by hand-checking failures → Claude as a strong external judge *for free-text semantic grading only*, never as ground truth for multiple-choice.

**Metrics** on a **dev-only** validation subset (no test contamination, rule 11):
- agreement rate, precision, recall, false-acceptance rate, false-rejection rate, confusion matrix, score correlation;
- broken down **per task category and per pruning level** (a grader that's accurate at n576 but blind at n64 is not a quality gate for aggressive pruning);
- targeted case list (your Phase 7 list): semantically-equivalent-but-different, partially correct, hallucinated detail, wrong OCR, count off by one, wrong spatial reasoning.

**Decision rule baked in:** "Is the lightweight grader reliable enough for the role we want?" If agreement with the reference is near-chance, the grader is a research *finding* (local LLMs can't replace validated scoring) and **not** a runtime gate. That negative result is valid and reportable.

---

## Step 11 — Error analysis (Phase 8)

On the dev sweep failures, classify *why* quality dropped: critical region/token removed, small object disappeared, OCR info lost, spatial relation collapsed, counting became incomplete, model already wrong at n576 (no causal relation to pruning), or the *scorer/gradger* mis-scored. Where feasible, inspect kept vs discarded patches.

Output: an **error taxonomy with representative examples per category**. This is what makes R6 ("OCR and counting break first") a *named mechanism* instead of a datapoint, and it feeds the writeup's "where the published claims held."

---

## Step 12 — Accuracy–pruning frontier (Phase 9)

Tables/plots: pruning % ↔ token count ↔ task accuracy ↔ dataset/category. Identify safe region, sharp cliff, dataset-dependent thresholds, tasks where aggressive pruning stays safe, tasks where modest pruning is risky.

This is where Sribhav's quality measurements **join Amay/Rithvik's** numbers: the final deliverable is the **accuracy–throughput frontier** chart (the single chart in the project's problem statement), and per CLAUDE.md only reportable numbers (Rithvik's harness) go on it. Sribhav supplies the accuracy axis (R6), they supply throughput (R1–R5).

---

## Step 13 — Runtime quality gate experiment (Phase 10)

Only after the grader has passed Step 11's validation. Configurations:

- **A. Baseline** — no pruning.
- **B. Fixed pruning** — VisPruner always at one budget.
- **C. Adaptive** — start aggressive/moderate; if the gate flags "risky," rerun at a larger budget or unpruned.

**Metrics (from your Phase 10 list):** final accuracy, first-pass accuracy, fallback rate, tokens processed, avg inference latency, p50/p95 if practical, **grader overhead**, **total pipeline latency**, % baseline accuracy recovered, % pruning efficiency retained.

The key engineering question, stated exactly as you framed it: **Can we get most of the efficiency benefit of pruning while selectively paying additional compute only for difficult examples?**

The honest failure condition (Phase 15): if adaptive fallback ends up costing nearly as much as never pruning, the idea isn't worthwhile and we say so.

---

## Step 14 — Clean GPU measurement (Phase 11)

Separate these **into five buckets, never fused**: LLaVA inference, VisPruner overhead, local-grader latency, fallback-rerun latency, total end-to-end pipeline. The grader must never be folded into "VisPruner's inference time" and then claimed as such. Final performance numbers come only from Rithvik's harness; the grader work coordinates with it rather than replacing it.

---

## Step 15 — Deliverables, structure, success/failure (Phases 12/13/15)

- **Deliverables:** multi-dataset benchmark; accuracy-vs-pruning curves; per-task sensitivity; error taxonomy; representative failures; grader + grader validation + comparison vs stronger judge + latency/resource; gate prototype + fallback strategy + fixed-vs-adaptive benchmark; configs, rerun scripts, saved raw results, analysis notebooks, final plots/tables.
- **Structure:** reuse repo conventions — eval configs under `sribhav/` (experiment configs, metrics, gate experiments), raw outputs under `results/`, plots under `sribhav/`; nothing sprawls into Amay's/Rithvik's directories.
- **Success/failure per experiment** (Phase 15, all pre-registered): flat curves across all datasets → *pruning generalizes, report that*; poor grader agreement → no gate; adaptive ≈ unpruned cost → drop the idea; different safe ratios per dataset → adaptive pruning is a real engineering problem worth building.

---

## Full execution sequence (final)

| # | Step | Depends on | Effort | Evidence it yields |
|---|---|---|---|---|
| 1 | ScienceQA 20-sample hand-check (Deliverable A) | — | small | Exact-match error pattern; scorer-trust decision |
| 2 | You fill the 20 judgments; we review disagreements | 1 | small (your time) | Human-graded labels for grader validation |
| 3 | Pod smoke run-book + smokes 1&2 (Deliverable B) | — | small | Runner verified; 576=unpruned confirmed |
| 4 | `question_type`-keyed scorer design (your call) | 2 | medium | Correct free-text metric |
| 5 | 3-budget dev sweep n576/288/144 (r=0.5) | 3,4 | medium (pod) | Per-category accuracy-vs-tokens curves |
| 6 | Random-answer baseline on outputs (rule 17) | 5 | small | Scorer discrimination sanity |
| 7 | Second-scorer agreement (rule 13) + thresholds | 5,6 | small | Scorer reliability |
| 8 | Per-category accuracy report (rule 14) | 5–7 | small | R6 result material |
| 9 | AI grader candidate comparison (Phase 5) | 8 | medium | Primary + fallback grader, with latency/memory |
| 10 | "Why not Claude" justification written (Phase 6) | 9 | small | Framing for the writeup |
| 11 | Validate grader vs labels/human/Claude (Phase 7) | 9,2 | medium | Grader agreement metrics; gate-readiness decision |
| 12 | Error analysis taxonomy (Phase 8) | 8,11 | medium | Failure mechanisms per category |
| 13 | Accuracy–pruning frontier (Phase 9) | 8–12 + team timings | medium | Feeds final accuracy–throughput chart |
| 14 | Runtime quality gate A/B/C experiment (Phase 10) | 11,12,13 | large | Recovery vs added-compute economics |
| 15 | Clean 5-bucket GPU measurement (Phase 11) | 14 | medium | Separated grader vs inference costs |
| 16 | Success/failure conditions applied (Phase 15) | 13–15 | small | Negative findings reported honestly |
| 17 | Deliverables + writeup (Phases 12/13) | all | large | The "one chart + honest writeup" |