You are helping me design and execute my part of a larger VisPruner / multimodal inference project.

My responsibility is primarily the **accuracy, evaluation, and quality-assurance side of VisPruner**. Another team member is handling low-level GPU profiling and optimization, so do not turn this into a CUDA/Triton optimization project.

I want you to first understand the problem deeply and then produce a **detailed implementation and research plan before writing code**.

The main research question is:

**How reliably does VisPruner preserve answer quality across varied datasets and pruning levels, and can we build a lightweight local AI grader that continuously detects when pruning has damaged answer quality?**

There is also a possible engineering extension:

**Can that grader become a runtime quality gate that triggers a fallback to a larger visual-token budget when aggressive pruning is likely to hurt accuracy?**

---

# PHASE 0 — FIRST UNDERSTAND THE EXISTING PROJECT

Before proposing implementation details, inspect the repository and understand what already exists.

Identify:

* where VisPruner is implemented
* how visual-token pruning currently works
* what pruning ratios / token counts are supported
* how LLaVA inference is currently invoked
* what datasets or evaluation scripts already exist
* whether answer scoring already exists
* whether there is already any Ollama/local-model evaluation infrastructure
* how outputs and experiment results are currently stored
* what code belongs to the GPU profiling work versus the evaluation work
* what pieces can be reused rather than rebuilt

Give me a concise architecture summary after inspecting the repository.

Do not start making major changes until you understand the existing implementation.

---

# PHASE 1 — PRODUCE A DETAILED PLAN BEFORE CODING

I want a substantial plan that I can review before implementation.

The plan should be detailed enough that another engineer could pick it up and execute it.

Structure the plan into the following sections.

## A. Objective

Explain exactly what we are trying to determine scientifically and what we are trying to build from an engineering perspective.

Separate these two things clearly.

For example:

### Research objective

Determine how answer quality changes under VisPruner across different visual tasks, datasets, and token budgets.

### Engineering objective

Develop a lightweight local grader capable of identifying cases where pruning appears to have damaged answer quality and investigate whether this can support adaptive fallback.

Do not assume that VisPruner fails or succeeds. The experiments must determine that.

---

## B. Hypotheses

Write explicit hypotheses that we can test.

For example:

H1:
Increasing pruning will reduce computation but eventually decrease task accuracy.

H2:
The amount of safe pruning will vary across task categories.

H3:
Tasks involving OCR, small objects, counting, or fine spatial relationships may be more sensitive to pruning than coarse visual questions.

H4:
A lightweight local grader can predict answer-quality degradation sufficiently well to act as a quality signal.

H5:
An adaptive fallback system can recover some accuracy lost under aggressive pruning while retaining a large portion of the efficiency benefit.

These are examples. Improve them if necessary.

For every hypothesis, specify:

* independent variable
* dependent variable
* control
* metrics
* experiment needed to test it

---

# PHASE 2 — DESIGN THE VISPRUNER ACCURACY BENCHMARK

Design a benchmark comparing several visual-token budgets.

At minimum consider:

* Full visual tokens / no pruning
* Light pruning
* Moderate pruning
* Aggressive pruning
* Current/default VisPruner configuration

If the normal LLaVA representation uses approximately 576 visual tokens, the full-token configuration should be included as the **unpruned reference configuration**.

However:

**Do not treat the unpruned model output as ground truth.**

If a dataset provides labels or reference answers, those labels are the primary ground truth.

The unpruned run instead answers this question:

**How much quality did pruning lose relative to what this model could achieve without pruning?**

For every pruning configuration collect:

* task accuracy
* semantic answer quality where required
* visual tokens retained
* pruning percentage
* inference latency if available
* GPU measurements supplied by the profiling side of the project
* failure category
* dataset/task category

Explain exactly how comparisons should be made.

---

# PHASE 3 — DATASET DIVERSITY

The benchmark should test whether VisPruner generalizes across visually different workloads.

Investigate appropriate datasets for several categories, such as:

* general VQA
* object recognition
* OCR / text-heavy images
* counting
* spatial reasoning
* small-object questions
* fine-grained visual detail
* chart/document understanding if feasible

Do not simply add datasets for quantity.

For each proposed dataset explain:

1. What capability it tests.
2. Why that capability may react differently to visual-token pruning.
3. Whether it provides deterministic labels/reference answers.
4. How difficult evaluation will be.
5. Dataset size.
6. Whether we need the whole dataset or a representative subset.
7. How much experiment runtime it is likely to require.

Then recommend the smallest useful dataset suite that gives us genuinely varied workloads.

---

# PHASE 4 — SCORING STRATEGY

Design a hierarchy of evaluation methods.

Prefer the most objective method available.

For example:

### Level 1 — deterministic scoring

Use exact match, multiple-choice accuracy, numeric comparison, etc. when possible.

### Level 2 — dataset-specific scoring

Use official evaluation metrics where appropriate.

### Level 3 — semantic grading

For open-ended answers where deterministic metrics are inadequate, use an AI grader.

Explain which datasets need which form of evaluation.

---

# PHASE 5 — LIGHTWEIGHT LOCAL AI GRADER

This is an important new component.

We want to investigate a **small, self-contained AI grader** that can run locally rather than relying on Claude/OpenAI/etc. for every evaluation.

The grader should ideally:

* run locally
* be reproducible
* require no network connection during normal operation
* be reasonably lightweight
* produce either a correctness judgment, score, or confidence
* handle semantically equivalent answers
* have sufficiently low overhead that repeated evaluation is realistic
* ideally be capable of running on our machine/GPU

Investigate possible approaches, including small local language models if appropriate.

Do not simply choose the smallest model.

Compare candidate graders based on:

* accuracy
* memory use
* latency
* GPU/CPU requirements
* context requirements
* reliability
* ease of deployment
* determinism/reproducibility

Recommend a primary candidate and one fallback candidate.

---

# PHASE 6 — ANSWER THE "WHY NOT JUST USE CLAUDE?" QUESTION

Explicitly address this design challenge.

An interviewer may reasonably ask:

**"If grading only needs to happen during benchmarking, why didn't you just send the outputs to Claude once?"**

The answer should distinguish between two use cases.

### Offline research evaluation

If we only wanted to compute benchmark scores once, then yes:

A strong external model such as Claude could be a perfectly reasonable grader.

### Continuous engineering quality assurance

Our larger engineering question is whether the evaluator can eventually remain part of the deployed inference pipeline.

A local grader may therefore provide:

* no external API dependency
* no network latency
* no recurring API cost
* reproducible grading
* potential offline deployment
* reduced external data exposure
* ability to run continuously rather than only during one experiment

Therefore the local grader should not exist merely because "local is cooler."

It must be justified by the continuous-quality-monitoring / runtime-gating use case.

Make this distinction clear in the project plan.

---

# PHASE 7 — VALIDATE THE LOCAL GRADER

We cannot simply trust a local grader because it produces a score.

Create a validation experiment.

Use one or more stronger references:

* ground-truth dataset labels
* human judgment
* Claude or another strong external judge

Create a representative validation subset.

Measure:

* agreement rate
* precision
* recall
* false acceptance rate
* false rejection rate
* confusion matrix where applicable
* score correlation if using continuous scores
* performance by task category
* performance by pruning level

Especially examine cases where:

* VisPruner output is slightly different but semantically correct
* the answer is partially correct
* the answer contains hallucinated details
* OCR is incorrect
* counting is off by one
* spatial reasoning is wrong

The important research question is:

**Is the lightweight grader reliable enough for the role we want it to perform?**

Do not claim that it is equivalent to Claude unless evidence supports that.

---

# PHASE 8 — ERROR ANALYSIS

For failures, classify why VisPruner lost quality.

Possible categories include:

* critical token/region was removed
* small object disappeared
* OCR information was lost
* spatial relationship became unclear
* counting information became incomplete
* aggressive pruning removed necessary context
* model was already wrong without pruning
* evaluator incorrectly scored the answer
* pruning had no causal relationship to the failure

Where feasible, compare visual tokens retained versus discarded.

The goal is not merely to report an overall accuracy number.

We want to learn:

**What kinds of visual information are most sensitive to token pruning?**

Produce examples for important failure categories.

---

# PHASE 9 — ACCURACY–PRUNING FRONTIER

Create plots/tables showing the relationship between:

* pruning percentage
* visual-token count
* task accuracy
* dataset/task category

Eventually this can be combined with the GPU profiling team's measurements to create an:

**accuracy–latency / accuracy–throughput frontier**

But my portion should primarily ensure that the quality measurements are rigorous.

Identify whether there appears to be:

* a generally safe pruning region
* a sharp quality cliff
* dataset-dependent thresholds
* workloads where aggressive pruning remains safe
* workloads where even modest pruning is risky

---

# PHASE 10 — RUNTIME QUALITY GATE EXPERIMENT

After the offline grader is validated, design an experiment for using it as a quality gate.

Concept:

Image + question
→ VisPruner
→ LLaVA
→ local quality/confidence grader
→ determine whether answer appears safe

If safe:

return answer

If unsafe:

rerun with a larger token budget or disable pruning

Test at least these configurations:

### A. Baseline

No pruning.

### B. Fixed pruning

VisPruner always uses the same token budget.

### C. Adaptive pruning

Start with aggressive/moderate pruning.

If the quality gate identifies a risky answer, retry using more visual tokens.

Compare:

* final accuracy
* first-pass accuracy
* fallback rate
* number of tokens processed
* average inference latency
* p50/p95 latency if practical
* grader overhead
* total pipeline latency
* percentage of baseline accuracy recovered
* percentage of pruning efficiency retained

The key engineering question is:

**Can we get most of the efficiency benefit of pruning while selectively paying additional compute only for difficult examples?**

---

# IMPORTANT LIMITATION

Be careful about one conceptual issue:

A grader that only sees the generated answer may not be able to know whether that answer is factually correct.

For runtime gating, determine exactly what information the grader would need.

Possible inputs might include:

* question + answer
* image + question + answer
* unpruned/pruned representation signals
* model confidence
* disagreement between multiple token budgets
* other quality proxies

Investigate this properly.

Do not design a runtime system that assumes an answer-only grader magically knows the correct answer.

If a multimodal grader would be too expensive, explore cheaper quality proxies or confidence signals.

This is an important part of the research.

---

# PHASE 11 — KEEP GPU MEASUREMENTS SCIENTIFICALLY CLEAN

The local grader may itself consume GPU resources.

Therefore separate:

1. LLaVA inference latency
2. VisPruner overhead
3. local grader latency
4. fallback latency
5. total end-to-end pipeline latency

Do not include the grader inside the raw VisPruner benchmark and then claim that number represents VisPruner's inference performance.

Coordinate the final performance measurements with the GPU profiling side of the project.

---

# PHASE 12 — DELIVERABLES

The plan should culminate in these deliverables.

## Research deliverables

1. Multi-dataset VisPruner benchmark
2. Accuracy versus pruning curves
3. Per-task sensitivity analysis
4. Error taxonomy
5. Representative failure examples
6. Analysis of whether pruning generalizes across distributions

## Grader deliverables

7. Lightweight local grader
8. Evaluation of grader accuracy
9. Comparison against a stronger judge / ground truth
10. Grader latency and resource measurements

## Adaptive-system deliverables

11. Prototype quality gate
12. Fallback strategy
13. Fixed-versus-adaptive pruning benchmark
14. Accuracy recovery versus added compute analysis

## Reproducibility deliverables

15. Experiment configuration files
16. Clear commands/scripts for rerunning benchmarks
17. Saved raw results
18. Analysis notebooks/scripts
19. Final plots/tables

---

# PHASE 13 — PROJECT STRUCTURE

Propose a sensible repository structure before implementing.

For example, determine where we should place:

* datasets
* benchmark runners
* evaluator interfaces
* local-grader code
* grader-validation scripts
* runtime-gate experiments
* metrics
* experiment configs
* result files
* visualization scripts

Avoid creating an unnecessarily complex framework.

Reuse existing project conventions wherever possible.

---

# PHASE 14 — EXECUTION SCHEDULE

Turn all of the above into an ordered execution plan.

For every step include:

* objective
* files/components likely affected
* implementation task
* experiment to run
* expected output
* dependency on previous work
* completion criterion
* approximate relative effort: small / medium / large

Organize the work into milestones.

A possible structure is:

### Milestone 1 — Reproduce baseline

Ensure the existing unpruned and VisPruner inference paths run reliably.

### Milestone 2 — Accuracy harness

Build reproducible evaluation across pruning ratios.

### Milestone 3 — Dataset expansion

Evaluate varied visual workloads.

### Milestone 4 — Local grader

Implement and validate lightweight semantic grading.

### Milestone 5 — Error analysis

Identify workload-specific weaknesses.

### Milestone 6 — Adaptive quality gate

Prototype fallback behavior.

### Milestone 7 — Final benchmark

Measure quality, efficiency, fallback rate, and grader overhead.

### Milestone 8 — Documentation

Produce reproducible results and the final research conclusion.

Modify these milestones if repository inspection suggests a better sequence.

---

# PHASE 15 — DEFINE SUCCESS AND FAILURE CONDITIONS

For every major experiment, specify what results would support or reject our hypothesis.

Examples:

If aggressive pruning produces almost no accuracy loss across every tested dataset, then we should report that rather than inventing a generalization problem.

If the local grader has poor agreement with ground truth, then it should not be presented as a reliable runtime quality gate.

If adaptive fallback costs nearly as much as simply running unpruned inference, then the idea may not be worthwhile.

If different datasets have substantially different safe pruning ratios, that supports adaptive pruning as a meaningful engineering problem.

The experiments must be capable of disproving our preferred story.

---

# FINAL PROJECT STORY WE ARE TRYING TO EARN

Do not assume this statement is true beforehand.

The experiments should determine whether we have enough evidence to eventually say something like:

> We investigated whether VisPruner's visual-token reduction generalized across different multimodal workloads rather than evaluating only aggregate performance on one distribution. We benchmarked multiple pruning levels against the full-token baseline and found that pruning sensitivity varied by task. We then developed and validated a lightweight local evaluator to detect quality degradation and explored using it as a runtime quality gate. An adaptive fallback system selectively increased the visual-token budget on risky examples, allowing us to study the tradeoff between quality preservation and inference efficiency.

Every part of this final story must be backed by measurements.

If the measurements show something different, rewrite the story to match the evidence.

---

# YOUR FIRST RESPONSE TO ME

Do **not** start by dumping implementation code.

First:

1. Inspect the existing repository.
2. Summarize the relevant architecture.
3. Identify what already exists versus what must be added.
4. Produce the detailed phased plan described above.
5. Identify technical risks and conceptual problems.
6. Recommend the smallest sensible first experiment.
7. Tell me exactly what evidence that experiment will give us.
8. Only after the plan is understood should we begin implementation step by step.

Keep the project scoped around **VisPruner accuracy, evaluation, local grading, and adaptive quality assurance**.