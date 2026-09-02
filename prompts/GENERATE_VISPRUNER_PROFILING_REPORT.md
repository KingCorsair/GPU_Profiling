# Prompt: Generate/Update the VisPruner Profiling Report

## Purpose

This prompt consolidates the VisPruner/LLaVA GPU-profiling investigation — every experiment, script, and raw output produced across that phase — into one canonical, traceable research report:

`/workspace/GPU_Profiling/AMAY_VISPRUNER_PROFILING_REPORT.md`

It exists so that document can be regenerated from scratch, or updated with new experiments, by a future Claude Code session that has no memory of the original conversation. The prompt is deliberately literal and exhaustive rather than a short summary — running it should require the session to actually re-inspect the repository's evidence rather than trust anyone's prior summary of it, including its own.

## When to use this

- The profiling/research phase gains a new experiment that should be folded into the canonical record.
- A conclusion in the report is later superseded by new evidence (e.g. an optimization is implemented and benchmarked, or a methodological issue is found).
- The report itself needs a structural refresh but the underlying experimental history hasn't changed.
- Do **not** use this to kick off new profiling work — it is a documentation/consolidation prompt. If GPU experiments need to be run first, run them separately, then use this prompt (or a lightly-edited version of it) to fold the results in.

## Output

Running this prompt produces or updates:

- `/workspace/GPU_Profiling/AMAY_VISPRUNER_PROFILING_REPORT.md` — the report itself
- A minimal, additive edit to `/workspace/GPU_Profiling/AMAY_SPEED_PLAN.md` linking to the report (only on first generation — don't re-apply if the link already exists)

It does **not** modify any file under `results/timing/`, `results/loadgen/`, or any script — those are read-only evidence for this prompt's purposes.

---

## The complete Claude prompt

```
I want you to consolidate the entire VisPruner/GPU profiling phase into one canonical research-style technical report.

Do NOT rerun GPU experiments.
Do NOT implement optimizations.
Do NOT delete or modify raw experiment evidence.

Create:

/workspace/GPU_Profiling/AMAY_VISPRUNER_PROFILING_REPORT.md

This should become the canonical written record of the profiling/research phase.

The goal is that months from now I should be able to open this one document and understand:

- what question we were investigating
- what experiments were conducted
- how each experiment was conducted
- where its scripts/raw outputs/traces are stored
- what each experiment found
- what methodological problems were discovered
- what numbers are trustworthy
- what conclusions survived those problems
- why those findings motivate the next engineering phase

It should read like a clear research/engineering paper rather than a collection of notes.

==================================================
1. INSPECT THE EXISTING EVIDENCE FIRST
==================================================

Before writing the report, inspect all relevant project-owned files and experiment outputs.

At minimum inspect:

- AMAY_SPEED_PLAN.md
- AMAY_TIMING_NOTES.md

Relevant scripts, including where present:

- scripts/profile_multimodal_prep.py
- scripts/profile_full_request.py
- scripts/run_concurrent_load_test.py
- scripts/profile_concurrent_load.py
- scripts/run_noload_ab_comparison.py
- scripts/run_ab_load_sweep.py
- scripts/profile_ab_trace.py
- any later scripts that clearly belong to this same profiling investigation

Relevant outputs:

- results/timing/*
- relevant results/loadgen/*
- profiler summaries
- stage breakdowns
- settings/metadata files
- raw A/B results
- traces
- monitoring outputs
- logs

Do not assume the old planning documents are completely correct.

Where possible, derive claims from the experiment outputs themselves.

Do not include unrelated work belonging to other contributors except where their existing harness/tool was used as part of these experiments.

==================================================
2. REPORT FORMAT
==================================================

Write the report using the following research-style structure.

# Title

Use a descriptive technical title centered on VisPruner, LLaVA, GPU profiling, and the gap between algorithmic/token reduction and real serving performance.

# Abstract

In simple English summarize:

- the system investigated
- the core question
- the main experiments
- the main result
- the main engineering implication

Keep this concise.

# 1. Motivation and Research Question

Explain why this investigation was performed.

The central question is approximately:

"Does reducing LLaVA visual tokens with VisPruner translate into proportional end-to-end latency and serving-performance improvements, and if not, where do the unrealized gains go?"

Refine this wording if the evidence suggests a more accurate version.

Clearly distinguish:

- algorithmic/token reduction
- prefill improvement
- complete request latency
- serving throughput/tail latency

Do not imply that a reduction in visual-token work theoretically guarantees an equal reduction in complete request latency.

# 2. System Under Test

Document:

- model
- VisPruner implementation
- GPU
- PyTorch version
- CUDA version
- Transformers version
- dtype
- attention implementation
- batch size
- important_ratio
- visual_token_num configurations
- server architecture
- load-generation architecture
- dataset/workload
- output-token settings where relevant

Use exact values from experiment metadata wherever available.

# 3. Experimental Methodology

Explain the profiling tools used:

- torch.profiler
- CUDA events where applicable
- record_function regions
- Chrome/Perfetto traces
- load generator
- GPU/CPU monitoring
- warmup
- A/B comparison methodology

Explain what each measurement means.

Clearly distinguish:

MEASURED
DERIVED
INTERPRETATION

==================================================
3. DOCUMENT EVERY MAJOR EXPERIMENT
==================================================

Create one full section per major experiment.

The experiment sequence appears to include approximately:

Experiment 1 — Narrow multimodal preparation profiling

Experiment 2 — Full isolated request profiling

Experiment 3 — Serving/load profiling

Experiment 4 — 576-vs-128 VisPruner no-load A/B comparison

Experiment 5 — 576-vs-128 load A/B comparison

Experiment 6 — 576-vs-128 profiler traces / isolated and near-saturation comparison

Experiment 7 — Any later correction/validation analysis that materially changed interpretation

Change this organization if repository evidence shows a better structure.

For EVERY experiment include:

## Research question

What exactly were we trying to learn?

## Experimental setup

- script
- model settings
- workload
- number of requests
- load/RPS if applicable
- visual_token_num
- output-token behavior
- relevant profiler settings

## Files produced

List exact repository paths for:

- script
- raw data
- profiler trace
- summary
- metadata
- logs

Explain what each file contains.

## Results

Give the important numerical results in tables.

## Interpretation

Explain what those numbers mean in simple English.

## Limitations

Explain what this experiment cannot establish.

## Status

Classify it as one of:

VALID
VALID WITH CAVEAT
DIAGNOSTIC ONLY
SUPERSEDED

Explain why.

==================================================
4. PRESERVE THE IMPORTANT MEASURED RESULTS
==================================================

Include a canonical results section collecting the important numbers.

Examples include, where supported:

- 576 → 128 visual tokens
- prefill latency reduction around 60%
- generate-only median latency improvement around 15.2%
- excluded image preprocessing cost around 86.7 ms
- mathematically corrected estimated full end-to-end improvement around 13.4%
- decode contribution to total latency
- vision tower contribution
- CUDA kernel launch count
- CPU CUDA-launch overhead
- KV-cache legacy rebuild behavior
- KV-cache torch.cat append behavior
- queueing/tail degradation beginning around 1.5 RPS
- observed throughput behavior at higher load
- approximately 3.8% observed capacity improvement, with its methodological caveat
- GPU memory findings
- serial serving behavior

Do NOT blindly use these values from this prompt.

Verify each one against the repository evidence.

For every important result give:

Metric
Value
Experiment
Source file
Measurement type
Confidence
Caveat

==================================================
5. HANDLE THE PART 1 TIMING BUG CORRECTLY
==================================================

Document the confirmed no-load timing bug carefully.

The original no-load timer began AFTER image preprocessing.

Therefore the reported 652.1 ms vs 553.0 ms values measured generate/model-path latency rather than complete request latency.

Image load + preprocessing + CUDA transfer was separately measured at approximately 86.7 ms average and should apply roughly equally to both configurations.

Preserve the raw numbers.

Then show the mathematical correction explicitly:

raw:
576 = 652.1 ms
128 = 553.0 ms

estimated complete request:
576 ≈ 652.1 + 86.7
128 ≈ 553.0 + 86.7

Then calculate the corrected percentage improvement.

Label this:

DERIVED ESTIMATE

not a newly measured result.

Explain why we deliberately chose not to rerun this profiling phase and instead will require a clean end-to-end benchmark during final optimization validation.

==================================================
6. DOCUMENT ALL KNOWN METHODOLOGY ISSUES
==================================================

Create a dedicated section:

# Experimental Limitations and Corrections

Include all confirmed issues, such as:

1. Part 1 preprocessing excluded from timer
2. 576 and 128 load sweeps were not temporally interleaved
3. output_token_count included a synthetic BOS token
4. isolated nvidia-smi utilization measurement was weak
5. low-RPS p95/p99 had tiny sample counts
6. near-saturation profiler experiment used back-to-back requests rather than real concurrent GPU execution
7. naive shared-model threaded generation previously crashed and is not evidence of successful concurrency
8. natural EOS caused small differences in generated output length in some comparisons
9. any other issue supported by the repository

For each issue include:

- affected experiment
- affected result
- severity
- whether relative conclusions change
- correction, if any
- whether a future clean benchmark is required

==================================================
7. SYNTHESIZE THE RESULTS
==================================================

Create a major section:

# Combined Findings

Explain the whole chain in simple English.

It may approximately be:

576 visual tokens
        ↓
128 visual tokens
        ↓
large reduction in prefill cost
        ↓
but decode remains dominant
        ↓
therefore much smaller complete-request latency improvement
        ↓
current serial serving architecture causes queueing
        ↓
therefore little additional serving capacity is realized

But derive the exact wording from evidence.

Answer explicitly:

1. What does VisPruner clearly improve?
2. What does it barely affect?
3. Why is end-to-end improvement much smaller than prefill improvement?
4. Why does the serving improvement become even smaller?
5. Which effects are VisPruner-specific?
6. Which effects are properties of the current inference/serving implementation?

==================================================
8. CONNECT THE EVIDENCE TO THE NEXT ENGINEERING PHASE
==================================================

Create:

# Engineering Implications

Do not design the final implementation roadmap here.

Instead connect each measured finding to an engineering question.

Organize findings under:

## Triton / custom kernels

## KV-cache optimization

## CUDA Graphs

## Batching / serving

## Additional categories

Only add an additional category if the profiling evidence justifies one.

For each:

Measured evidence
→ suspected systems problem
→ engineering question

Example format:

MEASURED:
34k+ CUDA launches during the representative request.

IMPLICATION:
Decode contains highly repetitive GPU dispatch.

ENGINEERING QUESTION:
Can CUDA Graph capture reduce CPU dispatch/launch overhead?

Do NOT say an optimization will work before it is implemented and benchmarked.

==================================================
9. EVIDENCE INDEX / APPENDIX
==================================================

Create an appendix:

# Appendix A — Experiment Artifact Index

This should solve the problem of finding files.

Give a repository tree showing:

GPU_Profiling/
│
├── AMAY_VISPRUNER_PROFILING_REPORT.md
├── AMAY_SPEED_PLAN.md
├── AMAY_TIMING_NOTES.md
│
├── scripts/
│   ├── ...
│
└── results/
    ├── timing/
    │   ├── ...
    └── loadgen/
        └── ...

For every important file give a one-line explanation.

Group repetitive loadgen result directories instead of listing hundreds of files.

Explain the difference between:

SCRIPT
RAW DATA
PROFILER TRACE
SUMMARY
METADATA
LOG

Also identify which files are normally worth opening manually and which are mainly archival/raw evidence.

==================================================
10. MAKE THE REPORT TRACEABLE
==================================================

An important requirement:

Every major numerical claim should point to the exact repository file supporting it.

Do not allow important results to exist only because Claude said them in a conversation.

If a conclusion can be reconstructed from raw files, reconstruct it.

If a claim cannot currently be supported by a repository artifact, explicitly write:

"Not currently preserved in a canonical repository artifact."

Do not fabricate support.

==================================================
11. ADD REGENERATION INSTRUCTIONS
==================================================

This report must be self-documenting.

Create:

# Appendix B — How to Regenerate or Update This Report

Explain step by step how another Claude Code session should regenerate or update this report later.

Include:

1. repository root
2. files/directories that should be inspected
3. which files must never be modified
4. how to distinguish my files from other contributors' files
5. how to preserve raw measurements
6. how to add new experiments
7. how to update old conclusions when later evidence supersedes them
8. how to keep MEASURED / DERIVED / INTERPRETATION distinct
9. how to update the evidence index
10. how to avoid rerunning GPU experiments unless explicitly requested

==================================================
12. EMBED THE EXACT CLAUDE PROMPT
==================================================

At the end of Appendix B include:

## Claude Prompt Used to Generate This Report

Include the complete exact prompt from this message verbatim.

The report should therefore contain its own regeneration prompt.

Do not summarize it.
Do not shorten it.
Preserve the full prompt.

This means someone should be able to:

1. open AMAY_VISPRUNER_PROFILING_REPORT.md
2. copy the prompt from Appendix B
3. give it to Claude Code in the repository
4. regenerate/update the report from the evidence

==================================================
13. ALSO SAVE THE GENERATION PROMPT SEPARATELY
==================================================

In addition to embedding the prompt in the report, create:

/workspace/GPU_Profiling/prompts/GENERATE_VISPRUNER_PROFILING_REPORT.md

Create the prompts/ directory if necessary.

This file should contain:

- purpose of the prompt
- when to use it
- the complete exact Claude prompt
- a note that AMAY_VISPRUNER_PROFILING_REPORT.md is the output

Do not place experimental results in this prompt file.

Its purpose is reproducibility.

==================================================
14. LINK THE DOCUMENTATION TOGETHER
==================================================

Update AMAY_SPEED_PLAN.md minimally near the beginning so it links to:

AMAY_VISPRUNER_PROFILING_REPORT.md

Explain:

AMAY_VISPRUNER_PROFILING_REPORT.md
= completed profiling/research evidence and experimental history

AMAY_SPEED_PLAN.md
= current engineering implementation roadmap

prompts/GENERATE_VISPRUNER_PROFILING_REPORT.md
= instructions for rebuilding/updating the profiling report

Do not substantially rewrite AMAY_SPEED_PLAN.md as part of this task.

==================================================
15. READABILITY
==================================================

The report should be technically rigorous but understandable to someone learning GPU inference.

Use:

- short paragraphs
- tables
- diagrams
- equations where useful
- simple English
- explicit definitions

Avoid writing it like raw profiler notes.

When introducing terms such as:

- prefill
- decode
- CUDA kernel launch
- KV cache
- queueing
- p50/p95
- throughput

briefly explain them.

==================================================
16. FINAL DOCUMENT ORGANIZATION
==================================================

The final report should approximately contain:

Title

Abstract

1. Motivation and Research Question
2. System Under Test
3. Experimental Methodology
4. Experiment 1
5. Experiment 2
6. Experiment 3
7. Experiment 4
8. Experiment 5 / later experiments as needed
9. Experimental Limitations and Corrections
10. Canonical Results
11. Combined Findings
12. Engineering Implications
13. Conclusion

Appendix A — Experiment Artifact Index
Appendix B — How to Regenerate or Update This Report
    - including the full Claude prompt

Change numbering if needed based on actual experiments.

==================================================
17. FINAL RESPONSE TO ME
==================================================

After creating everything, tell me:

1. exact report path
2. exact saved prompt path
3. experiments documented
4. major canonical findings
5. important caveated/superseded results
6. whether any evidence is missing
7. what was changed in AMAY_SPEED_PLAN.md
8. which document I should read first

Do NOT rerun experiments.
Do NOT implement performance optimizations.

This task is strictly consolidation, documentation, reproducibility, and research-report creation.
```
