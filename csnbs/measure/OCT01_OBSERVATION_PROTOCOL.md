# Follow-up: HTTP observation overhead and natural output policies

This follow-up is declared before its collection. It uses a separate clean source revision from the primary October 1 study. Both legs within each follow-up use the same revision, recorded in every run. Do not combine these trials with the original revision as if only token count changed.

## Incremental HTTP observation overhead

`campaigns/llava-oct01-http-overhead.json` prescribes five adjacent pairs of isolated full-HTTP trials, 90 measured requests and ten completed warmups per leg. Both legs use 576 visual tokens and the original 90 dev questions. The only changed control is `EXTENDED_HTTP_OBSERVATIONS=0` versus `1`, declared as `comparisonKind: "instrumentation"`. Primary metric: paired p50 HTTP latency. Positive latency effect in the generic report means the enabled condition was faster; interpret the signed result as an overhead experiment, never as a model optimization.

The enabled condition records host-wall stages, verified token-ID counts and answer character counts. It preserves the generation and output-decoding calls, with an additional output-ID host copy after decoding. Its cost is part of this experiment. The existing model-internal CUDA events, synchronizations and diagnostic writes remain active in both conditions. This experiment cannot measure their total cost.

The fixed budget is 900 measured requests plus 100 warmups. Use 10,000 seeded paired-block bootstrap draws. Five blocks may be insufficient to resolve a small overhead; a crossing interval is inconclusive. p95 and p99 remain unavailable with 90 observations per trial. Do not expand the budget until an overhead claim becomes favorable.

## Natural output-policy characterization

The three `llava-oct01-workload-{short,long,mixed}.json` specifications each prescribe two paired 576-versus-128-token repetitions at 1 request/second, 90 measured requests and ten completed warmups per leg. Keep importance ratio, natural EOS, greedy decoding, cache, maximum output budget of 64 tokens, and extended observation mode fixed. This is a descriptive sensitivity study; two blocks do not resolve an optimization claim.

The versioned workloads in `workloads/oct01/` derive from the same existing 90 dev examples. The short policy preserves the original question. The long policy appends an explicit three-sentence response request. The mixed policy assigns 45 questions to each policy, alternating within the six original categories. Preserve the original image, question identifier, category and source. Accuracy answers are omitted because these are serving workloads, not replacement accuracy datasets. The original dev and locked test files are unchanged.

Use a reproducible shuffled workload order from Python `random.Random(465)`. Each campaign also saves its own alternating A/B order. Inspect measured generated text-token counts, raw generated-step counts, EOS/cap hits and output characters separately for each variant, with observation coverage. More requested output is not proof of more produced output. If the longer prompt produces similar lengths, or token-contract checks fail, report that finding instead of changing the prompt during collection.

Total workload budget: 1,080 measured requests plus 120 warmups. At a given workload and rate the measured question order is identical across paired variants. Repeated examples are not independent population samples. Failures retain their prescribed denominator. A fixed 1 RPS may be below capacity for short outputs and overload the longer-output policy; use the observed backlog and delivery evidence, and do not call overloaded-window tails steady-state latency. These trials do not locate a precise capacity ceiling or replace broader multi-rate studies.

## Collection and evidence

Begin only after the primary and isolated experiments finish and their owned server stops. Verify the new source is clean, the checkpoint hashes match and no other GPU process is active. Confirm the expected instrumentation version in health before each leg. Keep ten completed warmups, 1 second settling, a 120-second per-request timeout and the saved fixed budget. Stop on failure; retain attempts and diagnose any follow-up without silently replacing evidence.

Verify all raw request/resource hashes and summaries, preserve original campaign bytes during export, and import only stopped finalized snapshots. Store actual observed lengths and unavailable reasons. These observations do not supply GPU prefill/decode attribution, queue admission timing, batching/KV implementations or validated accuracy.
