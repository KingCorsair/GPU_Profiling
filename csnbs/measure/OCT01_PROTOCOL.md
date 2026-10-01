# October 1 measurement protocol

Recorded before the primary token-count comparison. This protocol is Rithvik's measurement work; it does not change the serving implementation or evaluate accuracy.

## Question and fixed budget

Does the existing 128-token VisPruner configuration improve full-HTTP latency and completed work relative to the 576-token baseline on this A40, under the current 90-question dev workload and natural end-of-sequence policy?

The primary outcome is the paired percentage change in successful completions within the fixed arrival window at **3 requests/second**. Secondary outcomes are full-HTTP p50 latency at 1 and 3 requests/second, eligible per-trial p95 latency, outstanding work and drain time. An isolated full-HTTP experiment characterizes serial requests separately. Report all prescribed trials and failures, including negative or inconclusive findings.

| Stage | Specification | Trials | Measured requests/trial | Purpose |
|---|---|---:|---:|---|
| Baseline pilot | `campaigns/llava-baseline-pilot.json` | 6 identical baseline trials | 90 | Descriptive repeatability and arrival-delivery checks |
| Primary comparison | `campaigns/llava-oct01-primary.json` | 5 pairs at each of 1 and 3 RPS | 270 | Low-load and overloaded finite-window comparison |
| Isolated HTTP | `campaigns/llava-oct01-isolated.json` | 5 pairs, one request outstanding | 90 | Measured complete HTTP boundary without arrival queueing |

The broader four-rate `llava-token-ab.json` remains an unexecuted example protocol. The present two-rate study uses the pilot's 1 RPS operating point and the project's historical 3 RPS stress point. It has a bounded budget of 6,840 measured requests across all stages, plus 360 warmups; startup and drain add runtime. Budget approximately 90 minutes beyond the pilot. Do not expand this budget because an interval is unfavorable or crosses zero. A failed campaign stops for diagnosis; a retry retains the failure history and cannot restore an untouched fixed-budget claim.

## Controls and provenance

- Harness and serving source stay at clean commit `356df7d66f7cf4c7524c93d7c9d373e2b920fbfc` throughout GPU collection. Later analysis, export, documentation and test changes do not replace the recorded execution revision.
- GPU: the observed NVIDIA A40; each trial records the device UUID, driver and runtime. Use only this study's owned server, check for other compute processes before each trial, and preserve the shared checkout and pod.
- Checkpoint: `liuhaotian/llava-v1.5-7b`, revision `4481d270cc22fd5c4d1bb5df129622006ccd9234`, verified downloaded file hashes. An unavailable container digest remains unknown.
- Change only visual tokens 576 to 128. Keep `important_ratio=0.5`, FP16 model, batch size one, greedy decoding, cache enabled, `max_new_tokens=64`, natural EOS and `llava_v1` prompt template fixed.
- Ten completed warmups and 1 second settling before each trial. The measured sequence uses complete balanced cycles of the existing dev workload. Paired legs replay the same order; seed 465 advances by repetition.
- Restart the server for each leg. Use the saved randomized rate order and alternating adjacent A/B versus B/A order. Report startup separately from request latency.
- Deadline per request: 120 seconds. Preserve unsuccessful terminal outcomes and the full prescribed denominator. Do not substitute a success quota.

## Statistical and interpretation limits

Use nearest-rank percentiles, with p95 withheld below 200 successes and p99 withheld below 1,000. This study does not report GPU p99. Five independent paired blocks per rate are the resampling units for 10,000 seeded bootstrap draws. Desired practical resolution is **10%**, but neither the pilot nor five blocks guarantees that precision. An interval crossing zero is inconclusive, not equivalence. Treat secondary metrics as descriptive; do not select the most favorable metric as the headline.

Combine the six identical pilot trials only for descriptive repeatability, preserving the three paired blocks. A/A labels describe a sham comparison, never an optimization. Low-load throughput is constrained by offered arrivals, so equal throughput there does not imply equal capacity.

Capacity labels describe repeated finite windows. Two rates can produce a coarse passing/failing bracket; they do not locate an exact sustainable ceiling. Client outstanding requests include transport and service, and are not measured server queue depth. No latency SLO has been supplied.

Actual generated tokens, GPU prefill/decode, exact KV-cache allocation, asynchronous intake and batching occupancy require Amay's serving contract. The existing generator does not save answers or actual output lengths, so do not infer a controlled output-length study. Long-output/mixed-output experiments, instrumentation-overhead comparisons, random-scoring controls and the joint accuracy/throughput chart remain pending the corresponding supplied inputs. Sribhav retains eval design and scoring ownership.

Keep original raw files and campaign bytes. A portable export may rewrite only recorded artifact locations and must preserve a mapping and hashes. Reports and database imports verify the raw records before deriving conclusions.
