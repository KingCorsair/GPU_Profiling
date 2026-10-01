# Joining accuracy to serving measurements

`csnbs/join_accuracy_throughput.py` joins owner-supplied aggregate scores to an already verified canonical campaign report. It never reads answers or implements a scorer, evaluation split, or compression method. No matching accuracy has been supplied for the current pilot. The [example](campaigns/accuracy-handoff.example.json) is explicitly pending and contains no numeric scores.

Existing `sribhav/evaluation_results_*.json` and `sribhav/evaluation_with_baseline.json` are unmatched evaluation evidence. They include aggregates/predictions and a same-category random-answer baseline, but lack the matching server revision, checkpoint-file provenance, locked split hash and validated per-category handoff required here. Preserve those files; their existence does not supply a matched accuracy point for this pilot.

From the repository root, first generate the report using the canonical verifier. Then create a new output directory:

```sh
npm --prefix csnbs/measure run report -- path/to/campaign.json
python csnbs/join_accuracy_throughput.py --report path/to/report/report.json --output path/to/new-accuracy-handoff
# Once the accuracy owner supplies the completed aggregate artifact:
python csnbs/join_accuracy_throughput.py --report path/to/report/report.json \
  --accuracy path/to/owner-accuracy.json --output path/to/new-joint-results
```

The JSON records hashes of its exact report/accuracy inputs. Without accuracy, it contains `status:"pending"`, the required execution identities and no points; the CSV contains only column headings. A pending input must have an empty `evaluations` array. The tool never produces a placeholder chart. Existing output directories are refused. It checks input structure and matching identities; it does not repeat the canonical raw timing verification or independently adjudicate the owner's validation evidence.

## Owner aggregate contract

Use `schema:"owner-aggregate-accuracy"`, `schemaVersion:1`, `status:"validated"` and one `evaluations` entry per distinct measured execution identity. All fields below are required. No numbers in the integration tests are real accuracy evidence.

| Field in each evaluation | Required content |
|---|---|
| `evaluationId`, `scope` | Unique evaluation identity; `research` for model campaigns or `integration-only` for synthetic integration artifacts. Mixing scopes is rejected. |
| `executionIdentity` | Exactly the corresponding `identity` object emitted in the pending JSON, confirmed against the actual evaluated implementation. Never relabel different evaluation settings to make them match. |
| `evaluationGitCommit` | Full immutable 40-character evaluation-code commit. |
| `split` | `{datasetId, name, sha256, locked:true}` identifying the owner's locked evaluation population. Serving workload identity remains separate. |
| `scorer` | `{id, gitCommit, validated:true, validationMethod, validationEvidence:[{reference,sha256}]}`. References identify retained validation/agreement/hand-check evidence; a boolean without evidence is rejected. |
| `randomControl` | `{evaluationId, reference, sha256, validated:true}` identifying the owner's validated random-scoring control artifact. This join retains its provenance, not an invented control score. |
| `overall` | `{sampleCount, accuracy}`: positive integer count and the owner's reported accuracy fraction in `[0,1]`. |
| `perCategory` | Nonempty array of `{category,sampleCount,accuracy}` with unique names, positive counts no larger than the overall count, and reported fractions in `[0,1]`. Include the required project categories in the owner's evaluation; this helper does not design them. |
| `categorySemantics` | Explanation of whether categories overlap or form a partition. Scores are preserved; category scores are not pooled. |

Execution identity matches model, checkpoint revision, every checkpoint file's SHA-256/size, clean server commit, PyTorch/Transformers versions and effective configuration. This includes implementation, visual-token count, importance ratio, prompt template, decoding/EOS policy, dtype, batching and any extra effective controls. Only the local checkpoint path and download-provenance envelope are excluded; the checkpoint file inventory remains matched exactly. Input evaluation identities must retain canonical field structure and file ordering. Missing or mismatched identities are rejected, including an extra unmatched evaluation. Select one owner-approved locked artifact per identity; the tool does not choose the most favorable score.

All evaluations in a joint result must share the locked split, evaluation/scorer revisions, sample population and category-count semantics. A different split/scorer is a separate analysis. Scorer and random-control references must identify existing owner evidence; the tool validates their shape and retains their hashes, but does not retrieve them.

## Output interpretation

`accuracy-throughput.json` retains complete supplied evaluations, per-category aggregates, validation references and one point per serving run. `accuracy-throughput.csv` contains those chart-ready trial points, reported accuracy/sample count, within-window throughput, run kind, offered rate, p50, evaluation/split identity and request hash. Isolated trials report serial completion rate and have no offered load; offered rate also remains null when an older report does not identify its run kind. Repeated speed trials that reuse an aggregate score are not independent accuracy repetitions. There is no pooled percentile, accuracy confidence interval or significance claim.

Within-window throughput describes the offered workload and observation window, not maximum sustainable capacity. The completed baseline pilot has only one rate and identical serving configurations. Even after matching accuracy arrives, it cannot establish a pruning tradeoff by itself. Final accuracy-versus-throughput conclusions still require the matched baseline/candidate experiments and owner evidence described in [methodology.md](methodology.md).
