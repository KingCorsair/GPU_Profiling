# Matplotlib benchmark charts

The requested comparison has four conditions: 576 tokens with no pruning, stock
VisPruner without concurrent load, stock VisPruner under open-loop load, and
optimized VisPruner under the same load. The first slot is the mathematical 0%
reference. Each other slot needs its own matching 576-token control.

The default story uses p50 latency reduction:

```text
100 × (control latency − pruned latency) / control latency
```

Positive means faster; negative means slower. `--percentile p95` or `p99` changes
the story metric. The supporting figure always shows p50, p95, p99, and successful
throughput against offered requests per second. It reads the harness's existing
summaries; it does not generate load, time requests, recalculate percentiles, or
perform statistical significance tests.

## Run locally

From the repository root, create the optional plotting-only environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r csnbs/requirements-plotting.txt
.venv/bin/python csnbs/plot_load_comparison.py
```

On RunPod, use the project's Docker environment, where `requirements.txt` already
pins Matplotlib. Do not install packages into a running pod:

```bash
python csnbs/plot_load_comparison.py
```

The script uses Matplotlib's noninteractive Agg backend and saves PNG and SVG
files directly. No GPU, model, display server, or image dataset is needed. See
[Matplotlib backends](https://matplotlib.org/stable/users/explain/figure/backends.html)
and [savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html).

Default outputs under `results/loadgen/`:

- `vispruner_under_load_story.png` / `.svg`: the four conditions.
- `vispruner_under_load.png` / `.svg`: the supporting load curves.
- `vispruner_under_load_sources.json`: plotted values, matching control IDs,
  source-file hashes, GPU/commit/settings, and interpretation notes.

Options:

```bash
.venv/bin/python csnbs/plot_load_comparison.py --rps 2 --percentile p95 -o /tmp/vispruner-p95
.venv/bin/python csnbs/plot_load_comparison.py path/to/campaign.json --tokens 128 --dpi 240
.venv/bin/python -m unittest csnbs.test_plot_load_comparison
```

The positional campaign path and `-o` remain supported from the previous SVG
plotter. `-o chart.svg` or `chart.png` selects the stem `chart`; both formats and
its story/source companions are generated. Defaults and relative `runPath`
values are resolved against the repository, independent of the working directory.
A custom campaign path itself is relative to the caller's working directory.

## Current evidence

The local September 1 campaign has stock 576/128-token runs at 0.5, 1, 1.5, 2,
and 3 RPS. At 3 RPS its p50 latency reduction is approximately **−6.9%**. The
corresponding p95 reduction is about −0.34%, while successful throughput rises
about 3.83%. These are exploratory observations, not a demonstrated regression
or a proven speedup: each point has one run and only 5–80 successful measured
requests. A stable p99 needs roughly 1,000 or more samples under the project rules.

The campaign describes a serialized batch-one server. These results do not
measure continuous batching. The raw runs record the A40, workload, and commit,
but several model/pruning settings are null and rely on campaign metadata. The
figures disclose this limitation and the dirty source checkout. Historical
artifacts are used as recorded; chart generation does not certify the harness.

There are no isolated no-load or optimized measurements in this campaign. Their
slots show **Run required**, with no bar height. A low offered RPS is never silently
relabelled as no load. No optimized result is invented or interpolated.

## Add future measurements

Use `results/loadgen/2026-09-01/vispruner_stock_ab_campaign.json` as the campaign
format. Keep this historical campaign intact; create a new campaign for new runs.
Each series declares `id`, `label`, `servingPath` (`stock` or `engineered`),
`visualTokenNum`, and `runs`. A run entry needs its real `runId` and `runPath`.
Cached numeric fields are optional; when supplied they must match `run.json`.

The script accepts saved `loadgen-run` schema-version-1 summaries. For an
under-load series (the default `loadCondition`), `runKind` must be `open-loop`,
and each rate can appear once. Include both 576-token and pruned measurements
for each serving path. Optimized series can declare their own `gitCommit` and
`batchSize`; their pruned and control runs must match each other. Otherwise the
optimized story slot remains unmeasured. The JSON's control run IDs make this
normalization explicit: it measures the benefit of pruning within each engine,
not the total engine improvement relative to stock.

For genuinely isolated measurements, use `loadCondition: "isolated"` and one
saved summary per token setting. That summary must have `runKind: "isolated"`,
with the same source/workload/hardware/benchmark/summary structure; an offered
arrival rate is not used. This is a future reporting input contract: the current
open-loop harness does not produce isolated runs. Do not relabel its output.

Use the same workload, GPU, model/checkpoint, output cap, important ratio, duration,
and warmup policy for each pair. Record known model/pruning settings in the raw
run's `benchmark` object; contradictions and known pair mismatches are rejected.
Repeat baselines to establish noise and randomize acquisition order. This plotter
intentionally accepts only one summary per condition/rate; repeated-run inference
belongs in the measurement layer before a reportable comparison is added.

The project's eventual accuracy-versus-throughput chart additionally needs
matched, validated accuracy results. This serving comparison does not substitute
for that deliverable.
