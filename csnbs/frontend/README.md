# GPU Profiling frontend

A local benchmark dashboard built with React, TypeScript, Vite, and Recharts in
Rithvik's measurement slice. The historical presentation is preserved alongside a
searchable archive of saved run evidence.

## Run

From the repository root:

```bash
npm --prefix csnbs/frontend ci
npm --prefix csnbs/frontend run dev
```

Open **http://127.0.0.1:5173**. No GPU, model, API server, or credentials are needed.
The server binds to loopback and uses a fixed port; a port conflict is reported
instead of silently changing the URL. Node 20.19+ or 22.12+ is required by Vite.

```bash
npm --prefix csnbs/frontend test
npm --prefix csnbs/frontend run build
npm --prefix csnbs/frontend run preview
```

`build` syncs the data, typechecks the app, and writes a static production build
to `dist/`. `preview` serves that build locally. Nothing is published or deployed.

## Present the results

- Scroll through the latency plot, throughput plot, comparison, and run notes.
- Switch **Median / p95 / p99** above the latency plot to change the displayed
  percentile. Captions update with the selected saved values at 3 requests/s.
- The sun/moon button changes the theme. It follows the system theme until a
  preference is chosen; only this preference is stored locally.
- **Run details** expands the numeric table and provenance.
- Links at the end download the original SVG/PNG figures and source data.
  Original downloads do not change with the percentile selection.
- **Run archive** browses saved runs by campaign, evidence status, model, GPU,
  date, run ID, or commit. Select a run's name to inspect timing and provenance;
  select two checkboxes to compare their saved summaries side by side.
- The comparison identifies missing or mismatched conditions. It does not infer
  speedup, significance, or matched accuracy from two individual runs.
- Archive downloads preserve the exact original JSON and request JSONL bytes.
  Full SHA-256 identities are shown in each run's provenance panel.

## Evidence contract

The default view is deliberately a fixed historical campaign: September 1, 2026,
NVIDIA A40, LLaVA stock 576 versus 128 tokens at 0.5, 1, 1.5, 2, and 3 offered
requests per second. The four-condition story is fixed to **p50 at 3 req/s**,
independent of the latency plot's percentile selection.

`scripts/sync-data.mjs` reads the repository's existing
`results/loadgen/vispruner_under_load_sources.json`, checks its campaign hash and
all ten raw-run SHA-256 hashes, verifies the copied metrics against saved raw
summaries, and copies only the selected evidence and four original plots into
`public/`. The absolute author home path in `campaignPath` becomes a relative
repository path. Original run JSON values are preserved. Generated copies,
dependencies, and build output are ignored by Git.

The frontend does not generate load, measure time, compute percentiles, or infer
statistical significance. New archive evidence is not automatically merged into
the historical comparison. To replace that presentation with a future campaign,
update the source selection, contract, visible study context, and validation
cases together. If the existing plot source is stale or was generated
with another story rate, regenerate its defaults from the repository root:

```bash
.venv/bin/python csnbs/plot_load_comparison.py
```

See `../plotting.md` for reporting environment setup. The dev/build scripts stop
on stale evidence rather than serving mismatched numbers. The client validates
the complete paired-rate contract before rendering and offers a retry screen if
the data is missing or incompatible. Tests cover valid historical data, missing
pairs, duplicate rates, nonfinite metrics, wrong story configuration, and filled
missing experiments.

## Run archive contract

`scripts/archive.mjs` scans `results/loadgen/**/run.json` and
`results/campaigns/**/run.json` at data-sync/build time. It supports manifest
versions 1 and 2. Campaign-directory names group new trials; the historical
campaign retains its explicit source mapping. Model identity is never guessed
from a directory or token count. Values supplied only by historical campaign
metadata are marked as such.

Version 1 runs remain **Exploratory**. An `INTEGRATION_ONLY.txt` marker, smoke
run kind, or fake server marks a run **Integration only**. Version 2 quality
reasons remain visible; successful complete run checks are labeled **Run checks
passed**, which does not establish campaign significance or sustainable
capacity. Other V2 runs are labeled **Needs review**. Null percentiles remain
unavailable and their saved sample-limit reasons are retained. Server service
and queue measurements are distinct from client timing; within-window and
including-drain throughput are shown separately when saved.

V2 request records must match their manifest SHA-256. Unsupported schemas,
inconsistent request counts, duplicate evidence, corrupt request hashes, and
orphaned partial manifests appear under **Files needing review**, outside the
run table. The adapter neither recalculates metrics nor repairs source data.
The archive is a static snapshot; rerun `npm run sync-data` (and refresh the page)
or rebuild after saving new results. Synthetic records appear only in automated
tests, never in the generated dashboard data.

Tests cover the historical evidence contract plus archive quality labels,
unknown metadata, null metrics, incompatible comparisons, campaign discovery,
partial results, and request-record tampering.

## Interpretation

The campaign is exploratory: one run per point, 5–80 successful samples, unstable
p99, serialized batch-one serving, dirty source revision, and incomplete raw
configuration metadata. Fixed-duration runs change the question mix across rates;
the warmups were excluded by sequence rather than separated into a clean phase.
These limitations are visible on the page. The original downloaded plots
predate the workload-mix finding; retain the page's additional context when
presenting them.

No-load and optimized-serving comparisons remain **Not measured**. Validated
accuracy matched to these serving runs is unavailable, so the final
accuracy-throughput chart remains pending. Separate ScienceQA accuracy artifacts,
engineering diagnostics, and Gemma integration smoke runs are not mixed into it.

Stack references: [Vite setup](https://vite.dev/guide/) and
[Recharts sizing](https://recharts.github.io/en-US/guide/sizes/).
