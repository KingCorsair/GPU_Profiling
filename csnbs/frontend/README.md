# GPU Profiling frontend

A local benchmark dashboard built with React, TypeScript, Vite, and Recharts for
the project's measured results. The default view is the whole-project overview,
an interactive image-token demo, and an explicitly illustrative accuracy/throughput
explorer. The default sharing build has a browser-only image explanation. A separate
live build adds the protected model Playground. The real October and September
studies and searchable archive retain their saved evidence.

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

For the public sharing copy, run `npm --prefix csnbs/frontend run build:share`
from the repository root and host **`csnbs/frontend/dist-share/`**. The normal
`dist/` and development server contain the full local evidence and are not the
sharing output. The sharing build scans the referenced assets and downloads for
the owner's name, omits whole affected downloads, and marks them as local-only
in the UI. It keeps GitHub/source commit references, saved metrics, and all
three views. Downloads that remain preserve their original bytes and hashes.
An unexpected name in visible presentation data fails the export. Unreferenced
files and stale files from earlier sharing builds are not carried forward.

The page requests `noindex`; this is not access control or an anonymity guarantee.
Project results and source commits can still identify the project. Original
records under `results/` are never rewritten to remove historical folder names.

Shared snapshot: [GPU Profiling Results](https://gpu-profiling-results.vercel.app).
Anyone with the link can view it without signing in. Share this production
address; Vercel's individual deployment URLs can contain the account name.
Local changes do not update the hosted copy until rebuilt and deployed.

### Update the Vercel snapshot

Run from this `csnbs/frontend` directory with the Vercel CLI signed in:

```bash
npm run build:vercel
vercel deploy --prebuilt --prod
```

`build:vercel` runs the verified sharing build, then copies only `dist-share/`
into `.vercel/output/static/` using Vercel's Build Output API. It also sets an
`X-Robots-Tag: noindex, nofollow` response header. The `--prebuilt` flag uploads
that prepared output rather than the repository's unfiltered evidence. Stop
if the build fails; do not deploy stale output. Do not substitute a plain
`vercel deploy` or enable automatic Git builds using the ordinary build command.

The ignored `.vercel/project.json` links this checkout to the
`gpu-profiling-results` project. On another checkout, first use
`vercel link --project gpu-profiling-results` and choose the same account.
The previous ChatGPT Sites snapshot remains online at its existing address;
this deployment does not remove it.

## Image demo and Playground

The overview's **Try the image demo** button opens a side-by-side image preview.
Upload a JPG, PNG, or WebP, move the 1–576 token slider, choose a preset, and
toggle the 24 × 24 grid. The original image stays unchanged; the preview dims
unselected cells using a fixed, center-weighted illustrative pattern. It does
not run VisPruner, visualize actual retained indices or merged context, or
estimate accuracy or latency. The built-in picture is an original illustration,
not an evaluation example. Uploads remain in memory and carry into Playground
within the same tab. Refreshing clears them; they are never stored locally.

The default **showcase** build opens a browser-only explanation at `#/playground`.
It has no live client, automatic model checks, or GPU warm-up calls. It does not
pretend the illustrative image mask or invented preview values are recorded model
answers. A recorded-answer gallery requires actual captured outputs and is not part
of this live integration.

For the separate **live** build, run:

```bash
npm --prefix csnbs/frontend run dev:live
```

Open **http://127.0.0.1:5180/#/playground**. This starts the frontend and a local
Cloudflare-compatible gateway on loopback port 8787. It does not deploy anything
or start a GPU. Without a configured local model service, the interface reports
that live inference is unavailable and links to the saved studies.

Use [.env.example](.env.example) to configure an optional loopback model connection
in the ignored `.env.local`, then restart the local server. Leave the connection
unset to inspect the unavailable-service state without model requests.
The [gateway setup](gateway/README.md) documents Turnstile, request limits,
deployment settings, and the backend's required lifecycle controls. A compatible
model service must implement `/warmup` and `/generate` using the
[shared request and response types](src/live-contract.ts). The old fixed-setting
`/health` and `/infer` client remains in source for reference and its contract
tests, but is not the public live gateway.

`npm run build:live` produces the privacy-filtered **`dist-live-share/`** for the
team's live deployment. `npm run build:share` produces **`dist-share/`** for the
separate showcase deployment. `build:vercel` explicitly accepts only the showcase
build. Nothing in these commands publishes a site or creates a paid resource.

## Project overview and evidence routes

- `#/overview` (default): the research question, interactive image-token demo and illustrative results explorer,
  experiment architecture, evidence milestones, remaining questions, and recorded source.
- `#/playground`: a browser-only image-token explanation in the showcase build;
  image upload, question composer, verified model startup, and real baseline/pruned
  answer comparisons in the separate live build. The live image preview does not
  claim to show actual retained tokens.
- `#/october`: the repeated October study with real measurements.
- `#/september`: the exploratory September study with its original limitations.
- `#/archive`: recorded runs and reports. Selected runs, comparisons, and filters
  are encoded in the URL and survive reload; browser Back restores earlier selections.

The overview deliberately contains **invented UI fixtures**, isolated in
`src/demo-data.ts`. Its controls select a token budget, offered traffic, and a
hypothetical serving scenario. Every performance/accuracy chart and summary is
labeled synthetic. These values are neither measurements nor predictions and are
never inputs to evidence exports or the accuracy join. The real study views keep
using their verified data contracts. Replace the fixtures only after matched,
validated evidence exists; never combine fictional accuracy with real throughput
into an apparently measured point.

The project context was reconciled from the repository README, `AGENTS.md`,
[recorded measurement status](https://github.com/KingCorsair/GPU_Profiling/blob/31c92e89606908002decffdd89a41d68c76891f5/csnbs/MEASUREMENT_STATUS.md),
[October results](https://github.com/KingCorsair/GPU_Profiling/blob/31c92e89606908002decffdd89a41d68c76891f5/csnbs/measure/OCT01_RESULTS.md),
[October protocol](https://github.com/KingCorsair/GPU_Profiling/blob/31c92e89606908002decffdd89a41d68c76891f5/csnbs/measure/OCT01_PROTOCOL.md),
and accuracy requirements, plus serving status and the accuracy roadmap. These
measurement references are pinned to the recorded source revision. The older `DEMO_PLAN.md` explains
the intended final experience, but its expected outcomes and implementation
status are not current evidence. September and October show progress under
different protocols; they are not a controlled before/after optimization result.

No contributor attribution is displayed; the site presents the project itself.

## Present the results

- **October study** is the real-results page. It shows the seven-campaign collection
  totals, saved primary paired effects and confidence intervals, all primary
  trial throughput points, and the pending matched-accuracy handoff.
- Switch **1 request/s / 3 requests/s** to inspect individual trials; expand the
  trial table for outstanding work, drain, p50/p95 and raw-run downloads. p99
  remains unavailable. Neither rate is labeled a sustainable capacity ceiling.
- Download the required execution identities from the accuracy section. The
  [recorded accuracy requirements](https://github.com/KingCorsair/GPU_Profiling/blob/31c92e89606908002decffdd89a41d68c76891f5/csnbs/measure/accuracy-handoff.md) describe the locked split,
  scorer evidence and per-category aggregates required before joining scores.
- **September study** retains the September presentation described below.
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
- Archive filter menus match the dashboard theme. Use arrow keys or type to
  find an option, Enter to select, and Escape to close without changing the filter.
- The comparison identifies missing or mismatched conditions. It does not infer
  speedup, significance, or matched accuracy from two individual runs.
- Archive downloads preserve the exact original JSON and request JSONL bytes.
  Full SHA-256 identities are shown in each run's provenance panel.

## October presentation contract

`scripts/current-study.mjs` builds `public/data/current-study.json` from the
saved October collection index, canonical reports, indexed raw evidence and
primary accuracy handoff. It checks report/source hashes and matching campaign,
run and execution identities before publishing the presentation data. Saved
paired effect estimates and intervals are copied; the UI does not recompute
statistics. A missing or mismatched input fails the build rather than showing
partially trusted headline figures.

The browser validates the presentation contract before rendering. Original
reports, collection index, figure and pending accuracy handoff are downloadable
with their original bytes. This is an explicitly selected October study, not
a live "latest result" feed. Change the selection, validation and study context
together when a new campaign replaces it.

October, historical and archive views load independently: missing historical
presentation data does not prevent access to the October view or archive.
The historical figures and October findings remain separate studies.

## Historical evidence contract

The historical view is deliberately a fixed campaign: September 1, 2026,
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
versions 1 and 2. Nested portable exports retain their canonical campaign ID
from the nearest `campaign.json`, even when the destination folder is renamed;
directory names are the fallback. The historical campaign retains its explicit
source mapping. Model identity is never guessed
from a directory or token count. Values supplied only by historical campaign
metadata are marked as such.

Version 1 runs remain **Exploratory**. An integration campaign purpose, `INTEGRATION_ONLY.txt` marker, smoke
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

V2 run details count task categories and source datasets directly from the
hash-verified measured request records, including failures and excluding warmups.
Actual token ranges use only values recorded by the server for successful
requests. Output character lengths use only an optional explicit
`serverMetrics.output_characters` count (Unicode code points). Older records do
not preserve this count or the answer text, so those lengths remain unavailable.
Configured output caps and visual-token budgets never substitute for actual
lengths. These descriptions do not establish workload sensitivity or broad
dataset coverage.

Optional `results/campaigns/**/report.json` files add saved repeated capacity
screens when their run IDs, request hashes, and summaries match the indexed raw
evidence. The dashboard preserves report limitations and individual tested
rates; it does not interpolate a boundary or claim sustainable capacity. Reports
are available as original-byte downloads. Older reports without repeated
capacity screens retain an explicit unavailable message.

Isolated runs are sequential: they have no offered arrival rate or independent
arrival schedule. Their details show serial completion rate including drain,
omit scheduled-arrival statistics, and never display capacity screens derived
from a placeholder rate. Report arrival modes are checked against the raw runs.

Tests cover the historical evidence contract plus archive quality labels,
unknown metadata, null metrics, incompatible comparisons, campaign discovery,
partial results, request-record tampering, portable nested exports, raw workload
counts, missing output lengths, and report/evidence mismatches.

## Historical interpretation

The September campaign is exploratory: one run per point, 5–80 successful samples, unstable
p99, serialized batch-one serving, dirty source revision, and incomplete raw
configuration metadata. Fixed-duration runs change the question mix across rates;
the warmups were excluded by sequence rather than separated into a clean phase.
These limitations are visible on the page. The original downloaded plots
predate the workload-mix finding; retain the page's additional context when
presenting them.

Within the historical September study, no-load and optimized-serving comparisons
remain **Not measured**. The separate October isolated-HTTP study is available
in the current collection and archive. Validated
accuracy matched to these serving runs is unavailable, so the final
accuracy-throughput chart remains pending. Separate ScienceQA accuracy artifacts,
engineering diagnostics, and Gemma integration smoke runs are not mixed into it.

Stack references: [Vite setup](https://vite.dev/guide/) and
[Recharts sizing](https://recharts.github.io/en-US/guide/sizes/).
