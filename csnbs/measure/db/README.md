# Benchmark result storage

Raw files are the evidence source. PostgreSQL is a queryable index populated **after** collection; a database outage never stops a GPU experiment. The importer never changes files or recomputes latency, percentiles, throughput, or statistical significance.

From `csnbs/measure`, start an isolated local database:

```sh
export BENCHMARK_DB_PASSWORD='choose-a-local-password'
docker compose up -d --wait
export DATABASE_URL='postgresql://benchmark:choose-a-local-password@127.0.0.1:5433/benchmark'
npm run ingest -- --migrate-only
```

URL-encode reserved characters in the connection URL password. Alternatively set `DATABASE_URL` to the project's shared PostgreSQL service. The Compose listener binds only to localhost and uses a named volume. `docker compose down` stops the service without deleting results; do not add `-v` unless the stored database is disposable.

```sh
# Validation and hashing need no database.
npm run ingest -- ../../results/loadgen/2026-09-12/01-18-23Z_rps-0.5_8d4eaabf --validate-only
# Import a run directory (or its run.json).
npm run ingest -- ../../results/loadgen/2026-09-12/01-18-23Z_rps-0.5_8d4eaabf
# Import a finalized campaign and its linked runs in a single transaction.
npm run ingest -- /path/to/campaign.json
```

`DATABASE_URL` is an environment variable, never a stored run field. Do not commit credentials. The importer applies the namespaced `benchmark` schema automatically and checks the migration checksum. A changed applied migration fails; future schema changes need a new numbered migration.

## Input and identity

- `run.json` supports `schema: "loadgen-run"`, manifest versions 1 and 2. `requests: {file, schemaVersion, count}` supports request versions 1, 2, and 3. Each request must have a unique nonnegative `sequence` within its phase; optional `requestId` must also be unique within its run. An internal `record_key` encodes `[phase, sequence]`, preserving a null phase for legacy records while allowing warmup and measurement both to start at sequence zero. Missing timing/identity fields stay null; the full raw record is retained.
- Optional `resourceSamples: {file, count}` points to JSONL objects. `recordedAtUtc` is optional; metrics and units remain exactly as supplied in `raw`. Optional `cpuLoad` objects follow [the CPU-load contract](../contracts.md#optional-cpu-load-observations): present values are validated and retained, while missing historical fields stay absent. Unreferenced files are not silently imported.
- Artifact references stay within the run directory, including resolved symlinks. Each referenced file's SHA-256 and byte length are stored. Version 2 request/resource descriptors must supply a matching SHA-256; legacy descriptors may omit it, but any declared hash is checked before the first import. The run identity also hashes the ordered artifact inventory, so additions, removals, renames, and whitespace changes are detectable.
- A campaign uses `schema: "loadgen-campaign"`, version 1 or 2, `campaignId`, and `trials: [{trialId, runDirectory, ...}]`. Run directories resolve relative to the campaign file; absolute paths are also accepted. An optional trial `runId` must match `run.json`. A failed/unexecuted trial may have a null/missing directory and remains a trial with no run. Extra fields, including `spec`, `schedule`, statuses, and reasons, are retained.

Only import **finalized campaign snapshots**. The importer rejects a campaign lock, any running/active campaign, trial or attempt, and a manifest that changes while its files are read. A leftover lock needs verified runner cleanup; the importer never removes it. Stopped failed/interrupted snapshots can retain pending trials. During collection the campaign status may evolve; once imported, the exact campaign bytes are immutable. A changed snapshot with the same campaign ID is rejected. Resume collection before importing, or use a distinct explicitly named snapshot ID; never rewrite raw evidence to evade a conflict.

Identical reimport is a no-op. A changed artifact under an existing run ID or a changed campaign under an existing campaign ID is rejected. The entire import rolls back on conflict or database failure, including newly inserted runs preceding the failure. Run ingestion and later campaign association are supported because trials reference independent runs. Imports serialize with a transaction advisory lock to protect concurrent duplicate imports.

The schema retains `campaigns`, `trials`, `runs`, `artifacts`, `request_outcomes`, and `resource_samples`. Time, run/phase, workload, trial-run, and request identity indexes support retrieval. Full manifests preserve effective settings, source provenance, workload identity, summaries, and quality labels without flattening away future fields.

## Retrieval

```sh
psql "$DATABASE_URL" -v left_run='run-A-id' -v right_run='run-B-id' -f db/comparison.sql
```

The query retrieves original summaries and provenance for two runs plus record counts and hashes. It does not establish that those runs are scientifically comparable. Use the validated report for phase selection, matching controls, nearest-rank percentiles, sample-size gates, uncertainty, and interpretation. Legacy phase values remain null: earlier runs mixed warmup records with measurement records while their saved summaries excluded a prefix, so SQL counts of all saved rows are not automatically measured sample counts.

To retrieve a campaign with its original run summaries:

```sql
SELECT t.campaign_id, t.trial_id, t.run_id, t.manifest AS trial,
       r.manifest->'summary' AS summary, r.bundle_sha256
FROM benchmark.trials AS t
LEFT JOIN benchmark.runs AS r USING (run_id)
WHERE t.campaign_id = 'your-campaign-id'
ORDER BY t.trial_id;
```

## Validation

```sh
npm run db:test
DATABASE_URL='postgresql://...' npm run db:test
```

Without a database, file validation covers historical runs, v2 records/resources, unsupported versions, malformed JSONL, count/identity errors, escaping paths, and campaign links. The PostgreSQL test requires `DATABASE_URL`; it imports a campaign twice, verifies database/file summary equality and nullable legacy values, rejects changed raw bytes, checks transaction rollback after both an identity conflict and a real SQL constraint failure, and proves recovery. It cleans up only its random test IDs and leaves the schema intact. CI should supply a PostgreSQL service so the integration test cannot be skipped there.
