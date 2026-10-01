import { createHash } from "node:crypto";
import { readFile, realpath, stat } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import pg from "pg";

type JsonObject = Record<string, unknown>;
type Artifact = {
  relativePath: string;
  kind: "manifest" | "requests" | "resources";
  sha256: string;
  byteLength: number;
};
export type RunBundle = {
  runId: string;
  manifest: JsonObject;
  requests: JsonObject[];
  resources: JsonObject[];
  artifacts: Artifact[];
  sha256: string;
};
export type ImportBundle = {
  campaign: { manifest: JsonObject; sha256: string; trials: JsonObject[] } | null;
  runs: RunBundle[];
};
export type ImportResult = { insertedRuns: number; existingRuns: number; campaignId: string | null };

function object(value: unknown, label: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonObject;
}

function optionalObject(value: unknown): JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as JsonObject : {};
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`${label} must be a nonempty string`);
  return value;
}

function nullableString(value: unknown, label: string): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") throw new Error(`${label} must be a string or null`);
  return value;
}

function nullableNumber(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be finite or null`);
  return value;
}

function nonnegativeInteger(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > 2147483647) {
    throw new Error(`${label} must be a nonnegative 32-bit integer`);
  }
  return value;
}

function nullableDate(value: unknown, label: string): string | null {
  const text = nullableString(value, label);
  if (text !== null && (!/^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(text) || !Number.isFinite(Date.parse(text)))) {
    throw new Error(`${label} must be an ISO timestamp with a timezone or null`);
  }
  return text;
}

function sha256(bytes: Buffer | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseObject(bytes: Buffer, label: string): JsonObject {
  return object(JSON.parse(bytes.toString("utf8")), label);
}

function parseJsonl(bytes: Buffer, label: string): JsonObject[] {
  const lines = bytes.toString("utf8").split(/\r?\n/);
  if (lines.at(-1) === "") lines.pop();
  return lines.map((line, index) => {
    try { return object(JSON.parse(line), `${label}:${index + 1}`); }
    catch (error) { throw new Error(`Invalid JSONL at ${label}:${index + 1}`, { cause: error }); }
  });
}

function checkVersion(manifest: JsonObject, schema: string, label: string): void {
  if (manifest.schema !== schema || ![1, 2].includes(manifest.schemaVersion as number)) {
    throw new Error(`${label}: expected ${schema} schemaVersion 1 or 2`);
  }
}

async function readArtifact(directory: string, filename: string, kind: Artifact["kind"]): Promise<{ bytes: Buffer; artifact: Artifact }> {
  if (isAbsolute(filename) || filename.split(/[\\/]/).includes("..") || filename.includes("\\")) {
    throw new Error(`Artifact path must stay inside the run directory: ${filename}`);
  }
  const root = await realpath(directory);
  const path = await realpath(join(root, filename));
  const relativePath = relative(root, path);
  if (relativePath.startsWith(`..${sep}`) || relativePath === ".." || isAbsolute(relativePath)) {
    throw new Error(`Artifact symlink escapes the run directory: ${filename}`);
  }
  const bytes = await readFile(path);
  return { bytes, artifact: { relativePath: filename, kind, sha256: sha256(bytes), byteLength: bytes.length } };
}

const requestNumbers = [
  "scheduledAtMs", "sentAtMs", "completedAtMs", "latencyMs", "dispatchLatenessMs", "plannedToCompleteMs",
] as const;

function validateRequests(requests: JsonObject[]): void {
  const sequences = new Set<string>();
  const requestIds = new Set<string>();
  for (const request of requests) {
    const sequence = nonnegativeInteger(request.sequence, "request.sequence");
    const phase = nullableString(request.phase, "request.phase");
    const recordKey = JSON.stringify([phase, sequence]);
    if (sequences.has(recordKey)) throw new Error(`Duplicate request sequence ${sequence} in phase ${phase ?? "unknown"}`);
    sequences.add(recordKey);
    const requestId = nullableString(request.requestId, "request.requestId");
    if (requestId !== null && requestIds.has(requestId)) throw new Error(`Duplicate request ID ${requestId}`);
    if (requestId !== null) requestIds.add(requestId);
    for (const field of requestNumbers) nullableNumber(request[field], `request.${field}`);
    for (const field of ["phase", "category", "error"]) nullableString(request[field], `request.${field}`);
    if (request.questionId !== undefined && request.questionId !== null && typeof request.questionId !== "string" && typeof request.questionId !== "number") {
      throw new Error("request.questionId must be a string, number, or null");
    }
    const status = nullableNumber(request.status, "request.status");
    if (status !== null && (!Number.isInteger(status) || status < 100 || status > 599)) throw new Error("request.status is not an HTTP status");
  }
}

/** Validate and hash files without touching the database or changing raw artifacts. */
export async function readRunBundle(runDirectory: string): Promise<RunBundle> {
  const raw = await readArtifact(runDirectory, "run.json", "manifest");
  const manifest = parseObject(raw.bytes, "run.json");
  checkVersion(manifest, "loadgen-run", "run.json");
  const runId = requiredString(manifest.runId, "runId");
  nullableDate(manifest.recordedAtUtc, "recordedAtUtc");
  const descriptor = object(manifest.requests, "requests descriptor");
  if (![1, 2, 3].includes(descriptor.schemaVersion as number)) throw new Error("Unsupported requests schemaVersion");
  const requestFile = requiredString(descriptor.file, "requests.file");
  if (requestFile === "run.json") throw new Error("requests.file must differ from run.json");
  const requestRaw = await readArtifact(runDirectory, requestFile, "requests");
  const requests = parseJsonl(requestRaw.bytes, requestFile);
  if (requests.length !== nonnegativeInteger(descriptor.count, "requests.count")) throw new Error("requests.count does not match the raw file");
  validateRequests(requests);
  const artifacts = [raw.artifact, requestRaw.artifact];
  let resources: JsonObject[] = [];
  if (manifest.resourceSamples !== undefined && manifest.resourceSamples !== null) {
    const resourceDescriptor = object(manifest.resourceSamples, "resourceSamples descriptor");
    const resourceFile = requiredString(resourceDescriptor.file, "resourceSamples.file");
    if (artifacts.some((entry) => entry.relativePath === resourceFile)) throw new Error("Resource file duplicates another artifact");
    const resourceRaw = await readArtifact(runDirectory, resourceFile, "resources");
    resources = parseJsonl(resourceRaw.bytes, resourceFile);
    if (resources.length !== nonnegativeInteger(resourceDescriptor.count, "resourceSamples.count")) throw new Error("resourceSamples.count does not match the raw file");
    for (const sample of resources) nullableDate(sample.recordedAtUtc, "resource.recordedAtUtc");
    artifacts.push(resourceRaw.artifact);
  }
  // Include names and lengths as well as bytes' hashes; missing/added artifacts change identity.
  const bundleHash = sha256(JSON.stringify(artifacts));
  return { runId, manifest, requests, resources, artifacts, sha256: bundleHash };
}

/** Accept a run directory, run.json, or a finalized campaign.json snapshot. */
export async function readImportBundle(inputPath: string): Promise<ImportBundle> {
  const input = resolve(inputPath);
  if ((await stat(input)).isDirectory()) return { campaign: null, runs: [await readRunBundle(input)] };
  const bytes = await readFile(input);
  const manifest = parseObject(bytes, input);
  if (manifest.schema === "loadgen-run") {
    if (input !== join(dirname(input), "run.json")) throw new Error("Run manifest must be named run.json");
    return { campaign: null, runs: [await readRunBundle(dirname(input))] };
  }
  checkVersion(manifest, "loadgen-campaign", input);
  requiredString(manifest.campaignId, "campaignId");
  if (!Array.isArray(manifest.trials)) throw new Error("Campaign trials must be an array");
  const trials = manifest.trials.map((trial) => object(trial, "trial"));
  const trialIds = new Set<string>();
  const runs = new Map<string, RunBundle>();
  const importedTrials: JsonObject[] = [];
  for (const trial of trials) {
    const trialId = requiredString(trial.trialId, "trialId");
    if (trialIds.has(trialId)) throw new Error(`Duplicate trial ID ${trialId}`);
    trialIds.add(trialId);
    const runDirectory = nullableString(trial.runDirectory, "trial.runDirectory");
    let runId: string | null = null;
    if (runDirectory !== null) {
      const run = await readRunBundle(resolve(dirname(input), runDirectory));
      if (trial.runId !== undefined && trial.runId !== run.runId) throw new Error(`Trial ${trialId} runId differs from run.json`);
      const existing = runs.get(run.runId);
      if (existing !== undefined && existing.sha256 !== run.sha256) throw new Error(`Conflicting duplicate run ID ${run.runId}`);
      runs.set(run.runId, run);
      runId = run.runId;
    } else if (trial.runId !== undefined && trial.runId !== null) {
      throw new Error(`Trial ${trialId} has runId but no runDirectory`);
    }
    importedTrials.push({ ...trial, runId });
  }
  return { campaign: { manifest, sha256: sha256(bytes), trials: importedTrials }, runs: [...runs.values()] };
}

/** Apply only this package's namespaced schema, with a checked migration hash. */
export async function migrate(client: pg.Client): Promise<void> {
  const sql = await readFile(new URL("../db/001_initial.sql", import.meta.url), "utf8");
  await client.query("BEGIN");
  try {
    await client.query("SELECT pg_advisory_xact_lock(724062601)");
    await client.query("CREATE SCHEMA IF NOT EXISTS benchmark");
    await client.query(`CREATE TABLE IF NOT EXISTS benchmark.schema_migrations (
      version integer PRIMARY KEY, sha256 char(64) NOT NULL, applied_at timestamptz NOT NULL DEFAULT now()
    )`);
    const existing = await client.query<{ sha256: string }>("SELECT sha256 FROM benchmark.schema_migrations WHERE version = 1");
    if (existing.rowCount) {
      if (existing.rows[0]!.sha256 !== sha256(sql)) throw new Error("Migration 1 checksum differs; create a new migration instead of changing an applied schema");
    } else {
      await client.query(sql);
      await client.query("INSERT INTO benchmark.schema_migrations(version, sha256) VALUES (1, $1)", [sha256(sql)]);
    }
    await client.query("COMMIT");
  } catch (error) { await client.query("ROLLBACK"); throw error; }
}

async function insertRun(client: pg.Client, run: RunBundle): Promise<boolean> {
  const m = run.manifest;
  const config = optionalObject(m.config);
  const source = optionalObject(m.source);
  const workload = optionalObject(m.workload);
  if (source.gitDirty !== undefined && source.gitDirty !== null && typeof source.gitDirty !== "boolean") throw new Error("source.gitDirty must be boolean or null");
  const inserted = await client.query(`INSERT INTO benchmark.runs
    (run_id, schema_version, run_kind, recorded_at, workload_id, git_commit, git_dirty, offered_rps, request_count, manifest, bundle_sha256)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) ON CONFLICT (run_id) DO NOTHING RETURNING run_id`, [
    run.runId, m.schemaVersion, nullableString(m.runKind, "runKind"), nullableDate(m.recordedAtUtc, "recordedAtUtc"),
    nullableString(workload.id, "workload.id"), nullableString(source.gitCommit, "source.gitCommit"), source.gitDirty ?? null,
    nullableNumber(config.requestsPerSecond, "config.requestsPerSecond"), run.requests.length, JSON.stringify(m), run.sha256,
  ]);
  if (!inserted.rowCount) {
    const existing = await client.query<{ bundle_sha256: string }>("SELECT bundle_sha256 FROM benchmark.runs WHERE run_id = $1", [run.runId]);
    if (existing.rows[0]?.bundle_sha256 !== run.sha256) throw new Error(`Conflicting raw artifacts for run ${run.runId}`);
    return false;
  }
  for (const artifact of run.artifacts) {
    await client.query(`INSERT INTO benchmark.artifacts(run_id, relative_path, artifact_kind, sha256, byte_length)
      VALUES ($1,$2,$3,$4,$5)`, [run.runId, artifact.relativePath, artifact.kind, artifact.sha256, artifact.byteLength]);
  }
  // Bulk insert after collection; database latency cannot perturb the measured request schedule.
  const outcomes = run.requests.map((request) => {
    const status = nullableNumber(request.status, "status");
    return {
      run_id: run.runId, record_key: JSON.stringify([request.phase ?? null, request.sequence]), sequence: request.sequence, request_id: request.requestId ?? null,
      phase: request.phase ?? null, question_id: request.questionId == null ? null : String(request.questionId), category: request.category ?? null,
      scheduled_at_ms: request.scheduledAtMs ?? null, sent_at_ms: request.sentAtMs ?? null, completed_at_ms: request.completedAtMs ?? null,
      latency_ms: request.latencyMs ?? null, dispatch_lateness_ms: request.dispatchLatenessMs ?? null, planned_to_complete_ms: request.plannedToCompleteMs ?? null,
      http_status: status, error: request.error ?? null,
      successful: request.error !== undefined ? request.error === null && status !== null && status >= 200 && status < 300 : null,
      raw: request,
    };
  });
  if (outcomes.length) await client.query(`INSERT INTO benchmark.request_outcomes
    SELECT * FROM jsonb_populate_recordset(NULL::benchmark.request_outcomes, $1::jsonb)`, [JSON.stringify(outcomes)]);
  if (run.resources.length) await client.query(`INSERT INTO benchmark.resource_samples
    SELECT * FROM jsonb_populate_recordset(NULL::benchmark.resource_samples, $1::jsonb)`, [JSON.stringify(run.resources.map((sample, index) => ({
    run_id: run.runId, sample_index: index, recorded_at: sample.recordedAtUtc ?? null, raw: sample,
  })))]);
  return true;
}

/** One transaction for the whole import. An identity conflict rolls back every new row. */
export async function ingestBundle(client: pg.Client, bundle: ImportBundle): Promise<ImportResult> {
  const campaignId = bundle.campaign === null ? null : requiredString(bundle.campaign.manifest.campaignId, "campaignId");
  const result: ImportResult = { insertedRuns: 0, existingRuns: 0, campaignId };
  await client.query("BEGIN");
  try {
    // Serializes ingestion, including concurrent same-ID imports with different content.
    await client.query("SELECT pg_advisory_xact_lock(724062602)");
    const campaign = bundle.campaign;
    if (campaign !== null) {
      await client.query(`INSERT INTO benchmark.campaigns(campaign_id, schema_version, manifest, manifest_sha256)
        VALUES ($1,$2,$3,$4) ON CONFLICT (campaign_id) DO NOTHING`, [campaignId, campaign.manifest.schemaVersion, JSON.stringify(campaign.manifest), campaign.sha256]);
      const existing = await client.query<{ manifest_sha256: string }>("SELECT manifest_sha256 FROM benchmark.campaigns WHERE campaign_id = $1", [campaignId]);
      if (existing.rows[0]?.manifest_sha256 !== campaign.sha256) throw new Error(`Conflicting campaign manifest for ${campaignId}`);
    }
    for (const run of bundle.runs) {
      if (await insertRun(client, run)) result.insertedRuns++;
      else result.existingRuns++;
    }
    if (campaign !== null) {
      for (const trial of campaign.trials) {
        const stored = await client.query<{ manifest: JsonObject }>(`INSERT INTO benchmark.trials(campaign_id, trial_id, run_id, manifest)
          VALUES ($1,$2,$3,$4) ON CONFLICT (campaign_id, trial_id) DO NOTHING RETURNING manifest`, [campaignId, trial.trialId, trial.runId, JSON.stringify(trial)]);
        if (!stored.rowCount) {
          const matches = await client.query("SELECT 1 FROM benchmark.trials WHERE campaign_id=$1 AND trial_id=$2 AND manifest=$3::jsonb", [campaignId, trial.trialId, JSON.stringify(trial)]);
          if (!matches.rowCount) throw new Error(`Conflicting trial ${String(trial.trialId)}`);
        }
      }
    }
    await client.query("COMMIT");
    return result;
  } catch (error) { await client.query("ROLLBACK"); throw error; }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  if (args.includes("--help")) {
    console.log("Usage: DATABASE_URL=... npm run ingest -- <run-directory|run.json|campaign.json> [--validate-only]\n       DATABASE_URL=... npm run ingest -- --migrate-only\nCampaign imports are immutable finalized snapshots. Raw files are never modified.");
    return;
  }
  const flags = new Set(["--validate-only", "--migrate-only"]);
  if (args.some((arg) => arg.startsWith("--") && !flags.has(arg))) throw new Error("Unknown flag; use --help");
  const inputs = args.filter((arg) => !arg.startsWith("--"));
  if (args.includes("--migrate-only") && (inputs.length || args.includes("--validate-only"))) throw new Error("--migrate-only cannot be combined with an input or --validate-only");
  if (!args.includes("--migrate-only") && inputs.length !== 1) throw new Error("Expected one input path; use --help");
  const bundle = inputs.length ? await readImportBundle(inputs[0]!) : null;
  if (args.includes("--validate-only")) {
    console.log(JSON.stringify({ validated: true, campaignId: bundle?.campaign?.manifest.campaignId ?? null, runs: bundle?.runs.map((run) => ({ runId: run.runId, requests: run.requests.length, resources: run.resources.length, sha256: run.sha256 })) }, null, 2));
    return;
  }
  if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL is required (or use --validate-only)");
  const client = new pg.Client({ connectionString: process.env.DATABASE_URL, connectionTimeoutMillis: 10000 });
  await client.connect();
  try {
    await migrate(client);
    const result = bundle === null ? { migrated: true } : await ingestBundle(client, bundle);
    console.log(JSON.stringify(result, null, 2));
  } finally { await client.end(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error: unknown) => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
}
