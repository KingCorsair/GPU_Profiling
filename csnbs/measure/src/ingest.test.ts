import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import pg from "pg";
import { ingestBundle, migrate, readImportBundle, readRunBundle } from "./ingest.js";

type Fixture = { directory: string; manifest: Record<string, unknown>; requests: Record<string, unknown>[] };
async function fixture(root: string, name: string, version = 2): Promise<Fixture> {
  const directory = join(root, name);
  await mkdir(directory, { recursive: true });
  const requests = [{
    sequence: 0, requestId: version === 2 ? `${name}:0` : undefined,
    phase: version === 2 ? "measurement" : undefined,
    questionId: version === 2 ? "question-7" : undefined,
    scheduledAtMs: 100, sentAtMs: 101, completedAtMs: 110, latencyMs: 9,
    dispatchLatenessMs: 1, plannedToCompleteMs: 10, status: 200, error: null,
  }, {
    sequence: 1, requestId: version === 2 ? `${name}:1` : undefined,
    phase: version === 2 ? "measurement" : undefined,
    scheduledAtMs: 200, sentAtMs: 202, completedAtMs: 210, latencyMs: 8,
    dispatchLatenessMs: 2, plannedToCompleteMs: 10, status: null, error: "timeout",
  }];
  const manifest = {
    schema: "loadgen-run", schemaVersion: version, runId: name, runKind: "open-loop",
    recordedAtUtc: "2026-10-01T00:00:00.000Z", source: { gitCommit: "fixture", gitDirty: false },
    config: { requestsPerSecond: 10 }, workload: { id: "sha256:fixture" },
    summary: { totalRequests: 2, successfulRequests: 1, failedRequests: 1, successfulThroughputRps: 5, successfulRequestLatencyMs: { p50: 9, p95: 9, p99: 9 } },
    requests: { file: "requests.jsonl", schemaVersion: version === 2 ? 3 : 2, count: 2 },
    ...(version === 2 ? { resourceSamples: { file: "resources.jsonl", count: 1 } } : {}),
  };
  await writeFile(join(directory, "run.json"), JSON.stringify(manifest));
  await writeFile(join(directory, "requests.jsonl"), requests.map((request) => JSON.stringify(request)).join("\n") + "\n");
  if (version === 2) await writeFile(join(directory, "resources.jsonl"), JSON.stringify({ recordedAtUtc: "2026-10-01T00:00:00.000Z", gpuMemoryMiB: 8192 }) + "\n");
  const result = { directory, manifest, requests };
  if (version === 2) await refreshHashes(result);
  return result;
}

async function refreshHashes(source: Fixture): Promise<void> {
  for (const key of ["requests", "resourceSamples"]) {
    const descriptor = source.manifest[key] as Record<string, unknown> | undefined;
    if (descriptor) descriptor.sha256 = "sha256:" + createHash("sha256").update(await readFile(join(source.directory, descriptor.file as string))).digest("hex");
  }
  await writeFile(join(source.directory, "run.json"), JSON.stringify(source.manifest));
}

async function temp(t: { after: (fn: () => Promise<void>) => void }): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "measure-ingest-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("legacy and V2 files retain their raw metadata, phases, hashes, and optional resources", async (t) => {
  const root = await temp(t);
  for (const version of [1, 2]) {
    const source = await fixture(root, `run-v${version}`, version);
    const bundle = await readRunBundle(source.directory);
    assert.deepEqual(bundle.manifest, source.manifest);
    assert.equal(bundle.requests[0]?.phase, version === 1 ? undefined : "measurement");
    assert.equal(bundle.resources.length, version === 1 ? 0 : 1);
    assert.equal(bundle.artifacts.length, version === 1 ? 2 : 3);
    assert.match(bundle.sha256, /^[0-9a-f]{64}$/);
    assert.deepEqual(await readImportBundle(join(source.directory, "run.json")), { campaign: null, runs: [bundle] });
    const before = bundle.sha256;
    await writeFile(join(source.directory, "run.json"), JSON.stringify(source.manifest, null, 2));
    assert.notEqual((await readRunBundle(source.directory)).sha256, before, "even whitespace changes raw artifact identity");
  }
});

test("all checked-in historical loadgen runs are readable without rewriting them", async () => {
  const { readdir } = await import("node:fs/promises");
  const root = new URL("../../../results/loadgen/", import.meta.url);
  let count = 0;
  for (const day of await readdir(root, { withFileTypes: true })) {
    if (!day.isDirectory() || !/^\d{4}-\d{2}-\d{2}$/.test(day.name)) continue;
    for (const run of await readdir(new URL(`${day.name}/`, root), { withFileTypes: true })) {
      if (!run.isDirectory()) continue;
      const directory = new URL(`${day.name}/${run.name}/`, root);
      try { await readFile(new URL("run.json", directory)); }
      catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") continue; throw error; }
      const { fileURLToPath } = await import("node:url");
      await readRunBundle(fileURLToPath(directory));
      count++;
    }
  }
  assert.ok(count > 0);
});

test("rejects unsupported manifests and request record versions", async (t) => {
  const root = await temp(t);
  const source = await fixture(root, "bad-version");
  for (const change of [{ schemaVersion: 99 }, { requests: { file: "requests.jsonl", schemaVersion: 99, count: 2 } }]) {
    await writeFile(join(source.directory, "run.json"), JSON.stringify({ ...source.manifest, ...change }));
    await assert.rejects(readRunBundle(source.directory), /schemaVersion/);
  }
});

test("V2 warmup and measurement may restart sequence zero with distinct request identities", async (t) => {
  const root = await temp(t);
  const source = await fixture(root, "phase-local");
  await writeFile(join(source.directory, "requests.jsonl"), [
    { ...source.requests[0], phase: "warmup", requestId: "phase-local:warmup:0" },
    { ...source.requests[0], phase: "measurement", requestId: "phase-local:measurement:0" },
  ].map((request) => JSON.stringify(request)).join("\n") + "\n");
  await refreshHashes(source);
  assert.equal((await readRunBundle(source.directory)).requests.length, 2);
});

test("rejects mismatched counts, duplicate identities, malformed JSONL, and invalid HTTP status", async (t) => {
  const root = await temp(t);
  const source = await fixture(root, "invalid-records");
  const request = source.requests[0]!;
  for (const [records, pattern] of [
    [[request], /count/],
    [[request, request], /Duplicate request sequence/],
    [[request, { ...request, sequence: 1 }], /Duplicate request ID/],
    [[request, { ...source.requests[1], status: 999 }], /HTTP status/],
    [[request, { ...source.requests[1], latencyMs: "ten" }], /finite/],
  ] as const) {
    await writeFile(join(source.directory, "requests.jsonl"), records.map((record) => JSON.stringify(record)).join("\n") + "\n");
    await refreshHashes(source);
    await assert.rejects(readRunBundle(source.directory), pattern);
  }
  await writeFile(join(source.directory, "requests.jsonl"), "{invalid}\n");
  await refreshHashes(source);
  await assert.rejects(readRunBundle(source.directory), /Invalid JSONL/);
});

test("first import rejects tampered request/resource bytes and missing V2 hashes", async (t) => {
  const root = await temp(t);
  for (const file of ["requests.jsonl", "resources.jsonl"]) {
    const source = await fixture(root, `tampered-${file}`);
    await writeFile(join(source.directory, file), (await readFile(join(source.directory, file), "utf8")) + "\n");
    await assert.rejects(readRunBundle(source.directory), /SHA-256 mismatch/);
  }
  const missing = await fixture(root, "missing-v2-hash");
  delete (missing.manifest.requests as Record<string, unknown>).sha256;
  await writeFile(join(missing.directory, "run.json"), JSON.stringify(missing.manifest));
  await assert.rejects(readRunBundle(missing.directory), /sha256 is required/);
});

test("artifact references cannot traverse directories or follow an escaping symlink", async (t) => {
  const root = await temp(t);
  const source = await fixture(root, "path-check");
  await writeFile(join(root, "outside.jsonl"), "{}\n");
  await symlink(join(root, "outside.jsonl"), join(source.directory, "escape.jsonl"));
  for (const file of ["../outside.jsonl", "/tmp/outside.jsonl", "escape.jsonl"]) {
    await writeFile(join(source.directory, "run.json"), JSON.stringify({ ...source.manifest, requests: { file, count: 1, schemaVersion: 2 } }));
    await assert.rejects(readRunBundle(source.directory), /inside|escapes/);
  }
});

test("campaigns resolve relative directories, preserve failed trials, and reject identity mismatches", async (t) => {
  const root = await temp(t);
  await fixture(root, "trial-run");
  const path = join(root, "campaign.json");
  const campaign = { schema: "loadgen-campaign", schemaVersion: 2, campaignId: "campaign", trials: [
    { trialId: "complete", runDirectory: "trial-run", status: "completed" },
    { trialId: "failed", runDirectory: null, status: "failed", error: "startup failed" },
  ] };
  await writeFile(path, JSON.stringify(campaign));
  const bundle = await readImportBundle(path);
  assert.equal(bundle.runs.length, 1);
  assert.equal(bundle.campaign?.trials[1]?.runId, null);
  assert.equal(bundle.campaign?.trials[0]?.runId, "trial-run");
  await writeFile(path, JSON.stringify({ ...campaign, trials: [{ ...campaign.trials[0], runId: "wrong" }] }));
  await assert.rejects(readImportBundle(path), /differs/);
  await writeFile(path, JSON.stringify({ ...campaign, trials: [campaign.trials[0], campaign.trials[0]] }));
  await assert.rejects(readImportBundle(path), /Duplicate trial/);
});

test("real Postgres: campaign import is idempotent, preserves values, and rolls back conflicts", {
  skip: !process.env.DATABASE_URL ? "Set DATABASE_URL to run real PostgreSQL integration" : false,
}, async (t) => {
  const root = await temp(t);
  const prefix = `ingest-test-${randomUUID()}`;
  const client = new pg.Client({ connectionString: process.env.DATABASE_URL!, connectionTimeoutMillis: 10000 });
  await client.connect();
  t.after(async () => {
    try {
      await client.query("DELETE FROM benchmark.trials WHERE campaign_id LIKE $1", [prefix + "%"]);
      for (const table of ["request_outcomes", "resource_samples", "artifacts", "runs"]) {
        await client.query(`DELETE FROM benchmark.${table} WHERE run_id LIKE $1`, [prefix + "%"]);
      }
      await client.query("DELETE FROM benchmark.campaigns WHERE campaign_id LIKE $1", [prefix + "%"]);
    } finally { await client.end(); }
  });
  await migrate(client);
  await migrate(client);
  const a = await fixture(root, prefix + "-a", 1);
  const b = await fixture(root, prefix + "-b");
  const path = join(root, "campaign.json");
  const campaign = { schema: "loadgen-campaign", schemaVersion: 2, campaignId: prefix, trials: [
    { trialId: "a", runDirectory: prefix + "-a", status: "completed" },
    { trialId: "b", runDirectory: prefix + "-b", status: "completed" },
  ] };
  await writeFile(path, JSON.stringify(campaign));
  const bundle = await readImportBundle(path);
  assert.deepEqual(await ingestBundle(client, bundle), { insertedRuns: 2, existingRuns: 0, campaignId: prefix });
  assert.deepEqual(await ingestBundle(client, bundle), { insertedRuns: 0, existingRuns: 2, campaignId: prefix });
  assert.equal((await client.query("SELECT count(*)::int AS count FROM benchmark.request_outcomes WHERE run_id LIKE $1", [prefix + "%"])).rows[0].count, 4);
  assert.equal((await client.query("SELECT count(*)::int AS count FROM benchmark.resource_samples WHERE run_id LIKE $1", [prefix + "%"])).rows[0].count, 1);
  const stored = await client.query("SELECT manifest FROM benchmark.runs WHERE run_id=$1", [prefix + "-a"]);
  assert.deepEqual(stored.rows[0].manifest.summary, a.manifest.summary, "database reports use the exact file summary");
  const outcome = (await client.query("SELECT phase, request_id, latency_ms, successful FROM benchmark.request_outcomes WHERE run_id=$1 AND sequence=0", [prefix + "-a"])).rows[0];
  assert.deepEqual(outcome, { phase: null, request_id: null, latency_ms: 9, successful: true });
  const resource = (await client.query("SELECT raw FROM benchmark.resource_samples WHERE run_id=$1", [prefix + "-b"])).rows[0].raw;
  assert.equal(resource.gpuMemoryMiB, 8192);

  await writeFile(join(b.directory, "run.json"), JSON.stringify(b.manifest, null, 2));
  const changed = await readRunBundle(b.directory);
  await assert.rejects(ingestBundle(client, { campaign: null, runs: [changed] }), /Conflicting raw artifacts/);

  const newRun = await fixture(root, prefix + "-rollback");
  const failedCampaign = { ...campaign, campaignId: prefix + "-rollback", trials: [
    { trialId: "new", runDirectory: prefix + "-rollback" },
    { trialId: "changed", runDirectory: prefix + "-b" },
  ] };
  await writeFile(path, JSON.stringify(failedCampaign));
  await assert.rejects(ingestBundle(client, await readImportBundle(path)), /Conflicting raw artifacts/);
  assert.equal((await client.query("SELECT 1 FROM benchmark.runs WHERE run_id=$1", [prefix + "-rollback"])).rowCount, 0);
  assert.equal((await client.query("SELECT 1 FROM benchmark.campaigns WHERE campaign_id=$1", [prefix + "-rollback"])).rowCount, 0);
  assert.deepEqual(await ingestBundle(client, await readImportBundle(newRun.directory)), { insertedRuns: 1, existingRuns: 0, campaignId: null }, "a failed transaction does not poison the connection");

  const invalid = await readRunBundle((await fixture(root, prefix + "-sql-rollback")).directory);
  invalid.requests.push(invalid.requests[0]!); // Exercise an actual SQL constraint failure after run insertion.
  await assert.rejects(ingestBundle(client, { campaign: null, runs: [invalid] }), /duplicate key/);
  assert.equal((await client.query("SELECT 1 FROM benchmark.runs WHERE run_id=$1", [invalid.runId])).rowCount, 0);
});
