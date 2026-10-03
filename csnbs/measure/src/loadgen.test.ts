import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test, type TestContext } from "node:test";
import {
  buildWorkloadOrder, createJournal, describeWorkload, loadDataset, parseLoadConfig,
  percentile, runLoad, summarizeRun, summarizePercentiles, validateConfig,
  type DatasetPayload, type LoadConfig, type RequestResult,
} from "./loadgen.js";

const payloads: DatasetPayload[] = Array.from({ length: 12 }, (_, index) => ({
  image_b64: "aW1hZ2U=", question: `Question ${index}`, workloadIndex: index,
  questionId: `question-${index}`, category: ["ocr", "counting", "spatial"][Math.floor(index / 4)]!, sourceDataset: "fixture",
}));
function config(endpoint = "http://127.0.0.1:1/infer", overrides: Partial<LoadConfig> = {}): LoadConfig {
  return { endpoint, requestsPerSecond: 100, durationSeconds: 0.12, measuredRequests: 12,
    timeoutMs: 1000, datasetPath: "fixture.json", warmupRequests: 0, seed: 7, settleMs: 0,
    runKind: "open-loop", outputDir: tmpdir(), runDeadlineMs: null, ...overrides };
}
async function service(t: TestContext, handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<string> {
  const server = createServer(handler);
  await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  t.after(() => new Promise<void>((resolve, reject) => { server.close((error) => error ? reject(error) : resolve()); server.closeAllConnections(); }));
  const address = server.address(); assert.ok(address && typeof address !== "string");
  return `http://127.0.0.1:${address.port}/infer`;
}
function answer(req: IncomingMessage, res: ServerResponse, metrics?: Record<string, unknown>): void {
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify({ answer: "ok", request_id: req.headers["x-request-id"], ...(metrics ? { metrics } : {}) }));
}
async function temp(t: TestContext): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), "loadgen-v2-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test("legacy duration CLI and one-request smoke both normalize into valid measured budgets", () => {
  const base = ["--endpoint", "http://localhost:8000/infer", "--rps", "0.1", "--duration", "10", "--timeout", "100", "--dataset", "dev.json"];
  const legacy = parseLoadConfig(base);
  assert.equal(legacy.measuredRequests, 1); assert.equal(legacy.warmupRequests, 10);
  const smoke = parseLoadConfig([...base, "--run-kind", "smoke"]);
  assert.equal(smoke.measuredRequests, 1); assert.equal(smoke.warmupRequests, 0);
  const byCount = parseLoadConfig(["--endpoint", "http://localhost/infer", "--rps", "0.5", "--requests", "90", "--dataset", "dev.json"]);
  assert.equal(byCount.durationSeconds, 180);
  assert.throws(() => parseLoadConfig([...base, "--requests", "90"]), /different request counts/);
  assert.throws(() => parseLoadConfig([...base, "--rps", "2"]), /Duplicate/);
  assert.throws(() => parseLoadConfig([...base, "--unknown", "1"]), /Unknown/);
  for (const invalid of [NaN, Infinity, 0, -1]) assert.throws(() => validateConfig(config(undefined, { requestsPerSecond: invalid })));
});

test("seeded order balances category prefixes and visits each record once per complete cycle", () => {
  const order = buildWorkloadOrder(payloads, 24, 9);
  assert.deepEqual(order, buildWorkloadOrder(payloads, 24, 9));
  assert.notDeepEqual(order, buildWorkloadOrder(payloads, 24, 10));
  assert.deepEqual(order.slice(0, 6), buildWorkloadOrder(payloads, 6, 9));
  for (let offset = 0; offset < 24; offset += 12) assert.equal(new Set(order.slice(offset, offset + 12)).size, 12);
  for (let offset = 0; offset < 24; offset += 3) assert.equal(new Set(order.slice(offset, offset + 3).map((index) => payloads[index]!.category)).size, 3);
  const description = describeWorkload(payloads, 24, 9);
  assert.deepEqual(description.measuredCategoryCounts, { counting: 8, spatial: 8, ocr: 8 });
  assert.equal(description.completeDatasetCycles, 2);
  assert.throws(() => buildWorkloadOrder([], 1, 1), /empty/);
});

test("dataset preloading preserves IDs/categories and rejects malformed or duplicate records", async (t) => {
  const directory = await temp(t);
  await writeFile(join(directory, "image.jpg"), "image bytes");
  const path = join(directory, "dev.json");
  const record = { image: "image.jpg", question: "What?", question_id: "one", category: "ocr", source_dataset: "TEXT_VQA" };
  await writeFile(path, JSON.stringify([record]));
  const loaded = await loadDataset(path);
  assert.equal(loaded[0]!.questionId, "one"); assert.equal(loaded[0]!.sourceDataset, "TEXT_VQA");
  await writeFile(path, JSON.stringify([record, record]));
  await assert.rejects(loadDataset(path), /Duplicate/);
  await writeFile(path, JSON.stringify([{ image: "image.jpg" }]));
  await assert.rejects(loadDataset(path), /question/);
});

test("nearest-rank convention, non-finite rejection, and tail sample gates are explicit", () => {
  assert.equal(percentile([9, 1, 5, 3], 0.5), 3);
  assert.equal(percentile([], 0.95), null);
  assert.throws(() => percentile([NaN], 0.5), /finite/);
  assert.throws(() => percentile([1], 0), /fraction/);
  assert.equal(summarizePercentiles(Array.from({ length: 199 }, (_, i) => i)).p95, null);
  assert.equal(summarizePercentiles(Array.from({ length: 200 }, (_, i) => i + 1)).p95, 190);
  assert.equal(summarizePercentiles(Array.from({ length: 999 }, (_, i) => i)).p99, null);
  assert.equal(summarizePercentiles(Array.from({ length: 1000 }, (_, i) => i + 1)).p99, 990);
});

test("one measured request summarizes successfully without slicing away warmup rows", async (t) => {
  const endpoint = await service(t, (req, res) => answer(req, res));
  const run = await runLoad(config(endpoint, { measuredRequests: 1, durationSeconds: 0.01, warmupRequests: 2 }), payloads);
  const summary = summarizeRun(run);
  assert.equal(run.warmupResults.length, 2); assert.equal(summary.totalRequests, 1);
  assert.equal(summary.successfulRequests, 1); assert.equal(summary.achievedArrivalRateRps, null);
  assert.equal(summary.successfulRequestLatencyMs.sampleCount, 1);
  assert.equal(summary.successfulRequestLatencyMs.p95, null);
});

test("warmup completes and settling elapses before measurement starts", async (t) => {
  const endpoint = await service(t, (req, res) => setTimeout(() => answer(req, res), 30));
  const run = await runLoad(config(endpoint, { measuredRequests: 3, durationSeconds: 0.03, warmupRequests: 2, settleMs: 20 }), payloads);
  const warmupEnd = Math.max(...run.warmupResults.map((row) => row.completedAtMs));
  assert.ok(run.plannedStartMs >= warmupEnd + 18);
  assert.ok(run.results.every((row) => row.sentAtMs! >= warmupEnd));
  assert.equal(run.warmupResults[1]!.sentAtMs! >= run.warmupResults[0]!.completedAtMs, true);
});

test("open-loop overlaps slow responses while isolated diagnostic intentionally serializes", async (t) => {
  let active = 0; let maximum = 0;
  const endpoint = await service(t, (req, res) => {
    active++; maximum = Math.max(maximum, active);
    setTimeout(() => { active--; answer(req, res); }, 60);
  });
  const run = await runLoad(config(endpoint, { measuredRequests: 6, durationSeconds: 0.06 }), payloads);
  assert.ok(maximum >= 3, `expected overlap, saw ${maximum}`);
  assert.ok(run.results[1]!.sentAtMs! < run.results[0]!.completedAtMs);
  maximum = 0;
  const isolated = await runLoad(config(endpoint, { measuredRequests: 3, durationSeconds: 0.03, runKind: "isolated" }), payloads);
  assert.equal(maximum, 1);
  assert.ok(isolated.results[1]!.sentAtMs! >= isolated.results[0]!.completedAtMs);
  assert.equal(isolated.plannedEndMs, isolated.finishedMs);
});

test("measured workload identity and sequence stay identical when RPS changes", async (t) => {
  const endpoint = await service(t, (req, res) => answer(req, res));
  const fast = await runLoad(config(endpoint, { requestsPerSecond: 200, durationSeconds: 0.06 }), payloads);
  const slow = await runLoad(config(endpoint, { requestsPerSecond: 100, durationSeconds: 0.12 }), payloads);
  assert.deepEqual(fast.results.map((row) => row.questionId), slow.results.map((row) => row.questionId));
  assert.deepEqual(fast.results.map((row) => row.category), slow.results.map((row) => row.category));
  assert.equal(new Set(fast.results.map((row) => row.requestId)).size, 12);
  assert.ok(fast.results.every((row) => Math.abs(row.plannedToCompleteMs! - row.dispatchLatenessMs! - row.latencyMs!) < 1e-6));
});

test("HTTP failure, malformed response, timeout and missing server queue metrics stay explicit", async (t) => {
  let count = 0;
  const endpoint = await service(t, (req, res) => {
    count++;
    if (count === 1) { res.writeHead(503); res.end("busy"); }
    else if (count === 2) { res.end(JSON.stringify({ missing: "answer" })); }
    else if (count === 3) { /* wait for client timeout */ }
    else answer(req, res, { schema_version: 1, service_ms: 0.1, queue_ms: null, queue_unavailable_reason: "No intake clock" });
  });
  const run = await runLoad(config(endpoint, { measuredRequests: 4, durationSeconds: 0.04, timeoutMs: 80 }), payloads);
  assert.deepEqual(run.results.map((row) => row.outcome), ["http-error", "invalid-response", "timeout", "success"]);
  const summary = summarizeRun(run);
  assert.equal(summary.failedRequests, 3); assert.equal(summary.timedOutRequests, 1);
  assert.equal(summary.serverServiceMs.sampleCount, 1); assert.equal(summary.serverQueueMs.sampleCount, 0);
  assert.equal(run.results[3]!.serverMetrics!.queue_ms, null);
  assert.equal(run.results[3]!.unattributedClientMs, run.results[3]!.latencyMs! - 0.1);
});

test("bad correlation and invalid lifecycle durations are invalid responses", async (t) => {
  let count = 0;
  const endpoint = await service(t, (req, res) => {
    if (++count === 1) res.end(JSON.stringify({ answer: "ok", request_id: "wrong" }));
    else answer(req, res, { service_ms: -1 });
  });
  const run = await runLoad(config(endpoint, { measuredRequests: 2, durationSeconds: 0.02 }), payloads);
  assert.deepEqual(run.results.map((row) => row.outcome), ["invalid-response", "invalid-response"]);
});

test("warmup failure aborts before any measured request reaches the server", async (t) => {
  let received = 0;
  const endpoint = await service(t, (_req, res) => { received++; res.writeHead(500); res.end("failed"); });
  const run = await runLoad(config(endpoint, { warmupRequests: 2 }), payloads);
  assert.equal(received, 1); assert.equal(run.status, "aborted");
  assert.equal(run.results.length, 12); assert.ok(run.results.every((row) => row.sentAtMs === null && row.outcome === "cancelled"));
  assert.match(run.stopReason!, /Warmup failed/);
});

test("interruption preserves one terminal outcome for every predeclared measured request", async (t) => {
  const endpoint = await service(t, (req, res) => setTimeout(() => answer(req, res), 150));
  const controller = new AbortController();
  const saved: RequestResult[] = [];
  const timer = setTimeout(() => controller.abort("test interruption"), 25);
  const run = await runLoad(config(endpoint), payloads, { signal: controller.signal, onResult: (row) => saved.push(row) });
  clearTimeout(timer);
  assert.equal(run.status, "aborted"); assert.equal(saved.length, 12);
  assert.equal(new Set(saved.map((row) => row.requestId)).size, 12);
  assert.ok(saved.some((row) => row.sentAtMs === null));
  assert.ok(saved.every((row) => row.outcome === "cancelled"));
  assert.equal(summarizeRun(run).totalRequests, 12);
});

test("deadline aborts slow in-flight work and retains all terminal records", async (t) => {
  const endpoint = await service(t, () => {});
  const run = await runLoad(config(endpoint, { measuredRequests: 3, durationSeconds: 0.03, runDeadlineMs: 40 }), payloads);
  assert.equal(run.status, "aborted"); assert.match(run.stopReason!, /deadline/);
  assert.equal(run.results.length, 3); assert.ok(run.results.every((row) => row.outcome === "cancelled"));
});

test("window completions and drain throughput distinguish overloaded finite runs", async (t) => {
  const endpoint = await service(t, (req, res) => setTimeout(() => answer(req, res), 90));
  const run = await runLoad(config(endpoint, { measuredRequests: 3, durationSeconds: 0.03 }), payloads);
  const summary = summarizeRun(run);
  assert.equal(summary.successfulWithinWindow, 0); assert.equal(summary.outstandingAtWindowEnd, 3);
  assert.ok(summary.drainMs > 40); assert.ok(summary.successfulThroughputIncludingDrainRps > 0);
  assert.equal(summary.successfulThroughputWithinWindowRps, 0);
});

test("raw journal exists during a run, preserves completion order, and hashes exact stored bytes", async (t) => {
  const directory = await temp(t);
  const journal = await createJournal(directory);
  const endpoint = await service(t, (req, res) => answer(req, res));
  const run = await runLoad(config(endpoint, { measuredRequests: 2, durationSeconds: 0.02 }), payloads, { onResult: journal.append });
  const info = await journal.finish();
  const bytes = await readFile(join(directory, "requests.jsonl"));
  const rows = bytes.toString().trim().split("\n").map((line) => JSON.parse(line) as RequestResult);
  assert.equal(info.count, 2); assert.deepEqual(rows.map((row) => row.requestId).sort(), run.results.map((row) => row.requestId).sort());
  assert.equal(info.sha256, `sha256:${createHash("sha256").update(bytes).digest("hex")}`);
  await assert.rejects(createJournal(directory), /EEXIST/);
});
