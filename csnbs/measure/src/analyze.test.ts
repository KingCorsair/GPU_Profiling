import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test, type TestContext } from 'node:test';
import { baselineRepeatability, comparePairs, comparabilityReasons, readVerifiedRun, type VerifiedRun } from './analyze.js';
import { classifyCapacity } from './capacity.js';
import { buildWorkloadOrder, describeWorkload, runQualityReasons, summarizeRun, type DatasetPayload, type LoadConfig, type LoadRun, type RequestResult, type RunManifest } from './loadgen.js';

function fixture(id: string, latency = 20, serial = false): VerifiedRun {
  const payloads: DatasetPayload[] = Array.from({ length: 12 }, (_, index) => ({ image_b64: 'eA==', question: `Q${index}`, workloadIndex: index, questionId: `q${index}`, category: index % 2 ? 'ocr' : 'count', sourceDataset: 'fixture' }));
  const config: LoadConfig = { endpoint: 'http://localhost/infer', requestsPerSecond: 2, durationSeconds: 120, measuredRequests: 240, timeoutMs: 120000,
    datasetPath: 'fixture.json', warmupRequests: 10, seed: 7, settleMs: 0, runKind: 'open-loop', outputDir: '/tmp', runDeadlineMs: null };
  const workload = describeWorkload(payloads, 240, 7), order = buildWorkloadOrder(payloads, 240, 7);
  let previousEnd = 0;
  const makeRow = (phase: 'warmup' | 'measurement', sequence: number): RequestResult => {
    const index = phase === 'measurement' ? order[sequence]! : sequence % 12;
    const payload = payloads[index]!;
    const scheduledAtMs = phase === 'warmup' ? sequence * 2 : 1000 + sequence * 500;
    const sentAtMs = scheduledAtMs;
    const completedAtMs = phase === 'warmup' ? sentAtMs + 1 : (serial ? Math.max(sentAtMs, previousEnd) : sentAtMs) + latency;
    if (phase === 'measurement') previousEnd = completedAtMs;
    return { requestId: `${id}:${phase}:${sequence}`, phase, sequence, workloadIndex: index, questionId: payload.questionId,
      category: payload.category, sourceDataset: payload.sourceDataset, scheduledAtMs, sentAtMs, completedAtMs,
      latencyMs: completedAtMs - sentAtMs, dispatchLatenessMs: 0, plannedToCompleteMs: completedAtMs - scheduledAtMs,
      status: 200, error: null, outcome: 'success', serverRequestId: `${id}:${phase}:${sequence}`,
      serverMetrics: { service_ms: phase === 'warmup' ? 1 : latency, queue_ms: null },
      unattributedClientMs: phase === 'warmup' ? 0 : completedAtMs - sentAtMs - latency };
  };
  const warmupResults = Array.from({ length: 10 }, (_, sequence) => makeRow('warmup', sequence));
  const results = Array.from({ length: 240 }, (_, sequence) => makeRow('measurement', sequence));
  const run: LoadRun = { plannedStartMs: 1000, plannedEndMs: 121000, finishedMs: Math.max(121000, previousEnd), results, warmupResults, status: 'complete', stopReason: null };
  const requests = [...warmupResults, ...results];
  const manifest: RunManifest = {
    schema: 'loadgen-run', schemaVersion: 2, runId: id, runKind: 'open-loop', recordedAtUtc: '2026-10-01T12:00:00Z', status: 'complete', stopReason: null,
    timing: { clock: 'performance.now', performanceTimeOriginUnixMs: 0, plannedStartMs: 1000, plannedEndMs: 121000, finishedMs: run.finishedMs,
      plannedStartAtUtc: new Date(1000).toISOString(), plannedEndAtUtc: new Date(121000).toISOString(), finishedAtUtc: new Date(run.finishedMs).toISOString() },
    config, workload, source: { gitCommit: 'harness-revision', gitDirty: false }, hardware: { gpuModels: ['A40'], source: 'server-health' },
    server: { healthUrl: 'http://localhost/health', service: 'fixture', pid: 1, modelId: 'llava', checkpointRevision: 'weights-revision',
      configuration: { model_id: 'llava', visual_token_num: 576, important_ratio: 0.5, max_new_tokens: 64, implementation: 'fixture-v1', do_sample: false, use_cache: true, eos_policy: 'natural', prompt_template: 'fixture', batch_size: 1, dtype: 'float16' }, error: null,
      mode: 'model', modelLoaded: true, source: { gitCommit: 'server-revision', gitDirty: false },
      hardware: { gpuModels: ['A40'], devices: [{ name: 'A40', uuid: 'GPU-fixture' }] }, runtime: { python: '3.12' } },
    summary: summarizeRun(run), warmup: { requested: 10, completed: 10, successful: 10, isolated: true, settleMs: 0 },
    requests: { file: 'requests.jsonl', schemaVersion: 3, count: requests.length, sha256: '' },
    quality: { reportable: true, reasons: [], percentileConvention: 'nearest-rank' },
  };
  assert.deepEqual(runQualityReasons(manifest), []);
  return { manifest, requests, directory: `/tmp/${id}` };
}
async function save(t: TestContext, run: VerifiedRun): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'analysis-v2-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const bytes = run.requests.map((row) => JSON.stringify(row)).join('\n') + '\n';
  run.manifest.requests.sha256 = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  await writeFile(join(directory, 'requests.jsonl'), bytes);
  await writeFile(join(directory, 'run.json'), JSON.stringify(run.manifest));
  return directory;
}
function pairs(latencies: number[]) {
  return latencies.map((latency, index) => ({ baseline: fixture(`a-${index}`, 20), candidate: fixture(`b-${index}`, latency), blockId: `pair-${index}` }));
}

test('reader verifies raw hashes, phase-local sequences, workload and recomputed summary', async (t) => {
  const run = fixture('valid'); const directory = await save(t, run);
  const read = await readVerifiedRun(directory);
  assert.equal(read.requests.length, 250); assert.equal(read.manifest.summary.totalRequests, 240);
  assert.equal(read.manifest.summary.successfulRequestLatencyMs.p95, 20);
  assert.equal(read.manifest.summary.successfulRequestLatencyMs.p99, null);
});

test('reader rejects tampered bytes, cached summaries and unsupported versions', async (t) => {
  const raw = fixture('raw'); const rawDir = await save(t, raw);
  await writeFile(join(rawDir, 'requests.jsonl'), '{}\n');
  await assert.rejects(readVerifiedRun(rawDir), /SHA-256/);
  const badSummary = fixture('summary'); badSummary.manifest.summary.successfulRequests++;
  await assert.rejects(readVerifiedRun(await save(t, badSummary)), /summary/);
  const old = fixture('old'); (old.manifest as unknown as { schemaVersion: number }).schemaVersion = 99;
  await assert.rejects(readVerifiedRun(await save(t, old)), /Unsupported run schema/);
});

test('reader rejects missing coverage, altered workload, invalid timing, and fabricated quality', async (t) => {
  const missing = fixture('missing'); missing.requests.pop(); missing.manifest.requests.count--;
  await assert.rejects(readVerifiedRun(await save(t, missing)), /budget/);
  const order = fixture('order'); order.requests[11]!.questionId = 'fabricated';
  await assert.rejects(readVerifiedRun(await save(t, order)), /workload sequence/);
  const timing = fixture('timing'); timing.requests[10]!.latencyMs = -1;
  await assert.rejects(readVerifiedRun(await save(t, timing)), /Latency identity/);
  const quality = fixture('quality'); quality.manifest.server.mode = 'fake';
  await assert.rejects(readVerifiedRun(await save(t, quality)), /Reportability/);
});

test('paired inference resolves improvement and regression, while A/A remains inconclusive', () => {
  const faster = comparePairs(pairs([16, 15, 16, 17, 16]), 'p50', 12);
  assert.equal(faster.conclusion, 'improvement'); assert.equal(faster.effectPercent, 20); assert.ok(faster.ci95![0] > 0);
  const slower = comparePairs(pairs([24, 25, 24, 23, 24]), 'p50', 12);
  assert.equal(slower.conclusion, 'regression'); assert.equal(slower.effectPercent, -20);
  const unchanged = comparePairs(pairs([20, 20, 20, 20, 20]), 'p50', 12);
  assert.equal(unchanged.conclusion, 'inconclusive'); assert.deepEqual(unchanged.ci95, [0, 0]);
  assert.deepEqual(faster, comparePairs(pairs([16, 15, 16, 17, 16]), 'p50', 12));
});

test('mixed noisy effects, insufficient pairs, missing tails and correlated blocks cannot claim improvement', () => {
  assert.equal(comparePairs(pairs([10, 30, 10, 30, 20]), 'p50').conclusion, 'inconclusive');
  const short = comparePairs(pairs([10, 10]), 'p50'); assert.equal(short.ci95, null); assert.equal(short.conclusion, 'inconclusive');
  const missingTail = comparePairs(pairs([10, 10, 10, 10, 10]), 'p99'); assert.equal(missingTail.effectPercent, null);
  const correlated = pairs([10, 10, 10, 10, 10]); correlated[1]!.blockId = correlated[0]!.blockId;
  assert.equal(comparePairs(correlated, 'p50').conclusion, 'inconclusive');
  assert.ok(comparePairs(correlated, 'p50').reasons.some((reason) => reason.includes('Duplicate')));
});

test('comparisons reject workload/hardware/decoding changes and distinguish token and implementation comparisons', () => {
  const a = fixture('a'), b = fixture('b');
  b.manifest.server.configuration!.visual_token_num = 128;
  assert.deepEqual(comparabilityReasons(a.manifest, b.manifest), []);
  assert.ok(comparabilityReasons(a.manifest, b.manifest, 'implementation').includes('Unmatched effective configuration'));
  b.manifest.server.configuration!.max_new_tokens = 128;
  assert.ok(comparabilityReasons(a.manifest, b.manifest).includes('Unmatched effective configuration'));
  b.manifest.workload.measuredOrderHash = 'other';
  assert.ok(comparabilityReasons(a.manifest, b.manifest).includes('Unmatched measured request order'));
  b.manifest.server.hardware!.devices = [{ name: 'A40', uuid: 'another-GPU' }];
  assert.ok(comparabilityReasons(a.manifest, b.manifest).includes('Unmatched server hardware identity'));
  const mixed = pairs([10, 10, 10, 10, 10]); mixed[0]!.candidate.manifest.quality = { reportable: false, reasons: ['dirty'], percentileConvention: 'nearest-rank' };
  assert.equal(comparePairs(mixed, 'p50').conclusion, 'inconclusive');
});

test('baseline repeatability describes trial spread without claiming a detection threshold', () => {
  const repeatability = baselineRepeatability([16, 18, 20, 22, 24].map((latency, index) => fixture(`base-${index}`, latency)), 'p50');
  assert.equal(repeatability.median, 20); assert.equal(repeatability.relativeRangePercent, 40);
  assert.equal(repeatability.medianAbsoluteDeviationPercent, 10); assert.deepEqual(repeatability.reasons, []);
  assert.equal(baselineRepeatability([fixture('one')], 'p50').reasons.length, 1);
});

test('capacity screens delivery errors and growing backlog without declaring sustainable capacity', () => {
  const stable = classifyCapacity(fixture('stable'));
  assert.equal(stable.classification, 'finite-window-pass'); assert.equal(stable.sustainableCapacityEstablished, false);
  assert.equal(stable.queueObserved, false);
  const overloaded = classifyCapacity(fixture('overloaded', 700, true));
  assert.equal(overloaded.classification, 'overloaded'); assert.ok(overloaded.backlogGrowth > 0);
  const undelivered = fixture('undelivered'); undelivered.manifest.config.requestsPerSecond = 4;
  assert.equal(classifyCapacity(undelivered).classification, 'delivery-invalid');
  const isolated = fixture('isolated'); isolated.manifest.runKind = 'isolated';
  assert.equal(classifyCapacity(isolated).classification, 'inconclusive');
});
