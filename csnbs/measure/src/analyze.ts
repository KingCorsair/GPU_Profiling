import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { percentile, summarizeRun, runQualityReasons, type LoadRun, type RequestResult, type RunManifest } from './loadgen.js';

export type VerifiedResources = {
  file: 'resources.jsonl'; count: number; sha256: string; scope: string | null;
  samples: Record<string, unknown>[];
};
export type VerifiedRun = { manifest: RunManifest; requests: RequestResult[]; directory: string; resources?: VerifiedResources };
export type ComparisonMetric = 'successfulThroughputWithinWindowRps' | 'successfulThroughputIncludingDrainRps' | 'p50' | 'p95' | 'p99';
export type RunPair = { baseline: VerifiedRun; candidate: VerifiedRun; blockId: string };
export type PairComparison = {
  metric: ComparisonMetric; pairCount: number; effectPercent: number | null;
  ci95: [number, number] | null; conclusion: 'improvement' | 'regression' | 'inconclusive';
  reasons: string[]; pairEffectsPercent: number[]; bootstrapSeed: number;
  method: string; direction: 'higher-is-better' | 'lower-is-better';
};
function invariant(condition: unknown, message: string): asserts condition { if (!condition) throw Error(message); }
function close(a: number, b: number): boolean { return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= Math.max(1e-6, Math.abs(b) * 1e-10); }
function compareValue(actual: unknown, expected: unknown, path: string): void {
  if (typeof expected === 'number') { invariant(typeof actual === 'number' && close(actual, expected), `${path}: summary does not match raw requests`); return; }
  if (expected === null || typeof expected !== 'object') { invariant(actual === expected, `${path}: summary does not match raw requests`); return; }
  invariant(actual !== null && typeof actual === 'object', `${path}: expected object`);
  for (const [key, value] of Object.entries(expected)) compareValue((actual as Record<string, unknown>)[key], value, `${path}.${key}`);
}
const hash = (text: string | Buffer): string => `sha256:${createHash('sha256').update(text).digest('hex')}`;
const numeric = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** The final manifest is required. Partial journals remain evidence but are never completed runs. */
export async function readVerifiedRun(directory: string): Promise<VerifiedRun> {
  const manifest = JSON.parse(await readFile(join(directory, 'run.json'), 'utf8')) as RunManifest;
  invariant(manifest.schema === 'loadgen-run' && manifest.schemaVersion === 2, 'Unsupported run schema; use the historical reader for version 1');
  invariant(manifest.requests.file === 'requests.jsonl' && manifest.requests.schemaVersion === 3, 'Unsupported request schema');
  invariant(['complete', 'aborted'].includes(manifest.status), 'Invalid final run status');
  invariant(manifest.runKind === manifest.config.runKind, 'Run kind differs from configuration');
  const bytes = await readFile(join(directory, 'requests.jsonl'));
  invariant(hash(bytes) === manifest.requests.sha256, 'Raw request SHA-256 mismatch');
  const lines = bytes.toString('utf8').trim().split('\n').filter(Boolean);
  const requests = lines.map((line) => JSON.parse(line) as RequestResult);
  invariant(requests.length === manifest.requests.count, 'Raw request count mismatch');
  const ids = new Set<string>();
  const phases = { warmup: [] as RequestResult[], measurement: [] as RequestResult[] };
  for (const row of requests) {
    invariant(row.phase === 'warmup' || row.phase === 'measurement', 'Invalid request phase');
    invariant(Number.isSafeInteger(row.sequence) && row.sequence >= 0, 'Invalid sequence');
    invariant(row.requestId === `${manifest.runId}:${row.phase}:${row.sequence}`, 'Request identity mismatch');
    invariant(!ids.has(row.requestId), 'Duplicate request identity'); ids.add(row.requestId);
    invariant(typeof row.questionId === 'string' && typeof row.category === 'string', 'Missing workload identity');
    invariant(numeric(row.scheduledAtMs) && numeric(row.completedAtMs), 'Invalid request clock');
    invariant(['success', 'http-error', 'invalid-response', 'timeout', 'network-error', 'cancelled'].includes(row.outcome), 'Invalid terminal outcome');
    if (row.sentAtMs === null) {
      invariant(row.outcome === 'cancelled' && row.latencyMs === null && row.dispatchLatenessMs === null && row.plannedToCompleteMs === null, 'Undispatched request has timing/success claims');
    } else {
      invariant(numeric(row.sentAtMs) && row.completedAtMs >= row.sentAtMs, 'Invalid dispatched request timing');
      invariant(numeric(row.latencyMs) && close(row.latencyMs, row.completedAtMs - row.sentAtMs), 'Latency identity mismatch');
      invariant(numeric(row.dispatchLatenessMs) && close(row.dispatchLatenessMs, row.sentAtMs - row.scheduledAtMs), 'Dispatch identity mismatch');
      invariant(numeric(row.plannedToCompleteMs) && close(row.plannedToCompleteMs, row.completedAtMs - row.scheduledAtMs), 'Planned completion identity mismatch');
    }
    if (row.outcome === 'success') invariant(row.status !== null && row.status >= 200 && row.status < 300 && row.error === null && row.sentAtMs !== null, 'Success contradicts outcome/status');
    else invariant(typeof row.error === 'string', 'Failure is missing its error reason');
    if (row.serverRequestId !== null && row.outcome === 'success') invariant(row.serverRequestId === row.requestId, 'Server request correlation mismatch');
    if (row.serverMetrics !== null) {
      for (const key of ['service_ms', 'queue_ms', 'generation_wall_ms', 'preprocess_ms', 'postprocess_ms', 'generated_text_tokens', 'prompt_text_tokens', 'visual_tokens']) {
        const value = row.serverMetrics[key];
        invariant(value === undefined || value === null || (numeric(value) && value >= 0), `Invalid server metric ${key}`);
      }
    }
    const service = row.serverMetrics?.service_ms;
    if (typeof service === 'number' && row.latencyMs !== null) invariant(numeric(row.unattributedClientMs) && close(row.unattributedClientMs, row.latencyMs - service), 'Client/service residual mismatch');
    else invariant(row.unattributedClientMs === null, 'Unattributed client duration without service observation');
    phases[row.phase].push(row);
  }
  for (const rows of Object.values(phases)) {
    rows.sort((a, b) => a.sequence - b.sequence);
    rows.forEach((row, index) => invariant(row.sequence === index, 'Missing or duplicate phase sequence'));
  }
  invariant(phases.measurement.length === manifest.config.measuredRequests, 'Measured budget/raw outcome count mismatch');
  invariant(phases.warmup.length === manifest.config.warmupRequests, 'Warmup budget/raw outcome count mismatch');
  invariant(phases.warmup.length === manifest.warmup.completed, 'Warmup manifest count mismatch');
  invariant(phases.warmup.filter((row) => row.outcome === 'success').length === manifest.warmup.successful, 'Warmup success count mismatch');
  invariant(manifest.workload.orderSeed === manifest.config.seed, 'Workload seed differs from requested configuration');
  const questionIds = phases.measurement.map((row) => row.questionId);
  invariant(JSON.stringify(questionIds) === JSON.stringify(manifest.workload.measuredQuestionIds), 'Measured workload sequence mismatch');
  invariant(hash(JSON.stringify(questionIds)) === manifest.workload.measuredOrderHash, 'Measured order hash mismatch');
  const categories: Record<string, number> = {};
  for (const row of phases.measurement) categories[row.category] = (categories[row.category] ?? 0) + 1;
  compareValue(manifest.workload.measuredCategoryCounts, categories, 'workload.measuredCategoryCounts');
  const timing = manifest.timing;
  invariant(numeric(timing.plannedStartMs) && numeric(timing.plannedEndMs) && timing.plannedEndMs >= timing.plannedStartMs && numeric(timing.finishedMs), 'Invalid measurement window');
  for (const row of phases.measurement) {
    if (manifest.runKind !== 'isolated') invariant(close(row.scheduledAtMs, timing.plannedStartMs + row.sequence * 1000 / manifest.config.requestsPerSecond), 'Open-loop schedule differs from prescribed arrivals');
    if (row.sentAtMs !== null) invariant(row.sentAtMs >= timing.plannedStartMs, 'Measured request dispatched before measurement phase');
  }
  const latestWarmup = phases.warmup.reduce((latest, row) => Math.max(latest, row.completedAtMs), -Infinity);
  if (manifest.status === 'complete') {
    invariant(manifest.warmup.successful === manifest.config.warmupRequests, 'Completed run has unsuccessful warmup');
    invariant(timing.plannedStartMs + 2 >= latestWarmup + manifest.config.settleMs, 'Warmup or settling overlaps measurement');
    invariant(phases.measurement.every((row) => row.outcome !== 'cancelled'), 'Completed run has cancelled measured requests');
  }
  const run: LoadRun = { plannedStartMs: timing.plannedStartMs, plannedEndMs: timing.plannedEndMs, finishedMs: timing.finishedMs,
    results: phases.measurement, warmupResults: phases.warmup, status: manifest.status, stopReason: manifest.stopReason };
  compareValue(manifest.summary, summarizeRun(run), 'summary');
  const qualityReasons = runQualityReasons(manifest);
  invariant(manifest.quality.reportable === (qualityReasons.length === 0), 'Reportability contradicts provenance/outcomes');
  invariant(stable(manifest.quality.reasons) === stable(qualityReasons), 'Quality reasons contradict provenance/outcomes');
  const resourceDescriptor = (manifest as unknown as Record<string, unknown>).resourceSamples;
  const resources = resourceDescriptor === undefined || resourceDescriptor === null ? undefined : await readVerifiedResources(directory, resourceDescriptor);
  return { manifest, requests, directory: resolve(directory), ...(resources ? { resources } : {}) };
}

export async function readVerifiedResources(directory: string, descriptor: unknown): Promise<VerifiedResources> {
  invariant(descriptor !== null && typeof descriptor === 'object' && !Array.isArray(descriptor), 'Invalid resource descriptor');
  const metadata = descriptor as Record<string, unknown>;
  invariant(metadata.file === 'resources.jsonl', 'Unsupported resource file; expected resources.jsonl');
  invariant(Number.isSafeInteger(metadata.count) && (metadata.count as number) >= 0, 'Invalid resource sample count');
  invariant(typeof metadata.sha256 === 'string', 'Missing resource SHA-256');
  const bytes = await readFile(join(directory, 'resources.jsonl'));
  invariant(hash(bytes) === metadata.sha256, 'Resource sample SHA-256 mismatch');
  const lines = bytes.toString('utf8').trim().split('\n').filter(Boolean);
  invariant(lines.length === metadata.count, 'Resource sample count mismatch');
  const samples = lines.map((line) => {
    const parsed: unknown = JSON.parse(line);
    invariant(parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed), 'Invalid resource sample record');
    const row = parsed as Record<string, unknown>;
    invariant(typeof row.recordedAtUtc === 'string' && Number.isFinite(Date.parse(row.recordedAtUtc)), 'Invalid resource sample timestamp');
    if (row.kind === 'sampled-device') {
      invariant(Array.isArray(row.fields) && row.fields.every((key) => typeof key === 'string') && new Set(row.fields).size === row.fields.length, 'Invalid resource fields');
      invariant(Array.isArray(row.values) && row.values.length === row.fields.length && row.values.every((value) => typeof value === 'string' || typeof value === 'number'), 'Invalid resource values');
    } else invariant(typeof row.error === 'string', 'Unknown resource record kind');
    return row;
  });
  return { file: 'resources.jsonl', count: metadata.count as number, sha256: metadata.sha256,
    scope: typeof metadata.scope === 'string' ? metadata.scope : null, samples };
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${stable(item)}`).join(',')}}`;
  return JSON.stringify(value) ?? 'undefined';
}
function effectiveConfiguration(config: Record<string, unknown> | null, kind: 'token-count' | 'implementation'): unknown {
  if (!config) return null;
  return Object.fromEntries(Object.entries(config).filter(([key]) => kind === 'token-count' ? !['visual_token_num', 'visualTokenNum'].includes(key) : !['implementation', 'implementation_id', 'serving_variant'].includes(key)));
}
/** Reject known mismatches; missing provenance is a reporting reason, never assumed equality. */
export function comparabilityReasons(a: RunManifest, b: RunManifest, kind: 'token-count' | 'implementation' = 'token-count'): string[] {
  const reasons: string[] = [];
  const compare = (left: unknown, right: unknown, name: string) => { if (stable(left) !== stable(right)) reasons.push(`Unmatched ${name}`); };
  compare(a.runKind, b.runKind, 'run kind');
  for (const key of ['requestsPerSecond', 'durationSeconds', 'measuredRequests', 'timeoutMs', 'warmupRequests', 'settleMs'] as const) compare(a.config[key], b.config[key], key);
  compare(a.workload.id, b.workload.id, 'workload'); compare(a.workload.measuredOrderHash, b.workload.measuredOrderHash, 'measured request order');
  compare(a.source.gitCommit, b.source.gitCommit, 'measurement harness commit');
  compare(a.hardware.gpuModels, b.hardware.gpuModels, 'GPU model');
  const aServer = a.server as unknown as Record<string, unknown>, bServer = b.server as unknown as Record<string, unknown>;
  compare(aServer.hardware, bServer.hardware, 'server hardware identity'); compare(aServer.runtime, bServer.runtime, 'server runtime');
  compare(a.server.modelId, b.server.modelId, 'model identity'); compare(a.server.checkpointRevision, b.server.checkpointRevision, 'checkpoint revision');
  compare(effectiveConfiguration(a.server.configuration, kind), effectiveConfiguration(b.server.configuration, kind), 'effective configuration');
  if (kind === 'token-count') compare((aServer.source as Record<string, unknown> | null)?.gitCommit, (bServer.source as Record<string, unknown> | null)?.gitCommit, 'server implementation commit');
  return reasons;
}
function valueFor(run: VerifiedRun, metric: ComparisonMetric): number | null {
  return metric === 'p50' || metric === 'p95' || metric === 'p99'
    ? run.manifest.summary.successfulRequestLatencyMs[metric]
    : run.manifest.summary[metric];
}
function random(seed: number): () => number {
  let n = seed >>> 0;
  return () => { n += 0x6D2B79F5; let t = n; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
export function comparePairs(pairs: RunPair[], metric: ComparisonMetric = 'successfulThroughputWithinWindowRps', seed = 90, kind: 'token-count' | 'implementation' = 'token-count'): PairComparison {
  const reasons: string[] = [];
  const effects: number[] = [];
  const seen = new Set<string>();
  const runIds = new Set<string>();
  const higher = !['p50', 'p95', 'p99'].includes(metric);
  const rate = pairs[0]?.baseline.manifest.config.requestsPerSecond;
  for (const pair of pairs) {
    if (seen.has(pair.blockId)) reasons.push(`Duplicate independent block ${pair.blockId}`); seen.add(pair.blockId);
    for (const run of [pair.baseline, pair.candidate]) {
      if (runIds.has(run.manifest.runId)) reasons.push('Run reused across independent pairs'); runIds.add(run.manifest.runId);
      if (run.manifest.status !== 'complete') reasons.push('Incomplete run in comparison');
      if (!run.manifest.quality.reportable) reasons.push(...run.manifest.quality.reasons.map((reason) => `Run checks: ${reason}`));
      if (run.manifest.config.requestsPerSecond !== rate) reasons.push('Analyze each offered rate separately');
    }
    reasons.push(...comparabilityReasons(pair.baseline.manifest, pair.candidate.manifest, kind));
    const a = valueFor(pair.baseline, metric), b = valueFor(pair.candidate, metric);
    if (a === null || b === null || !Number.isFinite(a) || !Number.isFinite(b) || a <= 0) { reasons.push('Metric unavailable or baseline is nonpositive'); continue; }
    effects.push((higher ? b - a : a - b) / a * 100);
  }
  if (pairs.length < 5) reasons.push('Fewer than five independent paired blocks; uncertainty is not resolved');
  let ci95: [number, number] | null = null;
  if (effects.length >= 5 && seen.size === effects.length) {
    const rng = random(seed); const bootstraps: number[] = [];
    for (let sample = 0; sample < 10000; sample++) {
      const draw = Array.from({ length: effects.length }, () => effects[Math.floor(rng() * effects.length)]!);
      bootstraps.push(percentile(draw, 0.5)!);
    }
    ci95 = [percentile(bootstraps, 0.025)!, percentile(bootstraps, 0.975)!];
  }
  const uniqueReasons = [...new Set(reasons)];
  const conclusion = uniqueReasons.length === 0 && ci95 !== null ? ci95[0] > 0 ? 'improvement' : ci95[1] < 0 ? 'regression' : 'inconclusive' : 'inconclusive';
  return { metric, pairCount: pairs.length, effectPercent: percentile(effects, 0.5), ci95, conclusion,
    reasons: uniqueReasons, pairEffectsPercent: effects, bootstrapSeed: seed,
    method: 'Median paired percent effect; 10,000 bootstrap resamples of independent paired blocks. Requests are not independent replicates. Positive means better; interval crossing zero is inconclusive, not equivalence.',
    direction: higher ? 'higher-is-better' : 'lower-is-better' };
}
export function baselineRepeatability(runs: VerifiedRun[], metric: ComparisonMetric = 'successfulThroughputWithinWindowRps') {
  const values = runs.map((run) => valueFor(run, metric)).filter((value): value is number => value !== null && Number.isFinite(value));
  const median = percentile(values, 0.5);
  const absoluteDeviations = median === null ? [] : values.map((value) => Math.abs(value - median));
  const reasons = runs.length < 5 ? ['Fewer than five baseline repetitions; provisional spread only'] : [];
  const reference = runs[0]?.manifest;
  for (const run of runs) {
    if (!run.manifest.quality.reportable) reasons.push('Some baseline trials fail run-quality checks; spread is descriptive only');
    if (reference) {
      // Repetition seeds may shuffle the same complete workload; all scientific controls still match.
      reasons.push(...comparabilityReasons(reference, run.manifest).filter((reason) => reason !== 'Unmatched measured request order'));
      if (stable(reference.server.configuration) !== stable(run.manifest.server.configuration)) reasons.push('Baseline effective configurations differ');
    }
  }
  return { metric, trialCount: runs.length, validTrialCount: values.length, median,
    min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null,
    relativeRangePercent: median && values.length ? (Math.max(...values) - Math.min(...values)) / Math.abs(median) * 100 : null,
    medianAbsoluteDeviationPercent: median ? percentile(absoluteDeviations, 0.5)! / Math.abs(median) * 100 : null,
    reasons: [...new Set(reasons)],
    note: 'Observed trial spread is a pilot noise estimate, not a promised detection threshold.' };
}
