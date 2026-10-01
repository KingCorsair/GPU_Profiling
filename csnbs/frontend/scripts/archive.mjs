// An evidence adapter, not an analyzer: every metric comes from the saved summary.
import { createHash } from 'node:crypto';
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const number = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
const text = (value) => typeof value === 'string' && value.length > 0 ? value : null;
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function archiveRun(raw, { sourcePath, sha256, campaign = null, integrationNote = null }) {
  if (!object(raw) || raw.schema !== 'loadgen-run' || !text(raw.runId) || !object(raw.summary)) {
    throw new Error('Not a complete loadgen run');
  }
  // Fail closed for future schema versions until their meanings are reviewed.
  if (![1, 2].includes(raw.schemaVersion)) throw new Error(`Unsupported run schema ${raw.schemaVersion}`);
  const v2 = raw.schemaVersion === 2;
  if (v2 && (!object(raw.quality) || typeof raw.quality.reportable !== 'boolean' || !Array.isArray(raw.quality.reasons) || !raw.quality.reasons.every((reason) => typeof reason === 'string'))) throw new Error('V2 quality metadata missing');
  const summary = raw.summary;
  const successful = number(summary.successfulRequests);
  const failed = number(summary.failedRequests);
  const measured = number(summary.totalRequests);
  if (![successful, failed, measured].every((value) => value !== null && Number.isInteger(value)) || successful + failed !== measured) {
    throw new Error('Inconsistent request counts');
  }
  const server = object(raw.server) ? raw.server : {};
  const config = object(server.configuration) ? server.configuration : {};
  const benchmark = object(raw.benchmark) ? raw.benchmark : {};
  const notes = [];
  const modelId = text(server.modelId) ?? text(benchmark.modelId);
  if (!modelId) notes.push('Model identity is absent from the raw run.');
  if (v2) {
    notes.push(...raw.quality.reasons);
    if (raw.status !== 'complete') notes.push(`Run ${raw.status ?? 'status unknown'}: ${text(raw.stopReason) ?? 'no stop reason recorded'}.`);
    if (raw.warmup?.isolated === true) notes.push(`Separate warmup: ${raw.warmup.successful}/${raw.warmup.requested} requests succeeded before measurement.`);
    else notes.push('Separate warmup is unverified.');
    for (const metric of ['successfulRequestLatencyMs', 'dispatchLatenessMs', 'plannedToCompleteMs', 'serverServiceMs', 'serverQueueMs']) {
      for (const percentile of ['p95', 'p99']) if (text(summary[metric]?.[`${percentile}Reason`])) notes.push(`${metric} ${percentile}: ${summary[metric][`${percentile}Reason`]}`);
    }
    notes.push('Passing run checks does not establish campaign significance or sustainable capacity.');
  } else {
    notes.push('Legacy run: warmup was excluded by request sequence; a clean measurement phase is unverified.');
    notes.push('Legacy fixed-duration runs may use different questions at different arrival rates.');
  }
  if (successful < 1000) notes.push('Fewer than 1,000 successful requests: p99 is unstable.');
  if (raw.source?.gitDirty === true) notes.push('Uncommitted source changes were present.');
  if (!text(raw.source?.gitCommit)) notes.push('Source revision was not recorded.');
  if (failed > 0) notes.push('Some requests failed. Latency percentiles describe successful requests only.');
  const integration = Boolean(integrationNote) || raw.runKind === 'smoke' || server.mode === 'fake';
  if (integrationNote) notes.unshift(integrationNote);
  if (raw.runKind === 'isolated') notes.push('Isolated requests are sequential. Configured RPS is not an offered open-loop arrival rate.');
  const percentile = (field) => Object.fromEntries(['p50', 'p95', 'p99'].map((key) => [key, number(summary[field]?.[key])]));
  return {
    runId: raw.runId,
    sourcePath,
    sha256,
    downloadPath: `/data/archive/${sha256}.json`,
    requestDownloadPath: null,
    requestsSha256: null,
    schemaVersion: raw.schemaVersion,
    recordedAtUtc: text(raw.recordedAtUtc),
    runKind: text(raw.runKind) ?? 'unknown',
    evidence: integration ? 'integration' : !v2 ? 'exploratory' : raw.quality.reportable && raw.quality.reasons.length === 0 && raw.status === 'complete' ? 'eligible' : 'review',
    status: text(raw.status),
    notes,
    campaign: campaign?.name ?? null,
    modelId,
    visualTokenNum: number(config.visual_token_num) ?? campaign?.visualTokenNum ?? null,
    tokenSource: number(config.visual_token_num) !== null ? 'server' : number(campaign?.visualTokenNum) !== null ? 'campaign' : null,
    importantRatio: number(config.important_ratio) ?? campaign?.importantRatio ?? null,
    maxOutputTokens: number(config.max_new_tokens) ?? number(benchmark.maxOutputTokens) ?? campaign?.maxNewTokens ?? null,
    batchSize: number(config.batch_size) ?? number(benchmark.batchSize),
    offeredRps: raw.runKind === 'isolated' ? null : number(raw.config?.requestsPerSecond),
    durationSeconds: number(raw.config?.durationSeconds),
    workloadId: text(raw.workload?.id),
    measuredOrderHash: text(raw.workload?.measuredOrderHash),
    checkpointRevision: text(server.checkpointRevision),
    warmupRequests: number(raw.warmup?.successful),
    gitCommit: text(raw.source?.gitCommit),
    gitDirty: typeof raw.source?.gitDirty === 'boolean' ? raw.source.gitDirty : null,
    gpuModels: Array.isArray(raw.hardware?.gpuModels) ? raw.hardware.gpuModels.filter((value) => text(value)) : [],
    summary: {
      measuredRequests: measured,
      successfulRequests: successful,
      failedRequests: failed,
      successfulThroughputRps: number(summary.successfulThroughputRps),
      latencyMs: percentile('successfulRequestLatencyMs'),
      dispatchLatenessMs: percentile('dispatchLatenessMs'),
      scheduledToCompleteMs: percentile('plannedToCompleteMs'),
      serverServiceMs: percentile('serverServiceMs'),
      serverQueueMs: percentile('serverQueueMs'),
      withinWindowThroughputRps: number(summary.successfulThroughputWithinWindowRps),
      includingDrainThroughputRps: number(summary.successfulThroughputIncludingDrainRps),
      outstandingAtWindowEnd: number(summary.outstandingAtWindowEnd),
      drainMs: number(summary.drainMs),
    },
  };
}

async function runFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    // Do not follow symlinks outside the results directory.
    if (entry.isDirectory()) files.push(...await runFiles(fullPath));
    else if (entry.isFile() && (entry.name === 'run.json' || entry.name === 'run.partial.json' && !entries.some((sibling) => sibling.name === 'run.json'))) files.push(fullPath);
  }
  return files.sort();
}

export async function syncArchive(repository, frontend, source) {
  const selected = new Map(source.series.flatMap((series) => series.runs.map((run) => [run.runPath, {
    name: source.campaign.name,
    visualTokenNum: series.visualTokenNum,
    importantRatio: source.campaign.importantRatio,
    maxNewTokens: source.campaign.maxNewTokens,
  }])));
  const output = path.join(frontend, 'public/data/archive');
  await mkdir(output, { recursive: true });
  const archive = { schemaVersion: 1, generatedAtUtc: new Date().toISOString(), runs: [], issues: [] };
  const seenHashes = new Set();
  const files = [];
  for (const root of ['results/loadgen', 'results/campaigns']) {
    try { files.push(...await runFiles(path.join(repository, root))); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  for (const filename of files) {
    const sourcePath = path.relative(repository, filename).split(path.sep).join('/');
    try {
      const bytes = await readFile(filename);
      const raw = JSON.parse(bytes);
      if (raw.schema === 'loadgen-run-partial') throw new Error('Interrupted or still running: final run manifest unavailable');
      let integrationNote = null;
      try { integrationNote = (await readFile(path.join(path.dirname(filename), 'INTEGRATION_ONLY.txt'), 'utf8')).trim(); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
      const campaignName = sourcePath.startsWith('results/campaigns/') ? sourcePath.split('/')[2] : null;
      const entry = archiveRun(raw, { sourcePath, sha256: hash(bytes), campaign: selected.get(sourcePath) ?? (campaignName ? { name: campaignName } : null), integrationNote });
      if (seenHashes.has(entry.sha256)) throw new Error('Duplicate run bytes already indexed from another path');
      // Copies retain the exact original bytes and downloadable SHA-256 identity.
      await writeFile(path.join(output, `${entry.sha256}.json`), bytes);
      if (raw.requests?.file === 'requests.jsonl') {
        try {
          const records = await readFile(path.join(path.dirname(filename), 'requests.jsonl'));
          entry.requestsSha256 = hash(records);
          if (raw.schemaVersion === 2 && raw.requests.sha256 !== `sha256:${entry.requestsSha256}`) throw new Error('V2 request-record hash does not match the run manifest');
          entry.requestDownloadPath = `/data/archive/${entry.requestsSha256}.jsonl`;
          await writeFile(path.join(output, `${entry.requestsSha256}.jsonl`), records);
        } catch (error) {
          if (error.code !== 'ENOENT') throw error;
          if (raw.schemaVersion === 2) throw new Error('V2 request records are unavailable; integrity cannot be verified');
          entry.notes.push('The original per-request file is unavailable.');
        }
      }
      else if (raw.schemaVersion === 2) throw new Error('V2 request-record reference is missing');
      seenHashes.add(entry.sha256);
      archive.runs.push(entry);
    } catch (error) {
      archive.issues.push({ sourcePath, message: error.message });
    }
  }
  archive.runs.sort((a, b) => (b.recordedAtUtc ?? '').localeCompare(a.recordedAtUtc ?? '') || a.sourcePath.localeCompare(b.sourcePath));
  await writeFile(path.join(frontend, 'public/data/archive.json'), `${JSON.stringify(archive, null, 2)}\n`);
  return archive;
}
