// Fixed October presentation adapter. Copy saved effects/percentiles; never analyze requests.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const collectionPath = 'results/provenance/2026-10-01/collection-index.json';
const primaryId = 'llava-oct01-primary';
const primaryDirectory = `results/campaigns/2026-10-01/${primaryId}`;
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const nonempty = (value) => typeof value === 'string' && value.length > 0;
const expectedIds = ['llava-baseline-pilot', 'llava-oct01-http-overhead', 'llava-oct01-isolated', primaryId, 'llava-oct01-workload-long', 'llava-oct01-workload-mixed', 'llava-oct01-workload-short'];
const same = (a, b, description) => assert.deepEqual(a, b, `October study: ${description}`);

function savedEffect(effect, metric, direction) {
  assert(effect && effect.metric === metric && effect.direction === direction, 'October study: incompatible saved effect');
  same(effect.pairCount, 5, 'expected five paired blocks');
  assert(finite(effect.effectPercent) && Array.isArray(effect.ci95) && effect.ci95.length === 2 && effect.ci95.every(finite) && effect.ci95[0] <= effect.ci95[1], 'October study: missing saved uncertainty');
  assert(['improvement', 'inconclusive', 'regression'].includes(effect.conclusion) && nonempty(effect.method), 'October study: missing effect interpretation');
  assert(Array.isArray(effect.pairEffectsPercent) && effect.pairEffectsPercent.length === 5 && effect.pairEffectsPercent.every(finite), 'October study: missing paired effects');
  assert(Array.isArray(effect.reasons) && effect.reasons.every(nonempty) && Number.isSafeInteger(effect.bootstrapSeed), 'October study: incomplete effect provenance');
  return effect;
}

// Same identity projection as the existing join adapter: omit only the local
// checkpoint path and download envelope, retaining file hashes and all controls.
function executionIdentity(server) {
  same([server.mode, server.modelLoaded, server.error, server.source.gitDirty], ['model', true, null, false], 'unconfirmed execution identity');
  return {
    modelId: server.modelId,
    checkpointRevision: server.checkpointRevision,
    checkpointFiles: server.configuration.download_provenance.files.map(({ file, bytes, sha256 }) => ({ file, bytes, sha256: sha256.replace(/^sha256:/, '') })).sort((a, b) => a.file < b.file ? -1 : a.file > b.file ? 1 : 0),
    serverGitCommit: server.source.gitCommit,
    runtime: { torch: server.runtime.torch, transformers: server.runtime.transformers },
    configuration: Object.fromEntries(Object.entries(server.configuration).filter(([key]) => !['checkpoint', 'download_provenance'].includes(key))),
  };
}

/** The archive has already verified each report against its raw run/request evidence. */
export async function buildCurrentStudy(repository, archive) {
  const files = [];
  async function read(sourcePath, expectedHash) {
    const bytes = await readFile(path.join(repository, sourcePath));
    if (expectedHash !== undefined) same(hash(bytes), expectedHash, `changed evidence: ${sourcePath}`);
    return bytes;
  }
  function download(sourcePath, bytes) {
    const sha256 = hash(bytes);
    const downloadPath = `/data/current-study/${sha256}${path.extname(sourcePath)}`;
    files.push({ downloadPath, bytes });
    return { sourcePath, sha256, downloadPath };
  }
  const collectionBytes = await read(collectionPath);
  const collection = JSON.parse(collectionBytes);
  same(collection.schema, 'oct01-measurement-collection-index', 'wrong collection schema');
  same(collection.schemaVersion, 1, 'wrong collection version');
  same(collection.campaigns.map((campaign) => campaign.campaignId).sort(), expectedIds, 'seven frozen campaigns required');
  assert(nonempty(collection.verifiedAtUtc) && Array.isArray(archive?.campaignReports) && Array.isArray(archive.runs), 'October study: missing archive evidence');
  const reports = new Map();
  const campaigns = [];
  for (const campaign of collection.campaigns) {
    const directory = `results/campaigns/2026-10-01/${campaign.campaignId}`;
    same(campaign.report, `${directory}/report/report.json`, 'unexpected report path');
    const bytes = await read(campaign.report, campaign.reportSha256);
    const report = JSON.parse(bytes);
    const matches = archive.campaignReports.filter((entry) => entry.campaignId === campaign.campaignId && entry.sourcePath === campaign.report && entry.sha256 === campaign.reportSha256);
    same(matches.length, 1, `report must match verified archive: ${campaign.campaignId}`);
    same(report.campaignId, campaign.campaignId, 'report identity differs');
    same(report.runs.length, campaign.trialCount, 'trial count differs');
    same(matches[0].runCount, campaign.trialCount, 'archive trial count differs');
    assert(matches[0].purpose !== 'integration', 'October study: research evidence cannot be integration-only');
    const campaignManifest = JSON.parse(await read(`${directory}/campaign.json`, campaign.campaignSha256));
    same(campaignManifest.campaignId, campaign.campaignId, 'campaign identity differs');
    same(campaignManifest.trials.length, campaign.trialCount, 'campaign trial count differs');
    assert(campaignManifest.trials.every((trial) => trial.status === 'complete'), 'October study: incomplete campaign');
    assert(campaign.storageVerification.startsWith(`${directory}/report/`) && !campaign.storageVerification.includes('..'), 'October study: unexpected storage verification path');
    const storage = JSON.parse(await read(campaign.storageVerification, campaign.storageVerificationSha256));
    same(storage.reportSha256, campaign.reportSha256, 'storage report hash differs');
    same(storage.campaignId, campaign.campaignId, 'storage campaign differs');
    same(storage.totals.measurement, campaign.measuredRequests, 'storage measurement count differs');
    same(storage.totals.warmup, campaign.warmupRequests, 'storage warmup count differs');
    same(campaign.secondImportInsertedRuns, 0, 'duplicate storage import is not clean');
    const savedRuns = report.runs.map((run) => {
      const candidates = archive.runs.filter((entry) => entry.runId === run.runId && entry.campaign === report.campaignId && run.requestsSha256 === `sha256:${entry.requestsSha256}`);
      same(candidates.length, 1, 'missing or ambiguous raw run evidence');
      const saved = candidates[0];
      const stored = storage.runs.filter((entry) => entry.runId === run.runId);
      same(stored.length, 1, 'missing or ambiguous storage run inventory');
      const manifests = stored[0].artifacts.filter((artifact) => artifact.kind === 'manifest');
      const requests = stored[0].artifacts.filter((artifact) => artifact.kind === 'requests');
      same([manifests.length, requests.length], [1, 1], 'missing raw artifact storage hashes');
      same(saved.sha256, manifests[0].sha256, 'raw run manifest differs from verified storage inventory');
      same(saved.requestsSha256, requests[0].sha256, 'raw requests differ from verified storage inventory');
      same(saved.evidence, 'eligible', 'nonreportable research evidence');
      assert(run.quality?.reportable === true && run.quality.reasons.length === 0, 'October study: nonreportable saved run');
      same(saved.summary.measuredRequests, run.summary.totalRequests, 'saved measured count differs');
      same(saved.summary.failedRequests, run.summary.failedRequests, 'saved failed count differs');
      return saved;
    });
    same(savedRuns.reduce((sum, run) => sum + run.summary.measuredRequests, 0), campaign.measuredRequests, 'collection measurement count differs');
    same(savedRuns.reduce((sum, run) => sum + run.warmupRequests, 0), campaign.warmupRequests, 'collection warmup count differs');
    same(savedRuns.reduce((sum, run) => sum + run.summary.failedRequests, 0), campaign.failedMeasuredRequests, 'collection failure count differs');
    reports.set(campaign.campaignId, { report, manifest: campaignManifest, savedRuns });
    campaigns.push({ campaignId: campaign.campaignId, trialCount: campaign.trialCount, measuredRequests: campaign.measuredRequests, warmupRequests: campaign.warmupRequests, failedMeasuredRequests: campaign.failedMeasuredRequests, report: download(campaign.report, bytes) });
  }
  const totals = Object.fromEntries(['campaigns', 'trials', 'measuredRequests', 'warmupRequests', 'failedMeasuredRequests', 'failedWarmupRequests'].map((key) => [key, collection.totals[key]]));
  same(totals, { campaigns: 7, trials: 58, measuredRequests: 8820, warmupRequests: 580, failedMeasuredRequests: 0, failedWarmupRequests: 0 }, 'frozen collection totals differ');
  for (const [total, field] of [['trials', 'trialCount'], ['measuredRequests', 'measuredRequests'], ['warmupRequests', 'warmupRequests'], ['failedMeasuredRequests', 'failedMeasuredRequests']]) same(totals[total], campaigns.reduce((sum, campaign) => sum + campaign[field], 0), `collection ${total} differs`);

  const { report, manifest, savedRuns } = reports.get(primaryId);
  same(report.runKind, 'open-loop', 'primary arrival mode differs');
  same(report.comparisonKind, 'token-count', 'primary comparison differs');
  same(report.comparisons.map((comparison) => comparison.offeredRps).sort(), [1, 3], 'primary rates differ');
  const comparisons = report.comparisons.map((comparison) => {
    same(comparison.shamComparison, false, 'primary cannot be a sham comparison');
    return { offeredRps: comparison.offeredRps, throughput: savedEffect(comparison.throughput, 'successfulThroughputWithinWindowRps', 'higher-is-better'), latency: savedEffect(comparison.latency, 'p50', 'lower-is-better') };
  });
  const first = report.runs[0];
  const config = first.server.configuration;
  same(first.server.modelId, 'liuhaotian/llava-v1.5-7b', 'primary model differs');
  same(first.server.hardware.gpuModels, ['NVIDIA A40'], 'primary GPU differs');
  same([config.important_ratio, config.max_new_tokens, config.batch_size], [0.5, 64, 1], 'primary configuration differs');
  same([manifest.spec.measuredRequests, manifest.spec.warmupRequests, first.workload.recordCount], [270, 10, 90], 'primary workload budget differs');
  const trials = report.runs.map((run, index) => {
    const saved = savedRuns[index];
    same(run.source, first.source, 'primary client source differs');
    same(run.server.source, first.server.source, 'primary server source differs');
    same(run.server.hardware.gpuModels, first.server.hardware.gpuModels, 'primary GPU differs');
    same(run.server.modelId, first.server.modelId, 'primary model differs');
    same(run.server.checkpointRevision, first.server.checkpointRevision, 'primary checkpoint differs');
    same(run.workload.id, first.workload.id, 'primary workload differs');
    same(run.server.configuration.visual_token_num, run.variant === 'baseline-A' ? 576 : run.variant === 'pruned-128' ? 128 : undefined, 'primary variant differs');
    const summary = run.summary;
    same(summary.successfulRequestLatencyMs.p99, null, 'unavailable primary p99 must remain unavailable');
    assert(nonempty(summary.successfulRequestLatencyMs.p99Reason), 'October study: missing p99 explanation');
    same(run.capacity.sustainableCapacityEstablished, false, 'finite window cannot establish sustainable capacity');
    return { trialId: run.trialId, runId: run.runId, variant: run.variant, visualTokenNum: run.server.configuration.visual_token_num, offeredRps: run.offeredRps, measurementWindowSeconds: summary.measurementWindowSeconds, successfulRequests: summary.successfulRequests, failedRequests: summary.failedRequests, withinWindowThroughputRps: summary.successfulThroughputWithinWindowRps, includingDrainThroughputRps: summary.successfulThroughputIncludingDrainRps, outstandingAtWindowEnd: summary.outstandingAtWindowEnd, drainMs: summary.drainMs, latencyMs: { p50: summary.successfulRequestLatencyMs.p50, p95: summary.successfulRequestLatencyMs.p95, p99: null }, p99Reason: summary.successfulRequestLatencyMs.p99Reason, capacityClassification: run.capacity.classification, downloadPath: saved.downloadPath };
  });
  const primaryDownload = campaigns.find((campaign) => campaign.campaignId === primaryId).report;
  const handoffPath = `${primaryDirectory}/accuracy-handoff/accuracy-throughput.json`;
  const handoffBytes = await read(handoffPath);
  const handoff = JSON.parse(handoffBytes);
  same([handoff.schema, handoff.schemaVersion, handoff.scope, handoff.campaignId, handoff.status], ['accuracy-throughput-join', 1, 'research', primaryId, 'pending'], 'unsupported accuracy handoff; review new owner evidence before publishing');
  same(handoff.inputs.report.sha256, primaryDownload.sha256, 'accuracy handoff targets a different report');
  same([handoff.points, handoff.evaluations, handoff.inputs.accuracy], [[], [], null], 'pending accuracy must have no fabricated points');
  assert(nonempty(handoff.reason), 'October study: missing pending accuracy reason');
  same(handoff.requiredExecutionIdentities.length, 2, 'accuracy requires both execution configurations');
  for (const required of handoff.requiredExecutionIdentities) {
    same(required.variants.length, 1, 'ambiguous accuracy configuration');
    const runs = report.runs.filter((run) => run.variant === required.variants[0]);
    same(required.runIds.slice().sort(), runs.map((run) => run.runId).sort(), 'accuracy run identities differ');
    assert(runs.length > 0, 'October study: unknown accuracy variant');
    for (const run of runs) same(required.identity, executionIdentity(run.server), 'accuracy execution identity differs');
  }
  same(handoff.requiredExecutionIdentities.flatMap((required) => required.variants).sort(), ['baseline-A', 'pruned-128'], 'accuracy configurations are duplicated');
  const downloads = { collection: download(collectionPath, collectionBytes), report: primaryDownload };
  for (const [key, filename] of [['reportMarkdown', 'report.md'], ['plotSvg', 'campaign_overview.svg'], ['plotPng', 'campaign_overview.png']]) {
    const sourcePath = `${primaryDirectory}/report/${filename}`;
    downloads[key] = download(sourcePath, await read(sourcePath));
  }
  return { study: { schemaVersion: 1, studyId: 'oct01-a40-2026', dateUtc: '2026-10-01', verifiedAtUtc: collection.verifiedAtUtc, totals, campaigns, primary: { campaignId: primaryId, modelId: first.server.modelId, gpu: first.server.hardware.gpuModels[0], gitCommit: first.source.gitCommit, checkpointRevision: first.server.checkpointRevision, workloadId: first.workload.id, workloadRecords: first.workload.recordCount, importantRatio: config.important_ratio, maxNewTokens: config.max_new_tokens, batchSize: config.batch_size, measuredRequestsPerTrial: manifest.spec.measuredRequests, warmupsPerTrial: manifest.spec.warmupRequests, comparisons, trials, limitations: report.limitations }, accuracy: { status: handoff.status, reason: handoff.reason, requiredConfigurationCount: handoff.requiredExecutionIdentities.length, points: handoff.points, download: download(handoffPath, handoffBytes) }, downloads }, files };
}

export async function syncCurrentStudy(repository, frontend, archive) {
  // Validate everything before publishing the new landing-page data or downloads.
  const { study, files } = await buildCurrentStudy(repository, archive);
  await mkdir(path.join(frontend, 'public/data/current-study'), { recursive: true });
  for (const file of files) await writeFile(path.join(frontend, 'public', file.downloadPath), file.bytes);
  await writeFile(path.join(frontend, 'public/data/current-study.json'), `${JSON.stringify(study, null, 2)}\n`);
  return study;
}
