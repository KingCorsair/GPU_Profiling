export type StudyDownload = { sourcePath: string; sha256: string; downloadPath: string };
export type SavedEffect = {
  metric: string; pairCount: number; effectPercent: number; ci95: [number, number];
  conclusion: 'improvement' | 'inconclusive' | 'regression'; reasons: string[];
  pairEffectsPercent: number[]; bootstrapSeed: number; method: string;
  direction: 'higher-is-better' | 'lower-is-better';
};
export type StudyTrial = {
  trialId: string; runId: string; variant: 'baseline-A' | 'pruned-128'; visualTokenNum: 576 | 128;
  offeredRps: 1 | 3; measurementWindowSeconds: number; successfulRequests: number; failedRequests: number;
  withinWindowThroughputRps: number; includingDrainThroughputRps: number;
  outstandingAtWindowEnd: number; drainMs: number; latencyMs: { p50: number; p95: number; p99: null };
  p99Reason: string; capacityClassification: 'finite-window-pass' | 'overloaded'; downloadPath: string | null;
};
export type CurrentStudy = {
  publicPreview?: true;
  schemaVersion: 1; studyId: 'oct01-a40-2026'; dateUtc: '2026-10-01'; verifiedAtUtc: string;
  totals: { campaigns: number; trials: number; measuredRequests: number; warmupRequests: number; failedMeasuredRequests: number; failedWarmupRequests: number };
  campaigns: { campaignId: string; trialCount: number; measuredRequests: number; warmupRequests: number; failedMeasuredRequests: number; report: StudyDownload | null }[];
  primary: {
    campaignId: 'llava-oct01-primary'; modelId: string; gpu: 'NVIDIA A40'; gitCommit: string; checkpointRevision: string;
    workloadId: string; workloadRecords: 90; importantRatio: 0.5; maxNewTokens: 64; batchSize: 1;
    measuredRequestsPerTrial: 270; warmupsPerTrial: 10;
    comparisons: { offeredRps: 1 | 3; throughput: SavedEffect; latency: SavedEffect }[];
    trials: StudyTrial[]; limitations: string[];
  };
  accuracy: { status: 'pending'; reason: string; requiredConfigurationCount: 2; points: never[]; download: StudyDownload | null };
  downloads: { collection: StudyDownload | null; report: StudyDownload | null; reportMarkdown: StudyDownload | null; plotSvg: StudyDownload | null; plotPng: StudyDownload | null };
};

const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const nonnegative = (value: unknown): value is number => finite(value) && value >= 0;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(text);
const sha256 = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const download = (value: unknown): value is StudyDownload => object(value) && text(value.sourcePath) && value.sourcePath.startsWith('results/') && !value.sourcePath.includes('..') && sha256(value.sha256) && typeof value.downloadPath === 'string' && /^\/data\/current-study\/[a-f0-9]{64}\.(json|md|svg|png)$/.test(value.downloadPath) && value.downloadPath.includes(`/${value.sha256}.`);
const invalid = (): never => { throw new Error('Incomplete or incompatible October study'); };

function effect(value: unknown, metric: string, direction: string): value is SavedEffect {
  return object(value) && value.metric === metric && value.direction === direction && value.pairCount === 5 && finite(value.effectPercent)
    && Array.isArray(value.ci95) && value.ci95.length === 2 && value.ci95.every(finite) && value.ci95[0] <= value.ci95[1]
    && ['improvement', 'inconclusive', 'regression'].includes(String(value.conclusion)) && strings(value.reasons)
    && Array.isArray(value.pairEffectsPercent) && value.pairEffectsPercent.length === 5 && value.pairEffectsPercent.every(finite)
    && Number.isSafeInteger(value.bootstrapSeed) && text(value.method);
}

/** Guard the fixed saved-study contract before presenting research claims. */
export function parseCurrentStudy(input: unknown): CurrentStudy {
  if (!object(input) || input.schemaVersion !== 1 || input.studyId !== 'oct01-a40-2026' || input.dateUtc !== '2026-10-01' || !text(input.verifiedAtUtc)) return invalid();
  if (!object(input.totals) || !object(input.primary) || !object(input.accuracy) || !object(input.downloads) || !Array.isArray(input.campaigns)) return invalid();
  if (input.publicPreview !== undefined && input.publicPreview !== true) return invalid();
  const preview = input.publicPreview === true;
  const evidence = (value: unknown) => preview && value === null || download(value);
  const totals = input.totals;
  for (const [key, expected] of Object.entries({ campaigns: 7, trials: 58, measuredRequests: 8820, warmupRequests: 580, failedMeasuredRequests: 0, failedWarmupRequests: 0 })) if (totals[key] !== expected) return invalid();
  const expectedIds = ['llava-baseline-pilot', 'llava-oct01-http-overhead', 'llava-oct01-isolated', 'llava-oct01-primary', 'llava-oct01-workload-long', 'llava-oct01-workload-mixed', 'llava-oct01-workload-short'];
  if (input.campaigns.length !== expectedIds.length) return invalid();
  for (const id of expectedIds) {
    const matches = input.campaigns.filter((campaign) => object(campaign) && campaign.campaignId === id);
    if (matches.length !== 1) return invalid();
    const campaign = matches[0];
    if (!['trialCount', 'measuredRequests', 'warmupRequests', 'failedMeasuredRequests'].every((key) => Number.isSafeInteger(campaign[key]) && nonnegative(campaign[key])) || !evidence(campaign.report)) return invalid();
    if (campaign.report !== null && campaign.report.sourcePath !== `results/campaigns/2026-10-01/${id}/report/report.json`) return invalid();
  }
  for (const [total, field] of [['trials', 'trialCount'], ['measuredRequests', 'measuredRequests'], ['warmupRequests', 'warmupRequests'], ['failedMeasuredRequests', 'failedMeasuredRequests']]) if (input.campaigns.reduce((sum, campaign) => sum + campaign[field], 0) !== totals[total]) return invalid();
  const primary = input.primary;
  if (primary.campaignId !== 'llava-oct01-primary' || primary.modelId !== 'liuhaotian/llava-v1.5-7b' || primary.gpu !== 'NVIDIA A40' || primary.workloadRecords !== 90 || primary.importantRatio !== 0.5 || primary.maxNewTokens !== 64 || primary.batchSize !== 1 || primary.measuredRequestsPerTrial !== 270 || primary.warmupsPerTrial !== 10) return invalid();
  if (!['gitCommit', 'checkpointRevision'].every((key) => typeof primary[key] === 'string' && /^[a-f0-9]{40}$/.test(primary[key] as string)) || !text(primary.workloadId) || !strings(primary.limitations) || primary.limitations.length === 0) return invalid();
  if (!Array.isArray(primary.comparisons) || primary.comparisons.length !== 2) return invalid();
  for (const rate of [1, 3]) {
    const matches = primary.comparisons.filter((comparison) => object(comparison) && comparison.offeredRps === rate);
    if (matches.length !== 1 || !effect(matches[0].throughput, 'successfulThroughputWithinWindowRps', 'higher-is-better') || !effect(matches[0].latency, 'p50', 'lower-is-better')) return invalid();
  }
  if (!Array.isArray(primary.trials) || primary.trials.length !== 20) return invalid();
  const runIds = new Set<string>();
  const trialIds = new Set<string>();
  for (const trial of primary.trials) {
    if (!object(trial) || !text(trial.runId) || !text(trial.trialId) || runIds.has(trial.runId) || trialIds.has(trial.trialId)) return invalid();
    runIds.add(trial.runId); trialIds.add(trial.trialId);
    if (trial.variant !== 'baseline-A' && trial.variant !== 'pruned-128' || trial.visualTokenNum !== (trial.variant === 'baseline-A' ? 576 : 128)) return invalid();
    if (![1, 3].includes(Number(trial.offeredRps)) || trial.measurementWindowSeconds !== (trial.offeredRps === 3 ? 90 : 270) || trial.successfulRequests !== 270 || trial.failedRequests !== 0) return invalid();
    const identity = /^r([13])-p([1-5])-v([01])$/.exec(trial.trialId);
    if (!identity || Number(identity[1]) !== trial.offeredRps || Number(identity[3]) !== (trial.visualTokenNum === 576 ? 0 : 1)) return invalid();
    if (!['withinWindowThroughputRps', 'includingDrainThroughputRps', 'outstandingAtWindowEnd', 'drainMs'].every((key) => nonnegative(trial[key])) || !Number.isSafeInteger(trial.outstandingAtWindowEnd)) return invalid();
    if (!object(trial.latencyMs) || !nonnegative(trial.latencyMs.p50) || !nonnegative(trial.latencyMs.p95) || trial.latencyMs.p95 < trial.latencyMs.p50 || trial.latencyMs.p99 !== null || !text(trial.p99Reason)) return invalid();
    if (!['finite-window-pass', 'overloaded'].includes(String(trial.capacityClassification)) || !(preview && trial.downloadPath === null || typeof trial.downloadPath === 'string' && /^\/data\/archive\/[a-f0-9]{64}\.json$/.test(trial.downloadPath))) return invalid();
  }
  for (const rate of [1, 3]) for (const tokens of [576, 128]) if (primary.trials.filter((trial) => trial.offeredRps === rate && trial.visualTokenNum === tokens).length !== 5) return invalid();
  const accuracy = input.accuracy;
  if (accuracy.status !== 'pending' || !text(accuracy.reason) || accuracy.requiredConfigurationCount !== 2 || !Array.isArray(accuracy.points) || accuracy.points.length !== 0 || !evidence(accuracy.download)) return invalid();
  for (const key of ['collection', 'report', 'reportMarkdown', 'plotSvg', 'plotPng']) if (!evidence(input.downloads[key])) return invalid();
  const primaryCampaign = input.campaigns.find((campaign) => campaign.campaignId === primary.campaignId);
  if (input.downloads.report !== null && (!object(input.downloads.report) || !object(primaryCampaign.report) || input.downloads.report.sha256 !== primaryCampaign.report.sha256)) return invalid();
  return input as unknown as CurrentStudy;
}
