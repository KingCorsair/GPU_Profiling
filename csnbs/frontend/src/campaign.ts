export type Run = {
  runId: string;
  runPath: string;
  offeredRps: number;
  measuredRequests: number;
  successfulRequests: number;
  failedRequests: number;
  successfulThroughputRps: number;
  p50LatencyMs: number;
  p95LatencyMs: number;
  p99LatencyMs: number;
  sha256: string;
};
export type Campaign = {
  publicPreview?: true;
  campaign: {
    name: string;
    runDateUtc: string;
    gpu: string;
    gitCommit: string;
    gitDirty: boolean;
    workloadId: string;
    datasetRecordCount: number;
    durationSecondsPerLevel: number;
    warmupRequestsExcludedPerLevel: number;
    importantRatio: number;
    maxNewTokens: number;
    batchSize: number;
  };
  series: { id: string; visualTokenNum: number; runs: Run[] }[];
  conditions: { label: string; improvementPercent: number | null; note: string }[];
};

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const rates = [0.5, 1, 1.5, 2, 3];
const runMetrics = ['measuredRequests', 'successfulRequests', 'failedRequests', 'successfulThroughputRps', 'p50LatencyMs', 'p95LatencyMs', 'p99LatencyMs'];

/** Validate the fixed historical presentation contract before any chart dereferences it. */
export function parseCampaign(input: unknown): Campaign {
  const invalid = () => { throw new Error('Incomplete or incompatible benchmark campaign'); };
  if (!isObject(input) || !isObject(input.campaign)) return invalid();
  if (input.offeredRps !== 3 || input.visualTokenNum !== 128 || input.percentile !== 'p50') return invalid();
  if (input.publicPreview !== undefined && input.publicPreview !== true) return invalid();
  const campaign = input.campaign;
  if (campaign.runDateUtc !== '2026-09-01' || campaign.gpu !== 'NVIDIA A40' || campaign.batchSize !== 1 || campaign.importantRatio !== 0.5 || campaign.maxNewTokens !== 64 || campaign.status !== 'exploratory') return invalid();
  if (!['name', 'gitCommit', 'workloadId'].every((key) => typeof campaign[key] === 'string' && campaign[key].length > 0)) return invalid();
  if (typeof campaign.gitDirty !== 'boolean' || !['datasetRecordCount', 'durationSecondsPerLevel', 'warmupRequestsExcludedPerLevel'].every((key) => finite(campaign[key]) && campaign[key] > 0)) return invalid();
  if (!Array.isArray(input.series) || input.series.length !== 2) return invalid();
  for (const tokens of [576, 128]) {
    const series = input.series.find((value: unknown) => isObject(value) && value.visualTokenNum === tokens);
    if (!isObject(series) || typeof series.id !== 'string' || !Array.isArray(series.runs) || series.runs.length !== rates.length) return invalid();
    for (const rate of rates) {
      const matches = series.runs.filter((value: unknown) => isObject(value) && value.offeredRps === rate);
      if (matches.length !== 1 || !isObject(matches[0])) return invalid();
      const run = matches[0];
      if (!runMetrics.every((key) => finite(run[key]) && run[key] >= 0)) return invalid();
      if (typeof run.runId !== 'string' || !/^[\w.-]+$/.test(run.runId) || typeof run.runPath !== 'string' || typeof run.sha256 !== 'string') return invalid();
    }
  }
  if (!Array.isArray(input.conditions) || input.conditions.length !== 4 || !input.conditions.every((condition: unknown) => isObject(condition) && typeof condition.label === 'string' && typeof condition.note === 'string')) return invalid();
  if (input.conditions[0].improvementPercent !== 0 || input.conditions[1].improvementPercent !== null || input.conditions[3].improvementPercent !== null) return invalid();
  if (!finite(input.conditions[2].improvementPercent) || input.conditions[2].improvementPercent >= 0) return invalid();
  return input as unknown as Campaign;
}
