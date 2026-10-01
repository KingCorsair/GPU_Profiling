export type Percentiles = { p50: number | null; p95: number | null; p99: number | null };
export type LengthObservations = { observedCount: number; min: number | null; max: number | null };
export type WorkloadCharacterization = {
  scope: 'measurement-phase' | 'unavailable';
  measuredRecords: number | null;
  successfulRecords: number | null;
  categoryCounts: { label: string; count: number }[];
  sourceDatasetCounts: { label: string; count: number }[];
  outputCharacters: LengthObservations & { unit: string; reason: string | null };
  actualTokens: { generated_text_tokens: LengthObservations; prompt_text_tokens: LengthObservations; visual_tokens: LengthObservations };
  limitations: string[];
};
export type ArchivedRun = {
  runId: string;
  sourcePath: string;
  sha256: string;
  downloadPath: string;
  requestDownloadPath: string | null;
  requestsSha256: string | null;
  schemaVersion: number;
  recordedAtUtc: string | null;
  runKind: string;
  evidence: 'exploratory' | 'integration' | 'eligible' | 'review';
  status: string | null;
  notes: string[];
  campaign: string | null;
  workloadCharacterization: WorkloadCharacterization | null;
  modelId: string | null;
  visualTokenNum: number | null;
  tokenSource: 'server' | 'campaign' | null;
  importantRatio: number | null;
  maxOutputTokens: number | null;
  batchSize: number | null;
  offeredRps: number | null;
  durationSeconds: number | null;
  workloadId: string | null;
  measuredOrderHash: string | null;
  checkpointRevision: string | null;
  warmupRequests: number | null;
  gitCommit: string | null;
  gitDirty: boolean | null;
  gpuModels: string[];
  summary: {
    measuredRequests: number;
    successfulRequests: number;
    failedRequests: number;
    successfulThroughputRps: number | null;
    latencyMs: Percentiles;
    dispatchLatenessMs: Percentiles;
    scheduledToCompleteMs: Percentiles;
    serverServiceMs: Percentiles;
    serverQueueMs: Percentiles;
    withinWindowThroughputRps: number | null;
    includingDrainThroughputRps: number | null;
    outstandingAtWindowEnd: number | null;
    drainMs: number | null;
  };
};
export type CapacityBracket = {
  variant: string; status: 'screen-bracketed' | 'unbounded-above' | 'unbounded-below' | 'inconclusive';
  sustainableCapacityEstablished: false; highestAllPassRate: number | null; lowestAllFailRate: number | null;
  points: { offeredRps: number; trialCount: number; expectedTrialCount: number | null; classification: string; reasons: string[] }[];
  reasons: string[];
};
export type CampaignReport = { campaignId: string; purpose: string; sourcePath: string; sha256: string; downloadPath: string; runCount: number; capacityBrackets: CapacityBracket[]; limitations: string[] };
export type RunArchive = { schemaVersion: 1; generatedAtUtc: string; runs: ArchivedRun[]; campaignReports: CampaignReport[]; issues: { sourcePath: string; message: string }[] };
const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value);
const nullableNumber = (value: unknown) => value === null || typeof value === 'number' && Number.isFinite(value) && value >= 0;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every((item) => typeof item === 'string');

export function parseArchive(input: unknown): RunArchive {
  const invalid = () => { throw new Error('Incomplete or incompatible run archive'); };
  if (!object(input) || input.schemaVersion !== 1 || typeof input.generatedAtUtc !== 'string' || !Array.isArray(input.runs) || !Array.isArray(input.campaignReports) || !Array.isArray(input.issues)) return invalid();
  const hashes = new Set<string>();
  for (const run of input.runs) {
    if (!object(run) || !object(run.summary) || !strings(run.notes) || !strings(run.gpuModels)) return invalid();
    if (!['runId', 'sourcePath', 'sha256', 'downloadPath', 'runKind'].every((field) => typeof run[field] === 'string')) return invalid();
    if (typeof run.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(run.sha256) || hashes.has(run.sha256)) return invalid();
    hashes.add(run.sha256);
    if (run.downloadPath !== `/data/archive/${run.sha256}.json`) return invalid();
    if (![1, 2].includes(Number(run.schemaVersion)) || ![true, false, null].includes(run.gitDirty as boolean | null) || !['server', 'campaign', null].includes(run.tokenSource as string | null)) return invalid();
    if (run.requestDownloadPath !== null && (typeof run.requestsSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(run.requestsSha256) || run.requestDownloadPath !== `/data/archive/${run.requestsSha256}.jsonl`)) return invalid();
    if (!['exploratory', 'integration', 'eligible', 'review'].includes(String(run.evidence))) return invalid();
    if (!['status', 'recordedAtUtc', 'campaign', 'modelId', 'workloadId', 'gitCommit', 'measuredOrderHash', 'checkpointRevision'].every((field) => run[field] === null || typeof run[field] === 'string')) return invalid();
    if (!['visualTokenNum', 'importantRatio', 'maxOutputTokens', 'batchSize', 'offeredRps', 'durationSeconds', 'warmupRequests'].every((field) => nullableNumber(run[field]))) return invalid();
    for (const field of ['measuredRequests', 'successfulRequests', 'failedRequests']) if (!Number.isInteger(run.summary[field]) || Number(run.summary[field]) < 0) return invalid();
    if (Number(run.summary.successfulRequests) + Number(run.summary.failedRequests) !== run.summary.measuredRequests || !nullableNumber(run.summary.successfulThroughputRps)) return invalid();
    if (!['withinWindowThroughputRps', 'includingDrainThroughputRps', 'outstandingAtWindowEnd', 'drainMs'].every((field) => nullableNumber(run.summary && (run.summary as Record<string, unknown>)[field]))) return invalid();
    for (const field of ['latencyMs', 'dispatchLatenessMs', 'scheduledToCompleteMs', 'serverServiceMs', 'serverQueueMs']) {
      const percentiles = run.summary[field];
      if (!object(percentiles) || !['p50', 'p95', 'p99'].every((key) => nullableNumber(percentiles[key]))) return invalid();
    }
    const characterization = run.workloadCharacterization;
    if (characterization !== null) {
      if (!object(characterization) || !['measurement-phase', 'unavailable'].includes(String(characterization.scope)) || !strings(characterization.limitations)) return invalid();
      if (!nullableNumber(characterization.measuredRecords) || !nullableNumber(characterization.successfulRecords) || !object(characterization.outputCharacters) || !object(characterization.actualTokens)) return invalid();
      if (characterization.scope === 'measurement-phase' && (characterization.measuredRecords !== run.summary.measuredRequests || characterization.successfulRecords !== run.summary.successfulRequests)) return invalid();
      if (characterization.scope === 'unavailable' && (characterization.measuredRecords !== null || characterization.successfulRecords !== null)) return invalid();
      for (const field of ['categoryCounts', 'sourceDatasetCounts']) {
        const counts = characterization[field];
        if (!Array.isArray(counts) || !counts.every((row) => object(row) && typeof row.label === 'string' && Number.isSafeInteger(row.count) && Number(row.count) > 0)) return invalid();
        if (new Set(counts.map((row) => row.label)).size !== counts.length || characterization.scope === 'unavailable' && counts.length > 0) return invalid();
        if (characterization.scope === 'measurement-phase' && counts.reduce((total, row) => total + Number(row.count), 0) !== characterization.measuredRecords) return invalid();
      }
      for (const observations of [characterization.outputCharacters, ...['generated_text_tokens', 'prompt_text_tokens', 'visual_tokens'].map((key) => (characterization.actualTokens as Record<string, unknown>)[key])]) {
        if (!object(observations) || !Number.isSafeInteger(observations.observedCount) || Number(observations.observedCount) < 0 || !nullableNumber(observations.min) || !nullableNumber(observations.max)) return invalid();
        if (Number(observations.observedCount) > Number(characterization.successfulRecords ?? 0)) return invalid();
        if (observations.observedCount === 0 && (observations.min !== null || observations.max !== null)) return invalid();
        if (Number(observations.observedCount) > 0 && (observations.min === null || observations.max === null || Number(observations.min) > Number(observations.max))) return invalid();
      }
      if (typeof characterization.outputCharacters.unit !== 'string' || !(characterization.outputCharacters.reason === null || typeof characterization.outputCharacters.reason === 'string')) return invalid();
    }
  }
  for (const report of input.campaignReports) {
    if (!object(report) || !['campaignId', 'purpose', 'sourcePath'].every((field) => typeof report[field] === 'string') || typeof report.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(report.sha256) || report.downloadPath !== `/data/archive/${report.sha256}.json` || !Number.isSafeInteger(report.runCount) || Number(report.runCount) < 0 || !Array.isArray(report.capacityBrackets) || !strings(report.limitations)) return invalid();
    for (const bracket of report.capacityBrackets) {
      if (!object(bracket) || typeof bracket.variant !== 'string' || bracket.sustainableCapacityEstablished !== false || !['screen-bracketed', 'unbounded-above', 'unbounded-below', 'inconclusive'].includes(String(bracket.status)) || !nullableNumber(bracket.highestAllPassRate) || !nullableNumber(bracket.lowestAllFailRate) || !strings(bracket.reasons) || !Array.isArray(bracket.points)) return invalid();
      if (!bracket.points.every((point) => object(point) && typeof point.offeredRps === 'number' && nullableNumber(point.offeredRps) && Number.isSafeInteger(point.trialCount) && Number(point.trialCount) >= 0 && nullableNumber(point.expectedTrialCount) && typeof point.classification === 'string' && strings(point.reasons))) return invalid();
    }
  }
  if (!input.issues.every((issue) => object(issue) && typeof issue.sourcePath === 'string' && typeof issue.message === 'string')) return invalid();
  return input as RunArchive;
}

/** Explain missing/mismatched context; this is not a statistical comparison. */
export function comparisonWarnings(a: ArchivedRun, b: ArchivedRun): string[] {
  const warnings: string[] = [];
  const fields = [
    ['modelId', 'Model identity'], ['checkpointRevision', 'Checkpoint revision'], ['workloadId', 'Workload identity'], ['measuredOrderHash', 'Measured question order'],
    ['offeredRps', 'Offered load'], ['durationSeconds', 'Arrival duration'],
    ['maxOutputTokens', 'Output cap'], ['importantRatio', 'Important-token ratio'],
    ['batchSize', 'Batch size'],
  ] as const;
  if (a.runKind !== b.runKind) warnings.push('Load-generation modes differ.');
  if (a.schemaVersion !== b.schemaVersion) warnings.push('Harness schema versions differ; measurement boundaries and throughput definitions may differ.');
  for (const [key, label] of fields) {
    if (a[key] === null || b[key] === null) warnings.push(`${label} is unknown in at least one run.`);
    else if (a[key] !== b[key]) warnings.push(`${label} differs.`);
  }
  if (!a.gpuModels.length || !b.gpuModels.length) warnings.push('GPU identity is unknown in at least one run.');
  else if ([...a.gpuModels].sort().join() !== [...b.gpuModels].sort().join()) warnings.push('GPU models differ.');
  if (a.evidence === 'integration' || b.evidence === 'integration') warnings.push('An integration-only run cannot establish a benchmark result.');
  warnings.push('Saved summaries alone do not establish an improvement. Repeated matched trials and an uncertainty analysis are required.');
  return warnings;
}
