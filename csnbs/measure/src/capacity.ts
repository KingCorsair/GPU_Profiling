import { percentile } from './loadgen.js';
import type { VerifiedRun } from './analyze.js';

export type CapacityScreen = {
  classification: 'delivery-invalid' | 'overloaded' | 'finite-window-pass' | 'inconclusive';
  sustainableCapacityEstablished: false; offeredRps: number; actualDispatchRps: number | null;
  successfulWindowRps: number; outstandingAtWindowEnd: number; backlogGrowth: number;
  dispatchP95Ms: number | null; queueObserved: boolean; reasons: string[];
};
/** A conservative finite-window screen. A single run never establishes sustainable capacity. */
export function classifyCapacity(run: VerifiedRun): CapacityScreen {
  const { manifest } = run; const summary = manifest.summary;
  const rows = run.requests.filter((row) => row.phase === 'measurement' && row.sentAtMs !== null);
  const start = manifest.timing.plannedStartMs, end = manifest.timing.plannedEndMs;
  const outstanding = (at: number) => rows.filter((row) => row.sentAtMs! <= at && row.completedAtMs > at).length;
  const early = [0.2, 0.3, 0.4].map((fraction) => outstanding(start + (end - start) * fraction));
  const late = [0.6, 0.7, 0.8].map((fraction) => outstanding(start + (end - start) * fraction));
  const backlogGrowth = percentile(late, 0.5)! - percentile(early, 0.5)!;
  const dispatchP95Ms = percentile(rows.map((row) => row.dispatchLatenessMs!), 0.95);
  const queueObserved = rows.length > 0 && rows.filter((row) => typeof row.serverMetrics?.queue_ms === 'number').length === rows.length;
  const reasons: string[] = [];
  let classification: CapacityScreen['classification'] = 'inconclusive';
  const deliveryError = summary.achievedArrivalRateRps === null || Math.abs(summary.achievedArrivalRateRps / manifest.config.requestsPerSecond - 1) > 0.05;
  if (manifest.runKind !== 'open-loop') reasons.push('Capacity requires open-loop arrivals');
  else if (manifest.status !== 'complete') { classification = 'delivery-invalid'; reasons.push('Run was interrupted or aborted'); }
  else if (summary.dispatchedRequests !== summary.totalRequests || deliveryError || dispatchP95Ms === null || dispatchP95Ms > Math.max(10, 100 / manifest.config.requestsPerSecond)) {
    classification = 'delivery-invalid'; reasons.push('Client delivery deviates from the prescribed schedule');
  } else if (summary.failedRequests > 0) { classification = 'overloaded'; reasons.push('Observed request failures/timeouts; capacity criterion failed (cause needs diagnosis)'); }
  else if (backlogGrowth > Math.max(2, rows.length * 0.03) && summary.outstandingAtWindowEnd > Math.max(2, rows.length * 0.05)) {
    classification = 'overloaded'; reasons.push('Client outstanding work grows through the observation window');
  } else if (summary.measurementWindowSeconds < 30 || summary.totalRequests < 200) {
    reasons.push('Screen is too short or small: require at least 30 seconds and 200 measured requests');
  } else { classification = 'finite-window-pass'; reasons.push('Client delivery, failures and outstanding-work screen passed for this fixed observation window'); }
  if (!queueObserved) reasons.push('Server queue boundaries unavailable; client outstanding work includes transport and service');
  reasons.push('Repeat boundary rates over longer windows before a sustainable capacity claim');
  return { classification, sustainableCapacityEstablished: false, offeredRps: manifest.config.requestsPerSecond,
    actualDispatchRps: summary.achievedArrivalRateRps, successfulWindowRps: summary.successfulThroughputWithinWindowRps,
    outstandingAtWindowEnd: summary.outstandingAtWindowEnd, backlogGrowth, dispatchP95Ms, queueObserved, reasons };
}

export type RepeatedCapacityPoint = {
  offeredRps: number; trialCount: number; expectedTrialCount: number | null; runIds: string[];
  allPass: boolean; allFail: boolean;
  classification: 'all-pass' | 'all-fail' | 'mixed' | 'inconclusive';
  screens: CapacityScreen[]; reasons: string[];
};
export type CapacityBracket = {
  status: 'screen-bracketed' | 'unbounded-above' | 'unbounded-below' | 'inconclusive';
  sustainableCapacityEstablished: false; minimumRepetitions: number;
  highestAllPassRate: number | null; lowestAllFailRate: number | null;
  screenBracketRps: [number, number] | null; nonmonotonic: boolean;
  points: RepeatedCapacityPoint[]; reasons: string[];
};
function comparableCapacityIdentity(run: VerifiedRun): string {
  const { manifest: m } = run;
  // Rates, observation duration and repetition seed intentionally vary across a sweep.
  const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
    : value !== null && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, stable(item)])) : value;
  return JSON.stringify(stable({
    runKind: m.runKind, workload: m.workload.id, requestBudget: m.config.measuredRequests,
    warmup: m.config.warmupRequests, settleMs: m.config.settleMs, timeout: m.config.timeoutMs,
    harnessCommit: m.source.gitCommit, serverSource: m.server.source,
    serverHardware: m.server.hardware, gpuModels: m.hardware.gpuModels,
    model: m.server.modelId, checkpoint: m.server.checkpointRevision,
    configuration: m.server.configuration, runtime: m.server.runtime,
  }));
}
/** Aggregate one serving variant only. Repeated screens are not proof of sustainable capacity. */
export type CapacityBracketOptions = { minimumRepetitions?: number; expectedTrialsPerRate?: number; expectedRates?: number[] };
export function summarizeCapacityBracket(runs: VerifiedRun[], options: CapacityBracketOptions = {}): CapacityBracket {
  const minimumRepetitions = options.minimumRepetitions ?? 2;
  if (options.expectedTrialsPerRate !== undefined && (!Number.isSafeInteger(options.expectedTrialsPerRate) || options.expectedTrialsPerRate <= 0)) throw Error('Expected repetition count must be a positive integer');
  if (!Number.isSafeInteger(minimumRepetitions) || minimumRepetitions < 2) throw Error('Capacity aggregation requires at least two repetitions');
  const grouped = new Map<number, VerifiedRun[]>(), ids = new Set<string>(), reasons: string[] = [];
  if (options.expectedRates) for (const rate of options.expectedRates) {
    if (!Number.isFinite(rate) || rate <= 0 || grouped.has(rate)) throw Error('Expected rates must be unique positive finite values');
    grouped.set(rate, []);
  }
  let identity: string | null = null;
  for (const run of runs) {
    const currentIdentity = comparableCapacityIdentity(run);
    if (identity !== null && currentIdentity !== identity) reasons.push('Sweep changes workload, hardware, serving configuration, provenance or measurement policy');
    identity ??= currentIdentity;
    if (ids.has(run.manifest.runId)) reasons.push('Repeated run identity cannot count as independent replication');
    ids.add(run.manifest.runId);
    const rate = run.manifest.config.requestsPerSecond;
    if (options.expectedRates && !options.expectedRates.includes(rate)) reasons.push('Observed a rate outside the prescribed sweep');
    const group = grouped.get(rate) ?? []; group.push(run); grouped.set(rate, group);
  }
  const points: RepeatedCapacityPoint[] = [...grouped].sort(([a], [b]) => a - b).map(([offeredRps, group]) => {
    const screens = group.map(classifyCapacity), pointReasons: string[] = [];
    const pass = screens.filter((screen) => screen.classification === 'finite-window-pass').length;
    const fail = screens.filter((screen) => screen.classification === 'overloaded').length;
    const completePrescribedCount = options.expectedTrialsPerRate === undefined || group.length === options.expectedTrialsPerRate;
    const enough = group.length >= minimumRepetitions && completePrescribedCount;
    const allPass = enough && pass === group.length, allFail = enough && fail === group.length;
    const classification = allPass ? 'all-pass' : allFail ? 'all-fail' : pass > 0 && fail > 0 ? 'mixed' : 'inconclusive';
    if (group.length < minimumRepetitions) pointReasons.push(`Fewer than ${minimumRepetitions} independent trials at this rate`);
    if (!completePrescribedCount) pointReasons.push(`Expected ${options.expectedTrialsPerRate} trials at this rate, found ${group.length}; missing or additional trials cannot confirm the prescribed screen`);
    if (classification === 'mixed') pointReasons.push('Repeated trials disagree between pass and fail');
    if (screens.some((screen) => screen.classification === 'delivery-invalid')) pointReasons.push('At least one trial did not deliver its prescribed arrivals');
    if (screens.some((screen) => screen.classification === 'inconclusive')) pointReasons.push('At least one trial was too small/short or otherwise inconclusive');
    return { offeredRps, trialCount: group.length, expectedTrialCount: options.expectedTrialsPerRate ?? null, runIds: group.map((run) => run.manifest.runId), allPass, allFail, classification, screens, reasons: pointReasons };
  });
  const passRates = points.filter((point) => point.allPass).map((point) => point.offeredRps);
  const failRates = points.filter((point) => point.allFail).map((point) => point.offeredRps);
  const highestAllPassRate = passRates.length ? Math.max(...passRates) : null;
  const lowestAllFailRate = failRates.length ? Math.min(...failRates) : null;
  const nonmonotonic = highestAllPassRate !== null && lowestAllFailRate !== null && highestAllPassRate >= lowestAllFailRate;
  if (nonmonotonic) reasons.push('A higher rate passes while a lower rate fails; the sweep is nonmonotonic');
  if (points.some((point) => point.classification === 'mixed')) reasons.push('Mixed repeated outcomes need confirmation before choosing a bracket');
  if (points.some((point) => point.classification === 'inconclusive')) reasons.push('One or more tested rates remain inconclusive');
  if (!points.length) reasons.push('No completed trials supplied');
  let status: CapacityBracket['status'] = 'inconclusive', screenBracketRps: [number, number] | null = null;
  if (!reasons.length) {
    if (highestAllPassRate !== null && lowestAllFailRate !== null) { status = 'screen-bracketed'; screenBracketRps = [highestAllPassRate, lowestAllFailRate]; }
    else if (highestAllPassRate !== null) status = 'unbounded-above';
    else if (lowestAllFailRate !== null) status = 'unbounded-below';
  }
  reasons.push('Bounds describe repeated fixed-window client screens only; longer boundary trials and observable queue behavior are still required for sustainable capacity');
  return { status, sustainableCapacityEstablished: false, minimumRepetitions, highestAllPassRate, lowestAllFailRate,
    screenBracketRps, nonmonotonic, points, reasons: [...new Set(reasons)] };
}
