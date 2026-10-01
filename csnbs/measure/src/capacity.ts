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
