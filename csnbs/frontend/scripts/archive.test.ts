import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { archiveRun, archiveReport, syncArchive } from './archive.mjs';
import { characterizeRecords } from './workload.mjs';
import { comparisonWarnings, parseArchive } from '../src/archive.ts';
import { RunDetails } from '../src/RunArchive.tsx';
import CampaignReports from '../src/CampaignReports.tsx';

const sourcePath = 'results/loadgen/2026-09-01/02-37-44Z_rps-0.5_8beede1e/run.json';
const bytes = readFileSync(new URL(`../../../${sourcePath}`, import.meta.url));
const raw = JSON.parse(bytes.toString());
const options = { sourcePath, sha256: createHash('sha256').update(bytes).digest('hex') };
const entry = () => archiveRun(structuredClone(raw), options);
const archive = () => ({ schemaVersion: 1, generatedAtUtc: '2026-10-01T00:00:00Z', runs: [entry()], campaignReports: [], issues: [] });
const v2 = () => ({
  ...structuredClone(raw), schemaVersion: 2, status: 'complete', stopReason: null,
  quality: { reportable: false, reasons: ['Synthetic test only'], percentileConvention: 'nearest-rank' },
  warmup: { requested: 10, successful: 10, completed: 10, isolated: true, settleMs: 0 },
});

test('archive preserves real saved summaries and does not infer a missing model', () => {
  const run = entry();
  assert.equal(run.summary.latencyMs.p99, raw.summary.successfulRequestLatencyMs.p99);
  assert.equal(run.summary.successfulThroughputRps, raw.summary.successfulThroughputRps);
  assert.equal(run.modelId, null);
  assert.equal(run.evidence, 'exploratory');
  assert.equal(run.visualTokenNum, null);
  assert.equal(parseArchive(archive()).runs.length, 1);
});

test('campaign token metadata is explicitly distinguished from verified server metadata', () => {
  const run = archiveRun(raw, { ...options, campaign: { name: 'Historical', visualTokenNum: 576, importantRatio: 0.5, maxNewTokens: 64 } });
  assert.equal(run.visualTokenNum, 576);
  assert.equal(run.tokenSource, 'campaign');
  assert.equal(run.modelId, null);
});

test('integration markers cannot be silently promoted to benchmark results', () => {
  const run = archiveRun(raw, { ...options, integrationNote: 'Smoke run only.' });
  assert.equal(run.evidence, 'integration');
  assert.ok(run.notes.includes('Smoke run only.'));
});

test('unsupported schemas and inconsistent counts need review rather than appearing in charts', () => {
  assert.throws(() => archiveRun({ ...raw, schemaVersion: 99 }, options), /Unsupported/);
  const invalid = structuredClone(raw);
  invalid.summary.failedRequests = 1;
  assert.throws(() => archiveRun(invalid, options), /Inconsistent/);
});

test('missing or nonfinite metrics are unavailable, never zeros', () => {
  const missing = structuredClone(raw);
  missing.summary.successfulRequestLatencyMs.p99 = null;
  missing.summary.successfulThroughputRps = Infinity;
  const run = archiveRun(missing, options);
  assert.equal(run.summary.latencyMs.p99, null);
  assert.equal(run.summary.successfulThroughputRps, null);
});

test('archive rejects changed download URLs, duplicate records, and malformed summaries', () => {
  const changed = archive();
  changed.runs[0].downloadPath = 'https://example.com/untrusted';
  assert.throws(() => parseArchive(changed));
  const duplicate = archive();
  duplicate.runs.push(duplicate.runs[0]);
  assert.throws(() => parseArchive(duplicate));
  const malformed = archive();
  malformed.runs[0].summary.dispatchLatenessMs = {};
  assert.throws(() => parseArchive(malformed));
});

test('side-by-side comparison exposes unknown identities and mismatched conditions', () => {
  const a = entry();
  const b = entry();
  b.offeredRps = 3;
  b.evidence = 'integration';
  const warnings = comparisonWarnings(a, b);
  assert.ok(warnings.includes('Model identity is unknown in at least one run.'));
  assert.ok(warnings.includes('Offered load differs.'));
  assert.ok(warnings.some((warning) => warning.includes('integration-only')));
  assert.ok(warnings.some((warning) => warning.includes('uncertainty analysis')));
});

test('V2 preserves null percentiles and quality reasons without inventing values', () => {
  const saved = v2();
  saved.summary.successfulRequestLatencyMs.p99 = null;
  saved.summary.successfulRequestLatencyMs.p99Reason = 'Requires 1,000 successful samples';
  const run = archiveRun(saved, options);
  assert.equal(run.evidence, 'review');
  assert.equal(run.summary.latencyMs.p99, null);
  assert.ok(run.notes.includes('Synthetic test only'));
  assert.ok(run.notes.some((note) => note.includes('Requires 1,000')));
  assert.ok(run.notes.every((note) => !note.startsWith('Legacy')));
  assert.equal(parseArchive({ ...archive(), runs: [run] }).runs.length, 1);
});

test('V2 run checks do not override aborted, fake, or smoke evidence', () => {
  const saved = v2();
  saved.quality = { reportable: true, reasons: [], percentileConvention: 'nearest-rank' };
  assert.equal(archiveRun(saved, options).evidence, 'eligible');
  assert.equal(archiveRun({ ...saved, status: 'aborted' }, options).evidence, 'review');
  assert.equal(archiveRun({ ...saved, runKind: 'smoke' }, options).evidence, 'integration');
  assert.equal(archiveRun({ ...saved, server: { mode: 'fake' } }, options).evidence, 'integration');
  assert.equal(archiveRun({ ...saved, runKind: 'isolated' }, options).offeredRps, null);
});

test('archive imports campaign runs, verifies V2 request hashes, and surfaces partial runs', async () => {
  const repository = await mkdtemp(path.join(tmpdir(), 'frontend-archive-'));
  try {
    const directory = path.join(repository, 'results/campaigns/integration/trial/runs/example');
    const interrupted = path.join(repository, 'results/campaigns/integration/trial/runs/interrupted');
    const frontend = path.join(repository, 'frontend');
    await mkdir(directory, { recursive: true });
    await mkdir(interrupted, { recursive: true });
    const records = '{"phase":"measurement","outcome":"success","category":"OCR","sourceDataset":"TEXT_VQA"}\n';
    const saved = v2();
    saved.summary.totalRequests = 1;
    saved.summary.successfulRequests = 1;
    saved.requests = { file: 'requests.jsonl', schemaVersion: 3, count: 1, sha256: `sha256:${createHash('sha256').update(records).digest('hex')}` };
    await writeFile(path.join(directory, 'run.json'), JSON.stringify(saved));
    await writeFile(path.join(directory, 'requests.jsonl'), records);
    await writeFile(path.join(interrupted, 'run.partial.json'), JSON.stringify({ schema: 'loadgen-run-partial' }));
    const index = await syncArchive(repository, frontend, { series: [] });
    assert.equal(index.runs.length, 1);
    assert.equal(index.runs[0].campaign, 'integration');
    assert.equal(index.issues.length, 1);
    assert.match(index.issues[0].message, /Interrupted/);
    assert.equal(await readFile(path.join(frontend, 'public', index.runs[0].requestDownloadPath), 'utf8'), records);
    await writeFile(path.join(directory, 'requests.jsonl'), `${records}tampered`);
    const tampered = await syncArchive(repository, frontend, { series: [] });
    assert.equal(tampered.runs.length, 0);
    assert.ok(tampered.issues.some((issue) => issue.message.includes('hash does not match')));
  } finally {
    await rm(repository, { recursive: true, force: true });
  }
});

test('workload counts use measured outcomes only and never fabricate output lengths', () => {
  const saved = v2();
  saved.summary = { ...saved.summary, totalRequests: 3, successfulRequests: 2, failedRequests: 1 };
  const records = [
    { phase: 'warmup', outcome: 'success', category: 'Warmup only', sourceDataset: 'Warmup source', answer: 'never measured' },
    { phase: 'measurement', outcome: 'success', category: ' OCR ', sourceDataset: 'TEXT_VQA', serverMetrics: { generated_text_tokens: null } },
    { phase: 'measurement', outcome: 'success', category: 'counting', sourceDataset: 'GQA' },
    { phase: 'measurement', outcome: 'timeout', category: 'OCR', sourceDataset: null },
  ];
  const characterization = characterizeRecords(saved, records);
  assert.equal(characterization.measuredRecords, 3);
  assert.deepEqual(characterization.categoryCounts, [{ label: 'counting', count: 1 }, { label: 'OCR', count: 2 }]);
  assert.equal(characterization.sourceDatasetCounts.find((row) => row.label === 'Not recorded')?.count, 1);
  assert.equal(characterization.outputCharacters.observedCount, 0);
  assert.equal(characterization.outputCharacters.min, null);
  assert.equal(characterization.actualTokens.generated_text_tokens.observedCount, 0);
  assert.ok(!JSON.stringify(characterization.categoryCounts).includes('Warmup'));
});

test('actual token observations never substitute for unrecorded character lengths', () => {
  const saved = v2();
  saved.summary = { ...saved.summary, totalRequests: 1, successfulRequests: 1, failedRequests: 0 };
  const characterization = characterizeRecords(saved, [{ phase: 'measurement', outcome: 'success', category: 'OCR', sourceDataset: 'TEXT_VQA', serverMetrics: { generated_text_tokens: 4, prompt_text_tokens: 12, visual_tokens: 128 } }]);
  assert.equal(characterization.outputCharacters.observedCount, 0);
  assert.equal(characterization.outputCharacters.min, null);
  assert.match(characterization.outputCharacters.reason ?? '', /No output character counts/);
  assert.deepEqual(characterization.actualTokens.generated_text_tokens, { observedCount: 1, min: 4, max: 4 });
  assert.throws(() => characterizeRecords(saved, []), /count differs/);
  assert.equal(characterizeRecords(raw, []).scope, 'unavailable');
});

test('optional server character counts preserve partial coverage and remain distinct from tokens', () => {
  const saved = v2();
  saved.summary = { ...saved.summary, totalRequests: 3, successfulRequests: 2, failedRequests: 1 };
  const characterization = characterizeRecords(saved, [
    { phase: 'measurement', outcome: 'success', serverMetrics: { output_characters: 0, generated_text_tokens: 1 } },
    { phase: 'measurement', outcome: 'success', serverMetrics: { output_characters: -1, generated_text_tokens: null } },
    { phase: 'measurement', outcome: 'timeout', serverMetrics: { output_characters: 100 } },
  ]);
  assert.deepEqual(characterization.outputCharacters, { observedCount: 1, min: 0, max: 0, unit: 'Unicode code points', reason: 'Only some successful responses include a recorded character count.' });
  assert.equal(characterization.actualTokens.generated_text_tokens.min, 1);
});

test('portable campaign identity comes from its manifest despite nested date/run paths', async () => {
  const repository = await mkdtemp(path.join(tmpdir(), 'frontend-portable-'));
  try {
    const campaignRoot = path.join(repository, 'results/campaigns/renamed-export');
    const directory = path.join(campaignRoot, 'trial-attempt/runs/2026-10-01/example');
    await mkdir(directory, { recursive: true });
    await writeFile(path.join(campaignRoot, 'campaign.json'), JSON.stringify({ schema: 'loadgen-campaign', schemaVersion: 2, campaignId: 'canonical-id', spec: { purpose: 'integration' }, trials: [{ runDirectory: 'trial-attempt/runs/2026-10-01/example' }] }));
    const saved = v2();
    saved.summary = { ...saved.summary, totalRequests: 1, successfulRequests: 1, failedRequests: 0 };
    saved.quality = { reportable: true, reasons: [], percentileConvention: 'nearest-rank' };
    const records = '{"phase":"measurement","outcome":"success","category":"OCR","sourceDataset":"TEXT_VQA"}\n';
    saved.requests = { file: 'requests.jsonl', schemaVersion: 3, count: 1, sha256: `sha256:${createHash('sha256').update(records).digest('hex')}` };
    await writeFile(path.join(directory, 'run.json'), JSON.stringify(saved));
    await writeFile(path.join(directory, 'requests.jsonl'), records);
    const index = await syncArchive(repository, path.join(repository, 'frontend'), { series: [] });
    assert.equal(index.issues.length, 0);
    assert.equal(index.runs[0].campaign, 'canonical-id');
    assert.equal(index.runs[0].evidence, 'integration');
    assert.deepEqual(index.runs[0].workloadCharacterization.categoryCounts, [{ label: 'OCR', count: 1 }]);
    assert.equal(parseArchive(index).runs.length, 1);
  } finally { await rm(repository, { recursive: true, force: true }); }
});

test('saved campaign screens must match raw journals and summaries and cannot claim sustainable capacity', () => {
  const hash = 'a'.repeat(64);
  const evidence = [{ runId: raw.runId, campaign: 'canonical-id', requestsSha256: hash, rawSummary: raw.summary, evidence: 'integration', runKind: 'open-loop' }];
  const report = { schema: 'measurement-report', schemaVersion: 1, campaignId: 'canonical-id', purpose: 'integration', runs: [{ runId: raw.runId, requestsSha256: `sha256:${hash}`, summary: raw.summary }], limitations: ['Integration only'], capacityBrackets: [{ variant: 'A', status: 'inconclusive', sustainableCapacityEstablished: false, highestAllPassRate: null, lowestAllFailRate: null, points: [{ offeredRps: 1, trialCount: 1, expectedTrialCount: 5, classification: 'inconclusive', reasons: ['Missing trials'] }], reasons: ['Longer confirmation required'] }] };
  const saved = archiveReport(report, evidence, 'results/campaigns/export/report/report.json', hash);
  assert.equal(parseArchive({ ...archive(), campaignReports: [saved] }).campaignReports.length, 1);
  assert.equal(saved.capacityBrackets[0].sustainableCapacityEstablished, false);
  const stale = structuredClone(report);
  stale.runs[0].summary.successfulThroughputRps = 123;
  assert.throws(() => archiveReport(stale, evidence, '', hash), /does not match/);
  const optimistic = structuredClone(report);
  optimistic.capacityBrackets[0].sustainableCapacityEstablished = true;
  assert.throws(() => archiveReport(optimistic, evidence, '', hash), /Unsupported capacity/);
  assert.throws(() => archiveReport({ ...report, purpose: 'benchmark' }, evidence, '', hash), /integration-only/);
  assert.throws(() => archiveReport({ ...report, runKind: 'isolated' }, evidence, '', hash), /arrival mode differs/);
  const openLoopHtml = renderToStaticMarkup(createElement(CampaignReports, { reports: [saved] }));
  assert.match(openLoopHtml, /Offered \/ s/);
  // Even an older isolated report with placeholder-based screens must hide them.
  const isolated = archiveReport({ ...report, runKind: 'isolated' }, [{ ...evidence[0], runKind: 'isolated' }], '', hash);
  const html = renderToStaticMarkup(createElement(CampaignReports, { reports: [isolated] }));
  assert.match(html, /no offered arrival rate/);
  assert.doesNotMatch(html, /Offered \/ s/);
  assert.doesNotMatch(html, /Highest tested rate/);
});

test('isolated details show sequential timing without an arrival window or scheduled-arrival statistics', () => {
  const run = archiveRun({ ...v2(), runKind: 'isolated' }, options);
  const html = renderToStaticMarkup(createElement(RunDetails, { run }));
  assert.equal(run.offeredRps, null);
  assert.match(html, /Sequential \(no arrival schedule\)/);
  assert.match(html, /Serial completions including drain/);
  assert.doesNotMatch(html, /Completed within arrival window/);
  assert.doesNotMatch(html, /Dispatch lateness/);
  assert.doesNotMatch(html, /Scheduled to completion/);
  assert.ok(!comparisonWarnings(run, run).some((warning) => warning.includes('Offered load is unknown')));
});
