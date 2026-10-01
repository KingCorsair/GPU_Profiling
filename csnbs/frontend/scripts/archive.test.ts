import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { archiveRun, syncArchive } from './archive.mjs';
import { comparisonWarnings, parseArchive } from '../src/archive.ts';

const sourcePath = 'results/loadgen/2026-09-01/02-37-44Z_rps-0.5_8beede1e/run.json';
const bytes = readFileSync(new URL(`../../../${sourcePath}`, import.meta.url));
const raw = JSON.parse(bytes.toString());
const options = { sourcePath, sha256: createHash('sha256').update(bytes).digest('hex') };
const entry = () => archiveRun(structuredClone(raw), options);
const archive = () => ({ schemaVersion: 1, generatedAtUtc: '2026-10-01T00:00:00Z', runs: [entry()], issues: [] });
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
    const records = '{"phase":"measurement"}\n';
    const saved = v2();
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
