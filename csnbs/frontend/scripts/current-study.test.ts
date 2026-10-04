import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test, { after, before } from 'node:test';
import { syncArchive } from './archive.mjs';
import { buildCurrentStudy, syncCurrentStudy } from './current-study.mjs';
import { parseCurrentStudy, type CurrentStudy } from '../src/current-study.ts';

const repository = fileURLToPath(new URL('../../../', import.meta.url));
const primaryDirectory = 'results/campaigns/2026-10-01/llava-oct01-primary';
const collectionPath = 'results/provenance/2026-10-01/collection-index.json';
const handoffPath = `${primaryDirectory}/accuracy-handoff/accuracy-throughput.json`;
let scratch: string;
let archive: Awaited<ReturnType<typeof syncArchive>>;
let study: CurrentStudy;
let sourceFiles: string[];

before(async () => {
  scratch = await mkdtemp(path.join(os.tmpdir(), 'october-study-test-'));
  const legacy = JSON.parse(await readFile(path.join(repository, 'results/loadgen/vispruner_under_load_sources.json'), 'utf8'));
  archive = await syncArchive(repository, scratch, legacy);
  study = parseCurrentStudy((await buildCurrentStudy(repository, archive)).study);
  const collection = JSON.parse(await readFile(path.join(repository, collectionPath), 'utf8'));
  sourceFiles = [collectionPath, handoffPath, ...['report.md', 'campaign_overview.svg', 'campaign_overview.png'].map((name) => `${primaryDirectory}/report/${name}`)];
  for (const campaign of collection.campaigns) sourceFiles.push(campaign.report, campaign.storageVerification, `results/campaigns/2026-10-01/${campaign.campaignId}/campaign.json`);
});
after(async () => { if (scratch) await rm(scratch, { recursive: true, force: true }); });

async function withChangedEvidence(sourcePath: string, change: (bytes: Buffer) => Buffer | string, check: (root: string) => Promise<void>) {
  const root = await mkdtemp(path.join(scratch, 'changed-'));
  for (const file of sourceFiles) {
    const target = path.join(root, file);
    await mkdir(path.dirname(target), { recursive: true });
    if (file === sourcePath) await writeFile(target, change(await readFile(path.join(repository, file))));
    else await symlink(path.join(repository, file), target);
  }
  await check(root);
}

test('copies the saved paired inference, per-trial boundaries and pending accuracy without recomputing', async () => {
  const report = JSON.parse(await readFile(path.join(repository, primaryDirectory, 'report/report.json'), 'utf8'));
  for (const comparison of study.primary.comparisons) {
    const original = report.comparisons.find((row: { offeredRps: number }) => row.offeredRps === comparison.offeredRps);
    assert.deepEqual(comparison.throughput, original.throughput);
    assert.deepEqual(comparison.latency, original.latency);
  }
  assert.equal(study.totals.measuredRequests, 8820);
  assert.equal(study.primary.trials.length, 20);
  for (const trial of study.primary.trials) {
    const original = report.runs.find((run: { runId: string }) => run.runId === trial.runId);
    assert.equal(trial.withinWindowThroughputRps, original.summary.successfulThroughputWithinWindowRps);
    assert.equal(trial.includingDrainThroughputRps, original.summary.successfulThroughputIncludingDrainRps);
    assert.equal(trial.outstandingAtWindowEnd, original.summary.outstandingAtWindowEnd);
    assert.equal(trial.drainMs, original.summary.drainMs);
    assert.equal(trial.latencyMs.p95, original.summary.successfulRequestLatencyMs.p95);
    assert.equal(trial.latencyMs.p99, null);
  }
  assert.equal(study.accuracy.status, 'pending');
  assert.deepEqual(study.accuracy.points, []);
  assert.equal(study.accuracy.requiredConfigurationCount, 2);
});

test('publishes the original evidence bytes so the downloadable hashes remain valid', async () => {
  const target = path.join(scratch, 'published');
  const published = await syncCurrentStudy(repository, target, archive);
  for (const evidence of [...Object.values(published.downloads), published.accuracy.download, ...published.campaigns.map((campaign) => campaign.report)]) {
    assert.deepEqual(await readFile(path.join(target, 'public', evidence.downloadPath)), await readFile(path.join(repository, evidence.sourcePath)));
  }
  assert.deepEqual(parseCurrentStudy(JSON.parse(await readFile(path.join(target, 'public/data/current-study.json'), 'utf8'))), study);
});

test('rejects a report whose source bytes changed after collection verification', async () => {
  await withChangedEvidence(`${primaryDirectory}/report/report.json`, (bytes) => `${bytes.toString()}\n`, async (root) => {
    await assert.rejects(buildCurrentStudy(root, archive), /changed evidence/);
  });
});

test('rejects missing or nonresearch archive evidence', async () => {
  await assert.rejects(buildCurrentStudy(repository, { ...archive, campaignReports: archive.campaignReports.filter((report) => report.campaignId !== 'llava-oct01-primary') }), /verified archive/);
  await assert.rejects(buildCurrentStudy(repository, { ...archive, campaignReports: archive.campaignReports.map((report) => report.campaignId === 'llava-oct01-primary' ? { ...report, purpose: 'integration' } : report) }), /integration-only/);
  await assert.rejects(buildCurrentStudy(repository, { ...archive, runs: archive.runs.map((run) => run.campaign === 'llava-oct01-primary' ? { ...run, evidence: 'review' } : run) }), /nonreportable research evidence/);
});

test('rejects changed raw metadata even when the saved summary and request bytes match', async () => {
  const saved = archive.runs.find((run) => run.campaign === 'llava-oct01-primary');
  const changed = JSON.parse(await readFile(path.join(repository, saved.sourcePath), 'utf8'));
  changed.source.gitCommit = '0'.repeat(40);
  const changedHash = createHash('sha256').update(JSON.stringify(changed)).digest('hex');
  // This is the indexed manifest a fresh archive sync would produce: same summary
  // and request hash, changed metadata and therefore a different manifest hash.
  const changedArchive = { ...archive, runs: archive.runs.map((run) => run === saved ? { ...run, sha256: changedHash, gitCommit: changed.source.gitCommit } : run) };
  await assert.rejects(buildCurrentStudy(repository, changedArchive), /raw run manifest differs from verified storage inventory/);
});

for (const [name, mutate] of [
  ['another report hash', (handoff: any) => { handoff.inputs.report.sha256 = '0'.repeat(64); }],
  ['fabricated accuracy', (handoff: any) => { handoff.points = [{ accuracy: 1 }]; }],
  ['missing configuration', (handoff: any) => { handoff.requiredExecutionIdentities.pop(); }],
  ['mismatched execution identity', (handoff: any) => { handoff.requiredExecutionIdentities[0].identity.serverGitCommit = '0'.repeat(40); }],
  ['mismatched important ratio', (handoff: any) => { handoff.requiredExecutionIdentities[0].identity.configuration.important_ratio = 0.7; }],
  ['mismatched checkpoint file', (handoff: any) => { handoff.requiredExecutionIdentities[0].identity.checkpointFiles[0].sha256 = '0'.repeat(64); }],
  ['mismatched runtime', (handoff: any) => { handoff.requiredExecutionIdentities[0].identity.runtime.torch = 'different'; }],
] as const) {
  test(`rejects pending accuracy with ${name}`, async () => {
    await withChangedEvidence(handoffPath, (bytes) => {
      const changed = JSON.parse(bytes.toString()); mutate(changed); return JSON.stringify(changed);
    }, async (root) => { await assert.rejects(buildCurrentStudy(root, archive), /October study:/); });
  });
}

for (const [name, mutate] of [
  ['missing paired rate', (value: CurrentStudy) => { value.primary.comparisons.pop(); }],
  ['wrong throughput boundary', (value: CurrentStudy) => { value.primary.comparisons[0].throughput.metric = 'successfulThroughputIncludingDrainRps'; }],
  ['wrong effect direction', (value: CurrentStudy) => { value.primary.comparisons[0].latency.direction = 'higher-is-better'; }],
  ['reversed uncertainty bounds', (value: CurrentStudy) => { value.primary.comparisons[0].latency.ci95 = [15, 5]; }],
  ['missing saved uncertainty', (value: any) => { value.primary.comparisons[0].latency.ci95 = null; }],
  ['invented p99', (value: any) => { value.primary.trials[0].latencyMs.p99 = 123; }],
  ['nonfinite latency', (value: CurrentStudy) => { value.primary.trials[0].latencyMs.p50 = Infinity; }],
  ['duplicate trial', (value: CurrentStudy) => { value.primary.trials[1] = value.primary.trials[0]; }],
  ['malformed trial identity', (value: CurrentStudy) => { value.primary.trials[0].trialId = 'not-a-paired-trial'; }],
  ['mismatched trial rate', (value: CurrentStudy) => { value.primary.trials[0].trialId = 'r1-p1-v0'; }],
  ['fabricated accuracy point', (value: any) => { value.accuracy.points = [{ accuracy: 1 }]; }],
  ['wrong campaign total', (value: CurrentStudy) => { value.campaigns[0].measuredRequests += 1; }],
  ['mismatched token variant', (value: CurrentStudy) => { value.primary.trials[0].visualTokenNum = 128; }],
  ['external download URL', (value: CurrentStudy) => { value.downloads.report.downloadPath = 'https://example.com/report'; }],
] as const) {
  test(`rejects ${name} before rendering a research claim`, () => {
    const changed = structuredClone(study); mutate(changed);
    assert.throws(() => parseCurrentStudy(changed), /Incomplete or incompatible October study/);
  });
}
