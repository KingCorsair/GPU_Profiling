import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test, { after, before } from 'node:test';
import { syncArchive } from './archive.mjs';
import { syncCurrentStudy } from './current-study.mjs';
import { containsPersonalName, prepareSharedBuild, prepareSharedData } from './prepare-share.mjs';
import { parseCurrentStudy } from '../src/current-study.ts';
import { parseArchive } from '../src/archive.ts';

const repository = fileURLToPath(new URL('../../../', import.meta.url));
let scratch: string;
let input: any;

before(async () => {
  scratch = await mkdtemp(path.join(os.tmpdir(), 'dashboard-sharing-'));
  const source = JSON.parse(await readFile(path.join(repository, 'results/loadgen/vispruner_under_load_sources.json'), 'utf8'));
  const archive = await syncArchive(repository, scratch, source);
  const study = await syncCurrentStudy(repository, scratch, archive);
  const campaign = { ...source, campaignPath: 'results/loadgen/2026-09-01/vispruner_stock_ab_campaign.json', sourcePath: 'results/loadgen/vispruner_under_load_sources.json' };
  input = { study, archive, campaign };
  const build = path.join(scratch, 'dist');
  await cp(path.join(scratch, 'public'), build, { recursive: true });
  await mkdir(path.join(build, 'assets'));
  await mkdir(path.join(build, 'data/runs'), { recursive: true });
  await mkdir(path.join(build, 'plots'));
  await mkdir(path.join(build, 'licenses'));
  await cp(path.join(repository, 'csnbs/frontend/public/licenses/uiverse-smooth-termite-31.txt'), path.join(build, 'licenses/uiverse-smooth-termite-31.txt'));
  await writeFile(path.join(build, 'index.html'), '<meta name="robots" content="noindex, nofollow">');
  await writeFile(path.join(build, 'favicon.svg'), '<svg/>');
  await writeFile(path.join(build, 'assets/app.js'), 'console.log("dashboard");');
  await writeFile(path.join(build, 'data/campaign.json'), JSON.stringify(campaign));
  for (const series of source.series) for (const run of series.runs) await cp(path.join(repository, run.runPath), path.join(build, 'data/runs', `${run.runId}.json`));
  for (const name of ['vispruner_under_load', 'vispruner_under_load_story']) for (const ext of ['svg', 'png']) await cp(path.join(repository, 'results/loadgen', `${name}.${ext}`), path.join(build, 'plots', `${name}.${ext}`));
});
after(async () => { if (scratch) await rm(scratch, { recursive: true, force: true }); });

test('the sharing build excludes named downloads, preserves published bytes and source commits, and leaves the full build untouched', async () => {
  const build = path.join(scratch, 'dist');
  const output = path.join(scratch, 'dist-share');
  const original = structuredClone(input);
  await mkdir(output);
  await writeFile(path.join(output, 'stale-personal-file.json'), '{"path":"/workspace/rithvik-results/old"}');
  await writeFile(path.join(build, 'unreferenced-secret.txt'), 'rithvik');
  const result = await prepareSharedBuild(build, output);
  assert(result.copied.includes('/licenses/uiverse-smooth-termite-31.txt'));
  assert(result.excluded.length > 0);
  for (const url of result.excluded) {
    await assert.rejects(readFile(path.join(output, url)), { code: 'ENOENT' });
    assert(containsPersonalName((await readFile(path.join(build, url))).toString()));
  }
  for (const url of result.copied) assert.deepEqual(await readFile(path.join(output, url)), await readFile(path.join(build, url)));
  const publicStudy = parseCurrentStudy(JSON.parse(await readFile(path.join(output, 'data/current-study.json'), 'utf8')));
  const publicArchive = parseArchive(JSON.parse(await readFile(path.join(output, 'data/archive.json'), 'utf8')));
  assert.deepEqual(publicStudy.primary.comparisons, input.study.primary.comparisons);
  assert.equal(publicStudy.primary.gitCommit, input.study.primary.gitCommit);
  assert.deepEqual(publicStudy.totals, input.study.totals);
  assert.deepEqual(publicArchive.runs.map((run) => [run.summary, run.gitCommit]), input.archive.runs.map((run: any) => [run.summary, run.gitCommit]));
  assert(publicStudy.primary.trials.every((trial) => trial.downloadPath === null));
  assert(publicArchive.runs.some((run) => run.requestDownloadPath !== null));
  assert.deepEqual(input, original);
  const files = await readdir(output, { recursive: true });
  assert(!files.includes('stale-personal-file.json') && !files.includes('unreferenced-secret.txt'));
  for (const file of files) if (/\.(json|jsonl|svg|js|css|html)$/.test(file)) assert(!containsPersonalName(await readFile(path.join(output, file), 'utf8')), file);
});

test('unexpected names in visible metadata fail the export rather than being silently shared', () => {
  const changed = structuredClone(input);
  changed.study.primary.limitations.push('Collected in /workspace/rithvik-results');
  assert.throws(() => prepareSharedData(changed, new Set()), /personal name/);
});

test('missing downloads require an explicit public-preview contract', () => {
  const result = prepareSharedData(input, new Set([input.study.primary.trials[0].downloadPath, input.archive.runs[0].downloadPath]));
  delete result['current-study.json'].publicPreview;
  delete result['archive.json'].publicPreview;
  assert.throws(() => parseCurrentStudy(result['current-study.json']), /Incomplete/);
  assert.throws(() => parseArchive(result['archive.json']), /Incomplete/);
});

test('encoded personal names in downloadable JSON are excluded too', async () => {
  const build = path.join(scratch, 'encoded', 'dist');
  await cp(path.join(scratch, 'dist'), build, { recursive: true });
  const target = input.campaign.series[0].runs[0].runId;
  // Historical downloads are required, so the export must fail closed.
  await writeFile(path.join(build, 'data/runs', `${target}.json`), '{"name":"\\u0072ithvik"}');
  await assert.rejects(prepareSharedBuild(build, path.join(scratch, 'encoded', 'dist-share')), /personal name/);
});
