// Public export only: keep original evidence bytes or omit the whole download.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseCurrentStudy } from '../src/current-study.ts';
import { parseArchive } from '../src/archive.ts';
import { parseCampaign } from '../src/campaign.ts';

export function containsPersonalName(value) {
  return /rithvik|kancherla/i.test(value);
}

function artifactContainsName(bytes, filename) {
  if (containsPersonalName(bytes.toString('utf8'))) return true;
  // Also detect JSON-escaped names, rather than scanning only literal bytes.
  if (filename.endsWith('.json')) return containsPersonalName(JSON.stringify(JSON.parse(bytes.toString('utf8'))));
  if (filename.endsWith('.jsonl')) return bytes.toString('utf8').split('\n').filter(Boolean).some((line) => containsPersonalName(JSON.stringify(JSON.parse(line))));
  return false;
}

export function prepareSharedData(input, excludedPaths) {
  parseCurrentStudy(input.study); parseArchive(input.archive); parseCampaign(input.campaign);
  assert(!input.study.publicPreview && !input.archive.publicPreview && !input.campaign.publicPreview, 'Use the complete verified build as input.');
  const { study, archive, campaign } = structuredClone(input);
  study.publicPreview = archive.publicPreview = campaign.publicPreview = true;
  const available = (item) => item && !excludedPaths.has(item.downloadPath) ? item : null;
  for (const row of study.campaigns) row.report = available(row.report);
  for (const trial of study.primary.trials) if (excludedPaths.has(trial.downloadPath)) trial.downloadPath = null;
  study.accuracy.download = available(study.accuracy.download);
  for (const key of Object.keys(study.downloads)) study.downloads[key] = available(study.downloads[key]);
  for (const run of archive.runs) {
    if (excludedPaths.has(run.downloadPath)) run.downloadPath = null;
    if (excludedPaths.has(run.requestDownloadPath)) run.requestDownloadPath = null;
  }
  for (const report of archive.campaignReports) if (excludedPaths.has(report.downloadPath)) report.downloadPath = null;
  parseCurrentStudy(study); parseArchive(archive); parseCampaign(campaign);
  const result = { 'current-study.json': study, 'archive.json': archive, 'campaign.json': campaign };
  for (const value of Object.values(result)) assert(!containsPersonalName(JSON.stringify(value)), 'Presentation metadata contains a personal name; inspect locally before publishing.');
  return result;
}

export async function prepareSharedBuild(buildDirectory, destination) {
  assert(['dist-share', 'dist-live-share'].includes(path.basename(destination)) && path.dirname(destination) === path.dirname(buildDirectory) && destination !== buildDirectory, 'Sharing output must be a sibling dist-share or dist-live-share directory.');
  const read = async (name) => JSON.parse(await readFile(path.join(buildDirectory, 'data', name), 'utf8'));
  const input = { study: await read('current-study.json'), archive: await read('archive.json'), campaign: await read('campaign.json') };
  parseCurrentStudy(input.study); parseArchive(input.archive); parseCampaign(input.campaign);
  const staging = await mkdtemp(path.join(path.dirname(destination), '.share-build-'));
  const excluded = new Set();
  const copied = new Set();
  try {
    async function copyArtifact(url, required = false) {
      if (!url || copied.has(url) || excluded.has(url)) return;
      assert(/^\/(?:data|plots|assets)\/[\w./-]+$/.test(url) || ['/index.html', '/favicon.svg', '/licenses/uiverse-smooth-termite-31.txt'].includes(url), 'Unexpected public artifact path.');
      const relative = url.slice(1);
      const source = path.resolve(buildDirectory, relative);
      assert(source.startsWith(`${path.resolve(buildDirectory)}${path.sep}`) && !relative.split('/').includes('..'), 'Artifact escapes build directory.');
      assert((await lstat(source)).isFile(), 'Only regular files may be shared.');
      const bytes = await readFile(source);
      const expectedHash = /^\/data\/(?:archive|current-study)\/([a-f0-9]{64})\./.exec(url)?.[1];
      if (expectedHash) assert.equal(createHash('sha256').update(bytes).digest('hex'), expectedHash, 'Download differs from its original evidence hash.');
      if (artifactContainsName(bytes, relative)) {
        assert(!required, 'A required dashboard asset contains a personal name; inspect locally before publishing.');
        excluded.add(url);
        return;
      }
      const target = path.join(staging, relative);
      await mkdir(path.dirname(target), { recursive: true });
      await writeFile(target, bytes);
      copied.add(url);
    }
    for (const name of ['index.html', 'favicon.svg']) await copyArtifact(`/${name}`, true);
    await copyArtifact('/licenses/uiverse-smooth-termite-31.txt', true);
    assert((await readFile(path.join(staging, 'index.html'), 'utf8')).includes('name="robots" content="noindex, nofollow"'), 'Public preview must retain the noindex directive.');
    for (const entry of await readdir(path.join(buildDirectory, 'assets'), { withFileTypes: true })) {
      assert(entry.isFile() && /\.(js|css)$/.test(entry.name), 'Unexpected asset in public build.');
      await copyArtifact(`/assets/${entry.name}`, true);
    }
    const { study, archive, campaign } = input;
    for (const run of archive.runs) {
      await copyArtifact(run.downloadPath);
      await copyArtifact(run.requestDownloadPath);
    }
    for (const report of archive.campaignReports) await copyArtifact(report.downloadPath);
    for (const row of study.campaigns) await copyArtifact(row.report.downloadPath);
    for (const trial of study.primary.trials) await copyArtifact(trial.downloadPath);
    for (const item of [...Object.values(study.downloads), study.accuracy.download]) await copyArtifact(item.downloadPath);
    // Historical links are fixed in the UI; fail closed if any become unsafe.
    for (const series of campaign.series) for (const run of series.runs) await copyArtifact(`/data/runs/${run.runId}.json`, true);
    for (const name of ['vispruner_under_load', 'vispruner_under_load_story']) for (const ext of ['svg', 'png']) await copyArtifact(`/plots/${name}.${ext}`, true);
    const data = prepareSharedData(input, excluded);
    await mkdir(path.join(staging, 'data'), { recursive: true });
    for (const [name, value] of Object.entries(data)) await writeFile(path.join(staging, 'data', name), `${JSON.stringify(value, null, 2)}\n`);
    await rm(destination, { recursive: true, force: true });
    await rename(staging, destination);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
  console.log(`Public preview prepared: ${excluded.size} downloads with personal names omitted; source commits and measurements preserved.`);
  return { excluded: [...excluded], copied: [...copied] };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const frontend = fileURLToPath(new URL('../', import.meta.url));
  const live = process.argv.includes('--live');
  await prepareSharedBuild(path.join(frontend, live ? 'dist-live' : 'dist'), path.join(frontend, live ? 'dist-live-share' : 'dist-share'));
}
