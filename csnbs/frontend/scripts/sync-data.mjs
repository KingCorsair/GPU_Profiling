// Presentation adapter only: use saved summaries; never recalculate percentiles.
import { readFile, mkdir, writeFile, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
import { syncArchive } from './archive.mjs';

const frontend = fileURLToPath(new URL('../', import.meta.url));
const repository = path.resolve(frontend, '../..');
const sourcePath = 'results/loadgen/vispruner_under_load_sources.json';
const campaignPath = 'results/loadgen/2026-09-01/vispruner_stock_ab_campaign.json';
const read = (relative) => readFile(path.join(repository, relative));
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const source = JSON.parse(await read(sourcePath));
assert.equal(hash(await read(campaignPath)), source.campaignSha256, 'Campaign changed. Regenerate plotting artifacts first.');
assert.equal(source.campaign.status, 'exploratory');
assert.equal(source.percentile, 'p50');
assert.equal(source.offeredRps, 3, 'This presentation uses the 3 RPS story. Regenerate default plotting artifacts.');
assert.equal(source.visualTokenNum, 128);
assert.equal(source.conditions.length, 4);
assert.equal(source.conditions[0].improvementPercent, 0);
assert.equal(source.conditions[1].improvementPercent, null);
assert.equal(source.conditions[3].improvementPercent, null);
assert.equal(source.conditions[2].controlRunId, '02-35-58Z_rps-3_df52457e');
assert.equal(source.conditions[2].prunedRunId, '03-19-29Z_rps-3_4742a956');
assert.deepEqual(source.series.map((series) => series.visualTokenNum), [576, 128]);

const rawRuns = [];
for (const series of source.series) {
  for (const run of series.runs) {
    const bytes = await read(run.runPath);
    assert.equal(hash(bytes), run.sha256, `Stale plotted source: ${run.runId}`);
    const raw = JSON.parse(bytes);
    assert.equal(raw.runId, run.runId);
    assert.equal(raw.runKind, 'open-loop');
    assert.equal(raw.summary.successfulThroughputRps, run.successfulThroughputRps);
    for (const percentile of ['p50', 'p95', 'p99']) {
      assert.equal(raw.summary.successfulRequestLatencyMs[percentile], run[`${percentile}LatencyMs`]);
    }
    rawRuns.push({ file: `${run.runId}.json`, raw });
  }
}

await mkdir(path.join(frontend, 'public/data/runs'), { recursive: true });
await mkdir(path.join(frontend, 'public/plots'), { recursive: true });
// Keep the published provenance useful without exposing the author's home path.
await writeFile(path.join(frontend, 'public/data/campaign.json'), JSON.stringify({ ...source, campaignPath, sourcePath }, null, 2) + '\n');
for (const { file, raw } of rawRuns) {
  await writeFile(path.join(frontend, 'public/data/runs', file), JSON.stringify(raw, null, 2) + '\n');
}
for (const name of ['vispruner_under_load', 'vispruner_under_load_story']) {
  for (const extension of ['svg', 'png']) {
    await copyFile(path.join(repository, 'results/loadgen', `${name}.${extension}`), path.join(frontend, 'public/plots', `${name}.${extension}`));
  }
}
console.log(`Verified and copied ${rawRuns.length} saved runs and 4 original plots. No metrics recalculated.`);
const archive = await syncArchive(repository, frontend, source);
console.log(`Indexed ${archive.runs.length} saved runs; ${archive.issues.length} files need review.`);
