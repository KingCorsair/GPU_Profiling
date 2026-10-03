import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import test from 'node:test';
import { parseCampaign } from '../src/campaign.ts';

const fixture = JSON.parse(readFileSync(new URL('../../../results/loadgen/vispruner_under_load_sources.json', import.meta.url), 'utf8'));

test('loads the real saved A40 campaign without changing any metrics', () => {
  const campaign = parseCampaign(fixture);
  assert.equal(campaign, fixture);
  assert.equal(campaign.series.length, 2);
});

for (const [name, mutate] of [
  ['missing paired rate', (data: typeof fixture) => data.series[1].runs.pop()],
  ['missing token identity', (data: typeof fixture) => { data.series[1].visualTokenNum = 64; }],
  ['duplicate arrival rate', (data: typeof fixture) => { data.series[1].runs[0].offeredRps = 3; }],
  ['nonfinite latency', (data: typeof fixture) => { data.series[0].runs[0].p50LatencyMs = Infinity; }],
  ['changed story rate', (data: typeof fixture) => { data.offeredRps = 2; }],
  ['filled missing experiment', (data: typeof fixture) => { data.conditions[1].improvementPercent = 20; }],
  ['different model campaign', (data: typeof fixture) => { data.campaign.runDateUtc = '2026-09-12'; }],
] as const) {
  test(`rejects ${name} before chart rendering`, () => {
    const changed = structuredClone(fixture);
    mutate(changed);
    assert.throws(() => parseCampaign(changed), /Incomplete or incompatible/);
  });
}

test('rejects unavailable or malformed data', () => {
  for (const value of [null, [], {}, 'not JSON', { campaign: null }]) assert.throws(() => parseCampaign(value));
});
