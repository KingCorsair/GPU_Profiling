import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test, type TestContext } from 'node:test';
import { baselineRepeatability, comparePairs, comparabilityReasons, readVerifiedRun, type VerifiedRun } from './analyze.js';
import { classifyCapacity } from './capacity.js';
import { buildWorkloadOrder, describeWorkload, runQualityReasons, summarizeRun, type DatasetPayload, type LoadConfig, type LoadRun, type RequestResult, type RunManifest } from './loadgen.js';

function fixture(id: string, latency = 20, serial = false, seed = 7): VerifiedRun {
  const payloads: DatasetPayload[] = Array.from({ length: 12 }, (_, index) => ({ image_b64: 'eA==', question: `Q${index}`, workloadIndex: index, questionId: `q${index}`, category: index % 2 ? 'ocr' : 'count', sourceDataset: 'fixture' }));
  const config: LoadConfig = { endpoint: 'http://localhost/infer', requestsPerSecond: 2, durationSeconds: 120, measuredRequests: 240, timeoutMs: 120000,
    datasetPath: 'fixture.json', warmupRequests: 10, seed, settleMs: 0, runKind: 'open-loop', outputDir: '/tmp', runDeadlineMs: null };
  const workload = describeWorkload(payloads, 240, seed), order = buildWorkloadOrder(payloads, 240, seed);
  let previousEnd = 0;
  const makeRow = (phase: 'warmup' | 'measurement', sequence: number): RequestResult => {
    const index = phase === 'measurement' ? order[sequence]! : sequence % 12;
    const payload = payloads[index]!;
    const scheduledAtMs = phase === 'warmup' ? sequence * 2 : 1000 + sequence * 500;
    const sentAtMs = scheduledAtMs;
    const completedAtMs = phase === 'warmup' ? sentAtMs + 1 : (serial ? Math.max(sentAtMs, previousEnd) : sentAtMs) + latency;
    if (phase === 'measurement') previousEnd = completedAtMs;
    return { requestId: `${id}:${phase}:${sequence}`, phase, sequence, workloadIndex: index, questionId: payload.questionId,
      category: payload.category, sourceDataset: payload.sourceDataset, scheduledAtMs, sentAtMs, completedAtMs,
      latencyMs: completedAtMs - sentAtMs, dispatchLatenessMs: 0, plannedToCompleteMs: completedAtMs - scheduledAtMs,
      status: 200, error: null, outcome: 'success', serverRequestId: `${id}:${phase}:${sequence}`,
      serverMetrics: { service_ms: phase === 'warmup' ? 1 : latency, queue_ms: null },
      unattributedClientMs: phase === 'warmup' ? 0 : completedAtMs - sentAtMs - latency };
  };
  const warmupResults = Array.from({ length: 10 }, (_, sequence) => makeRow('warmup', sequence));
  const results = Array.from({ length: 240 }, (_, sequence) => makeRow('measurement', sequence));
  const run: LoadRun = { plannedStartMs: 1000, plannedEndMs: 121000, finishedMs: Math.max(121000, previousEnd), results, warmupResults, status: 'complete', stopReason: null };
  const requests = [...warmupResults, ...results];
  const manifest: RunManifest = {
    schema: 'loadgen-run', schemaVersion: 2, runId: id, runKind: 'open-loop', recordedAtUtc: '2026-10-01T12:00:00Z', status: 'complete', stopReason: null,
    timing: { clock: 'performance.now', performanceTimeOriginUnixMs: 0, plannedStartMs: 1000, plannedEndMs: 121000, finishedMs: run.finishedMs,
      plannedStartAtUtc: new Date(1000).toISOString(), plannedEndAtUtc: new Date(121000).toISOString(), finishedAtUtc: new Date(run.finishedMs).toISOString() },
    config, workload, source: { gitCommit: 'harness-revision', gitDirty: false }, hardware: { gpuModels: ['A40'], source: 'server-health' },
    server: { healthUrl: 'http://localhost/health', service: 'fixture', pid: 1, modelId: 'llava', checkpointRevision: 'weights-revision',
      configuration: { model_id: 'llava', visual_token_num: 576, important_ratio: 0.5, max_new_tokens: 64, implementation: 'fixture-v1', do_sample: false, use_cache: true, eos_policy: 'natural', prompt_template: 'fixture', batch_size: 1, dtype: 'float16' }, error: null,
      mode: 'model', modelLoaded: true, source: { gitCommit: 'server-revision', gitDirty: false },
      hardware: { gpuModels: ['A40'], devices: [{ name: 'A40', uuid: 'GPU-fixture' }] }, runtime: { python: '3.12' } },
    summary: summarizeRun(run), warmup: { requested: 10, completed: 10, successful: 10, isolated: true, settleMs: 0 },
    requests: { file: 'requests.jsonl', schemaVersion: 3, count: requests.length, sha256: '' },
    quality: { reportable: true, reasons: [], percentileConvention: 'nearest-rank' },
  };
  assert.deepEqual(runQualityReasons(manifest), []);
  return { manifest, requests, directory: `/tmp/${id}` };
}
async function save(t: TestContext, run: VerifiedRun): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'analysis-v2-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const bytes = run.requests.map((row) => JSON.stringify(row)).join('\n') + '\n';
  run.manifest.requests.sha256 = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  await writeFile(join(directory, 'requests.jsonl'), bytes);
  await writeFile(join(directory, 'run.json'), JSON.stringify(run.manifest));
  return directory;
}
function refreshSummary(run: VerifiedRun): void {
  const m=run.manifest,t=m.timing;
  t.plannedStartAtUtc=new Date(t.performanceTimeOriginUnixMs+t.plannedStartMs).toISOString();
  t.plannedEndAtUtc=new Date(t.performanceTimeOriginUnixMs+t.plannedEndMs).toISOString();
  t.finishedAtUtc=new Date(t.performanceTimeOriginUnixMs+t.finishedMs).toISOString();
  m.summary=summarizeRun({plannedStartMs:t.plannedStartMs,plannedEndMs:t.plannedEndMs,finishedMs:t.finishedMs,
    results:run.requests.filter(row=>row.phase==='measurement'),warmupResults:run.requests.filter(row=>row.phase==='warmup'),status:m.status,stopReason:m.stopReason});
  const reasons=runQualityReasons(m);m.quality={reportable:reasons.length===0,reasons,percentileConvention:'nearest-rank'};
}
function pairs(latencies: number[]) {
  return latencies.map((latency, index) => ({ baseline: fixture(`a-${index}`, 20), candidate: fixture(`b-${index}`, latency), blockId: `pair-${index}` }));
}

test('reader verifies raw hashes, phase-local sequences, workload and recomputed summary', async (t) => {
  const run = fixture('valid'); const directory = await save(t, run);
  const read = await readVerifiedRun(directory);
  assert.equal(read.requests.length, 250); assert.equal(read.manifest.summary.totalRequests, 240);
  assert.equal(read.manifest.summary.successfulRequestLatencyMs.p95, 20);
  assert.equal(read.manifest.summary.successfulRequestLatencyMs.p99, null);
});

test('reader rejects tampered bytes, cached summaries and unsupported versions', async (t) => {
  const raw = fixture('raw'); const rawDir = await save(t, raw);
  await writeFile(join(rawDir, 'requests.jsonl'), '{}\n');
  await assert.rejects(readVerifiedRun(rawDir), /SHA-256/);
  const badSummary = fixture('summary'); badSummary.manifest.summary.successfulRequests++;
  await assert.rejects(readVerifiedRun(await save(t, badSummary)), /summary/);
  const old = fixture('old'); (old.manifest as unknown as { schemaVersion: number }).schemaVersion = 99;
  await assert.rejects(readVerifiedRun(await save(t, old)), /Unsupported run schema/);
});

test('reader rejects missing coverage, altered workload, invalid timing, and fabricated quality', async (t) => {
  const missing = fixture('missing'); missing.requests.pop(); missing.manifest.requests.count--;
  await assert.rejects(readVerifiedRun(await save(t, missing)), /budget/);
  const order = fixture('order'); order.requests[11]!.questionId = 'fabricated';
  await assert.rejects(readVerifiedRun(await save(t, order)), /workload sequence/);
  const timing = fixture('timing'); timing.requests[10]!.latencyMs = -1;
  await assert.rejects(readVerifiedRun(await save(t, timing)), /Latency identity/);
  const quality = fixture('quality'); quality.manifest.server.mode = 'fake';
  await assert.rejects(readVerifiedRun(await save(t, quality)), /Reportability/);
});

test('reader binds window and terminal clocks to declared duration even when summaries are rewritten', async (t) => {
  const changed=fixture('rewritten-window');changed.manifest.timing.plannedEndMs+=90000;
  changed.manifest.timing.finishedMs=changed.manifest.timing.plannedEndMs;refreshSummary(changed);
  assert.equal(changed.manifest.quality.reportable,true);
  await assert.rejects(readVerifiedRun(await save(t,changed)),/window differs from configured duration/);
  changed.manifest.config.durationSeconds+=90;
  await assert.rejects(readVerifiedRun(await save(t,changed)),/duration\/rate differs/);
  const early=fixture('early-finish');early.manifest.timing.finishedMs=120600;refreshSummary(early);
  await assert.rejects(readVerifiedRun(await save(t,early)),/finished before prescribed window/);
  early.manifest.timing.finishedMs=120000;refreshSummary(early);
  await assert.rejects(readVerifiedRun(await save(t,early)),/Terminal request completes after/);
  const utc=fixture('utc-changed');utc.manifest.timing.plannedEndAtUtc=new Date(123000).toISOString();
  await assert.rejects(readVerifiedRun(await save(t,utc)),/UTC timestamp differs/);
});

test('clock validation preserves isolated actual windows, aborted future cancellations and floating point tolerance', async (t) => {
  const isolated=fixture('isolated-window');isolated.manifest.runKind='isolated';isolated.manifest.config.runKind='isolated';
  isolated.manifest.timing.finishedMs=120520;isolated.manifest.timing.plannedEndMs=120520;refreshSummary(isolated);
  assert.equal((await readVerifiedRun(await save(t,isolated))).manifest.summary.measurementWindowSeconds,119.52);
  const aborted=fixture('aborted-window');aborted.manifest.status='aborted';aborted.manifest.stopReason='test interruption';
  aborted.manifest.timing.finishedMs=1101;
  for(const row of aborted.requests.filter(row=>row.phase==='measurement'&&row.sequence>0)) Object.assign(row,{
    sentAtMs:null,completedAtMs:1100,latencyMs:null,dispatchLatenessMs:null,plannedToCompleteMs:null,status:null,error:'test interruption',
    outcome:'cancelled',serverRequestId:null,serverMetrics:null,unattributedClientMs:null});
  refreshSummary(aborted);
  assert.equal((await readVerifiedRun(await save(t,aborted))).manifest.status,'aborted');
  const precise=fixture('floating-window');precise.manifest.timing.plannedEndMs+=.00001;precise.manifest.timing.finishedMs+=.00001;refreshSummary(precise);
  await readVerifiedRun(await save(t,precise));
});

test('campaign rejects an extended standalone duration that still rounds to the same arrival count', async (t) => {
  const {planCampaign,verifyTrialRun}=await import('./campaign.js');
  const campaign=planCampaign({campaignId:'fixed-duration',purpose:'ab',endpoint:'http://localhost/infer',datasetPath:'fixture.json',rates:[2],
    repetitions:1,measuredRequests:240,warmupRequests:10,seed:7,timeoutMs:120000,startupTimeoutMs:1000,gpuExclusive:true,
    variants:[{id:'A',command:['unused'],expected:{model_id:'llava'}},{id:'B',command:['unused'],expected:{model_id:'llava'}}]});
  const run=fixture('duration-policy');run.manifest.config.durationSeconds=120.25;
  run.manifest.timing.plannedEndMs=121250;run.manifest.timing.finishedMs=121250;refreshSummary(run);
  const verified=await readVerifiedRun(await save(t,run));
  assert.throws(()=>verifyTrialRun(campaign,campaign.trials[0]!,verified),/duration differs from prescribed trial/);
});

test('paired inference resolves improvement and regression, while A/A remains inconclusive', () => {
  const faster = comparePairs(pairs([16, 15, 16, 17, 16]), 'p50', 12);
  assert.equal(faster.conclusion, 'improvement'); assert.equal(faster.effectPercent, 20); assert.ok(faster.ci95![0] > 0);
  const slower = comparePairs(pairs([24, 25, 24, 23, 24]), 'p50', 12);
  assert.equal(slower.conclusion, 'regression'); assert.equal(slower.effectPercent, -20);
  const unchanged = comparePairs(pairs([20, 20, 20, 20, 20]), 'p50', 12);
  assert.equal(unchanged.conclusion, 'inconclusive'); assert.deepEqual(unchanged.ci95, [0, 0]);
  assert.deepEqual(faster, comparePairs(pairs([16, 15, 16, 17, 16]), 'p50', 12));
});

test('mixed noisy effects, insufficient pairs, missing tails and correlated blocks cannot claim improvement', () => {
  assert.equal(comparePairs(pairs([10, 30, 10, 30, 20]), 'p50').conclusion, 'inconclusive');
  const short = comparePairs(pairs([10, 10]), 'p50'); assert.equal(short.ci95, null); assert.equal(short.conclusion, 'inconclusive');
  const missingTail = comparePairs(pairs([10, 10, 10, 10, 10]), 'p99'); assert.equal(missingTail.effectPercent, null);
  const correlated = pairs([10, 10, 10, 10, 10]); correlated[1]!.blockId = correlated[0]!.blockId;
  assert.equal(comparePairs(correlated, 'p50').conclusion, 'inconclusive');
  assert.ok(comparePairs(correlated, 'p50').reasons.some((reason) => reason.includes('Duplicate')));
});

test('comparisons reject workload/hardware/decoding changes and distinguish token and implementation comparisons', () => {
  const a = fixture('a'), b = fixture('b');
  b.manifest.server.configuration!.visual_token_num = 128;
  assert.deepEqual(comparabilityReasons(a.manifest, b.manifest), []);
  assert.ok(comparabilityReasons(a.manifest, b.manifest, 'implementation').includes('Unmatched effective configuration'));
  b.manifest.server.configuration!.max_new_tokens = 128;
  assert.ok(comparabilityReasons(a.manifest, b.manifest).includes('Unmatched effective configuration'));
  b.manifest.workload.measuredOrderHash = 'other';
  assert.ok(comparabilityReasons(a.manifest, b.manifest).includes('Unmatched measured request order'));
  b.manifest.server.hardware!.devices = [{ name: 'A40', uuid: 'another-GPU' }];
  assert.ok(comparabilityReasons(a.manifest, b.manifest).includes('Unmatched server hardware identity'));
  const mixed = pairs([10, 10, 10, 10, 10]); mixed[0]!.candidate.manifest.quality = { reportable: false, reasons: ['dirty'], percentileConvention: 'nearest-rank' };
  assert.equal(comparePairs(mixed, 'p50').conclusion, 'inconclusive');
});

test('instrumentation comparisons permit only the instrumentation field to change', () => {
  const a=fixture('observations-off'),b=fixture('observations-on');
  a.manifest.server.configuration!.instrumentation='handler-service-wall-v1';
  b.manifest.server.configuration!.instrumentation='handler-stage-token-observations-v2';
  assert.deepEqual(comparabilityReasons(a.manifest,b.manifest,'instrumentation'),[]);
  assert.ok(comparabilityReasons(a.manifest,b.manifest,'token-count').includes('Unmatched effective configuration'));
  assert.ok(comparabilityReasons(a.manifest,b.manifest,'implementation').includes('Unmatched effective configuration'));
  for(const [key,value] of Object.entries({visual_token_num:128,implementation:'other',max_new_tokens:128,important_ratio:.7})){
    const changed=structuredClone(b);changed.manifest.server.configuration![key]=value;
    assert.ok(comparabilityReasons(a.manifest,changed.manifest,'instrumentation').includes('Unmatched effective configuration'),key);
  }
  const changed=structuredClone(b);changed.manifest.server.source!.gitCommit='different-revision';
  assert.ok(comparabilityReasons(a.manifest,changed.manifest,'instrumentation').includes('Unmatched server implementation commit'));
});

test('baseline repeatability describes trial spread without claiming a detection threshold', () => {
  const repeatability = baselineRepeatability([16, 18, 20, 22, 24].map((latency, index) => fixture(`base-${index}`, latency)), 'p50');
  assert.equal(repeatability.median, 20); assert.equal(repeatability.relativeRangePercent, 40);
  assert.equal(repeatability.medianAbsoluteDeviationPercent, 10); assert.deepEqual(repeatability.reasons, []);
  assert.equal(baselineRepeatability([fixture('one')], 'p50').reasons.length, 1);
});

test('capacity screens delivery errors and growing backlog without declaring sustainable capacity', () => {
  const stable = classifyCapacity(fixture('stable'));
  assert.equal(stable.classification, 'finite-window-pass'); assert.equal(stable.sustainableCapacityEstablished, false);
  assert.equal(stable.queueObserved, false);
  const overloaded = classifyCapacity(fixture('overloaded', 700, true));
  assert.equal(overloaded.classification, 'overloaded'); assert.ok(overloaded.backlogGrowth > 0);
  const undelivered = fixture('undelivered'); undelivered.manifest.config.requestsPerSecond = 4;
  assert.equal(classifyCapacity(undelivered).classification, 'delivery-invalid');
  const isolated = fixture('isolated'); isolated.manifest.runKind = 'isolated';
  assert.equal(classifyCapacity(isolated).classification, 'inconclusive');
});

test('optional resource artifacts are verified by exact hash, count, format and safe filename', async (t) => {
  const {readVerifiedResources}=await import('./analyze.js');
  const run=fixture('resources'),directory=await save(t,run);
  const sample={recordedAtUtc:'2026-10-01T12:00:00Z',kind:'sampled-device',fields:['uuid','memory_mib'],values:['GPU-fixture','15000']};
  const bytes=JSON.stringify(sample)+'\n';
  const descriptor={file:'resources.jsonl',count:1,sha256:`sha256:${createHash('sha256').update(bytes).digest('hex')}`,scope:'sampled whole-device memory, not exact KV cache'};
  await writeFile(join(directory,'resources.jsonl'),bytes);
  Object.assign(run.manifest,{resourceSamples:descriptor});
  await writeFile(join(directory,'run.json'),JSON.stringify(run.manifest));
  const verified=await readVerifiedRun(directory);
  assert.equal(verified.resources!.count,1);assert.deepEqual(verified.resources!.samples[0],sample);
  await assert.rejects(readVerifiedResources(directory,{...descriptor,file:'../resources.jsonl'}),/Unsupported resource file/);
  await assert.rejects(readVerifiedResources(directory,{...descriptor,count:2}),/count mismatch/);
  await writeFile(join(directory,'resources.jsonl'),bytes.replace('15000','1'));
  await assert.rejects(readVerifiedRun(directory),/Resource sample SHA-256 mismatch/);
  await rm(join(directory,'resources.jsonl'));
  await assert.rejects(readVerifiedRun(directory),/ENOENT/);
});

function rateFixture(id:string,rate:number,overloaded=false):VerifiedRun {
  const run=fixture(id),m=run.manifest,start=m.timing.plannedStartMs;
  m.config.requestsPerSecond=rate;m.config.durationSeconds=m.config.measuredRequests/rate;
  const end=start+m.config.durationSeconds*1000;let previous=start;
  for(const row of run.requests.filter(row=>row.phase==='measurement')){
    row.scheduledAtMs=start+row.sequence*1000/rate;row.sentAtMs=row.scheduledAtMs;
    const service=overloaded?1500:20;
    row.completedAtMs=(overloaded?Math.max(row.sentAtMs,previous):row.sentAtMs)+service;
    previous=row.completedAtMs;row.latencyMs=row.completedAtMs-row.sentAtMs;row.dispatchLatenessMs=0;
    row.plannedToCompleteMs=row.latencyMs;row.serverMetrics!.service_ms=service;row.unattributedClientMs=row.latencyMs-service;
  }
  m.timing.plannedEndMs=end;m.timing.finishedMs=Math.max(end,previous);
  m.timing.plannedEndAtUtc=new Date(end).toISOString();m.timing.finishedAtUtc=new Date(m.timing.finishedMs).toISOString();
  m.summary=summarizeRun({plannedStartMs:start,plannedEndMs:end,finishedMs:m.timing.finishedMs,
    results:run.requests.filter(row=>row.phase==='measurement'),warmupResults:run.requests.filter(row=>row.phase==='warmup'),status:'complete',stopReason:null});
  return run;
}

test('repeated capacity screens bracket only complete monotonic prescribed rates', async () => {
  const {summarizeCapacityBracket}=await import('./capacity.js');
  const low=[rateFixture('low1',1),rateFixture('low2',1)];
  const high=[rateFixture('high1',3,true),rateFixture('high2',3,true)];
  const bracket=summarizeCapacityBracket([...low,...high],{expectedRates:[1,3],expectedTrialsPerRate:2});
  assert.equal(bracket.status,'screen-bracketed');assert.deepEqual(bracket.screenBracketRps,[1,3]);
  assert.equal(bracket.sustainableCapacityEstablished,false);assert.equal(bracket.highestAllPassRate,1);assert.equal(bracket.lowestAllFailRate,3);
  assert.equal(summarizeCapacityBracket(low).status,'unbounded-above');
  assert.equal(summarizeCapacityBracket(high).status,'unbounded-below');
  const missing=summarizeCapacityBracket(low,{expectedRates:[1,3],expectedTrialsPerRate:2});
  assert.equal(missing.status,'inconclusive');assert.equal(missing.points[1]!.trialCount,0);
  const incomplete=summarizeCapacityBracket([...low,...high],{expectedRates:[1,3],expectedTrialsPerRate:3});
  assert.equal(incomplete.status,'inconclusive');assert.equal(incomplete.highestAllPassRate,null);
});

test('mixed, nonmonotonic, duplicate or incomparable capacity trials remain inconclusive', async () => {
  const {summarizeCapacityBracket}=await import('./capacity.js');
  const mixed=summarizeCapacityBracket([rateFixture('mix1',2),rateFixture('mix2',2,true)]);
  assert.equal(mixed.status,'inconclusive');assert.equal(mixed.points[0]!.classification,'mixed');
  const nonmonotonic=summarizeCapacityBracket([rateFixture('fail1',1,true),rateFixture('fail2',1,true),rateFixture('pass1',3),rateFixture('pass2',3)]);
  assert.equal(nonmonotonic.status,'inconclusive');assert.equal(nonmonotonic.nonmonotonic,true);assert.equal(nonmonotonic.screenBracketRps,null);
  const duplicate=rateFixture('same',1);
  assert.equal(summarizeCapacityBracket([duplicate,duplicate]).status,'inconclusive');
  const changed=rateFixture('changed',1);changed.manifest.server.configuration!.visual_token_num=128;
  assert.equal(summarizeCapacityBracket([rateFixture('original',1),changed]).status,'inconclusive');
});

test('workload characterization preserves category/source counts and observed actual tokens only', async () => {
  const {characterizeWorkload}=await import('./report.js');
  const run=fixture('characters');
  const first=run.requests.find(row=>row.phase==='measurement')!;
  first.serverMetrics!.generated_text_tokens=7;
  first.serverMetrics!.prompt_text_tokens=11;
  const characterized=characterizeWorkload(run.requests);
  assert.equal(characterized.measuredRequests,240);
  assert.deepEqual(characterized.categoryCounts,{ocr:120,count:120});
  assert.equal(characterized.sourceDatasetCounts.fixture,240);
  assert.equal(characterized.generatedTextTokens.observedCount,1);assert.equal(characterized.generatedTextTokens.median,7);
  assert.equal(characterized.promptTextTokens.median,11);assert.equal(characterized.visualTokens.observedCount,0);
  assert.equal(characterized.outputCharacters.observedCount,0);assert.match(characterized.outputCharacters.reason!,/do not retain/);
  first.serverMetrics!.output_characters=0;
  const measured=run.requests.filter(row=>row.phase==='measurement');
  measured[1]!.serverMetrics!.output_characters=3;
  measured[2]!.serverMetrics!.output_characters=-2;
  measured[3]!.serverMetrics!.output_characters=1.5;
  measured[4]!.serverMetrics!.output_characters=null;
  const observed=characterizeWorkload(run.requests).outputCharacters;
  assert.equal(observed.observedCount,2);assert.equal(observed.min,0);assert.equal(observed.max,3);
  assert.match(observed.reason!,/only some/);
});

test('reports preserve instrumentation comparison scope and accept matched off/on configurations', async (t) => {
  const {planCampaign}=await import('./campaign.js');
  const {buildReport}=await import('./report.js');
  const campaign=planCampaign({campaignId:'observation-overhead',purpose:'ab',comparisonKind:'instrumentation',
    endpoint:'http://localhost/infer',datasetPath:'fixture.json',rates:[2],runKind:'isolated',repetitions:1,measuredRequests:240,
    warmupRequests:10,seed:7,timeoutMs:120000,startupTimeoutMs:1000,gpuExclusive:true,
    variants:[{id:'off',command:['unused'],expected:{instrumentation:'handler-service-wall-v1'}},
      {id:'on',command:['unused'],expected:{instrumentation:'handler-stage-token-observations-v2'}}]});
  const directory=await mkdtemp(join(tmpdir(),'observation-report-'));t.after(()=>rm(directory,{recursive:true,force:true}));
  for(const trial of campaign.trials){
    const run=fixture(trial.trialId,20);Object.assign(run.manifest.server.configuration!,campaign.spec.variants[trial.variantIndex]!.expected);
    run.manifest.runKind='isolated';run.manifest.config.runKind='isolated';
    run.manifest.timing.plannedEndMs=120520;run.manifest.timing.finishedMs=120520;refreshSummary(run);
    trial.status='complete';trial.runDirectory=await save(t,run);
  }
  const campaignPath=join(directory,'campaign.json');await writeFile(campaignPath,JSON.stringify(campaign));
  const report=await buildReport(campaignPath,join(directory,'report'));
  assert.equal(report.comparisonKind,'instrumentation');
  assert.equal(report.runKind,'isolated');assert.deepEqual(report.capacityBrackets,[]);
  assert.ok(report.runs.every(run=>run.runKind==='isolated'&&run.offeredRps===null));
  assert.equal(report.comparisons[0]!.latency.pairCount,1);
  assert.ok(report.comparisons[0]!.latency.reasons.every(reason=>!reason.includes('Unmatched')));
  assert.ok(report.limitations.some(reason=>reason.includes('does not estimate total instrumentation overhead')));
  const markdown=await readFile(join(directory,'report/report.md'),'utf8');
  assert.match(markdown,/unavailable \(isolated\)/);assert.match(markdown,/p50 latency: inconclusive/);
  assert.doesNotMatch(markdown,/2 RPS/);assert.match(markdown,/capacity not applicable/);
});

test('baseline reports combine both identical legs for latency spread and never narrate A/A noise as an optimization', async (t) => {
  const {planCampaign}=await import('./campaign.js');
  const {buildReport}=await import('./report.js');
  for(const repetitions of [3,5]){
    const campaign=planCampaign({campaignId:`sham-${repetitions}`,purpose:'baseline',endpoint:'http://localhost/infer',datasetPath:'fixture.json',rates:[2],
      repetitions,measuredRequests:240,warmupRequests:10,seed:7,timeoutMs:120000,startupTimeoutMs:1000,gpuExclusive:true,
      variants:[{id:'A',command:['unused'],expected:{model_id:'llava',visual_token_num:576}},{id:'B',command:['unused'],expected:{model_id:'llava',visual_token_num:576}}]});
    const directory=await mkdtemp(join(tmpdir(),'sham-report-'));t.after(()=>rm(directory,{recursive:true,force:true}));
    for(const trial of campaign.trials){
      const run=fixture(`${trial.trialId}-${repetitions}`,trial.variantIndex===0?20:10,false,7+trial.repetition);
      trial.status='complete';trial.runDirectory=await save(t,run);
    }
    const campaignPath=join(directory,'campaign.json');await writeFile(campaignPath,JSON.stringify(campaign));
    const report=await buildReport(campaignPath,join(directory,'report'));
    assert.equal(report.repeatability[0]!.trialCount,repetitions*2);
    assert.equal(report.repeatability[0]!.latencyP50.validTrialCount,repetitions*2);
    assert.equal(report.repeatability[0]!.independentPairedBlocks,repetitions);
    assert.equal(report.repeatability[0]!.latencyP50.relativeRangePercent,100);
    assert.equal(report.comparisons[0]!.latency.effectPercent,50);
    assert.equal(report.comparisons[0]!.latency.conclusion,'inconclusive');
    assert.ok(report.comparisons[0]!.latency.reasons.some(reason=>reason.includes('sham')));
    assert.equal(report.capacityBrackets[0]!.points[0]!.expectedTrialCount,repetitions);
  }
});
