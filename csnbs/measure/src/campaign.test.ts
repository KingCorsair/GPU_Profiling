import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {planCampaign,validateSpec,configurationMatches,type CampaignSpec} from './campaign.js';
const spec:CampaignSpec=JSON.parse(await readFile(new URL('../campaigns/cpu-integration.json',import.meta.url),'utf8'));
test('schedule is deterministic, adjacent, balanced and complete',()=>{const a=planCampaign(spec);const b=planCampaign(spec);assert.deepEqual(a.schedule,b.schedule);assert.equal(a.trials.length,8);for(let i=0;i<a.trials.length;i+=2){const one=a.trials[i]!,two=a.trials[i+1]!;assert.equal(one.blockId,two.blockId);assert.notEqual(one.variantIndex,two.variantIndex);}for(const rate of spec.rates){const blocks=a.trials.filter(t=>t.rate===rate).filter((_,i)=>i%2===0);assert.equal(blocks.filter(t=>t.variantIndex===0).length,1);}});
test('model campaigns require explicit exclusive GPU reservation',()=>assert.throws(()=>validateSpec({...spec,purpose:'ab'}),/exclusive/));
test('refuses invalid budget and duplicate rates',()=>{assert.throws(()=>validateSpec({...spec,measuredRequests:0}));assert.throws(()=>validateSpec({...spec,rates:[1,1]}));});
test('instrumentation comparison is an explicit validated campaign axis',()=>{
  validateSpec({...spec,comparisonKind:'instrumentation'});
  assert.throws(()=>validateSpec({...spec,comparisonKind:'all-config-changes' as NonNullable<CampaignSpec['comparisonKind']>}),/Invalid comparison kind/);
});
test('checks effective settings rather than desired labels',()=>{assert.equal(configurationMatches({visual_token_num:128},{visual_token_num:576}),false);assert.equal(configurationMatches({visual_token_num:576,extra:true},{visual_token_num:576}),true);});

test('Poisson campaigns require valid replay seeds and open-loop delivery',()=>{
  validateSpec({...spec,arrival:'poisson',arrivalSeed:123});
  assert.throws(()=>validateSpec({...spec,arrival:'poisson',runKind:'isolated'}),/open-loop/);
  assert.throws(()=>validateSpec({...spec,arrival:'burst' as NonNullable<CampaignSpec['arrival']>}),/arrival mode/);
  for(const arrivalSeed of [-1,0.5,NaN,0xffffffff])
    assert.throws(()=>validateSpec({...spec,arrival:'poisson',arrivalSeed}),/arrival seed/);
  assert.throws(()=>validateSpec({...spec,arrival:'uniform',arrivalSeed:1}),/arrival seed/);
});

test('Poisson campaigns replay paired arrivals, vary repetitions and verify persisted controls',async(t)=>{
  const {createServer}=await import('node:net');
  const {mkdtemp,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');
  const {join}=await import('node:path');
  const {fileURLToPath}=await import('node:url');
  const {executeCampaign,verifyTrialRun}=await import('./campaign.js');
  const {readVerifiedRun}=await import('./analyze.js');
  const {buildReport}=await import('./report.js');
  const directory=await mkdtemp(join(tmpdir(),'poisson-campaign-'));
  t.after(()=>rm(directory,{recursive:true,force:true}));
  const reservation=createServer();
  await new Promise<void>((resolve,reject)=>{reservation.once('error',reject);reservation.listen(0,'127.0.0.1',resolve);});
  const address=reservation.address();assert.ok(address&&typeof address!=='string');
  const port=address.port;await new Promise<void>(resolve=>reservation.close(()=>resolve()));
  const trialSpec:CampaignSpec={...structuredClone(spec),campaignId:'poisson-integration',
    endpoint:`http://127.0.0.1:${port}/infer`,datasetPath:fileURLToPath(new URL('../test/fixtures/workload.json',import.meta.url)),
    arrival:'poisson',arrivalSeed:123,rates:[100],repetitions:2,measuredRequests:12,warmupRequests:2,
    variants:spec.variants.map(variant=>({...variant,command:[process.execPath,fileURLToPath(new URL('../test/fixtures/serial-server.mjs',import.meta.url))],
      env:{...variant.env,PORT:String(port)}})) as unknown as CampaignSpec['variants']};
  const campaign=await executeCampaign(trialSpec,join(directory,'results'));
  assert.ok(campaign.trials.every(trial=>trial.status==='complete'),JSON.stringify(campaign.trials));
  const runs=await Promise.all(campaign.trials.map(trial=>readVerifiedRun(trial.runDirectory!)));
  const offsets=runs.map(run=>run.requests.filter(row=>row.phase==='measurement').sort((a,b)=>a.sequence-b.sequence)
    .map(row=>row.scheduledAtMs-run.manifest.timing.plannedStartMs));
  for(let i=0;i<runs.length;i++){
    const trial=campaign.trials[i]!,run=runs[i]!;
    assert.equal(run.manifest.config.arrival,'poisson');
    assert.equal(run.manifest.config.arrivalSeed,123+trial.repetition);
    assert.equal(run.manifest.config.seed,trialSpec.seed+trial.repetition);
    verifyTrialRun(campaign,trial,run);
    const altered=structuredClone(run);altered.manifest.config.arrivalSeed!++;
    assert.throws(()=>verifyTrialRun(campaign,trial,altered),/arrivals differ/);
    const uniform=structuredClone(run);uniform.manifest.config.arrival='uniform';delete uniform.manifest.config.arrivalSeed;
    assert.throws(()=>verifyTrialRun(campaign,trial,uniform),/arrivals differ/);
  }
  for(const i of [0,2])assert.ok(offsets[i]!.every((value,j)=>Math.abs(value-offsets[i+1]![j]!)<1e-6));
  assert.ok(offsets[0]!.some((value,j)=>Math.abs(value-offsets[2]![j]!)>1e-6));
  const report=await buildReport(join(directory,'results/campaign.json'),join(directory,'report'));
  assert.equal(report.runs.length,4);
});

test('saved campaigns reject changed specification, removed trials, or a rewritten schedule', async () => {
  const {verifyCampaign}=await import('./campaign.js');
  const valid=planCampaign(spec);verifyCampaign(valid);
  const changed=structuredClone(valid);changed.spec.measuredRequests++;
  assert.throws(()=>verifyCampaign(changed),/hash/);
  const missing=structuredClone(valid);missing.trials.pop();
  assert.throws(()=>verifyCampaign(missing),/missing or extra/);
  const schedule=structuredClone(valid);schedule.schedule.reverse();
  assert.throws(()=>verifyCampaign(schedule),/immutable schedule/);
  const completed=structuredClone(valid);completed.trials[0]!.status='complete';
  assert.throws(()=>verifyCampaign(completed),/exact run identity/);
});

test('failed startup, resumed attempt history, complete reports and missing saved runs stay auditable', async (t) => {
  const {createServer}=await import('node:net');
  const {mkdtemp,writeFile,rm}=await import('node:fs/promises');
  const {tmpdir}=await import('node:os');
  const {join}=await import('node:path');
  const {fileURLToPath,pathToFileURL}=await import('node:url');
  const {executeCampaign,verifyCampaign}=await import('./campaign.js');
  const {buildReport}=await import('./report.js');
  const directory=await mkdtemp(join(tmpdir(),'campaign-recovery-'));
  t.after(()=>rm(directory,{recursive:true,force:true}));
  const reservation=createServer();
  await new Promise<void>((resolve,reject)=>{reservation.once('error',reject);reservation.listen(0,'127.0.0.1',resolve);});
  const address=reservation.address();assert.ok(address&&typeof address!=='string');
  const port=address.port;await new Promise<void>(resolve=>reservation.close(()=>resolve()));
  const launcher=join(directory,'initially-missing-launcher');
  const trialSpec:CampaignSpec={...structuredClone(spec),campaignId:'recovery',endpoint:`http://127.0.0.1:${port}/infer`,
    datasetPath:fileURLToPath(new URL('../test/fixtures/workload.json',import.meta.url)),rates:[100],repetitions:1,measuredRequests:4,warmupRequests:0,startupTimeoutMs:3000,
    variants:spec.variants.map(variant=>({...variant,command:[launcher],env:{...variant.env,PORT:String(port)}})) as unknown as CampaignSpec['variants']};
  const first=await executeCampaign(trialSpec,join(directory,'results'));
  assert.equal(first.trials[0]!.status,'failed');assert.match(first.trials[0]!.error!,/ENOENT/);
  verifyCampaign(first);
  const fake=fileURLToPath(new URL('../test/fixtures/serial-server.mjs',import.meta.url));
  await writeFile(launcher,`#!${process.execPath}\nimport(${JSON.stringify(pathToFileURL(fake).href)});\n`,{mode:0o755});
  const resumed=await executeCampaign(trialSpec,join(directory,'results'));
  assert.ok(resumed.trials.every(trial=>trial.status==='complete'));
  assert.equal(resumed.trials[0]!.attempts!.length,2);
  assert.equal(resumed.trials[0]!.attempts![0]!.status,'failed');
  assert.notEqual(resumed.trials[0]!.attempts![0]!.directory,resumed.trials[0]!.attempts![1]!.directory);
  verifyCampaign(resumed);
  const report=await buildReport(join(directory,'results/campaign.json'),join(directory,'report'));
  assert.equal(report.runs.length,2);
  assert.ok(report.comparisons[0]!.throughput.reasons.some(reason=>reason.includes('Retried')));
  assert.equal(report.comparisons[0]!.throughput.conclusion,'inconclusive');
  // A vanished completed artifact is a verification failure, never a reason to create a new campaign.
  await rm(join(resumed.trials[0]!.runDirectory!,'run.json'));
  await assert.rejects(executeCampaign(trialSpec,join(directory,'results')),/ENOENT/);
});
