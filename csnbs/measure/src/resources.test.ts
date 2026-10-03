import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdtemp,readFile,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import test from 'node:test';
import {sampleCpuLoad,validateOptionalCpuLoad} from './cpu_resources.js';
import {monitorResources,sampleResources} from './resources.js';
import {readVerifiedResources} from './analyze.js';
import {readRunBundle} from './ingest.js';

const source=()=>({platform:'linux',loadavg:()=>[0,2.5,1.25],cpus:()=>[{}, {}, {}, {}],availableParallelism:()=>2});
test('CPU observer retains zero/fractional OS loads and distinct logical/available CPU counts',()=>{
 const observation=sampleCpuLoad(source());
 assert.deepEqual([observation.loadAverage1m,observation.loadAverage5m,observation.loadAverage15m],[0,2.5,1.25]);
 assert.equal(observation.logicalCpuCount,4);assert.equal(observation.availableParallelism,2);
 assert.equal(observation.scope,'os-visible-host-or-container');assert.equal(observation.source,'node:os');
 assert.deepEqual(observation.unavailableReasons,{loadAverage:null,logicalCpuCount:null,availableParallelism:null});
 validateOptionalCpuLoad(observation);
});

test('unsupported OS load averages remain unknown rather than Windows zero-load claims',()=>{
 for(const platform of ['win32','unknown-platform']){
  const observation=sampleCpuLoad({...source(),platform,loadavg:()=>{throw Error('must not be called');}});
  assert.equal(observation.loadAverage1m,null);assert.equal(observation.loadAverage5m,null);assert.equal(observation.loadAverage15m,null);
  assert.ok(observation.unavailableReasons.loadAverage!.includes(`unsupported on ${platform}`));
  assert.equal(observation.logicalCpuCount,4);assert.equal(observation.availableParallelism,2);
  validateOptionalCpuLoad(observation);
  assert.throws(()=>validateOptionalCpuLoad({...sampleCpuLoad(source()),platform}),/Unsupported platform/);
 }
});

test('OS failures, missing CPU info and invalid OS values are independently unavailable',()=>{
 const observation=sampleCpuLoad({...source(),loadavg:()=>{throw Error('no load API');},cpus:()=>[],availableParallelism:()=>{throw Error('no parallelism API');}});
 assert.equal(observation.loadAverage1m,null);assert.equal(observation.logicalCpuCount,null);assert.equal(observation.availableParallelism,null);
 assert.match(observation.unavailableReasons.loadAverage!,/no load API/);
 assert.match(observation.unavailableReasons.logicalCpuCount!,/unavailable/);
 assert.match(observation.unavailableReasons.availableParallelism!,/no parallelism API/);
 validateOptionalCpuLoad(observation);
 for(const averages of [[NaN,0,0],[-1,0,0],[0,Infinity,0],[0,0]]){
  const invalid=sampleCpuLoad({...source(),loadavg:()=>averages,availableParallelism:()=>0});
  assert.equal(invalid.loadAverage1m,null);assert.equal(invalid.availableParallelism,null);assert.equal(invalid.logicalCpuCount,4);
  validateOptionalCpuLoad(invalid);
 }
});

test('optional CPU validator preserves legacy absence and rejects corrupted numeric/availability claims',()=>{
 validateOptionalCpuLoad(undefined);
 const base=sampleCpuLoad(source());
 for(const change of [{loadAverage1m:-1},{loadAverage5m:Infinity},{loadAverage15m:'0'},
  {loadAverage1m:null},{logicalCpuCount:0},{logicalCpuCount:1.5},{availableParallelism:-1},{availableParallelism:'2'},
  {platform:'win32'},{schemaVersion:2},{scope:'process-utilization-percent'},
  {unavailableReasons:{loadAverage:'unknown',logicalCpuCount:null,availableParallelism:null}}]){
  assert.throws(()=>validateOptionalCpuLoad({...base,...change}),/Invalid CPU-load observation/);
 }
 assert.throws(()=>validateOptionalCpuLoad(null),/expected object/);
});

test('GPU samples/error records retain their existing fields plus one CPU observation per tick',async()=>{
 const cpuLoad=sampleCpuLoad(source());
 const lines='GPU-a, A40, 10, 1000, 40, 75, 1000\nGPU-b, A40, 20, 2000, 45, 80, 1100\n';
 const rows=await sampleResources(async()=>lines,cpuLoad);
 assert.equal(rows.length,2);assert.equal(rows[0]!.kind,'sampled-device');
 assert.deepEqual(rows[0]!.fields,['uuid','name','utilization_percent','memory_mib','temperature_c','power_w','sm_mhz']);
 assert.deepEqual(rows[0]!.values,['GPU-a','A40','10','1000','40','75','1000']);
 assert.equal(rows[0]!.cpuLoad,rows[1]!.cpuLoad);assert.equal(rows[0]!.recordedAtUtc,rows[1]!.recordedAtUtc);
 const errors=await sampleResources(async()=>{throw Error('GPU query failed');},cpuLoad);
 assert.equal(errors.length,1);assert.match(errors[0]!.error as string,/GPU query failed/);assert.equal(errors[0]!.cpuLoad,cpuLoad);
 assert.equal(errors[0]!.kind,undefined);
 const legacy=await sampleResources(async()=>lines);assert.equal(legacy[0]!.cpuLoad,undefined);
 for(const blank of ['', '  \n\t\n']){
  const empty=await sampleResources(async()=>blank,cpuLoad);
  assert.equal(empty.length,1);assert.match(empty[0]!.error as string,/no device rows/);
  assert.equal(empty[0]!.cpuLoad,cpuLoad);assert.equal(empty[0]!.kind,undefined);assert.equal(empty[0]!.values,undefined);
 }
});

test('future monitor captures OS-visible CPU observations by default even without a GPU tool',async(t)=>{
 const root=await mkdtemp(join(tmpdir(),'cpu-monitor-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const path=join(root,'resources.jsonl'),stop=await monitorResources(path);await stop();
 const rows=(await readFile(path,'utf8')).trim().split('\n').map(line=>JSON.parse(line) as Record<string,unknown>);
 assert.ok(rows.length>0);
 for(const row of rows){assert.ok(row.cpuLoad);validateOptionalCpuLoad(row.cpuLoad);}
});

test('verified reader and raw importer preserve CPU fields and reject a corrupt first import',async(t)=>{
 const root=await mkdtemp(join(tmpdir(),'cpu-resources-'));t.after(()=>rm(root,{recursive:true,force:true}));
 const observed=await sampleResources(async()=>'GPU-fixture,A40,0,1000,40,75,1000\n',sampleCpuLoad(source()));
 const legacy={recordedAtUtc:'2026-10-01T00:00:00Z',kind:'sampled-device',fields:['uuid'],values:['GPU-old']};
 const error=(await sampleResources(async()=>{throw Error('GPU unavailable');},sampleCpuLoad({...source(),platform:'win32'})))[0]!;
 const rows=[legacy,...observed,error];
 const requests='';await writeFile(join(root,'requests.jsonl'),requests);
 const persist=async()=>{
  const bytes=rows.map(row=>JSON.stringify(row)).join('\n')+'\n';await writeFile(join(root,'resources.jsonl'),bytes);
  const descriptor={file:'resources.jsonl',count:rows.length,sha256:`sha256:${createHash('sha256').update(bytes).digest('hex')}`};
  const manifest={schema:'loadgen-run',schemaVersion:2,runId:'cpu-fixture',requests:{file:'requests.jsonl',schemaVersion:3,count:0,
   sha256:`sha256:${createHash('sha256').update(requests).digest('hex')}`},resourceSamples:descriptor};
  await writeFile(join(root,'run.json'),JSON.stringify(manifest));return descriptor;
 };
 const descriptor=await persist();
 assert.deepEqual((await readVerifiedResources(root,descriptor)).samples,rows);
 assert.deepEqual((await readRunBundle(root)).resources,rows);
 assert.equal((JSON.parse((await readFile(join(root,'resources.jsonl'),'utf8')).split('\n')[0]!) as Record<string,unknown>).cpuLoad,undefined);
 (observed[0]!.cpuLoad as Record<string,unknown>).loadAverage1m=-1;
 const corrupt=await persist();
 await assert.rejects(readVerifiedResources(root,corrupt),/Invalid CPU-load observation/);
 await assert.rejects(readRunBundle(root),/Invalid CPU-load observation/);
});
