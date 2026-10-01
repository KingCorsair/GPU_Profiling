import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,join,resolve,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {readVerifiedRun,comparePairs,baselineRepeatability,type VerifiedRun} from './analyze.js';
import {classifyCapacity} from './capacity.js';
import {verifyCampaign,verifyTrialRun,type Campaign} from './campaign.js';
const format=(n:number|null|undefined,d=2)=>n==null?'unavailable':n.toFixed(d);
const escape=(s:string)=>s.replaceAll('|','\\|').replaceAll('\n',' ');
export async function buildReport(campaignPath:string,output:string){
 const campaign:Campaign=JSON.parse(await readFile(campaignPath,'utf8'));
 verifyCampaign(campaign);
 const runs=[];const verified=new Map<string,VerifiedRun>();const seenRunIds=new Set<string>();
 for(const trial of campaign.trials){
  if(trial.status!=='complete'||!trial.runDirectory)continue;
  const directory=resolve(dirname(campaignPath),trial.runDirectory);
  const run=await readVerifiedRun(directory);verifyTrialRun(campaign,trial,run);
  if(seenRunIds.has(run.manifest.runId))throw Error('Run identity reused across campaign trials');
  seenRunIds.add(run.manifest.runId);verified.set(trial.trialId,run);
  const rows=run.requests.filter(r=>r.phase==='measurement');const start=run.manifest.timing.plannedStartMs;
  const timeline=Array.from({length:41},(_,i)=>{const at=start+(run.manifest.timing.plannedEndMs-start)*i/40;return {scheduledSeconds:(at-start)/1000,latencyMs:null,outstanding:rows.filter(r=>r.sentAtMs!==null&&r.sentAtMs<=at&&r.completedAtMs>at).length};});
  runs.push({trialId:trial.trialId,variant:campaign.spec.variants[trial.variantIndex]!.id,rate:trial.rate,runId:run.manifest.runId,
   directory:relative(output,directory),summary:run.manifest.summary,capacity:classifyCapacity(run),
   latenciesMs:rows.filter(r=>r.outcome==='success').map(r=>r.latencyMs!),timeline,
   quality:run.manifest.quality,workload:run.manifest.workload,source:run.manifest.source,server:run.manifest.server,
   requestsSha256:run.manifest.requests.sha256,attempts:trial.attempts??[]});
 }
 const comparisons=[];const repeatability=[];
 for(const rate of campaign.spec.rates){
  const pairs=[];const baselines=[];
  for(let repetition=0;repetition<campaign.spec.repetitions;repetition++){
   const a=campaign.trials.find(t=>t.rate===rate&&t.repetition===repetition&&t.variantIndex===0);
   const b=campaign.trials.find(t=>t.rate===rate&&t.repetition===repetition&&t.variantIndex===1);
   const baseline=a?verified.get(a.trialId):undefined,candidate=b?verified.get(b.trialId):undefined;
   if(baseline)baselines.push(baseline);
   if(baseline&&candidate)pairs.push({baseline,candidate,blockId:a!.blockId});
  }
  const throughput=comparePairs(pairs,'successfulThroughputWithinWindowRps',campaign.spec.seed,campaign.spec.comparisonKind??'token-count'),latency=comparePairs(pairs,'p50',campaign.spec.seed,campaign.spec.comparisonKind??'token-count');
  const retried=campaign.trials.some(t=>t.rate===rate&&(t.attempts??[]).some(a=>a.status==='failed'||a.status==='interrupted'));
  const incomplete=campaign.trials.some(t=>t.rate===rate&&t.status!=='complete');
  for(const result of [throughput,latency]){
   if(retried){result.conclusion='inconclusive';result.reasons.push('Retried failed/interrupted trials remain in attempt history; this is not an untouched fixed-budget campaign');}
   if(incomplete){result.conclusion='inconclusive';result.reasons.push('Missing prescribed trials; incomplete evidence cannot establish the campaign comparison');}
  }
  comparisons.push({rate,throughput,latency});
  repeatability.push({rate,...baselineRepeatability(baselines)});
 }
 const limitations=[
  ...(campaign.spec.purpose==='integration'?['Synthetic CPU integration results validate the measurement pipeline and say nothing about GPU/model speed.']:[]),
  'Each capacity classification describes a fixed observation window. Sustainable capacity requires repeated longer boundary trials.',
  'Tails are conditional on successful requests. p95 requires 200 successes and p99 requires 1000; sample size does not guarantee precision.',
  'Paired uncertainty resamples independent trial blocks, not individual requests. Fewer than five pairs remains inconclusive.',
  'No validated matching accuracy scores are supplied. Accuracy versus throughput and random-scoring controls remain pending with the accuracy owner.',
  'Model batching, exact KV-cache allocation, prefill/decode GPU events, and actual token accounting depend on the serving owner.',
  'Service wall time is not CUDA event time. Client minus service is an unattributed residual, not measured server queue time.',
  'Longer-output sensitivity and instrumentation overhead require separately declared matched GPU campaigns.',
  ...(campaign.trials.some(t=>t.status!=='complete')?['Campaign has incomplete/failed trials; they are retained below and excluded from paired inference.']:[]),
 ];
 const report={schema:'measurement-report',schemaVersion:1,campaignId:campaign.campaignId,purpose:campaign.spec.purpose,generatedAtUtc:new Date().toISOString(),runs,comparisons,repeatability,limitations,trials:campaign.trials};
 await mkdir(output,{recursive:true});await writeFile(join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
 const lines=[`# Measurement report: ${campaign.campaignId}`,'',`Purpose: **${campaign.spec.purpose}**. ${runs.length}/${campaign.trials.length} trials completed and passed raw-artifact verification.`,
  '',campaign.spec.purpose==='integration'?'These are synthetic CPU service results, not GPU benchmark results.':'Interpret these results only within the measured workload and supplied server configuration.',
  '',`Measured requests per trial: ${campaign.spec.measuredRequests}; warmup: ${campaign.spec.warmupRequests}; seed: ${campaign.spec.seed}.`,'',
  '| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|'];
 for(const r of runs){const s=r.summary;lines.push(`| ${r.trialId} | ${escape(r.variant)} | ${r.rate} | ${s.successfulRequests} / ${s.totalRequests} | ${format(s.successfulThroughputWithinWindowRps)} | ${format(s.successfulThroughputIncludingDrainRps)} | ${format(s.successfulRequestLatencyMs.p50)} | ${format(s.successfulRequestLatencyMs.p95)} | ${format(s.successfulRequestLatencyMs.p99)} | ${format(s.drainMs)} | [raw manifest](${r.directory}/run.json) |`);}
 lines.push('','## Paired comparisons','');
 for(const comparison of comparisons){const c=comparison.throughput;lines.push(`- ${comparison.rate} RPS: ${c.conclusion}; ${c.pairCount} pairs; throughput effect ${format(c.effectPercent)}%; 95% interval ${c.ci95?c.ci95.map(v=>format(v)).join(' to ')+'%':'unavailable'}. ${c.reasons.join('; ')}`);}
 lines.push('','## Run quality and capacity screens','');for(const r of runs)lines.push(`- ${r.trialId}: ${r.capacity.classification}. ${[...r.quality.reasons,...r.capacity.reasons].join('; ')}`);
 const failed=campaign.trials.filter(t=>t.status!=='complete');if(failed.length){lines.push('','## Unfinished trials','');for(const t of failed)lines.push(`- ${t.trialId}: ${t.status}. ${t.error??''}`);}
 lines.push('','## Limits and outstanding owner inputs','',...limitations.map(s=>'- '+s),'','The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.','');
 await writeFile(join(output,'report.md'),lines.join('\n'));
 const header=['trialId','variant','offeredRps','successes','failures','windowThroughputRps','drainThroughputRps','p50Ms','p95Ms','p99Ms','drainMs'];
 const csv=[header.join(','),...runs.map(r=>[r.trialId,r.variant,r.rate,r.summary.successfulRequests,r.summary.failedRequests,r.summary.successfulThroughputWithinWindowRps,r.summary.successfulThroughputIncludingDrainRps,r.summary.successfulRequestLatencyMs.p50,r.summary.successfulRequestLatencyMs.p95,r.summary.successfulRequestLatencyMs.p99,r.summary.drainMs].map(v=>JSON.stringify(v??'')).join(','))];
 await writeFile(join(output,'trials.csv'),csv.join('\n')+'\n');
 await writeFile(join(output,'provenance.json'),JSON.stringify({campaignPath:relative(output,campaignPath),campaignSha256:createHash('sha256').update(await readFile(campaignPath)).digest('hex'),runs:runs.map(r=>({runId:r.runId,directory:r.directory,requestsSha256:r.requestsSha256}))},null,2)+'\n');
 return report;
}
async function main(){const path=process.argv[2];if(!path)throw Error('Usage: report <campaign.json> [output-directory]');const output=resolve(process.argv[3]??join(dirname(path),'report'));await buildReport(resolve(path),output);console.log(JSON.stringify({report:join(output,'report.md')}));}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{console.error(e);process.exitCode=1;});
