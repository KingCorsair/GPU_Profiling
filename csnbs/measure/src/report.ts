import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,join,resolve,relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {readVerifiedRun,comparePairs,baselineRepeatability,comparabilityReasons,type VerifiedRun} from './analyze.js';
import {classifyCapacity,summarizeCapacityBracket} from './capacity.js';
import {percentile,type RequestResult} from './loadgen.js';
import {verifyCampaign,verifyTrialRun,type Campaign} from './campaign.js';
const format=(n:number|null|undefined,d=2)=>n==null?'unavailable':n.toFixed(d);
const escape=(s:string)=>s.replaceAll('|','\\|').replaceAll('\n',' ');
export function characterizeWorkload(requests: RequestResult[]) {
 const measured=requests.filter(row=>row.phase==='measurement');
 const categoryCounts:Record<string,number>={},sourceDatasetCounts:Record<string,number>={};
 for(const row of measured){categoryCounts[row.category]=(categoryCounts[row.category]??0)+1;const source=row.sourceDataset??'<unavailable>';sourceDatasetCounts[source]=(sourceDatasetCounts[source]??0)+1;}
 const tokenSummary=(key:'generated_text_tokens'|'prompt_text_tokens'|'visual_tokens')=>{
  const values=measured.filter(row=>row.outcome==='success').map(row=>row.serverMetrics?.[key]).filter((value):value is number=>typeof value==='number'&&Number.isFinite(value));
  return {observedCount:values.length,min:values.length?Math.min(...values):null,median:percentile(values,0.5),max:values.length?Math.max(...values):null,
   reason:values.length===0?'Actual token counts were not supplied by the server':values.length<measured.length?'Actual token counts are available for only some measured requests':null};
 };
 return {measuredRequests:measured.length,categoryCounts,sourceDatasetCounts,generatedTextTokens:tokenSummary('generated_text_tokens'),promptTextTokens:tokenSummary('prompt_text_tokens'),visualTokens:tokenSummary('visual_tokens'),
  outputCharacters:{observedCount:0,min:null,median:null,max:null,reason:'Raw request records do not retain answer text or character counts'}};
}
function sameConfiguration(a:unknown,b:unknown):boolean {
 const sorted=(value:unknown):unknown=>Array.isArray(value)?value.map(sorted):value!==null&&typeof value==='object'?Object.fromEntries(Object.entries(value).sort(([left],[right])=>left.localeCompare(right)).map(([key,item])=>[key,sorted(item)])):value;
 return JSON.stringify(sorted(a))===JSON.stringify(sorted(b));
}
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
   requestsSha256:run.manifest.requests.sha256,attempts:trial.attempts??[],characterization:characterizeWorkload(run.requests),resourceSamples:run.resources??null});
 }
 const comparisons=[];const repeatability=[];
 const sham=campaign.spec.purpose==='baseline'||campaign.spec.purpose==='aa';
 for(const rate of campaign.spec.rates){
  const pairs=[];const baselines=[];
  for(let repetition=0;repetition<campaign.spec.repetitions;repetition++){
   const a=campaign.trials.find(t=>t.rate===rate&&t.repetition===repetition&&t.variantIndex===0);
   const b=campaign.trials.find(t=>t.rate===rate&&t.repetition===repetition&&t.variantIndex===1);
   const baseline=a?verified.get(a.trialId):undefined,candidate=b?verified.get(b.trialId):undefined;
   if(baseline)baselines.push(baseline);
   if(sham&&candidate&&baseline&&sameConfiguration(baseline.manifest.server.configuration,candidate.manifest.server.configuration)&&comparabilityReasons(baseline.manifest,candidate.manifest).length===0)baselines.push(candidate);
   if(baseline&&candidate)pairs.push({baseline,candidate,blockId:a!.blockId});
  }
  const throughput=comparePairs(pairs,'successfulThroughputWithinWindowRps',campaign.spec.seed,campaign.spec.comparisonKind??'token-count'),latency=comparePairs(pairs,'p50',campaign.spec.seed,campaign.spec.comparisonKind??'token-count');
  const retried=campaign.trials.some(t=>t.rate===rate&&(t.attempts??[]).some(a=>a.status==='failed'||a.status==='interrupted'));
  const incomplete=campaign.trials.some(t=>t.rate===rate&&t.status!=='complete');
  for(const result of [throughput,latency]){
   if(sham){result.conclusion='inconclusive';result.reasons.push('A/A sham comparison estimates noise; it cannot establish an optimization');}
   if(retried){result.conclusion='inconclusive';result.reasons.push('Retried failed/interrupted trials remain in attempt history; this is not an untouched fixed-budget campaign');}
   if(incomplete){result.conclusion='inconclusive';result.reasons.push('Missing prescribed trials; incomplete evidence cannot establish the campaign comparison');}
  }
  comparisons.push({rate,throughput,latency,shamComparison:sham});
  repeatability.push({rate,...baselineRepeatability(baselines),latencyP50:baselineRepeatability(baselines,'p50'),
   independentPairedBlocks:pairs.length,baselineLegsIncluded:sham?'Both legs when their effective configurations match':'Baseline leg only',runIds:baselines.map(run=>run.manifest.runId)});
 }
 const capacityBrackets=campaign.spec.variants.map((variant,variantIndex)=>{
  const trials=campaign.trials.filter(trial=>trial.variantIndex===variantIndex);
  const variantRuns=trials.map(trial=>verified.get(trial.trialId)).filter((run):run is VerifiedRun=>run!==undefined);
  const bracket=summarizeCapacityBracket(variantRuns,{expectedRates:campaign.spec.rates,expectedTrialsPerRate:campaign.spec.repetitions});
  if(trials.some(trial=>(trial.attempts??[]).some(attempt=>attempt.status==='failed'||attempt.status==='interrupted'))){bracket.status='inconclusive';bracket.screenBracketRps=null;bracket.reasons.push('A trial was retried after failure/interruption; the original fixed-budget sweep is incomplete');}
  return {variant:variant.id,...bracket};
 });
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
 const report={schema:'measurement-report',schemaVersion:1,campaignId:campaign.campaignId,purpose:campaign.spec.purpose,generatedAtUtc:new Date().toISOString(),runs,comparisons,repeatability,capacityBrackets,limitations,trials:campaign.trials};
 await mkdir(output,{recursive:true});await writeFile(join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
 const lines=[`# Measurement report: ${campaign.campaignId}`,'',`Purpose: **${campaign.spec.purpose}**. ${runs.length}/${campaign.trials.length} trials completed and passed raw-artifact verification.`,
  '',campaign.spec.purpose==='integration'?'These are synthetic CPU service results, not GPU benchmark results.':'Interpret these results only within the measured workload and supplied server configuration.',
  '',`Measured requests per trial: ${campaign.spec.measuredRequests}; warmup: ${campaign.spec.warmupRequests}; seed: ${campaign.spec.seed}.`,'',
  '| Trial | Variant | Offered RPS | Successes / offered | Window RPS | RPS including drain | p50 ms | p95 ms | p99 ms | Drain ms | Evidence |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|'];
 for(const r of runs){const s=r.summary;lines.push(`| ${r.trialId} | ${escape(r.variant)} | ${r.rate} | ${s.successfulRequests} / ${s.totalRequests} | ${format(s.successfulThroughputWithinWindowRps)} | ${format(s.successfulThroughputIncludingDrainRps)} | ${format(s.successfulRequestLatencyMs.p50)} | ${format(s.successfulRequestLatencyMs.p95)} | ${format(s.successfulRequestLatencyMs.p99)} | ${format(s.drainMs)} | [raw manifest](${r.directory}/run.json) |`);}
 lines.push('','## Paired comparisons','');
 for(const comparison of comparisons){const c=comparison.throughput;lines.push(`- ${comparison.rate} RPS: ${c.conclusion}; ${c.pairCount} pairs; throughput effect ${format(c.effectPercent)}%; 95% interval ${c.ci95?c.ci95.map(v=>format(v)).join(' to ')+'%':'unavailable'}. ${c.reasons.join('; ')}`);}
 lines.push('','## Baseline repeatability','');
 for(const row of repeatability){const latency=row.latencyP50;lines.push(`- ${row.rate} RPS: ${latency.validTrialCount} baseline trial observations, ${row.independentPairedBlocks} paired blocks. ${row.baselineLegsIncluded}. Trial p50 latency median ${format(latency.median)} ms; relative range ${format(latency.relativeRangePercent)}%; median absolute deviation ${format(latency.medianAbsoluteDeviationPercent)}%. Descriptive pilot spread, not a guaranteed detectable effect. ${latency.reasons.join('; ')}`);}
 lines.push('','## Repeated fixed-window capacity screens','');
 for(const bracket of capacityBrackets){lines.push(`- ${escape(bracket.variant)}: ${bracket.status}; highest all-pass tested rate ${format(bracket.highestAllPassRate)} RPS, lowest all-fail tested rate ${format(bracket.lowestAllFailRate)} RPS. ${bracket.reasons.join('; ')}`);for(const point of bracket.points)lines.push(`  - ${point.offeredRps} RPS: ${point.classification}, ${point.trialCount}/${point.expectedTrialCount??'?'} prescribed trials. ${point.reasons.join('; ')}`);}
 lines.push('','## Run quality and capacity screens','');for(const r of runs)lines.push(`- ${r.trialId}: ${r.capacity.classification}. ${[...r.quality.reasons,...r.capacity.reasons].join('; ')}`);
 const failed=campaign.trials.filter(t=>t.status!=='complete');if(failed.length){lines.push('','## Unfinished trials','');for(const t of failed)lines.push(`- ${t.trialId}: ${t.status}. ${t.error??''}`);}
 lines.push('','## Limits and outstanding owner inputs','',...limitations.map(s=>'- '+s),'','The machine-readable report preserves run IDs, hashes, server identity, workload order and comparison reasons. Historical evidence is unchanged.','');
 await writeFile(join(output,'report.md'),lines.join('\n'));
 const header=['trialId','variant','offeredRps','successes','failures','windowThroughputRps','drainThroughputRps','p50Ms','p95Ms','p99Ms','drainMs'];
 const csv=[header.join(','),...runs.map(r=>[r.trialId,r.variant,r.rate,r.summary.successfulRequests,r.summary.failedRequests,r.summary.successfulThroughputWithinWindowRps,r.summary.successfulThroughputIncludingDrainRps,r.summary.successfulRequestLatencyMs.p50,r.summary.successfulRequestLatencyMs.p95,r.summary.successfulRequestLatencyMs.p99,r.summary.drainMs].map(v=>JSON.stringify(v??'')).join(','))];
 await writeFile(join(output,'trials.csv'),csv.join('\n')+'\n');
 await writeFile(join(output,'provenance.json'),JSON.stringify({campaignPath:relative(output,campaignPath),campaignSha256:createHash('sha256').update(await readFile(campaignPath)).digest('hex'),runs:runs.map(r=>({runId:r.runId,directory:r.directory,requestsSha256:r.requestsSha256,resourceSamples:r.resourceSamples?{file:r.resourceSamples.file,count:r.resourceSamples.count,sha256:r.resourceSamples.sha256}:null}))},null,2)+'\n');
 return report;
}
async function main(){const path=process.argv[2];if(!path)throw Error('Usage: report <campaign.json> [output-directory]');const output=resolve(process.argv[3]??join(dirname(path),'report'));await buildReport(resolve(path),output);console.log(JSON.stringify({report:join(output,'report.md')}));}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{console.error(e);process.exitCode=1;});
