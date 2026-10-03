import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {open} from 'node:fs/promises';
import {sampleCpuLoad,type CpuLoadObservation} from './cpu_resources.js';
const exec=promisify(execFile);
const queryGpu=async()=> (await exec('nvidia-smi',['--query-gpu=uuid,name,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],{timeout:5000})).stdout;
export async function requireIdleGpu():Promise<void>{
 const {stdout}=await exec('nvidia-smi',['--query-compute-apps=pid,process_name','--format=csv,noheader'],{timeout:5000});
 if(stdout.trim())throw Error('GPU already has compute processes; reserve the GPU before starting the campaign');
 const devices=await exec('nvidia-smi',['--query-gpu=uuid','--format=csv,noheader'],{timeout:5000});
 if(!devices.stdout.trim())throw Error('No GPU identity available');
}
/** CPU metadata belongs to the sampling tick, repeated unchanged on each GPU row. */
export async function sampleResources(readGpu:()=>Promise<string>=queryGpu,cpuLoad?:CpuLoadObservation):Promise<Record<string,unknown>[]>{
 const recordedAtUtc=new Date().toISOString(),cpu=cpuLoad===undefined?{}:{cpuLoad};
 try{
  const stdout=await readGpu();
  if(!stdout.trim())throw Error('GPU query returned no device rows');
  return stdout.trim().split('\n').map(line=>({recordedAtUtc,kind:'sampled-device',
   fields:['uuid','name','utilization_percent','memory_mib','temperature_c','power_w','sm_mhz'],values:line.split(',').map(s=>s.trim()),...cpu}));
 }catch(error){return [{recordedAtUtc,error:String(error),...cpu}];}
}
export async function monitorResources(path:string):Promise<()=>Promise<void>>{
 const file=await open(path,'wx');let pending=Promise.resolve();let stopped=false;
 const sample=()=>{pending=pending.then(async()=>{
  if(stopped)return;
  const rows=await sampleResources(queryGpu,sampleCpuLoad());
  for(const row of rows)await file.write(JSON.stringify(row)+'\n');
 });};
 sample();const timer=setInterval(sample,1000);
 return async()=>{clearInterval(timer);await pending;stopped=true;await file.sync();await file.close();};
}
