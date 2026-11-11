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
export async function monitorResources(path:string,readGpu:()=>Promise<string>=queryGpu):Promise<()=>Promise<void>>{
 const file=await open(path,'wx');let pending=Promise.resolve();let stopped=false,sampling=false;
 let failure:unknown=null,stopping:Promise<void>|null=null;
 const sample=()=>{
  // Missed polling ticks are skipped: replaying them adds load without recovering past observations.
  if(stopped||sampling)return;
  sampling=true;
  pending=(async()=>{
   const rows=await sampleResources(readGpu,sampleCpuLoad());
   for(const row of rows)await file.write(JSON.stringify(row)+'\n');
  })().catch((error:unknown)=>{failure=error;stopped=true;clearInterval(timer);}).finally(()=>{sampling=false;});
 };
 const timer=setInterval(sample,1000);sample();
 return ()=>{
  stopped=true;clearInterval(timer);
  stopping??=(async()=>{
   try{await pending;if(failure!==null)throw failure;await file.sync();}
   finally{await file.close();}
  })();
  return stopping;
 };
}
