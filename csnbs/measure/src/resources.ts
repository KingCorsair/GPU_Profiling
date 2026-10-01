import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {open} from 'node:fs/promises';
const exec=promisify(execFile);
export async function requireIdleGpu():Promise<void>{
 const {stdout}=await exec('nvidia-smi',['--query-compute-apps=pid,process_name','--format=csv,noheader'],{timeout:5000});
 if(stdout.trim())throw Error('GPU already has compute processes; reserve the GPU before starting the campaign');
 const devices=await exec('nvidia-smi',['--query-gpu=uuid','--format=csv,noheader'],{timeout:5000});
 if(!devices.stdout.trim())throw Error('No GPU identity available');
}
export async function monitorResources(path:string):Promise<()=>Promise<void>>{
 const file=await open(path,'wx');let pending=Promise.resolve();let stopped=false;
 const sample=()=>{pending=pending.then(async()=>{
  if(stopped)return;
  const recordedAtUtc=new Date().toISOString();
  try{
   const {stdout}=await exec('nvidia-smi',['--query-gpu=uuid,name,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],{timeout:5000});
   for(const line of stdout.trim().split('\n')){const fields=line.split(',').map(s=>s.trim());await file.write(JSON.stringify({recordedAtUtc,kind:'sampled-device',fields:['uuid','name','utilization_percent','memory_mib','temperature_c','power_w','sm_mhz'],values:fields})+'\n');}
  }catch(error){await file.write(JSON.stringify({recordedAtUtc,error:String(error)})+'\n');}
 });};
 sample();const timer=setInterval(sample,1000);
 return async()=>{clearInterval(timer);await pending;stopped=true;await file.sync();await file.close();};
}
