import * as os from 'node:os';

export type CpuLoadObservation = {
 schemaVersion:1;source:'node:os';scope:'os-visible-host-or-container';platform:string;
 loadAverage1m:number|null;loadAverage5m:number|null;loadAverage15m:number|null;
 logicalCpuCount:number|null;availableParallelism:number|null;
 unavailableReasons:{loadAverage:string|null;logicalCpuCount:string|null;availableParallelism:string|null};
};
type CpuSource={platform:string;loadavg:()=>number[];cpus:()=>unknown[];availableParallelism:()=>number};
const nonnegative=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
const positiveInteger=(value:unknown):value is number=>nonnegative(value)&&Number.isSafeInteger(value)&&value>0;
const nonempty=(value:unknown):value is string=>typeof value==='string'&&value.trim().length>0;
const supportedLoadPlatforms=new Set(['aix','darwin','freebsd','linux','openbsd','sunos','netbsd','android']);

/** OS-visible system load, never a process/container CPU-utilization percentage. */
export function sampleCpuLoad(source:CpuSource={platform:os.platform(),loadavg:os.loadavg,cpus:os.cpus,availableParallelism:os.availableParallelism}):CpuLoadObservation{
 const observation:CpuLoadObservation={schemaVersion:1,source:'node:os',scope:'os-visible-host-or-container',platform:source.platform,
  loadAverage1m:null,loadAverage5m:null,loadAverage15m:null,logicalCpuCount:null,availableParallelism:null,
  unavailableReasons:{loadAverage:null,logicalCpuCount:null,availableParallelism:null}};
 try{
  if(!supportedLoadPlatforms.has(source.platform))throw Error(`OS load averages are unsupported on ${source.platform}`);
  const averages=source.loadavg();
  if(!Array.isArray(averages)||averages.length!==3||!averages.every(nonnegative))throw Error('OS returned invalid load averages');
  [observation.loadAverage1m,observation.loadAverage5m,observation.loadAverage15m]=averages as [number,number,number];
 }catch(error){observation.unavailableReasons.loadAverage=String(error);}
 try{
  const cpus=source.cpus();
  if(!Array.isArray(cpus)||!positiveInteger(cpus.length))throw Error('Logical CPU information unavailable');
  observation.logicalCpuCount=cpus.length;
 }catch(error){observation.unavailableReasons.logicalCpuCount=String(error);}
 try{
  const available=source.availableParallelism();
  if(!positiveInteger(available))throw Error('OS returned invalid available parallelism');
  observation.availableParallelism=available;
 }catch(error){observation.unavailableReasons.availableParallelism=String(error);}
 return observation;
}

/** Absent legacy observations stay absent; present records must be internally valid. */
export function validateOptionalCpuLoad(value:unknown):void{
 if(value===undefined)return;
 const require=(condition:unknown,message:string):void=>{if(!condition)throw Error(`Invalid CPU-load observation: ${message}`);};
 require(value!==null&&typeof value==='object'&&!Array.isArray(value),'expected object');
 const row=value as Record<string,unknown>;
 require(row.schemaVersion===1&&row.source==='node:os'&&row.scope==='os-visible-host-or-container'&&nonempty(row.platform),'schema/source/scope/platform');
 const reasons=row.unavailableReasons;
 require(reasons!==null&&typeof reasons==='object'&&!Array.isArray(reasons),'missing unavailable reasons');
 const errors=reasons as Record<string,unknown>;
 const averages=[row.loadAverage1m,row.loadAverage5m,row.loadAverage15m];
 require(averages.every(nonnegative)||averages.every(value=>value===null),'load averages must be three nonnegative finite numbers or three nulls');
 require(averages[0]===null?nonempty(errors.loadAverage):errors.loadAverage===null,'load-average availability reason');
 for(const key of ['logicalCpuCount','availableParallelism'] as const){
  require(row[key]===null||positiveInteger(row[key]),`${key} must be a positive integer or null`);
  require(row[key]===null?nonempty(errors[key]):errors[key]===null,`${key} availability reason`);
 }
 if(!supportedLoadPlatforms.has(row.platform as string))require(averages.every(value=>value===null),'Unsupported platform load averages must be unavailable, not zero');
}
