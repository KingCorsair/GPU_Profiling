// A deliberately serial synthetic service. Only for CPU integration and calibration.
import {createServer} from 'node:http';
const delayMs=Number(process.env.FAKE_DELAY_MS??20);
const configuration={model_id:'serial-fake',implementation:'serial-fake-v1',delay_ms:delayMs,visual_token_num:Number(process.env.VISUAL_TOKEN_NUM??576),important_ratio:0.5,max_new_tokens:64,batch_size:1,eos_policy:'natural'};
let queue=Promise.resolve();let active=0;
const server=createServer((req,res)=>{
 if(req.url==='/health'){res.setHeader('content-type','application/json');res.end(JSON.stringify({service:'serial-fake',pid:process.pid,mode:'fake',model_loaded:false,configuration,active,source:{gitCommit:'synthetic',gitDirty:false},hardware:{devices:[]},runtime:{node:process.version}}));return;}
 if(req.url!=='/infer'){res.writeHead(404);res.end();return;}
 let body='';req.on('data',chunk=>body+=chunk);req.on('end',()=>{
  const entered=performance.now();active++;
  queue=queue.then(async()=>{const admitted=performance.now();await new Promise(r=>setTimeout(r,delayMs));active--;try{const data=JSON.parse(body);res.setHeader('content-type','application/json');res.end(JSON.stringify({answer:'synthetic',request_id:data.request_id,metrics:{schema_version:1,service_ms:performance.now()-admitted,queue_ms:admitted-entered,generated_text_tokens:1,visual_tokens:configuration.visual_token_num}}));}catch{res.writeHead(400);res.end();}});
 });
});
server.listen(Number(process.env.PORT??8765),'127.0.0.1');
process.on('SIGTERM',()=>{server.close();server.closeAllConnections();});
