import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {planCampaign,validateSpec,configurationMatches,type CampaignSpec} from './campaign.js';
const spec:CampaignSpec=JSON.parse(await readFile(new URL('../campaigns/cpu-integration.json',import.meta.url),'utf8'));
test('schedule is deterministic, adjacent, balanced and complete',()=>{const a=planCampaign(spec);const b=planCampaign(spec);assert.deepEqual(a.schedule,b.schedule);assert.equal(a.trials.length,8);for(let i=0;i<a.trials.length;i+=2){const one=a.trials[i]!,two=a.trials[i+1]!;assert.equal(one.blockId,two.blockId);assert.notEqual(one.variantIndex,two.variantIndex);}for(const rate of spec.rates){const blocks=a.trials.filter(t=>t.rate===rate).filter((_,i)=>i%2===0);assert.equal(blocks.filter(t=>t.variantIndex===0).length,1);}});
test('model campaigns require explicit exclusive GPU reservation',()=>assert.throws(()=>validateSpec({...spec,purpose:'ab'}),/exclusive/));
test('refuses invalid budget and duplicate rates',()=>{assert.throws(()=>validateSpec({...spec,measuredRequests:0}));assert.throws(()=>validateSpec({...spec,rates:[1,1]}));});
test('checks effective settings rather than desired labels',()=>{assert.equal(configurationMatches({visual_token_num:128},{visual_token_num:576}),false);assert.equal(configurationMatches({visual_token_num:576,extra:true},{visual_token_num:576}),true);});
