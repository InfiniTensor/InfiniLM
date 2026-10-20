import ctypes,hashlib,json,time,os
import numpy as np

def run(g):
 r,tag,lib,ctx,cfg,forward,check,report,path=[g[k]for k in['r','tag','lib','ctx','cfg','forward','check','report','path']]
 policy=lib.modern_deep_precision_policy_r3443;policy.argtypes=[ctypes.c_int];stats=lib.modern_deep_precision_stats_r3443;stats.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def counters():
  v=[]
  for d in range(8):
   a=(ctypes.c_uint64*6)();assert stats(d,a,6)==0;v.append(list(a))
  return v
 def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 B=int(os.environ['DEEP_ATTENTION_BATCH_R3474']);assert B in[2,4];assert counters()==[[0]*6 for _ in range(8)]and policy(0)==0
 eos={cfg['eos_token_id']};vocab=cfg['vocab_size'];report.update(attention_F64_batch=B,attention_F64_cases=[],attention_F64_forwards=[],attention_F64_raw=[],scope='Latest3e0 full43 Deep Mode2 attention-only F64 plus matched synchronization at attention/FFN/indexer; FFN/indexer arithmetic native. B1 unchanged; B2/B4 QA/natural repeated firstEOS max512. Native+harness notHTTP. Full gate and independent audit required before adoption.')
 report['source_sha256']={n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()for n in['modern_deep_attention_f64_r3474.py','deep_attention_f64_serving_r3474.py','deep_attention_f64_r3474_inside.sh']}
 g['run_chat']('before');assert counters()==[[0]*6 for _ in range(8)]and policy(2)==0
 refs={}
 def generate(phase,prompts,limit,qa=None,layout=None):
  check(lib.modern_native_engine_reset(ctx));ids=[[]for _ in prompts];times=[[]for _ in prompts];hashes={};active=list(range(B));start=time.perf_counter();records=[]
  for step in range(limit):
   inputs=[prompts[i]if step==0 else[ids[i][-1]]for i in active];past=[0 if step==0 else len(prompts[i])+len(ids[i])-1 for i in active];widths=list(map(len,inputs));flat=sum(inputs,[]);pos=sum([list(range(p,p+w))for p,w in zip(past,widths)],[]);before=counters();code,x,chosen=forward(flat,pos,active,past,[p+w for p,w in zip(past,widths)],[0]+np.cumsum(widths).tolist());check(code);after=counters();delta=[[b-a for a,b in zip(v,w)]for v,w in zip(before,after)];f=(x.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all()and np.array_equal(f.argmax(-1),chosen)
   for d in delta:
    assert d[1]==(43 if len(active)>1 else 0)and d[2]==d[3]==0
    assert d[0]>=86 if len(active)>1 else d==[0]*6
   raw=x.tobytes();digest=hashlib.sha256(raw).hexdigest();hashes[str(step)]={'rows':active.copy(),'sha256':digest};duplicate=True
   for j,i in enumerate(active):
    for k,ii in enumerate(active[:j]):
     if prompts[i]==prompts[ii]and ids[i]==ids[ii]:duplicate&=np.array_equal(x[j],x[k])
   persist=step<4 or step in[63,127,255,383,511]or not duplicate
   if persist:
    check(lib.modern_native_engine_rank_logits_equal(ctx));file=f'deep-attention-r3474-{tag}-{phase}-s{step}.bin';p=r/'results'/file;assert not p.exists();p.write_bytes(raw);report['attention_F64_raw'].append({'file':file,'phase':phase,'step':step,'shape':list(x.shape),'rows':active.copy(),'selected_ids':chosen.tolist(),'sha256':digest})
   report['attention_F64_forwards'].append({'phase':phase,'step':step,'rows':active.copy(),'selected_ids':chosen.tolist(),'sha256':digest,'counter_delta':delta,'duplicate_same_input_history_raw_exact':bool(duplicate)});report['phase']={'phase':phase,'step':step,'active':len(active)}
   if not duplicate:save();raise AssertionError(('Deep attentionF64 duplicate raw',phase,step))
   stamp=time.perf_counter()-start
   for i,tok in zip(active,chosen):ids[i].append(int(tok));times[i].append(stamp)
   active=[i for i in active if ids[i][-1]not in eos]
   if step%64==0:save()
   if not active:break
  value=(ids,hashes);key=('qa',qa)if qa else('natural',layout);assert value==refs.setdefault(key,value),('Deep attentionF64 own fullID/allstepraw repeat',phase)
  assert all(not any(t in eos for t in v[:-1])and(v[-1]in eos or len(v)==limit)for v in ids)
  if qa:
   gold=next(v for v in g['baseline']['cases']if v['phase']==qa+'_repeat1')['generated_ids'];assert all(v==gold for v in ids)
  first=max(t[0]for t in times);end=max(t[-1]for t in times);useful=sum(tok not in eos for v in ids for tok in v);decoded=sum(t>first and tok not in eos for v,ts in zip(ids,times)for tok,t in zip(v,ts))
  c={'phase':phase,'batch':B,'qa':qa,'layout':layout,'input_lengths':list(map(len,prompts)),'max_output_tokens':limit,'generated_ids':ids,'raw_hashes_each_step':hashes,'step_times_per_row':times,'elapsed_seconds':end,'common_decode_seconds':end-first,'useful_output_tokens':useful,'common_decode_tokens':decoded,'aggregate_useful_decode_tok_s':decoded/(end-first)if end>first else None,'including_prefill_useful_tok_s':useful/end,'all_service_cards':8,'own_IDs_and_allstep_raw_repeat_exact':True,'no_forward_after_EOS':True};report['attention_F64_cases'].append(c);save()
 def qa(epoch):
  for c in g['chat_payload']['cases']:
   for repeat in[1,2]:generate(f'qa-{epoch}-{c["label"]}-r{repeat}',[c['input_ids']]*B,256,c['label'])
 qa('before');natural=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text());assert set(natural['eos_token_ids'])==eos and len(natural['cases'])==2
 report['natural_input_provenance']=natural
 for layout in[0,1]:
  for repeat in[1,2]:generate(f'natural-l{layout}-r{repeat}',[natural['cases'][(i+layout)%2]['input_ids']for i in range(B)],512,layout=layout)
 qa('after');before=counters();g['run_chat']('after');assert counters()==before;report.update(attention_F64_gate_complete=True,attention_F64_final_counters=counters());save()
