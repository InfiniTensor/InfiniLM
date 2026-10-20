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
 fp32_counter=lib.modern_deep_FP32_calls_r3597;fp32_counter.argtypes=[ctypes.c_int];fp32_counter.restype=ctypes.c_uint64
 def fp32_counts():return[int(fp32_counter(d))for d in range(8)]
 def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
 B=int(os.environ['DEEP_ATTENTION_BATCH_R3474']);assert B==16;mode=int(os.environ['DEEP_TOKEN_MODE']);assert mode in[0,5]
 reference_audit=json.loads((r/'results/deep-attention-r3511-B8-audit.json').read_text());assert reference_audit['completed'];reference=json.loads((r/'results'/f'deep-attention-r3509-{reference_audit["producer"][0]}.json').read_text());reference_cases={c['phase']:c for c in reference['attention_F64_cases']}
 for c in reference_cases.values():c['generated_ids']=[c['generated_ids'][i%8]for i in range(B)]
 report['precision_counter_names']=['synchronized_reductions','attention_promotions','FFN_promotions','indexer_promotions','cumulative_promoted_scratch_bytes','peak_promoted_scratch_bytes'];report['prior3616_failure']='Field5 is peak scratch bytes, not FFN promotion; original false assertion retained. Correct semantic checks without mathematical change.'
 report.update(token_policy={'mode':mode,'reference_producer':reference_audit['producer'][0],'full_generated_token_IDs_required':True,'logits_bitwise_equality_required':False,'F64_allreduce_enabled':False},token_reference_checks=[])
 for c in g['chat_payload']['cases']:assert len(c['input_ids'])*4<=4096
 assert counters()==[[0]*6 for _ in range(8)]and policy(0)==0
 eos={cfg['eos_token_id']};vocab=cfg['vocab_size'];report.update(attention_precision_batch=B,attention_precision_cases=[],attention_precision_forwards=[],attention_precision_raw=[],scope='Latest5f91/2b4/1ca/cd1/51cc full43 Deep native mode0 versus mode5 FP32 attention reduction; native FFN/indexer, no F64 allreduce. Exact full generated IDs to audited F64 reference; raw differences diagnostic only by human instruction. B1 unchanged; B8 QA/natural repeated firstEOS max512 with prefill_micro4; unchanged4096 CAPI limit, each chunk finite-logit/CPUargmax checked; raw SHA retained for integrity. Native+harness notHTTP. Full gate and independent audit required before adoption.')
 report['source_sha256']={n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()for n in['modern_deep_attention_precision_r3798.py','deep_attention_precision_serving_r3798.py','deep_attention_precision_r3798_inside.sh']}
 g['run_chat']('before');assert counters()==[[0]*6 for _ in range(8)]and policy(mode)==0
 refs={}
 def generate(phase,prompts,limit,qa=None,layout=None):
  check(lib.modern_native_engine_reset(ctx));ids=[[]for _ in prompts];times=[[]for _ in prompts];hashes={};active=list(range(B));start=time.perf_counter();records=[]
  for step in range(limit):
   before=counters();before_fp32=fp32_counts();chunks=[active[i:i+4]for i in range(0,len(active),4)]if step==0 else[active];logits_parts=[];chosen_parts=[];dispatches=[]
   for chunk in chunks:
    inputs=[prompts[i]if step==0 else[ids[i][-1]]for i in chunk];past=[0 if step==0 else len(prompts[i])+len(ids[i])-1 for i in chunk];widths=list(map(len,inputs));flat=sum(inputs,[]);pos=sum([list(range(p,p+w))for p,w in zip(past,widths)],[]);assert 0<len(flat)<=4096
    code,part,part_ids=forward(flat,pos,chunk,past,[p+w for p,w in zip(past,widths)],[0]+np.cumsum(widths).tolist());check(code);pf=(part.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(pf).all()and np.array_equal(pf.argmax(-1),part_ids);logits_parts.append(part);chosen_parts.append(part_ids);dispatches.append({'rows':chunk.copy(),'input_tokens':len(flat),'all8_rank_logits_exact':None,'raw_sha256':hashlib.sha256(part.tobytes()).hexdigest()})
   x=np.concatenate(logits_parts,axis=0);chosen=np.concatenate(chosen_parts);after=counters();delta=[[b-a for a,b in zip(v,w)]for v,w in zip(before,after)];f=(x.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all()and np.array_equal(f.argmax(-1),chosen)
   expected=43*sum(len(chunk)>1 for chunk in chunks)if mode==5 else 0;after_fp32=fp32_counts();assert[b-a for a,b in zip(before_fp32,after_fp32)]==[expected]*8
   for d in delta:
    if mode==5 and expected:assert d[0]>=2*expected and d[1]==expected and d[2]==d[3]==0 and d[4]>0 and d[5]>=0,('actual_FP32_attention_only',d)
    else:assert d==[0]*6,('native_or_B1_must_not_promote',d)
   for old,new in zip(before,after):assert new[5]>=old[5] and new[5]<=new[4],('peak_scratch_bytes_semantics',old,new)
   raw=x.tobytes();digest=hashlib.sha256(raw).hexdigest();hashes[str(step)]={'rows':active.copy(),'sha256':digest};duplicate=True
   for j,i in enumerate(active):
    for k,ii in enumerate(active[:j]):
     if prompts[i]==prompts[ii]and ids[i]==ids[ii]:duplicate&=np.array_equal(x[j],x[k])
   persist=step<4 or step in[63,127,255,383,511]
   if persist:
    file=f'deep-attention-r3798-{tag}-{phase}-s{step}.bin';p=r/'results'/file;assert not p.exists();p.write_bytes(raw);report['attention_precision_raw'].append({'file':file,'phase':phase,'step':step,'shape':list(x.shape),'rows':active.copy(),'selected_ids':chosen.tolist(),'sha256':digest})
   report['attention_precision_forwards'].append({'phase':phase,'step':step,'rows':active.copy(),'selected_ids':chosen.tolist(),'sha256':digest,'counter_delta':delta,'FP32_counter_delta':[b-a for a,b in zip(before_fp32,after_fp32)],'native_dispatches':dispatches,'duplicate_same_input_history_raw_exact':bool(duplicate)});report['phase']={'phase':phase,'step':step,'active':len(active)}
   # Duplicate raw values are diagnostic; complete generated IDs remain mandatory.
   stamp=time.perf_counter()-start
   for i,tok in zip(active,chosen):ids[i].append(int(tok));times[i].append(stamp)
   active=[i for i in active if ids[i][-1]not in eos]
   if step%64==0:save()
   if not active:break
  value=ids;key=('qa',qa)if qa else('natural',layout);assert value==refs.setdefault(key,value),('Deep token own fullID repeat',phase)
  assert all(not any(t in eos for t in v[:-1])and(v[-1]in eos or len(v)==limit)for v in ids)
  if qa:
   gold=next(v for v in g['baseline']['cases']if v['phase']==qa+'_repeat1')['generated_ids'];assert all(v==gold for v in ids)
  first=max(t[0]for t in times);end=max(t[-1]for t in times);useful=sum(tok not in eos for v in ids for tok in v);decoded=sum(t>first and tok not in eos for v,ts in zip(ids,times)for tok,t in zip(v,ts))
  exact=ids==reference_cases[phase]['generated_ids'];report['token_reference_checks'].append({'phase':phase,'all_generated_token_IDs_exact':exact,'actual':ids if not exact else None,'expected':reference_cases[phase]['generated_ids']if not exact else None});save();assert exact,('Deep full token reference mismatch',phase)
  c={'phase':phase,'batch':B,'qa':qa,'layout':layout,'input_lengths':list(map(len,prompts)),'max_output_tokens':limit,'generated_ids':ids,'raw_hashes_each_step':hashes,'step_times_per_row':times,'elapsed_seconds':end,'common_decode_seconds':end-first,'useful_output_tokens':useful,'common_decode_tokens':decoded,'aggregate_useful_decode_tok_s':decoded/(end-first)if end>first else None,'including_prefill_useful_tok_s':useful/end,'all_service_cards':8,'prefill_micro_batch':4,'own_IDs_repeat_exact':True,'no_forward_after_EOS':True};report['attention_precision_cases'].append(c);save()
 def qa(epoch):
  for c in g['chat_payload']['cases']:
   for repeat in[1,2]:generate(f'qa-{epoch}-{c["label"]}-r{repeat}',[c['input_ids']]*B,256,c['label'])
 qa('before');natural=json.loads((r/'results'/f'deep-serving-cases-r2130-{tag}.json').read_text());assert set(natural['eos_token_ids'])==eos and len(natural['cases'])==2
 report['natural_input_provenance']=natural
 for layout in[0,1]:
  for repeat in[1,2]:generate(f'natural-l{layout}-r{repeat}',[natural['cases'][(i+layout)%2]['input_ids']for i in range(B)],512,layout=layout)
 qa('after');before=counters();g['run_chat']('after');assert counters()==before;report.update(attention_precision_gate_complete=True,attention_precision_final_counters=counters(),actual_FP32_promotions_by_device=fp32_counts());save()
