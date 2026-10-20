import pathlib,json,datetime,subprocess,hashlib,math,collections,traceback
import numpy as np
from audit_affinity_masks_r2681 import audit_owner_affinity_r2681
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/deep-attention-r3494-B4-audit.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'scope':'Independent full43 Deep B4 Mode2 attentionF64 acceptance; frozenB1 before/after, B4 twelve QA and four long512/max512 actualEOS phases, allstep samegeometry hashes/duplicate-history gates, CPUcheckpoint argmax/rawSHA, actual all8 dispatch/affinity/normalclose and terminal. B2 failed separately; no B2/B3/dynamicarrival/HTTP or maximum performance claim.'}
def load(p):return json.loads(p.read_text())
def raw(name,shape):
 p=r/'results'/name;assert pathlib.Path(name).name==name;b=p.read_bytes();u=np.frombuffer(b,np.uint16).reshape(shape);x=(u.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(x).all();return b,u,x
try:
 rows=[a.split('|')for a in subprocess.check_output(['sacct','-j','910384','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed,NodeList%160'],text=True).splitlines()];found=[v for v in rows if v[1]=='deep-attention-r3475-B4'];assert len(found)==1 and found[0][2:4]==['COMPLETED','0:0'];step=found[0][0];d['producer']=found[0];p=r/'results'/f'deep-attention-r3474-{step}.json';q=load(p);assert q['completed']and q['engine_worker_shutdown_completed']and q['attention_F64_gate_complete']and not q.get('error');audit_owner_affinity_r2681(q,4)
 proof=load(r/'results/deep-wide-r3447-build.json');cast=load(r/'results/cast64-sdk-r3375-Deep-load-build.json');assert q['library_sha256']==proof['library_sha256']==sha(r/'candidates/deep-wide-3e0d89fe-r3443/libmodern_deep_wide_r3443.so');assert q['sdk_sha256']==cast['library_sha256']==sha(r/'install/hygon-deep-cast64-r3315/lib/libinfiniops.so');assert q['LM_source_commit']=='3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d'
 for n,h in q['source_sha256'].items():assert sha(r/'incoming'/n)==h
 B=q['attention_F64_batch'];assert B==4;V=129280;eos=set(q['natural_input_provenance']['eos_token_ids']);gold={c['phase']:c for c in load(r/'results/modern-deepseek-chat-routed-r20-910373.162-rank0.json')['cases']};raw_count=0;scalar_refs={}
 for epoch in['before','after']:
  assert len(q['chat_cases_by_epoch'][epoch])==6
  for c in q['chat_cases_by_epoch'][epoch]:
   assert c['generated_ids']==gold[c['phase']]['generated_ids']and c['stopped_at_eos']and c['generated_ids'][-1]in eos and c['all8_worker_full_logits_exact_first4'];key=c['phase'].rsplit('_repeat',1)[0];v=(c['generated_ids'],c['first4_logits_sha256']);assert v==scalar_refs.setdefault(key,v)
   for st,h in enumerate(c['first4_logits_sha256']):
    b,u,x=raw(f'deep-finite-r578-{step}-{epoch}-{c["phase"]}-step{st}-B1-bf16.bin',(1,V));assert hashlib.sha256(b).hexdigest()==h and int(x.argmax())==c['generated_ids'][st];raw_count+=1
 assert len(q['invalid_checks'])==12 and all(v['rejected_before_output_and_cache_mutation']for v in q['invalid_checks'])
 cases=q['attention_F64_cases'];assert len(cases)==16 and sum(c['qa']is not None for c in cases)==12;byphase={c['phase']:c for c in cases};assert len(byphase)==16;refs={};metrics=[]
 for c in cases:
  ids=c['generated_ids'];times=c['step_times_per_row'];assert c['batch']==B and len(ids)==len(times)==B and c['all_service_cards']==8 and c['own_IDs_and_allstep_raw_repeat_exact']and c['no_forward_after_EOS'];assert c['max_output_tokens']==(256 if c['qa']else 512)
  for v,ts in zip(ids,times):assert 1<=len(v)<=c['max_output_tokens']and len(v)==len(ts)and all(0<=tok<V for tok in v)and all(a<b for a,b in zip(ts,ts[1:]))and not any(tok in eos for tok in v[:-1])and(v[-1]in eos or len(v)==c['max_output_tokens'])
  hs=c['raw_hashes_each_step'];assert set(hs)=={str(st)for st in range(max(map(len,ids)))}
  for st in range(max(map(len,ids))):assert hs[str(st)]['rows']==[i for i,v in enumerate(ids)if len(v)>st]
  value=(ids,hs);key=('qa',c['qa'])if c['qa']else('natural',c['layout']);assert value==refs.setdefault(key,value)
  if c['qa']:assert all(v==gold[c['qa']+'_repeat1']['generated_ids']for v in ids)
  else:
   assert c['input_lengths']==[len(q['natural_input_provenance']['cases'][(i+c['layout'])%2]['input_ids'])for i in range(B)];first=max(ts[0]for ts in times);end=max(ts[-1]for ts in times);useful=sum(tok not in eos for v in ids for tok in v);decoded=sum(t>first and tok not in eos for v,ts in zip(ids,times)for tok,t in zip(v,ts));assert useful==c['useful_output_tokens']and decoded==c['common_decode_tokens']and end==c['elapsed_seconds']and end-first==c['common_decode_seconds'];assert math.isclose(c['aggregate_useful_decode_tok_s'],decoded/(end-first),rel_tol=1e-12)and math.isclose(c['including_prefill_useful_tok_s'],useful/end,rel_tol=1e-12)
   metrics.append({'phase':c['phase'],'batch':B,'input_lengths':c['input_lengths'],'output_lengths':list(map(len,ids)),'EOS_terminated_requests':sum(v[-1]in eos for v in ids),'decode_tok_s':decoded/(end-first),'including_prefill_tok_s':useful/end,'decode_tok_s_per_all8cards':decoded/(end-first)/8,'TTFT_seconds':[ts[0]for ts in times]})
 dispatch=q['attention_F64_forwards'];assert len(dispatch)==sum(max(map(len,c['generated_ids']))for c in cases);perphase=collections.defaultdict(list);totals=[[0]*6 for _ in range(8)]
 for c in dispatch:
  phase=byphase[c['phase']];st=c['step'];rows=c['rows'];assert rows==phase['raw_hashes_each_step'][str(st)]['rows']and c['sha256']==phase['raw_hashes_each_step'][str(st)]['sha256']and c['selected_ids']==[phase['generated_ids'][i][st]for i in rows]and c['duplicate_same_input_history_raw_exact'];perphase[c['phase']].append(st);assert len(c['counter_delta'])==8
  for rank,v in enumerate(c['counter_delta']):
   assert len(v)==6 and v[1]==(43 if len(rows)>1 else 0)and v[2]==v[3]==0
   if len(rows)>1:assert v[0]>=86 and v[4]>0
   else:assert v==[0]*6
   totals[rank]=[a+b for a,b in zip(totals[rank],v)]
 for phase,st in perphase.items():assert st==list(range(max(map(len,byphase[phase]['generated_ids']))))
 assert q['attention_F64_final_counters']==totals;actual_raw={}
 for c in q['attention_F64_raw']:
  phase=byphase[c['phase']];st=c['step'];rows=c['rows'];assert c['shape']==[len(rows),V]and rows==phase['raw_hashes_each_step'][str(st)]['rows'];b,u,x=raw(c['file'],c['shape']);assert hashlib.sha256(b).hexdigest()==c['sha256']==phase['raw_hashes_each_step'][str(st)]['sha256']and x.argmax(-1).tolist()==c['selected_ids']==[phase['generated_ids'][i][st]for i in rows];assert(phase['phase'],st)not in actual_raw;actual_raw[phase['phase'],st]=True;raw_count+=1
  if phase['qa']:assert all(np.array_equal(u[0],v)for v in u)
 for phase,c in byphase.items():
  for st in range(max(map(len,c['generated_ids']))):assert((phase,st)in actual_raw)==(st<4 or st in[63,127,255,383,511])
 owners=q['blas_policy_before_destroy'];assert owners['failures']==0 and all(v['created']==v['verified']for v in owners['owners']);assert len({v['tid']for v in owners['owners']if'libinfiniops.so'in v['caller']})==8
 d.update(completed=True,batch=4,service_cards=8,full_B1_QA=12,full_B4_QA=12,natural_phases=4,full_native_forwards=len(dispatch),CPU_fullvocab_raw_files=raw_count,all8_normal_workers_closed=True,affinity_verified=True,no_B1_F64_promotion=True,actual_attention_F64_with_native_FFN_indexer_verified=True,report_sha256=sha(p),source_sha256=q['source_sha256'],performance=metrics,limits='B4 only; separate B2 failure retained. Two frozen long prompts and fixed ceiling512, no mixed-length dynamicarrival/HTTP qualification; no hardwaremaximum claim.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('DEEP_ATTENTION_R3494_B4_FULL_AUDIT_COMPLETE')
