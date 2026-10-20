import pathlib,json,subprocess,datetime,hashlib,collections,math,traceback
import numpy as np
from audit_affinity_masks_r2681 import audit_owner_affinity_r2681
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/deep-FP32-token-r3729-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False};load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-j','910407','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()];rows=[a for a in rows if a[1]=='deep-token-r3699-mode5'];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'];step=rows[0][0];path=r/'results'/f'deep-attention-r3699-{step}.json';q=load(path);assert q['completed']and q['engine_worker_shutdown_completed']and q['attention_precision_gate_complete']and not q.get('error');audit_owner_affinity_r2681(q,4)
 proof=load(r/'results/deep-attention-FP32-r3688-build.json');sdk=load(r/'results/latest-sdk-r3672-Deep-build.json');assert q['library_sha256']==proof['library_sha256']==sha(r/'candidates/deep-attention-FP32-5f91ad7-r3688/libmodern_deep_attention_FP32_r3688.so');assert q['sdk_sha256']==sdk['library_sha256']==sha(r/'install/hygon-deep-5f91ad7-r3672/lib/libinfiniops.so');assert q['LM_source_commit']=='5f91ad738583db8f5617aeae53b76523f85d1ec6'
 for n,h in q['source_sha256'].items():assert sha(r/'incoming'/n)==h
 refaudit=load(r/'results/deep-attention-r3511-B8-audit.json');assert refaudit['completed'];ref=load(r/'results'/f'deep-attention-r3509-{refaudit["producer"][0]}.json');refs={c['phase']:c for c in ref['attention_F64_cases']};assert[c['input_ids']for c in q['natural_input_provenance']['cases']]==[c['input_ids']for c in ref['natural_input_provenance']['cases']];eos=set(q['natural_input_provenance']['eos_token_ids']);V=129280;raw_count=0;old=load(r/'results/modern-deepseek-chat-routed-r20-910373.162-rank0.json');gold={c['phase']:c for c in old['cases']}
 def raw(name,shape,h,ids):
  assert pathlib.Path(name).name==name;b=(r/'results'/name).read_bytes();u=np.frombuffer(b,np.uint16).reshape(shape);x=(u.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(x).all()and hashlib.sha256(b).hexdigest()==h and x.argmax(-1).tolist()==ids
 for epoch in['before','after']:
  chats=q['chat_cases_by_epoch'][epoch];assert len(chats)==6
  for c in chats:
   assert c['generated_ids']==gold[c['phase']]['generated_ids']and c['stopped_at_eos']and c['generated_ids'][-1]in eos
   for st,h in enumerate(c['first4_logits_sha256']):raw(f'deep-finite-r578-{step}-{epoch}-{c["phase"]}-step{st}-B1-bf16.bin',(1,V),h,[c['generated_ids'][st]]);raw_count+=1
 assert len(q['invalid_checks'])==12 and all(v['rejected_before_output_and_cache_mutation']for v in q['invalid_checks']);cases=q['attention_precision_cases'];assert len(cases)==16 and len(q['token_reference_checks'])==16;byphase={c['phase']:c for c in cases};own={};metrics=[]
 for c in cases:
  ids=c['generated_ids'];ts=c['step_times_per_row'];assert c['batch']==8 and len(ids)==len(ts)==8 and c['all_service_cards']==8 and c['no_forward_after_EOS']and c['own_IDs_repeat_exact']and ids==refs[c['phase']]['generated_ids'];key=('qa',c['qa'])if c['qa']else('natural',c['layout']);assert ids==own.setdefault(key,ids)
  for v,t in zip(ids,ts):assert len(v)==len(t)>0 and all(0<=x<V for x in v)and all(a<b for a,b in zip(t,t[1:]))and not any(x in eos for x in v[:-1])and(v[-1]in eos or len(v)==c['max_output_tokens'])
  if c['qa']:assert all(v==gold[c['qa']+'_repeat1']['generated_ids']for v in ids)
  else:
   first=max(t[0]for t in ts);end=max(t[-1]for t in ts);useful=sum(x not in eos for v in ids for x in v);decoded=sum(t>first and x not in eos for v,tt in zip(ids,ts)for x,t in zip(v,tt));assert decoded==c['common_decode_tokens']and useful==c['useful_output_tokens'];assert math.isclose(c['aggregate_useful_decode_tok_s'],decoded/(end-first),rel_tol=1e-12)and math.isclose(c['including_prefill_useful_tok_s'],useful/end,rel_tol=1e-12);metrics.append({'phase':c['phase'],'batch':8,'cards':8,'decode_tok_s':decoded/(end-first),'including_prefill_tok_s':useful/end,'output_lengths':list(map(len,ids))})
 dispatch=q['attention_precision_forwards'];assert len(dispatch)==sum(max(map(len,c['generated_ids']))for c in cases);totals=[0]*8;wide_totals=[[0]*6 for _ in range(8)]
 for v in dispatch:
  c=byphase[v['phase']];st=v['step'];active=[i for i,seq in enumerate(c['generated_ids'])if len(seq)>st];assert v['rows']==active and v['selected_ids']==[c['generated_ids'][i][st]for i in active];native=v['native_dispatches'];chunks=[active[i:i+4]for i in range(0,len(active),4)]if st==0 else[active];assert[z['rows']for z in native]==chunks and all(0<z['input_tokens']<=4096 for z in native);expected=43*sum(len(z)>1 for z in chunks);assert v['FP32_counter_delta']==[expected]*8;totals=[a+expected for a in totals]
  for rank,z in enumerate(v['counter_delta']):
   if expected:assert z[0]>=2*expected and z[1]==expected and z[2]==z[3]==0 and z[4]>0 and z[5]>=0,('actual_FP32_only_counter',rank,z)
   else:assert z==[0]*6
   wide_totals[rank]=[a+b for a,b in zip(wide_totals[rank],z)];assert wide_totals[rank][5]<=wide_totals[rank][4]
  assert v['sha256']==c['raw_hashes_each_step'][str(st)]['sha256'] and active==c['raw_hashes_each_step'][str(st)]['rows']
 assert q['attention_precision_final_counters']==wide_totals and q['actual_FP32_promotions_by_device']==totals;assert q['token_policy']['mode']==5 and q['token_policy']['F64_allreduce_enabled']is False and all(z['all_generated_token_IDs_exact']for z in q['token_reference_checks']);seen=set()
 for c in q['attention_precision_raw']:
  v=byphase[c['phase']];st=c['step'];assert(c['phase'],st)not in seen;seen.add((c['phase'],st));ids=[v['generated_ids'][i][st]for i in c['rows']];raw(c['file'],c['shape'],c['sha256'],ids);raw_count+=1
 for c in cases:
  for st in range(max(map(len,c['generated_ids']))):assert((c['phase'],st)in seen)==(st<4 or st in[63,127,255,383,511])
 d.update(completed=True,producer=rows[0],source_commit=q['LM_source_commit'],full_B1_QA=12,full_B8_QA=12,natural_phases=4,full_tokens_exact_to_frozen_F64_reference=True,logits_bitwise_equality_required=False,actual_FP32_attention_promotion_calls_per_device=totals,normal_owned_workers_closed=8,CPU_fullvocab_files=raw_count,report_sha256=sha(path),performance=metrics,scope='Latest5f91 Model3688/SDK3672/Common3673: B8 attention-outputBF16->FP32 SUM->BF16 only, nativeFFN/indexer, noF64AllReduce. Latest5f91 requalified under explicitfulltokenpolicy with unchanged mode5 math; field5 peak-scratch semantics independently checked. Two512input frozenprompts/max512, prefillmicro4; allcompleteIDs match F64reference. Static native+harness, notHTTP, broaderquality, broader quality or hardwaremaximum.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
