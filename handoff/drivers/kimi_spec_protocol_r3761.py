"""Experimental greedy prompt-lookup speculation; every emitted token is target verified."""
import ctypes,hashlib,json,time,os,pathlib
import numpy as np

def propose(history,maximum):
 if not maximum:return []
 for n in range(min(4,len(history)-1),0,-1):
  suffix=history[-n:]
  for i in range(len(history)-n-1,-1,-1):
   if history[i:i+n]==suffix:
    result=history[i+n:min(i+n+maximum,len(history))]
    if result:return result
 return []

def run(g):
 lib,ctx,control,r,report,save,check=[g[k]for k in ['lib','ctx','control','r','report','save','check']]
 node,tail,emitter,tag,vocab=[g[k]for k in ['node','tail','emitter','tag','vocab']]
 assert (g['tp_nodes'],g['pp'])==(1,4)
 verify=lib.modern_kimi_engine_spec_r3632;verify.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]+[ctypes.c_void_p]*11
 lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int];assert lib.modern_kimi_reduce_policy_r3226(0)==0
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());natural=json.loads((r/'results'/f'kimi-serving-cases-r1101-{tag}.json').read_text());eos=set(chat['eos_token_ids']);assert eos==set(natural['eos_token_ids'])
 report.update(scope='Latest5f91 PP-overlapped B16/32/64 serialized-versus-roundfeedback AR ABBA and prompt-lookup speculative actual multi-batch prompt-lookup greedy speculation; no trained draft/MTP weights, no oracle drafting. One load, nativeBF16 target, B1/2/4/8/16, K0/1/3/7. Full target CPUargmax/finite checks included in elapsed time. Compare complete IDs incl EOS to same-batch autoregressive paired-state baseline. Experimental results excluded from main throughput goal; no inheritance of old block qualification.',speculative_main_metric=False,spec_cases=[],full_logit_records=[],token_differences=[],QA=[])
 report['harness_files_sha256']={n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()for n in ['kimi_spec_protocol_r3761.py','modern_kimi_spec_r3761.py','kimi_spec_r3761_inside.sh']}
 math_policy=int(os.environ['INFINILM_KIMI_SPEC_MATH_R3676']);assert math_policy in[1,2]
 counter=lib.modern_kimi_spec_math_counts_r3676;counter.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def math_counts():
  values=[]
  for device in range(8):
   a=(ctypes.c_uint64*3)();assert counter(device,a,3)==0;values.append(list(a))
  return values
 start_layer,end_layer=g['ranges'][g['stage']];kda_layers=sum(start_layer<=v-1<end_layer for v in g['text']['linear_attn_config']['kda_layers']);full_layers=end_layer-start_layer-kda_layers
 frozen=json.loads(pathlib.Path(os.environ['KIMI_FROZEN_AR_R3679']).read_text());assert frozen['source_run']and frozen['AR']
 report.update(spec_math_policy=math_policy,local_KDA_layers=kda_layers,local_MLA_layers=full_layers,math_dispatch=[],frozen_AR_reference_sha256=hashlib.sha256(pathlib.Path(os.environ['KIMI_FROZEN_AR_R3679']).read_bytes()).hexdigest(),frozen_AR_complete_ID_checks=[])
 assert math_counts()==[[0]*3 for _ in range(8)]
 pipeline_enabled=False
 callno=0
 def forward(inputs,pasts,rows,reads,writes,label,initial=False):
  nonlocal callno
  B=len(rows);widths=list(map(len,inputs));N=sum(widths);assert 0<B<=8 and N<=8192 and all(w>0 for w in widths)
  row=np.array(rows,np.int32);past=np.array(pasts,np.int32);total=np.array([p+w for p,w in zip(pasts,widths)],np.int32);offset=np.array([0]+np.cumsum(widths).tolist(),np.int32);ids=np.ascontiguousarray(np.concatenate(inputs),dtype=np.int64);pos=np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64)for p,w in zip(pasts,widths)]));read=np.array(reads,np.int32);write=np.array(writes,np.int32);n_out=B if initial else N
  logits=np.full((n_out,vocab),0x7fc0,np.uint16);output=np.full(n_out,-999,np.int64);flags=np.array([-1],np.int32);args=[ctypes.c_void_p(v.ctypes.data)for v in [row,past,total,offset,ids,pos,logits,output,flags]]
  before_math=math_counts();tick=time.perf_counter()
  if initial:check(lib.modern_kimi_engine_forward_r2475(ctx,N,B,3,*args))
  else:check(verify(ctx,N,B,3,ctypes.c_void_p(read.ctypes.data),ctypes.c_void_p(write.ctypes.data),*args))
  elapsed=time.perf_counter()-tick;after_math=math_counts();delta=[[b-a for a,b in zip(x,y)]for x,y in zip(before_math,after_math)];expected=[0,0,0]if initial else[kda_layers,kda_layers*N,2*full_layers if math_policy==2 and max(widths)>1 else 0];assert delta==[expected]*8,('actual_verification_math_dispatch',label,delta,expected);report['math_dispatch'].append({'label':label,'rows':rows,'tokens':N,'initial':initial,'expected_per_device':expected,'delta':delta});assert flags[0]==((1 if tail else 0)+(2 if node==emitter else 0))
  selected=None;digest=None
  if tail:
   f32=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f32).all(),('nonfinite',label);selected=output.tolist();assert np.argmax(f32,-1).tolist()==selected,('CPUargmax',label);assert all(0<=x<vocab for x in selected);digest=hashlib.sha256(logits).hexdigest()
   entry={'sha256':digest,'outputs':selected,'shape':[n_out,vocab],'label':label,'finite':True,'CPUargmax_equals_GPU':True,'raw_persisted':initial or label.endswith(('-verify0','-replay0'))or callno%128==0}
   if entry['raw_persisted']:
    p=r/'results'/f'kimi-spec-r3761-{tag}-call{callno}.bin';assert not p.exists();p.write_bytes(logits.tobytes());entry['file']=p.name
   report['full_logit_records'].append(entry)
  if pipeline_enabled:
   callno+=1
   return (([[x]for x in selected]if initial else [selected[a:b]for a,b in zip(offset[:-1],offset[1:])])if tail else None),elapsed
  packets=control.collect('call'+str(callno),{'selected':selected,'seconds':elapsed,'raw_sha256':digest});selected=packets[emitter]['selected'];assert selected is not None;callno+=1
  result=[[x]for x in selected]if initial else [selected[a:b]for a,b in zip(offset[:-1],offset[1:])]
  return result,max(x['seconds']for x in packets)
 def generate(prompts,B,K,label,limit,teacher=None,pipeline=True):
  nonlocal pipeline_enabled
  pipeline_enabled=pipeline
  report['phase']=label;save();check(lib.modern_kimi_engine_reset_r2475(ctx));control.collect('reset:'+label,{})
  state=[i+1 for i in range(B)];shadow=[i+65 for i in range(B)];past=[0]*B;generated=[[]for _ in range(B)];times=[[]for _ in range(B)];rounds=[];drafted=accepted=verify_inputs=replayed=0;start=time.perf_counter();prefill_outputs=[]
  for a in range(0,B,4):
   ii=list(range(a,min(a+4,B)));out,_=forward([prompts[i]for i in ii],[0]*len(ii),[i+1 for i in ii],[state[i]for i in ii],[shadow[i]for i in ii],label+'-prefill',initial=True);stamp=time.perf_counter()-start
   if pipeline_enabled:
    if tail:prefill_outputs.extend([[i,z]for i,z in zip(ii,out)])
   else:
    for i,z in zip(ii,out):past[i]=len(prompts[i]);generated[i]=z;times[i]=[stamp]
  if pipeline_enabled:
   packets=control.collect('prefill-results:'+label,{'outputs':prefill_outputs});all_prefill=packets[emitter]['outputs'];assert[i for i,z in all_prefill]==list(range(B));stamp=time.perf_counter()-start
   for i,z in all_prefill:past[i]=len(prompts[i]);generated[i]=z;times[i]=[stamp]
  first=max(x[0]for x in times)
  for turn in range(limit):
   active=[i for i in range(B)if len(generated[i])<limit and generated[i][-1]not in eos]
   if not active:break
   proposals={i:propose(prompts[i]+generated[i],min(K,limit-len(generated[i])))for i in active}
   if teacher is not None:
    # Correctness-only oracle probes never become draft throughput results.
    proposals={i:teacher[i][len(generated[i]):min(len(generated[i])+K,len(teacher[i]))]for i in active}
   inputs={i:[generated[i][-1]]+proposals[i]for i in active};all_out={};before=time.perf_counter()
   for a in range(0,len(active),8):
    ii=active[a:a+8];out,_=forward([inputs[i]for i in ii],[past[i]for i in ii],[i+1 for i in ii],[state[i]for i in ii],[shadow[i]for i in ii],label+'-verify'+str(turn))
    if not pipeline_enabled or tail:all_out.update(zip(ii,out))
   if pipeline_enabled:
    packets=control.collect('verify-results:'+label+':'+str(turn),{'outputs':[[i,z]for i,z in all_out.items()]});all_out=dict(packets[emitter]['outputs']);assert sorted(all_out)==active
   retained={};emission={};matches={}
   for i in active:
    D=proposals[i];out=all_out[i];assert len(out)==len(D)+1;m=0
    while m<len(D)and D[m]==out[m]and D[m]not in eos:m+=1
    z=D[:m]+[out[m]];z=z[:limit-len(generated[i])]
    if any(x in eos for x in z):z=z[:next(j+1 for j,x in enumerate(z)if x in eos)]
    retained[i]=m+1;emission[i]=z;matches[i]=m;drafted+=len(D);accepted+=m;verify_inputs+=len(inputs[i])
   replay=[i for i in active if retained[i]<len(inputs[i]) and emission[i][-1]not in eos and len(generated[i])+len(emission[i])<limit]
   replay_mismatches=[];replay_outputs={}
   for a in range(0,len(replay),8):
    ii=replay[a:a+8];out,_=forward([inputs[i][:retained[i]]for i in ii],[past[i]for i in ii],[i+1 for i in ii],[state[i]for i in ii],[shadow[i]for i in ii],label+'-replay'+str(turn))
    if not pipeline_enabled or tail:replay_outputs.update(zip(ii,out))
   if pipeline_enabled and replay:
    packets=control.collect('replay-results:'+label+':'+str(turn),{'outputs':[[i,z]for i,z in replay_outputs.items()]});replay_outputs=dict(packets[emitter]['outputs']);assert sorted(replay_outputs)==replay
   for i,z in replay_outputs.items():
    replayed+=retained[i]
    if z[-1]!=all_out[i][retained[i]-1]:replay_mismatches.append({'row':i,'verified_correction':all_out[i][retained[i]-1],'replayed_correction':z[-1]})
   stamp=time.perf_counter()-start
   for i in active:
    generated[i].extend(emission[i]);times[i].extend([stamp]*len(emission[i]));past[i]+=retained[i];state[i],shadow[i]=shadow[i],state[i]
   rounds.append({'round':turn,'active':len(active),'drafted':sum(len(proposals[i])for i in active),'accepted':sum(matches.values()),'emitted':sum(len(z)for z in emission.values()),'verify_replay_seconds':time.perf_counter()-before,'replayed_rows':len(replay),'replay_token_differences':replay_mismatches})
   if turn%16==0:report['active_generation']={'label':label,'round':turn,'outputs':list(map(len,generated)),'elapsed':stamp};save()
  elapsed=time.perf_counter()-start;last=max(t[-1]for t in times);useful=sum(x not in eos for row in generated for x in row);decode=sum(t>first and x not in eos for row,ts in zip(generated,times)for x,t in zip(row,ts))
  assert all(not any(x in eos for x in v[:-1])and(v[-1]in eos or len(v)==limit)for v in generated)
  q={'label':label,'batch':B,'pipeline_feedback_once_per_round':pipeline,'draft_max_tokens':K,'draft_method':'teacher_forced_correctness_only'if teacher is not None else 'prompt_lookup_ngram'if K else 'autoregressive_width1_paired_state','generated_ids':generated,'timestamps':times,'input_lengths':list(map(len,prompts)),'max_output_tokens':limit,'drafted_tokens':drafted,'accepted_draft_tokens':accepted,'acceptance_fraction':accepted/drafted if drafted else None,'verification_input_tokens':verify_inputs,'replay_input_tokens':replayed,'rounds':rounds,'useful_outputs':useful,'elapsed_seconds':elapsed,'common_decode_seconds':last-first,'decode_outputs':decode,'aggregate_decode_tok_s':decode/(last-first)if last>first else None,'aggregate_including_prefill_tok_s':useful/last,'all_service_cards':32,'experimental_excluded_main_metric':True,'finite_and_full_vocab_CPUargmax_every_forward':True,'validation_raw_D2H_and_barriers_included':True,'per_request_mean_decode_tok_s':[(sum(x not in eos for x in row[1:])/(ts[-1]-ts[0])if ts[-1]>ts[0]else None)for row,ts in zip(generated,times)]}
  q['decode_tok_s_per_all_card']=q['aggregate_decode_tok_s']/32 if q['aggregate_decode_tok_s']is not None else None
  report['spec_cases'].append(q);save();return q
 # Complete known QA before experimentation, without imposing bitwise logit equality.
 for name,c in chat['cases'].items():
  q=generate([c['input_ids']],1,0,'QA-before-'+name,128);key='sort'if name.startswith('sort_probe')else name;expected=[v['generated_ids'][0]for v in known['cases']if v['task']==key and v['grouped']];assert q['generated_ids'][0]in expected,(name,q['generated_ids']);report['QA'].append({'task':name,'complete_frozen_IDs_exact':True});save()
 prompts=[x['input_ids']for x in natural['cases']];assert len(prompts)==2 and all(480<=len(p)<=512 for p in prompts)
 for B in [16,32,64]:
  ps=[prompts[i%2]for i in range(B)];baselines=[]
  for suffix,pipe in[('A1',False),('B1',True),('B2',True),('A2',False)]:
   q=generate(ps,B,0,f'B{B}-pipeline-AR-{suffix}',128,pipeline=pipe);gold=frozen['AR']['16']['generated_ids'];exact=all(v==gold[i%16]for i,v in enumerate(q['generated_ids']));report['frozen_AR_complete_ID_checks'].append({'batch':B,'phase':q['label'],'exact':exact});save();assert exact,('largerB_AR_fulltoken_compatibility',B,suffix);baselines.append(q)
  base=baselines[1]
  for K in[1,3]:
   reference=None
   for repeat_index in[1,2]:
    q=generate(ps,B,K,f'B{B}-pipeline-lookup-K{K}-B{repeat_index}',128,pipeline=True);q['all_generated_IDs_equal_AR']=q['generated_ids']==base['generated_ids'];q['relative_decode_speed_vs_AR']=q['aggregate_decode_tok_s']/base['aggregate_decode_tok_s'];q['same_configuration_repeat_IDs_exact']=reference is None or reference==q['generated_ids'];reference=q['generated_ids'];q['producer_token_checks_passed']=q['all_generated_IDs_equal_AR']and q['same_configuration_repeat_IDs_exact']and not any(z['replay_token_differences']for z in q['rounds']);q['independently_qualified']=False;save()
    if not q['all_generated_IDs_equal_AR']:report['token_differences'].append({'batch':B,'K':K,'repeat':repeat_index,'actual':q['generated_ids'],'AR':base['generated_ids']});save()
  final=generate(ps,B,0,f'B{B}-pipeline-AR-B3',128,pipeline=True);assert final['generated_ids']==base['generated_ids'],('post_spec_AR_repeat',B)
 for name,c in chat['cases'].items():
  q=generate([c['input_ids']],1,0,'QA-after-'+name,128);key='sort'if name.startswith('sort_probe')else name;expected=[v['generated_ids'][0]for v in known['cases']if v['task']==key and v['grouped']];assert q['generated_ids'][0]in expected,(name,q['generated_ids']);report['QA'].append({'task':name,'epoch':'after','complete_frozen_IDs_exact':True});save()
 report['verification_math_final_counts']=math_counts();report['spec_protocol_completed']=True;save()
