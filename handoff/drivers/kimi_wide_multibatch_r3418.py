import ctypes,hashlib,json,pathlib,time
import numpy as np

def forward_peer_result_r3412(packets,emitter,vocab):
 assert isinstance(packets,list)and 0<=emitter<len(packets)and all(isinstance(p,dict)for p in packets)
 selected=packets[emitter]['ids'];assert selected and all(0<=v<vocab for v in selected)
 assert packets[emitter]['CPU_argmax']==selected
 return selected,packets[emitter]['raw_sha256'],max(p['native_seconds']for p in packets)
def generation_peer_result_r3412(packets,emitter,generated,hashes):
 assert isinstance(packets,list)and 0<=emitter<len(packets)
 assert all(p['ids']==generated and p['hashes']==hashes for p in packets)
 return packets[emitter]

def run(g):
 lib,ctx,control,r,report,save=[g[k]for k in ['lib','ctx','control','r','report','save']];node,tail,emitter,tag,vocab=[g[k]for k in ['node','tail','emitter','tag','vocab']];check=g['check'];memory=g['memory'];moe=sum(i>0 for i in range(*g['ranges'][g['stage']]));assert g['tp_nodes']==1 and g['pp']==4
 report.update(topk_optimization_enabled=False,optimization_objective='maximize useful aggregate multi-request decode and end-to-end throughput; no single-user15 admission gate',batch_order=[8,4,2],measurement_scope='Full93 latest3e0 nativeengine+StageControlharness; 512input/max128output; distinct fromprevious512output results,realEOS,32targetcards. Existing F32prefill=1 numericalvariant; currentSDKbaseline ownrepeat, noTopkproviderpreload. Common3226 multi-request BF16SUM->FP32 numericalvariant, sync-retainedscratch; publicmulti-request KDA flags3; originalB1owned51 is a separate historical baseline.')
 report['integrated_Common_precision_numerical_variant']=True
 lib.modern_kimi_reduce_wide_stats_r3393.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def wide_stats():
  values=[]
  for d in range(8):
   v=(ctypes.c_uint64*6)();assert lib.modern_kimi_reduce_wide_stats_r3393(d,v,6)==0;values.append(list(v))
  return values
 report['precision_wide_default']=wide_stats();assert report['precision_wide_default']==[[0]*6 for _ in range(8)]
 report['harness_files_sha256']={n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()for n in ['kimi_wide_multibatch_r3418.py','modern_kimi_wide_multibatch_r3418.py','kimi_wide_multibatch_r3418_inside.sh']}
 lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int]
 lib.modern_kimi_reduce_stats_r3226.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 lib.modern_kimi_reduce_error_r3226.restype=ctypes.c_char_p
 def precision_stats():
  values=[]
  for d in range(8):
   v=(ctypes.c_uint64*5)();assert lib.modern_kimi_reduce_stats_r3226(d,v,5)==0;values.append(list(v))
  return values
 assert precision_stats()==[[0]*5 for _ in range(8)]
 assert lib.modern_kimi_reduce_policy_r3226(99)!=0 and precision_stats()==[[0]*5 for _ in range(8)]
 report['precision_defaultOFF_invalid_no_mutation']=True
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());natural=json.loads((r/'results'/f'kimi-serving-cases-r1101-{tag}.json').read_text());eos=set(chat['eos_token_ids']);assert eos==set(natural['eos_token_ids'])and len(natural['cases'])==2
 def forward(inputs,pasts,rows,phase,step,policy,persist):
  B=len(rows);widths=[len(v)for v in inputs];flat=np.ascontiguousarray(np.concatenate(inputs),dtype=np.int64);offsets=np.ascontiguousarray([0]+np.cumsum(widths).tolist(),dtype=np.int32);positions=np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64)for p,w in zip(pasts,widths)]));rows=np.ascontiguousarray(rows,dtype=np.int32);pasts=np.ascontiguousarray(pasts,dtype=np.int32);total=np.ascontiguousarray(pasts+widths,dtype=np.int32);logits=np.full((B,vocab),0x7fc0,np.uint16);ids=np.full(B,-999,np.int64);flags=np.array([-1],np.int32);before_reduce=precision_stats();before_wide=wide_stats();tick=time.perf_counter();check(lib.modern_kimi_engine_forward_r2475(ctx,len(flat),B,3,*[ctypes.c_void_p(a.ctypes.data)for a in[rows,pasts,total,offsets,flat,positions,logits,ids,flags]]));native=time.perf_counter()-tick
  after_reduce=precision_stats();delta=[[v-u for u,v in zip(a,b)]for a,b in zip(before_reduce,after_reduce)]
  assert all(a[0]==b[0]==policy and d[4]==0 for a,b,d in zip(before_reduce,after_reduce,delta))
  assert all((d[1]>0 and d[2]>0)if B>1 and policy else(d[1]==d[2]==0)for d in delta),('actual_precision_dispatch',phase,B,policy,delta)
  after_wide=wide_stats();dw=[[v-u for u,v in zip(a,b)]for a,b in zip(before_wide,after_wide)];assert all((d[0]+d[1]>0 and d[2]>0 and d[4]>0 and d[5]==0)if B>1 and policy==2 else(d[0]==d[1]==d[2]==0)for d in dw),('actual_F64_dispatch_by_observed_dtype',phase,B,dw)
  report.setdefault('wide_dispatch',[]).append({'phase':phase,'step':step,'batch':B,'delta':dw})
  report.setdefault('precision_dispatch',[]).append({'phase':phase,'step':step,'batch':B,'policy':policy,'calls_delta':[d[1]for d in delta],'bytes_delta':[d[2]for d in delta]})
  assert flags[0]==((1 if tail else 0)+(2 if node==emitter else 0))
  digest=argmax=None;record=None
  if tail:
   x=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(x).all();argmax=np.argmax(x,axis=1).tolist();digest=hashlib.sha256(logits.tobytes()).hexdigest()
   if persist:
    check(lib.modern_kimi_engine_rank_logits_r2475(ctx));name=f'kimi-multibatch-r3412-{tag}-node{node}-{phase}-step{step}.bin';p=r/'results'/name;assert not p.exists();p.write_bytes(logits.tobytes());record={'file':name,'sha256':digest,'rows':rows.tolist(),'output_ids':argmax,'shape':[B,vocab]};report.setdefault('raw_checkpoints',[]).append(record)
  packets=control.collect(phase+':'+str(step),{'ids':ids.tolist()if node==emitter else None,'raw_sha256':digest,'CPU_argmax':argmax,'native_seconds':native});selected,digest,max_native=forward_peer_result_r3412(packets,emitter,vocab);assert len(selected)==B
  return selected,digest,logits if tail else None,max_native

 def reset(phase,policy):
  check(lib.modern_kimi_engine_reset_r2475(ctx));assert lib.modern_kimi_reduce_policy_r3226(policy)==0;control.collect('reset:'+phase,{})
 def qa(batch,epoch,policy,refs):
  cases=dict(chat['cases'])
  for i in range(3):cases['sort_probe_'+str(i+1)]=cases['sort']
  for label,c in cases.items():
   for rep in [1,2]:
    phase=f'b{batch}-QA-{epoch}-{label}-r{rep}';report['phase']=phase;save();reset(phase,policy);generated=[];hashes=[];duplicate=True;prompt=c['input_ids']
    for step in range(128):
     selected,digest,raw,_=forward([prompt]*batch if step==0 else[[v]for v in generated[-1]],[0]*batch if step==0 else[len(prompt)+step-1]*batch,list(range(1,batch+1)),phase,step,policy,step<4);generated.append(selected)
     if step<4:hashes.append(digest)
     if tail:duplicate &= all(np.array_equal(raw[0],row)for row in raw[1:])
     packet=control.collect(phase+':quality'+str(step),{'duplicate_rows_exact':duplicate});assert packet[emitter]['duplicate_rows_exact'],('duplicate_rows_fullraw_mismatch',phase,step)
     assert len(set(selected))==1,('duplicate_row_IDS',phase,step)
     if selected[0]in eos:break
    ids=[v[0]for v in generated];key='sort'if label.startswith('sort_probe_')else label;expected=[v for v in known['cases']if v['task']==key and v['grouped']];known_ok=ids[-1]in eos and any(v['generated_ids']==[ids]for v in expected);value=(ids,hashes);assert known_ok and value==refs.setdefault(key,value),('frozenQA_or_repeat',phase)
    report.setdefault('chat_cases',[]).append({'batch':batch,'epoch':epoch,'policy':policy,'label':label,'repeat':rep,'generated_ids_per_row':np.asarray(generated).T.tolist(),'first4_logits_sha256':hashes,'known_answer_ids_exact':known_ok,'same_batch_fullraw_repeat_exact':True,'duplicate_rows_fullraw_exact':True,'stopped_at_EOS':True});save()
 def generation(batch,epoch,policy,layout,refs):
  phase=f'b{batch}-natural-{layout}-{epoch}';report['phase']=phase;save();reset(phase,policy);prompts=[natural['cases'][(i+layout)%2]['input_ids']for i in range(batch)];assert all(480<=len(v)<=512 for v in prompts);rows=list(range(1,batch+1));generated=[[]for i in rows];times=[[]for i in rows];active=rows.copy();hashes={};tick=time.perf_counter();prefill_done=None;native=[]
  for step in range(128):
   ins=[prompts[i-1]if step==0 else[generated[i-1][-1]]for i in active];pasts=[0 if step==0 else len(prompts[i-1])+len(generated[i-1])-1 for i in active];selected,digest,_,seconds=forward(ins,pasts,active,phase,step,policy,step in[0,1,2,3,63,127,255,383,511]);native.append(seconds);elapsed=time.perf_counter()-tick
   if prefill_done is None:prefill_done=elapsed
   if step in[0,1,2,3,63,127,255,383,511]:hashes[str(step)]={'rows':active.copy(),'raw_sha256':digest}
   next_active=[]
   for row,token in zip(active,selected):
    generated[row-1].append(token);times[row-1].append(elapsed)
    if token not in eos:next_active.append(row)
   active=next_active
   if step%64==0:report['active_generation']={'phase':phase,'step':step,'active_requests':len(active),'generated_total':sum(map(len,generated)),'elapsed':elapsed};save();print('KIMI_MULTIBATCH_R2976_STEP',node,phase,step,len(active),flush=True)
   if not active:break
  # Samplingnode wall determines a common measurement endpoint for every rank.
  packets=control.collect('finished:'+phase,{'ids':generated,'times':times,'hashes':hashes,'elapsed':time.perf_counter()-tick});canonical=generation_peer_result_r3412(packets,emitter,generated,hashes);times=canonical['times'];last=max(t[-1]for t in times);first=times[0][0];per=[]
  for i,(ids,ts)in enumerate(zip(generated,times)):
   assert not any(v in eos for v in ids[:-1]);useful=len(ids)-int(ids[-1]in eos);rate=(useful-1)/(ts[-1]-ts[0])if useful>1 else None;per.append({'row':i+1,'label':natural['cases'][(i+layout)%2]['label'],'input_tokens':len(prompts[i]),'generated_ids':ids,'useful_output_tokens':useful,'stopped_at_EOS':ids[-1]in eos,'decode_tok_s':rate,'TTFT_seconds':ts[0],'last_token_seconds':ts[-1]})
  useful=sum(v['useful_output_tokens']for v in per);decode_useful=sum(max(0,v['useful_output_tokens']-1)for v in per);value=(generated,hashes);assert value==refs.setdefault(layout,value),('natural_samebatch_ABBA_fullIDs_raw',phase)
  row={'batch':batch,'epoch':epoch,'policy':policy,'layout':layout,'phase':phase,'requests':per,'raw_checkpoint_hashes':hashes,'step_times_per_row':times,'all_requests_EOS_semantics_preserved':True,'same_batch_fullIDs_raw_repeat_exact':True,'all_serving_cards':32,'aggregate_decode_tok_s':decode_useful/(last-first),'aggregate_including_prefill_tok_s':useful/last,'decode_tok_s_per_card':decode_useful/(last-first)/32,'useful_outputs':useful,'useful_decode_outputs':decode_useful,'elapsed_seconds':last,'decode_elapsed_seconds':last-first,'min_request_decode_tok_s':min(v['decode_tok_s']for v in per if v['decode_tok_s']is not None),'max_native_seconds':max(native)};report.setdefault('natural_cases',[]).append(row);save();print('KIMI_MULTIBATCH_R3266_RESULT',node,phase,row['aggregate_decode_tok_s'],row['decode_tok_s_per_card'],flush=True)
 qa_refs_B1={};qa(1,'frozen_OFF',0,qa_refs_B1);qa(1,'frozen_ON',2,qa_refs_B1);report['B1_OFF_ON_frozen_QA_and_raw_exact']=True;save()
 for batch in report['batch_order']:
  qa_refs={};natural_refs={}
  for epoch,policy in [('candidate_r1',2),('candidate_r2',2)]:
   control.collect(f'policy:b{batch}:{epoch}',{'policy':policy});report.setdefault('policy_epochs',[]).append({'batch':batch,'epoch':epoch,'policy':policy});qa(batch,epoch,policy,qa_refs)
   for layout in [0,1]:generation(batch,epoch,policy,layout,natural_refs)
  report.setdefault('completed_batches',[]).append(batch);report['memory_after_batch']=memory();assert all(free>6*1024**3 for free,total in memory()),('6GiB_workspace_reserve',batch,memory());save()
 qa(1,'after_ON',2,qa_refs_B1);report['B1_after_multibatch_QA_raw_exact']=True;assert lib.modern_kimi_reduce_policy_r3226(0)==0;report['precision_final_stats']=precision_stats();report['multi_batch_protocol_complete']=True;save()
