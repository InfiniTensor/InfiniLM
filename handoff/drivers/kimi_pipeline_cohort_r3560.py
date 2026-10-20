"""Current full model: same M8 arithmetic, serial vs round feedback PP4."""
import ctypes,hashlib,json,time,os
import numpy as np

def run(g):
 lib,ctx,control,r,report,save=[g[k]for k in ['lib','ctx','control','r','report','save']]
 node,tail,emitter,tag,vocab,check=[g[k]for k in ['node','tail','emitter','tag','vocab','check']]
 assert (g['tp_nodes'],g['pp'])==(1,4)
 candidate_policy=int(os.environ['KIMI_REDUCE_POLICY']);assert candidate_policy in[0,1]
 ref_launch=json.loads((r/'kimi-pipeline-r3483-launch.json').read_text())
 reference=json.loads((r/'results'/f'kimi-pipeline-r3483-{ref_launch["run"]}-node3.json').read_text())
 ref_audit=json.loads((r/'results/kimi-pipeline-r3521-audit.json').read_text());assert ref_audit['completed'] and reference['completed']
 reference_cases={c['phase']:c for c in reference['pipeline_cases']}
 report['token_acceptance_policy']={'human_instruction':'Complete generated token IDs must match; logits bitwise equality is diagnostic only.','candidate_reduce_policy':candidate_policy,'reference_run':ref_launch['run'],'reference_audit':'kimi-pipeline-r3521-audit.json','logits_bitwise_equality_required':False}
 report['token_reference_checks']=[]
 report['logit_diagnostics']=[]
 lib.modern_kimi_reduce_policy_r3226.argtypes=[ctypes.c_int]
 lib.modern_kimi_reduce_stats_r3226.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 lib.modern_kimi_reduce_wide_stats_r3393.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def counts(wide=False):
  rows=[];size=6 if wide else 5;f=lib.modern_kimi_reduce_wide_stats_r3393 if wide else lib.modern_kimi_reduce_stats_r3226
  for dev in range(8):
   a=(ctypes.c_uint64*size)();assert f(dev,a,size)==0;rows.append(list(a))
  return rows
 assert counts()==[[0]*5 for _ in range(8)]and counts(True)==[[0]*6 for _ in range(8)]
 assert lib.modern_kimi_reduce_policy_r3226(99)!=0
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());natural=json.loads((r/'results'/f'kimi-serving-cases-r1101-{tag}.json').read_text());eos=set(chat['eos_token_ids']);assert eos==set(natural['eos_token_ids'])
 report.update(scope='Current3e0 Model3261/Common3393/Cast64SDK; native BF16 or FP32 candidate versus F64 control, all G32M8 round-feedback, same-process precision ABBA. Complete token IDs match frozen F64 reference; raw equality diagnostic only by explicit human instruction. Native+harness checkpoint costs included. No HTTP/continuousbatch/peak claim.',pipeline_protocol_complete=False,pipeline_cases=[],raw_checkpoints=[],precision_dispatch=[],wide_dispatch=[],chat_cases=[],natural_cases=[],single_user15_is_adoption_gate=False)
 report['harness_files_sha256']={n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()for n in ['kimi_pipeline_cohort_r3560.py','modern_kimi_pipeline_r3560.py','kimi_pipeline_r3560_inside.sh']}
 def forward(inputs,pasts,rows,phase,step,packet,policy,persist):
  B=len(rows);widths=[len(v)for v in inputs];flat=np.ascontiguousarray(np.concatenate(inputs),dtype=np.int64);off=np.ascontiguousarray([0]+np.cumsum(widths).tolist(),dtype=np.int32);pos=np.ascontiguousarray(np.concatenate([np.arange(p,p+w,dtype=np.int64)for p,w in zip(pasts,widths)]));row=np.ascontiguousarray(rows,dtype=np.int32);past=np.ascontiguousarray(pasts,dtype=np.int32);total=np.ascontiguousarray(past+widths,dtype=np.int32)
  assert len(flat)<=8192 and len(rows)<=8
  logits=np.full((B,vocab),0x7fc0,np.uint16);ids=np.full(B,-999,np.int64);flags=np.array([-1],np.int32);before=counts();bw=counts(True);tick=time.perf_counter()
  check(lib.modern_kimi_engine_forward_r2475(ctx,len(flat),B,3,*[ctypes.c_void_p(v.ctypes.data)for v in [row,past,total,off,flat,pos,logits,ids,flags]]));elapsed=time.perf_counter()-tick
  after=counts();aw=counts(True);delta=[[b-a for a,b in zip(x,y)]for x,y in zip(before,after)];wide=[[b-a for a,b in zip(x,y)]for x,y in zip(bw,aw)]
  assert flags[0]==((1 if tail else 0)+(2 if node==emitter else 0))
  for a,b,d,w in zip(before,after,delta,wide):
   assert a[0]==b[0]==policy and d[4]==0
   if B>1 and policy==2:assert d[1]>0 and d[2]>0 and w[0]+w[1]>0 and w[2]>0 and w[4]>0 and w[5]==0
   elif B>1 and policy==1:assert d[1]>0 and d[2]>0 and all(v==0 for v in w)
   else:assert d[1]==d[2]==0 and all(v==0 for v in w)
  report['precision_dispatch'].append({'phase':phase,'step':step,'packet':packet,'rows':rows,'policy':policy,'delta':delta})
  report['wide_dispatch'].append({'phase':phase,'step':step,'packet':packet,'rows':rows,'delta':wide})
  result={'rows':rows,'ids':None,'row_sha256':None,'duplicate_rows_exact':None,'native_seconds':elapsed}
  if tail:
   x=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(x).all();arg=np.argmax(x,-1).tolist();result.update(ids=ids.tolist(),CPU_argmax=arg,duplicate_rows_exact=all(np.array_equal(logits[0],v)for v in logits))
   assert node==emitter and ids.tolist()==arg
   if persist:
    raw=logits.tobytes();name=f'kimi-pipeline-r3560-{tag}-{phase}-s{step}-p{packet}.bin';p=r/'results'/name;assert not p.exists();p.write_bytes(raw);result['row_sha256']=[hashlib.sha256(v.tobytes()).hexdigest()for v in logits]
    report['raw_checkpoints'].append({'file':name,'phase':phase,'step':step,'packet':packet,'rows':rows,'shape':[B,vocab],'output_ids':arg,'sha256':hashlib.sha256(raw).hexdigest(),'row_sha256':result['row_sha256']})
  return result
 qa_refs={};natural_refs={}
 def generate(label,prompts,micro,round_feedback,policy,limit,qa_label=None,layout=0):
  B=len(prompts);assert 1<=B<=32 and micro<=8;report['phase']=label;save();check(lib.modern_kimi_engine_reset_r2475(ctx));assert lib.modern_kimi_reduce_policy_r3226(policy)==0;control.collect('reset:'+label,{})
  generated=[[]for _ in prompts];times=[[]for _ in prompts];hashes=[{}for _ in prompts];events=[];rounds=0;feedback_calls=0;start=time.perf_counter()
  def publish(local,step,serial_packet=None):
   nonlocal feedback_calls
   packets=control.collect(label+':round'+str(step)+(''if serial_packet is None else ':packet'+str(serial_packet)),{'groups':local});feedback_calls+=1;selected=packets[emitter]['groups'];assert len(selected)==len(local)
   stamp=time.perf_counter()-start
   for a,b in zip(local,selected):
    assert a['rows']==b['rows']and b['ids']==b['CPU_argmax']and all(0<=v<vocab for v in b['ids'])
    if qa_label is not None:assert len(set(b['ids']))==1
    if b['duplicate_rows_exact'] is False:report['logit_diagnostics'].append({'phase':label,'step':step,'raw_duplicate_rows_differ':True})
    for j,(row,tok)in enumerate(zip(b['rows'],b['ids'])):
     i=row-1;assert len(generated[i])==step;generated[i].append(tok);times[i].append(stamp)
     if b['row_sha256']is not None:hashes[i][str(step)]=b['row_sha256'][j]
  for step in range(limit):
   active=[i for i,v in enumerate(generated)if not v or v[-1]not in eos];local=[]
   if not active:break
   # Stable fixed cohorts; only completed rows within a cohort retire.
   groups=[[i for i in active if i//micro==group]for group in range((B+micro-1)//micro)];groups=[v for v in groups if v]
   for packet,indices in enumerate(groups):
    ins=[prompts[i]if step==0 else[generated[i][-1]]for i in indices];pasts=[0 if step==0 else len(prompts[i])+step-1 for i in indices];rows=[i+1 for i in indices];a=forward(ins,pasts,rows,label,step,packet,policy,step<4 or step in[63,127]);local.append(a);events.append({'step':step,'packet':packet,'rows':rows,'native_seconds':a['native_seconds']})
    if not round_feedback:publish([a],step,packet)
   if round_feedback:publish(local,step)
   rounds+=1
   if step%32==0:report['active_generation']={'phase':label,'step':step,'active':len(active),'elapsed_seconds':time.perf_counter()-start};save()
  packets=control.collect('finished:'+label,{'generated':generated,'times':times,'hashes':hashes});assert all(v['generated']==generated and v['hashes']==hashes for v in packets);canonical=packets[emitter];times=canonical['times'];first=max(v[0]for v in times);last=max(v[-1]for v in times);useful=sum(tok not in eos for v in generated for tok in v);decode=sum(t>first and tok not in eos for ids,ts in zip(generated,times)for tok,t in zip(ids,ts))
  assert all(not any(tok in eos for tok in v[:-1])and(v[-1]in eos or len(v)==limit)for v in generated)
  c={'phase':label,'batch':B,'micro':micro,'round_feedback':round_feedback,'policy':policy,'qa_label':qa_label,'layout':layout,'input_lengths':list(map(len,prompts)),'generated_ids':generated,'raw_hashes_per_row':hashes,'step_times_per_row':times,'events':events,'rounds':rounds,'feedback_calls':feedback_calls,'max_output_tokens':limit,'useful_outputs':useful,'common_decode_outputs':decode,'common_decode_seconds':last-first,'elapsed_seconds':last,'aggregate_decode_tok_s':decode/(last-first)if last>first else None,'aggregate_including_prefill_tok_s':useful/last,'all_serving_cards':32}
  if qa_label is not None:
   assert all(v==generated[0]for v in generated)and generated[0][-1]in eos;key='sort'if qa_label.startswith('sort_probe_')else qa_label;assert any(v['generated_ids']==[generated[0]]for v in known['cases']if v['task']==key and v['grouped']);value=generated[0];assert value==qa_refs.setdefault((B,key),value),('QA_token_repeat',label);report['chat_cases'].append(c)
  else:
   for i,v in enumerate(generated):
    key=(micro,i%micro,layout);value=v;assert value==natural_refs.setdefault(key,value),('same_micro_token_IDs',label,i)
   report['natural_cases'].append(c)
  ref_label=label if qa_label is not None or B==8 else 'g32_B1'
  expected=reference_cases[ref_label]['generated_ids']
  exact=generated==expected
  diagnosis={'phase':label,'reference_phase':ref_label,'all_generated_token_ids_exact':exact}
  if not exact:
   diagnosis['differences']=[{'row':i,'actual':a,'expected':b,'first_different_index':next((j for j,(x,y)in enumerate(zip(a,b))if x!=y),min(len(a),len(b)))}for i,(a,b)in enumerate(zip(generated,expected))if a!=b]
  report['token_reference_checks'].append(diagnosis)
  report['pipeline_cases'].append(c);save()
  assert exact,('full_token_reference_mismatch',label,ref_label)
 def qa(B,epoch,policy):
  cases=dict(chat['cases'])
  for i in range(3):cases['sort_probe_'+str(i+1)]=cases['sort']
  for name,c in cases.items():
   for rep in [1,2]:generate(f'QA-b{B}-{epoch}-{name}-r{rep}',[c['input_ids']]*B,B,True,policy,128,name)
 qa(1,'before_OFF',0);qa(1,'before_ON',candidate_policy);qa(8,'before_ON',candidate_policy)
 prompts=[c['input_ids']for c in natural['cases']];assert len(prompts)==2 and all(480<=len(p)<=512 for p in prompts)
 for label,B,policy in [('g8_ref1',8,candidate_policy),('g8_ref2',8,candidate_policy),('g32_F64_A1',32,2),('g32_candidate_B1',32,candidate_policy),('g32_candidate_B2',32,candidate_policy),('g32_F64_A2',32,2)]:generate(label,[prompts[i%2]for i in range(B)],8,True,policy,128)
 qa(8,'after_ON',candidate_policy);qa(1,'after_ON',candidate_policy);assert lib.modern_kimi_reduce_policy_r3226(0)==0;report.update(precision_final_stats=counts(),wide_final_stats=counts(True),pipeline_protocol_complete=True);save()
