import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,datetime,hashlib,json,math,pathlib,statistics,subprocess,traceback
import numpy as np
from audit_affinity_masks_r2681 import audit_owner_affinity_r2681
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';policy=int(os.environ['KIMI_REDUCE_POLICY']);assert policy in[0,1];out=r/'results'/f'kimi-token-r3562-P{policy}-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def host(p):return pathlib.Path(p.replace('/refactor',str(r),1))
try:
 launch=next(j for j in load(r/'kimi-token-policy-r3561-launch.json')['submitted_jobs']if j['policy']==policy);run=launch['run'];
 ref_launch=load(r/'kimi-pipeline-r3483-launch.json');ref_audit=load(r/'results/kimi-pipeline-r3521-audit.json');assert ref_audit['completed'];reference=load(r/'results'/f'kimi-pipeline-r3483-{ref_launch["run"]}-node3.json');refcases={c['phase']:c for c in reference['pipeline_cases']};
 rows=[a.split('|')for a in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()];a=[v for v in rows if v[1]==launch['label']];assert len(a)==1 and a[0][2:4]==['COMPLETED','0:0'];d['producer']=a[0]
 build=load(r/'results/kimi-reduce-model-r3262-build.json');base=r/'candidates/kimi-reduce-model-3e0d89fe-r3261';assert build['completed']and sha(base/'libmodern_kimi_reduce_r3261.so')==build['library_sha256']and build['pins']['InfiniLM']=='3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d'
 stage=load(r/'results/kimi-reduce-model-r3261-stage.json');assert stage['ready']
 for p,h in {**stage['unchanged_GPU_objects'],**stage['unchanged_GPU_helpers']}.items():assert sha(base/p)==h
 for p,h in build['mapped_libraries'].items():assert sha(host(p))==h
 common=load(r/'results/kimi-wide-common-r3395-build.json');cast=load(r/'results/cast64-sdk-r3375-Kimi-load-build.json');assert common['completed']and cast['completed']
 payload=load(r/'results'/f'kimi-serving-cases-r1101-{run}.json');chat=load(r/'results/kimi-chat-cases-r20.json');known=load(r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json');eos=set(chat['eos_token_ids']);assert eos==set(payload['eos_token_ids'])and len(payload['cases'])==2
 reports=[];raw_count=0;canonical=None;metrics=[]
 for node in range(4):
  p=r/'results'/f'kimi-pipeline-r3560-{run}-node{node}.json';q=load(p);assert q['completed']and q['pipeline_protocol_complete']and q['qualification_before_destroy_passed']and q['all8_owned_workers_shutdown_completed']and not q.get('error')and not q.get('shutdown_error');audit_owner_affinity_r2681(q,4)
  assert q['node']==node and(q['TP'],q['PP'])==(8,4)and q['loaded_keys']==q['parameter_keys']==q['source_parameter_keys_loaded']and q['LM_source_commit']==build['pins']['InfiniLM']and q['library_sha256']==build['library_sha256'];assert q['dense_prefill_optin']and not q['SDK_tile_optin']and q['SDK_tile_initial_counts']==[0]*8
  assert q['cache_geometry']=={'max_rows':32,'pages':132,'page_size':256,'KDA_pool_rows':33};assert len(q['hardware'])==8 and all(v['small_compute_passed']and v['CU']==80 and v['memory']>=63*1024**3 for v in q['hardware'])
  assert q['libinfiniops.so_sha256']==cast['library_sha256'];assert q['libkimi_grouped_prefill.so_sha256']==load(r/'results/model-latest-r2863-Kimi-build.json')['fresh_helpers']['helpers/libkimi_grouped_prefill.so']
  maps=q['actual_LM_native_libraries'];assert len(maps)==2
  for n,h in maps.items():assert '/candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM/python/'in n and sha(host(n))==h==(common['runtime_sha256']if'libinfinilm_runtime.so'in n else common['native_sha256'])
  assert q['harness_files_sha256']==launch['source_sha256']
  for n,h in q['harness_files_sha256'].items():assert sha(r/'incoming'/n)==h
  assert q['token_acceptance_policy']['candidate_reduce_policy']==policy and q['token_acceptance_policy']['logits_bitwise_equality_required']is False and len(q['token_reference_checks'])==116 and all(v['all_generated_token_ids_exact']for v in q['token_reference_checks']);
  assert len(q['chat_cases'])==110 and len(q['natural_cases'])==6 and len(q['pipeline_cases'])==116
  cases={c['phase']:c for c in q['pipeline_cases']};assert len(cases)==116 and set(cases)=={c['phase']for c in q['chat_cases']+q['natural_cases']};refs={};natrefs={};events={}
  for c in cases.values():
   ref_label=c['phase'] if c['qa_label']is not None or c['batch']==8 else 'g32_B1';assert c['generated_ids']==refcases[ref_label]['generated_ids'],('full_token_reference_mismatch',c['phase']);
   phase=c['phase'];B=c['batch'];M=c['micro'];ids=c['generated_ids'];times=c['step_times_per_row'];hashes=c['raw_hashes_per_row'];assert c['all_serving_cards']==32 and len(ids)==len(times)==len(hashes)==B and 0<M<=8 and c['policy']in[0,1,2];assert c['max_output_tokens']==128
   for v,t in zip(ids,times):assert 0<len(v)<=128 and len(v)==len(t)and t[0]>0 and all(a<b for a,b in zip(t,t[1:]))and all(0<=tok<163840 for tok in v)and not any(tok in eos for tok in v[:-1])and(v[-1]in eos or len(v)==128)
   L=max(map(len,ids));assert c['rounds']==L
   expected=[]
   for st in range(L):
    active=[i for i,v in enumerate(ids)if len(v)>st];groups=[[i for i in active if i//M==group]for group in range((B+M-1)//M)];groups=[z for z in groups if z]
    for packet,z in enumerate(groups):expected.append((st,packet,[i+1 for i in z]))
   actual=[(v['step'],v['packet'],v['rows'])for v in c['events']];assert actual==expected and all(v['native_seconds']>=0 for v in c['events']);assert c['feedback_calls']==(L if c['round_feedback']else len(expected))
   for ev in c['events']:key=(phase,ev['step'],ev['packet']);assert key not in events;events[key]=ev
   for i,h in enumerate(hashes):assert set(h)=={str(st)for st in range(len(ids[i]))if st<4 or st in[63,127]}
   if c['qa_label']is not None:
    label=c['qa_label'];key='sort'if label.startswith('sort_probe_')else label;assert all(v==ids[0]for v in ids)and ids[0][-1]in eos;assert any(v['generated_ids']==[ids[0]]for v in known['cases']if v['task']==key and v['grouped']);assert ids[0]==refs.setdefault((B,key),ids[0])
    assert B in[1,8]and M==B and c['round_feedback'];assert c['policy']==(0 if 'before_OFF'in phase else policy)
   else:
    assert M==8 and B in[8,32]and c['round_feedback'] and c['policy']==(2 if 'F64_A'in phase else policy) and c['input_lengths']==[len(payload['cases'][i%2]['input_ids'])for i in range(B)]
    for i,v in enumerate(ids):assert v==natrefs.setdefault((M,i%M,c['layout']),v)
    first=max(t[0]for t in times);last=max(t[-1]for t in times);useful=sum(tok not in eos for v in ids for tok in v);decode=sum(t>first and tok not in eos for v,ts in zip(ids,times)for tok,t in zip(v,ts));assert c['useful_outputs']==useful and c['common_decode_outputs']==decode and c['elapsed_seconds']==last and c['common_decode_seconds']==last-first
    assert math.isclose(c['aggregate_decode_tok_s'],decode/(last-first),rel_tol=1e-12)and math.isclose(c['aggregate_including_prefill_tok_s'],useful/last,rel_tol=1e-12)
    if node==3:metrics.append({'phase':phase,'G':B,'M':M,'round_feedback':c['round_feedback'],'max_output':128,'decode_tok_s':decode/(last-first),'including_prefill_tok_s':useful/last,'per_all32_cards':decode/(last-first)/32,'TTFT_seconds':[min(t[0]for t in times),first],'output_lengths':list(map(len,ids)),'EOS_terminated_requests':sum(v[-1]in eos for v in ids)})
  assert len(events)==q['output_metrics']['engine_calls']==len(q['precision_dispatch'])==len(q['wide_dispatch']);totals=[[0]*6 for _ in range(8)];calls=[0]*8;bytes_=[0]*8;observed=set()
  for a,b in zip(q['precision_dispatch'],q['wide_dispatch']):
   key=(a['phase'],a['step'],a['packet']);assert key not in observed and key in events;observed.add(key);ev=events[key];case=cases[a['phase']];assert a['rows']==b['rows']==ev['rows']and(a['phase'],a['step'],a['packet'])==(b['phase'],b['step'],b['packet'])and a['policy']==case['policy'];assert len(a['delta'])==len(b['delta'])==8
   for rank,(v,w)in enumerate(zip(a['delta'],b['delta'])):
    assert len(v)==5 and len(w)==6 and v[0]==v[4]==0
    if len(a['rows'])>1 and a['policy']==2:assert v[1]>0 and v[2]>0 and w[0]+w[1]>0 and w[2]>0 and w[4]>0 and w[5]==0
    elif len(a['rows'])>1 and a['policy']==1:assert v[1]>0 and v[2]>0 and all(z==0 for z in w)
    else:assert v[1]==v[2]==0 and all(z==0 for z in w)
    calls[rank]+=v[1];bytes_[rank]+=v[2];totals[rank]=[x+y for x,y in zip(totals[rank],w)]
  assert observed==set(events)and totals==q['wide_final_stats']
  for rank,v in enumerate(q['precision_final_stats']):assert v[0]==0 and v[1]==calls[rank]and v[2]==bytes_[rank]and v[4]>0
  raws={}
  for c in q['raw_checkpoints']:
   assert node==3;key=(c['phase'],c['step'],c['packet']);assert key in events and key not in raws;raws[key]=c;assert c['rows']==events[key]['rows']and c['shape']==[len(c['rows']),163840]and pathlib.Path(c['file']).name==c['file'];buf=(r/'results'/c['file']).read_bytes();u=np.frombuffer(buf,np.uint16).reshape(c['shape']);x=(u.astype(np.uint32)<<16).view(np.float32);assert hashlib.sha256(buf).hexdigest()==c['sha256']and np.isfinite(x).all()and x.argmax(-1).tolist()==c['output_ids'];case=cases[c['phase']];assert c['output_ids']==[case['generated_ids'][i-1][c['step']]for i in c['rows']]
   assert c['row_sha256']==[hashlib.sha256(v.tobytes()).hexdigest()for v in u]==[case['raw_hashes_per_row'][i-1][str(c['step'])]for i in c['rows']]
   # Raw buffer SHA is artifact integrity; cross-row/epoch/precision raw equality is diagnostic only.
   raw_count+=1
  if node==3:assert set(raws)=={k for k in events if k[1]<4 or k[1]in[63,127]}
  else:assert not raws
  shared=[(c['phase'],c['generated_ids'],c['raw_hashes_per_row'],c['step_times_per_row'])for c in cases.values()];assert shared==canonical if canonical is not None else True;canonical=shared;reports.append({'node':node,'sha256':sha(p),'native_forwards':len(events),'normal_closed_owners':8})
 assert len(metrics)==6 and raw_count>0
 serial=[v for v in metrics if v['phase']in['g32_F64_A1','g32_F64_A2']];parallel=[v for v in metrics if v['phase']in['g32_candidate_B1','g32_candidate_B2']];assert len(serial)==len(parallel)==2
 baseline=statistics.median(v['decode_tok_s']for v in serial);candidate=statistics.median(v['decode_tok_s']for v in parallel);incl=statistics.median(v['including_prefill_tok_s']for v in parallel)
 d.update(completed=True,source_pins=build['pins'],source_sha256=launch['source_sha256'],QA_per_node=110,all32_owners_closed=True,CPU_fullvocab_raw_files=raw_count,reports=reports,performance=metrics,ABBA_summary={'G':32,'M':8,'cards':32,'F64_control_decode_median':baseline,'candidate_decode_median':candidate,'candidate_including_prefill_median':incl,'per_all32_cards':candidate/32,'speed_ratio':candidate/baseline},candidate_policy=policy,all_full_token_sequences_exact_to_F64_reference=True,logits_bitwise_equality_required=False,scope='Explicit human token-only numerical acceptance; all G32 phases round-feedback, same-process precision ABBA. Locked3e0 full93 Kimi with Model3261/Common3393/Cast64SDK, nativeBF16 or FP32 candidate and F64 control. Literal known-answer IDs including sort unchanged at testedB1/B8, complete token sequences exact to frozen F64 reference, logits raw differences diagnostic only, 110QA/4nodes, all precision dispatch/CPUraw/32close/terminal checked.512input/max128output frozen two long prompts repeated inG8/G32 M8, round-feedback precision ABBA. Excludes HTTP/dynamicarrival/tokenizer/load/reset. Not B2 repair, not latestc3de model qualification, not broad quality or hardwareceiling.')
except BaseException as e:
 d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_TOKEN_R3562_FULL_AUDIT_COMPLETE',json.dumps(d['ABBA_summary']))
