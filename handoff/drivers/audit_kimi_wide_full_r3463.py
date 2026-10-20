import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
from audit_affinity_masks_r2681 import audit_owner_affinity_r2681
import collections,datetime,hashlib,json,math,pathlib,subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-wide-r3463-audit.json';assert not out.exists();launch=json.loads((r/'kimi-wide-full-r3419-launch.json').read_text());row={'run':launch['run']};producer=[]
for branch in launch['branches']:
 aa=[a.split('|') for a in subprocess.check_output(['sacct','-j',branch['parent'],'-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True).splitlines() if a.split('|')[1]==branch['label']];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'];producer.append(aa[0])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();base=r/'candidates/kimi-reduce-model-3e0d89fe-r3261';build=json.loads((r/'results/kimi-reduce-model-r3262-build.json').read_text());assert build['completed'];assert sha(base/'libmodern_kimi_reduce_r3261.so')==build['library_sha256'];assert build['pins']['InfiniLM']=='3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d'
stage=json.loads((r/'results/kimi-reduce-model-r3261-stage.json').read_text());assert stage['ready']
for p,h in {**stage['unchanged_GPU_objects'],**stage['unchanged_GPU_helpers']}.items():assert sha(base/p)==h,p
for p,h in build['mapped_libraries'].items():assert sha(pathlib.Path(p.replace('/refactor',str(r),1)))==h,p
payload=json.loads((r/'results'/f'kimi-serving-cases-r1101-{row["run"]}.json').read_text());chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());known=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text());eos=set(payload['eos_token_ids']);assert eos==set(chat['eos_token_ids']);phase_epochs=['candidate_r1','candidate_r2'];records=[];reports=[];rawfiles=0;canonical=None
for node in range(4):
 p=r/'results'/f'kimi-wide-multibatch-r3418-{row["run"]}-node{node}.json';q=json.loads(p.read_text());assert q['completed']and q['qualification_before_destroy_passed']and q['all8_owned_workers_shutdown_completed']and q['multi_batch_protocol_complete']and q['completed_batches']==[8,4,2];assert not q.get('error')and not q.get('shutdown_error');audit_owner_affinity_r2681(q,4);assert q['loaded_keys']==q['parameter_keys']and q['TP']==8 and q['PP']==4 and q['node']==node;assert q['library_sha256']==build['library_sha256']and q['topk_optimization_enabled']is False;assert q['cache_geometry']=={'max_rows':8,'pages':36,'page_size':256,'KDA_pool_rows':9};assert q['dense_prefill_optin']is True
 for n,h in q['harness_files_sha256'].items():assert sha(r/'incoming'/n)==h,n
 assert len(q['chat_cases'])==2*(len(chat['cases'])+3)*9 and len(q['natural_cases'])==12 and q['policy_epochs']==[{'batch':b,'epoch':e,'policy':2*int(e.startswith('candidate'))}for b in[8,4,2]for e in phase_epochs];assert all(v['small_compute_passed']and v['CU']==80 for v in q['hardware']);moe=sum(i>0 for i in range(*q['PP_ranges'][node]));phases={};refs={};natrefs={}
 for c in q['chat_cases']:
  B=c['batch'];policy=c['policy'];assert policy==2*int(c['epoch']!='frozen_OFF') and c['known_answer_ids_exact']and c['same_batch_fullraw_repeat_exact']and c['duplicate_rows_fullraw_exact']and c['stopped_at_EOS'];ids=c['generated_ids_per_row'];assert len(ids)==B and all(v==ids[0]for v in ids);assert ids[0][-1]in eos and not any(v in eos for v in ids[0][:-1]);key='sort'if c['label'].startswith('sort_probe_')else c['label'];assert any(v['generated_ids']==[ids[0]]for v in known['cases']if v['task']==key and v['grouped']);value=(ids[0],c['first4_logits_sha256']);assert value==refs.setdefault((B,key),value);prompt=(chat['cases']['sort']if key=='sort'else chat['cases'][key])['input_ids'];phase=f'b{B}-QA-{c["epoch"]}-{c["label"]}-r{c["repeat"]}';length=len(ids[0]);cached=policy*(length-1+int(len(prompt)*B<=512));single=policy*(length-1)*int(B==1);phases[phase]={'policy':policy,'forwards':length,'sum_counters_per_owner':[moe*length,moe*cached,moe*(length-cached),moe*single]}
 for c in q['natural_cases']:
  B=c['batch'];policy=c['policy'];assert policy==2*int(c['epoch']!='frozen_OFF') and c['same_batch_fullIDs_raw_repeat_exact']and c['all_requests_EOS_semantics_preserved']and c['all_serving_cards']==32;assert len(c['requests'])==len(c['step_times_per_row'])==B;generated=[];useful=decoded=0
  for i,(req,ts)in enumerate(zip(c['requests'],c['step_times_per_row'])):
   ids=req['generated_ids'];assert 1<len(ids)<=128 and len(ids)==len(ts)and ts[0]>0 and all(a<b for a,b in zip(ts,ts[1:]));assert not any(v in eos for v in ids[:-1]);n=len(ids)-int(ids[-1]in eos);assert n==req['useful_output_tokens']and req['stopped_at_EOS']==(ids[-1]in eos);assert math.isclose((n-1)/(ts[-1]-ts[0]),req['decode_tok_s'],rel_tol=1e-12);assert req['input_tokens']==len(payload['cases'][(i+c['layout'])%2]['input_ids'])and req['label']==payload['cases'][(i+c['layout'])%2]['label'];useful+=n;decoded+=n-1;generated.append(ids)
  value=(generated,c['raw_checkpoint_hashes']);assert value==natrefs.setdefault((B,c['layout']),value);last=max(t[-1]for t in c['step_times_per_row']);first=max(t[0]for t in c['step_times_per_row']);assert all(t[0]==first for t in c['step_times_per_row']);assert c['useful_outputs']==useful and c['useful_decode_outputs']==decoded and c['elapsed_seconds']==last and c['decode_elapsed_seconds']==last-first;assert math.isclose(c['aggregate_decode_tok_s'],decoded/(last-first),rel_tol=1e-12)and math.isclose(c['aggregate_including_prefill_tok_s'],useful/last,rel_tol=1e-12)and math.isclose(c['decode_tok_s_per_card'],decoded/(last-first)/32,rel_tol=1e-12)
  length=max(map(len,generated));active=[sum(len(ids)>t for ids in generated)for t in range(length)];input_tokens=sum(v['input_tokens']for v in c['requests']);cached=policy*(length-1+int(input_tokens<=512));single=policy*sum(n==1 for n in active[1:]);phases[c['phase']]={'policy':policy,'forwards':length,'sum_counters_per_owner':[moe*length,moe*cached,moe*(length-cached),moe*single]}
  if node==3:records.append({k:c[k]for k in ['batch','epoch','policy','layout','aggregate_decode_tok_s','aggregate_including_prefill_tok_s','decode_tok_s_per_card','min_request_decode_tok_s','elapsed_seconds','useful_outputs']})
 assert q['precision_defaultOFF_invalid_no_mutation']and q['B1_OFF_ON_frozen_QA_and_raw_exact']and q['B1_after_multibatch_QA_raw_exact']
 assert q['output_metrics']['engine_calls']==sum(v['forwards']for v in phases.values())==len(q['precision_dispatch'])
 common=json.loads((r/'results/kimi-wide-common-r3395-build.json').read_text());cast=json.loads((r/'results/cast64-sdk-r3375-Kimi-load-build.json').read_text());assert common['completed']and cast['completed'];assert q['libinfiniops.so_sha256']==cast['library_sha256'];assert q['libkimi_grouped_prefill.so_sha256']==json.loads((r/'results/model-latest-r2863-Kimi-build.json').read_text())['fresh_helpers']['helpers/libkimi_grouped_prefill.so']
 maps=q['actual_LM_native_libraries'];assert len(maps)==2
 for name,h in maps.items():
  assert '/candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM/python/'in name and sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h
  assert h==(common['runtime_sha256']if'libinfinilm_runtime.so'in name else common['native_sha256'])
 wide=q['wide_dispatch'];assert len(wide)==len(q['precision_dispatch']);wide_byphase=collections.defaultdict(list)
 for a,b in zip(wide,q['precision_dispatch']):
  assert (a['phase'],a['step'],a['batch'])==(b['phase'],b['step'],b['batch']);wide_byphase[a['phase']].append(a)
  assert len(a['delta'])==8
  for v in a['delta']:
   assert len(v)==6
   if b['batch']>1 and b['policy']==2:assert v[0]+v[1]>0 and v[2]>0 and v[4]>0 and v[5]==0
   else:assert v[0]==v[1]==v[2]==0
 assert set(wide_byphase)==set(phases)and q['precision_wide_default']==[[0]*6 for _ in range(8)]

 by_phase=collections.defaultdict(list);totals=[0]*8
 for a in q['precision_dispatch']:
  by_phase[a['phase']].append(a)
  assert a['policy']==phases[a['phase']]['policy']and len(a['calls_delta'])==len(a['bytes_delta'])==8
  assert all((c>0 and b>0)if a['batch']>1 and a['policy']else(c==b==0)for c,b in zip(a['calls_delta'],a['bytes_delta']))
  totals=[x+y for x,y in zip(totals,a['calls_delta'])]
 for name,values in by_phase.items():assert [v['step']for v in values]==list(range(phases[name]['forwards']))
 assert set(by_phase)==set(phases)
 assert all(v[0]==0 and v[1]==count and v[2]>0 and v[3]>0 for v,count in zip(q['precision_final_stats'],totals))
 for c in q.get('raw_checkpoints',[]):
  assert node==3;fn=c['file'];assert pathlib.Path(fn).name==fn;raw=(r/'results'/fn).read_bytes();assert hashlib.sha256(raw).hexdigest()==c['sha256'];x=(np.frombuffer(raw,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(c['shape']);assert c['shape'][1]==163840 and np.isfinite(x).all()and np.argmax(x,axis=1).tolist()==c['output_ids'];rawfiles+=1

 if node==3:
  indexed={(c['phase'],c['step']):[]for c in q['precision_dispatch']}
  case_by_phase={f'b{c["batch"]}-QA-{c["epoch"]}-{c["label"]}-r{c["repeat"]}':c for c in q['chat_cases']};natural_by_phase={c['phase']:c for c in q['natural_cases']}
  for c in q['raw_checkpoints']:
   stem=c['file'].removeprefix(f'kimi-multibatch-r3412-{row["run"]}-node3-').removesuffix('.bin');phase,st=stem.rsplit('-step',1);st=int(st);assert (phase,st)in indexed;indexed[phase,st].append(c)
   if phase in case_by_phase:
    qa=case_by_phase[phase];assert c['rows']==list(range(1,qa['batch']+1))and c['shape']==[qa['batch'],163840]and c['sha256']==qa['first4_logits_sha256'][st]and c['output_ids']==[v[st]for v in qa['generated_ids_per_row']]
    u=np.frombuffer((r/'results'/c['file']).read_bytes(),np.uint16).reshape(c['shape']);assert all(np.array_equal(u[0],v)for v in u)
   else:
    nat=natural_by_phase[phase];entry=nat['raw_checkpoint_hashes'][str(st)];assert c['sha256']==entry['raw_sha256']and c['rows']==entry['rows']==[i+1 for i,v in enumerate(nat['requests'])if len(v['generated_ids'])>st];assert c['output_ids']==[nat['requests'][i-1]['generated_ids'][st]for i in c['rows']]
  for (phase,st),values in indexed.items():
   expected=st<4 if phase in case_by_phase else st in[0,1,2,3,63,127,255,383,511]
   assert len(values)==int(expected),(phase,st,len(values))
 if node!=3:assert not q.get('raw_checkpoints')
 shared={'QA':q['chat_cases'],'natural':[(v['phase'],[x['generated_ids']for x in v['requests']],v['raw_checkpoint_hashes'])for v in q['natural_cases']]};assert shared==canonical if canonical is not None else True;canonical=shared
 reports.append({'node':node,'report_sha256':sha(p),'normal_owners_closed':8})
assert len(records)==12 and rawfiles>0
summary=[]
for B in[8,4,2]:
 for policy in[2]:
  a=[v for v in records if v['batch']==B and v['policy']==policy];summary.append({'batch':B,'policy':policy,'aggregate_decode_tok_s_range':[min(v['aggregate_decode_tok_s']for v in a),max(v['aggregate_decode_tok_s']for v in a)],'aggregate_including_prefill_tok_s_range':[min(v['aggregate_including_prefill_tok_s']for v in a),max(v['aggregate_including_prefill_tok_s']for v in a)],'decode_tok_s_per_card_range':[min(v['decode_tok_s_per_card']for v in a),max(v['decode_tok_s_per_card']for v in a)]})
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'producer_Slurm':producer,'pins':build['pins'],'cards':32,'batches':[8,4,2],'all32_owners_closed':True,'QA_per_batch_per_node':44,'CPU_full_logit_argmax_files':rawfiles,'Topk_optimization_enabled':False,'all_frozenQA_duplicate_raw_samebatch_repeat_and_EOS_passed':True,'no_B1_wide_and_actual_multirequest_F64_dispatch_verified':True,'single_user15_is_adoption_gate':False,'summary':summary,'records':records,'reports':reports,'scope':'Latestfull93 Kimi32GPU Common3393 mode2 BF16/F32 SUM via F64 scratch numerical variant;512input/max128output,separatefrom512output. FrozenB1OFF/ONrawbeforeandafter andactualallownerprecisiondispatch/noB1promotion. Native+harness. ConfiguredF32prefill=1 andpublicrequestpath, explicitnumericalvariantvsoldownedB1. CurrentSDKbaseline, noTopkpreload/candidateclaim. ActualF64per-forwardcalls/bytes andenginecountindependentlychecked. Rawquality/normalclose/Slurm/CPUargmax independentlychecked. DoesnotproveHTTP/concurrentarrivals/maxnodes/hardwarephysicalceiling.'};out.write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items()if k not in ['records','reports']}))
