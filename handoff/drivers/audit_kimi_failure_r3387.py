import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,datetime,hashlib,json,pathlib,subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-failure-r3387-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'producer_entire_matrix_qualified':False,'new_multibatch_throughput_qualified':False}
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 queue=load(r/'models-latest-r3267-queue.json');row=queue['rows'][0];branch=row['branches'][0];aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed,NodeList%160'],text=True).splitlines()if a.split('|')[1]==branch['label']];assert len(aa)==1 and aa[0][0]=='910373.766'and aa[0][2:4]==['FAILED','1:0'];d['producer_state']=aa[0]
 base=r/'candidates/kimi-reduce-model-3e0d89fe-r3261';build=load(r/'results/kimi-reduce-model-r3262-build.json');stage=load(r/'results/kimi-reduce-model-r3261-stage.json');assert build['completed']and stage['ready'];assert sha(base/'libmodern_kimi_reduce_r3261.so')==build['library_sha256']==row['library_sha256']
 for n,h in {**stage['unchanged_GPU_objects'],**stage['unchanged_GPU_helpers']}.items():assert sha(base/n)==h
 for p,h in build['mapped_libraries'].items():assert sha(pathlib.Path(p.replace('/refactor',str(r),1)))==h
 chat=load(r/'results/kimi-chat-cases-r20.json');known=load(r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json');eos=set(chat['eos_token_ids']);failure='b8-QA-candidate_r1-retrieval-r1';canonical=None;reports=[];rawfiles=0;failure_rows=[];dispatches=[]
 for node in range(4):
  p=r/'results'/f'kimi-multibatch-r3266-{row["run"]}-node{node}.json';q=load(p);assert not q['completed']and q['all8_owned_workers_shutdown_completed']and not q.get('shutdown_error')and 'duplicate_rows_fullraw_mismatch'in q['error']and failure in q['error']and q['error'].endswith(', 2))')
  assert q['node']==node and q['TP']==8 and q['PP']==4 and q['all_serving_cards']==32 and q['loaded_keys']==q['parameter_keys']and q['library_sha256']==build['library_sha256']
  assert q['LM_source_commit']==build['pins']['InfiniLM']and q['precision_defaultOFF_invalid_no_mutation']and q['B1_OFF_ON_frozen_QA_and_raw_exact']and not q.get('natural_cases')and not q.get('completed_batches')
  assert not q.get('owner_affinity')and not q.get('output_metrics')and not q.get('B1_after_multibatch_QA_raw_exact')
  for n,h in q['harness_files_sha256'].items():assert sha(r/'incoming'/n)==h
  for name,h in q['actual_LM_native_libraries'].items():assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h
  assert len(q['chat_cases'])==48;refs={};phases={};b1=collections.Counter()
  for c in q['chat_cases']:
   B=c['batch'];policy=c['policy'];assert B in[1,8]and policy==int(c['epoch']!='frozen_OFF')and c['known_answer_ids_exact']and c['same_batch_fullraw_repeat_exact']and c['duplicate_rows_fullraw_exact']and c['stopped_at_EOS']
   ids=c['generated_ids_per_row'];assert len(ids)==B and all(a==ids[0]for a in ids)and ids[0][-1]in eos and all(t not in eos for t in ids[0][:-1]);key='sort'if c['label'].startswith('sort_probe_')else c['label'];assert any(v['generated_ids']==[ids[0]]for v in known['cases']if v['task']==key and v['grouped'])
   value=(ids[0],c['first4_logits_sha256']);assert value==refs.setdefault((B,key),value)
   phase=f'b{B}-QA-{c["epoch"]}-{c["label"]}-r{c["repeat"]}';assert phase not in phases;phases[phase]=c
   if B==1:b1[c['epoch']]+=1
   else:assert c['epoch']=='candidate_r1'and c['label']in list(chat['cases'])[:2]
  assert b1=={'frozen_OFF':22,'frozen_ON':22}
  assert [(c['label'],c['repeat'])for c in q['chat_cases']if c['batch']==8]==[(label,repeat)for label in list(chat['cases'])[:2]for repeat in [1,2]]
  if canonical is None:canonical=q['chat_cases']
  else:assert q['chat_cases']==canonical
  byphase=collections.defaultdict(list);totals=[0]*8
  for c in q['precision_dispatch']:
   assert c['phase']in phases or c['phase']==failure;assert len(c['calls_delta'])==len(c['bytes_delta'])==8;byphase[c['phase']].append(c)
   assert all((a>0 and b>0)if c['batch']>1 and c['policy']else(a==b==0)for a,b in zip(c['calls_delta'],c['bytes_delta']))
   totals=[a+b for a,b in zip(totals,c['calls_delta'])]
  assert set(byphase)==set(phases)|{failure}
  for phase,c in phases.items():assert[v['step']for v in byphase[phase]]==list(range(len(c['generated_ids_per_row'][0])))
  assert[v['step']for v in byphase[failure]]==[0,1,2]and all(v['batch']==8 and v['policy']==1 for v in byphase[failure]);dispatches.append({'node':node,'recorded_forwards':len(q['precision_dispatch']),'promoted_calls_per_owner':totals,'no_B1_promotion_verified':True,'actual_positive_B8_promotion_verified':True})
  if node!=3:assert not q.get('raw_checkpoints')
  for c in q.get('raw_checkpoints',[]):
   fn=c['file'];assert pathlib.Path(fn).name==fn;raw=(r/'results'/fn).read_bytes();assert hashlib.sha256(raw).hexdigest()==c['sha256'];u=np.frombuffer(raw,np.uint16).reshape(c['shape']);x=(u.astype(np.uint32)<<16).view(np.float32);assert c['shape'][1]==163840 and np.isfinite(x).all()and np.argmax(x,axis=1).tolist()==c['output_ids'];rawfiles+=1
   label=fn.removeprefix(f'kimi-multibatch-r3266-{row["run"]}-node3-').removesuffix('.bin');phase,step=label.rsplit('-step',1);step=int(step)
   if phase in phases:
    case=phases[phase];assert c['shape'][0]==case['batch']and c['sha256']==case['first4_logits_sha256'][step]and c['output_ids']==[a[step]for a in case['generated_ids_per_row']]and np.array_equal(u,np.repeat(u[:1],len(u),axis=0))
   else:
    assert phase==failure and c['shape']==[8,163840]and step in[0,1,2];stats=[]
    for i in range(8):
     delta=x[i].astype(np.float64)-x[0].astype(np.float64);stats.append({'row':i,'changed_BF16_words_vs_row0':int(np.count_nonzero(u[i]!=u[0])),'changed_bytes_vs_row0':int(np.count_nonzero(u[i].view(np.uint8)!=u[0].view(np.uint8))),'max_abs':float(np.max(np.abs(delta))),'relative_L2':float(np.linalg.norm(delta)/np.linalg.norm(x[0].astype(np.float64))),'argmax':int(np.argmax(x[i]))})
    failure_rows.append({'step':step,'raw_file':fn,'sha256':c['sha256'],'rows':stats,'duplicate_rows_exact':np.array_equal(u,np.repeat(u[:1],8,axis=0)),'all_argmax_equal':len(set(c['output_ids']))==1})
  log=(r/'logs'/f'{branch["label"]}-{aa[0][0]}-{node}.log').read_text(errors='replace');assert 'duplicate_rows_fullraw_mismatch'in log and'Destroy COMPLETE'in log and 'Segmentation fault'not in log
  reports.append({'node':node,'file':p.name,'sha256':sha(p),'normal_owners_closed':8,'loaded_keys':q['loaded_keys']})
 assert len(failure_rows)==3 and [v['step']for v in failure_rows]==[0,1,2]and all(v['duplicate_rows_exact']for v in failure_rows[:2])and not failure_rows[2]['duplicate_rows_exact']
 d.update(completed=True,pins=build['pins'],QA_prefix_cases_per_node=48,B1_OFF_QA=22,B1_ON_QA=22,B8_prefix_QA=4,CPU_fullvocab_argmax_raw_files=rawfiles,normal_owners_closed=32,actual_precision_dispatch=dispatches,failed_window=failure_rows,reports=reports,whole_run_affinity_readback_persisted=False,whole_run_engine_counter_persisted=False,postmultibatch_B1_QA_completed=False,scope='Independentauditoffailedfull93model prefixandactualB8retrievalstep2negative; all48completedQA/node/frozenIDs/EOS/tailraw/noB1promotion/actual32ownerpromotion/normalshutdown checked. Onlyemitterfullrawarchived; priorproduceralltailrankchecknotindependentlyrerun. Missingfinalaffinity/enginecalls/afterB1 gates explicitlynotqualified. Argmaxequalitydoesnotreplacefullrawgate. NoTPS/causalroot/hardwarefaultclaim.')
except BaseException as e:
 import traceback
 d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_FAILURE_R3387_PREFIX_AND_NEGATIVE_AUDIT_COMPLETE',rawfiles,flush=True)
