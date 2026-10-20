import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,subprocess,datetime,statistics,math,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-spec-pipeline-r3706-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'experimental_excluded_main_metric':True};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
try:
 launch=load(r/'kimi-spec-pipeline-r3703-launch.json');aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if a.split('|')[1]==launch['label']];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'];tag=launch['run'];proof=load(r/'results/kimi-spec-math-r3683-build.json');base=r/'candidates/kimi-spec-math-5f91ad7-r3683';assert proof['completed']and sha(base/'libmodern_kimi_spec_math_r3683.so')==proof['library_sha256'];frozen=load(r/'results/kimi-spec-r3679-frozen-AR.json');gold=frozen['AR']['16']['generated_ids'];known=load(r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json');eos=set(load(r/'results/kimi-chat-cases-r20.json')['eos_token_ids']);reports=[];raw_count=0
 for node in range(4):
  q=load(r/'results'/f'kimi-spec-r3703-{tag}-node{node}.json');assert q['completed']and q['spec_protocol_completed']and q['all8_owned_workers_shutdown_completed']and q['owner_affinity']['verified']and q['library_sha256']==proof['library_sha256'];assert q['spec_math_policy']==2 and q['speculative_main_metric']is False and q['all_serving_cards']==32
  for n,h in q['harness_files_sha256'].items():assert sha(r/'incoming'/n)==h,n
  log=(r/'logs'/f'{launch["label"]}-{aa[0][0]}-{node}.log').read_text(errors='replace');assert'KIMI_SPEC_R3634_COMPLETE'in log and'SHARED_CACHED_R1780_COMPLETE'in log and not any(s in log for s in['Traceback','AssertionError','Segmentation fault'])
  assert len(q['frozen_AR_complete_ID_checks'])==12 and all(z['exact']for z in q['frozen_AR_complete_ID_checks']);counts=[[0]*3 for _ in range(8)]
  for z in q['math_dispatch']:
   expected=[0,0,0]if z['initial']else[q['local_KDA_layers'],q['local_KDA_layers']*z['tokens'],2*q['local_MLA_layers']if z['tokens']>len(z['rows'])else 0];assert z['expected_per_device']==expected and z['delta']==[expected]*8;counts=[[a+b for a,b in zip(x,y)]for x,y in zip(counts,z['delta'])]
  assert counts==q['verification_math_final_counts'];assert len(q['QA'])>=6
  for c in q['spec_cases']:
   if c['label'].startswith('QA-'):
    name=c['label'].split('-',2)[2];key='sort'if name.startswith('sort_probe')else name;assert c['generated_ids'][0]in[v['generated_ids'][0]for v in known['cases']if v['task']==key and v['grouped']]
  if node==3:
   for z in q['full_logit_records']:
    assert z['finite']and z['CPUargmax_equals_GPU']
    if not z['raw_persisted']:continue
    assert pathlib.Path(z['file']).name==z['file'];b=(r/'results'/z['file']).read_bytes();assert len(b)==math.prod(z['shape'])*2 and hashlib.sha256(b).hexdigest()==z['sha256'];x=(np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(z['shape']);assert np.isfinite(x).all()and x.argmax(-1).tolist()==z['outputs'];raw_count+=1
  reports.append(q)
 cases=reports[3]['spec_cases'];ids=[(c['label'],c['generated_ids'])for c in cases]
 for q in reports[:3]:assert[(c['label'],c['generated_ids'])for c in q['spec_cases']]==ids
 for c in cases:
  gen,ts=c['generated_ids'],c['timestamps'];assert len(gen)==len(ts)==c['batch']and c['all_service_cards']==32
  for v,t in zip(gen,ts):assert len(v)==len(t)>0 and all(a<=b for a,b in zip(t,t[1:]))and not any(x in eos for x in v[:-1])and(v[-1]in eos or len(v)==c['max_output_tokens'])
  first=max(v[0]for v in ts);end=max(v[-1]for v in ts);useful=sum(x not in eos for v in gen for x in v);decoded=sum(x not in eos and t>first for v,tt in zip(gen,ts)for x,t in zip(v,tt));assert useful==c['useful_outputs']and decoded==c['decode_outputs'];assert math.isclose(useful/end,c['aggregate_including_prefill_tok_s'],rel_tol=1e-12)
  if end>first:assert math.isclose(decoded/(end-first),c['aggregate_decode_tok_s'],rel_tol=1e-12)
 summaries=[];pipeline=[]
 for B in[16,32,64]:
  ar=[next(c for c in cases if c['label']==f'B{B}-pipeline-AR-{suffix}')for suffix in['A1','B1','B2','A2','B3']];assert[c['pipeline_feedback_once_per_round']for c in ar]==[False,True,True,False,True];assert all(all(v==gold[i%16]for i,v in enumerate(c['generated_ids']))for c in ar);A=statistics.median(c['aggregate_decode_tok_s']for c in[ar[0],ar[3]]);BB=statistics.median(c['aggregate_decode_tok_s']for c in[ar[1],ar[2]]);AR=statistics.median(c['aggregate_decode_tok_s']for c in[ar[1],ar[2],ar[4]])
  pipeline.append({'batch':B,'serialized_AR_decode_median':A,'pipeline_AR_ABBA_decode_median':BB,'pipeline_scheduling_speed_ratio':BB/A,'pipeline_AR3_decode_median':AR,'main_goal_metric':False})
  for K in[1,3]:
   cc=[next(c for c in cases if c['label']==f'B{B}-pipeline-lookup-K{K}-B{i}')for i in[1,2]];exact=all(c['generated_ids']==ar[1]['generated_ids']for c in cc);repeat=cc[0]['generated_ids']==cc[1]['generated_ids'];replay=all(not z['replay_token_differences']for c in cc for z in c['rounds']);rate=statistics.median(c['aggregate_decode_tok_s']for c in cc);inc=statistics.median(c['aggregate_including_prefill_tok_s']for c in cc);drafted=sum(c['drafted_tokens']for c in cc);accepted=sum(c['accepted_draft_tokens']for c in cc);per=[v for c in cc for v in c['per_request_mean_decode_tok_s']if v is not None]
   summaries.append({'batch':B,'proposal_max':K,'all_complete_IDs_equal_AR':exact,'repeats_exact':repeat,'replay_corrections_exact':replay,'qualified_experimental_comparison':exact and repeat and replay,'spec_decode_median':rate,'including_prefill_median':inc,'per_all32_cards':rate/32,'speed_ratio_vs_samepipeline_AR':rate/AR,'acceptance_fraction':accepted/drafted if drafted else None,'per_request_mean_decode_min_median_max':[min(per),statistics.median(per),max(per)]if per else None,'main_goal_metric':False})
 d.update(completed=True,producer=aa[0],pins=proof['pins'],all32_normal_owners_closed=True,CPU_fullvocab_samples=raw_count,all_complete_AR_IDs_exact_to_frozen_previous16rows=True,pipeline_AR_comparisons=pipeline,speculative_comparisons=summaries,qualified_speculative_comparisons=sum(c['qualified_experimental_comparison']for c in summaries),scope='Latest5f91 fullKimi93 PP4TP8. B16/32/64; width4initialprefill/8verify micros; at most8 proposalinputs perrow. One load; AR serialized/pipeline ABBA and separate greedy promptlookup speculative comparisons to samepipeline AR. No MTP or traineddraft, HTTP, mainmetric or hardwarepeak claim. Timings include CPUallvocabfinite/argmax, sampledraw/D2H, roundbarrier/feedback and rejection replay.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
