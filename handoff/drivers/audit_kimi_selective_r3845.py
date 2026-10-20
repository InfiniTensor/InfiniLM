import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,subprocess,datetime,math,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-selective-r3845-audit.json';assert not out.exists();d={'completed':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'main_goal_metric':False,'speculative_performance_qualified':False};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 launch=json.loads((r/'kimi-root-r3837-launch.json').read_text());aa=[v.split('|')for v in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if v.split('|')[1]==launch['label']];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'],aa;tag=launch['run'];reports=[];rawchecked=0;proof=json.loads((r/'results/kimi-MLA-trace-r3841-build.json').read_text());assert proof['completed']and sha(r/'candidates/kimi-MLA-trace-5f91ad7-r3841/libmodern_kimi_MLA_trace_r3841.so')==proof['library_sha256']
 for n,h in launch['source_sha256'].items():assert sha(r/'incoming'/n)==h
 for node in range(4):
  p=r/'results'/f'kimi-spec-r3837-{tag}-node{node}.json';q=json.loads(p.read_text());assert q['completed']and q['all8_owned_workers_shutdown_completed']and q['owner_affinity']['verified']and q['root_collection_completed']and q['root_trace_only']and not q['main_goal_metric']and not q['performance_qualified'];assert q['library_sha256']==proof['library_sha256']and q['harness_files_sha256']==launch['source_sha256'];assert len(q['QA'])==16 and all(v['complete_frozen_IDs_exact']for v in q['QA']);assert q['root_plan']==launch['selected_plan']
  cumulative=[[0]*3 for _ in range(8)]
  for ev in q['math_dispatch']:
   expected_math=[0,0,0]if ev['initial']else[q['local_KDA_layers'],q['local_KDA_layers']*ev['tokens'],2*q['local_MLA_layers']if ev['tokens']>len(ev['rows'])else 0]
   assert ev['expected_per_device']==expected_math and ev['delta']==[expected_math]*8
   cumulative=[[a+b for a,b in zip(x,y)]for x,y in zip(cumulative,ev['delta'])]
  assert cumulative==q['verification_math_final_counts']
  expected=q['root_plan']['expected_AR_ids'];cases=q['spec_cases'];assert next(v for v in cases if v['label']=='root-AR-before')['generated_ids']==[expected]==next(v for v in cases if v['label']=='root-AR-after')['generated_ids']
  paths=q['root_paths'];assert[v['label']for v in paths]==['serial-traced','block-traced','block-untraced-1','block-untraced-2'];assert paths[0]['generated_ids']==expected and paths[1]['generated_ids']==paths[2]['generated_ids']==paths[3]['generated_ids'];assert q['root_traced_vs_untraced_IDs_exact']and q['root_untraced_repeats_exact']
  log=(r/'logs'/f'{launch["label"]}-{aa[0][0]}-{node}.log').read_text(errors='replace');assert 'KIMI_SPEC_R3634_COMPLETE'in log and 'SHARED_CACHED_R1780_COMPLETE'in log and not any(v in log for v in ['Traceback','AssertionError','Segmentation fault'])
  for v in q['full_logit_records']:
   assert v['finite']and v['CPUargmax_equals_GPU']
   if node!=3 or not v['raw_persisted']:continue
   pp=r/'results'/v['file'];b=pp.read_bytes();assert pp.name==v['file']and len(b)==math.prod(v['shape'])*2 and hashlib.sha256(b).hexdigest()==v['sha256'];x=(np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(v['shape']);assert np.isfinite(x).all()and x.argmax(-1).tolist()==v['outputs'];rawchecked+=1
  reports.append(q)
 assert all(v['root_paths']==reports[0]['root_paths']for v in reports[1:]);virtualroot=pathlib.Path(reports[0]['trace_root']);assert virtualroot.parent==pathlib.Path('/refactor/results');root=r/'results'/virtualroot.name;journals=[];comparisons=[];totalbytes=0
 points=['entry','attn_res','attn_norm','attn_output','attn_added','ffn_res','ffn_norm','ffn_output','exit']
 def read_trace(label,call):
  nonlocal_dummy=None;directory=root/label/call;result={}
  for p in sorted(directory.glob('stage*-rank*.jsonl')):
   if '-kda-call'in p.name:continue
   rows=[json.loads(v)for v in p.read_text().splitlines()];assert rows
   journals.append({'file':p.relative_to(r).as_posix(),'sha256':sha(p),'entries':len(rows)})
   for row in rows:
    if row['point']not in points:continue
    key=(row['stage'],row['rank'],row['layer'],row['point']);assert key not in result;result[key]=(row,directory)
  return result
 serial1=read_trace('serial-traced','serial-first');serial2=read_trace('serial-traced','serial-second');block=read_trace('block-traced','block');assert set(serial1)==set(serial2)==set(block)
 assert{key[2]for key in block}=={0,1,3}
 for key in sorted(block,key=lambda k:(k[2],points.index(k[3]),k[1])):
  stage,rank,layer,point=key
  if rank not in[0,6]:continue
  mats=[];hashes=[]
  for mapping in[serial1,serial2,block]:
   row,directory=mapping[key];p=directory/f'stage{stage}-rank{rank}-layer{layer}-{point}-full-bf16.bin';b=p.read_bytes();assert len(b)==row['tokens']*row['hidden']*2;totalbytes+=len(b);x=np.frombuffer(b,np.uint16).reshape(row['tokens'],row['hidden']);f=(x.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(f).all();mats.append((x,f));hashes.append(hashlib.sha256(b).hexdigest())
  assert mats[0][0].shape==mats[1][0].shape==(1,block[key][0]['hidden'])and mats[2][0].shape==(2,block[key][0]['hidden'])
  for token in[0,1]:
   ref,ff=mats[token];actual,af=mats[2];neq=int(np.count_nonzero(ref[0]!=actual[token]));comparisons.append({'stage':stage,'rank':rank,'layer':layer,'point':point,'token_offset':token,'diagnostic_BF16_different_elements':neq,'max_absolute_difference':float(np.abs(ff[0]-af[token]).max()),'serial_sha256':hashes[token],'block_sha256':hashes[2]})
 first={}
 for v in comparisons:
  if v['diagnostic_BF16_different_elements']:
   first.setdefault(str(v['token_offset']),v)

 import statistics
 frozen=json.loads((r/'results/kimi-spec-r3759-frozen-AR.json').read_text());eos=set(json.loads((r/'results/kimi-chat-cases-r20.json').read_text())['eos_token_ids']);selective=[];tail=reports[3]
 assert all(q['selective_prompt_lookup_matrix_completed']and len(q['selective_prompt_lookup_experiment_r3843'])==12 for q in reports)
 for q in reports[:3]:assert[(v['label'],v['generated_ids'])for v in q['spec_cases']]==[(v['label'],v['generated_ids'])for v in tail['spec_cases']]
 for v in tail['spec_cases']:
  ids,ts=v['generated_ids'],v['timestamps'];assert len(ids)==len(ts)==v['batch']and v['all_service_cards']==32
  for seq,times in zip(ids,ts):assert len(seq)==len(times)and all(a<=b for a,b in zip(times,times[1:]))and not any(t in eos for t in seq[:-1])and(seq[-1]in eos or len(seq)==v['max_output_tokens'])
  first=max(t[0]for t in ts);last=max(t[-1]for t in ts);useful=sum(t not in eos for seq in ids for t in seq);decoded=sum(t>first and x not in eos for seq,times in zip(ids,ts)for x,t in zip(seq,times));assert useful==v['useful_outputs']and decoded==v['decode_outputs']
  if last>first:assert math.isclose(v['aggregate_decode_tok_s'],decoded/(last-first),rel_tol=1e-12)
  assert math.isclose(v['aggregate_including_prefill_tok_s'],useful/last,rel_tol=1e-12)
 for B in[1,8,16]:
  ar=[v for v in tail['spec_cases']if v['label']in[f'selective-B{B}-AR-A1',f'selective-B{B}-AR-A2']];assert len(ar)==2 and all(v['generated_ids']==frozen['AR'][str(B)]['generated_ids']for v in ar);ar_rate=statistics.median(v['aggregate_decode_tok_s']for v in ar)
  for minimum,K in[(3,3),(4,1)]:
   vs=[v for v in tail['spec_cases']if v['label']in[f'selective-B{B}-n{minimum}-K{K}-r1',f'selective-B{B}-n{minimum}-K{K}-r2']];assert len(vs)==2 and all(v['minimum_lookup_ngram']==minimum and v['draft_max_tokens']==K for v in vs);exact=all(v['generated_ids']==ar[0]['generated_ids']for v in vs);repeat=vs[0]['generated_ids']==vs[1]['generated_ids'];replay=all(not z['replay_token_differences']for v in vs for z in v['rounds']);drafted=sum(v['drafted_tokens']for v in vs);accepted=sum(v['accepted_draft_tokens']for v in vs);rate=statistics.median(v['aggregate_decode_tok_s']for v in vs);rates=[t for v in vs for t in v['per_request_mean_decode_tok_s']if t is not None]
   selective.append({'batch':B,'minimum_ngram':minimum,'draft_maximum':K,'complete_IDs_equal_frozen_AR':exact,'own_repeat_IDs_exact':repeat,'replay_correction_IDs_exact':replay,'actually_drafted_tokens':drafted,'actually_accepted_tokens':accepted,'effective_speculative_exercise':drafted>0 and accepted>0,'qualified_experimental_comparison':exact and repeat and replay and drafted>0 and accepted>0,'AR_decode_median':ar_rate,'spec_decode_median':rate,'decode_ratio':rate/ar_rate,'spec_including_prefill_median':statistics.median(v['aggregate_including_prefill_tok_s']for v in vs),'decode_per_all32_cards':rate/32,'per_request_decode_min_median_max':[min(rates),statistics.median(rates),max(rates)]if rates else None,'acceptance_fraction':accepted/drafted if drafted else None,'nonempty_proposal_round_fraction':sum(z['drafted']>0 for v in vs for z in v['rounds'])/sum(len(v['rounds'])for v in vs),'main_goal_metric':False,'scope':'One residenttarget,2bracketingAR/2selective repeats,512input/max128,static2families. CPUfullvocab checks/control included. Zeroaccepted or zerodrafted nevercounts as successful speculativeexercise. NoHTTP/productionpeak/traineddraft.'})
 d.update(selective_comparisons=selective,qualified_selective_count=sum(v['qualified_experimental_comparison']for v in selective),main_goal_metric=False,trace_layers=[0,1,3],target_KDA_layer=1)
 d.update(completed=True,producer=aa[0],normal_owned_workers_closed=32,CPU_fullvocab_raw_files=rawchecked,full_known_QA=16,serial_full_frozen_IDs_exact=True,traced_and_untraced_block_full_IDs_exact=True,block_complete_IDs_equal_serial=reports[0]['root_paths'][1]['generated_ids']==reports[0]['root_paths'][0]['generated_ids'],root_plan=launch['selected_plan'],root_paths=reports[0]['root_paths'],generic_trace_CPU_checked_bytes=totalbytes,generic_trace_journals=journals,first_generic_diagnostic_difference_by_token=first,diagnostic_comparisons=comparisons,scope='First actual acceptedblock fromidenticalprefix, teacher input only forcorrectness localization. Complete serial frozenAR, traced/untraced fullcontinuation exact, finiteCPUargmax/owner/Slurm audited. Rawdifferences diagnostic only, nottokenfailure or performance. Genericfulltrace rank0/6 layers0,1,3; otherTP ranks producerlastsnapshots andselectedKDA1 details persisted butnotindependentlycompared here. No draftperformance gain or mainmetric adoption.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_ROOT_R3818_AUDITED',d['block_complete_IDs_equal_serial'],first)
