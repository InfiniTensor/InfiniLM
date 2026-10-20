import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,datetime,statistics,math,subprocess,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';mode=int(os.environ['KIMI_SPEC_MATH_AUDIT_MODE']);assert mode in[1,2];out=r/'results'/f'kimi-spec-math-r3748-M{mode}-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'experimental_excluded_main_metric':True};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 launch=next(j for j in json.loads((r/'kimi-spec-r3747-launch.json').read_text())['jobs']if j['mode']==mode);aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()];aa=[a for a in aa if a[1]==launch['label']];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'],aa;step=aa[0][0];tag=launch['run'];proof=json.loads((r/'results/kimi-spec-math-r3712-build.json').read_text());stage=json.loads((r/'results/kimi-spec-math-r3712-stage.json').read_text());base=r/'candidates/kimi-spec-math-5f91ad7-r3712';assert proof['completed']and sha(base/'libmodern_kimi_spec_math_r3712.so')==proof['library_sha256']and proof['pins']==stage['pins'];assert proof['CPU_contract']['valid_metadata_cases']==56 and proof['CPU_contract']['invalid_metadata_cases']==10
 for n,h in stage['files_sha256'].items():assert sha(base/n)==h,n
 for n,h in launch['source_sha256'].items():assert sha(r/'incoming'/n)==h,n
 reports=[];raw_checked=0
 for node in range(4):
  p=r/'results'/f'kimi-spec-r3747-{tag}-node{node}.json';q=json.loads(p.read_text());assert q['completed']and q['spec_protocol_completed']and q['all8_owned_workers_shutdown_completed']and q['owner_affinity']['verified']and q['library_sha256']==proof['library_sha256'];assert q['harness_files_sha256']==launch['source_sha256'];assert q['all_serving_cards']==32 and q['speculative_main_metric']is False;assert len(q['QA'])>=6 and all(v['complete_frozen_IDs_exact']for v in q['QA'])
  log=(r/'logs'/f'{launch["label"]}-{step}-{node}.log').read_text(errors='replace');assert'KIMI_SPEC_R3634_COMPLETE'in log and'SHARED_CACHED_R1780_COMPLETE'in log and not any(v in log for v in ['Traceback','AssertionError','Segmentation fault','Fatal Python'])
  if node==3:
   for v in q['full_logit_records']:
    assert v['finite']and v['CPUargmax_equals_GPU']and len(v['outputs'])==v['shape'][0]
    if not v['raw_persisted']:continue
    assert pathlib.Path(v['file']).name==v['file'];p=r/'results'/v['file'];b=p.read_bytes();assert len(b)==math.prod(v['shape'])*2 and hashlib.sha256(b).hexdigest()==v['sha256'];x=(np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(v['shape']);assert np.isfinite(x).all()and np.argmax(x,axis=-1).tolist()==v['outputs'];raw_checked+=1
  assert q['spec_math_policy']==mode and len(q['frozen_AR_complete_ID_checks'])==5 and all(v['exact']for v in q['frozen_AR_complete_ID_checks'])
  counts=[[0]*3 for _ in range(8)]
  for event in q['math_dispatch']:
   expected=[0,0,0]if event['initial']else[q['local_KDA_layers'],q['local_KDA_layers']*event['tokens'],2*q['local_MLA_layers']if mode==2 and event['tokens']>len(event['rows'])else 0]
   assert event['expected_per_device']==expected and event['delta']==[expected]*8
   assert expected[2]in[0,2*q['local_MLA_layers']] and(mode==2 or expected[2]==0)
   counts=[[a+b for a,b in zip(v,w)]for v,w in zip(counts,event['delta'])]
  assert counts==q['verification_math_final_counts']
  frozen=r/'results/kimi-spec-r3747-frozen-AR.json';assert sha(frozen)==q['frozen_AR_reference_sha256']
  fr=json.loads(frozen.read_text())
  for B in[1,2,4,8,16]:assert next(v for v in q['spec_cases']if v['label']==f'B{B}-AR-A1')['generated_ids']==fr['AR'][str(B)]['generated_ids']
  reports.append(q)
 tail=reports[3];cases=tail['spec_cases'];eos=set(json.loads((r/'results/kimi-chat-cases-r20.json').read_text())['eos_token_ids']);summaries=[]
 for q in reports[:3]:assert [(v['label'],v['generated_ids'])for v in q['spec_cases']]==[(v['label'],v['generated_ids'])for v in cases]
 for v in cases:
  ids,ts=v['generated_ids'],v['timestamps'];assert len(ids)==len(ts)==v['batch']and v['all_service_cards']==32
  for row,times in zip(ids,ts):assert len(row)==len(times)and all(a<=b for a,b in zip(times,times[1:]))and not any(t in eos for t in row[:-1])and(row[-1]in eos or len(row)==v['max_output_tokens'])
  first=max(t[0]for t in ts);last=max(t[-1]for t in ts);useful=sum(t not in eos for row in ids for t in row);decoded=sum(t>first and x not in eos for row,times in zip(ids,ts)for x,t in zip(row,times));assert v['useful_outputs']==useful and v['decode_outputs']==decoded
  if last>first:assert math.isclose(v['aggregate_decode_tok_s'],decoded/(last-first),rel_tol=1e-12)
  assert math.isclose(v['aggregate_including_prefill_tok_s'],useful/last,rel_tol=1e-12)
 for B in [1,2,4,8,16]:
  ar=[v for v in cases if v['label']in[f'B{B}-AR-A{i}'for i in [1,2,3]]];assert len(ar)==3 and all(v['generated_ids']==ar[0]['generated_ids']for v in ar);base_decode=statistics.median(v['aggregate_decode_tok_s']for v in ar);base_incl=statistics.median(v['aggregate_including_prefill_tok_s']for v in ar)
  for K in [1,3,7]:
   spec=[v for v in cases if v['label']in[f'B{B}-lookup-K{K}-B{i}'for i in [1,2]]];assert len(spec)==2;exact=all(v['generated_ids']==ar[0]['generated_ids']for v in spec);replay_exact=all(not z['replay_token_differences']for v in spec for z in v['rounds']);repeat=spec[0]['generated_ids']==spec[1]['generated_ids'];decode=statistics.median(v['aggregate_decode_tok_s']for v in spec);incl=statistics.median(v['aggregate_including_prefill_tok_s']for v in spec);drafted=sum(v['drafted_tokens']for v in spec);accepted=sum(v['accepted_draft_tokens']for v in spec);rates=[x for v in spec for x in v['per_request_mean_decode_tok_s']if x is not None]
   summaries.append({'batch':B,'proposal_max_tokens':K,'all_complete_IDs_equal_AR':exact,'same_configuration_repeat_IDs_exact':repeat,'replayed_correction_IDs_exact':replay_exact,'qualified_experimental_comparison':exact and repeat and replay_exact,'AR_decode_median':base_decode,'spec_decode_median':decode,'decode_speed_ratio':decode/base_decode,'AR_including_prefill_median':base_incl,'spec_including_prefill_median':incl,'spec_decode_per_all32_cards':decode/32,'per_request_decode_min_median_max':[min(rates),statistics.median(rates),max(rates)]if rates else None,'acceptance_fraction':accepted/drafted if drafted else None,'drafted_tokens':drafted,'accepted_draft_tokens':accepted,'replay_input_tokens':sum(v['replay_input_tokens']for v in spec),'main_goal_metric':False})
 d.update(completed=True,verification_math_mode=mode,producer=aa[0],pins=proof['pins'],normal_owners_closed=32,full_logits_persisted_samples_independently_CPU_checked=raw_checked,all_forwards_producer_finite_and_CPUargmax_checked=True,all_nodes_complete_output_IDs_exact=True,comparisons=summaries,qualified_comparison_count=sum(x['qualified_experimental_comparison']for x in summaries),scope='Actual latest5f91 full93model nativeBF16 promptlookup speculative decoding; no MTP/draft weights and no oracle proposals. B1/2/4/8/16,K1/3/7,max128,2natural512prompts staticcohorts; eachcaseAR3/spec2 repeats. Only tokenexact comparisons usable experimentally; neither main target performance nor broad productionquality qualification. Elapsed includes CPUfullvocabargmax,D2H,control barriers and sampled raw persistence; no HTTP/continuousbatch/productionmax claim. Samebatch complete AR tokens additionally exact to frozen previous06d AR repeats; no F64quality equivalence claimed. Verifier-only KDAroundstep and optional MLAdecodeprojection actualdispatch counters checked onall32owners. Different early/late AR baselines bound drift, no automatic causality fromsinglemedian.',report_sha256={f'node{i}':sha(r/'results'/f'kimi-spec-r3747-{tag}-node{i}.json')for i in range(4)})
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
