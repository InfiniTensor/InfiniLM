import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,datetime,subprocess,math,statistics,collections
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/dspark-batch-r3646-audit.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910377','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed'],text=True).splitlines()if a.split('|')[1]=='dspark-batch-r3640'];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'];step=aa[0][0];featurepath=r/'results/kimi-dspark-features-r2773-audit.json';feature=json.loads(featurepath.read_text());assert feature['completed'];groups=collections.defaultdict(list);masks=[];sample_count=records=0;replicas={}
source=r/'models/Kimi-K3-DSpark-3c5bac30-r2699/model.safetensors';h=hashlib.sha256()
with source.open('rb')as f:
 while b:=f.read(16*1024**2):h.update(b)
assert h.hexdigest()=='ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca'
for device in range(8):
 p=r/'results'/f'dspark-batch-r3640-{step}-device{device}.json';q=json.loads(p.read_text());assert q['completed']and not q.get('error')and q['device']==device and q['loaded_draft_tensor_count']==62 and q['sampled_all62_weights_unchanged']and q['normal_Torch_tensor_cleanup']and q['all_model_parameters_buffers_head_weakrefs_released']and q['one_load_many_tests']and q['source_run']==feature['source_run']and q['feature_audit_sha256']==sha(featurepath)and q['actual_target_verification_executed']is False and len(q['cases'])==30;masks+=q['CPU_affinity']
 for c in q['cases']:
  B,K=c['batch'],c['block_size'];assert B in[1,2,4,8,16]and K in[2,4,7]and len(c['records'])==4;ref=next(x for x in feature['cases']if x['label']==c['label']and x['repeat']==1);previous=None
  for z in c['records']:
   ids=z['additional_proposed_ids'];assert len(ids)==B and all(len(v)==K for v in ids)and z['known_anchor_id']==ref['first_token']and z['native_reference_next_ids']==ref['target_greedy_reference_ids_first16'][1:K+1]and z['same_configuration_full_token_repeat_exact']and z['independent_CPUargmax_checked'];assert previous is None or ids==previous;previous=ids;key=(B,K,c['label']);assert ids==replicas.setdefault(key,ids);matches=[]
   for row,cs in zip(ids,z['confidence_probabilities']):
    assert len(cs)==K and all(math.isfinite(v)and 0<=v<=1 for v in cs);m=0
    for a,b in zip(row,z['native_reference_next_ids']):
     if a!=b or a in feature['eos_token_ids']:break
     m+=1
    matches.append(m)
   assert matches==z['contiguous_additional_greedy_matches']
   for name,v in z['files'].items():
    assert v['bytes']==B*K*(7168 if name=='hidden'else 163840)*2
    if not v['persisted']:continue
    p=r/'results'/v['file'];b=p.read_bytes();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256'];a=(np.frombuffer(b,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32);assert np.isfinite(a).all()
    if name=='Markov_logits':assert np.argmax(a.reshape(B,K,163840),axis=-1).tolist()==ids
    sample_count+=1
   assert z['host_completed_ms']>0 and z['GPU_completed_interval_ms']>0;records+=1
  groups[B,K].append({'device':device,'prompt':c['label'],'warm_host_ms':statistics.median(z['host_completed_ms']for z in c['records'][1:]),'warm_GPU_ms':statistics.median(z['GPU_completed_interval_ms']for z in c['records'][1:]),'offline_contiguous_matches_per_row':c['records'][1]['contiguous_additional_greedy_matches']})
assert len(masks)==len(set(masks))==64 and len(replicas)==30 and records==960;log=(r/'logs'/f'dspark-batch-r3640-{step}.log').read_text();assert'DSPARK_BATCH_R3640_ALL8_CLOSED'in log and'SHARED_CACHED_R1780_COMPLETE'in log
summary=[]
for(B,K),items in sorted(groups.items()):
 ms=statistics.median(x['warm_host_ms']for x in items);summary.append({'batch':B,'proposals_per_request':K,'warm_single_GPU_host_ms_median':ms,'warm_single_GPU_device_ms_median':statistics.median(x['warm_GPU_ms']for x in items),'proposed_draft_tokens_per_second_single_GPU':B*K*1000/ms,'offline_firstblock_match_min_max':[min(v for x in items for v in x['offline_contiguous_matches_per_row']),max(v for x in items for v in x['offline_contiguous_matches_per_row'])],'raw_results':items})
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'producer':aa[0],'GPUs':8,'unique_prompts':2,'records':records,'independently_checked_raw_files':sample_count,'normal_model_parameter_cleanup':True,'weights_sha256':h.hexdigest(),'same_batch_across8GPU_complete_proposal_IDs_exact':True,'summary':summary,'actual_online_target_verification':False,'end_to_end_speculative_speedup_measured':False,'excluded_from_main_goal_metric':True,'scope':'Full pinned DSpark draft weights, actual512-token frozen3e0 target features; B1..16/K2/4/7,4repeat/case,8independentreplicas,one load/many tests. Complete IDs repeat/crossdevice exact; raw differences diagnostic. Timings recompute full512 draftcontext; no steadycacheddecode, broad acceptance orlatest06d feature compatibility claim. Draft proposal tok/s is not useful targetmodel throughput.'};out.write_text(json.dumps(d,indent=2));print(json.dumps({'completed':True,'summary':[{k:v for k,v in x.items()if k!='raw_results'}for x in summary]}))
