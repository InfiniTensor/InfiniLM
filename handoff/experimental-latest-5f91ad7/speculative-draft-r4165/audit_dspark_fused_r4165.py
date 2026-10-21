import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,datetime,subprocess,math,statistics,collections
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/dspark-fused-r4165-audit.json';assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[v.split('|')for v in subprocess.check_output(['sacct','-j','910377','-n','-P','-o','JobID,JobName%120,State,ExitCode,Elapsed,NodeList%180'],text=True).splitlines()if v.split('|')[1]=='dspark-fused-r4164'];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'];step=rows[0][0]
featurepath=r/'results/kimi-dspark-features-r2773-audit.json';features=json.loads(featurepath.read_text());assert features['completed']
h=hashlib.sha256()
with(r/'models/Kimi-K3-DSpark-3c5bac30-r2699/model.safetensors').open('rb')as f:
 while b:=f.read(16*1024**2):h.update(b)
assert h.hexdigest()=='ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca'
groups=collections.defaultdict(list);cross_device={};masks=[];record_count=rawfiles=0;cache_same=True
for device in range(8):
 p=r/'results'/f'dspark-fused-r4164-{step}-device{device}.json';q=json.loads(p.read_text());assert q['completed']and not q.get('error')and q['device']==device and q['loaded_draft_tensor_count']==62 and q['sampled_all62_weights_unchanged']and q['normal_Torch_tensor_cleanup']and q['all_model_parameters_buffers_head_weakrefs_released']and q['one_load_many_tests']and q['source_run']==features['source_run']and q['feature_audit_sha256']==sha(featurepath)and not q['actual_target_verification_executed']and len(q['cases'])==32;masks+=q['CPU_affinity']
 for c in q['cases']:
  B,K=c['batch'],c['block_size'];assert B in[1,2,4,8]and K in[4,7,8,16]and c['cache_object_released']and c['K16_extrapolation_not_trained_length']==(K==16)and len(c['records'])==8 and c['draft_prefix_cache_init_ms']>0
  ref=next(v for v in features['cases']if v['label']==c['label']and v['repeat']==1);own={}
  for z in c['records']:
   enabled=z['cache_enabled'];ids=z['additional_proposed_ids'];assert len(ids)==B and all(len(v)==K for v in ids)and z['known_anchor_id']==ref['first_token']and z['gold_future_available']==len(ref['target_greedy_reference_ids_first16'])-1 and z['CPUargmax_checked']and z['finite']and z['own_ID_repeat_exact']and z['cache_length_after_forward']==(512+K if enabled else 512)
   assert ids==own.setdefault(enabled,ids);key=(B,K,c['label'],enabled);assert ids==cross_device.setdefault(key,ids)
   matches=[]
   for values,cs in zip(ids,z['confidence_probabilities']):
    assert len(cs)==K and all(math.isfinite(v)and 0<=v<=1 for v in cs);n=0
    for a,b in zip(values,ref['target_greedy_reference_ids_first16'][1:]):
     if a!=b or a in features['eos_token_ids']:break
     n+=1
    matches.append(n)
   assert matches==z['contiguous_greedy_matches_to_old_target']and z['acceptance_censored_at_reference_end']==[n==z['gold_future_available']and K>z['gold_future_available']for n in matches]
   v=z['file'];assert v['bytes']==B*K*163840*2
   if v['persisted']:
    raw=(r/'results'/v['file']).read_bytes();assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha256'];values=(np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32);assert np.isfinite(values).all()and values.reshape(B,K,163840).argmax(-1).tolist()==ids;rawfiles+=1
   assert z['host_completed_ms']>0 and z['GPU_completed_interval_ms']>0;record_count+=1
  same=own[True]==own[False];assert same==c['cache_vs_recompute_IDs_exact'];cache_same=cache_same and same
  for enabled in [False,True]:
   timings=[z for z in c['records']if z['cache_enabled']==enabled and z['repeat']>0]
   groups[B,K,enabled].append({'device':device,'prompt':c['label'],'host_ms':statistics.median(z['host_completed_ms']for z in timings),'device_ms':statistics.median(z['GPU_completed_interval_ms']for z in timings),'offline_old_feature_matches':timings[0]['contiguous_greedy_matches_to_old_target']})
assert len(masks)==len(set(masks))==64 and record_count==2048
log=(r/'logs'/f'dspark-fused-r4164-{step}.log').read_text();assert'DSPARK_FUSED_R4164_ALL8_CLOSED'in log and'SHARED_CACHED_R1780_COMPLETE'in log
summary=[{'batch':B,'proposals':K,'prefix_cache':enabled,'host_ms_median':statistics.median(x['host_ms']for x in items),'device_ms_median':statistics.median(x['device_ms']for x in items),'raw_results':items}for(B,K,enabled),items in sorted(groups.items())]
phase_groups=collections.defaultdict(list);graph_groups=collections.defaultdict(list);graph_ids={};graph_records=phase_records=0;graph_raw=0
for device in range(8):
 q=json.loads((r/'results'/f'dspark-fused-r4164-{step}-device{device}.json').read_text())
 for c in q['cases']:
  B,K=c['batch'],c['block_size'];profiles=c['performance_breakdown'];g=c['graph_experiment'];assert len(profiles)==6 and g['actual_device_graph']and g['normal_reset_before_cache_free']and g['prefix_sample_readonly']and g['capture_ms']>0 and len(g['records'])==16
  for z in profiles:
   assert z['CPUargmax']and z['finite']and z['IDs_exact_to_cached_eager']and all(z[k]>0 for k in ['host_ms','backbone_device_interval_ms','LMhead_device_interval_ms','sequential_Markov_confidence_concat_device_interval_ms']);phase_records+=1
  phase_groups[B,K].append({name:statistics.median(z[name]for z in profiles[1:])for name in ['host_ms','backbone_device_interval_ms','LMhead_device_interval_ms','sequential_Markov_confidence_concat_device_interval_ms']})
  own={}
  for z in g['records']:
   assert z['prefix_still_original_prompt']and z['IDs_match']and z['CPUargmax']and z['finite']and z['host_ms']>0 and z['device_ms']>0;ids=z['IDs'];assert ids==z['expected_eager_IDs']and len(ids)==B and all(len(v)==K for v in ids);assert ids==own.setdefault(z['variant'],ids);key=(B,K,c['label'],z['variant']);assert ids==graph_ids.setdefault(key,ids)
   if z['variant']==0:assert ids==next(v['additional_proposed_ids']for v in c['records']if v['cache_enabled'])
   v=z['raw'];assert v['bytes']==B*K*163840*2
   if v['persisted']:
    raw=(r/'results'/v['file']).read_bytes();assert len(raw)==v['bytes']and hashlib.sha256(raw).hexdigest()==v['sha256'];values=(np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32);assert np.isfinite(values).all()and values.reshape(B,K,163840).argmax(-1).tolist()==ids;graph_raw+=1
   graph_records+=1
  for enabled in[False,True]:
   costs=[z for z in g['records']if z['graph_enabled']==enabled and z['repeat']>0];graph_groups[B,K,enabled].append({'host_ms':statistics.median(z['host_ms']for z in costs),'device_ms':statistics.median(z['device_ms']for z in costs)})
assert phase_records==1536 and graph_records==4096
phase_summary=[{'batch':B,'proposals':K,**{name:statistics.median(x[name]for x in items)for name in items[0]}}for(B,K),items in sorted(phase_groups.items())]
graph_summary=[{'batch':B,'proposals':K,'graph':enabled,**{name:statistics.median(x[name]for x in items)for name in items[0]}}for(B,K,enabled),items in sorted(graph_groups.items())]
import struct
source=r/'models/Kimi-K3-DSpark-3c5bac30-r2699/model.safetensors'
with source.open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
def mapped(key):
 z=header[key];assert z['dtype']=='BF16'and z['shape']==[163840,256];return np.memmap(source,dtype=np.uint16,mode='r',offset=8+n+z['data_offsets'][0],shape=(163840,256))
w1=mapped('markov_head.markov_w1.weight');w2=mapped('markov_head.markov_w2.weight')
def f32(bits):return(np.asarray(bits,dtype=np.uint32)<<16).view(np.float32)
def rounded(x):
 raw=np.asarray(x,dtype=np.float32).copy().view(np.uint32);raw+=0x7fff+((raw>>16)&1);return f32(raw>>16)
proof=json.loads((r/'results/dspark-markov-r4163-build.json').read_text());assert proof['completed']and sha(r/'candidates/dspark-markov-r4163/libdspark_markov_r4163.so')==proof['library_sha256'];oracle_errors=[]
for device in range(8):
 q=json.loads((r/'results'/f'dspark-fused-r4164-{step}-device{device}.json').read_text());assert q['fused_library_sha256']==proof['library_sha256']and q['fused_source_sha256']==proof['source_sha256']
 for c in q['cases']:
  for z in c['graph_experiment']['records']:
   assert z['fused_guard_regions']
   v=z['CPU_oracle_sample']
   if v is None:continue
   assert c['batch']==1 and z['repeat']==1 and len(v['previous_ids'])==c['block_size'];ids=np.asarray(v['previous_ids']);columns=np.asarray(v['columns']);assert np.all(ids>=0)and np.all(ids<163840)and np.all(columns>=0)and np.all(columns<163840)
   bias=f32(w1[ids]).astype(np.float64)@f32(w2[columns]).astype(np.float64).T;gold=rounded(np.asarray(v['base'],dtype=np.float32)+rounded(bias));actual=np.asarray(v['actual'],dtype=np.float32);assert actual.shape==gold.shape==(c['block_size'],len(columns))and np.isfinite(actual).all();error=float(np.linalg.norm(actual-gold)/max(float(np.linalg.norm(gold)),1e-12));assert error<.005,('CPU_FP64_bias_oracle',device,c['block_size'],error);oracle_errors.append(error)
assert len(oracle_errors)==256
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'pureHIP_Markov_CPU_FP64_samples':len(oracle_errors),'CPU_FP64_max_relative_L2':max(oracle_errors),'all_fused_proposal_IDs_equal_original_Torch':True,'phase_records':phase_records,'graph_records':graph_records,'graph_raw_files_checked':graph_raw,'phase_summary':phase_summary,'graph_summary':graph_summary,'producer':rows[0],'records':record_count,'raw_logit_files_independently_checked':rawfiles,'cache_vs_fullcontext_IDs_exact_allcases':cache_same,'normal8GPU_cleanup':True,'summary':summary,'qualified_for_online_target_speedup':False,'scope':'Pinned3c5bac DSpark using two old3e0 firstblock feature fixtures, cache initialization excluded and recorded separately. Each K is independently initialized; no advancing-target or accepted-prefix transaction claim. K16 is extrapolation, goldfuture may censor at15. These are draft costs, not per-user useful target tokens/s.'};out.write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items()if k not in['summary','phase_summary','graph_summary']}))
