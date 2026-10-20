from model_affinity_check_r2628 import verify as verify_owner_affinity
import ctypes,hashlib,json,os,pathlib,re,time
import numpy as np
import resource
import torch
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
torch.set_num_threads(1)
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
path=r/'results'/f'deep-wide-r3457-{tag}.json'
lib=ctypes.CDLL(str(r/'candidates/deep-wide-3e0d89fe-r3443/libmodern_deep_wide_r3443.so'));lib.modern_native_engine_error.restype=ctypes.c_char_p
lib.modern_deepseek_sinkhorn_r89_counts.argtypes=[ctypes.c_void_p]
for n in ['modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_uint64
lib.modern_deep_allselected_set_r813.argtypes=[ctypes.c_int];assert lib.modern_deep_allselected_set_r813(99)==1 and lib.modern_deep_allselected_set_r813(1)==0
lib.modern_deep_allselected_calls_r813.argtypes=[ctypes.c_int]*2;lib.modern_deep_allselected_calls_r813.restype=ctypes.c_uint64
mapped=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfiniops.so'in line})
assert len(mapped)==1 and'/install/hygon-deep-cast64-r3315/lib/'in mapped[0]
assert hashlib.sha256(pathlib.Path(mapped[0]).read_bytes()).hexdigest()==json.loads((r/'results/cast64-sdk-r3375-Deep-load-build.json').read_text())['library_sha256']
lib.modern_native_engine_create.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_size_t];lib.modern_native_engine_create.restype=ctypes.c_void_p
lib.modern_native_engine_keys.argtypes=[ctypes.c_void_p];lib.modern_native_engine_keys.restype=ctypes.c_char_p
for name in ['finish','reset','destroy','rank_logits_equal']:getattr(lib,'modern_native_engine_'+name).argtypes=[ctypes.c_void_p]
lib.modern_native_engine_load.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
lib.modern_native_engine_forward.argtypes=[ctypes.c_void_p]+[ctypes.c_size_t]*2+[ctypes.c_void_p]*8
policy_lib=ctypes.CDLL(None);policy_lib.modern_blas_policy_r243_json.restype=ctypes.c_char_p
def policy_snapshot():
 value=json.loads(policy_lib.modern_blas_policy_r243_json().decode())
 assert value['failures']==0 and all(v['created']==v['verified']for v in value['owners'])
 assert len({v['tid']for v in value['owners']if'libinfiniops.so'in v['caller']})==8,('MissingownedBLASpolicyhandles',value)
 return value

def check(code):
 if code:raise RuntimeError(lib.modern_native_engine_error().decode())
class MallInfo2(ctypes.Structure):
 _fields_=[(name,ctypes.c_size_t)for name in ['arena','ordblks','smblks','hblks','hblkhd','usmblks','fsmblks','uordblks','fordblks','keepcost']]
host_libc=ctypes.CDLL(None);host_libc.mallinfo2.restype=MallInfo2
def memory_snapshot(label):
 heap=host_libc.mallinfo2();values={v.split(':',1)[0]:v.split(':',1)[1].strip()for v in pathlib.Path('/proc/self/status').read_text().splitlines()if v.startswith(('VmRSS:','VmHWM:','VmSize:'))}
 counts=(ctypes.c_size_t*8)();assert lib.modern_deepseek_sinkhorn_r89_counts(counts)==0
 values.update(label=label,ru_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,active_heap_bytes=int(heap.uordblks+heap.hblkhd),sinkhorn_calls_by_device=list(counts))
 print('NATIVE_ENGINE_MHC_R89_MEMORY',json.dumps(values),flush=True);return values

cfg=json.loads((model/'config.json').read_text());mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];start=time.perf_counter()
ctx=lib.modern_native_engine_create(json.dumps(cfg).encode(),16,2048)
if not ctx:raise RuntimeError(lib.modern_native_engine_error().decode())
report={'completed':False,'step':tag}
try:
 keys=set(json.loads(lib.modern_native_engine_keys(ctx)));assert len(keys)==67569 and keys<=set(mapping)
 report={'fused_sinkhorn_variant':1,'fused_scope':'Only mHC normalization is fused: pairwise4 row sums, sequential4 column sums, FP32 division. Actual all86 B4 parameter sets byte-exact to original on8 workers; complete model prefill/quality/performance and host-memory behavior require this run.','scope':'Actual normal C++ InferEngine/RankWorker/ModelFactory with8 owned worker threads on8DCUs; typed native sparse model/cache adapter. True raw weights loaded once in Python and broadcast by Engine loader. Three reference-framed known-answer cases repeated2, sampled IDs/logits compared to previously qualified native model. This is not public Python/HTTP, broad quality, per-request cache recycling or peakTPS qualification.','step':tag,'main_layers':43,'keys':len(keys),'cases':[],'invalid_checks':[],'completed':False}
 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'
 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'
 report['fused_scope']='Only latent attention data movement packed; pertoken F32 numerical order unchanged; retains qualified post/Sinkhorn/scatter.'
 report['sdk_sha256']=hashlib.sha256(pathlib.Path(mapped[0]).read_bytes()).hexdigest()
 report['library_sha256']=hashlib.sha256((r/'candidates/deep-wide-3e0d89fe-r3443/libmodern_deep_wide_r3443.so').read_bytes()).hexdigest();assert report['library_sha256']==json.loads((r/'results/deep-wide-r3447-build.json').read_text())['library_sha256']
 report['LM_source_commit']='3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d'
 report['latest_stack_scope']='2130 complete43Deep migrated1f598/Core9cdb/Ops0739/Runtime2117/model2112. Preserve frozen816 QA/fullIDs/raw checkpoint/CPUargmax/naturalEOS/actualowners/normalcleanup/Slurm0. Oldasync fence candidate not enabled. No oldperformance or15 inheritance.'
 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'
 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'
 grouped={}
 for key in keys:grouped.setdefault(mapping[key],[]).append(key)
 pending=[];pending_bytes=loaded=0
 def flush():
  global pending_bytes,loaded
  if not pending:return
  descriptors=[{'name':name,'pointer':value.data_ptr(),'shape':list(value.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.int8:2,torch.int64:3}[value.dtype]}for name,value in pending]
  check(lib.modern_native_engine_load(ctx,json.dumps(descriptors).encode()));loaded+=len(pending);pending.clear();pending_bytes=0
  print('NATIVE_ENGINE_LOAD',loaded,len(keys),time.perf_counter()-start,flush=True)
 for filename,names in sorted(grouped.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as f:
   for name in sorted(names):
    value=f.get_tensor(name).contiguous();pending.append((name,value));pending_bytes+=value.numel()*value.element_size()
    if len(pending)>=1024 or pending_bytes>=512*1024*1024:flush()
   flush()
 loaded_paths={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfinilm_runtime.so'in v or'_infinilm.cpython-'in v};assert any('/candidates/rebased-3e0d89fe-r2842/Common/Deep/InfiniLM/'in x and'libinfinilm_runtime.so'in x for x in loaded_paths),loaded_paths;assert all('/candidates/rebased-3e0d89fe-r2842/Common/Deep/InfiniLM/'in x for x in loaded_paths),loaded_paths;report['actual_new_runtime_paths']=sorted(loaded_paths)
 check(lib.modern_native_engine_finish(ctx));report['phase_memory']=[memory_snapshot('weights_ready')];report['loading_seconds']=time.perf_counter()-start;path.write_text(json.dumps(report,indent=2));print('NATIVE_ENGINE_WEIGHTS_READY',loaded,report['loading_seconds'],flush=True)
 payload=json.loads((r/'control/deep-chat-routed-r20-910373.162.json').read_text());baseline=json.loads((r/'results/modern-deepseek-chat-routed-r20-910373.162-rank0.json').read_text())
 tokenizer=PreTrainedTokenizerFast.from_pretrained(model,local_files_only=True)
 def forward(ids,positions,rows,past,total,offsets):
  values=[np.ascontiguousarray(x,dtype=np.int64 if i>=4 else np.int32)for i,x in enumerate([rows,past,total,offsets,ids,positions])]
  logits=np.full((len(rows),cfg['vocab_size']),0x7fc0,dtype=np.uint16);emitted=np.full(len(rows),-999,dtype=np.int64)
  code=lib.modern_native_engine_forward(ctx,len(ids),len(rows),*[ctypes.c_void_p(v.ctypes.data)for v in values],ctypes.c_void_p(logits.ctypes.data),ctypes.c_void_p(emitted.ctypes.data))
  return code,logits,emitted
 chat_payload=payload
 new_scalar_references={}
 report['reference_numeric_policy']='IntentionalF32attentionreductionorder: oldknownanswerIDs/EOS strict, oldlogits diagnostic, owngeometry rawfirst4 repeats/crossepoch strict. T<8 fallsback, but shortprefill8..64mayuse new math.'
 def run_chat(epoch):
  saved_cases=report['cases'];report['cases']=[]
  for case in chat_payload['cases']:
   for repeat in [1,2]:
    report.setdefault('phase_memory',[]).append(memory_snapshot('chat_before_reset'));check(lib.modern_native_engine_reset(ctx));report['phase_memory'].append(memory_snapshot('chat_after_reset'));phase=case['label']+'_repeat'+str(repeat);prompt=case['input_ids'];generated=[];hashes=[];times=[];begin=time.perf_counter()
    for step in range(256):
     ids=prompt if not step else[generated[-1]];past=0 if not step else len(prompt)+step-1;positions=list(range(past,past+len(ids)))
     tick=time.perf_counter();code,logits,emitted=forward(ids,positions,[0],[past],[past+len(ids)],[0,len(ids)]);check(code);times.append(time.perf_counter()-tick)
     values=torch.from_numpy(logits).view(torch.bfloat16);assert torch.isfinite(values).all()and int(values.argmax(-1).item())==int(emitted[0]),'Normal Engine greedy IDs differ from first-argmax semantics'
     if step<4:
      check(lib.modern_native_engine_rank_logits_equal(ctx));hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
      (r/'results'/f'deep-finite-r578-{tag}-{epoch}-{phase}-step{step}-B1-bf16.bin').write_bytes(logits.tobytes())
      old_case=next(c for c in baseline['cases']if c['phase']==phase)
      if hashes[-1]!=old_case['first4_logits_sha256'][step]:(r/'results'/f'native-engine-allselected-r815-{tag}-{phase}-step{step}-changed-bf16.bin').write_bytes(logits.tobytes())
     generated.append(int(emitted[0]))
     if step==0:
      badcode,badlogits,badids=forward([cfg['vocab_size']],[len(prompt)],[0],[len(prompt)],[len(prompt)+1],[0,1]);assert badcode and(badlogits==0x7fc0).all()and(badids==-999).all();report['invalid_checks'].append({'phase':phase,'case':'out_of_vocabulary_after_prefill','rejected_before_output_and_cache_mutation':True})
     if generated[-1]==cfg['eos_token_id']:break
    old=next(c for c in baseline['cases']if c['phase']==phase)
    row={'epoch':epoch,'phase':phase,'generated_ids':generated,'first4_logits_sha256':hashes,'all8_worker_full_logits_exact_first4':True,'baseline_ids_exact':generated==old['generated_ids'],'baseline_first4_logits_exact':hashes==old['first4_logits_sha256'],'stopped_at_eos':generated[-1]==cfg['eos_token_id'],'answer':tokenizer.decode(generated,skip_special_tokens=True).strip(),'elapsed_seconds':time.perf_counter()-begin,'step_seconds':times}
    row['blas_policy']=policy_snapshot();report['cases'].append(row);path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('NATIVE_ENGINE_CHAT_CASE',json.dumps(row,ensure_ascii=False),flush=True)
    value=(generated,hashes);key=case['label']
    if key not in new_scalar_references:new_scalar_references[key]=value
    assert row['baseline_ids_exact']and row['stopped_at_eos']and value==new_scalar_references[key],('685scalar knownanswer/repeat',epoch,phase)


  chats=report['cases'];assert len(chats)==6;report['cases']=saved_cases;report.setdefault('chat_cases_by_epoch',{})[epoch]=chats
  if epoch=='after':
   before=report['chat_cases_by_epoch']['before'];assert all(x['generated_ids']==y['generated_ids']and x['first4_logits_sha256']==y['first4_logits_sha256']for x,y in zip(before,chats))
  frozen=json.loads((r/'results/native-engine-allselected-r816-910373.442.json').read_text());assert frozen['completed']and frozen['engine_worker_shutdown_completed']
  for actual,gold in zip(chats,frozen['chat_cases_by_epoch'][epoch]):assert actual['generated_ids']==gold['generated_ids']and actual['first4_logits_sha256']==gold['first4_logits_sha256'],('1196frozen816QA',epoch,actual['phase'])
  report['finite_'+epoch+'_passed']=True;path.write_text(json.dumps(report,indent=2))


 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'
 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'
 natural_path=r/'results'/f'deep-serving-cases-r2130-{tag}.json';natural=json.loads(natural_path.read_text());report['natural_input_sha256']=hashlib.sha256(natural_path.read_bytes()).hexdigest();report['natural_cases']=[]
 report['scope']='3457 latest3e0 matchedsync native control and stagedF64 diagnostic; frozenB1/raw/invalid/owner gates retained. No B2throughput claim.'

 from deep_wide_protocol_r3457 import run
 run(globals());report['owner_affinity']=verify_owner_affinity(lib);report['blas_policy_before_destroy']=policy_snapshot();report['qualification_before_destroy_passed']=True
except BaseException as failure:
 report['error']=repr(failure);raise
finally:
 try:
  check(lib.modern_native_engine_destroy(ctx));report['engine_worker_shutdown_completed']=True;report['completed']=bool(report.get('qualification_before_destroy_passed'))
 except BaseException as close:report['completed']=False;report['shutdown_error']=repr(close);raise
 finally:path.write_text(json.dumps(report,indent=2))
print('DEEP_WIDE_R3457_COMPLETE',flush=True)
