import ctypes,hashlib,json,os,pathlib,re,time
import numpy as np
import resource
import torch
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
torch.set_num_threads(1)
r=pathlib.Path('/refactor');model=pathlib.Path('/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
path=r/'results'/f'legacy-deep-eos-r3421-{tag}.json'
lib=ctypes.CDLL(str(r/'candidates/native-engine-allselected-r815/libmodern_native_engine_allselected_r815.so'));lib.modern_native_engine_error.restype=ctypes.c_char_p
lib.modern_deepseek_sinkhorn_r89_counts.argtypes=[ctypes.c_void_p]
for n in ['modern_deep_latent_fused_calls_r681','modern_deep_latent_fallback_calls_r681']:
 f=getattr(lib,n);f.argtypes=[ctypes.c_int];f.restype=ctypes.c_uint64
lib.modern_deep_allselected_set_r813.argtypes=[ctypes.c_int];assert lib.modern_deep_allselected_set_r813(99)==1 and lib.modern_deep_allselected_set_r813(1)==0
lib.modern_deep_allselected_calls_r813.argtypes=[ctypes.c_int]*2;lib.modern_deep_allselected_calls_r813.restype=ctypes.c_uint64
mapped=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if'libinfiniops.so'in line})
assert len(mapped)==1 and'/install/hygon-v4-indexer-ops-r1/lib/'in mapped[0]
assert hashlib.sha256(pathlib.Path(mapped[0]).read_bytes()).hexdigest()=='8865b8d8e67832884911199d1a667856aadb2b438f5287d17fbaff1f8a3b85db'
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
 report['scope']='Full43 ordinary8thread Engine with qualified post/scatter and only packed latent KV gather/Cast. Six frozen knownanswers and twoB8 input512/output512 frozen IDs/full first4 logits/all8rank, repeats and owner shutdown required. Fixed horizon excludes loading/tokenizer/HTTP; no hardware maximum claim.'
 report['fused_scope']='Only latent attention data movement packed; pertoken F32 numerical order unchanged; retains qualified post/Sinkhorn/scatter.'
 report['sdk_sha256']=hashlib.sha256(pathlib.Path(mapped[0]).read_bytes()).hexdigest()
 report['library_sha256']={name:hashlib.sha256((r/'candidates/native-engine-allselected-r815'/name).read_bytes()).hexdigest()for name in ['libmodern_native_engine_allselected_r815.so']}
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
  report['finite_'+epoch+'_passed']=True;path.write_text(json.dumps(report,indent=2))

 batch16_references={}
 prior564=json.loads((r/'results/native-engine-geometry-r564-910373.370.json').read_text());check564=prior564['batch16_first4_checks'][0];assert check564['phase']=='beijing_repeat1'and check564['step']==0 and not check564['matched']
 prior_raw=(r/check564['raw'].removeprefix('/refactor/')).read_bytes();assert hashlib.sha256(prior_raw).hexdigest()==check564['sha256'];report['previous_cross_batch_logit_trial_failed']='910373.370';report['cross_batch_single_logit_bitidentity_qualified']=False

 def run_batch_chat(epoch):
  records=[]
  for case in chat_payload['cases']:
   prompt=case['input_ids'];assert 0<len(prompt)<=1024
   for repeat in [1,2]:
    phase=case['label']+'_repeat'+str(repeat);old=next(v for v in baseline['cases']if v['phase']==phase);check(lib.modern_native_engine_reset(ctx));generated=[];hashes=[]
    for step in range(256):
     past=0 if step==0 else len(prompt)+step-1;width=len(prompt)if step==0 else 1;chunk=next(v for v in [8,4,2,1]if v*width<=4096)if step==0 else 16;parts=[];ids_out=[]
     for offset in range(0,16,chunk):
      ids=prompt*chunk if step==0 else[generated[-1]]*chunk;code,logits,emitted=forward(ids,list(range(past,past+width))*chunk,list(range(offset,offset+chunk)),[past]*chunk,[past+width]*chunk,[i*width for i in range(chunk+1)]);check(code)
      values=torch.from_numpy(logits).view(torch.bfloat16);assert torch.isfinite(values).all()and torch.equal(values.argmax(-1),torch.from_numpy(emitted));ids_out.extend(map(int,emitted));parts.append(logits)
      if step<4:check(lib.modern_native_engine_rank_logits_equal(ctx))
     generated.append(ids_out[0]);duplicate_exact=ids_out==[ids_out[0]]*16;old_exact=generated==old['generated_ids'][:len(generated)]
     if not duplicate_exact or not old_exact:
      raw=np.concatenate(parts).tobytes();p=r/'results'/f'deep-allselected-r816-{tag}-{epoch}-{phase}-step{step}-failure-bf16.bin';p.write_bytes(raw);report['batch16_failure']={'epoch':epoch,'phase':phase,'step':step,'all_row_ids':ids_out,'generated':generated,'raw':str(p),'sha256':hashlib.sha256(raw).hexdigest()};path.write_text(json.dumps(report,indent=2))
     assert duplicate_exact and old_exact,('B16finiteanswer mismatch',epoch,phase,step)
     if step<4:
      raw=np.concatenate(parts).tobytes();digest=hashlib.sha256(raw).hexdigest();hashes.append(digest);p=r/'results'/f'deep-allselected-r816-{tag}-{epoch}-{phase}-step{step}-B16-bf16.bin';p.write_bytes(raw);stride=cfg['vocab_size']*2
      assert len(raw)==16*stride
      old_exact=all(hashlib.sha256(raw[i*stride:(i+1)*stride]).hexdigest()==old['first4_logits_sha256'][step]for i in range(16))
      # Matrix shape changes are measured independently of repeated same-shape
      # correctness. This trial never qualifies failed564 cross-batch identity.
      actual=torch.from_numpy(np.frombuffer(raw,dtype=np.uint16).copy()).view(torch.bfloat16).float().reshape(16,cfg['vocab_size'])
      solo_raw=(r/'results'/f'deep-finite-r578-{tag}-{epoch}-{phase}-step{step}-B1-bf16.bin').read_bytes();assert hashlib.sha256(solo_raw).hexdigest()==new_scalar_references[case['label']][1][step]
      solo=torch.from_numpy(np.frombuffer(solo_raw,dtype=np.uint16).copy()).view(torch.bfloat16).float();den=float(solo.square().mean().sqrt());error=float((actual-solo).square().mean().sqrt())/den
      cross={'epoch':epoch,'phase':phase,'step':step,'raw':str(p),'sha256':digest,'old_per_row_sha256':old['first4_logits_sha256'][step],'cross_batch_single_logits_exact':old_exact,'cross_batch_relative_RMS':error,'cross_batch_max_abs':float((actual-solo).abs().max()),'all16_duplicate_logits_exact':len({hashlib.sha256(raw[i*stride:(i+1)*stride]).hexdigest()for i in range(16)})==1}
      report.setdefault('batch16_first4_checks',[]).append(cross);path.write_text(json.dumps(report,indent=2)) # NativeBF16 reduce-order differences are retained as numeric diagnostics.
      if case['label']=='beijing'and step==0:report['old564_B16_prefill_raw_exact']=raw==prior_raw
      if case['label']in batch16_references:assert digest==batch16_references[case['label']]['hashes'][step],('B16samegeometry repeatedlogits differ',epoch,phase,step)
     if generated[-1]==cfg['eos_token_id']:break
    if case['label']not in batch16_references:batch16_references[case['label']]={'generated_ids':generated,'hashes':hashes,'reference_phase':phase,'reference_epoch':epoch}
    stable=generated==batch16_references[case['label']]['generated_ids']and hashes==batch16_references[case['label']]['hashes'];assert stable;report['batch16_references']=batch16_references
    row={'same_geometry_IDs_logits_repeated_exact':stable,'phase':phase,'batch':16,'generated_ids':generated,'all16_duplicate_generated_IDs_exact':True,'all16_duplicate_logits_exact':all(v['all16_duplicate_logits_exact']for v in report['batch16_first4_checks']if v['epoch']==epoch and v['phase']==phase),'first4_logits_sha256':hashes,'all16_rows_match_old_single_answer_logits':all(v['cross_batch_single_logits_exact']for v in report['batch16_first4_checks']if v['epoch']==epoch and v['phase']==phase),'stopped_at_eos':generated[-1]==cfg['eos_token_id'],'old_known_answer_IDs_exact':generated==old['generated_ids'],'blas_policy':policy_snapshot()};records.append(row);report.setdefault('batch16_chat_by_epoch',{})[epoch]=records;path.write_text(json.dumps(report,indent=2));assert row['stopped_at_eos']and row['old_known_answer_IDs_exact'];print('DEEP_GEOMETRY_R564_B16_QA',epoch,phase,flush=True)
  assert len(records)==6
  if epoch=='after':assert all(all(x[k]==y[k]for k in ['phase','generated_ids','first4_logits_sha256','stopped_at_eos'])for x,y in zip(records,report['batch16_chat_by_epoch']['before'])), 'B16finite cross-epoch differs'

 run_chat('before')
 run_batch_chat('before')
 report['chat_cases']=report['chat_cases_by_epoch']['before']

 import statistics
 reference_reports=[json.loads((r/'results'/f'deepseek-b8-reference-r114-910373.280-rank{k}.json').read_text())for k in range(8)]
 assert all(q['completed']and q['worker_shutdown_completed']and len(q['cases'])==4 for q in reference_reports)
 reference=next(c for c in reference_reports[0]['cases']if c['phase']=='context512_b8_repeat1')
 for q in reference_reports:
  for c in q['cases'][2:]:assert c['generated_ids']==reference['generated_ids']and c['first4_logits_sha256']==reference['first4_logits_sha256']
 payload=json.loads((r/'control'/'deepseek-b8-reference-r114-control-910373.280-ids.json').read_text());base_prompts=payload['prompts']*2
 assert len(base_prompts)==8 and all(len(v)==512 for v in base_prompts)
 report['scope']='Complete43 ordinary8thread Engine, qualified registerpost/Sinkhorn/scatter, new latent KV packing only. SameB8 cache8x2048, six finite answers, two full512 ID/first4 logits/all8worker comparisons to frozen original and owner shutdown. No throughput adoption from component alone; excludes loading/tokenizer/HTTP.'
 report['scope']='Candidate482 over completed440: replace bankconcat/IndexSelect BF16 temporary/Cast with component-qualified374 strided two-bank gather and exact widening; allperqueryF32math/cache/SDK8865/CPUmetadata retained. Same verified8ownedInfiniOpsBLAS handles initialized CUBLAS_ATOMICS_NOT_ALLOWED via unchanged271preload. Constructorget/set/get and TID/caller counts verified aftereach6finite/twofull512cases, not claim all proprietarykernel internals are atomicfree. Keep rawfirst4 and fullIDs/frozenchecks beforecomparison, all8rank/greedy/invalidinput/repeat/shutdown. Test BLAS-policy hypothesis after390instrumentation didnotreproduce356failure; no stability/hardwaremax promotion evenifthisprocesspasses. Earlier304/356failure remains.'
 report['scope']='System522 unchanged482gatherbinary/SDK8865/verified271BLASpolicy, cache16rowsx2048. Freshprocess full43 B8same-nodebaseline twice andB16 twice, prefill subbatches8 keep4096token cap; decode16. AlloriginalB8tiledfullIDs/first4full-logits/all8worker/greedy/raw/repeat retained. Sixfinitebeforeafter/invalidcachechecks/shutdown. Old304/356failures notdeclaredfixed; no hardwaremax/HTTP claim.'
 report['cache_rows']=16;canonical={}
 failed=json.loads((r/'results/native-engine-batch-r522-910373.365.json').read_text());old16=next(v for v in failed['cases']if v['phase']=='engine_b16_repeat1');assert not old16['baseline_ids_exact']and old16['baseline_first4_logits_exact']
 for step in range(4):
  canonical[step]=(r/'results'/f'deep-batch-r522-910373.365-engine_b8_repeat1-step{step}-logits-bf16.bin').read_bytes();assert hashlib.sha256(canonical[step]).hexdigest()==reference['first4_logits_sha256'][step]
 report['scope']='Separate578batchnumericaldiagnostic after564cross-B1/B16logitidentityFAILED. Same482DSO/8865SDK/271/cache16, originalsingle-row6answersandfullfirst4remainstrictbeforeafter; actualB16sixanswersmustmatcholdIDs/EOS/all16duplicates andsame-B16first4repeats/cross-epoch. Cross-B1/B16logitdelta reportednotacceptedasequivalent; firstB16beijingstep0mustmatchimmutable564rawinfreshprocess. SameB16full512twicerequiresfrozenfailed522 IDs/rawfirst4. All8workers/policy/invalidinput/shutdown unchanged. No broadquality,oldcrossbatchbitcompatibility,allhistoricalstability ormaxTPS claim.'
 report['old_B8_full_ID_compatibility_qualified']=False
 report['reference_failed_trial']='910373.365'

 report['precision_variant']='F32_KVsplit64_latent_nativeBF16_TP'
 report['scope']='686new685latentkernel variant: oldscalar/all16 knownanswerIDs/EOS andownfirst4repeat/crossepoch strict; B8/B16 ownfull512 two repeats andrawfirst4/all8/shutdown. Oldscalar/482/522syntheticlogits diagnostic. T<8/largeprefillfallback; shortprefill8..64mayuse new math. SameSDK8865/271/cache16/nativeBF16TP, noTree/FP32reduce change. 98component/240invalid/F64sampleaudited beforemodel. No broadquality/maximumclaim.'
 report['scope']='816full43 ordinary8threadEngine over815:813canonicalallselected shortcontext M8..64decode, samekeycompressor/cache/SDK8865/681attention/BF16TP. Selectionset equal but attentionFP32reductionorder changed. Oldscalar/all16knownanswerIDs/EOS strict, ownB8/B16full512twice/rawfirst4/all8/fused&allselectedcounters/owner/shutdown. Old686longIDs diagnostic. No broadquality/HTTP/maxclaim.'

 natural=json.loads((r/'results'/f'deep-serving-cases-r1196-{tag}.json').read_text());eos=set(natural['eos_token_ids']);assert len(natural['cases'])==2
 report.update(scope='Historical r815 model/SDK8865, actual long-form prompts480..512 and per-row firstEOS retirement at max512, B8/B16. Native+harness, excludes load/tokenizer/reset/HTTP. OwnfullIDs/fullraw hashes eachstep, archived9raw checkpoints/all8greedy and beforeafterQA/normalclose required.',natural_cases=[],natural_input_provenance=natural)
 refs={}
 for batch in [8,16]:
  for repeat in [1,2]:
   phase=f'b{batch}-natural-r{repeat}';check(lib.modern_native_engine_reset(ctx));prompts=[natural['cases'][i%2]['input_ids']for i in range(batch)];generated=[[]for _ in range(batch)];times=[[]for _ in range(batch)];active=list(range(batch));hashes={};tick=time.perf_counter()
   for step in range(512):
    chunk=8 if step==0 else len(active);selected=[];raws=[]
    for off in range(0,len(active),chunk):
     rows=active[off:off+chunk];inputs=[prompts[i]if step==0 else[generated[i][-1]]for i in rows];widths=[len(v)for v in inputs];pasts=[0 if step==0 else len(prompts[i])+len(generated[i])-1 for i in rows];flat=sum(inputs,[]);positions=sum([list(range(p,p+w))for p,w in zip(pasts,widths)],[]);offsets=[0]+np.cumsum(widths).tolist();code,logits,emitted=forward(flat,positions,rows,pasts,[p+w for p,w in zip(pasts,widths)],offsets);check(code)
     values=(logits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(values).all()and np.array_equal(np.argmax(values,-1),emitted)
     if step in[0,1,2,3,63,127,255,383,511]:check(lib.modern_native_engine_rank_logits_equal(ctx))
     selected.extend(map(int,emitted));raws.append(logits);elapsed=time.perf_counter()-tick
     for i,tok in zip(rows,emitted):generated[i].append(int(tok));times[i].append(elapsed)
    raw=np.concatenate(raws).tobytes();hashes[str(step)]={'rows':active.copy(),'sha256':hashlib.sha256(raw).hexdigest()}
    if step in[0,1,2,3,63,127,255,383,511]:
     file=f'legacy-deep-eos-r3421-{tag}-{phase}-step{step}.bin';p=r/'results'/file;assert not p.exists();p.write_bytes(raw);report.setdefault('raw_checkpoints',[]).append({'file':file,'bytes':len(raw),'shape':list(np.concatenate(raws).shape),'sha256':hashes[str(step)]['sha256'],'output_ids':selected,'rows':active.copy()})
    active=[i for i in active if generated[i][-1]not in eos]
    if step%64==0:report['active_generation']={'phase':phase,'step':step,'active':len(active),'elapsed':time.perf_counter()-tick};path.write_text(json.dumps(report,indent=2));print('LEGACY_DEEP_EOS_R3421',phase,step,len(active),flush=True)
    if not active:break
   end=max(t[-1]for t in times);first=min(t[0]for t in times);useful=sum(len(v)-int(v[-1]in eos)for v in generated);decode=sum(max(0,len(v)-int(v[-1]in eos)-1)for v in generated);value=(generated,hashes);assert value==refs.setdefault(batch,value),('legacyEOS_ID_raw_repeat',batch,repeat)
   assert all(not any(t in eos for t in v[:-1])for v in generated)
   row={'phase':phase,'batch':batch,'repeat':repeat,'generated_ids':generated,'raw_hashes_each_step':hashes,'step_times_per_row':times,'input_lengths':[len(v)for v in prompts],'useful_output_tokens':useful,'decode_output_tokens':decode,'aggregate_useful_decode_tok_s':decode/(end-first),'including_prefill_useful_tok_s':useful/end,'decode_tok_s_per_all8cards':decode/(end-first)/8,'elapsed_seconds':end,'decode_elapsed_seconds':end-first,'all_requests_firstEOS_or_limit':True,'own_IDs_fullraw_repeat_exact':True,'all8_rank_checkpoint_logits_exact':True,'example_text':[tokenizer.decode(generated[i],skip_special_tokens=False)for i in range(min(2,batch))]};report['natural_cases'].append(row);path.write_text(json.dumps(report,ensure_ascii=False,indent=2));print('LEGACY_DEEP_EOS_R3421_RESULT',phase,row['aggregate_useful_decode_tok_s'],row['including_prefill_useful_tok_s'],flush=True)
 run_chat('after');run_batch_chat('after');report['blas_policy_before_destroy']=policy_snapshot();report['qualification_before_destroy_passed']=True

except BaseException as failure:
 report['error']=repr(failure);raise
finally:
 try:
  check(lib.modern_native_engine_destroy(ctx));report['engine_worker_shutdown_completed']=True;report['completed']=bool(report.get('qualification_before_destroy_passed'))
 except BaseException as close:report['completed']=False;report['shutdown_error']=repr(close);raise
 finally:path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
