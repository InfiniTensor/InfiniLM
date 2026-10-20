import base64,collections,ctypes,hashlib,json,os,pathlib,time
import numpy as np
import torch
from safetensors import safe_open
from kimi_pipeline_split_prefill_r646 import SplitPrefillControlR646 as PipelineControl
torch.set_num_threads(1)
# Check actual templated QA lengths before creating/loading any GPU engine.
qa_payload_r669=json.loads((pathlib.Path('/refactor')/'results/kimi-chat-cases-r20.json').read_text())
qa_lengths_r669={label:len(qa_payload_r669['cases'][label]['input_ids'])for label in ['multiply','sort','retrieval']}
assert all(0<n and 8*n<=6144 and n+128<=1024 for n in qa_lengths_r669.values()),qa_lengths_r669
print('KIMI_BATCH669_PREFLIGHT',qa_lengths_r669,flush=True)
space_vocab_r691=pathlib.Path('/shared/Kimi-K3/tiktoken.model').read_bytes()
assert hashlib.sha256(space_vocab_r691).hexdigest()=='b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103'
space_rows_r691=[v.split()for v in space_vocab_r691.splitlines()if v.split()[-1:]==[b'220']]
assert len(space_rows_r691)==1 and base64.b64decode(space_rows_r691[0][0])==b' '

r=pathlib.Path('/refactor');model=pathlib.Path('/shared/Kimi-K3');stage=int(os.environ['SLURM_PROCID']);tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];head=os.environ['KIMI_ENGINE_HEAD'];world_port=18000+(int(os.environ['SLURM_JOB_ID'])*97+int(os.environ['SLURM_STEP_ID']))%4000
control=PipelineControl(stage,head,world_port+7000,os.environ['KIMI_ENGINE_NONCE']);control.collect('before_create',{'stage':stage})
gold834=json.loads((r/'results/kimi-batch-r691-910371.123-stage3.json').read_text());assert gold834['completed']
goldcases834={v['phase']:v for v in gold834['cases']}
import infinilm
free_before_create863=[list(torch.cuda.mem_get_info(device))for device in range(8)];assert all(free>58*1024**3 for free,total in free_before_create863),('863priorjobGPUmemorynotreleased',free_before_create863)
hardware918=[]
for device918 in range(8):
 p918=torch.cuda.get_device_properties(device918);assert ('gfx936'in p918.name or'gfx936'in getattr(p918,'gcnArchName',''))and p918.total_memory>=63*1024**3 and p918.multi_processor_count==80
 with torch.cuda.device(device918):
  x918=torch.arange(2048,device='cuda',dtype=torch.float32);assert int(x918.sum().item())==2047*2048//2;torch.cuda.synchronize();del x918
 hardware918.append({'device':device918,'name':p918.name,'total_memory':p918.total_memory,'multiprocessors':p918.multi_processor_count,'small_device_compute_passed':True})
library=r/'candidates/kimi-token-only-r963/libmodern_kimi_token_only_r963.so';lib=ctypes.CDLL(str(library));lib.modern_kimi_engine_error_r963.restype=ctypes.c_char_p
lib.modern_kimi_engine_create_r963.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_size_t];lib.modern_kimi_engine_create_r963.restype=ctypes.c_void_p
lib.modern_kimi_engine_keys_r963.argtypes=[ctypes.c_void_p];lib.modern_kimi_engine_keys_r963.restype=ctypes.c_char_p
lib.modern_kimi_engine_load_r963.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
for name in ['finish','reset','destroy','rank_logits']:getattr(lib,'modern_kimi_engine_'+name+'_r963').argtypes=[ctypes.c_void_p]
lib.modern_kimi_engine_forward_r963.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]+[ctypes.c_void_p]*9
def check(code):
 if code:raise RuntimeError(lib.modern_kimi_engine_error_r963().decode())
lib.modern_kimi_engine_partition_r963.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]
PP_ranges880=[]
for pp_stage880 in range(4):
 boundary880=(ctypes.c_size_t*2)();check(lib.modern_kimi_engine_partition_r963(pp_stage880,boundary880));PP_ranges880.append(list(boundary880))
assert PP_ranges880==[[0,24],[24,47],[47,71],[71,93]] and [n for a,b in PP_ranges880 for n in range(a,b)]==list(range(93))
cfg=json.loads((model/'config.json').read_text());text=cfg['text_config'];vocab=text['vocab_size'];begin=time.perf_counter();ctx=lib.modern_kimi_engine_create_r963(json.dumps(cfg).encode(),stage,head.encode(),world_port,128)
if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r963().decode())
keys=set(json.loads(lib.modern_kimi_engine_keys_r963(ctx)));mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert keys<=set(mapping)
report={'scope':'Ordinary full93 Kimi TP8PP4 with8owned worker threads per node, private config and224page/48request cache. Same qualified F32 helper/SDK197; deferred native tail feedback plus CPU219 persistent FIFO feedback for global48/micro12 pipeline. Original nativeM4 frozen gate and immutable staged fused22 M12 full IDs/first4 logits,9 phases and44 EOS finite answers before/after. NativeM12 differences remain explicitly recorded; fused arithmetic is not declared bit-identical to native. R295 proves prior223 and280 M12 match staged fused22 exactly, identifying prior wrong-reference failure,32worker shutdown. Timings exclude load/tokenizer/HTTP; fixed512 may include postEOS diagnostics. No dynamic batching or hardware maximum claim.','step':tag,'stage':stage,'cases':[],'chat_cases':[],'cache_pages':516,'max_requests':128,'completed':False,'create_seconds':time.perf_counter()-begin,'parameter_keys':len(keys)}
report['device_memory_before_create']=free_before_create863
report['PP_layer_ranges']=PP_ranges880
report['actual_device_preflight']=hardware918
report['cache_page_formula']='4*N+4'
report['partition_build']='kimi-token-only-r963-build.json'
report['capacity_memory_variant']='863same839cache128/math; prefill4 lowers transient peak after834prefill8 allocationfailure2 onstage2. Oldprefill8 longIDs/raw diagnostic; ownfull512raw/ID repeat strict.'
report['retry_provenance']={'failed_step':'910371.121','reason':'QA prompts are515..535tokens including template, old512assert invalid; no GPUmath or input truncation change; declared sortonly space220 normalization, perrowEOS stop','actual_QA_prompt_lengths':qa_lengths_r669,'prefill_token_cap':6144,'validated_total_sequence_cap':1024}
path=r/'results'/f'kimi-token-only-r964-{tag}-stage{stage}.json';path.write_text(json.dumps(report,indent=2))
grouped=collections.defaultdict(list)
for key in keys:grouped[mapping[key]].append(key)
pending=[];pending_bytes=loaded=0;start=time.perf_counter()
def flush():
 global pending_bytes,loaded
 if not pending:return
 descriptors=[{'name':name,'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]}for name,w in pending]
 check(lib.modern_kimi_engine_load_r963(ctx,json.dumps(descriptors).encode()));loaded+=len(pending);pending.clear();pending_bytes=0;print('KIMI_ENGINE_LOAD',stage,loaded,len(keys),round(time.perf_counter()-start,3),flush=True)
for filename,names in sorted(grouped.items()):
 with safe_open(model/filename,framework='pt',device='cpu')as shard:
  for name in sorted(names):
   w=shard.get_tensor(name)
   if name.endswith('.self_attn.A_log'):w=w[:text['linear_attn_config']['num_heads']]
   w=w.contiguous();pending.append((name,w));pending_bytes+=w.numel()*w.element_size()
   if len(pending)>=2048 or pending_bytes>=512*1024*1024:flush()
  flush()
check(lib.modern_kimi_engine_finish_r963(ctx));assert loaded==len(keys);report['device_memory_after_loading']=[list(torch.cuda.mem_get_info(device))for device in range(8)];report['loading_seconds']=time.perf_counter()-start
paths=pathlib.Path('/proc/self/maps').read_text().splitlines()
sdk=sorted({v.split()[-1]for v in paths if'libinfiniops.so'in v});helper=sorted({v.split()[-1]for v in paths if'libkimi_grouped_prefill.so'in v});assert len(sdk)==len(helper)==1
report.update(library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),sdk_sha256=hashlib.sha256(pathlib.Path(sdk[0]).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(pathlib.Path(helper[0]).read_bytes()).hexdigest());assert report['sdk_sha256']=='6210897fc1f077bc5fc2e10afc9b8b7141fc4c6c608be3f4cb2d896219560104'and report['helper_sha256']=='26cc4b1eb0e8c044c760db7d42a7374d9ebc98ce82591d1ae1e4a1292c4b7d82'
qualification=json.loads((r/'results/kimi-static-sdk-r343-summary.json').read_text());assert qualification['passed']and qualification['all_owner_threads_joined']and qualification['sdk_sha256']==report['sdk_sha256']
counter_lib=ctypes.CDLL(sdk[0]);counter_lib.modern_mxfp4_static_r330_counts.argtypes=[ctypes.POINTER(ctypes.c_ulonglong),ctypes.c_size_t]
lib.modern_kimi_mla_gate_fp32_count_r476.argtypes=[ctypes.c_int];lib.modern_kimi_mla_gate_fp32_count_r476.restype=ctypes.c_uint64
policy_lib=ctypes.CDLL(None);policy_lib.modern_blas_policy_r243_json.restype=ctypes.c_char_p
def counters():
 values=(ctypes.c_ulonglong*8)();assert counter_lib.modern_mxfp4_static_r330_counts(values,8)==0;static=list(values);gate=[lib.modern_kimi_mla_gate_fp32_count_r476(d)for d in range(8)]
 assert min(static)>0 and len(set(static))==1 and min(gate)>0 and len(set(gate))==1
 policy=json.loads(policy_lib.modern_blas_policy_r243_json().decode());assert policy['failures']==0 and all(v['created']==v['verified']for v in policy['owners'])and len({v['tid']for v in policy['owners']if'libinfiniops.so'in v['caller']})==8
 return {'static_W2_calls':static,'FP32_gate_prefill_calls':gate,'BLAS_policy':policy}
report['scope']='Intentional precision variant520: completed476MLAgateF32prefill candidate plus qualified621W2unroll(exactcomponent) and existing21F32groupedprefill. Same full93TP8PP4/cache224/48/G48M12. Original frozen22/oldnative all512 IDs remain strict; oldlogits flags retained diagnostically because gateprecision intentionally changes them. FirstnewM12full512 establishes separate immutable-in-run full first4logit reference; repeatM12 and allG48 serial/pipeline phases must match it.44finite knownanswer IDs/EOS and same-variant rawfirst4 repeat/cross-epoch/duplicate-sort gates plus32owner shutdown. No claim oldBF16logits match or all historicalfailures fixed; no broadquality/HTTP/max claim.'
control.collect('weights_ready',{'stage':stage,'library':report['library_sha256']});path.write_text(json.dumps(report,indent=2));print('KIMI_ENGINE_WEIGHTS_READY',stage,report['loading_seconds'],flush=True)
CHECKPOINTS963={0,1,2,3,63,127,255,383,511}
lib.modern_kimi_engine_output_metrics_r963.argtypes=[ctypes.c_void_p];lib.modern_kimi_engine_output_metrics_r963.restype=ctypes.c_char_p
def output_metrics963():
 raw=lib.modern_kimi_engine_output_metrics_r963(ctx);assert raw;return json.loads(raw.decode())
harness963=collections.defaultdict(lambda:{'logits_buffer_seconds':0.,'tail_validation_seconds':0.,'forward_calls':0,'omitted_logits_calls':0})
late_refs963={};late_checks963=[]
def forward(ids,past,width,grouped,phase,step,offset=0,defer=False,microbatch=0,expected_hashes=None):
 omit963=phase.startswith('tokens_only_')and step not in CHECKPOINTS963
 metrics963=harness963[phase];metrics963['forward_calls']+=1;metrics963['omitted_logits_calls']+=int(omit963)
 batch=len(past);
 if step==0:report['device_memory_before_last_prefill']={'phase':phase,'offset':offset,'batch':batch,'query_width':width,'devices':[list(torch.cuda.mem_get_info(device))for device in range(8)]};path.write_text(json.dumps(report,indent=2))
 flat=np.ascontiguousarray(ids,dtype=np.int64);
 if step==0 and phase.startswith(('reference_','long_pipeline_','tokens_only_')):report.setdefault('actual_input_packet_SHA834',{}).setdefault(phase,{})[microbatch]=hashlib.sha256(flat.tobytes()).hexdigest()
 positions=np.array([p+j for p in past for j in range(width)],dtype=np.int64);rows=np.arange(offset+1,offset+batch+1,dtype=np.int32);pasts=np.asarray(past,dtype=np.int32);total=pasts+width;offsets=np.arange(batch+1,dtype=np.int32)*width;buffer_tick963=time.perf_counter();logits=None if omit963 else np.full((batch,vocab),0x7fc0,dtype=np.uint16);metrics963['logits_buffer_seconds']+=time.perf_counter()-buffer_tick963;emitted=np.full(batch,-999,dtype=np.int64);flags=np.array([-1],dtype=np.int32)
 values=[rows,pasts,total,offsets,flat,positions,logits,emitted,flags];tick=time.perf_counter();check(lib.modern_kimi_engine_forward_r963(ctx,len(flat),batch,int(grouped)|(2 if defer else 0)|(4 if omit963 else 0),*[ctypes.c_void_p(0 if v is None else v.ctypes.data)for v in values]));elapsed=time.perf_counter()-tick
 hashes=[]
 if stage==3:
  if phase.startswith(('batch16_','large')):batch16_raw_parts[offset]=logits.tobytes()
  assert flags[0]==(2 if omit963 else 3)and all(0<=int(v)<vocab for v in emitted)
  if not omit963:
   validation_tick963=time.perf_counter();f32=torch.from_numpy(logits).view(torch.bfloat16).float();assert torch.isfinite(f32).all()and torch.equal(f32.argmax(-1),torch.from_numpy(emitted));metrics963['tail_validation_seconds']+=time.perf_counter()-validation_tick963
  if phase.startswith(('reference_','long_pipeline_','tokens_only_'))and step in CHECKPOINTS963 and step>=4:
   check(lib.modern_kimi_engine_rank_logits_r963(ctx));hash963=[hashlib.sha256(v.tobytes()).hexdigest()for v in logits];key963=(step,offset,batch);raw963=r/'results'/f'kimi-token-only-r964-{tag}-{phase}-q{microbatch}-step{step}-bf16.bin';raw963.write_bytes(logits.tobytes())
   if key963 not in late_refs963:late_refs963[key963]=hash963
   late963={'phase':phase,'step':step,'offset':offset,'packet':microbatch,'batch':batch,'per_row_SHA256':hash963,'argmax_IDs':emitted.tolist(),'all8_tail_rank_logits_exact':True,'baseline_checkpoint_bits_exact':hash963==late_refs963[key963],'raw':str(raw963),'raw_sha256':hashlib.sha256(logits.tobytes()).hexdigest()};late_checks963.append(late963);report['late_checkpoint_checks']=late_checks963;path.write_text(json.dumps(report,indent=2));assert late963['baseline_checkpoint_bits_exact'],('963latecheckpointdiffers',phase,step,offset)
  if step<4:
   check(lib.modern_kimi_engine_rank_logits_r963(ctx));hashes=[hashlib.sha256(v.tobytes()).hexdigest()for v in logits]
   if True:(r/'results'/f'kimi-token-only-r964-{tag}-{phase}-q{microbatch}-step{step}-bf16.bin').write_bytes(logits.tobytes())
 elif stage==0:assert flags[0]==(0 if defer else 2)
 else:assert flags[0]==0
 if defer:return emitted.tolist()if stage==3 else None,hashes,elapsed
 gathered=control.collect(phase+':'+str(step)+':'+str(microbatch),{'ids':emitted.tolist()if flags[0]&2 else None,'hashes':hashes,'elapsed_seconds':elapsed});assert gathered[0]['ids']==gathered[3]['ids'];next_ids=gathered[3]['ids'];assert all(0<=v<vocab for v in next_ids)
 return next_ids,gathered[3]['hashes'],max(v['elapsed_seconds']for v in gathered)

old_payload=json.loads((r/'results/modern-kimi-capacity-real93-910376.33-rendezvous.json').read_text());old_case=old_payload['cases']['b4_in512_out512'];prompts=old_case.get('input_ids_per_request',[old_case['input_ids']]*4);assert len(prompts)==4 and all(len(p)==512 for p in prompts)
frozen=json.loads((r/'results/modern-kimi-capacity-real93-long-summary.json').read_text())['cases']['queued_long_b4_baseline_repeat1'];frozen_hashes=[[]for _ in range(4)]
for step in range(4):
 raw=(r/'results'/f'kimi-capacity-real93-910376.33-queued_long_b4_baseline_repeat1-step{step}-logits-bf16.bin').read_bytes();assert len(raw)==4*vocab*2
 for row in range(4):frozen_hashes[row].append(hashlib.sha256(raw[row*vocab*2:(row+1)*vocab*2]).hexdigest())

prior=json.loads((r/'results/modern-kimi-pipeline-fused-reset-r22-910375.79-summary.json').read_text());assert prior['completed'] and prior['all16_finite_answers_passed']
native_prior=json.loads((r/'results/modern-kimi-pipeline-m12-trim-r19-summary.json').read_text());assert native_prior['completed'];native12=next(c for c in native_prior['phases']if c['phase']=='reference_b12_m12_r1')
report['reference_variant']='intentional_gateF32_grouped21_W2_621';report['old_reference_variant']='frozen_staged_fused_F32_r22';report['frozen_reference_sha256']=hashlib.sha256((r/'results/modern-kimi-pipeline-fused-reset-r22-910375.79-summary.json').read_bytes()).hexdigest();report['native_reference_sha256']=hashlib.sha256((r/'results/modern-kimi-pipeline-m12-trim-r19-summary.json').read_bytes()).hexdigest()
frozen12=next(c for c in prior['phases']if c['phase']=='reference_b12_m12_r1');assert len(frozen12['generated_ids'])==12 and all(len(v)==512 for v in frozen12['generated_ids'])
report['input_prompts_sha256']=hashlib.sha256(json.dumps(prompts).encode()).hexdigest();report['distinct_input_prompts']=len({tuple(v)for v in prompts});assert report['input_prompts_sha256']==prior['input_prompts_sha256']
new_finite={};new_references={}
qualified543=json.loads((r/'results/kimi-precision-r543-910375.129-stage3.json').read_text());assert qualified543['completed']
qualified_cases={v['phase']:v for v in qualified543['cases']};qualified_chats={(v['epoch'],v['label'],v['repeat']):v for v in qualified543['chat_cases']}

# The previous strict old-BF16 trial520 FAILED and remains failed. This is a
# separate numerical-variant experiment, never an old-bitidentity promotion.
failed=[json.loads((r/'results'/f'kimi-gate-perf-r520-910375.127-stage{s}.json').read_text())for s in range(4)]
failed_rows=[v['cases'][0]for v in failed]
assert all(len(v['chat_cases'])==22 and all(c['known_answer_ids_exact']and c['same_variant_first4_repeat_exact']for c in v['chat_cases'])for v in failed)
assert all(v['phase']=='reference_b4_m4'and not v['frozen_same_shape_ids_exact']and v['generated_ids']==failed_rows[0]['generated_ids']and v['per_row_first4_logits_sha256']==failed_rows[0]['per_row_first4_logits_sha256']for v in failed_rows)
new_references[4]={'generated_ids':failed_rows[0]['generated_ids'],'hashes':failed_rows[0]['per_row_first4_logits_sha256']}
for st in range(4):
 raw=(r/'results'/f'kimi-gate-perf-r520-910375.127-reference_b4_m4-q0-step{st}-bf16.bin').read_bytes();assert len(raw)==4*vocab*2
 for row in range(4):assert hashlib.sha256(raw[row*vocab*2:(row+1)*vocab*2]).hexdigest()==new_references[4]['hashes'][row][st]
report['scope']='Explicit F32gate numerical-variant experiment543. Strictold-BF16 trial520 failedonone M4row fromstep0 andisnotreclassified. Same476DSO/621SDK/helper21; M4IDs/fullfirst4 mustmatchimmutable520raw innewprocess plusrepeat. FirstM12full512 establishes samevariant reference; all512 IDs/fullfirst4 mustrepeat inM12/G48serial/pipeline. Oldnative/stagedIDs/logits recordeddiagnostically.44knownanswer/EOS/rawlogit/cross-epoch gates retained. Passing cannotcertifyoldprecisioncompatibility,broadquality,HTTP,maxTPS orallhistoricalstability.'
report['previous_strict_trial_failed']='910375.127'
report['old_precision_compatibility_qualified']=False
report['M4_reference_provenance']='immutable520failedtrial fourstageagreement andrawSHA'

def run_chat(epoch):
 chat=json.loads((r/'results/kimi-chat-cases-r20.json').read_text());cases=dict(chat['cases']);eos=chat['eos_token_ids'];reference=json.loads((r/'results/modern-kimi-quality-fp32-r20-910375.77-summary.json').read_text())
 for i in range(3):cases['sort_probe_'+str(i+1)]=cases['sort']
 for label,case in cases.items():
  for repeat in [1,2]:
   phase=epoch+':'+label+'_r'+str(repeat);check(lib.modern_kimi_engine_reset_r963(ctx));control.collect('reset:'+phase,{});prompt=case['input_ids'];generated=[];hashes=[];steps=[];start=time.perf_counter()
   for step in range(128):
    ids=prompt if step==0 else[generated[-1]];past=0 if step==0 else len(prompt)+step-1;emitted,h,seconds=forward(ids,[past],len(ids),True,phase,step);generated.append(emitted[0]);steps.append(seconds)
    if step<4:hashes.append(h[0])
    if generated[-1]in eos:break
   candidates=[v for v in reference['cases']if v['task']==('sort'if label.startswith('sort_probe_')else label)and v['grouped']];matched=generated[-1]in eos and any(v['generated_ids']==[generated]and v['first4_logits_sha256']==hashes for v in candidates)
   known_ids=generated[-1]in eos and any(v['generated_ids']==[generated]for v in candidates);finite_key='sort'if label.startswith('sort_probe_')else label
   if finite_key not in new_finite:new_finite[finite_key]={'generated_ids':generated,'hashes':hashes}
   stable=generated==new_finite[finite_key]['generated_ids']and hashes==new_finite[finite_key]['hashes']
   old543=qualified_chats[(epoch,label,repeat)];assert generated==old543['generated_ids']and hashes==old543['first4_logits_sha256'],('648cacheprofile changed scalar543',epoch,label,repeat)
   report['chat_cases'].append({'actual_counters':counters(),'known_answer_ids_exact':known_ids,'same_variant_first4_repeat_exact':stable,'epoch':epoch,'label':label,'repeat':repeat,'generated_ids':generated,'first4_logits_sha256':hashes,'stopped_at_eos':generated[-1]in eos,'same_staged_f32_ids_logits':matched,'elapsed_seconds':time.perf_counter()-start,'step_seconds':steps});path.write_text(json.dumps(report,indent=2));print('KIMI_ENGINE_CHAT',stage,label,repeat,len(generated),flush=True);assert known_ids and stable,('Newprecision finite ID/EOS/repeat failure',epoch,label,repeat)

batch16_refs={};batch16_raw_parts={}
# R669failed literal-ID gate remains failed. Only the sort task may omit
# standalone space token220; every other token/order/EOS stays exact.
def task_ids_r691(label,ids):return [v for v in ids if v!=220]if label=='sort'else ids

def run_batch16_chat(epoch):
 payload=json.loads((r/'results/kimi-chat-cases-r20.json').read_text())
 for label in ['multiply','sort','retrieval']:
  prompt=payload['cases'][label]['input_ids'];assert 0<len(prompt) and 8*len(prompt)<=6144 and len(prompt)+128<=1024
  old=qualified_chats[(epoch,label,1)]
  for repeat in [1,2]:
   phase='batch16_'+epoch+':'+label+'_r'+str(repeat);check(lib.modern_kimi_engine_reset_r963(ctx));control.collect('reset:'+phase,{});generated=[[]for _ in range(16)];hashes=[[]for _ in range(16)];finished=[False]*16
   for step in range(128):
    batch16_raw_parts.clear()
    chunk=4 if step==0 else 16;width=len(prompt)if step==0 else 1;past=0 if step==0 else len(prompt)+step-1;emitted_all=[]
    for offset in range(0,16,chunk):
     ids=prompt*chunk if step==0 else[v[-1]for v in generated[offset:offset+chunk]]
     emitted,h,seconds=forward(ids,[past]*chunk,width,True,phase,step,offset,False,offset//chunk)
     emitted_all.extend(emitted)
     if step<4:
      for i,value in enumerate(h):hashes[offset+i].append(value)
    for i,value in enumerate(emitted_all):
     if not finished[i]:
      generated[i].append(value);finished[i]=value in payload['eos_token_ids']
    if not all(task_ids_r691(label,v)==task_ids_r691(label,old['generated_ids'])[:len(task_ids_r691(label,v))]for v in generated):
     report['batch16_failure']={'epoch':epoch,'label':label,'repeat':repeat,'step':step,'generated_ids_per_row':generated,'expected':old['generated_ids']}
     if stage==3:
      raw=b''.join(batch16_raw_parts[k]for k in sorted(batch16_raw_parts));file=r/'results'/f'kimi-token-only-r964-{tag}-{phase}-step{step}-failure-bf16.bin';file.write_bytes(raw);report['batch16_failure'].update(raw_file=str(file),raw_sha256=hashlib.sha256(raw).hexdigest())
     path.write_text(json.dumps(report,indent=2));raise AssertionError(('691B16 task answer prefix',epoch,label,step,emitted_all))
    if all(finished):break
   assert all(task_ids_r691(label,v)==task_ids_r691(label,old['generated_ids'])and v[-1]in payload['eos_token_ids']for v in generated)
   if label not in batch16_refs:batch16_refs[label]=(generated,hashes)
   assert (generated,hashes)==batch16_refs[label],('669B16 samegeometry repeat',epoch,label,repeat)
   row={'epoch':epoch,'label':label,'repeat':repeat,'generated_ids_per_row':generated,'per_row_first4_logits_sha256':hashes,'all16_old_known_answer_IDs_EOS_exact':all(v==old['generated_ids']for v in generated),'all16_task_answer_IDs_EOS_exact_after_sort_space220_normalization':True,'sort_space220_normalization_allowed':label=='sort','same_geometry_raw_hashes_repeated_exact':True,'actual_counters':counters()}
   report.setdefault('batch16_chat_cases',[]).append(row);path.write_text(json.dumps(report,indent=2));print('KIMI_BATCH648_QA',epoch,label,repeat,flush=True)

large_refs={}
def run_large_chat(epoch,batch):
 payload=json.loads((r/'results/kimi-chat-cases-r20.json').read_text())
 for label in ['multiply','sort','retrieval']:
  prompt=payload['cases'][label]['input_ids'];assert 0<len(prompt) and 8*len(prompt)<=6144 and len(prompt)+128<=1024
  old=qualified_chats[(epoch,label,1)]
  for repeat in [1,2]:
   phase='large'+str(batch)+'_'+epoch+':'+label+'_r'+str(repeat);check(lib.modern_kimi_engine_reset_r963(ctx));control.collect('reset:'+phase,{});generated=[[]for _ in range(batch)];hashes=[[]for _ in range(batch)];finished=[False]*batch
   for step in range(128):
    batch16_raw_parts.clear()
    chunk=4 if step==0 else batch;width=len(prompt)if step==0 else 1;past=0 if step==0 else len(prompt)+step-1;emitted_all=[]
    for offset in range(0,batch,chunk):
     ids=prompt*chunk if step==0 else[v[-1]for v in generated[offset:offset+chunk]]
     emitted,h,seconds=forward(ids,[past]*chunk,width,True,phase,step,offset,False,offset//chunk)
     emitted_all.extend(emitted)
     if step<4:
      for i,value in enumerate(h):hashes[offset+i].append(value)
    for i,value in enumerate(emitted_all):
     if not finished[i]:
      generated[i].append(value);finished[i]=value in payload['eos_token_ids']
    if not all(task_ids_r691(label,v)==task_ids_r691(label,old['generated_ids'])[:len(task_ids_r691(label,v))]for v in generated):
     report['large_batch_failure']={'epoch':epoch,'label':label,'repeat':repeat,'step':step,'generated_ids_per_row':generated,'expected':old['generated_ids']}
     if stage==3:
      raw=b''.join(batch16_raw_parts[k]for k in sorted(batch16_raw_parts));file=r/'results'/f'kimi-token-only-r964-{tag}-{phase}-step{step}-failure-bf16.bin';file.write_bytes(raw);report['large_batch_failure'].update(raw_file=str(file),raw_sha256=hashlib.sha256(raw).hexdigest())
     path.write_text(json.dumps(report,indent=2));raise AssertionError(('691B16 task answer prefix',epoch,label,step,emitted_all))
    if all(finished):break
   assert all(task_ids_r691(label,v)==task_ids_r691(label,old['generated_ids'])and v[-1]in payload['eos_token_ids']for v in generated)
   if (batch,label) not in large_refs:large_refs[batch,label]=(generated,hashes)
   assert (generated,hashes)==large_refs[batch,label],('669B16 samegeometry repeat',epoch,label,repeat)
   row={'batch':batch,'epoch':epoch,'label':label,'repeat':repeat,'generated_ids_per_row':generated,'per_row_first4_logits_sha256':hashes,'all_rows_old_known_answer_IDs_EOS_exact':all(v==old['generated_ids']for v in generated),'all_rows_task_answer_IDs_EOS_exact_after_sort_space220_normalization':True,'sort_space220_normalization_allowed':label=='sort','same_geometry_raw_hashes_repeated_exact':True,'actual_counters':counters()}
   report.setdefault('large_batch_chat_cases',[]).append(row);path.write_text(json.dumps(report,indent=2));print('KIMI_CAPACITY834_LARGE_QA',epoch,label,repeat,flush=True)

report['finite_validation_policy']='691sortonly mayomit verifiedsingle-space token220; allother tokens/order/EOS strict. r669literal-ID failure remains failed. All16ownraw/ID repeats/crossepoch strict. Perrow history stopsatfirstEOS; completedslotscontinuefeedingEOS untilallrowsfinish.'
report['space220_vocab_sha256']=hashlib.sha256(space_vocab_r691).hexdigest()
run_batch16_chat('before')
# Capacity918 coversactualB24/B32 QA; thissameprocessA/B usesM16only.
frozen880=json.loads((r/'results/kimi-balanced-capacity-r880-910371.131-stage3.json').read_text());frozen880cases={v['phase']:v for v in frozen880['cases']};assert all(v in frozen880cases for v in ['reference_b16_m16_r1','reference_b16_m16_r2','long_pipeline_g64_m16_r1','long_pipeline_g64_m16_r2'])
run_chat('before')
report['scope']='691same669476/SDK621/helper21/cache288/64/defaultcommunication, no GPU math change. Literal-ID sort gate669FAILED retained. New declared finite policy: sortonly mayomit token220standalone space, allother tokens/order/EOS exact; captureall16completeEOS histories/ownfirst4+ID repeats/crossepoch. Multiply/retrieval and44scalar oldIDs strict. M4/M12/G48controls strict543; M16/G64own512refs/repeats/all32/raw/policy/shutdown. No generic whitespace relaxation or semantic/fullquality/hardwaremax claim.'
report['scope']='863cache128/544 memorytrial withprefill4; unchanged839/476math/SDK621/helper21/defaultcollectives. Preserve834stage2 pref8 OOM. Old691pref8longIDs/raw diagnostic, ownM16G64/M24G96/M32G128 full512x2 strict;44scalar frozen543 plus12B16/24B24B32 oldknownQA/EOS/rawrepeats/all32/shutdown. GPUfreebeforecreate/afterload/beforeeachprefill recorded. No hardwaremax/aggregate claim.'
report['scope']='880balanced93/PP4 boundaries0,24,47,71,93; cache128/544 andprefill4; immutable876/SDK621/helper21, all93layers andsameoperator math. OwnM16G64/M24G96/M32G128 full512x2 strict, frozen543scalar andallactualfiniteQA/raw/32owners/shutdown; old691prefill8 longdiagnostic. Preserve failed834and863 OOM; measureactualmemory ratherthanassumefit.'
report['scope']='918full93TP8PP4 new379nodegroup, private914same876partition/SDK621/helper21/math. Onlycache4N+4pages andKDAstateN+1, noownedrow/slotchange. All8gfx936/64GiB/80CU andsmallcomputepreflight;128rows516pages,prefill4, knownscalar543/actualB16B24B32/EOS/rawrepeats, full512x2M16G64M24G96M32G128/all32/shutdown strict. Sameprocessbatchcontrols; nodegroup differs880, no directcausalTPSclaimfromcrossnodegroups.'
PHASES=[('reference_b16_m16_r1',16,16,False,512),('reference_b16_m16_r2',16,16,False,512),('long_pipeline_g64_m16_r1',64,16,True,512),('long_pipeline_g64_m16_r2',64,16,True,512),('tokens_only_g64_m16_r1',64,16,True,512),('tokens_only_g64_m16_r2',64,16,True,512)]
report['scope']='964sameprocess baselinefullCPUlogits versus opt-inflag4token-onlyprivateCAPI, modelcsrc/SDK621/helper21/math unchanged914/876PP/cache128516/prefill4. Every512token mustmatchfrozen880; first4rawpluslate63/127/255/383/511 all8checks andfullbaselinebits exact;44scalar/12B16knownQA/actual32BWpreflight/shutdown. CPUlogits omission changes servingoutput/harness costs, notGPUkernel speed orphysicalpeak. Counters excludeextraall8debugcopies; benchmarkwallincludescheckpointcost.'
for phase,batch,micro,pipeline,count in PHASES:
 metrics_before963=output_metrics963()
 prefill_micro=4 if micro>=16 else micro
 check(lib.modern_kimi_engine_reset_r963(ctx));control.collect('reset:'+phase,{});generated=[[]for _ in range(batch)]if not pipeline or stage in [0,3]else None;hashes=[[]for _ in range(batch)]if stage==3 else None;times=[];events=[];start=time.perf_counter();feedback_start=control.feedback_seq
 for step in range(count):
  active_micro=prefill_micro if step==0 else micro
  for q in range(batch//active_micro):
   offset=q*active_micro
   if pipeline and stage==0 and step>0:
    values=control.receive_for_next_step(phase,step,q,micro,prefill_micro,vocab)
    for i,value in enumerate(values):generated[offset+i].append(value)
   width=512 if step==0 else 1;past=0 if step==0 else 512+step-1
   if step==0:ids=sum([prompts[(offset+i)%4]for i in range(active_micro)],[])
   elif stage==0 or not pipeline:ids=[generated[offset+i][-1]for i in range(active_micro)]
   else:ids=[0]*active_micro
   baseline_hashes=new_references[micro]['hashes']if micro in new_references else None;expected_step_hashes=[baseline_hashes[(offset+i)%micro][step]for i in range(active_micro)]if step<4 and baseline_hashes is not None else None
   emitted,h,seconds=forward(ids,[past]*active_micro,width,micro!=4,phase,step,offset,pipeline,q,expected_step_hashes)
   if not pipeline or stage==3:
    for i,value in enumerate(emitted):generated[offset+i].append(value)
   if stage==3:
    if step<4:
     for i,value in enumerate(h):hashes[offset+i].append(value)
    if pipeline:control.publish_feedback(phase,step,q,emitted)
   events.append({'step':step,'microbatch':q,'native_forward_seconds':seconds,'completion_seconds':time.perf_counter()-start})
  times.append(time.perf_counter()-start)
  if step%64==0 or step==count-1:print('KIMI_ENGINE_PIPELINE_STEP',stage,phase,step,round(times[-1],6),flush=True)
 if pipeline and stage==0:
  for q in range(batch//micro):
   values=control.receive_for_next_step(phase,count,q,micro,prefill_micro,vocab)
   for i,value in enumerate(values):generated[q*micro+i].append(value)
 elapsed=time.perf_counter()-start;feedback_count=control.feedback_seq-feedback_start
 gathered=control.collect('end:'+phase,{'stage':stage,'generated_ids':generated,'hashes':hashes,'elapsed_seconds':elapsed,'feedback_packets':feedback_count});head_ids=gathered[0]['generated_ids'];tail_ids=gathered[3]['generated_ids'];tail_hashes=gathered[3]['hashes'];expected=frozen['generated_ids']if micro==4 else frozen12['generated_ids'];expected_hashes=frozen_hashes if micro==4 else frozen12['per_row_first4_logits_sha256'];expected=[expected[i%len(expected)][:count]for i in range(batch)];expected_hashes=[expected_hashes[i%len(expected_hashes)]for i in range(batch)]
 if micro not in new_references:new_references[micro]={'generated_ids':tail_ids,'hashes':tail_hashes}
 stable_hashes=[new_references[micro]['hashes'][i%micro]for i in range(batch)];report['same_variant_references']=new_references
 variant_ids=[new_references[micro]['generated_ids'][i%micro][:count]for i in range(batch)]
 row={'prefill_micro_batch':prefill_micro,'same_variant_full_IDs_exact':tail_ids==variant_ids,'same_variant_first4_logits_exact':tail_hashes==stable_hashes,'actual_counters':counters(),'phase':phase,'global_batch':batch,'micro_batch':micro,'pipeline':pipeline,'input_tokens':512,'output_tokens':count,'generated_ids':tail_ids,'per_row_first4_logits_sha256':tail_hashes,'step_completion_seconds':times,'events':events,'elapsed_seconds':elapsed,'head_tail_ids_exact':head_ids==tail_ids,'frozen_same_shape_ids_exact':tail_ids==expected,'frozen_same_shape_first4_logits_exact':tail_hashes==expected_hashes,'all8_tail_first4_logits_checked':True,'feedback_packets_by_stage':[v['feedback_packets']for v in gathered]}
 if micro==12:
  row['native_original_ids_exact']=tail_ids==[native12['generated_ids'][i%12][:count]for i in range(batch)];row['native_original_first4_logits_exact']=tail_hashes==[native12['per_row_first4_logits_sha256'][i%12]for i in range(batch)]
 if micro==16:
  previous834=goldcases834[phase.replace('tokens_only_g64_','long_pipeline_g64_')];row['old691_prefill8_IDs_exact_diagnostic']=tail_ids==previous834['generated_ids'];row['old691_prefill8_raw_exact_diagnostic']=tail_hashes==previous834['per_row_first4_logits_sha256']
 metric_end963=output_metrics963();row['CAPI_output_metrics_delta']={k:metric_end963[k]-metrics_before963[k]for k in metric_end963 if k!='stage'};row['harness_metrics']=dict(harness963[phase]);row['late_checkpoint_count']=sum(v['phase']==phase for v in late_checks963);reference963=frozen880cases[phase.replace('tokens_only_g64_','long_pipeline_g64_')];row['frozen880_full512_IDs_raw_exact']=tail_ids==reference963['generated_ids']and tail_hashes==reference963['per_row_first4_logits_sha256']
 report['cases'].append(row);path.write_text(json.dumps(report,indent=2));assert row['head_tail_ids_exact']and row['same_variant_full_IDs_exact']and row['same_variant_first4_logits_exact'],('Explicit F32variant repeat mismatch',phase,stage)
 assert row['frozen880_full512_IDs_raw_exact'],('963outputmode changedmodel',stage,phase)
 if not pipeline:assert all(v['generated_ids']==variant_ids for v in gathered)
 assert row['feedback_packets_by_stage']==([batch//prefill_micro+(count-1)*(batch//micro),0,0,batch//prefill_micro+(count-1)*(batch//micro)]if pipeline else[0,0,0,0])
 print('KIMI_ENGINE_PIPELINE_PHASE_COMPLETE',stage,phase,elapsed,flush=True)

run_chat('after')
run_batch16_chat('after')
# Capacity918 coversactualB24/B32 QA; thissameprocessA/B usesM16only.
control.collect('before_destroy',{});check(lib.modern_kimi_engine_destroy_r963(ctx));control.collect('after_destroy',{});control.close();report.update(completed=True,all8_owned_workers_shutdown_completed=True);path.write_text(json.dumps(report,indent=2));print('KIMI_ENGINE_SERIAL_R198_COMPLETE',stage,flush=True)
