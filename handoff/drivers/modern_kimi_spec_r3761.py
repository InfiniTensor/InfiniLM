import os as _os,sys as _sys
assert _os.environ['PYTHONPYCACHEPREFIX'].startswith('/local/')
_sys.pycache_prefix=_os.environ['PYTHONPYCACHEPREFIX']+'/'+str(_os.getpid())
from kimi_spec_protocol_r3761 import run as _preimported_reduce_probe
from model_affinity_check_r2628 import verify as verify_owner_affinity
import collections,ctypes,hashlib,json,os,pathlib,time,threading,signal
import numpy as np
import torch
from safetensors import safe_open
from stage_control_r2386 import StageControl
import infinilm

torch.set_num_threads(1)
r=pathlib.Path('/refactor');model=pathlib.Path(os.environ.get('KIMI_MODEL_VIEW_R3679','/shared/Kimi-K3'));node=int(os.environ['SLURM_PROCID'])+int(os.environ['GLOBAL_OFFSET_R2354']);tp_nodes=int(os.environ['KIMI_TP_NODES']);pp=int(os.environ['KIMI_PP_SIZE']);participants=tp_nodes*pp;assert(tp_nodes,pp)in[(1,4),(2,2),(4,1),(4,2),(6,1)]
stage=node//tp_nodes;tail=stage==pp-1;emitter=(pp-1)*tp_nodes;tag=os.environ['GLOBAL_RUN_R2354'];head=os.environ['KIMI_ENGINE_HEAD'];port=int(os.environ['GLOBAL_PORT_R2354'])
control=StageControl(node,head,port+7000,os.environ['KIMI_ENGINE_NONCE'],participants,7200);control.collect('before_create',{'node':node,'TP':8*tp_nodes,'PP':pp})
path=r/'results'/f'kimi-spec-r3761-{tag}-node{node}.json';assert not path.exists();report={'node':node,'step':tag,'TP':8*tp_nodes,'PP':pp,'PP_stage':stage,'nodes':participants,'all_serving_cards':participants*8,'sampling_node':emitter,'completed':False,'target_achieved':False,'cases':[],'chat_cases':[],'natural_cases':[],'scope':'1966 latestfe998 complete93model with CPU-proven nested traversal linked locally with newlyrebuiltRuntime/Python/SDK/CCL, originaltiledcandidate and unchangedkernelobjects; no inherited performance. Original ABBA tiledW13off/on/on/off with qualified1477 ownedpolicy2 alwayson. Tile4/wave64/cached scratch; actualperforward92layercounts all32owners. Preserve all knownQA/fixed/natural fullID/raw exact and allcleanup. Prefill/newSDK416a/helper21/othermath unchanged. No speed/15 adoption before independent full audit. Parent1459full93 KimiTP8PP4/private1429 intentionalnumericalvariant: q_a/q_b prefillF32-output GEMM thenBF16, parentgateF32unchanged, newSDK416a/helper21/weights/nativeDecode/attention/MoE/Graphunchanged. No traceobserver. Isolated1422/1427matrixproof notmodelqualification. EveryforwardactualQdispatch counters, 44frozenknownIDs/EOS withsamevariant fullrawrepeats before/after; 2fixed512 fullIDs/raw9 ownrepeat diagnostics and4natural~512/max512 firstEOS cases, alltails8workers/CPUargmax/32ownerclose/Slurm0. Oldprecisionrawcompatibility notclaimed, original1283failurepreserved. NoHTTP/concurrency/15/bestnodecount adoptionbeforemeasurement.'}
def save():
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def memory():return [list(torch.cuda.mem_get_info(d))for d in range(8)]
report.update(numerical_variant='1429_q_a_q_b_prefill_F32_then_BF16',old_precision_raw_compatibility_claimed=False)
report['LM_source_commit']='5f91ad738583db8f5617aeae53b76523f85d1ec6'
report['loading_control_policy']={'initial_timeout_seconds':7200,'after_weights_ready_seconds':900,'prior_run':'r2361 failed waiting for uneven weight loading, no QA or performance qualification','math_ownership_and_QA_guards_unchanged':True}
owned_policy=2
tile_policy=0
report['dense_prefill_optin']=os.environ.get('INFINILM_DENSE_PREFILL_F32_R2475')=='1';report['candidate_source']='prefill-f32-b7b370-r2475';report['scope']='2489 topology frontier: fresh b7 model/runtime, opt-in prefill path OFF. TP16PP2/TP32PP1/TP48PP1 independently retain full frozen QA/own raw/EOS/CPUargmax/normal owner gates. No 15SLO adoption before independent audit.'
assert os.environ.get('INFINILM_DENSE_PREFILL_F32_R2475')in ['0','1']
report['SDK_tile_optin']=os.environ.get('INFINILM_MXFP4_TILE_TP_R2559')=='1'
report['loader_transaction_policy']={'max_parameter_keys':16384,'max_source_bytes':2*1024**3,'baseline_max_keys':2048,'baseline_max_bytes':512*1024**2,'owner_LoadBatch_capacity':256,'checkpoint_bytes_dtype_shape_and_fullQA_unchanged':True,'CPU_copy_build_receipt':'common-latest-r3673-Kimi-build.json'}
ctx=None;save()
try:
 report['phase']='hardware_preflight';report['hardware']=[];report['memory_before_create']=memory();assert all(free>58*1024**3 for free,total in report['memory_before_create'])
 for d in range(8):
  props=torch.cuda.get_device_properties(d);assert ('gfx936'in props.name or'gfx936'in getattr(props,'gcnArchName',''))and props.total_memory>=63*1024**3 and props.multi_processor_count==80
  with torch.cuda.device(d):
   x=torch.arange(2048,device='cuda',dtype=torch.float32);assert x.sum().item()==2047*2048//2;torch.cuda.synchronize();del x
  report['hardware'].append({'device':d,'name':props.name,'memory':props.total_memory,'CU':props.multi_processor_count,'small_compute_passed':True})
 preflight=json.loads((r/'results/kimi-resident-r2493-preflight.json').read_text());choice=next(v for v in preflight['topologies']if v['TP']==8*tp_nodes and v['PP']==pp);assert choice['all_stages_parameter_plus_6GiB_reserve_fits_64GiB'];report['planned_resident_memory']=choice['stages'][stage]
 library=r/'candidates/kimi-spec-math-5f91ad7-r3755/libmodern_kimi_spec_math_r3755.so';assert sha(library)==json.loads((r/'results/kimi-spec-math-r3755-build.json').read_text())['library_sha256'];lib=ctypes.CDLL(str(library));
 for method in ['error','create','keys','load','finish','forward','rank_logits','reset','destroy','partition','output_metrics','resize_cache','cache_geometry','layout']:assert getattr(lib,'modern_kimi_engine_'+method+'_r2475')
 lib.modern_kimi_engine_error_r2475.restype=ctypes.c_char_p
 def check(code):
  if code:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 lib.modern_kimi_engine_create_spec_r3632.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_size_t];lib.modern_kimi_engine_create_spec_r3632.restype=ctypes.c_void_p
 for name in ['keys','cache_geometry','output_metrics']:
  f=getattr(lib,'modern_kimi_engine_'+name+'_r2475');f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_char_p
 for name in ['finish','reset','destroy','rank_logits']:getattr(lib,'modern_kimi_engine_'+name+'_r2475').argtypes=[ctypes.c_void_p]
 lib.modern_kimi_engine_load_r2475.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.modern_kimi_engine_forward_r2475.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_int]+[ctypes.c_void_p]*9
 lib.modern_kimi_owned_stats_r1213.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
 def owned_stats():
  values=[]
  for d in range(8):
   v=(ctypes.c_uint64*12)();check(lib.modern_kimi_owned_stats_r1213(d,v));values.append(list(v))
  return values
 lib.modern_kimi_tiled_calls_r1861.argtypes=[ctypes.c_int];lib.modern_kimi_tiled_calls_r1861.restype=ctypes.c_uint64
 lib.modern_kimi_q_prefill_fp32_calls_r1429.argtypes=[ctypes.c_int];lib.modern_kimi_q_prefill_fp32_calls_r1429.restype=ctypes.c_uint64
 lib.modern_kimi_engine_partition_r2475.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]
 ranges=[]
 for p in range(pp):
  pair=(ctypes.c_size_t*2)();check(lib.modern_kimi_engine_partition_r2475(p,pp,pair));ranges.append(list(pair))
 assert [n for a,b in ranges for n in range(a,b)]==list(range(93));report['PP_ranges']=ranges
 # Prepare the exact existing raw GPU entry on each physical primary context
 # before engine creation or any pipeline receive/collective is submitted.
 # No real model weights/cache are touched; no graph or single-use guard changes.
 raw_tile=lib.modern_kimi_tiled_raw_r1861;raw_tile.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.c_size_t,ctypes.c_void_p];raw_tile.restype=ctypes.c_int;report['raw_tile_prewarm']=[];save()
 for device in range(8):
  torch.cuda.set_device(device);start_warm=time.perf_counter()
  warm_values=[torch.full((1,3584),float('nan'),device=device,dtype=torch.bfloat16),torch.full((1,16,384),float('nan'),device=device,dtype=torch.bfloat16),torch.zeros((1,3584),device=device,dtype=torch.bfloat16),torch.arange(16,device=device,dtype=torch.int32).view(1,16),torch.full((1,16),1/16,device=device,dtype=torch.float32),torch.zeros((16,768,1792),device=device,dtype=torch.uint8),torch.full((16,768,112),127,device=device,dtype=torch.uint8),torch.zeros((16,3584,192),device=device,dtype=torch.uint8),torch.full((16,3584,12),127,device=device,dtype=torch.uint8)]
  torch.cuda.synchronize(device);code=raw_tile(device,*[ctypes.c_void_p(v.data_ptr())for v in warm_values],16,ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream));assert code==0,code;torch.cuda.synchronize(device)
  assert bool((warm_values[0]==0).all())and bool((warm_values[1]==0).all())and bool((warm_values[2]==0).all())and torch.equal(warm_values[3],torch.arange(16,device=device,dtype=torch.int32).view(1,16))and bool((warm_values[4]==1/16).all())and bool((warm_values[5]==0).all())and bool((warm_values[6]==127).all())and bool((warm_values[7]==0).all())and bool((warm_values[8]==127).all())
  report['raw_tile_prewarm'].append({'device':device,'seconds':time.perf_counter()-start_warm,'valid_zero_outputs_and_all_inputs_readonly':True});save();del warm_values;torch.cuda.empty_cache()
 assert [int(lib.modern_kimi_tiled_calls_r1861(d))for d in range(8)]==[0]*8
 cfg=json.loads((model/'config.json').read_text());text=cfg['text_config'];vocab=text['vocab_size'];report['phase']='creating';save();begin=time.perf_counter();ctx=lib.modern_kimi_engine_create_spec_r3632(json.dumps(cfg).encode(),node,tp_nodes,pp,head.encode(),port,128)
 if not ctx:raise RuntimeError(lib.modern_kimi_engine_error_r2475().decode())
 report['create_seconds']=time.perf_counter()-begin;lib.modern_kimi_engine_layout_r2475.argtypes=[ctypes.c_void_p];lib.modern_kimi_engine_layout_r2475.restype=ctypes.c_char_p;layout=json.loads(lib.modern_kimi_engine_layout_r2475(ctx));assert layout=={'node':node,'local_workers':8,'TP':8*tp_nodes,'TP_node_rank':node%tp_nodes,'PP':pp,'PP_stage':stage,'sampling_node':emitter,'public_graph_enabled':False,'owned_metadata_enabled':False};report['layout']=layout
 geometry=json.loads(lib.modern_kimi_engine_cache_geometry_r2475(ctx));assert geometry=={'max_rows':128,'pages':516,'page_size':256,'KDA_pool_rows':129};report['cache_geometry']=geometry;report['memory_after_create']=memory();assert all(free>3*1024**3 for free,total in report['memory_after_create']),('insufficient_workspace_reserve_after_create',report['memory_after_create'])
 keys=set(json.loads(lib.modern_kimi_engine_keys_r2475(ctx)));mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];assert keys<=set(mapping);report['parameter_keys']=len(keys);report['phase']='loading_weights';save();groups=collections.defaultdict(list)
 for key in keys:groups[mapping[key]].append(key)
 pending=[];pending_bytes=loaded=flushes=0;start=time.perf_counter()
 def flush():
  global pending_bytes,loaded,flushes
  if not pending:return
  descriptors=[{'name':name,'pointer':w.data_ptr(),'shape':list(w.shape),'dtype':{torch.float32:0,torch.bfloat16:1,torch.uint8:2}[w.dtype]}for name,w in pending];check(lib.modern_kimi_engine_load_r2475(ctx,json.dumps(descriptors).encode()));loaded+=len(pending);pending.clear();pending_bytes=0;flushes+=1
  if flushes%16==0:report.update(loaded_keys=loaded,loading_elapsed_seconds=time.perf_counter()-start);save();print('KIMI_GLOBAL_R1482_LOAD',node,loaded,len(keys),round(time.perf_counter()-start,3),flush=True)
 for filename,names in sorted(groups.items()):
  with safe_open(model/filename,framework='pt',device='cpu')as shard:
   for name in sorted(names):
    w=shard.get_tensor(name)
    if name.endswith('.self_attn.A_log'):w=w[:text['linear_attn_config']['num_heads']]
    w=w.contiguous();pending.append((name,w));pending_bytes+=w.numel()*w.element_size()
    if len(pending)>=16384 or pending_bytes>=2*1024*1024*1024:flush()
   flush()
 check(lib.modern_kimi_engine_finish_r2475(ctx));assert loaded==len(keys);report.update(loaded_keys=loaded,loading_seconds=time.perf_counter()-start,memory_after_finish=memory())
 maps=pathlib.Path('/proc/self/maps').read_text().splitlines()
 for part,expected in [('libinfiniops.so',json.loads((r/'results/latest-sdk-r3672-Kimi-build.json').read_text())['library_sha256']),('libkimi_grouped_prefill.so',json.loads((r/'results/model-latest-r3675-Kimi-build.json').read_text())['fresh_helpers']['helpers/libkimi_grouped_prefill.so'])]:
  paths={line.split()[-1]for line in maps if part in line};assert len(paths)==1;actual=sha(pathlib.Path(next(iter(paths))));assert actual==expected;report[part+'_sha256']=actual
 report['library_sha256']=sha(library);report['phase']='weights_ready';save();control.collect('weights_ready',{'node':node,'layout':layout,'memory':report['memory_after_finish']});control.set_timeout(900);print('KIMI_GLOBAL_R1482_WEIGHTS_READY',node,report['loading_seconds'],flush=True)
 mappings={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfinilm_runtime.so'in v or '_infinilm.cpython'in v};assert mappings and all('/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/'in p for p in mappings);report['actual_LM_native_libraries']={p:sha(pathlib.Path(p))for p in mappings};save()
 rtproof=json.loads((r/'results/common-latest-r3673-Kimi-build.json').read_text());assert rtproof['completed'];assert[sha(pathlib.Path(p))for p in mappings if'libinfinilm_runtime.so'in p]==[rtproof['runtime_sha256']];cpylib=ctypes.CDLL(next(p for p in mappings if'libinfinilm_runtime.so'in p));cpylib.modern_cpu_copy_stats_r2515.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];values=(ctypes.c_uint64*5)();assert cpylib.modern_cpu_copy_stats_r2515(values,5)==0;report['CPU_copy_stats_after_loading']=list(values);assert values[0]>0 and values[1]>0;report['loader_flushes']=flushes;report['source_parameter_keys_loaded']=loaded;save()
 sdk=ctypes.CDLL(str(r/'install/hygon-kimi-5f91ad7-r3672/lib/libinfiniops.so'));sdk.modern_mxfp4_tile_tp_counts_r2559.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
 def sdk_tile_stats():
  values=(ctypes.c_uint64*8)();assert sdk.modern_mxfp4_tile_tp_counts_r2559(values,8)==0;return list(values)
 report['SDK_tile_initial_counts']=sdk_tile_stats();assert report['SDK_tile_initial_counts']==[0]*8
 from kimi_spec_protocol_r3761 import run
 assert all(free>6*1024**3 for free,total in memory()),('6GiB_reserve_before_multi',memory())
 run(globals())
 report['owner_affinity']=verify_owner_affinity(lib);report['output_metrics']=json.loads(lib.modern_kimi_engine_output_metrics_r2475(ctx));report['qualification_before_destroy_passed']=True;control.collect('before_destroy',{})

except BaseException as failure:
 report['error']=repr(failure);save();raise
finally:
 try:
  if ctx:check(lib.modern_kimi_engine_destroy_r2475(ctx));report['all8_owned_workers_shutdown_completed']=True
  if report.get('qualification_before_destroy_passed'):control.collect('after_destroy',{})
  report['completed']=bool(report.get('qualification_before_destroy_passed'));report['phase']='completed'if report['completed']else'failed'
 except BaseException as failure:report['shutdown_error']=repr(failure);report['completed']=False;raise
 finally:save();control.close()
print('KIMI_SPEC_R3634_COMPLETE',node,flush=True)
