import os as _cache_os,sys as _cache_sys
_cache_base=_cache_os.environ['PYTHONPYCACHEPREFIX'];assert _cache_base.startswith('/local/infini-pycache-')
_cache_sys.pycache_prefix=_cache_base+'/'+str(_cache_os.getpid())
from glm_idle_probe_r3772 import run as _preloaded_probe_r3091
from model_affinity_check_r2628 import verify as verify_owner_affinity
import base64
import ctypes
import hashlib
import json
import os
import pathlib
import time
import traceback
import socket
import re
from glm_idle_chat_r3772 import run_chat_checks

import torch
import torch.multiprocessing as mp

r = pathlib.Path('/refactor')
# Reuse the exact raw-checkpoint loader and CAPI signatures. Do not execute the
# old diagnostic worker or its __main__ block.
baseline = (r / 'incoming/modern_glm_full_model_probe_r2.py').read_text()
ns = {}
exec(baseline[:baseline.index('def worker(rank,payload):')].replace('libmodern_glm_full_model_probe.so', '../candidates/glm-idle-trim-5f91ad7-r3771/libmodern_glm_idle_trim_r3771.so'), ns)
lib, ptr, check, unique, model, stage, tag = (ns[k] for k in ['lib', 'ptr', 'check', 'unique', 'model', 'stage', 'tag'])
ROWS=int(os.environ['GLM_CACHE_ROWS_R3627']);CAPACITY=int(os.environ['GLM_CACHE_CAP_R3627']);assert(ROWS,CAPACITY)in[(16,640)];PP=int(os.environ['GLM_PP_SIZE_R3603']);assert PP in[2,4];CARDS=8*PP;BOUNDS=[0,38,78]if PP==2 else[0,18,38,58,78]
cache_overlay=pathlib.Path('/dev/shm')/('iloader-r3604-'+str(os.getuid()))/('overlay-glm-pp'+str(PP)+'-s'+str(stage))
if os.environ.get('USE_WEIGHT_CACHE_R3604')=='1':
 assert cache_overlay.is_dir(),str(cache_overlay);model=cache_overlay;ns['model']=model

lib.modern_glm_model_tp_control.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
lib.modern_glm_pipeline_last_profile.restype = ctypes.POINTER(ctypes.c_double)
lib.modern_glm_direct_calls_r601.argtypes=[ctypes.c_int];lib.modern_glm_direct_calls_r601.restype=ctypes.c_uint64
lib.modern_glm_dense_mode_r48_set.argtypes=[ctypes.c_int]
lib.modern_glm_fp8_calls_r780.argtypes=[ctypes.c_int];lib.modern_glm_fp8_calls_r780.restype=ctypes.c_uint64
lib.modern_glm_fp8_runtime_stats_r809.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
lib.modern_glm_fp8_runtime_error_r809.restype=ctypes.c_char_p
def cache_stats809(device):
    v=(ctypes.c_uint64*10)();code=lib.modern_glm_fp8_runtime_stats_r809(device,v,10)
    if code:raise RuntimeError(lib.modern_glm_fp8_runtime_error_r809().decode())
    return dict(zip(['created','destroyed','hits','misses','evictions','active','active_bytes','highwater_bytes','runs','invalid'],list(v)))

lib.modern_glm_cache_view_calls_r927.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_cache_view_calls_r927.restype=ctypes.c_uint64
lib.modern_glm_model_indexer_policy_r998.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_model_decode_leases_policy_r1356.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_model_feature_bypass_r1406.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_indexer_metadata_calls_r1406.argtypes=[ctypes.c_int];lib.modern_glm_indexer_metadata_calls_r1406.restype=ctypes.c_uint64
lib.modern_glm_indexer_select_all_calls_r980.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_indexer_select_all_calls_r980.restype=ctypes.c_uint64
lib.modern_glm_model_owned_prefix_policy_r1669.argtypes=[ctypes.c_void_p,ctypes.c_int]
lib.modern_glm_prefix_calls_r1669.argtypes=[ctypes.c_int,ctypes.c_int];lib.modern_glm_prefix_calls_r1669.restype=ctypes.c_uint64
policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p
adoption=json.loads((r/'results/glm-balanced-micro8-r153-lock.json').read_text())
mode=adoption['mode'];assert mode in [0,2]

def worker(rank, payload):
    torch.cuda.set_device(rank)
    assert torch.cuda.mem_get_info(rank)[0]>58*1024**3
    assert lib.modern_glm_dense_mode_r48_set(mode)==0
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    cfg = json.loads((model / 'config.json').read_text())
    buffers = [ctypes.create_string_buffer(base64.b64decode(payload[k] if k != 'tp' else payload[k][stage])) for k in ['world', 'tp', 'control']]
    ctx = lib.modern_glm_model_create(rank, stage, ROWS, CAPACITY, json.dumps(cfg).encode(), *buffers)
    if not ctx:
        raise RuntimeError(lib.modern_glm_model_error().decode())
    report={'TP':8,'PP':PP,'all_serving_cards':CARDS,'weight_source_path':str(model),'cache_capacity':CAPACITY,'cache_rows':ROWS,'mode':mode,'step':tag,'stage':stage,'rank':rank,'cases':[],'completed':False,'scope':'Current3072 fullGLM multi-request fastpath ABBA; startup markers and bounded1200s sharedpayload visibility wait. All original numeric/owner guards retained; failed768 preserved.'}
    report['memory_after_create']=torch.cuda.mem_get_info(rank);early=torch.tensor([float(report['memory_after_create'][0]>10*1024**3)],device='cuda');check(lib.modern_glm_model_control(ctx,1,ptr(early),stream()))
    if early.item()!=CARDS:
        report.update(capacity_rejected_before_disk_loading=True,completed=False);check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown());report['all_contexts_closed_after_memory_rejection']=True;(r/'results'/f'glm-large-preflight-r3772-{tag}-stage{stage}-rank{rank}.json').write_text(json.dumps(report,indent=2));return
    # The existing loader is executed verbatim and updates this report.
    loader = baseline[baseline.index(" sdk=sorted("):baseline.index(' def control(x):')].replace('hygon-runtime-fixes-r1', 'hygon-glm-5f91ad7-r3672')
    wrapped = 'def load(ctx, report):\n' + loader + '\n return report\n'
    loader_ns = dict(ns)
    loader_ns.update(stage=stage, rank=rank, report=report, ctx=ctx)
    exec(wrapped, loader_ns)
    loader_ns['load'](ctx, report)
    lib.modern_glm_idle_trim_r3771.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    check(lib.modern_glm_model_reset(ctx));before_trim=torch.cuda.mem_get_info(rank)[0];stats=(ctypes.c_uint64*6)()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1)as executor:assert executor.submit(lib.modern_glm_idle_trim_r3771,ctx,stats,6).result()!=0
    assert lib.modern_glm_idle_trim_r3771(ctx,stats,5)!=0
    check(lib.modern_glm_idle_trim_r3771(ctx,stats,6));assert list(stats)[3:]==[rank,stage,PP]
    torch.cuda.empty_cache();report['idle_trim_after_loading']={'native_before_after_total_device_stage_PP':list(stats),'torch_before_bytes':before_trim,'torch_after_bytes':torch.cuda.mem_get_info(rank)[0],'foreignthread_and_invalid_rejected':True}
    print('GLM_IDLE_TRIM_R3771_AFTER_LOADING',stage,rank,report['idle_trim_after_loading'],flush=True)
    barrier=torch.tensor([float(torch.cuda.mem_get_info(rank)[0]>6*1024**3)],device='cuda');check(lib.modern_glm_model_control(ctx,1,ptr(barrier),stream()))
    if barrier.item()!=CARDS:
        report.update(completed=False,capacity_rejected_after_idle_trim=True);check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown());report['all_contexts_closed_after_memory_rejection']=True;(r/'results'/f'glm-idle-capacity-r3772-{tag}-stage{stage}-rank{rank}.json').write_text(json.dumps(report,indent=2));return

    keys=set(json.loads(lib.modern_glm_model_keys(ctx)))
    layers=sorted({int(k.split('.')[2])for k in keys if k.startswith('model.layers.')});bounds=BOUNDS
    assert layers==list(range(bounds[stage],bounds[stage+1]));report['pipeline_boundaries']=bounds;report['actual_layer_indices']=layers

    sdk_paths=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfiniops.so'in line})
    assert len(sdk_paths)==1 and '/install/hygon-glm-5f91ad7-r3672/lib/' in sdk_paths[0]
    report['actual_sdk_sha256']=hashlib.sha256(pathlib.Path(sdk_paths[0]).read_bytes()).hexdigest()
    assert report['actual_sdk_sha256']==json.loads((r/'results/latest-sdk-r3672-GLM-build.json').read_text())['library_sha256']
    report['library_sha256'] = hashlib.sha256((r / 'build/../candidates/glm-idle-trim-5f91ad7-r3771/libmodern_glm_idle_trim_r3771.so').read_bytes()).hexdigest()
    assert report['library_sha256']==json.loads((r/'results/glm-idle-trim-r3771-build.json').read_text())['library_sha256']
    report['LM_source_commit']='5f91ad738583db8f5617aeae53b76523f85d1ec6'
    report['scope']='Latest5f91/2b4/1ca/cd1/51cc GLM TP8PP2/4, actual38/40or18/20/20/20layers, ctx16rows. Complete tokens exact vs frozen knownQA and own same-microABBA, rawdiagnostic; finite/CPUargmax/owner/normalclosure/Slurm0. No inherited throughput or pureTP/EP claim.'
    report['scope']='Latest5f91/2b4/1ca/cd1/51cc GLM TP8PP2/4, actual38/40or18/20/20/20layers, ctx16rows. Complete tokens exact vs frozen knownQA and own same-microABBA, rawdiagnostic; finite/CPUargmax/owner/normalclosure/Slurm0. No inherited throughput or pureTP/EP claim.'
    mappings={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfinilm_runtime.so'in v or '_infinilm.cpython'in v};assert mappings and all('/candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/'in p for p in mappings);report['actual_LM_native_libraries']={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()for p in mappings}
    report['scope']='Latest5f91/2b4/1ca/cd1/51cc GLM TP8PP2/4, actual38/40or18/20/20/20layers, ctx16rows. Complete tokens exact vs frozen knownQA and own same-microABBA, rawdiagnostic; finite/CPUargmax/owner/normalclosure/Slurm0. No inherited throughput or pureTP/EP claim.'
    report['scope']='Latest5f91/2b4/1ca/cd1/51cc GLM TP8PP2/4, actual38/40or18/20/20/20layers, ctx16rows. Complete tokens exact vs frozen knownQA and own same-microABBA, rawdiagnostic; finite/CPUargmax/owner/normalclosure/Slurm0. No inherited throughput or pureTP/EP claim.'
    report.update(old_precision_compatibility_qualified=False, numerical_variant='780_per128K_BF16_GEMM_F32dots_Mge32', component_report='glm-fp8-batched-r776-summary.json')
    report['cache_library_sha256']=json.loads((r/'results/glm-idle-trim-r3771-build.json').read_text())['fresh_helpers']['helpers/libglm_fp8_cache_r807.so']
    report['same_variant_references']={}
    report['fp8_helper_sha256']=hashlib.sha256((r/'candidates/glm-idle-trim-5f91ad7-r3771/helpers/libglm_fp8_batched_r776.so').read_bytes()).hexdigest()
    assert report['fp8_helper_sha256']==json.loads((r/'results/model-latest-r3675-GLM-build.json').read_text())['fresh_helpers']['helpers/libglm_fp8_batched_r776.so']
    report_path = r / 'results' / f'glm-fastpath-r3772-{tag}-stage{stage}-rank{rank}.json'

    def control(x):
        check(lib.modern_glm_model_control(ctx, x.numel(), ptr(x), stream()))

    def tp_control(x):
        check(lib.modern_glm_model_tp_control(ctx, x.numel(), ptr(x), stream()))

    lib.modern_glm_model_batch_cache_policy_r3068.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.modern_glm_model_batch_cache_stats_r3068.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
    def batch_stats3075():
        a=(ctypes.c_uint64*5)();check(lib.modern_glm_model_batch_cache_stats_r3068(ctx,a,5));return list(a)
    initial=batch_stats3075();assert initial==[0]*5,('not defaultOFF',initial)
    assert lib.modern_glm_model_batch_cache_policy_r3068(ctx,2)!=0 and batch_stats3075()==initial
    sentinel=(ctypes.c_uint64*5)(77,77,77,77,77)
    assert lib.modern_glm_model_batch_cache_stats_r3068(ctx,sentinel,4)!=0 and list(sentinel)==[77]*5
    import threading
    codes=[]
    wrong=threading.Thread(target=lambda:codes.append(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1)))
    wrong.start();wrong.join(20);assert not wrong.is_alive()and codes==[1]and batch_stats3075()==initial
    report.update(batch_cache_default_OFF_verified=True,batch_cache_wrong_thread_and_invalid_untouched=True)
    helper_maps={v.split()[-1]for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libglm_fp8_'in v};assert len(helper_maps)==2 and all('/candidates/glm-idle-trim-5f91ad7-r3771/helpers/'in v for v in helper_maps)
    report['actual_FP8_helpers']={v:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest()for v in helper_maps}
    report['Python_bytecode_cache_prefix']=_cache_sys.pycache_prefix;report['probe_preimport_before_CCL_verified']=True
    report['natural_input_sha256']=payload['natural_input_sha256'];report['natural_cases']=[]
    vocab=cfg['vocab_size'];barrier=torch.ones(1,device='cuda');control(barrier);assert barrier.item()==CARDS
    def chat_gate(epoch):
        run_chat_checks(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path)
        records=report.pop('chat_cases');assert len(records)==6
        assert all(records[i]['generated_ids']==records[i+1]['generated_ids'] for i in [0,2,4])
        report[epoch+'_raw_repeats_diagnostic'] = all(records[i]['first4_logits_sha256']==records[i+1]['first4_logits_sha256']for i in[0,2,4])if stage==PP-1 else None
        gate=torch.zeros(1,device='cuda')
        if stage==PP-1 and rank==0:
            source=r/'results'/f'glm-routed-fp8-chat-r20-{tag}-decoded.json'
            decoded=json.loads(source.read_text());assert decoded['known_answer_smoke_passed']and decoded['all_tail_rank0_repeats_exact']
            target=r/'results'/f'glm-fastpath-r3772-{tag}-{epoch}-decoded.json';assert not target.exists();source.replace(target);gate.fill_(1)
        control(gate);assert gate.item()==1
        report[epoch+'_chat_cases']=records;report[epoch+'_chat_gate_passed']=True;report_path.write_text(json.dumps(report,indent=2))

        frozen=json.loads((r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{3 if stage==PP-1 else 0}-rank{rank}.json').read_text());assert frozen['completed'];ref=frozen[epoch+'_chat_cases'];assert len(ref)==len(records)==6
        for actual,gold in zip(records,ref):
            assert actual['generated_ids']==gold['generated_ids'],('1186frozen999_QA',stage,rank,epoch)

    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,2)!=0
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0
    check(lib.modern_glm_model_indexer_policy_r998(ctx,1))
    check(lib.modern_glm_model_decode_leases_policy_r1356(ctx,1))
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0
    check(lib.modern_glm_model_feature_bypass_r1406(ctx,1))
    check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx,0));chat_gate('before_select');report['owner_affinity']=verify_owner_affinity(lib,rank);report['affinity_build_receipt']='glm-idle-trim-r3771-build.json';refs={}
    before_busy=batch_stats3075();assert lib.modern_glm_model_batch_cache_policy_r3068(ctx,1)!=0 and batch_stats3075()==before_busy;report['batch_cache_live_sequence_policy_rejected']=True
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0
    check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_feature_bypass_r1406(ctx,0))
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0
    check(lib.modern_glm_model_feature_bypass_r1406(ctx,1));check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1))
    assert lib.modern_glm_model_feature_bypass_r1406(ctx,0)!=0
    assert lib.modern_glm_model_indexer_policy_r998(ctx,0)!=0
    check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx,0))
    report['policy_invalid2_no_selection_no_feature_live_cache_and_disable_rejected']=True
    from glm_idle_probe_r3772 import run as run_cohorts
    run_cohorts(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,chat_gate)
    report['BLAS_policy']=json.loads(policy.modern_blas_policy_r243_json());assert report['BLAS_policy']['failures']==0 and all(v['created']==v['verified']for v in report['BLAS_policy']['owners'])
    report['cache_before_destroy']=cache_stats809(rank);check(lib.modern_glm_model_destroy(ctx));report['cache_after_destroy']=cache_stats809(rank);assert report['cache_after_destroy']['created']==report['cache_after_destroy']['destroyed']and report['cache_after_destroy']['active']==report['cache_after_destroy']['active_bytes']==0
    assert report['cohort_phases_completed']
    check(lib.modern_glm_model_shutdown());report.update(completed=True,worker_shutdown_completed=True,serving_target_qualified=False);report_path.write_text(json.dumps(report,indent=2))

def guarded(rank, payload):
    try:
        worker(rank, payload)
    except BaseException:
        traceback.print_exc()
        os._exit(2)


if __name__ == '__main__':
    directory = r / 'control'
    p = directory / f'glm-fastpath-r3772-control-{tag}-ids.json'
    if stage == 0:
        verified=json.loads((r/'results/glm-long-prompts-r21.json').read_text());assert verified['passed']
        print('GLM3323_BEFORE_UNIQUE',stage,tag,time.time(),flush=True)
        payload={'world':unique(),'tp':[unique()for _ in range(PP)],'control':unique(),'prompts':verified['prompts'],'eos_token_ids':verified['template_provenance']['eos_token_ids'],'feedback_host':os.environ['GLM_PIPELINE_TAIL_NODE'],'feedback_nonce':os.urandom(16).hex()}
        print('GLM3323_AFTER_UNIQUE',stage,tag,time.time(),flush=True)
        chat=json.loads((r/'results/glm-chat-template-r20.json').read_text());assert chat['passed']
        payload.update(chat_cases=chat['cases'],chat_eos_token_ids=chat['provenance']['eos_token_ids'],chat_template_provenance=chat['provenance'])
        natural_path=r/'results'/f'glm-serving-cases-r2130-{tag}.json';natural=json.loads(natural_path.read_text());payload.update(natural_cases=natural['cases'],natural_eos_token_ids=natural['eos_token_ids'],natural_input_sha256=hashlib.sha256(natural_path.read_bytes()).hexdigest())
        assert not p.exists()
        temporary = p.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload))
        temporary.replace(p)
        print('GLM3323_PAYLOAD_PUBLISHED',stage,tag,time.time(),flush=True)
    print('GLM3323_WAIT_PAYLOAD',stage,tag,time.time(),flush=True)
    begin = time.monotonic()
    while not p.exists():
        if time.monotonic() - begin > 1200:
            raise TimeoutError('GLM pipeline identities missing')
        time.sleep(.1)
    print('GLM3323_PAYLOAD_VISIBLE',stage,tag,time.time(),flush=True)
    mp.spawn(guarded, args=(json.loads(p.read_text()),), nprocs=8, join=True)
    print('GLM_PIPELINE_NODE_COMPLETE', stage, flush=True)
