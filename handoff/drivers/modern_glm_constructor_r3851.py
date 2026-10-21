import os as _cache_os,sys as _cache_sys
_cache_base=_cache_os.environ['PYTHONPYCACHEPREFIX'];assert _cache_base.startswith('/local/infini-pycache-')
_cache_sys.pycache_prefix=_cache_base+'/'+str(_cache_os.getpid())
from glm_capacity_probe_r3820 import run as _preloaded_probe_r3091
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
from glm_idle_chat_r3784 import run_chat_checks

import torch
import torch.multiprocessing as mp

r = pathlib.Path('/refactor')
# Reuse the exact raw-checkpoint loader and CAPI signatures. Do not execute the
# old diagnostic worker or its __main__ block.
baseline = (r / 'incoming/modern_glm_full_model_probe_r2.py').read_text()
ns = {}
exec(baseline[:baseline.index('def worker(rank,payload):')].replace('libmodern_glm_full_model_probe.so', '../candidates/glm-prefill-capacity-5f91ad7-r3822/libmodern_glm_prefill_capacity_r3822.so'), ns)
lib, ptr, check, unique, model, stage, tag = (ns[k] for k in ['lib', 'ptr', 'check', 'unique', 'model', 'stage', 'tag'])
ROWS=int(os.environ['GLM_CACHE_ROWS_R3627']);CAPACITY=int(os.environ['GLM_CACHE_CAP_R3627']);assert(ROWS,CAPACITY)==(32,640);PP=int(os.environ['GLM_PP_SIZE_R3603']);assert PP in[2,4];CARDS=8*PP;BOUNDS=[0,38,78]if PP==2 else[0,18,38,58,78]
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

def worker(rank,payload):
    assert PP==2;torch.cuda.set_device(rank);assert torch.cuda.mem_get_info(rank)[0]>58*1024**3
    assert lib.modern_glm_dense_mode_r48_set(mode)==0;stream=lambda:ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    cfg=json.loads((model/'config.json').read_text());buffers=[ctypes.create_string_buffer(base64.b64decode(payload[k]if k!='tp'else payload[k][stage]))for k in['world','tp','control']]
    tick=time.perf_counter();ctx=lib.modern_glm_model_create(rank,stage,ROWS,CAPACITY,json.dumps(cfg).encode(),*buffers)
    if not ctx:raise RuntimeError(lib.modern_glm_model_error().decode())
    path=r/'results'/f'glm-PP2-constructor-r3851-{tag}-stage{stage}-rank{rank}.json';assert not path.exists()
    report={'stage':stage,'rank':rank,'rows':ROWS,'capacity':CAPACITY,'PP':PP,'TP':8,'cards':CARDS,'step':tag,'completed':False,'weights_loaded':False,'performance_qualified':False,'scope':'Constructor-only actual16owner capacity/completeoriginal1MRoPE/normalclose probe. No checkpoint weights loaded or modeltoken/performance qualification.'}
    try:
        report['create_seconds']=time.perf_counter()-tick;report['free_total_after_create']=list(torch.cuda.mem_get_info(rank));report['original_RoPE_config']=[cfg['max_position_embeddings'],cfg['rope_parameters']];report['owner_affinity']=verify_owner_affinity(lib,rank)
        lib.modern_glm_shared_rope_stats_r3783.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];v=(ctypes.c_uint64*5)();check(lib.modern_glm_shared_rope_stats_r3783(ctx,v,5));report['shared_RoPE_stats']=list(v);assert list(v)[1]==2 and list(v)[4]==2
        report['local_8GiB_constructor_admitted']=report['free_total_after_create'][0]>8*1024**3;admission=torch.tensor([float(report['local_8GiB_constructor_admitted'])],device='cuda');check(lib.modern_glm_model_control(ctx,1,ptr(admission),stream()));report['admitted_ranks_out_of16']=int(admission.item());report['all16_memory_admitted']=admission.item()==16
        lib.modern_glm_shared_rope_live_r3783.restype=ctypes.c_uint64;check(lib.modern_glm_model_destroy(ctx));ctx=None;assert lib.modern_glm_shared_rope_live_r3783()==0;check(lib.modern_glm_model_shutdown());report['all_contexts_closed_without_loading']=True;report['free_total_after_shutdown']=list(torch.cuda.mem_get_info(rank));report['completed']=True
    finally:
        if ctx:check(lib.modern_glm_model_destroy(ctx));check(lib.modern_glm_model_shutdown())
        path.write_text(json.dumps(report,indent=2))


def guarded(rank, payload):
    try:
        worker(rank, payload)
    except BaseException:
        traceback.print_exc()
        os._exit(2)


if __name__ == '__main__':
    directory = r / 'control'
    p = directory / f'glm-constructor-r3851-control-{tag}-ids.json'
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
