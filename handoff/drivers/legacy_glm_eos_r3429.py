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
from glm_native_chat_checks_r20 import run_chat_checks

import torch
import torch.multiprocessing as mp

r = pathlib.Path('/refactor')
# Reuse the exact raw-checkpoint loader and CAPI signatures. Do not execute the
# old diagnostic worker or its __main__ block.
baseline = (r / 'incoming/modern_glm_full_model_probe_r2.py').read_text()
ns = {}
exec(baseline[:baseline.index('def worker(rank,payload):')].replace('libmodern_glm_full_model_probe.so', '../candidates/glm-cache-view-r927/libmodern_glm_cache_view_r927.so'), ns)
lib, ptr, check, unique, model, stage, tag = (ns[k] for k in ['lib', 'ptr', 'check', 'unique', 'model', 'stage', 'tag'])
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
policy=ctypes.CDLL(None);policy.modern_blas_policy_r243_json.restype=ctypes.c_char_p
adoption=json.loads((r/'results/glm-balanced-micro8-r153-lock.json').read_text())
mode=adoption['mode'];assert mode in [0,2]

PHASES = [
 ('reference_g4_m4_r1',4,4,False,16),
 ('reference_g4_m4_r2',4,4,False,16),
 ('reference_g8_m8_r1',8,8,False,512),
 ('reference_g8_m8_r2',8,8,False,512),
 ('long_pipeline_g32_m8_r1',32,8,True,512),
 ('long_pipeline_g32_m8_r2',32,8,True,512),
 ('reference_g16_m16_r1',16,16,False,512),
 ('reference_g16_m16_r2',16,16,False,512),
 ('long_pipeline_g64_m16_r1',64,16,True,512),
 ('long_pipeline_g64_m16_r2',64,16,True,512),
]


def worker(rank, payload):
    torch.cuda.set_device(rank)
    assert lib.modern_glm_dense_mode_r48_set(mode)==0
    stream = lambda: ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    cfg = json.loads((model / 'config.json').read_text())
    buffers = [ctypes.create_string_buffer(base64.b64decode(payload[k] if k != 'tp' else payload[k][stage])) for k in ['world', 'tp', 'control']]
    ctx = lib.modern_glm_model_create(rank, stage, 64, 1024, json.dumps(cfg).encode(), *buffers)
    if not ctx:
        raise RuntimeError(lib.modern_glm_model_error().decode())
    report={'cache_rows':64,'micro8_reference_scope':'Ownnew785projectionvariant M8 andM16full512refs, ownrepeats andpipeline eachgeometry strict. Old98diagnostic.','scope':'Actual full78 GLM TP8PP4, mode0/2 exact full64-logit gate against original, SDK58735. Four512-input prompts repeated across micro8/global32. Frozen original micro4 prefix16 gate; original mode0/micro8 serial512 twice establishes same-shape512 reference; mode2 pipelined32/micro8 must match it at64/512 twice. Cross32 repetitions required. Six finite EOS-aware answers both before and after performance. Fixed outputs may contain postEOS diagnostics. Excludes loading/tokenizer/HTTP, broad independent quality, ordinary Engine and maximumTPS.', 'mode':mode,'step':tag,'stage':stage,'rank':rank,'cases':[],'completed':False}
    # The existing loader is executed verbatim and updates this report.
    loader = baseline[baseline.index(" sdk=sorted("):baseline.index(' def control(x):')].replace('hygon-runtime-fixes-r1', 'hygon-fp8-expert-r19')
    wrapped = 'def load(ctx, report):\n' + loader + '\n return report\n'
    loader_ns = dict(ns)
    loader_ns.update(stage=stage, rank=rank, report=report, ctx=ctx)
    exec(wrapped, loader_ns)
    loader_ns['load'](ctx, report)
    keys=set(json.loads(lib.modern_glm_model_keys(ctx)))
    layers=sorted({int(k.split('.')[2])for k in keys if k.startswith('model.layers.')});bounds=[0,18,38,58,78]
    assert layers==list(range(bounds[stage],bounds[stage+1]));report['pipeline_boundaries']=bounds;report['actual_layer_indices']=layers
    frozen8_report=json.loads((r/'results/glm-micro8-r98-910376.50-stage3-rank0.json').read_text())
    assert frozen8_report['completed'] and frozen8_report['worker_shutdown_completed']
    frozen8=next(c for c in frozen8_report['cases']if c['phase']=='original_g8_m8_r1')

    sdk_paths=sorted({line.split()[-1]for line in pathlib.Path('/proc/self/maps').read_text().splitlines()if 'libinfiniops.so'in line})
    assert len(sdk_paths)==1 and '/install/hygon-fp8-expert-r19/lib/' in sdk_paths[0]
    report['actual_sdk_sha256']=hashlib.sha256(pathlib.Path(sdk_paths[0]).read_bytes()).hexdigest()
    assert report['actual_sdk_sha256']=='58735f6730d7092efe8cfb44cfd57170d36c0be31d89b3145a79a3504477ae7f'
    report['library_sha256'] = hashlib.sha256((r / 'build/../candidates/glm-cache-view-r927/libmodern_glm_cache_view_r927.so').read_bytes()).hexdigest()
    assert report['library_sha256']==json.loads((r/'results/glm-cache-view-r927-build.json').read_text())['library_sha256']
    report.update(old_precision_compatibility_qualified=False, numerical_variant='780_per128K_BF16_GEMM_F32dots_Mge32', component_report='glm-fp8-batched-r776-summary.json')
    report['scope']='786 full78 GLM TP8PP4, sameSDK587/FFN19/cache605/direct601/pair544/dense48, onlyattention FP8 projections M>=32 use776 newdot reductionorder. Plancreation/destruction included; decode<32 unchanged. OwnM4prefix16/M8full512/M16full512 twice andG32/G64pipeline mustmatchownrefs/fullraw/all32. Frozenoldfull512 isdiagnostic, beforeafter3EOSknownanswersx2 strict. Fixed512 mayinclude postEOS; noHTTP/broadquality/hardwaremax/aggregate.'
    report['scope']='811 full78GLM TP8PP4:809adds807boundedM8/16workspacecache, currentweightsunpackedpercall, M1/M4legacy andM>=32same790. Newdecode dotreductionorder needsownM8/M16full512refs/repeats/G32G64pipeline.32rankbeforeafterknownQA/raw/CPUargmax/BLASpolicy/cachecountersandownerrelease beforeRuntime/comm teardown strict. Old98diagnostic; nohardwaremax orcomponentgain==modelgain.'
    report['cache_library_sha256']=json.loads((r/'results/glm-cache-view-r927-build.json').read_text())['cache_library_sha256']
    report['same_variant_references']={}
    report['scope']='928full78TP8PP4 same809/SDK587/807/776math, only927zero-copyKVviews whenvalidatedlogicalselection equalswholevalidprefix. Subsetfallback, metadata/poison/reset/ownerlifetimes retained. All10phases/frozen811full512IDs/raw andownrepeats/32rankknownbeforeafterQA/cachecleardestroy/shutdown strict. No newprecisionvariant orcomponent-to-modelgain assumption.' 
    report['fp8_helper_sha256']=hashlib.sha256((r/'candidates/glm-fp8-batched-r776/libglm_fp8_batched_r776.so').read_bytes()).hexdigest()
    assert report['fp8_helper_sha256']==json.loads((r/'results/glm-cache-view-r927-build.json').read_text())['helper_sha256']
    report_path = r / 'results' / f'legacy-glm-eos-r3429-{tag}-stage{stage}-rank{rank}.json'

    def control(x):
        check(lib.modern_glm_model_control(ctx, x.numel(), ptr(x), stream()))

    def tp_control(x):
        check(lib.modern_glm_model_tp_control(ctx, x.numel(), ptr(x), stream()))

    connection = None
    reader = None

    def wait_feedback(path, count):
        match=re.fullmatch(r'q(\d+)-s(\d+)\.json',path.name)
        assert match and stage==0 and rank==0 and reader is not None
        requested_q,previous_step=map(int,match.groups())
        previous_micro=min(micro,8) if previous_step==0 else micro
        assert count%previous_micro==0
        parts=count//previous_micro;value=[]
        for part in range(parts):
            packet=json.loads(reader.readline())
            expected_q=requested_q*parts+part
            assert packet['phase']==phase and packet['q']==expected_q and packet['step']==previous_step,packet
            assert len(packet['ids'])==previous_micro and all(0<=int(v)<cfg['vocab_size']for v in packet['ids'])
            value.extend(packet['ids'])
        assert len(value)==count
        return value

    barrier = torch.ones(1, device='cuda')
    control(barrier)
    assert barrier.item() == 32
    address = (os.environ['INFINILM_PIPELINE_FEEDBACK_ADDRESS'],
               24000 + (int(os.environ['SLURM_JOB_ID']) * 97 + int(os.environ['SLURM_STEP_ID'])) % 8000)
    report['feedback_address'] = list(address)
    report['feedback_bootstrap'] = 'verified_ib0_address_and_step_derived_port_no_filesystem_discovery'
    if stage == 3 and rank == 0:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.settimeout(180)
        listener.bind(address)
        listener.listen(1)
        print('FEEDBACK_LISTENER_READY', stage, address, flush=True)
        connection, peer_address = listener.accept()
        listener.close()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(300)
        reader = connection.makefile('rb')
        assert json.loads(reader.readline()) == {'nonce': payload['feedback_nonce']}
        connection.sendall((json.dumps({'accepted': payload['feedback_nonce']}) + '\n').encode())
    elif stage == 0 and rank == 0:
        begin = time.monotonic()
        while True:
            try:
                connection = socket.create_connection(address, timeout=10)
                break
            except ConnectionRefusedError:
                if time.monotonic() - begin > 180:
                    raise
                time.sleep(.05)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(300)
        reader = connection.makefile('rb')
        connection.sendall((json.dumps({'nonce': payload['feedback_nonce']}) + '\n').encode())
        assert json.loads(reader.readline()) == {'accepted': payload['feedback_nonce']}
        print('FEEDBACK_CONNECTED', stage, address, flush=True)

    report['feedback_transport'] = 'persistent_tcp'
    report['feedback_filesystem_rendezvous_outside_timing'] = False
    report['feedback_rendezvous_no_filesystem'] = True
    prompts = payload['prompts']
    assert len(prompts)==4 and all(len(p)==512 for p in prompts)
    length, vocab = 512, cfg['vocab_size']
    def chat_gate(epoch):
        run_chat_checks(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path)
        records=report.pop('chat_cases');assert len(records)==6
        assert all(records[i]['generated_ids']==records[i+1]['generated_ids'] for i in [0,2,4])
        if stage==3:assert all(records[i]['first4_logits_sha256']==records[i+1]['first4_logits_sha256'] for i in [0,2,4])
        gate=torch.zeros(1,device='cuda')
        if stage==3 and rank==0:
            source=r/'results'/f'glm-routed-fp8-chat-r20-{tag}-decoded.json'
            decoded=json.loads(source.read_text());assert decoded['known_answer_smoke_passed']and decoded['all_tail_rank0_repeats_exact']
            target=r/'results'/f'legacy-glm-eos-r3429-{tag}-{epoch}-decoded.json';assert not target.exists();source.replace(target);gate.fill_(1)
        control(gate);assert gate.item()==1
        report[epoch+'_chat_cases']=records;report[epoch+'_chat_gate_passed']=True;report_path.write_text(json.dumps(report,indent=2))

    chat_gate('before_performance')
    from legacy_glm_eos_probe_r3429 import run
    run(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path,chat_gate)
    if reader is not None:
        reader.close()
    if connection is not None:
        connection.close()
    report['cache_before_destroy']=cache_stats809(rank)
    check(lib.modern_glm_model_destroy(ctx))
    report['cache_after_destroy']=cache_stats809(rank);assert report['cache_after_destroy']['created']==report['cache_after_destroy']['destroyed']and report['cache_after_destroy']['active']==report['cache_after_destroy']['active_bytes']==0
    check(lib.modern_glm_model_shutdown())
    report.update(completed=True, worker_shutdown_completed=True, complete_model_quality_qualified=False, maximum_model_tps_established=False)
    report_path.write_text(json.dumps(report, indent=2))


def guarded(rank, payload):
    try:
        worker(rank, payload)
    except BaseException:
        traceback.print_exc()
        os._exit(2)


if __name__ == '__main__':
    directory = r / 'control'
    p = directory / f'legacy-glm-eos-r3429-control-{tag}-ids.json'
    if stage == 0:
        verified=json.loads((r/'results/glm-long-prompts-r21.json').read_text());assert verified['passed']
        payload={'world':unique(),'tp':[unique()for _ in range(4)],'control':unique(),'prompts':verified['prompts'],'eos_token_ids':verified['template_provenance']['eos_token_ids'],'feedback_host':os.environ['GLM_PIPELINE_TAIL_NODE'],'feedback_nonce':os.urandom(16).hex()}
        natural=json.loads((r/'results'/f'glm-serving-cases-r1186-{tag}.json').read_text());payload.update(natural_cases=natural['cases'],natural_eos_token_ids=natural['eos_token_ids'],natural_input_provenance=natural)
        chat=json.loads((r/'results/glm-chat-template-r20.json').read_text());assert chat['passed']
        payload.update(chat_cases=chat['cases'],chat_eos_token_ids=chat['provenance']['eos_token_ids'],chat_template_provenance=chat['provenance'])
        assert not p.exists()
        temporary = p.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload))
        temporary.replace(p)
    begin = time.monotonic()
    while not p.exists():
        if time.monotonic() - begin > 300:
            raise TimeoutError('GLM pipeline identities missing')
        time.sleep(.1)
    mp.spawn(guarded, args=(json.loads(p.read_text()),), nprocs=8, join=True)
    print('GLM_PIPELINE_NODE_COMPLETE', stage, flush=True)
