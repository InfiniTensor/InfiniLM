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
    report_path = r / 'results' / f'glm-fp8-cached-model-r928-{tag}-stage{stage}-rank{rank}.json'

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
            target=r/'results'/f'glm-fp8-cached-model-r928-{tag}-{epoch}-decoded.json';assert not target.exists();source.replace(target);gate.fill_(1)
        control(gate);assert gate.item()==1
        report[epoch+'_chat_cases']=records;report[epoch+'_chat_gate_passed']=True;report_path.write_text(json.dumps(report,indent=2))

    chat_gate('before_performance')
    reference_ids,reference_hashes=None,None
    reference8_ids,reference8_hashes=None,None
    previous = json.loads((r / 'results/modern-glm78-initial-summary.json').read_text())
    for phase, batch, micro, pipeline, steps_in_phase in PHASES:
        phase_mode=0 if phase.startswith('reference_')else mode
        assert lib.modern_glm_dense_mode_r48_set(phase_mode)==0
        check(lib.modern_glm_model_reset(ctx))
        directory = r / 'control' / f'glm-fp8-cached-model-r928-control-{tag}-{phase}'
        barrier.fill_(1)
        control(barrier)
        generated = [[] for _ in range(batch)] if stage in (0, 3) or not pipeline else None
        per_row_hashes = [[] for _ in range(batch)] if stage == 3 else None
        events, completion_times, first_token_times = [], [], [None] * batch
        start = time.perf_counter()
        for step in range(steps_in_phase):
            active_micro=min(micro,8) if step==0 else micro
            for q in range(batch // active_micro):
                qstart = time.perf_counter()
                offset = q * active_micro
                if pipeline and step > 0 and stage == 0:
                    emitted = torch.tensor(wait_feedback(directory / f'q{q}-s{step-1}.json', active_micro), dtype=torch.float32, device='cuda') if rank == 0 else torch.zeros(active_micro, device='cuda')
                    tp_control(emitted)
                    for i, value in enumerate(emitted.cpu().tolist()):
                        generated[offset + i].append(int(value))
                past, width = (0, length) if step == 0 else (length + step - 1, 1)
                if step == 0:
                    current = sum((prompts*((active_micro+3)//4))[:active_micro],[])
                elif stage == 0 or not pipeline:
                    current = [generated[offset + i][-1] for i in range(active_micro)]
                else:
                    # Intermediate/last stages receive hidden states from PP.
                    # IDs are validated but never embedded on these stages.
                    current = [0] * active_micro
                ids = torch.tensor(current, device='cuda', dtype=torch.int64)
                pos = torch.arange(past, past + width, device='cuda', dtype=torch.int64).repeat(active_micro)
                requests = [(offset + i, past, width) for i in range(active_micro)]
                desc = (ctypes.c_size_t * (3 * active_micro))(*[v for req in requests for v in req])
                logits = torch.empty((active_micro, vocab), device='cuda', dtype=torch.bfloat16)
                f0 = time.perf_counter()
                check(lib.modern_glm_model_forward(ctx, len(ids), active_micro, desc, ptr(ids), ptr(pos), ptr(logits), 0, stream()))
                forward = time.perf_counter() - f0
                native_profile = list(lib.modern_glm_pipeline_last_profile()[:6])
                if stage == 3:
                    assert torch.isfinite(logits).all()
                    if step < 4:
                        raw = logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        if rank==0:(r/'results'/f'glm-fp8-cached-model-r928-{tag}-{phase}-q{q}-s{step}-bf16.bin').write_bytes(raw)
                        row_bytes = vocab * 2
                        for i in range(active_micro):
                            per_row_hashes[offset + i].append(hashlib.sha256(raw[i * row_bytes:(i + 1) * row_bytes]).hexdigest())
                    next_ids = logits.argmax(-1).cpu().tolist()
                    if pipeline:
                        for i, value in enumerate(next_ids):
                            generated[offset + i].append(int(value))
                        if rank == 0:
                            connection.sendall((json.dumps({'phase': phase, 'q': q, 'step': step, 'ids': next_ids}) + '\n').encode())
                if not pipeline:
                    emitted = torch.tensor(next_ids, dtype=torch.float32, device='cuda') if stage == 3 and rank == 0 else torch.zeros(active_micro, device='cuda')
                    control(emitted)
                    for i, value in enumerate(emitted.cpu().tolist()):
                        generated[offset + i].append(int(value))
                if step == 0 and stage == 3:
                    for i in range(active_micro):
                        first_token_times[offset + i] = time.perf_counter() - start
                events.append({'step': step, 'microbatch': q, 'forward_seconds': forward, 'micro_wall_seconds': time.perf_counter() - qstart,
                               'native_incoming_seconds': native_profile[0], 'native_layers_seconds': native_profile[1], 'native_outgoing_seconds': native_profile[2],
                               'gpu_incoming_seconds': native_profile[3], 'gpu_layers_seconds': native_profile[4], 'gpu_outgoing_seconds': native_profile[5]})
            completion_times.append(time.perf_counter() - start)
            if rank == 0:
                print('GLM_PIPELINE_STEP', stage, phase, step, round(completion_times[-1], 6), flush=True)
        elapsed = time.perf_counter() - start
        barrier.fill_(1)
        control(barrier)
        if pipeline and stage == 0:
            for q in range(batch // micro):
                values = wait_feedback(directory / f'q{q}-s{steps_in_phase-1}.json', micro) if rank == 0 else [0] * micro
                emitted = torch.tensor(values, dtype=torch.float32, device='cuda')
                tp_control(emitted)
                for i, value in enumerate(emitted.cpu().tolist()):
                    generated[q * micro + i].append(int(value))
        if generated is not None:
            assert all(len(values) == steps_in_phase for values in generated)
        frozen = json.loads((r / 'results/modern-glm-pipeline-r16-r4-summary.json').read_text())
        direct_calls=lib.modern_glm_direct_calls_r601(rank);assert direct_calls>0
        report['cases'].append({'actual_direct_decode_calls':direct_calls,'phase_mode':phase_mode,'phase': phase, 'global_batch': batch, 'micro_batch': micro, 'prefill_micro_batch':min(micro,8), 'pipeline': pipeline,
                                'input_tokens_per_request': length, 'output_tokens_per_request': steps_in_phase,
                                'generated_ids': generated, 'per_row_first4_logits_sha256': per_row_hashes,
                                'elapsed_seconds': elapsed, 'step_completion_seconds': completion_times,
                                'first_token_seconds_by_row': first_token_times if stage == 3 else None,
                                'events': events, 'free_total_gpu_bytes_after': torch.cuda.mem_get_info()})
        report_path.write_text(json.dumps(report, indent=2))
        refs=report['same_variant_references'];key=str(micro)
        if phase.startswith('reference_')and phase.endswith('_r1'):
            refs[key]={'generated_ids':generated,'hashes':per_row_hashes}
        assert key in refs
        expected=refs[key]['generated_ids'];expected_hashes=refs[key]['hashes']
        if generated is not None:assert generated==[expected[i%micro]for i in range(batch)],('781 ownfullIDs differ',phase,stage,rank)
        if stage==3:assert per_row_hashes==[expected_hashes[i%micro]for i in range(batch)],('781 ownraw logits differ',phase,rank)
        report['cases'][-1]['same_variant_full_IDs_first4_logits_exact']=True
        frozen928=json.loads((r/'results/glm-fp8-cached-model-r811-910375.139-stage3-rank0.json').read_text());assert frozen928['completed'];old928=next(v for v in frozen928['cases']if v['phase']==phase)
        if generated is not None:assert generated==old928['generated_ids'],('928frozen811fullIDs',phase,stage,rank)
        if stage==3:assert per_row_hashes==old928['per_row_first4_logits_sha256'],('928frozen811raw',phase,rank)
        report['cases'][-1].update(frozen811_full_IDs_raw_exact=True,cache_view_calls=int(lib.modern_glm_cache_view_calls_r927(rank,0)),cache_copy_calls=int(lib.modern_glm_cache_view_calls_r927(rank,1)))
        assert report['cases'][-1]['cache_view_calls']>0
        if generated is not None:report['cases'][-1]['old_frozen98_IDs_exact_diagnostic']=generated==[frozen8['generated_ids'][i%8][:steps_in_phase]for i in range(batch)]
        if stage==3:report['cases'][-1]['old_frozen98_raw_exact_diagnostic']=per_row_hashes==[frozen8['per_row_first4_logits_sha256'][i%8]for i in range(batch)]
        report['cases'][-1]['fp8_batched_calls']=int(lib.modern_glm_fp8_calls_r780(rank));assert report['cases'][-1]['fp8_batched_calls']>0
        counts=json.loads(policy.modern_blas_policy_r243_json());assert counts['failures']==0 and counts['owners']and all(v['created']==v['verified']for v in counts['owners'])and any('libglm_fp8_batched_r776.so'in v['caller']for v in counts['owners'])
        report['cases'][-1]['BLAS_policy']=counts
        cstats=cache_stats809(rank);assert cstats['active']<=10 and cstats['active_bytes']<=192*1024**2 and cstats['invalid']==0
        if micro in [8,16]:assert cstats['runs']>0 and cstats['hits']>0
        report['cases'][-1]['cache_stats']=cstats

        report['cases'][-1]['first_eos_index_by_request']=[next((j for j,t in enumerate(ids)if t in payload['eos_token_ids']),None)for ids in generated]if generated is not None else None
        report_path.write_text(json.dumps(report, indent=2))
        print('GLM_PIPELINE_PHASE_COMPLETE', stage, rank, phase, elapsed, flush=True)
    chat_gate('after_performance')
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
    p = directory / f'glm-fp8-cached-model-r928-control-{tag}-ids.json'
    if stage == 0:
        verified=json.loads((r/'results/glm-long-prompts-r21.json').read_text());assert verified['passed']
        payload={'world':unique(),'tp':[unique()for _ in range(4)],'control':unique(),'prompts':verified['prompts'],'eos_token_ids':verified['template_provenance']['eos_token_ids'],'feedback_host':os.environ['GLM_PIPELINE_TAIL_NODE'],'feedback_nonce':os.urandom(16).hex()}
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
