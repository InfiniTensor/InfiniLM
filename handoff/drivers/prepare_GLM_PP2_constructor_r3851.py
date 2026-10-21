import pathlib,ast
r=pathlib.Path(__file__).resolve().parent
s=(r/'modern_glm_capacity_r3820.py').read_text(encoding='utf8').replace("assert(ROWS,CAPACITY)in[(16,640)]","assert(ROWS,CAPACITY)==(32,640)")
start=s.index('def worker(rank, payload):');end=s.index('\ndef guarded(rank, payload):',start)
worker='''def worker(rank,payload):
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

'''
s=s[:start]+worker+s[end:];s=s.replace('glm-fastpath-r3820-control-','glm-constructor-r3851-control-');ast.parse(s);(r/'modern_glm_constructor_r3851.py').write_text(s,encoding='utf8',newline='\n')
s=(r/'glm_capacity_r3820_inside.sh').read_text(encoding='utf8').replace('modern_glm_capacity_r3820.py','modern_glm_constructor_r3851.py');(r/'glm_constructor_r3851_inside.sh').write_text(s,encoding='utf8',newline='\n');print('PP2 rows32 constructor-only, no checkpoint loading,8GiB gate and16normalowner closes prepared.')
