import os,sys
assert os.environ['PYTHONPYCACHEPREFIX'].startswith('/local/')
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import ctypes as C,datetime,hashlib,json,pathlib,socket,time
import numpy as np
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];base=r/'candidates/kimi-wide-replay-3e0d89fe-r3409';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def library():
 lib=C.CDLL(str(base/'libkimi_projection_r3409.so'));lib.kimi_projection_r3409_error.restype=C.c_char_p;lib.kimi_projection_r3409_id_size.restype=C.c_size_t;lib.kimi_projection_r3409_id.argtypes=[C.c_void_p];lib.kimi_projection_r3409_create.argtypes=[C.c_int]*3+[C.c_void_p];lib.kimi_projection_r3409_create.restype=C.c_void_p;lib.kimi_projection_r3409_allreduce.argtypes=[C.c_void_p,C.c_int,C.c_size_t]+[C.c_void_p]*3;lib.kimi_projection_r3409_destroy.argtypes=[C.c_void_p];lib.kimi_projection_r3409_gemm.argtypes=[C.c_void_p,C.c_size_t,C.c_int]+[C.c_void_p]*4;lib.kimi_projection_r3409_frame.argtypes=[C.c_void_p,C.c_size_t,C.c_int];lib.modern_kimi_reduce_policy_r3226.argtypes=[C.c_int];lib.modern_kimi_reduce_stats_r3226.argtypes=[C.c_int,C.POINTER(C.c_uint64),C.c_size_t];return lib
def worker(rank,uid):
 torch.set_num_threads(1);torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));mask=cpus[rank*4:rank*4+4];assert len(mask)==4;os.sched_setaffinity(0,set(mask));stream=torch.cuda.Stream(device=rank);torch.cuda.set_stream(stream);lib=library();ctx=None
 p=r/'results'/f'kimi-projection-r3409-{tag}-rank{rank}.json';assert not p.exists();q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'rank':rank,'CPU_affinity':sorted(os.sched_getaffinity(0)),'cases':[],'helper_sha256':sha(base/'libkimi_projection_r3409.so'),'source_sha256':sha(pathlib.Path(__file__)),'scope':'IntegratedCommon3393 Mode2 BF16->F64 policy with8realCCLranks, defaultOFF+busy/nested+realrequestB1guard+actualpromotioncounters. RealKimi firstlayerKDA gated andcheckpoint o_proj replay through currentSDKGemm andcurrentLMRuntimeCCL. CompareBF16 native reduction vsF32reduction aftersame BF16localGemm. Bothpackedlayouts; B4/B8 realrecords, B16 tiledB8gated stress only. Fullmodeltracebytecompat recorded, ownrepeats/readonly/FP64samples andnormalCCLclose; no wholemodel speed orB8fixadoption.'}
 def save():
  t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
 def ck(code):
  if code:raise RuntimeError(lib.kimi_projection_r3409_error().decode())
 def raw(t):return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def record(name,t):
  blob=raw(t);fn=f'kimi-projection-r3409-{tag}-rank{rank}-{name}.bin';out=r/'results'/fn;assert not out.exists();out.write_bytes(blob);return{'file':fn,'shape':list(t.shape),'dtype':str(t.dtype),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}
 save()
 try:
  assert torch.cuda.mem_get_info(rank)[0]>56*1024**3
  proof=json.loads((r/'results/kimi-row-boundary-r3153-audit.json').read_text());assert proof['completed']and proof['all_trace_vs_plain_raw_exact'];run=json.loads((r/'models-latest-r3118-queue.json').read_text())['rows'][0]['run'];folder=r/'results'/f'kimi-row-boundary-r3117-{run}'
  mapping=json.loads((pathlib.Path('/shared/Kimi-K3')/'model.safetensors.index.json').read_text())['weight_map'];keys=[k for k in mapping if'.layers.0.'in k and k.endswith('o_proj.weight')];assert len(keys)==1,keys;key=keys[0]
  with safe_open(pathlib.Path('/shared/Kimi-K3')/mapping[key],framework='pt',device='cpu')as f:whole=f.get_tensor(key)
  assert whole.dtype==torch.bfloat16 and whole.shape==(7168,12288)
  weight=whole[:,rank*1536:(rank+1)*1536].contiguous().to(rank);weight_before=raw(weight);q['weight']={'key':key,'global_shape':list(whole.shape),'K_slice':[rank*1536,(rank+1)*1536],'sha256':hashlib.sha256(weight_before).hexdigest()};del whole
  buf=C.create_string_buffer(uid);ctx=lib.kimi_projection_r3409_create(rank,8,rank,buf);assert ctx,lib.kimi_projection_r3409_error()
  for m in[4,8,16]:
   source_m=4 if m==4 else 8;directory=folder/(f'trace_B{source_m}_stage0-step1');journals=list(directory.glob(f'stage0-rank{rank}-layer0-kda-call*.jsonl'));assert len(journals)==1;rows=[json.loads(s)for s in journals[0].read_text().splitlines()];gated=next(v for v in rows if v['point']=='gated');native=next(v for v in rows if v['point']=='output');b=(directory/gated['file']).read_bytes();arr=np.frombuffer(b,np.uint16).copy();x=torch.from_numpy(arr).view(torch.bfloat16).reshape(source_m,1536)
   if m==16:x=x[:1].repeat(16,1)
   x=x.to(rank);xbefore=raw(x);assert all(torch.equal(x[i],x[0])for i in range(m));recorded=(directory/native['file']).read_bytes()if m==source_m else None
   for packed in[0,1]:
    w=weight if not packed else weight.T.contiguous();local_guard=torch.full((m*7168*2+128,),0x37,device=rank,dtype=torch.uint8);local=local_guard[64:-64].view(torch.bfloat16).reshape(m,7168)
    def project():ck(lib.kimi_projection_r3409_gemm(ctx,m,packed,x.data_ptr(),w.data_ptr(),local.data_ptr(),stream.cuda_stream))
    project();first=raw(local);assert bool(torch.isfinite(local).all());repeat=[]
    for _ in range(8):project();repeat.append(raw(local)==first)
    ts=sorted({0,m//2,m-1});rs=[0,3584,7167];a=x[ts].double().cpu().numpy();b=weight[rs].double().cpu().numpy();oracle=a@b.T;actual=local[ts][:,rs].double().cpu().numpy();err=float(np.linalg.norm(actual-oracle)/max(1e-12,np.linalg.norm(oracle)))
    c={'M':m,'packed':packed,'input_trace':gated,'trace_directory':str(directory),'local_repeat8_exact':all(repeat),'local_duplicate_rows_raw_exact':all(raw(row)==raw(local[0])for row in local),'local_CPU_FP64_sample_NRMSE':err,'local_raw':record(f'B{m}-p{packed}-local',local),'input_raw':record(f'B{m}-p{packed}-input',x),'weight_sample_raw':record(f'B{m}-p{packed}-weightrows',weight[rs]),'sample_output_raw':record(f'B{m}-p{packed}-sample',local[ts][:,rs]),'epochs':[]}
    yg=torch.full((m*7168+64,),0x3555,device=rank,dtype=torch.uint16);y=yg[32:-32].view(torch.bfloat16).reshape(m,7168);xf=local.float();yf=torch.full_like(local,float('nan'),dtype=torch.float32);by_mode={}
    # Same packed M, policyON but one real request: BF16 path must remain unchanged.
    ck(lib.modern_kimi_reduce_policy_r3226(2));ck(lib.kimi_projection_r3409_frame(ctx,1,0))
    assert lib.modern_kimi_reduce_policy_r3226(0)==2
    assert lib.kimi_projection_r3409_frame(ctx,1,0)!=0
    ck(lib.kimi_projection_r3409_allreduce(ctx,2,local.numel(),local.data_ptr(),y.data_ptr(),stream.cuda_stream))
    ck(lib.kimi_projection_r3409_frame(ctx,0,1));b1raw=raw(y)
    for epoch,mode in enumerate([0,2,2,0]):
     ck(lib.modern_kimi_reduce_policy_r3226(mode));ck(lib.kimi_projection_r3409_frame(ctx,m,0))
     startstats=(C.c_uint64*5)();ck(lib.modern_kimi_reduce_stats_r3226(rank,startstats,5))
     before=time.perf_counter();hashes=[]
     for loop in range(8):
      ck(lib.kimi_projection_r3409_allreduce(ctx,2,local.numel(),local.data_ptr(),y.data_ptr(),stream.cuda_stream))
      stream.synchronize();hashes.append(hashlib.sha256(raw(y)).hexdigest())
     ck(lib.kimi_projection_r3409_frame(ctx,0,1))
     endstats=(C.c_uint64*5)();ck(lib.modern_kimi_reduce_stats_r3226(rank,endstats,5));assert endstats[1]-startstats[1]==8*bool(mode) and endstats[2]-startstats[2]==8*bool(mode)*local.numel()*8
     blob=raw(y);by_mode.setdefault(mode,blob);assert by_mode[mode]==blob
     if not mode:assert blob==b1raw
     c['epochs'].append({'epoch':epoch,'mode':'F64_sum_BF16_final'if mode else'BF16_sum','wall_ms_per_call_castinclusive':(time.perf_counter()-before)*1000/8,'own_repeat8_exact':len(set(hashes))==1,'duplicate_rows_fullraw_exact':all(raw(v)==raw(y[0])for v in y),'model_trace_output_exact':blob==recorded if recorded is not None else None,'raw':record(f'B{m}-p{packed}-epoch{epoch}',y)})
    assert raw(x)==xbefore and raw(weight)==weight_before and bool((local_guard[:64]==0x37).all()and(local_guard[-64:]==0x37).all())and bool((yg[:32]==0x3555).all()and(yg[-32:]==0x3555).all());c['readonly_and_output_guards']=True;q['cases'].append(c);save();print('KIMI_PROJECTION_REPLAY_R3233_CASE',rank,m,packed,err,flush=True)
  policy=C.CDLL(None);policy.modern_blas_policy_r243_json.restype=C.c_char_p;q['BLAS_policy']=json.loads(policy.modern_blas_policy_r243_json());assert q['BLAS_policy']['failures']==0 and all(v['created']==v['verified']for v in q['BLAS_policy']['owners']);ck(lib.modern_kimi_reduce_policy_r3226(0));q['integrated_Common_request_policy_and_actual_counters_passed']=True;q['numeric_collection_before_close']=True
 except BaseException as e:q['error']=repr(e);save();raise
 finally:
  if ctx:ck(lib.kimi_projection_r3409_destroy(ctx));q['communicator_normally_closed']=True
  q['completed']=bool(q.get('numeric_collection_before_close'));save()
if __name__=='__main__':
 assert torch.cuda.device_count()==8;lib=library();buf=C.create_string_buffer(lib.kimi_projection_r3409_id_size());assert lib.kimi_projection_r3409_id(buf)==0;mp.spawn(worker,args=(buf.raw,),nprocs=8,join=True);print('KIMI_PROJECTION_REPLAY_R3233_ALL8_CLOSED',flush=True)
