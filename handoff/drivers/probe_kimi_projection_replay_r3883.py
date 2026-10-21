import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import ctypes as C,datetime,hashlib,json,pathlib,time,traceback
import numpy as np
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');base=r/'candidates/kimi-KDA1-replay-5f91ad7-r3883';tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def library():
 l=C.CDLL(str(base/'libkimi_projection_r3883.so'));l.kimi_projection_r3883_error.restype=C.c_char_p;l.kimi_projection_r3883_id_size.restype=C.c_size_t;l.kimi_projection_r3883_id.argtypes=[C.c_void_p];l.kimi_projection_r3883_create.argtypes=[C.c_int]*3+[C.c_void_p];l.kimi_projection_r3883_create.restype=C.c_void_p;l.kimi_projection_r3883_gemm.argtypes=[C.c_void_p,C.c_size_t,C.c_int]+[C.c_void_p]*4;l.kimi_projection_r3883_allreduce.argtypes=[C.c_void_p,C.c_int,C.c_size_t]+[C.c_void_p]*3;l.kimi_projection_r3883_destroy.argtypes=[C.c_void_p];l.kimi_projection_r3883_frame.argtypes=[C.c_void_p,C.c_size_t,C.c_int];l.modern_kimi_reduce_policy_r3226.argtypes=[C.c_int];return l
def worker(rank,uid):
 torch.set_num_threads(1);torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32;os.sched_setaffinity(0,set(cpus[4*rank:4*rank+4]));stream=torch.cuda.Stream(device=rank);torch.cuda.set_stream(stream);l=library();ctx=None
 p=r/'results'/f'kimi-KDA1-replay-r3883-{tag}-rank{rank}.json';assert not p.exists();q={'completed':False,'rank':rank,'cases':[],'main_goal_metric':False,'wholemodel_qualified':False,'CPU_affinity':sorted(os.sched_getaffinity(0)),'helper_sha256':sha(base/'libkimi_projection_r3883.so')}
 def ck(code):
  if code:raise RuntimeError(l.kimi_projection_r3883_error().decode())
 def raw(x):stream.synchronize();return x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def record(name,x):
  b=raw(x);fn=f'kimi-KDA1-replay-r3883-{tag}-rank{rank}-{name}.bin';pp=r/'results'/fn;assert not pp.exists();pp.write_bytes(b);return {'file':fn,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(b).hexdigest()}
 try:
  assert torch.cuda.mem_get_info(rank)[0]>56*1024**3
  proof=json.loads((r/'results/kimi-KDA1-partial-r3861-analysis.json').read_text());assert proof['completed'];fixture=r/'results'/proof['fixture'];assert sha(fixture)==proof['fixture_sha256'];f=np.load(fixture);q['fixture_sha256']=sha(fixture)
  mapping=json.loads((pathlib.Path('/shared/Kimi-K3')/'model.safetensors.index.json').read_text())['weight_map'];keys=[k for k in mapping if'.layers.1.'in k and k.endswith('o_proj.weight')];assert len(keys)==1;key=keys[0]
  with safe_open(pathlib.Path('/shared/Kimi-K3')/mapping[key],framework='pt',device='cpu')as sf:whole=sf.get_tensor(key)
  assert whole.shape==(7168,12288)and whole.dtype==torch.bfloat16
  w=whole[:,rank*1536:(rank+1)*1536].contiguous().to(rank);del whole;whash=hashlib.sha256(raw(w)).hexdigest();q['weight']={'key':key,'local_sha256':whash}
  ctx=l.kimi_projection_r3883_create(rank,8,rank,C.create_string_buffer(uid));assert ctx,l.kimi_projection_r3883_error()
  for label,m in [('first',1),('second',1),('block',2)]:
   a=f[label+'_gated'][rank];assert a.dtype==np.uint16 and a.size==m*1536;x=torch.from_numpy(a.copy()).view(torch.bfloat16).reshape(m,1536).to(rank);xb=raw(x);local=torch.empty((m,7168),device=rank,dtype=torch.bfloat16)
   def project():ck(l.kimi_projection_r3883_gemm(ctx,m,0,x.data_ptr(),w.data_ptr(),local.data_ptr(),stream.cuda_stream))
   project();lb=raw(local);assert bool(torch.isfinite(local).all());repeats=[]
   for _ in range(3):project();repeats.append(raw(local)==lb)
   c={'label':label,'M':m,'local':record(label+'-local',local),'local_repeats_equal':all(repeats),'reductions':[],'source_output_sha256':hashlib.sha256(f[label+'_output'][rank].tobytes()).hexdigest()}
   expected=f[label+'_output'][rank].reshape(m,7168).tobytes()
   serial=[]
   for row in range(m):
    yy=torch.empty((1,7168),device=rank,dtype=torch.bfloat16)
    ck(l.kimi_projection_r3883_gemm(ctx,1,0,x[row:row+1].data_ptr(),w.data_ptr(),yy.data_ptr(),stream.cuda_stream));serial.append(yy)
   golden=torch.cat(serial);gb=raw(golden);c['serial_M1_reference']=record(label+'-M1-reference',golden);c['projection_variants']=[]
   for variant in[0,2]:
    yy=torch.full_like(local,float('nan'));tick=time.perf_counter()
    ck(l.kimi_projection_r3883_gemm(ctx,m,variant,x.data_ptr(),w.data_ptr(),yy.data_ptr(),stream.cuda_stream));stream.synchronize()
    times=[]
    for loop in range(8):
     begin=time.perf_counter();ck(l.kimi_projection_r3883_gemm(ctx,m,variant,x.data_ptr(),w.data_ptr(),yy.data_ptr(),stream.cuda_stream));stream.synchronize();times.append(time.perf_counter()-begin)
    assert bool(torch.isfinite(yy).all())
    c['projection_variants'].append({'variant':variant,'name':['ordinary_2D','broadcast_batched_M1','sequential_M1'][variant],'raw_equal_serial_M1_diagnostic':raw(yy)==gb,'different_elements':int((yy.view(torch.int16)!=golden.view(torch.int16)).sum().item()),'wall_seconds_host_inclusive':times,'output':record(label+'-variant'+str(variant),yy)})
   assert raw(x)==xb and hashlib.sha256(raw(w)).hexdigest()==whash;c['input_weight_readonly']=True;q['cases'].append(c)
  ck(l.modern_kimi_reduce_policy_r3226(0));q['collected']=True;q['scope']='Actual KDA1 inputs/weights, local GEMM shape component only; no collective or wholemodel token/performance qualification.'
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if ctx:ck(l.kimi_projection_r3883_destroy(ctx));q['Runtime_and_communicator_normally_closed']=True
  q['completed']=bool(q.get('collected'));p.write_text(json.dumps(q,indent=2))
if __name__=='__main__':
 assert torch.cuda.device_count()==8;l=library();buf=C.create_string_buffer(l.kimi_projection_r3883_id_size());ck=l.kimi_projection_r3883_id(buf);assert ck==0,l.kimi_projection_r3883_error();mp.spawn(worker,args=(buf.raw,),nprocs=8,join=True);print('KIMI_KDA1_REPLAY_R3866_ALL8_CLOSED',flush=True)
