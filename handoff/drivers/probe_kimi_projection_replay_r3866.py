import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import ctypes as C,datetime,hashlib,json,pathlib,time,traceback
import numpy as np
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');base=r/'candidates/kimi-KDA1-replay-5f91ad7-r3866';tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def library():
 l=C.CDLL(str(base/'libkimi_projection_r3866.so'));l.kimi_projection_r3866_error.restype=C.c_char_p;l.kimi_projection_r3866_id_size.restype=C.c_size_t;l.kimi_projection_r3866_id.argtypes=[C.c_void_p];l.kimi_projection_r3866_create.argtypes=[C.c_int]*3+[C.c_void_p];l.kimi_projection_r3866_create.restype=C.c_void_p;l.kimi_projection_r3866_gemm.argtypes=[C.c_void_p,C.c_size_t,C.c_int]+[C.c_void_p]*4;l.kimi_projection_r3866_allreduce.argtypes=[C.c_void_p,C.c_int,C.c_size_t]+[C.c_void_p]*3;l.kimi_projection_r3866_destroy.argtypes=[C.c_void_p];l.kimi_projection_r3866_frame.argtypes=[C.c_void_p,C.c_size_t,C.c_int];l.modern_kimi_reduce_policy_r3226.argtypes=[C.c_int];return l
def worker(rank,uid):
 torch.set_num_threads(1);torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=32;os.sched_setaffinity(0,set(cpus[4*rank:4*rank+4]));stream=torch.cuda.Stream(device=rank);torch.cuda.set_stream(stream);l=library();ctx=None
 p=r/'results'/f'kimi-KDA1-replay-r3866-{tag}-rank{rank}.json';assert not p.exists();q={'completed':False,'rank':rank,'cases':[],'main_goal_metric':False,'wholemodel_qualified':False,'CPU_affinity':sorted(os.sched_getaffinity(0)),'helper_sha256':sha(base/'libkimi_projection_r3866.so')}
 def ck(code):
  if code:raise RuntimeError(l.kimi_projection_r3866_error().decode())
 def raw(x):stream.synchronize();return x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def record(name,x):
  b=raw(x);fn=f'kimi-KDA1-replay-r3866-{tag}-rank{rank}-{name}.bin';pp=r/'results'/fn;assert not pp.exists();pp.write_bytes(b);return {'file':fn,'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(b).hexdigest()}
 try:
  assert torch.cuda.mem_get_info(rank)[0]>56*1024**3
  proof=json.loads((r/'results/kimi-KDA1-partial-r3861-analysis.json').read_text());assert proof['completed'];fixture=r/'results'/proof['fixture'];assert sha(fixture)==proof['fixture_sha256'];f=np.load(fixture);q['fixture_sha256']=sha(fixture)
  mapping=json.loads((pathlib.Path('/shared/Kimi-K3')/'model.safetensors.index.json').read_text())['weight_map'];keys=[k for k in mapping if'.layers.1.'in k and k.endswith('o_proj.weight')];assert len(keys)==1;key=keys[0]
  with safe_open(pathlib.Path('/shared/Kimi-K3')/mapping[key],framework='pt',device='cpu')as sf:whole=sf.get_tensor(key)
  assert whole.shape==(7168,12288)and whole.dtype==torch.bfloat16
  w=whole[:,rank*1536:(rank+1)*1536].contiguous().to(rank);del whole;whash=hashlib.sha256(raw(w)).hexdigest();q['weight']={'key':key,'local_sha256':whash}
  ctx=l.kimi_projection_r3866_create(rank,8,rank,C.create_string_buffer(uid));assert ctx,l.kimi_projection_r3866_error()
  for label,m in [('first',1),('second',1),('block',2)]:
   a=f[label+'_gated'][rank];assert a.dtype==np.uint16 and a.size==m*1536;x=torch.from_numpy(a.copy()).view(torch.bfloat16).reshape(m,1536).to(rank);xb=raw(x);local=torch.empty((m,7168),device=rank,dtype=torch.bfloat16)
   def project():ck(l.kimi_projection_r3866_gemm(ctx,m,0,x.data_ptr(),w.data_ptr(),local.data_ptr(),stream.cuda_stream))
   project();lb=raw(local);assert bool(torch.isfinite(local).all());repeats=[]
   for _ in range(3):project();repeats.append(raw(local)==lb)
   c={'label':label,'M':m,'local':record(label+'-local',local),'local_repeats_equal':all(repeats),'reductions':[],'source_output_sha256':hashlib.sha256(f[label+'_output'][rank].tobytes()).hexdigest()}
   expected=f[label+'_output'][rank].reshape(m,7168).tobytes()
   for mode in[0,1,2]:
    ck(l.modern_kimi_reduce_policy_r3226(mode));ck(l.kimi_projection_r3866_frame(ctx,2,0));y=torch.empty_like(local);t=time.perf_counter()
    ck(l.kimi_projection_r3866_allreduce(ctx,2,local.numel(),local.data_ptr(),y.data_ptr(),stream.cuda_stream));ck(l.kimi_projection_r3866_frame(ctx,0,1));stream.synchronize();dt=time.perf_counter()-t
    c['reductions'].append({'policy':mode,'F64_reference_only':mode==2,'actual_captured_output_equal':raw(y)==expected,'finite':bool(torch.isfinite(y).all()),'wall_seconds_diagnostic':dt,'output':record(label+'-policy'+str(mode),y)})
   assert raw(x)==xb and hashlib.sha256(raw(w)).hexdigest()==whash;c['input_weight_readonly']=True;q['cases'].append(c)
  ck(l.modern_kimi_reduce_policy_r3226(0));q['collected']=True
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if ctx:ck(l.kimi_projection_r3866_destroy(ctx));q['Runtime_and_communicator_normally_closed']=True
  q['completed']=bool(q.get('collected'));p.write_text(json.dumps(q,indent=2))
if __name__=='__main__':
 assert torch.cuda.device_count()==8;l=library();buf=C.create_string_buffer(l.kimi_projection_r3866_id_size());ck=l.kimi_projection_r3866_id(buf);assert ck==0,l.kimi_projection_r3866_error();mp.spawn(worker,args=(buf.raw,),nprocs=8,join=True);print('KIMI_KDA1_REPLAY_R3866_ALL8_CLOSED',flush=True)
