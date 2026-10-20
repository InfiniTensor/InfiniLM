import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import base64,ctypes as C,datetime,hashlib,json,pathlib,socket,time,traceback
import torch,torch.multiprocessing as mp
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];node=int(os.environ['SLURM_PROCID']);base=r/'candidates/parallel-collectives-5f91ad7-r3685';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def library():
 lib=C.CDLL(str(base/'libparallel_r3612.so'));lib.parallel_r3612_error.restype=C.c_char_p;lib.parallel_r3612_id_size.restype=C.c_size_t;lib.parallel_r3612_id.argtypes=[C.c_void_p];lib.parallel_r3612_create.argtypes=[C.c_int]*3+[C.c_void_p];lib.parallel_r3612_create.restype=C.c_void_p;lib.parallel_r3612_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_size_t]+[C.c_void_p]*4+[C.c_int];lib.parallel_r3612_destroy.argtypes=[C.c_void_p];return lib
def worker(device,payload):
 torch.set_num_threads(1);torch.cuda.set_device(device);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[device*4:device*4+4]));stream=torch.cuda.Stream();torch.cuda.set_stream(stream);lib=library();path=r/'results'/f'parallel-r3685-{tag}-node{node}-gpu{device}.json';assert not path.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':socket.gethostname(),'node_rank':node,'device':device,'completed':False,'cases':[],'groups_destroyed':[],'CPU_affinity':sorted(os.sched_getaffinity(0)),'library_sha256':sha(base/'libparallel_r3612.so'),'scope':'Latest5f91 SDK3672/Common3673/CCL51cc communication component: real8/16/32rank groups, AR, AG,RS,RS+AG. Shapes correspond6144/7168hidden andB1/8/32/128/512. Bounded positive integer oracle rankmod4+1, all intermediate sums <=80, exactly representable evenBF16. Prior rank1..32 oracle can round intermediate sums above256, despite final528 beingrepresentable; original failure retained, not a proved communication bug. Not actualexpert dispatch, fullmodel EP, pureTP modelthroughput or physicalnetworkpeak.'}
 def save():
  p=path.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(path)
 def ck(code):
  if code:raise RuntimeError(lib.parallel_r3612_error().decode())
 maps={v.split()[-1]:sha(pathlib.Path(v.split()[-1]))for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(n in v for n in ['libinfinilm_runtime.so','libinfiniccl.so','libinfinirt.so','libinfiniops.so'])};sdk=json.loads((r/'results/latest-sdk-r3672-GLM-build.json').read_text());runtime=json.loads((r/'results/common-latest-r3673-GLM-build.json').read_text())
 for part,h in [('libinfiniops.so',sdk['library_sha256']),('libinfiniccl.so',sdk['CCL_sha256']),('libinfinirt.so',sdk['RT_sha256']),('libinfinilm_runtime.so',runtime['runtime_sha256'])]:assert[v for p,v in maps.items()if part in p]==[h]
 d['actual_mapped_libraries']=maps;d['source_sha256']={n:sha(r/'incoming'/n)for n in['parallel_collectives_r3612.cpp','probe_parallel_collectives_r3685.py','parallel_collectives_r3685_inside.sh']};save();ctx=None;contexts={}
 try:
  globalrank=node*8+device
  # Keep all groups alive through the benchmark; isolate last-group-recreation.
  for size in[8,16,32]:
   rank=globalrank%size;group=globalrank//size;uid=C.create_string_buffer(base64.b64decode(payload[str(size)][group]));created=lib.parallel_r3612_create(device,size,rank,uid);assert created,lib.parallel_r3612_error();contexts[size]=created
  for size in[8,16,32]:
   rank=globalrank%size;group=globalrank//size;ctx=contexts[size]
   for hidden in[6144,7168]:
    for batch in[1,8,32,128,512]:
     count=batch*hidden
     for code,dtype in[(2,torch.bfloat16),(0,torch.float32),(3,torch.float64)]:
      for mode in[0,1,2,3]:
       x=torch.full((count*(size if mode==2 else 1),),rank%4+1,device='cuda',dtype=dtype);y=torch.full((count*(size if mode==1 else 1),),float('nan'),device='cuda',dtype=dtype);scratch=torch.full((max(1,count//size),),float('nan'),device='cuda',dtype=dtype);repeat=20 if count<512000 else 6
       def run(n):ck(lib.parallel_r3612_run(ctx,mode,code,count,x.data_ptr(),y.data_ptr(),scratch.data_ptr(),stream.cuda_stream,n))
       run(2);expected=(size//4)*10
       if mode==1:
        blocks=y.view(size,count);assert all(bool((blocks[k]==k%4+1).all())for k in range(size))
       else:assert bool(torch.isfinite(y).all())and bool((y==expected).all()),(size,mode,code)
       samples=[]
       for rep in range(3):
        start=time.perf_counter();run(repeat);samples.append((time.perf_counter()-start)*1000/repeat)
       assert bool((x==rank%4+1).all())
       d['cases'].append({'group_size':size,'group_rank':rank,'hidden':hidden,'batch':batch,'dtype':str(dtype),'mode':['AllReduce','AllGather','ReduceScatter','ReduceScatter+AllGather'][mode],'local_elements':count,'input_bytes':x.numel()*x.element_size(),'output_bytes':y.numel()*y.element_size(),'repeat':repeat,'three_wall_ms_per_call':samples,'finite_exact_oracle_and_input_untouched':True});del x,y,scratch
       if len(d['cases'])%24==0:save()
  ctx=None
  for size in[32,16,8]:ck(lib.parallel_r3612_destroy(contexts.pop(size)));d['groups_destroyed'].append(size);save()
  d['completed']=True;save()
 except BaseException as e:
  d.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:
  for remaining in list(contexts.values()):ck(lib.parallel_r3612_destroy(remaining))
def guarded(device,payload):
 try:worker(device,payload)
 except BaseException:traceback.print_exc();os._exit(2)
if __name__=='__main__':
 p=r/'control'/f'parallel-r3685-{tag}.json'
 if node==0:
  lib=library();payload={}
  for size in[8,16,32]:
   payload[str(size)]=[]
   for _ in range(32//size):
    b=C.create_string_buffer(lib.parallel_r3612_id_size());assert lib.parallel_r3612_id(b)==0;payload[str(size)].append(base64.b64encode(b.raw).decode())
  assert not p.exists();q=p.with_suffix('.tmp');q.write_text(json.dumps(payload));q.replace(p)
 deadline=time.monotonic()+300
 while not p.exists():assert time.monotonic()<deadline;time.sleep(.2)
 mp.spawn(guarded,args=(json.loads(p.read_text()),),nprocs=8,join=True);print('PARALLEL_R3612_NODE_COMPLETE',node,flush=True)
