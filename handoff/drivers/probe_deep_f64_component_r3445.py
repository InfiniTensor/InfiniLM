import os,sys
assert os.environ['PYTHONPYCACHEPREFIX'].startswith('/local/');sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import ctypes as C,datetime,hashlib,json,pathlib,threading
import numpy as np
import torch,torch.multiprocessing as mp
r=pathlib.Path('/refactor');base=r/'candidates/deep-f64-3e0d89fe-r3445';tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def library():
 lib=C.CDLL(str(base/'libdeep_f64_r3445.so'));lib.deep_projection_r3445_error.restype=C.c_char_p;lib.deep_projection_r3445_id_size.restype=C.c_size_t;lib.deep_projection_r3445_id.argtypes=[C.c_void_p];lib.deep_projection_r3445_create.argtypes=[C.c_int]*3+[C.c_void_p];lib.deep_projection_r3445_create.restype=C.c_void_p;lib.deep_projection_r3445_destroy.argtypes=[C.c_void_p];lib.deep_projection_r3445_allreduce.argtypes=[C.c_void_p,C.c_int,C.c_size_t]+[C.c_void_p]*3;lib.deep_projection_r3445_wide.argtypes=[C.c_void_p,C.c_int,C.c_size_t]+[C.c_void_p]*3;return lib
def worker(rank,uid):
 torch.set_num_threads(1);torch.cuda.set_device(rank);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[rank*4:rank*4+4]));stream=torch.cuda.Stream(device=rank);torch.cuda.set_stream(stream);lib=library();ctx=None;p=r/'results'/f'deep-f64-r3445-{tag}-rank{rank}.json';assert not p.exists();q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'rank':rank,'CPU_affinity':sorted(os.sched_getaffinity(0)),'library_sha256':sha(base/'libdeep_f64_r3445.so'),'source_sha256':sha(pathlib.Path(__file__)),'cases':[],'scope':'SyntheticFP32partial rows: randomdynamicrange+nearBF16roundingboundary; nativeFP32sum vs exactSDKCastFP32->FP64/nativeCCLsum/CastFP32, thenTorchBF16 final. M4/8/16,N7168,aligned+offset64,in/outplace,8ownrepeats,guards/owner/normalCCLclose. NoactualDeep remainingwindowrootcause/fullmodelrepair/TPSclaim.'}
 def save():
  t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
 def ck(code):
  if code:raise RuntimeError(lib.deep_projection_r3445_error().decode())
 def raw(x):return x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def record(label,x):
  b=raw(x);n=f'deep-f64-r3445-{tag}-rank{rank}-{label}.bin';v=r/'results'/n;assert not v.exists();v.write_bytes(b);return{'file':n,'shape':list(x.shape),'dtype':str(x.dtype),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 save()
 try:
  assert torch.cuda.mem_get_info(rank)[0]>56*1024**3;buf=C.create_string_buffer(uid);ctx=lib.deep_projection_r3445_create(rank,8,rank,buf);assert ctx,lib.deep_projection_r3445_error()
  for m in[4,8,16]:
   for fixture in['random','BF16_boundary']:
    rng=np.random.default_rng(3385000+rank*100+m);n=7168;exponent=(np.arange(n)%9-4).astype(np.int32)
    row=(rng.uniform(-1,1,n)*np.exp2((np.arange(n)+rank)%17-8)).astype(np.float32)if fixture=='random'else(np.exp2(exponent)*(1+1/256)if rank==0 else np.exp2(exponent-24)*[0,1,1,1,-1,-1,-1,1][rank]).astype(np.float32)
    source=torch.from_numpy(np.tile(row,(m,1))).to(rank)
    for shift in[0,64]:
     for inplace in[False,True]:
      label=f'M{m}-{fixture}-a{shift}-i{int(inplace)}';guard=torch.full((m*n*4+512+shift,),0x6d,device=rank,dtype=torch.uint8);x=guard[256+shift:-256].view(torch.float32).reshape(m,n);outguard=torch.full_like(guard,0x53);y=outguard[256+shift:-256].view(torch.float32).reshape(m,n);c={'M':m,'N':n,'fixture':fixture,'shift_bytes':shift,'inplace':inplace,'input':record(label+'-input',source),'outputs':[]}
      for mode in[0,1,1,0]:
       hashes=[];bfhashes=[];target=x if inplace else y
       for loop in range(8):
        x.copy_(source);target=x if inplace else y
        fn=lib.deep_projection_r3445_wide if mode else lib.deep_projection_r3445_allreduce;ck(fn(ctx,0,x.numel(),x.data_ptr(),target.data_ptr(),stream.cuda_stream));b=raw(target);hashes.append(hashlib.sha256(b).hexdigest());bfhashes.append(hashlib.sha256(raw(target.bfloat16())).hexdigest())
        assert bool(torch.isfinite(target).all())
       output=target.clone();stream.synchronize();duplicate=all(raw(v)==raw(output[0])for v in output);c['outputs'].append({'mode':mode,'epoch':len(c['outputs']),'F32_sha256_repeat8':hashes,'BF16_sha256_repeat8':bfhashes,'own_repeat8_exact':len(set(hashes))==1,'duplicate_rows_fullraw_exact':duplicate,'raw':record(label+'-e'+str(len(c['outputs']))+'-out',output),'BF16_raw':record(label+'-e'+str(len(c['outputs']))+'-bf16',output.bfloat16())})
       assert bool((guard[:256+shift]==0x6d).all()and(guard[-256:]==0x6d).all()and(outguard[:256+shift]==0x53).all()and(outguard[-256:]==0x53).all())
       if not inplace:assert raw(x)==raw(source)
      before=raw(y);assert lib.deep_projection_r3445_wide(ctx,99,x.numel(),x.data_ptr(),y.data_ptr(),stream.cuda_stream)!=0 and raw(y)==before
      bad=[];thread=threading.Thread(target=lambda:bad.append(lib.deep_projection_r3445_wide(ctx,0,x.numel(),x.data_ptr(),y.data_ptr(),stream.cuda_stream)));thread.start();thread.join(20);assert not thread.is_alive()and bad==[1]and raw(y)==before
      c['guards_invalid_wrongowner_readonly_passed']=True;q['cases'].append(c);save();print('DEEP_F64_R3445_CASE',rank,label,flush=True)
  assert len(q['cases'])==24;q['numeric_collection_before_close']=True
 except BaseException as e:q['error']=repr(e);save();raise
 finally:
  if ctx:ck(lib.deep_projection_r3445_destroy(ctx));q['communicator_normally_closed']=True
  q['completed']=bool(q.get('numeric_collection_before_close'));save()
if __name__=='__main__':
 assert torch.cuda.device_count()==8;lib=library();uid=C.create_string_buffer(lib.deep_projection_r3445_id_size());assert lib.deep_projection_r3445_id(uid)==0;mp.spawn(worker,args=(uid.raw,),nprocs=8,join=True);print('DEEP_F64_R3445_ALL8_CLOSED',flush=True)
