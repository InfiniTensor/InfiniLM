import concurrent.futures,ctypes as C,datetime,hashlib,json,os,pathlib,threading
import numpy as np
import torch
torch.set_num_threads(1)
r=pathlib.Path('/refactor');name=os.environ['CAST64_VARIANT_R3376'];tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];base=r/'candidates'/f'cast64-latest-r3552-{name}';lib=C.CDLL(str(base/'libcast64_contract_r3376.so'))
lib.cast64_contract_create_r3376.argtypes=[C.c_int];lib.cast64_contract_create_r3376.restype=C.c_void_p;lib.cast64_contract_error_r3376.restype=C.c_char_p;lib.cast64_contract_destroy_r3376.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
lib.unified_ops_run_r3552.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_size_t,C.c_size_t,C.c_double,C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert torch.cuda.device_count()==8
def raw(x):return x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
def worker(dev):
 torch.cuda.set_device(dev);cpu=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpu[dev*4:dev*4+4]));stream=torch.cuda.Stream(device=dev);torch.cuda.set_stream(stream);ctx=lib.cast64_contract_create_r3376(dev);assert ctx,lib.cast64_contract_error_r3376();p=r/'results'/f'unified-ops-r3567-{name}-{tag}-gpu{dev}.json';assert not p.exists();q={'completed':False,'variant':name,'device':dev,'cases':[],'invalid_untouched':0,'wrongowner_untouched':0};finish=False
 try:
  for typ,dtype in[(1,torch.float32),(2,torch.float16),(3,torch.bfloat16)]:
   for rows,cols in[(0,5),(3,5),(1,128),(8,513)]:
    x=((torch.arange(rows*cols,dtype=torch.float32)%49-24)/8).reshape(rows,cols).to(dtype);b=((torch.arange(cols,dtype=torch.float32)%9-4)/4).reshape(1,cols).to(dtype);xp=torch.full((rows*cols+64,),11.,dtype=dtype,device=dev);bp=torch.full((cols+64,),12.,dtype=dtype,device=dev);xp[32:32+rows*cols].copy_(x.flatten());bp[32:32+cols].copy_(b.flatten());xf=raw(xp);bf=raw(bp)
    for op in[0,1,2]:
     for scalar in[0.,0.5,-2.,1.25]:
      reference=(x.float()*scalar).to(dtype)if op==0 else(x.float()*b.float()).to(dtype)if op==1 else torch.nn.functional.pad(x,(2,3),'constant',scalar);n=reference.numel();yp=torch.full((n+64,),13.,dtype=dtype,device=dev);outptr=yp[32:].data_ptr();records=[]
      for repeat in range(4):
       yp.fill_(13.);code=lib.unified_ops_run_r3552(ctx,op,typ,rows,cols,scalar,xp[32:].data_ptr(),bp[32:].data_ptr(),outptr,stream.cuda_stream,0);assert code==0,(op,typ,rows,cols,lib.cast64_contract_error_r3376().decode());actual=raw(yp[32:32+n]);gold=raw(reference);numeric=yp[32:32+n].detach().cpu().reshape(reference.shape);assert torch.isfinite(numeric).all()and torch.equal(numeric,reference),('unified_numeric_math',op,typ,rows,cols,scalar,float((numeric.float()-reference.float()).abs().max())if n else 0);q.setdefault('bitwise_differences_with_equal_numeric_values',0);q['bitwise_differences_with_equal_numeric_values']+=int(actual!=gold);assert raw(xp)==xf and raw(bp)==bf and bool((yp[:32]==13).all()and(yp[-32:]==13).all());records.append(hashlib.sha256(actual).hexdigest())
      q['cases'].append({'op':op,'dtype':typ,'rows':rows,'cols':cols,'scalar':scalar,'source_sha256':hashlib.sha256(raw(x)).hexdigest(),'other_sha256':hashlib.sha256(raw(b)).hexdigest(),'output_sha256':records[0],'repeat4_sha256':records,'CPU_reference_numeric_exact':True,'guard_input_untouched':True})
      if op==2 and rows:
       for invalid in[1,2]:
        yp.fill_(13.);code=lib.unified_ops_run_r3552(ctx,op,typ,rows,cols,scalar,xp[32:].data_ptr(),bp[32:].data_ptr(),outptr,stream.cuda_stream,invalid);assert code!=0 and bool((yp==13).all())and raw(xp)==xf and raw(bp)==bf;q['invalid_untouched']+=1
  assert len(q['cases'])==144;q['library_sha256']=sha(base/'libcast64_contract_r3376.so');finish=True
 except BaseException as e:q['error']=repr(e);raise
 finally:
  closed=C.c_int(77);code=lib.cast64_contract_destroy_r3376(ctx,C.byref(closed));assert code==0 and closed.value==1;q['normal_Runtime_weak_close']=True;q['completed']=finish;p.write_text(json.dumps(q,indent=2))
 return{'file':p.name,'sha256':sha(p),'device':dev}
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:owners=list(pool.map(worker,range(8)))
p=r/'results'/f'unified-ops-r3567-{name}-{tag}.json';assert not p.exists();p.write_text(json.dumps({'completed':True,'variant':name,'owners':owners,'scope':'Fresh c3deCommon/SDK math: Mul scalar/tensor broadcast and runtime constantPad on8Hygon owners,FP32/FP16/BF16 incl zeroelements/guard/repeat/negativePad rejection/normalclose. Only fixtures here, not wholemodel.'},indent=2));print('UNIFIED_OPS_R3523_ALL8_CLOSED',name)
