import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,datetime,hashlib,ctypes,gc
import numpy as np
import torch
r=pathlib.Path('/refactor');base=r/'candidates/deep-collapse-5f91ad7-r4367';out=r/'results/deep-collapse-r4367.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lib=ctypes.CDLL(str(base/'libmodern_deep_collapse_r4367.so'));lib.modern_deep_collapse_probe_r4367.argtypes=[ctypes.c_char_p,ctypes.c_int];lib.modern_deep_collapse_probe_r4367.restype=ctypes.c_int;lib.modern_deep_collapse_result_r4367.restype=ctypes.c_char_p
rawdir=r/'results/deep-collapse-r4367-raw';rawdir.mkdir(exist_ok=False)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'Slurm_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'library_sha256':sha(base/'libmodern_deep_collapse_r4367.so'),'source_sha256':{p.name:sha(p)for p in base.glob('*')if p.suffix in['.cpp','.cu','.py','.sh']},'cases':[],'normal_owner_contexts_closed':0,'model_performance_qualified':False,'scope':'Standalone latestRuntime/nativeOps Mul+fourstreamSum+Cast collapse tail vs fused FP32. Randomsignedinputwithcancellation, BF16roundedinputs, realdimensions, fullCPUbitcompares+sampledindependentCPUFP64oracle. No modelweights/IDs/entire mHC or physicalpeak claim.'}
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(out)
save()
try:
 for low in [False,True]:
  for T in [1,8,16,32,64,128,512,4096]:
   rng=np.random.default_rng(4367+T+10000*int(low));x=rng.standard_normal((T,4,4096),dtype=np.float32);x*=np.asarray([.001,.1,1.,100.],np.float32).reshape(1,4,1)
   x[:,2,::17]=-x[:,3,::17];pre=rng.uniform(1e-6,1.000001,(T,4)).astype(np.float32)
   if low:
    u=x.view(np.uint32);x=((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32).copy()
   originals=[np.empty((T,4096),np.uint16 if low else np.float32)for _ in range(8)];candidates=[np.empty_like(z)for z in originals]
   q={'tokens':T,'bf16':low,'residual_fp32':x.ctypes.data,'pre':pre.ctypes.data,'original_out':[v.ctypes.data for v in originals],'candidate_out':[v.ctypes.data for v in candidates]}
   status=lib.modern_deep_collapse_probe_r4367(json.dumps(q).encode(),8);row=json.loads(lib.modern_deep_collapse_result_r4367());assert status==0 and row['completed'],row
   row.update(tokens=T,bf16=low,input_SHA256=hashlib.sha256(x.tobytes()).hexdigest(),pre_SHA256=hashlib.sha256(pre.tobytes()).hexdigest(),CPU_full_variant0_checks=[],CPU_oracle=[])
   cols=np.arange(0,4096,67);samples=x[:,:,cols].astype(np.float64)*pre[:,:,None].astype(np.float64);expected=samples.sum(1);bound=8*np.finfo(np.float32).eps*np.abs(samples).sum(1)+1e-30
   if low:bound+=.004*np.abs(expected)
   for rank,(a,b,rr)in enumerate(zip(originals,candidates,row['reports'])):
    actual_a=(a.astype(np.uint32)<<16).view(np.float32)if low else a;actual_b=(b.astype(np.uint32)<<16).view(np.float32)if low else b
    assert np.isfinite(actual_a).all()and np.isfinite(actual_b).all()and rr['owner_shutdown_completed'];changes=int(np.count_nonzero(a.view(np.uint16 if low else np.uint32)!=b.view(np.uint16 if low else np.uint32)));assert changes==rr['variants'][0]['different_elements'];row['CPU_full_variant0_checks'].append({'rank':rank,'different_elements':changes,'finite':True});errors=np.abs(actual_b[:,cols].astype(np.float64)-expected);assert np.all(errors<=bound),('CPUoracle',rank,T,low,float(errors.max()),float(bound.max()));row['CPU_oracle'].append({'rank':rank,'sampled_points':int(errors.size),'max_abs_error':float(errors.max()),'passed':True})
    if T in[16,128]or(T==4096 and rank in[0,6]):
     for kind,arr in [('native',a),('fused0',b)]:
      p=rawdir/f'T{T}-low{int(low)}-rank{rank}-{kind}.bin';arr.tofile(p);row.setdefault('raw',[]).append({'file':p.relative_to(r/'results').as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'rank':rank,'kind':kind,'shape':[T,4096],'dtype':'BF16'if low else'F32'})
   d['cases'].append(row);d['normal_owner_contexts_closed']+=8;save();del x,pre,originals,candidates,samples;gc.collect()
 d['all_variant0_full_outputs_exact']=all(z['different_elements']==0 for c in d['cases']for z in c['CPU_full_variant0_checks']);d['all_variants_exact']=[v for v in range(3)if all(c['reports'][rank]['variants'][v]['different_elements']==0 and c['reports'][rank]['variants'][v]['repeat_differences']==0 for c in d['cases']for rank in range(8))];d['completed']=True;save();print('DEEP_COLLAPSE_R4367_COMPLETE',json.dumps({'all_exact_variants':d['all_variants_exact'],'normal_contexts':d['normal_owner_contexts_closed']}))
except BaseException as exc:d['error']=repr(exc);save();raise
