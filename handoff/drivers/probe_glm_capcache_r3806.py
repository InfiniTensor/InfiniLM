import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import pathlib,json,ctypes,time,datetime,hashlib,gc,statistics
import numpy as np
import torch,torch.multiprocessing as mp
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];base=r/'candidates/glm-capcache-r3806';oldpath=r/'candidates/models-5f91ad7-r3675/GLM/helpers/libglm_fp8_batched_r776.so';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def worker(device):
 torch.set_num_threads(1);torch.cuda.set_device(device);stream=torch.cuda.Stream();torch.cuda.set_stream(stream);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[device*4:device*4+4]));torch.manual_seed(3806);old=ctypes.CDLL(str(oldpath));new=ctypes.CDLL(str(base/'libglm_capcache_r3806.so'));vp=ctypes.c_void_p;sz=ctypes.c_size_t
 old.glm_fp8_error_r776.restype=ctypes.c_char_p;new.glm_fp8_capcache_error_r3806.restype=ctypes.c_char_p
 create=old.glm_fp8_create_r776;create.argtypes=[ctypes.c_int]*2+[sz]*7+[vp];create.restype=vp;run=old.glm_fp8_run_r776;run.argtypes=[vp,ctypes.c_int]+[vp]*5;destroy=old.glm_fp8_destroy_r776;destroy.argtypes=[vp]
 cc=new.glm_fp8_capcache_create_r3806;cc.argtypes=[ctypes.c_int,vp,vp];cc.restype=vp;cr=new.glm_fp8_capcache_run_r3806;cr.argtypes=[vp]*3+[sz]*7+[vp]*5;cs=new.glm_fp8_capcache_stats_r3806;cs.argtypes=[vp,ctypes.POINTER(ctypes.c_uint64),sz]
 clear=new.glm_fp8_capcache_clear_r3806;clear.argtypes=[vp]*3;delete=new.glm_fp8_capcache_destroy_r3806;delete.argtypes=[vp]*3
 identity=ctypes.c_uint64(3806);ident=vp(ctypes.addressof(identity));st=vp(stream.cuda_stream);cache=cc(device,st,ident);assert cache,new.glm_fp8_capcache_error_r3806();path=r/'results'/f'glm-capcache-r3806-{tag}-gpu{device}.json';assert not path.exists();report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':device,'completed':False,'cases':[],'baseline_helper_sha256':sha(oldpath),'candidate_helper_sha256':sha(base/'libglm_capcache_r3806.so'),'scope':'Actual latest3675 BF16 per128K baselinehelper. Cachedcapacity onlystorage; actual M/chunk/strides/math unchanged. FullGPU tensor equality +persistedsamplemath, fixed768MiB16LRU; nativeoperator/harness only, no modeltoken/HTTP/peak claim.'}
 def save():path.write_text(json.dumps(report,indent=2))
 def check(code,baseline=False):assert code==0,(old.glm_fp8_error_r776()if baseline else new.glm_fp8_capcache_error_r3806())
 def stats():v=(ctypes.c_uint64*12)();check(cs(cache,v,12));return list(v)
 def ptr(x):return vp(x.data_ptr())
 def data(shape):x=torch.randint(0,256,shape,device=device,dtype=torch.uint8);x[(x&127)==127]=0;return x
 def timing(fn,reps=5):
  hs=[];gs=[]
  for _ in range(reps):
   torch.cuda.synchronize(device);a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);tick=time.perf_counter();a.record(stream);fn();b.record(stream);b.synchronize();hs.append((time.perf_counter()-tick)*1000);gs.append(a.elapsed_time(b))
  return{'host_ms_median':statistics.median(hs),'stream_elapsed_ms_median':statistics.median(gs),'host_repetitions_ms':hs,'stream_repetitions_ms':gs}
 try:
  sentinel=(ctypes.c_uint64*12)(*[77]*12);assert cs(cache,sentinel,11)!=0 and list(sentinel)==[77]*12
  import concurrent.futures
  with concurrent.futures.ThreadPoolExecutor(max_workers=1)as ex:assert ex.submit(cs,cache,sentinel,12).result()!=0
  assert clear(cache,vp(99),st)!=0 and stats()[5]==0
  configs=[(name,m,n,k,0,0)for name,n,k in [('qa',2048,6144),('output',6144,2048),('up',1536,6144),('down',6144,768)]for m in[32,33,64,65,128,256,511,1024,2048]]+[('kv',4096,3584,512,0,0),('offset',127,384,384,129,1)]
  for index,cfg in enumerate(configs):
   name,m,n,k,ro,co=(*cfg[:4],0,0)if len(cfg)==4 else cfg;sr=(ro+n+127)//128;sc=(co+k+127)//128
   x=data((m,k));w=data((n,k));xs=torch.rand((m,sc),device=device)*.03+.005;ws=torch.rand((sr,sc),device=device)*.03+.005;y=torch.empty((m,n),device=device,dtype=torch.bfloat16);z=torch.empty_like(y);xx=x.clone();ww=w.clone();ss=xs.clone();tt=ws.clone();torch.cuda.synchronize(device)
   p=create(device,1,m,n,k,sr,sc,ro,co,st);assert p,old.glm_fp8_error_r776()
   def reused():check(run(p,1,ptr(y),ptr(x),ptr(xs),ptr(w),ptr(ws)),True)
   def cached():check(cr(cache,ident,st,m,n,k,sr,sc,ro,co,ptr(z),ptr(x),ptr(xs),ptr(w),ptr(ws)))
   def fresh():
    q=create(device,1,m,n,k,sr,sc,ro,co,st);assert q
    try:check(run(q,1,ptr(y),ptr(x),ptr(xs),ptr(w),ptr(ws)),True)
    finally:check(destroy(q),True)
   reused();cached();torch.cuda.synchronize(device);assert torch.equal(y,z),('actualM_BF16_full_matrix',cfg);assert torch.isfinite(y).all()
   t_reuse=timing(reused);t_cached=timing(cached);t_fresh=timing(fresh);check(destroy(p),True);assert torch.equal(y,z)and torch.equal(x,xx)and torch.equal(w,ww)and torch.equal(xs,ss)and torch.equal(ws,tt)
   # Same bucket with changed content must unpack the current input/weight.
   w.copy_(data((n,k)));xs.mul_(1.03125);torch.cuda.synchronize(device);p2=create(device,1,m,n,k,sr,sc,ro,co,st);assert p2
   try:check(run(p2,1,ptr(y),ptr(x),ptr(xs),ptr(w),ptr(ws)),True);cached();torch.cuda.synchronize(device);assert torch.equal(y,z),('updated_content',cfg)
   finally:check(destroy(p2),True)
   assert torch.equal(x,xx)and torch.equal(ws,tt);points=[(0,0),(m-1,n-1),(m//2,min(128,n-1))];rr=[v[0]for v in points];ccol=[v[1]for v in points];sample=r/'results'/f'glm-capcache-r3806-{tag}-gpu{device}-case{index}.npz';np.savez_compressed(sample,x=x[rr].cpu().numpy(),w=w[ccol].cpu().numpy(),xs=xs[rr].cpu().numpy(),ws=ws[[(ro+j)//128 for j in ccol]].cpu().numpy(),baseline=y[rr,ccol].contiguous().view(torch.uint8).cpu().numpy().view(np.uint16),candidate=z[rr,ccol].contiguous().view(torch.uint8).cpu().numpy().view(np.uint16))
   s=stats();assert s[5]<=16 and s[6]<=768*1024**2 and s[7]<=768*1024**2
   report['cases'].append({'name':name,'M':m,'N':n,'K':k,'row_offset':ro,'column_offset':co,'scale_rows':sr,'scale_columns':sc,'complete_BF16_matrix_exact_to_baseline':True,'changed_weight_and_scale_exact_to_baseline':True,'readonly_x_ws':True,'reused_baseline':t_reuse,'uncached_baseline':t_fresh,'capacity_cache':t_cached,'host_speed_ratio_uncached_to_capacity':t_fresh['host_ms_median']/t_cached['host_ms_median'],'stream_ratio_uncached_to_capacity':t_fresh['stream_elapsed_ms_median']/t_cached['stream_elapsed_ms_median'],'sample_file':sample.name,'sample_sha256':sha(sample),'cache_stats':s});save();del x,w,xs,ws,y,z,xx,ww,ss,tt;gc.collect()
  before=stats();check(clear(cache,ident,st));after=stats();assert after[0]==after[1]and after[5]==after[6]==0 and before[2]>0;report['before_clear']=before;report['after_clear']=after;check(delete(cache,ident,st));cache=None;report.update(completed=True,all_plans_destroyed=True,source_files_sha256={n:sha(r/'incoming'/n)for n in ['glm_fp8_capplan_r3806.cu','glm_fp8_capcache_r3806.cpp','glm_fp8_capcache_r3806.hpp','probe_glm_capcache_r3806.py','glm_capcache_r3806_inside.sh']});save()
 except BaseException as e:report['error']=repr(e);save();raise
 finally:
  if cache:check(delete(cache,ident,st))
if __name__=='__main__':mp.spawn(worker,nprocs=8,join=True);print('GLM_CAPCACHE_R3806_ALL8_COMPLETE',flush=True)
