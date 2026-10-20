import os,sys
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import datetime,hashlib,json,pathlib,subprocess,time,traceback,socket
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');model=pathlib.Path('/old/models/ZhipuAI-GLM-5.3');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];base=pathlib.Path('/dev/shm')/('iloader-r3613-'+str(os.getuid()));name='glm-one-shard';output=r/'results'/f'weight-reload-r3613-{tag}.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def cli(args,timeout=300):
 p=subprocess.run([sys.executable,'-m','infiniloader',*args],capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError((args,p.returncode,p.stdout[-1500:],p.stderr[-1500:]))
 return p
def worker(device,payload,barrier):
 torch.set_num_threads(1);torch.cuda.set_device(device);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[device*4:device*4+4]));out=r/'results'/f'weight-reload-r3613-{tag}-gpu{device}.json';assert not out.exists();q={'device':device,'node':socket.gethostname(),'completed':False,'source_file':payload['file'],'source_bytes':payload['bytes'],'epochs':[],'CPU_affinity':sorted(os.sched_getaffinity(0))}
 def save():out.write_text(json.dumps(q,indent=2))
 save()
 try:
  for epoch,kind in enumerate(['original_shared','resident_memfd','resident_memfd','original_shared']):
   path=model/payload['file']if kind=='original_shared'else pathlib.Path(payload['view'])/payload['file'];barrier.wait(300);tick=time.perf_counter();values=[];types={};total=0
   with safe_open(path,framework='pt',device='cpu')as f:
    names=sorted(f.keys())
    for key in names:
     cpu=f.get_tensor(key);types[str(cpu.dtype)]=types.get(str(cpu.dtype),0)+1;raw=cpu.contiguous().view(torch.uint8);gpu=raw.to(device);values.append((key,gpu));total+=raw.numel()
   torch.cuda.synchronize(device);elapsed=time.perf_counter()-tick
   with safe_open(path,framework='pt',device='cpu')as f:
    samples=[]
    for i in sorted(set([0,len(values)//4,len(values)//2,3*len(values)//4,len(values)-1])):
     key,gpu=values[i];cpu=f.get_tensor(key).contiguous().view(torch.uint8).flatten();got=gpu.flatten();positions=sorted(set([0,len(cpu)//2,max(0,len(cpu)-64)]))
     for pos in positions:assert torch.equal(got[pos:pos+64].cpu(),cpu[pos:pos+64])
     samples.append({'key':key,'shape':list(gpu.shape),'positions':positions,'byte_samples_exact':True})
   q['epochs'].append({'epoch':epoch,'source_kind':kind,'tensor_keys':len(names),'actual_tensor_bytes':total,'H2D_and_loader_wall_seconds':elapsed,'payload_GiB_s':total/2**30/elapsed,'checkpoint_dtypes':types,'GPU_bytes_allocated':torch.cuda.memory_allocated(device),'GPU_reserved':torch.cuda.memory_reserved(device),'CPU_smaps_rollup':pathlib.Path('/proc/self/smaps_rollup').read_text(),'samples':samples});save();del values,gpu,raw,cpu,got;torch.cuda.empty_cache();barrier.wait(300)
  q['completed']=True;save()
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc());save();raise
def guarded(device,payload,barrier):
 try:worker(device,payload,barrier)
 except BaseException:traceback.print_exc();os._exit(2)
if __name__=='__main__':
 d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':tag,'completed':False,'loader_commit':'88d6bac6a80a424201da1688989d1d87e3217a93','scope':'Actual complete oneGLMcheckpointfile parsed andcopied byte-preservingtoall8GPUs, shared/memfd/memfd/sharedABBA. Preload keepssource OS cache warm: not a coldNFS benchmark, fullmodel loading speedup, collective or inference throughput. Per-file prep, memoryPSS andH2D latency separate.'};assert not output.exists();output.write_text(json.dumps(d,indent=2));started=False
 try:
  files=sorted(model.glob('*.safetensors'));p=next(p for p in files if 1024**3<=p.stat().st_size<=16*1024**3);d.update(file=p.name,bytes=p.stat().st_size);tick=time.perf_counter();args=['start',str(model),'--base-dir',str(base),'--name',name,'--pattern',p.name,'--workers','1','--max-memory-bytes','20GiB','--max-resident-bytes','20GiB','--reserve-bytes','64GiB','--wait-timeout','2400'];res=cli(args,2500);started=True;d.update(preload_seconds=time.perf_counter()-tick,view=res.stdout.strip().splitlines()[-1],status=json.loads(cli(['status',name,'--base-dir',str(base),'--json']).stdout));output.write_text(json.dumps(d,indent=2));payload={k:d[k]for k in['file','bytes','view']};barrier=mp.get_context('spawn').Barrier(8);mp.spawn(guarded,args=(payload,barrier),nprocs=8,join=True);d['completed']=True
 except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());raise
 finally:
  if started:d['cache_stop']=cli(['stop',name,'--base-dir',str(base)]).stdout
  output.write_text(json.dumps(d,indent=2))
