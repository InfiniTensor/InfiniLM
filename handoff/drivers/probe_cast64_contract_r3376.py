import os,sys
assert os.environ['PYTHONPYCACHEPREFIX'].startswith('/local/');sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']+'/'+str(os.getpid())
import concurrent.futures,ctypes as C,datetime,hashlib,json,math,pathlib,struct,threading
import numpy as np
import torch
torch.set_num_threads(1)
r=pathlib.Path('/refactor');name=os.environ['CAST64_VARIANT_R3376'];mode=os.environ['CAST64_MODE_R3376'];assert mode in ['old','new'];tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];base=r/'candidates'/f'cast64-contract-r3376-{name}';lib=C.CDLL(str(base/'libcast64_contract_r3376.so'))
lib.cast64_contract_create_r3376.argtypes=[C.c_int];lib.cast64_contract_create_r3376.restype=C.c_void_p;lib.cast64_contract_error_r3376.restype=C.c_char_p;lib.cast64_contract_run_r3376.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_size_t,C.c_int]+[C.c_void_p]*3;lib.cast64_contract_destroy_r3376.argtypes=[C.c_void_p,C.POINTER(C.c_int)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ops={p for p in{a.split()[-1]for a in pathlib.Path('/proc/self/maps').read_text().splitlines()}if'libinfiniops.so'in p};assert len(ops)==1;actual=next(iter(ops));prefix=(r/'install'/f'hygon-{name.lower()}-cast64-r3315'if mode=='new'else r/'install'/f'hygon-{name.lower()}-3e0d89fe-r2844');assert pathlib.Path(actual).resolve()==(prefix/'lib/libinfiniops.so').resolve()
width=[8,4,2,2];ndtypes=['<u8','<u4','<u2','<u2'];formats=[(52,11,1023),(23,8,127),(10,5,15),(7,8,127)]
def fixture(dtype,count,dev):
 p,e,bias=formats[dtype];sign=1<<(p+e);inf=((1<<e)-1)<<p;one=bias<<p
 edges=[0,sign,one,one|sign,1,2,3,sign|1,(1<<p)-1,1<<p,(1<<p)+1,inf-1,sign|(inf-1),inf,sign|inf,inf|1,sign|inf|1,inf|(1<<(p-1)),inf|((1<<p)-1)]
 if dtype==0:
  for power,precision in [(0,7),(0,10),(0,23),(-126,23),(-14,10),(-126,7)]:
   midpoint=math.ldexp(1+2**(-precision-1),power)
   edges.extend(struct.unpack('<Q',struct.pack('<d',v))[0]for v in[midpoint,math.nextafter(midpoint,-math.inf),math.nextafter(midpoint,math.inf),-midpoint])
  for exponent in [-150,-25,-134,-149,-24,-133]:
   v=math.ldexp(1.,exponent);edges.extend(struct.unpack('<Q',struct.pack('<d',a))[0]for a in[v,math.nextafter(v,0.),math.nextafter(v,math.inf),-v])
  for v in [65504.,65520.,math.nextafter(65520.,0.),math.ldexp(2-2**-23,127),math.ldexp(2-2**-7,127)]:edges.append(struct.unpack('<Q',struct.pack('<d',v))[0])
 rng=np.random.default_rng(3376000+dev*100+dtype);random=rng.integers(0,2**64 if dtype==0 else 2**(width[dtype]*8),size=max(count,1),dtype=np.uint64).astype(ndtypes[dtype]);k=min(count,len(edges));random[:k]=np.asarray(edges[:k],dtype=ndtypes[dtype]);return random[:count].tobytes()
def worker(dev):
 torch.cuda.set_device(dev);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[dev*4:dev*4+4]));assert torch.cuda.mem_get_info(dev)[0]>56*1024**3;stream=torch.cuda.Stream(device=dev);torch.cuda.set_stream(stream);ctx=lib.cast64_contract_create_r3376(dev);assert ctx,lib.cast64_contract_error_r3376()
 path=r/'results'/f'cast64-contract-r3376-{name}-{tag}-{mode}-gpu{dev}.json';archive=r/'results'/f'cast64-contract-r3376-{name}-{tag}-{mode}-gpu{dev}.bin';assert not path.exists()and not archive.exists();journal=archive.open('xb');q={'completed':False,'mode':mode,'variant':name,'device':dev,'cases':[],'invalid_checks':0,'identity_inplace_checks':0,'wrongowner_untouched':False,'library_sha256':sha(base/'libcast64_contract_r3376.so'),'actual_Op_path':actual,'actual_Op_sha256':sha(pathlib.Path(actual)),'CPU_affinity':sorted(os.sched_getaffinity(0)),'scope':'OldSDK9typepairs/newSDK16typepairs,rawbit edge+random fixtures,count0/1/17/513/4096,alignment0/64,8repeats,alias/metadata/owner/empty/normalRuntimeweakclose. NewSDKisolated; graphnotqualified.'}
 def ck(v):
  if v:raise RuntimeError(lib.cast64_contract_error_r3376().decode())
 def record(b):
  off=journal.tell();journal.write(b);return{'file':archive.name,'offset':off,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 def gpu_bytes(v):return v.detach().contiguous().cpu().numpy().tobytes()
 try:
  types=range(4)if mode=='new'else range(1,4)
  for it in types:
   for ot in types:
    for count in [0,1,17,513,4096]:
     for shift in [0,64]:
      source=fixture(it,count,dev);pad=256+shift;x=torch.full((pad+len(source)+256,),0x6d,device=dev,dtype=torch.uint8);y=torch.full((pad+count*width[ot]+256,),0x53,device=dev,dtype=torch.uint8)
      if source:x[pad:pad+len(source)].copy_(torch.from_numpy(np.frombuffer(source,np.uint8).copy()).to(dev))
      xp=x.data_ptr()+pad;yp=y.data_ptr()+pad;frozen=None;hashes=[]
      for repeat in range(8):
       y.fill_(0x53);ck(lib.cast64_contract_run_r3376(ctx,it,ot,count,0,xp,yp,stream.cuda_stream));b=gpu_bytes(y[pad:pad+count*width[ot]]);h=hashlib.sha256(b).hexdigest();hashes.append(h)
       if frozen is None:frozen=b
       assert b==frozen,('repeat',dev,it,ot,count,shift)
       assert bool((x[:pad]==0x6d).all()and(x[-256:]==0x6d).all()and(y[:pad]==0x53).all()and(y[-256:]==0x53).all())
       assert gpu_bytes(x[pad:pad+len(source)])==source
      row={'input_type':it,'output_type':ot,'count':count,'shift_bytes':shift,'input':record(source),'output':record(frozen),'repeat8_sha256':hashes,'input_readonly_and_guard_untouched':True}
      if count==4096 and shift==0:
       for bit,bout,bc,bgeom,ip,op in[(99,ot,count,0,xp,yp),(it,99,count,0,xp,yp),(it,ot,8193,0,xp,yp),(it,ot,count,1,xp,yp),(it,ot,count,2,xp,yp),(it,ot,count,3,xp,yp),(it,ot,count,4,xp,yp),(it,ot,count,0,xp+1,yp)]:
        y.fill_(0x53);code=lib.cast64_contract_run_r3376(ctx,bit,bout,bc,bgeom,ip,op,stream.cuda_stream);assert code!=0 and bool((y==0x53).all())and gpu_bytes(x[pad:pad+len(source)])==source,('invalid',dev,it,ot,bc,bgeom,code);q['invalid_checks']+=1
       if it!=ot:
        code=lib.cast64_contract_run_r3376(ctx,it,ot,64,0,xp,xp,stream.cuda_stream);assert code!=0 and gpu_bytes(x[pad:pad+len(source)])==source;q['invalid_checks']+=1
       bad=[];y.fill_(0x53);thread=threading.Thread(target=lambda:bad.append(lib.cast64_contract_run_r3376(ctx,it,ot,count,0,xp,yp,stream.cuda_stream)));thread.start();thread.join(20);assert not thread.is_alive()and bad==[1]and bool((y==0x53).all());q['wrongowner_untouched']=True
      if it==ot and count==513:
       ck(lib.cast64_contract_run_r3376(ctx,it,ot,count,0,xp,xp,stream.cuda_stream));assert gpu_bytes(x[pad:pad+len(source)])==source;q['identity_inplace_checks']+=1
      q['cases'].append(row)
    journal.flush();path.write_text(json.dumps(q,indent=2));print('CAST64_CONTRACT_R3376_PAIR',name,mode,dev,it,ot,flush=True)
  expected=160 if mode=='new'else 90;assert len(q['cases'])==expected;q['numeric_collection_complete']=True
 except BaseException as e:q['error']=repr(e);raise
 finally:
  closed=C.c_int(77);ck(lib.cast64_contract_destroy_r3376(ctx,C.byref(closed)));assert closed.value==1;q['Runtime_normally_closed_weak_expired']=True;journal.close();q['archive_sha256']=sha(archive);q['archive_bytes']=archive.stat().st_size;q['completed']=bool(q.get('numeric_collection_complete'));path.write_text(json.dumps(q,indent=2))
 return{'device':dev,'file':path.name,'sha256':sha(path),'cases':len(q['cases'])}
if __name__=='__main__':
 assert torch.cuda.device_count()==8
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:owners=list(pool.map(worker,range(8)))
 out=r/'results'/f'cast64-contract-r3376-{name}-{tag}-{mode}.json';assert not out.exists();out.write_text(json.dumps({'completed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':name,'mode':mode,'step':tag,'owners':owners},indent=2));print('CAST64_CONTRACT_R3376_ALL8_CLOSED',name,mode,flush=True)
