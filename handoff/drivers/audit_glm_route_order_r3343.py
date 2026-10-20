import datetime,hashlib,json,math,mmap,pathlib,statistics,struct,subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/glm-route-order-r3343-audit.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'full_model_qualified':False}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
try:
 rows=[v.split('|')for v in subprocess.check_output(['sacct','-j','910374','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed,NodeList%160'],text=True).splitlines()if 'glm-route-order-r3328-component'in v];assert len(rows)==1 and rows[0][2:4]==['COMPLETED','0:0'];step=rows[0][0];d['producer_state']=rows[0]
 q=load(r/'results'/f'glm-route-order-r3328-{step}.json');assert q['completed']and not q.get('error')and len(q['owners'])==8
 proof=load(r/'results/glm-route-order-r3328-build.json');assert proof['completed'];base=r/'candidates/glm-route-order-r3328';assert sha(base/'libglm_route_order_r3328.so')==proof['library_sha256']and sha(base/'order.o')==proof['new_GPU_object_sha256']
 for n,h in proof['source_sha256'].items():assert sha(r/'incoming'/n)==h
 assert sha(r/'candidates/glm-batch-model-r3072/helpers/routed.o')==proof['current_routed_object_sha256']
 model=pathlib.Path.home()/'hygon-inference-20261013/models/ZhipuAI-GLM-5.3';index=load(model/'model.safetensors.index.json')['weight_map'];opened={}
 def tensor(key):
  filename=index[key]
  if filename not in opened:
   f=(model/filename).open('rb');mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);n=struct.unpack('<Q',mm[:8])[0];header=json.loads(mm[8:8+n]);opened[filename]=(f,mm,header,8+n)
  f,mm,header,start=opened[filename];v=header[key];a,b=v['data_offsets'];return mm,start+a,v['shape'],v['dtype']
 lut=[]
 for b in range(256):
  e=(b>>3)&15;m=b&7;x=(math.nan if e==15 and m==7 else math.ldexp(1+m/8,e-7))if e else math.ldexp(m,-9);lut.append(-x if b&128 else x)
 lut=np.asarray(lut,np.float64);summary={};samples=0;cases=0;normal=0;checks=0;refs=[]
 def archive(v):
  p=r/'results'/v['file'];assert p.stat().st_size==v['bytes']and sha(p)==v['sha256'];return p
 for owner in q['owners']:
  dev=owner['device'];assert owner['completed']and owner['all_weight_banks_readonly']and owner['Runtime_closures']==80 and len(owner['cases'])==10;normal+=80
  p=r/'results'/f'glm-route-order-r3328-{step}-gpu{dev}.json';assert load(p)==owner;refs.append({'file':p.name,'sha256':sha(p)})
  assert{(v['tokens'],v['pattern'],v['op'])for v in owner['cases']}=={(m,p,o)for m,p in[(8,'mixed'),(32,'mixed'),(512,'mixed'),(4096,'mixed'),(512,'one_expert')]for o in[0,1]}
  for c in owner['cases']:
   cases+=1;op=c['op'];tokens=c['tokens'];routes=tokens*8;offset=dev*256;assert c['offset']==offset and c['output_shape']==[routes,6144 if op else 256]
   assert c['fullraw_old_new_and_repeat_exact']and c['stable_permutation_verified']and c['invalid_untouched_count']==6;checks+=6
   m,k,n=(routes,256,6144)if op else(tokens,6144,256)
   x=np.memmap(archive(c['inputs']['x']),mode='r',dtype=np.uint8,shape=(m,k));xs=np.memmap(archive(c['inputs']['xs']),mode='r',dtype=np.float32,shape=(m,k//128));ids=np.memmap(archive(c['inputs']['ids']),mode='r',dtype=np.int64,shape=(routes,));assert np.isfinite(xs).all()and(ids>=0).all()and(ids<256).all()
   outputs=[]
   for j in range(1 if op else 2):
    a=c['outputs']['0'][j];b=c['outputs']['1'][j];assert a['sha256']==b['sha256']
    aa=np.memmap(archive(a),mode='r',dtype=np.uint16,shape=(routes,n));bb=np.memmap(archive(b),mode='r',dtype=np.uint16,shape=(routes,n));assert np.array_equal(aa,bb);assert np.isfinite((aa.astype(np.uint32)<<16).view(np.float32)).all();outputs.append(aa)
    for mode in ['0','1']:
     assert len(c['output_hashes_by_repeat'][mode])==4 and all(h[j]==a['sha256']for h in c['output_hashes_by_repeat'][mode])
   for j,y in enumerate(outputs):
    for route in sorted({0,routes//2,routes-1}):
     e=int(ids[route]);proj='down_proj'if op else('gate_proj'if j==0 else'up_proj');prefix=f'model.layers.{owner["layer"]}.mlp.experts.{e}.{proj}';wm,wa,wshape,wdtype=tensor(prefix+'.weight');sm,sa,sshape,sdtype=tensor(prefix+'.weight_scale_inv');assert wshape==([6144,2048]if op else[2048,6144])and sdtype=='F32'
     for column in sorted({0,n//2,n-1}):
      wr=column if op else offset+column;kc=offset if op else 0;raw=np.frombuffer(wm,dtype=np.uint8,count=k,offset=wa+wr*wshape[1]+kc);v=lut[raw];inp=lut[x[route if op else route//8]];assert np.isfinite(v).all()and np.isfinite(inp).all();expected=0.
      for b in range(k//128):
       scindex=(wr//128)*sshape[1]+(offset//128+b if op else b);scale=struct.unpack_from('<f',sm,sa+scindex*4)[0];expected+=float(np.dot(inp[b*128:(b+1)*128],v[b*128:(b+1)*128]))*float(xs[route if op else route//8,b])*scale
      actual=struct.unpack('<f',struct.pack('<I',int(y[route,column])<<16))[0];assert math.isfinite(expected)and abs(actual-expected)<=.005*abs(expected)+.001,('CPU_FP64_sample',dev,c['label'],j,route,column,actual,expected);samples+=1
   A=statistics.median(c['GPU_ms']['0']);B=statistics.median(c['GPU_ms']['1']);assert A>0 and B>0 and math.isclose(A/B,c['kernel_plus_stable_order_speed_ratio'],rel_tol=1e-12);summary.setdefault((tokens,c['pattern'],op),[]).append(A/B)
 del raw
 for f,mm,h,s in opened.values():mm.close();f.close()
 log=(r/'logs'/f'glm-route-order-r3328-component-{step}.log').read_text();assert'SHARED_CACHED_R1780_COMPLETE'in log and not any(s in log for s in ['Traceback','AssertionError','Segmentation fault'])
 d.update(completed=True,cases=cases,CPU_independently_recomputed_actual_weight_dot_samples=samples,Runtime_normal_closures=normal,invalid_untouched_checks=checks,fullraw_old_new_repeat_exact=True,summary=[{'tokens':k[0],'pattern':k[1],'op':k[2],'median_speed_ratio':statistics.median(v),'min_speed_ratio':min(v),'max_speed_ratio':max(v)}for k,v in sorted(summary.items())],build_sha256=sha(r/'results/glm-route-order-r3328-build.json'),report_sha256=sha(r/'results'/f'glm-route-order-r3328-{step}.json'),owners=refs,scope='Actual256expert layer3 FP8 TP8local components,80cases/8owners. Allarchivedinputs/outputhashes/fullrawcomparison/finite/CPUFP64dot samples fromoriginalcheckpoint. Exactcurrentr19dot tree, stableexpert launchorder; GPUevent includessort, excludesRuntimecreate/CPUvalidation/scratchallocation (wall includes). No complete78model/sustainedHBM/hardwarepeak/TPS adoption.')
except BaseException as e:
 import traceback
 d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('GLM_ROUTE_ORDER_R3343_CPU_AUDIT_COMPLETE',cases,samples,flush=True)
