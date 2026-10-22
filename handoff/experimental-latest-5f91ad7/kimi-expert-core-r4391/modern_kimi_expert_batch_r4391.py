import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,ctypes,time,statistics,hashlib,datetime,contextlib,gc,math
import numpy as np
import torch,torch.multiprocessing as mp
from safetensors import safe_open
r=pathlib.Path('/refactor');base=r/'candidates/kimi-expert-batch-5f91ad7-r4391';model=pathlib.Path('/shared/Kimi-K3');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def worker(rank):
 torch.set_num_threads(2);torch.cuda.set_device(rank);torch.backends.cuda.matmul.allow_tf32=False
 lib=ctypes.CDLL(str(base/'libmodern_kimi_expert_batch_r4391.so'));lib.modern_expert_error_r4391.restype=ctypes.c_char_p;lib.modern_expert_create_r4391.argtypes=[ctypes.c_int];lib.modern_expert_create_r4391.restype=ctypes.c_void_p;lib.modern_expert_destroy_r4391.argtypes=[ctypes.c_void_p]
 lib.modern_expert_forward_r4391.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_size_t]+[ctypes.c_void_p]*9
 lib.modern_kimi_tiled_raw_r1861.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.c_size_t,ctypes.c_void_p]
 lib.modern_kimi_tiled_batch_r4391.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.c_size_t,ctypes.c_size_t,ctypes.c_void_p]
 out=r/'results'/f'kimi-expert-batch-r4391-rank{rank}.json';assert not out.exists();rawdir=r/'results'/f'kimi-expert-batch-r4391-rank{rank}-raw';rawdir.mkdir(exist_ok=False)
 d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'rank':rank,'cases':[],'library_sha256':sha(base/'libmodern_kimi_expert_batch_r4391.so'),'full_model_qualified':False,'scope':'ExpertcorepreTP-reduction only, realfirstMoE selected64experts/currentTP8localshard geometry, seededBF16latentinput/routingfixtures notcapturedtargetroutes. Exactcurrenttile.oN1 vsbatchedsamekernels; nativecurrentFusedMoeprovider1comparison diagnostic. No router/up/shared/TP/acceptance orModelperformancequalification.'}
 def save():
  p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(out)
 def check(code):
  if code:raise RuntimeError((code,lib.modern_expert_error_r4391().decode()))
 def ptr(t):return ctypes.c_void_p(t.data_ptr())
 def bits(t):return t.detach().cpu().view(torch.uint16).numpy().copy()
 def fp(bits):return(bits.astype(np.uint32)<<16).view(np.float32)
 ctx=None;graphs=[];save()
 try:
  cfg=json.loads((model/'config.json').read_text())['text_config'];assert cfg['hidden_size']==7168 and cfg['routed_expert_hidden_size']==3584 and cfg['moe_intermediate_size']//8==384
  mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];prefix='language_model.model.layers.1.block_sparse_moe.experts.';E=64
  host={'w13':torch.empty((E,768,1792),dtype=torch.uint8),'s13':torch.empty((E,768,112),dtype=torch.uint8),'w2':torch.empty((E,3584,192),dtype=torch.uint8),'s2':torch.empty((E,3584,12),dtype=torch.uint8)}
  tick=time.perf_counter();files={}
  with contextlib.ExitStack()as stack:
   def get(expert,name,axis,width):
    key=prefix+str(expert)+'.'+name;fn=mapping[key]
    if fn not in files:files[fn]=stack.enter_context(safe_open(model/fn,framework='pt',device='cpu'))
    view=files[fn].get_slice(key);sl=[slice(None)]*len(view.get_shape());sl[axis]=slice(rank*width,(rank+1)*width);v=view[tuple(sl)].contiguous();assert v.dtype==torch.uint8;return v
   for e in range(E):
    host['w13'][e,:384].copy_(get(e,'w1.weight_packed',0,384));host['w13'][e,384:].copy_(get(e,'w3.weight_packed',0,384));host['s13'][e,:384].copy_(get(e,'w1.weight_scale',0,384));host['s13'][e,384:].copy_(get(e,'w3.weight_scale',0,384));host['w2'][e].copy_(get(e,'w2.weight_packed',1,192));host['s2'][e].copy_(get(e,'w2.weight_scale',1,12))
  d['partial_weights_load_seconds']=time.perf_counter()-tick;d['selected_checkpoint_tensors']=E*6;d['host_weights_sha256']={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()for k,v in host.items()};d['config_sha256']=sha(model/'config.json');d['index_sha256']=sha(model/'model.safetensors.index.json');weights={k:v.cuda()for k,v in host.items()};torch.cuda.synchronize();ctx=lib.modern_expert_create_r4391(rank);assert ctx,lib.modern_expert_error_r4391();save()
  table=np.asarray([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],np.float64)
  def decode(p,s):
   z=p.numpy();v=np.stack([table[z&15],table[z>>4]],-1).reshape(-1);sc=np.exp2(s.numpy().astype(np.float64)-127).repeat(32);assert len(v)==len(sc);return v*sc
  for T in[1,2,4,8,16]:
   for layout in[0,1]:
    rng=np.random.default_rng(4391+T*10+layout);x_cpu=torch.from_numpy(rng.standard_normal((T,3584),dtype=np.float32)*.1).to(torch.bfloat16);ids_cpu=np.stack([np.arange(16,dtype=np.int32)if layout==0 else rng.choice(E,16,replace=False).astype(np.int32)for _ in range(T)]);rw=rng.uniform(.01,1,(T,16)).astype(np.float32);rw/=rw.sum(-1,keepdims=True);x=x_cpu.cuda();ids=torch.from_numpy(ids_cpu).cuda();routes=torch.from_numpy(rw).cuda();a0=torch.empty((T,16,384),device='cuda',dtype=torch.bfloat16);a1=torch.empty_like(a0);y0=torch.empty_like(x);y1=torch.empty_like(x);yn=torch.empty_like(x);stream=ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    wp=[ptr(weights[k])for k in['w13','s13','w2','s2']]
    def serial():
     for t in range(T):check(lib.modern_kimi_tiled_raw_r1861(rank,ptr(y0[t]),ptr(a0[t]),ptr(x[t]),ptr(ids[t]),ptr(routes[t]),*wp,E,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
    def batch():check(lib.modern_kimi_tiled_batch_r4391(rank,ptr(y1),ptr(a1),ptr(x),ptr(ids),ptr(routes),*wp,E,T,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)))
    def native():check(lib.modern_expert_forward_r4391(ctx,T,E,ptr(yn),ptr(x),ptr(ids),ptr(routes),*wp,stream))
    serial();batch();native();torch.cuda.synchronize();aa,bb,yy,zz,nn=bits(a0),bits(a1),bits(y0),bits(y1),bits(yn);assert np.isfinite(fp(aa)).all()and np.isfinite(fp(yy)).all()and np.isfinite(fp(nn)).all();assert np.array_equal(aa,bb)and np.array_equal(yy,zz),('Batchdiff',rank,T,layout,int(np.count_nonzero(aa!=bb)),int(np.count_nonzero(yy!=zz)))
    sampled=[];af=fp(bb);xf=x_cpu.float().numpy().astype(np.float64)
    for t,k,j in[(0,0,0),(T-1,15,17),(T-1,0,383)]:
     e=int(ids_cpu[t,k]);g=float(xf[t]@decode(host['w13'][e,j],host['s13'][e,j]));u=float(xf[t]@decode(host['w13'][e,384+j],host['s13'][e,384+j]));expected=4*math.tanh(g/4)*(1/(1+math.exp(-max(-80,min(80,g)))))*25*math.tanh(u/25);actual=float(af[t,k,j]);assert abs(actual-expected)<=.01+.02*abs(expected),('W13oracle',actual,expected);sampled.append({'part':'W13activation','actual':actual,'CPU_F64':expected})
    for t,h in[(0,0),(T-1,17),(T-1,3583)]:
     parts=[float(af[t,k].astype(np.float64)@decode(host['w2'][int(ids_cpu[t,k]),h],host['s2'][int(ids_cpu[t,k]),h]))*float(rw[t,k])for k in range(16)];expected=sum(parts);actual=float(fp(zz)[t,h]);assert abs(actual-expected)<=.004*abs(expected)+5e-5*sum(map(abs,parts))+1e-6,('W2oracle',actual,expected);sampled.append({'part':'W2output','actual':actual,'CPU_F64':expected})
    for _ in range(3):serial();batch()
    torch.cuda.synchronize();eager=[[],[]];gpu=[[],[]]
    for rep in range(10):
     for mode in([0,1]if rep%2==0 else[1,0]):
      a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);tick=time.perf_counter();a.record();[serial,batch][mode]();b.record();b.synchronize();eager[mode].append((time.perf_counter()-tick)*1000);gpu[mode].append(a.elapsed_time(b))
    current_graphs=[]
    for fn in[serial,batch]:
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):fn()
     current_graphs.append(graph);graphs.append(graph)
    # Prove both graphs contain work, write their full outputs, and read updated inputs.
    for mode in[0,1]:
     [a0,a1][mode].fill_(float('nan'));[y0,y1][mode].fill_(float('nan'));current_graphs[mode].replay();torch.cuda.synchronize();assert np.array_equal(bits([a0,a1][mode]),aa)and np.array_equal(bits([y0,y1][mode]),yy),('graph_sentinel_not_overwritten',mode)
    x.mul_(-.5);serial();torch.cuda.synchronize();changed_a,changed_y=bits(a0),bits(y0);assert not np.array_equal(changed_y,yy)
    for mode in[0,1]:
     [a0,a1][mode].fill_(float('nan'));[y0,y1][mode].fill_(float('nan'));current_graphs[mode].replay();torch.cuda.synchronize();assert np.array_equal(bits([a0,a1][mode]),changed_a)and np.array_equal(bits([y0,y1][mode]),changed_y),('graph_input_update_mismatch',mode)
    x.copy_(x_cpu);current_graphs[0].replay();current_graphs[1].replay();torch.cuda.synchronize();assert np.array_equal(bits(a0),aa)and np.array_equal(bits(a1),aa)and np.array_equal(bits(y0),yy)and np.array_equal(bits(y1),yy)
    grapht=[[],[]]
    for rep in range(10):
     for mode in([0,1]if rep%2==0 else[1,0]):tick=time.perf_counter();current_graphs[mode].replay();torch.cuda.synchronize();grapht[mode].append((time.perf_counter()-tick)*1000)
    assert np.array_equal(bits(a0),bits(a1))and np.array_equal(bits(y0),bits(y1));row={'tokens':T,'layout':layout,'batch_vs_serial_activations_exact':True,'batch_vs_serial_outputs_exact':True,'native_provider1_differences_vs_serial':int(np.count_nonzero(nn!=yy)),'native_provider1_max_abs_difference':float(np.abs(fp(nn)-fp(yy)).max()),'graph_sentinel_write_verified':True,'graph_updated_inputs_match_fresh_eager':True,'sampled_independent_CPU_F64_oracle':sampled,'serial_eager_host_ms':statistics.median(eager[0]),'batch_eager_host_ms':statistics.median(eager[1]),'serial_GPU_event_span_ms':statistics.median(gpu[0]),'batch_GPU_event_span_ms':statistics.median(gpu[1]),'serial_graph_host_ms':statistics.median(grapht[0]),'batch_graph_host_ms':statistics.median(grapht[1]),'eager_host_samples_ms':eager,'graph_host_samples_ms':grapht,'raw':[]}
    for label,arr in[('serial_activation',aa),('batch_activation',bb),('serial_output',yy),('batch_output',zz),('native_provider1_output',nn)]:
     p=rawdir/f'T{T}-layout{layout}-{label}.bin';arr.tofile(p);row['raw'].append({'file':p.relative_to(r/'results').as_posix(),'bytes':p.stat().st_size,'sha256':sha(p),'shape':list(arr.shape),'label':label,'dtype':'BF16'})
    d['cases'].append(row);save();torch.cuda.synchronize();current_graphs.clear();graphs.clear();del graph,x,ids,routes,a0,a1,y0,y1,yn;gc.collect();torch.cuda.empty_cache()
  check(lib.modern_expert_destroy_r4391(ctx));ctx=None;d['native_Runtime_owner_shutdown_completed']=True;del weights;torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();d['completed']=True;save();print('KIMI_EXPERT_BATCH_R4391_COMPLETE',rank,flush=True)
 except BaseException as exc:d['error']=repr(exc);save();raise
 finally:
  torch.cuda.synchronize();graphs.clear()
  if ctx:check(lib.modern_expert_destroy_r4391(ctx))
if __name__=='__main__':
 mp.spawn(worker,nprocs=8,join=True)
 rows=[json.loads((r/'results'/f'kimi-expert-batch-r4391-rank{d}.json').read_text())for d in range(8)];assert all(q['completed']and q['native_Runtime_owner_shutdown_completed']and len(q['cases'])==10 for q in rows);out=r/'results/kimi-expert-batch-r4391.json';assert not out.exists();out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'normal_native_Runtime_owners':8,'rank_reports':[f'kimi-expert-batch-r4391-rank{d}.json'for d in range(8)],'scope':'Expertcorecomponent only/currenttile.o serialvsbatched andnativeprovider1; nofullTP/MoE/gating/up/shared/targettoken/speedqualification.'},indent=2));print('KIMI_EXPERT_BATCH_R4391_ALL8_COMPLETE',flush=True)
