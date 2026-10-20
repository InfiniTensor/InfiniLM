import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,subprocess,hashlib,statistics,datetime,traceback,math
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/glm-capcache-r3814-audit.json';assert not out.exists();d={'completed':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'whole_model_performance_qualified':False};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
try:
 aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='glm-capcache-r3812'];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'],aa;tag=aa[0][0];reports=[];samples=0;max_relative=0.;ratios=[]
 code=np.arange(256,dtype=np.uint16);ex=(code>>3)&15;ma=code&7;domain=np.where(ex,np.ldexp(1+ma.astype(np.float64)/8,ex.astype(int)-7),np.ldexp(ma.astype(np.float64),-9));domain=np.where(code&128,-domain,domain);domain[(ex==15)&(ma==7)]=np.nan
 expected=[(name,m,n,k,0,0)for name,n,k in [('qa',2048,6144),('output',6144,2048),('up',1536,6144),('down',6144,768)]for m in[32,33,64,65,128,256,511,1024,2048]]+[('kv',4096,3584,512,0,0),('offset',127,384,384,129,1)]
 for device in range(8):
  p=r/'results'/f'glm-capcache-r3806-{tag}-gpu{device}.json';q=json.loads(p.read_text());assert q['completed']and q['all_plans_destroyed']and q['device']==device
  assert q['baseline_helper_sha256']==sha(r/'candidates/models-5f91ad7-r3675/GLM/helpers/libglm_fp8_batched_r776.so')and q['candidate_helper_sha256']==sha(r/'candidates/glm-capcache-r3806/libglm_capcache_r3806.so')
  for n,h in q['source_files_sha256'].items():assert pathlib.Path(n).name==n and sha(r/'incoming'/n)==h
  assert len(q['cases'])==38
  for c,exp in zip(q['cases'],expected):
   assert tuple(c[k]for k in ['name','M','N','K','row_offset','column_offset'])==exp
   assert c['complete_BF16_matrix_exact_to_baseline']and c['changed_weight_and_scale_exact_to_baseline']and c['readonly_x_ws']
   s=c['cache_stats'];assert s[5]<=16 and s[6]<=768*1024**2 and s[7]<=768*1024**2 and s[0]-s[1]==s[5]and s[8]>0
   for key in ['capacity_cache','uncached_baseline','reused_baseline']:
    t=c[key]
    for prefix,values in [('host','host_repetitions_ms'),('stream_elapsed','stream_repetitions_ms')]:
     assert len(t[values])==5 and all(math.isfinite(v)and v>0 for v in t[values]);assert math.isclose(t[prefix+'_ms_median'],statistics.median(t[values]),rel_tol=1e-12)
   sample=r/'results'/c['sample_file'];assert sample.name==c['sample_file']and sha(sample)==c['sample_sha256']
   with np.load(sample,allow_pickle=False)as v:
    x,w=domain[v['x']],domain[v['w']];xs,ws=v['xs'],v['ws'];old,new=v['baseline'],v['candidate'];assert np.array_equal(old,new)and len(old)==3
    # Independent FP64 mathematical oracle; FP32 GEMM accumulation can differ.
    result=np.zeros(3,np.float64);co=c['column_offset'];k=c['K']
    for b in range(co//128,(co+k+127)//128):
     lo=max(0,b*128-co);hi=min(k,(b+1)*128-co);result+=np.sum(x[:,lo:hi]*w[:,lo:hi],axis=1)*xs[:,b]*ws[:,b]
    gpu=(old.astype(np.uint32)<<16).view(np.float32).astype(np.float64);assert np.isfinite(gpu).all()and np.isfinite(result).all();rel=np.abs(gpu-result)/np.maximum(np.abs(result),1.0);max_relative=max(max_relative,float(rel.max()));assert np.all(rel<=.012),('sample_CPU_math',device,exp,gpu.tolist(),result.tolist());samples+=3
   ratios.append({'device':device,'name':c['name'],'M':c['M'],'host_uncached_to_cache':c['host_speed_ratio_uncached_to_capacity'],'stream_uncached_to_cache':c['stream_ratio_uncached_to_capacity']})
  before,after=q['before_clear'],q['after_clear'];assert before[2]>0 and before[4]>0 and after[0]==after[1]and after[5]==after[6]==0
  reports.append({'device':device,'file':p.name,'sha256':sha(p),'before_clear':before,'after_clear':after})
 log=(r/'logs'/f'glm-capcache-r3812-{tag}.log').read_text(errors='replace');assert 'GLM_CAPCACHE_R3806_ALL8_COMPLETE'in log and not any(v in log for v in ['Traceback','AssertionError','oom_kill'])
 d.update(completed=True,producer=aa[0],devices=8,cases=304,independent_CPU_math_points=samples,CPU_math_max_relative_error=max_relative,complete_matrix_baseline_equality_producer_checked=True,changed_content_checked=True,all_plans_destroyed=True,reports=reports,ratios=ratios,host_ratio_median=statistics.median(v['host_uncached_to_cache']for v in ratios),stream_ratio_median=statistics.median(v['stream_uncached_to_cache']for v in ratios),source_sha256={n:sha(r/'incoming'/n)for n in ['audit_glm_capcache_r3814.py','glm_capcache_r3812_inside.sh']},scope='Cache storage/handle reuse only; full matrix equivalence and read-only guards reported by GPU producer, independent persisted samples and terminal/lifecycle audited. Timing contains launch/allocate/synchronization gaps, no kernel/hardware/model peak claim. CPU FP64 sample oracle uses 1.2% tolerance for BF16 result, not a replacement for full token model acceptance.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('GLM_CAPCACHE_R3814_AUDIT_COMPLETED',samples,d['host_ratio_median'],d['stream_ratio_median'])
