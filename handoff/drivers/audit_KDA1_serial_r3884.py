import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,subprocess,datetime,traceback,statistics
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-KDA1-serial-r3884-audit.json';assert not out.exists();d={'completed':False,'main_goal_metric':False,'wholemodel_qualified':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
 aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-KDA1-replay-r3883'];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'],aa
 tag=aa[0][0];rows=[];all_serial=True;control=True;ratios=[]
 def read(z):
  p=r/'results'/z['file'];b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==z['sha256'];a=np.frombuffer(b,np.uint16).reshape(z['shape']);assert np.isfinite((a.astype(np.uint32)<<16).view(np.float32)).all();return a
 for k in range(8):
  q=json.loads((r/'results'/f'kimi-KDA1-replay-r3883-{tag}-rank{k}.json').read_text());assert q['completed']and q['Runtime_and_communicator_normally_closed']and q['rank']==k
  old=json.loads((r/'results'/f'kimi-KDA1-replay-r3866-910373.831-rank{k}.json').read_text())
  for c,oc in zip(q['cases'],old['cases']):
   assert c['label']==oc['label']and c['input_weight_readonly'];ref=read(c['serial_M1_reference']);vvs={}
   for v in c['projection_variants']:
    a=read(v['output']);dif=int(np.count_nonzero(a!=ref));assert dif==v['different_elements'];vvs[v['variant']]=v
    if v['variant']==2:all_serial&=dif==0
    if v['variant']==0:control&=np.array_equal(a,read(oc['local']))
    rows.append({'rank':k,'label':c['label'],'variant':v['variant'],'name':v['name'],'M1_diagnostic_different_elements':dif,'host_inclusive_median_us':statistics.median(v['wall_seconds_host_inclusive'])*1e6})
   if c['M']==2:ratios.append(statistics.median(vvs[2]['wall_seconds_host_inclusive'])/statistics.median(vvs[0]['wall_seconds_host_inclusive']))
 d.update(completed=True,producer=aa[0],normal_owners_closed=8,all_sequential_M1_component_outputs_equal_serial_reference=bool(all_serial),ordinary_2D_control_reproduces_prior_real_input_replay=bool(control),local_sequential_to_2D_host_cost_ratio_median=statistics.median(ratios),rows=rows,scope='Actual KDA1 projection fixture only. Raw comparisons diagnose counterfactual local projection, not model acceptance. Host-inclusive component ratio is not a model speedup. Complete frozen IDs/EOS/owner/Slurm and throughput after any model integration remain untested.',model_fix_adopted=False)
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
