import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'tip-serving-r3698-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_builds','jobs':[]};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(120):
  ready=all((r/'results'/n).exists()and json.loads((r/'results'/n).read_text())['completed']for n in['glm-scale-r3694-build.json','deep-attention-FP32-r3688-build.json'])
  if ready:break
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if any(x in a.split('|')[1]for x in['tip-trial-build-r3694','tip-trial-build-r3688'])];assert not any(a[2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING','COMPLETED']for a in aa),aa;time.sleep(60)
 else:raise TimeoutError('finite2h build wait')
 for parent,nodes,label,env in[
  ('910371',['COMPUTE_NODE_b169aebc','COMPUTE_NODE_0bc6a5de','COMPUTE_NODE_66463f9f','COMPUTE_NODE_f4161461'],'glm-large-r3698-rows128',{'GLM_CACHE_ROWS_R3627':'128','GLM_CACHE_CAP_R3627':'640','GLM_PP_SIZE_R3603':'4','GLM_NETWORK_TRIAL':'shca','WAVE_INSIDE_R2323':'glm_large_r3698_inside.sh'}),
  ('910376',['COMPUTE_NODE_4424e77b','COMPUTE_NODE_34a8f993','COMPUTE_NODE_ba45ccc4','COMPUTE_NODE_8b8806cd'],'glm-large-r3698-rows96',{'GLM_CACHE_ROWS_R3627':'96','GLM_CACHE_CAP_R3627':'640','GLM_PP_SIZE_R3603':'4','GLM_NETWORK_TRIAL':'shca','WAVE_INSIDE_R2323':'glm_large_r3698_inside.sh'}),
  ('910407',['COMPUTE_NODE_fb2bb4d3'],'deep-token-r3699-mode5',{'DEEP_ATTENTION_BATCH_R3474':'8','DEEP_TOKEN_MODE':'5','DEEP_USE_RAMCACHE_R3616':'0','WAVE_INSIDE_R2323':'deep_attention_precision_r3699_inside.sh'})]:
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',parent,'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()];assert not any('.'in a[0]and not a[0].endswith(('.batch','.extern'))and a[2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']for a in aa),aa
  env.update(INFINILM_OWNER_CPU_WIDTH_R2392='4',GLM_PIPELINE_TAIL_NODE=nodes[-1]);args=['srun','--export=ALL,'+','.join(k+'='+v for k,v in env.items()),'--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N'+str(len(nodes)),'-n'+str(len(nodes)),'--ntasks-per-node=1','-c64','-w',','.join(nodes),'--gres=dcu:8','--mem=240000M','-t','08:00:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J-%t.log')),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  with(r/'logs'/(label+'-controller.log')).open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  d['jobs'].append({'parent':parent,'nodes':nodes,'label':label,'pid':p.pid,'args':args});out.write_text(json.dumps(d,indent=2))
 d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
