import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-shared-rope-component-r3788-launch.json';assert not out.exists();node='COMPUTE_NODE_08e5cdc7';aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%180'],text=True).splitlines()]
for a in aa:
 if '.'in a[0]and not a[0].endswith(('.batch','.extern'))and a[2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:assert node not in subprocess.check_output(['scontrol','show','hostnames',a[4]],text=True).splitlines(),a
label='glm-shared-rope-component-r3788';args=['srun','--jobid=910373','--overlap','--exact','-N1','-n1','-c32','-w',node,'--gres=dcu:8','--mem=16000M','-t','00:20:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=glm_shared_rope_contract_r3788_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs'/(label+'-controller.log')).open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'label':label,'args':args},indent=2));print(json.dumps({'submitted':True,'pid':p.pid}))
