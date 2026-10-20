import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-alias-runtime-r3776-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_build'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(90):
  p=r/'results/kimi-spec-alias-r3775-build.json'
  if p.exists():assert json.loads(p.read_text())['completed'];break
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%100,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='kimi-alias-r3775-build'];assert not any(a[2]not in['RUNNING','PENDING','COMPLETING','COMPLETED']for a in aa),aa;time.sleep(30)
 else:raise TimeoutError('45min guard build wait')
 aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%100,State,ExitCode,NodeList%180'],text=True).splitlines()]
 for a in aa:
  if '.'in a[0]and not a[0].endswith(('.batch','.extern'))and a[2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:assert'COMPUTE_NODE_a2faf214'not in subprocess.check_output(['scontrol','show','hostnames',a[4]],text=True).splitlines()
 label='kimi-alias-runtime-r3776';args=['srun','--jobid=910373','--overlap','--exact','-N1','-n1','-c32','-w','COMPUTE_NODE_a2faf214','--gres=dcu:8','--mem=16000M','-t','00:25:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=kda_alias_runtime_r3775_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(r/'logs'/(label+'-controller.log')).open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='submitted',pid=p.pid,args=args,label=label)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
