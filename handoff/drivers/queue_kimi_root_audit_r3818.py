import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-root-r3818-audit-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_root_producer'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(250):
  aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910374','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-root-r3810'];assert len(aa)==1
  if aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert aa[0][2:4]==['COMPLETED','0:0'],aa;label='kimi-root-r3818-audit';args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=12000M','-t','00:40:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(r/'incoming/audit_kimi_root_r3818.py')];p=subprocess.run(args,timeout=2500);assert p.returncode==0;d.update(state='audited',producer=aa[0]);break
  time.sleep(60)
 else:raise TimeoutError('Finite roottrace wait elapsed')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
