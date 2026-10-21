import pathlib,json,subprocess,time,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-KDA1-serial-r3884-audit-queue.json';assert not out.exists();d={'state':'waiting_producer'};out.write_text(json.dumps(d))
try:
 for _ in range(40):
  aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-KDA1-replay-r3883'];assert len(aa)<=1
  if aa and aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert aa[0][2:4]==['COMPLETED','0:0'],aa
   args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=10000M','-t','00:15:00','--job-name=kimi-KDA1-serial-r3884-audit','--output='+str(r/'logs/kimi-KDA1-serial-r3884-audit-%J.log'),'python3',str(r/'incoming/audit_KDA1_serial_r3884.py')];p=subprocess.run(args,timeout=1000);assert p.returncode==0;d.update(state='audited',producer=aa[0]);break
  time.sleep(30)
 else:raise TimeoutError('Finite component wait elapsed')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
