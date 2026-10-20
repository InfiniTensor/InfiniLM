import pathlib,json,time,subprocess,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'deep-token-r3729-audit-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_producer','completed':False};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(180):
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910407','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='deep-token-r3699-mode5']
  if aa and aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:break
  time.sleep(30)
 else:raise TimeoutError('finite90min waiting producer')
 d['producer']=aa
 if aa[0][2:4]!=['COMPLETED','0:0']:d.update(state='producer_failed',completed=True)
 else:
  args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=8000M','-t','00:25:00','--job-name=deep-FP32-token-r3729-audit','--output='+str(r/'logs/deep-FP32-token-r3729-audit-%J.log'),'python3',str(r/'incoming/audit_deep_FP32_token_r3729.py')];p=subprocess.run(args,timeout=1600);d.update(state='audited'if p.returncode==0 else'audit_failed',completed=True,returncode=p.returncode)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
