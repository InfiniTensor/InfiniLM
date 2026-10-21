import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-PP3-r3873-audit-queue.json';assert not out.exists();d={'state':'waiting_producer','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};out.write_text(json.dumps(d))
try:
 for _ in range(500):
  p=r/'glm-PP3-serving-r3871-queue.json'
  if p.exists():assert json.loads(p.read_text())['state']!='failed',json.loads(p.read_text())
  aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910381','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='glm-PP3-serving-r3871'];assert len(aa)<=1
  if aa and aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert aa[0][2:4]==['COMPLETED','0:0'],aa
   args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=16000M','-t','00:40:00','--job-name=glm-PP3-r3873-audit','--output='+str(r/'logs/glm-PP3-r3873-audit-%J.log'),'python3',str(r/'incoming/audit_GLM_PP3_r3873.py')];p=subprocess.run(args,timeout=2500);assert p.returncode==0;d.update(state='audited',producer=aa[0]);break
  time.sleep(60)
 else:raise TimeoutError('Finite producer wait expired')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
