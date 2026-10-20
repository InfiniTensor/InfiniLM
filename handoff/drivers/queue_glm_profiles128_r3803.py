import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-profile-r3803-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_exact_producer910371.381'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(180):
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910371','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='glm-large-r3721-rows128'];assert len(aa)==1 and aa[0][0]=='910371.381',aa
  if aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert aa[0][2:4]==['COMPLETED','0:0'],aa;label='glm-profile-r3803-audit-rows128';args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=10000M','-t','00:40:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','GLM_LARGE_AUDIT_ROWS=128','python3',str(r/'incoming/audit_glm_profiles_r3797.py')];p=subprocess.run(args,timeout=2500);assert p.returncode==0;d.update(state='audited',producer=aa[0],result='results/glm-large-r3797-rows128-audit.json');break
  time.sleep(60)
 else:raise TimeoutError('finite3h waiting existing128rowjob')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
