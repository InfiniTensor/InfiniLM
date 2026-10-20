import pathlib,json,time,subprocess,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-spec-pipeline-r3706-audit-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting','completed':False};out.write_text(json.dumps(d,indent=2));launch=json.loads((r/'kimi-spec-pipeline-r3703-launch.json').read_text())
try:
 for _ in range(510):
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]==launch['label']]
  if aa and aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:break
  time.sleep(60)
 else:raise TimeoutError('finite8.5h pipeline trial wait')
 d['producer']=aa[0]
 if aa[0][2:4]!=['COMPLETED','0:0']:d.update(state='producer_failed',completed=True)
 else:
  args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=8000M','-t','00:30:00','--job-name=kimi-spec-pipeline-r3706-audit','--output='+str(r/'logs/kimi-spec-pipeline-r3706-audit-%J.log'),'python3',str(r/'incoming/audit_kimi_pipeline_spec_r3706.py')];p=subprocess.run(args,timeout=1900);d.update(state='audited'if p.returncode==0 else'audit_failed',completed=True,returncode=p.returncode)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
