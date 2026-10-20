import pathlib,json,time,subprocess,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-pipeline-r3762-audit-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_launch'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(660):
  p=r/'kimi-spec-pipeline-r3761-launch.json'
  if p.exists():
   j=json.loads(p.read_text());aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',j['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]==j['label']]
   if aa and aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
    assert aa[0][2:4]==['COMPLETED','0:0'],aa;label='kimi-pipeline-r3762-audit';args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=10000M','-t','00:40:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(r/'incoming/audit_kimi_pipeline_spec_r3762.py')];p=subprocess.run(args,timeout=2500);assert p.returncode==0;d.update(state='audited',producer=aa[0]);break
  q=r/'kimi-pipeline-r3761-queue.json'
  if q.exists():assert json.loads(q.read_text())['state']!='failed',json.loads(q.read_text())
  time.sleep(60)
 else:raise TimeoutError('finite11h pipeline admission+producer wait')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
