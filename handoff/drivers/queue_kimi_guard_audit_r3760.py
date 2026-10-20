import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-spec-math-r3760-audit-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_launch','modes':{},'completed':False};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(180):
  p=r/'kimi-spec-r3759-launch.json'
  if p.exists()and len(json.loads(p.read_text())['jobs'])==2:break
  q=r/'kimi-guard-r3759-trial-queue.json'
  if q.exists():assert json.loads(q.read_text())['state']!='failed',json.loads(q.read_text())
  time.sleep(60)
 else:raise TimeoutError('finite3h waiting launch')
 jobs=json.loads(p.read_text())['jobs'];d['state']='waiting_producers';out.write_text(json.dumps(d,indent=2))
 for _ in range(510):
  for j in jobs:
   mode=j['mode']
   if str(mode)in d['modes']:continue
   aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',j['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]==j['label']]
   if not aa or aa[0][2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:continue
   if aa[0][2:4]!=['COMPLETED','0:0']:d['modes'][str(mode)]={'state':'producer_failed','producer':aa[0]}
   else:
    label=f'kimi-spec-math-r3760-audit-M{mode}';args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=8000M','-t','00:30:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','KIMI_SPEC_MATH_AUDIT_MODE='+str(mode),'python3',str(r/'incoming/audit_kimi_guard_math_r3760.py')];p=subprocess.run(args,timeout=1900);d['modes'][str(mode)]={'state':'audited'if p.returncode==0 else'audit_failed','producer':aa[0],'returncode':p.returncode}
   out.write_text(json.dumps(d,indent=2))
  if len(d['modes'])==2:d.update(state='finished',completed=True);break
  time.sleep(60)
 else:raise TimeoutError('finite8.5h waiting model trials')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
