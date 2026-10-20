import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-idle-r3773-audit-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_launch','jobs':{},'completed':False};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(120):
  p=r/'glm-idle-serving-r3772-launch.json'
  if p.exists()and len(json.loads(p.read_text())['jobs'])==2:break
  q=r/'glm-idle-serving-r3772-queue.json'
  if q.exists():assert json.loads(q.read_text())['state']!='failed',json.loads(q.read_text())
  time.sleep(30)
 else:raise TimeoutError('finite1h build/launch wait')
 jobs=json.loads(p.read_text())['jobs'];d['state']='waiting_producers';out.write_text(json.dumps(d,indent=2))
 for _ in range(540):
  for j in jobs:
   key=str(j['PP'])
   if key in d['jobs']:continue
   aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',j['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]==j['label']]
   if not aa or aa[0][2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:continue
   if aa[0][2:4]!=['COMPLETED','0:0']:d['jobs'][key]={'state':'producer_failed','producer':aa[0]}
   else:
    rejection=sorted((r/'results').glob(f'glm-idle-capacity-r3772-{aa[0][0]}-stage*-rank*.json'))
    if rejection:
     assert len(rejection)==8*j['PP'] and all(json.loads(p.read_text())['capacity_rejected_after_idle_trim']and json.loads(p.read_text())['all_contexts_closed_after_memory_rejection']for p in rejection);d['jobs'][key]={'state':'capacity_rejected_all_owners_closed','producer':aa[0],'normal_owners':len(rejection),'performance_qualified':False}
    else:
     label='glm-idle-r3773-audit-PP'+key;args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=10000M','-t','00:40:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','GLM_IDLE_AUDIT_PP='+key,'python3',str(r/'incoming/audit_glm_idle_r3773.py')];p=subprocess.run(args,timeout=2500);d['jobs'][key]={'state':'audited'if p.returncode==0 else'audit_failed','producer':aa[0],'returncode':p.returncode}
   out.write_text(json.dumps(d,indent=2))
  if len(d['jobs'])==2:d.update(state='finished',completed=True);break
  time.sleep(60)
 else:raise TimeoutError('finite9h producer wait')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
