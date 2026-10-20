import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'followup-audits-r3756-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_producers','completed':False,'jobs':{}};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(600):
  targets=[];p=r/'glm-capacity-r3721-launch.json'
  if p.exists():targets +=[('GLM'+j['label'].split('rows')[1],j,'audit_glm_large_r3756.py',['GLM_LARGE_AUDIT_ROWS='+j['label'].split('rows')[1]])for j in json.loads(p.read_text())['jobs']]
  p=r/'kimi-spec-pipeline-r3736-launch.json'
  if p.exists():targets.append(('KimiPipeline',json.loads(p.read_text()),'audit_kimi_pipeline_spec_r3753.py',[]))
  for key,j,script,env in targets:
   if key in d['jobs']:continue
   aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',j['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]==j['label']]
   if not aa or aa[0][2]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:continue
   if aa[0][2:4]!=['COMPLETED','0:0']:d['jobs'][key]={'state':'producer_failed','producer':aa[0]}
   else:
    label='followup-audit-r3756-'+key;args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=10000M','-t','00:40:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env',*env,'python3',str(r/'incoming'/script)];p=subprocess.run(args,timeout=2500);d['jobs'][key]={'state':'audited'if p.returncode==0 else'audit_failed','producer':aa[0],'returncode':p.returncode}
   out.write_text(json.dumps(d,indent=2))
  p=r/'kimi-pipeline-r3750-queue.json';pipeline_closed=p.exists()and json.loads(p.read_text())['state']=='failed'
  if 'GLM96'in d['jobs']and'GLM128'in d['jobs']and('KimiPipeline'in d['jobs']or pipeline_closed):d.update(state='finished',completed=True);break
  time.sleep(60)
 else:raise TimeoutError('finite10h audit wait')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
