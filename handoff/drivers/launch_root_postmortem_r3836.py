import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-root-r3836-postmortem-start.json';assert not out.exists();label='kimi-root-r3836-postmortem';args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=12000M','-t','00:40:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(r/'incoming/audit_kimi_root_postmortem_r3836.py')]
with(r/'logs'/(label+'-controller.log')).open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'args':args},indent=2));print(json.dumps({'submitted':True,'pid':p.pid}))
