import pathlib,json,time,subprocess,traceback,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-KDA-pair-r3830-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_complete_root_audit'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(260):
  p=r/'results/kimi-root-r3818-audit.json'
  if p.exists():
   try:q=json.loads(p.read_text())
   except json.JSONDecodeError:time.sleep(1);continue
   assert q['completed'],q.get('error');label='kimi-KDA-pair-r3830';args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=12000M','-t','00:25:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(r/'incoming/analyze_kimi_KDA_pair_r3830.py')];p=subprocess.run(args,timeout=1600);assert p.returncode==0;d['state']='analyzed';break
  a=r/'kimi-root-r3818-audit-queue.json'
  if a.exists():assert json.loads(a.read_text())['state']!='failed'
  time.sleep(60)
 else:raise TimeoutError('Finite selectedKDA analysis wait elapsed')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
