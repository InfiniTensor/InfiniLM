import pathlib,json,time,subprocess,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-KDA-pair-r3838-queue.json';assert not out.exists();d={'state':'waiting_postmortem_analysis'};out.write_text(json.dumps(d))
try:
 for _ in range(80):
  p=r/'results/kimi-root-r3836-postmortem.json'
  if p.exists():
   try:q=json.loads(p.read_text())
   except json.JSONDecodeError:time.sleep(1);continue
   assert q['completed']and q['diagnostic_trace_analysis_completed'],q.get('error');label='kimi-KDA-pair-r3838';p=subprocess.run(['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=12000M','-t','00:25:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'python3',str(r/'incoming/analyze_kimi_KDA_pair_r3838.py')],timeout=1600);assert p.returncode==0;d['state']='analyzed';break
  time.sleep(30)
 else:raise TimeoutError('Finite postmortem CPU wait elapsed')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
