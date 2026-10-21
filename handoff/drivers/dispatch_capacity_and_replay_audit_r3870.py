import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'followups-r3870-launch.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910381,910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()];assert any(v[:4]==['910381.323','glm-PP3-capacity-r3865','COMPLETED','0:0']for v in aa)
allq={}
for step in['910381.322','910381.323']:
 rr=[json.loads(p.read_text())for p in(r/'results').glob(f'glm-PP3-constructor-r3857-{step}-stage*-rank*.json')];assert len(rr)==24 and all(q['completed']and q['all24_memory_admitted']and q['all_contexts_closed_without_loading']and q['owner_affinity']['verified']for q in rr);allq[step]={(q['stage'],q['rank']):q for q in rr}
predictions=[]
for k,q32 in allq['910381.323'].items():
 q16=allq['910381.322'][k];slope=(q16['free_total_after_create'][0]-q32['free_total_after_create'][0])/16;predictions.append({'stage':k[0],'rank':k[1],'free_GiB_rows32':q32['free_total_after_create'][0]/1024**3,'predicted_free_GiB_rows64':(q32['free_total_after_create'][0]-32*slope)/1024**3,'predicted_free_GiB_rows96':(q32['free_total_after_create'][0]-64*slope)/1024**3})
assert min(v['predicted_free_GiB_rows64']for v in predictions)>9
d['capacity_forecast']=predictions
base=json.loads((r/'glm-PP3-capacity-r3857-launch.json').read_text());args=[v.replace('glm-PP3-capacity-r3857','glm-PP3-capacity-r3870').replace('GLM_CACHE_ROWS_R3627=16','GLM_CACHE_ROWS_R3627=64')for v in base['args']]
with(r/'logs/glm-PP3-capacity-r3870-controller.log').open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['GLM_rows64']={'pid':p.pid,'args':args,'rows':64,'performance_qualified':False}
assert any(v[1:4]==['kimi-KDA1-replay-r3866','COMPLETED','0:0']for v in aa),aa[-5:]
args=['srun','--jobid=910377','--overlap','--exact','-N1','-n1','-c2','-w','COMPUTE_NODE_3181eb8b','--gres=none','--mem=12000M','-t','00:15:00','--job-name=kimi-KDA1-replay-r3868-audit','--output='+str(r/'logs/kimi-KDA1-replay-r3868-audit-%J.log'),'python3',str(r/'incoming/audit_KDA1_replay_r3868.py')]
with(r/'logs/kimi-KDA1-replay-r3868-audit-controller.log').open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['Kimi_replay_audit']={'pid':p.pid,'args':args};out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
