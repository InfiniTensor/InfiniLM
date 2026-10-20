import pathlib,json,subprocess,time,datetime,hashlib,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-pipeline-r3761-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_wholemodel_B1_QA_and_frozen_AR_repeat'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(360):
  lp=r/'kimi-spec-r3759-launch.json'
  if lp.exists()and len(json.loads(lp.read_text())['jobs'])==2:break
  tq=r/'kimi-guard-r3759-trial-queue.json'
  if tq.exists():assert json.loads(tq.read_text())['state']!='failed',json.loads(tq.read_text())
  time.sleep(30)
 else:raise TimeoutError('finite3h guarded model launch wait')
 launch=json.loads(lp.read_text());j=next(v for v in launch['jobs']if v['mode']==2);required=len(json.loads((r/'results/kimi-chat-cases-r20.json').read_text())['cases']);snapshot=None
 for _ in range(360):
  p=r/'results'/f'kimi-spec-r3759-{j["run"]}-node3.json'
  if p.exists():
   try:
    raw=p.read_bytes();q=json.loads(raw)
   except json.JSONDecodeError:time.sleep(30);continue
   if q.get('error'):raise RuntimeError(('M2producer_failed',q['error']))
   qa=q.get('QA',[]);ar=q.get('frozen_AR_complete_ID_checks',[]);rep=q.get('baseline_repeats',[])
   if len(qa)>=required and all(v['complete_frozen_IDs_exact']for v in qa)and any(v['batch']==1 and v['exact']for v in ar)and any(v['batch']==1 and v['exact']for v in rep):snapshot=(q,hashlib.sha256(raw).hexdigest());break
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j',j['parent'],'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]==j['label']];assert not any(a[2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING','COMPLETED']for a in aa),aa;time.sleep(30)
 else:raise TimeoutError('finite3h wholemodelQA+AR admission wait')
 q,digest=snapshot;proof={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':True,'library_sha256':q['library_sha256'],'source_run':j['run'],'first_full_QA':qa,'B1_frozen_AR_checks':ar,'B1_AR_repeat_checks':rep,'source_report_snapshot_sha256':digest,'all_matrix_and_owner_closure_complete':q['completed'],'speculative_performance_qualified':False,'scope':'Admission only: actualwholemodel firstQA andfull128 B1 autoregressive repeat/frozen06d compatibility resolvefirstforwardCausalConv blocker. Otherbatches/speculation/ownerclosure/terminal unqualified; notperformanceadoption.'};p=r/'results/kimi-forward-guard-verified-r3761.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2))
 p=subprocess.run(['python3',str(r/'incoming/launch_kimi_pipeline_spec_r3761.py')],text=True,capture_output=True,timeout=240);d.update(stdout=p.stdout[-4000:],stderr=p.stderr[-4000:]);assert p.returncode==0,d;d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
