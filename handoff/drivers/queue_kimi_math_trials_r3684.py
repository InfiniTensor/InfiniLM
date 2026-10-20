import pathlib,json,time,subprocess,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-spec-r3684-trial-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_build_and_frozen_AR','excluded_main_metric':True};out.write_text(json.dumps(d,indent=2))
try:
 old=json.loads((r/'kimi-spec-r3634-launch.json').read_text())
 for _ in range(180):
  p=r/'results/kimi-spec-math-r3683-build.json';q=r/'results'/f'kimi-spec-r3634-{old["run"]}-node3.json';ready=False
  if p.exists()and json.loads(p.read_text())['completed']and q.exists():
   report=json.loads(q.read_text());ready=True
   for B in[1,2,4,8,16]:
    cc=[v for v in report.get('spec_cases',[])if v['label']in[f'B{B}-AR-A{i}'for i in[1,2,3]]];ready &=len(cc)==3 and all(v['generated_ids']==cc[0]['generated_ids']for v in cc)
  if ready:break
  aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910377','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-spec-math-r3683-build'];assert not any(v[2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING','COMPLETED']for v in aa),aa
  if q.exists()and json.loads(q.read_text()).get('error'):raise RuntimeError('oldARproducer_failed; do not freeze incomplete reference')
  time.sleep(60)
 else:raise TimeoutError('finite3h build+AR wait')
 p=subprocess.run(['python3',str(r/'incoming/launch_kimi_math_trials_r3679.py')],text=True,capture_output=True,timeout=400);d.update(stdout=p.stdout[-5000:],stderr=p.stderr[-5000:]);assert p.returncode==0,d;d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
