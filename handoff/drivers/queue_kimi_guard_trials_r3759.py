import pathlib,json,time,subprocess,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-guard-r3759-trial-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_actual_Runtime_guard_component'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(180):
  p=r/'kimi-guard-r3757-queue.json'
  if p.exists():assert json.loads(p.read_text())['state']!='failed',json.loads(p.read_text())
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='kimi-guard-r3757']
  if aa and aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert aa[0][2:4]==['COMPLETED','0:0'],aa
   p=subprocess.run(['python3',str(r/'incoming/audit_kimi_guard_runtime_r3758.py')],capture_output=True,text=True,timeout=180);assert p.returncode==0,(p.stdout,p.stderr);break
  time.sleep(30)
 else:raise TimeoutError('finite90min guard component wait')
 p=subprocess.run(['python3',str(r/'incoming/launch_kimi_guard_trials_r3759.py')],capture_output=True,text=True,timeout=400);d.update(stdout=p.stdout[-4000:],stderr=p.stderr[-4000:]);assert p.returncode==0,d;d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
