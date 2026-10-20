import pathlib,json,time,subprocess,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-trace-r3731-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_trace_build'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(30):
  p=r/'results/throw-trace-r3730-build.json'
  if p.exists()and json.loads(p.read_text())['completed']:break
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='throw-trace-r3730-build'];assert not any(a[2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING','COMPLETED']for a in aa),aa;time.sleep(30)
 else:raise TimeoutError('finite15min trace build')
 p=subprocess.run(['python3',str(r/'incoming/launch_kimi_trace_trial_r3731.py')],text=True,capture_output=True,timeout=240);d.update(stdout=p.stdout[-4000:],stderr=p.stderr[-4000:]);assert p.returncode==0,d;d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
