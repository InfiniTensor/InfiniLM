import pathlib,json,subprocess,time,traceback,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-idle-serving-r3772-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_build'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(90):
  p=r/'results/glm-idle-trim-r3771-build.json'
  if p.exists():assert json.loads(p.read_text())['completed'];break
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='glm-idle-trim-r3771-build'];assert not any(a[2]not in['RUNNING','PENDING','COMPLETING','COMPLETED']for a in aa),aa;time.sleep(30)
 else:raise TimeoutError('finite45min build wait')
 p=subprocess.run(['python3',str(r/'incoming/launch_glm_idle_serving_r3772.py')],capture_output=True,text=True,timeout=180);d.update(stdout=p.stdout[-4000:],stderr=p.stderr[-4000:]);assert p.returncode==0,d;d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
