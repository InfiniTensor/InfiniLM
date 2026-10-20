import pathlib,json,subprocess,time,traceback,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-idle-serving-r3784-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_build'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(90):
  p=r/'results/glm-shared-rope-r3783-build.json'
  comp=r/'results/glm-shared-rope-component-r3783.json'
  if p.exists()and comp.exists():
   assert json.loads(p.read_text())['completed']and json.loads(comp.read_text())['completed']
   states=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]in['glm-idle-trim-r3783-build','glm-shared-rope-component-r3783']]
   assert not any(a[2]not in['RUNNING','PENDING','COMPLETING','COMPLETED']for a in states),states
   if len(states)==2 and all(a[2:4]==['COMPLETED','0:0']for a in states):break
  aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if a.split('|')[1]=='glm-idle-trim-r3783-build'];assert not any(a[2]not in['RUNNING','PENDING','COMPLETING','COMPLETED']for a in aa),aa;time.sleep(30)
 else:raise TimeoutError('finite45min build wait')
 p=subprocess.run(['python3',str(r/'incoming/launch_glm_shared_rope_r3784.py')],capture_output=True,text=True,timeout=180);d.update(stdout=p.stdout[-4000:],stderr=p.stderr[-4000:]);assert p.returncode==0,d;d.update(state='submitted',completed_dispatch=True)
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
