import pathlib,json,subprocess,time,datetime,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-targeted-r3847-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_actual_trace_only_build'};out.write_text(json.dumps(d,indent=2))
try:
 for _ in range(180):
  states=[v.split('|')for v in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-MLA-trace-r3841-build'];assert len(states)==1
  if states[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert states[0][2:4]==['COMPLETED','0:0'],states;assert json.loads((r/'results/kimi-MLA-trace-r3841-build.json').read_text())['completed']
   for name in ['launch_kimi_root_r3837.py','start_kimi_selective_audit_r3845.py']:
    p=subprocess.run(['python3',str(r/'incoming'/name)],capture_output=True,text=True,timeout=190);d[name]={'stdout':p.stdout[-2500:],'stderr':p.stderr[-2500:],'exit_code':p.returncode};assert p.returncode==0,d;out.write_text(json.dumps(d,indent=2))
   d.update(state='submitted_targeted_trace_and_selective_lookup',build_state=states[0]);break
  time.sleep(30)
 else:raise TimeoutError('Finite90min tracebuild wait elapsed')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
