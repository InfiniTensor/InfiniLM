import pathlib,json,subprocess,time,datetime,traceback,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-PP3-serving-r3871-queue.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'waiting_actual_rows64_constructor'};out.write_text(json.dumps(d))
try:
 for _ in range(30):
  aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910381','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='glm-PP3-capacity-r3870'];assert len(aa)==1
  if aa[0][2]not in['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   assert aa[0][2:4]==['COMPLETED','0:0'],aa
   rr=[json.loads(p.read_text())for p in(r/'results').glob(f'glm-PP3-constructor-r3857-{aa[0][0]}-stage*-rank*.json')];assert len(rr)==24 and all(q['completed']and q['all24_memory_admitted']and q['all_contexts_closed_without_loading']and q['owner_affinity']['verified']and q['rows']==64 for q in rr)
   base=json.loads((r/'glm-PP3-capacity-r3857-launch.json').read_text());args=[v.replace('glm-PP3-capacity-r3857','glm-PP3-serving-r3871').replace('GLM_CACHE_ROWS_R3627=16','GLM_CACHE_ROWS_R3627=64').replace('00:15:00','08:00:00').replace('glm_PP3_constructor_r3857_inside.sh','glm_large_r3871_inside.sh')for v in base['args']]
   with(r/'logs/glm-PP3-serving-r3871-controller.log').open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   launch={'pid':p.pid,'args':args,'parent':'910381','label':'glm-PP3-serving-r3871','rows':64,'constructor_min_free_GiB':min(q['free_total_after_create'][0]for q in rr)/1024**3,'source_sha256':{n:hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()for n in['modern_glm_large_r3871.py','glm_large_probe_r3871.py','glm_large_r3871_inside.sh']}}
   d.update(state='submitted',constructor_terminal=aa[0],launch=launch);(r/'glm-PP3-serving-r3871-launch.json').write_text(json.dumps(launch,indent=2));break
  time.sleep(30)
 else:raise TimeoutError('Finite constructor wait expired')
except BaseException as e:d.update(state='failed',error=repr(e),traceback=traceback.format_exc())
out.write_text(json.dumps(d,indent=2))
