import pathlib,json,subprocess,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-KDA1-replay-r3883-launch.json';assert not out.exists();node='COMPUTE_NODE_e121dfd5';parent='910373';label='kimi-KDA1-replay-r3883'
health=[v.split('|')for v in subprocess.check_output(['sacct','-j',parent,'-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-post-shape-health-r3882'];assert len(health)==1 and health[0][2:4]==['COMPLETED','0:0'],health
hp=r/'results'/('kimi-post-shape-health-r3882-'+health[0][0]+'.json');hq=json.loads(hp.read_text());assert hq['completed']and len(hq['cards'])==8 and all(v['small_copy_sync_passed']for v in hq['cards'])
aa=[v.split('|')for v in subprocess.check_output(['sacct','-j',parent,'-n','-P','-o','JobID,State,NodeList%180'],text=True).splitlines()]
for a in aa:
 if '.'in a[0]and not a[0].endswith(('.batch','.extern'))and a[1]in['RUNNING','PENDING','CONFIGURING','COMPLETING']:assert node not in subprocess.check_output(['scontrol','show','hostnames',a[2]],text=True).splitlines(),a
args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w',node,'--gres=dcu:8','--mem=96000M','-t','00:25:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=kimi_projection_replay_r3883_inside.sh','USE_WEIGHT_CACHE_R3604=0','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs'/(label+'-controller.log')).open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'submitted':True,'pid':p.pid,'args':args,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print(out.read_text())
