import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-post-shape-health-r3882-launch.json';assert not out.exists()
aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()];assert any(v[1:4]==['kimi-KDA1-replay-r3875','FAILED','1:0']for v in aa)
args=['srun','--jobid=910373','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','COMPUTE_NODE_e121dfd5','--gres=dcu:8','--mem=16000M','-t','00:08:00','--job-name=kimi-post-shape-health-r3882','--output='+str(r/'logs/kimi-post-shape-health-r3882-%J.log'),'env','WAVE_INSIDE_R2323=kimi_post_shape_health_r3882_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
with(r/'logs/kimi-post-shape-health-r3882-controller.log').open('w')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'submitted':True,'pid':p.pid,'args':args},indent=2));print(out.read_text())
