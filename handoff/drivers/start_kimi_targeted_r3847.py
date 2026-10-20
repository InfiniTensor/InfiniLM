import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-targeted-r3847-start.json';assert not out.exists()
with(r/'logs/kimi-targeted-r3847-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_targeted_r3847.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'pid':p.pid,'submitted':True}));print(out.read_text())
