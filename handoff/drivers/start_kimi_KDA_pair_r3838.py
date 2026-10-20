import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-KDA-pair-r3838-start.json';assert not out.exists()
with(r/'logs/kimi-KDA-pair-r3838-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_KDA_pair_r3838.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'pid':p.pid,'submitted':True}));print(out.read_text())
