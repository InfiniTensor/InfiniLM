import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-guard-r3757-start.json';assert not out.exists()
with(r/'logs/kimi-guard-r3757-queue-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_guard_test_r3757.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
