import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-shared-rope-r3789-start.json';assert not out.exists()
with(r/'logs/glm-shared-rope-r3789-queue-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_glm_shared_rope_r3789.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
