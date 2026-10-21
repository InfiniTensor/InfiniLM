import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-PP3-serving-r3871-start.json';assert not out.exists()
with(r/'logs/glm-PP3-serving-r3871-queue-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_glm_PP3_serving_r3871.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'submitted':True,'pid':p.pid}));print(out.read_text())
