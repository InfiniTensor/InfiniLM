import pathlib,subprocess,json
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-pipeline-r3761-queue-start.json';assert not out.exists()
with(r/'logs/kimi-pipeline-r3761-queue-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_pipeline_admission_r3761.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
