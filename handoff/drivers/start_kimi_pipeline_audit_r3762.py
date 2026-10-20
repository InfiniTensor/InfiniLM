import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-pipeline-r3762-audit-start.json';assert not out.exists()
with(r/'logs/kimi-pipeline-r3762-audit-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_pipeline_audit_r3762.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
