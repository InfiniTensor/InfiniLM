import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-idle-r3790-audit-start.json';assert not out.exists()
with(r/'logs/glm-idle-r3790-audit-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_glm_shared_rope_audit_r3790.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
