import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'glm-PP3-r3873-audit-start.json';assert not out.exists()
with(r/'logs/glm-PP3-r3873-audit-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_GLM_PP3_audit_r3873.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'submitted':True,'pid':p.pid}));print(out.read_text())
