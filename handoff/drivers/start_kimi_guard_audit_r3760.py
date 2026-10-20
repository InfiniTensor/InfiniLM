import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'kimi-spec-math-r3760-audit-queue-start.json';assert not out.exists()
with(r/'logs/kimi-spec-math-r3760-audit-queue-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_kimi_guard_audit_r3760.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
