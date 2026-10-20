import pathlib,json,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'followup-audits-r3756-start.json';assert not out.exists()
with(r/'logs/followup-audits-r3756-controller.log').open('w')as f:p=subprocess.Popen(['python3',str(r/'incoming/queue_followup_audits_r3756.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out.write_text(json.dumps({'started':True,'pid':p.pid},indent=2));print(json.dumps({'started':True,'pid':p.pid}))
