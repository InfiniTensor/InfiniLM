import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,subprocess,datetime,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-KDA1-replay-r3868-audit.json';assert not out.exists();d={'completed':False,'main_goal_metric':False,'wholemodel_qualified':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
 aa=[v.split('|')for v in subprocess.check_output(['sacct','-j','910373','-n','-P','-o','JobID,JobName%120,State,ExitCode'],text=True).splitlines()if v.split('|')[1]=='kimi-KDA1-replay-r3866'];assert len(aa)==1 and aa[0][2:4]==['COMPLETED','0:0'],aa
 tag=aa[0][0];qq=[json.loads((r/'results'/f'kimi-KDA1-replay-r3866-{tag}-rank{k}.json').read_text())for k in range(8)];assert all(q['completed']and q['Runtime_and_communicator_normally_closed']and q['rank']==k for k,q in enumerate(qq));proof=json.loads((r/'results/kimi-projection-r3866-build.json').read_text());assert proof['completed']and all(q['helper_sha256']==proof['library_sha256']for q in qq)
 def read(z):
  p=r/'results'/z['file'];b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==z['sha256'];x=np.frombuffer(b,np.uint16).reshape(z['shape']);assert np.isfinite((x.astype(np.uint32)<<16).view(np.float32)).all();return x
 local=[];reduced={i:[]for i in[0,1,2]};capture={i:True for i in[0,1,2]}
 for q in qq:
  assert[v['label']for v in q['cases']]==['first','second','block'];xs=[];ys={i:[]for i in[0,1,2]}
  for c in q['cases']:
   assert c['input_weight_readonly'];xs.append(read(c['local']))
   for z in c['reductions']:
    assert z['finite'];ys[z['policy']].append(read(z['output']));capture[z['policy']]&=z['actual_captured_output_equal']
  local.append({'rank':q['rank'],'different_local_elements':int(np.count_nonzero(np.concatenate(xs[:2])!=xs[2])),'shape':list(xs[2].shape)})
  for mode in[0,1,2]:reduced[mode].append({'rank':q['rank'],'different_reduced_elements':int(np.count_nonzero(np.concatenate(ys[mode][:2])!=ys[mode][2]))})
 exact_reproduction=capture[0];local_equal=all(v['different_local_elements']==0 for v in local);native_diff=any(v['different_reduced_elements']for v in reduced[0]);fp32_equal=all(v['different_reduced_elements']==0 for v in reduced[1]);f64_equal=all(v['different_reduced_elements']==0 for v in reduced[2])
 d.update(completed=True,producer=aa[0],local_projection_serial_vs_block=local,reduction_serial_vs_block=reduced,replay_equal_actual_captured_output_by_policy=capture,native_replay_reproduces_real_model_output=exact_reproduction,local_projection_equal_in_this_fixture=local_equal,FP32_eliminates_this_fixture_difference=fp32_equal,F64_eliminates_this_fixture_difference=f64_equal,causal_scope='This captured KDA1 projection boundary only. Raw equality diagnoses boundary; whole output token parity must be revalidated after any model change.',diagnosis='Native collective shape-dependent reduction reproduced with identical local projections'if exact_reproduction and local_equal and native_diff else'No collective-only causal conclusion; inspect local projection or capture mismatch',performance_gain_adopted=False)
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
