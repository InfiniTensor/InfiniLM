import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,datetime,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-KDA-pair-r3838-analysis.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'performance_qualified':False}
try:
 proof=json.loads((r/'results/kimi-root-r3836-postmortem.json').read_text());assert proof['completed']and proof['diagnostic_trace_analysis_completed']and proof['actual_before_after_AR_complete_IDs_exact']and proof['traced_and_untraced_block_full_IDs_exact'];launch=json.loads((r/'kimi-root-r3810-launch.json').read_text());root=r/'results'/('kimi-paired-root-r3810-'+launch['run']);rows=[];files=[]
 def read(call,rank):
  directory=root/('block-traced/block'if call=='block'else'serial-traced/'+call);paths=list(directory.glob(f'stage0-rank{rank}-layer*-kda-call0.jsonl'));assert len(paths)==1;records=[json.loads(v)for v in paths[0].read_text().splitlines()];assert len({v['point']for v in records})==len(records);files.append({'file':paths[0].relative_to(r).as_posix(),'sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest()});result={}
  for v in records:
   p=directory/v['file'];b=p.read_bytes();assert p.name==v['file']and len(b)==v['bytes'];dtype={'bf16':np.uint16,'f32':np.float32,'i32':np.int32}[v['dtype']];x=np.frombuffer(b,dtype).reshape(v['shape']);result[v['point']]=(v,x);files.append({'file':p.relative_to(r).as_posix(),'sha256':hashlib.sha256(b).hexdigest()})
  return result
 def compare(a,b,label,rank,point):
  assert a.shape==b.shape
  diff=int(np.count_nonzero(a!=b));rows.append({'rank':rank,'point':point,'comparison':label,'shape':list(a.shape),'diagnostic_different_elements':diff,'total_elements':a.size})
 for rank in range(8):
  first,second,block=[read(call,rank)for call in['serial-first','serial-second','block']];assert set(first)==set(second)==set(block)
  for point in first:
   va,a=first[point];vb,b=second[point];vc,c=block[point];assert va['dtype']==vb['dtype']==vc['dtype']
   if point in['ssm_before','conv_before']:
    readfirst=int(first['read_rows'][1].reshape(-1)[0]);readblock=int(block['read_rows'][1].reshape(-1)[0]);compare(a[readfirst],c[readblock],'active_prefix_before_block_vs_serial_first',rank,point)
   elif point in['ssm_after','conv_after']:
    writesecond=int(second['write_rows'][1].reshape(-1)[0]);writeblock=int(block['write_rows'][1].reshape(-1)[0]);compare(b[writesecond],c[writeblock],'committed_state_after_two_tokens',rank,point)
   elif a.shape==b.shape and len(a.shape)==len(c.shape):
    axes=[i for i in range(a.ndim)if c.shape[i]==2*a.shape[i]and all(a.shape[j]==c.shape[j]for j in range(a.ndim)if j!=i)]
    if len(axes)==1:compare(np.concatenate([a,b],axis=axes[0]),c,'serial_two_tokens_concatenated_vs_block',rank,point)
    elif a.shape==c.shape and point in['a_log','dt_bias']:compare(a,c,'readonly_parameter',rank,point)
 first_by_rank={}
 for v in rows:
  if v['diagnostic_different_elements']and v['comparison']=='serial_two_tokens_concatenated_vs_block':first_by_rank.setdefault(str(v['rank']),v)
 d.update(completed=True,root_audit_sha256=hashlib.sha256((r/'results/kimi-root-r3836-postmortem.json').read_bytes()).hexdigest(),comparisons=rows,first_projection_or_operator_diagnostic_difference_per_rank=first_by_rank,files=files,scope='Selected firstKDA stage0,all8ranks,identicalprefix. Compare only active initialstates and committed finalstates using actualmetadata read/write rows; never compare unused/uninitialized poolrows. Shapealigned serialtokenprojections versus block, rawdifference diagnostic only. Traced/untraced completeIDs previouslyaudited; not tokenqualification or speedgain fromrawdifference.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_KDA_PAIR_R3830_ANALYZED',first_by_rank)
