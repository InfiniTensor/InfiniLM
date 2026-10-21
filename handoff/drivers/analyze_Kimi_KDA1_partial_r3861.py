import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,json,hashlib,datetime,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-KDA1-partial-r3861-analysis.json';assert not out.exists();d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'wholemodel_qualified':False,'scope':'Read-only CPUdiagnosis ofcompleted tracedcalls anduntracedrootpaths in a stillrunning producer. No finalSlurm/afterQA/servingperformance qualification. Rawdifferences diagnostic only.'}
try:
 launch=json.loads((r/'kimi-root-r3837-launch.json').read_text());p=r/'results'/f'kimi-spec-r3837-{launch["run"]}-node3.json';raw=p.read_bytes();q=json.loads(raw);assert len(q['root_paths'])==4 and q['root_traced_vs_untraced_IDs_exact'];assert q['root_paths'][0]['generated_ids']==q['root_plan']['expected_AR_ids'];root=r/'results'/pathlib.Path(q['trace_root']).name;data={};files=[];arrays={}
 for label,call in [('first','serial-traced/serial-first'),('second','serial-traced/serial-second'),('block','block-traced/block')]:
  data[label]=[];directory=root/call
  for rank in range(8):
   journal=directory/f'stage0-rank{rank}-layer1-kda-call0.jsonl';rows=[json.loads(v)for v in journal.read_text().splitlines()];assert len({v['point']for v in rows})==len(rows);entry={}
   for row in rows:
    pp=directory/row['file'];b=pp.read_bytes();assert len(b)==row['bytes'];dtype={'bf16':np.uint16,'f32':np.float32,'i32':np.int32}[row['dtype']];a=np.frombuffer(b,dtype).reshape(row['shape']);entry[row['point']]=(row,a);files.append({'file':pp.relative_to(r).as_posix(),'sha256':hashlib.sha256(b).hexdigest()})
   data[label].append(entry)
 summaries=[]
 for rank in range(8):
  first,second,block=[data[k][rank]for k in ['first','second','block']]
  for point in first:
   va,a=first[point];vb,b=second[point];vc,c=block[point];assert va['dtype']==vb['dtype']==vc['dtype']
   if point in['conv_before','ssm_before']:a=a[int(first['read_rows'][1][0])];c=c[int(block['read_rows'][1][0])];kind='active_initial_state';expected=a
   elif point in['conv_after','ssm_after']:b=b[int(second['write_rows'][1][0])];c=c[int(block['write_rows'][1][0])];kind='committed_state_after_two';expected=b
   else:
    axes=[i for i in range(a.ndim)if c.shape[i]==2*a.shape[i]and a.shape==b.shape and all(c.shape[j]==a.shape[j]for j in range(a.ndim)if j!=i)]
    if len(axes)!=1:continue
    expected=np.concatenate([a,b],axis=axes[0]);kind='serial_concatenated_vs_block'
   assert expected.shape==c.shape;summaries.append({'rank':rank,'point':point,'comparison':kind,'dtype':va['dtype'],'different_elements':int(np.count_nonzero(expected!=c)),'elements':c.size})
 for point in ['input','qkv_projected','conv_output','silu_qkv','f_a','gate_projected','beta_projected','kda_output','normalized_f32','output_gate_input','gate_sigmoid','gated','output','row_output_local','row_output_reduced']:
  if any(point not in data[label][rank]for label in ['first','second','block']for rank in range(8)):continue
  for label in ['first','second','block']:
   entries=[data[label][rank][point]for rank in range(8)];assert len({v[0]['dtype']for v in entries})==1;arrays[label+'_'+point]=np.stack([v[1]for v in entries]);arrays[label+'_'+point+'_dtype']=np.array(entries[0][0]['dtype'])
 fixture=r/'results'/('kimi-KDA1-projection-fixture-r3861-'+launch['run']+'.npz');assert not fixture.exists();np.savez_compressed(fixture,**arrays)
 d.update(completed=True,row_projection_local_hook_observed=all('row_output_local'in data[label][rank]for label in ['first','second','block']for rank in range(8)),prior_CPU_analysis_failure='3858 expected row_output_local hook absent in actual emitted journals; preserve failure and inspect actual Common symbol path. No GPU producer rerun.',producer_still_running_at_snapshot=not q['completed'],source_report_sha256=hashlib.sha256(raw).hexdigest(),comparisons=summaries,nonzero_comparisons=[v for v in summaries if v['different_elements']],fixture=fixture.name,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),files=files,root_paths=[{k:v[k]for k in['label','first_token_difference','generated_ids']}for v in q['root_paths']])
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_KDA1_PARTIAL_R3858',d['nonzero_comparisons'])
