import datetime,hashlib,json,pathlib,struct,subprocess,traceback
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/kimi-projection-r3411-audit.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'scope':'Independent CPU fullpartial matrices andallrankreduce raws, realcheckpoint TPweightSHA/read-onlymemmap andFP64projection samples. IntegratedCommon3393 nativeBF16 publicAPI withrequestframe-controlledFP64sum; B1packedMguard,busy/nestedguards and actualperownerpromotioncounters verified. Walltimings include D2H/hashobserver andinterop, notisolatedCCLlatency ormodelTPS. No fullmodelB8quality/fixadoption.'}
def read(v):
 p=r/'results'/v['file'];assert p.parent==r/'results';b=p.read_bytes();assert len(b)==v['bytes']and hashlib.sha256(b).hexdigest()==v['sha256'];u=np.frombuffer(b,np.uint16).reshape(v['shape']);return b,(u.astype(np.uint32)<<16).view(np.float32)
def bf16_rne(a):
 b=a.astype(np.float32).view(np.uint32);return((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
try:
 aa=[a.split('|')for a in subprocess.check_output(['sacct','-j','910406','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True).splitlines()];found=[a for a in aa if a[1]=='kimi-wide-projection-r3410'];assert len(found)==1 and found[0][2:4]==['COMPLETED','0:0'];step=found[0][0];d['producer']=found[0]
 build=json.loads((r/'results/kimi-projection-r3409-build.json').read_text());assert build['completed']and sha(r/'candidates/kimi-wide-replay-3e0d89fe-r3409/libkimi_projection_r3409.so')==build['library_sha256']
 for n,h in build['source_sha256'].items():assert sha(r/'incoming'/n)==h
 for n,h in build['mapped_libraries'].items():assert sha(pathlib.Path(n.replace('/refactor',str(r),1)))==h
 model=pathlib.Path.home()/'shared/Kimi-K3';key='language_model.model.layers.0.self_attn.o_proj.weight';mapping=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];file=model/mapping[key]
 with file.open('rb')as f:hlen=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(hlen))
 meta=header[key];assert meta['dtype']=='BF16'and meta['shape']==[7168,12288];whole=np.memmap(file,dtype=np.uint16,mode='r',offset=8+hlen+meta['data_offsets'][0],shape=(7168,12288))
 groups={};reports=[];affinity=[];rawcount=0
 for rank in range(8):
  p=r/'results'/f'kimi-projection-r3409-{step}-rank{rank}.json';q=json.loads(p.read_text());assert q['completed']and q['communicator_normally_closed']and q['numeric_collection_before_close']and len(q['cases'])==6 and not q.get('error')
  assert q['integrated_Common_request_policy_and_actual_counters_passed'];assert q['helper_sha256']==build['library_sha256']and q['source_sha256']==build['source_sha256']['probe_kimi_projection_replay_r3409.py']
  assert len(q['CPU_affinity'])==4 and not set(q['CPU_affinity'])&set(affinity);affinity+=q['CPU_affinity'];policy=q['BLAS_policy'];assert policy['failures']==0 and policy['owners']and all(v['created']==v['verified']for v in policy['owners'])
  weight=np.array(whole[:,rank*1536:(rank+1)*1536],copy=True);assert hashlib.sha256(weight.tobytes()).hexdigest()==q['weight']['sha256']
  for c in q['cases']:
   assert c['local_repeat8_exact']and c['local_duplicate_rows_raw_exact']and c['readonly_and_output_guards']
   localraw,local=read(c['local_raw']);xraw,x=read(c['input_raw']);wraw,w=read(c['weight_sample_raw']);_,sample=read(c['sample_output_raw']);rawcount+=4;assert wraw==weight[[0,3584,7167]].tobytes()
   directory=pathlib.Path(c['trace_directory'].replace('/refactor',str(r),1));source=(directory/c['input_trace']['file']).read_bytes()
   assert xraw==(source if c['M']!=16 else source[:1536*2]*16)
   m=c['M'];ts=sorted({0,m//2,m-1});expected=x[ts].astype(np.float64)@w.astype(np.float64).T;err=float(np.linalg.norm(sample-expected)/max(1e-12,np.linalg.norm(expected)));assert abs(err-c['local_CPU_FP64_sample_NRMSE'])<1e-12 and err<.005
   assert np.array_equal(sample,local[np.ix_(ts,[0,3584,7167])])and np.isfinite(local).all()and all(np.array_equal(row,local[0])for row in local)
   outputs={}
   for e in c['epochs']:
    b,a=read(e['raw']);rawcount+=1;assert np.isfinite(a).all()and e['own_repeat8_exact'];dupe=all(np.array_equal(row,a[0])for row in a);assert dupe==e['duplicate_rows_fullraw_exact']
    if c['M']!=16:
     native=next(json.loads(s)for s in next(directory.glob(f'stage0-rank{rank}-layer0-kda-call*.jsonl')).read_text().splitlines()if json.loads(s)['point']=='output');assert(b==(directory/native['file']).read_bytes())==e['model_trace_output_exact']
    old=outputs.setdefault(e['mode'],b);assert old==b
   groups.setdefault((m,c['packed']),[]).append({'rank':rank,'local':local,'outputs':outputs,'local_FP64_error':err})
  reports.append({'rank':rank,'report_sha256':sha(p),'communicator_normally_closed':True})
 summary=[];all_f32_oracle=True
 for(m,packed),rows in sorted(groups.items()):
  assert len(rows)==8;gold=np.sum([v['local'].astype(np.float64)for v in rows],axis=0);oracle=bf16_rne(gold).tobytes();modes={}
  for mode in['BF16_sum','F64_sum_BF16_final']:
   assert len({v['outputs'][mode]for v in rows})==1;b=rows[0]['outputs'][mode];u=np.frombuffer(b,np.uint16).reshape(m,7168);a=(u.astype(np.uint32)<<16).view(np.float32);dupe=all(np.array_equal(v,u[0])for v in u);changed=int(np.count_nonzero(np.frombuffer(b,np.uint16)!=np.frombuffer(oracle,np.uint16)));modes[mode]={'duplicate_rows_fullraw_exact':dupe,'CPU_FP64_sum_RNE_BF16_exact':b==oracle,'changed_words_vs_oracle':changed,'relative_L2_to_FP64_sum':float(np.linalg.norm(a.astype(np.float64)-gold)/max(1e-12,np.linalg.norm(gold)))}
   if mode=='F64_sum_BF16_final':all_f32_oracle&=dupe and b==oracle
  assert modes['BF16_sum']['duplicate_rows_fullraw_exact']==(m==4)
  summary.append({'M':m,'packed':packed,'ranks':8,'all_local_GEMM_rows_and_ownrepeats_exact':True,'max_local_CPU_FP64_sample_NRMSE':max(v['local_FP64_error']for v in rows),'modes':modes})
 d.update(completed=True,normal_communicators_closed=8,cases=48,raw_files=rawcount,source_checkpoint=str(file),weight_key=key,source_checkpoint_meta=meta,reports=reports,summary=summary,F64_sum_all_cases_match_FP64_RNE_BF16_and_duplicate=bool(all_f32_oracle),root_boundary='Real local GEMM partials replayed through new Common Mode2; F64 reduction is qualified only for these projection fixtures. Wholemodel retrieval-step2 failure remains a separate gate.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,indent=2));raise
out.write_text(json.dumps(d,indent=2));print('KIMI_PROJECTION_R3234_AUDIT_COMPLETE',d['F64_sum_all_cases_match_FP64_RNE_BF16_and_duplicate'],flush=True)
