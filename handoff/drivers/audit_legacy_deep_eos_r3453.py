import datetime,hashlib,json,pathlib,subprocess,traceback,statistics
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/legacy-deep-eos-r3453-audit.json';assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'scope':'Historical r815 actual firstEOS retirement, B8/B16 two frozen templated long prompts repeated, native+harness not HTTP. Independent CPU full IDs/EOS/active schedule/raw checkpoints/argmax/own geometry repeat/knownanswers/binary+SDK+sources/BLAS owners/8normalclose/terminal. Historical duplicate-row raw differences remain diagnostic, not new strict numerical qualification. Canonical decode uses common window after latest request first token, excludes EOS.'}
def raw(p,shape=None):
 p=r/p.removeprefix('/refactor/');assert p.resolve().is_relative_to(r.resolve());b=p.read_bytes();a=np.frombuffer(b,np.uint16);v=(a.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(v).all()
 if shape is not None:v=v.reshape(shape)
 return b,v
def owners(q):
 assert q['failures']==0 and all(v['created']==v['verified'] for v in q['owners'])
 assert len({v['tid'] for v in q['owners'] if 'libinfiniops.so' in v['caller']})==8
try:
 rows=[a.split('|') for a in subprocess.check_output(['sacct','-j','910407','-n','-P','-o','JobID,JobName%110,State,ExitCode,Elapsed'],text=True).splitlines()]
 found=[a for a in rows if a[1]=='legacy-deep-realEOS-r3422'];assert len(found)==1 and found[0][2:4]==['COMPLETED','0:0'];step=found[0][0];d['producer']=found[0]
 p=r/'results'/f'legacy-deep-eos-r3421-{step}.json';q=json.loads(p.read_text());assert q['completed'] and q['engine_worker_shutdown_completed'] and q['qualification_before_destroy_passed'] and not q.get('error');d['report_sha256']=sha(p)
 build=json.loads((r/'results/deepseek-allselected-r815-build.json').read_text());assert list(q['library_sha256'].values())==[build['library_sha256']];assert sha(r/'candidates/native-engine-allselected-r815/libmodern_native_engine_allselected_r815.so')==build['library_sha256']
 assert q['sdk_sha256']==sha(r/'install/hygon-v4-indexer-ops-r1/lib/libinfiniops.so')=='8865b8d8e67832884911199d1a667856aadb2b438f5287d17fbaff1f8a3b85db'
 for name,h in build['inputs_sha256'].items():assert sha(r/name)==h
 expected=json.loads((r/'incoming/legacy-deep-eos-r3453-source-lock.json').read_text())
 for name,h in expected.items():assert sha(r/'incoming'/name)==h
 d['source_sha256']=expected
 cfg=json.loads((r.parent/'shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json').read_text());V=cfg['vocab_size'];eos=set(q['natural_input_provenance']['eos_token_ids']);assert cfg['eos_token_id'] in eos
 gold={c['phase']:c for c in json.loads((r/'results/modern-deepseek-chat-routed-r20-910373.162-rank0.json').read_text())['cases']};refs={};checked=0
 assert set(q['chat_cases_by_epoch'])==set(q['batch16_chat_by_epoch'])=={'before','after'}
 checks={(c['epoch'],c['phase'],c['step']):c for c in q['batch16_first4_checks']};dups=0;batchrefs={}
 for epoch in ['before','after']:
  assert len(q['chat_cases_by_epoch'][epoch])==len(q['batch16_chat_by_epoch'][epoch])==6
  for c in q['chat_cases_by_epoch'][epoch]:
   ids=c['generated_ids'];assert ids==gold[c['phase']]['generated_ids'] and ids[-1]==cfg['eos_token_id'] and c['stopped_at_eos'] and c['all8_worker_full_logits_exact_first4'];owners(c['blas_policy'])
   for st,h in enumerate(c['first4_logits_sha256']):
    b,a=raw('results/'+f'deep-finite-r578-{step}-{epoch}-{c["phase"]}-step{st}-B1-bf16.bin',(1,V));assert hashlib.sha256(b).hexdigest()==h and int(a.argmax())==ids[st];checked+=1
   value=(ids,c['first4_logits_sha256']);key=c['phase'].rsplit('_repeat',1)[0];assert value==refs.setdefault(key,value)
  for c in q['batch16_chat_by_epoch'][epoch]:
   ids=c['generated_ids'];assert ids==gold[c['phase']]['generated_ids'] and ids[-1]==cfg['eos_token_id'] and c['stopped_at_eos'] and c['all16_duplicate_generated_IDs_exact'];owners(c['blas_policy'])
   for st,h in enumerate(c['first4_logits_sha256']):
    v=checks[epoch,c['phase'],st];b,a=raw(v['raw'],(16,V));assert hashlib.sha256(b).hexdigest()==h==v['sha256'] and (a.argmax(-1)==ids[st]).all();dups+=not all(np.array_equal(a[0],v) for v in a);checked+=1
   value=(ids,c['first4_logits_sha256']);key=c['phase'].rsplit('_repeat',1)[0];assert value==batchrefs.setdefault(key,value)
 assert len(q['invalid_checks'])==12 and all(c['rejected_before_output_and_cache_mutation'] for c in q['invalid_checks']);owners(q['blas_policy_before_destroy'])
 natural=q['natural_input_provenance'];assert len(natural['cases'])==2 and len(q['natural_cases'])==4;refs={};metrics=[];raw_by_phase={};examples=[]
 for c in q['natural_cases']:
  batch=c['batch'];ids=c['generated_ids'];ts=c['step_times_per_row'];assert batch in [8,16] and len(ids)==len(ts)==batch and c['input_lengths']==[len(natural['cases'][i%2]['input_ids']) for i in range(batch)]
  for v,t in zip(ids,ts):
   assert 1<=len(v)<=512 and len(v)==len(t) and not any(x in eos for x in v[:-1]) and (v[-1] in eos or len(v)==512)
   assert all(0<=x<V for x in v) and all(0<x<=c['elapsed_seconds'] for x in t) and all(a<b for a,b in zip(t,t[1:]))
  hs=c['raw_hashes_each_step'];assert set(hs)=={str(i) for i in range(max(map(len,ids)))}
  for st in range(max(map(len,ids))):assert hs[str(st)]['rows']==[i for i,v in enumerate(ids) if len(v)>st]
  value=(ids,hs);assert value==refs.setdefault(batch,value);start=max(t[0] for t in ts);end=max(t[-1] for t in ts);useful=sum(tok not in eos for v in ids for tok in v);decode=sum(t>start and tok not in eos for v,tt in zip(ids,ts) for tok,t in zip(v,tt));assert end==c['elapsed_seconds'] and useful==c['useful_output_tokens']
  metrics.append({'phase':c['phase'],'batch':batch,'useful_output_tokens':useful,'common_decode_output_tokens':decode,'common_decode_seconds':end-start,'canonical_decode_tok_s':decode/(end-start),'including_prefill_tok_s':useful/end,'decode_tok_s_per_all8cards':decode/(end-start)/8,'TTFT_seconds_per_request':[t[0] for t in ts],'output_lengths':list(map(len,ids)),'EOS_terminated_requests':sum(v[-1] in eos for v in ids),'legacy_report_decode_diagnostic_tok_s':c['aggregate_useful_decode_tok_s']})
  raw_by_phase[c['phase']]=c;examples.append({'phase':c['phase'],'text':c['example_text']})
 for c in q['raw_checkpoints']:
  phase,st=c['file'].split(step+'-',1)[1].rsplit('-step',1);st=int(st.removesuffix('.bin'));v=raw_by_phase[phase];b,a=raw('results/'+c['file'],c['shape']);assert len(b)==c['bytes'] and hashlib.sha256(b).hexdigest()==c['sha256']==v['raw_hashes_each_step'][str(st)]['sha256'];assert c['rows']==v['raw_hashes_each_step'][str(st)]['rows'];assert a.shape==(len(c['rows']),V) and a.argmax(-1).tolist()==c['output_ids']==[v['generated_ids'][i][st] for i in c['rows']];checked+=1
 assert len(q['raw_checkpoints'])==sum(sum(st<len(v['generated_ids'][0]) or any(st<len(w) for w in v['generated_ids']) for st in [0,1,2,3,63,127,255,383,511]) for v in q['natural_cases'])
 d.update(completed=True,normal_owned_workers_closed=8,scalar_QA=12,B16_QA=12,CPU_raw_files=checked,B16_QA_checkpoints_with_duplicate_row_logit_difference=dups,performance=metrics,examples=examples,canonical_metric_correction='Original driver uses earliest first-token boundary; CPU audit recomputes latest first token and counts actual useful tokens after boundary. Original report preserved unchanged.',limitations='Two frozen long prompts repeated, historical owngeometry raw gate permits duplicate-row differences. Not broad semantic quality/continuous batching/HTTP/strict current B2 numerical qualification.')
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc());out.write_text(json.dumps(d,ensure_ascii=False,indent=2));raise
out.write_text(json.dumps(d,ensure_ascii=False,indent=2));print('LEGACY_DEEP_EOS_R3453_AUDIT_COMPLETE',flush=True)
