import argparse,array,collections,datetime,hashlib,json,pathlib,statistics,subprocess
p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();a.variant='fused';r=pathlib.Path.home()/'hygon-inference-refactor-20261014';v=False;prefix='native-engine-allselected-r816';rawprefix='deep-allselected-r816'
state=subprocess.check_output(['sacct','-j',a.tag,'-n','-P','-o','JobID,JobName%65,State,ExitCode,Elapsed'],text=True);d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'variant':a.variant,'completed':False,'scalar_cases':[],'batch16_cases':[],'first4_checks':[],'phases':[],'scope':'Independent rawSHA/argmax/fullID/repeat/knownanswer/owner audit for635nativeBF16 geometry or634intentionalFP32attentionreduce. Beyondfirst4 B16knownanswerall-rowID checks are enforced bydriver, onlyrow0fullIDs arepersisted there. Crossbatch logits andoldprecision differences staydiagnostic. No broadquality/modelmaximum.'}
d['scope']='817canonicalallselected attentionorder variant. OldscalarandactualB16knownanswerIDs/EOS strict; ownB8/B16full512repeats/raw/all8; fast/fallbackcounters/SDK/source/owner/shutdown/terminal required. Parent687independentSDK/DSO/source/rawSHA/CPUargmax/knownanswers/EOS/ownsamegeometryrepeats/all8/policy/actualfusedandfallbackcounters/shutdownandSlurm audit. IntentionalKVsplit681F32 reduction order; oldscalar andsyntheticlogitsdiagnostic. Fourfull512 B8/B16 phases, sixB1+sixB16 beforeafter. Beyondfirst4 B16QA allrowIDs enforcedindriver butonlyrow0fullIDspersisted. Notfullquality/HTTP/hardwaremaximum.'
path=r/'results'/f'{prefix}-{a.tag}.json'
if not path.exists():print(json.dumps(d));raise SystemExit
q=json.loads(path.read_text());d['failure']=q.get('batch16_failure');assert q['sdk_sha256']=='8865b8d8e67832884911199d1a667856aadb2b438f5287d17fbaff1f8a3b85db'
build=json.loads((r/'results/deepseek-allselected-r815-build.json').read_text());assert list(q['library_sha256'].values())==[build['library_sha256']]
for name,digest in build['inputs_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest

gold=json.loads((r/'results/modern-deepseek-chat-routed-r20-910373.162-rank0.json').read_text());goldcases={c['phase']:c for c in gold['cases']};eos=json.loads((r.parent/'shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8/config.json').read_text())['eos_token_id'];V=129280
def policy(value):
 assert value['failures']==0 and all(x['created']==x['verified']for x in value['owners'])and len({x['tid']for x in value['owners']if'libinfiniops.so'in x['caller']})==8
def readraw(name):
 path=r/name.removeprefix('/refactor/');assert path.resolve().is_relative_to(r.resolve());return path.read_bytes()
def argmaxes(raw,batch):
 assert len(raw)==batch*V*2;result=[]
 for row in range(batch):
  words=array.array('h');words.frombytes(raw[row*V*2:(row+1)*V*2]);best=max(words)
  if best<0:best=min(words)
  assert (best&0x7f80)!=0x7f80
  # BF16 logits are finite by the runtime driver's full-vocabulary check.
  result.append(words.index(best))
 return result
scalar={}
for epoch,cases in q.get('chat_cases_by_epoch',{}).items():
 for c in cases:
  old=goldcases[c['phase']];assert c['generated_ids']==old['generated_ids']and c['stopped_at_eos']and c['generated_ids'][-1]==eos and c['all8_worker_full_logits_exact_first4'];policy(c['blas_policy'])
  for step,digest in enumerate(c['first4_logits_sha256']):
   raw=(r/'results'/f'deep-finite-r578-{a.tag}-{epoch}-{c["phase"]}-step{step}-B1-bf16.bin').read_bytes();assert hashlib.sha256(raw).hexdigest()==digest and argmaxes(raw,1)==[c['generated_ids'][step]]
  value=(c['generated_ids'],c['first4_logits_sha256']);label=c['phase'].rsplit('_repeat',1)[0];assert label not in scalar or scalar[label]==value;scalar[label]=value
  d['scalar_cases'].append({'epoch':epoch,'phase':c['phase'],'old_known_answer_IDs_EOS_exact':True,'same_variant_repeated_exact':True,'old_logits_exact':c['first4_logits_sha256']==old['first4_logits_sha256']})
checks={(x['epoch'],x['phase'],x['step']):x for x in q.get('batch16_first4_checks',[])}
for key,c in checks.items():
 raw=readraw(c['raw']);assert hashlib.sha256(raw).hexdigest()==c['sha256']and argmaxes(raw,16)==[goldcases[c['phase']]['generated_ids'][c['step']]]*16
 hashes=[hashlib.sha256(raw[j*V*2:(j+1)*V*2]).hexdigest()for j in range(16)];assert (len(set(hashes))==1)==c['all16_duplicate_logits_exact']
 d['first4_checks'].append({'epoch':key[0],'phase':key[1],'step':key[2],'all16_known_answer_argmax_exact':True,'raw_sha_verified':True,'duplicate_logits_exact':len(set(hashes))==1,'cross_batch_relative_RMS':c['cross_batch_relative_RMS'],'row_groups':[[j for j,h in enumerate(hashes)if h==digest]for digest in dict.fromkeys(hashes)]})
batchrefs={}
for epoch,cases in q.get('batch16_chat_by_epoch',{}).items():
 for c in cases:
  assert c['generated_ids']==goldcases[c['phase']]['generated_ids']and c['generated_ids'][-1]==eos and c['stopped_at_eos']and c['old_known_answer_IDs_exact']and c['same_geometry_IDs_logits_repeated_exact'];policy(c['blas_policy'])
  for step,digest in enumerate(c['first4_logits_sha256']):assert checks[epoch,c['phase'],step]['sha256']==digest
  label=c['phase'].rsplit('_repeat',1)[0];value=(c['generated_ids'],c['first4_logits_sha256']);assert label not in batchrefs or batchrefs[label]==value;batchrefs[label]=value
  d['batch16_cases'].append({'epoch':epoch,'phase':c['phase'],'old_answer_IDs_EOS_and_samegeometry_repeats_exact':True})
old16=next(c for c in json.loads((r/'results/native-engine-batch-r522-910373.365.json').read_text())['cases']if c['phase']=='engine_b16_repeat1');ownrefs={}
for c in q.get('cases',[]):
 if c.get('output_tokens')!=512:continue
 batch=c['batch'];assert batch in [8,16]and len(c['generated_ids'])==batch and all(len(ids)==512 for ids in c['generated_ids'])and len(c['steps'])==512 and c['all8_worker_full_logits_exact_first4'];policy(c['blas_policy'])
 for step,digest in enumerate(c['first4_logits_sha256']):
  raw=(r/'results'/f'{rawprefix}-{a.tag}-{c["phase"]}-step{step}-logits-bf16.bin').read_bytes();assert hashlib.sha256(raw).hexdigest()==digest and argmaxes(raw,batch)==[ids[step]for ids in c['generated_ids']]
 value=(c['generated_ids'],c['first4_logits_sha256']);assert batch not in ownrefs or ownrefs[batch]==value;ownrefs[batch]=value
 tps=batch/statistics.mean(x['wall_seconds']for x in c['steps'][9:]);inc=batch*512/c['elapsed_seconds'];assert abs(tps-c['diagnostic_steady_total_tps'])<1e-10 and abs(inc-c['including_prefill_total_tps'])<1e-10
 d['phases'].append({'phase':c['phase'],'batch':batch,'steady_total_tps':tps,'including_prefill_total_tps':inc,'same_geometry_repeated_exact':True,'old_B8_ID_compatibility_flag':c['baseline_ids_exact']})
need=4;scalar_counts=collections.Counter(c['epoch']for c in d['scalar_cases']);batch_counts=collections.Counter(c['epoch']for c in d['batch16_cases'])
if q.get('completed'):
 assert scalar_counts=={'before':6,'after':6}and batch_counts=={'before':6,'after':6}and len(d['phases'])==need and q['own_long_repeat_exact']and q['engine_worker_shutdown_completed'];policy(q['blas_policy_before_destroy'])
 for key in ['actual_fused_latent_calls','actual_fallback_latent_calls']:
  counts=q[key];assert len(counts)==8 and min(counts)>0 and len(set(counts))==1
 d['actual_fused_latent_calls']=q['actual_fused_latent_calls'];d['actual_fallback_latent_calls']=q['actual_fallback_latent_calls']
 d['completed']='|COMPLETED|0:0|'in state;d['single_matrix_answers_repeats_raw_owner_shutdown_passed']=d['completed']
if q.get('completed'):
 for key in['allselected_fast_calls','allselected_full_calls']:
  v=q[key];assert len(v)==8 and min(v)>0 and len(set(v))==1;d[key]=v
(r/'results'/f'deep-allselected-r817-{a.tag}-summary.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
