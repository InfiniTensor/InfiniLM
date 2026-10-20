import argparse,array,collections,datetime,hashlib,json,pathlib,statistics,struct,subprocess
a=argparse.Namespace(tag='910373.774');r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
state=subprocess.check_output(['sacct','-j',a.tag,'-n','-P','-o','JobID,JobName%65,State,ExitCode,Elapsed,NodeList%50'],text=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'completed':False,'phases':[],'scalar_QA':[],'batch_QA':[],'scope':'864prefill4 memorytrial, failed834pref8OOM retained, old691longdiagnostic; ownfull512/raw/IDs andallknownQA strict. 835independent full93Kimi capacity128/pages544 on833, unchanged476math/SDK621/helper21/defaultcollectives. Full512x2serial/pipeline eachM16G64/M24G96/M32G128, frozen691M16 controls, actualinputSHA, all4stage/8tailrank IDs/rawfirst4 CPUargmax;44scalar+12B16+24B24B32knownQA/EOS/rawrepeats/owner-shutdown. No absolutepeak/aggregate/HTTP or broadquality claim.'}
reports=[]
for stage in range(4):
 path=r/'results'/f'legacy-kimi-eos-r3472-{a.tag}-stage{stage}.json'
 if path.exists():reports.append(json.loads(path.read_text()))
d['reports_available']=len(reports)
vocab=163840
def raw_check(phase,step,packet,width,hashes,ids=None):
 raw=(r/'results'/f'kimi-token-only-r964-{a.tag}-{phase}-q{packet}-step{step}-bf16.bin').read_bytes();assert len(raw)==width*vocab*2
 for row in range(width):
  part=raw[row*vocab*2:(row+1)*vocab*2];assert hashlib.sha256(part).hexdigest()==hashes[row]
  if ids is not None:
   words=array.array('h');words.frombytes(part);assert all((v&0x7f80)!=0x7f80 for v in words);best=max(words);best=min(words)if best<0 else best;index=words.index(best)
   if best in[0,-32768]:index=min(words.index(v)for v in[0,-32768]if v in words)
   assert index==ids[row]
def counters(q):
 for k in['static_W2_calls','FP32_gate_prefill_calls']:
  v=q[k];assert len(v)==8 and min(v)>0 and len(set(v))==1
 policy=q['BLAS_policy'];assert policy['failures']==0 and all(v['created']==v['verified']for v in policy['owners'])and len({v['tid']for v in policy['owners']if'libinfiniops.so'in v['caller']})==8
if len(reports)==4 and all('sdk_sha256'in q for q in reports):
 build=json.loads((r/'results/kimi-token-only-r963-build.json').read_text());assert hashlib.sha256((r/'candidates/kimi-token-only-r963/libmodern_kimi_token_only_r963.so').read_bytes()).hexdigest()==build['library_sha256']
 manifest=r/'candidates/kimi-token-only-r963/InfiniLM/source-manifest-r963.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==build['source_manifest_sha256']
 for n,h in json.loads((r/'legacy-kimi-eos-r3473-launch.json').read_text())['sources_sha256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h
 assert build['PP_boundaries']==[0,24,47,71,93]
 assert all(q['PP_layer_ranges']==[[0,24],[24,47],[47,71],[71,93]]for q in reports)
 for q in reports:assert q['library_sha256']==build['library_sha256']and q['sdk_sha256']==build['sdk_sha256']=='6210897fc1f077bc5fc2e10afc9b8b7141fc4c6c608be3f4cb2d896219560104'and q['helper_sha256']==build['helper_sha256']=='26cc4b1eb0e8c044c760db7d42a7374d9ebc98ce82591d1ae1e4a1292c4b7d82'and q['cache_pages']==516 and q['max_requests']==128
 gold=json.loads((r/'results/kimi-batch-r691-910371.123-stage3.json').read_text());assert gold['completed'];goldcases={v['phase']:v for v in gold['cases']}
 eos=json.loads((r/'results/kimi-chat-cases-r20.json').read_text())['eos_token_ids'];oldscalar=json.loads((r/'results/kimi-precision-r543-910375.129-stage3.json').read_text());known={(v['epoch'],v['label'],v['repeat']):v for v in oldscalar['chat_cases']}
 scalar=[{(v['epoch'],v['label'],v['repeat']):v for v in q['chat_cases']}for q in reports]
 for key in sorted(set.intersection(*(set(v)for v in scalar))):
  rows=[v[key]for v in scalar];c=rows[3];assert all(v['generated_ids']==c['generated_ids']and v['first4_logits_sha256']==c['first4_logits_sha256']and v['known_answer_ids_exact']and v['same_variant_first4_repeat_exact']for v in rows);assert c['generated_ids']==known[key]['generated_ids']and c['first4_logits_sha256']==known[key]['first4_logits_sha256']and c['generated_ids'][-1]in eos
  for step,h in enumerate(c['first4_logits_sha256']):raw_check(key[0]+':'+key[1]+'_r'+str(key[2]),step,0,1,[h],[c['generated_ids'][step]])
  d['scalar_QA'].append({'epoch':key[0],'label':key[1],'repeat':key[2],'known_IDs_raw_repeat_EOS_verified':True})
 same={}
 for field,batches in[('batch16_chat_cases',[16]),('large_batch_chat_cases',[24,32])]:
  mappings=[{((v.get('batch',16)),v['epoch'],v['label'],v['repeat']):v for v in q.get(field,[])}for q in reports]
  for key in sorted(set.intersection(*(set(v)for v in mappings))):
   batch,epoch,label,repeat=key;assert batch in batches;rows=[v[key]for v in mappings];c=rows[3];ids=c['generated_ids_per_row'];hashes=c['per_row_first4_logits_sha256'];assert len(ids)==len(hashes)==batch
   normalize=lambda seq:[t for t in seq if t!=220]if label=='sort'else seq
   assert all(v['generated_ids_per_row']==ids and v['per_row_first4_logits_sha256']==hashes and v['same_geometry_raw_hashes_repeated_exact']and v['sort_space220_normalization_allowed']==(label=='sort')for v in rows)
   assert all(normalize(seq)==normalize(known[epoch,label,1]['generated_ids'])and seq[-1]in eos for seq in ids);value=(ids,hashes);assert (batch,label)not in same or same[batch,label]==value;same[batch,label]=value
   phase=('batch16_'if batch==16 else'large'+str(batch)+'_')+epoch+':'+label+'_r'+str(repeat)
   for step in range(len(hashes[0])):
    width=4 if step==0 else batch
    for packet in range(batch//width):raw_check(phase,step,packet,width,[v[step]for v in hashes[packet*width:(packet+1)*width]])
   d['batch_QA'].append({'batch':batch,'epoch':epoch,'label':label,'repeat':repeat,'known_IDs_EOS_raw_repeat_verified':True})
 assert len(d['scalar_QA'])==44 and len(d['batch_QA'])==12
 assert all(q.get('completed')and q.get('qualification_before_destroy_passed')and q.get('all8_owned_workers_shutdown_completed')and not q.get('error')and not q.get('shutdown_error')for q in reports)
 assert '|COMPLETED|0:0|'in state
 provenance=reports[3]['natural_input_provenance'];assert all(q['natural_input_provenance']==provenance for q in reports);assert set(provenance['eos_token_ids'])==set(eos)and len(provenance['cases'])==2
 natural=reports[3]['natural_cases'];assert len(natural)==4 and all(q['natural_cases']==natural for q in reports)
 refs={};checkpoints=[0,1,2,3,63,127,255,383,511];raw_files=0
 for c in natural:
  batch=c['batch'];phase=c['phase'];assert batch in[16,64]and c['micro']==16 and c['pipeline']==(batch==64)and c['all_requests_firstEOS_or_limit']and c['own_IDs_checkpoint_raw_repeat_exact']and c['static_cohort_with_EOS_padding']
  ids=c['generated_ids'];ts=c['step_times_per_row'];hs=c['raw_checkpoint_hashes'];inputs=[provenance['cases'][(i//16)%2]['input_ids']for i in range(batch)];assert c['input_lengths']==list(map(len,inputs))and all(480<=len(v)<=512 for v in inputs)
  assert len(ids)==len(ts)==len(hs)==batch
  # This acceptance is deliberately scoped to this all512-cap long-output run.
  assert all(len(v)==512 and len(t)==512 and len(h)==9 and not any(tok in eos for tok in v)and all(0<=tok<vocab for tok in v)and all(a<b for a,b in zip(t,t[1:]))for v,t,h in zip(ids,ts,hs))
  assert (ids,hs)==refs.setdefault(batch,(ids,hs))and c['rounds']==512
  for step in checkpoints:
   width=4 if step==0 else 16
   for packet in range(batch//width):
    lo=packet*width;raw_check(phase,step,packet,width,[v[checkpoints.index(step)]for v in hs[lo:lo+width]],[v[step]for v in ids[lo:lo+width]]);raw_files+=1
    if step==0:
     actual=sum(inputs[lo:lo+width],[]);expected=hashlib.sha256(struct.pack('<'+str(len(actual))+'q',*actual)).hexdigest();assert all(q['actual_input_packet_SHA834'][phase][str(packet)]==expected for q in reports)
  first=max(t[0]for t in ts);end=max(t[-1]for t in ts);useful=batch*512;decoded=sum(t>first for times in ts for t in times);assert useful==c['useful_output_tokens']and decoded==c['decode_output_tokens']and end==c['elapsed_seconds']and end-first==c['decode_elapsed_seconds']
  assert abs(c['aggregate_useful_decode_tok_s']-decoded/(end-first))<1e-9 and abs(c['including_prefill_useful_tok_s']-useful/end)<1e-9
  assert abs(c['decode_tok_s_per_all32cards']-decoded/(end-first)/32)<1e-10
  d['phases'].append({'phase':phase,'G':batch,'M':16,'decode_tok_s':decoded/(end-first),'including_prefill_tok_s':useful/end,'per_all32_cards':decoded/(end-first)/32,'TTFT_seconds':[t[0]for t in ts],'output_lengths':list(map(len,ids)),'EOS_terminated_requests':0})
 d.update(completed=True,natural_raw_files=raw_files,all32_owners_normal_close=True,source_sha256=json.loads((r/'legacy-kimi-eos-r3473-launch.json').read_text())['sources_sha256'],scope='Independent historical r963 actual EOS first retirement on two480..512 long prompts repeated G16/M16 andG64/M16; alloutputs reached512 ceiling. Frozen44scalar and12B16QA/own first4+late raw/CPUargmax/inputSHA/SDK/model/32normalclose/terminal passed. Historical standalone space220 sort allowance retained. Static cohort with EOS polling and feedback wall, no HTTP or mixed length continuous serving/SLO/physical maximum claim.')
out=r/'results/legacy-kimi-eos-r3512-audit.json';assert not out.exists();out.write_text(json.dumps(d,indent=2));assert d['completed'];print('LEGACY_KIMI_EOS_R3512_FULL_AUDIT_COMPLETE')
