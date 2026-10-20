import argparse,array,collections,datetime,hashlib,json,pathlib,statistics,struct,subprocess
p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
state=subprocess.check_output(['sacct','-j',a.tag,'-n','-P','-o','JobID,JobName%65,State,ExitCode,Elapsed,NodeList%50'],text=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'completed':False,'phases':[],'scalar_QA':[],'batch_QA':[],'scope':'864prefill4 memorytrial, failed834pref8OOM retained, old691longdiagnostic; ownfull512/raw/IDs andallknownQA strict. 835independent full93Kimi capacity128/pages544 on833, unchanged476math/SDK621/helper21/defaultcollectives. Full512x2serial/pipeline eachM16G64/M24G96/M32G128, frozen691M16 controls, actualinputSHA, all4stage/8tailrank IDs/rawfirst4 CPUargmax;44scalar+12B16+24B24B32knownQA/EOS/rawrepeats/owner-shutdown. No absolutepeak/aggregate/HTTP or broadquality claim.'}
reports=[]
for stage in range(4):
 path=r/'results'/f'kimi-token-only-r964-{a.tag}-stage{stage}.json'
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
 for n,h in json.loads((r/'kimi-token-only-r964-r1-launch.json').read_text())['sources_sha256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h
 assert build['PP_boundaries']==[0,24,47,71,93]
 assert all(q['PP_layer_ranges']==[[0,24],[24,47],[47,71],[71,93]]for q in reports)
 for q in reports:assert q['library_sha256']==build['library_sha256']and q['sdk_sha256']==build['sdk_sha256']=='6210897fc1f077bc5fc2e10afc9b8b7141fc4c6c608be3f4cb2d896219560104'and q['helper_sha256']==build['helper_sha256']=='26cc4b1eb0e8c044c760db7d42a7374d9ebc98ce82591d1ae1e4a1292c4b7d82'and q['cache_pages']==516 and q['max_requests']==128
 gold=json.loads((r/'results/kimi-batch-r691-910371.123-stage3.json').read_text());assert gold['completed'];goldcases={v['phase']:v for v in gold['cases']}
 original=json.loads((r/'results/modern-kimi-capacity-real93-910376.33-rendezvous.json').read_text())['cases']['b4_in512_out512'];inputs=original.get('input_ids_per_request',[original['input_ids']]*4);assert len(inputs)==4 and all(len(v)==512 for v in inputs)
 for i in range(min(len(q['cases'])for q in reports)):
  rows=[q['cases'][i]for q in reports];c=rows[3];batch,micro,count=c['global_batch'],c['micro_batch'],c['output_tokens'];assert micro in[16,24,32]and batch==(micro*4 if c['pipeline']else micro)and count==512 and c['prefill_micro_batch']==4
  refs=reports[3]['same_variant_references'][str(micro)];ids=[refs['generated_ids'][j%micro]for j in range(batch)];hashes=[refs['hashes'][j%micro]for j in range(batch)]
  for q,v in zip(reports,rows):
   assert q['same_variant_references'][str(micro)]==refs and v['phase']==c['phase']and v['generated_ids']==ids and v['per_row_first4_logits_sha256']==hashes and v['head_tail_ids_exact']and v['same_variant_full_IDs_exact']and v['same_variant_first4_logits_exact']and v['all8_tail_first4_logits_checked'];counters(v['actual_counters']);assert len(v['step_completion_seconds'])==512 and len(v['events'])==batch//4+511*(batch//micro)
   assert v['feedback_packets_by_stage']==([batch//4+511*(batch//micro),0,0,batch//4+511*(batch//micro)]if c['pipeline']else[0,0,0,0])
   for packet in range(batch//4):
    actual=sum([inputs[(packet*4+j)%4]for j in range(4)],[]);assert q['actual_input_packet_SHA834'][c['phase']][str(packet)]==hashlib.sha256(struct.pack('<'+str(len(actual))+'q',*actual)).hexdigest()
  for step in range(4):
   width=4 if step==0 else micro
   for packet in range(batch//width):
    offset=packet*width;raw_check(c['phase'],step,packet,width,[v[step]for v in hashes[offset:offset+width]],[v[step]for v in ids[offset:offset+width]])
  frozen=json.loads((r/'results/kimi-balanced-capacity-r880-910371.131-stage3.json').read_text());reference=next(v for v in frozen['cases']if v['phase']==c['phase'].replace('tokens_only_g64_','long_pipeline_g64_'));assert all(v['frozen880_full512_IDs_raw_exact']for v in rows)and c['generated_ids']==reference['generated_ids']and c['per_row_first4_logits_sha256']==reference['per_row_first4_logits_sha256']
  if micro==16:d.setdefault('old691_prefill8_diagnostic',[]).append({'phase':c['phase'],'IDs_exact':c['generated_ids']==goldcases[c['phase'].replace('tokens_only_g64_','long_pipeline_g64_')]['generated_ids'],'raw_exact':c['per_row_first4_logits_sha256']==goldcases[c['phase'].replace('tokens_only_g64_','long_pipeline_g64_')]['per_row_first4_logits_sha256']})
  times=c['step_completion_seconds'];steady=batch/statistics.mean(times[j]-times[j-1]for j in range(9,512));d['phases'].append({'phase':c['phase'],'global_batch':batch,'micro_batch':micro,'steady_total_tps':steady,'steady_per_request_tps':steady/batch,'including_prefill_total_tps':batch*512/max(v['elapsed_seconds']for v in rows),'first_TTFT_seconds':c['events'][0]['completion_seconds'],'last_TTFT_seconds':c['events'][batch//4-1]['completion_seconds'],'all4stage_IDs_rawfirst4_CPUargmax_inputSHA_verified':True})
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
 d['completed']=len(d['phases'])==6 and len(d['scalar_QA'])==44 and len(d['batch_QA'])==12 and collections.Counter(v['batch']for v in d['batch_QA'])=={16:12}and all(q.get('completed')and q.get('all8_owned_workers_shutdown_completed')for q in reports)and'|COMPLETED|0:0|'in state

d['scope']='965independent964 sameprocessfullCPUlogits/token-only audit: frozen880all512IDs/first4raw pluslate63/127/255/383/511rawbits/CPUargmax/all8,44scalar/12B16QA/32BWpreflight/shutdown. Same914csrc andGPUgreedy/waits. Actualforwardbytes/calls/enginewall/copywall andharnesscost; rankdebugcopies excludedfromCAPIcounts butincludedinbenchmarkwall. NotGPUkernel peak/HTTP/aggregate.'
d['output_metrics']=[];d['late_checkpoints']=[]
if len(reports)==4 and all('sdk_sha256'in q for q in reports):
 manifest_path=r/'candidates/kimi-token-only-r963/InfiniLM/source-manifest-r963.json';manifest=json.loads(manifest_path.read_text());assert hashlib.sha256((manifest_path.parent.parent/'kimi_engine_probe_r963.cpp').read_bytes()).hexdigest()==manifest['probe_sha256']
 parent=json.loads((r/'candidates/kimi-minpages-r914/InfiniLM/source-manifest-r914.json').read_text());assert manifest['files']==parent['files']
 assert all(len(q['actual_device_preflight'])==8 and all(v['small_device_compute_passed']and v['multiprocessors']==80 for v in q['actual_device_preflight'])for q in reports)
 checkpoints={0,1,2,3,63,127,255,383,511};phase_cases={q['stage']:{v['phase']:v for v in q['cases']}for q in reports};available=set.intersection(*(set(v)for v in phase_cases.values()))
 for phase in sorted(available):
  c=phase_cases[3][phase];batch=c['global_batch'];calls=batch//4+511*(batch//16);only=phase.startswith('tokens_only_');expected_omitted=(512-len(checkpoints))*(batch//16)if only else 0;expected_full=calls-expected_omitted;expected_bytes=(batch*len(checkpoints)if only else batch*512)*163840*2
  for stage in range(4):
   v=phase_cases[stage][phase];metrics=v['CAPI_output_metrics_delta'];host=v['harness_metrics'];assert metrics['engine_calls']==calls and host['forward_calls']==calls and host['omitted_logits_calls']==expected_omitted
   for k in ['engine_wall_seconds','logits_copy_seconds','id_copy_seconds']:assert metrics[k]>=0
   native=sum(e['native_forward_seconds']for e in v['events']);assert sum(metrics[k]for k in ['engine_wall_seconds','logits_copy_seconds','id_copy_seconds'])<=native+.003
   if stage==3:
    assert metrics['full_logits_calls']==expected_full and metrics['token_only_calls']==expected_omitted and metrics['forward_logits_bytes']==expected_bytes and metrics['forward_id_bytes']==batch*512*8
    assert v['late_checkpoint_count']==5*(batch//16)
   else:
    assert metrics['full_logits_calls']==metrics['token_only_calls']==metrics['forward_logits_bytes']==0 and v['late_checkpoint_count']==0
   assert host['logits_buffer_seconds']>=0 and host['tail_validation_seconds']>=0
   d['output_metrics'].append({'phase':phase,'stage':stage,'CAPI':metrics,'harness':host,'CAPI_native_forward_seconds':native})
 late_refs={}
 for c in reports[3].get('late_checkpoint_checks',[]):
  if c['phase']not in available:continue
  phase=c['phase'];pcase=phase_cases[3][phase];step=c['step'];offset=c['offset'];width=c['batch'];assert step in checkpoints and step>=4 and width==16 and offset==16*c['packet']and offset+16<=pcase['global_batch']and c['all8_tail_rank_logits_exact']and c['baseline_checkpoint_bits_exact']
  raw=(r/'results'/f'kimi-token-only-r964-{a.tag}-{phase}-q{c["packet"]}-step{step}-bf16.bin').read_bytes();assert len(raw)==16*163840*2 and hashlib.sha256(raw).hexdigest()==c['raw_sha256']
  ids=[v[step]for v in pcase['generated_ids'][offset:offset+16]];assert c['argmax_IDs']==ids;raw_check(phase,step,c['packet'],16,c['per_row_SHA256'],ids)
  key=(step,offset,width);value=c['per_row_SHA256'];assert key not in late_refs or late_refs[key]==value;late_refs[key]=value
  d['late_checkpoints'].append({'phase':phase,'step':step,'offset':offset,'rawSHA_CPUargmax_and_baseline_bits_verified':True})
 if d['completed']:assert len(d['output_metrics'])==24 and len(d['late_checkpoints'])==90
 if all(name in available for name in ['long_pipeline_g64_m16_r1','long_pipeline_g64_m16_r2','tokens_only_g64_m16_r1','tokens_only_g64_m16_r2']):
  values={v['phase']:v for v in d['phases']};normal=[values[f'long_pipeline_g64_m16_r{j}']['steady_total_tps']for j in[1,2]];tokens=[values[f'tokens_only_g64_m16_r{j}']['steady_total_tps']for j in[1,2]];d['same_process_comparison']={'baseline_tps':normal,'token_only_tps':tokens,'mean_speedup':statistics.mean(tokens)/statistics.mean(normal),'scope':'Sameprocess/sameGPUmodelbinary/G64M16; changedCPUoutput/harness costs withdeclaredrawcheckpoints, notkernelspeedup.'}
(r/'results'/f'kimi-token-only-r965-audit-{a.tag}.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
