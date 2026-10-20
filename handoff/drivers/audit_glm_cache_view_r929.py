import argparse,array,datetime,hashlib,json,pathlib,statistics,subprocess
p=argparse.ArgumentParser();p.add_argument('tag');a=p.parse_args();r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
state=subprocess.check_output(['sacct','-j',a.tag,'-n','-P','-o','JobID,JobName%65,State,ExitCode,Elapsed'],text=True)
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'completed':False,'phases':[],'repeat_pairs':[],'scope':'812independent811 full78TP8PP4/809decodecache807variant, cachecountersandownerrelease: immutable776helper/SDK587/source/DSO, ownM4prefix16 M8/M16full512refs/repeats/G32G64pipeline, all32IDs/8tailraw/CPUargmax/actualhelperBLASpolicy/knownQA/crossepoch/shutdown/Slurm. Old98diagnostic; no oldprecision, modelmax or aggregate claim.'}
reports=[]
for s in range(4):
 for rank in range(8):
  path=r/'results'/f'glm-fp8-cached-model-r928-{a.tag}-stage{s}-rank{rank}.json'
  if path.exists():
   try:reports.append(json.loads(path.read_text()))
   except json.JSONDecodeError as e:d.setdefault('live_read_errors',[]).append({'path':path.name,'error':str(e)})
d['reports_available']=len(reports)
if len(reports)==32:
 build=json.loads((r/'results/glm-cache-view-r927-build.json').read_text());assert all(q['library_sha256']==build['library_sha256']and q['actual_sdk_sha256']==build['sdk_sha256']=='58735f6730d7092efe8cfb44cfd57170d36c0be31d89b3145a79a3504477ae7f'and q['fp8_helper_sha256']==build['helper_sha256']for q in reports)
 assert hashlib.sha256((r/'candidates/glm-cache-view-r927/libmodern_glm_cache_view_r927.so').read_bytes()).hexdigest()==build['library_sha256']
 assert hashlib.sha256((r/'candidates/glm-fp8-batched-r776/libglm_fp8_batched_r776.so').read_bytes()).hexdigest()==build['helper_sha256']
 for n,h in build['source_sha256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h
 assert all(q['cache_rows']==64 and q['pipeline_boundaries']==[0,18,38,58,78]and q['actual_layer_indices']==list(range([0,18,38,58][q['stage']],[18,38,58,78][q['stage']]))for q in reports)
 vocab=json.loads((r.parent/'hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json').read_text())['vocab_size']
 for index in range(min(len(q['cases'])for q in reports)):
  rows=[q['cases'][index]for q in reports];c=rows[24];batch,micro,count=c['global_batch'],c['micro_batch'],c['output_tokens_per_request'];assert all(v['phase']==c['phase']and v['same_variant_full_IDs_first4_logits_exact']and v['fp8_batched_calls']>0 and v['actual_direct_decode_calls']>0 for v in rows)
  frozen929=json.loads((r/'results/glm-fp8-cached-model-r811-910375.139-stage3-rank0.json').read_text());old929=next(v for v in frozen929['cases']if v['phase']==c['phase']);assert all(v['frozen811_full_IDs_raw_exact']and v['cache_view_calls']>0 for v in rows)
  assert all(v['generated_ids']==old929['generated_ids']and v['per_row_first4_logits_sha256']==old929['per_row_first4_logits_sha256']for v in rows[24:])
  refs=reports[24]['same_variant_references'][str(micro)];ids=[refs['generated_ids'][i%micro]for i in range(batch)];hashes=[refs['hashes'][i%micro]for i in range(batch)]
  assert all(v['generated_ids']==ids and v['per_row_first4_logits_sha256']==hashes for v in rows[24:])and all(v['generated_ids']==ids for v in rows[:8])
  if not c['pipeline']:assert all(v['generated_ids']==ids for v in rows)
  for v in rows:
   cs=v['cache_stats'];assert cs['active']<=10 and cs['active_bytes']<=192*1024**2 and cs['invalid']==0
   if micro in[8,16]:assert cs['runs']>0 and cs['hits']>0
   policy=v['BLAS_policy'];assert policy['failures']==0 and policy['owners']and all(t['created']==t['verified']for t in policy['owners'])and any('libglm_fp8_batched_r776.so'in t['caller']for t in policy['owners'])
   assert len(v['events'])==batch//min(micro,8)+(count-1)*(batch//micro)and len(v['step_completion_seconds'])==count
  for step in range(4):
   width=min(micro,8)if step==0 else micro
   for packet in range(batch//width):
    raw=(r/'results'/f'glm-fp8-cached-model-r928-{a.tag}-{c["phase"]}-q{packet}-s{step}-bf16.bin').read_bytes();assert len(raw)==width*vocab*2
    for j in range(width):
     part=raw[j*vocab*2:(j+1)*vocab*2];assert hashlib.sha256(part).hexdigest()==hashes[packet*width+j][step];words=array.array('h');words.frombytes(part);best=max(words);best=min(words)if best<0 else best;assert words.index(best)==ids[packet*width+j][step]
  times=c['step_completion_seconds'];steady=batch/statistics.mean(times[j]-times[j-1]for j in range(9,count))
  d['phases'].append({'phase':c['phase'],'global_batch':batch,'micro_batch':micro,'output_tokens':count,'steady_total_tps':steady,'including_prefill_total_tps':batch*count/max(v['elapsed_seconds']for v in rows),'first_TTFT_seconds':min(c['first_token_seconds_by_row']),'last_TTFT_seconds':max(c['first_token_seconds_by_row']),'full_IDs_rawfirst4_CPUargmax_all32_verified':True,'old98_IDs_exact_diagnostic':c['old_frozen98_IDs_exact_diagnostic'],'old98_raw_exact_diagnostic':c['old_frozen98_raw_exact_diagnostic']})
 byname={v['phase']:v for v in d['phases']}
 for prefix in['reference_g4_m4','reference_g8_m8','long_pipeline_g32_m8','reference_g16_m16','long_pipeline_g64_m16']:
  pair=[byname.get(prefix+'_r'+str(j))for j in[1,2]]
  if all(pair):d['repeat_pairs'].append({'label':prefix,'steady_total_tps':[v['steady_total_tps']for v in pair],'including_prefill_total_tps':[v['including_prefill_total_tps']for v in pair]})
 if all(q.get('completed')and q.get('worker_shutdown_completed')for q in reports):
  for q in reports:
   assert q['cache_library_sha256']==build['cache_library_sha256'];after=q['cache_after_destroy'];before=q['cache_before_destroy'];assert before['runs']>0 and before['hits']>0 and after['created']==after['destroyed']and after['active']==after['active_bytes']==after['invalid']==0
  for epoch in['before_performance','after_performance']:
   assert all(q.get(epoch+'_chat_gate_passed')and len(q[epoch+'_chat_cases'])==6 for q in reports)
   chats=[q[epoch+'_chat_cases']for q in reports];assert all([v['generated_ids']for v in c]==[v['generated_ids']for v in chats[0]]for c in chats)
   assert all([v['first4_logits_sha256']for v in c]==[v['first4_logits_sha256']for v in chats[24]]for c in chats[24:])
   decoded=json.loads((r/'results'/f'glm-fp8-cached-model-r928-{a.tag}-{epoch}-decoded.json').read_text());assert decoded['known_answer_smoke_passed']and decoded['all_tail_rank0_repeats_exact']
  assert all([v['generated_ids']for v in q['before_performance_chat_cases']]==[v['generated_ids']for v in q['after_performance_chat_cases']]for q in reports)
  assert all([v['first4_logits_sha256']for v in q['before_performance_chat_cases']]==[v['first4_logits_sha256']for v in q['after_performance_chat_cases']]for q in reports[24:])
  d['before_after_QA_and_shutdown_verified']=True
  d['completed']=len(d['phases'])==10 and len(d['repeat_pairs'])==5 and '|COMPLETED|0:0|'in state
(r/'results'/f'glm-fp8-cached-model-r929-{a.tag}-audit.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
