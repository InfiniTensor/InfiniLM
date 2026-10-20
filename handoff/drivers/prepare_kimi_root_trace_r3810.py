import pathlib,ast
r=pathlib.Path(__file__).resolve().parent
s=(r/'kimi_spec_protocol_r3759.py').read_text(encoding='utf8').replace('r3759','r3810');start=s.index(' # Complete known QA before experimentation,')
s=s[:start]+'''
 plan=json.loads(pathlib.Path(os.environ['KIMI_ROOT_PLAN_R3810']).read_text());expected=plan['expected_AR_ids'];prefix_length=plan['prefix_length'];assert 1<=prefix_length<40 and len(expected)==128
 report.update(scope='Correctness-only firstacceptedblock versus serial onidenticalgreedy prefix, all93generic layertrace andselectedfirstKDA detailedtrace. NativeMode2/latest3775 aliasOFF. Full finite/CPUargmax/outputIDs andtraced/untraced continuation collected, no draftspeed or performanceadoption.',root_trace_only=True,main_goal_metric=False,performance_qualified=False,root_plan=plan,root_paths=[])
 def known(epoch):
  for name,c in chat['cases'].items():
   q=generate([c['input_ids']],1,0,'QA-'+epoch+'-'+name,128);key='sort'if name.startswith('sort_probe')else name;gold=[v['generated_ids'][0]for v in known_gold['cases']if v['task']==key and v['grouped']];assert q['generated_ids'][0]in gold,(name,q['generated_ids']);report['QA'].append({'task':name,'epoch':epoch,'complete_frozen_IDs_exact':True});save()
 known_gold=known;known('before')
 prompt=natural['cases'][0]['input_ids'];assert prompt==plan['prompt_ids'];baseline=generate([prompt],1,0,'root-AR-before',128);assert baseline['generated_ids'][0]==expected
 trace_root=r/'results'/('kimi-paired-root-r3810-'+tag);trace_root.mkdir(parents=True,exist_ok=True)
 os.environ['INFINILM_KIMI_TRACE_LAYERS']=','.join(str(i)for i in range(93));os.environ['INFINILM_KIMI_TRACE_FULL_FIRST']='0';os.environ['INFINILM_KIMI_TRACE_FULL_LAST']='92';os.environ['INFINILM_KIMI_KDA_TRACE_LAYER']=str(min(g['text']['linear_attn_config']['kda_layers'])-1)
 def root_path(label,block,tracing):
  check(lib.modern_kimi_engine_reset_r2475(ctx));control.collect('root-reset:'+label,{});active,spare=1,17;past=0;ids=[];details=[]
  def call(inputs,initial=False,trace_label=None):
   nonlocal active,spare,past
   directory=None
   if trace_label is not None:
    directory=trace_root/label/trace_label;directory.mkdir(parents=True,exist_ok=True);os.environ['INFINILM_KIMI_TRACE_DIR']=str(directory)
   else:os.environ.pop('INFINILM_KIMI_TRACE_DIR',None)
   try:out,_=forward([inputs],[past],[1],[active],[spare],label+'-'+(trace_label or 'untraced')+'-'+str(past),initial)
   finally:os.environ.pop('INFINILM_KIMI_TRACE_DIR',None)
   before_active,before_spare=active,spare
   if not initial:active,spare=spare,active
   past+=len(inputs);details.append({'past_after':past,'read_slot':before_active,'write_slot':before_spare if not initial else before_active,'inputs':inputs if not initial else None,'outputs':out[0],'trace_directory':str(directory)if directory else None});return out[0]
  ids=call(prompt,True)
  while len(ids)<prefix_length:
   out=call([ids[-1]]);ids+=out;assert ids==expected[:len(ids)],('identicalprefix',label,len(ids))
  assert past==len(prompt)+prefix_length-1 and ids==expected[:prefix_length]
  prefix_active,prefix_spare,prefix_past=active,spare,past
  if block:
   out=call([ids[-1],expected[prefix_length]],trace_label='block'if tracing else None)
   if out[0]==expected[prefix_length]:ids+=out
   else:
    # Same rejection semantics as the real verifier: replay untouched active.
    active,spare,past=prefix_active,prefix_spare,prefix_past;ids+=call([ids[-1]])
  else:
   ids+=call([ids[-1]],trace_label='serial-first'if tracing else None)
   ids+=call([ids[-1]],trace_label='serial-second'if tracing else None)
  while len(ids)<128 and ids[-1]not in eos:ids+=call([ids[-1]])
  first=next((i for i,(a,b)in enumerate(zip(ids,expected))if a!=b),min(len(ids),len(expected))if len(ids)!=len(expected)else None)
  q={'label':label,'used_block':block,'traced':tracing,'generated_ids':ids,'expected_AR_complete_IDs_exact':ids==expected,'first_token_difference':first,'prefix_length':prefix_length,'calls':details,'final_state_slot':active,'performance_qualified':False};report['root_paths'].append(q);save();return q
 serial=root_path('serial-traced',False,True);assert serial['generated_ids']==expected
 blocked=root_path('block-traced',True,True);untraced=root_path('block-untraced-1',True,False);repeat=root_path('block-untraced-2',True,False)
 report['root_traced_vs_untraced_IDs_exact']=blocked['generated_ids']==untraced['generated_ids']==repeat['generated_ids'];report['root_untraced_repeats_exact']=untraced['generated_ids']==repeat['generated_ids'];report['trace_root']=str(trace_root)
 after=generate([prompt],1,0,'root-AR-after',128);assert after['generated_ids']==expected;known('after')
 report['verification_math_final_counts']=math_counts();report['root_collection_completed']=True;report['spec_protocol_completed']=True;save()
'''
# Avoid shadowing the preloaded frozen known-answer object by the helper name.
s=s.replace(" def known(epoch):"," def run_known(epoch):").replace(" known_gold=known;known('before')"," known_gold=known;run_known('before')").replace(";known('after')",";run_known('after')")
ast.parse(s);(r/'kimi_root_protocol_r3810.py').write_text(s,encoding='utf8',newline='\n')
s=(r/'modern_kimi_spec_r3759.py').read_text(encoding='utf8').replace('r3759','r3810').replace('kimi_spec_protocol_r3810','kimi_root_protocol_r3810').replace('kimi-spec-math-5f91ad7-r3755','kimi-spec-alias-5f91ad7-r3775').replace('libmodern_kimi_spec_math_r3755.so','libmodern_kimi_spec_alias_r3775.so').replace('kimi-spec-math-r3755-build.json','kimi-spec-alias-r3775-build.json');ast.parse(s);(r/'modern_kimi_root_r3810.py').write_text(s,encoding='utf8',newline='\n')
s=(r/'kimi_spec_r3759_inside.sh').read_text(encoding='utf8').replace('modern_kimi_spec_r3759.py','modern_kimi_root_r3810.py');(r/'kimi_root_r3810_inside.sh').write_text(s,encoding='utf8',newline='\n')
# Hash exactly the executed harness names.
p=r/'kimi_root_protocol_r3810.py';s=p.read_text(encoding='utf8').replace('kimi_spec_protocol_r3810.py','kimi_root_protocol_r3810.py').replace('modern_kimi_spec_r3810.py','modern_kimi_root_r3810.py').replace('kimi_spec_r3810_inside.sh','kimi_root_r3810_inside.sh');p.write_text(s,encoding='utf8',newline='\n')
print('One-load identicalprefix accepted-block trace, untracedreplays, fullbefore/afterQA prepared; no oracleperformance.')
