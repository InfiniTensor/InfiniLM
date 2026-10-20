import pathlib,ast
r=pathlib.Path(__file__).resolve().parent;p=r/'kimi_root_protocol_r3837.py';s=p.read_text(encoding='utf8')
needle=' for n in range(min(4,len(history)-1),0,-1):';assert s.count(needle)==1;s=s.replace(needle," minimum=int(os.environ.get('KIMI_LOOKUP_MIN_NGRAM_R3843','1'));assert minimum in[1,3,4]\n for n in range(min(4,len(history)-1),minimum-1,-1):")
s=s.replace("INFINILM_KIMI_TRACE_LAYERS']='0,1'","INFINILM_KIMI_TRACE_LAYERS']='0,1,3'").replace("INFINILM_KIMI_TRACE_FULL_LAST']='1'","INFINILM_KIMI_TRACE_FULL_LAST']='3'").replace('targetlayers0and1 generictrace plus selectedMLA1 rawlocal/reduced/internal trace','targetlayers0and1and3 generictrace plus selectedKDA1 detailedtrace andMLA3 W1/W2 internaltrace')
needle=" after=generate([prompt],1,0,'root-AR-after',128);";assert s.count(needle)==1
s=s.replace(needle,'''
 # One resident target also measures selective prompt lookup. No trained draft.
 report['selective_prompt_lookup_experiment_r3843']=[];os.environ.pop('INFINILM_KIMI_TRACE_DIR',None)
 for B in[1,8,16]:
  prompts=[natural['cases'][i%2]['input_ids']for i in range(B)]
  ar_before=generate(prompts,B,0,f'selective-B{B}-AR-A1',128);assert ar_before['generated_ids']==frozen['AR'][str(B)]['generated_ids']
  for minimum,K in[(3,3),(4,1)]:
   os.environ['KIMI_LOOKUP_MIN_NGRAM_R3843']=str(minimum)
   for repetition in[1,2]:
    trial=generate(prompts,B,K,f'selective-B{B}-n{minimum}-K{K}-r{repetition}',128);trial['minimum_lookup_ngram']=minimum;trial['complete_IDs_equal_frozen_AR']=trial['generated_ids']==ar_before['generated_ids'];trial['nonempty_proposal_rounds']=sum(z['drafted']>0 for z in trial['rounds']);report['selective_prompt_lookup_experiment_r3843'].append({'label':trial['label'],'batch':B,'minimum_ngram':minimum,'K':K,'complete_IDs_equal_frozen_AR':trial['complete_IDs_equal_frozen_AR'],'drafted':trial['drafted_tokens'],'accepted':trial['accepted_draft_tokens'],'main_goal_metric':False});save()
  os.environ.pop('KIMI_LOOKUP_MIN_NGRAM_R3843',None)
  ar_after=generate(prompts,B,0,f'selective-B{B}-AR-A2',128);assert ar_after['generated_ids']==ar_before['generated_ids'];save()
 after=generate([prompt],1,0,'root-AR-after',128);''')
s=s.replace("report['verification_math_final_counts']=math_counts();", "report['selective_prompt_lookup_matrix_completed']=len(report['selective_prompt_lookup_experiment_r3843'])==12;assert report['selective_prompt_lookup_matrix_completed'];report['verification_math_final_counts']=math_counts();")
ast.parse(s)
module=ast.parse(s);scope={};exec(compile(ast.Module(body=[module.body[0]]+[x for x in module.body if isinstance(x,(ast.Import,ast.ImportFrom))]+[x for x in module.body if isinstance(x,ast.FunctionDef)and x.name=='propose'],type_ignores=[]),'<propose-only>','exec'),scope)
scope['os'].environ['KIMI_LOOKUP_MIN_NGRAM_R3843']='3';assert scope['propose']([1,2,3,4,1,2,3],3)==[4,1,2];assert scope['propose']([1,2,1],1)==[]
scope['os'].environ['KIMI_LOOKUP_MIN_NGRAM_R3843']='4';assert scope['propose']([1,2,3,4,1,2,3],3)==[];scope['os'].environ.pop('KIMI_LOOKUP_MIN_NGRAM_R3843',None)
p.write_text(s,encoding='utf8',newline='\n');print('Selective n3/K3,n4/K1 B1/8/16, AR2/spec2 each, same resident target prepared; all speculative cases remain excluded mainmetric.')
