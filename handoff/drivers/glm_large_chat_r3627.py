import ctypes,hashlib,json,pathlib,re,time
import torch

def run_chat_checks(lib,ctx,stage,rank,payload,control,check,ptr,stream,vocab,report,report_path):
 r=pathlib.Path('/refactor');tag=report['step'];records=[]
 for case in payload['chat_cases']:
  prompt=case['input_ids'];length=len(prompt)
  for repeat in [1,2]:
   check(lib.modern_glm_model_reset(ctx))
   barrier=torch.ones(1,device='cuda');control(barrier);assert barrier.item()==report['all_serving_cards']
   generated=[];hashes=[];step_times=[];begin=time.perf_counter()
   for step in range(256):
    tick=time.perf_counter();current=prompt if step==0 else [generated[-1]]
    past,width=(0,length) if step==0 else (length+step-1,1)
    ids=torch.tensor(current,device='cuda',dtype=torch.int64)
    positions=torch.arange(past,past+width,device='cuda',dtype=torch.int64)
    desc=(ctypes.c_size_t*3)(0,past,width)
    logits=torch.empty(1,vocab,device='cuda',dtype=torch.bfloat16)
    check(lib.modern_glm_model_forward(ctx,len(current),1,desc,ptr(ids),ptr(positions),ptr(logits),0,stream()))
    if stage==report['PP']-1:
     assert torch.isfinite(logits).all()
     if step<4:hashes.append(hashlib.sha256(logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    emitted=logits.argmax(-1).float() if stage==report['PP']-1 and rank==0 else torch.zeros(1,device='cuda')
    control(emitted);value=int(emitted.item());assert 0<=value<vocab
    generated.append(value);step_times.append(time.perf_counter()-tick)
    if value in payload['chat_eos_token_ids']:break
   row={'label':case['label'],'repeat':repeat,'input_tokens':length,'generated_ids':generated,'first4_logits_sha256':hashes,
        'stopped_at_eos':generated[-1]in payload['chat_eos_token_ids'],'elapsed_seconds':time.perf_counter()-begin,'step_seconds':step_times,
        'generation_starts_in_thinking':case['generation_starts_in_thinking']}
   records.append(row);report['chat_cases']=records;report['chat_template_provenance']=payload['chat_template_provenance']
   report_path.write_text(json.dumps(report,indent=2))
   print('GLM_NATIVE_CHAT_R20_CASE',stage,rank,case['label'],repeat,len(generated),row['stopped_at_eos'],flush=True)
 report['chat_cases_completed']=len(records)==6
 if stage==report['PP']-1 and rank==0:
  from glm_chat_template_r20 import prepare_tokenizer,decode_answer
  tokenizer,kwargs,provenance=prepare_tokenizer();assert provenance==payload['chat_template_provenance']
  decoded=[]
  for row in records:
   answer,raw,thinking_closed=decode_answer(tokenizer,row['generated_ids'],row['generation_starts_in_thinking'])
   known=('北京'in answer and ('中国'in answer or '首都'in answer))if row['label']=='beijing' else (bool(re.search(r'(?<!\d)3973(?!\d)',answer.replace(',','')))if row['label']=='arithmetic' else '待发货'in answer)
   decoded.append({'label':row['label'],'repeat':row['repeat'],'answer':answer,'raw_decoded':raw,'stopped_at_eos':row['stopped_at_eos'],'thinking_closed':thinking_closed,'known_answer_smoke_passed':bool(known and thinking_closed and row['stopped_at_eos'])})
  repeats=[{'label':records[i]['label'],'all_generated_ids_exact':records[i]['generated_ids']==records[i+1]['generated_ids'],'first4_logits_exact':records[i]['first4_logits_sha256']==records[i+1]['first4_logits_sha256']}for i in [0,2,4]]
  result={'scope':'Tail-rank0 decoded known-answer smoke only; cross-rank reports and terminal Slurm state require separate audit. All3 independent model quality/normal Engine/peakTPS remain unqualified.',
          'step':tag,'decoded':decoded,'repeat_checks':repeats,'known_answer_smoke_passed':all(v['known_answer_smoke_passed']for v in decoded),'all_tail_rank0_repeats_exact':all(v['all_generated_ids_exact']for v in repeats),'complete_model_quality_qualified':False}
  (r/'results'/f'glm-routed-fp8-chat-r20-{tag}-decoded.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  print('GLM_NATIVE_CHAT_R20_COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)
