import hashlib,json,pathlib
from collections.abc import Mapping

MODEL=pathlib.Path('/old/models/ZhipuAI-GLM-5.3')
def normalize_ids(value):
 if isinstance(value,Mapping):value=value['input_ids']
 if hasattr(value,'tolist'):value=value.tolist()
 if len(value)==1 and isinstance(value[0],(list,tuple)):value=value[0]
 return [int(v)for v in value]
def prepare_tokenizer():
 from transformers import PreTrainedTokenizerFast
 tokenizer=PreTrainedTokenizerFast.from_pretrained(MODEL,local_files_only=True)
 template=(MODEL/'chat_template.jinja').read_text()
 # Use the actual checkpoint template, including its reasoning controls.
 tokenizer.chat_template=template
 kwargs={'reasoning_effort':'low'}
 if 'enable_thinking' in template:kwargs['enable_thinking']=False
 generation=json.loads((MODEL/'generation_config.json').read_text())
 eos=generation['eos_token_id'];eos=eos if isinstance(eos,list)else[eos]
 assert tokenizer.eos_token_id in eos
 provenance={'source':str(MODEL/'chat_template.jinja'),'template_sha256':hashlib.sha256(template.encode()).hexdigest(),
             'tokenizer_config_sha256':hashlib.sha256((MODEL/'tokenizer_config.json').read_bytes()).hexdigest(),'kwargs':kwargs,'eos_token_ids':eos}
 return tokenizer,kwargs,provenance
def build_cases():
 tokenizer,kwargs,provenance=prepare_tokenizer()
 def encode(text):
  messages=[{'role':'user','content':text}]
  ids=normalize_ids(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**kwargs))
  rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**kwargs)
  assert ids[:2]==[154822,154824] and 154827 in ids and 154828 in ids
  assert all(0<=v<154880 for v in ids)
  return ids,rendered
 background='背景资料：订单状态包括待付款、待发货、已发货和已签收。不同订单编号互不影响。只根据给定记录回答，不要推测。\n'
 query='系统记录：订单 A17 的状态为待发货，订单 B21 已签收。只输出订单 A17 的状态，不要解释。'
 count=next(n for n in range(1,100) if len(encode(background*n+query)[0])>=512)
 cases=[{'label':'beijing','text':'请直接给出最终回答，不要展开推理。用一句话介绍北京。'},
        {'label':'arithmetic','text':'请直接给出最终结果，只输出十进制整数：137乘以29等于多少？'},
        {'label':'retrieval','text':background*count+query}]
 for case in cases:
  case['input_ids'],case['rendered']=encode(case['text'])
  case['input_tokens']=len(case['input_ids'])
  case['generation_starts_in_thinking']=case['rendered'].rstrip().endswith('<think>')
 assert 512<=cases[2]['input_tokens']<768
 return tokenizer,cases,provenance
def decode_answer(tokenizer,generated,started_in_thinking):
 raw=tokenizer.decode(generated,skip_special_tokens=True).strip()
 if '</think>' in raw:return raw.split('</think>')[-1].strip(),raw,True
 return raw,raw,not started_in_thinking
