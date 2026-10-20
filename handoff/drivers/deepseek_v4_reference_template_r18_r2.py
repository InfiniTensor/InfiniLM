from collections.abc import Mapping
import hashlib
from types import SimpleNamespace

REFERENCE_URL = 'https://github.com/pengcheng888/InfiniLM/blob/main_skill/python/infinilm/processors/deepseek_v4_processor.py'
REFERENCE_SOURCE_SHA256 = '16aecc7e369f37c1885ac8aed422f8760912af26a8499f1176e537b20e776e68'

def _fix_missing_chat_template(self):
        if getattr(self.tokenizer, "chat_template", None):
            return

        user_token = self.tokenizer.convert_ids_to_tokens(128803)
        assistant_token = self.tokenizer.convert_ids_to_tokens(128804)
        thinking_end_token = "</think>"

        self.tokenizer.chat_template = (
            "{{ bos_token }}"
            "{%- for message in messages -%}"
            "{%- if message['role'] == 'system' -%}"
            "{{ message['content'] }}"
            "{%- elif message['role'] == 'developer' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if loop.last and add_generation_prompt -%}"
            f"{assistant_token}{thinking_end_token}"
            "{%- endif -%}"
            "{%- elif message['role'] == 'user' -%}"
            f"{user_token}{{{{ message['content'] }}}}"
            "{%- if (not loop.last and messages[loop.index0 + 1]['role'] == 'assistant') or (loop.last and add_generation_prompt) -%}"
            f"{assistant_token}{thinking_end_token}"
            "{%- endif -%}"
            "{%- elif message['role'] == 'assistant' -%}"
            "{{ message['content'] }}{{ eos_token }}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )

def apply_reference_fallback(tokenizer):
    assert tokenizer.bos_token_id == 0 and tokenizer.eos_token_id == 1
    for value in [128803,128804]:
        text = tokenizer.convert_ids_to_tokens(value)
        assert isinstance(text,str) and text and tokenizer.encode(text, add_special_tokens=False) == [value], (value,text)
    _fix_missing_chat_template(SimpleNamespace(tokenizer=tokenizer))
    assert tokenizer.chat_template
    return {'reference_url':REFERENCE_URL,'source_sha256':REFERENCE_SOURCE_SHA256,
            'template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}

def encode_reference_single_user(tokenizer,text):
    value = tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True)
    if isinstance(value,Mapping):
        value = value['input_ids']
    if hasattr(value,'tolist'):
        value = value.tolist()
    if len(value)==1 and isinstance(value[0],(list,tuple)):
        value = value[0]
    assert isinstance(value,list) and all(isinstance(v,int) for v in value)
    assert value[0]==0 and value.count(128803)==1 and value.count(128804)==1
    assert 1 not in value
    return value
