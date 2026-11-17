import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from infinilm.llm.llm import LLMEngine

class Tokenizer:
    vocab_size = 3
    def __init__(self): self.length_calls = 0
    def __len__(self):
        self.length_calls += 1
        return 5  # Includes two added tokens.
    def decode(self, ids): return ''.join(str(i) for i in ids)

class Request:
    def __init__(self):
        self.generated_token_ids=[];self._token_decode_offset=0;self.generated_text=''
        self._output_queue=None;self.request_id='test'
        self.sampling_params=SimpleNamespace(max_tokens=100,ignore_eos=True)
    def is_aborted(self): return False
    def is_finished(self): return False
    def get_total_length(self): return 2+len(self.generated_token_ids)
    def append_generated_token_id(self,i): self.generated_token_ids.append(i)
    def get_num_generated_tokens(self): return len(self.generated_token_ids)

class VocabularyCache(unittest.TestCase):
    def test_added_tokens_and_bounds_without_per_token_vocab_queries(self):
        tokenizer=Tokenizer()
        runner=SimpleNamespace(device='cpu',dtype='float32',eos_token_id=[],processor=SimpleNamespace(get_tokenizer=lambda:tokenizer))
        config=SimpleNamespace(model_path='unused',enable_prefix_caching=False,pipeline_parallel_size=1,pipeline_parallel_stage=0,cache_type='static',max_cache_len=128,device='cpu',enable_graph=False)
        with patch('infinilm.llm.llm.read_hf_config',return_value={'model_type':'test'}), patch('infinilm.llm.llm.model_uses_mamba_cache',return_value=False), patch('infinilm.llm.llm.ModelRunner',return_value=runner), patch('infinilm.llm.llm.StaticScheduler',return_value=MagicMock()), patch('infinilm.compile.env.prefill_chunked_enabled',return_value=False):
            engine=LLMEngine(config)
        self.assertEqual(tokenizer.length_calls,1)
        request=Request()
        engine._apply_sampled_token_ids(request,4)
        engine._apply_sampled_token_ids(request,3)
        self.assertEqual(request.generated_token_ids,[4,3])
        with self.assertRaisesRegex(ValueError,'Out-of-range'):
            engine._apply_sampled_token_ids(request,5)
        with self.assertRaisesRegex(ValueError,'Out-of-range'):
            engine._apply_sampled_token_ids(request,-1)
        self.assertEqual(tokenizer.length_calls,1)

if __name__=='__main__':unittest.main()
