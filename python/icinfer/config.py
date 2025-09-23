import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 16
    max_kvcache_tokens: int = -1
    num_kvcache_blocks: int = -1
    trust_remote_code: bool = False
    attention_bias: bool = False
    enable_paged_attn: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 4 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.model_path = self.model
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=self.trust_remote_code)
        print(self.model_path)
        self.check_hf_config()
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.num_kvcache_blocks < 0 and self.max_kvcache_tokens > 0:
            self.num_kvcache_blocks = self.max_kvcache_tokens // self.kvcache_block_size
        assert self.max_num_batched_tokens >= self.max_model_len
    
    def check_hf_config(self):
        if getattr(self.hf_config, "head_dim", None) is None:
            self.hf_config.head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
        if getattr(self.hf_config, "attention_bias", None) is None:
            self.hf_config.attention_bias = self.attention_bias
        if getattr(self.hf_config, "kvcache_block_size", None) is None:
            self.hf_config.kvcache_block_size = self.kvcache_block_size
