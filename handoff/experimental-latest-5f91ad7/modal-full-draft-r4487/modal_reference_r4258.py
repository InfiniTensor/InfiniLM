# Extracted from MIT-licensed z-lab/dflash, commit 07ebd93db9f472af339b644bb70221ad8428328a.
# First-generation model definitions only; no downloaded generation, training or CLI code.
from typing import ClassVar
import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
 ALL_ATTENTION_FUNCTIONS, GradientCheckpointingLayer, Qwen3Config, Qwen3MLP,
 Qwen3PreTrainedModel, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half)
def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [round(start + i * span / (num_draft_layers - 1)) for i in range(num_draft_layers)]

def _draft_config(config):
    return getattr(config, 'dflash_config', {})

def _draft_value(config, name, default=None):
    return _draft_config(config).get(name, getattr(config, name, default))

def _attention_mask(query, key, *, is_causal, sliding_window):
    query_position = key.shape[-2] - query.shape[-2] + torch.arange(query.shape[-2], device=query.device)[:, None]
    key_position = torch.arange(key.shape[-2], device=query.device)[None, :]
    visible = torch.ones((query.shape[-2], key.shape[-2]), dtype=torch.bool, device=query.device)
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    return visible[None, None]

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = q * cos[..., -q_len:, :] + rotate_half(q) * sin[..., -q_len:, :]
    k_embed = k * cos + rotate_half(k) * sin
    return (q_embed, k_embed)

class Qwen3DFlashAttention(nn.Module):

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** (-0.5)
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        layer_types = getattr(config, 'layer_types', None)
        layer_type = layer_types[layer_idx] if layer_types else 'full_attention'
        is_causal = getattr(config, 'is_causal', None)
        self.is_causal = layer_type == 'sliding_attention' if is_causal is None else bool(is_causal)
        self.sliding_window = config.sliding_window if layer_type == 'sliding_attention' else None

    def forward(self, hidden_states: torch.Tensor, target_hidden: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: torch.Tensor | None, past_key_values: Cache | None=None, cache_position: torch.LongTensor | None=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        if attention_mask is None and (self.is_causal or self.sliding_window is not None):
            attention_mask = _attention_mask(q, k, is_causal=self.is_causal, sliding_window=self.sliding_window)
        attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS['sdpa'](self, q, k, v, attention_mask, dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)

class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_conv = None
        self.mlp_conv = None

    def forward(self, target_hidden: torch.Tensor | None=None, hidden_states: torch.Tensor | None=None, attention_mask: torch.Tensor | None=None, position_ids: torch.LongTensor | None=None, past_key_value: Cache | None=None, output_attentions: bool | None=False, use_cache: bool | None=False, cache_position: torch.LongTensor | None=None, position_embeddings: tuple[torch.Tensor, torch.Tensor] | None=None, **kwargs) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, target_hidden=target_hidden, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_value, output_attentions=output_attentions, use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)[0]
        if attention_kernel is not None:
            hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        hidden_states = residual + hidden_states
        return hidden_states

class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules: ClassVar[list[str]] = ['Qwen3DFlashDecoderLayer']

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList([Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.target_layer_ids = _draft_value(config, 'target_layer_ids', build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = int(_draft_value(config, 'block_size', 16))
        self.mask_token_id = _draft_value(config, 'mask_token_id')
        self.post_init()

    def forward(self, position_ids: torch.LongTensor, attention_mask: torch.Tensor | None=None, noise_embedding: torch.Tensor | None=None, target_hidden: torch.Tensor | None=None, past_key_values: Cache | None=None, use_cache: bool=False, **kwargs) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, target_hidden=target_hidden, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        return self.norm(hidden_states)

    def compute_logits(self, hidden, output_head):
        logits = output_head(hidden)
        logits = logits * float(_draft_value(self.config, 'output_multiplier', 1.0))
        softcap = _draft_value(self.config, 'final_logit_softcapping')
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits
