#!/usr/bin/env python3
"""
Forward validation for the Qwen3.5 MTP draft model (C++ qwen3_5_mtp).

Compares the C++ draft forward against a torch reference built from the
HF transformers Qwen3.5 building blocks (Qwen3_5Attention/MLP/RMSNorm) plus
the MTP fusion defined by DeepSeek-V3 Multi-Token Prediction: the draft fuses
Norm(Emb(next token)) with Norm(previous hidden state), projects through fc,
runs one full-attention decoder layer, applies the final norm, and predicts
with the target-tied lm_head.

Inputs are produced by an HF transformers prefill of a fixed token sequence
(target last hidden states); both sides run on CUDA and their logits and
hidden states are compared within declared tolerances.
"""

import argparse
import gc
import json
import os
import sys
import tempfile

try:
    import torch
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5MLP,
        Qwen3_5RMSNorm,
        Qwen3_5TextRotaryEmbedding,
    )
except ImportError as e:
    print(f"Error: Required packages not found. Please install: {e}")
    sys.exit(1)

try:
    import infinicore
    from infinilm.cache.cache import PagedKVCacheConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
except ImportError as e:
    print("Error: InfiniLM package not found. Please install it:")
    print(f"  Error: {e}")
    sys.exit(1)

# Reuse the generic tensor helpers from test/models/llama/.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))

from utils import infinicore_to_torch_tensor, tensor_all_close  # noqa: E402

DEFAULT_MODEL_DIR = os.path.expanduser("~/models/Qwen3.5-2B")
DEFAULT_DEVICE = "cuda"
DEFAULT_SEED = 20260915
DEFAULT_PROMPT = (
    "The capital of France is Paris. The largest planet in the solar system is"
)
DEFAULT_MAX_NEW_TOKENS = 24
# bf16 both sides: tolerance covers kernel-order accumulation differences on
# logits whose values reach ~30 (see printed stats for the measured margins).
RTOL = 1e-2
ATOL = 1.0


def build_target_hidden(model_dir, prompt, max_new_tokens, device):
    """Prefill with HF transformers; return (token ids, last hidden states)."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    config = model.config

    input_ids = tokenizer.encode(prompt)
    with torch.no_grad():
        generated = model.generate(
            input_ids=torch.tensor([input_ids], device=device),
            attention_mask=torch.ones(
                1, len(input_ids), dtype=torch.long, device=device
            ),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=None,
        )
    seq = generated[0].tolist()
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([seq], device=device),
            output_hidden_states=True,
        )
    # hidden_states[-1] is the post-final-norm hidden state, the representation
    # the MTP module consumes as its "previous hidden state" input.
    target_hidden = output.hidden_states[-1].detach()
    embed_weight = model.model.embed_tokens.weight.detach().clone()

    del model, output
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return seq, target_hidden, embed_weight, config


class TorchMtpReference(torch.nn.Module):
    """MTP module reference: HF layers + DeepSeek-V3-style fusion."""

    def __init__(self, config, embed_weight, model_dir):
        super().__init__()
        self.model_dir = model_dir
        self.embed_weight = torch.nn.Parameter(embed_weight, requires_grad=False)
        self.fc = torch.nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Qwen3_5Attention(config, layer_idx=0)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rope = Qwen3_5TextRotaryEmbedding(config)

    def load_mtp_weights(self):
        """Load the checkpoint mtp.* weights directly (zero-centered norms)."""
        import glob as _glob

        # checkpoint suffix after "mtp." -> parameter of this module
        weight_map = {
            "fc.weight": self.fc.weight,
            "pre_fc_norm_embedding.weight": self.pre_fc_norm_embedding.weight,
            "pre_fc_norm_hidden.weight": self.pre_fc_norm_hidden.weight,
            "norm.weight": self.norm.weight,
            "layers.0.input_layernorm.weight": self.input_layernorm.weight,
            "layers.0.post_attention_layernorm.weight": self.post_attention_layernorm.weight,
        }
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            weight_map[f"layers.0.self_attn.{proj}.weight"] = getattr(
                self.attn, proj
            ).weight
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weight_map[f"layers.0.mlp.{proj}.weight"] = getattr(self.mlp, proj).weight

        loaded = 0
        for fp in sorted(_glob.glob(os.path.join(self.model_dir, "*.safetensors"))):
            with safe_open(fp, "pt", "cpu") as f:
                for key in f.keys():
                    if not key.startswith("mtp."):
                        continue
                    param = weight_map[key[len("mtp.") :]]
                    param.data = f.get_tensor(key).to(torch.bfloat16)
                    loaded += 1
        if loaded != 15:
            raise RuntimeError(f"expected 15 mtp.* weights, loaded {loaded}")

    def forward(self, input_ids, target_hidden, position_ids):
        embeds = torch.nn.functional.embedding(input_ids, self.embed_weight)
        normed_embed = self.pre_fc_norm_embedding(embeds)
        normed_hidden = self.pre_fc_norm_hidden(target_hidden)
        # Concat order is the one used by the shipped checkpoints (verified:
        # the swapped order collapses next-token accuracy to chance).
        fused = torch.cat([normed_embed, normed_hidden], dim=-1)
        hidden_states = self.fc(fused)

        seq_len = hidden_states.shape[1]
        # HF eager attention applies only the mask it is given; build the
        # causal mask explicitly (the full model does the same internally).
        causal = torch.full(
            (1, 1, seq_len, seq_len),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).triu(diagonal=1)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        cos_sin = self.rope(hidden_states, position_ids)
        attn_out, _ = self.attn(
            hidden_states=hidden_states,
            position_embeddings=cos_sin,
            attention_mask=causal,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        out_hidden = self.norm(hidden_states)
        logits = torch.nn.functional.linear(out_hidden, self.embed_weight)
        return logits, out_hidden


def run_cpp_draft(model_dir, seq, target_hidden, device):
    """Build the C++ qwen3_5_mtp engine on a fixture dir and run one forward."""
    fixture = tempfile.mkdtemp(prefix="qwen3_5_mtp_fixture_")
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    # Standalone single-layer full-attention draft model (mirrors the C++
    # config post-processing applied to the embedded mtp.* weights).
    config["model_type"] = "qwen3_5_mtp"
    text_config = dict(config["text_config"])
    text_config["num_hidden_layers"] = 1
    text_config["layer_types"] = ["full_attention"]
    config["text_config"] = text_config
    with open(os.path.join(fixture, "config.json"), "w") as f:
        json.dump(config, f)
    # The draft weights live inside the target checkpoint shards.
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            os.symlink(os.path.join(model_dir, name), os.path.join(fixture, name))

    seq_len = len(seq) - 1
    block_size = 256
    max_blocks = (seq_len + block_size - 1) // block_size
    engine = InferEngine(
        model_path=fixture,
        device=infinicore.device(device, 0),
        cache_config=PagedKVCacheConfig(
            num_blocks=max_blocks, block_size=block_size, max_batch_size=1
        ),
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, fixture, dtype=engine.dtype)

    draft_input_ids = infinicore.from_list([seq[1:]], dtype=infinicore.int64).to(
        infinicore.device(device, 0)
    )
    # mrope models take one position axis per section; text-only inputs repeat
    # the same positions on every axis.
    position_ids = infinicore.from_list(
        [list(range(seq_len)) for _ in range(3)], dtype=infinicore.int64
    ).to(infinicore.device(device, 0))
    target_hidden_infini = infinicore.from_torch(
        target_hidden[:, :-1, :].contiguous().to(device)
    )
    past_kv = infinicore.from_list([0], dtype=infinicore.int32)
    total_kv = infinicore.from_list([seq_len], dtype=infinicore.int32)
    block_tables = infinicore.from_list(
        [list(range(max_blocks))], dtype=infinicore.int32
    )
    slot_mapping = infinicore.from_list(list(range(seq_len)), dtype=infinicore.int64)
    input_offsets = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
    cu_seqlens = infinicore.from_list([0, seq_len], dtype=infinicore.int32)

    output = engine.forward_raw(
        draft_input_ids,
        position_ids=position_ids,
        past_kv_lengths=past_kv,
        total_kv_lengths=total_kv,
        input_offsets=input_offsets,
        cu_seqlens=cu_seqlens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        target_hidden_states=target_hidden_infini,
    )
    torch_reference = torch.empty(0)
    logits = infinicore_to_torch_tensor(output["logits"], torch_reference)
    hidden_states = infinicore_to_torch_tensor(output["hidden_states"], torch_reference)
    return logits.float(), hidden_states.float()


def compare(name, reference, actual, rtol, atol):
    """Compare a torch reference tensor with the C++ output; print stats."""
    is_close, stats = tensor_all_close(reference, actual, rtol=rtol, atol=atol)
    status = "✓" if is_close else "✗"
    print(
        f"   {status} {name}: max_abs_diff={stats['max_abs_diff']:.6f} "
        f"mean_abs_diff={stats['mean_abs_diff']:.6f} "
        f"ref_absmax={reference.abs().max().item():.4f}"
    )
    if not is_close:
        diff = (reference - actual).abs()
        worst = torch.unravel_index(torch.argmax(diff), diff.shape)
        print(
            f"       worst at {tuple(worst)}: ref={reference[worst].item():.4f} "
            f"actual={actual[worst].item():.4f}"
        )
    return is_close


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 MTP draft forward validation (C++ vs torch reference)"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3.5 MTP Draft Forward Validation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("✗ CUDA requested but torch.cuda is not available")
        return 1
    if args.device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.manual_seed(DEFAULT_SEED)

    print("\n1. Building inputs with HF transformers prefill...")
    seq, target_hidden, embed_weight, config = build_target_hidden(
        args.model, args.prompt, args.max_new_tokens, args.device
    )
    print(
        f"   sequence length: {len(seq)}, target hidden: {tuple(target_hidden.shape)}"
    )

    print("\n2. Running torch MTP reference (embed_first fusion)...")
    config._attn_implementation = "eager"
    reference = TorchMtpReference(config, embed_weight, args.model)
    reference.load_mtp_weights()
    reference = reference.to(args.device).eval()
    seq_len = len(seq) - 1
    draft_ids = torch.tensor([seq[1:]], device=args.device)
    positions = (
        torch.arange(seq_len, device=args.device).view(1, 1, -1).expand(3, 1, -1)
    )
    with torch.no_grad():
        ref_logits, ref_hidden = reference(
            draft_ids, target_hidden[:, :-1, :], positions
        )
    ref_logits = ref_logits.float().cpu()
    ref_hidden = ref_hidden.float().cpu()
    print(
        f"   ref logits: {tuple(ref_logits.shape)} absmax={ref_logits.abs().max().item():.4f}"
    )
    print(
        f"   ref hidden: {tuple(ref_hidden.shape)} absmax={ref_hidden.abs().max().item():.4f}"
    )
    del reference
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("\n3. Running C++ qwen3_5_mtp draft (InferEngine, paged attention)...")
    cpp_logits, cpp_hidden = run_cpp_draft(args.model, seq, target_hidden, args.device)
    print(f"   cpp logits: {tuple(cpp_logits.shape)}")
    print(f"   cpp hidden: {tuple(cpp_hidden.shape)}")

    print(f"\n4. Comparing (rtol={args.rtol}, atol={args.atol})...")
    # The last position predicts beyond the sequence; drop it on both sides.
    ok = True
    ok &= compare(
        "logits", ref_logits[:, :-1, :], cpp_logits[:, :-1, :], args.rtol, args.atol
    )
    ok &= compare("hidden_states", ref_hidden, cpp_hidden, args.rtol, args.atol)

    print("\n" + "=" * 70)
    if ok:
        print("✓ C++ draft forward matches the torch reference")
    else:
        print("✗ C++ draft forward deviates from the torch reference")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
