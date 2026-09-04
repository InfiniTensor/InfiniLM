#!/usr/bin/env python3
"""Single source of truth for GGUF-to-InfiniLM tensor mapping.

InfiniLM weights use [out, in] orientation, matching GGUF packed row order, so
conversion does not transpose weight data. Transform semantics follow
llama.cpp's Qwen conversion: recover A_log from -exp(A_log), preserve baked
normalization offsets, restore grouped value-head rows, and describe runtime
activation permutation for column-reordered output projections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Transform semantics
# ---------------------------------------------------------------------------
T_NONE = ""  # Preserve packed bytes, or only cast dense values.
T_VROWS = "vrows"  # Restore value-head row blocks to grouped order.
T_VELEM = "velem"  # One scalar per head; implemented by the same row transform.
T_ALOG = "alog"  # Recover A_log = log(-ssm_a), then permute.
T_DENSE = "dense"  # Dequantize to BF16 for parameters without a packed path.

# Value-head permutation scope along dimension 0:
#   all    = the entire dimension contains value heads
#   v_tail = only the trailing value_dim segment contains value heads
VPERM_ALL, VPERM_TAIL = "all", "v_tail"

# Checkpoint suffix shared with the C++ packed-weight layout.
BLOB_SUFFIX = "weight_bytes"

# Both forms share one implementation; elements per head are derived from shape.
VPERM_TRANSFORMS = (T_VROWS, T_VELEM)


def needs_vperm(e: "Entry") -> bool:
    return bool(set(e.transforms) & set(VPERM_TRANSFORMS))


# ---------------------------------------------------------------------------
# GGML type ids from ggml.h. Keep this module independent of gguf-py.
# ---------------------------------------------------------------------------
F32, Q8_0, Q4_K, Q5_K, Q6_K = "F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K"
IQ4_NL, IQ4_XS = "IQ4_NL", "IQ4_XS"

# Packed block types supported by the runtime kernel.
NATIVE_BLOB_TYPES = (Q8_0, Q4_K, Q5_K, Q6_K)
# I-quants converted to dense BF16 until native kernels are available.
V1_IQUANT_DENSE = (IQ4_NL, IQ4_XS)
DENSE_SRC_TYPES = (F32, Q8_0, Q6_K) + V1_IQUANT_DENSE


def apply_v1_exceptions(plan, gguf_types, enabled=True):
    """Convert I-quant entries without native kernels to dense BF16 in place.

    ``gguf_types`` maps tensor names to GGML type names collected by the caller.
    Set ``enabled=False`` when native IQ4 kernels become available.
    """
    n = 0
    if enabled:
        for e in plan:
            if e.blob and gguf_types.get(e.gguf) in V1_IQUANT_DENSE:
                e.blob = False
                e.transforms = e.transforms + (T_DENSE,)
                e.note = (
                    (e.note + "；" if e.note else "")
                    + "dense fallback for source %s; remove when a native kernel is available"
                    % gguf_types[e.gguf]
                )
                n += 1
    return n


@dataclass
class Entry:
    """Map one GGUF tensor to one InfiniLM parameter."""

    infinilm: str  # InfiniLM parameter name including model.language_model prefix.
    gguf: str  # GGUF tensor name.
    shape: tuple  # Full InfiniLM shape before TP, in [out, in] orientation.
    blob: bool  # Preserve original blocks as U8 [out, row_bytes].
    transforms: tuple = ()
    types: tuple = ()  # Allowed source types; empty accepts any reported type.
    slices: tuple = ()  # [start, end) ranges along output dimension.
    vperm: str = VPERM_ALL  # Scope for T_VROWS.
    # Column permutations cross quantization blocks and would require
    # requantization. Record them as runtime activation-permutation rules rather
    # than conversion-time transforms.
    act_vperm: bool = False
    note: str = ""


@dataclass
class Dims:
    hidden: int
    n_q_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    lin_k_heads: int
    lin_v_heads: int
    lin_k_dim: int
    lin_v_dim: int
    conv_kernel: int
    vocab: int
    n_layers: int
    interval: int
    mrope_section: tuple = (11, 11, 10)
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    architectures: str = "Qwen3_5ForConditionalGeneration"

    # Derived dimensions.
    @property
    def q_rows(self) -> int:  # q_proj rows with interleaved query and gate values.
        return self.n_q_heads * self.head_dim * 2

    @property
    def kv_rows(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def o_in(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def key_dim(self) -> int:
        return self.lin_k_heads * self.lin_k_dim

    @property
    def value_dim(self) -> int:
        return self.lin_v_heads * self.lin_v_dim

    @property
    def qkv_rows(self) -> int:  # Fused q | k | v rows matching GGUF attn_qkv.
        return self.key_dim * 2 + self.value_dim

    @property
    def conv_channels(self) -> int:
        return self.qkv_rows

    @property
    def v_per_k(self) -> int:
        return self.lin_v_heads // self.lin_k_heads

    def layer_types(self) -> list:
        """Match prepare_qwen3_5_model_config: (i + 1) % interval == 0."""
        return [
            "full_attention" if (i + 1) % self.interval == 0 else "linear_attention"
            for i in range(self.n_layers)
        ]


REAL = Dims(
    hidden=5120,
    n_q_heads=24,
    n_kv_heads=4,
    head_dim=256,
    ffn=17408,
    lin_k_heads=16,
    lin_v_heads=48,
    lin_k_dim=128,
    lin_v_dim=128,
    conv_kernel=4,
    vocab=248320,
    n_layers=64,
    interval=4,
)

MINI = Dims(
    hidden=512,
    n_q_heads=2,
    n_kv_heads=1,
    head_dim=256,
    ffn=1024,
    lin_k_heads=2,
    lin_v_heads=6,
    lin_k_dim=128,
    lin_v_dim=128,
    conv_kernel=4,
    vocab=1024,
    n_layers=8,
    interval=4,
)


PREFIX = "model.language_model."


def layer_entries(d: Dims, i: int, role: str) -> list:
    """Return entries for layer ``i`` and its attention role.

    Source types are discovered from the input because the same suffix may use
    different quantization types across layers.
    """
    L = f"{PREFIX}layers.{i}."
    G = f"blk.{i}."
    kd, vd = d.key_dim, d.value_dim
    out = [
        Entry(
            L + "input_layernorm.weight",
            G + "attn_norm.weight",
            (d.hidden,),
            False,
            (T_DENSE,),
            note="GGUF already contains the baked +1 offset",
        ),
        Entry(
            L + "post_attention_layernorm.weight",
            G + "post_attention_norm.weight",
            (d.hidden,),
            False,
            (T_DENSE,),
            note="GGUF already contains the baked +1 offset",
        ),
        Entry(
            L + "mlp.gate_proj.weight", G + "ffn_gate.weight", (d.ffn, d.hidden), True
        ),
        Entry(L + "mlp.up_proj.weight", G + "ffn_up.weight", (d.ffn, d.hidden), True),
        Entry(
            L + "mlp.down_proj.weight", G + "ffn_down.weight", (d.hidden, d.ffn), True
        ),
    ]
    if role == "full_attention":
        out += [
            Entry(
                L + "self_attn.q_proj.weight",
                G + "attn_q.weight",
                (d.q_rows, d.hidden),
                True,
                (),
                note="rows contain interleaved query and gate values per head",
            ),
            Entry(
                L + "self_attn.k_proj.weight",
                G + "attn_k.weight",
                (d.kv_rows, d.hidden),
                True,
            ),
            Entry(
                L + "self_attn.v_proj.weight",
                G + "attn_v.weight",
                (d.kv_rows, d.hidden),
                True,
            ),
            Entry(
                L + "self_attn.o_proj.weight",
                G + "attn_output.weight",
                (d.hidden, d.o_in),
                True,
            ),
            Entry(
                L + "self_attn.q_norm.weight",
                G + "attn_q_norm.weight",
                (d.head_dim,),
                False,
                (T_DENSE,),
                note="GGUF already contains the baked +1 offset",
            ),
            Entry(
                L + "self_attn.k_norm.weight",
                G + "attn_k_norm.weight",
                (d.head_dim,),
                False,
                (T_DENSE,),
                note="GGUF already contains the baked +1 offset",
            ),
        ]
    else:
        out += [
            Entry(
                L + "linear_attn.in_proj_q.weight",
                G + "attn_qkv.weight",
                (kd, d.hidden),
                True,
                (),
                slices=((0, kd),),
                note="attn_qkv rows [0:kd]",
            ),
            Entry(
                L + "linear_attn.in_proj_k.weight",
                G + "attn_qkv.weight",
                (kd, d.hidden),
                True,
                (),
                slices=((kd, 2 * kd),),
                note="attn_qkv rows [kd:2kd]",
            ),
            Entry(
                L + "linear_attn.in_proj_v.weight",
                G + "attn_qkv.weight",
                (vd, d.hidden),
                True,
                (T_VROWS,),
                slices=((2 * kd, 2 * kd + vd),),
                note="attn_qkv rows [2kd:] with tiled-to-grouped value heads",
            ),
            Entry(
                L + "linear_attn.in_proj_z.weight",
                G + "attn_gate.weight",
                (vd, d.hidden),
                True,
                (T_VROWS,),
                note="row permutation from qwen.py using head_v_dim",
            ),
            Entry(
                L + "linear_attn.in_proj_a.weight",
                G + "ssm_alpha.weight",
                (d.lin_v_heads, d.hidden),
                False,
                (T_DENSE, T_VROWS),
                note="dense fallback plus qwen.py row permutation with head_dim=1",
            ),
            Entry(
                L + "linear_attn.in_proj_b.weight",
                G + "ssm_beta.weight",
                (d.lin_v_heads, d.hidden),
                False,
                (T_DENSE, T_VROWS),
                note="dense fallback plus row permutation with head_dim=1",
            ),
            Entry(
                L + "linear_attn.A_log",
                G + "ssm_a",
                (d.lin_v_heads,),
                False,
                (T_ALOG, T_VROWS),
                note="GGUF stores -exp(A_log); recover it with log(-x)",
            ),
            Entry(
                L + "linear_attn.dt_bias",
                G + "ssm_dt.bias",
                (d.lin_v_heads,),
                False,
                (T_VELEM,),
                note="per-head qwen.py permutation without changing values",
            ),
            Entry(
                L + "linear_attn.conv1d.weight",
                G + "ssm_conv1d.weight",
                (d.conv_channels, 1, d.conv_kernel),
                False,
                (T_DENSE, T_VROWS),
                vperm=VPERM_TAIL,
                note="restore squeezed [C,K] shape and permute only trailing V channels",
            ),
            Entry(
                L + "linear_attn.norm.weight",
                G + "ssm_norm.weight",
                (d.lin_v_dim,),
                False,
                (T_DENSE,),
                note="not permuted by qwen.py and no normalization offset",
            ),
            Entry(
                L + "linear_attn.out_proj.weight",
                G + "ssm_out.weight",
                (d.hidden, vd),
                True,
                (),
                act_vperm=True,
                note="column permutation requires grouped-to-tiled runtime activation mapping",
            ),
        ]
    return out


def build_plan(d: Dims) -> list:
    """Build full-model mapping entries, including root-level tensors."""
    entries = [
        Entry(
            PREFIX + "embed_tokens.weight",
            "token_embd.weight",
            (d.vocab, d.hidden),
            False,
            (T_DENSE,),
            note="dequantize embedding to dense BF16",
        ),
    ]
    for i, role in enumerate(d.layer_types()):
        entries += layer_entries(d, i, role)
    entries += [
        Entry(
            PREFIX + "norm.weight",
            "output_norm.weight",
            (d.hidden,),
            False,
            (T_DENSE,),
            note="GGUF already contains the baked +1 offset",
        ),
        Entry(
            "lm_head.weight",
            "output.weight",
            (d.vocab, d.hidden),
            False,
            (T_DENSE,),
            note="dequantize output head to dense BF16",
        ),
    ]
    return entries


def activation_vperm_suffix(e: "Entry") -> str:
    """Return the layer-independent checkpoint-stem suffix for C++ matching."""
    name = re.sub(r"^" + re.escape(PREFIX) + r"layers\.\d+\.", "", e.infinilm)
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name + "."


def activation_vperm_rules(d: "Dims", plan: list) -> list:
    """Derive runtime value-head activation permutations for quantization_config.

    llama.cpp exports selected output-projection columns in tiled order, while
    the GDN kernel produces grouped activations. Packed columns cannot be moved
    across blocks without requantization, so the runtime permutes activations.
    """
    n_k, r, hd = d.lin_k_heads, d.v_per_k, d.lin_v_dim
    rules, seen = [], set()
    for e in plan:
        if not e.act_vperm:
            continue
        in_dim = int(e.shape[1])
        if in_dim != n_k * r * hd:
            raise ValueError(
                "%s: input dimension %d != num_k_heads*num_v_per_k*head_dim %d; "
                "cannot permute complete heads" % (e.infinilm, in_dim, n_k * r * hd)
            )
        suffix = activation_vperm_suffix(e)
        if suffix in seen:
            continue
        seen.add(suffix)
        rules.append(
            {"suffix": suffix, "num_k_heads": n_k, "num_v_per_k": r, "head_dim": hd}
        )
    return rules


def expected_keys(d: Dims) -> list:
    return [e.infinilm for e in build_plan(d)]


# GGUF tensor prefixes excluded from the main model, including the MTP block.
DROP_PREFIXES = ("blk.64.",)
MTP_BLOCK = 64


def compress(shape: tuple) -> tuple:
    """Remove singleton dimensions when comparing squeezed GGUF tensors."""
    return tuple(int(x) for x in shape if int(x) != 1)


# ---------------------------------------------------------------------------
# Derived checkpoint names, type-table keys, packed row sizes, and config data.
# ---------------------------------------------------------------------------
def ckpt_name(e: "Entry") -> str:
    """Return the safetensors and framework state-dict parameter name."""
    if e.blob and e.infinilm.endswith(".weight"):
        return e.infinilm[: -len(".weight")] + "." + BLOB_SUFFIX
    return e.infinilm


def type_table_key(name: str) -> str:
    """Return the exact checkpoint name used as the ggml_types table key."""
    return name


def row_bytes(n_in: int, block_size: int, type_size: int) -> int:
    """Return bytes per packed row using caller-provided GGML block metadata."""
    if n_in % block_size:
        raise ValueError(
            "input size %d is not divisible by block size %d" % (n_in, block_size)
        )
    return n_in // block_size * type_size


def make_text_config(d: "Dims") -> dict:
    """Build the Qwen3.5 text_config consumed by InfiniLM."""
    return {
        "model_type": "qwen3_5_text",
        "hidden_size": d.hidden,
        "num_hidden_layers": d.n_layers,
        "num_attention_heads": d.n_q_heads,
        "num_key_value_heads": d.n_kv_heads,
        "head_dim": d.head_dim,
        "intermediate_size": d.ffn,
        "rms_norm_eps": d.rms_norm_eps,
        "max_position_embeddings": d.max_position_embeddings,
        "vocab_size": d.vocab,
        "full_attention_interval": d.interval,
        "linear_num_key_heads": d.lin_k_heads,
        "linear_num_value_heads": d.lin_v_heads,
        "linear_key_head_dim": d.lin_k_dim,
        "linear_value_head_dim": d.lin_v_dim,
        "linear_conv_kernel_dim": d.conv_kernel,
        "attention_bias": False,
        "rope_parameters": {
            "rope_type": "mrope",
            "rope_theta": d.rope_theta,
            "partial_rotary_factor": d.partial_rotary_factor,
            # InfiniLM requires three MRoPE sections.
            "mrope_section": list(d.mrope_section),
            # Qwen3.5 always uses interleaved MRoPE.
            "mrope_interleaved": True,
        },
    }


def make_root_config(d: "Dims", ggml_types: dict, act_vperm: list = None) -> dict:
    """Build root config with top-level quantization metadata.

    ModelConfig reads quantization_config before merging text_config, so placing
    it inside text_config would silently select NoneQuantization.
    """
    return {
        "model_type": "qwen3_5",
        "torch_dtype": "bfloat16",
        # BF16 logits collapse close top candidates into exact ties.  Keep the
        # dense BF16 head weights, but accumulate/write its output in FP32.
        "lm_head_output_dtype": "float32",
        "tie_word_embeddings": False,
        "text_config": make_text_config(d),
        "quantization_config": {
            "quant_method": "gguf",
            # Nested C++ modules remove this prefix before type-table lookup.
            "key_prefix": PREFIX,
            "ggml_types": ggml_types,
            # Empty means that no runtime activation permutation is required.
            "activation_vperm": act_vperm or [],
        },
    }
