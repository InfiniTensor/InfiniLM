#!/usr/bin/env python3
"""探针：用 mini qwen3_5 config 构造 InferEngine，导出 C++ 侧权威参数键与 shape。"""

import json
import os
import sys
import tempfile

import infinicore
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine

CFG = {
    "model_type": "qwen3_5",
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 1024,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "vocab_size": 1024,
        "full_attention_interval": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 6,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "attention_bias": False,
        "rope_parameters": {
            "rope_type": "mrope",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    },
}


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    d = infinicore.device(dev, 0)
    tmp = tempfile.mkdtemp(prefix="mini_qwen35_")
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)
    eng = InferEngine(
        model_path=tmp,
        device=d,
        distributed_config=DistConfig(1),
        cache_config=StaticKVCacheConfig(max_batch_size=1, max_cache_len=16),
    )
    keys = list(eng.state_dict_keyname())
    sd = eng.state_dict()[0]
    print("# device=%s  参数总数=%d" % (dev, len(keys)))
    for k in sorted(keys):
        t = sd.get(k)
        shape = tuple(t.shape) if t is not None else "<missing>"
        dt = getattr(t, "dtype", "")
        print("%-58s %-22s %s" % (k, str(shape), dt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
