"""Exercise the public Python-to-native graph configuration constructor on CPU."""

import argparse
import json
from pathlib import Path


def main():
    from infinilm import _core as core
    from infinilm.cache.cache import PagedKVCacheConfig
    from infinilm.infer_engine import InferEngine

    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()
    folder = Path(args.scratch)
    folder.mkdir(exist_ok=False)
    config = json.loads(Path(args.model_config).read_text())
    config.update(
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=32,
        torch_dtype="float32",
        dtype="float32",
    )
    (folder / "config.json").write_text(json.dumps(config))
    for profiles in [None, [], [(1, 2048), (8, 0)]]:
        engine = InferEngine(
            str(folder),
            device=core.device("cpu"),
            cache_config=PagedKVCacheConfig(8, 256, 8),
            enable_graph_compiling=True,
            attention_backend="paged-attn",
            graph_capture_profiles=profiles,
            graph_metadata_upload=True,
        )
        assert engine.model_type == "qwen3"
        del engine
    try:
        InferEngine(
            str(folder),
            device=core.device("cpu"),
            cache_config=PagedKVCacheConfig(8, 256, 8),
            graph_capture_profiles=[(9, 0)],
        )
        raise AssertionError("native accepted a batch beyond its cache")
    except ValueError:
        pass
    print(
        "public graph config passed Python/pybind/native CPU construction and rejection"
    )


if __name__ == "__main__":
    main()
