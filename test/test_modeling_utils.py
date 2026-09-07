from infinilm.modeling_utils import _is_glm_base_inference_unused_weight


def test_glm_dsa_indexer_weights_are_loaded():
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
    }

    assert not _is_glm_base_inference_unused_weight(
        "model.layers.34.self_attn.indexer.wk.weight", config
    )
    assert not _is_glm_base_inference_unused_weight(
        "model.layers.77.self_attn.indexer.wq_b.weight_scale", config
    )
    assert _is_glm_base_inference_unused_weight(
        "model.layers.78.self_attn.q_proj.weight", config
    )
    assert not _is_glm_base_inference_unused_weight(
        "model.layers.78.self_attn.q_proj.weight", {"model_type": "llama"}
    )
