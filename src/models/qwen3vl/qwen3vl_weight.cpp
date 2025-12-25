#include "qwen3vl_impl.hpp"

#include <cmath>

inline std::shared_ptr<Tensor> getInEmbd(
    const Qwen3vlMeta *meta) {
    auto shape = std::vector<size_t>({meta->text_meta.vocab_size, meta->text_meta.hidden_size});
    return Tensor::weight(nullptr, meta->dtype, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    const Qwen3vlMeta *meta) {
    auto shape = std::vector<size_t>({meta->text_meta.hidden_size});
    return Tensor::weight(nullptr, meta->dtype, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    const Qwen3vlMeta *meta) {

    auto shape = std::vector<size_t>({meta->text_meta.vocab_size, meta->text_meta.hidden_size});
    return Tensor::weight(nullptr, meta->dtype, shape)
        ->permute({1, 0});
}

inline void getLayerWeight(
    const Qwen3vlMeta *meta,LayerWeight& layer, int ndev) {
    auto nkvh = meta->text_meta.num_key_value_heads;
    auto nh = meta->text_meta.num_attention_heads;
    auto dh = meta->text_meta.head_dim;
    auto d = meta->text_meta.hidden_size;
    auto di = meta->text_meta.intermediate_size;

    auto dh_shape = std::vector<size_t>({meta->text_meta.hidden_size});
    layer.attn_norm = Tensor::weight(nullptr, meta->dtype, dh_shape);
    auto qk_norm_shape = std::vector<size_t>({meta->text_meta.head_dim});
    layer.attn_q_norm = Tensor::weight(nullptr, meta->dtype, qk_norm_shape);
    layer.attn_k_norm = Tensor::weight(nullptr, meta->dtype, qk_norm_shape);
    auto qkv_proj_shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
    layer.attn_qkv_proj = Tensor::weight(nullptr, meta->dtype, qkv_proj_shape);
    auto o_proj_shape = std::vector<size_t>({d, nh / ndev * dh});
    layer.attn_o_proj = Tensor::weight(nullptr, meta->dtype, o_proj_shape);
    
    layer.mlp_norm = Tensor::weight(nullptr, meta->dtype, dh_shape);
    auto up_shape = std::vector<size_t>({2 * di / ndev, d});
    layer.mlp_gate_up = Tensor::weight(nullptr, meta->dtype, up_shape);
    auto down_shape = std::vector<size_t>({d, di / ndev});
    layer.mlp_down = Tensor::weight(nullptr, meta->dtype, down_shape);
}


inline void getVisualWeight(
    const Qwen3vlMeta *meta, std::shared_ptr<VisualEncoderWeight> w_vis) {
    Qwen3vlVisMeta vis_meta = meta->vis_meta;
    auto patch_embed_shape = std::vector<size_t>({vis_meta.hidden_size , vis_meta.in_channels, vis_meta.temporal_patch_size, vis_meta.patch_size, vis_meta.patch_size});
    w_vis->patch_embed_weight = Tensor::weight(nullptr, meta->dtype, patch_embed_shape);
    w_vis->patch_embed_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
    w_vis->pos_embed_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.num_position_embeddings, vis_meta.hidden_size});
    w_vis->merger = std::make_shared<MergerWeight>();
    w_vis->merger->linear_fc1_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size, vis_meta.intermediate_size});
    w_vis->merger->linear_fc2_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.out_hidden_size, vis_meta.intermediate_size});
    w_vis->merger->linear_fc1_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size});
    w_vis->merger->linear_fc2_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.out_hidden_size});
    w_vis->merger->norm_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
    w_vis->merger->norm_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
    w_vis->blocks = std::vector<VisBlockWeight>(vis_meta.depth);
    for (size_t i = 0; i < vis_meta.depth; i++) { 
        w_vis->blocks[i].attn_proj_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size,vis_meta.hidden_size});
        w_vis->blocks[i].attn_proj_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
        w_vis->blocks[i].attn_qkv_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.in_channels*vis_meta.hidden_size,vis_meta.hidden_size});
        w_vis->blocks[i].attn_qkv_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.in_channels*vis_meta.hidden_size});
        w_vis->blocks[i].mlp_linear_fc1_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size, vis_meta.hidden_size});
        w_vis->blocks[i].mlp_linear_fc1_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size});
        w_vis->blocks[i].mlp_linear_fc2_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size, vis_meta.intermediate_size});
        w_vis->blocks[i].mlp_linear_fc2_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
        w_vis->blocks[i].norm1_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
        w_vis->blocks[i].norm1_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
        w_vis->blocks[i].norm2_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
        w_vis->blocks[i].norm2_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.hidden_size});
    }
    w_vis->deepstack_mergers = std::vector<DeepstackMergerWeight>(3);
    for (size_t i = 0; i < 3; i++){
        w_vis->deepstack_mergers[i].linear_fc1_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size,vis_meta.intermediate_size});
        w_vis->deepstack_mergers[i].linear_fc2_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.out_hidden_size,vis_meta.intermediate_size});
        w_vis->deepstack_mergers[i].linear_fc1_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size});
        w_vis->deepstack_mergers[i].linear_fc2_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.out_hidden_size});
        w_vis->deepstack_mergers[i].norm_weight = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size});
        w_vis->deepstack_mergers[i].norm_bias = Tensor::weight(nullptr, meta->dtype, {vis_meta.intermediate_size});
    }
    
}


inline std::shared_ptr<Tensor> getSinTable(const Qwen3vlMeta *meta) {
    auto half_dh = meta->text_meta.head_dim / 2;
    auto unit = dsize(meta->dtype);
    void *table = std::malloc(meta->text_meta.max_tokens * half_dh * unit);

    for (size_t i = 0; i < meta->text_meta.max_tokens; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->text_meta.rope_theta, static_cast<float>(j) / half_dh));
            if (meta->dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->text_meta.max_tokens, half_dh});
    auto tensor = Tensor::weight(table, meta->dtype, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(const Qwen3vlMeta *meta) {
    auto half_dh = meta->text_meta.head_dim / 2;
    auto unit = dsize(meta->dtype);
    void *table = std::malloc(meta->text_meta.max_tokens * half_dh * unit);

    for (size_t i = 0; i < meta->text_meta.max_tokens; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->text_meta.rope_theta, static_cast<float>(j) / half_dh));
            if (meta->dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->text_meta.max_tokens, half_dh});
    auto tensor = Tensor::weight(table, meta->dtype, shape);
    std::free(table);
    return tensor;
}

Qwen3vlWeights::Qwen3vlWeights(
    const Qwen3vlMeta *_meta, infiniDevice_t device, int ndev, const int *dev_ids, bool _transpose_weight) {
    meta = _meta;
    transpose_weight = _transpose_weight;
    device_weights = std::vector<std::shared_ptr<Qwen3vlDeviceWeights>>(ndev);
    for (int dev = 0; dev < ndev; dev++) {
        int dev_id = dev_ids[dev];
        RUN_INFINI(infinirtSetDevice(device, dev_id));
        device_weights[dev] = std::make_shared<Qwen3vlDeviceWeights>();
        device_weights[dev]->device = device;
        device_weights[dev]->dev_id = dev_id;
        RUN_INFINI(infinirtStreamCreate(&device_weights[dev]->load_stream));

        device_weights[dev]->w_lang = std::make_shared<LanguageModelWeight>();
        device_weights[dev]->w_vis = std::make_shared<VisualEncoderWeight>();

        device_weights[dev]->w_lang->in_embd = getInEmbd(meta);
        device_weights[dev]->w_lang->out_norm = getOutNorm(meta);
        device_weights[dev]->w_lang->out_embd = getOutEmbd(meta);
        device_weights[dev]->sin_table = getSinTable(meta);
        device_weights[dev]->cos_table = getCosTable(meta);

        device_weights[dev]->w_lang->layers = std::vector<LayerWeight>(meta->text_meta.num_hidden_layers);

        for (size_t layer = 0; layer < meta->text_meta.num_hidden_layers; layer++) {
            getLayerWeight(meta, device_weights[dev]->w_lang->layers[layer], ndev);
        }

        getVisualWeight(meta, device_weights[dev]->w_vis);

    }
}

//--- Lang Global
void load_input_embd(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading input embedding from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->in_embd->load(cpu_ptr, weight->load_stream);
    }
}

void load_output_norm(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading output norm from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->out_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_output_embd(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading output embedding from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->out_embd->load(cpu_ptr, weight->load_stream);
        if(weights->transpose_weight) {
            weight->w_lang->out_embd->permute({1,0}); //[d,voc]
        }
    }
}

// --- Attention
void load_attn_norm(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention norm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].attn_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_q_norm(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention q_norm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].attn_q_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_qkv_proj(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention q_proj " << layer << " from " << cpu_ptr << std::endl;
    int ndev = int(weights->device_weights.size());
    auto nkvh = weights->meta->text_meta.num_key_value_heads;
    auto nh = weights->meta->text_meta.num_attention_heads;
    auto dh = weights->meta->text_meta.head_dim;
    auto d = weights->meta->text_meta.hidden_size;
    //[ndev,nh+2*nkvh,dh,d]
    for (int idev = 0; idev < ndev; idev++) {
        auto weight = weights->device_weights[idev];
        size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(weights->meta->dtype);
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].attn_qkv_proj->load((char *)cpu_ptr + offset, weight->load_stream);
        if(weights->transpose_weight) {
            weight->w_lang->layers[layer].attn_qkv_proj = 
                weight->w_lang->layers[layer].attn_qkv_proj->permute({1,0}); //[d, (nh+2*nkvh)*dh]
        }
    }
}

void load_attn_k_norm(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention k_norm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].attn_k_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_o_proj(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading attention o_proj " << layer << " from " << cpu_ptr << std::endl;
    int ndev = int(weights->device_weights.size());
    auto nh = weights->meta->text_meta.num_attention_heads;
    auto dh = weights->meta->text_meta.head_dim;
    auto d = weights->meta->text_meta.hidden_size;
    // [ndev, d, nh // ndev * dh]
    for (int idev = 0; idev < ndev; idev++) {
        auto weight = weights->device_weights[idev];
        size_t offset = idev * d * (nh / ndev * dh) * dsize(weights->meta->dtype);
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].attn_o_proj->load((char *)cpu_ptr + offset, weight->load_stream);
        if(weights->transpose_weight) {
            weight->w_lang->layers[layer].attn_o_proj = 
                weight->w_lang->layers[layer].attn_o_proj->permute({1,0}); //[nh/ndev*dh, d]
        }
    }
}

// --- MLP
void load_mlp_norm(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading mlp norm " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].mlp_norm->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_gate_up(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading mlp gate " << layer << " from " << cpu_ptr << std::endl;
    int ndev = int(weights->device_weights.size());
    auto di = weights->meta->text_meta.head_dim;
    auto d = weights->meta->text_meta.hidden_size;
    // [ndev, 2*di // ndev, d]
    for (int idev = 0; idev < ndev; idev++) {
        auto weight = weights->device_weights[idev];
        size_t offset = idev * (2 * di / ndev) * d * dsize(weights->meta->dtype);
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].mlp_gate_up->load((char *)cpu_ptr + offset, weight->load_stream);
        if(weights->transpose_weight) {
            weight->w_lang->layers[layer].mlp_gate_up = 
                weight->w_lang->layers[layer].mlp_gate_up->permute({1,0}); //[d, 2*di/ndev]
        }
    }
}

void load_mlp_down(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading mlp down " << layer << " from " << cpu_ptr << std::endl;
    int ndev = int(weights->device_weights.size());
    auto di = weights->meta->text_meta.head_dim;
    auto d = weights->meta->text_meta.hidden_size;
    //[ndev, d, di // ndev]
    for (int idev = 0; idev < ndev; idev++) {
        auto weight = weights->device_weights[idev];
        size_t offset = idev * d * (di / ndev) * dsize(weights->meta->dtype);
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_lang->layers[layer].mlp_down->load((char *)cpu_ptr + offset, weight->load_stream);
        if(weights->transpose_weight) {
            weight->w_lang->layers[layer].mlp_down = 
                weight->w_lang->layers[layer].mlp_down->permute({1,0}); //[di/ndev, d]
        } 
    }
}

// --- Vision weights
void load_patch_embed_weight(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading patch embed weight from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->patch_embed_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_patch_embed_bias(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading patch embed bias from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->patch_embed_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_pos_embed_weight(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading pos embed weight from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->pos_embed_weight->load(cpu_ptr, weight->load_stream);
    }
}

// Vision block attention
void load_attn_proj_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision attn proj weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].attn_proj_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_proj_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision attn proj bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].attn_proj_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_qkv_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision attn qkv weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].attn_qkv_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_attn_qkv_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision attn qkv bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].attn_qkv_bias->load(cpu_ptr, weight->load_stream);
    }
}

// Vision block mlp
void load_mlp_linear_fc1_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision mlp fc1 weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].mlp_linear_fc1_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_linear_fc1_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision mlp fc1 bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].mlp_linear_fc1_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_linear_fc2_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision mlp fc2 weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].mlp_linear_fc2_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_mlp_linear_fc2_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision mlp fc2 bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].mlp_linear_fc2_bias->load(cpu_ptr, weight->load_stream);
    }
}

// Vision block norm
void load_norm1_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision norm1 weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].norm1_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_norm1_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision norm1 bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].norm1_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_norm2_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision norm2 weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].norm2_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_norm2_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading vision norm2 bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->blocks[layer].norm2_bias->load(cpu_ptr, weight->load_stream);
    }
}

// Deepstack merger
void load_deepstack_merger_linear_fc1_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading deepstack merger fc1 weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->deepstack_mergers[layer].linear_fc1_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_deepstack_merger_linear_fc1_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading deepstack merger fc1 bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->deepstack_mergers[layer].linear_fc1_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_deepstack_merger_linear_fc2_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading deepstack merger fc2 weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->deepstack_mergers[layer].linear_fc2_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_deepstack_merger_linear_fc2_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading deepstack merger fc2 bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->deepstack_mergers[layer].linear_fc2_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_deepstack_merger_norm_weight(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading deepstack merger norm weight " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->deepstack_mergers[layer].norm_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_deepstack_merger_norm_bias(Qwen3vlWeights *weights, void *cpu_ptr, size_t layer) {
    std::cout << "Loading deepstack merger norm bias " << layer << " from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->deepstack_mergers[layer].norm_bias->load(cpu_ptr, weight->load_stream);
    }
}

// Merger
void load_merger_linear_fc1_weight(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading merger fc1 weight from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->merger->linear_fc1_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_merger_linear_fc1_bias(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading merger fc1 bias from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->merger->linear_fc1_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_merger_linear_fc2_weight(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading merger fc2 weight from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->merger->linear_fc2_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_merger_linear_fc2_bias(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading merger fc2 bias from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->merger->linear_fc2_bias->load(cpu_ptr, weight->load_stream);
    }
}

void load_merger_norm_weight(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading merger norm weight from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->merger->norm_weight->load(cpu_ptr, weight->load_stream);
    }
}

void load_merger_norm_bias(Qwen3vlWeights *weights, void *cpu_ptr) {
    std::cout << "Loading merger norm bias from " << cpu_ptr << std::endl;
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_vis->merger->norm_bias->load(cpu_ptr, weight->load_stream);
    }
}


static Qwen3vlWeightLoader weight_loader = {
    // Language model loaders
    .lang_loader = {
        .load_input_embd = load_input_embd,
        .load_output_norm = load_output_norm,
        .load_output_embd = load_output_embd,
        .load_attn_norm = load_attn_norm,
        .load_attn_q_norm = load_attn_q_norm,
        .load_attn_k_norm = load_attn_k_norm,
        .load_attn_qkv_proj = load_attn_qkv_proj,
        .load_attn_o_proj = load_attn_o_proj,
        .load_mlp_norm = load_mlp_norm,
        .load_mlp_gate_up = load_mlp_gate_up,
        .load_mlp_down = load_mlp_down,
    },
    // Vision model loaders
    .vis_loader = {
        .load_patch_embed_weight = load_patch_embed_weight,
        .load_patch_embed_bias = load_patch_embed_bias,
        .load_pos_embed_weight = load_pos_embed_weight,
        .load_attn_proj_weight = load_attn_proj_weight,
        .load_attn_proj_bias = load_attn_proj_bias,
        .load_attn_qkv_weight = load_attn_qkv_weight,
        .load_attn_qkv_bias = load_attn_qkv_bias,
        .load_mlp_linear_fc1_weight = load_mlp_linear_fc1_weight,
        .load_mlp_linear_fc1_bias = load_mlp_linear_fc1_bias,
        .load_mlp_linear_fc2_weight = load_mlp_linear_fc2_weight,
        .load_mlp_linear_fc2_bias = load_mlp_linear_fc2_bias,
        .load_norm1_weight = load_norm1_weight,
        .load_norm1_bias = load_norm1_bias,
        .load_norm2_weight = load_norm2_weight,
        .load_norm2_bias = load_norm2_bias,
        .load_deepstack_merger_linear_fc1_weight = load_deepstack_merger_linear_fc1_weight,
        .load_deepstack_merger_linear_fc1_bias = load_deepstack_merger_linear_fc1_bias,
        .load_deepstack_merger_linear_fc2_weight = load_deepstack_merger_linear_fc2_weight,
        .load_deepstack_merger_linear_fc2_bias = load_deepstack_merger_linear_fc2_bias,
        .load_deepstack_merger_norm_weight = load_deepstack_merger_norm_weight,
        .load_deepstack_merger_norm_bias = load_deepstack_merger_norm_bias,
        .load_merger_linear_fc1_weight = load_merger_linear_fc1_weight,
        .load_merger_linear_fc1_bias = load_merger_linear_fc1_bias,
        .load_merger_linear_fc2_weight = load_merger_linear_fc2_weight,
        .load_merger_linear_fc2_bias = load_merger_linear_fc2_bias,
        .load_merger_norm_weight = load_merger_norm_weight,
        .load_merger_norm_bias = load_merger_norm_bias,
    }
};

__C Qwen3vlWeights *
createQwen3vlWeights(const Qwen3vlMeta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids,
                        bool transpose_weight) {
    auto weights = new Qwen3vlWeights(meta, device, ndev, dev_ids, transpose_weight);
    return weights;
};

__C Qwen3vlWeightLoader *
createQwen3vlWeightLoader() {
    return &weight_loader;
}