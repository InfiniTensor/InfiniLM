#pragma once
#include "infinicore_infer/models/qwen3_vl.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

struct Qwen3VLDeviceWeight {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, b_attn_q, b_attn_k, b_attn_v, w_ffn_norm;
    std::vector<std::shared_ptr<Tensor>> w_attn_q, w_attn_k, w_attn_v, w_attn_out, w_ffn_gate, w_ffn_up, w_ffn_down;
    std::vector<std::shared_ptr<Tensor>> w_q_norm, w_k_norm;

    // Vision encoder weights
    std::shared_ptr<Tensor> sin_table_v, cos_table_v; // ViT 专用 2D mRoPE 表
    std::vector<std::shared_ptr<Tensor>> b_v_attn_proj, w_v_attn_proj;
    std::vector<std::shared_ptr<Tensor>> b_v_attn_qkv, w_v_attn_qkv;
    std::vector<std::shared_ptr<Tensor>> b_v_mlp_fc1, w_v_mlp_fc1;
    std::vector<std::shared_ptr<Tensor>> b_v_mlp_fc2, w_v_mlp_fc2;
    std::vector<std::shared_ptr<Tensor>> b_v_norm1, w_v_norm1;
    std::vector<std::shared_ptr<Tensor>> b_v_norm2, w_v_norm2;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_ln_q, w_v_merger_ln_q;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_mlp_0, w_v_merger_mlp_0;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_mlp_2, w_v_merger_mlp_2;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_0_ln_q, w_v_merger_list_0_ln_q;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_0_mlp_0, w_v_merger_list_0_mlp_0;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_0_mlp_2, w_v_merger_list_0_mlp_2;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_1_ln_q, w_v_merger_list_1_ln_q;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_1_mlp_0, w_v_merger_list_1_mlp_0;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_1_mlp_2, w_v_merger_list_1_mlp_2;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_2_ln_q, w_v_merger_list_2_ln_q;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_2_mlp_0, w_v_merger_list_2_mlp_0;
    std::vector<std::shared_ptr<Tensor>> b_v_merger_list_2_mlp_2, w_v_merger_list_2_mlp_2;
    std::vector<std::shared_ptr<Tensor>> b_v_patch_embed_proj, w_v_patch_embed_proj;
    std::vector<std::shared_ptr<Tensor>> w_v_pos_embed;
};

class Qwen3VLWeights : public infinicore::weights::Loader {
private:
    std::vector<std::shared_ptr<Qwen3VLDeviceWeight>> _device_weights;

public:
    Qwen3VLWeights(const Qwen3VLMeta *meta,
                   infiniDevice_t device,
                   const std::vector<int> &dev_ids);
    std::vector<std::shared_ptr<Qwen3VLDeviceWeight>> &device_weights() {
        return _device_weights;
    }
};

struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Qwen3VLDeviceWeight> weights;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    const uint32_t *pos_ids; // ViT/vision positions (e.g., [patches,2] or [patches,3])
    uint32_t pos_ids_len;
    const float *pixel_values;
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;

    // LLM 3D mRoPE positions and rope_section
    const uint32_t *llm_pos_ids;  // shape (3, ntok) flattened, or nullptr
    uint32_t llm_pos_ids_len;     // must be 3*ntok if provided
    const uint32_t *rope_section; // shape (3,), or nullptr
    uint32_t rope_section_len;    // must be 3 if provided
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct Qwen3VLModel {
    Qwen3VLMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    Qwen3VLModel(const Qwen3VLMeta *, const ModelWeights *);
};
