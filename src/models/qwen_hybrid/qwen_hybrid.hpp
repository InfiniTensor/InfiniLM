#pragma once
#include "infinicore_infer/models/qwen_hybrid.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"
#include "../../modules/modules.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

struct DeviceResource;

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_start, cv_done;
    bool proceed = false;
    bool exit_flag = false;
};

struct QwenHybridLayer {
    std::shared_ptr<infinicore::nn::module::RMSNorm> input_norm;
    std::shared_ptr<infinicore::nn::module::MultiHeadAttention> multi_head_attn;
    std::shared_ptr<infinicore::nn::module::RMSNorm> post_attn_norm;
    std::shared_ptr<infinicore::nn::module::MLP> mlp;

    QwenHybridLayer(const QwenHybridMeta *meta, size_t layer, int rank, int nranks, infinicore::weights::Loader &weights_loader);
};

struct QwenHybridDeviceModel {
    std::shared_ptr<Tensor> input_embedding, sin_table, cos_table;
    std::vector<std::shared_ptr<QwenHybridLayer>> layers;
    std::shared_ptr<infinicore::nn::module::RMSNorm> output_norm;
    std::shared_ptr<infinicore::nn::module::Linear> output_embedding;

    QwenHybridDeviceModel(const QwenHybridMeta *meta, int rank, int nranks, infinicore::weights::Loader &weights_loader);

    void infer(InferRequest *req, DeviceResource &rsrc);
};

struct QwenHybridModel {
    QwenHybridMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    infinicore::weights::Loader weights_loader;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    QwenHybridModel(const QwenHybridMeta *meta_, infiniDevice_t device_, const std::vector<int> &dev_ids_);
};