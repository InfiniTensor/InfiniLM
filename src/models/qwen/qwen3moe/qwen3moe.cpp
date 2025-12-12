#include "../../../tensor.hpp"
#include "../../../utils.hpp"
#include "../../inference_context.hpp"
#include "../qwen_device_resource.hpp"
#include "../qwen_kv_cache.hpp"
#include "../qwen_model.hpp"
#include "../qwen_weight.hpp"
#include "infinicore_infer.h"
#include <random>
#include <thread>
#include <vector>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           Model API            ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

__C Qwen3MoE::Model *Qwen3MoEcreateModel(const Qwen3MoE::Meta *meta,
                                         const Qwen3MoE::Weights *weight,
                                         infiniDevice_t device,
                                         int ndev,
                                         const int *dev_ids) {
    return createModel<Qwen3MoE::Model, Qwen3MoE::Meta, Qwen3MoE::Weights>(meta, weight, device, ndev, dev_ids);
}

/// @brief 销毁模型
__C void Qwen3MoEdestroyModel(struct Qwen3MoE::Model *model) {
    destroyModel<Qwen3MoE::Model>(model);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           KVCache API            ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @brief 创建 KV Cache
__C KVCache *Qwen3MoEcreateKVCache(const Qwen3MoE::Model *model) {
    return createKVCache<Qwen3MoE::Model>(model);
}

/// @brief 复制 KV Cache
__C KVCache *
Qwen3MoEduplicateKVCache(const Qwen3MoE::Model *model,
                         const KVCache *kv_cache, uint32_t seq_len) {
    return duplicateKVCache<Qwen3MoE::Model>(model, kv_cache, seq_len);
}

/// @brief 销毁 KV Cache
__C void Qwen3MoEdropKVCache(const Qwen3MoE::Model *model, KVCache *kv_cache) {
    dropKVCache<Qwen3MoE::Model>(model, kv_cache);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           infer API            //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
__C void Qwen3MoEinferBatch(struct Qwen3MoE::Model *model,
                            const uint32_t *tokens, uint32_t ntok,
                            const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                            KVCache **kv_caches,
                            const float *temperature, const uint32_t *topk, const float *topp,
                            uint32_t *output) {
    inferBatch<Qwen3MoE::Model>(model, tokens, ntok,
                                req_lens, nreq, req_pos,
                                kv_caches, temperature, topk, topp, output);
}

__C void Qwen3MoEforwardBatch(Qwen3MoE::Model *model,
                              const uint32_t *tokens, uint32_t ntok,
                              const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                              KVCache **kv_caches,
                              void *logits) {

    forwardBatch<Qwen3MoE::Model>(model,
                                  tokens, ntok,
                                  req_lens, nreq, req_pos,
                                  kv_caches,
                                  logits);
}