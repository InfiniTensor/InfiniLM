#include "../../../tensor.hpp"
#include "../../../utils.hpp"
#include "../../inference_context.hpp"
#include "../qwen_device_resource.hpp"
#include "../../../../include/infinicore_infer/cache.h"
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
