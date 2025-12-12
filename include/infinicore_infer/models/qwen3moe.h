#ifndef _QWEN3MOE_H_
#define _QWEN3MOE_H_

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>
#include <stdint.h>
#include <stdio.h>
namespace Qwen3MoE {
struct Weights;
struct Model;

struct Meta {
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
    //
    size_t _moe_intermediate_size;
    size_t _shared_expert_intermediate_size;
    size_t _num_experts;
    size_t _num_experts_per_tok;
    bool _norm_topk_prob;

public:
    void print_info() const {
        printf("\n");
        printf(" dt_logits : %d\n", dt_logits);
        printf(" nlayer : %ld\n", nlayer);
        printf(" d : %ld\n", d);
        printf(" nh : %ld\n", nh);
        printf(" nkvh : %ld\n", nkvh);
        printf(" dh : %ld\n", dh);
        printf(" di : %ld\n", di);
        printf(" dvoc : %ld\n", dvoc);
        printf(" nkvh : %ld\n", nkvh);

        printf(" epsilon : %f\n", epsilon);
        printf(" theta : %f\n", theta);

        printf(" end_token : %d\n", end_token);

        printf(" _moe_intermediate_size : %ld\n", _moe_intermediate_size);
        printf(" _shared_expert_intermediate_size : %ld\n", _shared_expert_intermediate_size);
        printf(" _num_experts : %ld\n", _num_experts);
        printf(" _num_experts_per_tok : %ld\n", _num_experts_per_tok);
        printf(" _norm_topk_prob : %d\n", _norm_topk_prob);
    }
};

}; // namespace Qwen3MoE

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////           Qwen3 APIs            /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3MoE::Model *
Qwen3MoEcreateModel(const Qwen3MoE::Meta *,
                    const Qwen3MoE::Weights *,
                    infiniDevice_t device,
                    int ndev,
                    const int *dev_ids);

/// @brief 销毁模型
__C __export void
Qwen3MoEdestroyModel(struct Qwen3MoE::Model *);

/// @brief 创建 KV Cache
__C __export struct KVCache *
Qwen3MoEcreateKVCache(const struct Qwen3MoE::Model *);

/// @brief 复制 KV Cache
__C __export struct KVCache *
Qwen3MoEduplicateKVCache(const struct Qwen3MoE::Model *,
                         const struct KVCache *, uint32_t seq_len);

/// @brief 销毁 KV Cache
__C __export void
Qwen3MoEdropKVCache(const struct Qwen3MoE::Model *,
                    struct KVCache *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
Qwen3MoEinferBatch(struct Qwen3MoE::Model *,
                   const uint32_t *tokens, uint32_t ntok,
                   const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                   struct KVCache **kv_caches,
                   const float *temperature, const uint32_t *topk, const float *topp,
                   uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
Qwen3MoEforwardBatch(struct Qwen3MoE::Model *,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct KVCache **kv_caches,
                     void *logits);

#endif
