#ifndef MODEL_LLADAMOE_H
#define MODEL_LLADAMOE_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stddef.h>
#include <stdint.h>

struct LLaDAModel;

typedef struct {
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh;
    size_t di_dense, di_expert;
    size_t dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
    size_t num_experts;
} LLaDAMeta;


typedef struct{
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    // 0 if linear weights are passed as W, any other value if passed as W^T (default format in pytorch)
    int transpose_linear_weights;
    // [dvoc, d]
    const void *input_embd;
    // [d]
    const void *output_norm;
    // [dvoc, d]
    const void *output_embd;
    // nlayer * [d]
    const void *const *attn_norm;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh, d] each devide deal with equal head
    const void *const *attn_qkv;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh]
    const void *const *attn_qkv_b;
    // nlayer * [dh]
    const void *const *attn_q_norm;
    // nlayer * [dh]
    const void *const *attn_k_norm;
    // nlayer * [ndev, d, nkvh / ndev * dh]
    const void *const *attn_o;
    // nlayer * [d]
    const void *const *ffn_norm;
    // nlayer * [ndev, 2 * di / ndev, d]
    const void *const *ffn_gate_up;
    // nlayer * [ndev, d, di / ndev]
    const void *const *ffn_down;
} LLaDAWeights;

// 改进后的权重加载器结构体
typedef struct {

} LLaDAWeightLoader; // # TODO: 



__C LLaDAModel *
createLLaDAModel(   const LLaDAMeta *,
                    const LLaDAWeights *,
                    infiniDevice_t device,
                    int ndev,
                    const int *dev_ids);

__C void
destroyLLaDAModel();

__C void
inferBatchLLaDA();

__C void
forwardBatchLLaDA();

#endif