#ifndef __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__
#define __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__

#include "recurrent_delta_rule_common.cuh"

template <typename Tgate, typename Tcompute, size_t Dk>
struct ScalarGatedDeltaRulePolicy {
    const Tgate *g;
    const Tgate *beta;
    ptrdiff_t g_s0;
    ptrdiff_t g_s1;
    ptrdiff_t g_s2;
    ptrdiff_t beta_s0;
    ptrdiff_t beta_s1;
    ptrdiff_t beta_s2;

    __device__ void prepare(int token_batch,
                            int64_t token_idx,
                            int,
                            int value_head_idx,
                            Tcompute *decay,
                            Tcompute *beta_out) const {
        if (threadIdx.x == 0) {
            const ptrdiff_t gate_offset = static_cast<ptrdiff_t>(token_batch) * g_s0 + static_cast<ptrdiff_t>(token_idx) * g_s1 + static_cast<ptrdiff_t>(value_head_idx) * g_s2;
            const ptrdiff_t beta_offset = static_cast<ptrdiff_t>(token_batch) * beta_s0 + static_cast<ptrdiff_t>(token_idx) * beta_s1 + static_cast<ptrdiff_t>(value_head_idx) * beta_s2;
            decay[0] = expf(static_cast<Tcompute>(
                op::recurrent_gated_delta_rule::cuda::loadAsFloat(g, gate_offset)));
            beta_out[0] = static_cast<Tcompute>(
                op::recurrent_gated_delta_rule::cuda::loadAsFloat(beta, beta_offset));
        }
        __syncthreads();
        const Tcompute scalar_decay = decay[0];
        for (int key_dim_idx = threadIdx.x; key_dim_idx < static_cast<int>(Dk);
             key_dim_idx += blockDim.x) {
            decay[key_dim_idx] = scalar_decay;
        }
    }
};

template <typename Tdata,
          typename Tgate,
          typename Tcompute,
          size_t Dk,
          size_t Dv,
          size_t WARPS_PER_BLOCK>
__device__ void recurrentGatedDeltaRuleIndexedPoolWarpKernel(
    Tdata *out,
    Tdata *initial_state,
    Tdata *final_state,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const Tgate *g,
    const Tgate *beta,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
    bool indexed_state_pool,
    size_t pool_size,
    size_t num_key_heads,
    size_t value_heads_per_key_head,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t initial_s0,
    ptrdiff_t initial_s1,
    ptrdiff_t initial_s2,
    ptrdiff_t initial_s3,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t final_s3,
    ptrdiff_t q_s0,
    ptrdiff_t q_s1,
    ptrdiff_t q_s2,
    ptrdiff_t k_s0,
    ptrdiff_t k_s1,
    ptrdiff_t k_s2,
    ptrdiff_t v_s0,
    ptrdiff_t v_s1,
    ptrdiff_t v_s2,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2,
    Tcompute *shared) {
    const ScalarGatedDeltaRulePolicy<Tgate, Tcompute, Dk> gate_policy{
        g,
        beta,
        g_s0,
        g_s1,
        g_s2,
        beta_s0,
        beta_s1,
        beta_s2,
    };
    op::recurrent_gated_delta_rule::cuda::recurrentDeltaRuleWarpSequence<
        Tdata,
        Tcompute,
        Dk,
        Dv,
        WARPS_PER_BLOCK>(
        out,
        initial_state,
        final_state,
        q,
        k,
        v,
        nullptr,
        initial_state_indices,
        final_state_indices,
        false,
        initial_state_indices_i64,
        final_state_indices_i64,
        use_qk_l2norm,
        false,
        indexed_state_pool,
        1,
        pool_size,
        num_key_heads,
        value_heads_per_key_head,
        rsqrtf(static_cast<Tcompute>(Dk)),
        out_s0,
        out_s1,
        out_s2,
        initial_s0,
        initial_s1,
        initial_s2,
        initial_s3,
        final_s0,
        final_s1,
        final_s2,
        final_s3,
        q_s0,
        q_s1,
        q_s2,
        k_s0,
        k_s1,
        k_s2,
        v_s0,
        v_s1,
        v_s2,
        gate_policy,
        shared);
}

#endif // __RECURRENT_GATED_DELTA_RULE_KERNEL_CUH__
