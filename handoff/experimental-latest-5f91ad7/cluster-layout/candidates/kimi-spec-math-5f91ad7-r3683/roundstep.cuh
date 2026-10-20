template <typename Tdata, typename Tgate>
__global__ void kimiDeltaAttentionRoundStepR3655(
    Tdata *out,
    Tdata *initial_state,
    Tdata *final_state,
    Tdata *indexed_state_output,
    const Tdata *q,
    const Tdata *k,
    const Tdata *v,
    const Tgate *g,
    const Tgate *beta,
    const float *A_log,
    const float *dt_bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool cu_seqlens_i64,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool use_qk_l2norm,
    bool has_cu_seqlens,
    bool indexed_state_pool,
    size_t T,
    size_t D,
    size_t pool_size,
    float scale,
    float lower_bound,
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
    ptrdiff_t A_log_s0,
    ptrdiff_t dt_bias_s0) {

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int value_dim_idx = blockIdx.z;

    extern __shared__ float shared[];
    float *state = shared;
    float *q_vec = state + D;
    float *k_vec = q_vec + D;
    float *scratch = k_vec + D;

    int64_t token_begin = 0;
    int64_t token_end = static_cast<int64_t>(T);
    if (has_cu_seqlens) {
        token_begin = kdaLoadOptionalIndex(cu_seqlens, cu_seqlens_i64, batch_idx, 0);
        token_end = kdaLoadOptionalIndex(cu_seqlens, cu_seqlens_i64, batch_idx + 1, 0);
        if (token_begin < 0 || token_end < token_begin || token_end > static_cast<int64_t>(T)) {
            return;
        }
    }

    int64_t read_slot = batch_idx;
    int64_t write_slot = batch_idx;
    if (indexed_state_pool) {
        read_slot = kdaLoadOptionalIndex(initial_state_indices, initial_state_indices_i64, batch_idx, batch_idx);
        write_slot = final_state_indices == nullptr
                       ? static_cast<int64_t>(batch_idx)
                       : kdaLoadOptionalIndex(final_state_indices, final_state_indices_i64, batch_idx, batch_idx);
        if (read_slot < 0 || write_slot < 0 || read_slot >= static_cast<int64_t>(pool_size) || write_slot >= static_cast<int64_t>(pool_size)) {
            return;
        }
    }

    const ptrdiff_t initial_base = static_cast<ptrdiff_t>(read_slot) * initial_s0
                                 + static_cast<ptrdiff_t>(head_idx) * initial_s1
                                 + static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;

    Tdata *final_state_target = final_state_indices == nullptr ? final_state : indexed_state_output;
    const ptrdiff_t final_base = final_state_indices == nullptr
                                   ? static_cast<ptrdiff_t>(batch_idx) * final_s0
                                         + static_cast<ptrdiff_t>(head_idx) * final_s1
                                         + static_cast<ptrdiff_t>(value_dim_idx) * final_s2
                                   : static_cast<ptrdiff_t>(write_slot) * initial_s0
                                         + static_cast<ptrdiff_t>(head_idx) * initial_s1
                                         + static_cast<ptrdiff_t>(value_dim_idx) * initial_s2;
    const ptrdiff_t final_k_stride = final_state_indices == nullptr ? final_s3 : initial_s3;

    for (int dk = threadIdx.x; dk < static_cast<int>(D); dk += blockDim.x) {
        state[dk] = kdaLoadAsFloat(initial_state, initial_base + static_cast<ptrdiff_t>(dk) * initial_s3);
    }
    __syncthreads();

    const int token_batch = has_cu_seqlens ? 0 : batch_idx;
    const float a_log_exp = expf(A_log[static_cast<ptrdiff_t>(head_idx) * A_log_s0]);

    for (int64_t token_idx = token_begin; token_idx < token_end; ++token_idx) {
        const ptrdiff_t q_base = static_cast<ptrdiff_t>(token_batch) * q_s0 + static_cast<ptrdiff_t>(token_idx) * q_s1 + static_cast<ptrdiff_t>(head_idx) * q_s2;
        const ptrdiff_t k_base = static_cast<ptrdiff_t>(token_batch) * k_s0 + static_cast<ptrdiff_t>(token_idx) * k_s1 + static_cast<ptrdiff_t>(head_idx) * k_s2;
        const ptrdiff_t g_base = static_cast<ptrdiff_t>(token_batch) * g_s0 + static_cast<ptrdiff_t>(token_idx) * g_s1 + static_cast<ptrdiff_t>(head_idx) * g_s2;

        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int dk = threadIdx.x; dk < static_cast<int>(D); dk += blockDim.x) {
            float q_raw = kdaLoadAsFloat(q, q_base + dk);
            float k_raw = kdaLoadAsFloat(k, k_base + dk);
            q_vec[dk] = q_raw;
            k_vec[dk] = k_raw;
            q_sum += q_raw * q_raw;
            k_sum += k_raw * k_raw;
        }
        q_sum = kdaBlockReduceSum(q_sum, scratch);
        k_sum = kdaBlockReduceSum(k_sum, scratch);

        const float q_scale = use_qk_l2norm ? rsqrtf(q_sum + 1e-6f) * scale : scale;
        const float k_scale = use_qk_l2norm ? rsqrtf(k_sum + 1e-6f) : 1.0f;
        for (int dk = threadIdx.x; dk < static_cast<int>(D); dk += blockDim.x) {
            q_vec[dk] *= q_scale;
            k_vec[dk] *= k_scale;
        }
        __syncthreads();

        float kv_mem = 0.0f;
        float hq_mem = 0.0f;
        float kq_mem = 0.0f;
        for (int dk = threadIdx.x; dk < static_cast<int>(D); dk += blockDim.x) {
            float raw_gate = kdaLoadAsFloat(g, g_base + dk) + dt_bias[static_cast<ptrdiff_t>(head_idx) * dt_bias_s0 + dk];
            float decay = expf(lower_bound * kdaSigmoid(a_log_exp * raw_gate));
            kv_mem += state[dk] * decay * k_vec[dk];
            hq_mem += state[dk] * decay * q_vec[dk];
            kq_mem += k_vec[dk] * q_vec[dk];
        }
        kv_mem = kdaBlockReduceSum(kv_mem, scratch);
        hq_mem = kdaBlockReduceSum(hq_mem, scratch);
        kq_mem = kdaBlockReduceSum(kq_mem, scratch);

        const ptrdiff_t beta_offset = static_cast<ptrdiff_t>(token_batch) * beta_s0 + static_cast<ptrdiff_t>(token_idx) * beta_s1 + static_cast<ptrdiff_t>(head_idx) * beta_s2;
        const float beta_t = kdaSigmoid(kdaLoadAsFloat(beta, beta_offset));
        const ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0 + static_cast<ptrdiff_t>(token_idx) * v_s1 + static_cast<ptrdiff_t>(head_idx) * v_s2;
        const float v_t = kdaLoadAsFloat(v, v_base + value_dim_idx);
        const float delta = (v_t - kv_mem) * beta_t;

        if (threadIdx.x == 0) {
            const ptrdiff_t out_base = static_cast<ptrdiff_t>(token_batch) * out_s0 + static_cast<ptrdiff_t>(token_idx) * out_s1 + static_cast<ptrdiff_t>(head_idx) * out_s2;
            out[out_base + value_dim_idx] = static_cast<Tdata>(hq_mem + delta * kq_mem);
        }

        for (int dk = threadIdx.x; dk < static_cast<int>(D); dk += blockDim.x) {
            float raw_gate = kdaLoadAsFloat(g, g_base + dk) + dt_bias[static_cast<ptrdiff_t>(head_idx) * dt_bias_s0 + dk];
            float decay = expf(lower_bound * kdaSigmoid(a_log_exp * raw_gate));
            state[dk] = static_cast<float>(static_cast<Tdata>(state[dk] * decay + k_vec[dk] * delta));
        }
        __syncthreads();
    }

    for (int dk = threadIdx.x; dk < static_cast<int>(D); dk += blockDim.x) {
        final_state_target[final_base + static_cast<ptrdiff_t>(dk) * final_k_stride] = static_cast<Tdata>(state[dk]);
    }
}

