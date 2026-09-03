#include "minimax_text_01_linear_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <infinicore/ops.hpp>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/broadcast_to.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/ops/float_power.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <infinicore/ops/silu.hpp>
#include <infinicore/ops/var_mean.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::minimax_text_01 {

namespace {

// ALiBi-style slope sequence: for 2^a heads the i-th slope is
// start * ratio^i with start = 2^(-(2^(-(log2(n) - 3)))), ratio = start.
// Non power-of-two counts interpolate with the next power of two.
std::vector<double> get_slopes_power_of_2(size_t n) {
    std::vector<double> slopes(n);
    const double start = std::pow(2.0, -(std::pow(2.0, -(std::log2(static_cast<double>(n)) - 3.0))));
    const double ratio = start;
    for (size_t i = 0; i < n; ++i) {
        slopes[i] = start * std::pow(ratio, static_cast<double>(i));
    }
    return slopes;
}

std::vector<double> get_slopes(size_t n) {
    const double log2n = std::log2(static_cast<double>(n));
    if (std::floor(log2n) == log2n) {
        return get_slopes_power_of_2(n);
    }
    const size_t closest = static_cast<size_t>(std::pow(2.0, std::floor(log2n)));
    const auto first = get_slopes_power_of_2(closest);
    const auto second = get_slopes(2 * closest);
    std::vector<double> slopes = first;
    for (size_t i = 0; slopes.size() < n; ++i) {
        slopes.push_back(second[2 * i]); // take every other slope of the larger set
    }
    return slopes;
}

// Round-to-nearest bit-level float -> bf16 conversion. infinicore's cast_
// requires ATen, which this build does not link, so bf16 scalars/constants
// are converted manually here (matching how the decay is built).
uint16_t float_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

// ------------------------------------------------------------------ //
//  Chunked-prefill decay tables.                                      //
//                                                                     //
//  Prefill of the Lightning recurrence S_t = a_h*S_{t-1} + k_t^T v_t  //
//  (a_h = exp(-slope_h*layer_scale), a per-head scalar) is evaluated  //
//  in blocks of C tokens, exactly as the HF reference does:           //
//    * within a block the pairwise causal decay is a dense table      //
//      D[i][j] = a_h^{i-j} (i>=j, else 0)                             //
//    * the cross-block query/k/block scales use the powers a_h^k      //
//  All powers are static per-head constants, so they are precomputed  //
//  here on the host (in bf16) and shipped to the device once.         //
// ------------------------------------------------------------------ //
struct ChunkDecayTables {
    // a_h^k for k = 0..C, shaped [1, heads, C+1, 1].
    infinicore::Tensor pow;
    // a_h^{pos+1} for pos = 0..C-1, shaped [1, heads, C, 1].
    infinicore::Tensor qdec;
    // a_h^{C-1-pos} for pos = 0..C-1, shaped [1, heads, C, 1]
    // (take the last m entries to get a_h^{m-1-pos} for an m-token block).
    infinicore::Tensor kdec;
    // a_h^{i-j} (i>=j) else 0, shaped [1, heads, C, C].
    infinicore::Tensor diag;
};

ChunkDecayTables build_chunk_decay_tables(size_t chunk_size,
                                          const std::vector<double> &decays,
                                          const infinicore::DataType &dtype,
                                          const infinicore::Device &device) {
    const size_t H = decays.size();
    const size_t C = chunk_size;
    // Powers a_h^k on the host, one row per head (reuse of one std::pow per
    // head is avoided by multiplying up, keeping bf16 rounding deterministic).
    std::vector<std::vector<uint16_t>> a_pow(H, std::vector<uint16_t>(C + 1));
    for (size_t h = 0; h < H; ++h) {
        double v = 1.0;
        for (size_t k = 0; k <= C; ++k) {
            a_pow[h][k] = float_to_bf16(static_cast<float>(v));
            v *= decays[h];
        }
    }
    const auto from_host = [&](std::vector<uint16_t> &buf,
                               const std::vector<size_t> &shape) {
        auto cpu = infinicore::Tensor::from_blob(
            buf.data(), shape, infinicore::DataType::BF16, infinicore::Device::cpu());
        return cpu->to(device);
    };

    ChunkDecayTables out;
    std::vector<uint16_t> buf;
    // pow [1, H, C+1, 1]
    buf.resize(H * (C + 1));
    for (size_t h = 0; h < H; ++h) {
        for (size_t k = 0; k <= C; ++k) {
            buf[h * (C + 1) + k] = a_pow[h][k];
        }
    }
    out.pow = from_host(buf, {1, H, C + 1, 1});
    // qdec [1, H, C, 1]: a^{pos+1}
    buf.resize(H * C);
    for (size_t h = 0; h < H; ++h) {
        for (size_t pos = 0; pos < C; ++pos) {
            buf[h * C + pos] = a_pow[h][pos + 1];
        }
    }
    out.qdec = from_host(buf, {1, H, C, 1});
    // kdec [1, H, C, 1]: a^{C-1-pos}
    for (size_t h = 0; h < H; ++h) {
        for (size_t pos = 0; pos < C; ++pos) {
            buf[h * C + pos] = a_pow[h][C - 1 - pos];
        }
    }
    out.kdec = from_host(buf, {1, H, C, 1});
    // diag [1, H, C, C]: a^{i-j} for i>=j else 0
    buf.resize(H * C * C);
    for (size_t h = 0; h < H; ++h) {
        for (size_t i = 0; i < C; ++i) {
            for (size_t j = 0; j < C; ++j) {
                buf[(h * C + i) * C + j] = (i >= j) ? a_pow[h][i - j] : 0;
            }
        }
    }
    out.diag = from_host(buf, {1, H, C, C});
    return out;
}

} // namespace

MiniMaxText01LinearAttention::MiniMaxText01LinearAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    num_heads_ = model_config->get<size_t>("num_attention_heads");
    head_dim_ = model_config->get<size_t>("head_dim");
    num_hidden_layers_ = model_config->get<size_t>("num_hidden_layers");
    const size_t projection_size = num_heads_ * head_dim_;

    // Weight quantization method (GPTQ / AWQ / MXFP4 / None), sourced from the
    // config's "quantization" field and passed to every linear layer so the
    // Lightning path stays consistent with the full-attention path.
    auto quantization_method = model_config->get_quantization_method();

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();

    // TP-local head count: `QKVParallelLinear` splits heads contiguously, so
    // rank r owns the heads [r*local, (r+1)*local) and the local projection
    // size shrinks accordingly.
    tp_size_ = rank_info.tp_size;
    tp_rank_ = rank_info.tp_rank;
    communicator_ = rank_info.comm;
    if (tp_size_ == 0 || num_heads_ % tp_size_ != 0) {
        throw std::runtime_error(
            "MiniMaxText01LinearAttention: num_heads_ (" + std::to_string(num_heads_)
            + ") must be divisible by tp_size (" + std::to_string(tp_size_) + ")");
    }
    local_num_heads_ = num_heads_ / tp_size_;
    local_projection_size_ = local_num_heads_ * head_dim_;

    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_,
        head_dim_,
        num_heads_,
        num_heads_,
        "q_proj",
        "k_proj",
        "v_proj",
        register_fn,
        quantization_method,
        false,
        dtype,
        device,
        rank_info);

    if (rank_info.tp_size > 1) {
        // TP global RMSNorm: the variance is all-reduced across ranks (see
        // forward) and the weight is split along the projection dimension and
        // registered under the checkpoint name "norm.weight". Note that
        // constructing an `nn::Parameter` with a `tp_dim` takes a full shape
        // and splits it by `tp_size` internally, so the full projection_size
        // is passed instead of local_projection_size_.
        norm_weight_ = infinicore::nn::Parameter(
            {projection_size}, dtype, device,
            /*tp_dim=*/0, rank_info.tp_rank, rank_info.tp_size);
        register_parameter("norm.weight", norm_weight_);
    } else {
        // Single GPU keeps the standard RMSNorm (fused, f32 precision) so
        // that the existing single-GPU validation results stay identical.
        // The reference `MiniMaxText01LightningAttention` builds its norm with
        // `MiniMaxText01RMSNorm` default eps (1e-6), unlike the other RMSNorms.
        INFINICORE_NN_MODULE_INIT(norm, projection_size, 1e-6, dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(output_gate, hidden_size_, projection_size, quantization_method,
                              false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(out_proj, projection_size, hidden_size_, quantization_method,
                              false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size, rank_info.comm);

    // Per-head ALiBi-style slopes, scaled by layer depth:
    // slope_layer = slope_base * (1 - layer_idx / (L - 1) + 1e-5).
    // Under TP each rank only handles its local heads, so only this rank's
    // decay heads are kept (matching the q/k/v split).
    const auto slopes = get_slopes(num_heads_);
    const double layer_scale = 1.0 - static_cast<double>(layer_idx) / static_cast<double>(num_hidden_layers_ - 1) + 1e-5;
    const size_t local_head_begin = tp_rank_ * local_num_heads_;
    std::vector<uint16_t> decay_bf16(local_num_heads_);
    std::vector<double> decays(local_num_heads_);
    for (size_t i = 0; i < local_num_heads_; ++i) {
        const double decay = std::exp(-slopes[local_head_begin + i] * layer_scale);
        decays[i] = decay;
        decay_bf16[i] = float_to_bf16(static_cast<float>(decay));
    }
    // Wrap the buffer as a bf16 tensor so element-wise multiplication with the
    // bf16 recurrent state never mixes dtypes.
    auto decay_cpu = infinicore::Tensor::from_blob(
        decay_bf16.data(), {1, local_num_heads_, 1, 1},
        infinicore::DataType::BF16, infinicore::Device::cpu());
    decay_tensor_ = decay_cpu->to(device);

    // Chunked prefill: enabled by config `lightning_chunk_size` (>= 2, default
    // 64) so prefill of long sequences does not run one serial token at a time.
    // Env INFINILM_LIGHTNING_CHUNK overrides it (0/1 disables and keeps the
    // exact legacy token-by-token path for A/B comparison); decode always uses
    // the single-token recurrence regardless of this setting.
    int64_t chunk_size = model_config->get_or<int64_t>("lightning_chunk_size", 64);
    if (const char *env = std::getenv("INFINILM_LIGHTNING_CHUNK")) {
        chunk_size = std::atoll(env);
    }
    if (chunk_size >= 2) {
        chunk_size_ = static_cast<size_t>(std::min<int64_t>(chunk_size, 512));
        ChunkDecayTables tables = build_chunk_decay_tables(chunk_size_, decays, dtype, device);
        chunk_pow_ = std::move(tables.pow);
        chunk_qdec_ = std::move(tables.qdec);
        chunk_kdec_ = std::move(tables.kdec);
        chunk_diag_ = std::move(tables.diag);
    }
}

infinicore::Tensor MiniMaxText01LinearAttention::forward(
    const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    // q/k/v come from `QKVParallelLinear` and are already the local heads of
    // this TP rank; the projection dimension is scaled accordingly.
    const size_t local_projection_size = local_projection_size_;
    const auto &dtype = hidden_states->dtype();
    const auto &device = hidden_states->device();

    // 1. Fused QKV projection + SiLU, then split into q / k / v.
    auto hidden_states_mutable = hidden_states;
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);
    q = infinicore::op::silu(q);
    k = infinicore::op::silu(k);
    v = infinicore::op::silu(v);

    // 2. Reshape to [batch, seq, local_heads, head_dim].
    auto q_heads = q->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto k_heads = k->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto v_heads = v->view({batch_size, seq_len, local_num_heads_, head_dim_});

    // Recurrent state [batch, local_heads, head_dim, head_dim], initialised to zero.
    auto state = infinicore::Tensor::zeros(
        {batch_size, local_num_heads_, head_dim_, head_dim_}, dtype, device);

    // Pool plumbing shared by both paths: decoding requests seed `state` from
    // the shared pool; afterwards the final state is persisted back.
    auto &forward_context = infinilm::global_state::get_forward_context();
    const auto &mamba_metadata = forward_context.mamba_metadata;
    // A linear attention layer always has its recurrent state allocated, so the
    // pool exists whenever the vector covers this layer.
    const bool has_pool = forward_context.ssm_state_vec.size() > layer_idx_;
    const bool has_indices = mamba_metadata.init_state_indices.has_value()
                          && mamba_metadata.final_state_indices.has_value();
    bool is_decode = false;
    if (has_pool && has_indices) {
        const auto pool = forward_context.ssm_state_vec.at(layer_idx_);
        const size_t num_requests = mamba_metadata.input_offsets.value()->shape()[0] - 1;
        is_decode = num_requests == seq_len;
        if (is_decode) {
            auto init_idx_cpu = mamba_metadata.init_state_indices.value()->to(infinicore::Device::cpu());
            const auto *init_idx = reinterpret_cast<const int32_t *>(init_idx_cpu->data());
            for (size_t r = 0; r < num_requests; ++r) {
                state->narrow({{0, r, 1}})
                    ->copy_from(pool->narrow({{0, static_cast<size_t>(init_idx[r]), 1}}));
            }
        }
    }

    // The chunked scan handles long prefill in parallel blocks; decode (exactly
    // one recurrent step per request) and short sequences fall back to the
    // token-by-token recurrence. Both are mathematically identical and share
    // `state` as the carry, so the persisted final state is the same.
    const bool use_chunked = chunk_size_ >= 2 && !is_decode && seq_len >= 2;
    infinicore::Tensor output;
    if (use_chunked) {
        output = forward_chunked_(q_heads, k_heads, v_heads, state);
    } else {
        // Per-head decay broadcast over the [batch, local_heads, head_dim, head_dim]
        // state (only needed by the recurrence path).
        auto decay = infinicore::op::broadcast_to(
                         decay_tensor_,
                         std::vector<int64_t>{static_cast<int64_t>(batch_size),
                                              static_cast<int64_t>(local_num_heads_),
                                              static_cast<int64_t>(head_dim_),
                                              static_cast<int64_t>(head_dim_)})
                         ->contiguous();
        output = forward_recurrent_(q_heads, k_heads, v_heads, decay, state);
    }

    // Optionally persist the final states back to the shared pool.
    if (has_pool && has_indices) {
        const auto pool = forward_context.ssm_state_vec.at(layer_idx_);
        const size_t num_requests = mamba_metadata.input_offsets.value()->shape()[0] - 1;
        auto final_idx_cpu = mamba_metadata.final_state_indices.value()->to(infinicore::Device::cpu());
        const auto *final_idx = reinterpret_cast<const int32_t *>(final_idx_cpu->data());
        for (size_t r = 0; r < num_requests; ++r) {
            pool->narrow({{0, static_cast<size_t>(final_idx[r]), 1}})
                ->copy_from(state->narrow({{0, r, 1}}));
        }
    }

    // 4. RMSNorm, output gate and output projection.
    //    A "global RMSNorm" (variance all-reduced across ranks, weight split
    //    by TP) is used when TP > 1; when TP == 1 the standard nn::RMSNorm is
    //    kept so that the single-GPU validation results stay unchanged.
    infinicore::Tensor normalized;
    if (tp_size_ > 1) {
        // mean(x^2) over the local projection dim, per token: [B, S, 1].
        auto x2 = infinicore::op::mul(output, output);
        auto mean_x2 = infinicore::op::var_mean(x2, {2}, false, true).second;
        // All-reduce the per-token local mean and divide by tp_size. Local
        // projection dims are disjoint across ranks, so the global mean equals
        // the average of the per-rank local means.
        if (communicator_ != nullptr) {
            infinicore::op::distributed::allreduce_(
                mean_x2, mean_x2, INFINICCL_SUM, communicator_);
            mean_x2 = infinicore::op::mul_scalar(
                mean_x2, 1.0 / static_cast<double>(tp_size_));
        }
        // mean + eps (add needs equal shapes: broadcast the eps scalar first).
        uint16_t eps_bits = float_to_bf16(1e-6f);
        auto eps_cpu = infinicore::Tensor::from_blob(
            &eps_bits, {1, 1, 1}, infinicore::DataType::BF16,
            infinicore::Device::cpu());
        auto eps_t = eps_cpu->to(device);
        const std::vector<int64_t> mean_shape{
            static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len), 1};
        auto mean_eps = infinicore::op::add(
            mean_x2, infinicore::op::broadcast_to(eps_t, mean_shape));
        // rsqrt = (mean + eps)^(-0.5); the convenient float_power returns F64,
        // so use the out-version to force a bf16 output (no mixed-dtype mul).
        auto inv = infinicore::Tensor::empty(
            {batch_size, seq_len, 1}, dtype, device);
        infinicore::op::float_power_(inv, mean_eps, -0.5);
        const std::vector<int64_t> full_shape{
            static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len),
            static_cast<int64_t>(local_projection_size)};
        auto inv_b = infinicore::op::broadcast_to(inv, full_shape);
        auto w_b = infinicore::op::broadcast_to(norm_weight_, full_shape);
        normalized = infinicore::op::mul(
            infinicore::op::mul(output, inv_b), w_b);
    } else {
        normalized = norm_->forward(output);
    }
    auto gate_input = hidden_states; // non-const copy for the linear API
    auto output_gate = infinicore::op::sigmoid(output_gate_->forward(gate_input));
    auto gated = infinicore::op::mul(normalized, output_gate);
    return out_proj_->forward(gated);
}

infinicore::Tensor MiniMaxText01LinearAttention::forward_recurrent_(
    const infinicore::Tensor &q_heads,
    const infinicore::Tensor &k_heads,
    const infinicore::Tensor &v_heads,
    const infinicore::Tensor &decay,
    infinicore::Tensor &state) const {
    // Token-by-token recurrence (the exact legacy path, kept for decode and for
    // A/B comparison):
    //    S = decay * S + outer(k, v);  o = q @ S.
    const auto shape = q_heads->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t bh = batch_size * local_num_heads_;
    auto output = infinicore::Tensor::empty(
        {batch_size, seq_len, local_projection_size_}, q_heads->dtype(),
        q_heads->device());
    for (size_t t = 0; t < seq_len; ++t) {
        auto k_t = k_heads->narrow({{1, t, 1}})->contiguous(); // [batch, 1, heads, head_dim]
        auto v_t = v_heads->narrow({{1, t, 1}})->contiguous();
        auto q_t = q_heads->narrow({{1, t, 1}})->contiguous();

        // outer = k^T @ v: [b*h, d, 1] @ [b*h, 1, d] -> [b*h, d, d].
        auto outer = infinicore::op::matmul(
            k_t->view({bh, head_dim_, 1}),
            v_t->view({bh, 1, head_dim_}));

        // state = decay * state + outer.
        state = infinicore::op::add(
            infinicore::op::mul(state, decay),
            outer->view({batch_size, local_num_heads_, head_dim_, head_dim_}));

        // o = q @ state: [b*h, 1, d] @ [b*h, d, d] -> [b*h, 1, d].
        auto o = infinicore::op::matmul(
            q_t->view({bh, 1, head_dim_}),
            state->view({bh, head_dim_, head_dim_}));

        output->narrow({{1, t, 1}})
            ->copy_from(o->view({batch_size, 1, local_projection_size_}));
    }
    return output;
}

infinicore::Tensor MiniMaxText01LinearAttention::forward_chunked_(
    const infinicore::Tensor &q_heads,
    const infinicore::Tensor &k_heads,
    const infinicore::Tensor &v_heads,
    infinicore::Tensor &state) const {
    // Chunked prefill (HF-style block scan). q/k/v are [b, S, h, d]; prefill
    // evaluates the recurrence S_t = a_h*S_{t-1} + k_t^T v_t in blocks of C
    // tokens. Within a block the causal decayed attention is computed densely
    // in parallel; only the block-to-block carry is serial. `state` carries the
    // exact post-block recurrent state, so it finishes identical to the
    // token-by-token recurrence up to floating-point accumulation order.
    const auto shape = q_heads->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t bh = batch_size * local_num_heads_;
    const size_t C = chunk_size_;
    const auto &dtype = q_heads->dtype();
    const auto &device = q_heads->device();
    const std::vector<int64_t> state_shape{
        static_cast<int64_t>(batch_size), static_cast<int64_t>(local_num_heads_),
        static_cast<int64_t>(head_dim_), static_cast<int64_t>(head_dim_)};

    // Move the head axis next to batch so (batch, head) fold onto one GEMM
    // batch dimension with no per-block permute: [b, S, h, d] -> [b, h, S, d].
    auto q4 = q_heads->permute({0, 2, 1, 3})->contiguous();
    auto k4 = k_heads->permute({0, 2, 1, 3})->contiguous();
    auto v4 = v_heads->permute({0, 2, 1, 3})->contiguous();
    auto output4 = infinicore::Tensor::empty(
        {batch_size, local_num_heads_, seq_len, head_dim_}, dtype, device);

    // Helper: broadcast a [1, h, X, 1] chunk table onto a [b, h, m, d] tensor.
    // `table` is taken by value (a shared-handle copy) so narrow() may return a
    // fresh view; the slice is made contiguous before broadcasting.
    const auto scale_chunk = [&](infinicore::Tensor table,
                                 const infinicore::Tensor &chunk,
                                 size_t begin, size_t len) {
        auto slice = table->narrow({{2, begin, len}})->contiguous();
        const std::vector<int64_t> tshape{
            static_cast<int64_t>(batch_size), static_cast<int64_t>(local_num_heads_),
            static_cast<int64_t>(len), static_cast<int64_t>(head_dim_)};
        return infinicore::op::mul(
            chunk, infinicore::op::broadcast_to(slice, tshape)->contiguous());
    };

    for (size_t si = 0; si < seq_len; si += C) {
        const size_t m = std::min(C, seq_len - si);
        auto qb = q4->narrow({{2, si, m}}); // [b, h, m, d]
        auto kb = k4->narrow({{2, si, m}});
        auto vb = v4->narrow({{2, si, m}});
        auto qf = qb->view({bh, m, head_dim_}); // folded GEMM operands
        auto vf = vb->view({bh, m, head_dim_});

        // (a) Cross-block term: o_cross = (q * a^{pos+1}) @ carry.
        //     (b, h) rows align between qb and state because both fold the same
        //     (batch, head) axis.
        auto o_cross = infinicore::op::matmul(
            scale_chunk(chunk_qdec_, qb, 0, m)->view({bh, m, head_dim_}),
            state->view({bh, head_dim_, head_dim_}));

        // (b) Intra-block dense causal term:
        //     o_diag = ((q @ k^T) .* a^{i-j}) @ v.
        auto kT = kb->view({bh, m, head_dim_})->permute({0, 2, 1})->contiguous();
        auto qk = infinicore::op::matmul(qf, kT); // [bh, m, m]
        auto diag_m = chunk_diag_->narrow({{2, 0, m}})
                          ->narrow({{3, 0, m}})
                          ->contiguous();
        const std::vector<size_t> qkview{batch_size, local_num_heads_, m, m};
        const std::vector<int64_t> qkshape{
            static_cast<int64_t>(batch_size), static_cast<int64_t>(local_num_heads_),
            static_cast<int64_t>(m), static_cast<int64_t>(m)};
        auto qkm = infinicore::op::mul(
            qk->view(qkview),
            infinicore::op::broadcast_to(diag_m, qkshape)->contiguous());
        auto o_diag = infinicore::op::matmul(qkm->view({bh, m, m}), vf);

        // o = o_cross + o_diag, written back as [b, h, m, d].
        auto o_blk = infinicore::op::add(o_cross, o_diag);
        output4->narrow({{2, si, m}})
            ->copy_from(o_blk->view({batch_size, local_num_heads_, m, head_dim_}));

        // (c) Fold the block into the carry:
        //     state = a^m * state + (k * a^{m-1-pos})^T @ v.
        auto kTs = scale_chunk(chunk_kdec_, kb, C - m, m)
                       ->view({bh, m, head_dim_})
                       ->permute({0, 2, 1})
                       ->contiguous();
        auto state_outer = infinicore::op::matmul(kTs, vf);         // [bh, d, d]
        auto pow_m = chunk_pow_->narrow({{2, m, 1}})->contiguous(); // [1, h, 1, 1] a^m
        auto decay_m = infinicore::op::broadcast_to(pow_m, state_shape)->contiguous();
        state = infinicore::op::add(
            infinicore::op::mul(state, decay_m),
            state_outer->view({batch_size, local_num_heads_, head_dim_, head_dim_}));
    }

    // Back to the [b, S, h*d] layout expected by the caller.
    return output4->permute({0, 2, 1, 3})
        ->contiguous()
        ->view({batch_size, seq_len, local_projection_size_});
}

} // namespace infinilm::models::minimax_text_01
