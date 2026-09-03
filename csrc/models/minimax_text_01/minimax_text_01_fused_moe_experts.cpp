#include "minimax_text_01_fused_moe_experts.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "../../layers/quantization/quantization_scheme.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::minimax_text_01 {

MiniMaxText01FusedMoeExperts::MiniMaxText01FusedMoeExperts(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    num_experts_ = model_config->get<size_t>("num_experts");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto dtype = model_config->get_dtype();
    ASSERT(num_experts_ > 0);

    const auto ep_config = infinilm::layers::moe::make_ep_config();
    const auto expert_placement = infinilm::layers::moe::make_expert_placement(ep_config, num_experts_);
    const size_t num_local_experts = expert_placement.local_num_experts;

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const bool ep_enabled = ep_config.backend != infinilm::layers::moe::EPBackend::Disabled;
    if (ep_enabled) {
        intermediate_size_per_partition_ = intermediate_size;
    } else {
        ASSERT(intermediate_size % tp_size == 0);
        intermediate_size_per_partition_ = intermediate_size / tp_size;
    }
    const size_t expert_tp_rank = ep_enabled ? 0 : tp_rank;
    const size_t expert_tp_size = ep_enabled ? 1 : tp_size;
    device_ = device;

    // MXFP4 path: per-expert packed registration (w1/w2/w3.weight_packed +
    // weight_scale), consumed by the fused_moe_mxfp4 kernel for real 4-bit
    // memory compression.
    const auto quant = model_config->get_quantization_method();
    use_mxfp4_ = quant != nullptr
              && quant->get_quant_scheme() == infinilm::quantization::QuantScheme::MXFP4_W4A16;
    if (use_mxfp4_) {
        register_mxfp4_experts();
        return;
    }

    // Key difference: the shape is passed in full (intermediate_size rather
    // than per-partition) together with a tp_dim. Constructing an
    // `nn::Parameter` with a `tp_dim` splits the tensor along that dimension
    // by `tp_size` automatically, and loading narrows the corresponding slice
    // per tp_rank.
    w13_weight_ = infinicore::nn::Parameter(
        {num_local_experts, intermediate_size * 2, hidden_size_},
        dtype,
        device,
        /*tp_dim=*/1,
        expert_tp_rank,
        expert_tp_size);
    w2_weight_ = infinicore::nn::Parameter(
        {num_local_experts, hidden_size_, intermediate_size},
        dtype,
        device,
        /*tp_dim=*/2,
        expert_tp_rank,
        expert_tp_size);
    this->register_parameter("w13_weight", w13_weight_);
    this->register_parameter("w2_weight", w2_weight_);

    for (size_t local_expert = 0; local_expert < num_local_experts; ++local_expert) {
        const size_t global_expert = expert_placement.local_expert_start + local_expert;
        auto gate_weight = w13_weight_->narrow(
                                          {{0, local_expert, 1}, {1, 0, intermediate_size_per_partition_}})
                               ->squeeze(0);
        auto up_weight = w13_weight_
                             ->narrow({{0, local_expert, 1},
                                       {1, intermediate_size_per_partition_, intermediate_size_per_partition_}})
                             ->squeeze(0);
        auto down_weight = w2_weight_->narrow({{0, local_expert, 1}})->squeeze(0);

        const std::string prefix = std::to_string(global_expert) + ".";
        this->register_parameter(
            prefix + "gate_proj.weight",
            infinicore::nn::Parameter(gate_weight, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(
            prefix + "up_proj.weight",
            infinicore::nn::Parameter(up_weight, 0, expert_tp_rank, expert_tp_size));
        this->register_parameter(
            prefix + "down_proj.weight",
            infinicore::nn::Parameter(down_weight, 1, expert_tp_rank, expert_tp_size));
    }

    moe_weights_.packed_w13 = w13_weight_;
    moe_weights_.packed_w2 = w2_weight_;
}

const infinilm::layers::moe::MoeWeights &MiniMaxText01FusedMoeExperts::moe_weights() const {
    return moe_weights_;
}

void MiniMaxText01FusedMoeExperts::register_mxfp4_experts() {
    if (hidden_size_ % 32 != 0 || intermediate_size_per_partition_ % 32 != 0) {
        throw std::runtime_error(
            "MiniMaxText01FusedMoeExperts: MXFP4 dimensions must be divisible by 32");
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);

    // w1/w3 share the w13 storage; all shapes are the local (per-partition) ones.
    auto w13 = infinicore::Tensor::empty(
        {num_experts_, 2 * intermediate_size_per_partition_, hidden_size_ / 2},
        infinicore::DataType::U8, device_);
    auto w13_scale = infinicore::Tensor::empty(
        {num_experts_, 2 * intermediate_size_per_partition_, hidden_size_ / 32},
        infinicore::DataType::U8, device_);
    auto w2 = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, intermediate_size_per_partition_ / 2},
        infinicore::DataType::U8, device_);
    auto w2_scale = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, intermediate_size_per_partition_ / 32},
        infinicore::DataType::U8, device_);

    auto register_packed = [&](const std::string &name,
                               const infinicore::Tensor &storage,
                               size_t expert,
                               size_t row_start,
                               size_t row_count,
                               size_t tp_dim) {
        auto expert_view = storage
                               ->narrow({{0, expert, 1}, {1, row_start, row_count}})
                               ->squeeze(0);
        this->register_parameter(
            name,
            infinicore::nn::Parameter(expert_view, tp_dim, tp_rank, tp_size));
    };
    for (size_t expert = 0; expert < num_experts_; ++expert) {
        const std::string prefix = std::to_string(expert) + ".";
        register_packed(prefix + "w1.weight_packed", w13, expert,
                        0, intermediate_size_per_partition_, 0);
        register_packed(prefix + "w1.weight_scale", w13_scale, expert,
                        0, intermediate_size_per_partition_, 0);
        register_packed(prefix + "w3.weight_packed", w13, expert,
                        intermediate_size_per_partition_, intermediate_size_per_partition_, 0);
        register_packed(prefix + "w3.weight_scale", w13_scale, expert,
                        intermediate_size_per_partition_, intermediate_size_per_partition_, 0);
        register_packed(prefix + "w2.weight_packed", w2, expert,
                        0, hidden_size_, 1);
        register_packed(prefix + "w2.weight_scale", w2_scale, expert,
                        0, hidden_size_, 1);
    }

    mxfp4_weights_.packed_w13 = std::move(w13);
    mxfp4_weights_.w13_scale = std::move(w13_scale);
    mxfp4_weights_.packed_w2 = std::move(w2);
    mxfp4_weights_.w2_scale = std::move(w2_scale);
}

} // namespace infinilm::models::minimax_text_01
