#include "kimi_k25_moe.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <infinicore/ops/mxfp4_dequantize.hpp>

#include <stdexcept>
#include <string>

namespace infinilm::models::kimi_k25 {
namespace {

bool uses_mxfp4(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const auto &config = model_config->get_config_json();
    return config.contains("quantization_config")
        && config.at("quantization_config").is_object()
        && config.at("quantization_config").value("quant_method", "") == "quark";
}

} // namespace

KimiK25MXFP4Experts::KimiK25MXFP4Experts(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : num_experts_(model_config->get<size_t>("num_experts")),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      dtype_(model_config->get_dtype()),
      device_(device) {
    const size_t intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const size_t tp_rank = static_cast<size_t>(rank_info.tp_rank);
    if (hidden_size_ % 32 != 0 || intermediate_size % (tp_size * 32) != 0) {
        throw std::runtime_error(
            "KimiK25MXFP4Experts: expert dimensions must be divisible by TP and MXFP4 group size");
    }
    local_intermediate_size_ = intermediate_size / tp_size;
    packed_parameters_.reserve(num_experts_ * 6);

    auto register_packed = [&](const std::string &name,
                               const infinicore::Shape &shape,
                               size_t tp_dim) {
        packed_parameters_.emplace_back(
            shape, infinicore::DataType::U8, device, tp_dim, tp_rank, tp_size);
        this->register_parameter(name, packed_parameters_.back());
    };

    for (size_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
        const std::string prefix = std::to_string(expert_idx) + ".";
        for (const std::string projection : {"gate_proj", "up_proj"}) {
            register_packed(prefix + projection + ".weight",
                            {intermediate_size, hidden_size_ / 2}, 0);
            register_packed(prefix + projection + ".weight_scale",
                            {intermediate_size, hidden_size_ / 32}, 0);
        }
        register_packed(prefix + "down_proj.weight",
                        {hidden_size_, intermediate_size / 2}, 1);
        register_packed(prefix + "down_proj.weight_scale",
                        {hidden_size_, intermediate_size / 32}, 1);
    }
}

const infinilm::layers::moe::MoeWeights &KimiK25MXFP4Experts::moe_weights() const {
    if (!moe_weights_.packed_w13 || !moe_weights_.packed_w2) {
        throw std::runtime_error(
            "KimiK25MXFP4Experts: weights have not been processed after loading");
    }
    return moe_weights_;
}

void KimiK25MXFP4Experts::process_weights_after_loading() {
    if (moe_weights_.packed_w13 && moe_weights_.packed_w2) {
        return;
    }
    auto w13 = infinicore::Tensor::empty(
        {num_experts_, local_intermediate_size_ * 2, hidden_size_},
        dtype_, device_);
    auto w2 = infinicore::Tensor::empty(
        {num_experts_, hidden_size_, local_intermediate_size_},
        dtype_, device_);

    for (size_t expert_idx = 0; expert_idx < num_experts_; ++expert_idx) {
        const std::string prefix = std::to_string(expert_idx) + ".";
        auto dequantize = [&](const std::string &projection) {
            const std::string weight_name = prefix + projection + ".weight";
            const std::string scale_name = prefix + projection + ".weight_scale";
            const auto weight_it = parameters_.find(weight_name);
            const auto scale_it = parameters_.find(scale_name);
            if (weight_it == parameters_.end() || scale_it == parameters_.end()) {
                throw std::runtime_error(
                    "KimiK25MXFP4Experts: missing packed parameter "
                    + (weight_it == parameters_.end() ? weight_name : scale_name));
            }
            return infinicore::op::mxfp4_dequantize(
                weight_it->second,
                scale_it->second,
                dtype_);
        };

        auto gate = dequantize("gate_proj");
        auto up = dequantize("up_proj");
        auto down = dequantize("down_proj");
        w13->narrow({{0, expert_idx, 1}, {1, 0, local_intermediate_size_}})
            ->squeeze(0)
            ->copy_from(gate);
        w13->narrow({{0, expert_idx, 1},
                     {1, local_intermediate_size_, local_intermediate_size_}})
            ->squeeze(0)
            ->copy_from(up);
        w2->narrow({{0, expert_idx, 1}})->squeeze(0)->copy_from(down);
    }

    parameters_.clear();
    packed_parameters_.clear();
    packed_parameters_.shrink_to_fit();
    moe_weights_.packed_w13 = std::move(w13);
    moe_weights_.packed_w2 = std::move(w2);
}

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_mlp_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t intermediate_size) {
    auto config_json = model_config->get_config_json();
    config_json["intermediate_size"] = intermediate_size;
    return std::make_shared<infinilm::config::ModelConfig>(std::move(config_json));
}

KimiK25MoE::KimiK25MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    routed_scaling_factor_ = model_config->get_or<float>("routed_scaling_factor", 1.0f);
    uses_mxfp4_experts_ = uses_mxfp4(model_config);
    if (uses_mxfp4_experts_) {
        mxfp4_experts_ = this->register_module<KimiK25MXFP4Experts>(
            "experts", model_config, device);
    } else {
        INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    }
    INFINICORE_NN_MODULE_INIT(fused_moe, model_config, device, layer_idx);

    const size_t shared_intermediate_size = model_config->get<size_t>("moe_intermediate_size")
                                          * model_config->get_or<size_t>("n_shared_experts", 1);
    auto shared_config = make_kimi_mlp_config(model_config, shared_intermediate_size);
    INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
}

infinicore::Tensor KimiK25MoE::forward(const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    auto flattened = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto [routing_weights, selected_experts] = gate_->forward(flattened);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };
    const auto &weights = uses_mxfp4_experts_
                            ? mxfp4_experts_->moe_weights()
                            : experts_->moe_weights();
    auto routed = fused_moe_->forward(flattened, topk_output, weights)->view(shape);
    routed = infinicore::op::mul_scalar(routed, routed_scaling_factor_);
    return infinicore::op::add(routed, shared_experts_->forward(hidden_states));
}

} // namespace infinilm::models::kimi_k25
