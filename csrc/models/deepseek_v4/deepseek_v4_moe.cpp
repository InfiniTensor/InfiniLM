#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_profile.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/deepseek_v4_router.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#include "spdlog/spdlog.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {
namespace {

bool supports_fused_deepseek_moe(infinicore::Device::Type device_type) {
    switch (device_type) {
    case infinicore::Device::Type::NVIDIA:
    case infinicore::Device::Type::ALI:
    case infinicore::Device::Type::HYGON:
    case infinicore::Device::Type::ILUVATAR:
    case infinicore::Device::Type::METAX:
    case infinicore::Device::Type::MOORE:
        return true;
    default:
        return false;
    }
}

bool env_flag(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::string(value) == "1";
}

bool disable_fused_hash_topk_env() {
    static const bool value = env_flag("DSV4_DISABLE_FUSED_HASH_TOPK");
    return value;
}

} // namespace

DeepseekV4TopK::DeepseekV4TopK(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")),
      routed_scaling_(static_cast<float>(model_config->get_or<double>("routed_scaling_factor", 1.0))),
      scoring_func_(model_config->get_or<std::string>("scoring_func", "sqrtsoftplus")),
      norm_topk_prob_(model_config->get_or<bool>("norm_topk_prob", true)),
      layer_idx_(layer_idx) {
    ASSERT(scoring_func_ == "sqrtsoftplus");
    const auto &dtype = model_config->get_dtype();
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size_}, dtype, device));

    const std::string topk_method = model_config->get_or<std::string>("topk_method", "");
    if (topk_method == "noaux_tc") {
        INFINICORE_NN_PARAMETER_INIT(bias, ({num_experts_}, infinicore::DataType::F32, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4TopK::forward(const infinicore::Tensor &hidden_states) const {
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE router: expected hidden_states shape [N,D]");
    }

    infinicore::Tensor router_bias;
    if (has_bias()) {
        router_bias = bias_;
    }

    auto gate_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);

    return infinicore::op::deepseek_v4_topk_router(gate_logits,
                                                   num_experts_per_tok_,
                                                   norm_topk_prob_,
                                                   router_bias);
}

DeepseekV4HashTopK::DeepseekV4HashTopK(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")),
      routed_scaling_(static_cast<float>(model_config->get_or<double>("routed_scaling_factor", 1.0))),
      scoring_func_(model_config->get_or<std::string>("scoring_func", "sqrtsoftplus")),
      norm_topk_prob_(model_config->get_or<bool>("norm_topk_prob", true)),
      layer_idx_(layer_idx) {
    ASSERT(scoring_func_ == "sqrtsoftplus");
    const auto &dtype = model_config->get_dtype();
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size_}, dtype, device));

    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab_size, num_experts_per_tok_}, infinicore::DataType::I64, device));
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4HashTopK::forward(const infinicore::Tensor &hidden_states,
                            const infinicore::Tensor &input_ids) const {

    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE router: expected hidden_states shape [N,D]");
    }
    if (input_ids.empty()) {
        throw std::runtime_error("DeepseekV4HashTopK: hash-routed layer requires input_ids");
    }

    if (input_ids->device() != hidden_states->device()) {
        throw std::runtime_error("DeepseekV4HashTopK: input_ids must be on the same device as hidden_states");
    }

    auto input_ids_view = input_ids->is_contiguous() ? input_ids : input_ids->contiguous();
    if (!disable_fused_hash_topk_env()) {
        try {
            return infinicore::op::deepseek_v4_hash_topk_router(
                hidden_states,
                weight_,
                input_ids_view,
                tid2eid_,
                norm_topk_prob_);
        } catch (const std::exception &e) {
            spdlog::warn("DeepseekV4HashTopK: fused hash topk unavailable, falling back to linear + hash router: {}", e.what());
        }
    }

    auto gate_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);

    return infinicore::op::deepseek_v4_hash_router(
        gate_logits,
        input_ids_view,
        tid2eid_,
        norm_topk_prob_);
}

DeepseekV4Experts::DeepseekV4Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device)
    : DeepseekV4Experts(std::move(model_config), 0, device) {
}

DeepseekV4Experts::DeepseekV4Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      moe_intermediate_size_(model_config->get<size_t>("moe_intermediate_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    if (!use_deepseek_v4_w8a8_linear(model_config)) {
        throw std::runtime_error(std::string{});
    }
    init_packed_w8a8_experts_(device, rank_info.tp_rank, rank_info.tp_size);
    refresh_expert_weight_views_();
}

void DeepseekV4Experts::init_packed_w8a8_experts_(const infinicore::Device &device,
                                                  int tp_rank,
                                                  int tp_size) {
    if (tp_size <= 0 || moe_intermediate_size_ % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error("DeepseekV4Experts: moe_intermediate_size must be divisible by tp_size for packed experts");
    }
    local_moe_intermediate_size_ = moe_intermediate_size_ / static_cast<size_t>(tp_size);

    gate_weight_packed_ = infinicore::nn::Parameter(
        {num_experts_, local_moe_intermediate_size_, hidden_size_},
        infinicore::DataType::I8,
        device);
    up_weight_packed_ = infinicore::nn::Parameter(
        {num_experts_, local_moe_intermediate_size_, hidden_size_},
        infinicore::DataType::I8,
        device);
    down_weight_packed_ = infinicore::nn::Parameter(
        {num_experts_, hidden_size_, local_moe_intermediate_size_},
        infinicore::DataType::I8,
        device);
    gate_weight_scale_packed_ = infinicore::nn::Parameter(
        {num_experts_, local_moe_intermediate_size_, 1},
        infinicore::DataType::F32,
        device);
    up_weight_scale_packed_ = infinicore::nn::Parameter(
        {num_experts_, local_moe_intermediate_size_, 1},
        infinicore::DataType::F32,
        device);
    down_weight_scale_packed_ = infinicore::nn::Parameter(
        {num_experts_, hidden_size_, 1},
        infinicore::DataType::F32,
        device);

    register_packed_expert_aliases_(tp_rank, tp_size);
}

void DeepseekV4Experts::register_packed_expert_aliases_(int tp_rank, int tp_size) {
    for (size_t i = 0; i < num_experts_; ++i) {
        const std::string prefix = std::to_string(i) + ".";
        auto gate_weight = gate_weight_packed_->narrow({{0, i, 1}})->squeeze(0);
        auto up_weight = up_weight_packed_->narrow({{0, i, 1}})->squeeze(0);
        auto down_weight = down_weight_packed_->narrow({{0, i, 1}})->squeeze(0);
        auto gate_scale = gate_weight_scale_packed_->narrow({{0, i, 1}})->squeeze(0);
        auto up_scale = up_weight_scale_packed_->narrow({{0, i, 1}})->squeeze(0);
        auto down_scale = down_weight_scale_packed_->narrow({{0, i, 1}})->squeeze(0);

        this->register_parameter(prefix + "w1.weight",
                                 infinicore::nn::Parameter(gate_weight, 0, tp_rank, tp_size));
        this->register_parameter(prefix + "w1.weight_scale",
                                 infinicore::nn::Parameter(gate_scale, 0, tp_rank, tp_size));
        this->register_parameter(prefix + "w2.weight",
                                 infinicore::nn::Parameter(down_weight, 1, tp_rank, tp_size));
        this->register_parameter(prefix + "w2.weight_scale",
                                 infinicore::nn::Parameter(down_scale, 0, 0, 1));
        this->register_parameter(prefix + "w3.weight",
                                 infinicore::nn::Parameter(up_weight, 0, tp_rank, tp_size));
        this->register_parameter(prefix + "w3.weight_scale",
                                 infinicore::nn::Parameter(up_scale, 0, tp_rank, tp_size));
    }
}

void DeepseekV4Experts::refresh_expert_weight_views_() {
    gate_weights_.clear();
    up_weights_.clear();
    down_weights_.clear();
    gate_weight_scales_.clear();
    up_weight_scales_.clear();
    down_weight_scales_.clear();

    gate_weights_.reserve(num_experts_);
    up_weights_.reserve(num_experts_);
    down_weights_.reserve(num_experts_);
    gate_weight_scales_.reserve(num_experts_);
    up_weight_scales_.reserve(num_experts_);
    down_weight_scales_.reserve(num_experts_);

    for (size_t i = 0; i < num_experts_; ++i) {
        gate_weights_.push_back(gate_weight_packed_->narrow({{0, i, 1}})->squeeze(0));
        up_weights_.push_back(up_weight_packed_->narrow({{0, i, 1}})->squeeze(0));
        down_weights_.push_back(down_weight_packed_->narrow({{0, i, 1}})->squeeze(0));
        gate_weight_scales_.push_back(gate_weight_scale_packed_->narrow({{0, i, 1}})->squeeze(0));
        up_weight_scales_.push_back(up_weight_scale_packed_->narrow({{0, i, 1}})->squeeze(0));
        down_weight_scales_.push_back(down_weight_scale_packed_->narrow({{0, i, 1}})->squeeze(0));
    }
    local_moe_intermediate_size_ = gate_weights_.empty() ? moe_intermediate_size_ : gate_weights_.front()->shape()[0];
}

bool DeepseekV4Experts::has_w8a8_weights_() const {
    return !gate_weights_.empty()
        && gate_weights_.front()->dtype() == infinicore::DataType::I8
        && up_weights_.front()->dtype() == infinicore::DataType::I8
        && down_weights_.front()->dtype() == infinicore::DataType::I8
        && !gate_weight_scales_.empty()
        && !up_weight_scales_.empty()
        && !down_weight_scales_.empty()
        && gate_weight_scales_.front()->dtype() == infinicore::DataType::F32
        && up_weight_scales_.front()->dtype() == infinicore::DataType::F32
        && down_weight_scales_.front()->dtype() == infinicore::DataType::F32;
}

void DeepseekV4Experts::process_weights_after_loading() {
    refresh_expert_weight_views_();
    expert_ptr_tables_.reset();
    if (std::getenv("DSV4_DISABLE_LAYER_PTR_TABLES") == nullptr && has_w8a8_weights_()) {
        expert_ptr_tables_ = build_w8a8_ptr_tables_(gate_weights_.front()->device());
    }
}

infinicore::Tensor DeepseekV4Experts::build_w8a8_ptr_tables_(const infinicore::Device &device) const {
    std::vector<const void *> ptrs;
    ptrs.reserve(num_experts_ * 6);
    auto append = [&ptrs](const std::vector<infinicore::Tensor> &tensors) {
        for (const auto &tensor : tensors) {
            ptrs.push_back(tensor->data());
        }
    };
    append(gate_weights_);
    append(up_weights_);
    append(down_weights_);
    append(gate_weight_scales_);
    append(up_weight_scales_);
    append(down_weight_scales_);

    const size_t bytes = ptrs.size() * sizeof(void *);
    auto host = infinicore::Tensor::empty({bytes}, infinicore::DataType::U8, infinicore::Device::Type::CPU, true);
    std::memcpy(host->data(), ptrs.data(), bytes);
    return host->to(device);
}

infinicore::Tensor DeepseekV4Experts::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &top_k_index,
                                              const infinicore::Tensor &top_k_weights) const {
    ASSERT(hidden_states->ndim() == 2 && hidden_states->shape()[1] == hidden_size_);
    ASSERT(has_w8a8_weights_());
    ASSERT(supports_fused_deepseek_moe(hidden_states->device().getType()));
    ASSERT(top_k_index->dtype() == infinicore::DataType::I32);
    ASSERT(top_k_weights->dtype() == infinicore::DataType::F32);
    ASSERT(hidden_states->dtype() == infinicore::DataType::BF16 || hidden_states->dtype() == infinicore::DataType::F16);

    if (!expert_ptr_tables_.empty()) {
        return infinicore::op::deepseek_moe_w8a8i8_with_ptr_tables(hidden_states, top_k_index, top_k_weights,
                                                                   expert_ptr_tables_,
                                                                   local_moe_intermediate_size_, num_experts_);
    }
    return infinicore::op::deepseek_moe_w8a8i8(hidden_states, top_k_index, top_k_weights,
                                               gate_weights_, up_weights_, down_weights_,
                                               gate_weight_scales_, up_weight_scales_, down_weight_scales_,
                                               local_moe_intermediate_size_, num_experts_);
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device)
    : DeepseekV4MoE(std::move(model_config), 0, device) {
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t layer_idx,
                             const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      layer_idx_(layer_idx) {
    const size_t moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    const size_t num_shared_experts = model_config->get_or<size_t>("n_shared_experts", 0);
    const size_t num_hash_layers = model_config->get_or<size_t>("num_hash_layers", 0);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;

    if (layer_idx < num_hash_layers) {
        topk_ = this->register_module<DeepseekV4HashTopK>("gate", model_config, layer_idx, device);
    } else {
        topk_ = this->register_module<DeepseekV4TopK>("gate", model_config, layer_idx, device);
    }
    INFINICORE_NN_MODULE_INIT(experts, model_config, layer_idx_, device);
    has_shared_experts_ = num_shared_experts > 0;
    if (num_shared_experts > 0) {
        INFINICORE_NN_MODULE_INIT(shared_experts, model_config, moe_intermediate_size * num_shared_experts, device);
    }

    f32_allreduce_ = env_flag("DSV4_FFN_F32_ALLREDUCE");
}

infinicore::Tensor DeepseekV4MoE::forward(const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &input_ids) const {
    const auto shape = hidden_states->shape();
    if (shape.size() != 3 || shape[2] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE: expected hidden_states shape [B,S,D]");
    }
    auto hidden_flat = hidden_states->view({shape[0] * shape[1], hidden_size_});
    infinicore::Tensor topk_weights;
    infinicore::Tensor topk_ids;
    if (std::holds_alternative<std::shared_ptr<DeepseekV4HashTopK>>(topk_)) {
        std::tie(topk_weights, topk_ids) = std::get<std::shared_ptr<DeepseekV4HashTopK>>(topk_)->forward(hidden_flat, input_ids);
    } else {
        std::tie(topk_weights, topk_ids) = std::get<std::shared_ptr<DeepseekV4TopK>>(topk_)->forward(hidden_flat);
    }
    auto final_hidden_states = experts_->forward(hidden_flat, topk_ids, topk_weights);
    if (has_shared_experts_) {
        final_hidden_states = infinicore::op::add(final_hidden_states, shared_experts_->forward_without_allreduce(hidden_flat));
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {

        if (f32_allreduce_) {

            const auto local_values = tensor_to_float_vector(final_hidden_states);
            auto reduced = float_vector_to_tensor(local_values, final_hidden_states->shape(),
                                                  infinicore::DataType::F32, final_hidden_states->device());
            infinicore::op::distributed::allreduce_(reduced, reduced, INFINICCL_SUM, communicator_);
            final_hidden_states = float_vector_to_tensor(tensor_to_float_vector(reduced), final_hidden_states->shape(),
                                                         final_hidden_states->dtype(), final_hidden_states->device());
        } else {
            infinicore::op::distributed::allreduce_(final_hidden_states, final_hidden_states, INFINICCL_SUM, communicator_);
        }
    }
    return final_hidden_states->view(shape);
}

// infinicore::Tensor DeepseekV4MoE::forward(const infinicore::Tensor &hidden_states,
//                                           const infinicore::Tensor &input_ids) const {
//     const auto shape = hidden_states->shape();
//     if (shape.size() != 3 || shape[2] != hidden_size_) {
//         throw std::runtime_error("DeepseekV4MoE: expected hidden_states shape [B,S,D]");
//     }
//     auto hidden_flat = hidden_states->view({shape[0] * shape[1], hidden_size_});
//     profile::ScopedTimer moe_timer(profile::Event::MoeForward);
//     infinicore::Tensor topk_weights;
//     infinicore::Tensor topk_ids;
//     {
//         profile::ScopedTimer timer(profile::Event::MoeTopk);
//         if (std::holds_alternative<std::shared_ptr<DeepseekV4HashTopK>>(topk_)) {
//             std::tie(topk_weights, topk_ids) = std::get<std::shared_ptr<DeepseekV4HashTopK>>(topk_)->forward(hidden_flat, input_ids);
//         } else {
//             std::tie(topk_weights, topk_ids) = std::get<std::shared_ptr<DeepseekV4TopK>>(topk_)->forward(hidden_flat);
//         }
//     }
//     infinicore::Tensor final_hidden_states;
//     {
//         profile::ScopedTimer timer(profile::Event::MoeExperts);
//         final_hidden_states = experts_->forward(hidden_flat, topk_ids, topk_weights);
//     }
//     if (has_shared_experts_) {
//         profile::ScopedTimer timer(profile::Event::MoeSharedExperts);
//         final_hidden_states = infinicore::op::add(final_hidden_states, shared_experts_->forward_without_allreduce(hidden_flat));
//     }
//     if (tp_size_ > 1 && communicator_ != nullptr) {
//         profile::ScopedTimer timer(profile::Event::MoeAllReduce);

//         if (f32_allreduce_) {

//             const auto local_values = tensor_to_float_vector(final_hidden_states);
//             auto reduced = float_vector_to_tensor(local_values, final_hidden_states->shape(),
//                                                   infinicore::DataType::F32, final_hidden_states->device());
//             infinicore::op::distributed::allreduce_(reduced, reduced, INFINICCL_SUM, communicator_);
//             final_hidden_states = float_vector_to_tensor(tensor_to_float_vector(reduced), final_hidden_states->shape(),
//                                                          final_hidden_states->dtype(), final_hidden_states->device());
//         } else {
//             infinicore::op::distributed::allreduce_(final_hidden_states, final_hidden_states, INFINICCL_SUM, communicator_);
//         }
//     }
//     return final_hidden_states->view(shape);
// }

} // namespace infinilm::models::deepseek_v4
