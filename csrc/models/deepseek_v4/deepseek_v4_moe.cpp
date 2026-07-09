#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
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

bool require_fused_moe_env() {
    static const bool value = env_flag("DSV4_REQUIRE_FUSED_MOE");
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
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      moe_intermediate_size_(model_config->get<size_t>("moe_intermediate_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;
    const auto &config_json = model_config->get_config_json();
    use_fused_moe_ = !(config_json.contains("swiglu_limit") && !config_json.at("swiglu_limit").is_null());
    if (const char *force_fused = std::getenv("DSV4_FORCE_FUSED_MOE"); force_fused != nullptr && std::string(force_fused) == "1") {
        use_fused_moe_ = true;
    }

    experts_.reserve(num_experts_);
    gate_weights_.reserve(num_experts_);
    up_weights_.reserve(num_experts_);
    down_weights_.reserve(num_experts_);
    gate_weight_scales_.reserve(num_experts_);
    up_weight_scales_.reserve(num_experts_);
    down_weight_scales_.reserve(num_experts_);
    for (size_t i = 0; i < num_experts_; ++i) {
        auto expert = this->register_module<DeepseekV4MLP>(std::to_string(i), model_config, moe_intermediate_size_, device);
        gate_weights_.push_back(expert->gate_weight());
        up_weights_.push_back(expert->up_weight());
        down_weights_.push_back(expert->down_weight());
        gate_weight_scales_.push_back(expert->gate_weight_scale());
        up_weight_scales_.push_back(expert->up_weight_scale());
        down_weight_scales_.push_back(expert->down_weight_scale());
        experts_.push_back(std::move(expert));
    }
    local_moe_intermediate_size_ = gate_weights_.empty() ? moe_intermediate_size_ : gate_weights_.front()->shape()[0];
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
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4Experts: expected hidden_states shape [N,D]");
    }
    const bool dense_weights = !gate_weights_.empty() && gate_weights_.front()->dtype() == hidden_states->dtype()
                            && up_weights_.front()->dtype() == hidden_states->dtype()
                            && down_weights_.front()->dtype() == hidden_states->dtype();
    const bool w8a8_weights = !gate_weights_.empty()
                           && gate_weights_.front()->dtype() == infinicore::DataType::I8
                           && up_weights_.front()->dtype() == infinicore::DataType::I8
                           && down_weights_.front()->dtype() == infinicore::DataType::I8
                           && !gate_weight_scales_.empty()
                           && !up_weight_scales_.empty()
                           && !down_weight_scales_.empty()
                           && gate_weight_scales_.front()->dtype() == infinicore::DataType::F32
                           && up_weight_scales_.front()->dtype() == infinicore::DataType::F32
                           && down_weight_scales_.front()->dtype() == infinicore::DataType::F32;

    const bool fused_device_dtype_ready = supports_fused_deepseek_moe(hidden_states->device().getType())
                                       && top_k_index->dtype() == infinicore::DataType::I32
                                       && top_k_weights->dtype() == infinicore::DataType::F32
                                       && (hidden_states->dtype() == infinicore::DataType::BF16
                                           || hidden_states->dtype() == infinicore::DataType::F16);
    const bool fused_common_ready = use_fused_moe_ && fused_device_dtype_ready;
    const bool require_fused_moe = require_fused_moe_env();

    if (fused_device_dtype_ready && w8a8_weights) {
        try {
            if (std::getenv("DSV4_DISABLE_LAYER_PTR_TABLES") != nullptr) {
                return infinicore::op::deepseek_moe_w8a8i8(hidden_states, top_k_index, top_k_weights,
                                                           gate_weights_, up_weights_, down_weights_,
                                                           gate_weight_scales_, up_weight_scales_, down_weight_scales_,
                                                           local_moe_intermediate_size_, num_experts_);
            }
            if (expert_ptr_tables_.empty()) {
                expert_ptr_tables_ = build_w8a8_ptr_tables_(hidden_states->device());
            }
            return infinicore::op::deepseek_moe_w8a8i8_with_ptr_tables(hidden_states, top_k_index, top_k_weights,
                                                                       expert_ptr_tables_,
                                                                       local_moe_intermediate_size_, num_experts_);
        } catch (const std::exception &e) {
            if (require_fused_moe) {
                throw;
            }
            spdlog::warn("DeepseekV4Experts: deepseek_moe_w8a8i8 unavailable on {}, falling back to routed experts: {}", static_cast<int>(hidden_states->device().getType()), e.what());
        }
    }

    if (fused_common_ready && dense_weights) {
        try {
            return infinicore::op::deepseek_moe(hidden_states, top_k_index, top_k_weights,
                                                gate_weights_, up_weights_, down_weights_,
                                                local_moe_intermediate_size_, num_experts_);
        } catch (const std::exception &e) {
            if (require_fused_moe) {
                throw;
            }
            spdlog::warn("DeepseekV4Experts: deepseek_moe unavailable on {}, falling back to routed experts: {}", static_cast<int>(hidden_states->device().getType()), e.what());
        }
    }

    if (require_fused_moe) {
        throw std::runtime_error("DeepseekV4Experts: DSV4_REQUIRE_FUSED_MOE=1 but no fused MoE path matched");
    }
    return forward_cpu_routed_(hidden_states, top_k_index, top_k_weights);
}

infinicore::Tensor DeepseekV4Experts::forward_cpu_routed_(const infinicore::Tensor &hidden_states,
                                                          const infinicore::Tensor &top_k_index,
                                                          const infinicore::Tensor &top_k_weights) const {
    const size_t ntoken = hidden_states->shape()[0];
    auto top_k_weights_host = top_k_weights->to(infinicore::Device::Type::CPU);
    auto top_k_index_host = top_k_index->to(infinicore::Device::Type::CPU);
    if (hidden_states->device().getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    auto *top_k_index_ptr = reinterpret_cast<int32_t *>(top_k_index_host->data());
    auto *top_k_weights_ptr = reinterpret_cast<float *>(top_k_weights_host->data());

    auto final_hidden_states = infinicore::Tensor::empty(hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
    for (size_t token = 0; token < ntoken; ++token) {
        auto hidden_i = hidden_states->narrow({{0, token, 1}});
        const size_t route_offset = token * num_experts_per_tok_;

        infinicore::Tensor final_hidden_states_i;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            const int index = top_k_index_ptr[route_offset + k];
            const float score = top_k_weights_ptr[route_offset + k];
            ASSERT(index >= 0 && static_cast<size_t>(index) < num_experts_);
            experts_[index]->set_alpha(score);
            auto expert_out = experts_[index]->forward_without_allreduce(hidden_i);
            experts_[index]->set_alpha(1.0f);
            if (k == 0) {
                final_hidden_states_i = expert_out;
            } else {
                infinicore::op::add_(final_hidden_states_i, final_hidden_states_i, expert_out);
            }
        }
        final_hidden_states->narrow({{0, token, 1}})->copy_from(final_hidden_states_i);
    }
    return final_hidden_states;
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
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
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

} // namespace infinilm::models::deepseek_v4
