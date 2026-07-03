#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#include "spdlog/spdlog.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

std::vector<float> softmax_row(const std::vector<float> &logits, size_t row_offset, size_t width) {
    float max_value = logits[row_offset];
    for (size_t i = 1; i < width; ++i) {
        max_value = std::max(max_value, logits[row_offset + i]);
    }
    std::vector<float> out(width);
    double denom = 0.0;
    for (size_t i = 0; i < width; ++i) {
        out[i] = std::exp(logits[row_offset + i] - max_value);
        denom += out[i];
    }
    for (auto &value : out) {
        value = static_cast<float>(value / denom);
    }
    return out;
}

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


std::vector<float> router_logits_fp32_reference(const infinicore::Tensor &hidden_states,
                                                const infinicore::Tensor &weight,
                                                size_t ntoken,
                                                size_t hidden_size,
                                                size_t num_experts) {
    const auto hidden_values = tensor_to_float_vector(hidden_states);
    const auto weight_values = tensor_to_float_vector(weight);
    if (hidden_values.size() != ntoken * hidden_size
        || weight_values.size() != num_experts * hidden_size) {
        throw std::runtime_error("DeepseekV4Gate: router logits shape mismatch");
    }

    std::vector<float> logits(ntoken * num_experts, 0.0f);
    for (size_t token = 0; token < ntoken; ++token) {
        const size_t hidden_offset = token * hidden_size;
        for (size_t expert = 0; expert < num_experts; ++expert) {
            const size_t weight_offset = expert * hidden_size;
            float acc = 0.0f;
            for (size_t hidden = 0; hidden < hidden_size; ++hidden) {
                acc += hidden_values[hidden_offset + hidden] * weight_values[weight_offset + hidden];
            }
            logits[token * num_experts + expert] = acc;
        }
    }
    return logits;
}

infinicore::Tensor int32_vector_to_tensor(const std::vector<int64_t> &values,
                                          const infinicore::Shape &shape,
                                          const infinicore::Device &device) {
    size_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (values.size() != numel) {
        throw std::runtime_error("DeepseekV4MoE: int32_vector_to_tensor shape mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, infinicore::DataType::I32, infinicore::Device::cpu());
    auto *out = reinterpret_cast<int32_t *>(cpu->data());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<int32_t>(values[i]);
    }
    auto tensor = cpu->to(device);
    if (device.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }
    return tensor;
}

} // namespace

DeepseekV4Gate::DeepseekV4Gate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device)
    : DeepseekV4Gate(std::move(model_config), 0, device) {
}

DeepseekV4Gate::DeepseekV4Gate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device)
    : hidden_size_(model_config->get<size_t>("hidden_size")),
      num_experts_(model_config->get<size_t>("n_routed_experts")),
      num_experts_per_tok_(model_config->get<size_t>("num_experts_per_tok")),
      routed_scaling_(static_cast<float>(model_config->get_or<double>("routed_scaling_factor", 1.0))),
      scoring_func_(model_config->get_or<std::string>("scoring_func", "sqrtsoftplus")),
      norm_topk_prob_(model_config->get_or<bool>("norm_topk_prob", true)),
      layer_idx_(layer_idx) {
    const auto &dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_hash_layers = model_config->get_or<size_t>("num_hash_layers", 0);
    const std::string topk_method = model_config->get_or<std::string>("topk_method", "");

    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size_}, dtype, device));
    if (layer_idx < num_hash_layers) {
        INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab_size, num_experts_per_tok_}, infinicore::DataType::I64, device));
    } else if (topk_method == "noaux_tc") {
        INFINICORE_NN_PARAMETER_INIT(bias, ({num_experts_}, infinicore::DataType::F32, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekV4Gate::forward(const infinicore::Tensor &hidden_states,
                        const infinicore::Tensor &input_ids) const {
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4Gate: expected hidden_states shape [N,D]");
    }
    const size_t ntoken = hidden_states->shape()[0];

    // SGLang computes router logits with BF16 inputs and FP32 accumulation.
    // Keep this reference path for precision alignment of DeepSeek-V4 routing.
    auto logits_values = router_logits_fp32_reference(hidden_states, weight_, ntoken, hidden_size_, num_experts_);
    auto scores = logits_values;
    for (size_t token = 0; token < ntoken; ++token) {
        const size_t offset = token * num_experts_;
        if (scoring_func_ == "softmax") {
            auto row = softmax_row(logits_values, offset, num_experts_);
            std::copy(row.begin(), row.end(), scores.begin() + offset);
        } else if (scoring_func_ == "sigmoid") {
            for (size_t i = 0; i < num_experts_; ++i) {
                scores[offset + i] = sigmoid(logits_values[offset + i]);
            }
        } else {
            for (size_t i = 0; i < num_experts_; ++i) {
                scores[offset + i] = std::sqrt(std::log1p(std::exp(-std::abs(logits_values[offset + i])))
                                               + std::max(logits_values[offset + i], 0.0f));
            }
        }
    }

    std::vector<int64_t> token_ids;
    std::vector<int64_t> tid2eid;
    if (has_tid2eid()) {
        if (input_ids.empty()) {
            throw std::runtime_error("DeepseekV4Gate: hash-routed layer requires input_ids");
        }
        token_ids = tensor_to_int64_vector(input_ids);
        if (token_ids.size() != ntoken) {
            throw std::runtime_error("DeepseekV4Gate: input_ids size mismatch");
        }
        tid2eid = tensor_to_int64_vector(tid2eid_);
    }

    std::vector<float> bias;
    if (has_bias()) {
        bias = tensor_to_float_vector(bias_);
    }

    std::vector<int64_t> selected(ntoken * num_experts_per_tok_, 0);
    std::vector<float> weights(ntoken * num_experts_per_tok_, 0.0f);
    for (size_t token = 0; token < ntoken; ++token) {
        const size_t score_offset = token * num_experts_;
        const size_t route_offset = token * num_experts_per_tok_;
        if (has_tid2eid()) {
            const int64_t token_id = token_ids[token];
            if (token_id < 0 || static_cast<size_t>(token_id) * num_experts_per_tok_ + num_experts_per_tok_ > tid2eid.size()) {
                throw std::runtime_error("DeepseekV4Gate: token id out of tid2eid range");
            }
            for (size_t k = 0; k < num_experts_per_tok_; ++k) {
                selected[route_offset + k] = tid2eid[static_cast<size_t>(token_id) * num_experts_per_tok_ + k];
            }
        } else {
            std::vector<std::pair<float, int64_t>> ranked;
            ranked.reserve(num_experts_);
            for (size_t expert = 0; expert < num_experts_; ++expert) {
                const float biased = scores[score_offset + expert] + (bias.empty() ? 0.0f : bias[expert]);
                ranked.emplace_back(biased, static_cast<int64_t>(expert));
            }
            std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(num_experts_per_tok_), ranked.end(),
                              [](const auto &a, const auto &b) {
                                  if (a.first == b.first) {
                                      return a.second < b.second;
                                  }
                                  return a.first > b.first;
                              });
            for (size_t k = 0; k < num_experts_per_tok_; ++k) {
                selected[route_offset + k] = ranked[k].second;
            }
        }

        float denom = 0.0f;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            const int64_t expert = selected[route_offset + k];
            if (expert < 0 || static_cast<size_t>(expert) >= num_experts_) {
                throw std::runtime_error("DeepseekV4Gate: selected expert id out of range");
            }
            weights[route_offset + k] = scores[score_offset + static_cast<size_t>(expert)];
            denom += weights[route_offset + k];
        }
        if (norm_topk_prob_ && scoring_func_ != "softmax") {
            denom += 1e-9f;
            for (size_t k = 0; k < num_experts_per_tok_; ++k) {
                weights[route_offset + k] /= denom;
            }
        }
        // Match the current SGLang slimquant_marlin reference: keep top-k
        // routing probabilities normalized here and do not bake
        // routed_scaling_factor into the gate weights.
        (void)routed_scaling_;
    }    auto top_k_weights = float_vector_to_tensor(weights, {ntoken, num_experts_per_tok_},
                                                infinicore::DataType::F32, hidden_states->device());
    auto top_k_index = int32_vector_to_tensor(selected, {ntoken, num_experts_per_tok_},
                                              hidden_states->device());
    return {top_k_weights, top_k_index};
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
    use_fused_moe_ = !(model_config->get_config_json().contains("swiglu_limit") && !model_config->get_ref("swiglu_limit").is_null());

    experts_.reserve(num_experts_);
    gate_weights_.reserve(num_experts_);
    up_weights_.reserve(num_experts_);
    down_weights_.reserve(num_experts_);
    for (size_t i = 0; i < num_experts_; ++i) {
        auto expert = this->register_module<DeepseekV4MLP>(std::to_string(i), model_config, moe_intermediate_size_, device);
        gate_weights_.push_back(expert->gate_weight());
        up_weights_.push_back(expert->up_weight());
        down_weights_.push_back(expert->down_weight());
        experts_.push_back(std::move(expert));
    }
    local_moe_intermediate_size_ = gate_weights_.empty() ? moe_intermediate_size_ : gate_weights_.front()->shape()[0];
}

infinicore::Tensor DeepseekV4Experts::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &top_k_index,
                                              const infinicore::Tensor &top_k_weights) const {
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4Experts: expected hidden_states shape [N,D]");
    }
    const bool dense_weights =
        !gate_weights_.empty()
        && gate_weights_.front()->dtype() == hidden_states->dtype()
        && up_weights_.front()->dtype() == hidden_states->dtype()
        && down_weights_.front()->dtype() == hidden_states->dtype();
    if (use_fused_moe_
        && dense_weights
        && supports_fused_deepseek_moe(hidden_states->device().getType())
        && top_k_index->dtype() == infinicore::DataType::I32
        && top_k_weights->dtype() == infinicore::DataType::F32
        && (hidden_states->dtype() == infinicore::DataType::BF16
            || hidden_states->dtype() == infinicore::DataType::F16)) {
        try {
            auto output = infinicore::op::deepseek_moe(hidden_states, top_k_index, top_k_weights,
                                                       gate_weights_, up_weights_, down_weights_,
                                                       local_moe_intermediate_size_, num_experts_);
            return output;
        } catch (const std::exception &e) {
            spdlog::warn("DeepseekV4Experts: deepseek_moe unavailable on {}, falling back to CPU-routed experts: {}",
                         static_cast<int>(hidden_states->device().getType()), e.what());
        }
    }
    return forward_cpu_routed_(hidden_states, top_k_index, top_k_weights);
}

infinicore::Tensor DeepseekV4Experts::forward_cpu_routed_(const infinicore::Tensor &hidden_states,
                                                          const infinicore::Tensor &top_k_index,
                                                          const infinicore::Tensor &top_k_weights) const {
    const auto shape = hidden_states->shape();
    if (shape.size() != 2 || shape[1] != hidden_size_) {
        throw std::runtime_error("DeepseekV4Experts: expected hidden_states shape [N,D]");
    }
    const size_t ntoken = shape[0];
    auto selected = tensor_to_int64_vector(top_k_index);
    auto weights = tensor_to_float_vector(top_k_weights);
    auto final_hidden_states = infinicore::Tensor::empty({ntoken, hidden_size_}, hidden_states->dtype(), hidden_states->device());
    for (size_t token = 0; token < ntoken; ++token) {
        auto hidden_i = hidden_states->narrow({{0, token, 1}});
        std::vector<float> token_values(hidden_size_, 0.0f);
        const size_t route_offset = token * num_experts_per_tok_;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            const int64_t expert = selected[route_offset + k];
            if (expert < 0 || static_cast<size_t>(expert) >= num_experts_) {
                throw std::runtime_error("DeepseekV4Experts: selected expert id out of range");
            }
            const float route_weight = weights[route_offset + k];
            experts_[expert]->set_alpha(route_weight);
            auto expert_out = experts_[expert]->forward_without_allreduce(hidden_i);
            experts_[expert]->set_alpha(1.0f);
            auto expert_values = tensor_to_float_vector(expert_out);
            if (expert_values.size() != hidden_size_) {
                throw std::runtime_error("DeepseekV4Experts: expert output size mismatch");
            }
            for (size_t i = 0; i < hidden_size_; ++i) {
                token_values[i] += expert_values[i];
            }
        }
        auto token_out = float_vector_to_tensor(token_values, {1, hidden_size_}, hidden_states->dtype(), hidden_states->device());
        final_hidden_states->narrow({{0, token, 1}})->copy_from(token_out);
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
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    communicator_ = rank_info.comm;

    INFINICORE_NN_MODULE_INIT(gate, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    has_shared_experts_ = num_shared_experts > 0;
    if (num_shared_experts > 0) {
        INFINICORE_NN_MODULE_INIT(shared_experts, model_config, moe_intermediate_size * num_shared_experts, device);
    }
}

infinicore::Tensor DeepseekV4MoE::forward(const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &input_ids) const {
    const auto shape = hidden_states->shape();
    if (shape.size() != 3 || shape[2] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE: expected hidden_states shape [B,S,D]");
    }
    auto hidden_flat = hidden_states->view({shape[0] * shape[1], hidden_size_});
    auto [routing_weights, selected_experts] = gate_->forward(hidden_flat, input_ids);
    auto final_hidden_states = experts_->forward(hidden_flat, selected_experts, routing_weights);
    if (has_shared_experts_) {
        final_hidden_states = infinicore::op::add(
            final_hidden_states, shared_experts_->forward_without_allreduce(hidden_flat));
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {
        const char *f32_allreduce = std::getenv("DSV4_FFN_F32_ALLREDUCE");
        if (f32_allreduce != nullptr && std::string(f32_allreduce) == "1") {
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
