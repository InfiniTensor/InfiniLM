#include "ernie4_5_moe.hpp"

#include "../../config/model_config.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {
namespace {

size_t json_first_size(const nlohmann::json &value, size_t fallback) {
    if (value.is_array() && !value.empty()) {
        return value.at(0).get<size_t>();
    }
    if (value.is_number_unsigned() || value.is_number_integer()) {
        return value.get<size_t>();
    }
    return fallback;
}

size_t json_size_at(const nlohmann::json &value, size_t index, size_t fallback) {
    if (value.is_array() && index < value.size()) {
        return value.at(index).get<size_t>();
    }
    if (index == 0 && (value.is_number_unsigned() || value.is_number_integer())) {
        return value.get<size_t>();
    }
    return fallback;
}

std::shared_ptr<infinilm::config::ModelConfig>
clone_with_moe_intermediate(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                            size_t moe_intermediate_size) {
    auto json = model_config->get_config_json();
    json["moe_intermediate_size"] = moe_intermediate_size;
    return std::make_shared<infinilm::config::ModelConfig>(json);
}

size_t tensor_offset_1d(const infinicore::Tensor &tensor, size_t index) {
    ASSERT(tensor->ndim() == 1);
    const auto stride = tensor->stride(0);
    ASSERT(stride >= 0);
    return index * static_cast<size_t>(stride);
}

size_t tensor_offset_2d(const infinicore::Tensor &tensor, size_t row, size_t col) {
    ASSERT(tensor->ndim() == 2);
    const auto row_stride = tensor->stride(0);
    const auto col_stride = tensor->stride(1);
    ASSERT(row_stride >= 0 && col_stride >= 0);
    return row * static_cast<size_t>(row_stride) + col * static_cast<size_t>(col_stride);
}

float read_as_f32_offset(const infinicore::Tensor &tensor, size_t offset) {
    const auto dtype = tensor->dtype();
    const auto *data = tensor->data();
    if (dtype == infinicore::DataType::F32) {
        return reinterpret_cast<const float *>(data)[offset];
    }
    if (dtype == infinicore::DataType::F16) {
        return f16_to_f32(reinterpret_cast<const uint16_t *>(data)[offset]);
    }
    if (dtype == infinicore::DataType::BF16) {
        return bf16_to_f32(reinterpret_cast<const uint16_t *>(data)[offset]);
    }
    ASSERT(false);
    return 0.0f;
}

float read_as_f32_1d(const infinicore::Tensor &tensor, size_t index) {
    return read_as_f32_offset(tensor, tensor_offset_1d(tensor, index));
}

float read_as_f32_2d(const infinicore::Tensor &tensor, size_t row, size_t col) {
    return read_as_f32_offset(tensor, tensor_offset_2d(tensor, row, col));
}

int64_t read_as_i64_offset(const infinicore::Tensor &tensor, size_t offset) {
    const auto dtype = tensor->dtype();
    const auto *data = tensor->data();
    if (dtype == infinicore::DataType::I64) {
        return reinterpret_cast<const int64_t *>(data)[offset];
    }
    if (dtype == infinicore::DataType::I32) {
        return reinterpret_cast<const int32_t *>(data)[offset];
    }
    ASSERT(false);
    return 0;
}

int64_t read_as_i64_1d(const infinicore::Tensor &tensor, size_t index) {
    return read_as_i64_offset(tensor, tensor_offset_1d(tensor, index));
}

int64_t read_as_i64_2d(const infinicore::Tensor &tensor, size_t row, size_t col) {
    return read_as_i64_offset(tensor, tensor_offset_2d(tensor, row, col));
}

} // namespace

Ernie4_5MoeTopKRouter::Ernie4_5MoeTopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &json = model_config->get_config_json();
    const auto experts_json = json.value("moe_num_experts", nlohmann::json(64));
    num_experts_ = json_first_size(experts_json, 64);
    num_expert_groups_ = experts_json.is_array() ? experts_json.size() : 1;
    num_experts_per_tok_ = model_config->get_or<size_t>("moe_k", 6);
    norm_topk_prob_ = model_config->get_or<bool>("moe_norm_gate_logits", true);
    use_correction_bias_ = model_config->get_or<bool>("moe_use_aux_free", false);
    ASSERT((num_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts_));
    ASSERT(num_expert_groups_ >= 1);
    ASSERT(num_expert_groups_ <= 2);
    if (num_expert_groups_ > 1) {
        ASSERT_EQ(json_size_at(experts_json, 1, num_experts_), num_experts_);
    }

    const auto &dtype{model_config->get_dtype()};
    const bool use_gpu_router = std::getenv("INFINILM_ERNIE_GPU_ROUTER") != nullptr;
    const auto gate_dtype = use_gpu_router ? dtype : infinicore::DataType::F32;
    INFINICORE_NN_PARAMETER_INIT(weight, ({num_experts_, hidden_size}, gate_dtype, device));
    if (num_expert_groups_ > 1) {
        INFINICORE_NN_PARAMETER_INIT(weight_1, ({num_experts_, hidden_size}, gate_dtype, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
Ernie4_5MoeTopKRouter::forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &score_correction_bias,
                               const std::vector<size_t> *token_groups) const {
    ASSERT(hidden_states->ndim() == 2);

    size_t ntoken = hidden_states->shape()[0];
    size_t hidden_size = hidden_states->shape()[1];
    auto correction_bias_cpu = use_correction_bias_
                                 ? score_correction_bias->to(infinicore::Device::Type::CPU)
                                 : infinicore::Tensor();
    if (use_correction_bias_) {
        ASSERT(correction_bias_cpu);
        ASSERT(correction_bias_cpu->numel() >= num_experts_);
    }
    if (token_groups != nullptr) {
        ASSERT_EQ(token_groups->size(), ntoken);
    }

    auto router_scores_cpu = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::F32, infinicore::Device::Type::CPU);
    auto router_indices_cpu = infinicore::Tensor::empty({ntoken, num_experts_per_tok_}, infinicore::DataType::I32, infinicore::Device::Type::CPU);
    auto *router_scores_ptr = reinterpret_cast<float *>(router_scores_cpu->data());
    auto *router_indices_ptr = reinterpret_cast<int *>(router_indices_cpu->data());

    std::vector<float> logits(num_experts_);
    std::vector<float> probs(num_experts_);
    std::vector<size_t> expert_order(num_experts_);
    std::iota(expert_order.begin(), expert_order.end(), 0);

    infinicore::Tensor router_logits_cpu;
    infinicore::Tensor hidden_states_cpu;
    infinicore::Tensor weight_cpu;
    infinicore::Tensor weight_1_cpu;
    const bool use_gpu_gate_linear = token_groups == nullptr && std::getenv("INFINILM_ERNIE_GPU_ROUTER") != nullptr;
    if (use_gpu_gate_linear) {
        ASSERT(weight_->ndim() == 2);
        ASSERT(weight_->shape()[0] == num_experts_);
        ASSERT(weight_->shape()[1] == hidden_size);
        auto router_logits = infinicore::op::linear(hidden_states, weight_, std::nullopt, 1.0f);
        router_logits_cpu = router_logits->to(infinicore::Device::Type::CPU);
        ASSERT(router_logits_cpu->ndim() == 2);
        ASSERT(router_logits_cpu->shape()[0] == ntoken);
        ASSERT(router_logits_cpu->shape()[1] == num_experts_);
    } else {
        hidden_states_cpu = hidden_states->to(infinicore::Device::Type::CPU);
        weight_cpu = weight_->to(infinicore::Device::Type::CPU);
        weight_1_cpu = num_expert_groups_ > 1 ? weight_1_->to(infinicore::Device::Type::CPU) : infinicore::Tensor();
        ASSERT(weight_cpu->ndim() == 2);
        ASSERT(weight_cpu->shape()[0] == num_experts_);
        ASSERT(weight_cpu->shape()[1] == hidden_size);
        if (num_expert_groups_ > 1) {
            ASSERT(weight_1_cpu->ndim() == 2);
            ASSERT(weight_1_cpu->shape()[0] == num_experts_);
            ASSERT(weight_1_cpu->shape()[1] == hidden_size);
        }
    }

    for (size_t itok = 0; itok < ntoken; ++itok) {
        const size_t expert_group = token_groups == nullptr ? 0 : (*token_groups)[itok];
        ASSERT(expert_group < num_expert_groups_);

        if (use_gpu_gate_linear) {
            for (size_t iexpert = 0; iexpert < num_experts_; ++iexpert) {
                logits[iexpert] = read_as_f32_2d(router_logits_cpu, itok, iexpert);
            }
        } else {
            const auto &gate_weight = expert_group == 0 ? weight_cpu : weight_1_cpu;

            for (size_t iexpert = 0; iexpert < num_experts_; ++iexpert) {
                float acc = 0.0f;
                for (size_t ihidden = 0; ihidden < hidden_size; ++ihidden) {
                    acc += read_as_f32_2d(hidden_states_cpu, itok, ihidden) * read_as_f32_2d(gate_weight, iexpert, ihidden);
                }
                logits[iexpert] = acc;
            }
        }

        float max_logit = logits[0];
        for (size_t iexpert = 1; iexpert < num_experts_; ++iexpert) {
            max_logit = std::max(max_logit, logits[iexpert]);
        }

        float denom = 0.0f;
        for (size_t iexpert = 0; iexpert < num_experts_; ++iexpert) {
            probs[iexpert] = std::exp(logits[iexpert] - max_logit);
            denom += probs[iexpert];
        }
        if (!std::isfinite(max_logit) || !std::isfinite(denom) || denom <= 0.0f) {
            SPDLOG_ERROR("ERNIE MoE router invalid CPU softmax: token={} max_logit={} denom={} hidden_dtype={} weight_dtype={}",
                         itok,
                         max_logit,
                         denom,
                         static_cast<int>(use_gpu_gate_linear ? hidden_states->dtype() : hidden_states_cpu->dtype()),
                         static_cast<int>(use_gpu_gate_linear ? weight_->dtype() : weight_cpu->dtype()));
            for (size_t i = 0; i < std::min<size_t>(num_experts_, 8); ++i) {
                SPDLOG_ERROR("ERNIE MoE router cpu_logit[{},{}]={}", itok, i, logits[i]);
            }
            if (!use_gpu_gate_linear) {
                for (size_t i = 0; i < std::min<size_t>(hidden_states_cpu->shape()[1], 8); ++i) {
                    SPDLOG_ERROR("ERNIE MoE router hidden[{},{}]={}", itok, i, read_as_f32_2d(hidden_states_cpu, itok, i));
                }
            }
        }
        ASSERT(denom > 0.0f);
        for (float &prob : probs) {
            prob /= denom;
        }

        auto route_score = [&](size_t expert) {
            float score = probs[expert];
            if (use_correction_bias_) {
                score += correction_bias_cpu->ndim() == 1
                           ? read_as_f32_1d(correction_bias_cpu, expert)
                           : read_as_f32_2d(correction_bias_cpu, expert_group, expert);
            }
            return score;
        };
        std::iota(expert_order.begin(), expert_order.end(), 0);
        std::partial_sort(
            expert_order.begin(),
            expert_order.begin() + num_experts_per_tok_,
            expert_order.end(),
            [&](size_t lhs, size_t rhs) {
                const float lhs_score = route_score(lhs);
                const float rhs_score = route_score(rhs);
                if (lhs_score == rhs_score) {
                    return lhs < rhs;
                }
                return lhs_score > rhs_score;
            });

        float selected_sum = 0.0f;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            selected_sum += probs[expert_order[k]];
        }
        if (norm_topk_prob_) {
            selected_sum = std::max(selected_sum, 1e-12f);
        }

        const size_t topk_offset = itok * num_experts_per_tok_;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            const size_t expert = expert_order[k];
            router_indices_ptr[topk_offset + k] = static_cast<int>(expert + expert_group * num_experts_);
            router_scores_ptr[topk_offset + k] = norm_topk_prob_ ? (probs[expert] / selected_sum) : probs[expert];
        }
    }

    return std::make_tuple(router_scores_cpu->to(hidden_states->device()),
                           router_indices_cpu->to(hidden_states->device()));
}

Ernie4_5MoeStatics::Ernie4_5MoeStatics(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    const auto &json = model_config->get_config_json();
    size_t num_experts = json_first_size(json.value("moe_num_experts", nlohmann::json(64)), 64);
    size_t num_groups = 1;
    if (json.contains("moe_num_experts") && json["moe_num_experts"].is_array()) {
        num_groups = json["moe_num_experts"].size();
    }
    INFINICORE_NN_PARAMETER_INIT(e_score_correction_bias, ({num_groups, num_experts}, infinicore::DataType::F32, device));
}

Ernie4_5TextMoeBlock::Ernie4_5TextMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    const auto &json = model_config->get_config_json();
    const auto experts_json = json.value("moe_num_experts", nlohmann::json(64));
    const auto intermediate_json = json.value("moe_intermediate_size", nlohmann::json(1536));
    num_experts_ = json_first_size(experts_json, 64);
    num_expert_groups_ = experts_json.is_array() ? experts_json.size() : 1;
    num_experts_per_tok_ = model_config->get_or<size_t>("moe_k", 6);
    size_t num_shared_experts = model_config->get_or<size_t>("moe_num_shared_experts", 0);
    ASSERT(num_expert_groups_ >= 1);
    ASSERT(num_expert_groups_ <= 2);
    for (size_t group = 1; group < num_expert_groups_; ++group) {
        ASSERT_EQ(json_size_at(experts_json, group, num_experts_), num_experts_);
    }

    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(moe_statics, model_config, device);

    experts_.reserve(num_experts_ * num_expert_groups_);
    size_t expert_idx = 0;
    for (size_t group = 0; group < num_expert_groups_; ++group) {
        const size_t intermediate_size = json_size_at(intermediate_json, group, json_first_size(intermediate_json, 1536));
        auto expert_config = clone_with_moe_intermediate(model_config, intermediate_size);
        for (size_t i = 0; i < num_experts_; ++i) {
            experts_.push_back(this->register_module<infinilm::layers::moe::legacy::MoeMLP>(
                "experts." + std::to_string(expert_idx), expert_config, device));
            ++expert_idx;
        }
    }

    if (num_shared_experts > 0) {
        const size_t intermediate_size = json_first_size(intermediate_json, 1536);
        auto shared_config = clone_with_moe_intermediate(model_config, intermediate_size * num_shared_experts);
        INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
    }
}

std::vector<size_t>
Ernie4_5TextMoeBlock::token_groups_(const infinicore::Tensor &token_type_ids,
                                    const std::vector<size_t> &hidden_shape) const {
    std::vector<size_t> groups;
    if (!token_type_ids || num_expert_groups_ <= 1) {
        return groups;
    }

    const bool restore_3d = hidden_shape.size() == 3;
    const size_t batch = restore_3d ? hidden_shape[0] : 1;
    const size_t seq_len = restore_3d ? hidden_shape[1] : hidden_shape[0];
    const size_t ntoken = batch * seq_len;

    auto token_types_cpu = token_type_ids->to(infinicore::Device::Type::CPU);
    const auto tt_shape = token_types_cpu->shape();
    groups.assign(ntoken, 0);
    bool any_multimodal = false;

    if (restore_3d && tt_shape.size() == 2) {
        ASSERT_EQ(tt_shape[0], batch);
        ASSERT(tt_shape[1] >= seq_len);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                const int64_t token_type = read_as_i64_2d(token_types_cpu, b, s);
                const size_t group = token_type == 0 ? 0 : 1;
                groups[b * seq_len + s] = group;
                any_multimodal = any_multimodal || group != 0;
            }
        }
    } else {
        ASSERT(token_types_cpu->numel() >= ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            const int64_t token_type = token_types_cpu->ndim() == 1
                                         ? read_as_i64_1d(token_types_cpu, i)
                                         : read_as_i64_offset(token_types_cpu, i);
            const size_t group = token_type == 0 ? 0 : 1;
            groups[i] = group;
            any_multimodal = any_multimodal || group != 0;
        }
    }

    if (!any_multimodal) {
        groups.clear();
    }
    return groups;
}

infinicore::Tensor Ernie4_5TextMoeBlock::forward(const infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &token_type_ids) const {
    ASSERT((hidden_states->ndim() == 2) || (hidden_states->ndim() == 3));

    auto shape = hidden_states->shape();
    bool restore_3d = hidden_states->ndim() == 3;
    auto hidden_states_2d = restore_3d ? hidden_states->view({shape[0] * shape[1], shape[2]}) : hidden_states;

    auto token_groups = token_groups_(token_type_ids, shape);
    auto [routing_weights, selected_experts] = gate_->forward(
        hidden_states_2d,
        moe_statics_->e_score_correction_bias(),
        token_groups.empty() ? nullptr : &token_groups);
    auto routing_weights_cpu = routing_weights->to(infinicore::Device::Type::CPU);
    auto selected_experts_cpu = selected_experts->to(infinicore::Device::Type::CPU);

    float *routing_weights_ptr = reinterpret_cast<float *>(routing_weights_cpu->data());
    int *selected_experts_ptr = reinterpret_cast<int *>(selected_experts_cpu->data());

    size_t ntoken = hidden_states_2d->shape()[0];
    auto final_hidden_states = infinicore::Tensor::empty(hidden_states_2d->shape(), hidden_states_2d->dtype(), hidden_states_2d->device());
    infinicore::Tensor shared_out;
    if (shared_experts_) {
        shared_out = shared_experts_->forward(hidden_states_2d);
    }

    for (size_t itok = 0; itok < ntoken; ++itok) {
        auto hidden_states_i = hidden_states_2d->narrow({{0, itok, 1}});
        const size_t route_row = itok * num_experts_per_tok_;

        infinicore::Tensor final_hidden_states_i;
        for (size_t k = 0; k < num_experts_per_tok_; ++k) {
            int index = selected_experts_ptr[route_row + k];
            float score = routing_weights_ptr[route_row + k];
            ASSERT(index >= 0 && static_cast<size_t>(index) < experts_.size());

            experts_[index]->set_alpha(score);
            auto expert_out = experts_[index]->forward(hidden_states_i);
            if (k == 0) {
                final_hidden_states_i = expert_out;
            } else {
                infinicore::op::add_(final_hidden_states_i, final_hidden_states_i, expert_out);
            }
        }

        if (shared_out) {
            auto shared_i = shared_out->narrow({{0, itok, 1}});
            infinicore::op::add_(final_hidden_states_i, final_hidden_states_i, shared_i);
        }
        final_hidden_states->narrow({{0, itok, 1}})->copy_from(final_hidden_states_i);
    }

    if (restore_3d) {
        return final_hidden_states->view({shape[0], shape[1], shape[2]});
    }
    return final_hidden_states;
}

} // namespace infinilm::models::ernie4_5_moe_vl
