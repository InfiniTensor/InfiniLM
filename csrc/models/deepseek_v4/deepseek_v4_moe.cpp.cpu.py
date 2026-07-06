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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

float sqrtsoftplus(float logit) {
    return std::sqrt(std::log1p(std::exp(-std::abs(logit))) + std::max(logit, 0.0f));
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

bool force_deepseek_v4_router_cpu() {
    const char *flag = std::getenv("DSV4_ROUTER_CPU");
    return flag != nullptr && std::string(flag) == "1";
}

bool should_use_deepseek_v4_router_gpu(const infinicore::Device &device) {
    return device.getType() != infinicore::Device::Type::CPU && !force_deepseek_v4_router_cpu();
}

void warn_router_gpu_fallback_once(const char *router_name, const std::exception &e) {
    static bool warned_topk = false;
    static bool warned_hash = false;
    bool &warned = std::strcmp(router_name, "DeepseekV4HashRouter") == 0 ? warned_hash : warned_topk;
    if (!warned) {
        spdlog::warn("{}: DeepSeek V4 GPU router unavailable, falling back to CPU router: {}",
                     router_name, e.what());
        warned = true;
    }
}

void apply_sqrtsoftplus_scores(std::vector<float> &scores,
                               const std::vector<float> &logits,
                               size_t ntoken,
                               size_t num_experts) {
    scores.resize(logits.size());
    for (size_t token = 0; token < ntoken; ++token) {
        const size_t offset = token * num_experts;
        for (size_t i = 0; i < num_experts; ++i) {
            scores[offset + i] = sqrtsoftplus(logits[offset + i]);
        }
    }
}

void select_topk_experts(size_t ntoken,
                         size_t num_experts,
                         size_t num_experts_per_tok,
                         const std::vector<float> &scores,
                         const std::vector<float> &bias,
                         std::vector<int64_t> &selected) {
    selected.assign(ntoken * num_experts_per_tok, 0);

    for (size_t token = 0; token < ntoken; ++token) {
        const size_t score_offset = token * num_experts;
        const size_t route_offset = token * num_experts_per_tok;

        std::vector<std::pair<float, int64_t>> ranked;
        ranked.reserve(num_experts);
        for (size_t expert = 0; expert < num_experts; ++expert) {
            const float biased = scores[score_offset + expert] + (bias.empty() ? 0.0f : bias[expert]);
            ranked.emplace_back(biased, static_cast<int64_t>(expert));
        }
        std::partial_sort(ranked.begin(),
                          ranked.begin() + static_cast<std::ptrdiff_t>(num_experts_per_tok),
                          ranked.end(),
                          [](const auto &a, const auto &b) {
                              if (a.first == b.first) {
                                  return a.second < b.second;
                              }
                              return a.first > b.first;
                          });
        for (size_t k = 0; k < num_experts_per_tok; ++k) {
            selected[route_offset + k] = ranked[k].second;
        }
    }
}

void select_hash_experts(size_t ntoken,
                         size_t num_experts_per_tok,
                         const std::vector<int64_t> &token_ids,
                         const std::vector<int64_t> &tid2eid,
                         std::vector<int64_t> &selected) {
    selected.assign(ntoken * num_experts_per_tok, 0);

    for (size_t token = 0; token < ntoken; ++token) {
        const int64_t token_id = token_ids[token];
        const size_t route_offset = token * num_experts_per_tok;
        if (token_id < 0
            || static_cast<size_t>(token_id) * num_experts_per_tok + num_experts_per_tok > tid2eid.size()) {
            throw std::runtime_error("DeepseekV4HashRouter: token id out of tid2eid range");
        }
        for (size_t k = 0; k < num_experts_per_tok; ++k) {
            selected[route_offset + k] = tid2eid[static_cast<size_t>(token_id) * num_experts_per_tok + k];
        }
    }
}

void weights_from_selected(size_t ntoken,
                           size_t num_experts,
                           size_t num_experts_per_tok,
                           bool norm_topk_prob,
                           const std::vector<float> &scores,
                           const std::vector<int64_t> &selected,
                           std::vector<float> &weights) {
    weights.assign(ntoken * num_experts_per_tok, 0.0f);

    for (size_t token = 0; token < ntoken; ++token) {
        const size_t score_offset = token * num_experts;
        const size_t route_offset = token * num_experts_per_tok;
        float denom = 0.0f;
        for (size_t k = 0; k < num_experts_per_tok; ++k) {
            const int64_t expert = selected[route_offset + k];
            if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
                throw std::runtime_error("DeepseekV4MoE router: selected expert id out of range");
            }
            weights[route_offset + k] = scores[score_offset + static_cast<size_t>(expert)];
            denom += weights[route_offset + k];
        }
        if (norm_topk_prob) {
            denom += 1e-9f;
            for (size_t k = 0; k < num_experts_per_tok; ++k) {
                weights[route_offset + k] /= denom;
            }
        }
    }
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

void validate_router_hidden_states(const infinicore::Tensor &hidden_states, size_t hidden_size) {
    if (hidden_states->ndim() != 2 || hidden_states->shape()[1] != hidden_size) {
        throw std::runtime_error("DeepseekV4MoE router: expected hidden_states shape [N,D]");
    }
}

infinicore::Tensor compute_router_logits(const infinicore::Tensor &hidden_states,
                                         const infinicore::Tensor &weight) {
    return infinicore::op::linear(hidden_states, weight, std::nullopt, 1.0f);
}

} // namespace

DeepseekV4TopKRouter::DeepseekV4TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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
DeepseekV4TopKRouter::forward(const infinicore::Tensor &hidden_states,
                              const infinicore::Tensor &input_ids) const {
    (void)input_ids;
    validate_router_hidden_states(hidden_states, hidden_size_);
    const size_t ntoken = hidden_states->shape()[0];
    const auto device = hidden_states->device();

    auto logits = compute_router_logits(hidden_states, weight_);
    if (should_use_deepseek_v4_router_gpu(device)) {
        try {
            infinicore::Tensor router_bias;
            if (has_bias()) {
                router_bias = bias_;
            }
            auto [top_k_weights, top_k_index] = infinicore::op::deepseek_v4_topk_router(
                logits,
                num_experts_per_tok_,
                norm_topk_prob_,
                router_bias);
            return std::make_tuple(top_k_weights, top_k_index);
        } catch (const std::exception &e) {
            warn_router_gpu_fallback_once("DeepseekV4TopKRouter", e);
        }
    }

    auto logits_values = tensor_to_float_vector(logits);
    if (logits_values.size() != ntoken * num_experts_) {
        throw std::runtime_error("DeepseekV4TopKRouter: router logits shape mismatch");
    }

    std::vector<float> scores;
    apply_sqrtsoftplus_scores(scores, logits_values, ntoken, num_experts_);

    std::vector<float> bias;
    if (has_bias()) {
        bias = tensor_to_float_vector(bias_);
    }

    std::vector<int64_t> selected;
    std::vector<float> weights;
    select_topk_experts(ntoken, num_experts_, num_experts_per_tok_, scores, bias, selected);
    weights_from_selected(ntoken, num_experts_, num_experts_per_tok_,
                          norm_topk_prob_, scores, selected, weights);
    // Match SGLang: keep normalized top-k weights; do not apply routed_scaling_factor here.
    (void)routed_scaling_;

    auto top_k_weights = float_vector_to_tensor(weights, {ntoken, num_experts_per_tok_},
                                                infinicore::DataType::F32, device);
    auto top_k_index = int32_vector_to_tensor(selected, {ntoken, num_experts_per_tok_}, device);
    return {top_k_weights, top_k_index};
}

DeepseekV4HashRouter::DeepseekV4HashRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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
DeepseekV4HashRouter::forward(const infinicore::Tensor &hidden_states,
                              const infinicore::Tensor &input_ids) const {
    validate_router_hidden_states(hidden_states, hidden_size_);
    if (input_ids.empty()) {
        throw std::runtime_error("DeepseekV4HashRouter: hash-routed layer requires input_ids");
    }

    const size_t ntoken = hidden_states->shape()[0];
    const auto device = hidden_states->device();
    auto logits = compute_router_logits(hidden_states, weight_);
    if (should_use_deepseek_v4_router_gpu(device) && input_ids->device() == device) {
        try {
            auto input_ids_contiguous = input_ids->contiguous();
            auto [top_k_weights, top_k_index] = infinicore::op::deepseek_v4_hash_router(
                logits,
                input_ids_contiguous,
                tid2eid_,
                norm_topk_prob_);
            return std::make_tuple(top_k_weights, top_k_index);
        } catch (const std::exception &e) {
            warn_router_gpu_fallback_once("DeepseekV4HashRouter", e);
        }
    }

    auto logits_values = tensor_to_float_vector(logits);
    if (logits_values.size() != ntoken * num_experts_) {
        throw std::runtime_error("DeepseekV4HashRouter: router logits shape mismatch");
    }

    std::vector<float> scores;
    apply_sqrtsoftplus_scores(scores, logits_values, ntoken, num_experts_);

    auto token_ids = tensor_to_int64_vector(input_ids);
    if (token_ids.size() != ntoken) {
        throw std::runtime_error("DeepseekV4HashRouter: input_ids size mismatch");
    }
    auto tid2eid = tensor_to_int64_vector(tid2eid_);

    std::vector<int64_t> selected;
    std::vector<float> weights;
    select_hash_experts(ntoken, num_experts_per_tok_, token_ids, tid2eid, selected);
    weights_from_selected(ntoken, num_experts_, num_experts_per_tok_,
                          norm_topk_prob_, scores, selected, weights);
    (void)routed_scaling_;

    auto top_k_weights = float_vector_to_tensor(weights, {ntoken, num_experts_per_tok_},
                                                infinicore::DataType::F32, device);
    auto top_k_index = int32_vector_to_tensor(selected, {ntoken, num_experts_per_tok_}, device);
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
    const auto &config_json = model_config->get_config_json();
    use_fused_moe_ = !(config_json.contains("swiglu_limit") && !config_json.at("swiglu_limit").is_null());
    if (const char *force_fused = std::getenv("DSV4_FORCE_FUSED_MOE"); force_fused != nullptr && std::string(force_fused) == "1") {
        use_fused_moe_ = true;
    }

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
    const bool dense_weights = !gate_weights_.empty() && gate_weights_.front()->dtype() == hidden_states->dtype()
                            && up_weights_.front()->dtype() == hidden_states->dtype()
                            && down_weights_.front()->dtype() == hidden_states->dtype();

    // std::cout << "----------->  use_fused_moe_ " << use_fused_moe_ << std::endl;                                                                                                  // 0
    // std::cout << "----------->  dense_weights " << dense_weights << std::endl;                                                                                                    // 0
    // std::cout << "----------->   supports_fused_deepseek_moe(hidden_states->device().getType()) " << supports_fused_deepseek_moe(hidden_states->device().getType()) << std::endl; // 1

    if (use_fused_moe_
        && dense_weights
        && supports_fused_deepseek_moe(hidden_states->device().getType())
        && top_k_index->dtype() == infinicore::DataType::I32
        && top_k_weights->dtype() == infinicore::DataType::F32
        && (hidden_states->dtype() == infinicore::DataType::BF16
            || hidden_states->dtype() == infinicore::DataType::F16)) {
        try {
            // printf("-----------> 111 gpu fused moe \n");
            return infinicore::op::deepseek_moe(hidden_states, top_k_index, top_k_weights,
                                                gate_weights_, up_weights_, down_weights_,
                                                local_moe_intermediate_size_, num_experts_);
        } catch (const std::exception &e) {
            spdlog::warn("DeepseekV4Experts: deepseek_moe unavailable on {}, falling back to routed experts: {}",
                         static_cast<int>(hidden_states->device().getType()), e.what());
        }
    }
    //  printf("-----------> 222 cpu fused moe \n");
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
        gate_ = this->register_module<DeepseekV4HashRouter>("gate", model_config, layer_idx, device);
    } else {
        gate_ = this->register_module<DeepseekV4TopKRouter>("gate", model_config, layer_idx, device);
    }
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    has_shared_experts_ = num_shared_experts > 0;
    if (num_shared_experts > 0) {
        INFINICORE_NN_MODULE_INIT(shared_experts, model_config, moe_intermediate_size * num_shared_experts, device);
    }

    f32_allreduce_env = std::getenv("DSV4_FFN_F32_ALLREDUCE");
}

infinicore::Tensor DeepseekV4MoE::forward(const infinicore::Tensor &hidden_states,
                                          const infinicore::Tensor &input_ids) const {
    const auto shape = hidden_states->shape();
    if (shape.size() != 3 || shape[2] != hidden_size_) {
        throw std::runtime_error("DeepseekV4MoE: expected hidden_states shape [B,S,D]");
    }
    auto hidden_flat = hidden_states->view({shape[0] * shape[1], hidden_size_});
    auto [routing_weights, selected_experts] = std::visit(
        [&](const auto &gate) {
            return gate->forward(hidden_flat, input_ids);
        },
        gate_);
    auto final_hidden_states = experts_->forward(hidden_flat, selected_experts, routing_weights);
    if (has_shared_experts_) {
        final_hidden_states = infinicore::op::add(final_hidden_states, shared_experts_->forward_without_allreduce(hidden_flat));
    }
    if (tp_size_ > 1 && communicator_ != nullptr) {

        if (f32_allreduce_env != nullptr && std::string(f32_allreduce_env) == "1") {

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
