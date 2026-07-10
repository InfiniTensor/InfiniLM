#include "ernie4_5_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace infinilm::models::ernie4_5_vl {
namespace {

std::vector<size_t> get_size_list(const nlohmann::json &config, const char *key) {
    std::vector<size_t> values;
    if (!config.contains(key) || config.at(key).is_null()) {
        return values;
    }
    const auto &value = config.at(key);
    if (value.is_array()) {
        values.reserve(value.size());
        for (const auto &item : value) {
            values.push_back(item.get<size_t>());
        }
    } else {
        values.push_back(value.get<size_t>());
    }
    return values;
}

} // namespace

Ernie45TopKRouter::Ernie45TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_experts = model_config->get<size_t>("num_experts");
    const auto expert_counts = get_size_list(model_config->get_config_json(), "moe_num_experts");
    const size_t num_router_groups = expert_counts.empty() ? 1 : expert_counts.size();
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    norm_topk_prob_ = model_config->get<bool>("norm_topk_prob");

    ASSERT((num_experts > 0) && (num_router_groups <= 2) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_experts * num_router_groups));

    // ERNIE stores router weights as [hidden_size, group_num_experts], while Qwen3 stores
    // [num_experts, hidden_size]. matmul(hidden, weight) matches ERNIE directly.
    INFINICORE_NN_PARAMETER_INIT(weight, ({hidden_size, num_experts}, dtype, device));
    if (num_router_groups > 1) {
        INFINICORE_NN_PARAMETER_INIT(weight_1, ({hidden_size, num_experts}, dtype, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Ernie45TopKRouter::forward(const infinicore::Tensor &hidden_states, const infinicore::Tensor &correction_bias) const {
    ASSERT(hidden_states->ndim() == 2);
    ASSERT(hidden_states->dtype() == weight_->dtype());

    auto router_logits = infinicore::op::matmul(hidden_states, weight_);
    return infinicore::op::moe_topk_softmax(router_logits, num_experts_per_tok_, norm_topk_prob_, 0.0f, correction_bias);
}

Ernie45ExpertMLP::Ernie45ExpertMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t intermediate_size,
                                   const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const bool use_bias = model_config->get_or<bool>("mlp_bias", false);
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    auto quantization_method = model_config->get_quantization_method();

    INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size, intermediate_size, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(up_proj, hidden_size, intermediate_size, quantization_method, use_bias, dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size, hidden_size, quantization_method, use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor Ernie45ExpertMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(hidden_states_mutable);
    auto up = up_proj_->forward(hidden_states_mutable);
    auto intermediate = infinicore::op::swiglu(up, gate);
    return down_proj_->forward(intermediate);
}

Ernie45Experts::Ernie45Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device) {
    const auto &config_json = model_config->get_config_json();
    auto expert_counts = get_size_list(config_json, "moe_num_experts");
    auto intermediate_sizes = get_size_list(config_json, "moe_intermediate_size");
    if (expert_counts.empty()) {
        expert_counts.push_back(model_config->get<size_t>("num_experts"));
    }
    if (intermediate_sizes.empty()) {
        intermediate_sizes.push_back(model_config->get<size_t>("moe_intermediate_size"));
    }
    if (intermediate_sizes.size() == 1 && expert_counts.size() > 1) {
        intermediate_sizes.resize(expert_counts.size(), intermediate_sizes.front());
    }
    if (expert_counts.size() != intermediate_sizes.size()) {
        throw std::runtime_error("Ernie45Experts: moe_num_experts and moe_intermediate_size group counts differ");
    }

    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t text_intermediate_size = intermediate_sizes.front();
    num_experts_per_tok_ = model_config->get<size_t>("num_experts_per_tok");
    num_text_experts_ = expert_counts.front();
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    ASSERT((num_text_experts_ > 0) && (num_experts_per_tok_ > 0) && (num_experts_per_tok_ <= num_text_experts_));
    INFINICORE_NN_PARAMETER_INIT(w1, ({num_text_experts_, 2 * text_intermediate_size, hidden_size}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(w2, ({num_text_experts_, hidden_size, text_intermediate_size}, dtype, device));
    if (use_bias_) {
        INFINICORE_NN_PARAMETER_INIT(b1, ({num_text_experts_, 2 * text_intermediate_size}, dtype, device));
        INFINICORE_NN_PARAMETER_INIT(b2, ({num_text_experts_, hidden_size}, dtype, device));
    }
}

infinicore::Tensor Ernie45Experts::forward(const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &top_k_index,
                                           const infinicore::Tensor &top_k_weights) const {
    ASSERT(hidden_states->ndim() == 2);
    ASSERT(top_k_index->ndim() == 2 && top_k_weights->ndim() == 2);

    std::optional<infinicore::Tensor> b1 = std::nullopt;
    std::optional<infinicore::Tensor> b2 = std::nullopt;
    if (use_bias_) {
        b1 = b1_;
        b2 = b2_;
    }
    return infinicore::op::fused_moe(hidden_states, top_k_index, top_k_weights, w1_, w2_, b1, b2,
                                    infinicore::op::FusedMoeActivation::Swiglu);
}

Ernie45MoE::Ernie45MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device) {
    num_experts_ = model_config->get<size_t>("num_experts");
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);

    if (model_config->get_or<bool>("moe_use_aux_free", false)) {
        size_t num_expert_groups = 1;
        const auto &config_json = model_config->get_config_json();
        if (config_json.contains("moe_num_experts") && config_json["moe_num_experts"].is_array()) {
            num_expert_groups = config_json["moe_num_experts"].size();
        }
        const size_t num_experts = model_config->get<size_t>("num_experts");
        e_score_correction_bias_ = infinicore::nn::Parameter({num_expert_groups, num_experts}, infinicore::DataType::F32, device);
        this->register_parameter("moe_statics.e_score_correction_bias", e_score_correction_bias_);
    }

    const size_t n_shared_experts = model_config->get_or<size_t>("moe_num_shared_experts", 0);
    has_shared_experts_ = n_shared_experts > 0;
    if (has_shared_experts_) {
        auto shared_config_json = model_config->get_config_json();
        auto intermediate_sizes = get_size_list(shared_config_json, "moe_intermediate_size");
        if (intermediate_sizes.empty()) {
            throw std::runtime_error("Ernie45MoE: moe_intermediate_size is required for shared experts");
        }
        shared_config_json["intermediate_size"] = intermediate_sizes.front() * n_shared_experts;
        auto shared_config = std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
        INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
    }
}

infinicore::Tensor Ernie45MoE::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);
    const auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});
    auto correction_bias = e_score_correction_bias_ ? e_score_correction_bias_->narrow({{0, 0, 1}})->view({num_experts_}) : infinicore::Tensor();
    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped, correction_bias);
    auto final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights)->view(shape);
    if (has_shared_experts_) {
        final_hidden_states = infinicore::op::add(final_hidden_states, shared_experts_->forward(hidden_states));
    }
    return final_hidden_states;
}

} // namespace infinilm::models::ernie4_5_vl

