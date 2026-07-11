#include "ernie4_5_sparse_moe_block.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>

namespace infinilm::models::ernie4_5_moe_vl {

namespace {

std::shared_ptr<infinilm::config::ModelConfig> make_shared_experts_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    auto shared_config_json = model_config->get_config_json();
    const size_t num_shared_experts = model_config->get_or<size_t>("moe_num_shared_experts", 0);
    const size_t moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    shared_config_json["intermediate_size"] = moe_intermediate_size * num_shared_experts;
    return std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
}

size_t first_size_t_or(const nlohmann::json &config_json,
                       const std::string &key,
                       size_t default_value) {
    if (!config_json.contains(key) || config_json.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config_json.at(key);
    if (value.is_array()) {
        if (value.empty()) {
            return default_value;
        }
        return value.at(0).get<size_t>();
    }
    return value.get<size_t>();
}

size_t second_size_t_or(const nlohmann::json &config_json,
                        const std::string &key,
                        size_t default_value) {
    if (!config_json.contains(key) || config_json.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config_json.at(key);
    if (value.is_array()) {
        if (value.size() < 2) {
            return first_size_t_or(config_json, key, default_value);
        }
        return value.at(1).get<size_t>();
    }
    return value.get<size_t>();
}

std::shared_ptr<infinilm::config::ModelConfig> make_vision_experts_config(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    auto vision_config_json = model_config->get_config_json();
    vision_config_json["num_experts"] = vision_config_json.value(
        "vision_num_experts",
        second_size_t_or(vision_config_json, "moe_num_experts", model_config->get<size_t>("num_experts")));
    vision_config_json["moe_intermediate_size"] = vision_config_json.value(
        "vision_moe_intermediate_size",
        second_size_t_or(
            vision_config_json,
            "moe_intermediate_size",
            model_config->get<size_t>("moe_intermediate_size")));
    return std::make_shared<infinilm::config::ModelConfig>(vision_config_json);
}

} // namespace

Ernie4_5SparseMoeBlock::Ernie4_5SparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device,
                                               size_t layer_id)
    : infinilm::layers::moe::SparseMoeBlock(model_config, device, layer_id) {
    has_shared_experts_ = model_config->get_or<size_t>("moe_num_shared_experts", 0) > 0;
    if (has_shared_experts_) {
        auto shared_config = make_shared_experts_config(model_config);
        INFINICORE_NN_MODULE_INIT(shared_experts, shared_config, device);
    }

    const auto &config_json = model_config->get_config_json();
    has_vision_experts_ = model_config->get_or<bool>("enable_ernie_vision_experts", false) && config_json.contains("moe_num_experts") && config_json.at("moe_num_experts").is_array() && config_json.at("moe_num_experts").size() > 1;
    if (has_vision_experts_) {
        auto vision_config = make_vision_experts_config(model_config);
        INFINICORE_NN_MODULE_INIT(vision_gate, vision_config, device);
        INFINICORE_NN_MODULE_INIT(vision_experts, vision_config, device);
        INFINICORE_NN_MODULE_INIT(vision_fused_moe, vision_config, device, layer_id);
    }
}

infinicore::Tensor Ernie4_5SparseMoeBlock::forward_routed_(
    const infinicore::Tensor &hidden_states,
    const infinilm::layers::moe::TopKRouter &gate,
    const infinilm::layers::moe::FusedMoeExperts &experts,
    const infinilm::layers::moe::FusedMoE &fused_moe) const {
    auto [routing_weights, selected_experts] = gate.forward(hidden_states);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };

    auto output = fused_moe.forward(
        hidden_states,
        topk_output,
        experts.moe_weights());
    if (has_shared_experts_) {
        auto shared_output = shared_experts_->forward(hidden_states);
        output = infinicore::op::add(output, shared_output);
    }
    return output;
}

infinicore::Tensor Ernie4_5SparseMoeBlock::forward_text_(const infinicore::Tensor &hidden_states) const {
    return forward_routed_(hidden_states, *gate_, *experts_, *fused_moe_);
}

infinicore::Tensor Ernie4_5SparseMoeBlock::forward_vision_(const infinicore::Tensor &hidden_states) const {
    return forward_routed_(hidden_states, *vision_gate_, *vision_experts_, *vision_fused_moe_);
}

infinicore::Tensor Ernie4_5SparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);
    const auto &visual_ranges = infinilm::global_state::get_forward_context().mm_metadata.visual_token_ranges;
    auto flat_hidden = hidden_states->view({hidden_states->size(0) * hidden_states->size(1), hidden_states->size(2)})->contiguous();
    if (!has_vision_experts_ || !visual_ranges.has_value() || visual_ranges->empty()) {
        return forward_text_(flat_hidden)->view(hidden_states->shape());
    }

    auto flat_output = infinicore::Tensor::empty(flat_hidden->shape(), flat_hidden->dtype(), flat_hidden->device());

    size_t current = 0;
    const size_t ntokens = flat_hidden->size(0);
    const auto &ranges = visual_ranges.value();
    for (size_t i = 0; i + 1 < ranges.size(); i += 2) {
        const size_t range_start = ranges[i];
        const size_t range_end = ranges[i + 1];
        const size_t start = std::min(range_start, ntokens);
        const size_t end = std::min(range_end, ntokens);
        if (start > current) {
            const size_t len = start - current;
            auto text_output = forward_text_(flat_hidden->narrow({{0, current, len}})->contiguous());
            flat_output->narrow({{0, current, len}})->copy_from(text_output);
        }
        if (end > start) {
            const size_t len = end - start;
            auto vision_output = forward_vision_(flat_hidden->narrow({{0, start, len}})->contiguous());
            flat_output->narrow({{0, start, len}})->copy_from(vision_output);
        }
        current = std::max(current, end);
    }
    if (current < ntokens) {
        const size_t len = ntokens - current;
        auto text_output = forward_text_(flat_hidden->narrow({{0, current, len}})->contiguous());
        flat_output->narrow({{0, current, len}})->copy_from(text_output);
    }
    return flat_output->view(hidden_states->shape());
}

} // namespace infinilm::models::ernie4_5_moe_vl
