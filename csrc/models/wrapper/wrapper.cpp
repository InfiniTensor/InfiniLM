#include "wrapper.hpp"
#include <stdexcept> // Add this for std::runtime_error

namespace infinilm::models::llama {

LlamaForCausalLMWrapper::LlamaForCausalLMWrapper(
    const LlamaConfig &config,
    const infinicore::Device &device,
    infinicore::DataType dtype)
    : model_(std::make_shared<LlamaForCausalLM>(config, device, dtype)) {}

infinicore::Tensor LlamaForCausalLMWrapper::forward(
    const infinicore::Tensor &input_ids,
    const infinicore::Tensor &position_ids,
    std::vector<void *> *kv_caches) const {
    return model_->forward(input_ids, position_ids, kv_caches);
}

const LlamaConfig &LlamaForCausalLMWrapper::config() const {
    return model_->config();
}

LlamaForCausalLM &LlamaForCausalLMWrapper::model() {
    return *model_;
}

const LlamaForCausalLM &LlamaForCausalLMWrapper::model() const {
    return *model_;
}

std::unordered_map<std::string, infinicore::nn::Parameter>
LlamaForCausalLMWrapper::state_dict() const {
    return model_->state_dict();
}

infinicore::nn::Parameter LlamaForCausalLMWrapper::get_parameter(const std::string &name) const {
    auto state_dict = model_->state_dict();
    auto it = state_dict.find(name);
    if (it != state_dict.end()) {
        return it->second; // Return the Parameter object
    }
    throw std::runtime_error("Parameter '" + name + "' not found in model");
}

void LlamaForCausalLMWrapper::load_state_dict(
    const std::unordered_map<std::string, infinicore::Tensor> &state_dict) {
    model_->load_state_dict(state_dict);
}

} // namespace infinilm::models::llama
