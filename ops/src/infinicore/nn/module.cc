#include "infinicore/nn/module.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {
const std::unordered_map<std::string, Parameter> &Module::state_dict() const {
    static std::unordered_map<std::string, Parameter> result;
    result.clear();

    collect_all_parameters(result, "");

    return result;
}

void Module::load_state_dict(const std::unordered_map<std::string, Tensor> &_state_dict) {
    load_state_dict_recursively(_state_dict, "");
}

void Module::load_parameter(const std::string &name, const Tensor &param) {
    // This function only handles direct parameters (no hierarchical traversal)
    auto all_params = state_dict();
    auto it = all_params.find(name);
    if (it != all_params.end()) {
        auto existing_param = it->second;
        try {
            existing_param.load(param);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error loading parameter '" + name + "'. \n" + e.what());
        }
        return;
    }

    // Parameter not found
    spdlog::debug("load_parameter_: Parameter '{}' not found. Available: {} params",
                  name, parameters_.size());
    throw std::runtime_error("Parameter '" + name + "' not found in module.");
}

void Module::load_parameter_(const std::string &name, const Tensor &param) {
    // This function only handles direct parameters (no hierarchical traversal)
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {
        auto existing_param = it->second;
        try {
            existing_param.load(param);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error loading parameter '" + name + "'. \n" + e.what());
        }
        return;
    }

    // Parameter not found
    spdlog::debug("load_parameter_: Parameter '{}' not found. Available: {} params",
                  name, parameters_.size());
    throw std::runtime_error("Parameter '" + name + "' not found in module.");
}

void Module::load_parameter_from_blob(const std::string &name, const void *data) {
    auto param = parameters_[name];
    param.load_blob(data);
}

Tensor Module::register_parameter(const std::string &name, Parameter param) {
    parameters_[name] = param;
    return param;
}

Tensor Module::register_buffer(const std::string &name, Parameter buffer) {
    buffers_[name] = buffer;
    return buffer;
}

void Module::load_state_dict_recursively(const std::unordered_map<std::string, Tensor> &_state_dict, const std::string &prefix) {
    // Load direct parameters with the given prefix
    for (const auto &[param_name, param] : parameters_) {
        std::string full_name = prefix.empty() ? param_name : prefix + "." + param_name;
        auto it = _state_dict.find(full_name);
        if (it != _state_dict.end()) {
            load_parameter_(param_name, it->second);
        }
    }

    // Recursively load parameters from submodules with extended prefix
    for (const auto &[sub_name, submodule] : submodules_) {
        std::string sub_prefix = prefix.empty() ? sub_name : prefix + "." + sub_name;
        submodule->load_state_dict_recursively(_state_dict, sub_prefix);
    }
}

void Module::collect_all_parameters(std::unordered_map<std::string, Parameter> &all_params, const std::string &prefix) const {
    // Add direct parameters with the given prefix
    for (const auto &[param_name, param] : parameters_) {
        std::string full_name = prefix.empty() ? param_name : prefix + "." + param_name;
        all_params[full_name] = param;
    }

    // Recursively collect parameters from submodules with extended prefix
    for (const auto &[sub_name, submodule] : submodules_) {
        std::string sub_prefix = prefix.empty() ? sub_name : prefix + "." + sub_name;
        submodule->collect_all_parameters(all_params, sub_prefix);
    }
}

} // namespace infinicore::nn
