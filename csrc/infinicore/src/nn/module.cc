#include "infinicore/nn/module.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {
namespace {
Parameter get_state_dict_parameter(
    const std::unordered_map<std::string, Parameter> &state_dict,
    const std::string &name) {
    auto it = state_dict.find(name);
    if (it == state_dict.end()) {
        throw std::runtime_error("Parameter '" + name + "' not found in module.");
    }
    return it->second;
}
} // namespace

std::unordered_map<std::string, Parameter> Module::state_dict() const {
    std::unordered_map<std::string, Parameter> result;
    collect_all_parameters(result, "");
    return result;
}

std::vector<std::string> Module::state_dict_keys() const {
    std::vector<std::string> result;
    collect_all_parameter_names(result, "");
    return result;
}

void Module::load_state_dict(const std::unordered_map<std::string, Tensor> &_state_dict) {
    load_state_dict_recursively(_state_dict, "");
}

void Module::load_parameter(const std::string &name, const Tensor &param) {
    auto param_it = parameters_.find(name);
    if (param_it != parameters_.end()) {
        try {
            param_it->second.load(param);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error loading parameter '" + name + "'. \n" + e.what());
        }
        return;
    }

    std::shared_ptr<Module> matched_submodule;
    std::string matched_prefix;
    for (const auto &[sub_name, submodule] : submodules_) {
        if (name.size() <= sub_name.size() || name.compare(0, sub_name.size(), sub_name) != 0 || name[sub_name.size()] != '.') {
            continue;
        }
        if (sub_name.size() > matched_prefix.size()) {
            matched_prefix = sub_name;
            matched_submodule = submodule;
        }
    }

    if (matched_submodule) {
        try {
            matched_submodule->load_parameter(name.substr(matched_prefix.size() + 1), param);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error loading parameter '" + name + "'. \n" + e.what());
        }
        return;
    }

    spdlog::debug("load_parameter: Parameter '{}' not found. Available direct params={}, submodules={}",
                  name, parameters_.size(), submodules_.size());
    throw std::runtime_error("Parameter '" + name + "' not found in module.");
}

void Module::load_parameters_no_sync(const std::unordered_map<std::string, Tensor> &params, bool strict) {
    auto all_params = state_dict();
    for (const auto &[name, param] : params) {
        auto it = all_params.find(name);
        if (it == all_params.end()) {
            if (strict) {
                throw std::runtime_error("Parameter '" + name + "' not found in module.");
            }
            continue;
        }
        auto existing_param = it->second;
        try {
            existing_param.load_no_sync(param);
        } catch (const std::exception &e) {
            throw std::runtime_error("Error loading parameter '" + name + "'. \n" + e.what());
        }
    }
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

void Module::collect_all_parameter_names(std::vector<std::string> &all_names, const std::string &prefix) const {
    for (const auto &[param_name, _] : parameters_) {
        all_names.push_back(prefix.empty() ? param_name : prefix + "." + param_name);
    }

    for (const auto &[sub_name, submodule] : submodules_) {
        std::string sub_prefix = prefix.empty() ? sub_name : prefix + "." + sub_name;
        submodule->collect_all_parameter_names(all_names, sub_prefix);
    }
}

std::unordered_map<std::string, Module *> Module::modules_dict() const {
    std::unordered_map<std::string, Module *> result;
    collect_all_modules(result, "");
    return result;
}

void Module::collect_all_modules(std::unordered_map<std::string, Module *> &out, const std::string &prefix) const {
    // 记录当前模块（跳过根节点的空前缀，可按需改为 "root"）
    if (!prefix.empty()) {
        out[prefix] = const_cast<Module *>(this);
    }

    // 递归遍历子模块
    for (const auto &[name, sub] : submodules_) {
        std::string sub_prefix = prefix.empty() ? name : prefix + "." + name;
        sub->collect_all_modules(out, sub_prefix);
    }
}

} // namespace infinicore::nn
