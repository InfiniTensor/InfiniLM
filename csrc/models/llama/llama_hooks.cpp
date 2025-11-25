#include "llama_hooks.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::llama {

void HookRegistry::register_hook(const std::string &name, HookCallback callback) {
    hooks_[name] = callback;
    SPDLOG_DEBUG("HookRegistry: Registered hook '{}'", name);
}

void HookRegistry::call_hook(const std::string &name, const infinicore::Tensor &tensor, int layer_idx) const {
    // Try exact match first
    auto it = hooks_.find(name);
    if (it != hooks_.end()) {
        try {
            it->second(name, tensor, layer_idx);
        } catch (const std::exception &e) {
            SPDLOG_ERROR("HookRegistry: Error calling hook '{}': {}", name, e.what());
        }
        return;
    }

    // Try pattern matching (e.g., "layer0_*" matches "layer0_q_after_proj")
    for (const auto &[pattern, callback] : hooks_) {
        if (pattern.back() == '*' && name.size() >= pattern.size() - 1) {
            std::string prefix = pattern.substr(0, pattern.size() - 1);
            if (name.substr(0, prefix.size()) == prefix) {
                try {
                    callback(name, tensor, layer_idx);
                } catch (const std::exception &e) {
                    SPDLOG_ERROR("HookRegistry: Error calling hook pattern '{}' for '{}': {}", pattern, name, e.what());
                }
                return;
            }
        }
    }
}

void HookRegistry::clear() {
    hooks_.clear();
    SPDLOG_DEBUG("HookRegistry: Cleared all hooks");
}

} // namespace infinilm::models::llama
