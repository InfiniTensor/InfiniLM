#pragma once

#include "infinicore/tensor.hpp"
#include <functional>
#include <string>
#include <memory>
#include <unordered_map>

namespace infinilm::models::llama {

/**
 * @brief Hook callback type for capturing intermediate values
 *
 * Hook functions are called with:
 * - name: Identifier for the intermediate value (e.g., "layer0_q_after_proj")
 * - tensor: The intermediate tensor value
 * - layer_idx: Layer index (for layer-specific hooks, -1 if not applicable)
 */
using HookCallback = std::function<void(const std::string &name, const infinicore::Tensor &tensor, int layer_idx)>;

/**
 * @brief Hook registry for managing hooks
 */
class HookRegistry {
public:
    /**
     * @brief Register a hook callback
     *
     * @param name Hook name (can be pattern like "layer0_*" or specific name)
     * @param callback Hook callback function
     */
    void register_hook(const std::string &name, HookCallback callback);

    /**
     * @brief Call hook if registered
     *
     * @param name Full hook name
     * @param tensor Tensor to pass to hook
     * @param layer_idx Layer index (-1 if not applicable)
     */
    void call_hook(const std::string &name, const infinicore::Tensor &tensor, int layer_idx = -1) const;

    /**
     * @brief Clear all hooks
     */
    void clear();

    /**
     * @brief Check if any hooks are registered
     */
    bool has_hooks() const { return !hooks_.empty(); }

private:
    std::unordered_map<std::string, HookCallback> hooks_;
};

} // namespace infinilm::models::llama
